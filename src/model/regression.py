import warnings

import numpy as np
import xarray as xr
from src.data.validation import validate_returns_da, validate_factor_da
from src.model.time_weights import time_spec_attrs, time_weights, validate_time_spec

ESTIMATORS = ("ols", "huber")

# Normal-consistency constant for the MAD scale estimate, norm.ppf(0.75).
_MAD_NORMAL = 0.6744897501960817


# ---------------------------------------------------------------------------
# Fitting core
# ---------------------------------------------------------------------------
# All regressions solve for every asset at once. Time weights w (oldest first,
# summing to one) enter as weighted least squares; for Huber they multiply the
# IRLS robustness weights, so an observation's influence is (time weight) x
# (outlier downweight).

def _design(F: np.ndarray, include_intercept: bool) -> np.ndarray:
    if include_intercept:
        return np.column_stack([np.ones(F.shape[0]), F])
    return F


def _wls(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares for all columns of Y; returns coef (N, p)."""
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
    return coef.T


def _weighted_median(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Column-wise weighted median of values (n, N).

    Interpolates on cumulative-weight midpoints, which reduces to np.median
    (including the even-n average) when the weights are equal.
    """
    n = values.shape[0]
    if n == 1:
        return values[0].copy()
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_w = w[order]
    midpoints = np.cumsum(sorted_w, axis=0) - 0.5 * sorted_w
    half = 0.5 * sorted_w.sum(axis=0)

    # Vectorized np.interp(half, midpoints[:, j], sorted_values[:, j]) per column.
    cols = np.arange(values.shape[1])
    hi = np.clip((midpoints < half).sum(axis=0), 1, n - 1)
    lo = hi - 1
    m_lo, m_hi = midpoints[lo, cols], midpoints[hi, cols]
    frac = np.clip((half - m_lo) / (m_hi - m_lo), 0.0, 1.0)
    v_lo, v_hi = sorted_values[lo, cols], sorted_values[hi, cols]
    return v_lo + frac * (v_hi - v_lo)


def _robust_scale(resid: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Time-weighted MAD scale about zero (statsmodels RLM's default)."""
    return _weighted_median(np.abs(resid), w) / _MAD_NORMAL


def _huber_irls(X, Y, w, epsilon, max_iter, tol, update_scale=False, init=None):
    """Huber M-estimate by IRLS with a time-weighted MAD scale.

    With ``update_scale=False`` (default) the scale is fixed at the MAD of the
    initial weighted-least-squares residuals, so the IRLS solves a convex
    problem and always converges. With ``update_scale=True`` the scale is
    re-estimated every iteration; under EWMA weights the weighted median can
    jump as residuals reorder, and the iteration may cycle.

    ``init`` optionally warm-starts the coefficients (the scale still comes
    from the WLS residuals); with a fixed scale the solution does not depend
    on the starting point.

    Matches statsmodels ``RLM(M=HuberT(epsilon)).fit(scale_est="mad",
    update_scale=...)`` when the time weights are equal.
    Returns (coef (N, p), scale (N,), converged).
    """
    coef = _wls(X, Y, w)
    scale = _robust_scale(Y - X @ coef.T, w)
    if init is not None and not update_scale:
        coef = init
    resid = Y - X @ coef.T

    for _ in range(max_iter):
        # A zero scale means a perfect fit; treat every residual as an inlier.
        safe_scale = np.where(scale > 0, scale, np.inf)
        u = np.abs(resid) / safe_scale
        psi_weights = np.minimum(1.0, epsilon / np.maximum(u, np.finfo(float).tiny))
        v = w[:, None] * psi_weights

        Xv = v.T[:, :, None] * X[None]              # (N, n, p)
        A = Xv.transpose(0, 2, 1) @ X               # (N, p, p)
        c = (Xv * Y.T[:, :, None]).sum(axis=1)      # (N, p)
        new_coef = np.linalg.solve(A, c[..., None])[..., 0]

        resid = Y - X @ new_coef.T
        if update_scale:
            scale = _robust_scale(resid, w)
        converged = np.all(np.abs(new_coef - coef) <= tol * np.maximum(1.0, np.abs(coef)))
        coef = new_coef
        if converged:
            return coef, scale, True

    return coef, scale, False


def _fit(X, Y, w, estimator, huber_epsilon, max_iter, tol, update_scale, init=None):
    if estimator == "ols":
        return _wls(X, Y, w), None, True
    return _huber_irls(X, Y, w, huber_epsilon, max_iter, tol, update_scale, init)


def _validate_spec(estimator, weighting, span, halflife, huber_epsilon):
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator!r}; expected one of {ESTIMATORS}")
    if estimator == "huber" and huber_epsilon <= 0:
        raise ValueError("huber_epsilon must be > 0")
    return validate_time_spec(weighting, span, halflife, stacklevel=4)


def _spec_attrs(estimator, weighting, span, halflife, huber_epsilon, update_scale) -> dict:
    attrs = {"estimator": estimator, **time_spec_attrs(weighting, span, halflife)}
    if estimator == "huber":
        attrs["huber_epsilon"] = float(huber_epsilon)
        attrs["huber_scale_method"] = "mad_updated" if update_scale else "mad_fixed"
    return attrs


def _warn_unconverged(n_failed: int, n_total: int, max_iter: int) -> None:
    if n_failed:
        warnings.warn(
            f"Huber IRLS did not converge in {max_iter} iterations for "
            f"{n_failed} of {n_total} regressions",
            RuntimeWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_forward_decompose(asset_da: xr.DataArray,
                              factor_da: xr.DataArray,
                              window: int = 252,
                              include_intercept: bool = True,
                              estimator: str = "ols",
                              weighting: str = "equal",
                              span: float | None = None,
                              halflife: float | None = None,
                              huber_epsilon: float = 1.345,
                              max_iter: int = 50,
                              tol: float = 1e-8,
                              update_scale: bool = False,
                              ) -> xr.Dataset:
    """
    Perform rolling linear regression of asset returns on factor returns.

    Parameters
    ----------
    asset_da : xr.DataArray
        Must contain:
            returns(time, asset)

    factor_da : xr.DataArray
        Must contain:
            factor_returns(time, factor)

    window : int
        Rolling regression window length.

    include_intercept: bool
        Include intercept (alpha) in regression.

    estimator : {"ols", "huber"}
        "ols" is (weighted) least squares. "huber" is a Huber M-estimate fit
        by IRLS with a time-weighted MAD scale, which downweights observations
        whose residuals exceed ``huber_epsilon`` scale units.

    weighting : {"equal", "ewma"}
        Time-weighting of observations within each window.

    span, halflife : float, optional
        EWMA decay; exactly one is required when weighting is "ewma".

    huber_epsilon : float
        Huber threshold in units of the residual scale. 1.345 gives 95%
        efficiency relative to OLS under normal errors.

    max_iter, tol : int, float
        IRLS iteration limit and coefficient tolerance (Huber only); the
        tolerance is absolute for |coef| < 1 and relative above.

    update_scale : bool
        Huber only. False (default) fixes the MAD scale from the initial
        weighted-least-squares residuals, which guarantees convergence; True
        re-estimates it every iteration (statsmodels RLM's default), which can
        cycle under EWMA weights.

    Returns
    -------
    xr.Dataset with variables:

        betas(time, asset, factor)
        alphas(time, asset)           (if include_intercept)
        fitted(time, asset)
        residuals(time, asset)
        huber_scale(time, asset)      (if estimator="huber")

    and attrs recording ``estimator``, ``weighting`` and ``span``/``halflife``,
    which ``src.optimizer.estimators`` reads to keep the factor-model moments
    on the same time scale as the betas.

    Notes
    -----
    - Betas are aligned to the *end* of each window.
    - First (window - 1) observations will be NaN.
    """
    weighting, span, halflife = _validate_spec(estimator, weighting, span, halflife, huber_epsilon)

    # Validate data structures and align
    validate_returns_da(asset_da)
    validate_factor_da(factor_da)

    asset_da, factor_da = xr.align(asset_da, factor_da, join="inner")

    R = asset_da.values    # (T, N)
    F = factor_da.values   # (T, K)

    T, N = R.shape
    _, K = F.shape

    X_all = _design(F, include_intercept)
    w = time_weights(window, weighting, span=span, halflife=halflife)

    betas = np.full((T, N, K), np.nan)
    alphas = np.full((T, N), np.nan) if include_intercept else None
    fitted = np.full((T, N), np.nan)
    residuals = np.full((T, N), np.nan)
    scales = np.full((T, N), np.nan) if estimator == "huber" else None
    n_unconverged = 0
    coef = None

    for t in range(window - 1, T):
        rows = slice(t - window + 1, t + 1)
        # Adjacent windows share window - 1 observations, so the previous
        # solution is a close warm start for Huber.
        coef, scale, converged = _fit(
            X_all[rows], R[rows], w, estimator, huber_epsilon, max_iter, tol, update_scale, init=coef
        )
        n_unconverged += not converged

        if include_intercept:
            alphas[t] = coef[:, 0]
            betas[t] = coef[:, 1:]
        else:
            betas[t] = coef
        if scales is not None:
            scales[t] = scale

        fitted[t] = coef @ X_all[t]
        residuals[t] = R[t] - fitted[t]

    _warn_unconverged(n_unconverged, T - window + 1, max_iter)

    # Construct Outputs
    data_vars = {
        "betas": (("time", "asset", "factor"), betas),
        "fitted": (("time", "asset"), fitted),
        "residuals": (("time", "asset"), residuals),
    }
    if include_intercept:
        data_vars["alphas"] = (("time", "asset"), alphas)
    if scales is not None:
        data_vars["huber_scale"] = (("time", "asset"), scales)

    result = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": asset_da.time,
            "asset": asset_da.asset,
            "factor": factor_da.factor,
        },
        attrs={
            "window": window,
            "regression_type": f"rolling_{estimator}",
            "include_intercept": include_intercept,
            "factor_model": "sector_etf",
            **_spec_attrs(estimator, weighting, span, halflife, huber_epsilon, update_scale),
        },
    )

    return result


def sample_forward_decompose(asset_da: xr.DataArray,
                             factor_da: xr.DataArray,
                             include_intercept: bool = True,
                             estimator: str = "ols",
                             weighting: str = "equal",
                             span: float | None = None,
                             halflife: float | None = None,
                             huber_epsilon: float = 1.345,
                             max_iter: int = 50,
                             tol: float = 1e-8,
                             update_scale: bool = False,
                             ) -> xr.Dataset:
    """
    Perform full-sample linear regression of asset returns on factor returns.

    Parameters
    ----------
    asset_da : xr.DataArray
        returns(time, asset)

    factor_da : xr.DataArray
        factor_returns(time, factor)

    include_intercept : bool
        Include intercept (alpha) in regression.

    estimator, weighting, span, halflife, huber_epsilon, max_iter, tol, update_scale
        As in :func:`rolling_forward_decompose`; EWMA weights span the full
        sample, so the most recent observations dominate.

    Returns
    -------
    xr.Dataset with variables:

        betas(asset, factor)
        alphas(asset)                (optional)
        fitted(time, asset)
        residuals(time, asset)
        huber_scale(asset)           (if estimator="huber")

    Notes
    -----
    - Betas are estimated using the entire sample.
    - Fitted values and residuals are computed across all dates using the
      full-sample parameter estimates.
    """
    weighting, span, halflife = _validate_spec(estimator, weighting, span, halflife, huber_epsilon)

    # Validate data structures and align
    validate_returns_da(asset_da)
    validate_factor_da(factor_da)

    asset_da, factor_da = xr.align(asset_da, factor_da, join="inner")

    R = asset_da.values      # (T, N)
    F = factor_da.values     # (T, K)

    T, N = R.shape

    X = _design(F, include_intercept)
    w = time_weights(T, weighting, span=span, halflife=halflife)

    coef, scale, converged = _fit(X, R, w, estimator, huber_epsilon, max_iter, tol, update_scale)
    _warn_unconverged(int(not converged), 1, max_iter)

    if include_intercept:
        alphas = coef[:, 0]
        betas = coef[:, 1:]
    else:
        betas = coef

    # Compute fitted returns across all time
    fitted = X @ coef.T
    residuals = R - fitted

    # Construct dataset
    data_vars = {
        "betas": (("asset", "factor"), betas),
        "fitted": (("time", "asset"), fitted),
        "residuals": (("time", "asset"), residuals),
    }

    if include_intercept:
        data_vars["alphas"] = (("asset",), alphas)
    if scale is not None:
        data_vars["huber_scale"] = (("asset",), scale)

    result = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": asset_da.time,
            "asset": asset_da.asset,
            "factor": factor_da.factor,
        },
        attrs={
            "regression_type": f"full_sample_{estimator}",
            "include_intercept": include_intercept,
            "factor_model": "sector_etf",
            **_spec_attrs(estimator, weighting, span, halflife, huber_epsilon, update_scale),
        },
    )

    return result
