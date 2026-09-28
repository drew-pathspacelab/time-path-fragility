from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.model.time_weights import resolve_decay, time_weights, validate_time_spec
from src.optimizer._data import to_returns_frame

"""Moment estimators for the optimizer.

Two independent choices describe an estimate:

1. ``method`` -- model structure
   - "sample":       asset-to-asset moments estimated directly from asset returns
   - "factor_model": mu = B f + alpha + e_bar,  Sigma = B F B^T + D, built from a
                     forward decomposition (``src.model.regression``)

2. ``weighting`` -- time-weighting of the moments ("equal" or "ewma", with
   exactly one of ``span`` / ``halflife`` for "ewma"). Kernels come from
   ``src.model.time_weights`` so they match the regression layer.

For "factor_model" the betas were already estimated upstream with their own
weighting (and estimator, e.g. OLS or Huber), recorded in ``forward_ds.attrs``.
With ``weighting=None`` the estimator inherits that weighting for f, F, and the
residual moments; an explicit weighting that differs raises a
:class:`TimeScaleMismatchWarning`, since B and F/D would then describe different
time scales. The regression *estimator* (OLS vs Huber) is not checked: it
changes B, alpha, and the residuals, but not how their moments are weighted here.
"""

METHODS = ("sample", "factor_model")
_DEPRECATED_METHODS = {"hist": "sample", "ewma": "sample"}


class TimeScaleMismatchWarning(UserWarning):
    """Factor-model moments weighted differently from the regression that produced B."""


# ---------------------------------------------------------------------------
# Shrinkage and stabilization
# ---------------------------------------------------------------------------

def _shrink_mean(raw_mu: pd.Series, alpha: float, target: str) -> pd.Series:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("shrinkage alpha must be in [0, 1]")

    if target == "zero":
        target_mu = pd.Series(0.0, index=raw_mu.index)
    elif target == "grand_mean":
        target_mu = pd.Series(float(raw_mu.mean()), index=raw_mu.index)
    else:
        raise ValueError(f"Unknown mean shrinkage target: {target}")

    return (1.0 - alpha) * raw_mu + alpha * target_mu


def _covariance_target(raw_cov: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "diagonal":
        target_cov = pd.DataFrame(
            np.diag(np.diag(raw_cov.values)),
            index=raw_cov.index,
            columns=raw_cov.columns,
        )
    elif target == "identity":
        avg_var = float(np.trace(raw_cov.values) / raw_cov.shape[0])
        target_cov = pd.DataFrame(
            np.eye(raw_cov.shape[0]) * avg_var,
            index=raw_cov.index,
            columns=raw_cov.columns,
        )
    else:
        raise ValueError(f"Unknown covariance shrinkage target: {target}")

    return target_cov


def _stabilize_covariance(cov: pd.DataFrame, ridge: float = 1e-8) -> pd.DataFrame:
    values = 0.5 * (cov.values + cov.values.T)
    values = values + np.eye(values.shape[0]) * ridge
    return pd.DataFrame(values, index=cov.index, columns=cov.columns)


# ---------------------------------------------------------------------------
# Method / weighting resolution
# ---------------------------------------------------------------------------

def _none_if_nan(value):
    if value is None:
        return None
    try:
        return None if np.isnan(value) else float(value)
    except TypeError:
        return value


def _regression_time_spec(attrs) -> tuple[str, float | None, float | None]:
    """Weighting recorded by the regression that produced ``forward_ds``.

    Decompositions without a ``weighting`` attr (plain OLS) are equal-weighted.
    """
    weighting = attrs.get("weighting", "equal")
    span = _none_if_nan(attrs.get("span"))
    halflife = _none_if_nan(attrs.get("halflife"))
    if weighting == "ewma" and (span is None) == (halflife is None):
        raise ValueError("forward_ds.attrs declare weighting='ewma' without exactly one of span/halflife")
    return weighting, span, halflife


def _resolve_method(method: str, weighting: str | None, kind: str) -> tuple[str, str | None]:
    if method in _DEPRECATED_METHODS:
        replacement = _DEPRECATED_METHODS[method]
        if method == "ewma":
            if weighting not in (None, "ewma"):
                raise ValueError(f"method='ewma' conflicts with weighting={weighting!r}")
            weighting = "ewma"
            hint = "method='sample', weighting='ewma'"
        else:
            hint = f"method={replacement!r}"
        warnings.warn(
            f"method={method!r} is deprecated; use {hint}",
            FutureWarning,
            stacklevel=4,
        )
        method = replacement

    if method not in METHODS:
        raise ValueError(f"Unknown {kind} estimation method: {method}")
    return method, weighting


def _resolve_time_spec(
    method: str,
    weighting: str | None,
    span: float | None,
    halflife: float | None,
    regression_spec: tuple[str, float | None, float | None] | None = None,
) -> tuple[str, float | None, float | None]:
    """Resolve and validate (weighting, span, halflife) for one fit."""
    if weighting is None:
        if method == "factor_model":
            reg_weighting, reg_span, reg_halflife = regression_spec
            weighting = reg_weighting
            if span is None and halflife is None:
                span, halflife = reg_span, reg_halflife
        else:
            weighting = "equal"

    weighting, span, halflife = validate_time_spec(weighting, span, halflife, stacklevel=5)

    if method == "factor_model":
        _check_time_scale_match(weighting, span, halflife, regression_spec)

    return weighting, span, halflife


def _check_time_scale_match(weighting, span, halflife, regression_spec) -> None:
    reg_weighting, reg_span, reg_halflife = regression_spec
    if weighting == reg_weighting == "equal":
        return
    if weighting == reg_weighting == "ewma" and np.isclose(
        resolve_decay(span=span, halflife=halflife),
        resolve_decay(span=reg_span, halflife=reg_halflife),
    ):
        return

    def describe(w, s, h):
        if w == "equal":
            return "equal"
        return f"ewma(span={s})" if s is not None else f"ewma(halflife={h})"

    warnings.warn(
        "factor-model moments use "
        f"{describe(weighting, span, halflife)} weighting but the betas were estimated with "
        f"{describe(*regression_spec)} weighting; B and F/D describe different time scales",
        TimeScaleMismatchWarning,
        stacklevel=5,
    )


# ---------------------------------------------------------------------------
# Weighted moments
# ---------------------------------------------------------------------------
# Weights are positional (oldest first). Missing values drop out and the
# remaining weights are renormalized per column (per column pair for the
# covariance), matching pandas ``ewm(adjust=True, ignore_na=False)`` and the
# pairwise-complete behaviour of ``DataFrame.cov``. Variances use the
# reliability-weight correction 1 / (1 - sum w^2), which reduces to ddof=1 for
# equal weights and matches pandas' bias-corrected EWM variance.

def _weighted_mean(frame: pd.DataFrame, w: np.ndarray) -> pd.Series:
    x = frame.to_numpy(dtype=float)
    mask = np.isfinite(x)
    wm = w[:, None] * mask
    total = wm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (wm * np.where(mask, x, 0.0)).sum(axis=0) / total
    mean[total <= 0] = np.nan
    return pd.Series(mean, index=frame.columns)


def _pair_cov(xi: np.ndarray, xj: np.ndarray, w: np.ndarray) -> float:
    valid = np.isfinite(xi) & np.isfinite(xj)
    if valid.sum() < 2:
        return np.nan
    wv = w[valid] / w[valid].sum()
    correction = 1.0 - (wv ** 2).sum()
    if correction <= 0:
        return np.nan
    di = xi[valid] - wv @ xi[valid]
    dj = xj[valid] - wv @ xj[valid]
    return float(wv @ (di * dj) / correction)


def _weighted_cov(frame: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    x = frame.to_numpy(dtype=float)
    n_cols = x.shape[1]

    if np.isfinite(x).all() and x.shape[0] >= 2:
        centered = x - w @ x
        cov = (centered * w[:, None]).T @ centered / (1.0 - (w ** 2).sum())
    else:
        cov = np.full((n_cols, n_cols), np.nan)
        for i in range(n_cols):
            for j in range(i, n_cols):
                cov[i, j] = cov[j, i] = _pair_cov(x[:, i], x[:, j], w)

    return pd.DataFrame(cov, index=frame.columns, columns=frame.columns)


def _weighted_var(frame: pd.DataFrame, w: np.ndarray) -> pd.Series:
    x = frame.to_numpy(dtype=float)
    var = np.array([_pair_cov(x[:, i], x[:, i], w) for i in range(x.shape[1])])
    return pd.Series(var, index=frame.columns)


# ---------------------------------------------------------------------------
# Factor-model inputs
# ---------------------------------------------------------------------------

def _align_factor_model_inputs(forward_ds, factor_returns):
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise TypeError("factor-model estimation requires xarray") from exc

    if not isinstance(forward_ds, xr.Dataset):
        raise TypeError("forward_ds must be an xarray.Dataset")
    if not isinstance(factor_returns, xr.DataArray):
        raise TypeError("factor_returns must be an xarray.DataArray")

    required_vars = {"betas", "residuals"}
    missing = required_vars - set(forward_ds.data_vars)
    if missing:
        raise ValueError(f"forward_ds missing required variables: {sorted(missing)}")
    if factor_returns.dims != ("time", "factor"):
        raise ValueError("factor_returns must have dims ('time', 'factor')")

    forward_ds, factor_returns = xr.align(forward_ds, factor_returns, join="inner")
    if forward_ds.sizes["time"] == 0:
        raise ValueError("aligned factor-model inputs must not be empty")

    return forward_ds, factor_returns


def _latest_betas(forward_ds) -> pd.DataFrame:
    betas = forward_ds["betas"]
    if betas.dims == ("asset", "factor"):
        return betas.to_pandas()
    if betas.dims == ("time", "asset", "factor"):
        latest = betas.dropna(dim="time", how="all").isel(time=-1)
        return latest.to_pandas()
    raise ValueError("forward_ds['betas'] must have dims ('asset', 'factor') or ('time', 'asset', 'factor')")


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class _MomentEstimator(BaseEstimator):
    """Shared method/weighting resolution for the mean and covariance estimators."""

    _kind = "moment"

    def _prepare(self, X, y):
        """Resolve the spec and load inputs; returns (frames, weights).

        frames is ``returns`` for "sample" and ``(betas, factor_returns, residuals)``
        for "factor_model".
        """
        method, weighting = _resolve_method(self.method, self.weighting, self._kind)

        if method == "sample":
            weighting, span, halflife = _resolve_time_spec(method, weighting, self.span, self.halflife)
            returns = to_returns_frame(X, variable=self.variable)
            self.returns_ = returns
            frames = returns
            n_obs = len(returns)
        else:
            forward_ds, factor_returns = _align_factor_model_inputs(X, y)
            regression_spec = _regression_time_spec(forward_ds.attrs)
            weighting, span, halflife = _resolve_time_spec(
                method, weighting, self.span, self.halflife, regression_spec
            )
            betas = _latest_betas(forward_ds)
            factor_returns_df = factor_returns.to_pandas()
            residuals_df = forward_ds["residuals"].to_pandas().reindex(columns=betas.index)
            self.forward_ds_ = forward_ds
            self.factor_returns_ = factor_returns_df
            frames = (betas, factor_returns_df, residuals_df)
            n_obs = len(factor_returns_df)

        self.method_ = method
        self.weighting_ = weighting
        self.span_ = span
        self.halflife_ = halflife
        self.time_weights_ = time_weights(n_obs, weighting, span=span, halflife=halflife)
        return frames, self.time_weights_


class MeanEstimator(_MomentEstimator):
    """Expected-return estimator.

    Parameters
    ----------
    method : {"sample", "factor_model"}
        Model structure. "hist" and "ewma" are deprecated aliases for "sample"
        (the latter with ``weighting="ewma"``).
    weighting : {None, "equal", "ewma"}
        Time-weighting of the moments. None means "equal" for "sample" and
        "inherit from forward_ds.attrs" for "factor_model".
    span, halflife : float, optional
        EWMA decay; exactly one is required when weighting is "ewma".
    """

    _kind = "mean"

    def __init__(
        self,
        method: str = "sample",
        weighting: str | None = None,
        span: float | None = None,
        halflife: float | None = None,
        shrinkage: float = 0.0,
        shrinkage_target: str = "grand_mean",
        variable: str = "asset_returns",
    ):
        self.method = method
        self.weighting = weighting
        self.span = span
        self.halflife = halflife
        self.shrinkage = shrinkage
        self.shrinkage_target = shrinkage_target
        self.variable = variable

    def fit(self, X, y=None):
        frames, w = self._prepare(X, y)

        if self.method_ == "sample":
            mu = _weighted_mean(frames, w)
        else:
            betas, factor_returns_df, residuals_df = frames
            factor_mu = _weighted_mean(factor_returns_df, w)
            residual_mu = _weighted_mean(residuals_df, w)
            mu = betas @ factor_mu.reindex(betas.columns) + residual_mu.reindex(betas.index).fillna(0.0)

            if "alphas" in self.forward_ds_:
                alphas = self.forward_ds_["alphas"]
                if alphas.dims == ("asset",):
                    mu = mu + alphas.to_pandas().reindex(betas.index).fillna(0.0)
                elif alphas.dims == ("time", "asset"):
                    alpha_latest = alphas.dropna(dim="time", how="all").isel(time=-1).to_pandas()
                    mu = mu + alpha_latest.reindex(betas.index).fillna(0.0)

        self.mean_ = _shrink_mean(mu.astype(float), self.shrinkage, self.shrinkage_target)
        return self


class CovarianceEstimator(_MomentEstimator):
    """Covariance estimator.

    Parameters
    ----------
    method : {"sample", "factor_model"}
        Model structure. "hist" and "ewma" are deprecated aliases for "sample"
        (the latter with ``weighting="ewma"``).
    weighting : {None, "equal", "ewma"}
        Time-weighting of the moments. None means "equal" for "sample" and
        "inherit from forward_ds.attrs" for "factor_model".
    span, halflife : float, optional
        EWMA decay; exactly one is required when weighting is "ewma".
    """

    _kind = "covariance"

    def __init__(
        self,
        method: str = "sample",
        weighting: str | None = None,
        span: float | None = None,
        halflife: float | None = None,
        shrinkage: float = 0.0,
        shrinkage_target: str = "diagonal",
        variable: str = "asset_returns",
        ridge: float = 1e-8,
    ):
        self.method = method
        self.weighting = weighting
        self.span = span
        self.halflife = halflife
        self.shrinkage = shrinkage
        self.shrinkage_target = shrinkage_target
        self.variable = variable
        self.ridge = ridge

    def fit(self, X, y=None):
        frames, w = self._prepare(X, y)

        if self.method_ == "sample":
            raw_cov = _weighted_cov(frames, w)
        else:
            betas, factor_returns_df, residuals_df = frames
            factor_cov = _weighted_cov(factor_returns_df, w)
            residual_var = _weighted_var(residuals_df, w).fillna(0.0)

            b_vals = betas.values.astype(float)
            sigma_f = factor_cov.reindex(index=betas.columns, columns=betas.columns).values.astype(float)
            d_vals = np.diag(residual_var.reindex(betas.index).values.astype(float))
            raw_cov = pd.DataFrame(
                b_vals @ sigma_f @ b_vals.T + d_vals,
                index=betas.index,
                columns=betas.index,
            )

        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage alpha must be in [0, 1]")

        target_cov = _covariance_target(raw_cov, self.shrinkage_target)
        cov = (1.0 - self.shrinkage) * raw_cov + self.shrinkage * target_cov

        self.covariance_ = _stabilize_covariance(cov.astype(float), ridge=self.ridge)
        return self


def compute_mu(data, factor_returns=None, **kwargs) -> pd.Series:
    """Estimate the expected return (mean) vector for a set of assets.

    Thin functional wrapper around :class:`MeanEstimator`: constructs the
    estimator with the given keyword arguments, fits it to ``data`` (and
    optionally ``factor_returns``), and returns the fitted mean vector.

    Parameters
    ----------
    data : pd.DataFrame or xr.Dataset
        Input data used to estimate the mean. For ``method="sample"``, this
        is passed to ``to_returns_frame`` to build a returns frame. For
        ``method="factor_model"``, this must be an ``xarray.Dataset``
        containing ``betas`` and ``residuals`` (and optionally ``alphas``).
    factor_returns : xr.DataArray, optional
        Factor return series with dims ``("time", "factor")``. Required
        when ``method="factor_model"``; ignored otherwise.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`MeanEstimator`, e.g. ``method`` ("sample" or
        "factor_model"), ``weighting`` ("equal" or "ewma"), ``span`` or
        ``halflife`` (exactly one for "ewma"), ``shrinkage``,
        ``shrinkage_target`` ("zero" or "grand_mean"), and ``variable``.

    Returns
    -------
    pd.Series
        Estimated (and shrunk) mean return for each asset, indexed by
        asset.
    """
    estimator = MeanEstimator(**kwargs)
    estimator.fit(data, y=factor_returns)
    return estimator.mean_.copy()


def compute_cov(data, factor_returns=None, **kwargs) -> pd.DataFrame:
    """Estimate the covariance matrix of asset returns.

    Thin functional wrapper around :class:`CovarianceEstimator`:
    constructs the estimator with the given keyword arguments, fits it to
    ``data`` (and optionally ``factor_returns``), and returns the fitted,
    shrunk, and numerically stabilized covariance matrix.

    Parameters
    ----------
    data : pd.DataFrame or xr.Dataset
        Input data used to estimate the covariance. For ``method="sample"``,
        this is passed to ``to_returns_frame`` to build a returns frame. For
        ``method="factor_model"``, this must be an ``xarray.Dataset``
        containing ``betas`` and ``residuals``.
    factor_returns : xr.DataArray, optional
        Factor return series with dims ``("time", "factor")``. Required
        when ``method="factor_model"``; ignored otherwise.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`CovarianceEstimator`, e.g. ``method`` ("sample" or
        "factor_model"), ``weighting`` ("equal" or "ewma"), ``span`` or
        ``halflife`` (exactly one for "ewma"), ``shrinkage``,
        ``shrinkage_target`` ("diagonal" or "identity"), ``variable``, and
        ``ridge`` (diagonal loading for numerical stability).

    Returns
    -------
    pd.DataFrame
        Symmetric, positive-definite (ridge-stabilized) covariance
        matrix, indexed and columned by asset.

    Raises
    ------
    AssertionError
        If the resulting covariance matrix contains any non-finite
        values.
    """
    estimator = CovarianceEstimator(**kwargs)
    estimator.fit(data, y=factor_returns)
    cov = estimator.covariance_.copy()
    assert np.all(np.isfinite(cov))
    return cov
