import numpy as np
import xarray as xr
from src.data.validation import validate_returns_da, validate_factor_da


def rolling_forward_decompose(asset_da: xr.DataArray,
                              factor_da: xr.DataArray,
                              window: int = 252,
                              include_intercept: bool = True,
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

    Returns
    -------
    xr.Dataset with variables:

        betas(time, asset, factor)
        fitted(time, asset)
        residuals(time, asset)

    Notes
    -----
    - Betas are aligned to the *end* of each window.
    - First (window - 1) observations will be NaN.
    """

    # Validate data structures and align
    validate_returns_da(asset_da)
    validate_factor_da(factor_da)

    asset_da, factor_da = xr.align(asset_da, factor_da, join="inner")

    R = asset_da.values    # (T, N)
    F = factor_da.values   # (T, K)

    T, N = R.shape
    _, K = F.shape

    betas = np.full((T, N, K), np.nan)
    alphas = np.full((T, N), np.nan) if include_intercept else None
    fitted = np.full((T, N), np.nan)
    residuals = np.full((T, N), np.nan)

    for t in range(window - 1, T):

        R_window = R[t - window + 1 : t + 1]   # (window, N)
        F_window = F[t - window + 1 : t + 1]   # (window, K)

        if include_intercept:
            X = np.column_stack([np.ones(window), F_window])
        else:
            # Pure factor model
            X = F_window

        # Solve for each asset separately
        for n in range(N):
            y = R_window[:, n]

            beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)

            if include_intercept:
                alpha = beta_full[0]
                beta = beta_full[1:]
                alphas[t, n] = alpha
            else:
                beta = beta_full

            betas[t, n, :] = beta
            fitted[t, n] = (F[t] @ beta) + (alpha if include_intercept else 0.0)
            residuals[t, n] = R[t, n] - fitted[t, n]

    # Construct Outputs
    if include_intercept:
        data_vars={
            "betas": (("time", "asset", "factor"), betas),
            "alphas": (("time", "asset"), alphas),
            "fitted": (("time", "asset"), fitted),
            "residuals": (("time", "asset"), residuals),
        }
    else:
        data_vars={
            "betas": (("time", "asset", "factor"), betas),
            "fitted": (("time", "asset"), fitted),
            "residuals": (("time", "asset"), residuals),
        },

    result = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": asset_da.time,
            "asset": asset_da.asset,
            "factor": factor_da.factor,
        },
        attrs={
            "window": window,
            "regression_type": "rolling_ols",
            "include_intercept": include_intercept,
            "factor_model": "sector_etf",
        },
    )

    return result


def sample_forward_decompose(asset_da: xr.DataArray,
                             factor_da: xr.DataArray,
                             include_intercept: bool = True,
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

    Returns
    -------
    xr.Dataset with variables:

        betas(asset, factor)
        alphas(asset)                (optional)
        fitted(time, asset)
        residuals(time, asset)

    Notes
    -----
    - Betas are estimated using the entire sample.
    - Fitted values and residuals are computed across all dates using the
      full-sample parameter estimates.
    """

    # Validate data structures and align
    validate_returns_da(asset_da)
    validate_factor_da(factor_da)

    asset_da, factor_da = xr.align(asset_da, factor_da, join="inner")

    R = asset_da.values      # (T, N)
    F = factor_da.values     # (T, K)

    T, N = R.shape
    _, K = F.shape

    betas = np.zeros((N, K))
    alphas = np.zeros(N) if include_intercept else None

    # Build regression matrix
    if include_intercept:
        X = np.column_stack([np.ones(T), F])
    else:
        X = F

    for n in range(N):
        y = R[:, n]

        beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)

        if include_intercept:
            alphas[n] = beta_full[0]
            betas[n, :] = beta_full[1:]
        else:
            betas[n, :] = beta_full

    # Compute fitted returns across all time
    fitted = F @ betas.T

    if include_intercept:
        fitted += alphas

    residuals = R - fitted

    # Construct dataset
    data_vars = {
        "betas": (("asset", "factor"), betas),
        "fitted": (("time", "asset"), fitted),
        "residuals": (("time", "asset"), residuals),
    }

    if include_intercept:
        data_vars["alphas"] = (("asset",), alphas)

    result = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": asset_da.time,
            "asset": asset_da.asset,
            "factor": factor_da.factor,
        },
        attrs={
            "regression_type": "full_sample_ols",
            "include_intercept": include_intercept,
            "factor_model": "sector_etf",
        },
    )

    return result
