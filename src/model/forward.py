import numpy as np
import xarray as xr


def forward_decompose(asset_ds: xr.Dataset,
                      factor_ds: xr.Dataset,
                      window: int = 252,
                      include_intercept: bool = True,
                      ) -> xr.Dataset:
    """
    Perform rolling linear regression of asset returns on factor returns.

    Parameters
    ----------
    asset_ds : xr.Dataset
        Must contain:
            returns(time, asset)

    factor_ds : xr.Dataset
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

    asset_ds, factor_ds = xr.align(asset_ds, factor_ds, join="inner")

    R = asset_ds["returns"].values           # (T, N)
    F = factor_ds["factor_returns"].values   # (T, K)

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
            "time": asset_ds.time,
            "asset": asset_ds.asset,
            "factor": factor_ds.factor,
        },
        attrs={
            "window": window,
            "model": "rolling_ols",
            "include_intercept": include_intercept,
        },
    )

    return result
