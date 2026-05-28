import numpy as np
import xarray as xr

def rolling_risk_decomposition(forward_ds: xr.Dataset,
                               factor_da: xr.DataArray,
                               ) -> xr.Dataset:
    """
    Compute rolling variance decomposition from forward factor model.

    Parameters
    ----------
    forward_ds : xr.Dataset
        Must contain:
            betas(time, asset, factor)
            residuals(time, asset)

    factor_da : xr.DataArray
        Must contain:
            returns(time, factor)

    Returns
    -------
    xr.Dataset with variables:

        sys_var(time, asset)
        idio_var(time, asset)
        total_var(time, asset)
        sys_share(time, asset)
        factor_contributions(time, asset, factor)
    """
    window = forward_ds.window
    forward_ds, factor_da = xr.align(forward_ds, factor_da, join="inner")

    betas = forward_ds["betas"].values           # (T, N, K)
    residuals = forward_ds["residuals"].values   # (T, N)
    F = factor_da.values                         # (T, K)

    T, N, K = betas.shape

    sys_var = np.full((T, N), np.nan)
    idio_var = np.full((T, N), np.nan)
    total_var = np.full((T, N), np.nan)
    sys_share = np.full((T, N), np.nan)
    factor_contributions = np.full((T, N, K), np.nan)

    for t in range(window - 1, T):

        F_window = F[t - window + 1 : t + 1]   # (window, K)
        eps_window = residuals[t - window + 1 : t + 1]  # (window, N)

        # Factor covariance
        Sigma_f = np.cov(F_window, rowvar=False)

        for i in range(N):

            beta = betas[t, i, :]  # (K,)

            if np.any(np.isnan(beta)):
                # if the window is incomplete skip ahead
                continue

            # Systematic variance
            systematic_var = beta @ Sigma_f @ beta

            # Residual variance
            eps_i = eps_window[:, i]
            residuals_var = np.var(eps_i, ddof=1)

            total = systematic_var + residuals_var

            sys_var[t, i] = systematic_var
            idio_var[t, i] = residuals_var
            total_var[t, i] = total

            if total > 0:
                sys_share[t, i] = systematic_var / total

            # Factor-level contributions
            # beta_k * (Sigma_f beta)_k
            marginal = Sigma_f @ beta
            factor_contributions[t, i, :] = beta * marginal

    result = xr.Dataset(
        data_vars={
            "sys_var": (("time", "asset"), sys_var),
            "idio_var": (("time", "asset"), idio_var),
            "total_var": (("time", "asset"), total_var),
            "sys_share": (("time", "asset"), sys_share),
            "factor_contributions": (
                ("time", "asset", "factor"),
                factor_contributions,
            ),
        },
        coords={
            "time": forward_ds.time,
            "asset": forward_ds.asset,
            "factor": forward_ds.factor,
        },
        attrs={
            "window": window,
            "model_type": "rolling_variance_decomposition",
        },
    )

    return result


def sample_risk_decomposition(forward_ds: xr.Dataset,
                              factor_da: xr.DataArray,
                              ) -> xr.Dataset:
    """
    Compute variance decomposition over the full sample forward factor model.

    Parameters
    ----------
    forward_ds : xr.Dataset
        Must contain:
            betas(asset, factor)
            residuals(time, asset)

    factor_da : xr.DataArray
        Must contain:
            returns(time, factor)

    Returns
    -------
    xr.Dataset with variables:

        sys_var(asset)
        idio_var(asset)
        total_var(asset)
        sys_share(asset)
        factor_contributions(asset, factor)
        Sigma_f(factor, factor)
    """

    forward_ds, factor_da = xr.align(forward_ds, factor_da, join="inner")

    betas = forward_ds["betas"].values           # (N, K)
    residuals = forward_ds["residuals"].values   # (T, N)
    F = factor_da.values                         # (T, K)

    T = forward_ds.sizes['time']
    N = forward_ds.sizes['asset']
    K = forward_ds.sizes['factor']

    sys_var = np.full((N), np.nan)
    idio_var = np.full((N), np.nan)
    total_var = np.full((N), np.nan)
    sys_share = np.full((N), np.nan)
    factor_contributions = np.full((N, K), np.nan)

    # Factor covariance
    Sigma_f = np.cov(F, rowvar=False)

    for i in range(N):

        beta = betas[i, :]  # (K,)

        if np.any(np.isnan(beta)):
            raise ValueError("NaNs detected in full-sample regression coefficients")

        # Systematic variance
        systematic_var = beta @ Sigma_f @ beta

        # Residual variance
        residuals_var = np.var(residuals[:,i], ddof=1)

        total = systematic_var + residuals_var

        sys_var[i] = systematic_var
        idio_var[i] = residuals_var
        total_var[i] = total

        if total > 0:
            sys_share[i] = systematic_var / total

        # Factor-level contributions
        # beta_k * (Sigma_f beta)_k
        marginal = Sigma_f @ beta
        factor_contributions[i, :] = beta * marginal

    result = xr.Dataset(
        data_vars={
            "sys_var": ("asset", sys_var),
            "idio_var": ("asset", idio_var),
            "total_var": ("asset", total_var),
            "sys_share": ("asset", sys_share),
            "factor_contributions": (("asset", "factor"), factor_contributions),
            "Sigma_f": (("factor", "factor_p"), Sigma_f),
        },
        coords={
            "asset": forward_ds.asset,
            "factor": forward_ds.factor,
            "factor_p": forward_ds.factor,
        },
        attrs={
            "window": None,
            "sample_start": str(forward_ds.time[0].values),
            "sample_end": str(forward_ds.time[-1].values),
            "sample_size": T,
            "model_type": "sample_variance_decomposition",
            "factor_model": "sector_etf",
        },
    )

    return result
