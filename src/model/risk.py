import numpy as np
import xarray as xr

def rolling_risk_decomposition(forward_ds: xr.Dataset,
                               factor_ds: xr.Dataset,
                               window: int = 252,
                               ) -> xr.Dataset:
    """
    Compute rolling variance decomposition from forward factor model.

    Parameters
    ----------
    forward_ds : xr.Dataset
        Must contain:
            betas(time, asset, factor)
            residuals(time, asset)

    factor_ds : xr.Dataset
        Must contain:
            factor_returns(time, factor)

    window : int
        Rolling window length used for covariance estimation.

    Returns
    -------
    xr.Dataset with variables:

        systematic_var(time, asset)
        idiosyncratic_var(time, asset)
        total_var(time, asset)
        systematic_share(time, asset)
        factor_contributions(time, asset, factor)
    """
    forward_ds, factor_ds = xr.align(forward_ds, factor_ds, join="inner")

    betas = forward_ds["betas"].values           # (T, N, K)
    residuals = forward_ds["residuals"].values   # (T, N)
    F = factor_ds["factor_returns"].values       # (T, K)

    T, N, K = betas.shape

    systematic_var = np.full((T, N), np.nan)
    idiosyncratic_var = np.full((T, N), np.nan)
    total_var = np.full((T, N), np.nan)
    systematic_share = np.full((T, N), np.nan)
    factor_contributions = np.full((T, N, K), np.nan)

    for t in range(window - 1, T):

        F_window = F[t - window + 1 : t + 1]   # (window, K)
        eps_window = residuals[t - window + 1 : t + 1]  # (window, N)

        # Factor covariance
        Sigma_f = np.cov(F_window, rowvar=False)

        for i in range(N):

            beta = betas[t, i, :]  # (K,)

            if np.any(np.isnan(beta)):
                continue

            # Systematic variance
            sys_var = beta @ Sigma_f @ beta

            # Residual variance
            eps_i = eps_window[:, i]
            idio_var = np.var(eps_i, ddof=1)

            total = sys_var + idio_var

            systematic_var[t, i] = sys_var
            idiosyncratic_var[t, i] = idio_var
            total_var[t, i] = total

            if total > 0:
                systematic_share[t, i] = sys_var / total

            # Factor-level contributions
            # beta_k * (Sigma_f beta)_k
            marginal = Sigma_f @ beta
            factor_contributions[t, i, :] = beta * marginal

    result = xr.Dataset(
        data_vars={
            "systematic_var": (("time", "asset"), systematic_var),
            "idiosyncratic_var": (("time", "asset"), idiosyncratic_var),
            "total_var": (("time", "asset"), total_var),
            "systematic_share": (("time", "asset"), systematic_share),
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
            "model": "rolling_variance_decomposition",
        },
    )

    return result
