from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.optimizer._data import to_returns_frame

"""TODO: method="factor_model" needs to be separated into:
         1. Specifying model structure i.e. direct-asset model vs factor model
         2. Time-weighting (hist, ewma) for the moment estimation method
            {mu: [f, alphas, residuals], cov: [F,D]}

         Currently hard-coded to historical estimates for f, F, and residual moments.
"""
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


def _factor_returns_mean(factor_returns: pd.DataFrame, method: str, halflife: float | None) -> pd.Series:
    if method == "hist":
        return factor_returns.mean(axis=0)
    if method == "ewma":
        if halflife is None:
            raise ValueError("halflife is required for method='ewma'")
        return factor_returns.ewm(halflife=halflife, adjust=False).mean().iloc[-1]
    raise ValueError(f"Unknown factor-model estimation method: {method}")


def _factor_returns_cov(factor_returns: pd.DataFrame, method: str, halflife: float | None) -> pd.DataFrame:
    if method == "hist":
        return factor_returns.cov()
    if method == "ewma":
        if halflife is None:
            raise ValueError("halflife is required for method='ewma'")
        ewm_cov = factor_returns.ewm(halflife=halflife, adjust=False).cov()
        n_factors = factor_returns.shape[1]
        cov = ewm_cov.iloc[-n_factors:].copy()
        cov.index = factor_returns.columns
        return cov
    raise ValueError(f"Unknown factor-model estimation method: {method}")


def _latest_betas(forward_ds) -> pd.DataFrame:
    betas = forward_ds["betas"]
    if betas.dims == ("asset", "factor"):
        return betas.to_pandas()
    if betas.dims == ("time", "asset", "factor"):
        latest = betas.dropna(dim="time", how="all").isel(time=-1)
        return latest.to_pandas()
    raise ValueError("forward_ds['betas'] must have dims ('asset', 'factor') or ('time', 'asset', 'factor')")


def _residual_mean(residuals: pd.DataFrame, method: str, halflife: float | None) -> pd.Series:
    if method == "hist":
        return residuals.mean(axis=0)
    if method == "ewma":
        if halflife is None:
            raise ValueError("halflife is required for method='ewma'")
        return residuals.ewm(halflife=halflife, adjust=False).mean().iloc[-1]
    raise ValueError(f"Unknown residual estimation method: {method}")


def _residual_variance(residuals: pd.DataFrame, method: str, halflife: float | None) -> pd.Series:
    if method == "hist":
        return residuals.var(axis=0, ddof=1)
    if method == "ewma":
        if halflife is None:
            raise ValueError("halflife is required for method='ewma'")
        return residuals.ewm(halflife=halflife, adjust=False).var().iloc[-1]
    raise ValueError(f"Unknown residual estimation method: {method}")


class MeanEstimator(BaseEstimator):
    def __init__(
        self,
        method: str = "hist",
        halflife: float | None = None,
        shrinkage: float = 0.0,
        shrinkage_target: str = "grand_mean",
        variable: str = "asset_returns",
    ):
        self.method = method
        self.halflife = halflife
        self.shrinkage = shrinkage
        self.shrinkage_target = shrinkage_target
        self.variable = variable

    def fit(self, X, y=None):
        if self.method in {"hist", "ewma"}:
            returns = to_returns_frame(X, variable=self.variable)

            if self.method == "hist":
                mu = returns.mean(axis=0)
            else:
                if self.halflife is None:
                    raise ValueError("halflife is required for method='ewma'")
                mu = returns.ewm(halflife=self.halflife, adjust=False).mean().iloc[-1]

            self.returns_ = returns
        elif self.method == "factor_model":
            factor_returns = y if y is not None else None
            forward_ds, factor_returns = _align_factor_model_inputs(X, factor_returns)

            betas = _latest_betas(forward_ds)
            factor_returns_df = factor_returns.to_pandas()
            residuals_df = forward_ds["residuals"].to_pandas()

            factor_mu = _factor_returns_mean(factor_returns_df, "hist", None)
            residual_mu = _residual_mean(residuals_df, "hist", None)
            mu = betas @ factor_mu + residual_mu.reindex(betas.index).fillna(0.0)

            if "alphas" in forward_ds:
                alphas = forward_ds["alphas"]
                if alphas.dims == ("asset",):
                    mu = mu + alphas.to_pandas().reindex(betas.index).fillna(0.0)
                elif alphas.dims == ("time", "asset"):
                    alpha_latest = alphas.dropna(dim="time", how="all").isel(time=-1).to_pandas()
                    mu = mu + alpha_latest.reindex(betas.index).fillna(0.0)

            self.forward_ds_ = forward_ds
            self.factor_returns_ = factor_returns_df
        else:
            raise ValueError(f"Unknown mean estimation method: {self.method}")

        self.mean_ = _shrink_mean(mu.astype(float), self.shrinkage, self.shrinkage_target)
        return self


class CovarianceEstimator(BaseEstimator):
    def __init__(
        self,
        method: str = "hist",
        halflife: float | None = None,
        shrinkage: float = 0.0,
        shrinkage_target: str = "diagonal",
        variable: str = "asset_returns",
        ridge: float = 1e-8,
    ):
        self.method = method
        self.halflife = halflife
        self.shrinkage = shrinkage
        self.shrinkage_target = shrinkage_target
        self.variable = variable
        self.ridge = ridge

    def fit(self, X, y=None):
        if self.method in {"hist", "ewma"}:
            returns = to_returns_frame(X, variable=self.variable)

            if self.method == "hist":
                raw_cov = returns.cov()
            else:
                if self.halflife is None:
                    raise ValueError("halflife is required for method='ewma'")
                ewm_cov = returns.ewm(halflife=self.halflife, adjust=False).cov()
                n_assets = returns.shape[1]
                raw_cov = ewm_cov.iloc[-n_assets:].copy()
                raw_cov.index = returns.columns

            self.returns_ = returns
        elif self.method == "factor_model":
            factor_returns = y if y is not None else None
            forward_ds, factor_returns = _align_factor_model_inputs(X, factor_returns)

            betas = _latest_betas(forward_ds)
            factor_returns_df = factor_returns.to_pandas()
            residuals_df = forward_ds["residuals"].to_pandas().reindex(columns=betas.index)

            factor_cov = _factor_returns_cov(factor_returns_df, "hist", None)
            residual_var = _residual_variance(residuals_df, "hist", None).fillna(0.0)

            b_vals = betas.values.astype(float)
            sigma_f = factor_cov.reindex(index=betas.columns, columns=betas.columns).values.astype(float)
            d_vals = np.diag(residual_var.reindex(betas.index).values.astype(float))
            raw_cov = pd.DataFrame(
                b_vals @ sigma_f @ b_vals.T + d_vals,
                index=betas.index,
                columns=betas.index,
            )

            self.forward_ds_ = forward_ds
            self.factor_returns_ = factor_returns_df
        else:
            raise ValueError(f"Unknown covariance estimation method: {self.method}")

        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage alpha must be in [0, 1]")

        target_cov = _covariance_target(raw_cov, self.shrinkage_target)
        cov = (1.0 - self.shrinkage) * raw_cov + self.shrinkage * target_cov

        self.covariance_ = _stabilize_covariance(cov.astype(float), ridge=self.ridge)
        return self


def compute_mu(data, factor_returns=None, **kwargs) -> pd.Series:
    estimator = MeanEstimator(**kwargs)
    estimator.fit(data, y=factor_returns)
    return estimator.mean_.copy()


def compute_cov(data, factor_returns=None, **kwargs) -> pd.DataFrame:
    estimator = CovarianceEstimator(**kwargs)
    estimator.fit(data, y=factor_returns)

    cov = estimator.covariance_.copy()

    # Final checks
    assert np.all(np.isfinite(cov))

    return cov
