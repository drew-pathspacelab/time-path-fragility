import xarray as xr


"""Data structure validation functions."""
def validate_returns_da(da: xr.DataArray):
    """Validate asset return DataArray."""
    assert isinstance(da, xr.DataArray)
    assert da.name == "asset_returns"
    assert da.dims == ("time", "asset")
    assert da.ndim == 2
    assert da.time.to_index().is_monotonic_increasing
    assert da.time.size > 0
    assert da.asset.size > 0


def validate_factor_da(da: xr.DataArray):
    """Validate factor return DataArray."""
    assert isinstance(da, xr.DataArray)
    assert da.name == "factor_returns"
    assert da.dims == ("time", "factor")
    assert da.ndim == 2
    assert da.time.to_index().is_monotonic_increasing
    assert da.time.size > 0
    assert da.factor.size > 0


def validate_bootstrap_inputs(
    asset_da: xr.DataArray,
    factor_da: xr.DataArray,
    block_size: int,
    n_paths: int,
    method: str = "block",
):
    """Validate inputs for bootstrap path sampling."""
    validate_returns_da(asset_da)
    validate_factor_da(factor_da)

    assert method == "block"
    assert isinstance(block_size, int)
    assert block_size > 0
    assert isinstance(n_paths, int)
    assert n_paths > 0

    aligned_asset_da, aligned_factor_da = xr.align(asset_da, factor_da, join="inner")

    assert aligned_asset_da.time.size > 0
    assert aligned_factor_da.time.size == aligned_asset_da.time.size
    assert block_size <= aligned_asset_da.time.size

    return aligned_asset_da, aligned_factor_da


def validate_bootstrap_dataset(paths_ds: xr.Dataset):
    """Validate a bootstrapped dataset with path, time, asset, and factor axes."""
    assert isinstance(paths_ds, xr.Dataset)
    assert {"asset_returns", "factor_returns"} <= set(paths_ds.data_vars)

    asset_returns = paths_ds["asset_returns"]
    factor_returns = paths_ds["factor_returns"]

    assert asset_returns.dims == ("path", "time", "asset")
    assert factor_returns.dims == ("path", "time", "factor")
    assert paths_ds.path.size > 0
    assert paths_ds.time.to_index().is_monotonic_increasing
    assert paths_ds.asset.size > 0
    assert paths_ds.factor.size > 0


"""Variance validation functions."""
def validate_variance_identity(returns, fitted, residuals, tol=1e-2):
    direct = returns.var("time", ddof=1)
    model = fitted.var("time", ddof=1) + residuals.var("time", ddof=1)
    rel_err = abs(direct - model) / direct
    assert (rel_err < tol).all()


def validate_forward_decomposition_identity(
    returns_da: xr.DataArray,
    factor_da: xr.DataArray,
    forward_ds: xr.Dataset,
    tol: float = 1e-10,
) -> xr.Dataset:
    """
    Validate the pointwise forward-model reconstruction:

        returns ~= alpha + beta @ factor + residual

    This is intended as an optional diagnostic for both rolling and
    full-sample decompositions. It returns error terms so callers can inspect
    magnitude and timing instead of only receiving a pass/fail signal.
    """

    validate_returns_da(returns_da)
    validate_factor_da(factor_da)

    returns_da, factor_da = xr.align(returns_da, factor_da, join="inner")

    fitted_from_beta = (forward_ds["betas"] * factor_da).sum("factor")

    if "alphas" in forward_ds:
        fitted_from_beta = fitted_from_beta + forward_ds["alphas"]

    reconstruction_error = returns_da - fitted_from_beta - forward_ds["residuals"]
    fitted_error = forward_ds["fitted"] - fitted_from_beta

    valid_reconstruction = reconstruction_error.notnull()
    valid_fitted = fitted_error.notnull()

    max_abs_reconstruction_error = float(
        abs(reconstruction_error).where(valid_reconstruction).max(skipna=True).item()
    ) if bool(valid_reconstruction.any()) else 0.0

    max_abs_fitted_error = float(
        abs(fitted_error).where(valid_fitted).max(skipna=True).item()
    ) if bool(valid_fitted.any()) else 0.0

    assert max_abs_reconstruction_error < tol, (
        f"Forward decomposition reconstruction error exceeded tolerance: "
        f"{max_abs_reconstruction_error} >= {tol}"
    )
    assert max_abs_fitted_error < tol, (
        f"Forward fitted-value error exceeded tolerance: "
        f"{max_abs_fitted_error} >= {tol}"
    )

    return xr.Dataset(
        data_vars={
            "reconstruction_error": reconstruction_error,
            "fitted_error": fitted_error,
        },
        attrs={
            "tolerance": tol,
            "max_abs_reconstruction_error": max_abs_reconstruction_error,
            "max_abs_fitted_error": max_abs_fitted_error,
        },
    )
