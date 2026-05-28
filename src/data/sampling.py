import numpy as np
import xarray as xr

from src.data.cache import bootstrap_cache_name, cache_exists, load_cache, save_cache
from src.data.validation import validate_bootstrap_dataset, validate_bootstrap_inputs


def bootstrap_time_paths(
    asset_da: xr.DataArray,
    factor_da: xr.DataArray,
    block_size: int = 20,
    n_paths: int = 100,
    seed: int | None = None,
    method: str = "block",
    align: str = "inner",
    return_dataset: bool = True,
    use_cache: bool = True,
) -> list[xr.Dataset] | xr.Dataset:
    """
    Generate block-bootstrapped time paths for asset and factor returns.

    Parameters
    ----------
    asset_da : xr.DataArray
        Asset returns with dimensions ``(time, asset)``.
    factor_da : xr.DataArray
        Factor returns with dimensions ``(time, factor)``.
    block_size : int, default 20
        Length of each contiguous time block sampled with replacement.
    n_paths : int, default 100
        Number of bootstrap paths to generate.
    seed : int, optional
        Seed passed to ``np.random.default_rng`` for reproducible resampling.
    method : {"block"}, default "block"
        Bootstrap sampling method. Only fixed-length block bootstrap is
        implemented in v1.
    align : {"inner", "left", "right", "outer", "exact"}, default "inner"
        Alignment rule passed to ``xr.align`` before resampling.
    return_dataset : bool, default True
        If True, return one dataset with a ``path`` dimension. If False,
        return a list of per-path datasets.
    use_cache : bool, default True
        If True, load/save the stacked dataset representation from cache.

    Returns
    -------
    xr.Dataset or list[xr.Dataset]
        Bootstrapped paths containing:
            asset_returns(path, time, asset)
            factor_returns(path, time, factor)
    """
    if align != "inner":
        raise ValueError("bootstrap_time_paths currently supports align='inner' only")

    try:
        asset_da, factor_da = validate_bootstrap_inputs(
            asset_da=asset_da,
            factor_da=factor_da,
            block_size=block_size,
            n_paths=n_paths,
            method=method,
        )
    except AssertionError as exc:
        raise ValueError("Invalid bootstrap inputs") from exc

    n_time = asset_da.sizes["time"]

    cache_name = bootstrap_cache_name(
        assets=asset_da["asset"].values.tolist(),
        factors=factor_da["factor"].values.tolist(),
        start=str(asset_da["time"].values[0]),
        end=str(asset_da["time"].values[-1]),
        block_size=block_size,
        n_paths=n_paths,
        seed=seed,
        method=method,
        align=align,
        time_dim="time",
    )

    if use_cache and cache_exists(cache_name):
        cached = load_cache(cache_name)
        if not isinstance(cached, xr.Dataset):
            raise TypeError(f"Expected Dataset cache entry for {cache_name}")
        validate_bootstrap_dataset(cached)
        return _split_bootstrap_dataset(cached) if not return_dataset else cached

    asset_values = asset_da.values
    factor_values = factor_da.values

    n_assets = asset_values.shape[1]
    n_factors = factor_values.shape[1]
    n_block_starts = n_time - block_size + 1
    n_blocks_per_path = int(np.ceil(n_time / block_size))
    rng = np.random.default_rng(seed)

    boot_asset_returns = np.empty((n_paths, n_time, n_assets), dtype=asset_values.dtype)
    boot_factor_returns = np.empty((n_paths, n_time, n_factors), dtype=factor_values.dtype)

    for path_idx in range(n_paths):
        start_idxs = rng.integers(0, n_block_starts, size=n_blocks_per_path)

        sampled_asset_blocks = [
            asset_values[start_idx : start_idx + block_size] for start_idx in start_idxs
        ]
        sampled_factor_blocks = [
            factor_values[start_idx : start_idx + block_size] for start_idx in start_idxs
        ]

        boot_asset_returns[path_idx] = np.concatenate(sampled_asset_blocks, axis=0)[:n_time]
        boot_factor_returns[path_idx] = np.concatenate(sampled_factor_blocks, axis=0)[:n_time]

    result = xr.Dataset(
        data_vars={
            "asset_returns": (
                ("path", "time", "asset"),
                boot_asset_returns,
                asset_da.attrs,
            ),
            "factor_returns": (
                ("path", "time", "factor"),
                boot_factor_returns,
                factor_da.attrs,
            ),
        },
        coords={
            "path": np.arange(n_paths),
            "time": asset_da["time"],
            "asset": asset_da["asset"],
            "factor": factor_da["factor"],
        },
        attrs={
            "sampling_method": "time_block_bootstrap",
            "block_size": block_size,
            "n_paths": n_paths,
            "seed": seed,
            "align": align,
            "cache_name": cache_name,
        },
    )

    validate_bootstrap_dataset(result)

    if use_cache:
        save_cache(result, cache_name)

    return _split_bootstrap_dataset(result) if not return_dataset else result


def _split_bootstrap_dataset(paths_ds: xr.Dataset) -> list[xr.Dataset]:
    return [paths_ds.isel(path=path_idx).drop_vars("path") for path_idx in paths_ds["path"].values]
