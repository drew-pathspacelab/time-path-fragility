import hashlib
import json
from pathlib import Path
from typing import Any

import xarray as xr


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_SCHEMA_VERSION = "0.1"
_CACHE_OBJECT_TYPE_ATTR = "_cache_object_type"
_CACHE_DATAARRAY_NAME_ATTR = "_cache_dataarray_name"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.zarr"


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_freeze(item) for item in value]
    if isinstance(value, dict):
        return {key: _freeze(value[key]) for key in sorted(value)}
    return value


def make_cache_key(kind: str, **params: Any) -> str:
    frozen_params = _freeze(params)
    payload = json.dumps(frozen_params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{kind}_v{CACHE_SCHEMA_VERSION}_{digest}"


def price_cache_name(
    tickers,
    start: str,
    end: str,
    frequency: str,
    is_factor: bool = False,
) -> str:
    return make_cache_key(
        "factor_prices" if is_factor else "asset_prices",
        tickers=sorted(tickers),
        start=start,
        end=end,
        frequency=frequency,
    )


def bootstrap_cache_name(
    assets,
    factors,
    start: str,
    end: str,
    block_size: int,
    n_paths: int,
    seed: int | None,
    method: str = "block",
    align: str = "inner",
    time_dim: str = "time",
) -> str:
    return make_cache_key(
        "bootstrap_paths",
        assets=list(assets),
        factors=list(factors),
        start=start,
        end=end,
        block_size=block_size,
        n_paths=n_paths,
        seed=seed,
        method=method,
        align=align,
        time_dim=time_dim,
    )


def cache_exists(name: str) -> bool:
    return _cache_path(name).exists()


def load_cache(name: str) -> xr.Dataset | xr.DataArray:
    dataset = xr.open_zarr(_cache_path(name), consolidated=False)
    object_type = dataset.attrs.get(_CACHE_OBJECT_TYPE_ATTR, "dataset")

    if object_type == "dataarray":
        variable_name = dataset.attrs.get(_CACHE_DATAARRAY_NAME_ATTR)
        if variable_name is None:
            if len(dataset.data_vars) != 1:
                raise ValueError(
                    f"Cache entry {name!r} does not identify a DataArray variable"
                )
            variable_name = next(iter(dataset.data_vars))
        return dataset[variable_name]

    return dataset


def save_cache(data: xr.Dataset | xr.DataArray, name: str) -> None:
    path = _cache_path(name)

    if isinstance(data, xr.DataArray):
        variable_name = data.name or "data"
        dataset = data.to_dataset(name=variable_name)
        dataset.attrs[_CACHE_OBJECT_TYPE_ATTR] = "dataarray"
        dataset.attrs[_CACHE_DATAARRAY_NAME_ATTR] = variable_name
    elif isinstance(data, xr.Dataset):
        dataset = data.copy()
        dataset.attrs[_CACHE_OBJECT_TYPE_ATTR] = "dataset"
    else:
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    dataset.to_zarr(path, mode="w", consolidated=False)
