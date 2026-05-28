from __future__ import annotations

import pandas as pd


def to_returns_frame(data, variable: str = "asset_returns") -> pd.DataFrame:
    """
    Convert project data objects into a canonical returns DataFrame.

    Accepted inputs
    ---------------
    - pandas.DataFrame with shape (time, asset | factor)
    - xarray.DataArray with dims (time, asset | factor)
    - xarray.Dataset containing ``variable`` with dims (time, asset | factor)
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("returns DataFrame must not be empty")
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        return data.copy()

    try:
        import xarray as xr
    except ModuleNotFoundError:
        xr = None

    if xr is not None and isinstance(data, xr.DataArray):
        if data.dims != ("time", "asset") and data.dims != ("time", "factor"):
            raise ValueError("returns DataArray must have dims ('time', 'asset | factor')")
        frame = data.to_pandas()
        return frame.sort_index()

    if xr is not None and isinstance(data, xr.Dataset):
        if variable not in data:
            raise ValueError(f"dataset must contain variable {variable!r}")
        array = data[variable]
        if array.dims != ("time", "asset") and array.dims != ("time", "factor"):
            raise ValueError(
                f"dataset variable {variable!r} must have dims ('time', 'asset')"
            )
        frame = array.to_pandas()
        return frame.sort_index()

    raise TypeError(
        "data must be a pandas.DataFrame, xarray.DataArray, or xarray.Dataset"
    )
