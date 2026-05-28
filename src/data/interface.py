from typing import List
import pandas as pd
import numpy as np
import xarray as xr

from .backends.yfinance_backend import fetch_adjusted_prices, BACKEND_METADATA
from .cache import cache_exists, load_cache, save_cache, price_cache_name


INTERFACE_METADATA = {'schema_version': "0.1"}


def load_prices(tickers: List[str],
                start: str,
                end: str,
                frequency: str = "1d",
                is_factor: bool = False,
                use_cache: bool = True,
                ) -> xr.DataArray:

    name = price_cache_name(
        tickers=tickers,
        start=start,
        end=end,
        frequency=frequency,
        is_factor=is_factor,
    )

    if use_cache and cache_exists(name):
        print(f"Loading cached data: {name}")
        cached = load_cache(name)
        if not isinstance(cached, xr.DataArray):
            raise TypeError(f"Expected DataArray cache entry for {name}")
        return cached

    print(f"Fetching data from backend: {name}")
    prices = fetch_adjusted_prices(tickers=tickers,
                                   start=start,
                                   end=end,
                                   frequency=frequency,
                                   )
    dim_name = "factor" if is_factor else "asset"

    prices_da = xr.DataArray(prices.values,
                      coords={"time": prices.index.values,
                              dim_name: prices.columns.values,
                              },
                      dims=("time", dim_name),
                      name="price"
                      )

    # Attach dataset metadata
    metadata = BACKEND_METADATA.copy()
    metadata["created"] = str(pd.Timestamp.utcnow())
    metadata["frequency"] = "daily" if frequency=="1d" else frequency
    metadata.update(INTERFACE_METADATA)

    prices_da.attrs.update(metadata)

    if use_cache:
        save_cache(prices_da, name)

    return prices_da


def get_returns(tickers: List[str],
                start: str,
                end: str,
                frequency: str = "1d",
                is_factor: bool = False,
                use_cache: bool = True,
                ) -> xr.DataArray:
    """
    Fetch adjusted close prices and convert to log returns.

    Returns
    -------
    xr.DataArray
        dims: ("time", "asset")
    """

    prices = load_prices(tickers=tickers,
                         start=start,
                         end=end,
                         frequency=frequency,
                         is_factor=is_factor,
                         use_cache=use_cache,
                         )

    log_returns = np.log(prices / prices.shift(time=1))
    log_returns = log_returns.dropna('time')
    if not is_factor:
        log_returns.name = 'asset_returns'
    else:
        log_returns.name = 'factor_returns'

    return log_returns
