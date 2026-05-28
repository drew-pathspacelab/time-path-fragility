from dataclasses import dataclass
from typing import List, Dict
import xarray as xr
import pandas as pd
import numpy as np


"""
Boilerplate dataclass auto generates dunder methods and structure.
TODO: Delete this comment, I keep forgetting what it is doing.
"""
@dataclass(frozen=True)
class Universe:
    assets: List[str]
    sector_map: Dict[str, str]
    market: List[str]


def load_universe() -> Universe:
    """
    Canonical asset universe for Artifact 01.
    Fixed and versioned.
    """
    assets = [
        "AAPL", "MSFT", "NVDA", "META",
        "JPM", "CAT", "XOM",
        "JNJ", "PG", "KO",
        "AMT",
    ]

    # Sector ETF tickers (SPDR)
    sector_map = {
        "AAPL": "XLK",
        "MSFT": "XLK",
        "NVDA": "XLK",
        "META": "XLK",

        "JPM": "XLF",
        "CAT": "XLI",
        "XOM": "XLE",

        "JNJ": "XLV",
        "PG": "XLP",
        "KO": "XLP",

        "AMT": "XLRE",
    }

    market = [
                "XLB",
                "XLE",
                "XLF",
                "XLI",
                "XLK",
                "XLP",
                "XLU",
                "XLV",
                "XLY",
                "XLRE",
    ]

    return Universe(assets=assets, sector_map=sector_map, market=market)


def load_asset_returns(start: str,
                       end: str,
                       assets: list[str] | None = None,
                       source: str = 'mock',
                       ) -> xr.Dataset:

    if source=='mock':
        # NOTE: freq="B" ≠ US trading days; real loaders must supply actual calendars
        time = pd.date_range(start=start, end=end, freq='B')
        data = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=(len(time), len(assets))
        )
    else:
        raise NotImplementedError(f"Source {source} not implemented.")

    ds = xr.Dataset(
        data_vars={
            'returns': (('time', 'asset'), data)
        },
        coords={
            'time':time,
            'asset': assets,
        },
        attrs={
            'source': source,
            'frequency': "D"
        }
    )

    return ds

def load_factor_returns (start: str,
                         end: str,
                         factors: list[str] | None = None,
                         source: str ="mock",
                        ) -> xr.Dataset:

    if source == "mock":
        time = pd.date_range(start=start, end=end, freq="B")
        data = np.random.normal(
            loc=0.0,
            scale=0.008,
            size=(len(time), len(factors)),
        )
    else:
        raise NotImplementedError(f"Source {source} not implemented")

    ds = xr.Dataset(
        data_vars={
            "factor_returns": (("time", "factor"), data)
        },
        coords={
            "time": time,
            "factor": factors,
        },
        attrs={
            "source": source,
            "frequency": "D",
        },
    )

    return ds
