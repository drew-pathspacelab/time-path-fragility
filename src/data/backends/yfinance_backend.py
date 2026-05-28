from typing import List
import yfinance as yf
import pandas as pd

BACKEND_METADATA = {"source": "yfinance",
                    "provider": "Yahoo Finance",
                    "price_field": "adjusted_close",
                    "currency": "USD",
                    }


def fetch_adjusted_prices(tickers: List[str],
                          start: str,
                          end: str,
                          frequency: str = "1d",
                          ) -> pd.DataFrame:
    """
    Fetch adjusted daily closing prices for a set of assets.

    Parameters
    ----------
    tickers : list[str]
        List of asset tickers to download (e.g. ["AAPL", "MSFT", "XOM"]).

    start : str or datetime-like
        Start date for the price history.

    end : str or datetime-like
        End date for the price history.

    Returns
    -------
    pandas.DataFrame
        DataFrame of adjusted closing prices with

        index:
            date (trading days)

        columns:
            asset tickers

        shape:
            (n_dates, n_assets)

    Notes
    -----
    The returned data is *raw provider output* and has not yet been
    converted into the project's canonical xarray schema. Normalization
    occurs in the data interface layer before caching.

    Missing data (e.g. IPO dates or halted trading) may appear as NaN.
    """

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=frequency,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices.sort_index(inplace=True)

    return prices
