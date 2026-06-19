"""Stage 0 — typed data access layer.

Thin wrapper over the existing free-source helpers in `forex.fx_data`
(FRED CSV + yfinance + CFTC). Everything downstream pulls through here so
the universe and macro series are fetched once and cached in-process.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from forex.fx_data import fetch_fred, get_ohlc
from macro_engine.config import AUX_SYMBOLS, UNIVERSE


@lru_cache(maxsize=1)
def get_universe_prices(lookback_days: int = 9000) -> dict[str, pd.DataFrame]:
    """{symbol -> OHLCV} for the ranked universe + auxiliary series.

    ~25y of daily bars (9000d) so Stage 2 can label decades of returns by
    regime. ETFs younger than that simply return what history they have.
    """
    symbols = [u["sym"] for u in UNIVERSE] + AUX_SYMBOLS
    return get_ohlc(symbols, lookback_days=lookback_days)


def close_frame(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide close-price frame (columns = symbols), forward-filled, date index."""
    cols = {}
    for sym, df in prices.items():
        if df is not None and not df.empty and "Close" in df:
            cols[sym] = df["Close"]
    out = pd.DataFrame(cols).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def daily_returns(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple returns for every symbol."""
    return close_frame(prices).pct_change(fill_method=None)


@lru_cache(maxsize=32)
def fred_series(series_id: str) -> pd.Series:
    """Single FRED series (cached). Index tz-naive datetime."""
    s = fetch_fred(series_id)
    if not s.empty:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    return s
