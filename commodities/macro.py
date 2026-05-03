"""Macro backdrop block: DXY (USD), US 10y yield, VIX.

Pulled from yfinance; very small download. The dashboard uses this as a
context bar at the top — falling DXY is a tailwind for commodities priced
in USD; high VIX often coincides with risk-off / commodity drawdowns.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",   # quoted as percent * 1, e.g. 4.30 = 4.30%
    "VIX": "^VIX",
}


def get_macro() -> dict:
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=1100)  # ~3y for VIX percentile

    out = {}
    for label, sym in MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(sym).history(start=start, end=end, auto_adjust=True)["Close"].dropna()
            if hist.empty:
                continue
            last = float(hist.iloc[-1])
            chg_1m = float(hist.iloc[-1] / hist.iloc[-21] - 1) * 100 if len(hist) >= 21 else 0.0
            pctile = float(hist.rank(pct=True).iloc[-1] * 100)
            out[label] = {
                "last": last,
                "chg_1m_pct": chg_1m,
                "pctile_3y": pctile,
                "direction": "DOWN" if chg_1m < -0.5 else ("UP" if chg_1m > 0.5 else "FLAT"),
            }
        except Exception as e:
            print(f"  ! macro {label}: {e}")
    return out
