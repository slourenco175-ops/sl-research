"""Macro backdrop for FX: DXY, US10Y, US 2y real, oil, gold, copper, VIX.

These are the cross-asset levers that move major FX pairs:
  DXY    — broad USD direction
  US10Y  — global risk-free rate
  US2YR  — US real-rate proxy (front-end)
  CL=F   — WTI: tailwind for CAD/NOK, headwind for JPY/EUR (importers)
  GC=F   — Gold: safe haven; correlates with CHF, sometimes AUD
  HG=F   — Copper: industrial cycle proxy, drives AUD/CLP
  VIX    — risk-on/risk-off; high VIX → JPY/CHF/USD bid
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

MACRO_TICKERS = {
    "DXY":    "DX-Y.NYB",
    "US10Y":  "^TNX",   # quoted as percent * 1, e.g. 4.30 = 4.30%
    "US2YR":  "^FVX",   # 5y as a 2y proxy (yfinance lacks ^IRX as 2y); approximate front-end
    "OIL":    "CL=F",
    "GOLD":   "GC=F",
    "COPPER": "HG=F",
    "VIX":    "^VIX",
}


def get_macro() -> dict:
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=1100)  # ~3y for percentiles

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
