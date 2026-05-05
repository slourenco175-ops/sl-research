"""Weekly inventory levels for energy commodities, direct from EIA.

Inventories are the single most important fundamental for storables: low/falling
stocks → tightness → bullish, high/rising stocks → glut → bearish. The classic
COT + technicals stack ignores them entirely, which is the biggest hole in a
commodities model.

Source: EIA's public xls history files (no key required, free, public). FRED's
EIA petroleum mirror has been retired so we fetch the original xls files.

  WCESTUS1  — Weekly U.S. Ending Stocks of Crude Oil ex-SPR (k bbl)
  WGTSTUS1  — Weekly U.S. Ending Stocks of Total Gasoline (k bbl)
  WDISTUS1  — Weekly U.S. Ending Stocks of Distillate Fuel Oil (k bbl)
  NW2_EPG0_SWO_R48_BCF — Weekly Lower-48 NG Working Storage (Bcf)

For each series we compute:
  level         : latest absolute level (printed for context)
  seasonal_z    : how many σ today's level is above/below the historical
                  same-week average over the past 5 years (de-seasonalized)
  build_4w      : 4-week change in level
  build_4w_norm : 4-week change MINUS the 5y average 4-week change for this
                  same calendar window — a "build vs typical" measure
                  (positive = builds bigger than seasonal norm = bearish)
  weeks_ago     : staleness check — EIA usually prints Wed (oil) or Thu (NG).

A negative seasonal_z is bullish for the linked commodity.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

# key → (EIA xls URL, human label)
INVENTORY_SERIES = {
    "cl": (
        "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls",
        "U.S. crude oil ex-SPR",
    ),
    "rb": (
        "https://www.eia.gov/dnav/pet/hist_xls/WGTSTUS1w.xls",
        "U.S. total gasoline",
    ),
    "ho": (
        "https://www.eia.gov/dnav/pet/hist_xls/WDISTUS1w.xls",
        "U.S. distillate fuel",
    ),
    "ng": (
        "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls",
        "U.S. working nat gas (L48)",
    ),
}


@lru_cache(maxsize=16)
def _fetch_eia_xls(url: str) -> pd.Series:
    """EIA history xls layout: 'Data 1' sheet, with date col 0, value col 1,
    starting at row 3 (rows 0-2 are header / source key)."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", header=None)
    except Exception as e:
        print(f"  ! EIA inventory {url.rsplit('/',1)[-1]}: {e}")
        return pd.Series(dtype=float)

    # Drop the 3 header rows; col 0 = date, col 1 = value
    if len(df) < 4:
        return pd.Series(dtype=float)
    body = df.iloc[3:].copy()
    body.columns = ["date", "value"]
    body["date"] = pd.to_datetime(body["date"], errors="coerce")
    body["value"] = pd.to_numeric(body["value"], errors="coerce")
    return body.dropna().set_index("date")["value"].sort_index()


def _seasonal_z(level: pd.Series, years_back: int = 5) -> float:
    """How many σ above/below the same-week mean of the past N years."""
    if len(level) < 52:
        return float("nan")
    last_date = level.index[-1]
    last_val = float(level.iloc[-1])

    same_week = []
    for yr_offset in range(1, years_back + 1):
        start = last_date - pd.DateOffset(years=yr_offset) - pd.Timedelta(days=10)
        end = last_date - pd.DateOffset(years=yr_offset) + pd.Timedelta(days=10)
        window = level.loc[(level.index >= start) & (level.index <= end)]
        same_week.extend(window.tolist())

    same_week = [v for v in same_week if pd.notna(v)]
    if len(same_week) < 4:
        return float("nan")

    mu = float(np.mean(same_week))
    sd = float(np.std(same_week, ddof=0))
    if sd <= 0:
        return float("nan")
    return float((last_val - mu) / sd)


def _build_vs_norm(level: pd.Series, weeks: int = 4, years_back: int = 5) -> tuple[float, float]:
    """(latest N-week change, deviation from same-window 5y average change)."""
    if len(level) < weeks + 1:
        return float("nan"), float("nan")
    latest_change = float(level.iloc[-1] - level.iloc[-(weeks + 1)])
    last_date = level.index[-1]

    historical = []
    for yr_offset in range(1, years_back + 1):
        anchor = last_date - pd.DateOffset(years=yr_offset)
        idx = level.index.get_indexer([anchor], method="nearest")
        if len(idx) == 0 or idx[0] < weeks:
            continue
        i = int(idx[0])
        historical.append(float(level.iloc[i] - level.iloc[i - weeks]))
    if len(historical) < 3:
        return latest_change, float("nan")
    norm = float(np.mean(historical))
    return latest_change, latest_change - norm


def get_inventories() -> dict:
    """{key: {label, level, level_date, weeks_ago, seasonal_z, build_4w, build_4w_norm}}.

    Returns an empty dict on total network failure; partial results otherwise.
    """
    out = {}
    today = pd.Timestamp.utcnow().tz_localize(None)
    for key, (url, label) in INVENTORY_SERIES.items():
        s = _fetch_eia_xls(url)
        if s.empty:
            continue
        last_date = s.index[-1]
        weeks_ago = max(0, (today - last_date).days // 7)
        sz = _seasonal_z(s)
        b4, b4_norm = _build_vs_norm(s)
        out[key] = {
            "label": label,
            "level": float(s.iloc[-1]),
            "level_date": last_date.strftime("%Y-%m-%d"),
            "weeks_ago": int(weeks_ago),
            "seasonal_z": round(sz, 2) if pd.notna(sz) else None,
            "build_4w": round(b4, 0) if pd.notna(b4) else None,
            "build_4w_norm": round(b4_norm, 0) if pd.notna(b4_norm) else None,
        }
    return out
