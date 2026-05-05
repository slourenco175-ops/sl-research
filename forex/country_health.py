"""Country-level fundamentals snapshot per currency.

Pulls from FRED (free, no API key) the series defined in `universe_fx.CURRENCIES`:
  y10, policy, unemp, cpi (level), gdp (level)

Computes:
  y10            : latest 10y sovereign yield (%)
  policy         : latest policy-rate proxy (%)
  real_rate      : y10 - cpi_yoy (%)
  unemp          : latest unemployment rate (%)
  unemp_6m_chg   : change in unemp over the last 6 months (pp)
  cpi_yoy        : CPI YoY (%)
  cpi_6m_chg     : change in YoY over 6m (pp) — disinflation tracker
  gdp_yoy        : real GDP YoY (%)
  health_score   : -100..+100 composite (higher = stronger fundamentals)

Note: many series are monthly or quarterly with publication lag (1-3 months).
That's expected — FX fundamentals move slowly; this isn't an intraday tool.
"""
from __future__ import annotations

import pandas as pd

from forex.fx_data import fetch_fred
from forex.universe_fx import CURRENCIES


def _yoy(series: pd.Series) -> float:
    """Year-over-year % change of the latest observation, given a level series."""
    if len(series) < 13:
        return float("nan")
    last = series.iloc[-1]
    yr_ago = series.iloc[-13]
    if yr_ago == 0 or pd.isna(yr_ago):
        return float("nan")
    return float(last / yr_ago - 1) * 100


def _months_back(series: pd.Series, months: int) -> float | None:
    """Value `months` observations ago (assumes monthly cadence)."""
    if len(series) <= months:
        return None
    return float(series.iloc[-1 - months])


def _compute_one(ccy: str, meta: dict) -> dict:
    fred = meta.get("fred", {})

    y10_s    = fetch_fred(fred["y10"])    if "y10"    in fred else pd.Series(dtype=float)
    policy_s = fetch_fred(fred["policy"]) if "policy" in fred else pd.Series(dtype=float)
    unemp_s  = fetch_fred(fred["unemp"])  if "unemp"  in fred else pd.Series(dtype=float)
    cpi_s    = fetch_fred(fred["cpi"])    if "cpi"    in fred else pd.Series(dtype=float)
    gdp_s    = fetch_fred(fred["gdp"])    if "gdp"    in fred else pd.Series(dtype=float)

    y10    = float(y10_s.iloc[-1])    if not y10_s.empty    else float("nan")
    policy = float(policy_s.iloc[-1]) if not policy_s.empty else float("nan")
    unemp  = float(unemp_s.iloc[-1])  if not unemp_s.empty  else float("nan")

    cpi_yoy = _yoy(cpi_s) if not cpi_s.empty else float("nan")
    gdp_yoy = _yoy(gdp_s) if not gdp_s.empty else float("nan")

    real_rate = y10 - cpi_yoy if pd.notna(y10) and pd.notna(cpi_yoy) else float("nan")

    unemp_6m_prev = _months_back(unemp_s, 6)
    unemp_6m_chg = (unemp - unemp_6m_prev) if (unemp_6m_prev is not None and pd.notna(unemp)) else float("nan")

    cpi_6m_chg = float("nan")
    if len(cpi_s) >= 13 + 6:
        cpi_yoy_6m_ago = float(cpi_s.iloc[-7] / cpi_s.iloc[-19] - 1) * 100 if cpi_s.iloc[-19] else float("nan")
        if pd.notna(cpi_yoy_6m_ago):
            cpi_6m_chg = cpi_yoy - cpi_yoy_6m_ago

    return {
        "ccy": ccy,
        "country": meta.get("country", ccy),
        "y10": round(y10, 2) if pd.notna(y10) else float("nan"),
        "policy": round(policy, 2) if pd.notna(policy) else float("nan"),
        "real_rate": round(real_rate, 2) if pd.notna(real_rate) else float("nan"),
        "unemp": round(unemp, 2) if pd.notna(unemp) else float("nan"),
        "unemp_6m_chg": round(unemp_6m_chg, 2) if pd.notna(unemp_6m_chg) else float("nan"),
        "cpi_yoy": round(cpi_yoy, 2) if pd.notna(cpi_yoy) else float("nan"),
        "cpi_6m_chg": round(cpi_6m_chg, 2) if pd.notna(cpi_6m_chg) else float("nan"),
        "gdp_yoy": round(gdp_yoy, 2) if pd.notna(gdp_yoy) else float("nan"),
    }


def _health_score(row: dict) -> float:
    """Composite -100..+100. Each axis contributes ±25.

      + real rate (high real rate attracts capital)
      + growth (GDP YoY)
      - unemployment trend (rising = weak)
      - inflation overshoot (high CPI YoY hurts purchasing power once policy lags)
    """
    s = 0.0
    n = 0

    rr = row.get("real_rate")
    if pd.notna(rr):
        s += max(-25, min(25, rr * 12.5)); n += 1   # ±2% real rate ≈ ±25

    g = row.get("gdp_yoy")
    if pd.notna(g):
        s += max(-25, min(25, g * 8)); n += 1        # ±3% growth ≈ ±24

    du = row.get("unemp_6m_chg")
    if pd.notna(du):
        s += max(-25, min(25, -du * 25)); n += 1     # +1pp unemp = -25

    cpi = row.get("cpi_yoy")
    if pd.notna(cpi):
        # Sweet spot ~2%. Both deflation and >5% are bad.
        gap = abs(cpi - 2.0)
        s += max(-25, min(25, 25 - gap * 8)); n += 1

    return round(s, 1) if n else 0.0


def build_health_table() -> pd.DataFrame:
    rows = []
    for ccy, meta in CURRENCIES.items():
        try:
            r = _compute_one(ccy, meta)
        except Exception as e:
            print(f"  ! country health {ccy}: {e}")
            continue
        r["health_score"] = _health_score(r)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("ccy")
    return df
