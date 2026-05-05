"""Futures curve module — pulls deferred-month contracts via yfinance.

Replaces the heuristic "spot vs 252d avg" carry proxy with a real annualized
roll yield computed from the actual M1 → M_n forward curve.

Coverage notes:
  Energy (CL, BZ, NG, HO, RB) trade all 12 months — full curve usually available.
  Metals (GC, SI, HG, PL, PA) and grains (ZC, ZS, ZW, ZL, ZM) trade fewer
  active months; we try a comprehensive set and use whatever yfinance returns.
  Softs (KC, SB, CC, CT) and livestock (LE) — same.

For tickers/months yfinance can't resolve we silently get an empty frame.
We require ≥ 2 contracts to compute a slope; below that, the function
returns a graceful empty dict and the scorer falls back to the v1 carry proxy.

Symbol format used by yfinance:
  <ROOT><MonthCode><YY>.<Exchange>
  Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
  Exchanges:   .NYM (NYMEX/ICE energy), .CMX (COMEX metals),
               .CBT (CBOT grains), .NYB (ICE softs)
"""
from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

# yfinance prints "possibly delisted" stderr warnings for every missing month
# code in our scan. We expect a lot of misses (deferred-month coverage is
# sparse). Silence them so the run logs stay readable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


# Per-root metadata: (yfinance exchange suffix, list of typical active month codes)
# When in doubt we list all months — yfinance just returns empty for ones that don't trade.
_CURVE_META = {
    # energy: full strip available
    "CL": (".NYM", MONTH_CODES),
    "BZ": (".NYM", MONTH_CODES),
    "NG": (".NYM", MONTH_CODES),
    "HO": (".NYM", MONTH_CODES),
    "RB": (".NYM", MONTH_CODES),
    # COMEX metals — gold/silver/copper/platinum/palladium have distinct cycles
    "GC": (".CMX", ["G", "J", "M", "Q", "V", "Z"]),
    "SI": (".CMX", ["F", "H", "K", "N", "U", "Z"]),
    "HG": (".CMX", ["F", "H", "K", "N", "U", "Z"]),
    "PL": (".NYM", ["F", "J", "N", "V"]),
    "PA": (".NYM", ["H", "M", "U", "Z"]),
    # CBOT grains — distinct cycles per crop
    "ZC": (".CBT", ["H", "K", "N", "U", "Z"]),
    "ZS": (".CBT", ["F", "H", "K", "N", "Q", "U", "X"]),
    "ZW": (".CBT", ["H", "K", "N", "U", "Z"]),
    "ZL": (".CBT", ["F", "H", "K", "N", "Q", "U", "V", "Z"]),
    "ZM": (".CBT", ["F", "H", "K", "N", "Q", "U", "V", "Z"]),
    # ICE softs
    "KC": (".NYB", ["H", "K", "N", "U", "Z"]),
    "SB": (".NYB", ["H", "K", "N", "V"]),
    "CC": (".NYB", ["H", "K", "N", "U", "Z"]),
    "CT": (".NYB", ["H", "K", "N", "V", "Z"]),
    # CME livestock
    "LE": (".CME", ["G", "J", "M", "Q", "V", "Z"]),
}


def _expiry_from_code(code: str, yr_full: int) -> pd.Timestamp:
    """Approximate expiry as last day of the contract month (good enough for slope math)."""
    month = MONTH_CODES.index(code) + 1
    return pd.Timestamp(year=yr_full, month=month, day=28)  # 28th avoids month-overflow edge cases


@lru_cache(maxsize=128)
def _fetch_one(symbol: str) -> float:
    """Latest close for a yfinance symbol, or NaN if not available."""
    try:
        hist = yf.Ticker(symbol).history(period="10d", auto_adjust=True)["Close"].dropna()
        if hist.empty:
            return float("nan")
        return float(hist.iloc[-1])
    except Exception:
        return float("nan")


def fetch_curve(root: str, n_months_ahead: int = 14) -> list[dict]:
    """Try to pull up to N future month contracts. Returns a list of
    {symbol, code, year, expiry, price, months_ahead}, sorted by expiry.

    months_ahead is a continuous fractional-months figure used for slope math.
    """
    if root not in _CURVE_META:
        return []
    suffix, months = _CURVE_META[root]
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    out = []
    yr = today.year
    seen = set()
    # walk forward until we cover ~n_months_ahead months, scanning across years
    for yr_offset in (0, 1, 2):
        for code in months:
            year = yr + yr_offset
            yy = str(year)[-2:]
            sym = f"{root}{code}{yy}{suffix}"
            if sym in seen:
                continue
            seen.add(sym)
            expiry = _expiry_from_code(code, year)
            if expiry < today:
                continue
            months_ahead = (expiry - today).days / 30.4375
            if months_ahead > n_months_ahead:
                continue
            price = _fetch_one(sym)
            if pd.notna(price) and price > 0:
                out.append({
                    "symbol": sym,
                    "code": code,
                    "year": year,
                    "expiry": expiry,
                    "price": price,
                    "months_ahead": float(months_ahead),
                })

    out.sort(key=lambda d: d["expiry"])
    return out


def curve_slope(curve: list[dict]) -> dict:
    """Annualized roll yield from a list of contracts.

    Uses a log-linear OLS on (price ~ months_ahead). The slope is the implied
    monthly forward log-return; multiply by 12 to annualize. Negative slope =
    backwardation (bullish), positive = contango (bearish drag for longs).

    Also reports M1→M2 and M1→M6 spreads as %.
    """
    if not curve or len(curve) < 2:
        return {}
    arr = np.array([(c["months_ahead"], c["price"]) for c in curve], dtype=float)
    t, p = arr[:, 0], arr[:, 1]
    if (p <= 0).any():
        return {}
    log_p = np.log(p)
    t_mean = t.mean()
    var_t = ((t - t_mean) ** 2).sum()
    if var_t <= 0:
        return {}
    b = ((t - t_mean) * (log_p - log_p.mean())).sum() / var_t  # log-return per month
    slope_pct_yr = float(b * 12 * 100)

    # M1 / M2 / M6 (closest to those expiries)
    m1 = curve[0]
    m2 = curve[1] if len(curve) >= 2 else None
    m6 = next((c for c in curve if c["months_ahead"] >= 5.5), None) or curve[-1]

    spread_m1_m2 = float((m2["price"] - m1["price"]) / m1["price"] * 100) if m2 else float("nan")
    spread_m1_m6 = float((m6["price"] - m1["price"]) / m1["price"] * 100) if m6 != m1 else float("nan")

    if slope_pct_yr < -2:
        state = "BACKWARDATION"
    elif slope_pct_yr > 2:
        state = "CONTANGO"
    else:
        state = "FLAT"

    return {
        "n_months": len(curve),
        "front_symbol": m1["symbol"],
        "front_price": m1["price"],
        "deferred_months_ahead": curve[-1]["months_ahead"],
        "slope_pct_yr": round(slope_pct_yr, 2),
        "spread_m1_m2_pct": round(spread_m1_m2, 2) if pd.notna(spread_m1_m2) else None,
        "spread_m1_m6_pct": round(spread_m1_m6, 2) if pd.notna(spread_m1_m6) else None,
        "state": state,
    }


def get_curves(roots: list[str], n_months_ahead: int = 14) -> dict[str, dict]:
    """Build curve dict for a list of contract roots. Empty dict per failed root."""
    out = {}
    for root in roots:
        try:
            curve = fetch_curve(root, n_months_ahead=n_months_ahead)
            slope = curve_slope(curve)
        except Exception as e:
            print(f"  ! curve {root}: {e}")
            slope = {}
        out[root] = slope
    return out
