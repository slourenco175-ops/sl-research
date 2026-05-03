"""Per-asset factor + signal computation for the commodities model.

v2 outputs (in addition to v1's mom/trend/composite/regime):
  RSI(14), MACD label, 20d Donchian breakout, ATR(14), stops L/S,
  ST/MT/LT trend bucket (UP/DOWN/FLAT vs 20d/50d/200d),
  vol_20d / vol_60d annualized, vol percentile (3y), vol regime label,
  CTA proxy (-100..+100) + 1m/3m/12m breakdown,
  carry proxy (clearly labeled — replace with real F_near/F_far in prod),
  seasonality for current calendar month (avg + hit rate over available history).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


# -------------------- helpers --------------------

def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.std(ddof=0) == 0 or s.dropna().empty:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0


def _macd_label(close: pd.Series) -> str:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = (macd - signal).dropna()
    if len(hist) < 5:
        return "FLAT"
    last, prev3 = hist.iloc[-1], hist.iloc[-4]
    if last > 0 and prev3 <= 0:
        return "BULL CROSS"
    if last < 0 and prev3 >= 0:
        return "BEAR CROSS"
    return "BULL" if last > 0 else "BEAR"


def _breakout(close: pd.Series, lookback: int = 20) -> str:
    if len(close) < lookback + 2:
        return "INSIDE"
    window = close.iloc[-(lookback + 1):-1]
    last = float(close.iloc[-1])
    if last >= float(window.max()):
        return "BREAK UP"
    if last <= float(window.min()):
        return "BREAK DOWN"
    return "INSIDE"


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if not {"High", "Low", "Close"}.issubset(df.columns) or len(df) < period + 1:
        return float("nan")
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def _trend_bucket(price: float, ma: float, tol_pct: float = 1.0) -> str:
    if ma <= 0 or pd.isna(ma):
        return "FLAT"
    diff = (price / ma - 1) * 100
    if diff > tol_pct:
        return "UP"
    if diff < -tol_pct:
        return "DOWN"
    return "FLAT"


def _vol_regime(pctile: float) -> str:
    if pd.isna(pctile):
        return "NORMAL"
    if pctile >= 90:
        return "EXTREME"
    if pctile >= 65:
        return "HIGH"
    if pctile <= 25:
        return "LOW"
    return "NORMAL"


def _seasonality(close: pd.Series, month: int) -> dict:
    """Average return and hit rate for the given calendar month over history."""
    monthly = close.resample("ME").last().pct_change().dropna()
    same_month = monthly[monthly.index.month == month]
    if same_month.empty:
        return {"avg_pct": 0.0, "hit_rate": 0.5, "n_years": 0}
    return {
        "avg_pct": float(same_month.mean() * 100),
        "hit_rate": float((same_month > 0).mean()),
        "n_years": int(len(same_month)),
    }


def _vol_pctile(close: pd.Series, window: int = 20, lookback: int = 756) -> tuple[float, float, float]:
    """Return (vol_20d_ann_pct, vol_60d_ann_pct, vol_pctile_3y)."""
    rets = close.pct_change().dropna()
    if len(rets) < window:
        return float("nan"), float("nan"), float("nan")
    rolling = rets.rolling(window).std() * np.sqrt(252) * 100
    vol_20 = float(rolling.iloc[-1])
    vol_60 = float((rets.rolling(60).std() * np.sqrt(252) * 100).iloc[-1]) if len(rets) >= 60 else vol_20
    recent = rolling.dropna().iloc[-lookback:]
    pct = float(recent.rank(pct=True).iloc[-1] * 100) if not recent.empty else float("nan")
    return vol_20, vol_60, pct


def _cta_score(close: pd.Series) -> dict:
    """Trend-following CTA proxy. Vol-scaled trend at 1m / 3m / 12m, mapped to -100..+100.

    A composite of multi-horizon trends — what most CTA replication papers use as
    the position signal.
    """
    rets = close.pct_change().dropna()
    if len(rets) < 252:
        return {"score": 0, "z_1m": 0, "z_3m": 0, "z_12m": 0}

    def tr(n):
        if len(close) <= n:
            return 0.0
        ret = close.iloc[-1] / close.iloc[-n] - 1
        vol = rets.iloc[-n:].std() * np.sqrt(252)
        if vol <= 0:
            return 0.0
        # vol-scaled trend, clipped to [-3, 3], mapped to [-100, +100]
        z = max(-3.0, min(3.0, ret / vol))
        return z * 100 / 3

    s1, s3, s12 = tr(21), tr(63), tr(252)
    composite = (s1 + s3 + s12) / 3
    return {
        "score": round(composite),
        "z_1m": round(s1),
        "z_3m": round(s3),
        "z_12m": round(s12),
    }


def _cta_label(score: float) -> str:
    if score >= 60:
        return "VERY LONG"
    if score >= 20:
        return "LONG"
    if score <= -60:
        return "VERY SHORT"
    if score <= -20:
        return "SHORT"
    return "FLAT"


def _carry_proxy(close: pd.Series) -> dict:
    """Heuristic carry/curve proxy — NOT a real roll yield.

    Replace with (F_near - F_far) / F_near * (12 / months_apart) once a real
    futures curve feed is wired in (Barchart, Nasdaq Data Link, EODHD).

    Heuristic used: spot vs 252d trailing average — strong premium often
    coincides with backwardated curves (scarcity). Annualized.
    """
    if len(close) < 252:
        return {"state": "FLAT", "value_pct": 0.0, "is_proxy": True}
    avg_252 = float(close.iloc[-252:].mean())
    last = float(close.iloc[-1])
    if avg_252 <= 0:
        return {"state": "FLAT", "value_pct": 0.0, "is_proxy": True}
    val = (last / avg_252 - 1) * 100
    state = "BACKWARDATION" if val > 5 else ("CONTANGO" if val < -5 else "FLAT")
    return {"state": state, "value_pct": round(val, 1), "is_proxy": True}


# -------------------- main --------------------

def compute_per_asset(ohlc: pd.DataFrame) -> dict:
    """All single-asset factors + indicators. Expects a DataFrame with OHLC."""
    if ohlc is None or ohlc.empty or "Close" not in ohlc.columns:
        return {}
    p = ohlc["Close"].dropna()
    if len(p) < 252:
        return {}

    last = float(p.iloc[-1])

    # momentum
    mom_12_1 = float(p.iloc[-21] / p.iloc[-252] - 1) if len(p) >= 252 + 21 else float(p.iloc[-1] / p.iloc[-252] - 1)
    mom_3m = float(p.iloc[-1] / p.iloc[-63] - 1) if len(p) > 63 else 0.0
    mom_1m = float(p.iloc[-1] / p.iloc[-21] - 1) if len(p) > 21 else 0.0

    # MAs + multi-horizon trend buckets (ST=20d, MT=50d, LT=200d)
    ma20 = float(p.iloc[-20:].mean())
    ma50 = float(p.iloc[-50:].mean())
    ma200 = float(p.iloc[-200:].mean())
    trend_st = _trend_bucket(last, ma20)
    trend_mt = _trend_bucket(last, ma50)
    trend_lt = _trend_bucket(last, ma200)

    trend_vs_50d = (last / ma50 - 1) if ma50 > 0 else 0.0
    trend_vs_200d = (last / ma200 - 1) if ma200 > 0 else 0.0
    slope_50 = (ma50 / float(p.iloc[-50:-25].mean()) - 1) if len(p) >= 50 else 0.0

    # vol
    vol_20, vol_60, vol_pctile = _vol_pctile(p)
    vol_regime = _vol_regime(vol_pctile)

    # indicators
    rsi = _rsi(p)
    macd = _macd_label(p)
    bo = _breakout(p)

    # ATR + stops
    atr = _atr(ohlc)
    stop_dist = atr * 2 if pd.notna(atr) else last * 0.02
    long_stop = round(last - stop_dist, 4)
    short_stop = round(last + stop_dist, 4)

    # CTA proxy
    cta = _cta_score(p)

    # Carry proxy
    carry = _carry_proxy(p)

    # Seasonality (current month)
    seas = _seasonality(p, month=datetime.utcnow().month)

    return {
        "price": round(last, 4),
        "mom_12_1": mom_12_1, "mom_3m": mom_3m, "mom_1m": mom_1m,
        "trend_vs_200d": trend_vs_200d, "trend_vs_50d": trend_vs_50d,
        "slope_50d": slope_50, "ma200": ma200, "ma50": ma50, "ma20": ma20,
        "trend_st": trend_st, "trend_mt": trend_mt, "trend_lt": trend_lt,
        "vol_20d_ann": vol_20 / 100 if pd.notna(vol_20) else float("nan"),
        "vol_60d_ann_pct": vol_60,
        "vol_20d_ann_pct": vol_20,
        "vol_pctile_3y": vol_pctile,
        "vol_regime": vol_regime,
        "rsi_14": round(rsi, 1),
        "macd": macd,
        "breakout_20d": bo,
        "atr_14": atr,
        "stop_dist": stop_dist,
        "long_stop": long_stop,
        "short_stop": short_stop,
        "cta_score": cta["score"],
        "cta_label": _cta_label(cta["score"]),
        "cta_z_1m": cta["z_1m"],
        "cta_z_3m": cta["z_3m"],
        "cta_z_12m": cta["z_12m"],
        "carry_state": carry["state"],
        "carry_proxy_pct": carry["value_pct"],
        "carry_is_proxy": carry["is_proxy"],
        "seas_avg_pct": round(seas["avg_pct"], 2),
        "seas_hit_rate": seas["hit_rate"],
        "seas_n_years": seas["n_years"],
    }


def build_factor_table(ohlc: dict[str, pd.DataFrame], universe: list[dict]) -> pd.DataFrame:
    """Per-asset factor rows + cross-sectional composite z + bucket + regime."""
    rows = []
    for u in universe:
        sym = u["yf"]
        if sym not in ohlc:
            continue
        f = compute_per_asset(ohlc[sym])
        if not f:
            continue
        spark = ohlc[sym]["Close"].dropna().iloc[-120:].tolist()
        rows.append({**u, **f, "spark": spark})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cross-sectional composite (kept identical to v1 so v1 dashboard still works)
    df["z_mom_12_1"] = _zscore(df["mom_12_1"])
    df["z_mom_1m"] = _zscore(df["mom_1m"])
    df["z_trend"] = _zscore(df["trend_vs_200d"])
    df["composite_z"] = df["z_mom_12_1"] - 0.3 * df["z_mom_1m"] + 0.5 * df["z_trend"]

    n = len(df)
    cutoff = max(1, n // 3)
    ranked = df["composite_z"].rank(ascending=False, method="first")
    df["signal"] = "FLAT"
    df.loc[ranked <= cutoff, "signal"] = "LONG"
    df.loc[ranked > n - cutoff, "signal"] = "SHORT"

    def regime(row):
        if row["trend_vs_200d"] > 0 and row["composite_z"] > 0.3:
            return "BULL"
        if row["trend_vs_200d"] < 0 and row["composite_z"] < -0.3:
            return "BEAR"
        return "NEUTRAL"

    df["regime"] = df.apply(regime, axis=1)

    return df.sort_values("composite_z", ascending=False).reset_index(drop=True)
