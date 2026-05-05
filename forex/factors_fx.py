"""Per-pair factor + signal computation for the forex monitor.

Builds a row per pair containing:
  - technicals on the pair price (mom, trend, RSI, MACD, breakout, ATR, vol)
  - CTA proxy (vol-scaled multi-horizon trend)
  - seasonality (avg + hit rate for the current calendar month)
  - COT positioning (Leveraged Money) on the BASE currency futures
  - macro deltas (carry, real-rate, growth, unemployment, commodity overlay)
    computed as base - quote so a positive delta favors the base ccy

Most of the technical primitives are duplicated from `commodities/factors.py`
on purpose — keeping forex self-contained means changes to one model don't
silently affect the other.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from forex.universe_fx import CURRENCIES


# -------------------- technical helpers --------------------

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


def _trend_bucket(price: float, ma: float, tol_pct: float = 0.5) -> str:
    """Tighter tolerance than commodities (FX moves are smaller)."""
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


def _vol_pctile(close: pd.Series, window: int = 20, lookback: int = 756) -> tuple[float, float, float]:
    rets = close.pct_change().dropna()
    if len(rets) < window:
        return float("nan"), float("nan"), float("nan")
    rolling = rets.rolling(window).std() * np.sqrt(252) * 100
    vol_20 = float(rolling.iloc[-1])
    vol_60 = float((rets.rolling(60).std() * np.sqrt(252) * 100).iloc[-1]) if len(rets) >= 60 else vol_20
    recent = rolling.dropna().iloc[-lookback:]
    pct = float(recent.rank(pct=True).iloc[-1] * 100) if not recent.empty else float("nan")
    return vol_20, vol_60, pct


def _vol_ratio_5y(close: pd.Series, window: int = 60, lookback: int = 1260) -> float:
    """Current 60d ann vol / 5y median 60d ann vol — used to damp aggregate scores."""
    rets = close.pct_change().dropna()
    if len(rets) < window:
        return 1.0
    rolling = rets.rolling(window).std() * np.sqrt(252)
    series = rolling.dropna().iloc[-lookback:]
    if series.empty or series.median() <= 0:
        return 1.0
    return float(series.iloc[-1] / series.median())


def _vol_scaled_return(close: pd.Series, n_days: int) -> float:
    """Return over n_days / asset's realized vol over the same window. Clipped to ±3."""
    if len(close) <= n_days + 1:
        return 0.0
    rets = close.pct_change().dropna()
    if len(rets) < n_days:
        return 0.0
    ret = float(close.iloc[-1] / close.iloc[-n_days] - 1)
    sigma_window = float(rets.iloc[-n_days:].std()) * np.sqrt(n_days)
    if sigma_window <= 0:
        return 0.0
    return float(np.clip(ret / sigma_window, -3.0, 3.0))


def _cta_score(close: pd.Series) -> dict:
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
        z = max(-3.0, min(3.0, ret / vol))
        return z * 100 / 3

    s1, s3, s12 = tr(21), tr(63), tr(252)
    composite = (s1 + s3 + s12) / 3
    return {"score": round(composite), "z_1m": round(s1), "z_3m": round(s3), "z_12m": round(s12)}


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


def _regression_analysis(close: pd.Series, lookback: int = 1260) -> dict:
    """5y OLS regression of price on time + residual statistics + inline SVG chart.

    A flat-mean valuation anchor is a poor fit when a pair has been trending
    for years (USDJPY, USDBRL). A linear regression captures the trend, then
    measures how far spot is from that trend in residual standard deviations.

    Returns:
      z              : (last_price − fitted_last) / σ_residual; >0 = spot above trend
      r2             : goodness-of-fit of the trend line
      slope_pct_yr   : annualized drift implied by the trend (% of mean / yr)
      sigma_resid    : residual standard deviation (price units)
      fitted_last    : the model's "fair value" today
      svg            : embeddable SVG showing price + trend + ±1/±2σ bands
    """
    s = close.dropna().iloc[-lookback:]
    if len(s) < 252:
        return {}

    n = len(s)
    t = np.arange(n, dtype=float)
    y = s.values.astype(float)

    t_mean = t.mean()
    y_mean = y.mean()
    var_t = ((t - t_mean) ** 2).sum()
    if var_t <= 0:
        return {}
    b = ((t - t_mean) * (y - y_mean)).sum() / var_t
    a = y_mean - b * t_mean
    fitted = a + b * t
    resid = y - fitted
    sigma = float(resid.std(ddof=0))
    if sigma <= 0:
        return {}

    z = float((y[-1] - fitted[-1]) / sigma)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    slope_pct_yr = float(b * 252 / fitted.mean() * 100)  # %/yr drift of fair value

    svg = _regression_svg(t, y, fitted, sigma)

    return {
        "z": round(z, 2),
        "r2": round(r2, 3),
        "slope_pct_yr": round(slope_pct_yr, 2),
        "sigma_resid": sigma,
        "fitted_last": float(fitted[-1]),
        "svg": svg,
    }


def _regression_svg(t: np.ndarray, y: np.ndarray, fitted: np.ndarray, sigma: float) -> str:
    """Inline SVG: price line + regression trend + ±1σ / ±2σ residual bands.

    Self-contained (no external scripts); colors match the dashboard theme.
    Width 720 × height 220 keeps it readable inside a detail card.
    """
    W, H = 720, 220
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 12, 10, 28
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    n = len(t)

    upper2 = fitted + 2 * sigma
    upper1 = fitted + sigma
    lower1 = fitted - sigma
    lower2 = fitted - 2 * sigma

    y_min = float(min(y.min(), lower2.min()))
    y_max = float(max(y.max(), upper2.max()))
    y_pad = (y_max - y_min) * 0.04 if y_max > y_min else 1.0
    y_min -= y_pad
    y_max += y_pad
    y_range = y_max - y_min

    def x_at(i: int) -> float:
        return PAD_L + (i / (n - 1)) * plot_w

    def y_at(v: float) -> float:
        return PAD_T + (1 - (v - y_min) / y_range) * plot_h

    def line_path(values: np.ndarray) -> str:
        return "M " + " L ".join(f"{x_at(i):.1f},{y_at(values[i]):.1f}" for i in range(n))

    def two_point(values: np.ndarray) -> str:
        return f"M {x_at(0):.1f},{y_at(values[0]):.1f} L {x_at(n - 1):.1f},{y_at(values[n - 1]):.1f}"

    price_d = line_path(y)
    reg_d = two_point(fitted)
    p1u = two_point(upper1)
    p1d = two_point(lower1)
    p2u = two_point(upper2)
    p2d = two_point(lower2)

    # y-axis labels at 5 evenly spaced grid lines
    y_labels = []
    for k in range(5):
        v = y_min + (4 - k) / 4 * y_range
        y_labels.append(
            f'<text x="{PAD_L - 4:.0f}" y="{y_at(v) + 3:.0f}" '
            f'fill="#777" font-size="9" font-family="monospace" text-anchor="end">{v:.4f}</text>'
            f'<line x1="{PAD_L}" y1="{y_at(v):.1f}" x2="{W - PAD_R}" y2="{y_at(v):.1f}" '
            f'stroke="#1f1f23" stroke-width="0.5"/>'
        )

    # x-axis: time markers every ~year
    x_labels = []
    n_years_approx = max(1, round(n / 252))
    for k in range(n_years_approx + 1):
        idx = min(n - 1, int(round(k * n / n_years_approx)))
        years_ago = (n - 1 - idx) / 252
        label = "now" if years_ago < 0.1 else f"-{years_ago:.0f}y"
        x_labels.append(
            f'<text x="{x_at(idx):.0f}" y="{H - 8:.0f}" fill="#777" font-size="9" '
            f'font-family="monospace" text-anchor="middle">{label}</text>'
        )

    return (
        f'<svg width="100%" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;background:#0e0e10;border:1px solid #2a2a2d">'
        f'{"".join(y_labels)}'
        f'<path d="{p2u}" fill="none" stroke="#d9534f" stroke-width="0.8" stroke-dasharray="4,3" opacity="0.85"/>'
        f'<path d="{p1u}" fill="none" stroke="#aaa" stroke-width="0.6" stroke-dasharray="2,3" opacity="0.45"/>'
        f'<path d="{reg_d}" fill="none" stroke="#aaa" stroke-width="1"/>'
        f'<path d="{p1d}" fill="none" stroke="#aaa" stroke-width="0.6" stroke-dasharray="2,3" opacity="0.45"/>'
        f'<path d="{p2d}" fill="none" stroke="#5cb85c" stroke-width="0.8" stroke-dasharray="4,3" opacity="0.85"/>'
        f'<path d="{price_d}" fill="none" stroke="#d4af37" stroke-width="1.3"/>'
        f'{"".join(x_labels)}'
        f'</svg>'
    )


def _seasonality(close: pd.Series, month: int) -> dict:
    monthly = close.resample("ME").last().pct_change().dropna()
    same_month = monthly[monthly.index.month == month]
    if same_month.empty:
        return {"avg_pct": 0.0, "hit_rate": 0.5, "n_years": 0}
    return {
        "avg_pct": float(same_month.mean() * 100),
        "hit_rate": float((same_month > 0).mean()),
        "n_years": int(len(same_month)),
    }


# -------------------- commodity overlay --------------------

def commodity_tailwind(ccy: str, ohlc_commodities: dict[str, pd.DataFrame]) -> float:
    """Weighted commodity-trend tailwind for a currency in [-1, +1].

    Uses 3m % change of each linked commodity * its sensitivity weight, then
    squashes via tanh so a single big move can't dominate.
    """
    weights = CURRENCIES.get(ccy, {}).get("commodities", {})
    if not weights:
        return 0.0
    contribs = []
    for sym, w in weights.items():
        df = ohlc_commodities.get(sym)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        p = df["Close"].dropna()
        if len(p) < 64:
            continue
        ret_3m = float(p.iloc[-1] / p.iloc[-63] - 1)
        contribs.append(w * ret_3m)
    if not contribs:
        return 0.0
    return float(np.tanh(sum(contribs) * 4))  # 4x scale so 5% comm move ≈ 0.2 tailwind


# -------------------- main --------------------

def compute_pair_technicals(ohlc: pd.DataFrame) -> dict:
    """Pair-level technicals + indicators."""
    if ohlc is None or ohlc.empty or "Close" not in ohlc.columns:
        return {}
    p = ohlc["Close"].dropna()
    if len(p) < 252:
        return {}

    last = float(p.iloc[-1])

    mom_12_1 = float(p.iloc[-21] / p.iloc[-252] - 1) if len(p) >= 252 + 21 else float(p.iloc[-1] / p.iloc[-252] - 1)
    mom_3m = float(p.iloc[-1] / p.iloc[-63] - 1) if len(p) > 63 else 0.0
    mom_1m = float(p.iloc[-1] / p.iloc[-21] - 1) if len(p) > 21 else 0.0

    ma20 = float(p.iloc[-20:].mean())
    ma50 = float(p.iloc[-50:].mean())
    ma200 = float(p.iloc[-200:].mean())
    trend_st = _trend_bucket(last, ma20)
    trend_mt = _trend_bucket(last, ma50)
    trend_lt = _trend_bucket(last, ma200)

    trend_vs_50d = (last / ma50 - 1) if ma50 > 0 else 0.0
    trend_vs_200d = (last / ma200 - 1) if ma200 > 0 else 0.0

    vol_20, vol_60, vol_pctile = _vol_pctile(p)
    vol_regime = _vol_regime(vol_pctile)
    vol_ratio = _vol_ratio_5y(p)

    mom_1m_z = _vol_scaled_return(p, 21)
    mom_3m_z = _vol_scaled_return(p, 63)
    mom_12m_z = _vol_scaled_return(p, 252)

    rsi = _rsi(p)
    macd = _macd_label(p)
    bo = _breakout(p)

    atr = _atr(ohlc)
    stop_dist = atr * 2 if pd.notna(atr) else last * 0.01
    long_stop = round(last - stop_dist, 5)
    short_stop = round(last + stop_dist, 5)

    cta = _cta_score(p)
    seas = _seasonality(p, month=datetime.utcnow().month)
    reg = _regression_analysis(p)

    return {
        "price": round(last, 5),
        "mom_12_1": mom_12_1, "mom_3m": mom_3m, "mom_1m": mom_1m,
        "mom_1m_z": mom_1m_z, "mom_3m_z": mom_3m_z, "mom_12m_z": mom_12m_z,
        "trend_vs_200d": trend_vs_200d, "trend_vs_50d": trend_vs_50d,
        "ma200": ma200, "ma50": ma50, "ma20": ma20,
        "trend_st": trend_st, "trend_mt": trend_mt, "trend_lt": trend_lt,
        "vol_20d_ann_pct": vol_20,
        "vol_60d_ann_pct": vol_60,
        "vol_pctile_3y": vol_pctile,
        "vol_regime": vol_regime,
        "vol_ratio_5y": vol_ratio,
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
        "seas_avg_pct": round(seas["avg_pct"], 2),
        "seas_hit_rate": seas["hit_rate"],
        "seas_n_years": seas["n_years"],
        # 5y OLS regression: residual z-score, R², annualized trend slope, fitted FV, chart SVG
        "valuation_z":     reg.get("z", float("nan")),
        "regression_r2":   reg.get("r2", float("nan")),
        "regression_slope_pct_yr": reg.get("slope_pct_yr", float("nan")),
        "regression_fitted": reg.get("fitted_last", float("nan")),
        "regression_svg":  reg.get("svg", ""),
    }


def build_factor_table(
    ohlc_pairs: dict[str, pd.DataFrame],
    ohlc_commodities: dict[str, pd.DataFrame],
    universe: list[dict],
    health_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-pair rows with technicals + base/quote fundamentals + deltas."""

    # Pre-compute commodity tailwind per currency
    ccy_tail = {ccy: commodity_tailwind(ccy, ohlc_commodities) for ccy in CURRENCIES}

    def health_get(ccy: str, col: str) -> float:
        if health_df is None or health_df.empty or ccy not in health_df.index:
            return float("nan")
        return float(health_df.loc[ccy, col]) if pd.notna(health_df.loc[ccy, col]) else float("nan")

    rows = []
    for u in universe:
        sym = u["yf"]
        if sym not in ohlc_pairs:
            continue
        f = compute_pair_technicals(ohlc_pairs[sym])
        if not f:
            continue

        base, quote = u["base"], u["quote"]

        carry_base = health_get(base, "y10")
        carry_quote = health_get(quote, "y10")
        carry_diff = (carry_base - carry_quote) if (pd.notna(carry_base) and pd.notna(carry_quote)) else float("nan")

        rr_base = health_get(base, "real_rate")
        rr_quote = health_get(quote, "real_rate")
        rr_diff = (rr_base - rr_quote) if (pd.notna(rr_base) and pd.notna(rr_quote)) else float("nan")

        gdp_base = health_get(base, "gdp_yoy")
        gdp_quote = health_get(quote, "gdp_yoy")
        gdp_diff = (gdp_base - gdp_quote) if (pd.notna(gdp_base) and pd.notna(gdp_quote)) else float("nan")

        unemp_base = health_get(base, "unemp")
        unemp_quote = health_get(quote, "unemp")
        unemp_diff = (unemp_base - unemp_quote) if (pd.notna(unemp_base) and pd.notna(unemp_quote)) else float("nan")

        cpi_base = health_get(base, "cpi_yoy")
        cpi_quote = health_get(quote, "cpi_yoy")
        # Positive cpi_diff = base ccy has higher inflation (PPP drag on base).
        cpi_diff = (cpi_base - cpi_quote) if (pd.notna(cpi_base) and pd.notna(cpi_quote)) else float("nan")

        health_base = health_get(base, "health_score")
        health_quote = health_get(quote, "health_score")
        health_diff = (health_base - health_quote) if (pd.notna(health_base) and pd.notna(health_quote)) else float("nan")

        comm_diff = ccy_tail.get(base, 0.0) - ccy_tail.get(quote, 0.0)

        spark = ohlc_pairs[sym]["Close"].dropna().iloc[-120:].tolist()

        rows.append({
            **u,
            **f,
            "carry_base": carry_base, "carry_quote": carry_quote, "carry_diff": carry_diff,
            "rr_base": rr_base, "rr_quote": rr_quote, "rr_diff": rr_diff,
            "gdp_base": gdp_base, "gdp_quote": gdp_quote, "gdp_diff": gdp_diff,
            "unemp_base": unemp_base, "unemp_quote": unemp_quote, "unemp_diff": unemp_diff,
            "cpi_base": cpi_base, "cpi_quote": cpi_quote, "cpi_diff": cpi_diff,
            "health_base": health_base, "health_quote": health_quote, "health_diff": health_diff,
            "comm_base": ccy_tail.get(base, 0.0),
            "comm_quote": ccy_tail.get(quote, 0.0),
            "comm_diff": comm_diff,
            "spark": spark,
        })

    return pd.DataFrame(rows)
