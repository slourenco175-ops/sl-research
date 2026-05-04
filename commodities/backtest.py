"""Vectorized backtester for the v2 cross-sectional commodity score.

Design choices (all explicit, all auditable):
  - Weekly rebalance (Friday close), to align with COT updates.
  - At each rebalance: rebuild factors from prices ONLY up to that date,
    look up the latest COT report dated <= that date, score every contract,
    take the top N longs and bottom N shorts.
  - Position sizing: equal-risk ('vol parity'). Each leg gets weight
    target_vol / asset_realized_vol(20d). Total notional capped at 200% (2x).
  - Cost model: 5 bps per trade per side (flat, conservative for liquid
    futures via futures-on-stocks proxies). Cost = bps * |Δweight|.
  - No regime overlay, no macro filter — pure score-driven.
  - Returns are computed on close-to-close of the held weights.

Honest limitations:
  - yfinance =F symbols are continuous front-month, NOT roll-adjusted, so
    backtest returns include some roll noise (typically -0.5 to -2% / yr).
  - COT history is 3y from cached fetch; backtests start from when full COT
    is available (~2022-01).
  - Carry is the proxy (no real curve). When we wire a real feed in, we
    should re-run this and compare.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from commodities.factors import (
    _atr,
    _breakout,
    _carry_proxy,
    _cta_score,
    _macd_label,
    _rsi,
    _seasonality,
    _trend_bucket,
    _vol_pctile,
    _vol_regime,
)
from commodities.scoring import score_row


@dataclass
class BacktestConfig:
    start: str = "2022-01-07"     # first Friday with full COT history
    n_per_side: int = 5            # 5 long + 5 short
    target_vol_per_leg: float = 0.10   # 10% annualized per asset
    max_gross_leverage: float = 2.0    # cap total notional at 2x
    cost_bps_per_side: float = 5.0
    rebalance: str = "W-FRI"


# -------------------- point-in-time helpers --------------------

def _factors_at(ohlc: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Compute the same v2 factors but using ONLY data up to `asof`."""
    sub = ohlc.loc[:asof]
    if sub.empty or len(sub) < 252:
        return {}
    p = sub["Close"].dropna()
    if len(p) < 252:
        return {}

    last = float(p.iloc[-1])
    mom_12_1 = float(p.iloc[-21] / p.iloc[-252] - 1) if len(p) >= 252 + 21 else 0.0
    mom_3m = float(p.iloc[-1] / p.iloc[-63] - 1) if len(p) > 63 else 0.0
    mom_1m = float(p.iloc[-1] / p.iloc[-21] - 1) if len(p) > 21 else 0.0

    ma20 = float(p.iloc[-20:].mean())
    ma50 = float(p.iloc[-50:].mean())
    ma200 = float(p.iloc[-200:].mean())

    vol_20, vol_60, vol_pctile = _vol_pctile(p)
    rsi = _rsi(p)
    macd = _macd_label(p)
    bo = _breakout(p)
    atr = _atr(sub)
    cta = _cta_score(p)
    carry = _carry_proxy(p)
    seas = _seasonality(p, month=asof.month)

    return {
        "price": last,
        "mom_12_1": mom_12_1, "mom_3m": mom_3m, "mom_1m": mom_1m,
        "trend_vs_200d": (last / ma200 - 1) if ma200 > 0 else 0.0,
        "trend_vs_50d": (last / ma50 - 1) if ma50 > 0 else 0.0,
        "trend_st": _trend_bucket(last, ma20),
        "trend_mt": _trend_bucket(last, ma50),
        "trend_lt": _trend_bucket(last, ma200),
        "rsi_14": rsi, "macd": macd, "breakout_20d": bo,
        "atr_14": atr,
        "vol_20d_ann_pct": vol_20, "vol_60d_ann_pct": vol_60,
        "vol_pctile_3y": vol_pctile, "vol_regime": _vol_regime(vol_pctile),
        "cta_score": cta["score"], "cta_z_1m": cta["z_1m"],
        "cta_z_3m": cta["z_3m"], "cta_z_12m": cta["z_12m"],
        "carry_state": carry["state"], "carry_proxy_pct": carry["value_pct"],
        "carry_is_proxy": True,
        "seas_avg_pct": seas["avg_pct"], "seas_hit_rate": seas["hit_rate"],
        "_realized_vol_20d_ann": (vol_20 / 100) if pd.notna(vol_20) else 0.30,
    }


def _cot_at(cot_df: pd.DataFrame, contract_substring: str, asof: pd.Timestamp) -> dict:
    if cot_df.empty or contract_substring is None:
        return {}
    mask = cot_df["Market_and_Exchange_Names"].str.contains(
        contract_substring, case=False, na=False, regex=False
    )
    sub = cot_df[mask & (cot_df["report_date"] <= asof)].sort_values("report_date")
    if sub.empty:
        return {}

    def _num(col):
        return pd.to_numeric(sub.get(col, 0), errors="coerce").fillna(0) \
            if col in sub.columns else pd.Series(0, index=sub.index)

    mm_long = pd.to_numeric(sub["M_Money_Positions_Long_All"], errors="coerce")
    mm_short = pd.to_numeric(sub["M_Money_Positions_Short_All"], errors="coerce")
    oi = pd.to_numeric(sub["Open_Interest_All"], errors="coerce")
    pm_long = _num("Prod_Merc_Positions_Long_All")
    pm_short = _num("Prod_Merc_Positions_Short_All")
    sd_long = _num("Swap_Positions_Long_All")
    sd_short = _num("Swap_Positions_Short_All")

    mm_net = (mm_long - mm_short)
    mm_net_pct = (mm_net / oi * 100).dropna()
    comm_net_pct = (((pm_long + sd_long) - (pm_short + sd_short)) / oi * 100).dropna()
    if mm_net_pct.empty:
        return {}

    latest = float(mm_net_pct.iloc[-1])
    pctile = float(mm_net_pct.rank(pct=True).iloc[-1] * 100)
    one_wk = float(mm_net_pct.iloc[-1] - mm_net_pct.iloc[-2]) if len(mm_net_pct) >= 2 else 0.0
    four_wk = float(mm_net_pct.iloc[-1] - mm_net_pct.iloc[-5]) if len(mm_net_pct) >= 5 else one_wk

    if (latest >= 0 and four_wk < 0) or (latest < 0 and four_wk > 0):
        traj = "REVERSING"
    elif abs(four_wk) >= abs(one_wk) * 3:
        traj = "ACCELERATING"
    else:
        traj = "DECELERATING"

    return {
        "mm_net_pct_oi": latest,
        "mm_3y_percentile": pctile,
        "mm_1w_change": one_wk,
        "mm_4w_change": four_wk,
        "mm_trajectory": traj,
        "comm_3y_percentile": float(comm_net_pct.rank(pct=True).iloc[-1] * 100) if not comm_net_pct.empty else None,
    }


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.std(ddof=0) == 0 or s.dropna().empty:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


# -------------------- main backtest --------------------

def run_backtest(
    ohlc: dict[str, pd.DataFrame],
    cot_df: pd.DataFrame,
    universe: list[dict],
    cfg: BacktestConfig | None = None,
) -> dict:
    """Run the cross-sectional backtest. Returns dict with equity curve,
    weights history, trades, and aggregated stats."""
    cfg = cfg or BacktestConfig()

    # Daily closes panel for return computation
    closes = pd.DataFrame({s: df["Close"] for s, df in ohlc.items()}).ffill(limit=3)
    daily_rets = closes.pct_change().fillna(0)

    # Rebalance dates
    end = closes.index.max()
    all_fridays = pd.date_range(cfg.start, end, freq=cfg.rebalance)
    rebal_dates = [d for d in all_fridays if d in closes.index or d <= closes.index.max()]
    # Snap each rebal date to the closest preceding trading day actually present.
    snapped = []
    for d in rebal_dates:
        idx = closes.index.searchsorted(d, side="right") - 1
        if idx >= 0:
            snapped.append(closes.index[idx])
    rebal_dates = sorted(set(snapped))

    # State
    weights_history = []   # list of (date, dict[symbol -> weight])
    score_history = []
    prev_w = pd.Series(0.0, index=closes.columns)

    # Empty macro for backtest (we don't have historical DXY series cached
    # consistently — keeping it neutral makes results comparable). The scoring
    # function handles missing macro gracefully.
    macro_neutral = {}

    for asof in rebal_dates:
        # 1. Build factor table at this date
        rows = []
        for u in universe:
            sym = u["yf"]
            if sym not in ohlc:
                continue
            f = _factors_at(ohlc[sym], asof)
            if not f:
                continue
            cot = _cot_at(cot_df, u.get("cot_name"), asof)
            rows.append({**u, **f, **cot})
        if not rows:
            continue

        df = pd.DataFrame(rows)
        # Cross-sectional composite z (same as live model)
        df["z_mom_12_1"] = _zscore(df["mom_12_1"])
        df["z_mom_1m"] = _zscore(df["mom_1m"])
        df["z_trend"] = _zscore(df["trend_vs_200d"])
        df["composite_z"] = df["z_mom_12_1"] - 0.3 * df["z_mom_1m"] + 0.5 * df["z_trend"]

        # 2. Score every row
        df["_score"] = [score_row(r.to_dict(), macro_neutral)[0] for _, r in df.iterrows()]
        df = df.sort_values("_score", ascending=False).reset_index(drop=True)

        score_history.append({
            "date": asof,
            "scores": df.set_index("yf")["_score"].to_dict(),
        })

        # 3. Pick top N longs, bottom N shorts
        n = min(cfg.n_per_side, len(df) // 3)
        longs = df.head(n)
        shorts = df.tail(n)

        # 4. Vol-parity sizing per leg
        new_w = pd.Series(0.0, index=closes.columns)
        for _, r in longs.iterrows():
            v = max(r.get("_realized_vol_20d_ann", 0.30), 0.05)
            new_w[r["yf"]] = cfg.target_vol_per_leg / v / n  # /n so total long weight ~target_vol/avg_vol
        for _, r in shorts.iterrows():
            v = max(r.get("_realized_vol_20d_ann", 0.30), 0.05)
            new_w[r["yf"]] = -cfg.target_vol_per_leg / v / n

        # 5. Cap gross leverage
        gross = new_w.abs().sum()
        if gross > cfg.max_gross_leverage:
            new_w *= cfg.max_gross_leverage / gross

        # 6. Costs from turnover
        turnover = (new_w - prev_w).abs().sum()
        cost = turnover * (cfg.cost_bps_per_side / 10000)

        weights_history.append({"date": asof, "weights": new_w.copy(), "cost": cost})
        prev_w = new_w

    # ---- mark-to-market: between rebalances, hold weights and accrue ----
    if not weights_history:
        return {"equity": pd.Series(dtype=float), "stats": {}, "trades": pd.DataFrame()}

    wdf = pd.DataFrame([w["weights"] for w in weights_history],
                       index=[w["date"] for w in weights_history])
    cost_series = pd.Series([w["cost"] for w in weights_history],
                            index=[w["date"] for w in weights_history])

    # Forward-fill weights to daily, shift by 1 (use weights set at close to
    # earn the next day's return)
    wdf = wdf.reindex(daily_rets.index, method="ffill").fillna(0).shift(1).fillna(0)
    daily_pnl = (wdf * daily_rets).sum(axis=1)

    # Apply cost on rebalance days
    cost_daily = cost_series.reindex(daily_rets.index).fillna(0)
    daily_pnl = daily_pnl - cost_daily

    daily_pnl = daily_pnl.loc[cfg.start:]
    equity = (1 + daily_pnl).cumprod()

    stats = _compute_stats(daily_pnl, equity)

    # Trades: one row per rebalance per non-zero position
    trade_rows = []
    for w in weights_history:
        for sym, wt in w["weights"].items():
            if abs(wt) > 1e-6:
                side = "LONG" if wt > 0 else "SHORT"
                trade_rows.append({
                    "date": w["date"], "symbol": sym, "side": side, "weight": round(wt, 4),
                })
    trades_df = pd.DataFrame(trade_rows)

    return {
        "equity": equity,
        "daily_returns": daily_pnl,
        "weights": wdf,
        "stats": stats,
        "trades": trades_df,
        "score_history": score_history,
        "config": cfg,
    }


def _compute_stats(daily_pnl: pd.Series, equity: pd.Series) -> dict:
    if len(daily_pnl) < 30:
        return {}
    ann_factor = 252
    mu_d = float(daily_pnl.mean())
    sd_d = float(daily_pnl.std(ddof=0))
    cagr = float(equity.iloc[-1] ** (ann_factor / len(daily_pnl)) - 1) if equity.iloc[-1] > 0 else -1.0
    vol_ann = sd_d * np.sqrt(ann_factor)
    sharpe = (mu_d * ann_factor) / vol_ann if vol_ann > 0 else 0.0

    # downside
    neg = daily_pnl[daily_pnl < 0]
    sortino = (mu_d * ann_factor) / (neg.std(ddof=0) * np.sqrt(ann_factor)) if len(neg) > 5 else 0.0

    # drawdown
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1
    max_dd = float(dd.min())

    # hit rate (positive days)
    hit = float((daily_pnl > 0).mean())

    # monthly returns table
    monthly = (1 + daily_pnl).resample("ME").prod() - 1
    months_pos = float((monthly > 0).mean())

    # rolling 1y Sharpe (last)
    if len(daily_pnl) >= 252:
        last_year = daily_pnl.iloc[-252:]
        roll_sharpe = (last_year.mean() * 252) / (last_year.std(ddof=0) * np.sqrt(252))
    else:
        roll_sharpe = sharpe

    return {
        "n_days": len(daily_pnl),
        "start": daily_pnl.index[0].strftime("%Y-%m-%d"),
        "end": daily_pnl.index[-1].strftime("%Y-%m-%d"),
        "cagr": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "hit_rate_daily": hit,
        "months_positive": months_pos,
        "rolling_1y_sharpe": float(roll_sharpe),
        "final_equity": float(equity.iloc[-1]),
        "monthly_returns": monthly,
    }
