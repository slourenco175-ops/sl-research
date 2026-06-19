"""Point-in-time backtester for the macro idea engine.

Tests the core thesis the live engine asks you to take on faith: *does ranking
the cross-asset universe by regime-conditional Sharpe, tilted by macro
divergences, actually earn out of sample?*

Method (all explicit, all auditable):
  - Monthly rebalance (month-end).
  - At each rebalance `asof` we use ONLY data dated ≤ asof:
      • Regime label = the Stage 1 16-state / super-regime at asof.
      • Conditional Sharpe per asset = pooled (Stage 2 James-Stein shrink of the
        16-state cell toward its super-regime parent) on the return history up
        to asof. Rank → signed regime_score in [-1, 1].
      • Divergence z per asset from the PIT-computable Stage 3 lenses
        (inflation, growth, term_premium, credit_equity, fx_carry,
        terms_of_trade). Rolling z's are trailing; regression residuals are
        re-fit on the trailing window at each asof (no look-ahead betas).
      • Surface = the two directional lenses agree in sign (mirrors the live
        agreement≥2, no-conflict rule once COT is excluded).
  - Sizing: vol-parity legs (trailing 60d vol), gross capped at 2x, then scaled
    by trailing covariance to a target book vol (correlation-aware, same spirit
    as portfolio.py). Hold to next month-end. Cost = bps × turnover.
  - Benchmarks: equal-weight long-only universe and a 60/40 SPY/IEF book.

Honest limitations:
  - The Fed **policy_path** divergence is a single dated snapshot, not a
    historical series, so it is EXCLUDED from the backtest. Live runs get one
    extra lens the backtest never sees.
  - Stage 1 pillars use as-reported FRED (no publication lag), so the regime
    labels carry the same mild look-ahead flagged for the live engine. This
    backtest inherits it; treat results as optimistic by that margin.
  - COT positioning history is only ~3y, so the crowding lens is omitted.
  - yfinance ETF closes are auto-adjusted total-return proxies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macro_engine.config import COUNTRY_COMMODITY, SHRINK_K, UNIVERSE, WEIGHTS
from macro_engine.data import close_frame, daily_returns, fred_series, get_universe_prices
from macro_engine.stage1_regime import detect_regime
from macro_engine.stage2_conditional import _ann_sharpe, _shrink, TRADING_DAYS
from macro_engine.stage3_divergence import ASSET_SIGNS, _daily_yoy_from_monthly, _z

SYMBOLS = [u["sym"] for u in UNIVERSE]
BUCKET = {u["sym"]: u["bucket"] for u in UNIVERSE}


@dataclass
class BTConfig:
    start: str = "2010-12-31"          # enough universe + regime history by here
    rebalance: str = "ME"
    target_book_vol: float = 0.10      # annualized, correlation-aware
    target_vol_per_leg: float = 0.10
    max_gross: float = 2.0
    cost_bps: float = 5.0
    vol_lookback: int = 60
    cov_lookback: int = 120
    div_clip: float = 3.0


# ----------------------------- PIT signal helpers -----------------------------

def _pit_resid_z(y: pd.Series, xs: list[pd.Series], asof: pd.Timestamp,
                 window: int = 504) -> float:
    """Standardized current OLS residual of y on xs, fit on the trailing window
    up to asof (no look-ahead). Positive = y rich vs fair value."""
    frame = pd.concat([y.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(xs)],
                      axis=1).dropna().loc[:asof]
    if len(frame) < 60:
        return np.nan
    fit = frame.tail(window)
    X = np.column_stack([np.ones(len(fit))]
                        + [fit[f"x{i}"].to_numpy() for i in range(len(xs))])
    beta, *_ = np.linalg.lstsq(X, fit["y"].to_numpy(), rcond=None)
    resid = fit["y"].to_numpy() - X @ beta
    sd = resid.std(ddof=0)
    if sd <= 1e-12:
        return np.nan
    return float((resid[-1] - resid.mean()) / sd)


def _last(series: pd.Series, asof: pd.Timestamp) -> float:
    s = series.loc[:asof].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def _precompute_rolling_gaps(closes: pd.DataFrame) -> dict[str, pd.Series]:
    """Trailing-z gap series that are PIT-safe to sample at any asof."""
    idx = closes.index
    gaps: dict[str, pd.Series] = {}

    be = fred_series("T10YIE")
    if not be.empty:
        be = be.reindex(idx, method="ffill")
        infl = _daily_yoy_from_monthly("CPIAUCSL", idx)
        gaps["inflation"] = (_z(infl) - _z(be))           # +ve = more inflation than priced

    if "HG=F" in closes and "GC=F" in closes:
        cg = (closes["HG=F"] / closes["GC=F"]).dropna()
        grow = _daily_yoy_from_monthly("INDPRO", idx)
        gaps["growth"] = (_z(grow) - _z(cg))              # +ve = more growth than priced

    foreign = []
    for fid in ("ECBDFR", "IRSTCB01JPM156N"):
        s = fred_series(fid)
        if not s.empty:
            foreign.append(s.resample("ME").last())
    dgs2 = fred_series("DGS2")
    if not dgs2.empty and foreign:
        us = dgs2.resample("ME").last()
        favg = pd.concat(foreign, axis=1).mean(axis=1)
        gaps["fx_carry"] = _z((us - favg).dropna(), window=36)
    return gaps


def _divergence_z(asof, gaps, closes, dgs10, dgs2, be10, oas, vix) -> dict[str, float]:
    """Per-asset aggregated signed divergence z at asof (policy_path excluded)."""
    out: dict[str, float] = {}

    def add(name: str, gap: float):
        if np.isnan(gap):
            return
        for asset, sign in ASSET_SIGNS.get(name, {}).items():
            out[asset] = out.get(asset, 0.0) + gap * sign

    for name in ("inflation", "growth", "fx_carry"):
        if name in gaps:
            add(name, _last(gaps[name], asof))

    if dgs10 is not None:
        add("term_premium", _pit_resid_z(dgs10, [dgs2, be10], asof))
    if oas is not None and vix is not None:
        add("credit_equity", _pit_resid_z(oas, [vix], asof))

    # terms_of_trade — each country its own +1 signed cheap/rich residual
    for country, comm in COUNTRY_COMMODITY.items():
        if country in closes and comm in closes:
            rz = _pit_resid_z(np.log(closes[country].replace(0, np.nan)),
                              [np.log(closes[comm].replace(0, np.nan))], asof)
            if not np.isnan(rz):
                out[country] = out.get(country, 0.0) + (-rz)   # cheap ⇒ long
    return out


def _regime_scores(asof, daily_rets, state_label, super_label) -> dict[str, float]:
    """PIT pooled conditional Sharpe rank → signed score in [-1, 1]."""
    cur_state = _label_at(state_label, asof)
    cur_super = _label_at(super_label, asof)
    if cur_state is None:
        return {}
    rets = daily_rets.loc[:asof]
    cell_mask = (state_label.reindex(rets.index, method="ffill") == cur_state).to_numpy()
    parent_mask = (super_label.reindex(rets.index, method="ffill") == cur_super).to_numpy()

    sharpes = {}
    for sym in SYMBOLS:
        if sym not in rets:
            continue
        col = rets[sym]
        parent = col[parent_mask].dropna()
        if len(parent) < 60:
            continue
        pm, pv = float(parent.mean()), float(parent.var(ddof=0))
        cell = col[cell_mask].dropna()
        n = len(cell)
        cm = float(cell.mean()) if n else pm
        cv = float(cell.var(ddof=0)) if n else pv
        pooled_m, pooled_v = _shrink(cm, cv, n, pm, pv, SHRINK_K)
        sharpes[sym] = _ann_sharpe(pooled_m, pooled_v)
    if not sharpes:
        return {}
    ser = pd.Series(sharpes).sort_values(ascending=False)
    ranks = np.arange(1, len(ser) + 1)
    score = 1.0 - 2.0 * (ranks - 1) / max(len(ser) - 1, 1)
    return dict(zip(ser.index, score))


def _label_at(label: pd.Series, asof: pd.Timestamp):
    s = label.loc[:asof]
    return s.iloc[-1] if len(s) else None


# ------------------------------- the backtest --------------------------------

def run_backtest(cfg: BTConfig | None = None) -> dict:
    cfg = cfg or BTConfig()
    prices = get_universe_prices()
    closes = close_frame(prices)
    daily_rets = daily_returns(prices)

    regime = detect_regime()
    state_label = regime["monthly_state"]
    super_label = regime["monthly_super"]

    gaps = _precompute_rolling_gaps(closes)
    dgs10, dgs2, be10 = fred_series("DGS10"), fred_series("DGS2"), fred_series("T10YIE")
    oas = fred_series("BAMLH0A0HYM2")
    vix = closes["^VIX"] if "^VIX" in closes else None
    dgs10 = dgs10 if not dgs10.empty else None
    oas = oas if not oas.empty else None

    rebals = pd.date_range(cfg.start, closes.index.max(), freq=cfg.rebalance)
    rebals = [closes.index[closes.index.searchsorted(d, side="right") - 1] for d in rebals]
    rebals = sorted(set(d for d in rebals if d >= pd.Timestamp(cfg.start)))

    weights_hist, prev_w = [], pd.Series(0.0, index=closes.columns)
    book_label = {}

    for asof in rebals:
        rscore = _regime_scores(asof, daily_rets, state_label, super_label)
        dz = _divergence_z(asof, gaps, closes, dgs10, dgs2, be10, oas, vix)

        picks = {}
        for sym in SYMBOLS:
            rs = rscore.get(sym, 0.0)
            d = float(np.clip(dz.get(sym, 0.0), -cfg.div_clip, cfg.div_clip))
            if rs == 0.0 or d == 0.0 or np.sign(rs) != np.sign(d):
                continue                                  # need agreement (≥2 lenses)
            conv = WEIGHTS["regime"] * rs + WEIGHTS["divergence"] * d
            picks[sym] = conv

        new_w = pd.Series(0.0, index=closes.columns)
        for sym, conv in picks.items():
            sub = daily_rets[sym].loc[:asof].dropna().iloc[-cfg.vol_lookback:]
            if len(sub) < cfg.vol_lookback // 2:
                continue
            vol = float(sub.std(ddof=0) * np.sqrt(TRADING_DAYS))
            if vol <= 1e-6:
                continue
            new_w[sym] = np.sign(conv) * (cfg.target_vol_per_leg / vol)

        gross = new_w.abs().sum()
        if gross > cfg.max_gross:
            new_w *= cfg.max_gross / gross

        # correlation-aware scale to target book vol (trailing covariance)
        held = [s for s in new_w.index if abs(new_w[s]) > 1e-9]
        if held:
            cov = daily_rets[held].loc[:asof].iloc[-cfg.cov_lookback:].cov() * TRADING_DAYS
            w = new_w[held].to_numpy()
            bv = float(np.sqrt(max(w @ cov.fillna(0).to_numpy() @ w, 0.0)))
            if bv > 1e-6:
                new_w *= min(cfg.max_gross / max(new_w.abs().sum(), 1e-9),
                             cfg.target_book_vol / bv)

        cost = (new_w - prev_w).abs().sum() * (cfg.cost_bps / 10000)
        weights_hist.append({"date": asof, "weights": new_w.copy(), "cost": cost})
        book_label[asof] = _label_at(super_label, asof)
        prev_w = new_w

    return _assemble(cfg, weights_hist, daily_rets, closes, book_label)


def _assemble(cfg, weights_hist, daily_rets, closes, book_label) -> dict:
    if not weights_hist:
        return {"stats": {}, "equity": pd.Series(dtype=float)}
    wdf = pd.DataFrame([w["weights"] for w in weights_hist],
                       index=[w["date"] for w in weights_hist])
    costs = pd.Series([w["cost"] for w in weights_hist],
                      index=[w["date"] for w in weights_hist])

    held_w = wdf.reindex(daily_rets.index, method="ffill").fillna(0).shift(1).fillna(0)
    pnl = (held_w * daily_rets.fillna(0)).sum(axis=1)
    pnl = pnl - costs.reindex(daily_rets.index).fillna(0)
    pnl = pnl.loc[cfg.start:]
    equity = (1 + pnl).cumprod()

    # benchmarks
    avail = [s for s in SYMBOLS if s in daily_rets]
    ew = daily_rets[avail].loc[cfg.start:].mean(axis=1)
    bench_ew = (1 + ew).cumprod()
    sixty40 = pd.Series(0.0, index=daily_rets.index)
    if "SPY" in daily_rets and "IEF" in daily_rets:
        sixty40 = 0.6 * daily_rets["SPY"].fillna(0) + 0.4 * daily_rets["IEF"].fillna(0)
    bench_6040 = (1 + sixty40.loc[cfg.start:]).cumprod()

    stats = _stats(pnl, equity)
    stats["bench_ew"] = _stats(ew.loc[cfg.start:], bench_ew)
    stats["bench_6040"] = _stats(sixty40.loc[cfg.start:], bench_6040)

    # regime attribution — monthly pnl grouped by the book's super-regime label
    monthly = (1 + pnl).resample("ME").prod() - 1
    lbl = pd.Series(book_label).reindex(monthly.index, method="ffill")
    attr = {}
    for reg, grp in monthly.groupby(lbl):
        attr[reg] = {"n_months": int(len(grp)), "mean_pct": round(float(grp.mean()) * 100, 2),
                     "total_pct": round(float((1 + grp).prod() - 1) * 100, 1),
                     "hit": round(float((grp > 0).mean()) * 100, 0)}

    return {"stats": stats, "equity": equity, "daily": pnl, "monthly": monthly,
            "weights": wdf, "attribution": attr, "bench_ew": bench_ew,
            "bench_6040": bench_6040, "config": cfg}


def _stats(daily: pd.Series, equity: pd.Series) -> dict:
    daily = daily.dropna()
    if len(daily) < 30:
        return {}
    af = 252
    mu, sd = float(daily.mean()), float(daily.std(ddof=0))
    cagr = float(equity.iloc[-1] ** (af / len(daily)) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = sd * np.sqrt(af)
    sharpe = (mu * af) / vol if vol > 0 else 0.0
    neg = daily[daily < 0]
    sortino = (mu * af) / (neg.std(ddof=0) * np.sqrt(af)) if len(neg) > 5 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    monthly = (1 + daily).resample("ME").prod() - 1
    return {
        "start": daily.index[0].strftime("%Y-%m-%d"), "end": daily.index[-1].strftime("%Y-%m-%d"),
        "n_days": len(daily), "cagr_pct": round(cagr * 100, 1), "vol_pct": round(vol * 100, 1),
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "max_dd_pct": round(dd * 100, 1), "hit_daily_pct": round(float((daily > 0).mean()) * 100, 0),
        "months_pos_pct": round(float((monthly > 0).mean()) * 100, 0),
        "final_equity": round(float(equity.iloc[-1]), 3), "calmar": round(cagr / abs(dd), 2) if dd < 0 else None,
    }


if __name__ == "__main__":
    from macro_engine.backtest_report import main
    main()
