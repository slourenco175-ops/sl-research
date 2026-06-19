"""Stage 2 — conditional return engine (Phase 2).

For the *current* 16-state regime we estimate, per universe asset, the
conditional return distribution and rank by conditional Sharpe. Two pieces of
statistical rigor beyond Phase 1:

  1. Hierarchical (James-Stein-style) pooling. The exact 16-state cell is often
     thin, so we shrink its daily mean and variance toward its 2x2 super-regime
     parent: pooled = (n·cell + k·parent) / (n + k). Thin cells lean on the
     parent prior; data-rich cells barely move.

  2. Block-bootstrap confidence interval on the *pooled* Sharpe (block ≈ 1
     trading month to preserve autocorrelation). Cells whose CI straddles zero
     are flagged so Stage 5 can treat them as low-conviction.

Returns one row per asset for the current state, carrying raw cell Sharpe,
parent Sharpe, pooled Sharpe (the decision value reported as `sharpe`), the
bootstrap CI, and a confidence flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_engine.config import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_REPS,
    CI_PCT,
    MIN_OBS_FOR_BOOTSTRAP,
    MIN_OBS_PER_CELL,
    SEED,
    SHRINK_K,
    UNIVERSE,
)
from macro_engine.data import daily_returns, get_universe_prices

TRADING_DAYS = 252


def _max_drawdown(rets: pd.Series) -> float:
    curve = (1 + rets.fillna(0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1).min())


def _ann_sharpe(mean_d: float, var_d: float) -> float:
    vol_d = np.sqrt(var_d)
    if vol_d <= 1e-12:
        return 0.0
    return float((mean_d / vol_d) * np.sqrt(TRADING_DAYS))


def _shrink(cell_mean: float, cell_var: float, n: int,
            parent_mean: float, parent_var: float, k: float) -> tuple[float, float]:
    """Shrink a cell's daily mean & variance toward the parent prior."""
    if n <= 0:
        return parent_mean, parent_var
    w = n / (n + k)
    pooled_mean = w * cell_mean + (1 - w) * parent_mean
    pooled_var = w * cell_var + (1 - w) * parent_var
    return pooled_mean, pooled_var


def _block_bootstrap_sharpe(cell: np.ndarray, parent_mean: float, parent_var: float,
                            rng: np.random.Generator) -> tuple[float, float]:
    """CI on the pooled Sharpe via circular block bootstrap of the cell returns."""
    n = len(cell)
    block = min(BOOTSTRAP_BLOCK, n)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(BOOTSTRAP_REPS)
    for b in range(BOOTSTRAP_REPS):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sample = cell[idx[:n]]
        pm, pv = _shrink(float(sample.mean()), float(sample.var(ddof=0)), n,
                         parent_mean, parent_var, SHRINK_K)
        sharpes[b] = _ann_sharpe(pm, pv)
    lo, hi = np.percentile(sharpes, CI_PCT)
    return float(lo), float(hi)


def conditional_returns(regime: dict) -> pd.DataFrame:
    """Per-asset conditional stats for the *current* 16-state regime.

    Needs Stage 1 `monthly_state` (16-state) and `monthly_super` (parent).
    """
    prices = get_universe_prices()
    rets = daily_returns(prices)
    rng = np.random.default_rng(SEED)

    cur_state = regime["state_code"]
    cur_super = regime["super_regime"]

    state_label = regime["monthly_state"].reindex(rets.index, method="ffill")
    super_label = regime["monthly_super"].reindex(rets.index, method="ffill")
    cell_mask = (state_label == cur_state).to_numpy()
    parent_mask = (super_label == cur_super).to_numpy()

    names = {u["sym"]: u["name"] for u in UNIVERSE}
    buckets = {u["sym"]: u["bucket"] for u in UNIVERSE}

    rows = []
    for sym in [u["sym"] for u in UNIVERSE]:
        if sym not in rets:
            continue
        col = rets[sym]
        parent = col[parent_mask].dropna()
        if parent.empty:
            continue
        parent_mean, parent_var = float(parent.mean()), float(parent.var(ddof=0))
        parent_sharpe = _ann_sharpe(parent_mean, parent_var)

        cell = col[cell_mask].dropna()
        n = int(len(cell))
        if n > 0:
            cell_mean, cell_var = float(cell.mean()), float(cell.var(ddof=0))
            raw_sharpe = _ann_sharpe(cell_mean, cell_var)
        else:
            cell_mean, cell_var, raw_sharpe = parent_mean, parent_var, float("nan")

        pooled_mean, pooled_var = _shrink(cell_mean, cell_var, n,
                                          parent_mean, parent_var, SHRINK_K)
        pooled_sharpe = _ann_sharpe(pooled_mean, pooled_var)

        if n >= MIN_OBS_FOR_BOOTSTRAP:
            ci_low, ci_high = _block_bootstrap_sharpe(cell.to_numpy(), parent_mean,
                                                      parent_var, rng)
        else:
            ci_low, ci_high = float("nan"), float("nan")

        if n < MIN_OBS_FOR_BOOTSTRAP:
            confidence = "thin"
        elif np.isnan(ci_low) or (ci_low < 0.0 < ci_high):
            confidence = "weak"      # bootstrap CI straddles zero
        elif n < MIN_OBS_PER_CELL:
            confidence = "ok"
        else:
            confidence = "strong"

        stat_src = cell if n > 0 else parent
        rows.append({
            "asset": sym, "name": names.get(sym, sym), "bucket": buckets.get(sym, ""),
            "state": cur_state, "super_regime": cur_super,
            "mean_ann": round(pooled_mean * TRADING_DAYS * 100, 2),
            "vol_ann": round(np.sqrt(pooled_var * TRADING_DAYS) * 100, 2),
            "sharpe": round(pooled_sharpe, 2),          # pooled = decision value
            "raw_sharpe": round(raw_sharpe, 2) if not np.isnan(raw_sharpe) else float("nan"),
            "parent_sharpe": round(parent_sharpe, 2),
            "ci_low": round(ci_low, 2) if not np.isnan(ci_low) else float("nan"),
            "ci_high": round(ci_high, 2) if not np.isnan(ci_high) else float("nan"),
            "hit_rate": round(float((stat_src > 0).mean()) * 100, 1),
            "max_dd": round(_max_drawdown(stat_src) * 100, 1),
            "n_obs": n,
            "n_parent": int(len(parent)),
            "confidence": confidence,
        })
    return pd.DataFrame(rows)


def rank_current_regime(cond: pd.DataFrame, regime: dict) -> pd.DataFrame:
    """Rank the current-regime assets by pooled conditional Sharpe.

    Phase 2: cond already holds only the current 16-state rows.
    """
    if cond.empty:
        return cond
    sub = cond.sort_values("sharpe", ascending=False).reset_index(drop=True)
    sub["regime_rank"] = np.arange(1, len(sub) + 1)
    if len(sub) > 1:
        sub["regime_score"] = 1.0 - 2.0 * (sub["regime_rank"] - 1) / (len(sub) - 1)
    else:
        sub["regime_score"] = 0.0
    # Down-weight low-confidence cells so Stage 5 conviction reflects sampling risk.
    conf_mult = {"strong": 1.0, "ok": 0.85, "weak": 0.5, "thin": 0.4}
    sub["regime_score"] = sub.apply(
        lambda r: r["regime_score"] * conf_mult.get(r["confidence"], 0.6), axis=1
    )
    return sub
