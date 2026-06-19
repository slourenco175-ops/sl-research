"""Correlation-aware portfolio risk.

Sizing in Stage 6 is per-idea (standalone vol budget). That ignores the fact
that the surfaced book is usually full of correlated bets — long several equity
markets, short several bonds — so the *naive* sum of standalone vols badly
overstates real risk and a stack of look-alike trades masquerades as
diversification.

This module computes the annualized covariance of the universe from daily
returns and provides:
  - `vol_scale_factor` — a single multiplier Stage 6 applies so the correlated
    book vol targets `PORTFOLIO_VOL_CAP` (never scales *up*, only trims).
  - `portfolio_risk` — a risk panel for the dashboard: correlated book vol vs
    the naive standalone sum, the diversification ratio, net exposure by bucket,
    and the correlation matrix among surfaced ideas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_engine.config import PORTFOLIO_VOL_CAP, UNIVERSE
from macro_engine.data import daily_returns, get_universe_prices

TRADING_DAYS = 252
_BUCKET = {u["sym"]: u["bucket"] for u in UNIVERSE}


def _signed_weights(sheet: pd.DataFrame, surfaced_only: bool = True) -> dict[str, float]:
    if sheet.empty:
        return {}
    sub = sheet[sheet["surfaced"]] if surfaced_only and "surfaced" in sheet else sheet
    w = {}
    for _, r in sub.iterrows():
        wpct = r.get("suggested_weight_pct")
        if wpct is not None and not pd.isna(wpct):
            w[r["asset"]] = float(wpct) / 100.0
    return w


def _ann_cov(assets: list[str]) -> pd.DataFrame:
    rets = daily_returns(get_universe_prices())
    cols = [a for a in assets if a in rets.columns]
    # Pairwise-complete covariance tolerates the younger country ETFs.
    return rets[cols].cov() * TRADING_DAYS


def book_vol(weights: dict[str, float]) -> tuple[float, pd.DataFrame]:
    """Annualized vol of the signed-weight book; returns (vol, ann-cov)."""
    assets = list(weights)
    if not assets:
        return 0.0, pd.DataFrame()
    cov = _ann_cov(assets)
    assets = [a for a in assets if a in cov.index]
    if not assets:
        return 0.0, cov
    w = np.array([weights[a] for a in assets])
    C = cov.loc[assets, assets].fillna(0.0).to_numpy()
    var = float(w @ C @ w)
    return float(np.sqrt(max(var, 0.0))), cov


def vol_scale_factor(weights: dict[str, float]) -> float:
    """Multiplier so the correlated book vol meets PORTFOLIO_VOL_CAP (trim only)."""
    vol, _ = book_vol(weights)
    if vol <= 1e-9:
        return 1.0
    return float(min(1.0, PORTFOLIO_VOL_CAP / vol))


def portfolio_risk(sheet: pd.DataFrame) -> dict:
    """Risk panel for the surfaced book (post-sizing)."""
    weights = _signed_weights(sheet, surfaced_only=True)
    if not weights:
        return {}
    vol, cov = book_vol(weights)
    assets = [a for a in weights if a in cov.index]

    standalone = {a: float(np.sqrt(max(cov.loc[a, a], 0.0))) for a in assets}
    sum_standalone = sum(abs(weights[a]) * standalone[a] for a in assets)
    div_ratio = (sum_standalone / vol) if vol > 1e-9 else float("nan")

    gross = sum(abs(w) for w in weights.values())
    net = sum(weights.values())
    net_by_bucket: dict[str, float] = {}
    for a, w in weights.items():
        b = _BUCKET.get(a, "other")
        net_by_bucket[b] = net_by_bucket.get(b, 0.0) + w

    rets = daily_returns(get_universe_prices())
    corr = rets[assets].corr() if len(assets) > 1 else pd.DataFrame()

    return {
        "book_vol_pct": round(vol * 100, 1),
        "sum_standalone_pct": round(sum_standalone * 100, 1),
        "diversification_ratio": round(div_ratio, 2) if not np.isnan(div_ratio) else None,
        "gross_pct": round(gross * 100, 1),
        "net_pct": round(net * 100, 1),
        "net_by_bucket": {k: round(v * 100, 1) for k, v in sorted(net_by_bucket.items())},
        "corr": corr,
        "assets": assets,
        "vol_cap_pct": round(PORTFOLIO_VOL_CAP * 100, 1),
    }
