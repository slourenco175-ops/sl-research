"""Stage 6 — expression + sizing.

Convert each surviving view into a clean instrument and a vol-aware size:
  - directional, high conviction, no flagged reversal → futures / cash
  - reversal risk flagged (crowded / turning positioning) → options (defined risk)
Sizing: fractional Kelly (capped) scaled to a per-idea vol budget, using the
asset's in-regime vol from Stage 2. Correlation-aware marginal sizing via the
DCC-GARCH optimizer is Phase 3 — here we report a standalone vol contribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_engine.config import (KELLY_FRACTION, MAX_WEIGHT, TARGET_PORTFOLIO_VOL,
                                  UNIVERSE)
from macro_engine.portfolio import vol_scale_factor


def _instrument(bucket: str, reversal: bool, conviction: float) -> str:
    if reversal:
        return "options (defined risk)"
    if bucket in ("rates", "fx", "credit"):
        return "futures / cash"
    return "futures / ETF"


def express_and_size(scored: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    vol_map = dict(zip(ranked["asset"], ranked["vol_ann"])) if not ranked.empty else {}

    # Pass 1 — per-idea standalone-vol-budgeted weights (pre-correlation).
    rows = []
    for _, r in scored.iterrows():
        bucket = r["bucket"]
        sign = 1.0 if r["direction"] == "LONG" else -1.0
        conviction = abs(float(r["convergence_score"]))
        conviction_scale = float(min(1.5, conviction / 2.0))

        vol_ann = float(vol_map.get(r["asset"], np.nan))   # % annualized
        if np.isnan(vol_ann) or vol_ann <= 0:
            vol_target_w = 0.5                              # fallback if no vol
        else:
            vol_target_w = TARGET_PORTFOLIO_VOL / (vol_ann / 100.0)

        raw_w = KELLY_FRACTION * conviction_scale * vol_target_w
        weight = float(np.clip(raw_w, 0, MAX_WEIGHT)) * sign
        rows.append({"row": r.to_dict(), "bucket": bucket, "conviction": conviction,
                     "vol_ann": vol_ann, "weight": weight})

    # Pass 2 — correlation-aware book scaling. Trim every weight by one factor
    # so the *correlated* vol of the surfaced book meets PORTFOLIO_VOL_CAP.
    surfaced_w = {x["row"]["asset"]: x["weight"] for x in rows if x["row"].get("surfaced")}
    scale = vol_scale_factor(surfaced_w)

    out = []
    for x in rows:
        weight = x["weight"] * scale
        vol_ann = x["vol_ann"]
        vol_contrib = abs(weight) * (vol_ann if not np.isnan(vol_ann) else 0.0)
        out.append({
            **x["row"],
            "instrument": _instrument(x["bucket"], bool(x["row"]["reversal_flag"]), x["conviction"]),
            "suggested_weight_pct": round(weight * 100, 1),
            "est_vol_contribution_pct": round(vol_contrib, 1),
            "book_vol_scale": round(scale, 2),
            "horizon": "1-3m",
        })
    return pd.DataFrame(out)
