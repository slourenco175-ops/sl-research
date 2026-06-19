"""Stage 5 — convergence scoring + ranking.

Combine the orthogonal lenses into one signed conviction score per asset:
  regime     — Stage 2 conditional-Sharpe rank (signed, long-favourable > 0)
  divergence — Stage 3 gap_z aggregated across divergences (signed)
  crowding   — Stage 4, applied *against* the idea's own direction (a modifier)

Convergence = conviction; conflict = haircut/discard. Surface only candidates
where ≥ MIN_AGREEMENT lenses point the same way and no hard lens conflict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_engine.config import MIN_AGREEMENT, UNIVERSE, WEIGHTS

CONFLICT_THRESH = 0.5   # both lenses stronger than this, opposite signs → hard conflict


def converge(ranked: pd.DataFrame, divergences: pd.DataFrame, positioning: pd.DataFrame) -> pd.DataFrame:
    names = {u["sym"]: u["name"] for u in UNIVERSE}
    buckets = {u["sym"]: u["bucket"] for u in UNIVERSE}

    regime_score = dict(zip(ranked["asset"], ranked["regime_score"])) if not ranked.empty else {}
    regime_sharpe = dict(zip(ranked["asset"], ranked["sharpe"])) if not ranked.empty else {}
    regime_conf = dict(zip(ranked["asset"], ranked["confidence"])) if not ranked.empty else {}

    # Aggregate divergence signal per asset (sum of signed per-asset z's).
    if not divergences.empty:
        div_agg = divergences.groupby("asset")["divergence_z"].sum().to_dict()
        div_n = divergences.groupby("asset")["divergence_z"].size().to_dict()
        div_detail = {a: g for a, g in divergences.groupby("asset")}
    else:
        div_agg, div_n, div_detail = {}, {}, {}

    crowd = dict(zip(positioning["asset"], positioning["crowd_long"])) if not positioning.empty else {}
    reversal = dict(zip(positioning["asset"], positioning["reversal_flag"])) if not positioning.empty else {}

    assets = sorted(set(regime_score) | set(div_agg) | set(crowd))
    rows = []
    for a in assets:
        rs = float(regime_score.get(a, 0.0))
        dz = float(div_agg.get(a, 0.0))
        dz_clip = float(np.clip(dz, -3, 3))

        # Provisional direction before crowding, from the two directional lenses.
        provisional = WEIGHTS["regime"] * rs + WEIGHTS["divergence"] * dz_clip
        direction_sign = 1.0 if provisional >= 0 else -1.0

        # Crowding works against the idea's own direction (crowded-with-you = haircut).
        cl = float(crowd.get(a, 0.0))
        crowding_score = -cl * direction_sign           # signed, same frame as direction

        convergence = (WEIGHTS["regime"] * rs
                       + WEIGHTS["divergence"] * dz_clip
                       + WEIGHTS["crowding"] * crowding_score)
        net_sign = 1.0 if convergence >= 0 else -1.0

        # Agreement: lenses whose signed contribution matches the net direction.
        lenses = []
        if a in regime_score:
            lenses.append(("regime", rs))
        if a in div_agg:
            lenses.append(("divergence", dz_clip))
        if a in crowd:
            lenses.append(("crowding", crowding_score))
        agreement = sum(1 for _, v in lenses if v != 0 and np.sign(v) == net_sign)

        # Hard conflict: regime and divergence both strong and opposite.
        conflict = (a in regime_score and a in div_agg
                    and abs(rs) > CONFLICT_THRESH and abs(dz_clip) > CONFLICT_THRESH
                    and np.sign(rs) != np.sign(dz_clip))

        surfaced = (agreement >= MIN_AGREEMENT) and not conflict

        rows.append({
            "asset": a, "name": names.get(a, a), "bucket": buckets.get(a, ""),
            "direction": "LONG" if net_sign > 0 else "SHORT",
            "regime_score": round(rs, 2), "regime_sharpe": round(regime_sharpe.get(a, float("nan")), 2),
            "regime_confidence": regime_conf.get(a, "n/a"),
            "divergence_z": round(dz_clip, 2), "n_divergences": int(div_n.get(a, 0)),
            "crowd_long": round(cl, 2), "crowding_score": round(crowding_score, 2),
            "reversal_flag": bool(reversal.get(a, False)),
            "convergence_score": round(float(convergence), 2),
            "agreement_count": int(agreement),
            "conflict": bool(conflict),
            "surfaced": bool(surfaced),
            "_div_detail": div_detail.get(a),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Rank by absolute conviction (strongest signed view first).
    df["abs_conv"] = df["convergence_score"].abs()
    df = df.sort_values(["surfaced", "abs_conv"], ascending=[False, False]).reset_index(drop=True)
    return df.drop(columns="abs_conv")
