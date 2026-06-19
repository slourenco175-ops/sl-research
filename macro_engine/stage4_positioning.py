"""Stage 4 — positioning overlay.

Tells you whether a convergent idea is clean or crowded. Crowding doesn't kill a
thesis; it changes the payoff (crowded longs cap upside, fatten the left tail).
We read CFTC net-spec positioning percentiles — disaggregated (Managed Money)
for commodities, TFF (Leveraged Money) for FX / equity-index futures — and emit
a per-asset crowding read. Stage 5 turns it into a directional multiplier.

Positioning is noisy and lagged (COT is T+3, weekly) — a modifier, never a
trigger (design doc §6).
"""
from __future__ import annotations

import pandas as pd

from commodities.cmdty_data import get_cot_history, latest_managed_money
from forex.fx_data import get_tff_history, latest_lev_money
from macro_engine.config import UNIVERSE


def _crowd_read(mm: dict) -> dict:
    pctile = mm.get("mm_3y_percentile")
    if pctile is None or pd.isna(pctile):
        return {}
    crowd_long = round((pctile - 50.0) / 50.0, 2)      # +1 crowded long, -1 crowded short
    reversal = mm.get("mm_trajectory") == "REVERSING" or pctile >= 90 or pctile <= 10
    return {
        "spec_pctile": pctile,
        "crowd_long": crowd_long,
        "reversal_flag": bool(reversal),
        "trajectory": mm.get("mm_trajectory"),
        "report_date": mm.get("report_date"),
    }


def positioning_overlay() -> pd.DataFrame:
    try:
        disagg = get_cot_history(years_back=3)
    except Exception as e:
        print(f"  ! Stage 4 disaggregated COT: {e}")
        disagg = pd.DataFrame()
    try:
        tff = get_tff_history(years_back=3)
    except Exception as e:
        print(f"  ! Stage 4 TFF COT: {e}")
        tff = pd.DataFrame()

    rows = []
    for u in UNIVERSE:
        sub = u.get("cot")
        if not sub:
            continue
        mm = None
        if not disagg.empty:
            mm = latest_managed_money(disagg, sub)
        if mm is None and not tff.empty:
            mm = latest_lev_money(tff, sub)
        read = _crowd_read(mm) if mm else {}
        if read:
            rows.append({"asset": u["sym"], **read})
    return pd.DataFrame(rows)
