"""Assemble the final ranked trade sheet (design doc §3 schema)."""
from __future__ import annotations

import pandas as pd


def _thesis(r: pd.Series, regime: dict) -> str:
    d = r["direction"].lower()
    sh = r.get("regime_sharpe")
    sh_txt = f"Sharpe {sh:+.2f} in {regime['super_regime']}" if pd.notna(sh) else regime["super_regime"]
    nd = int(r.get("n_divergences", 0))
    return f"{d.title()} {r['name']} — {sh_txt}; {nd} divergence(s) aligned, {r['agreement_count']} lenses agree"


def _catalyst(r: pd.Series, regime: dict) -> str:
    detail = r.get("_div_detail")
    divs = []
    if isinstance(detail, pd.DataFrame) and not detail.empty:
        divs = sorted(detail["divergence"].unique().tolist())
    parts = []
    if divs:
        parts.append(" / ".join(divs) + " gap closes")
    tr = regime["transition"]
    parts.append(f"regime P(→{tr['most_likely_next']}) {tr['next_3m'].get(tr['most_likely_next'], 0):.0%} over 3m")
    return "; ".join(parts)


def build_sheet(falsified: pd.DataFrame, regime: dict) -> pd.DataFrame:
    if falsified.empty:
        return falsified
    rows = []
    for i, r in falsified.reset_index(drop=True).iterrows():
        size = (f'{r.get("suggested_weight_pct", 0):+.1f}% '
                f'(~{r.get("est_vol_contribution_pct", 0):.1f}% vol)')
        rows.append({
            "id": f"M{i+1:02d}",
            "thesis": _thesis(r, regime),
            "direction": r["direction"],
            "asset": r["asset"],
            "instrument": r.get("instrument", "—"),
            "regime_score": r["regime_score"],
            "regime_confidence": r.get("regime_confidence", "n/a"),
            "divergence_z": r["divergence_z"],
            "crowding": r["crowding_score"],
            "convergence_score": r["convergence_score"],
            "agreement_count": r["agreement_count"],
            "catalyst": _catalyst(r, regime),
            "invalidation": f'{r.get("price_stop", "—")} · {r.get("macro_invalidation", "—")} · {r.get("time_stop_days")}d',
            "horizon": r.get("horizon", "1-3m"),
            "suggested_size": size,
            "suggested_weight_pct": r.get("suggested_weight_pct"),
            "time_stop_days": r.get("time_stop_days"),
            "surfaced": r["surfaced"],
            "conflict": r["conflict"],
            "reversal_flag": r["reversal_flag"],
            "entry": r.get("entry"),
            "price_stop": r.get("price_stop"),
            "macro_invalidation": r.get("macro_invalidation"),
            "_div_detail": r.get("_div_detail"),
        })
    return pd.DataFrame(rows)
