"""Fair-value-first conviction scorer for FX pairs.

The model's primary question is: *what does the data say the pair should be
doing?* Fundamentals (carry, real rates, growth, unemployment, inflation
differential, country health composite, commodity overlay) drive the FV
score, which alone determines the verdict. Technicals (trend, MACD, RSI,
breakout) and 5y valuation distance are tracked separately as a confirmation
/ timing layer that adjusts the *composite* used for ranking — but never the
verdict label itself.

This split lets you read the dashboard two ways:
  1. "Is the model saying this currency is mispriced?" → look at FV bias.
  2. "Is price action confirming or fighting the model?" → compare tech_score
     against fv_score, and check the valuation bucket (CHEAP/FAIR/RICH).

Every contributing rule is appended to `reasons` so the dashboard can show
*why* a pair scores where it does — same explicit pattern as the commodities
scorer.
"""
from __future__ import annotations

import pandas as pd


def _fv_verdict(fv_score: float) -> str:
    if fv_score >= 5:
        return "BULLISH"
    if fv_score >= 2:
        return "MILDLY BULLISH"
    if fv_score <= -5:
        return "BEARISH"
    if fv_score <= -2:
        return "MILDLY BEARISH"
    return "NEUTRAL"


def _fv_bias_label(fv_score: float) -> str:
    """Short label for the FV column. Says which leg of the pair fundamentals favor."""
    if fv_score >= 4:
        return "STRONG BASE"
    if fv_score >= 1.5:
        return "MILD BASE"
    if fv_score <= -4:
        return "STRONG QUOTE"
    if fv_score <= -1.5:
        return "MILD QUOTE"
    return "BALANCED"


def _valuation_label(val_z: float) -> str:
    if pd.isna(val_z):
        return "—"
    if val_z >= 1.5:
        return "RICH"
    if val_z >= 0.75:
        return "MILDLY RICH"
    if val_z <= -1.5:
        return "CHEAP"
    if val_z <= -0.75:
        return "MILDLY CHEAP"
    return "FAIR"


def score_row(row: dict, macro: dict) -> dict:
    """Returns: {score, fv_score, tech_score, val_score, cot_score,
                 verdict, fv_bias, valuation, reasons}."""
    fv_score = 0.0
    tech_score = 0.0
    val_score = 0.0
    cot_score = 0.0
    reasons: list[str] = []

    # =======================================================================
    # FUNDAMENTALS — drive the verdict
    # =======================================================================

    # --- carry differential (10y base − 10y quote) — heaviest single driver ---
    cd = row.get("carry_diff")
    if pd.notna(cd):
        if cd >= 3.0:
            fv_score += 2.5; reasons.append(f"+2.5 FV carry diff {cd:+.2f}% (large base advantage)")
        elif cd >= 1.5:
            fv_score += 1.5; reasons.append(f"+1.5 FV carry diff {cd:+.2f}% favors base")
        elif cd >= 0.5:
            fv_score += 0.75; reasons.append(f"+0.75 FV carry diff {cd:+.2f}% mild base edge")
        elif cd <= -3.0:
            fv_score -= 2.5; reasons.append(f"-2.5 FV carry diff {cd:+.2f}% (large quote advantage)")
        elif cd <= -1.5:
            fv_score -= 1.5; reasons.append(f"-1.5 FV carry diff {cd:+.2f}% favors quote")
        elif cd <= -0.5:
            fv_score -= 0.75; reasons.append(f"-0.75 FV carry diff {cd:+.2f}% mild quote edge")

    # --- real-rate differential ---
    rr = row.get("rr_diff")
    if pd.notna(rr):
        if rr >= 2.0:
            fv_score += 2.0; reasons.append(f"+2 FV real-rate diff {rr:+.2f}% favors base")
        elif rr >= 0.5:
            fv_score += 1.0; reasons.append(f"+1 FV real-rate diff {rr:+.2f}% mild base")
        elif rr <= -2.0:
            fv_score -= 2.0; reasons.append(f"-2 FV real-rate diff {rr:+.2f}% favors quote")
        elif rr <= -0.5:
            fv_score -= 1.0; reasons.append(f"-1 FV real-rate diff {rr:+.2f}% mild quote")

    # --- growth differential (real GDP YoY) ---
    gd = row.get("gdp_diff")
    if pd.notna(gd):
        if gd >= 1.5:
            fv_score += 1.0; reasons.append(f"+1 FV growth gap {gd:+.2f}pp favors base")
        elif gd >= 0.5:
            fv_score += 0.5; reasons.append(f"+0.5 FV growth gap {gd:+.2f}pp")
        elif gd <= -1.5:
            fv_score -= 1.0; reasons.append(f"-1 FV growth gap {gd:+.2f}pp favors quote")
        elif gd <= -0.5:
            fv_score -= 0.5; reasons.append(f"-0.5 FV growth gap {gd:+.2f}pp")

    # --- unemployment differential (lower base unemp = stronger base) ---
    ud = row.get("unemp_diff")
    if pd.notna(ud):
        if ud <= -2.0:
            fv_score += 0.75; reasons.append(f"+0.75 FV base unemp {ud:+.2f}pp lower")
        elif ud <= -0.5:
            fv_score += 0.5; reasons.append(f"+0.5 FV base unemp {ud:+.2f}pp lower")
        elif ud >= 2.0:
            fv_score -= 0.75; reasons.append(f"-0.75 FV base unemp {ud:+.2f}pp higher")
        elif ud >= 0.5:
            fv_score -= 0.5; reasons.append(f"-0.5 FV base unemp {ud:+.2f}pp higher")

    # --- inflation differential (PPP drag) ---
    cpi_d = row.get("cpi_diff")
    if pd.notna(cpi_d):
        # Positive cpi_diff means base ccy has higher inflation → erodes purchasing power → bearish base.
        if cpi_d >= 2.0:
            fv_score -= 0.75; reasons.append(f"-0.75 FV base inflation {cpi_d:+.2f}pp higher (PPP drag)")
        elif cpi_d >= 1.0:
            fv_score -= 0.5; reasons.append(f"-0.5 FV base inflation {cpi_d:+.2f}pp higher")
        elif cpi_d <= -2.0:
            fv_score += 0.75; reasons.append(f"+0.75 FV base inflation {cpi_d:+.2f}pp lower (PPP tailwind)")
        elif cpi_d <= -1.0:
            fv_score += 0.5; reasons.append(f"+0.5 FV base inflation {cpi_d:+.2f}pp lower")

    # --- country health composite (catches anything the explicit rules miss) ---
    hd = row.get("health_diff")
    if pd.notna(hd):
        if hd >= 30:
            fv_score += 1.0; reasons.append(f"+1 FV base fundamentals stronger (Δhealth {hd:+.0f})")
        elif hd >= 15:
            fv_score += 0.5; reasons.append(f"+0.5 FV base fundamentals mildly stronger (Δhealth {hd:+.0f})")
        elif hd <= -30:
            fv_score -= 1.0; reasons.append(f"-1 FV quote fundamentals stronger (Δhealth {hd:+.0f})")
        elif hd <= -15:
            fv_score -= 0.5; reasons.append(f"-0.5 FV quote fundamentals mildly stronger (Δhealth {hd:+.0f})")

    # --- commodity overlay (base − quote tailwind) ---
    cd2 = row.get("comm_diff", 0) or 0
    if cd2 >= 0.25:
        fv_score += 1.0; reasons.append(f"+1 FV strong commodity tailwind for base ({cd2:+.2f})")
    elif cd2 >= 0.10:
        fv_score += 0.5; reasons.append(f"+0.5 FV commodity tailwind for base ({cd2:+.2f})")
    elif cd2 <= -0.25:
        fv_score -= 1.0; reasons.append(f"-1 FV strong commodity tailwind for quote ({cd2:+.2f})")
    elif cd2 <= -0.10:
        fv_score -= 0.5; reasons.append(f"-0.5 FV commodity tailwind for quote ({cd2:+.2f})")

    # --- VIX-driven safe-haven kicker (still counts as fundamentals) ---
    vix = macro.get("VIX", {})
    if vix and vix.get("pctile_3y", 50) >= 80:
        base, quote = row.get("base"), row.get("quote")
        safe_havens = {"USD", "CHF", "JPY"}
        if base in safe_havens and quote not in safe_havens:
            fv_score += 0.5; reasons.append(f"+0.5 FV VIX {vix['pctile_3y']:.0f}%ile, base safe-haven")
        elif quote in safe_havens and base not in safe_havens:
            fv_score -= 0.5; reasons.append(f"-0.5 FV VIX {vix['pctile_3y']:.0f}%ile, quote safe-haven")

    # =======================================================================
    # VALUATION (5y mean-reversion anchor) — adjusts composite, not verdict
    # =======================================================================
    val_z = row.get("valuation_z")
    if pd.notna(val_z):
        if val_z >= 1.5:
            val_score -= 1.0; reasons.append(f"-1 VAL spot RICH vs 5y mean (z {val_z:+.2f}σ)")
        elif val_z >= 0.75:
            val_score -= 0.5; reasons.append(f"-0.5 VAL spot mildly rich (z {val_z:+.2f}σ)")
        elif val_z <= -1.5:
            val_score += 1.0; reasons.append(f"+1 VAL spot CHEAP vs 5y mean (z {val_z:+.2f}σ)")
        elif val_z <= -0.75:
            val_score += 0.5; reasons.append(f"+0.5 VAL spot mildly cheap (z {val_z:+.2f}σ)")

    # =======================================================================
    # COT — positioning extremes are a contrarian fade (composite-only)
    # =======================================================================
    mm_pct = row.get("mm_3y_percentile")
    if pd.notna(mm_pct):
        if mm_pct >= 90:
            cot_score -= 1.0; reasons.append(f"-1 COT LevMoney crowded long base ({mm_pct:.0f}%ile)")
        elif mm_pct <= 10:
            cot_score += 1.0; reasons.append(f"+1 COT LevMoney crowded short base ({mm_pct:.0f}%ile)")

    traj = row.get("mm_trajectory")
    mm_4w = row.get("mm_4w_change", 0) or 0
    if traj == "REVERSING":
        if mm_4w > 0:
            cot_score += 0.5; reasons.append("+0.5 COT LevMoney reversing higher")
        else:
            cot_score -= 0.5; reasons.append("-0.5 COT LevMoney reversing lower")

    # =======================================================================
    # TECHNICALS — confirmation/timing only (down-weighted in composite)
    # =======================================================================
    for h, key in [("st", "trend_st"), ("mt", "trend_mt"), ("lt", "trend_lt")]:
        v = row.get(key)
        if v == "UP":
            tech_score += 1.0; reasons.append(f"+1 TECH trend_{h}=UP")
        elif v == "DOWN":
            tech_score -= 1.0; reasons.append(f"-1 TECH trend_{h}=DOWN")

    # vol-scaled trend strength on the pair itself (clipped ±3σ)
    z3 = row.get("mom_3m_z", 0) or 0
    z12 = row.get("mom_12m_z", 0) or 0
    if abs(z3) >= 0.5:
        contrib = round(0.5 * z3, 2)
        tech_score += contrib; reasons.append(f"{contrib:+.2f} TECH 3m vol-scaled trend ({z3:+.2f}σ)")
    if abs(z12) >= 0.5:
        contrib = round(0.5 * z12, 2)
        tech_score += contrib; reasons.append(f"{contrib:+.2f} TECH 12m vol-scaled trend ({z12:+.2f}σ)")

    macd = row.get("macd")
    if macd == "BULL CROSS":
        tech_score += 1.5; reasons.append("+1.5 TECH MACD bull cross")
    elif macd == "BEAR CROSS":
        tech_score -= 1.5; reasons.append("-1.5 TECH MACD bear cross")

    bo = row.get("breakout_20d")
    if bo == "BREAK UP":
        tech_score += 1.0; reasons.append("+1 TECH 20d breakout up")
    elif bo == "BREAK DOWN":
        tech_score -= 1.0; reasons.append("-1 TECH 20d breakdown")

    rsi = row.get("rsi_14", 50) or 50
    if rsi >= 75:
        tech_score -= 0.75; reasons.append(f"-0.75 TECH RSI {rsi:.0f} overbought")
    elif rsi <= 25:
        tech_score += 0.75; reasons.append(f"+0.75 TECH RSI {rsi:.0f} oversold")

    # CTA flags as tech for the composite
    cta = row.get("cta_score", 0) or 0
    if cta >= 60:
        tech_score += 0.5; reasons.append("+0.5 TECH CTAs heavy long")
    elif cta <= -60:
        tech_score -= 0.5; reasons.append("-0.5 TECH CTAs heavy short")

    # Seasonality (very mild)
    s_avg = row.get("seas_avg_pct", 0) or 0
    s_hit = row.get("seas_hit_rate", 0.5) or 0.5
    if s_avg >= 0.5 and s_hit >= 0.6:
        tech_score += 0.25; reasons.append(f"+0.25 TECH seasonal tailwind (+{s_avg:.1f}%, hit {s_hit*100:.0f}%)")
    elif s_avg <= -0.5 and s_hit <= 0.4:
        tech_score -= 0.25; reasons.append(f"-0.25 TECH seasonal headwind ({s_avg:.1f}%, hit {s_hit*100:.0f}%)")

    # =======================================================================
    # Composite + verdict
    # =======================================================================
    composite = fv_score + val_score + cot_score + 0.4 * tech_score

    if row.get("vol_regime") == "EXTREME":
        composite *= 0.7
        reasons.append("vol regime EXTREME (composite reduced 30%)")

    # vol-ratio damp: if 60d vol is N× the 5y median, shrink the composite by 1/N
    vr = row.get("vol_ratio_5y", 1.0) or 1.0
    if vr > 1.2:
        damp = 1.0 / min(vr, 3.0)
        composite *= damp
        reasons.append(f"vol {vr:.2f}× 5y norm → composite × {damp:.2f}")

    return {
        "score": round(composite, 1),
        "fv_score": round(fv_score, 1),
        "tech_score": round(tech_score, 1),
        "val_score": round(val_score, 1),
        "cot_score": round(cot_score, 1),
        "verdict": _fv_verdict(fv_score),       # FV drives the label
        "fv_bias": _fv_bias_label(fv_score),
        "valuation": _valuation_label(row.get("valuation_z", float("nan"))),
        "reasons": reasons,
    }


VERDICT_COLOR = {
    "BULLISH": "#0a7d28",
    "MILDLY BULLISH": "#5cb85c",
    "NEUTRAL": "#888",
    "MILDLY BEARISH": "#e8a33d",
    "BEARISH": "#d9534f",
}
