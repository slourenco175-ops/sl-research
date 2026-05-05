"""Rules-based conviction scorer.

Given a row of factors + positioning + macro context, returns:
  score   : float (sum of fired rule weights)
  reasons : list[str] of rule descriptions
  verdict : BULLISH / MILDLY BULLISH / NEUTRAL / MILDLY BEARISH / BEARISH

This is intentionally explicit (every contributing rule is listed in the
dashboard card) so you can argue with it instead of trusting a black box.
"""
from __future__ import annotations

import pandas as pd


def score_row(row: dict, macro: dict) -> tuple[float, list[str], str]:
    score = 0.0
    reasons: list[str] = []

    # --- trend (ST/MT/LT)
    for h, key in [("st", "trend_st"), ("mt", "trend_mt"), ("lt", "trend_lt")]:
        v = row.get(key)
        if v == "UP":
            score += 1; reasons.append(f"+1 trend_{h}=UP")
        elif v == "DOWN":
            score -= 1; reasons.append(f"-1 trend_{h}=DOWN")

    # --- vol-scaled trend strength: continuous magnitude on top of buckets ---
    # Each z-score is in [-3, +3]; weight roughly matches the bucket weights.
    z3 = row.get("mom_3m_z", 0) or 0
    z12 = row.get("mom_12m_z", 0) or 0
    if abs(z3) >= 0.5:
        contrib = round(0.5 * z3, 2)
        score += contrib; reasons.append(f"{contrib:+.2f} 3m vol-scaled trend ({z3:+.2f}σ)")
    if abs(z12) >= 0.5:
        contrib = round(0.5 * z12, 2)
        score += contrib; reasons.append(f"{contrib:+.2f} 12m vol-scaled trend ({z12:+.2f}σ)")

    # --- breakout
    bo = row.get("breakout_20d")
    if bo == "BREAK UP":
        score += 1; reasons.append("+1 20d breakout up")
    elif bo == "BREAK DOWN":
        score -= 1; reasons.append("-1 20d breakdown")

    # --- MACD cross (heavier weight than steady-state)
    macd = row.get("macd")
    if macd == "BULL CROSS":
        score += 1.5; reasons.append("+1.5 MACD bull cross")
    elif macd == "BEAR CROSS":
        score -= 1.5; reasons.append("-1.5 MACD bear cross")

    # --- CTA / trend follower positioning
    cta = row.get("cta_score", 0)
    if cta >= 60:
        score += 1; reasons.append("+1 CTAs heavy long")
    elif cta <= -60:
        score -= 1; reasons.append("-1 CTAs heavy short")

    # --- COT extremes (contrarian fade)
    mm_pct = row.get("mm_3y_percentile")
    if pd.notna(mm_pct):
        if mm_pct >= 90:
            score -= 1; reasons.append(f"-1 MM crowded long ({mm_pct:.0f}%ile)")
        elif mm_pct <= 10:
            score += 1; reasons.append(f"+1 MM crowded short ({mm_pct:.0f}%ile)")

    # --- COT trajectory
    traj = row.get("mm_trajectory")
    mm_4w = row.get("mm_4w_change", 0) or 0
    if traj == "REVERSING":
        if mm_4w > 0:
            score += 0.5; reasons.append("+0.5 MM reversing higher")
        else:
            score -= 0.5; reasons.append("-0.5 MM reversing lower")

    # --- commercials net long is bullish (smart hedgers buying)
    comm_pct = row.get("comm_3y_percentile")
    if pd.notna(comm_pct):
        if comm_pct >= 80:
            score += 1; reasons.append(f"+1 Commercials net long ({comm_pct:.0f}%ile)")
        elif comm_pct <= 20:
            score -= 1; reasons.append(f"-1 Commercials hedging ({comm_pct:.0f}%ile)")

    # --- real curve slope (preferred when available) ---
    # When the deferred-month curve fetch succeeded we score the real annualized
    # roll yield; otherwise fall back to the spot-vs-252d carry proxy below.
    curve_slope = row.get("curve_slope_pct_yr")
    if pd.notna(curve_slope):
        if curve_slope <= -8:
            score += 1.5; reasons.append(f"+1.5 curve steeply backwardated ({curve_slope:+.1f}%/yr)")
        elif curve_slope <= -3:
            score += 1.0; reasons.append(f"+1 curve backwardated ({curve_slope:+.1f}%/yr)")
        elif curve_slope >= 8:
            score -= 1.0; reasons.append(f"-1 curve steeply contangoed ({curve_slope:+.1f}%/yr)")
        elif curve_slope >= 3:
            score -= 0.5; reasons.append(f"-0.5 curve contangoed ({curve_slope:+.1f}%/yr)")
    else:
        # --- carry (proxy) — used only if real curve unavailable
        cstate = row.get("carry_state")
        if cstate == "BACKWARDATION":
            score += 1; reasons.append("+1 Backwardated curve (proxy)")
        elif cstate == "CONTANGO":
            score -= 0.5; reasons.append("-0.5 Contango drag (proxy)")

    # --- inventory seasonal-z (only set for energy contracts with EIA data) ---
    inv_z = row.get("inv_seasonal_z")
    if pd.notna(inv_z):
        if inv_z <= -1.5:
            score += 1.5; reasons.append(f"+1.5 inventories deep below seasonal norm (z {inv_z:+.2f})")
        elif inv_z <= -0.75:
            score += 1.0; reasons.append(f"+1 inventories below seasonal norm (z {inv_z:+.2f})")
        elif inv_z >= 1.5:
            score -= 1.5; reasons.append(f"-1.5 inventories deep above seasonal norm (z {inv_z:+.2f})")
        elif inv_z >= 0.75:
            score -= 1.0; reasons.append(f"-1 inventories above seasonal norm (z {inv_z:+.2f})")

    # --- inventory build vs typical: "is this week's draw bigger/smaller than usual?"
    inv_b4n = row.get("inv_build_4w_norm")
    if pd.notna(inv_b4n):
        # We don't have a universal threshold per commodity — use level-relative scaling.
        # Convert the build-vs-norm into a fraction of the latest level.
        inv_lvl = row.get("inv_level") or 0
        if inv_lvl > 0:
            frac = inv_b4n / inv_lvl  # e.g. 0.04 = 4% of stockpile bigger build than typical
            if frac <= -0.04:
                score += 0.5; reasons.append(f"+0.5 4w draw bigger than seasonal norm ({frac*100:+.1f}%)")
            elif frac >= 0.04:
                score -= 0.5; reasons.append(f"-0.5 4w build bigger than seasonal norm ({frac*100:+.1f}%)")

    # --- seasonality
    s_avg = row.get("seas_avg_pct", 0) or 0
    s_hit = row.get("seas_hit_rate", 0.5) or 0.5
    if s_avg >= 1.0 and s_hit >= 0.6:
        score += 0.5; reasons.append(f"+0.5 seasonal tailwind (+{s_avg:.1f}%, hit {s_hit*100:.0f}%)")
    elif s_avg <= -1.0 and s_hit <= 0.4:
        score -= 0.5; reasons.append(f"-0.5 seasonal headwind ({s_avg:.1f}%, hit {s_hit*100:.0f}%)")

    # --- macro overlay (USD direction)
    dxy = macro.get("DXY", {})
    if dxy:
        if dxy.get("direction") == "DOWN":
            score += 0.5; reasons.append("+0.5 DXY falling (USD tailwind)")
        elif dxy.get("direction") == "UP":
            score -= 0.5; reasons.append("-0.5 DXY rising (USD headwind)")

    # --- vol regime damps conviction in extreme vol
    if row.get("vol_regime") == "EXTREME":
        score *= 0.7
        reasons.append("vol regime EXTREME (conviction reduced 30%)")

    # --- vol-ratio damp: if today's vol is N× the 5y median, shrink the score by 1/N.
    # CTA recipe — equates conviction across assets at different vol regimes.
    vr = row.get("vol_ratio_5y", 1.0) or 1.0
    if vr > 1.2:
        damp = 1.0 / min(vr, 3.0)
        score *= damp
        reasons.append(f"vol {vr:.2f}× 5y norm → conviction × {damp:.2f}")

    # --- verdict
    if score >= 5:
        verdict = "BULLISH"
    elif score >= 2:
        verdict = "MILDLY BULLISH"
    elif score <= -5:
        verdict = "BEARISH"
    elif score <= -2:
        verdict = "MILDLY BEARISH"
    else:
        verdict = "NEUTRAL"

    return round(score, 1), reasons, verdict


VERDICT_COLOR = {
    "BULLISH": "#0a7d28",
    "MILDLY BULLISH": "#5cb85c",
    "NEUTRAL": "#888",
    "MILDLY BEARISH": "#e8a33d",
    "BEARISH": "#d9534f",
}
