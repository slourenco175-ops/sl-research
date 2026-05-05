"""Rules-based conviction scorer for FX pairs.

Blends pair technicals, COT extremes, and base-vs-quote fundamentals deltas
into a single signed score, then maps to a verdict. Every contributing rule
is appended to `reasons` so the dashboard can show *why* a pair scores where
it does — same explicit pattern as `commodities/scoring.py`.
"""
from __future__ import annotations

import pandas as pd


def score_row(row: dict, macro: dict) -> tuple[float, list[str], str]:
    score = 0.0
    reasons: list[str] = []

    # --- pair technical trend (ST/MT/LT) ---
    for h, key in [("st", "trend_st"), ("mt", "trend_mt"), ("lt", "trend_lt")]:
        v = row.get(key)
        if v == "UP":
            score += 1; reasons.append(f"+1 trend_{h}=UP")
        elif v == "DOWN":
            score -= 1; reasons.append(f"-1 trend_{h}=DOWN")

    # --- breakout ---
    bo = row.get("breakout_20d")
    if bo == "BREAK UP":
        score += 1; reasons.append("+1 20d breakout up")
    elif bo == "BREAK DOWN":
        score -= 1; reasons.append("-1 20d breakdown")

    # --- MACD cross (heavier weight than steady-state) ---
    macd = row.get("macd")
    if macd == "BULL CROSS":
        score += 1.5; reasons.append("+1.5 MACD bull cross")
    elif macd == "BEAR CROSS":
        score -= 1.5; reasons.append("-1.5 MACD bear cross")

    # --- CTA / trend follower positioning ---
    cta = row.get("cta_score", 0)
    if cta >= 60:
        score += 1; reasons.append("+1 CTAs heavy long")
    elif cta <= -60:
        score -= 1; reasons.append("-1 CTAs heavy short")

    # --- COT extremes on BASE ccy futures (contrarian fade) ---
    mm_pct = row.get("mm_3y_percentile")
    if pd.notna(mm_pct):
        if mm_pct >= 90:
            score -= 1; reasons.append(f"-1 LevMoney crowded long base ({mm_pct:.0f}%ile)")
        elif mm_pct <= 10:
            score += 1; reasons.append(f"+1 LevMoney crowded short base ({mm_pct:.0f}%ile)")

    traj = row.get("mm_trajectory")
    mm_4w = row.get("mm_4w_change", 0) or 0
    if traj == "REVERSING":
        if mm_4w > 0:
            score += 0.5; reasons.append("+0.5 LevMoney reversing higher")
        else:
            score -= 0.5; reasons.append("-0.5 LevMoney reversing lower")

    # --- carry differential (10y base − 10y quote) ---
    carry_diff = row.get("carry_diff")
    if pd.notna(carry_diff):
        if carry_diff >= 2.0:
            score += 1.5; reasons.append(f"+1.5 carry diff {carry_diff:+.2f}% (base pays)")
        elif carry_diff >= 0.75:
            score += 0.75; reasons.append(f"+0.75 carry diff {carry_diff:+.2f}% (mild base advantage)")
        elif carry_diff <= -2.0:
            score -= 1.5; reasons.append(f"-1.5 carry diff {carry_diff:+.2f}% (quote pays)")
        elif carry_diff <= -0.75:
            score -= 0.75; reasons.append(f"-0.75 carry diff {carry_diff:+.2f}% (mild quote advantage)")

    # --- real-rate differential ---
    rr_diff = row.get("rr_diff")
    if pd.notna(rr_diff):
        if rr_diff >= 1.5:
            score += 1; reasons.append(f"+1 real-rate diff {rr_diff:+.2f}% favors base")
        elif rr_diff <= -1.5:
            score -= 1; reasons.append(f"-1 real-rate diff {rr_diff:+.2f}% favors quote")

    # --- growth differential (GDP YoY) ---
    gdp_diff = row.get("gdp_diff")
    if pd.notna(gdp_diff):
        if gdp_diff >= 1.0:
            score += 0.5; reasons.append(f"+0.5 growth gap {gdp_diff:+.2f}pp favors base")
        elif gdp_diff <= -1.0:
            score -= 0.5; reasons.append(f"-0.5 growth gap {gdp_diff:+.2f}pp favors quote")

    # --- unemployment differential (lower unemp = stronger; sign flipped) ---
    unemp_diff = row.get("unemp_diff")
    if pd.notna(unemp_diff):
        if unemp_diff <= -1.5:
            score += 0.5; reasons.append(f"+0.5 base unemp {unemp_diff:+.2f}pp lower")
        elif unemp_diff >= 1.5:
            score -= 0.5; reasons.append(f"-0.5 base unemp {unemp_diff:+.2f}pp higher")

    # --- composite country health delta ---
    health_diff = row.get("health_diff")
    if pd.notna(health_diff):
        if health_diff >= 25:
            score += 1; reasons.append(f"+1 base fundamentals stronger (Δhealth {health_diff:+.0f})")
        elif health_diff <= -25:
            score -= 1; reasons.append(f"-1 quote fundamentals stronger (Δhealth {health_diff:+.0f})")

    # --- commodity overlay (base − quote tailwind) ---
    comm_diff = row.get("comm_diff", 0) or 0
    if comm_diff >= 0.15:
        score += 0.75; reasons.append(f"+0.75 commodity tailwind for base ({comm_diff:+.2f})")
    elif comm_diff <= -0.15:
        score -= 0.75; reasons.append(f"-0.75 commodity tailwind for quote ({comm_diff:+.2f})")

    # --- seasonality ---
    s_avg = row.get("seas_avg_pct", 0) or 0
    s_hit = row.get("seas_hit_rate", 0.5) or 0.5
    if s_avg >= 0.5 and s_hit >= 0.6:
        score += 0.5; reasons.append(f"+0.5 seasonal tailwind (+{s_avg:.1f}%, hit {s_hit*100:.0f}%)")
    elif s_avg <= -0.5 and s_hit <= 0.4:
        score -= 0.5; reasons.append(f"-0.5 seasonal headwind ({s_avg:.1f}%, hit {s_hit*100:.0f}%)")

    # --- macro overlay: VIX risk-off lifts USD/CHF/JPY ---
    vix = macro.get("VIX", {})
    if vix and vix.get("pctile_3y", 50) >= 80:
        base = row.get("base"); quote = row.get("quote")
        safe_havens = {"USD", "CHF", "JPY"}
        if base in safe_havens and quote not in safe_havens:
            score += 0.5; reasons.append(f"+0.5 VIX {vix['pctile_3y']:.0f}%ile, base is safe-haven")
        elif quote in safe_havens and base not in safe_havens:
            score -= 0.5; reasons.append(f"-0.5 VIX {vix['pctile_3y']:.0f}%ile, quote is safe-haven")

    # --- vol regime damps conviction in extreme vol ---
    if row.get("vol_regime") == "EXTREME":
        score *= 0.7
        reasons.append("vol regime EXTREME (conviction reduced 30%)")

    # --- verdict ---
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
