"""Relative-value spreads — ratios and cracks across the commodity complex.

Each spread: current value, 3y z-score, 3y percentile, 1m % change, signal label.
ELEVATED = >70th pct, DEPRESSED = <30th pct, NEUTRAL otherwise.
"""
from __future__ import annotations

import pandas as pd

# (label, formula_callable_on_closes_dict, kind)
# kind: 'ratio' for A/B, 'crack' for HO*42 - CL etc. (we keep ratios for simplicity)
SPREAD_DEFS = [
    ("Brent-WTI",   ("BZ=F", "CL=F"), "ratio"),
    ("Gold/Silver", ("GC=F", "SI=F"), "ratio"),
    ("Gold/Copper", ("GC=F", "HG=F"), "ratio"),
    ("HO-CL Crack", ("HO=F", "CL=F"), "ratio"),
    ("RB-CL Crack", ("RB=F", "CL=F"), "diff"),    # gasoline - crude
    ("Corn-Wheat",  ("ZC=F", "ZW=F"), "ratio"),
    ("Soy-Corn",    ("ZS=F", "ZC=F"), "ratio"),
]


def _signal(pctile: float) -> str:
    if pctile >= 70:
        return "ELEVATED"
    if pctile <= 30:
        return "DEPRESSED"
    return "NEUTRAL"


def build_spreads(ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    closes = {s: df["Close"] for s, df in ohlc.items()}
    for label, (a, b), kind in SPREAD_DEFS:
        if a not in closes or b not in closes:
            continue
        ca, cb = closes[a].dropna(), closes[b].dropna()
        joined = pd.concat([ca, cb], axis=1, join="inner").dropna()
        if joined.empty or len(joined) < 90:
            continue
        joined.columns = ["a", "b"]
        spread = joined["a"] / joined["b"] if kind == "ratio" else joined["a"] - joined["b"]
        s3y = spread.iloc[-min(len(spread), 756):]  # last ~3y
        last = float(s3y.iloc[-1])
        mu, sd = s3y.mean(), s3y.std(ddof=0)
        z = float((last - mu) / sd) if sd > 0 else 0.0
        pct = float(s3y.rank(pct=True).iloc[-1] * 100)
        chg_1m = float(s3y.iloc[-1] / s3y.iloc[-21] - 1) * 100 if len(s3y) > 21 else 0.0
        rows.append({
            "spread": label,
            "last": last,
            "z_3y": z,
            "pctile_3y": pct,
            "chg_1m_pct": chg_1m,
            "signal": _signal(pct),
        })
    return pd.DataFrame(rows)
