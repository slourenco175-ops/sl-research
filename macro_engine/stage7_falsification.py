"""Stage 7 — falsification + journal.

Pre-register kill conditions before an idea goes live — the structural defence
against confirmation bias. Each idea gets a price stop (ATR-based), a macro-data
invalidation derived from the pillar its divergence rests on, and a time stop.
A decision-journal line is appended at generation for the post-mortem loop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from macro_engine.config import (ATR_STOP_MULT, DEFAULT_TIME_STOP_DAYS,
                                  JOURNAL_PATH)
from macro_engine.data import get_universe_prices

MACRO_INVALIDATION = {
    "inflation": "thesis dies if CPI/PCE YoY reverses its current trend for 2 consecutive prints",
    "growth":    "thesis dies if INDPRO YoY flips sign vs the current nowcast",
    "rates":     "thesis dies if the 2y yield round-trips the move / Fed guidance reverses",
}


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return float("nan")
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def falsify(sized: pd.DataFrame) -> pd.DataFrame:
    if sized.empty:
        return sized
    prices = get_universe_prices()
    out = []
    for _, r in sized.iterrows():
        df = prices.get(r["asset"])
        entry = float(df["Close"].iloc[-1]) if df is not None and not df.empty else float("nan")
        atr = _atr(df)
        sign = 1.0 if r["direction"] == "LONG" else -1.0
        if not np.isnan(entry) and not np.isnan(atr):
            stop = entry - sign * ATR_STOP_MULT * atr
            price_stop = f"{stop:.2f} ({ATR_STOP_MULT:g}×ATR {'below' if sign > 0 else 'above'} {entry:.2f})"
        else:
            price_stop = "—"

        pillars = []
        detail = r.get("_div_detail")
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            for p in ",".join(detail["supporting_pillars"].astype(str)).split(","):
                if p and p not in pillars:
                    pillars.append(p)
        macro_inval = "; ".join(MACRO_INVALIDATION[p] for p in pillars if p in MACRO_INVALIDATION) or "—"

        out.append({
            **{k: v for k, v in r.to_dict().items() if k != "_div_detail"},
            "entry": round(entry, 2) if not np.isnan(entry) else None,
            "price_stop": price_stop,
            "macro_invalidation": macro_inval,
            "time_stop_days": DEFAULT_TIME_STOP_DAYS,
        })
    return pd.DataFrame(out)


def write_journal(sheet: pd.DataFrame, regime: dict) -> None:
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(JOURNAL_PATH, "a") as f:
        for _, r in sheet[sheet.get("surfaced", True)].iterrows():
            entry = {
                "ts": ts,
                "regime": regime["super_regime"],
                "state_code": regime["state_code"],
                "asset": r["asset"], "direction": r["direction"],
                "convergence_score": r["convergence_score"],
                "agreement_count": r["agreement_count"],
                "instrument": r.get("instrument"),
                "suggested_weight_pct": r.get("suggested_weight_pct"),
                "price_stop": r.get("price_stop"),
                "macro_invalidation": r.get("macro_invalidation"),
                "time_stop_days": r.get("time_stop_days"),
            }
            f.write(json.dumps(entry) + "\n")
