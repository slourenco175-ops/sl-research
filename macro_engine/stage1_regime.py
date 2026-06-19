"""Stage 1 — macro state.

Four binary pillars (Growth / Inflation / Rates / Liquidity) → a 4-bit code
(16 states). For Stage 2 pooling we also collapse to a 2x2 super-regime on
growth × inflation. Emits, as the design doc requires, not just the discrete
label but the transition matrix, P(transition) over 1-3m, and per-pillar
nowcast value + trend.

Note (Phase 1): pillars are computed on as-reported FRED series without an
explicit publication lag. Macro releases lag 1-2 months, so the *latest* state
is near-real-time but the labelled history has mild lookahead. Point-in-time
lagging is a Phase 2 rigor item (see design doc §6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_engine.config import PILLARS, REGIME_RESAMPLE
from macro_engine.data import fred_series

SUPER = {
    (1, 1): "Reflation",   # growth up, inflation up
    (1, 0): "Goldilocks",  # growth up, inflation down
    (0, 1): "Stagflation", # growth down, inflation up
    (0, 0): "Slowdown",    # growth down, inflation down
}
SUPER_ORDER = ["Goldilocks", "Reflation", "Stagflation", "Slowdown"]


def _monthly(s: pd.Series) -> pd.Series:
    return s.resample(REGIME_RESAMPLE).last().dropna()


def _pillar_value(name: str, cfg: dict) -> pd.Series:
    """Native-frequency nowcast value for a pillar (the number, not the bit)."""
    raw = fred_series(cfg["fred"][0])
    if raw.empty:
        return raw
    m = _monthly(raw)
    method = cfg["method"]
    if method in ("yoy_trend", "yoy_rising"):
        return (m / m.shift(12) - 1.0) * 100.0   # YoY %
    return m                                      # level (rates, liquidity)


def _pillar_bits(values: dict[str, pd.Series]) -> pd.DataFrame:
    """Align pillar nowcasts on a common monthly index and derive binary bits."""
    df = pd.DataFrame({k: v for k, v in values.items() if not v.empty}).sort_index()
    df = df.ffill().dropna()
    bits = pd.DataFrame(index=df.index)

    # growth: YoY momentum positive (YoY rising over 3m)
    if "growth" in df:
        bits["growth"] = (df["growth"].diff(3) > 0).astype(int)
    # inflation: YoY above target band OR re-accelerating over 6m
    if "inflation" in df:
        tgt = PILLARS["inflation"].get("target_yoy", 2.5)
        bits["inflation"] = ((df["inflation"] > tgt) | (df["inflation"].diff(6) > 0)).astype(int)
    # rates: 2y yield rising over 3m (tightening impulse)
    if "rates" in df:
        bits["rates"] = (df["rates"].diff(3) > 0).astype(int)
    # liquidity: NFCI positive = tight conditions
    if "liquidity" in df:
        bits["liquidity"] = (df["liquidity"] > 0).astype(int)
    return bits.dropna(), df


def _state_code(row: pd.Series, cols: list[str]) -> str:
    """4-bit state code, e.g. 'G+/I-/R+/L-' (pillar order from config)."""
    return "/".join(f"{c[0].upper()}{'+' if int(row[c]) else '-'}" for c in cols)


def _state_to_super(code: str) -> str:
    """Map a full 16-state code to its 2x2 super-regime parent (growth × inflation).

    Segments look like 'G+', 'I-', 'R+', 'L-' — only G and I select the parent.
    """
    bits = {seg[0]: (seg[1] == "+") for seg in code.split("/")}
    return SUPER[(int(bits.get("G", 0)), int(bits.get("I", 0)))]


def _transition_matrix(seq: pd.Series) -> pd.DataFrame:
    """Row-normalized 1-step empirical transition matrix over super-regimes."""
    states = SUPER_ORDER
    M = pd.DataFrame(0.0, index=states, columns=states)
    s = seq.dropna().tolist()
    for a, b in zip(s[:-1], s[1:]):
        if a in states and b in states:
            M.loc[a, b] += 1.0
    M = M.div(M.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return M


def _trend(series: pd.Series) -> str:
    if len(series) < 4:
        return "flat"
    chg = series.iloc[-1] - series.iloc[-4]
    if chg > 0.05:
        return "rising"
    if chg < -0.05:
        return "falling"
    return "flat"


def detect_regime() -> dict:
    values = {name: _pillar_value(name, cfg) for name, cfg in PILLARS.items()}
    bits, vals = _pillar_bits(values)
    if bits.empty:
        raise SystemExit("Stage 1: no pillar data — check FRED access.")

    latest = bits.iloc[-1]
    state_bits = {k: int(latest[k]) for k in bits.columns}

    labels = {
        k: (PILLARS[k]["hot_label"] if state_bits.get(k) == 1 else PILLARS[k]["cold_label"])
        for k in bits.columns
    }
    cols = list(bits.columns)
    code = _state_code(latest, cols)

    # Full 16-state history (for Stage 2 hierarchical pooling) and its parent
    # super-regime history (for Stage 2 prior + transition stats).
    monthly_state = bits.apply(lambda r: _state_code(r, cols), axis=1)
    super_seq = monthly_state.map(_state_to_super)
    current_super = super_seq.iloc[-1]

    M = _transition_matrix(super_seq)
    next_1m = M.loc[current_super].to_dict()
    M3 = pd.DataFrame(np.linalg.matrix_power(M.values, 3), index=M.index, columns=M.columns)
    next_3m = M3.loc[current_super].to_dict()
    p_change_1m = float(1.0 - next_1m.get(current_super, 0.0))
    p_change_3m = float(1.0 - next_3m.get(current_super, 0.0))

    nowcast = {}
    for k in bits.columns:
        v = vals[k]
        nowcast[k] = {"value": round(float(v.iloc[-1]), 2), "trend": _trend(v)}

    return {
        "asof": bits.index[-1].strftime("%Y-%m-%d"),
        "state_bits": state_bits,
        "state_code": code,
        "labels": labels,
        "super_regime": current_super,
        "nowcast": nowcast,
        "monthly_super": super_seq,          # date -> super_regime (Phase 1 pooling)
        "monthly_state": monthly_state,      # date -> full 16-state code (Phase 2 pooling)
        "state_to_super": _state_to_super,   # code -> super_regime parent
        "transition": {
            "matrix": M,
            "next_1m": next_1m,
            "next_3m": next_3m,
            "p_change_1m": p_change_1m,
            "p_change_3m": p_change_3m,
            "most_likely_next": max(
                {s: p for s, p in next_3m.items() if s != current_super}.items(),
                key=lambda kv: kv[1], default=(current_super, 0.0),
            )[0],
        },
    }
