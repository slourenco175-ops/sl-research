"""Fed policy-path snapshots — seed of Stage 3 (divergence scanner).

Point-in-time record of each FOMC decision: the statement bias, the dot-plot
median, and the curve's reaction. This is the raw material the policy-path
divergence node consumes (market-implied path vs. our nowcast). It is stored
dated and append-only so there is no lookahead: each entry reflects only what
was known on its meeting date.

Schema per meeting (maps toward the Stage 3 output schema in the design doc):
  meeting        — FOMC decision date (ISO)
  chair          — presiding chair
  decision       — action taken
  target_range   — funds-rate target after the meeting (%)
  statement_bias — "easing" | "neutral" | "tightening"
  dots_median_cy — SEP median funds rate for the current year (%)
  hike_votes     — participants projecting a hike this year, "n/total"
  curve_bp       — same-day yield change by tenor (bp), our FMP treasury pull
  pillars        — regime pillar flags this print implies (Stage 1 hand-off)
  divergence     — qualitative policy-path read until the pipeline z-scores it
  invalidation   — Stage 7 seed: what would kill the implied stance

gap_z is intentionally absent: it requires a trailing window the Stage 3 node
computes against the SOFR/FF strip. Recorded as None here, filled by the engine.
"""
from __future__ import annotations

# Append-only, newest last. Keyed by meeting date.
SNAPSHOTS: dict[str, dict] = {
    "2026-06-17": {
        "meeting": "2026-06-17",
        "chair": "Warsh",
        "decision": "hold (4th consecutive)",
        "target_range": (3.50, 3.75),
        "statement_bias": "neutral",  # was "easing"; removed "additional rate adjustments"
        "dots_median_cy": 3.8,         # up from 3.4 in March SEP — swung from a cut to a hike
        "hike_votes": "9/18",          # chair (Warsh) submitted no projection
        "inflation_yoy": 4.2,
        "curve_bp": {                  # 6/16 -> 6/17 close, FMP treasury-rates
            "1y": 14, "2y": 15, "3y": 15, "5y": 11,
            "10y": 6, "30y": 0,
        },
        "curve_levels": {              # (prior 6/16, post 6/17) yield %, FMP treasury-rates
            "1m": (3.67, 3.68), "2m": (3.71, 3.74), "3m": (3.79, 3.83),
            "6m": (3.81, 3.91), "1y": (3.84, 3.98), "2y": (4.05, 4.20),
            "3y": (4.08, 4.23), "5y": (4.16, 4.27), "7y": (4.28, 4.37),
            "10y": (4.43, 4.49), "20y": (4.92, 4.95), "30y": (4.93, 4.93),
        },
        "curve_shape": "bear_flattener",   # front/belly repriced hike, long end anchored
        "spread_2s10s_bp": (38, 29),       # (prior, post) — flattened 9bp
        "pillars": {
            "rates": "tightening-bias",
            "inflation": "high-sticky",
            "liquidity": "unchanged",
        },
        "divergence": {
            "name": "policy_path",
            "asset": "front-end rates / USD",
            "gap_raw": "market had cuts priced; Fed + curve repriced toward a hike",
            "gap_z": None,                 # filled by Stage 3 vs SOFR/FF strip
            "persistence_days": 0,         # gap resolved on the day, not a standing mispricing
            "direction": "resolved",       # not an entry — fade residual cut-priced longs
            "supporting_pillars": ["rates", "inflation"],
            "note": (
                "Long-end unchanged (30y +0bp) = market credits Warsh's 2% "
                "credibility pitch. Hawkish and believed."
            ),
        },
        "invalidation": "stance softens if core PCE prints below 3.0% twice, or the energy spike unwinds",
        "fx_note": "FX feed tier-gated; reporting had DXY ~99.5, EURUSD struggling >1.16, USDJPY ~160",
    },
}


def latest() -> dict:
    """Most recent FOMC snapshot."""
    return SNAPSHOTS[max(SNAPSHOTS)]


def get_policy_path() -> dict:
    """All recorded snapshots, keyed by meeting date (append-only history)."""
    return SNAPSHOTS
