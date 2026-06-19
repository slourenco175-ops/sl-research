"""SL Research — Macro Idea Engine.

A convergence-based macro idea generator: regime read (Stage 1) → conditional
base-rate returns (Stage 2) → market-vs-view divergences (Stage 3) → positioning
overlay (Stage 4) → convergence scoring (Stage 5) → expression + sizing (Stage 6)
→ falsification + journal (Stage 7) → ranked trade sheet.

Phase 1 (this build): runs end-to-end on free data (FRED + yfinance + CFTC),
super-regime pooling in Stage 2, three divergences in Stage 3, simple sizing.
Research only. Does NOT trade.
"""
