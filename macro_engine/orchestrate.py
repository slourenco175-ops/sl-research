"""DAG runner for the macro idea engine.

Runs the stages in dependency order, assembles the ranked trade sheet, writes
the journal + CSV, renders the dashboard, and opens it.

    cd "/Users/samuellourenco/SL Research"
    source .venv/bin/activate
    python -m macro_engine.orchestrate

Research only. Does NOT trade.
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime

from macro_engine.dashboard import render
from macro_engine.portfolio import portfolio_risk
from macro_engine.sheet import build_sheet
from macro_engine.stage1_regime import detect_regime
from macro_engine.stage2_conditional import conditional_returns, rank_current_regime
from macro_engine.stage3_divergence import scan_divergences
from macro_engine.stage4_positioning import positioning_overlay
from macro_engine.stage5_scoring import converge
from macro_engine.stage6_expression import express_and_size
from macro_engine.stage7_falsification import falsify, write_journal

OUTPUT_DIR = "macro_engine/runs"


def run() -> dict:
    print("macro idea engine — convergence pipeline\n")

    print("[1/7] Stage 1 — macro regime ...")
    regime = detect_regime()
    print(f"      {regime['super_regime']} ({regime['state_code']}) asof {regime['asof']}")

    print("[2/7] Stage 2 — conditional returns ...")
    cond = conditional_returns(regime)
    ranked = rank_current_regime(cond, regime)
    print(f"      {len(ranked)} assets ranked in {regime['super_regime']}")

    print("[3/7] Stage 3 — divergence scan ...")
    divergences = scan_divergences(regime)
    print(f"      {len(divergences)} divergence rows")

    print("[4/7] Stage 4 — positioning (COT) ...")
    positioning = positioning_overlay()
    print(f"      {len(positioning)} assets with COT")

    print("[5/7] Stage 5 — convergence scoring ...")
    scored = converge(ranked, divergences, positioning)
    n_surf = int(scored["surfaced"].sum()) if not scored.empty else 0
    print(f"      {len(scored)} candidates, {n_surf} surfaced")

    print("[6/7] Stage 6 — expression + sizing ...")
    sized = express_and_size(scored, ranked)

    print("[7/7] Stage 7 — falsification + journal ...")
    falsified = falsify(sized)
    sheet = build_sheet(falsified, regime)
    prisk = portfolio_risk(sheet)
    if prisk:
        print(f"      book vol {prisk['book_vol_pct']:.1f}% (cap {prisk['vol_cap_pct']:.0f}%) "
              f"vs standalone {prisk['sum_standalone_pct']:.1f}% · div {prisk.get('diversification_ratio')}")
    write_journal(sheet, regime)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(OUTPUT_DIR, f"macro_engine_{stamp}.html")
    csv_path = os.path.join(OUTPUT_DIR, f"macro_engine_{stamp}.csv")
    render(html_path, regime, ranked, divergences, sheet, positioning, prisk)
    if not sheet.empty:
        sheet.drop(columns=["_div_detail"], errors="ignore").to_csv(csv_path, index=False)

    print(f"\n  HTML -> {html_path}")
    print(f"  CSV  -> {csv_path}")
    if not sheet.empty:
        cols = ["id", "asset", "direction", "convergence_score", "agreement_count",
                "suggested_size", "surfaced"]
        print("\nTop ideas:")
        print(sheet[cols].head(10).to_string(index=False))

    abs_path = os.path.abspath(html_path)
    if os.getenv("NO_OPEN") != "1":
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "Safari", abs_path], check=False)
        else:
            webbrowser.open(f"file://{abs_path}")
    return {"regime": regime, "ranked": ranked, "divergences": divergences, "sheet": sheet}


if __name__ == "__main__":
    run()
