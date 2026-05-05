"""End-to-end forex pipeline (renders the static HTML monitor).

  1. Pull ~5y of OHLC for the 12-pair universe (yfinance, free)
  2. Pull OHLC for linked commodities (oil, gold, copper)
  3. Pull last 3y of CFTC Traders in Financial Futures (TFF) — FX positioning
  4. Pull macro backdrop: DXY, US10Y, oil, gold, copper, VIX
  5. Pull country health for each currency from FRED (yields/CPI/GDP/unemp)
  6. Compute per-pair technicals + base/quote fundamentals deltas + COT
  7. Run the rules-based conviction scorer per pair
  8. Render the HTML dashboard
  9. Open it in the default browser

Research only. Does NOT trade.

    cd "/Users/samuellourenco/SL Research"
    source .venv/bin/activate
    python -m forex.run_fx
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime

import pandas as pd

from forex.country_health import build_health_table
from forex.dashboard_fx import render
from forex.factors_fx import build_factor_table
from forex.fx_data import get_ohlc, get_tff_history, latest_lev_money
from forex.macro_fx import get_macro
from forex.scoring_fx import score_row
from forex.universe_fx import CURRENCIES, UNIVERSE

OUTPUT_DIR = "forex/runs"
COMMODITY_SYMBOLS = ["CL=F", "GC=F", "HG=F", "ZS=F"]


def main() -> None:
    print(f"forex monitor — {len(UNIVERSE)} pairs\n")

    print("[1/6] Pulling FX OHLC from yfinance (~5y) ...")
    pair_symbols = [u["yf"] for u in UNIVERSE]
    ohlc_pairs = get_ohlc(pair_symbols, lookback_days=1800)
    print(f"      got {len(ohlc_pairs)} pair OHLC frames")

    print("[2/6] Pulling linked commodities (oil/gold/copper) ...")
    ohlc_comm = get_ohlc(COMMODITY_SYMBOLS, lookback_days=600)
    print(f"      got {len(ohlc_comm)} commodity OHLC frames")

    print("[3/6] Pulling COT TFF (financial futures, last 3y) ...")
    try:
        tff_df = get_tff_history(years_back=3)
        print(f"      got {len(tff_df):,} TFF rows")
    except SystemExit as e:
        print(f"      ! {e}")
        tff_df = pd.DataFrame()

    print("[4/6] Pulling macro (DXY/US10Y/oil/gold/copper/VIX) ...")
    macro = get_macro()
    print(f"      {', '.join(macro.keys()) or 'no macro data'}")

    print("[5/6] Pulling country health from FRED ...")
    health_df = build_health_table()
    print(f"      got health rows for: {', '.join(health_df.index.tolist())}")

    print("[6/6] Computing factors, scoring, rendering ...")
    df = build_factor_table(ohlc_pairs, ohlc_comm, UNIVERSE, health_df)
    if df.empty:
        raise SystemExit("No factor rows produced — check yfinance access.")

    # Attach COT (Leveraged Money) on the BASE currency futures
    if not tff_df.empty:
        cot_extras = []
        for _, r in df.iterrows():
            base_meta = CURRENCIES.get(r["base"], {})
            # USD has no own contract — proxy with the DXY index in TFF
            cot_name = base_meta.get("cot_name") or base_meta.get("dxy_cot_name")
            mm = latest_lev_money(tff_df, cot_name) if cot_name else None
            cot_extras.append(mm or {})
        df = pd.concat([df, pd.DataFrame(cot_extras, index=df.index)], axis=1)
    else:
        for c in ["mm_net_pct_oi", "mm_3y_percentile", "mm_1w_change",
                  "mm_4w_change", "mm_net_contracts", "mm_trajectory",
                  "comm_3y_percentile", "report_date"]:
            df[c] = pd.NA

    scored = [score_row(r.to_dict(), macro) for _, r in df.iterrows()]
    df["_score"]      = [x["score"]      for x in scored]
    df["_fv_score"]   = [x["fv_score"]   for x in scored]
    df["_tech_score"] = [x["tech_score"] for x in scored]
    df["_val_score"]  = [x["val_score"]  for x in scored]
    df["_cot_score"]  = [x["cot_score"]  for x in scored]
    df["_verdict"]    = [x["verdict"]    for x in scored]
    df["_fv_bias"]    = [x["fv_bias"]    for x in scored]
    df["_valuation"]  = [x["valuation"]  for x in scored]
    df["_reasons"]    = [x["reasons"]    for x in scored]
    # Sort by FV score (the model's "fair value" view), composite as tiebreaker.
    df = df.sort_values(["_fv_score", "_score"], ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(OUTPUT_DIR, f"forex_{stamp}.html")
    csv_path = os.path.join(OUTPUT_DIR, f"forex_{stamp}.csv")

    render(df, health_df, macro, html_path)
    df.drop(columns=["spark", "_reasons", "regression_svg"], errors="ignore").to_csv(csv_path, index=False)

    print(f"\n  HTML -> {html_path}")
    print(f"  CSV  -> {csv_path}\n")

    print("Top fair-value bias:")
    cols = ["yf", "name", "_fv_score", "_verdict", "_fv_bias", "_valuation",
            "_tech_score", "_score", "carry_diff", "rr_diff", "health_diff"]
    print(df[cols].head(8).to_string(index=False))

    abs_path = os.path.abspath(html_path)
    if os.getenv("NO_OPEN") == "1":
        return
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Safari", abs_path], check=False)
    else:
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
