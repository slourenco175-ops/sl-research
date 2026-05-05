"""End-to-end commodities pipeline (renders the v2 dashboard).

  1. Pull ~5y of OHLC for the universe (yfinance, free)
  2. Pull last 3y of CFTC disaggregated COT (free, direct from cftc.gov)
  3. Pull DXY / US10Y / VIX for the macro backdrop
  4. Compute factors + indicators + carry-proxy + CTA proxy + seasonality
  5. Compute relative-value spreads
  6. Run the rules-based conviction scorer per asset
  7. Render the v2 HTML dashboard and the v1 dashboard (kept for fallback)
  8. Open the v2 dashboard in the default browser

Research only. Does NOT trade.

    cd "/Users/samuellourenco/SL Research"
    source .venv/bin/activate
    python -m commodities.run_cmdty
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime

import pandas as pd

from commodities.cmdty_data import (
    closes_from_ohlc,
    get_cot_history,
    get_ohlc,
    latest_managed_money,
)
from commodities.curve import get_curves
from commodities.dashboard import render as render_v1
from commodities.dashboard_v2 import render_v2
from commodities.factors import build_factor_table
from commodities.inventories import get_inventories
from commodities.macro import get_macro
from commodities.scoring import score_row
from commodities.spreads import build_spreads
from commodities.universe_cmdty import UNIVERSE

OUTPUT_DIR = "commodities/runs"


def main() -> None:
    print(f"v2 commodities run — {len(UNIVERSE)} contracts\n")

    print("[1/8] Pulling OHLC from yfinance (~5y) ...")
    symbols = [u["yf"] for u in UNIVERSE]
    ohlc = get_ohlc(symbols, lookback_days=1800)
    print(f"      got {len(ohlc)} OHLC frames")

    print("[2/8] Pulling COT (CFTC disaggregated, last 3y) ...")
    try:
        cot_df = get_cot_history(years_back=3)
        print(f"      got {len(cot_df):,} COT rows")
    except SystemExit as e:
        print(f"      ! {e}")
        cot_df = pd.DataFrame()

    print("[3/8] Pulling macro (DXY / US10Y / VIX) ...")
    macro = get_macro()
    print(f"      {', '.join(macro.keys()) or 'no macro data'}")

    print("[4/8] Pulling EIA weekly inventories (FRED) ...")
    inventories = get_inventories()
    print(f"      got {len(inventories)} inventory series: {', '.join(inventories.keys()) or 'none'}")

    print("[5/8] Pulling deferred-month curves from yfinance ...")
    curve_roots = sorted({u["curve_root"] for u in UNIVERSE if u.get("curve_root")})
    curves = get_curves(curve_roots)
    n_with_slope = sum(1 for c in curves.values() if c.get("slope_pct_yr") is not None)
    print(f"      got {n_with_slope}/{len(curves)} curves with computable slope")

    print("[6/8] Computing factors + indicators ...")
    df = build_factor_table(ohlc, UNIVERSE)
    if df.empty:
        raise SystemExit("No factor rows produced — check yfinance access.")

    if not cot_df.empty:
        cot_extras = []
        for _, r in df.iterrows():
            mm = latest_managed_money(cot_df, r.get("cot_name")) or {}
            cot_extras.append(mm)
        df = pd.concat([df, pd.DataFrame(cot_extras, index=df.index)], axis=1)
    else:
        for c in ["mm_net_pct_oi", "mm_3y_percentile", "mm_1w_change",
                  "mm_4w_change", "mm_net_contracts", "mm_trajectory",
                  "comm_3y_percentile", "report_date"]:
            df[c] = pd.NA

    # Attach inventory + curve fields per row (NaN where unavailable)
    inv_extras, curve_extras = [], []
    for _, r in df.iterrows():
        inv_key = r.get("inventory_key")
        inv = inventories.get(inv_key) if inv_key else None
        inv_extras.append({
            "inv_label": inv["label"] if inv else None,
            "inv_level": inv["level"] if inv else float("nan"),
            "inv_level_date": inv["level_date"] if inv else None,
            "inv_seasonal_z": inv["seasonal_z"] if inv else float("nan"),
            "inv_build_4w": inv["build_4w"] if inv else float("nan"),
            "inv_build_4w_norm": inv["build_4w_norm"] if inv else float("nan"),
            "inv_weeks_ago": inv["weeks_ago"] if inv else None,
        })

        cv = curves.get(r.get("curve_root")) or {}
        curve_extras.append({
            "curve_n_months": cv.get("n_months"),
            "curve_slope_pct_yr": cv.get("slope_pct_yr", float("nan")),
            "curve_spread_m1_m2_pct": cv.get("spread_m1_m2_pct"),
            "curve_spread_m1_m6_pct": cv.get("spread_m1_m6_pct"),
            "curve_state": cv.get("state"),
            "curve_deferred_months_ahead": cv.get("deferred_months_ahead"),
        })
    df = pd.concat([
        df,
        pd.DataFrame(inv_extras, index=df.index),
        pd.DataFrame(curve_extras, index=df.index),
    ], axis=1)

    print("[7/8] Computing spreads + scoring ...")
    spreads_df = build_spreads(ohlc)

    scores, all_reasons, verdicts = [], [], []
    for _, r in df.iterrows():
        s, reasons, v = score_row(r.to_dict(), macro)
        scores.append(s)
        all_reasons.append(reasons)
        verdicts.append(v)
    df["_score"] = scores
    df["_verdict"] = verdicts
    df["_reasons"] = all_reasons
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)

    print("[8/8] Rendering dashboards ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    v2_path = os.path.join(OUTPUT_DIR, f"dashboard_v2_{stamp}.html")
    v1_path = os.path.join(OUTPUT_DIR, f"dashboard_{stamp}.html")
    csv_path = os.path.join(OUTPUT_DIR, f"signals_{stamp}.csv")

    # The v2 renderer expects df to already be sorted, with _score/_verdict/_reasons attached
    scored = list(zip(df["_score"], df["_reasons"], df["_verdict"]))
    render_v2(df, spreads_df, macro, scored, v2_path, ohlc=ohlc)
    render_v1(df, v1_path)

    df.drop(columns=["spark", "_reasons"], errors="ignore").to_csv(csv_path, index=False)

    print(f"\n  v2 HTML -> {v2_path}")
    print(f"  v1 HTML -> {v1_path}")
    print(f"  CSV     -> {csv_path}\n")

    print("Top conviction:")
    cols = ["name", "sector", "_score", "_verdict", "trend_st", "trend_mt", "trend_lt", "macd", "carry_state"]
    print(df[cols].head(8).to_string(index=False))

    abs_path = os.path.abspath(v2_path)
    if os.getenv("NO_OPEN") == "1":
        return  # CI / headless: skip opening a browser
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Safari", abs_path], check=False)
    else:
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
