"""End-to-end backtest runner.

  1. Pull ~5y OHLC for the universe (yfinance, free)
  2. Pull last 3y of CFTC disaggregated COT (free)
  3. Run vectorized weekly-rebalance backtest with point-in-time scoring
  4. Render the backtest HTML report + CSV exports
  5. Open the report in Safari (skip with NO_OPEN=1)

    cd "/Users/samuellourenco/SL Research"
    source .venv/bin/activate
    python -m commodities.run_backtest
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime

import pandas as pd

from commodities.backtest import BacktestConfig, run_backtest
from commodities.backtest_report import render_backtest_report
from commodities.cmdty_data import get_cot_history, get_ohlc
from commodities.universe_cmdty import UNIVERSE

OUTPUT_DIR = "commodities/runs"


def main() -> None:
    print("commodities backtest — v2 cross-sectional model\n")

    print("[1/4] Pulling OHLC (~5y) ...")
    symbols = [u["yf"] for u in UNIVERSE]
    ohlc = get_ohlc(symbols, lookback_days=2200)
    print(f"      {len(ohlc)} OHLC frames")

    print("[2/4] Pulling COT (3y) ...")
    cot_df = get_cot_history(years_back=3)
    print(f"      {len(cot_df):,} rows")

    print("[3/4] Running backtest (weekly rebalance) ...")
    cfg = BacktestConfig()
    result = run_backtest(ohlc, cot_df, UNIVERSE, cfg)

    if result["equity"].empty:
        raise SystemExit("No equity curve produced — check date range / data.")

    s = result["stats"]
    print(f"\n      Sharpe:  {s['sharpe']:.2f}")
    print(f"      CAGR:    {s['cagr']*100:+.1f}%   Vol: {s['vol_ann']*100:.1f}%")
    print(f"      Max DD:  {s['max_drawdown']*100:.1f}%")
    print(f"      Sortino: {s['sortino']:.2f}")
    print(f"      Hit:     {s['hit_rate_daily']*100:.0f}% days, {s['months_positive']*100:.0f}% months")
    print(f"      Period:  {s['start']} → {s['end']}  ({s['n_days']} days)")

    print("\n[4/4] Rendering report ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(OUTPUT_DIR, f"backtest_{stamp}.html")
    eq_path = os.path.join(OUTPUT_DIR, f"backtest_equity_{stamp}.csv")
    trades_path = os.path.join(OUTPUT_DIR, f"backtest_trades_{stamp}.csv")

    render_backtest_report(result, html_path)
    pd.DataFrame({
        "equity": result["equity"],
        "daily_return": result["daily_returns"],
        "drawdown": result["equity"] / result["equity"].cummax() - 1,
    }).to_csv(eq_path)
    if not result["trades"].empty:
        result["trades"].to_csv(trades_path, index=False)

    print(f"\n  Report -> {html_path}")
    print(f"  Equity -> {eq_path}")
    print(f"  Trades -> {trades_path}\n")

    if os.getenv("NO_OPEN") == "1":
        return
    abs_path = os.path.abspath(html_path)
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Safari", abs_path], check=False)
    else:
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
