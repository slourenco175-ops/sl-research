"""End-to-end v1 pipeline:
  1. Pull snapshots for the universe (yfinance, free)
  2. Score each with Claude Haiku
  3. Save full results + a top-N table to CSV
  4. Print the top 10 to the terminal

This does NOT trade. Output is research only."""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from data import get_snapshot
from screener import score_stock
from universe import DOW_30

OUTPUT_DIR = "runs"
TOP_N = 10


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing. Copy .env.example to .env and fill it in.")

    client = Anthropic()
    universe = DOW_30
    print(f"Screening {len(universe)} tickers with Claude Haiku 4.5\n")

    results = []
    for i, ticker in enumerate(universe, 1):
        print(f"[{i:>2}/{len(universe)}] {ticker} ...", end=" ", flush=True)
        snapshot = get_snapshot(ticker)
        if not snapshot:
            print("no data, skipping")
            continue

        score = score_stock(client, snapshot)
        if not score:
            print("scoring failed")
            continue

        print(f"score={score['score']} ({score['confidence']})")
        results.append(score)
        time.sleep(0.3)  # be polite to the API

    if not results:
        raise SystemExit("No stocks were scored. Check your network / API key.")

    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"scores_{stamp}.csv")
    df.to_csv(out_path, index=False)

    print(f"\nFull results -> {out_path}\n")
    print(f"Top {TOP_N}:")
    print(df.head(TOP_N)[["ticker", "name", "sector", "score", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
