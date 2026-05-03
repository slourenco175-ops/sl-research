# SL Research — AI Portfolio Manager (v1: Screener)

Research-only. **Does not trade.** Outputs a ranked CSV of stock scores.

## What v1 does

For each ticker in the Dow 30:
1. Pulls a fundamentals snapshot from Yahoo Finance (free, no API key).
2. Sends it to Claude Haiku 4.5, which returns a `score` (0-100), `confidence`, `thesis`, and `key_risks`.
3. Saves all results to `runs/scores_YYYYMMDD_HHMMSS.csv` and prints the top 10.

Cost per full run: roughly **$0.02-0.05** (30 Haiku calls, ~1k tokens each).

## Setup

```bash
cd "/Users/samuellourenco/SL Research"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste your Anthropic API key
```

## Run

```bash
source .venv/bin/activate
python run.py
```

## Files

- `universe.py` — list of tickers to screen (currently Dow 30; swap for Russell 1000 later)
- `data.py` — yfinance snapshot fetcher
- `screener.py` — Claude-based scoring agent
- `run.py` — orchestrator that ties it together

## Roadmap

v1 (this) — screener only.
v2 — add bull/bear adversarial research swarm on top 10 (web search, last 7 days).
v3 — scenario modeling (bull/base/bear price targets at 1/3/6/12 months) with self-debate.
v4 — optimizer that picks ~15 positions under sector + risk constraints (cvxpy, called as a tool).
v5 — rebalancing loop on a schedule, paper-trading harness (Alpaca paper API). Still no live trading.
