"""Screener agent: asks Claude Haiku to score one stock 0-100 from its snapshot.
Cheap (Haiku 4.5), structured JSON output, one ticker per call."""

from __future__ import annotations

import json
import re

from anthropic import Anthropic

MODEL = "claude-haiku-4-5"

SYSTEM = """You are a buy-side equity analyst screener.

You score one stock at a time on a 0-100 scale based ONLY on the fundamentals snapshot you are given.
- 80-100: strong fundamentals, attractive valuation, durable growth
- 60-79: solid quality with some concerns
- 40-59: mixed / fairly valued / no clear edge
- 20-39: weak or expensive
- 0-19: avoid

You weigh: profitability and margins, growth, valuation vs growth (PEG), balance sheet (debt/equity),
cash generation (FCF), and 6-month price momentum. Missing data should lower your confidence, not your score.

Respond with ONLY valid JSON in this exact shape:
{"score": <int 0-100>, "confidence": <"low"|"medium"|"high">, "thesis": "<2-3 sentences>", "key_risks": "<1-2 sentences>"}
"""


def score_stock(client: Anthropic, snapshot: dict) -> dict | None:
    """Send one snapshot to Claude, return parsed score dict or None."""
    prompt = (
        f"Score this stock based on the snapshot below.\n\n"
        f"```json\n{json.dumps(snapshot, indent=2, default=str)}\n```"
    )

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()

        # Strip code fences if Claude wrapped them
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)

        return {
            "ticker": snapshot["ticker"],
            "name": snapshot.get("name"),
            "sector": snapshot.get("sector"),
            "score": int(parsed["score"]),
            "confidence": parsed.get("confidence", "medium"),
            "thesis": parsed.get("thesis", ""),
            "key_risks": parsed.get("key_risks", ""),
        }
    except Exception as e:
        print(f"  ! score failed for {snapshot.get('ticker')}: {e}")
        return None
