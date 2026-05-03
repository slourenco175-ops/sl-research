"""Pulls a compact financial snapshot per ticker from yfinance.
Free, no API key. Returns a dict that fits in one Claude prompt cheaply."""

from __future__ import annotations

import yfinance as yf


def get_snapshot(ticker: str) -> dict | None:
    """Return a small dict of fundamentals + recent price action, or None on error."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        hist = t.history(period="6mo")
        if hist.empty:
            return None

        price_now = float(hist["Close"].iloc[-1])
        price_6mo_ago = float(hist["Close"].iloc[0])
        ret_6mo = (price_now / price_6mo_ago - 1) * 100

        return {
            "ticker": ticker,
            "name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap_b": _to_billions(info.get("marketCap")),
            "price": round(price_now, 2),
            "return_6mo_pct": round(ret_6mo, 2),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "peg": info.get("pegRatio"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "pb": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow_b": _to_billions(info.get("freeCashflow")),
            "dividend_yield": info.get("dividendYield"),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "business_summary": (info.get("longBusinessSummary") or "")[:600],
        }
    except Exception as e:
        print(f"  ! {ticker}: {e}")
        return None


def _to_billions(x) -> float | None:
    if x is None:
        return None
    try:
        return round(float(x) / 1e9, 2)
    except (TypeError, ValueError):
        return None
