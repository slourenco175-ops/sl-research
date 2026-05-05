"""Data layer for the forex monitor.

Three free sources, no API keys:
  - yfinance for FX pair OHLC + linked-commodity prices
  - cftc.gov direct ZIPs for the Traders in Financial Futures (TFF) report
    (the FX equivalent of the disaggregated commodities COT report)
  - FRED CSV endpoint (https://fred.stlouisfed.org/graph/fredgraph.csv)
    for sovereign yields, unemployment, CPI, GDP — no key required

The TFF report comes out Friday afternoon for Tuesday's snapshot, same cadence
as the disaggregated commodities one. We pull a multi-year history and slice in
memory; far cheaper than re-downloading per ticker.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf


# -------------------------- prices --------------------------

def get_ohlc(symbols: list[str], lookback_days: int = 1800) -> dict[str, pd.DataFrame]:
    """Return {symbol -> OHLCV DataFrame} for yfinance symbols (FX or commodities).

    1800 calendar days ~= 5y of daily bars: enough for 200d trend, 3y vol
    percentile, and ~5y monthly seasonality, all in one network call.
    """
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=int(lookback_days * 1.6))

    raw = yf.download(
        tickers=symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for s in symbols:
            if s in raw.columns.levels[0]:
                df = raw[s].dropna(how="all").ffill(limit=3)
                if not df.empty:
                    out[s] = df
    else:
        out[symbols[0]] = raw.dropna(how="all").ffill(limit=3)
    return out


# -------------------------- COT (TFF / financial futures) --------------------------

@lru_cache(maxsize=1)
def get_tff_history(years_back: int = 3) -> pd.DataFrame:
    """CFTC Traders in Financial Futures — the FX/equity-index COT report."""
    this_year = datetime.utcnow().year
    frames = []
    for yr in range(this_year - years_back, this_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{yr}.zip"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
                with zf.open(txt_name) as f:
                    df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"  ! TFF {yr}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    date_candidates = [
        "Report_Date_as_YYYY-MM-DD",
        "Report Date as YYYY-MM-DD",
        "As_of_Date_In_Form_YYMMDD",
    ]
    date_col = next((c for c in date_candidates if c in out.columns), None)
    if date_col:
        out["report_date"] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def latest_lev_money(tff_df: pd.DataFrame, contract_substring: str) -> dict | None:
    """Latest Leveraged Money positioning for an FX contract.

    'Leveraged Money' in the TFF report = hedge funds + CTAs — the cleanest
    speculator proxy in FX. We mirror the commodities `latest_managed_money`
    output schema (mm_*) so the dashboard can reuse formatting.
    """
    if tff_df.empty or contract_substring is None:
        return None

    mask = tff_df["Market_and_Exchange_Names"].str.contains(
        contract_substring, case=False, na=False, regex=False
    )
    sub = tff_df[mask].sort_values("report_date")
    if sub.empty:
        return None

    long = pd.to_numeric(sub["Lev_Money_Positions_Long_All"], errors="coerce")
    short = pd.to_numeric(sub["Lev_Money_Positions_Short_All"], errors="coerce")
    oi = pd.to_numeric(sub["Open_Interest_All"], errors="coerce")

    am_long = pd.to_numeric(sub.get("Asset_Mgr_Positions_Long_All", 0), errors="coerce").fillna(0)
    am_short = pd.to_numeric(sub.get("Asset_Mgr_Positions_Short_All", 0), errors="coerce").fillna(0)

    net = (long - short)
    net_pct = (net / oi * 100).dropna()
    am_net_pct = (((am_long - am_short)) / oi * 100).dropna()
    if net_pct.empty:
        return None

    latest = float(net_pct.iloc[-1])
    pctile = float(net_pct.rank(pct=True).iloc[-1] * 100)
    one_wk = float(net_pct.iloc[-1] - net_pct.iloc[-2]) if len(net_pct) >= 2 else 0.0
    four_wk = float(net_pct.iloc[-1] - net_pct.iloc[-5]) if len(net_pct) >= 5 else one_wk

    if (latest >= 0 and four_wk < 0) or (latest < 0 and four_wk > 0):
        trajectory = "REVERSING"
    elif abs(four_wk) >= abs(one_wk) * 3:
        trajectory = "ACCELERATING"
    else:
        trajectory = "DECELERATING"

    am_pctile = (
        float(am_net_pct.rank(pct=True).iloc[-1] * 100) if not am_net_pct.empty else float("nan")
    )

    return {
        "mm_net_pct_oi": round(latest, 2),
        "mm_3y_percentile": round(pctile, 1),
        "mm_1w_change": round(one_wk, 2),
        "mm_4w_change": round(four_wk, 2),
        "mm_net_contracts": int(net.iloc[-1]) if pd.notna(net.iloc[-1]) else 0,
        "mm_trajectory": trajectory,
        "comm_3y_percentile": round(am_pctile, 1) if pd.notna(am_pctile) else None,
        "report_date": sub["report_date"].iloc[-1].strftime("%Y-%m-%d"),
    }


# -------------------------- FRED --------------------------

@lru_cache(maxsize=512)
def _fetch_fred_one(series_id: str) -> pd.Series:
    """Fetch a single FRED series via the public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"  ! FRED {series_id}: {e}")
        return pd.Series(dtype=float)

    if df.empty or df.shape[1] < 2:
        return pd.Series(dtype=float)

    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index()


def fetch_fred(series: str | list[str]) -> pd.Series:
    """Fetch a FRED series. Accepts either a single ID or a list of fallbacks.

    With a list, returns the first non-empty series. This handles FRED's habit
    of renaming/retiring OECD-derived IDs without warning.
    """
    if isinstance(series, str):
        return _fetch_fred_one(series)
    for sid in series:
        s = _fetch_fred_one(sid)
        if not s.empty:
            return s
    return pd.Series(dtype=float)
