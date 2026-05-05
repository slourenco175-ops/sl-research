"""Data layer for the commodities model.

Two free sources, no API keys:
  - yfinance for front-month continuous futures prices
  - cot_reports for CFTC Commitments of Traders (disaggregated, futures-only)

COT comes out Friday afternoon for Tuesday's snapshot — so it lags ~3 days.
We pull a single multi-year history and slice in memory; that's far cheaper
than re-downloading per ticker.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import pandas as pd
import yfinance as yf


# -------------------------- prices --------------------------

def get_ohlc(symbols: list[str], lookback_days: int = 1800) -> dict[str, pd.DataFrame]:
    """Return {symbol -> OHLCV DataFrame} for the given yfinance symbols.

    1800 calendar days ~= 5 years of daily bars: enough for 200d trend,
    3y vol percentile, ~5y monthly seasonality, all in one network call.
    """
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=int(lookback_days * 1.6))
    # yfinance treats `end` as exclusive — bump by 1 day so today's bar is included
    end_inclusive = end + pd.Timedelta(days=1)

    raw = yf.download(
        tickers=symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end_inclusive.strftime("%Y-%m-%d"),
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


def get_prices(symbols: list[str], lookback_days: int = 400) -> pd.DataFrame:
    """Backward-compat: closes-only DataFrame derived from get_ohlc()."""
    ohlc = get_ohlc(symbols, lookback_days=lookback_days)
    if not ohlc:
        return pd.DataFrame()
    closes = pd.DataFrame({s: df["Close"] for s, df in ohlc.items()})
    return closes.dropna(how="all").ffill(limit=3)


def closes_from_ohlc(ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: df["Close"] for s, df in ohlc.items()}).dropna(how="all").ffill(limit=3)


# -------------------------- COT --------------------------

def _fetch_cftc_zip(url_template: str, years_back: int) -> pd.DataFrame:
    """Generic helper: fetch & concat annual CFTC zip files."""
    import io
    import zipfile

    import requests

    this_year = datetime.utcnow().year
    frames = []
    for yr in range(this_year - years_back, this_year + 1):
        url = url_template.format(year=yr)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
                with zf.open(txt_name) as f:
                    df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"  ! COT {url.split('/')[-1]}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Different CFTC reports use either underscored or space-and-paren column names.
    # Normalize the date column.
    date_candidates = [
        "Report_Date_as_YYYY-MM-DD",
        "As_of_Date_In_Form_YYMMDD",
        "Report Date as YYYY-MM-DD",
        "As of Date In Form YYMMDD",
        "As of Date in Form YYYY-MM-DD",
        "As of Date in Form YYMMDD",
    ]
    date_col = next((c for c in date_candidates if c in out.columns), None)
    if date_col:
        out["report_date"] = pd.to_datetime(out[date_col], errors="coerce")
    return out


@lru_cache(maxsize=1)
def get_cot_history(years_back: int = 3) -> pd.DataFrame:
    """Disaggregated futures-only — Managed Money + Producer/Merchant + Swap dealers."""
    return _fetch_cftc_zip(
        "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
        years_back,
    )


def latest_managed_money(cot_df: pd.DataFrame, contract_substring: str) -> dict | None:
    """Latest Managed Money + Commercial positioning for a contract.

    Returns:
      mm_net_pct_oi, mm_3y_percentile, mm_1w_change, mm_4w_change,
      mm_net_contracts, mm_trajectory ('ACCELERATING'|'DECELERATING'|'REVERSING'),
      comm_3y_percentile (Producer+Swap dealers net % OI percentile),
      report_date
    """
    if cot_df.empty or contract_substring is None:
        return None

    mask = cot_df["Market_and_Exchange_Names"].str.contains(
        contract_substring, case=False, na=False, regex=False
    )
    sub = cot_df[mask].sort_values("report_date")
    if sub.empty:
        return None

    def _num(col_name):
        if col_name in sub.columns:
            return pd.to_numeric(sub[col_name], errors="coerce").fillna(0)
        return pd.Series(0, index=sub.index)

    mm_long = pd.to_numeric(sub["M_Money_Positions_Long_All"], errors="coerce")
    mm_short = pd.to_numeric(sub["M_Money_Positions_Short_All"], errors="coerce")
    oi = pd.to_numeric(sub["Open_Interest_All"], errors="coerce")
    pm_long = _num("Prod_Merc_Positions_Long_All")
    pm_short = _num("Prod_Merc_Positions_Short_All")
    sd_long = _num("Swap_Positions_Long_All")
    sd_short = _num("Swap_Positions_Short_All")

    mm_net = (mm_long - mm_short)
    mm_net_pct = (mm_net / oi * 100).dropna()
    comm_net_pct = (((pm_long + sd_long) - (pm_short + sd_short)) / oi * 100).dropna()

    if mm_net_pct.empty:
        return None

    latest = float(mm_net_pct.iloc[-1])
    pctile = float(mm_net_pct.rank(pct=True).iloc[-1] * 100)
    one_wk = float(mm_net_pct.iloc[-1] - mm_net_pct.iloc[-2]) if len(mm_net_pct) >= 2 else 0.0
    four_wk = float(mm_net_pct.iloc[-1] - mm_net_pct.iloc[-5]) if len(mm_net_pct) >= 5 else one_wk

    # trajectory: same sign + accelerating, same sign + slowing, or reversing
    if (latest >= 0 and four_wk < 0) or (latest < 0 and four_wk > 0):
        trajectory = "REVERSING"
    elif abs(four_wk) >= abs(one_wk) * 3:
        trajectory = "ACCELERATING"
    else:
        trajectory = "DECELERATING"

    comm_pctile = (
        float(comm_net_pct.rank(pct=True).iloc[-1] * 100) if not comm_net_pct.empty else float("nan")
    )

    return {
        "mm_net_pct_oi": round(latest, 2),
        "mm_3y_percentile": round(pctile, 1),
        "mm_1w_change": round(one_wk, 2),
        "mm_4w_change": round(four_wk, 2),
        "mm_net_contracts": int(mm_net.iloc[-1]) if pd.notna(mm_net.iloc[-1]) else 0,
        "mm_trajectory": trajectory,
        "comm_3y_percentile": round(comm_pctile, 1) if pd.notna(comm_pctile) else None,
        "report_date": sub["report_date"].iloc[-1].strftime("%Y-%m-%d"),
    }


