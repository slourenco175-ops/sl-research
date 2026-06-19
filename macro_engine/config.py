"""Single config surface for the macro engine: universe, regime series,
thresholds, scoring weights. Same spirit as the Lynch/Buffett scorer config —
one place to tune the engine without touching stage logic.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Stage 0 — universe (cross-asset, instrument-agnostic proxies via yfinance).
# ETFs chosen for long, clean, auto-adjusted history. Futures continuations
# (=F) used where the ETF proxy is poor (copper).
# --------------------------------------------------------------------------
UNIVERSE: list[dict] = [
    # symbol, bucket, human name, COT contract substring (None = no COT map)
    {"sym": "SHY",   "bucket": "rates",     "name": "UST 1-3y",         "cot": None},
    {"sym": "IEF",   "bucket": "rates",     "name": "UST 7-10y",        "cot": None},
    {"sym": "TLT",   "bucket": "rates",     "name": "UST 20y+",         "cot": None},
    {"sym": "TIP",   "bucket": "inflation", "name": "TIPS",             "cot": None},
    {"sym": "UUP",   "bucket": "fx",        "name": "USD index (DXY)",  "cot": "U.S. DOLLAR INDEX"},
    {"sym": "FXE",   "bucket": "fx",        "name": "Euro",             "cot": "EURO FX"},
    {"sym": "FXY",   "bucket": "fx",        "name": "Japanese yen",     "cot": "JAPANESE YEN"},
    {"sym": "SPY",   "bucket": "equity",    "name": "S&P 500",          "cot": "E-MINI S&P 500"},
    {"sym": "ACWI",  "bucket": "equity",    "name": "MSCI ACWI",        "cot": None},
    {"sym": "EEM",   "bucket": "equity",    "name": "EM equity",        "cot": None},
    {"sym": "DBC",   "bucket": "commodity", "name": "Broad commodities","cot": None},
    {"sym": "GLD",   "bucket": "commodity", "name": "Gold",             "cot": "GOLD"},
    {"sym": "USO",   "bucket": "commodity", "name": "WTI crude",        "cot": "CRUDE OIL, LIGHT SWEET"},
    {"sym": "CPER",  "bucket": "commodity", "name": "Copper",           "cot": "COPPER"},
    {"sym": "HYG",   "bucket": "credit",    "name": "US high yield",    "cot": None},
    {"sym": "LQD",   "bucket": "credit",    "name": "US IG credit",     "cot": None},

    # --- Global single-country equity sleeve (USD-unhedged MSCI ETFs) --------
    # Ranked the same way as everything else; the cross-country relative value
    # comes from Stage 2 (regime-conditional Sharpe) + Stage 3 terms-of-trade.
    {"sym": "EWP",   "bucket": "global_equity", "name": "Spain",        "cot": None},
    {"sym": "EWG",   "bucket": "global_equity", "name": "Germany",      "cot": None},
    {"sym": "EWJ",   "bucket": "global_equity", "name": "Japan",        "cot": None},
    {"sym": "GREK",  "bucket": "global_equity", "name": "Greece",       "cot": None},
    {"sym": "EPOL",  "bucket": "global_equity", "name": "Poland",       "cot": None},
    {"sym": "FXI",   "bucket": "global_equity", "name": "China (large)","cot": None},
    {"sym": "INDA",  "bucket": "global_equity", "name": "India",        "cot": None},
    {"sym": "EWY",   "bucket": "global_equity", "name": "South Korea",  "cot": None},
    {"sym": "EWT",   "bucket": "global_equity", "name": "Taiwan",       "cot": None},
    {"sym": "EIDO",  "bucket": "global_equity", "name": "Indonesia",    "cot": None},
    {"sym": "EWZ",   "bucket": "global_equity", "name": "Brazil",       "cot": None},
    {"sym": "EWW",   "bucket": "global_equity", "name": "Mexico",       "cot": None},
    {"sym": "GXG",   "bucket": "global_equity", "name": "Colombia",     "cot": None},
    {"sym": "ECH",   "bucket": "global_equity", "name": "Chile",        "cot": None},
    {"sym": "EPU",   "bucket": "global_equity", "name": "Peru",         "cot": None},
    {"sym": "NORW",  "bucket": "global_equity", "name": "Norway",       "cot": None},
    {"sym": "EZA",   "bucket": "global_equity", "name": "South Africa", "cot": None},
]

# Extra price series used for divergence ratios / vol, not ranked as candidates.
AUX_SYMBOLS = ["^VIX", "HG=F", "GC=F", "XLY", "XLP"]

# Cross-country terms-of-trade map: commodity-exporting countries whose equity
# index should track a key export. Stage 3 regresses the country on this proxy
# (already in the universe) and trades the residual. Note Portugal has no liquid
# US-listed single-country ETF (PGAL delisted 2024) so it is intentionally absent.
COUNTRY_COMMODITY: dict[str, str] = {
    "NORW": "USO",   # Norway — oil & gas
    "GXG":  "USO",   # Colombia — oil
    "EWW":  "USO",   # Mexico — oil
    "EWZ":  "DBC",   # Brazil — broad commodities (soy, iron ore, oil)
    "ECH":  "CPER",  # Chile — copper
    "EPU":  "CPER",  # Peru — copper
    "EIDO": "DBC",   # Indonesia — coal / palm / broad
    "EZA":  "GLD",   # South Africa — gold & precious metals
}

# --------------------------------------------------------------------------
# Cross-asset DRIVER model (macro_engine/correlations.py).
# Drivers are explanatory series. We measure each tradable's sensitivity (beta /
# correlation) to every driver, keep only the statistically robust + stable
# relationships, then turn recent driver moves into directional signals
# (e.g. crude↑ → Norway↑, broad commodities↑ → Brazil↑).
#   kind: 'px'  = price, use pct-change return
#         'yld' = FRED yield, use daily change in basis points
#         'vix' = vol index, use pct-change
# --------------------------------------------------------------------------
DRIVERS: list[dict] = [
    # --- commodities ---
    {"name": "Crude WTI",     "sym": "CL=F", "kind": "px",  "grp": "commodity"},
    {"name": "Crude Brent",   "sym": "BZ=F", "kind": "px",  "grp": "commodity"},
    {"name": "Nat gas",       "sym": "NG=F", "kind": "px",  "grp": "commodity"},
    {"name": "Copper",        "sym": "HG=F", "kind": "px",  "grp": "commodity"},
    {"name": "Gold",          "sym": "GC=F", "kind": "px",  "grp": "commodity"},
    {"name": "Silver",        "sym": "SI=F", "kind": "px",  "grp": "commodity"},
    {"name": "Platinum",      "sym": "PL=F", "kind": "px",  "grp": "commodity"},
    {"name": "Wheat",         "sym": "ZW=F", "kind": "px",  "grp": "commodity"},
    {"name": "Soybeans",      "sym": "ZS=F", "kind": "px",  "grp": "commodity"},
    {"name": "Agriculture",   "sym": "DBA",  "kind": "px",  "grp": "commodity"},
    {"name": "Broad cmdty",   "sym": "DBC",  "kind": "px",  "grp": "commodity"},
    {"name": "Energy equity", "sym": "XLE",  "kind": "px",  "grp": "commodity"},
    # --- FX (USDxxx: rising = USD up / local currency down) ---
    {"name": "USD index",     "sym": "UUP",      "kind": "px", "grp": "fx"},
    {"name": "EUR/USD",       "sym": "EURUSD=X", "kind": "px", "grp": "fx"},
    {"name": "USD/JPY",       "sym": "USDJPY=X", "kind": "px", "grp": "fx"},
    {"name": "GBP/USD",       "sym": "GBPUSD=X", "kind": "px", "grp": "fx"},
    {"name": "AUD/USD",       "sym": "AUDUSD=X", "kind": "px", "grp": "fx"},
    {"name": "USD/CAD",       "sym": "USDCAD=X", "kind": "px", "grp": "fx"},
    {"name": "USD/NOK",       "sym": "USDNOK=X", "kind": "px", "grp": "fx"},
    {"name": "USD/BRL",       "sym": "USDBRL=X", "kind": "px", "grp": "fx"},
    {"name": "USD/CLP",       "sym": "USDCLP=X", "kind": "px", "grp": "fx"},
    {"name": "USD/COP",       "sym": "USDCOP=X", "kind": "px", "grp": "fx"},
    {"name": "USD/ZAR",       "sym": "USDZAR=X", "kind": "px", "grp": "fx"},
    {"name": "USD/MXN",       "sym": "USDMXN=X", "kind": "px", "grp": "fx"},
    {"name": "USD/IDR",       "sym": "USDIDR=X", "kind": "px", "grp": "fx"},
    # --- rates / credit (FRED, daily Δbp) ---
    {"name": "UST 2y Δbp",    "fred": "DGS2",          "kind": "yld", "grp": "rates"},
    {"name": "UST 10y Δbp",   "fred": "DGS10",         "kind": "yld", "grp": "rates"},
    {"name": "UST 30y Δbp",   "fred": "DGS30",         "kind": "yld", "grp": "rates"},
    {"name": "Breakeven Δbp", "fred": "T10YIE",        "kind": "yld", "grp": "rates"},
    {"name": "HY OAS Δbp",    "fred": "BAMLH0A0HYM2",  "kind": "yld", "grp": "rates"},
    # --- risk ---
    {"name": "S&P 500",       "sym": "SPY",  "kind": "px",  "grp": "risk"},
    {"name": "EM equity",     "sym": "EEM",  "kind": "px",  "grp": "risk"},
    {"name": "VIX",           "sym": "^VIX", "kind": "vix", "grp": "risk"},
]

# Robustness gates for keeping a driver→asset relationship.
CORR_MIN_YEARS = 6        # require this much overlapping daily history
CORR_ABS_MIN = 0.12       # |full-sample correlation| floor
CORR_T_MIN = 3.0          # |t-stat| floor (significance of the correlation)
CORR_STABLE_MIN = 0.60    # fraction of rolling-1y windows whose corr sign agrees
SIGNAL_MOVE_WINDOW = 21   # 'recent move' lookback for a driver (trading days)
SIGNAL_Z_WINDOW = 252     # window to z-score that move
SIGNAL_FWD_DAYS = 21      # forward horizon for IC validation (~1 month)

# Lead-lag hunt: does a trailing driver move PREDICT forward asset returns?
LEAD_HORIZONS = (1, 3, 5, 10, 21, 42, 63)   # forward asset-return horizons (days)
LEAD_IC_MIN = 0.06        # |forward IC| floor to call a lead 'real'
LEAD_MIN_SAMPLES = 60     # min de-overlapped observations for the IC

# --------------------------------------------------------------------------
# Stage 1 — regime pillars. One FRED series each (fallback chains allowed).
# Each pillar is reduced to a binary bit; the four bits make the 16-state code.
# `hot_label` is the bit==1 meaning. Super-regime uses growth × inflation.
# --------------------------------------------------------------------------
PILLARS: dict[str, dict] = {
    "growth": {
        "fred": ["INDPRO"],                 # industrial production
        "method": "yoy_trend",              # bit=1 if YoY momentum positive
        "hot_label": "EXPANDING", "cold_label": "CONTRACTING",
    },
    "inflation": {
        "fred": ["CPIAUCSL"],               # headline CPI level
        "method": "yoy_rising",             # bit=1 if YoY rising or > target band
        "hot_label": "RISING", "cold_label": "FALLING",
        "target_yoy": 2.5,
    },
    "rates": {
        "fred": ["DGS2"],                   # 2y Treasury (policy-path proxy)
        "method": "level_3m_change",        # bit=1 if rising (tightening)
        "hot_label": "TIGHTENING", "cold_label": "EASING",
    },
    "liquidity": {
        "fred": ["NFCI"],                   # Chicago Fed financial conditions
        "method": "level_positive",         # bit=1 if NFCI>0 (tight)
        "hot_label": "TIGHT", "cold_label": "LOOSE",
    },
}

REGIME_HISTORY_YEARS = 25        # how far back to label history for Stage 2
REGIME_RESAMPLE = "ME"           # month-end regime snapshots (reduce daily noise)

# --------------------------------------------------------------------------
# Stage 2 — conditional returns.
# --------------------------------------------------------------------------
MIN_OBS_PER_CELL = 60            # daily obs; below this → low-confidence flag

# Phase 2 — full 16-state estimation with hierarchical (James-Stein-style)
# shrinkage toward the 2x2 super-regime parent, and block-bootstrap CIs on
# the conditional Sharpe so we can discard cells whose CI straddles zero.
SHRINK_K = 120                   # pseudo-obs of parent prior; pooled = (n·cell + k·parent)/(n+k)
BOOTSTRAP_REPS = 400             # block-bootstrap resamples for the Sharpe CI
BOOTSTRAP_BLOCK = 21             # ~1 trading month per block (preserves autocorrelation)
CI_PCT = (5.0, 95.0)             # percentile bounds reported for the bootstrapped Sharpe
MIN_OBS_FOR_BOOTSTRAP = 40       # below this, skip bootstrap (CI = NaN, flag "thin")
SEED = 7                         # deterministic bootstrap

# --------------------------------------------------------------------------
# Stage 3 — divergences. Trailing window for z-scoring the gap.
# --------------------------------------------------------------------------
DIVERGENCE_Z_WINDOW = 252        # ~1y of daily obs
FAIRVALUE_REG_WINDOW = 504       # ~2y rolling OLS window for fair-value regressions
WIDE_STALE_Z = 2.0               # |z| above this AND ...
WIDE_STALE_DAYS = 63             # ... open longer than this many days → "wide & stale" flag

# --------------------------------------------------------------------------
# Stage 5 — convergence scoring weights + surface rule.
# --------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "regime":     1.0,   # Stage 2 conditional-Sharpe rank (normalized)
    "divergence": 1.2,   # Stage 3 gap_z — primary alpha source
    "crowding":   0.6,   # Stage 4 multiplier-as-signed-score
}
MIN_AGREEMENT = 2        # lenses must agree (Phase 1 has 3 lenses; doc target is 3)

# --------------------------------------------------------------------------
# Stage 6 — sizing.
# --------------------------------------------------------------------------
KELLY_FRACTION = 0.25            # fractional Kelly cap
TARGET_PORTFOLIO_VOL = 0.10      # 10% annualized vol budget per idea (standalone proxy)
MAX_WEIGHT = 0.15                # cap any single idea notional
PORTFOLIO_VOL_CAP = 0.12         # correlation-aware cap on the *whole book* (Stage 6 scales to this)

# --------------------------------------------------------------------------
# Stage 7 — falsification.
# --------------------------------------------------------------------------
ATR_STOP_MULT = 2.5              # price stop = entry ± mult × ATR(14)
DEFAULT_TIME_STOP_DAYS = 63      # ~one quarter catalyst window
JOURNAL_PATH = "macro_engine/runs/journal.jsonl"
