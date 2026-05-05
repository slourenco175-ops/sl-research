"""Forex universe — majors + key crosses.

Each pair entry maps:
  yf:        yfinance symbol (e.g. EURUSD=X)
  base:      base currency (3-letter ISO)
  quote:     quote currency (3-letter ISO)
  name:      human-readable label
  group:     'major' | 'cross'

CURRENCIES holds per-currency metadata used for fundamentals + commodity overlay:
  cot_name:    CFTC TFF contract substring (None = USD, since DXY tracks it)
  fred:        FRED series IDs for {y10, policy, unemp, cpi, gdp}
  commodities: dict of {yf_symbol: weight in [-1, +1]} commodity tailwind score
"""

UNIVERSE = [
    # --- majors (7) ---
    {"yf": "EURUSD=X", "base": "EUR", "quote": "USD", "name": "Euro / US Dollar",        "group": "major"},
    {"yf": "GBPUSD=X", "base": "GBP", "quote": "USD", "name": "Pound / US Dollar",       "group": "major"},
    {"yf": "USDJPY=X", "base": "USD", "quote": "JPY", "name": "US Dollar / Yen",         "group": "major"},
    {"yf": "USDCHF=X", "base": "USD", "quote": "CHF", "name": "US Dollar / Swiss Franc", "group": "major"},
    {"yf": "AUDUSD=X", "base": "AUD", "quote": "USD", "name": "Aussie / US Dollar",      "group": "major"},
    {"yf": "USDCAD=X", "base": "USD", "quote": "CAD", "name": "US Dollar / Canadian",    "group": "major"},
    {"yf": "NZDUSD=X", "base": "NZD", "quote": "USD", "name": "Kiwi / US Dollar",        "group": "major"},

    # --- key crosses (5) ---
    {"yf": "EURGBP=X", "base": "EUR", "quote": "GBP", "name": "Euro / Pound",            "group": "cross"},
    {"yf": "EURJPY=X", "base": "EUR", "quote": "JPY", "name": "Euro / Yen",              "group": "cross"},
    {"yf": "GBPJPY=X", "base": "GBP", "quote": "JPY", "name": "Pound / Yen",             "group": "cross"},
    {"yf": "AUDJPY=X", "base": "AUD", "quote": "JPY", "name": "Aussie / Yen",            "group": "cross"},
    {"yf": "EURCHF=X", "base": "EUR", "quote": "CHF", "name": "Euro / Swiss Franc",      "group": "cross"},
]


# Per-currency metadata. FRED IDs are OECD-harmonized series where possible
# so the same column names work cross-country. All series are public + free
# (no API key needed via the fredgraph CSV endpoint).
CURRENCIES = {
    "USD": {
        "name": "US Dollar",
        "country": "United States",
        "cot_name": None,  # for USD use DXY index (USD INDEX in TFF)
        "dxy_cot_name": "USD INDEX",
        "fred": {
            "y10":    "DGS10",                 # 10y Treasury (daily, %)
            "policy": "DFF",                   # Effective fed funds (daily, %)
            "unemp":  "UNRATE",                # Civilian unemployment (monthly, %)
            "cpi":    "CPIAUCSL",              # CPI level (monthly) — YoY computed
            "gdp":    "GDPC1",                 # Real GDP (quarterly) — YoY computed
        },
        "commodities": {},  # USD doesn't have a single commodity tie
    },
    "EUR": {
        "name": "Euro",
        "country": "Eurozone",
        "cot_name": "EURO FX",
        "fred": {
            "y10":    "IRLTLT01EZM156N",       # 10y EZ benchmark (monthly, %)
            "policy": "ECBDFR",                # ECB deposit facility (daily, %)
            "unemp":  "LRHUTTTTEZM156S",       # EZ harmonized unemployment (monthly, %)
            "cpi":    "CP0000EZ19M086NEST",    # EZ HICP level (monthly) — YoY computed
            "gdp":    "CLVMNACSCAB1GQEA19",    # EZ real GDP (quarterly) — YoY computed
        },
        "commodities": {},
    },
    "GBP": {
        "name": "British Pound",
        "country": "United Kingdom",
        "cot_name": "BRITISH POUND",
        "fred": {
            "y10":    "IRLTLT01GBM156N",
            "policy": "IRSTCB01GBM156N",
            "unemp":  "LMUNRRTTGBM156S",
            "cpi":    "GBRCPIALLMINMEI",
            "gdp":    "CLVMNACSCAB1GQUK",
        },
        "commodities": {},
    },
    "JPY": {
        "name": "Japanese Yen",
        "country": "Japan",
        "cot_name": "JAPANESE YEN",
        "fred": {
            "y10":    "IRLTLT01JPM156N",
            "policy": "IRSTCB01JPM156N",
            "unemp":  "LRUN64TTJPM156S",
            "cpi":    "JPNCPIALLMINMEI",
            "gdp":    "JPNRGDPEXP",
        },
        # JPY is a major energy importer; rising oil = JPY headwind.
        "commodities": {"CL=F": -0.5},
    },
    "CHF": {
        "name": "Swiss Franc",
        "country": "Switzerland",
        "cot_name": "SWISS FRANC",
        "fred": {
            "y10":    "IRLTLT01CHM156N",
            "policy": "IRSTCB01CHM156N",
            "unemp":  "LMUNRRTTCHM156S",
            "cpi":    "CHECPIALLMINMEI",
            "gdp":    "CLVMNACSCAB1GQCH",
        },
        # CHF often trades with gold as a safe-haven asset.
        "commodities": {"GC=F": 0.4},
    },
    "CAD": {
        "name": "Canadian Dollar",
        "country": "Canada",
        "cot_name": "CANADIAN DOLLAR",
        "fred": {
            "y10":    "IRLTLT01CAM156N",
            "policy": "IRSTCB01CAM156N",
            "unemp":  "LRHUTTTTCAM156S",
            "cpi":    "CANCPIALLMINMEI",
            "gdp":    "NAEXKP01CAQ652S",
        },
        # WTI crude is the dominant CAD commodity link.
        "commodities": {"CL=F": 0.7},
    },
    "AUD": {
        "name": "Australian Dollar",
        "country": "Australia",
        "cot_name": "AUSTRALIAN DOLLAR",
        "fred": {
            "y10":    "IRLTLT01AUM156N",
            "policy": "IRSTCB01AUM156N",
            "unemp":  "LRHUTTTTAUM156S",
            "cpi":    "AUSCPIALLQINMEI",
            "gdp":    "AUSGDPRQDSMEI",
        },
        # No iron-ore yfinance proxy; copper + gold are reasonable stand-ins.
        "commodities": {"HG=F": 0.5, "GC=F": 0.3},
    },
    "NZD": {
        "name": "New Zealand Dollar",
        "country": "New Zealand",
        "cot_name": "NZ DOLLAR",
        "fred": {
            "y10":    "IRLTLT01NZM156N",
            "policy": "IRSTCB01NZM156N",
            "unemp":  "LRHUTTTTNZQ156S",
            "cpi":    "NZLCPIALLQINMEI",
            "gdp":    "NZLGDPRQDSMEI",
        },
        # No free dairy futures proxy on yfinance — use copper as a risk/AU proxy.
        "commodities": {"HG=F": 0.4},
    },
}


def all_currencies() -> list[str]:
    """Distinct list of currency codes used across the universe."""
    seen = []
    for u in UNIVERSE:
        for c in (u["base"], u["quote"]):
            if c not in seen:
                seen.append(c)
    return seen
