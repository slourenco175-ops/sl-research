"""Liquid commodity futures universe — energy, metals, ags, softs, livestock.

Each entry maps:
  yf:        yfinance front-month continuous symbol
  cot_name:  the contract name as it appears in CFTC disaggregated reports
  sector:    energy / metals / ags / softs / livestock
  name:      human-readable label for the dashboard

If a contract isn't appearing in COT data, set cot_name=None and we'll
just skip the positioning column for that one (the model still works).
"""

UNIVERSE = [
    # --- energy ---
    {"yf": "CL=F", "cot_name": "WTI-PHYSICAL",         "sector": "energy",    "name": "WTI Crude"},
    {"yf": "BZ=F", "cot_name": "BRENT LAST DAY",       "sector": "energy",    "name": "Brent Crude"},
    {"yf": "NG=F", "cot_name": "NAT GAS NYME",         "sector": "energy",    "name": "Nat Gas (HH)"},
    {"yf": "HO=F", "cot_name": "#2 HEATING OIL",       "sector": "energy",    "name": "Heating Oil"},
    {"yf": "RB=F", "cot_name": "GASOLINE RBOB",        "sector": "energy",    "name": "RBOB Gasoline"},

    # --- metals ---
    {"yf": "GC=F", "cot_name": "GOLD",                 "sector": "metals",    "name": "Gold"},
    {"yf": "SI=F", "cot_name": "SILVER",               "sector": "metals",    "name": "Silver"},
    {"yf": "HG=F", "cot_name": "COPPER-",              "sector": "metals",    "name": "Copper"},
    {"yf": "PL=F", "cot_name": "PLATINUM",             "sector": "metals",    "name": "Platinum"},
    {"yf": "PA=F", "cot_name": "PALLADIUM",            "sector": "metals",    "name": "Palladium"},

    # --- grains / oilseeds ---
    {"yf": "ZC=F", "cot_name": "CORN",                 "sector": "ags",       "name": "Corn"},
    {"yf": "ZS=F", "cot_name": "SOYBEANS",             "sector": "ags",       "name": "Soybeans"},
    {"yf": "ZW=F", "cot_name": "WHEAT-SRW",            "sector": "ags",       "name": "Wheat (SRW)"},
    {"yf": "ZL=F", "cot_name": "SOYBEAN OIL",          "sector": "ags",       "name": "Soybean Oil"},
    {"yf": "ZM=F", "cot_name": "SOYBEAN MEAL",         "sector": "ags",       "name": "Soybean Meal"},

    # --- softs ---
    {"yf": "KC=F", "cot_name": "COFFEE C",             "sector": "softs",     "name": "Coffee"},
    {"yf": "SB=F", "cot_name": "SUGAR NO. 11",         "sector": "softs",     "name": "Sugar"},
    {"yf": "CC=F", "cot_name": "COCOA",                "sector": "softs",     "name": "Cocoa"},
    {"yf": "CT=F", "cot_name": "COTTON NO. 2",         "sector": "softs",     "name": "Cotton"},

    # --- livestock ---
    {"yf": "LE=F", "cot_name": "LIVE CATTLE",          "sector": "livestock", "name": "Live Cattle"},
]
