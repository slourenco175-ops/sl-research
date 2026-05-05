"""Liquid commodity futures universe — energy, metals, ags, softs, livestock.

Each entry maps:
  yf:             yfinance front-month continuous symbol
  cot_name:       the contract name as it appears in CFTC disaggregated reports
  sector:         energy / metals / ags / softs / livestock
  name:           human-readable label for the dashboard
  curve_root:     yfinance symbol root for deferred-month contracts (None to skip)
  inventory_key:  matches a key in commodities.inventories.INVENTORY_SERIES
                  (None for contracts without a clean weekly inventory feed)

If a contract isn't appearing in COT data, set cot_name=None and we'll
just skip the positioning column for that one (the model still works).
"""

UNIVERSE = [
    # --- energy ---
    {"yf": "CL=F", "cot_name": "WTI-PHYSICAL",   "curve_root": "CL", "inventory_key": "cl",   "sector": "energy",    "name": "WTI Crude"},
    {"yf": "BZ=F", "cot_name": "BRENT LAST DAY", "curve_root": "BZ", "inventory_key": "cl",   "sector": "energy",    "name": "Brent Crude"},
    {"yf": "NG=F", "cot_name": "NAT GAS NYME",   "curve_root": "NG", "inventory_key": "ng",   "sector": "energy",    "name": "Nat Gas (HH)"},
    {"yf": "HO=F", "cot_name": "#2 HEATING OIL", "curve_root": "HO", "inventory_key": "ho",   "sector": "energy",    "name": "Heating Oil"},
    {"yf": "RB=F", "cot_name": "GASOLINE RBOB",  "curve_root": "RB", "inventory_key": "rb",   "sector": "energy",    "name": "RBOB Gasoline"},

    # --- metals ---
    {"yf": "GC=F", "cot_name": "GOLD",           "curve_root": "GC", "inventory_key": None,   "sector": "metals",    "name": "Gold"},
    {"yf": "SI=F", "cot_name": "SILVER",         "curve_root": "SI", "inventory_key": None,   "sector": "metals",    "name": "Silver"},
    {"yf": "HG=F", "cot_name": "COPPER-",        "curve_root": "HG", "inventory_key": None,   "sector": "metals",    "name": "Copper"},
    {"yf": "PL=F", "cot_name": "PLATINUM",       "curve_root": "PL", "inventory_key": None,   "sector": "metals",    "name": "Platinum"},
    {"yf": "PA=F", "cot_name": "PALLADIUM",      "curve_root": "PA", "inventory_key": None,   "sector": "metals",    "name": "Palladium"},

    # --- grains / oilseeds ---
    {"yf": "ZC=F", "cot_name": "CORN",           "curve_root": "ZC", "inventory_key": None,   "sector": "ags",       "name": "Corn"},
    {"yf": "ZS=F", "cot_name": "SOYBEANS",       "curve_root": "ZS", "inventory_key": None,   "sector": "ags",       "name": "Soybeans"},
    {"yf": "ZW=F", "cot_name": "WHEAT-SRW",     "curve_root": "ZW", "inventory_key": None,   "sector": "ags",       "name": "Wheat (SRW)"},
    {"yf": "ZL=F", "cot_name": "SOYBEAN OIL",    "curve_root": "ZL", "inventory_key": None,   "sector": "ags",       "name": "Soybean Oil"},
    {"yf": "ZM=F", "cot_name": "SOYBEAN MEAL",   "curve_root": "ZM", "inventory_key": None,   "sector": "ags",       "name": "Soybean Meal"},

    # --- softs ---
    {"yf": "KC=F", "cot_name": "COFFEE C",       "curve_root": "KC", "inventory_key": None,   "sector": "softs",     "name": "Coffee"},
    {"yf": "SB=F", "cot_name": "SUGAR NO. 11",   "curve_root": "SB", "inventory_key": None,   "sector": "softs",     "name": "Sugar"},
    {"yf": "CC=F", "cot_name": "COCOA",          "curve_root": "CC", "inventory_key": None,   "sector": "softs",     "name": "Cocoa"},
    {"yf": "CT=F", "cot_name": "COTTON NO. 2",   "curve_root": "CT", "inventory_key": None,   "sector": "softs",     "name": "Cotton"},

    # --- livestock ---
    {"yf": "LE=F", "cot_name": "LIVE CATTLE",    "curve_root": "LE", "inventory_key": None,   "sector": "livestock", "name": "Live Cattle"},
]
