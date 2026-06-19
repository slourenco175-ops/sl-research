"""Stage 3 — divergence scanner (primary alpha source).

Each divergence yields a signed gap (our macro view minus what the market price
embeds), z-scored over a trailing window. We then map each divergence to the
universe assets it implicates, with a per-asset sign (+1 long / -1 short when the
gap resolves in our favour). Two diagnostics per divergence: magnitude (gap_z)
and persistence (how long the sign has stood — wide-and-stale flags a trap).

Library (Phase 2): policy_path (Fed vs our path), inflation (breakeven vs
nowcast), growth (copper/gold + cyclicals vs growth nowcast), term_premium
(10y vs fair value on 2y + breakeven), credit_equity (HY OAS vs VIX-implied),
fx_carry (US 2y vs foreign policy rates). Fair-value rows use a rolling OLS
residual, z-scored and persistence-tagged. A "wide & stale" flag marks gaps
that are large yet have failed to close for a long time (possible value trap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forex.policy_path import latest as latest_policy
from macro_engine.config import (
    COUNTRY_COMMODITY,
    DIVERGENCE_Z_WINDOW,
    FAIRVALUE_REG_WINDOW,
    WIDE_STALE_DAYS,
    WIDE_STALE_Z,
)
from macro_engine.data import close_frame, fred_series, get_universe_prices

W = DIVERGENCE_Z_WINDOW

# Which assets each divergence implicates, and the sign of their exposure when
# the gap resolves toward OUR view (signed_gap > 0).
ASSET_SIGNS = {
    "policy_path":   {"UUP": +1, "TLT": -1, "IEF": -1, "GLD": -1},
    "inflation":     {"TIP": +1, "DBC": +1, "GLD": +1, "TLT": -1},
    "growth":        {"SPY": +1, "EEM": +1, "CPER": +1, "HYG": +1, "TLT": -1, "GLD": -1},
    # term_premium signed_gap>0 ⇒ 10y yield rich vs fair value (price low) ⇒ long duration
    "term_premium":  {"TLT": +1, "IEF": +1},
    # credit_equity signed_gap>0 ⇒ HY spread wide vs VIX-implied (credit cheap) ⇒ long credit, short equity
    "credit_equity": {"HYG": +1, "LQD": +1, "SPY": -1},
    # fx_carry signed_gap>0 ⇒ US carry rich vs foreign ⇒ long USD
    "fx_carry":      {"UUP": +1, "FXE": -1, "FXY": -1},
}


def _z(series: pd.Series, window: int = W) -> pd.Series:
    s = series.dropna()
    m = s.rolling(window, min_periods=window // 3).mean()
    sd = s.rolling(window, min_periods=window // 3).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def _persistence_days(signed_series: pd.Series) -> int:
    """Trailing run length where the sign of the gap is unchanged."""
    s = np.sign(signed_series.dropna())
    if s.empty:
        return 0
    last = s.iloc[-1]
    run = 0
    for v in s.iloc[::-1]:
        if v == last and v != 0:
            run += 1
        else:
            break
    return int(run)


def _wide_stale(gap_z: float, persistence_days: int) -> bool:
    """Large gap that has failed to close for a long time → possible value trap."""
    return abs(gap_z) >= WIDE_STALE_Z and persistence_days >= WIDE_STALE_DAYS


def _fairvalue_resid(y: pd.Series, xs: list[pd.Series],
                     window: int = FAIRVALUE_REG_WINDOW) -> pd.Series:
    """Residual of `y` vs an OLS fair value on `xs` (with intercept).

    Betas are fit on the trailing `window` aligned observations, then applied
    over the full aligned history so we get a residual *series* to z-score and
    measure persistence on. Positive residual = `y` rich vs its fair value.
    """
    df = pd.concat([y.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(xs)],
                   axis=1).dropna()
    if len(df) < max(window // 2, 30):
        return pd.Series(dtype=float)
    fit = df.tail(window)
    X = np.column_stack([np.ones(len(fit))] + [fit[f"x{i}"].to_numpy() for i in range(len(xs))])
    beta, *_ = np.linalg.lstsq(X, fit["y"].to_numpy(), rcond=None)
    Xall = np.column_stack([np.ones(len(df))] + [df[f"x{i}"].to_numpy() for i in range(len(xs))])
    fitted = Xall @ beta
    return pd.Series(df["y"].to_numpy() - fitted, index=df.index)


def _daily_yoy_from_monthly(series_id: str, index: pd.DatetimeIndex) -> pd.Series:
    s = fred_series(series_id)
    if s.empty:
        return pd.Series(dtype=float, index=index)
    m = s.resample("ME").last()
    yoy = (m / m.shift(12) - 1.0) * 100.0
    return yoy.reindex(index, method="ffill")


def scan_divergences(regime: dict) -> pd.DataFrame:
    prices = get_universe_prices()
    closes = close_frame(prices)
    idx = closes.index
    rows: list[dict] = []

    # ---- 1. policy_path — recorded Fed snapshot vs our rule/nowcast path -----
    snap = latest_policy()
    # Hawkish surprise magnitude: 2y move normalized by a ~5bp typical daily move.
    two_y_bp = snap.get("curve_bp", {}).get("2y", 0)
    signed_gap_pp = (two_y_bp / 5.0)            # +ve = hawkish (short duration / long USD)
    if snap.get("statement_bias") == "easing":
        signed_gap_pp *= -1
    for asset, sign in ASSET_SIGNS["policy_path"].items():
        rows.append({
            "divergence": "policy_path", "asset": asset,
            "gap_raw": f"2y {two_y_bp:+d}bp, dots {snap.get('dots_median_cy')}%",
            "gap_z": round(signed_gap_pp, 2),
            "divergence_z": round(signed_gap_pp * sign, 2),
            "persistence_days": snap.get("divergence", {}).get("persistence_days", 0),
            "direction": "long" if sign > 0 else "short",
            "supporting_pillars": "rates,inflation",
        })

    # ---- 2. inflation — market breakeven (T10YIE) vs our inflation nowcast ---
    be = fred_series("T10YIE")
    if not be.empty:
        be = be.reindex(idx, method="ffill")
        infl_now = _daily_yoy_from_monthly("CPIAUCSL", idx)
        bz, iz = _z(be), _z(infl_now)
        gap = (iz - bz).dropna()               # +ve = we see MORE inflation than priced
        if not gap.empty:
            signed_gap_infl = float(gap.iloc[-1])
            persist = _persistence_days(gap)
            for asset, sign in ASSET_SIGNS["inflation"].items():
                rows.append({
                    "divergence": "inflation", "asset": asset,
                    "gap_raw": f"breakeven z {bz.iloc[-1]:+.2f} vs nowcast z {iz.iloc[-1]:+.2f}",
                    "gap_z": round(signed_gap_infl, 2),
                    "divergence_z": round(signed_gap_infl * sign, 2),
                    "persistence_days": persist,
                    "direction": "long" if signed_gap_infl * sign > 0 else "short",
                    "supporting_pillars": "inflation",
                })

    # ---- 3. growth — copper/gold ratio vs our growth nowcast ----------------
    if "HG=F" in closes and "GC=F" in closes:
        cg = (closes["HG=F"] / closes["GC=F"]).dropna()
        grow_now = _daily_yoy_from_monthly("INDPRO", idx)
        cgz, gz = _z(cg), _z(grow_now)
        gap = (gz - cgz).dropna()              # +ve = we see MORE growth than priced
        if not gap.empty:
            signed_gap_grow = float(gap.iloc[-1])
            persist = _persistence_days(gap)
            for asset, sign in ASSET_SIGNS["growth"].items():
                rows.append({
                    "divergence": "growth", "asset": asset,
                    "gap_raw": f"copper/gold z {cgz.iloc[-1]:+.2f} vs growth z {gz.iloc[-1]:+.2f}",
                    "gap_z": round(signed_gap_grow, 2),
                    "divergence_z": round(signed_gap_grow * sign, 2),
                    "persistence_days": persist,
                    "direction": "long" if signed_gap_grow * sign > 0 else "short",
                    "supporting_pillars": "growth",
                })

    # ---- 4. term_premium — 10y yield vs fair value on 2y + 10y breakeven -----
    dgs10, dgs2, be10 = fred_series("DGS10"), fred_series("DGS2"), fred_series("T10YIE")
    if not dgs10.empty and not dgs2.empty and not be10.empty:
        resid = _fairvalue_resid(dgs10, [dgs2, be10])
        rz = _z(resid)
        if not rz.dropna().empty:
            signed = float(rz.dropna().iloc[-1])    # +ve = 10y yield rich (price cheap)
            persist = _persistence_days(rz)
            for asset, sign in ASSET_SIGNS["term_premium"].items():
                rows.append({
                    "divergence": "term_premium", "asset": asset,
                    "gap_raw": f"10y resid {resid.dropna().iloc[-1]:+.2f}pp vs fair(2y,BE)",
                    "gap_z": round(signed, 2),
                    "divergence_z": round(signed * sign, 2),
                    "persistence_days": persist,
                    "direction": "long" if signed * sign > 0 else "short",
                    "supporting_pillars": "rates",
                })

    # ---- 5. credit_equity — HY OAS vs fair value implied by equity vol (VIX) -
    oas = fred_series("BAMLH0A0HYM2")
    if not oas.empty and "^VIX" in closes:
        resid = _fairvalue_resid(oas, [closes["^VIX"]])
        rz = _z(resid)
        if not rz.dropna().empty:
            signed = float(rz.dropna().iloc[-1])    # +ve = HY spread wide vs VIX-implied (credit cheap)
            persist = _persistence_days(rz)
            for asset, sign in ASSET_SIGNS["credit_equity"].items():
                rows.append({
                    "divergence": "credit_equity", "asset": asset,
                    "gap_raw": f"HY OAS resid {resid.dropna().iloc[-1]:+.2f}pp vs VIX-fair",
                    "gap_z": round(signed, 2),
                    "divergence_z": round(signed * sign, 2),
                    "persistence_days": persist,
                    "direction": "long" if signed * sign > 0 else "short",
                    "supporting_pillars": "liquidity,growth",
                })

    # ---- 6. fx_carry — US 2y vs foreign policy rates (guarded; skip if stale) -
    foreign = []
    for fid in ("ECBDFR", "IRSTCB01JPM156N"):     # ECB deposit rate, BoJ overnight
        s = fred_series(fid)
        if not s.empty:
            foreign.append(s.resample("ME").last())
    if not dgs2.empty and foreign:
        us = dgs2.resample("ME").last()
        favg = pd.concat(foreign, axis=1).mean(axis=1)
        carry = (us - favg).dropna()              # +ve = US carry advantage
        cz = _z(carry, window=36)                 # monthly window ~3y
        if not cz.dropna().empty:
            signed = float(cz.dropna().iloc[-1])
            persist = _persistence_days(cz)
            for asset, sign in ASSET_SIGNS["fx_carry"].items():
                rows.append({
                    "divergence": "fx_carry", "asset": asset,
                    "gap_raw": f"US 2y {us.iloc[-1]:.2f}% vs foreign {favg.iloc[-1]:.2f}% (z {signed:+.2f})",
                    "gap_z": round(signed, 2),
                    "divergence_z": round(signed * sign, 2),
                    "persistence_days": persist,
                    "direction": "long" if signed * sign > 0 else "short",
                    "supporting_pillars": "rates",
                })

    # ---- 7. terms_of_trade — commodity-exporter equity vs its export proxy ---
    # Regress log(country) on log(commodity) over a rolling window; the residual
    # is the country's mispricing vs what its key export implies. Cheap (residual
    # < 0) ⇒ long country; rich ⇒ short. One row per country (its own sign +1).
    for country, comm in COUNTRY_COMMODITY.items():
        if country not in closes or comm not in closes:
            continue
        y = np.log(closes[country].replace(0, np.nan))
        x = np.log(closes[comm].replace(0, np.nan))
        resid = _fairvalue_resid(y, [x])
        rz = _z(resid)
        if rz.dropna().empty:
            continue
        signed = -float(rz.dropna().iloc[-1])     # +ve = country cheap vs export ⇒ long
        persist = _persistence_days(-rz)          # run length of the (cheap/rich) sign
        rows.append({
            "divergence": "terms_of_trade", "asset": country,
            "gap_raw": f"{country} resid {-resid.dropna().iloc[-1]:+.2f} vs {comm} fair (z {signed:+.2f})",
            "gap_z": round(signed, 2),
            "divergence_z": round(signed, 2),     # country sign is +1
            "persistence_days": persist,
            "direction": "long" if signed > 0 else "short",
            "supporting_pillars": "growth",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["wide_stale"] = df.apply(
            lambda r: _wide_stale(r["gap_z"], r["persistence_days"]), axis=1
        )
    return df
