"""Cross-asset correlation / driver-signal engine.

Three jobs:
  1. Measure how every tradable asset responds to a panel of macro DRIVERS
     (commodities, FX, rates, risk) — correlation, beta, significance (t-stat)
     and *stability* (does the sign hold across rolling windows?).
  2. Keep only the robust + stable relationships (the real economic links, e.g.
     crude→Norway, broad-commodities→Brazil, copper→Chile) and discard noise.
  3. Turn recent driver moves into directional signals on each asset, then
     validate them with a forward-return information coefficient (IC).

The signal for an asset is the correlation-weighted sum of its robust drivers'
recent (z-scored) moves: an asset positively linked to a driver that just
rallied gets a long tilt. Relationship weights are full-sample (in-sample), but
the move timing and the forward return used for IC are out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forex.fx_data import get_ohlc
from macro_engine.config import (
    CORR_ABS_MIN,
    CORR_MIN_YEARS,
    CORR_STABLE_MIN,
    CORR_T_MIN,
    DRIVERS,
    LEAD_HORIZONS,
    LEAD_IC_MIN,
    LEAD_MIN_SAMPLES,
    SIGNAL_FWD_DAYS,
    SIGNAL_MOVE_WINDOW,
    SIGNAL_Z_WINDOW,
    UNIVERSE,
)
from macro_engine.data import daily_returns, fred_series, get_universe_prices

TRADING_DAYS = 252
NAMES = {u["sym"]: u["name"] for u in UNIVERSE}
BUCKET = {u["sym"]: u["bucket"] for u in UNIVERSE}
DRIVER_GRP = {d["name"]: d["grp"] for d in DRIVERS}


# ------------------------------- data panels ---------------------------------

def driver_panel() -> pd.DataFrame:
    """Daily transformed driver series (columns = driver display names)."""
    yf_syms = [d["sym"] for d in DRIVERS if "sym" in d]
    ohlc = get_ohlc(yf_syms, lookback_days=9000)
    cols: dict[str, pd.Series] = {}
    for d in DRIVERS:
        if "sym" in d:
            df = ohlc.get(d["sym"])
            if df is None or df.empty or "Close" not in df:
                continue
            s = df["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            cols[d["name"]] = s.pct_change()              # px / vix
        else:
            raw = fred_series(d["fred"])
            if raw.empty:
                continue
            cols[d["name"]] = raw.reindex(
                raw.index.union(pd.bdate_range(raw.index.min(), raw.index.max()))
            ).ffill().diff() * 100.0                       # yld → Δbp
    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    return panel


def target_panel() -> pd.DataFrame:
    """Daily returns of the tradable universe."""
    return daily_returns(get_universe_prices())


# --------------------------- relationship measurement ------------------------

def _pair_stats(y: pd.Series, x: pd.Series) -> dict | None:
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(j)
    if n < CORR_MIN_YEARS * TRADING_DAYS:
        return None
    corr = float(j["y"].corr(j["x"]))
    if np.isnan(corr):
        return None
    var_x = float(j["x"].var(ddof=0))
    beta = float(j[["y", "x"]].cov().iloc[0, 1] / var_x) if var_x > 0 else 0.0
    t = corr * np.sqrt(max(n - 2, 1) / max(1 - corr ** 2, 1e-9))
    roll = j["y"].rolling(TRADING_DAYS).corr(j["x"]).dropna()
    stable = float((np.sign(roll) == np.sign(corr)).mean()) if len(roll) else 0.0
    return {"corr": corr, "beta": beta, "t": float(t), "n_years": round(n / TRADING_DAYS, 1),
            "stable_frac": round(stable, 2)}


def relationship_table(targets: list[str] | None = None) -> pd.DataFrame:
    """Long table of every (asset × driver) relationship with a robustness flag."""
    dpan = driver_panel()
    tpan = target_panel()
    targets = targets or [u["sym"] for u in UNIVERSE]
    rows = []
    for sym in targets:
        if sym not in tpan:
            continue
        y = tpan[sym]
        for drv in dpan.columns:
            if drv == NAMES.get(sym):       # skip self-ish (rare overlap)
                continue
            st = _pair_stats(y, dpan[drv])
            if st is None:
                continue
            robust = (abs(st["corr"]) >= CORR_ABS_MIN and abs(st["t"]) >= CORR_T_MIN
                      and st["stable_frac"] >= CORR_STABLE_MIN)
            rows.append({"asset": sym, "name": NAMES.get(sym, sym), "bucket": BUCKET.get(sym, ""),
                         "driver": drv, "driver_grp": DRIVER_GRP.get(drv, ""), **st, "robust": robust})
    df = pd.DataFrame(rows)
    return df.sort_values(["asset", "corr"], key=lambda c: c.abs() if c.name == "corr" else c,
                          ascending=[True, False]).reset_index(drop=True)


def correlation_matrix(targets: list[str] | None = None) -> pd.DataFrame:
    """Asset × driver full-sample correlation matrix (for the heatmap)."""
    rel = relationship_table(targets)
    if rel.empty:
        return pd.DataFrame()
    return rel.pivot(index="asset", columns="driver", values="corr")


# ------------------------------ signal building ------------------------------

def _driver_move_z(dpan: pd.DataFrame) -> pd.DataFrame:
    """Z-scored trailing move of each driver (period return / net Δ)."""
    move = dpan.rolling(SIGNAL_MOVE_WINDOW).sum()           # ~period return / net bp
    mu = move.rolling(SIGNAL_Z_WINDOW).mean()
    sd = move.rolling(SIGNAL_Z_WINDOW).std(ddof=0)
    return (move - mu) / sd.replace(0, np.nan)


def build_signals(rel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Current driver-implied signal per asset (correlation-weighted move-z)."""
    rel = rel if rel is not None else relationship_table()
    robust = rel[rel["robust"]]
    if robust.empty:
        return pd.DataFrame()
    movez = _driver_move_z(driver_panel())
    latest_move = movez.ffill().iloc[-1]

    rows = []
    for sym, grp in robust.groupby("asset"):
        num, den, contribs = 0.0, 0.0, []
        for _, r in grp.iterrows():
            mz = latest_move.get(r["driver"], np.nan)
            if pd.isna(mz):
                continue
            c = r["corr"] * float(mz)        # corr carries sign + weight
            num += c
            den += abs(r["corr"])
            contribs.append((r["driver"], r["corr"], float(mz), c))
        if den == 0:
            continue
        signal = num / den
        contribs.sort(key=lambda t: -abs(t[3]))
        top = contribs[0]
        rows.append({
            "asset": sym, "name": NAMES.get(sym, sym), "bucket": BUCKET.get(sym, ""),
            "signal": round(signal, 3),
            "direction": "LONG" if signal > 0 else ("SHORT" if signal < 0 else "—"),
            "n_drivers": len(contribs),
            "top_driver": top[0],
            "top_corr": round(top[1], 2),
            "top_move_z": round(top[2], 2),
            "drivers": "; ".join(f"{d}({co:+.2f})" for d, co, _, _ in contribs[:4]),
        })
    return pd.DataFrame(rows).sort_values("signal", key=lambda c: c.abs(),
                                          ascending=False).reset_index(drop=True)


# ------------------------------ IC validation --------------------------------

def validate_ic(rel: pd.DataFrame | None = None) -> dict:
    """Forward-return information coefficient of the driver signals.

    Builds each asset's signal *series* (in-sample corr weights, out-of-sample
    move timing), samples month-end, and correlates with the next ~21d return.
    """
    rel = rel if rel is not None else relationship_table()
    robust = rel[rel["robust"]]
    if robust.empty:
        return {}
    dpan = driver_panel()
    movez = _driver_move_z(dpan)
    tpan = target_panel()

    per_asset, all_sig, all_fwd = {}, [], []
    for sym, grp in robust.groupby("asset"):
        if sym not in tpan:
            continue
        w = grp.set_index("driver")["corr"]
        cols = [d for d in w.index if d in movez.columns]
        if not cols:
            continue
        sig = (movez[cols] * w[cols]).sum(axis=1) / w[cols].abs().sum()
        fwd = tpan[sym].rolling(SIGNAL_FWD_DAYS).sum().shift(-SIGNAL_FWD_DAYS)
        j = pd.concat([sig.rename("s"), fwd.rename("f")], axis=1).dropna()
        j = j.resample("ME").last().dropna()             # reduce overlap
        if len(j) < 24:
            continue
        ic = float(j["s"].corr(j["f"]))
        hit = float((np.sign(j["s"]) == np.sign(j["f"])).mean())
        per_asset[sym] = {"ic": round(ic, 3), "hit": round(hit * 100, 0), "n": len(j)}
        all_sig.append(j["s"]); all_fwd.append(j["f"])

    if not all_sig:
        return {}
    S, F = pd.concat(all_sig), pd.concat(all_fwd)
    pooled_ic = float(S.corr(F))
    pooled_hit = float((np.sign(S) == np.sign(F)).mean())
    ics = pd.Series({k: v["ic"] for k, v in per_asset.items()})
    return {"pooled_ic": round(pooled_ic, 3), "pooled_hit": round(pooled_hit * 100, 0),
            "n_obs": int(len(S)), "mean_asset_ic": round(float(ics.mean()), 3),
            "pct_positive_ic": round(float((ics > 0).mean()) * 100, 0),
            "per_asset": per_asset}


# ------------------------------ lead-lag hunt --------------------------------

def _fwd_ic(drv_move: pd.Series, ast_ret: pd.Series, h: int) -> tuple[float, int, float, float]:
    """IC between a trailing driver move and the asset's *forward* h-day return.

    De-overlapped by sampling every h-th observation, so each point is a roughly
    independent (driver-as-of-t, asset-return-t→t+h) pair. Positive horizon = the
    driver's past move leading the asset's future return. Also returns the IC on
    each half of the sample — a real lead must survive out of sample, not just
    show up once. Returns (ic_full, n, ic_first_half, ic_second_half).
    """
    fwd = ast_ret.rolling(h).sum().shift(-h)
    j = pd.concat([drv_move.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(j) < h:
        return float("nan"), 0, float("nan"), float("nan")
    j = j.iloc[::h]                                # reduce overlap to ~independent
    if len(j) < LEAD_MIN_SAMPLES:
        return float("nan"), len(j), float("nan"), float("nan")
    full = float(j["s"].corr(j["f"]))
    mid = len(j) // 2
    h1 = float(j["s"].iloc[:mid].corr(j["f"].iloc[:mid]))
    h2 = float(j["s"].iloc[mid:].corr(j["f"].iloc[mid:]))
    return full, len(j), h1, h2


def lead_lag_table(rel: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each robust contemporaneous pair, scan whether the driver *leads* the
    asset. Compares same-day correlation against forward IC at several horizons
    and flags pairs where a past driver move genuinely predicts forward returns
    (not just co-moves)."""
    rel = rel if rel is not None else relationship_table()
    robust = rel[rel["robust"]]
    if robust.empty:
        return pd.DataFrame()
    dpan = driver_panel()
    tpan = target_panel()
    move = dpan.rolling(SIGNAL_MOVE_WINDOW).sum()      # trailing driver move

    rows = []
    for _, r in robust.iterrows():
        sym, drv = r["asset"], r["driver"]
        if sym not in tpan or drv not in move.columns:
            continue
        ar = tpan[sym]
        dm = move[drv]
        ics = {}
        for h in LEAD_HORIZONS:
            ic, n, h1, h2 = _fwd_ic(dm, ar, h)
            if not np.isnan(ic):
                ics[h] = (ic, n, h1, h2)
        if not ics:
            continue
        # best forward horizon (exclude h=1 noise) by |IC|, same sign as the link
        cand = {h: v for h, v in ics.items() if h >= 3}
        if not cand:
            continue
        best_h = max(cand, key=lambda h: abs(cand[h][0]))
        best_ic, best_n, ic_h1, ic_h2 = cand[best_h]
        contemp = float(r["corr"])
        # split-sample survival: same sign in BOTH halves and same sign as full IC
        stable_oos = (np.sign(ic_h1) == np.sign(ic_h2) == np.sign(best_ic)
                      and min(abs(ic_h1), abs(ic_h2)) >= LEAD_IC_MIN / 2)
        predictive = (abs(best_ic) >= LEAD_IC_MIN
                      and np.sign(best_ic) == np.sign(contemp)
                      and best_n >= LEAD_MIN_SAMPLES
                      and stable_oos)
        rows.append({
            "asset": sym, "name": NAMES.get(sym, sym), "bucket": BUCKET.get(sym, ""),
            "driver": drv, "driver_grp": DRIVER_GRP.get(drv, ""),
            "contemp_corr": round(contemp, 2),
            "best_h": best_h, "lead_ic": round(best_ic, 3), "n": best_n,
            "ic_h1": round(ic_h1, 3), "ic_h2": round(ic_h2, 3),
            "edge": round(abs(best_ic) - abs(contemp), 3),   # forward vs same-day
            "stable_oos": stable_oos, "predictive": predictive,
            "ic_by_h": {h: round(v[0], 3) for h, v in ics.items()},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("lead_ic", key=lambda c: c.abs(),
                          ascending=False).reset_index(drop=True)
