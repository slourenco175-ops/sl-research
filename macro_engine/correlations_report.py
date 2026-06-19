"""HTML report for the cross-asset correlation / driver-signal engine.

Sections:
  1. Verdict — what the correlations are good for (and what they are NOT).
  2. IC validation — does the driver-move signal predict forward returns? (honest)
  3. Driver × asset correlation heatmap (the real deliverable).
  4. Robust relationships table (crude→Norway, copper→Chile, ...).
  5. Current driver-implied signals (contemporaneous read).

House dark-gold style; written to macro_engine/runs/; opened in Safari unless NO_OPEN=1.
"""
from __future__ import annotations

import datetime as dt
import html
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from macro_engine.config import (
    CORR_ABS_MIN,
    CORR_MIN_YEARS,
    CORR_STABLE_MIN,
    CORR_T_MIN,
    LEAD_IC_MIN,
    SIGNAL_MOVE_WINDOW,
    SIGNAL_FWD_DAYS,
)
from macro_engine.correlations import (
    build_signals,
    correlation_matrix,
    lead_lag_table,
    relationship_table,
    validate_ic,
)
from macro_engine.dashboard import CSS

RUNS = Path(__file__).resolve().parent / "runs"


def _esc(x) -> str:
    return html.escape(str(x))


def _corr_cell(v: float) -> str:
    """Green = positive (moves with driver), red = negative; intensity by |corr|."""
    if pd.isna(v):
        return '<td class="muted" style="text-align:center">·</td>'
    a = min(abs(v), 1.0)
    if v >= 0:
        bg = f"rgba(92,184,92,{0.12 + 0.6 * a:.2f})"      # green
    else:
        bg = f"rgba(255,107,107,{0.12 + 0.6 * a:.2f})"    # red
    fg = "#fff" if a > 0.45 else "#cbd"
    return f'<td style="background:{bg};color:{fg};text-align:center">{v:+.2f}</td>'


def _heatmap(cm: pd.DataFrame, rel: pd.DataFrame) -> str:
    if cm.empty:
        return '<p class="muted">no relationships</p>'
    # order drivers by group then name; order assets by bucket then symbol
    grp = rel.drop_duplicates("driver").set_index("driver")["driver_grp"].to_dict()
    drivers = sorted(cm.columns, key=lambda d: (str(grp.get(d, "")), d))
    bkt = rel.drop_duplicates("asset").set_index("asset")["bucket"].to_dict()
    assets = sorted(cm.index, key=lambda a: (str(bkt.get(a, "")), a))

    head = "".join(f'<th class="rot">{_esc(d)}</th>' for d in drivers)
    rows = ""
    for a in assets:
        cells = "".join(_corr_cell(cm.loc[a, d]) for d in drivers)
        rows += f'<tr><td><b>{_esc(a)}</b> <span class="muted">{_esc(bkt.get(a,""))}</span></td>{cells}</tr>'
    return (f'<div style="overflow-x:auto"><table class="heat"><thead><tr>'
            f'<th style="text-align:left">asset \\ driver</th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<div class="muted" style="margin-top:6px">'
            f'<span class="legpos">■</span> positive (moves <i>with</i> driver) &nbsp;·&nbsp; '
            f'<span class="legneg">■</span> negative (moves <i>against</i>) &nbsp;·&nbsp; '
            f'full-sample daily correlation, all pairs shown (robust or not).</div>')


def _robust_table(rel: pd.DataFrame) -> str:
    r = rel[rel["robust"]].copy()
    if r.empty:
        return '<p class="muted">no robust relationships</p>'
    r = r.reindex(r["corr"].abs().sort_values(ascending=False).index)
    body = ""
    for _, x in r.head(60).iterrows():
        cc = "long" if x["corr"] > 0 else "short"
        body += (f'<tr><td><b>{_esc(x["asset"])}</b> {_esc(x["name"])}</td>'
                 f'<td style="text-align:left">{_esc(x["driver"])} '
                 f'<span class="muted">{_esc(x["driver_grp"])}</span></td>'
                 f'<td class="{cc}">{x["corr"]:+.2f}</td>'
                 f'<td>{x["beta"]:+.2f}</td>'
                 f'<td>{x["t"]:.0f}</td>'
                 f'<td>{x["n_years"]:.1f}y</td>'
                 f'<td>{x["stable_frac"]:.0%}</td></tr>')
    return (f'<table><thead><tr><th style="text-align:left">asset</th>'
            f'<th style="text-align:left">driver</th><th>corr</th><th>beta</th>'
            f'<th>t</th><th>span</th><th>stable</th></tr></thead><tbody>{body}</tbody></table>'
            f'<div class="muted" style="margin-top:4px">Robust gate: |corr|≥{CORR_ABS_MIN}, '
            f'|t|≥{CORR_T_MIN:.0f}, sign-stable ≥{CORR_STABLE_MIN:.0%} of rolling windows, '
            f'≥{CORR_MIN_YEARS}y history. Top 60 of {int(rel["robust"].sum())} robust links by |corr|.</div>')


def _signals_table(sig: pd.DataFrame) -> str:
    if sig.empty:
        return '<p class="muted">no signals</p>'
    body = ""
    for _, x in sig.iterrows():
        dcls = "long" if x["direction"] == "LONG" else ("short" if x["direction"] == "SHORT" else "muted")
        body += (f'<tr><td><b>{_esc(x["asset"])}</b> {_esc(x["name"])}</td>'
                 f'<td class="{dcls}">{x["direction"]}</td>'
                 f'<td data-sort="{x["signal"]}">{x["signal"]:+.2f}</td>'
                 f'<td>{x["n_drivers"]}</td>'
                 f'<td style="text-align:left">{_esc(x["top_driver"])} '
                 f'<span class="muted">({x["top_corr"]:+.2f}×z{x["top_move_z"]:+.1f})</span></td>'
                 f'<td style="text-align:left">{_esc(x["drivers"])}</td></tr>')
    return (f'<table><thead><tr><th style="text-align:left">asset</th><th>dir</th>'
            f'<th>signal</th><th>#drv</th><th style="text-align:left">top driver</th>'
            f'<th style="text-align:left">contributors (corr)</th></tr></thead>'
            f'<tbody>{body}</tbody></table>'
            f'<div class="muted" style="margin-top:4px">signal = Σ(corr·move-z)/Σ|corr| over robust drivers; '
            f'move = trailing {SIGNAL_MOVE_WINDOW}d, z-scored. '
            f'<b>Contemporaneous read</b> — see IC panel before trusting it as a forward signal.</div>')


def _leadlag_panel(ll: pd.DataFrame) -> str:
    if ll is None or ll.empty:
        return '<p class="muted">no lead-lag results</p>'
    genuine = ll[ll["predictive"] & (ll["edge"] > 0)].copy()
    genuine = genuine.reindex(genuine["lead_ic"].abs().sort_values(ascending=False).index)
    n_scan, n_oos, n_gen = len(ll), int(ll["predictive"].sum()), len(genuine)

    cards = (
        f'<div class="pillars" style="grid-template-columns:repeat(3,1fr)">'
        f'<div class="pill"><div class="lab">{n_scan}</div><div class="v">robust pairs scanned</div></div>'
        f'<div class="pill"><div class="lab">{n_oos}</div><div class="v">survive both half-samples</div></div>'
        f'<div class="pill"><div class="lab">{n_gen}</div>'
        f'<div class="v">genuine leads (fwd IC &gt; same-day)</div></div></div>')

    body = ""
    for _, x in genuine.head(30).iterrows():
        cc = "long" if x["lead_ic"] > 0 else "short"
        body += (f'<tr><td><b>{_esc(x["driver"])}</b> '
                 f'<span class="muted">{_esc(x["driver_grp"])}</span></td>'
                 f'<td style="text-align:left">→ <b>{_esc(x["asset"])}</b> {_esc(x["name"])}</td>'
                 f'<td>{x["best_h"]}d</td>'
                 f'<td class="{cc}">{x["lead_ic"]:+.2f}</td>'
                 f'<td>{x["ic_h1"]:+.2f}</td><td>{x["ic_h2"]:+.2f}</td>'
                 f'<td>{x["contemp_corr"]:+.2f}</td>'
                 f'<td class="long">+{x["edge"]:.2f}</td>'
                 f'<td>{x["n"]}</td></tr>')
    tbl = (f'<table><thead><tr><th style="text-align:left">driver leads</th>'
           f'<th style="text-align:left">asset</th><th>horizon</th><th>lead IC</th>'
           f'<th>IC ½₁</th><th>IC ½₂</th><th>same-day</th><th>edge</th><th>n</th>'
           f'</tr></thead><tbody>{body}</tbody></table>')

    note = (f'<div class="muted" style="margin-top:6px"><b>Lead IC</b> = corr(driver\'s trailing '
            f'{SIGNAL_MOVE_WINDOW}d move, asset\'s <i>forward</i> return over the horizon), '
            f'de-overlapped. <b>IC ½₁/½₂</b> are the two half-samples — only pairs whose lead holds '
            f'in <i>both</i> halves are shown (kills most multiple-testing noise). <b>edge</b> = '
            f'|lead IC| − |same-day corr|: positive means the driver carries <i>forward</i> '
            f'information beyond co-moving. Gate |IC|≥{LEAD_IC_MIN}. Small n (~70–110 independent '
            f'obs) — treat as a hypothesis shortlist, not certainty.</div>')
    return cards + tbl + note


def _ic_panel(ic: dict) -> str:
    if not ic:
        return '<p class="muted">IC not available</p>'
    pa = pd.Series({k: v["ic"] for k, v in ic["per_asset"].items()}).sort_values(ascending=False)
    cards = (
        f'<div class="pillars" style="grid-template-columns:repeat(4,1fr)">'
        f'<div class="pill"><div class="lab">{ic["pooled_ic"]:+.3f}</div>'
        f'<div class="v">pooled IC (all assets)</div></div>'
        f'<div class="pill"><div class="lab">{ic["mean_asset_ic"]:+.3f}</div>'
        f'<div class="v">mean per-asset IC</div></div>'
        f'<div class="pill"><div class="lab">{ic["pct_positive_ic"]:.0f}%</div>'
        f'<div class="v">assets with IC&gt;0</div></div>'
        f'<div class="pill"><div class="lab">{ic["pooled_hit"]:.0f}%</div>'
        f'<div class="v">pooled hit rate</div></div></div>')

    def _mini(s: pd.Series) -> str:
        out = ""
        for sym, v in s.items():
            cc = "long" if v > 0 else "short"
            out += f'<tr><td><b>{_esc(sym)}</b></td><td class="{cc}">{v:+.3f}</td></tr>'
        return f'<table><thead><tr><th style="text-align:left">asset</th><th>IC</th></tr></thead><tbody>{out}</tbody></table>'

    return (cards + '<div class="card-grid" style="grid-template-columns:1fr 1fr">'
            f'<div><h4>BEST FORWARD IC</h4>{_mini(pa.head(8))}</div>'
            f'<div><h4>WORST FORWARD IC</h4>{_mini(pa.tail(8))}</div></div>'
            f'<div class="muted" style="margin-top:6px">IC = corr(signal, next-{SIGNAL_FWD_DAYS}d return), '
            f'month-end sampled, n≥24 months per asset. Relationship weights are full-sample (in-sample); '
            f'move timing and forward return are out-of-sample.</div>')


def render_report(out_path: Path, rel, cm, sig, ic, ll=None) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_robust = int(rel["robust"].sum())
    n_total = len(rel)
    pooled = ic.get("pooled_ic", float("nan")) if ic else float("nan")
    n_lead = int((ll["predictive"] & (ll["edge"] > 0)).sum()) if ll is not None and not ll.empty else 0

    verdict = (
        f'<b>What this is.</b> {n_robust} of {n_total} asset×driver links pass the robustness gate '
        f'(significant, sign-stable, long history). These are <b>real, contemporaneous</b> economic '
        f'relationships — crude→Norway, broad-commodities→Brazil, copper→Chile, rates→bonds. '
        f'Use them for <b>hedging, beta-neutralisation, risk decomposition and replication</b>.<br>'
        f'<b>What it is NOT.</b> Turning a recent driver move into a same-direction forward signal does '
        f'<b>not</b> earn broadly: pooled forward IC = {pooled:+.3f} (≈0). Most drivers and assets move '
        f'<i>together same-day</i>, so the "signal" column below is a contemporaneous read, not momentum.<br>'
        f'<b>Where there IS forward edge.</b> The lead-lag scan finds {n_lead} pairs where a driver\'s '
        f'past move predicts the asset\'s <i>future</i> return beyond same-day co-movement and survives '
        f'both halves of history — clustered in <b>copper/commodities → rates &amp; credit</b> and '
        f'<b>USD → credit</b> (growth-bellwether commodities leading the rates complex by ~1 month). '
        f'Small samples — a hypothesis shortlist, not a guarantee.')

    extra_css = (
        ".heat th.rot{writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;"
        "padding:6px 2px;font-size:9px;cursor:default}"
        ".heat td{padding:4px 6px}.heat td:first-child{white-space:nowrap}"
        ".legpos{color:#5cb85c}.legneg{color:#ff6b6b}"
        ".pillars .pill .lab{font-size:22px;color:#d4af37}")

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SL Research · Cross-Asset Correlations & Driver Signals</title>
<style>{CSS}{extra_css}</style></head><body>
<h1>CROSS-ASSET CORRELATIONS &amp; DRIVER SIGNALS</h1>
<div class="timestamp">{ts} · macro idea engine · correlation/driver module</div>
<div class="macro">{verdict}</div>

<h2>LEAD-LAG — DOES THE DRIVER PREDICT THE ASSET?</h2>
{_leadlag_panel(ll)}

<h2>SIGNAL VALIDATION — FORWARD INFORMATION COEFFICIENT</h2>
{_ic_panel(ic)}

<h2>DRIVER × ASSET CORRELATION MAP</h2>
{_heatmap(cm, rel)}

<h2>ROBUST RELATIONSHIPS</h2>
{_robust_table(rel)}

<h2>CURRENT DRIVER-IMPLIED SIGNALS (CONTEMPORANEOUS)</h2>
{_signals_table(sig)}
</body></html>"""

    out_path.write_text(html_doc)


def main() -> Path:
    print("building relationship table ...")
    rel = relationship_table()
    print(f"  {len(rel)} pairs, {int(rel['robust'].sum())} robust")
    cm = rel.pivot(index="asset", columns="driver", values="corr")
    print("building signals ...")
    sig = build_signals(rel)
    print("validating forward IC ...")
    ic = validate_ic(rel)
    if ic:
        print(f"  pooled IC {ic['pooled_ic']:+.3f} · mean-asset IC {ic['mean_asset_ic']:+.3f} "
              f"· {ic['pct_positive_ic']:.0f}% assets IC>0")
    print("scanning lead-lag (driver predicts asset?) ...")
    ll = lead_lag_table(rel)
    if not ll.empty:
        print(f"  {int(ll['predictive'].sum())} survive both halves · "
              f"{int((ll['predictive'] & (ll['edge'] > 0)).sum())} genuine leads (fwd IC > same-day)")

    RUNS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"correlations_{stamp}.html"
    render_report(out, rel, cm, sig, ic, ll)
    print(f"wrote {out}")
    if not os.environ.get("NO_OPEN"):
        subprocess.run(["open", "-a", "Safari", str(out)], check=False)
    return out


if __name__ == "__main__":
    main()
