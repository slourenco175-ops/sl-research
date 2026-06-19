"""SL Research — Macro Idea Engine dashboard (single self-contained HTML).

House style (dark, gold accent, click-to-expand). Renders: the regime hero
(super-regime, pillar nowcasts, transition probabilities), the ranked trade
sheet with expandable falsification detail, the Stage 2 conditional-return
table for the current regime, the Stage 3 divergence table, and the recorded
Fed policy-path yield curve.
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from forex.policy_dashboard import _curve_svg
from forex.policy_path import latest as latest_policy
from macro_engine.config import MIN_OBS_PER_CELL

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=IBM+Plex+Mono:wght@400;500;700&display=swap');
body{font-family:'IBM Plex Mono',monospace;background:#0e0e10;color:#e6e6e6;padding:24px;max-width:1500px;margin:auto}
h1{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:3px;font-size:34px;margin:0 0 6px 0}
h2{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:2px;font-size:23px;margin:30px 0 12px 0;color:#d4af37}
h4{font-size:11px;color:#999;margin:0 0 8px 0;letter-spacing:1px}
.timestamp{color:#888;font-size:11px;margin-bottom:22px}
.hero{background:#15151a;border-left:3px solid #d4af37;padding:20px 22px;margin-bottom:18px}
.hero .reg{font-size:30px;font-weight:bold;letter-spacing:1px;color:#d4af37}
.hero .code{font-size:14px;color:#aaa;margin-left:10px}
.pillars{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.pill{background:#1a1a1d;padding:12px 14px;border-left:3px solid #444}
.pill.hot{border-left-color:#ff6b6b}.pill.cold{border-left-color:#5cb85c}
.pill .lab{font-size:13px;font-weight:bold}.pill .v{font-size:11px;color:#999;margin-top:4px}
.trans{margin-top:10px;font-size:12px}
.bar{display:inline-block;height:10px;background:#d4af37;border-radius:1px;vertical-align:middle}
.barbg{display:inline-block;width:180px;height:10px;background:#222;border-radius:1px;vertical-align:middle}
table{border-collapse:collapse;font-size:11px;width:100%;margin-bottom:10px}
th{background:#1a1a1d;padding:8px 8px;text-align:right;border-bottom:2px solid #d4af37;font-size:10px;letter-spacing:1px;cursor:pointer}
th:first-child,td:first-child{text-align:left}
th:hover{color:#d4af37}
td{padding:6px 8px;border-bottom:1px solid #2a2a2d;text-align:right}
tbody tr.summary-row{cursor:pointer}
tbody tr.summary-row:hover,tbody tr.summary-row.open{background:#1a1a1d}
tbody tr.summary-row .chev{display:inline-block;width:10px;color:#d4af37;transition:transform .15s}
tbody tr.summary-row.open .chev{transform:rotate(90deg)}
tr.detail-row{display:none}tr.detail-row.open{display:table-row}
tr.detail-row>td{padding:0;background:#0e0e10}
.card{background:#15151a;border-left:3px solid #444;padding:16px 18px}
.card-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px}
.card-grid p{font-size:12px;line-height:1.7;color:#ccc;margin:0}
.long{color:#5cb85c;font-weight:bold}.short{color:#ff6b6b;font-weight:bold}
.up{color:#ff6b6b}.down{color:#5cb85c}.muted{color:#666;font-size:11px;font-style:italic}
.badge{font-size:10px;padding:2px 7px;border-radius:2px;letter-spacing:1px}
.badge.surf{background:#14331a;color:#5cb85c}.badge.watch{background:#2a2a16;color:#d4af37}
.badge.conf{background:#3a1414;color:#ff6b6b}
.macro{background:#1a1a1d;padding:12px 18px;border-left:3px solid #d4af37;margin-bottom:18px;font-size:12px;line-height:1.8}
.cf-strong{color:#5cb85c;font-weight:bold}.cf-ok{color:#9acd32}.cf-weak{color:#d4af37}.cf-thin{color:#ff6b6b}
.ci{color:#888}.stale{color:#ff6b6b;font-weight:bold}
"""

POS = lambda v: "up" if v > 0 else ("down" if v < 0 else "muted")


def _pillar_chips(regime: dict) -> str:
    out = ""
    for k, lab in regime["labels"].items():
        bit = regime["state_bits"].get(k, 0)
        nc = regime["nowcast"].get(k, {})
        cls = "hot" if bit else "cold"
        out += (f'<div class="pill {cls}"><div class="lab">{html.escape(k.upper())}: {html.escape(lab)}</div>'
                f'<div class="v">nowcast {nc.get("value", "—")} · {nc.get("trend", "—")}</div></div>')
    return out


def _transition_bars(regime: dict) -> str:
    tr = regime["transition"]
    nxt = tr["next_3m"]
    rows = ""
    for state, p in sorted(nxt.items(), key=lambda kv: -kv[1]):
        w = int(180 * p)
        cur = " (current)" if state == regime["super_regime"] else ""
        rows += (f'<div class="trans">{html.escape(state)}{cur}: '
                 f'<span class="barbg"><span class="bar" style="width:{w}px"></span></span> {p:.0%}</div>')
    return (f'{rows}<div class="muted" style="margin-top:6px">P(regime change within 3m): '
            f'{tr["p_change_3m"]:.0%} · most likely next: {html.escape(tr["most_likely_next"])}</div>')


def _sheet_table(sheet: pd.DataFrame) -> str:
    if sheet.empty:
        return '<p class="muted">No candidates produced.</p>'
    body = ""
    for _, r in sheet.iterrows():
        dcls = "long" if r["direction"] == "LONG" else "short"
        if r["conflict"]:
            badge = '<span class="badge conf">CONFLICT</span>'
        elif r["surfaced"]:
            badge = '<span class="badge surf">SURFACED</span>'
        else:
            badge = '<span class="badge watch">WATCH</span>'

        body += f"""
        <tr class="summary-row">
          <td><span class="chev">▶</span> <b>{html.escape(r['id'])}</b> {html.escape(r['asset'])}</td>
          <td class="{dcls}">{r['direction']}</td>
          <td>{html.escape(str(r['instrument']))}</td>
          <td data-sort="{r['regime_score']}">{r['regime_score']:+.2f}
            <span class="cf-{html.escape(str(r.get('regime_confidence', 'n/a')))}" style="font-size:9px">
              {html.escape(str(r.get('regime_confidence', '')))}</span></td>
          <td data-sort="{r['divergence_z']}">{r['divergence_z']:+.2f}</td>
          <td data-sort="{r['crowding']}">{r['crowding']:+.2f}</td>
          <td data-sort="{r['convergence_score']}"><b>{r['convergence_score']:+.2f}</b></td>
          <td>{r['agreement_count']}</td>
          <td>{html.escape(str(r['suggested_size']))}</td>
          <td>{badge}</td>
        </tr>"""

        div_rows = ""
        detail = r.get("_div_detail")
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            for _, d in detail.iterrows():
                div_rows += (f'<tr><td>{html.escape(d["divergence"])}</td>'
                             f'<td class="{POS(d["divergence_z"])}">{d["divergence_z"]:+.2f}</td>'
                             f'<td>{d["persistence_days"]}d</td>'
                             f'<td>{html.escape(str(d["direction"]))}</td>'
                             f'<td style="text-align:left">{html.escape(str(d["gap_raw"]))}</td></tr>')
        div_table = (f'<table><thead><tr><th>Divergence</th><th>z (signed)</th><th>Persist</th>'
                     f'<th>Dir</th><th style="text-align:left">Gap</th></tr></thead>'
                     f'<tbody>{div_rows}</tbody></table>') if div_rows else '<p class="muted">no divergence rows</p>'

        body += f"""
        <tr class="detail-row"><td colspan="10"><div class="card">
          <p><b>Thesis.</b> {html.escape(r['thesis'])}</p>
          <div class="card-grid">
            <div><h4>CATALYST</h4><p>{html.escape(r['catalyst'])}</p></div>
            <div><h4>INVALIDATION (STAGE 7)</h4><p>Price: {html.escape(str(r['price_stop']))}<br>
              Macro: {html.escape(str(r['macro_invalidation']))}<br>
              Time: {r['horizon']} window</p></div>
            <div><h4>SIZE</h4><p>{html.escape(str(r['suggested_size']))}<br>
              entry {r['entry']}<br>reversal risk: {r['reversal_flag']}</p></div>
          </div>
          <h4 style="margin-top:14px">DIVERGENCE BREAKDOWN (STAGE 3)</h4>{div_table}
        </div></td></tr>"""
    return f"""<table id="sheet"><thead><tr>
      <th>Idea / Asset</th><th>Dir</th><th>Instrument</th><th>Regime</th><th>Div z</th>
      <th>Crowd</th><th>Conv</th><th>Agree</th><th>Size</th><th>Status</th>
    </tr></thead><tbody>{body}</tbody></table>"""


def _ci_cell(r: pd.Series) -> str:
    lo, hi = r.get("ci_low"), r.get("ci_high")
    if pd.isna(lo) or pd.isna(hi):
        return '<span class="ci">—</span>'
    return f'<span class="ci">[{lo:+.2f}, {hi:+.2f}]</span>'


def _cond_table(ranked: pd.DataFrame) -> str:
    if ranked.empty:
        return '<p class="muted">no conditional stats</p>'
    rows = ""
    for _, r in ranked.iterrows():
        raw = r.get("raw_sharpe")
        raw_txt = "—" if pd.isna(raw) else f"{raw:+.2f}"
        conf = str(r["confidence"])
        rows += (f'<tr><td>{html.escape(r["asset"])} <span class="muted">{html.escape(r["name"])}</span></td>'
                 f'<td>{r["sharpe"]:+.2f}</td><td class="muted">{raw_txt}</td>'
                 f'<td class="muted">{r.get("parent_sharpe", float("nan")):+.2f}</td>'
                 f'<td>{_ci_cell(r)}</td>'
                 f'<td>{r["mean_ann"]:+.1f}%</td><td>{r["vol_ann"]:.1f}%</td>'
                 f'<td>{r["hit_rate"]:.0f}%</td><td class="down">{r["max_dd"]:.0f}%</td>'
                 f'<td>{r["n_obs"]:,}</td>'
                 f'<td class="cf-{html.escape(conf)}">{html.escape(conf)}</td></tr>')
    return (f'<table><thead><tr><th>Asset</th><th>Sharpe (pooled)</th><th>Raw</th><th>Parent</th>'
            f'<th>90% CI</th><th>Mean(ann)</th><th>Vol(ann)</th>'
            f'<th>Hit</th><th>MaxDD</th><th>n</th><th>Conf</th></tr></thead><tbody>{rows}</tbody></table>')


def _div_table(divergences: pd.DataFrame) -> str:
    if divergences.empty:
        return '<p class="muted">no divergences</p>'
    rows = ""
    for _, d in divergences.iterrows():
        stale = '<span class="stale">⚠ stale</span>' if d.get("wide_stale") else ""
        rows += (f'<tr><td>{html.escape(d["divergence"])}</td><td>{html.escape(d["asset"])}</td>'
                 f'<td class="{POS(d["divergence_z"])}">{d["divergence_z"]:+.2f}</td>'
                 f'<td>{d["persistence_days"]}d {stale}</td><td>{html.escape(str(d["direction"]))}</td>'
                 f'<td style="text-align:left">{html.escape(str(d["gap_raw"]))}</td></tr>')
    return (f'<table><thead><tr><th>Divergence</th><th>Asset</th><th>z (signed)</th><th>Persist</th>'
            f'<th>Dir</th><th style="text-align:left">Gap</th></tr></thead><tbody>{rows}</tbody></table>')


def _positioning_table(positioning: pd.DataFrame) -> str:
    if positioning is None or positioning.empty:
        return '<p class="muted">no COT positioning available</p>'
    rows = ""
    for _, p in positioning.iterrows():
        cl = float(p.get("crowd_long", 0.0))
        crowd = "crowded long" if cl > 0.3 else ("crowded short" if cl < -0.3 else "neutral")
        rev = '<span class="stale">⚠ reversing</span>' if p.get("reversal_flag") else ""
        rows += (f'<tr><td>{html.escape(str(p["asset"]))}</td>'
                 f'<td>{p.get("spec_pctile", float("nan")):.0f}%</td>'
                 f'<td class="{POS(cl)}">{cl:+.2f}</td>'
                 f'<td>{html.escape(crowd)}</td>'
                 f'<td>{html.escape(str(p.get("trajectory", "—")))} {rev}</td>'
                 f'<td>{html.escape(str(p.get("report_date", "—")))}</td></tr>')
    return (f'<table><thead><tr><th>Asset</th><th>Spec %ile (3y)</th><th>Crowd</th>'
            f'<th>Read</th><th>Trajectory</th><th>COT date</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _transition_heatmap(regime: dict) -> str:
    M = regime["transition"].get("matrix")
    if M is None or getattr(M, "empty", True):
        return ""
    states = list(M.index)
    head = "".join(f"<th>{html.escape(s[:4])}</th>" for s in states)
    body = ""
    for frm in states:
        cur = ' style="color:#d4af37"' if frm == regime["super_regime"] else ""
        cells = ""
        for to in states:
            p = float(M.loc[frm, to])
            # gold intensity ~ probability
            bg = f"rgba(212,175,55,{min(0.85, p):.2f})"
            txt = "#0e0e10" if p > 0.4 else "#e6e6e6"
            cells += f'<td style="background:{bg};color:{txt}">{p:.0%}</td>'
        body += f'<tr><td{cur}>{html.escape(frm)}</td>{cells}</tr>'
    return (f'<table style="max-width:560px"><thead><tr><th>from \\ to</th>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table>'
            f'<p class="muted">1-step monthly transition probabilities (empirical, 25y).</p>')


def _corr_heatmap(corr: pd.DataFrame) -> str:
    if corr is None or corr.empty:
        return '<p class="muted">need ≥2 surfaced ideas for a correlation matrix</p>'
    cols = list(corr.columns)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = ""
    for a in cols:
        cells = ""
        for b in cols:
            v = float(corr.loc[a, b])
            # green negative (diversifying) → red positive (stacking)
            if v >= 0:
                bg = f"rgba(255,107,107,{min(0.8, abs(v)):.2f})"
            else:
                bg = f"rgba(92,184,92,{min(0.8, abs(v)):.2f})"
            body_txt = "#0e0e10" if abs(v) > 0.5 else "#e6e6e6"
            cells += f'<td style="background:{bg};color:{body_txt}">{v:+.2f}</td>'
        body += f'<tr><td>{html.escape(a)}</td>{cells}</tr>'
    return (f'<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>')


def _portfolio_panel(prisk: dict) -> str:
    if not prisk:
        return '<p class="muted">no surfaced book to risk-check</p>'
    buckets = " · ".join(f'{html.escape(k)} {v:+.1f}%' for k, v in prisk["net_by_bucket"].items())
    dr = prisk.get("diversification_ratio")
    dr_txt = f"{dr:.2f}×" if dr is not None else "—"
    macro = (f'<div class="macro"><b>PORTFOLIO RISK (SURFACED BOOK)</b> &nbsp; '
             f'correlated vol <b>{prisk["book_vol_pct"]:.1f}%</b> '
             f'(cap {prisk["vol_cap_pct"]:.0f}%) vs naïve standalone sum '
             f'{prisk["sum_standalone_pct"]:.1f}% · diversification {dr_txt} · '
             f'gross {prisk["gross_pct"]:.0f}% / net {prisk["net_pct"]:+.0f}% · '
             f'net by bucket: {buckets}</div>')
    return macro + _corr_heatmap(prisk.get("corr"))


def render(out_path: str, regime: dict, ranked: pd.DataFrame,
           divergences: pd.DataFrame, sheet: pd.DataFrame,
           positioning: pd.DataFrame | None = None,
           prisk: dict | None = None) -> None:
    snap = latest_policy()
    n_surf = int(sheet["surfaced"].sum()) if not sheet.empty else 0

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SL Research — Macro Idea Engine</title>
<style>{CSS}</style></head><body>

<h1>SL RESEARCH — MACRO IDEA ENGINE</h1>
<div class="timestamp">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} ·
  regime asof {html.escape(regime['asof'])} · {n_surf} surfaced idea(s) ·
  convergence engine · research only, does not trade</div>

<div class="hero">
  <span class="reg">{html.escape(regime['super_regime'])}</span>
  <span class="code">state {html.escape(regime['state_code'])}</span>
  <div class="pillars">{_pillar_chips(regime)}</div>
  <h4>REGIME TRANSITION (3M, EMPIRICAL)</h4>
  {_transition_bars(regime)}
</div>

<div class="macro"><b>FED POLICY PATH</b> &nbsp; {html.escape(snap['meeting'])} ·
  {html.escape(snap['chair'])} · target {snap['target_range'][0]:.2f}–{snap['target_range'][1]:.2f}% ·
  dots {snap['dots_median_cy']:.1f}% · {html.escape(snap['statement_bias'])} bias ·
  {html.escape(snap['curve_shape'].replace('_', ' '))}</div>

<h2>RANKED TRADE SHEET</h2>
{_sheet_table(sheet)}

<h2>PORTFOLIO RISK (CORRELATION-AWARE)</h2>
{_portfolio_panel(prisk or {})}

<h2>POSITIONING — CFTC NET SPEC (STAGE 4)</h2>
{_positioning_table(positioning)}

<h2>REGIME TRANSITION MATRIX</h2>
{_transition_heatmap(regime)}

<h2>CONDITIONAL RETURNS — STATE {html.escape(regime['state_code'])} (STAGE 2)</h2>
<p class="muted">16-state cell shrunk toward {html.escape(regime['super_regime'])} parent (James-Stein pooling);
  90% CI from block bootstrap. Conf: strong = CI clears zero &amp; n≥{MIN_OBS_PER_CELL}, weak = CI straddles zero, thin = sparse cell.</p>
{_cond_table(ranked)}

<h2>DIVERGENCES (STAGE 3)</h2>
{_div_table(divergences)}

<h2>FED YIELD CURVE — PRIOR vs POST ({html.escape(snap['meeting'])})</h2>
{_curve_svg(snap['curve_levels'])}

<script>
document.querySelectorAll('#sheet tr.summary-row').forEach(row => {{
  row.addEventListener('click', () => {{
    const d = row.nextElementSibling;
    if (!d || !d.classList.contains('detail-row')) return;
    row.classList.toggle('open'); d.classList.toggle('open');
  }});
}});
document.querySelectorAll('table th').forEach((th, i) => {{
  let asc = false;
  th.addEventListener('click', () => {{
    const tb = th.closest('table').querySelector('tbody');
    const all = Array.from(tb.querySelectorAll('tr'));
    const pairs = [];
    for (let k = 0; k < all.length; k++)
      if (all[k].classList.contains('summary-row')) pairs.push([all[k], all[k+1]]);
    if (!pairs.length) return;
    pairs.sort((a,b) => {{
      const ax=a[0].children[i]?.dataset.sort ?? a[0].children[i]?.innerText ?? '';
      const bx=b[0].children[i]?.dataset.sort ?? b[0].children[i]?.innerText ?? '';
      const an=parseFloat(ax), bn=parseFloat(bx);
      if(!isNaN(an)&&!isNaN(bn)) return asc?an-bn:bn-an;
      return asc?String(ax).localeCompare(bx):String(bx).localeCompare(ax);
    }});
    asc=!asc;
    pairs.forEach(p => {{ tb.appendChild(p[0]); if(p[1]) tb.appendChild(p[1]); }});
  }});
}});
</script>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)
