"""SL Research — FOMC policy-path dashboard (single self-contained HTML file).

Renders the recorded policy-path snapshots (forex/policy_path.py) into an
interactive page in the house style: macro backdrop bar, a decision hero,
an inline yield-curve chart (prior vs post), and click-to-expand sections for
the dot plot, regime pillars (Stage 1 hand-off), the divergence read (Stage 3),
and the falsification seed (Stage 7).

    cd "/Users/samuellourenco/SL Research"
    source .venv/bin/activate
    python -m forex.policy_dashboard
"""
from __future__ import annotations

import html
import os
import subprocess
import sys
import webbrowser
from datetime import datetime

from forex.policy_path import SNAPSHOTS, latest

OUTPUT_DIR = "forex/runs"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=IBM+Plex+Mono:wght@400;500;700&display=swap');
body{font-family:'IBM Plex Mono',monospace;background:#0e0e10;color:#e6e6e6;padding:24px;max-width:1100px;margin:auto}
h1{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:3px;font-size:32px;margin:0 0 8px 0}
h2{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:2px;font-size:22px;margin:28px 0 12px 0;color:#d4af37}
h4{font-size:11px;color:#999;margin:0 0 8px 0;letter-spacing:1px}
.timestamp{color:#888;font-size:11px;margin-bottom:24px}
.macro{background:#1a1a1d;padding:12px 18px;border-left:3px solid #d4af37;margin-bottom:24px;font-size:12px;line-height:1.8}
.hero{background:#15151a;border-left:3px solid #d4af37;padding:20px 22px;margin-bottom:20px}
.hero .row{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.hero .decision{font-size:26px;font-weight:bold;letter-spacing:1px}
.hero .chair{font-size:15px;color:#aaa}
.hero .tag{font-size:11px;padding:3px 9px;border-radius:2px;letter-spacing:1px;font-weight:700}
.tag.hawk{background:#3a1414;color:#ff6b6b}
.tag.neutral{background:#2a2a16;color:#d4af37}
.hero p{font-size:13px;line-height:1.7;color:#ccc;margin:8px 0 0 0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0 22px 0}
.kpi{background:#1a1a1d;padding:14px 16px;border-left:3px solid #444}
.kpi .v{font-size:22px;font-weight:bold;color:#d4af37}
.kpi .l{font-size:10px;color:#999;letter-spacing:1px;margin-top:4px}
.kpi .s{font-size:11px;color:#888;margin-top:2px}
table{border-collapse:collapse;font-size:12px;width:100%;margin-bottom:8px}
th{background:#1a1a1d;padding:8px 10px;text-align:right;border-bottom:2px solid #d4af37;font-size:10px;letter-spacing:1px}
th:first-child{text-align:left}
td{padding:6px 10px;border-bottom:1px solid #2a2a2d;text-align:right}
td:first-child{text-align:left;font-weight:bold}
.up{color:#ff6b6b}.down{color:#5cb85c}.flat{color:#888}
.section{background:#15151a;border-left:3px solid #444;margin-bottom:12px}
.section-head{cursor:pointer;user-select:none;padding:14px 18px;display:flex;align-items:center;gap:10px;font-size:13px;letter-spacing:1px}
.section-head:hover{background:#1a1a1d}
.section-head .chev{display:inline-block;width:10px;color:#d4af37;transition:transform .15s}
.section.open .section-head .chev{transform:rotate(90deg)}
.section-body{display:none;padding:0 18px 18px 18px;font-size:13px;line-height:1.8;color:#ccc}
.section.open .section-body{display:block}
.section-body ul{margin:6px 0;padding-left:18px;list-style-type:none}
.section-body li:before{content:"• ";color:#d4af37}
.pill{display:inline-block;padding:4px 10px;margin:3px 4px 3px 0;border-radius:2px;font-size:11px;letter-spacing:1px;background:#1a1a1d;border-left:2px solid #d4af37}
.note{color:#666;font-size:11px;font-style:italic}
.curve-wrap{padding:16px 0 6px 0}
.legend{font-size:11px;color:#888;margin-top:6px}
"""


def _curve_svg(levels: dict) -> str:
    """Two-line yield curve (prior grey, post gold) with hover tooltips."""
    tenors = list(levels.keys())
    prior = [levels[t][0] for t in tenors]
    post = [levels[t][1] for t in tenors]
    W, H = 1000, 320
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 40
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    lo = min(min(prior), min(post)) - 0.1
    hi = max(max(prior), max(post)) + 0.1

    def x(i):
        return pad_l + (plot_w * i / (len(tenors) - 1))

    def y(v):
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    # horizontal gridlines + y labels
    grid = ""
    step = 0.2
    g = round(lo + (step - (lo % step)), 2)
    while g < hi:
        gy = y(g)
        grid += (f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{W - pad_r}" y2="{gy:.1f}" '
                 f'stroke="#2a2a2d" stroke-width="1"/>'
                 f'<text x="{pad_l - 8}" y="{gy + 3:.1f}" fill="#777" font-size="10" '
                 f'text-anchor="end">{g:.1f}</text>')
        g = round(g + step, 2)

    def polyline(vals, color, width):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>'

    def dots(vals, color, label):
        out = ""
        for i, v in enumerate(vals):
            out += (f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" fill="{color}">'
                    f'<title>{tenors[i]} {label}: {v:.2f}%</title></circle>')
        return out

    xlabels = "".join(
        f'<text x="{x(i):.1f}" y="{H - pad_b + 18:.1f}" fill="#999" font-size="10" '
        f'text-anchor="middle">{t}</text>' for i, t in enumerate(tenors))

    return f"""<svg viewBox="0 0 {W} {H}" width="100%" style="background:#0e0e10">
      {grid}
      {polyline(prior, "#888", 1.5)}
      {polyline(post, "#d4af37", 2.5)}
      {dots(prior, "#888", "prior")}
      {dots(post, "#d4af37", "post")}
      {xlabels}
    </svg>"""


def _curve_table(levels: dict) -> str:
    rows = ""
    for t, (p0, p1) in levels.items():
        bp = round((p1 - p0) * 100)
        cls = "up" if bp > 0 else ("down" if bp < 0 else "flat")
        rows += (f'<tr><td>{html.escape(t)}</td><td>{p0:.2f}</td><td>{p1:.2f}</td>'
                 f'<td class="{cls}">{bp:+d}</td></tr>')
    return (f'<table><thead><tr><th>Tenor</th><th>Prior</th><th>Post</th>'
            f'<th>Δ bp</th></tr></thead><tbody>{rows}</tbody></table>')


def _section(title: str, body: str, open_: bool = False) -> str:
    cls = "section open" if open_ else "section"
    return (f'<div class="{cls}"><div class="section-head"><span class="chev">▶</span>'
            f'{title}</div><div class="section-body">{body}</div></div>')


def _render_snapshot(s: dict) -> str:
    lo, hi = s["target_range"]
    bias = s["statement_bias"]
    bias_tag = "hawk" if bias == "tightening" else ("neutral" if bias == "neutral" else "neutral")
    div = s["divergence"]

    kpis = f"""
    <div class="kpis">
      <div class="kpi"><div class="v">{lo:.2f}–{hi:.2f}%</div><div class="l">TARGET RANGE</div><div class="s">{html.escape(s['decision'])}</div></div>
      <div class="kpi"><div class="v">{s['dots_median_cy']:.1f}%</div><div class="l">DOTS MEDIAN (CY)</div><div class="s">{html.escape(s['hike_votes'])} see a hike</div></div>
      <div class="kpi"><div class="v">{s['inflation_yoy']:.1f}%</div><div class="l">INFLATION YoY</div><div class="s">above 2% target</div></div>
      <div class="kpi"><div class="v up">{s['curve_bp'].get('2y', 0):+d}bp</div><div class="l">2Y YIELD MOVE</div><div class="s">{html.escape(s['curve_shape'].replace('_', ' '))}</div></div>
    </div>"""

    curve = f"""
    <div class="curve-wrap">{_curve_svg(s['curve_levels'])}
      <div class="legend"><span style="color:#888">━━ prior ({list(SNAPSHOTS)[0] if False else 'pre-meeting'})</span>
        &nbsp;&nbsp;<span style="color:#d4af37">━━ post-decision</span>
        &nbsp;&nbsp;· 2s10s {s['spread_2s10s_bp'][0]}→{s['spread_2s10s_bp'][1]}bp (flattened {s['spread_2s10s_bp'][0] - s['spread_2s10s_bp'][1]}bp)</div>
    </div>
    {_curve_table(s['curve_levels'])}"""

    dots_body = (
        f"<ul><li>SEP median current-year funds rate: <b>{s['dots_median_cy']:.1f}%</b> "
        f"(swung up from 3.4% in March — from pricing a cut to pricing a hike)</li>"
        f"<li>Participants projecting a hike this year: <b>{html.escape(s['hike_votes'])}</b></li>"
        f"<li>Chair {html.escape(s['chair'])} submitted no projection</li></ul>")

    pillars = "".join(
        f'<span class="pill">{html.escape(k.upper())}: {html.escape(v)}</span>'
        for k, v in s["pillars"].items())
    pillars_body = (
        f"<div>{pillars}</div>"
        f'<p class="note">Stage 1 hand-off — these flags condition the regime read.</p>')

    div_body = (
        f"<ul>"
        f"<li><b>{html.escape(div['name'])}</b> · {html.escape(div['asset'])}</li>"
        f"<li>Gap: {html.escape(div['gap_raw'])}</li>"
        f"<li>Direction: <b>{html.escape(div['direction'])}</b> · persistence {div['persistence_days']}d "
        f"· gap_z {div['gap_z'] if div['gap_z'] is not None else '— (filled by Stage 3 vs SOFR/FF strip)'}</li>"
        f"<li>Supporting pillars: {html.escape(', '.join(div['supporting_pillars']))}</li>"
        f"</ul><p>{html.escape(div['note'])}</p>")

    falsify_body = (
        f"<ul><li>Invalidation: {html.escape(s['invalidation'])}</li>"
        f'<li class="note">{html.escape(s["fx_note"])}</li></ul>')

    return f"""
    <div class="hero">
      <div class="row">
        <span class="decision">FOMC · {html.escape(s['meeting'])}</span>
        <span class="chair">Chair {html.escape(s['chair'])}</span>
        <span class="tag {bias_tag}">{html.escape(bias.upper())} BIAS</span>
        <span class="tag hawk">HAWKISH HOLD</span>
      </div>
      <p>Statement flipped from an easing bias to <b>{html.escape(bias)}</b> — dropped the
      "additional rate adjustments" language. Chair {html.escape(s['chair'])} leaned on
      "data-dependent / preserve flexibility" while anchoring an aggressive 2% inflation-credibility
      message and launching institutional task forces.</p>
    </div>
    {kpis}
    <h2>YIELD CURVE — PRIOR vs POST</h2>
    {curve}
    <h2>DETAIL</h2>
    {_section("DOT PLOT / SEP", dots_body, open_=True)}
    {_section("REGIME PILLARS — STAGE 1 HAND-OFF", pillars_body)}
    {_section("DIVERGENCE READ — STAGE 3 (POLICY PATH)", div_body)}
    {_section("FALSIFICATION SEED — STAGE 7", falsify_body)}
    """


def render(out_path: str) -> None:
    s = latest()
    body = _render_snapshot(s)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SL Research — Policy Path</title>
<style>{CSS}</style></head><body>

<h1>SL RESEARCH — FED POLICY PATH</h1>
<div class="timestamp">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} ·
  latest meeting {html.escape(s['meeting'])} · {len(SNAPSHOTS)} snapshot(s) recorded ·
  research only, does not trade</div>

<div class="macro"><b>MACRO BACKDROP</b> &nbsp;
  Target {s['target_range'][0]:.2f}–{s['target_range'][1]:.2f}% &nbsp;|&nbsp;
  Dots median {s['dots_median_cy']:.1f}% &nbsp;|&nbsp;
  Inflation {s['inflation_yoy']:.1f}% YoY &nbsp;|&nbsp;
  {html.escape(s['fx_note'])}</div>

{body}

<script>
document.querySelectorAll('.section-head').forEach(h => {{
  h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
}});
</script>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(OUTPUT_DIR, f"policy_path_{stamp}.html")
    render(html_path)
    abs_path = os.path.abspath(html_path)
    print(f"  HTML -> {html_path}")
    if os.getenv("NO_OPEN") == "1":
        return
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Safari", abs_path], check=False)
    else:
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
