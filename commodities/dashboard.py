"""Render the commodities signal table to a single self-contained HTML file.

No JS frameworks, no CDN. Sortable table via a tiny vanilla JS helper.
Sparklines are inline SVG so the file works fully offline.
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd


def _sparkline_svg(values: list[float], width: int = 110, height: int = 28) -> str:
    if not values or len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = i * (width / (len(values) - 1))
        y = height - ((v - lo) / rng) * height
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#16a34a" if values[-1] >= values[0] else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" style="display:block">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.4" '
        f'points="{" ".join(pts)}"/></svg>'
    )


def _bar(value_pct: float, color: str) -> str:
    """Horizontal mini-bar, value_pct in [0,100]."""
    v = max(0.0, min(100.0, float(value_pct)))
    return (
        f'<div class="bar-wrap"><div class="bar-fill" '
        f'style="width:{v:.1f}%;background:{color}"></div>'
        f'<span class="bar-lbl">{v:.0f}</span></div>'
    )


def _signal_chip(sig: str) -> str:
    color = {"LONG": "#16a34a", "SHORT": "#dc2626", "FLAT": "#6b7280"}.get(sig, "#6b7280")
    return f'<span class="chip" style="background:{color}">{sig}</span>'


def _regime_chip(reg: str) -> str:
    color = {"BULL": "#15803d", "BEAR": "#b91c1c", "NEUTRAL": "#475569"}.get(reg, "#475569")
    return f'<span class="chip" style="background:{color}">{reg}</span>'


def _fmt_pct(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{x * 100:+.{digits}f}%"


def _fmt_num(x, digits=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def render(df: pd.DataFrame, out_path: str) -> None:
    rows_html = []
    for _, r in df.iterrows():
        mm_pct = r.get("mm_3y_percentile")
        mm_color = (
            "#dc2626" if pd.notna(mm_pct) and mm_pct >= 80
            else "#16a34a" if pd.notna(mm_pct) and mm_pct <= 20
            else "#3b82f6"
        )
        mm_cell = (
            _bar(mm_pct, mm_color)
            + f'<div class="sub">net {_fmt_num(r.get("mm_net_pct_oi"), 1)}% OI · '
            + f'Δ1w {_fmt_num(r.get("mm_1w_change"), 2)}</div>'
            if pd.notna(mm_pct) else '<span class="muted">no COT</span>'
        )

        rows_html.append(f"""
        <tr>
          <td><b>{html.escape(r['name'])}</b><div class="sub">{html.escape(r['yf'])}</div></td>
          <td>{html.escape(r['sector'])}</td>
          <td class="num">{_fmt_num(r['price'], 2)}</td>
          <td>{_sparkline_svg(r['spark'])}</td>
          <td class="num" data-sort="{r['mom_12_1']:.6f}">{_fmt_pct(r['mom_12_1'])}</td>
          <td class="num" data-sort="{r['mom_3m']:.6f}">{_fmt_pct(r['mom_3m'])}</td>
          <td class="num" data-sort="{r['mom_1m']:.6f}">{_fmt_pct(r['mom_1m'])}</td>
          <td class="num" data-sort="{r['trend_vs_200d']:.6f}">{_fmt_pct(r['trend_vs_200d'])}</td>
          <td class="num" data-sort="{r['composite_z']:.6f}"><b>{r['composite_z']:+.2f}</b></td>
          <td>{_regime_chip(r['regime'])}</td>
          <td>{_signal_chip(r['signal'])}</td>
          <td>{mm_cell}</td>
          <td class="num">{_fmt_pct(r['vol_20d_ann'], 1)}</td>
        </tr>
        """)

    longs = df[df['signal'] == 'LONG']['name'].tolist()
    shorts = df[df['signal'] == 'SHORT']['name'].tolist()
    bulls = df[df['regime'] == 'BULL']['name'].tolist()
    bears = df[df['regime'] == 'BEAR']['name'].tolist()

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Commodities Signal Dashboard</title>
<style>
  :root {{
    --bg:#0b1020; --card:#111a33; --text:#e5e7eb; --muted:#94a3b8;
    --border:#1e293b; --accent:#60a5fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,Inter,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); padding:24px; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .meta {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .summary {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; }}
  .card h3 {{ margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); }}
  .card .list {{ font-size:13px; line-height:1.6; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; font-size:13px; }}
  th, td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; vertical-align:middle; }}
  th {{ background:#152045; color:#cbd5e1; font-weight:600; font-size:11px; text-transform:uppercase;
         letter-spacing:0.05em; cursor:pointer; user-select:none; white-space:nowrap; }}
  th:hover {{ color:var(--accent); }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:hover {{ background:#1a2a55; }}
  .sub {{ color:var(--muted); font-size:11px; margin-top:2px; }}
  .muted {{ color:var(--muted); }}
  .chip {{ display:inline-block; padding:3px 8px; border-radius:6px; color:#fff;
           font-size:11px; font-weight:700; letter-spacing:0.04em; }}
  .bar-wrap {{ position:relative; background:#1e293b; border-radius:4px; height:14px; width:100px; }}
  .bar-fill {{ height:100%; border-radius:4px; }}
  .bar-lbl {{ position:absolute; right:4px; top:-1px; font-size:10px; color:#fff; line-height:14px; }}
  .legend {{ font-size:12px; color:var(--muted); margin-top:18px; }}
  .legend code {{ background:#1e293b; padding:2px 6px; border-radius:4px; color:var(--text); }}
</style></head><body>

<h1>Commodities Signal Dashboard</h1>
<div class="meta">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} · {len(df)} contracts ·
  cross-sectional momentum + trend + COT positioning · v1</div>

<div class="summary">
  <div class="card"><h3>Long signals ({len(longs)})</h3><div class="list">{", ".join(longs) or "—"}</div></div>
  <div class="card"><h3>Short signals ({len(shorts)})</h3><div class="list">{", ".join(shorts) or "—"}</div></div>
  <div class="card"><h3>Bull regime ({len(bulls)})</h3><div class="list">{", ".join(bulls) or "—"}</div></div>
  <div class="card"><h3>Bear regime ({len(bears)})</h3><div class="list">{", ".join(bears) or "—"}</div></div>
</div>

<table id="grid">
  <thead><tr>
    <th>Contract</th><th>Sector</th><th>Price</th><th>120d</th>
    <th>12-1m</th><th>3m</th><th>1m</th><th>vs 200d</th>
    <th>Composite z</th><th>Regime</th><th>Signal</th>
    <th>Mgd Money (3y %ile)</th><th>Vol 20d</th>
  </tr></thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>

<div class="legend">
  <p><b>How to read:</b>
  <code>Composite z</code> blends 12-1m momentum (+), 1m return (−, reversal), and trend vs 200d (+).
  Top tercile → <span class="chip" style="background:#16a34a">LONG</span>,
  bottom → <span class="chip" style="background:#dc2626">SHORT</span>.
  <code>Regime</code> requires trend AND composite to agree.
  <code>Mgd Money</code> shows CFTC speculative positioning percentile over 3y —
  ≥80 (red) = crowded long, ≤20 (green) = crowded short. Treat extremes as a fade overlay,
  not a primary signal.</p>
  <p><b>Caveats:</b> v1 has no carry/term-structure factor (yfinance doesn't expose curves cleanly).
  COT lags ~3 days. Yfinance =F symbols are continuous front-month and not roll-adjusted, so
  reported returns include some roll noise. Backtest before sizing.</p>
</div>

<script>
// minimal sortable table: click a th to sort by that column
document.querySelectorAll('#grid th').forEach((th, i) => {{
  let asc = true;
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const ax = a.children[i].dataset.sort ?? a.children[i].innerText;
      const bx = b.children[i].dataset.sort ?? b.children[i].innerText;
      const an = parseFloat(ax), bn = parseFloat(bx);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? ax.localeCompare(bx) : bx.localeCompare(ax);
    }});
    asc = !asc;
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html_doc)
