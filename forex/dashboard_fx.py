"""SL Research forex dashboard — single self-contained HTML file.

Mirrors `commodities/dashboard_v2.py`: macro backdrop bar, summary table
ranked by conviction score with click-to-expand detail cards, then a
country-health table at the bottom.
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from forex.scoring_fx import VERDICT_COLOR

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=IBM+Plex+Mono:wght@400;500;700&display=swap');
body{font-family:'IBM Plex Mono',monospace;background:#0e0e10;color:#e6e6e6;
     padding:24px;max-width:1900px;margin:auto}
h1{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:3px;font-size:32px;margin:0 0 8px 0}
h2{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:2px;font-size:22px;margin:32px 0 12px 0;color:#d4af37}
h4{font-size:11px;color:#999;margin:0 0 8px 0;letter-spacing:1px}
.timestamp{color:#888;font-size:11px;margin-bottom:24px}
.macro{background:#1a1a1d;padding:12px 18px;border-left:3px solid #d4af37;margin-bottom:24px;font-size:12px}
table{border-collapse:collapse;font-size:11px;width:100%;margin-bottom:16px}
th{background:#1a1a1d;padding:8px 6px;text-align:left;border-bottom:2px solid #d4af37;font-size:10px;letter-spacing:1px;cursor:pointer;user-select:none}
th:hover{color:#d4af37}
td{padding:6px;border-bottom:1px solid #2a2a2d}
tbody tr.summary-row{cursor:pointer}
tbody tr.summary-row:hover{background:#1a1a1d}
tbody tr.summary-row.open{background:#1a1a1d}
tbody tr.summary-row .chev{display:inline-block;width:10px;color:#d4af37;transition:transform .15s}
tbody tr.summary-row.open .chev{transform:rotate(90deg)}
tr.detail-row{display:none}
tr.detail-row.open{display:table-row}
tr.detail-row > td{padding:0;background:#0e0e10;border-bottom:1px solid #2a2a2d}
.card{background:#15151a;border-left:3px solid #444;padding:18px;margin:0}
.card-head{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid #333;padding-bottom:10px;margin-bottom:14px;flex-wrap:wrap}
.sym{font-size:24px;font-weight:bold;letter-spacing:2px}
.name{font-size:16px;color:#aaa}
.sec{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px}
.last{font-size:18px;color:#d4af37;margin-left:auto}
.verdict{font-size:14px;font-weight:bold;letter-spacing:1px}
.card-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:14px}
.card-grid p{font-size:12px;line-height:1.6;color:#ccc;margin:0}
.reasons{border-top:1px solid #333;padding-top:10px}
.reasons ul{font-size:11px;color:#999;margin:0;padding-left:18px;line-height:1.7;list-style-type:none}
.reasons li:before{content:"• ";color:#d4af37}
.note{color:#666;font-size:10px;font-style:italic}
.up{color:#5cb85c}
.down{color:#d9534f}
.flat{color:#888}
.chart-row{margin:0 0 16px 0;padding:0 4px}
.chart-row h4{margin-bottom:6px}
.chart-legend{font-size:10px;color:#777;margin-top:4px;letter-spacing:0.5px}
"""


def _fmt_pct(x, digits=1, signed=True):
    if x is None or pd.isna(x):
        return "—"
    fmt = f"%+.{digits}f%%" if signed else f"%.{digits}f%%"
    return fmt % x


def _fmt_pct_ratio(x, digits=2, signed=True):
    if x is None or pd.isna(x):
        return "—"
    return _fmt_pct(x * 100, digits, signed)


def _fmt(x, digits=4):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{digits}f}"


def _trend_chips(st: str, mt: str, lt: str) -> str:
    return f"{st[0]}/{mt[0]}/{lt[0]}"


def _macro_block(macro: dict) -> str:
    if not macro:
        return ""
    parts = []
    if "DXY" in macro:
        m = macro["DXY"]
        parts.append(f'DXY {m["last"]:.2f} ({m["direction"]}, {m["chg_1m_pct"]:+.1f}% 1m)')
    if "US10Y" in macro:
        m = macro["US10Y"]
        parts.append(f'US10Y {m["last"]:.2f}% ({m["chg_1m_pct"]:+.2f}% 1m)')
    if "OIL" in macro:
        m = macro["OIL"]
        parts.append(f'WTI {m["last"]:.1f} ({m["chg_1m_pct"]:+.1f}% 1m)')
    if "GOLD" in macro:
        m = macro["GOLD"]
        parts.append(f'Gold {m["last"]:.0f} ({m["chg_1m_pct"]:+.1f}% 1m)')
    if "VIX" in macro:
        m = macro["VIX"]
        parts.append(f'VIX {m["last"]:.1f} ({m["pctile_3y"]:.0f}%ile)')
    return f'<div class="macro"><b>MACRO BACKDROP</b> &nbsp; {" &nbsp;|&nbsp; ".join(parts)}</div>'


def _val_color(label: str) -> str:
    if "RICH" in label:
        return "#d9534f"
    if "CHEAP" in label:
        return "#5cb85c"
    return "#888"


def _summary_row(r: pd.Series) -> str:
    verdict = r["_verdict"]
    color = VERDICT_COLOR.get(verdict, "#888")
    sym_short = r["yf"].replace("=X", "")

    if pd.notna(r.get("mm_3y_percentile")):
        mm_cell = f'{r["mm_3y_percentile"]:.0f} / {r.get("mm_trajectory", "—")}'
    else:
        mm_cell = "—"

    carry_cell = f'{r["carry_diff"]:+.2f}%' if pd.notna(r.get("carry_diff")) else "—"
    rr_cell    = f'{r["rr_diff"]:+.2f}%'    if pd.notna(r.get("rr_diff"))    else "—"
    cpi_cell   = f'{r["cpi_diff"]:+.2f}pp'  if pd.notna(r.get("cpi_diff"))   else "—"
    gdp_cell   = f'{r["gdp_diff"]:+.2f}pp'  if pd.notna(r.get("gdp_diff"))   else "—"
    health_cell = f'{r["health_diff"]:+.0f}' if pd.notna(r.get("health_diff")) else "—"
    comm_cell  = f'{r.get("comm_diff", 0):+.2f}'

    val_label = r.get("_valuation", "—")
    val_z = r.get("valuation_z")
    val_z_str = f"{val_z:+.2f}σ" if pd.notna(val_z) else "—"
    val_color = _val_color(val_label)

    fv_score = float(r["_fv_score"])
    tech_score = float(r["_tech_score"])
    score = float(r["_score"])
    fv_bias = r["_fv_bias"]

    return f"""
    <tr class="summary-row" data-score="{score}">
      <td><span class="chev">▶</span> <b>{html.escape(sym_short)}</b></td>
      <td>{html.escape(r["name"])}</td>
      <td>{html.escape(r["group"].title())}</td>
      <td>{_fmt(r["price"], 4)}</td>
      <td data-sort="{fv_score}"><b>{fv_score:+.1f}</b></td>
      <td>{html.escape(fv_bias)}</td>
      <td style="color:{color};font-weight:bold">{verdict}</td>
      <td data-sort="{val_z if pd.notna(val_z) else 0}" style="color:{val_color}">{val_label} ({val_z_str})</td>
      <td data-sort="{r.get('carry_diff', 0) if pd.notna(r.get('carry_diff')) else 0}">{carry_cell}</td>
      <td data-sort="{r.get('rr_diff', 0) if pd.notna(r.get('rr_diff')) else 0}">{rr_cell}</td>
      <td data-sort="{r.get('cpi_diff', 0) if pd.notna(r.get('cpi_diff')) else 0}">{cpi_cell}</td>
      <td data-sort="{r.get('gdp_diff', 0) if pd.notna(r.get('gdp_diff')) else 0}">{gdp_cell}</td>
      <td data-sort="{r.get('health_diff', 0) if pd.notna(r.get('health_diff')) else 0}">{health_cell}</td>
      <td data-sort="{r.get('comm_diff', 0)}">{comm_cell}</td>
      <td>{_trend_chips(r['trend_st'], r['trend_mt'], r['trend_lt'])}</td>
      <td>{r['macd']}</td>
      <td>{r['rsi_14']:.1f}</td>
      <td data-sort="{tech_score}">{tech_score:+.1f}</td>
      <td>{mm_cell}</td>
      <td data-sort="{score}"><b>{score:+.1f}</b></td>
    </tr>
    """


def _country_row(ccy: str, r: pd.Series) -> str:
    score = r.get("health_score", 0) or 0
    if score >= 25:
        col = "#5cb85c"
    elif score <= -25:
        col = "#d9534f"
    else:
        col = "#888"

    def f2(v):
        return f"{v:.2f}" if pd.notna(v) else "—"

    return f"""
    <tr>
      <td><b>{html.escape(ccy)}</b></td>
      <td>{html.escape(r.get("country", ""))}</td>
      <td>{f2(r.get("y10"))}</td>
      <td>{f2(r.get("policy"))}</td>
      <td>{f2(r.get("real_rate"))}</td>
      <td>{f2(r.get("cpi_yoy"))}</td>
      <td>{f2(r.get("cpi_6m_chg"))}</td>
      <td>{f2(r.get("unemp"))}</td>
      <td>{f2(r.get("unemp_6m_chg"))}</td>
      <td>{f2(r.get("gdp_yoy"))}</td>
      <td style="color:{col};font-weight:bold">{score:+.0f}</td>
    </tr>
    """


def _detail_card(r: pd.Series) -> str:
    verdict = r["_verdict"]
    color = VERDICT_COLOR.get(verdict, "#888")
    sym_short = r["yf"].replace("=X", "")
    fv_score = float(r["_fv_score"])
    tech_score = float(r["_tech_score"])
    val_score = float(r["_val_score"])
    cot_score = float(r["_cot_score"])
    score = float(r["_score"])
    reasons = r.get("_reasons", []) or []
    val_label = r.get("_valuation", "—")
    val_z = r.get("valuation_z")

    mm_block = (
        f'LevMoney net: {r.get("mm_net_contracts", 0):,}<br>'
        f'%ile 3y: {r["mm_3y_percentile"]:.0f}<br>'
        f'4w change: {r.get("mm_4w_change", 0):+.2f}<br>'
        f'Trajectory: {r.get("mm_trajectory", "—")}<br>'
        f'AssetMgr %ile: {r.get("comm_3y_percentile", 0) or 0:.0f}'
    ) if pd.notna(r.get("mm_3y_percentile")) else '<span class="note">no COT data</span>'

    def f2(v): return f"{v:.2f}" if pd.notna(v) else "—"

    fundamentals_block = f"""
      <b>{html.escape(r['base'])}</b> — y10 {f2(r.get('carry_base'))}%, real {f2(r.get('rr_base'))}%, CPI {f2(r.get('cpi_base'))}%, GDP {f2(r.get('gdp_base'))}%, unemp {f2(r.get('unemp_base'))}%, health {f2(r.get('health_base'))}<br>
      <b>{html.escape(r['quote'])}</b> — y10 {f2(r.get('carry_quote'))}%, real {f2(r.get('rr_quote'))}%, CPI {f2(r.get('cpi_quote'))}%, GDP {f2(r.get('gdp_quote'))}%, unemp {f2(r.get('unemp_quote'))}%, health {f2(r.get('health_quote'))}<br><br>
      <b>Δ (base − quote)</b> — carry {f2(r.get('carry_diff'))}%, real {f2(r.get('rr_diff'))}%, CPI {f2(r.get('cpi_diff'))}pp, GDP {f2(r.get('gdp_diff'))}pp, health {f2(r.get('health_diff'))}
    """

    comm_block = (
        f'Base ({html.escape(r["base"])}): {r.get("comm_base", 0):+.2f}<br>'
        f'Quote ({html.escape(r["quote"])}): {r.get("comm_quote", 0):+.2f}<br>'
        f'Δ tailwind: {r.get("comm_diff", 0):+.2f}<br>'
        f'<span class="note">tanh-squashed weighted 3m comm. moves</span>'
    )

    reg_r2 = r.get("regression_r2")
    reg_slope = r.get("regression_slope_pct_yr")
    reg_fitted = r.get("regression_fitted")
    val_block = (
        f'<b style="color:{_val_color(val_label)}">{val_label}</b><br>'
        f'Residual z: {f2(val_z)}σ<br>'
        f'Trend FV: {_fmt(reg_fitted, 4)}<br>'
        f'Spot: {_fmt(r["price"], 4)}<br>'
        f'R²: {f2(reg_r2)} · slope {f2(reg_slope)}%/yr<br>'
        f'<span class="note">distance from 5y OLS trend</span>'
    )

    score_breakdown_block = (
        f'<b>FV (fundamentals): {fv_score:+.1f}</b><br>'
        f'Valuation (5y MR): {val_score:+.1f}<br>'
        f'COT positioning: {cot_score:+.1f}<br>'
        f'Technicals × 0.4: {0.4 * tech_score:+.2f} <span class="note">(raw {tech_score:+.1f})</span><br>'
        f'<b>Composite: {score:+.1f}</b>'
    )

    reasons_html = "".join(f"<li>{html.escape(t)}</li>" for t in reasons) or "<li>(no rules fired)</li>"

    chart_svg = r.get("regression_svg", "") or ""
    chart_block = f"""
      <div class="chart-row">
        <h4>5Y LINEAR REGRESSION · price vs trend ±1σ / ±2σ</h4>
        {chart_svg}
        <div class="chart-legend">
          <span style="color:#d4af37">━━ price</span> &nbsp;
          <span style="color:#aaa">━━ trend (OLS)</span> &nbsp;
          <span style="color:#aaa">┄┄ ±1σ</span> &nbsp;
          <span style="color:#d9534f">┄┄ +2σ (rich)</span> &nbsp;
          <span style="color:#5cb85c">┄┄ −2σ (cheap)</span>
        </div>
      </div>
    """ if chart_svg else ""

    return f"""
    <div class="card" style="border-left-color:{color}">
      <div class="card-head">
        <span class="sym">{html.escape(sym_short)}</span>
        <span class="name">{html.escape(r["name"])}</span>
        <span class="sec">{html.escape(r["group"])}</span>
        <span class="last">{_fmt(r["price"], 4)}</span>
        <span class="verdict" style="color:{color}">{verdict} · FV {fv_score:+.1f}</span>
      </div>
      {chart_block}
      <div class="card-grid">
        <div><h4>FUNDAMENTALS (FV)</h4><p>{fundamentals_block}</p></div>
        <div><h4>VALUATION (5Y MR)</h4><p>{val_block}</p></div>
        <div><h4>COMMODITY OVERLAY</h4><p>{comm_block}</p></div>
        <div><h4>COT (TFF)</h4><p>{mm_block}</p></div>
        <div><h4>SCORE BREAKDOWN</h4><p>{score_breakdown_block}</p></div>
        <div><h4>TREND (TECH)</h4><p>
          ST: <b class="{r['trend_st'].lower()}">{r['trend_st']}</b><br>
          MT: <b class="{r['trend_mt'].lower()}">{r['trend_mt']}</b><br>
          LT: <b class="{r['trend_lt'].lower()}">{r['trend_lt']}</b><br>
          vs 50d: {_fmt_pct_ratio(r['trend_vs_50d'])}<br>
          vs 200d: {_fmt_pct_ratio(r['trend_vs_200d'])}
        </p></div>
        <div><h4>SIGNALS (TECH)</h4><p>
          RSI: {r['rsi_14']:.1f}<br>
          MACD: {r['macd']}<br>
          Breakout: {r['breakout_20d']}<br>
          1m: {_fmt_pct_ratio(r['mom_1m'])}<br>
          3m: {_fmt_pct_ratio(r['mom_3m'])}
        </p></div>
        <div><h4>VOL / RISK</h4><p>
          RV 20d: {r.get('vol_20d_ann_pct', 0):.1f}%<br>
          RV 60d: {r.get('vol_60d_ann_pct', 0):.1f}%<br>
          %ile (3y): {r.get('vol_pctile_3y', 0):.0f}<br>
          Regime: {r['vol_regime']}<br>
          ATR(14): {_fmt(r.get('atr_14'), 5)}
        </p></div>
        <div><h4>STOPS</h4><p>
          Long stop: {_fmt(r['long_stop'], 4)}<br>
          Short stop: {_fmt(r['short_stop'], 4)}<br>
          Stop dist: {_fmt(r.get('stop_dist'), 5)}<br>
          CTA: {r['cta_label']} ({r['cta_score']:+d})
        </p></div>
      </div>
      <div class="reasons"><h4>RULE-LEVEL BREAKDOWN</h4><ul>{reasons_html}</ul></div>
    </div>
    """


def render(
    df: pd.DataFrame,
    health_df: pd.DataFrame,
    macro: dict,
    out_path: str,
) -> None:
    df = df.copy().reset_index(drop=True)

    summary_rows = []
    n_cols = 20
    for _, row in df.iterrows():
        summary_rows.append(_summary_row(row))
        detail_html = _detail_card(row)
        summary_rows.append(
            f'<tr class="detail-row"><td colspan="{n_cols}">{detail_html}</td></tr>'
        )

    country_rows = ""
    if health_df is not None and not health_df.empty:
        country_rows = "".join(_country_row(ccy, r) for ccy, r in health_df.iterrows())

    n_bull = (df["_verdict"].isin(["BULLISH", "MILDLY BULLISH"])).sum()
    n_bear = (df["_verdict"].isin(["BEARISH", "MILDLY BEARISH"])).sum()
    n_neut = (df["_verdict"] == "NEUTRAL").sum()

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SL Research — Forex Monitor</title>
<style>{CSS}</style></head><body>

<h1>SL RESEARCH — FOREX MONITOR</h1>
<div class="timestamp">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} ·
  {len(df)} pairs &nbsp;·&nbsp; FV verdict: {n_bull} bullish / {n_neut} neutral / {n_bear} bearish
  &nbsp;·&nbsp; verdict reflects fundamentals only — composite score also folds in valuation, COT, and technicals (×0.4)</div>
{_macro_block(macro)}

<h2>SUMMARY — RANKED BY FAIR-VALUE BIAS</h2>
<table id="summary"><thead><tr>
  <th>Pair</th><th>Name</th><th>Group</th><th>Last</th>
  <th>FV</th><th>FV Bias</th><th>Verdict</th><th>Valuation (5y)</th>
  <th>Carry Δ</th><th>Real Δ</th><th>CPI Δ</th><th>GDP Δ</th><th>Health Δ</th><th>Comm Δ</th>
  <th>Trend</th><th>MACD</th><th>RSI</th><th>Tech</th>
  <th>LM %ile / Traj</th><th>Score</th>
</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>

<h2>COUNTRY HEALTH</h2>
<table><thead><tr>
  <th>Ccy</th><th>Country</th><th>10Y%</th><th>Policy%</th><th>Real%</th>
  <th>CPI YoY%</th><th>CPI Δ6m</th><th>Unemp%</th><th>Unemp Δ6m</th>
  <th>GDP YoY%</th><th>Health</th>
</tr></thead><tbody>{country_rows}</tbody></table>

<script>
// Click-to-expand detail rows
document.querySelectorAll('#summary tr.summary-row').forEach(row => {{
  row.addEventListener('click', () => {{
    const detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains('detail-row')) return;
    row.classList.toggle('open');
    detail.classList.toggle('open');
  }});
}});

// Column sorting (keeps summary+detail rows paired)
document.querySelectorAll('#summary th').forEach((th, i) => {{
  let asc = false;
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const allRows = Array.from(tbody.querySelectorAll('tr'));
    const pairs = [];
    for (let k = 0; k < allRows.length; k++) {{
      if (allRows[k].classList.contains('summary-row')) {{
        pairs.push([allRows[k], allRows[k + 1]]);
      }}
    }}
    pairs.sort((a, b) => {{
      const ax = a[0].children[i].dataset.sort ?? a[0].children[i].innerText;
      const bx = b[0].children[i].dataset.sort ?? b[0].children[i].innerText;
      const an = parseFloat(ax), bn = parseFloat(bx);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? ax.localeCompare(bx) : bx.localeCompare(ax);
    }});
    asc = !asc;
    pairs.forEach(p => {{ tbody.appendChild(p[0]); if (p[1]) tbody.appendChild(p[1]); }});
  }});
}});
</script>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html_doc)
