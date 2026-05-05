"""SL Research v2 commodities dashboard — single self-contained HTML file.

Mirrors the layout from the user's reference: macro backdrop, summary table
ranked by conviction score, relative-value spreads, then per-asset detail
cards with score breakdown.
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from commodities.scoring import VERDICT_COLOR

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
.spark-wrap{margin-bottom:14px;background:#0e0e10;border:1px solid #2a2a2d;padding:10px 12px}
.spark-wrap h4{margin-bottom:6px}
.spark-wrap svg{display:block;width:100%;height:90px}
.spark-meta{font-size:11px;color:#999;margin-top:4px;display:flex;gap:14px;flex-wrap:wrap}
.spark-meta b{color:#d4af37}
"""


def _fmt_pct(x, digits=1, signed=True):
    if x is None or pd.isna(x):
        return "—"
    fmt = f"%+.{digits}f%%" if signed else f"%.{digits}f%%"
    return fmt % x


def _fmt_pct_ratio(x, digits=1, signed=True):
    """For values stored as ratios (0.05 = 5%)."""
    if x is None or pd.isna(x):
        return "—"
    return _fmt_pct(x * 100, digits, signed)


def _fmt(x, digits=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{digits}f}"


def _trend_chips(st: str, mt: str, lt: str) -> str:
    return f"{st[0]}/{mt[0]}/{lt[0]}"


def _sparkline_svg(closes: pd.Series, width: int = 600, height: int = 90) -> str:
    """Inline SVG line chart of the last ~22 trading days of close prices."""
    s = closes.dropna().tail(22)
    if len(s) < 2:
        return '<div class="note">no price history</div>'

    vals = s.values.astype(float)
    lo, hi = float(vals.min()), float(vals.max())
    rng = (hi - lo) or 1.0
    pad_x, pad_y = 4, 6
    n = len(vals)
    inner_w = width - 2 * pad_x
    inner_h = height - 2 * pad_y

    pts = []
    for i, v in enumerate(vals):
        x = pad_x + (i / (n - 1)) * inner_w
        y = pad_y + inner_h - ((v - lo) / rng) * inner_h
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)

    first, last = vals[0], vals[-1]
    color = "#5cb85c" if last >= first else "#d9534f"
    fill = "rgba(92,184,92,0.10)" if last >= first else "rgba(217,83,79,0.10)"

    # area fill polygon (line + drop to baseline)
    area_pts = poly + f" {pad_x + inner_w:.1f},{pad_y + inner_h:.1f} {pad_x:.1f},{pad_y + inner_h:.1f}"

    last_x = pad_x + inner_w
    last_y = pad_y + inner_h - ((last - lo) / rng) * inner_h

    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polygon points="{area_pts}" fill="{fill}" stroke="none"/>'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5" fill="{color}"/>'
        f'</svg>'
    )


def _spark_block(symbol: str, ohlc: dict[str, pd.DataFrame] | None) -> str:
    if not ohlc or symbol not in ohlc:
        return ""
    df = ohlc[symbol]
    if df is None or df.empty or "Close" not in df.columns:
        return ""
    closes = df["Close"].dropna().tail(22)
    if len(closes) < 2:
        return ""
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    chg_pct = (last / first - 1.0) * 100.0
    lo, hi = float(closes.min()), float(closes.max())
    start_dt = closes.index[0].strftime("%Y-%m-%d")
    end_dt = closes.index[-1].strftime("%Y-%m-%d")
    chg_color = "#5cb85c" if chg_pct >= 0 else "#d9534f"
    svg = _sparkline_svg(closes)
    return f"""
      <div class="spark-wrap">
        <h4>1-MONTH PRICE ({start_dt} → {end_dt})</h4>
        {svg}
        <div class="spark-meta">
          <span>Last: <b>{last:,.2f}</b></span>
          <span>1M chg: <b style="color:{chg_color}">{chg_pct:+.2f}%</b></span>
          <span>1M low: <b>{lo:,.2f}</b></span>
          <span>1M high: <b>{hi:,.2f}</b></span>
          <span>Bars: {len(closes)}</span>
        </div>
      </div>
    """


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
    if "VIX" in macro:
        m = macro["VIX"]
        parts.append(f'VIX {m["last"]:.1f} ({m["pctile_3y"]:.0f}%ile)')
    return f'<div class="macro"><b>MACRO BACKDROP</b> &nbsp; {" &nbsp;|&nbsp; ".join(parts)}</div>'


def _summary_row(r: pd.Series, score: float, verdict: str) -> str:
    color = VERDICT_COLOR.get(verdict, "#888")
    sym_short = r["yf"].replace("=F", "")
    cta_lbl = f'{r["cta_label"]} ({r["cta_score"]:+d})'

    if pd.notna(r.get("mm_3y_percentile")):
        mm_cell = f'{r["mm_3y_percentile"]:.0f} / {r.get("mm_trajectory", "—")}'
    else:
        mm_cell = "—"

    carry_val = r.get("carry_proxy_pct", 0)
    carry_cell = f'{r["carry_state"]} ({carry_val:+.1f})'

    seas_cell = (
        f'{r["seas_avg_pct"]:+.1f}% ({r["seas_hit_rate"]*100:.0f}%)'
        if pd.notna(r.get("seas_avg_pct")) else "—"
    )

    return f"""
    <tr class="summary-row" data-score="{score}">
      <td><span class="chev">▶</span> <b>{html.escape(sym_short)}</b></td>
      <td>{html.escape(r["name"])}</td>
      <td>{html.escape(r["sector"].title())}</td>
      <td>{_fmt(r["price"], 2)}</td>
      <td data-sort="{r['mom_1m']:.6f}">{_fmt_pct_ratio(r['mom_1m'])}</td>
      <td data-sort="{r['mom_3m']:.6f}">{_fmt_pct_ratio(r['mom_3m'])}</td>
      <td data-sort="{r['mom_12_1']:.6f}">{_fmt_pct_ratio(r['mom_12_1'])}</td>
      <td>{_trend_chips(r['trend_st'], r['trend_mt'], r['trend_lt'])}</td>
      <td>{r['rsi_14']:.1f}</td>
      <td>{r['macd']}</td>
      <td>{r['breakout_20d']}</td>
      <td>{carry_cell}</td>
      <td>{r['vol_regime']} ({r.get('vol_pctile_3y', 0):.0f}%ile)</td>
      <td>{cta_lbl}</td>
      <td>{mm_cell}</td>
      <td>{seas_cell}</td>
      <td>L:{_fmt(r['long_stop'], 2)} S:{_fmt(r['short_stop'], 2)}</td>
      <td style="color:{color};font-weight:bold">{verdict}</td>
      <td data-sort="{score}"><b>{score:+.1f}</b></td>
    </tr>
    """


def _spread_row(r: pd.Series) -> str:
    return f"""
    <tr>
      <td><b>{html.escape(r['spread'])}</b></td>
      <td>{_fmt(r['last'], 3)}</td>
      <td>{r['z_3y']:+.2f}</td>
      <td>{r['pctile_3y']:.0f}</td>
      <td>{r['chg_1m_pct']:+.2f}%</td>
      <td>{r['signal']}</td>
    </tr>
    """


def _detail_card(r: pd.Series, score: float, reasons: list[str], verdict: str,
                 ohlc: dict[str, pd.DataFrame] | None = None) -> str:
    color = VERDICT_COLOR.get(verdict, "#888")
    sym_short = r["yf"].replace("=F", "")
    spark_html = _spark_block(r["yf"], ohlc)

    mm_block = ""
    if pd.notna(r.get("mm_3y_percentile")):
        mm_block = (
            f'MM net: {r.get("mm_net_contracts", 0):,}<br>'
            f'MM %ile 3y: {r["mm_3y_percentile"]:.0f}<br>'
            f'4w change: {r.get("mm_4w_change", 0):+.2f}<br>'
            f'Trajectory: {r.get("mm_trajectory", "—")}<br>'
            f'Comm. %ile: {r.get("comm_3y_percentile", 0) or 0:.0f}'
        )
    else:
        mm_block = '<span class="note">no COT data</span>'

    carry_note = '<span class="note">v1 proxy — replace with F_near/F_far for prod</span>' \
        if r.get("carry_is_proxy") else ""

    reasons_html = "".join(f"<li>{html.escape(t)}</li>" for t in reasons) or "<li>(no rules fired)</li>"

    return f"""
    <div class="card" style="border-left-color:{color}">
      <div class="card-head">
        <span class="sym">{html.escape(sym_short)}</span>
        <span class="name">{html.escape(r["name"])}</span>
        <span class="sec">{html.escape(r["sector"])}</span>
        <span class="last">{_fmt(r["price"], 2)}</span>
        <span class="verdict" style="color:{color}">{verdict} ({score:+.1f})</span>
      </div>
      {spark_html}
      <div class="card-grid">
        <div><h4>TREND</h4><p>
          ST: <b class="{r['trend_st'].lower()}">{r['trend_st']}</b><br>
          MT: <b class="{r['trend_mt'].lower()}">{r['trend_mt']}</b><br>
          LT: <b class="{r['trend_lt'].lower()}">{r['trend_lt']}</b><br>
          vs 50d: {_fmt_pct_ratio(r['trend_vs_50d'])}<br>
          vs 200d: {_fmt_pct_ratio(r['trend_vs_200d'])}
        </p></div>
        <div><h4>SIGNALS</h4><p>
          RSI: {r['rsi_14']:.1f}<br>
          MACD: {r['macd']}<br>
          Breakout: {r['breakout_20d']}<br>
          1m: {_fmt_pct_ratio(r['mom_1m'])}<br>
          3m: {_fmt_pct_ratio(r['mom_3m'])}
        </p></div>
        <div><h4>CARRY</h4><p>
          State: {r['carry_state']}<br>
          Proxy: {r.get('carry_proxy_pct', 0):+.1f}%<br><br>
          {carry_note}
        </p></div>
        <div><h4>VOL REGIME</h4><p>
          RV 20d: {r.get('vol_20d_ann_pct', 0):.1f}%<br>
          RV 60d: {r.get('vol_60d_ann_pct', 0):.1f}%<br>
          %ile (3y): {r.get('vol_pctile_3y', 0):.0f}<br>
          Regime: {r['vol_regime']}
        </p></div>
        <div><h4>CTA POSITIONING</h4><p>
          {r['cta_label']} ({r['cta_score']:+d})<br>
          1m: {r['cta_z_1m']:+d}<br>
          3m: {r['cta_z_3m']:+d}<br>
          12m: {r['cta_z_12m']:+d}
        </p></div>
        <div><h4>COT</h4><p>{mm_block}</p></div>
        <div><h4>SEASONALITY ({datetime.utcnow().strftime("%b")})</h4><p>
          Avg ({r.get('seas_n_years', 0)}y): {r.get('seas_avg_pct', 0):+.2f}%<br>
          Hit rate: {(r.get('seas_hit_rate', 0) or 0)*100:.0f}%
        </p></div>
        <div><h4>RISK / SIZING</h4><p>
          ATR(14): {_fmt(r.get('atr_14'), 4)}<br>
          Stop dist: {_fmt(r.get('stop_dist'), 4)}<br>
          Long stop: {_fmt(r['long_stop'], 2)}<br>
          Short stop: {_fmt(r['short_stop'], 2)}
        </p></div>
      </div>
      <div class="reasons"><h4>SCORE BREAKDOWN</h4><ul>{reasons_html}</ul></div>
    </div>
    """


def render_v2(
    df: pd.DataFrame,
    spreads_df: pd.DataFrame,
    macro: dict,
    scored: list[tuple[float, list[str], str]],
    out_path: str,
    ohlc: dict[str, pd.DataFrame] | None = None,
) -> None:
    # Caller passes df pre-sorted by score, with _score/_verdict/_reasons attached.
    df = df.copy().reset_index(drop=True)

    summary_rows = []
    n_cols = 19  # number of <th> in summary header
    for _, row in df.iterrows():
        score = float(row["_score"])
        verdict = row["_verdict"]
        reasons = row.get("_reasons", []) if "_reasons" in df.columns else []
        summary_rows.append(_summary_row(row, score, verdict))
        detail_html = _detail_card(row, score, reasons, verdict, ohlc)
        summary_rows.append(
            f'<tr class="detail-row"><td colspan="{n_cols}">{detail_html}</td></tr>'
        )

    spread_rows = "".join(_spread_row(r) for _, r in spreads_df.iterrows()) if not spreads_df.empty else ""

    n_bull = (df["_verdict"].isin(["BULLISH", "MILDLY BULLISH"])).sum()
    n_bear = (df["_verdict"].isin(["BEARISH", "MILDLY BEARISH"])).sum()
    n_neut = (df["_verdict"] == "NEUTRAL").sum()

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SL Research — Commodity Dashboard v2</title>
<style>{CSS}</style></head><body>

<h1>SL RESEARCH — COMMODITY DASHBOARD</h1>
<div class="timestamp">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} ·
  {len(df)} contracts &nbsp;·&nbsp; {n_bull} bullish / {n_neut} neutral / {n_bear} bearish</div>
{_macro_block(macro)}

<h2>SUMMARY — RANKED BY CONVICTION SCORE</h2>
<table id="summary"><thead><tr>
  <th>Sym</th><th>Name</th><th>Sec</th><th>Last</th>
  <th>1M</th><th>3M</th><th>12M</th>
  <th>Trend</th><th>RSI</th><th>MACD</th><th>Break</th>
  <th>Curve</th><th>Vol Regime</th>
  <th>CTA</th><th>MM %ile / Traj</th><th>Seasonal</th>
  <th>Stops L/S</th><th>Verdict</th><th>Score</th>
</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>

<h2>RELATIVE VALUE / SPREADS</h2>
<table><thead><tr>
  <th>Spread</th><th>Last</th><th>Z(3y)</th><th>%ile 3y</th><th>1M%</th><th>Signal</th>
</tr></thead><tbody>{spread_rows}</tbody></table>

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
