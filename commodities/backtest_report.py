"""HTML report for the backtest: stats panel, equity curve, drawdown,
monthly heatmap, recent trades. Same dark/gold theme as the dashboard."""
from __future__ import annotations

from datetime import datetime

import pandas as pd


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=IBM+Plex+Mono:wght@400;500;700&display=swap');
body{font-family:'IBM Plex Mono',monospace;background:#0e0e10;color:#e6e6e6;
     padding:24px;max-width:1700px;margin:auto}
h1{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:3px;font-size:32px;margin:0 0 8px 0}
h2{font-family:'Cormorant Garamond',serif;font-weight:300;letter-spacing:2px;font-size:22px;margin:32px 0 12px 0;color:#d4af37}
.timestamp{color:#888;font-size:11px;margin-bottom:24px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:28px}
.stat{background:#1a1a1d;padding:14px;border-left:3px solid #d4af37}
.stat .label{color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.stat .value{color:#e6e6e6;font-size:22px;font-weight:700}
.stat .sub{color:#aaa;font-size:11px;margin-top:4px}
.chart-wrap{background:#15151a;padding:18px;margin-bottom:18px;border-left:3px solid #444}
.note{color:#888;font-size:11px;margin-top:8px;font-style:italic}
table{border-collapse:collapse;font-size:11px;width:100%;margin-bottom:16px}
th{background:#1a1a1d;padding:8px 6px;text-align:left;border-bottom:2px solid #d4af37;font-size:10px;letter-spacing:1px}
td{padding:6px;border-bottom:1px solid #2a2a2d;font-variant-numeric:tabular-nums}
tr:hover{background:#1a1a1d}
.up{color:#5cb85c}.down{color:#d9534f}
.heatmap td{text-align:right;padding:5px 8px;font-size:10px}
.heatmap th{padding:5px 8px;text-align:center}
"""


def _color_for_return(ret: float) -> str:
    """Green/red gradient for monthly heatmap."""
    if pd.isna(ret):
        return "#15151a"
    if ret > 0:
        intensity = min(abs(ret) / 0.05, 1.0)
        g = int(50 + 130 * intensity)
        return f"rgb(20,{g},40)"
    intensity = min(abs(ret) / 0.05, 1.0)
    r = int(60 + 150 * intensity)
    return f"rgb({r},25,25)"


def _equity_svg(equity: pd.Series, drawdown: pd.Series, width: int = 1600, height: int = 280) -> str:
    """Inline SVG of equity curve + shaded drawdown area."""
    if equity.empty:
        return ""
    n = len(equity)
    pad = 40
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    eq_min, eq_max = float(equity.min()), float(equity.max())
    rng = (eq_max - eq_min) or 0.01

    pts_eq = []
    for i, v in enumerate(equity.values):
        x = pad + i * plot_w / max(n - 1, 1)
        y = pad + plot_h - (v - eq_min) / rng * plot_h
        pts_eq.append(f"{x:.1f},{y:.1f}")

    # drawdown shaded area (bottom 60px of chart)
    dd_h = 60
    dd_pts = [f"{pad},{height - pad}"]
    for i, v in enumerate(drawdown.values):
        x = pad + i * plot_w / max(n - 1, 1)
        y = (height - pad) - (-v / max(abs(drawdown.min()), 0.01)) * dd_h
        dd_pts.append(f"{x:.1f},{y:.1f}")
    dd_pts.append(f"{pad + plot_w},{height - pad}")

    # axis labels — equity start/end and dates
    start_lbl = equity.index[0].strftime("%Y-%m")
    end_lbl = equity.index[-1].strftime("%Y-%m")

    return f'''
    <svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block">
      <rect width="{width}" height="{height}" fill="#15151a"/>
      <polygon points="{" ".join(dd_pts)}" fill="#5a1a1a" opacity="0.45"/>
      <polyline points="{" ".join(pts_eq)}" fill="none" stroke="#d4af37" stroke-width="1.8"/>
      <line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" stroke="#444" stroke-width="0.5"/>
      <text x="{pad}" y="{pad - 8}" fill="#888" font-size="11" font-family="IBM Plex Mono">
        equity {eq_min:.2f}x → {eq_max:.2f}x
      </text>
      <text x="{pad}" y="{height - 12}" fill="#888" font-size="10" font-family="IBM Plex Mono">{start_lbl}</text>
      <text x="{width - pad - 50}" y="{height - 12}" fill="#888" font-size="10" font-family="IBM Plex Mono">{end_lbl}</text>
      <text x="{width - pad - 100}" y="{pad - 8}" fill="#d9534f" font-size="11" font-family="IBM Plex Mono">
        max DD {drawdown.min():.1%}
      </text>
    </svg>
    '''


def _monthly_heatmap_html(monthly: pd.Series) -> str:
    if monthly.empty:
        return "<p class='note'>No monthly data</p>"
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header = "<tr><th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>YTD</th></tr>"
    rows = []
    for year, row in pivot.iterrows():
        cells = [f"<td><b>{year}</b></td>"]
        ytd = 1.0
        for m in range(1, 13):
            v = row.get(m, float("nan"))
            if pd.notna(v):
                cells.append(f'<td style="background:{_color_for_return(v)}">{v*100:+.1f}</td>')
                ytd *= (1 + v)
            else:
                cells.append('<td style="color:#444">—</td>')
        ytd_pct = ytd - 1
        ytd_color = "#5cb85c" if ytd_pct > 0 else "#d9534f"
        cells.append(f'<td style="color:{ytd_color};font-weight:700">{ytd_pct*100:+.1f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f'<table class="heatmap">{header}{"".join(rows)}</table>'


def render_backtest_report(result: dict, out_path: str) -> None:
    stats = result.get("stats", {})
    equity = result.get("equity", pd.Series())
    daily = result.get("daily_returns", pd.Series())
    trades = result.get("trades", pd.DataFrame())
    cfg = result.get("config")

    if equity.empty:
        with open(out_path, "w") as f:
            f.write("<html><body><p>No backtest data — check date range / data availability.</p></body></html>")
        return

    drawdown = equity / equity.cummax() - 1

    sharpe = stats.get("sharpe", 0)
    sharpe_color = "#5cb85c" if sharpe >= 1 else ("#e8a33d" if sharpe >= 0.5 else "#d9534f")
    cagr_color = "#5cb85c" if stats.get("cagr", 0) > 0 else "#d9534f"

    # last 20 rebalances
    if not trades.empty:
        recent = trades.sort_values("date", ascending=False).head(60)
        trade_rows = "".join(
            f'<tr><td>{r["date"].strftime("%Y-%m-%d")}</td><td>{r["symbol"]}</td>'
            f'<td class="{"up" if r["side"]=="LONG" else "down"}">{r["side"]}</td>'
            f'<td>{r["weight"]:+.2%}</td></tr>'
            for _, r in recent.iterrows()
        )
    else:
        trade_rows = "<tr><td colspan=4>No trades</td></tr>"

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SL Research — Backtest Report</title>
<style>{CSS}</style></head><body>

<h1>SL RESEARCH — BACKTEST REPORT</h1>
<div class="timestamp">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} ·
  Strategy: cross-sectional, top {cfg.n_per_side if cfg else 5} long / bottom {cfg.n_per_side if cfg else 5} short ·
  Rebalance: weekly ({cfg.rebalance if cfg else "W-FRI"}) ·
  Costs: {cfg.cost_bps_per_side if cfg else 5} bps/side ·
  Vol target: {(cfg.target_vol_per_leg*100) if cfg else 10:.0f}%/leg
</div>

<h2>HEADLINE STATS — {stats.get("start", "?")} to {stats.get("end", "?")}</h2>
<div class="stats">
  <div class="stat"><div class="label">Sharpe</div>
    <div class="value" style="color:{sharpe_color}">{stats.get("sharpe", 0):.2f}</div>
    <div class="sub">rolling 1y: {stats.get("rolling_1y_sharpe", 0):.2f}</div></div>
  <div class="stat"><div class="label">CAGR</div>
    <div class="value" style="color:{cagr_color}">{stats.get("cagr", 0)*100:+.1f}%</div>
    <div class="sub">vol {stats.get("vol_ann", 0)*100:.1f}%</div></div>
  <div class="stat"><div class="label">Max Drawdown</div>
    <div class="value" style="color:#d9534f">{stats.get("max_drawdown", 0)*100:.1f}%</div></div>
  <div class="stat"><div class="label">Sortino</div>
    <div class="value">{stats.get("sortino", 0):.2f}</div></div>
  <div class="stat"><div class="label">Daily Hit Rate</div>
    <div class="value">{stats.get("hit_rate_daily", 0)*100:.0f}%</div></div>
  <div class="stat"><div class="label">Months Positive</div>
    <div class="value">{stats.get("months_positive", 0)*100:.0f}%</div>
    <div class="sub">{stats.get("n_days", 0)} trading days</div></div>
</div>

<h2>EQUITY CURVE & DRAWDOWN</h2>
<div class="chart-wrap">
  {_equity_svg(equity, drawdown)}
  <div class="note">Gold = strategy equity (gross of taxes, includes 5bps/side cost on rebalance turnover).
  Red shaded = drawdown from peak. yfinance front-month continuous prices include some roll noise.</div>
</div>

<h2>MONTHLY RETURNS</h2>
{_monthly_heatmap_html(stats.get("monthly_returns", pd.Series(dtype=float)))}

<h2>RECENT POSITIONS (LAST 60)</h2>
<table>
  <thead><tr><th>Rebal Date</th><th>Symbol</th><th>Side</th><th>Weight</th></tr></thead>
  <tbody>{trade_rows}</tbody>
</table>

<h2>WHAT TO TAKE AWAY</h2>
<div class="chart-wrap">
  <div class="note" style="font-size:13px;color:#ccc;line-height:1.7">
  <p><b>How to read this:</b> the strategy ranks all ~20 commodities every Friday by the v2 conviction score
  and goes long the top {cfg.n_per_side if cfg else 5} / short the bottom {cfg.n_per_side if cfg else 5},
  weighted to {(cfg.target_vol_per_leg*100) if cfg else 10:.0f}% volatility per leg.
  Sharpe ≥1 = real edge worth scaling. 0.5–1.0 = noisy but plausible. &lt;0.5 = re-think.</p>
  <p><b>Honest caveats:</b> (1) yfinance =F symbols are continuous front-month, NOT roll-adjusted.
  Roll noise probably costs the strategy ~50–200 bps/yr. (2) Carry is the heuristic proxy, not a real curve —
  the strongest commodity factor is therefore underweighted. (3) Backtest uses point-in-time COT but no
  point-in-time macro overlay. (4) Costs assume liquid execution; real futures slippage may be higher
  in less-liquid contracts (e.g., palladium).</p>
  <p><b>If results look strong:</b> next steps are (a) wire in a real curve feed, (b) paper-trade for 3-6 months
  to validate, (c) parameter-sweep n_per_side and rebalance frequency.<br>
  <b>If results look weak:</b> the score weights need re-calibration — likely candidates are reducing trend weight,
  amplifying COT trajectory and carry once real, and adding a vol-regime filter.</p>
  </div>
</div>

</body></html>"""

    with open(out_path, "w") as f:
        f.write(html_doc)
