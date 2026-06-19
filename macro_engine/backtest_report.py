"""HTML report for the macro-engine backtest (house style, self-contained)."""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

from macro_engine.backtest import run_backtest
from macro_engine.dashboard import CSS

OUTPUT_DIR = "macro_engine/runs"


def _equity_svg(curves: dict[str, pd.Series], w: int = 1100, h: int = 320) -> str:
    """Multi-line equity curve (each normalized to 1.0 at its first point)."""
    series = {k: v.dropna() for k, v in curves.items() if v is not None and not v.dropna().empty}
    if not series:
        return '<p class="muted">no equity curve</p>'
    all_idx = sorted(set().union(*[set(s.index) for s in series.values()]))
    norm = {k: (s / s.iloc[0]) for k, s in series.items()}
    lo = min(float(s.min()) for s in norm.values())
    hi = max(float(s.max()) for s in norm.values())
    pad, t = 46, 12
    x0, x1, y0, y1 = pad, w - t, t, h - pad
    n = len(all_idx)
    xmap = {d: x0 + (x1 - x0) * i / max(n - 1, 1) for i, d in enumerate(all_idx)}

    def ymap(v):
        return y1 - (y1 - y0) * (v - lo) / (hi - lo if hi > lo else 1)

    colors = {"strategy": "#d4af37", "equal-weight": "#5b9bd5", "60/40": "#888"}
    paths = ""
    for k, s in norm.items():
        pts = " ".join(f"{xmap[d]:.1f},{ymap(float(s.loc[d])):.1f}" for d in s.index)
        c = colors.get(k, "#aaa")
        paths += f'<polyline fill="none" stroke="{c}" stroke-width="1.8" points="{pts}"/>'
    # gridlines + y labels
    grid = ""
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        val = lo + (hi - lo) * frac
        y = ymap(val)
        grid += (f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#2a2a2d"/>'
                 f'<text x="{x0-6}" y="{y+3:.1f}" fill="#777" font-size="9" text-anchor="end">{val:.2f}x</text>')
    # x year labels
    years, seen = "", set()
    for d in all_idx:
        y = pd.Timestamp(d).year
        if y not in seen and pd.Timestamp(d).month <= 2:
            seen.add(y)
            x = xmap[d]
            years += f'<text x="{x:.1f}" y="{h-pad+16}" fill="#777" font-size="9" text-anchor="middle">{y}</text>'
    legend = ""
    for i, (k, c) in enumerate(colors.items()):
        if k in norm:
            lx = x0 + i * 150
            legend += (f'<rect x="{lx}" y="{t-4}" width="11" height="11" fill="{c}"/>'
                       f'<text x="{lx+16}" y="{t+5}" fill="#ccc" font-size="10">{k}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#15151a;border-left:3px solid #d4af37">'
            f'{grid}{paths}{years}{legend}</svg>')


def _stats_table(stats: dict) -> str:
    keys = [("cagr_pct", "CAGR %"), ("vol_pct", "Vol %"), ("sharpe", "Sharpe"),
            ("sortino", "Sortino"), ("max_dd_pct", "Max DD %"), ("calmar", "Calmar"),
            ("hit_daily_pct", "Hit (daily) %"), ("months_pos_pct", "Months + %"),
            ("final_equity", "Final equity")]
    cols = [("strategy", stats), ("equal-weight", stats.get("bench_ew", {})),
            ("60/40", stats.get("bench_6040", {}))]
    head = "".join(f"<th>{html_escape(name)}</th>" for name, _ in cols)
    rows = ""
    for k, lab in keys:
        cells = ""
        for _, st in cols:
            v = st.get(k)
            cells += f"<td>{v if v is not None else '—'}</td>"
        rows += f"<tr><td>{lab}</td>{cells}</tr>"
    span = stats.get("start", "")
    return (f'<p class="muted">{span} → {stats.get("end","")} · {stats.get("n_days","")} trading days</p>'
            f'<table style="max-width:640px"><thead><tr><th>Metric</th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _attr_table(attr: dict) -> str:
    if not attr:
        return ""
    order = ["Goldilocks", "Reflation", "Stagflation", "Slowdown"]
    rows = ""
    for reg in [r for r in order if r in attr] + [r for r in attr if r not in order]:
        a = attr[reg]
        cls = "up" if a["total_pct"] >= 0 else "down"
        rows += (f'<tr><td>{html_escape(reg)}</td><td>{a["n_months"]}</td>'
                 f'<td class="{cls}">{a["mean_pct"]:+.2f}%</td>'
                 f'<td class="{cls}">{a["total_pct"]:+.1f}%</td><td>{a["hit"]:.0f}%</td></tr>')
    return (f'<table style="max-width:520px"><thead><tr><th>Regime</th><th>Months</th>'
            f'<th>Avg/mo</th><th>Total</th><th>Hit</th></tr></thead><tbody>{rows}</tbody></table>')


def html_escape(s) -> str:
    import html as _h
    return _h.escape(str(s))


def render_report(out_path: str, res: dict) -> None:
    stats = res["stats"]
    curves = {"strategy": res.get("equity"), "equal-weight": res.get("bench_ew"),
              "60/40": res.get("bench_6040")}
    cfg = res["config"]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SL Research — Macro Engine Backtest</title><style>{CSS}</style></head><body>
<h1>SL RESEARCH — MACRO ENGINE BACKTEST</h1>
<div class="timestamp">Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} ·
  monthly rebalance · {html_escape(stats.get("start",""))}–{html_escape(stats.get("end",""))} ·
  point-in-time · research only</div>

<div class="macro"><b>HEADLINE</b> &nbsp; Sharpe <b>{stats.get("sharpe","—")}</b> ·
  CAGR {stats.get("cagr_pct","—")}% · vol {stats.get("vol_pct","—")}% ·
  max DD {stats.get("max_dd_pct","—")}% · Calmar {stats.get("calmar","—")} &nbsp;|&nbsp;
  vs equal-weight Sharpe {stats.get("bench_ew",{}).get("sharpe","—")} ·
  60/40 Sharpe {stats.get("bench_6040",{}).get("sharpe","—")}</div>

<h2>EQUITY CURVE (GROWTH OF 1)</h2>
{_equity_svg(curves)}

<h2>PERFORMANCE vs BENCHMARKS</h2>
{_stats_table(stats)}

<h2>RETURN ATTRIBUTION BY REGIME</h2>
{_attr_table(res.get("attribution", {}))}

<div class="macro" style="border-left-color:#888;margin-top:24px"><b>METHOD &amp; LIMITATIONS</b><br>
  Monthly rebalance, vol-parity legs scaled to a {cfg.target_book_vol:.0%} correlation-aware book-vol
  target, gross capped {cfg.max_gross:.0f}x, {cfg.cost_bps:.0f}bps/turnover. Surfaces names where the
  regime-conditional Sharpe rank and the aggregated divergence z <b>agree in sign</b>.<br>
  <b>Excluded vs live engine:</b> the Fed policy-path divergence (single dated snapshot, not history) and
  the COT crowding lens (~3y history). Regime pillars use as-reported FRED (no publication lag) so results
  carry the same mild look-ahead as the live engine — treat as optimistic by that margin.</div>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)


def main() -> None:
    print("macro engine — point-in-time backtest\n")
    res = run_backtest()
    s = res["stats"]
    if not s:
        print("  no results (insufficient history).")
        return
    print(f"  {s['start']} → {s['end']}  ({s['n_days']} days)")
    print(f"  Sharpe {s['sharpe']}  CAGR {s['cagr_pct']}%  vol {s['vol_pct']}%  "
          f"maxDD {s['max_dd_pct']}%  Calmar {s['calmar']}")
    print(f"  vs EW Sharpe {s.get('bench_ew',{}).get('sharpe')}  |  "
          f"60/40 Sharpe {s.get('bench_6040',{}).get('sharpe')}")
    print("\n  By regime:")
    for reg, a in res.get("attribution", {}).items():
        print(f"    {reg:12s} {a['n_months']:3d}mo  avg {a['mean_pct']:+.2f}%/mo  "
              f"total {a['total_pct']:+.1f}%  hit {a['hit']:.0f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"macro_backtest_{stamp}.html")
    render_report(path, res)
    abs_path = os.path.abspath(path)
    print(f"\n  HTML -> {path}")
    if os.getenv("NO_OPEN") != "1":
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "Safari", abs_path], check=False)
        else:
            webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
