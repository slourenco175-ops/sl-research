import { useState, useEffect, useRef } from "react";

const MODEL = "claude-sonnet-4-20250514";

const FLAGS = {
  US:"🇺🇸", EU:"🇪🇺", UK:"🇬🇧", JP:"🇯🇵", CN:"🇨🇳",
  CH:"🇨🇭", CA:"🇨🇦", AU:"🇦🇺", BR:"🇧🇷", TR:"🇹🇷",
  DE:"🇩🇪", FR:"🇫🇷", IT:"🇮🇹", ES:"🇪🇸"
};

// ─── COMMODITY × COUNTRY BETAS (terms of trade) ────────────────────────────
// Rough β: country economy sensitivity to a 1% move in each commodity.
// Sign = net exporter (+) vs net importer (-). Magnitude = trade/GDP exposure.
// Stylised, calibrated from BIS / World Bank / IEA / USDA / EITI data.
// Tickers must match `commodities[].ticker` in the feed.
const COMMODITY_BETAS = {
  US: { BRENT: 0.1,  WTI: 0.2,  NG: 0.4,  GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.1, CORN: 0.4,  WHEAT: 0.3,  SOY: 0.5 },
  EU: { BRENT: -0.6, WTI: -0.5, NG: -0.7, GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.3, CORN: -0.2, WHEAT: 0.0,  SOY: -0.2 },
  UK: { BRENT: -0.3, WTI: -0.3, NG: -0.4, GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.3, CORN: -0.2, WHEAT: -0.1, SOY: -0.2 },
  JP: { BRENT: -0.8, WTI: -0.8, NG: -0.7, GOLD: 0.0,  SILVER: -0.1, COPPER: -0.5, CORN: -0.3, WHEAT: -0.5, SOY: -0.3 },
  CN: { BRENT: -0.5, WTI: -0.5, NG: -0.4, GOLD: 0.1,  SILVER: 0.0,  COPPER: -0.3, CORN: -0.3, WHEAT: -0.2, SOY: -0.6 },
  CH: { BRENT: -0.2, WTI: -0.2, NG: -0.2, GOLD: 0.2,  SILVER: 0.1,  COPPER: -0.1, CORN: -0.1, WHEAT: -0.1, SOY: -0.1 },
  CA: { BRENT: 0.7,  WTI: 0.7,  NG: 0.5,  GOLD: 0.3,  SILVER: 0.2,  COPPER: 0.3,  CORN: 0.2,  WHEAT: 0.4,  SOY: 0.2 },
  AU: { BRENT: 0.0,  WTI: 0.0,  NG: 0.4,  GOLD: 0.4,  SILVER: 0.3,  COPPER: 0.5,  CORN: 0.0,  WHEAT: 0.3,  SOY: 0.0 },
  BR: { BRENT: 0.6,  WTI: 0.6,  NG: 0.1,  GOLD: 0.2,  SILVER: 0.1,  COPPER: 0.3,  CORN: 0.5,  WHEAT: 0.0,  SOY: 0.7 },
  TR: { BRENT: -0.7, WTI: -0.7, NG: -0.6, GOLD: 0.1,  SILVER: 0.0,  COPPER: -0.2, CORN: -0.2, WHEAT: -0.3, SOY: -0.2 },
  DE: { BRENT: -0.5, WTI: -0.5, NG: -0.7, GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.4, CORN: -0.1, WHEAT: 0.0,  SOY: -0.2 },
  FR: { BRENT: -0.5, WTI: -0.5, NG: -0.4, GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.2, CORN: 0.2,  WHEAT: 0.4,  SOY: -0.1 },
  IT: { BRENT: -0.6, WTI: -0.6, NG: -0.6, GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.3, CORN: -0.1, WHEAT: -0.2, SOY: -0.2 },
  ES: { BRENT: -0.5, WTI: -0.5, NG: -0.5, GOLD: 0.0,  SILVER: 0.0,  COPPER: -0.2, CORN: -0.1, WHEAT: -0.2, SOY: -0.2 },
};

// ─── HEALTH SCORING ENGINE ─────────────────────────────────────────────────
// Each component scored on its own scale, then summed. Total /100.
//   GROWTH (22)         : GDP YoY — higher = better
//   INFLATION (22)      : distance from 2% target
//   PMI (18)            : composite, 50 = neutral
//   REAL RATES (13)     : 0–3% sweet spot
//   DEBT / GDP (13)     : <60% best, >130% punitive
//   TERMS OF TRADE (12) : commodity tailwind/headwind via β × 1W move
const WEIGHTS = { growth: 22, inflation: 22, pmi: 18, realRate: 13, debt: 13, termsOfTrade: 12 };

function scoreGrowth(v) {
  if (v == null) return null;
  if (v >= 4) return 22;
  if (v >= 0) return 11 + (v / 4) * 11;         // 0% → 11, 4% → 22
  return Math.max(0, 11 + v * 3.7);              // ~-3% → 0
}
function scoreInflation(v) {
  if (v == null) return null;
  return Math.max(0, 22 * (1 - Math.abs(v - 2) / 4));
}
function scorePMI(v) {
  if (v == null) return null;
  return Math.max(0, Math.min(18, ((v - 40) / 20) * 18));
}
function scoreRealRate(rate, cpi) {
  if (rate == null || cpi == null) return null;
  const r = rate - cpi;
  if (r >= 0 && r <= 3) return 13;
  if (r > 3 && r <= 6) return 13 - (r - 3) * 1.7;
  if (r > 6) return 3;
  if (r < 0 && r >= -2) return 13 + r * 4.3;     // -2 → ~4.4
  return 0;
}
function scoreDebt(v) {
  if (v == null) return null;
  if (v < 60) return 13;
  if (v < 90) return 13 - ((v - 60) / 30) * 4.3;       // 90% → ~8.7
  if (v < 130) return 8.7 - ((v - 90) / 40) * 4.3;     // 130% → ~4.4
  if (v < 200) return Math.max(0, 4.4 - ((v - 130) / 70) * 4.4);
  return 0;
}
function computeTermsOfTrade(code, d) {
  const m = COMMODITY_BETAS[code];
  if (!m || !d?.commodities) return null;
  let signal = 0, count = 0;
  for (const c of d.commodities) {
    const beta = m[c.ticker];
    if (beta == null || c.change_1w == null) continue;
    signal += beta * c.change_1w;
    count++;
  }
  return count > 0 ? +signal.toFixed(2) : null;
}
function scoreTermsOfTrade(signal) {
  if (signal == null) return null;
  // Signal in pp (e.g. +2 = strong tailwind). Clamp to [-3, +3] → [0, 12].
  const clamped = Math.max(-3, Math.min(3, signal));
  return ((clamped + 3) / 6) * 12;
}

function computeHealthScore(code, d) {
  const cb   = d.central_banks?.find(x => x.code === code);
  const inf  = d.inflation?.find(x => x.code === code);
  const gdp  = d.gdp_growth?.find(x => x.code === code);
  const pmi  = d.pmi?.find(x => x.code === code);
  const debt = d.debt_to_gdp?.find(x => x.code === code);

  const pmiVal = pmi?.composite ?? pmi?.manufacturing;
  const realRate = (cb?.rate != null && inf?.value != null) ? +(cb.rate - inf.value).toFixed(2) : null;
  const totSignal = computeTermsOfTrade(code, d);

  const c = {
    growth:       scoreGrowth(gdp?.value),
    inflation:    scoreInflation(inf?.value),
    pmi:          scorePMI(pmiVal),
    realRate:     scoreRealRate(cb?.rate, inf?.value),
    debt:         scoreDebt(debt?.value),
    termsOfTrade: scoreTermsOfTrade(totSignal),
  };

  let sum = 0, maxAvail = 0;
  for (const k of Object.keys(WEIGHTS)) {
    if (c[k] != null) { sum += c[k]; maxAvail += WEIGHTS[k]; }
  }
  const score = maxAvail > 0 ? (sum / maxAvail) * 100 : null;

  return {
    code,
    country: cb?.country ?? gdp?.country ?? inf?.country ?? pmi?.country ?? debt?.country ?? code,
    score: score != null ? +score.toFixed(1) : null,
    components: c,
    coverage: maxAvail / 100,
    raw: {
      gdp: gdp?.value, cpi: inf?.value, pmi: pmiVal,
      rate: cb?.rate, realRate, debt: debt?.value,
      tot: totSignal,
    },
  };
}

function buildRanking(d) {
  if (!d) return [];
  const codes = new Set();
  ["central_banks","inflation","gdp_growth","pmi","debt_to_gdp"].forEach(k =>
    (d[k] || []).forEach(x => codes.add(x.code))
  );
  return [...codes]
    .map(c => computeHealthScore(c, d))
    .filter(x => x.score != null && x.coverage >= 0.4)
    .sort((a, b) => b.score - a.score);
}

function healthLabel(s) {
  if (s == null) return { label: "—", color: "#4a5f72" };
  if (s >= 75) return { label: "STRONG",   color: "#22d87a" };
  if (s >= 60) return { label: "HEALTHY",  color: "#7ed957" };
  if (s >= 45) return { label: "MIXED",    color: "#f5a623" };
  if (s >= 30) return { label: "WEAK",     color: "#f08c4d" };
  return         { label: "STRESSED", color: "#f04d5e" };
}

const PROMPT = `You are a real-time macro data aggregator. Use web search to find the LATEST available macro data from Trading Economics, central bank websites, and financial sources. Today is April 2026.

Return ONLY a valid JSON object — no markdown, no backticks, no explanation. Just raw JSON starting with { and ending with }.

{
  "central_banks": [
    {"country":"United States","code":"US","bank":"Fed","rate":4.5,"prev_rate":4.5,"last_change":"Dec 2024","next_meeting":"May 2025","bias":"neutral"},
    {"country":"Euro Area","code":"EU","bank":"ECB","rate":2.5,"prev_rate":2.75,"last_change":"Mar 2025","next_meeting":"Apr 2025","bias":"dovish"},
    {"country":"United Kingdom","code":"UK","bank":"BoE","rate":4.5,"prev_rate":4.75,"last_change":"Feb 2025","next_meeting":"May 2025","bias":"dovish"},
    {"country":"Japan","code":"JP","bank":"BoJ","rate":0.5,"prev_rate":0.25,"last_change":"Jan 2025","next_meeting":"May 2025","bias":"hawkish"},
    {"country":"China","code":"CN","bank":"PBoC","rate":3.1,"prev_rate":3.35,"last_change":"Oct 2024","next_meeting":null,"bias":"dovish"},
    {"country":"Switzerland","code":"CH","bank":"SNB","rate":0.25,"prev_rate":0.5,"last_change":"Mar 2025","next_meeting":"Jun 2025","bias":"dovish"},
    {"country":"Canada","code":"CA","bank":"BoC","rate":2.75,"prev_rate":3.0,"last_change":"Mar 2025","next_meeting":"Apr 2025","bias":"neutral"},
    {"country":"Australia","code":"AU","bank":"RBA","rate":4.1,"prev_rate":4.35,"last_change":"Feb 2025","next_meeting":"May 2025","bias":"neutral"},
    {"country":"Brazil","code":"BR","bank":"BCB","rate":14.75,"prev_rate":13.75,"last_change":"Mar 2025","next_meeting":"May 2025","bias":"hawkish"},
    {"country":"Turkey","code":"TR","bank":"TCMB","rate":42.5,"prev_rate":45.0,"last_change":"Mar 2025","next_meeting":"Apr 2025","bias":"dovish"}
  ],
  "inflation": [
    {"country":"United States","code":"US","value":2.4,"prev":2.8,"date":"Mar 2025"},
    {"country":"Euro Area","code":"EU","value":2.2,"prev":2.3,"date":"Mar 2025"},
    {"country":"United Kingdom","code":"UK","value":2.6,"prev":2.8,"date":"Feb 2025"},
    {"country":"Japan","code":"JP","value":3.6,"prev":4.0,"date":"Feb 2025"},
    {"country":"China","code":"CN","value":-0.1,"prev":0.5,"date":"Mar 2025"},
    {"country":"Canada","code":"CA","value":2.3,"prev":1.9,"date":"Mar 2025"},
    {"country":"Australia","code":"AU","value":2.4,"prev":2.5,"date":"Q4 2024"},
    {"country":"Brazil","code":"BR","value":5.5,"prev":5.1,"date":"Mar 2025"},
    {"country":"Turkey","code":"TR","value":38.1,"prev":42.1,"date":"Mar 2025"}
  ],
  "gdp_growth": [
    {"country":"United States","code":"US","value":2.4,"prev":3.1,"period":"Q4 2024","type":"YoY"},
    {"country":"Euro Area","code":"EU","value":1.2,"prev":0.9,"period":"Q4 2024","type":"YoY"},
    {"country":"United Kingdom","code":"UK","value":1.5,"prev":1.0,"period":"Q4 2024","type":"YoY"},
    {"country":"Japan","code":"JP","value":-0.2,"prev":0.9,"period":"Q4 2024","type":"YoY"},
    {"country":"China","code":"CN","value":5.4,"prev":4.6,"period":"Q4 2024","type":"YoY"},
    {"country":"Germany","code":"DE","value":-0.2,"prev":-0.3,"period":"Q4 2024","type":"YoY"},
    {"country":"Brazil","code":"BR","value":3.5,"prev":3.3,"period":"Q4 2024","type":"YoY"},
    {"country":"Australia","code":"AU","value":1.3,"prev":0.8,"period":"Q4 2024","type":"YoY"}
  ],
  "pmi": [
    {"country":"United States","code":"US","composite":51.2,"manufacturing":50.2,"services":54.4,"date":"Mar 2025"},
    {"country":"Euro Area","code":"EU","composite":50.9,"manufacturing":48.6,"services":51.0,"date":"Mar 2025"},
    {"country":"United Kingdom","code":"UK","composite":50.5,"manufacturing":44.9,"services":53.0,"date":"Mar 2025"},
    {"country":"Japan","code":"JP","composite":48.4,"manufacturing":48.4,"services":50.0,"date":"Mar 2025"},
    {"country":"China","code":"CN","composite":51.8,"manufacturing":51.2,"services":52.0,"date":"Mar 2025"},
    {"country":"Germany","code":"DE","composite":50.9,"manufacturing":48.3,"services":53.0,"date":"Mar 2025"},
    {"country":"France","code":"FR","composite":48.0,"manufacturing":46.8,"services":47.9,"date":"Mar 2025"}
  ],
  "bond_yields": [
    {"country":"United States","code":"US","tenor":"10Y","yield":4.35,"change_1d":-3.2},
    {"country":"Germany","code":"DE","tenor":"10Y","yield":2.52,"change_1d":-1.8},
    {"country":"United Kingdom","code":"UK","tenor":"10Y","yield":4.48,"change_1d":-2.1},
    {"country":"Japan","code":"JP","tenor":"10Y","yield":1.53,"change_1d":1.5},
    {"country":"Italy","code":"IT","tenor":"10Y","yield":3.62,"change_1d":-1.4},
    {"country":"Spain","code":"ES","tenor":"10Y","yield":3.18,"change_1d":-1.2},
    {"country":"France","code":"FR","tenor":"10Y","yield":3.25,"change_1d":-1.5}
  ],
  "fx": [
    {"pair":"EUR/USD","rate":1.1387,"change_1d":0.42,"change_1w":1.85},
    {"pair":"GBP/USD","rate":1.2905,"change_1d":0.21,"change_1w":0.94},
    {"pair":"USD/JPY","rate":142.30,"change_1d":-0.85,"change_1w":-3.20},
    {"pair":"USD/CHF","rate":0.8185,"change_1d":-0.55,"change_1w":-2.10},
    {"pair":"AUD/USD","rate":0.6325,"change_1d":0.31,"change_1w":0.75},
    {"pair":"USD/CNY","rate":7.285,"change_1d":-0.12,"change_1w":-0.45},
    {"pair":"USD/BRL","rate":5.845,"change_1d":-0.35,"change_1w":-1.20},
    {"pair":"DXY","rate":99.35,"change_1d":-0.45,"change_1w":-2.10}
  ],
  "commodities": [
    {"name":"Brent Crude","ticker":"BRENT","price":66.80,"unit":"USD/bbl","change_1d":-1.20,"change_1w":-3.50},
    {"name":"WTI Crude","ticker":"WTI","price":63.20,"unit":"USD/bbl","change_1d":-1.35,"change_1w":-3.80},
    {"name":"Natural Gas","ticker":"NG","price":3.45,"unit":"USD/MMBtu","change_1d":0.85,"change_1w":2.10},
    {"name":"Gold","ticker":"GOLD","price":3315.0,"unit":"USD/oz","change_1d":0.65,"change_1w":3.20},
    {"name":"Silver","ticker":"SILVER","price":32.50,"unit":"USD/oz","change_1d":0.42,"change_1w":1.85},
    {"name":"Copper","ticker":"COPPER","price":4.68,"unit":"USD/lb","change_1d":-0.95,"change_1w":-2.40},
    {"name":"Corn","ticker":"CORN","price":485.0,"unit":"USc/bu","change_1d":0.30,"change_1w":0.85},
    {"name":"Wheat","ticker":"WHEAT","price":545.0,"unit":"USc/bu","change_1d":-0.55,"change_1w":-1.20},
    {"name":"Soybeans","ticker":"SOY","price":1015.0,"unit":"USc/bu","change_1d":0.22,"change_1w":0.60}
  ],
  "debt_to_gdp": [
    {"country":"United States","code":"US","value":123.0,"period":"2024"},
    {"country":"Euro Area","code":"EU","value":88.0,"period":"2024"},
    {"country":"United Kingdom","code":"UK","value":98.0,"period":"2024"},
    {"country":"Japan","code":"JP","value":250.0,"period":"2024"},
    {"country":"China","code":"CN","value":84.0,"period":"2024"},
    {"country":"Switzerland","code":"CH","value":38.0,"period":"2024"},
    {"country":"Canada","code":"CA","value":107.0,"period":"2024"},
    {"country":"Australia","code":"AU","value":57.0,"period":"2024"},
    {"country":"Brazil","code":"BR","value":85.0,"period":"2024"},
    {"country":"Turkey","code":"TR","value":28.0,"period":"2024"},
    {"country":"Germany","code":"DE","value":63.0,"period":"2024"},
    {"country":"France","code":"FR","value":112.0,"period":"2024"},
    {"country":"Italy","code":"IT","value":138.0,"period":"2024"},
    {"country":"Spain","code":"ES","value":107.0,"period":"2024"}
  ]
}

Search for the most current values above and update them with live data. Return ONLY the JSON.`;

// ─── helpers ───────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => (v == null || isNaN(v)) ? "—" : Number(v).toFixed(d);

function DeltaBadge({ v, suffix = "" }) {
  if (v == null || isNaN(v)) return <span style={{ color: "#4a5f72" }}>—</span>;
  const pos = v > 0.009;
  const neg = v < -0.009;
  const color = pos ? "#22d87a" : neg ? "#f04d5e" : "#4a5f72";
  return (
    <span style={{ color, fontFamily: "var(--mono)", fontSize: 11, fontWeight: 500 }}>
      {pos ? "▲" : neg ? "▼" : "●"} {Math.abs(v).toFixed(2)}{suffix}
    </span>
  );
}

function BiasBadge({ bias }) {
  const map = {
    hawkish: { bg: "rgba(240,77,94,.12)", color: "#f04d5e", border: "rgba(240,77,94,.25)" },
    dovish:  { bg: "rgba(34,216,122,.1)", color: "#22d87a", border: "rgba(34,216,122,.25)" },
    neutral: { bg: "rgba(245,166,35,.1)", color: "#f5a623", border: "rgba(245,166,35,.22)" },
  };
  const s = map[(bias||"neutral").toLowerCase()] || map.neutral;
  return (
    <span style={{
      background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      fontFamily: "var(--mono)", fontSize: 9, letterSpacing: "0.08em",
      padding: "2px 7px", textTransform: "uppercase"
    }}>{bias || "—"}</span>
  );
}

// ─── sub-panels ────────────────────────────────────────────────────────────
function StripTile({ label, value, delta, deltaLabel }) {
  return (
    <div style={{
      background: "var(--bg1)", border: "1px solid var(--border)",
      padding: "10px 14px", minWidth: 110, flexShrink: 0,
      cursor: "default", transition: "border-color .15s"
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = "var(--amber)"}
      onMouseLeave={e => e.currentTarget.style.borderColor = "var(--border)"}
    >
      <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 16, fontWeight: 500, color: "var(--text)", lineHeight: 1 }}>{value}</div>
      <div style={{ marginTop: 3 }}><DeltaBadge v={delta} suffix={deltaLabel || ""} /></div>
    </div>
  );
}

function SectionHeader({ title }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, marginTop: 24 }}>
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--amber)", letterSpacing: "0.12em", textTransform: "uppercase", flexShrink: 0 }}>{title}</span>
      <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
    </div>
  );
}

function DataTable({ headers, rows }) {
  return (
    <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} style={{
                fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)",
                letterSpacing: "0.1em", textTransform: "uppercase",
                textAlign: h.right ? "right" : "left",
                padding: "8px 12px", background: "var(--bg)", borderBottom: "1px solid var(--border)",
                fontWeight: 400, whiteSpace: "nowrap"
              }}>{h.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}
              onMouseEnter={e => { for (const td of e.currentTarget.cells) td.style.background = "var(--bg2)"; }}
              onMouseLeave={e => { for (const td of e.currentTarget.cells) td.style.background = ""; }}
            >
              {row.map((cell, j) => (
                <td key={j} style={{
                  fontFamily: "var(--mono)", fontSize: 12, color: "var(--text2)",
                  padding: "9px 12px",
                  borderBottom: i < rows.length - 1 ? "1px solid var(--border)" : "none",
                  textAlign: headers[j].right ? "right" : "left",
                  ...cell.style
                }}>{cell.node || cell.value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── OVERVIEW TAB ──────────────────────────────────────────────────────────
function OverviewTab({ d }) {
  const ranking = buildRanking(d);
  return (
    <div>
      {ranking.length > 0 && (
        <>
          <SectionHeader title="GLOBAL HEALTH INDEX" />
          <GlobalHealthGauge ranking={ranking} />
        </>
      )}
      <SectionHeader title="KEY POLICY RATES" />
      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
        {(d.central_banks || []).map(b => (
          <StripTile key={b.code} label={`${FLAGS[b.code]||""} ${b.bank}`}
            value={`${fmt(b.rate, 2)}%`} delta={b.rate - b.prev_rate} deltaLabel="%" />
        ))}
      </div>

      <SectionHeader title="INFLATION (CPI YoY)" />
      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
        {(d.inflation || []).map(i => (
          <StripTile key={i.code} label={`${FLAGS[i.code]||""} ${i.code} CPI`}
            value={`${fmt(i.value, 1)}%`} delta={i.value - i.prev} deltaLabel="pp" />
        ))}
      </div>

      <SectionHeader title="FX SNAPSHOT" />
      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
        {(d.fx || []).map(f => (
          <StripTile key={f.pair} label={f.pair}
            value={fmt(f.rate, f.pair.includes("JPY") || f.pair === "DXY" ? 2 : 4)}
            delta={f.change_1d} deltaLabel="%" />
        ))}
      </div>

      <SectionHeader title="COMMODITIES" />
      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
        {(d.commodities || []).map(c => (
          <StripTile key={c.ticker} label={c.name}
            value={fmt(c.price, c.price > 100 ? 1 : 3)}
            delta={c.change_1d} deltaLabel="%" />
        ))}
      </div>

      <SectionHeader title="REAL RATES (POLICY RATE − CPI)" />
      <RealRatesPanel d={d} />
    </div>
  );
}

function RealRatesPanel({ d }) {
  const combined = (d.central_banks || []).map(b => {
    const inf = (d.inflation || []).find(i => i.code === b.code);
    return { ...b, real_rate: inf ? +(b.rate - inf.value).toFixed(2) : null };
  }).filter(x => x.real_rate != null).sort((a, b) => b.real_rate - a.real_rate);

  const maxAbs = Math.max(...combined.map(x => Math.abs(x.real_rate)), 5);

  return (
    <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", padding: 16 }}>
      {combined.map(x => {
        const pct = Math.abs(x.real_rate) / maxAbs * 100;
        const color = x.real_rate > 0 ? "#22d87a" : "#f04d5e";
        return (
          <div key={x.code} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text2)", width: 44, textAlign: "right", flexShrink: 0 }}>
              {FLAGS[x.code]||""} {x.code}
            </div>
            <div style={{ flex: 1, height: 8, background: "var(--bg2)", position: "relative", border: "1px solid var(--border)" }}>
              <div style={{
                position: "absolute", [x.real_rate >= 0 ? "left" : "right"]: 0,
                width: `${pct}%`, height: "100%", background: color, opacity: 0.6,
                transition: "width .8s cubic-bezier(.4,0,.2,1)"
              }} />
              <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: "var(--border2)" }} />
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color, width: 46, flexShrink: 0 }}>
              {x.real_rate > 0 ? "+" : ""}{fmt(x.real_rate, 2)}%
            </div>
          </div>
        );
      })}
      <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", marginTop: 8, letterSpacing: "0.06em" }}>
        Real Rate = Policy Rate − CPI Inflation (YoY) · Green = positive real rate · Red = negative real rate
      </div>
    </div>
  );
}

// ─── RATES TAB ─────────────────────────────────────────────────────────────
function RatesTab({ d }) {
  const headers = [
    { label: "Country" }, { label: "Bank" },
    { label: "Rate", right: true }, { label: "Prev", right: true },
    { label: "Δ", right: true }, { label: "Last Change" },
    { label: "Next Meeting" }, { label: "Bias" }
  ];
  const rows = (d.central_banks || []).map(b => {
    const diff = b.rate - b.prev_rate;
    return [
      { value: `${FLAGS[b.code]||""} ${b.country}`, style: { color: "var(--text)" } },
      { value: b.bank, style: { color: "var(--text3)" } },
      { node: <span style={{ background: "rgba(245,166,35,.1)", color: "var(--amber)", border: "1px solid rgba(245,166,35,.2)", fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, padding: "2px 8px" }}>{fmt(b.rate, 2)}%</span> },
      { value: `${fmt(b.prev_rate, 2)}%`, style: { color: "var(--text3)" } },
      { node: <DeltaBadge v={diff} suffix="%" /> },
      { value: b.last_change || "—", style: { color: "var(--text3)" } },
      { value: b.next_meeting || "—", style: { color: "var(--text2)" } },
      { node: <BiasBadge bias={b.bias} /> },
    ];
  });
  return (
    <div>
      <SectionHeader title="CENTRAL BANK POLICY RATES" />
      <DataTable headers={headers} rows={rows} />
      <SectionHeader title="REAL RATES SPREAD" />
      <RealRatesPanel d={d} />
    </div>
  );
}

// ─── INFLATION TAB ─────────────────────────────────────────────────────────
function InflationTab({ d }) {
  const infHeaders = [
    { label: "Country" }, { label: "CPI YoY%", right: true },
    { label: "Prev", right: true }, { label: "Δ MoM", right: true }, { label: "Period" }
  ];
  const infRows = [...(d.inflation || [])].sort((a, b) => b.value - a.value).map(i => {
    const diff = i.value - i.prev;
    const valColor = i.value > 4 ? "#f04d5e" : i.value > 2 ? "#f5a623" : "#22d87a";
    return [
      { value: `${FLAGS[i.code]||""} ${i.country}` },
      { node: <b style={{ color: valColor }}>{fmt(i.value, 1)}%</b> },
      { value: `${fmt(i.prev, 1)}%`, style: { color: "var(--text3)" } },
      { node: <DeltaBadge v={diff} suffix="pp" /> },
      { value: i.date || "—", style: { color: "var(--text3)" } }
    ];
  });

  const gdpHeaders = [
    { label: "Country" }, { label: "GDP Growth", right: true },
    { label: "Prev", right: true }, { label: "Type" }, { label: "Period" }
  ];
  const gdpRows = (d.gdp_growth || []).map(g => {
    const valColor = g.value > 2 ? "#22d87a" : g.value > 0 ? "#f5a623" : "#f04d5e";
    return [
      { value: `${FLAGS[g.code]||""} ${g.country}` },
      { node: <b style={{ color: valColor }}>{fmt(g.value, 1)}%</b> },
      { value: `${fmt(g.prev, 1)}%`, style: { color: "var(--text3)" } },
      { value: g.type || "—", style: { color: "var(--text3)" } },
      { value: g.period || "—", style: { color: "var(--text3)" } }
    ];
  });

  return (
    <div>
      <SectionHeader title="CONSUMER PRICE INFLATION (YoY %)" />
      <DataTable headers={infHeaders} rows={infRows} />
      <SectionHeader title="GDP GROWTH" />
      <DataTable headers={gdpHeaders} rows={gdpRows} />
    </div>
  );
}

// ─── PMI TAB ───────────────────────────────────────────────────────────────
function PMITab({ d }) {
  return (
    <div>
      <SectionHeader title="PURCHASING MANAGERS INDEX" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10 }}>
        {(d.pmi || []).map(p => {
          const val = p.composite || p.manufacturing;
          const color = val > 50 ? "#22d87a" : val < 50 ? "#f04d5e" : "#f5a623";
          const signal = val > 50 ? "EXPANSION" : val < 50 ? "CONTRACTION" : "NEUTRAL";
          return (
            <div key={p.code} style={{
              background: "var(--bg1)", border: "1px solid var(--border)",
              padding: "14px 16px", position: "relative", overflow: "hidden"
            }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text3)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>
                {FLAGS[p.code]||""} {p.country}
              </div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 28, fontWeight: 500, color, lineHeight: 1, marginBottom: 4 }}>
                {fmt(val, 1)}
              </div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 9, color, letterSpacing: "0.08em" }}>{signal}</div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", marginTop: 6 }}>
                {p.manufacturing != null && `MFG ${fmt(p.manufacturing, 1)}`}
                {p.services != null && `  SVC ${fmt(p.services, 1)}`}
              </div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", marginTop: 2 }}>{p.date}</div>
              <div style={{
                position: "absolute", bottom: 0, left: 0,
                width: `${Math.max(0, Math.min(100, (val - 30) / 40 * 100))}%`,
                height: 2, background: color, opacity: 0.5
              }} />
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 16, fontFamily: "var(--mono)", fontSize: 10, color: "var(--text3)", letterSpacing: "0.06em" }}>
        PMI REFERENCE: &gt;50 = EXPANSION · 50 = NEUTRAL · &lt;50 = CONTRACTION
      </div>
    </div>
  );
}

// ─── BONDS TAB ─────────────────────────────────────────────────────────────
function BondsTab({ d }) {
  const bondHeaders = [
    { label: "Country" }, { label: "Tenor" },
    { label: "Yield", right: true }, { label: "1D Chg (bp)", right: true }
  ];
  const bondRows = (d.bond_yields || []).map(b => [
    { value: `${FLAGS[b.code]||""} ${b.country}` },
    { value: b.tenor, style: { color: "var(--text3)" } },
    { node: <b style={{ color: "var(--amber)" }}>{fmt(b.yield, 3)}%</b> },
    { node: <DeltaBadge v={b.change_1d} suffix="bp" /> }
  ]);

  const fxHeaders = [
    { label: "Pair" }, { label: "Rate", right: true },
    { label: "1D %", right: true }, { label: "1W %", right: true }
  ];
  const fxRows = (d.fx || []).map(f => {
    const dec = f.pair.includes("JPY") || f.pair === "DXY" ? 2 : 4;
    return [
      { value: f.pair, style: { fontWeight: 500, color: "var(--text)" } },
      { node: <b style={{ color: "var(--text)" }}>{fmt(f.rate, dec)}</b> },
      { node: <DeltaBadge v={f.change_1d} suffix="%" /> },
      { node: <DeltaBadge v={f.change_1w} suffix="%" /> }
    ];
  });

  return (
    <div>
      <SectionHeader title="10Y SOVEREIGN BOND YIELDS" />
      <DataTable headers={bondHeaders} rows={bondRows} />
      <SectionHeader title="FX RATES" />
      <DataTable headers={fxHeaders} rows={fxRows} />
    </div>
  );
}

// ─── COMMODITIES TAB ───────────────────────────────────────────────────────
function CommoditiesTab({ d }) {
  const headers = [
    { label: "Commodity" }, { label: "Price", right: true },
    { label: "Unit" }, { label: "1D %", right: true }, { label: "1W %", right: true }
  ];
  const rows = (d.commodities || []).map(c => [
    { value: c.name, style: { fontWeight: 500, color: "var(--text)" } },
    { node: <b style={{ color: "var(--amber)" }}>{fmt(c.price, c.price > 100 ? 1 : 3)}</b> },
    { value: c.unit, style: { color: "var(--text3)" } },
    { node: <DeltaBadge v={c.change_1d} suffix="%" /> },
    { node: <DeltaBadge v={c.change_1w} suffix="%" /> }
  ]);
  return (
    <div>
      <SectionHeader title="COMMODITIES" />
      <DataTable headers={headers} rows={rows} />
    </div>
  );
}

// ─── GLOBAL HEALTH GAUGE ───────────────────────────────────────────────────
function GlobalHealthGauge({ ranking }) {
  if (!ranking?.length) return null;
  const avg = ranking.reduce((s, r) => s + r.score, 0) / ranking.length;
  const lab = healthLabel(avg);
  const top3 = ranking.slice(0, 3);
  const bot3 = ranking.slice(-3).reverse();

  // arc gauge
  const angle = Math.min(180, Math.max(0, (avg / 100) * 180));
  const r = 70, cx = 90, cy = 80;
  const rad = (a) => ((a - 180) * Math.PI) / 180;
  const x = cx + r * Math.cos(rad(angle));
  const y = cy + r * Math.sin(rad(angle));

  return (
    <div style={{
      background: "var(--bg1)", border: "1px solid var(--border)",
      padding: 18, display: "grid", gridTemplateColumns: "200px 1fr 1fr", gap: 18, alignItems: "center"
    }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <svg width="180" height="100" viewBox="0 0 180 100">
          {/* arc background */}
          <path d={`M 20 80 A 70 70 0 0 1 160 80`} stroke="var(--bg2)" strokeWidth="10" fill="none" strokeLinecap="round" />
          {/* coloured segments behind */}
          <path d="M 20 80 A 70 70 0 0 1 55 22" stroke="#f04d5e" strokeWidth="10" fill="none" opacity="0.3" />
          <path d="M 55 22 A 70 70 0 0 1 90 10" stroke="#f5a623" strokeWidth="10" fill="none" opacity="0.3" />
          <path d="M 90 10 A 70 70 0 0 1 125 22" stroke="#7ed957" strokeWidth="10" fill="none" opacity="0.3" />
          <path d="M 125 22 A 70 70 0 0 1 160 80" stroke="#22d87a" strokeWidth="10" fill="none" opacity="0.3" />
          {/* needle */}
          <line x1={cx} y1={cy} x2={x} y2={y} stroke={lab.color} strokeWidth="2" strokeLinecap="round" />
          <circle cx={cx} cy={cy} r="4" fill={lab.color} />
          {/* score */}
          <text x={cx} y={cy + 18} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 22, fontWeight: 600, fill: lab.color }}>{avg.toFixed(1)}</text>
        </svg>
        <div style={{
          fontFamily: "var(--mono)", fontSize: 10, color: lab.color,
          letterSpacing: "0.12em", marginTop: 2,
          background: `${lab.color}1f`, border: `1px solid ${lab.color}55`,
          padding: "3px 10px"
        }}>{lab.label}</div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", letterSpacing: "0.08em", marginTop: 8 }}>
          GLOBAL HEALTH INDEX · N={ranking.length}
        </div>
      </div>

      <div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--green)", letterSpacing: "0.12em", marginBottom: 8 }}>▲ TOP 3 ECONOMIES</div>
        {top3.map((r, i) => (
          <MiniRankRow key={r.code} rank={i + 1} r={r} />
        ))}
      </div>

      <div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--red)", letterSpacing: "0.12em", marginBottom: 8 }}>▼ BOTTOM 3 ECONOMIES</div>
        {bot3.map((r, i) => (
          <MiniRankRow key={r.code} rank={ranking.length - i} r={r} />
        ))}
      </div>
    </div>
  );
}

function MiniRankRow({ rank, r }) {
  const lab = healthLabel(r.score);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, fontFamily: "var(--mono)", fontSize: 11 }}>
      <span style={{ color: "var(--text3)", width: 22, fontSize: 10 }}>#{rank}</span>
      <span style={{ width: 18 }}>{FLAGS[r.code] || ""}</span>
      <span style={{ color: "var(--text2)", flex: 1 }}>{r.country}</span>
      <span style={{ color: lab.color, fontWeight: 600 }}>{r.score.toFixed(1)}</span>
    </div>
  );
}

// ─── RANKING TAB ───────────────────────────────────────────────────────────
function RankingTab({ d }) {
  const ranking = buildRanking(d);
  if (!ranking.length) return <div style={{ color: "var(--text3)", fontFamily: "var(--mono)" }}>Insufficient data for ranking.</div>;

  return (
    <div>
      <SectionHeader title="GLOBAL HEALTH INDEX" />
      <GlobalHealthGauge ranking={ranking} />

      <SectionHeader title="ECONOMIC HEALTH RANKING" />
      <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {[
                {l:"#"},{l:"Country"},{l:"Score",r:true},{l:"Status"},
                {l:"Growth",r:true},{l:"Inflation",r:true},{l:"PMI",r:true},
                {l:"Real Rate",r:true},{l:"Debt/GDP",r:true},{l:"ToT",r:true}
              ].map((h, i) => (
                <th key={i} style={{
                  fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)",
                  letterSpacing: "0.1em", textTransform: "uppercase",
                  textAlign: h.r ? "right" : "left",
                  padding: "8px 12px", background: "var(--bg)",
                  borderBottom: "1px solid var(--border)", fontWeight: 400
                }}>{h.l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ranking.map((r, i) => {
              const lab = healthLabel(r.score);
              return (
                <tr key={r.code}
                  onMouseEnter={e => { for (const td of e.currentTarget.cells) td.style.background = "var(--bg2)"; }}
                  onMouseLeave={e => { for (const td of e.currentTarget.cells) td.style.background = ""; }}
                >
                  <td style={cellStyle(i, ranking.length, { color: "var(--text3)", width: 30 })}>
                    <span style={{
                      display: "inline-block", width: 22, textAlign: "center",
                      background: i < 3 ? "var(--amber-dim)" : "transparent",
                      color: i < 3 ? "var(--amber)" : "var(--text3)",
                      padding: "1px 0", fontWeight: i < 3 ? 600 : 400
                    }}>{i + 1}</span>
                  </td>
                  <td style={cellStyle(i, ranking.length)}>{FLAGS[r.code]||""} {r.country}</td>
                  <td style={cellStyle(i, ranking.length, { textAlign: "right" })}>
                    <ScoreBar score={r.score} color={lab.color} />
                  </td>
                  <td style={cellStyle(i, ranking.length)}>
                    <span style={{
                      display: "inline-block", fontFamily: "var(--mono)", fontSize: 9,
                      letterSpacing: "0.08em", padding: "2px 7px",
                      background: `${lab.color}1c`, color: lab.color, border: `1px solid ${lab.color}44`
                    }}>{lab.label}</span>
                  </td>
                  <ComponentCell i={i} n={ranking.length} val={r.raw.gdp}     suffix="%"  comp={r.components.growth}    max={WEIGHTS.growth} />
                  <ComponentCell i={i} n={ranking.length} val={r.raw.cpi}     suffix="%"  comp={r.components.inflation} max={WEIGHTS.inflation} />
                  <ComponentCell i={i} n={ranking.length} val={r.raw.pmi}     suffix=""   comp={r.components.pmi}       max={WEIGHTS.pmi} pmi />
                  <ComponentCell i={i} n={ranking.length} val={r.raw.realRate} suffix="%" comp={r.components.realRate}  max={WEIGHTS.realRate} />
                  <ComponentCell i={i} n={ranking.length} val={r.raw.debt}    suffix="%"  comp={r.components.debt}      max={WEIGHTS.debt} debt />
                  <ComponentCell i={i} n={ranking.length} val={r.raw.tot}     suffix="pp" comp={r.components.termsOfTrade} max={WEIGHTS.termsOfTrade} />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <SectionHeader title="DEBT / GDP RATIOS" />
      <DebtPanel d={d} />

      <div style={{ marginTop: 18, padding: "12px 14px", background: "var(--bg1)", border: "1px dashed var(--border2)" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--amber)", letterSpacing: "0.12em", marginBottom: 6 }}>METHODOLOGY</div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text2)", lineHeight: 1.6 }}>
          Score = Growth (22) + Inflation prox. to 2% (22) + PMI (18) + Real Rate (13) + Debt/GDP (13) + Terms of Trade (12) = 100.<br/>
          <b style={{ color: "var(--amber)" }}>Terms of Trade</b> = Σ (β × commodity 1W % chg). β reflects net-exporter exposure: e.g. CA β<sub>oil</sub>=+0.7, JP β<sub>oil</sub>=−0.8, AU β<sub>copper</sub>=+0.5, BR β<sub>soybeans</sub>=+0.7. Output is a weekly tailwind/headwind in pp; clamped ±3pp → 0–12 score.<br/>
          Inflation penalises both deflation and high inflation symmetrically. Real-rate sweet spot is 0–3%. Debt/GDP: &lt;60% full marks, &gt;130% steep penalty. Recomputes on every refresh.
        </div>
      </div>
    </div>
  );
}

const cellStyle = (i, n, extra = {}) => ({
  fontFamily: "var(--mono)", fontSize: 12, color: "var(--text2)",
  padding: "9px 12px",
  borderBottom: i < n - 1 ? "1px solid var(--border)" : "none",
  ...extra
});

function ScoreBar({ score, color }) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, justifyContent: "flex-end", width: "100%" }}>
      <div style={{ width: 90, height: 6, background: "var(--bg2)", border: "1px solid var(--border)", position: "relative" }}>
        <div style={{ width: `${score}%`, height: "100%", background: color, opacity: 0.75, transition: "width .8s cubic-bezier(.4,0,.2,1)" }} />
      </div>
      <span style={{ color, fontWeight: 600, minWidth: 36, textAlign: "right" }}>{score.toFixed(1)}</span>
    </div>
  );
}

function ComponentCell({ i, n, val, suffix, comp, max, pmi, debt }) {
  if (val == null) return <td style={cellStyle(i, n, { textAlign: "right", color: "var(--text3)" })}>—</td>;
  const pct = comp != null ? comp / max : 0;
  // colour by sub-score health
  let color;
  if (comp == null) color = "var(--text3)";
  else if (pct >= 0.7) color = "#22d87a";
  else if (pct >= 0.4) color = "#f5a623";
  else color = "#f04d5e";

  let display;
  if (pmi)       display = val.toFixed(1);
  else if (debt) display = val.toFixed(0) + suffix;
  else           display = (val > 0 ? "+" : "") + val.toFixed(1) + suffix;

  return (
    <td style={cellStyle(i, n, { textAlign: "right" })}>
      <span style={{ color, fontWeight: 500 }}>{display}</span>
      <span style={{ color: "var(--text3)", fontSize: 9, marginLeft: 4 }}>
        ({comp != null ? comp.toFixed(0) : "—"}/{max})
      </span>
    </td>
  );
}

function DebtPanel({ d }) {
  const debt = [...(d.debt_to_gdp || [])].sort((a, b) => b.value - a.value);
  if (!debt.length) return <div style={{ color: "var(--text3)", fontFamily: "var(--mono)", fontSize: 11 }}>No debt data.</div>;
  const max = Math.max(...debt.map(x => x.value), 100);

  return (
    <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", padding: 16 }}>
      {debt.map(x => {
        const pct = (x.value / max) * 100;
        let color = "#22d87a";
        if (x.value > 130) color = "#f04d5e";
        else if (x.value > 90) color = "#f08c4d";
        else if (x.value > 60) color = "#f5a623";
        return (
          <div key={x.code} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text2)", width: 60, flexShrink: 0 }}>
              {FLAGS[x.code] || ""} {x.code}
            </div>
            <div style={{ flex: 1, height: 10, background: "var(--bg2)", border: "1px solid var(--border)", position: "relative" }}>
              <div style={{ width: `${pct}%`, height: "100%", background: color, opacity: 0.7, transition: "width .8s cubic-bezier(.4,0,.2,1)" }} />
              {/* threshold markers */}
              <div style={{ position: "absolute", left: `${(60/max)*100}%`, top: -2, width: 1, height: 14, background: "rgba(245,166,35,.4)" }} />
              <div style={{ position: "absolute", left: `${(90/max)*100}%`, top: -2, width: 1, height: 14, background: "rgba(240,140,77,.4)" }} />
              <div style={{ position: "absolute", left: `${(130/max)*100}%`, top: -2, width: 1, height: 14, background: "rgba(240,77,94,.4)" }} />
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color, width: 60, flexShrink: 0, textAlign: "right" }}>
              {x.value.toFixed(0)}%
            </div>
          </div>
        );
      })}
      <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", marginTop: 10, letterSpacing: "0.06em" }}>
        Thresholds: 60% (Maastricht) · 90% (Reinhart-Rogoff) · 130% (high stress)
      </div>
    </div>
  );
}

// ─── TERMS OF TRADE TAB ────────────────────────────────────────────────────
function TermsOfTradeTab({ d }) {
  const commodities = d?.commodities || [];
  const codes = Object.keys(COMMODITY_BETAS);

  // current tailwind ranking
  const tot = codes
    .map(code => {
      const sig = computeTermsOfTrade(code, d);
      const country = d.central_banks?.find(x => x.code === code)?.country
        || d.gdp_growth?.find(x => x.code === code)?.country
        || d.inflation?.find(x => x.code === code)?.country
        || code;
      return { code, country, signal: sig };
    })
    .filter(x => x.signal != null)
    .sort((a, b) => b.signal - a.signal);

  // colour for β cell
  const betaColor = (b) => {
    if (b == null) return { bg: "var(--bg2)", fg: "var(--text3)" };
    const t = Math.min(1, Math.abs(b));
    if (b > 0)  return { bg: `rgba(34,216,122,${0.08 + t * 0.45})`, fg: t > 0.5 ? "#fff" : "#22d87a" };
    if (b < 0)  return { bg: `rgba(240,77,94,${0.08 + t * 0.45})`,  fg: t > 0.5 ? "#fff" : "#f04d5e" };
    return { bg: "var(--bg2)", fg: "var(--text3)" };
  };

  // colour for tailwind value
  const sigColor = (s) => s > 0.5 ? "#22d87a" : s > 0 ? "#7ed957" : s > -0.5 ? "#f5a623" : "#f04d5e";

  const maxAbsSig = Math.max(...tot.map(x => Math.abs(x.signal)), 0.5);

  return (
    <div>
      <SectionHeader title="CURRENT TAILWIND / HEADWIND (1W)" />
      <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", padding: 16 }}>
        {tot.map(t => {
          const pct = (Math.abs(t.signal) / maxAbsSig) * 50;
          const color = sigColor(t.signal);
          const side = t.signal >= 0 ? "left" : "right";
          return (
            <div key={t.code} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text2)", width: 130, flexShrink: 0 }}>
                {FLAGS[t.code] || ""} {t.country}
              </div>
              <div style={{ flex: 1, height: 9, background: "var(--bg2)", border: "1px solid var(--border)", position: "relative" }}>
                <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: "var(--border2)" }} />
                <div style={{
                  position: "absolute", [side]: "50%", width: `${pct}%`, height: "100%",
                  background: color, opacity: 0.7,
                  transform: t.signal < 0 ? "translateX(0)" : "none",
                  transition: "width .8s cubic-bezier(.4,0,.2,1)"
                }} />
              </div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 11, color, width: 70, flexShrink: 0, textAlign: "right", fontWeight: 600 }}>
                {t.signal >= 0 ? "+" : ""}{t.signal.toFixed(2)}pp
              </div>
            </div>
          );
        })}
        <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", marginTop: 10, letterSpacing: "0.06em" }}>
          Signal = Σ(β × commodity 1W % change). Positive = commodity tailwind. Negative = headwind.
        </div>
      </div>

      <SectionHeader title="β MATRIX · COUNTRY × COMMODITY" />
      <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 720 }}>
          <thead>
            <tr>
              <th style={{
                fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)",
                letterSpacing: "0.1em", textTransform: "uppercase",
                padding: "8px 12px", background: "var(--bg)",
                borderBottom: "1px solid var(--border)", fontWeight: 400,
                position: "sticky", left: 0, zIndex: 1, textAlign: "left"
              }}>Country</th>
              {commodities.map(c => (
                <th key={c.ticker} style={{
                  fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)",
                  letterSpacing: "0.06em", textTransform: "uppercase",
                  padding: "8px 6px", background: "var(--bg)",
                  borderBottom: "1px solid var(--border)", fontWeight: 400,
                  textAlign: "center", minWidth: 56
                }}>
                  <div>{c.ticker}</div>
                  <div style={{
                    fontSize: 9, fontWeight: 500, marginTop: 2,
                    color: c.change_1w > 0 ? "#22d87a" : c.change_1w < 0 ? "#f04d5e" : "var(--text3)"
                  }}>
                    {c.change_1w > 0 ? "+" : ""}{(c.change_1w ?? 0).toFixed(1)}%
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {codes.map((code, i) => (
              <tr key={code}>
                <td style={{
                  fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)",
                  padding: "8px 12px", borderBottom: i < codes.length - 1 ? "1px solid var(--border)" : "none",
                  position: "sticky", left: 0, background: "var(--bg1)",
                  whiteSpace: "nowrap"
                }}>
                  {FLAGS[code] || ""} {code}
                </td>
                {commodities.map(c => {
                  const b = COMMODITY_BETAS[code]?.[c.ticker];
                  const col = betaColor(b);
                  return (
                    <td key={c.ticker} style={{
                      fontFamily: "var(--mono)", fontSize: 11, fontWeight: 500,
                      padding: "8px 6px", textAlign: "center",
                      color: col.fg, background: col.bg,
                      borderBottom: i < codes.length - 1 ? "1px solid var(--border)" : "none"
                    }}>
                      {b == null ? "—" : (b > 0 ? "+" : "") + b.toFixed(1)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text3)", marginTop: 8, letterSpacing: "0.06em" }}>
        β interpretation: +1.0 = country GDP rises ~1% for every 1% commodity rally. Sign = exporter (+) vs importer (−). Magnitude = trade exposure.
      </div>

      <SectionHeader title="LINKAGE NOTES" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          { t: "ENERGY EXPORTERS", c: "#22d87a", txt: "CA, BR positive on Brent/WTI (β≈+0.6 to +0.7). US neutral-positive on WTI, strong-positive on NG via LNG exports. Russia/Saudi proxies if added." },
          { t: "ENERGY IMPORTERS", c: "#f04d5e", txt: "JP, EU, DE, IT all deeply negative on oil & gas (β≈−0.5 to −0.8). Turkey acutely exposed via current account. Energy spike → ToT shock → currency pressure." },
          { t: "METALS EXPORTERS", c: "#22d87a", txt: "AU dominant on copper/iron-ore proxy (β≈+0.5). CA strong on copper/gold (β≈+0.3). BR also gains via iron-ore correlation channel." },
          { t: "AG EXPORTERS", c: "#7ed957", txt: "BR β<sub>soy</sub>=+0.7 (largest exporter), β<sub>corn</sub>=+0.5. US β<sub>soy</sub>=+0.5, β<sub>corn</sub>=+0.4. FR β<sub>wheat</sub>=+0.4 (top EU exporter)." },
          { t: "AG IMPORTERS", c: "#f04d5e", txt: "CN β<sub>soy</sub>=−0.6 (huge feed/crush demand). JP, TR, MENA generally negative on grains. Food inflation passthrough is a risk." },
          { t: "GOLD-LINKED", c: "#f5a623", txt: "AU and CA mining sectors gain (β≈+0.3 to +0.4). CH benefits via refining/banking. Generally a portfolio hedge — moderate β across exporters." },
        ].map(b => (
          <div key={b.t} style={{ background: "var(--bg1)", border: "1px solid var(--border)", padding: "12px 14px" }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: b.c, letterSpacing: "0.12em", marginBottom: 6 }}>{b.t}</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text2)", lineHeight: 1.6 }}
                 dangerouslySetInnerHTML={{ __html: b.txt }} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── MAIN ──────────────────────────────────────────────────────────────────
export default function MacroDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("overview");
  const [ts, setTs] = useState(null);
  const [loadMsg, setLoadMsg] = useState("Connecting to Trading Economics...");

  const LOAD_MSGS = [
    "Connecting to Trading Economics...",
    "Fetching central bank rates...",
    "Pulling inflation & GDP data...",
    "Loading PMI indices...",
    "Assembling bond yields & FX...",
    "Finalising commodities...",
  ];

  async function fetchData() {
    setLoading(true);
    setError(null);
    setData(null);
    let msgIdx = 0;
    setLoadMsg(LOAD_MSGS[0]);
    const interval = setInterval(() => {
      msgIdx = (msgIdx + 1) % LOAD_MSGS.length;
      setLoadMsg(LOAD_MSGS[msgIdx]);
    }, 3500);

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: MODEL,
          max_tokens: 4000,
          tools: [{ type: "web_search_20250305", name: "web_search" }],
          messages: [{ role: "user", content: PROMPT }]
        })
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const result = await res.json();
      clearInterval(interval);

      const text = (result.content || []).filter(b => b.type === "text").map(b => b.text).join("");
      const match = text.match(/\{[\s\S]*\}/);
      if (!match) throw new Error("No JSON in response");

      let parsed;
      try { parsed = JSON.parse(match[0]); }
      catch { parsed = JSON.parse(match[0].replace(/,\s*([}\]])/g, "$1")); }

      setData(parsed);
      setTs(new Date());
    } catch (e) {
      clearInterval(interval);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchData(); }, []);

  const TABS = [
    { id: "overview", label: "OVERVIEW" },
    { id: "ranking", label: "★ RANKING" },
    { id: "tot", label: "TERMS OF TRADE" },
    { id: "rates", label: "CENTRAL BANKS" },
    { id: "inflation", label: "INFLATION / GDP" },
    { id: "pmi", label: "PMI" },
    { id: "bonds", label: "BONDS / FX" },
    { id: "commodities", label: "COMMODITIES" },
  ];

  const statusColor = loading ? "#f5a623" : error ? "#f04d5e" : "#22d87a";

  return (
    <div style={{
      "--bg": "#06080b", "--bg1": "#0d1117", "--bg2": "#141923", "--bg3": "#1c2433",
      "--border": "#1e2a3a", "--border2": "#243040",
      "--amber": "#f5a623", "--amber-dim": "rgba(245,166,35,0.12)",
      "--green": "#22d87a", "--red": "#f04d5e", "--cyan": "#38bdf8",
      "--text": "#e2eaf5", "--text2": "#8899aa", "--text3": "#4a5f72",
      "--mono": "'IBM Plex Mono', monospace", "--sans": "'IBM Plex Sans', sans-serif",
      background: "var(--bg)", color: "var(--text)",
      fontFamily: "var(--sans)", minHeight: "100vh"
    }}>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { height: 4px; width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #243040; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
        @keyframes loadCell { 0%,100%{background:#141923} 50%{background:rgba(245,166,35,.15)} }
      `}</style>

      {/* HEADER */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "11px 24px", background: "var(--bg1)", borderBottom: "1px solid var(--border)",
        position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, background: "var(--amber)",
            clipPath: "polygon(0 0, 100% 0, 100% 60%, 60% 100%, 0 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600, color: "#000"
          }}>SL</div>
          <div>
            <span style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 600, color: "var(--amber)", letterSpacing: "0.08em" }}>SL RESEARCH</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text3)", letterSpacing: "0.06em", marginLeft: 6 }}>// GLOBAL MACRO TERMINAL</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%", background: statusColor,
            boxShadow: `0 0 6px ${statusColor}`,
            animation: (!loading && !error) ? "pulse 2s infinite" : "none"
          }} />
          {ts && !loading && !error && (
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text3)", letterSpacing: "0.04em" }}>
              {ts.toUTCString().substring(0, 25).toUpperCase()} UTC
            </span>
          )}
          <button
            onClick={fetchData}
            disabled={loading}
            style={{
              background: "var(--bg2)", border: "1px solid var(--border2)",
              color: "var(--amber)", fontFamily: "var(--mono)", fontSize: 11,
              padding: "5px 12px", cursor: loading ? "not-allowed" : "pointer",
              letterSpacing: "0.06em", opacity: loading ? 0.4 : 1,
              transition: "all .15s"
            }}
            onMouseEnter={e => { if (!loading) { e.target.style.background = "rgba(245,166,35,.12)"; e.target.style.borderColor = "var(--amber)"; }}}
            onMouseLeave={e => { e.target.style.background = "var(--bg2)"; e.target.style.borderColor = "var(--border2)"; }}
          >↻ REFRESH</button>
        </div>
      </div>

      {/* TABS */}
      <div style={{
        display: "flex", padding: "0 24px", background: "var(--bg1)",
        borderBottom: "1px solid var(--border)", overflowX: "auto"
      }}>
        {TABS.map(t => (
          <div key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              fontFamily: "var(--mono)", fontSize: 11,
              color: tab === t.id ? "var(--amber)" : "var(--text3)",
              padding: "10px 16px", cursor: "pointer",
              borderBottom: `2px solid ${tab === t.id ? "var(--amber)" : "transparent"}`,
              letterSpacing: "0.08em", whiteSpace: "nowrap", transition: "all .15s"
            }}
          >{t.label}</div>
        ))}
      </div>

      {/* CONTENT */}
      <div style={{ padding: "20px 24px", maxWidth: 1400 }}>
        {/* Loading */}
        {loading && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 80, gap: 20 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 3, width: 80 }}>
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} style={{
                  height: 14, background: "var(--bg2)",
                  animation: `loadCell 1.5s ${(i % 3) * 0.3}s infinite`
                }} />
              ))}
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--amber)", letterSpacing: "0.1em", animation: "blink 1.5s infinite" }}>
              FETCHING MACRO DATA
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text3)", letterSpacing: "0.06em" }}>{loadMsg}</div>
          </div>
        )}

        {/* Error */}
        {!loading && error && (
          <div>
            <div style={{
              fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)",
              background: "rgba(240,77,94,.08)", border: "1px solid rgba(240,77,94,.2)",
              padding: "14px 18px", letterSpacing: "0.04em"
            }}>⚠ ERROR: {error}</div>
            <button onClick={fetchData} style={{
              marginTop: 12, background: "var(--bg2)", border: "1px solid var(--border2)",
              color: "var(--amber)", fontFamily: "var(--mono)", fontSize: 11,
              padding: "6px 14px", cursor: "pointer", letterSpacing: "0.06em"
            }}>↻ RETRY</button>
          </div>
        )}

        {/* Dashboard */}
        {!loading && !error && data && (
          <>
            {tab === "overview"     && <OverviewTab    d={data} />}
            {tab === "ranking"      && <RankingTab     d={data} />}
            {tab === "tot"          && <TermsOfTradeTab d={data} />}
            {tab === "rates"        && <RatesTab       d={data} />}
            {tab === "inflation"    && <InflationTab   d={data} />}
            {tab === "pmi"          && <PMITab         d={data} />}
            {tab === "bonds"        && <BondsTab       d={data} />}
            {tab === "commodities"  && <CommoditiesTab d={data} />}
          </>
        )}
      </div>
    </div>
  );
}
