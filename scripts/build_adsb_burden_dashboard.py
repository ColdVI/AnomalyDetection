"""ADS-B contextual_physics_v1 dogal-yuk gorsel paneli.

ADR-041'de uretilen iki ham rapor dosyasini (LSTM alfa egrileri + CUSUM esik
egrileri) okuyup docs/adsb_contextual_physics_v1_burden_dashboard.html olarak
grafikli, tek-dosyalik bir panel uretir. Sayilar elle kopyalanmiyor -- ham JSON
rapordan okunuyor, bu yuzden yeni bir kosu sonrasi script yeniden calistirmak
yeterli.

Bu panel bir DETECTION/RECALL panosu DEGIL -- yalniz normal veride olculen
dogal yanlis-alarm davranisini gosterir (bkz. sayfadaki durum uyarisi).

Kullanim:
    python scripts/build_adsb_burden_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUDGET_PATH = ROOT / "configs" / "adsb_contextual_physics_v1_alarm_budget.json"
LSTM_REPORT_PATH = ROOT / "artifacts/adsb/runs/20260714_contextual_physics_v1_development_burden_v2/development_burden_curves.json"
CUSUM_REPORT_PATH = ROOT / "artifacts/adsb/runs/20260714_contextual_physics_v1_cusum_burden_v1/cusum_burden_report.json"
OUT_PATH = ROOT / "docs" / "adsb_contextual_physics_v1_burden_dashboard.html"


def load_payload() -> dict:
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    lstm = json.loads(LSTM_REPORT_PATH.read_text(encoding="utf-8"))
    cusum = json.loads(CUSUM_REPORT_PATH.read_text(encoding="utf-8"))

    channel_shares = budget["budget_shares_of_total"]
    pareto_grid = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]

    lstm_series = []
    for channel, cdata in lstm["channels"].items():
        share = channel_shares.get(channel, 0.0)
        target_per_hour_at_1x = share * 1.0 / 100.0
        for profile_name, curve in cdata["profiles"].items():
            points = [
                {
                    "alpha": row["alpha"],
                    "rate": row["alert_episodes_per_scoreable_flight_hour"] or 0.0,
                    "flight_fraction": row["alerted_flight_fraction"] or 0.0,
                    "n_episodes": row["n_alert_episodes"],
                }
                for row in curve
            ]
            lstm_series.append({
                "channel": channel,
                "profile": profile_name,
                "points": points,
                "target_per_hour_at_pareto_1x": target_per_hour_at_1x,
                "channel_budget_share": share,
            })

    cusum_channels = cusum["channels"]
    combined_share = cusum["combined_budget_share"]
    cal_points = [
        {
            "h": row["threshold_h"],
            "rate": row["alert_episodes_per_scoreable_flight_hour"] or 0.0,
            "flight_fraction": row["alerted_flight_fraction"] or 0.0,
            "n_episodes": row["n_alert_episodes"],
        }
        for row in cusum["calibration_curve"]
    ]
    dev_points = [
        {
            "h": row["threshold_h"],
            "rate": row["alert_episodes_per_scoreable_flight_hour"] or 0.0,
            "flight_fraction": row["alerted_flight_fraction"] or 0.0,
            "n_episodes": row["n_alert_episodes"],
        }
        for row in cusum["development_curve"]
    ]
    pareto_markers = [
        {
            "v": float(v),
            "h": h,
            "target_per_hour": combined_share * float(v) / 100.0,
            "n_episodes_at_h_calibration": next(
                r["n_alert_episodes"] for r in cusum["calibration_curve"] if r["threshold_h"] == h
            ),
        }
        for v, h in cusum["derived_h_by_pareto_point"].items()
    ]

    return {
        "provenance": {
            "lstm_calibration_day": lstm["calibration_day"],
            "lstm_development_day": lstm["development_day"],
            "lstm_n_calibration_parts": lstm["n_calibration_parts_used"],
            "lstm_n_development_parts": lstm["n_development_parts_used"],
            "cusum_n_fit_parts": cusum["n_fit_parts_used"],
            "cusum_n_calibration_parts": cusum["n_calibration_parts_used"],
            "cusum_n_development_parts": cusum["n_development_parts_used"],
        },
        "pareto_grid": pareto_grid,
        "channel_shares": channel_shares,
        "lstm_series": lstm_series,
        "cusum": {
            "channels": cusum_channels,
            "combined_budget_share": combined_share,
            "calibration_points": cal_points,
            "development_points": dev_points,
            "pareto_markers": pareto_markers,
        },
    }


TEMPLATE = """<title>contextual_physics_v1 -- Dogal Alarm Yuku Paneli</title>
<style>
:root {
  --bg: #f5f3ee;
  --bg-elevated: #ffffff;
  --ink: #1b1f24;
  --ink-dim: #565f68;
  --border: #ddd8cc;
  --accent: #1f7d76;
  --accent-soft: #e2efee;
  --good: #3f8f5c;
  --good-soft: #e4f3e8;
  --warn: #a3781f;
  --warn-soft: #f3ecd9;
  --grid-line: #e4e0d3;
  --mono: "Cascadia Mono", "Consolas", ui-monospace, "SFMono-Regular", monospace;
  --sans: "Segoe UI", ui-sans-serif, system-ui, -apple-system, sans-serif;
  --c1: #c9634f;
  --c2: #c99a3f;
  --c3: #1f7d76;
  --c4: #5b6bc9;
  --c5: #6b8f4e;
  --cal: #8891a0;
  --dev: #9a4f7d;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #10141a; --bg-elevated: #171c24; --ink: #e8e6de; --ink-dim: #96a0ab;
    --border: #2a3138; --accent: #5ec7bd; --accent-soft: #1a2c2c;
    --good: #6cc78e; --good-soft: #16281d; --warn: #d9b95e; --warn-soft: #2b2413;
    --grid-line: #232a32;
    --c1: #e0897a; --c2: #d9b95e; --c3: #5ec7bd; --c4: #8b97e0; --c5: #93bd78;
    --cal: #7c8798; --dev: #c98cb0;
  }
}
:root[data-theme="dark"] {
  --bg: #10141a; --bg-elevated: #171c24; --ink: #e8e6de; --ink-dim: #96a0ab;
  --border: #2a3138; --accent: #5ec7bd; --accent-soft: #1a2c2c;
  --good: #6cc78e; --good-soft: #16281d; --warn: #d9b95e; --warn-soft: #2b2413;
  --grid-line: #232a32;
  --c1: #e0897a; --c2: #d9b95e; --c3: #5ec7bd; --c4: #8b97e0; --c5: #93bd78;
  --cal: #7c8798; --dev: #c98cb0;
}
:root[data-theme="light"] {
  --bg: #f5f3ee; --bg-elevated: #ffffff; --ink: #1b1f24; --ink-dim: #565f68;
  --border: #ddd8cc; --accent: #1f7d76; --accent-soft: #e2efee;
  --good: #3f8f5c; --good-soft: #e4f3e8; --warn: #a3781f; --warn-soft: #f3ecd9;
  --grid-line: #e4e0d3;
  --c1: #c9634f; --c2: #c99a3f; --c3: #1f7d76; --c4: #5b6bc9; --c5: #6b8f4e;
  --cal: #8891a0; --dev: #9a4f7d;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.45; }
.page { max-width: 1040px; margin: 0 auto; padding: 28px 20px 80px; }
a { color: var(--accent); }
a:hover { text-decoration: none; }
header.top { margin-bottom: 20px; }
header.top h1 { font-size: 1.55rem; margin: 0 0 4px; letter-spacing: -0.01em; text-wrap: balance; }
header.top p { margin: 0; color: var(--ink-dim); font-size: 0.92rem; max-width: 68ch; }

.status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 20px 0 28px; }
@media (max-width: 720px) { .status-grid { grid-template-columns: 1fr; } }
.status-card {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px; border-left: 4px solid var(--good);
}
.status-card.warn { border-left-color: var(--warn); }
.status-card h3 { margin: 0 0 8px; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); }
.status-card ul { margin: 0; padding-left: 18px; font-size: 0.9rem; }
.status-card li { margin-bottom: 6px; }
.status-card li:last-child { margin-bottom: 0; }

section.chart-section {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px 20px 14px; margin-bottom: 20px;
}
section.chart-section h2 { font-size: 1.02rem; margin: 0 0 2px; }
section.chart-section .sub { color: var(--ink-dim); font-size: 0.84rem; margin: 0 0 12px; max-width: 72ch; }
.chart-wrap { overflow-x: auto; }
svg.chart { display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 12px 18px; margin-top: 8px; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 6px; color: var(--ink-dim); }
.legend-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.axis-label { font-size: 0.68rem; fill: var(--ink-dim); font-family: var(--sans); }
.grid-line { stroke: var(--grid-line); stroke-width: 1; }
.series-line { fill: none; stroke-width: 2; }
.series-dot { stroke: var(--bg-elevated); stroke-width: 1.2; }
.ref-line { stroke-dasharray: 4 3; stroke-width: 1.3; opacity: 0.75; }

.callout {
  margin-top: 10px; font-size: 0.82rem; color: var(--ink-dim);
  border-top: 1px dashed var(--border); padding-top: 10px;
}
.callout b { color: var(--ink); }

table.freq { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 6px; }
table.freq th, table.freq td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }
table.freq th { color: var(--ink-dim); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.03em; }
table.freq td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
.swatch-cell { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }

footer.prov { margin-top: 24px; font-size: 0.78rem; color: var(--ink-dim); border-top: 1px solid var(--border); padding-top: 14px; }
footer.prov code { font-family: var(--mono); background: var(--accent-soft); color: var(--accent); padding: 1px 5px; border-radius: 4px; }
</style>
<div class="page">
  <header class="top">
    <h1>contextual_physics_v1 -- Dogal Alarm Yuku</h1>
    <p>Bes fiziksel-sapma sinyalinin, TAMAMEN NORMAL (etiketsiz, olay enjekte edilmemis) ucus
    verisinde esik/hassasiyet degistikce urettigi dogal yanlis-alarm oranini gosterir.
    Tum deney kayitlari icin: <a href="experiment_dashboard.html">deney kayit paneli</a>.</p>
  </header>

  <div class="status-grid" id="status-grid"></div>

  <section class="chart-section" id="lstm-section">
    <h2>Hiz / yon / dikey-hiz -- hassasiyet egrileri</h2>
    <p class="sub">Yatay eksen: esik gevsekligi (alfa, log olcek -- kucuk = siki/az alarm). Dikey
    eksen: normal ucusta saatte kac alarm dogdugu (log olcek). Kesikli cizgiler her kanalin
    Pareto=1.0 (100 saatte 1 alarm) hedefini gosterir.</p>
    <div class="chart-wrap" id="lstm-chart"></div>
    <div class="legend" id="lstm-legend"></div>
  </section>

  <section class="chart-section" id="sensitivity-section">
    <h2>Ayni esikte, sinyaller arasi fark</h2>
    <p class="sub" id="sensitivity-sub"></p>
    <div class="chart-wrap" id="sensitivity-chart"></div>
  </section>

  <section class="chart-section" id="cusum-section">
    <h2>Yatay konum sapmasi (dogu/kuzey, ortak CUSUM) -- kalibrasyon vs gelisim</h2>
    <p class="sub">Iki farkli gunun ayni esik degerlerinde ne kadar farkli davrandigini gosterir --
    farkin buyuklugu, esigin ne kadar guvenilir oldugunun bir olcusudur.</p>
    <div class="chart-wrap" id="cusum-chart"></div>
    <div class="legend" id="cusum-legend"></div>
    <div class="callout" id="cusum-callout"></div>
  </section>

  <footer class="prov" id="provenance"></footer>
</div>

<script>
const DATA = __DATA_JSON__;

const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function logPos(value, domainMin, domainMax, rangeMin, rangeMax) {
  const v = Math.max(value, domainMin);
  const t = (Math.log10(v) - Math.log10(domainMin)) / (Math.log10(domainMax) - Math.log10(domainMin));
  return rangeMin + t * (rangeMax - rangeMin);
}

function niceLogTicks(min, max) {
  const startExp = Math.floor(Math.log10(min));
  const endExp = Math.ceil(Math.log10(max));
  const ticks = [];
  for (let e = startExp; e <= endExp; e++) ticks.push(Math.pow(10, e));
  return ticks;
}

function fmtCompact(v) {
  if (v === 0) return "0";
  if (v >= 1) return v.toFixed(v >= 10 ? 0 : 1);
  return v.toExponential(0).replace("e-0", "e-").replace("e+0", "e");
}

function drawLogLogChart(container, {
  series, xDomain, yDomain, width = 960, height = 320,
  xLabel, yLabel, refLines = [],
}) {
  const margin = { top: 14, right: 24, bottom: 40, left: 54 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const svg = el("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}` }, container);
  const plot = el("g", { transform: `translate(${margin.left},${margin.top})` }, svg);

  const xTicks = niceLogTicks(xDomain[0], xDomain[1]);
  const yTicks = niceLogTicks(yDomain[0], yDomain[1]);

  for (const t of yTicks) {
    const y = plotH - logPos(t, yDomain[0], yDomain[1], 0, plotH);
    el("line", { class: "grid-line", x1: 0, x2: plotW, y1: y, y2: y }, plot);
    const label = el("text", { class: "axis-label", x: -8, y: y + 3, "text-anchor": "end" }, plot);
    label.textContent = fmtCompact(t);
  }
  for (const t of xTicks) {
    const x = logPos(t, xDomain[0], xDomain[1], 0, plotW);
    el("line", { class: "grid-line", x1: x, x2: x, y1: 0, y2: plotH }, plot);
    const label = el("text", { class: "axis-label", x, y: plotH + 16, "text-anchor": "middle" }, plot);
    label.textContent = fmtCompact(t);
  }

  for (const ref of refLines) {
    const y = plotH - logPos(ref.value, yDomain[0], yDomain[1], 0, plotH);
    const line = el("line", {
      class: "ref-line", x1: 0, x2: plotW, y1: y, y2: y, stroke: ref.color,
    }, plot);
    if (ref.label) el("title", {}, line).textContent = ref.label;
  }

  for (const s of series) {
    const pts = s.points.map(p => [
      logPos(p.x, xDomain[0], xDomain[1], 0, plotW),
      plotH - logPos(Math.max(p.y, yDomain[0]), yDomain[0], yDomain[1], 0, plotH),
    ]);
    const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
    el("path", { class: "series-line", d, stroke: s.color, "stroke-dasharray": s.dashed ? "5 3" : "" }, plot);
    pts.forEach((p, i) => {
      const dot = el("circle", {
        class: "series-dot", cx: p[0], cy: p[1], r: 3, fill: s.color,
      }, plot);
      const title = el("title", {}, dot);
      title.textContent = s.points[i].tooltip || "";
    });
  }

  el("text", {
    class: "axis-label", x: plotW / 2, y: plotH + 34, "text-anchor": "middle",
  }, plot).textContent = xLabel;
  const yl = el("text", {
    class: "axis-label", x: -plotH / 2, y: -40, "text-anchor": "middle",
    transform: `rotate(-90)`,
  }, plot);
  yl.textContent = yLabel;
}

function renderStatus() {
  const grid = document.getElementById("status-grid");
  grid.innerHTML = `
    <div class="status-card">
      <h3>Dogrulanan (saglamlasan temel)</h3>
      <ul>
        <li>Model ham sinyal buyuklugune degil gercek zamansal oruntuye tepki veriyor (bagimsiz
        kontrolden gecti) -- Isolation Forest ayni kontrolu gecemedi.</li>
        <li>Normal ucusta yanlis-alarm davranisi artik gercek veriyle olculuyor; 5 sinyalin 4'unde
        hedeflenen en siki butcelerde bile oran dusuk.</li>
        <li>Karar katmani ~150x hizlandi, esdegerligi kanitlandi -- artik daha genis olcekte
        olculebiliyor.</li>
      </ul>
    </div>
    <div class="status-card warn">
      <h3>Henuz olculmedi (tespit basarisi)</h3>
      <ul>
        <li>Gercek anomalileri YAKALAMA orani (recall) bu turlarda HIC test edilmedi -- burada
        gosterilen her sey "normalde ne kadar sik yanlis alarm veriyor" sorusuna cevap.</li>
        <li>Eski kural-bazli sisteme karsi esit-butceli kiyas henuz yapilmadi.</li>
        <li>Yani "anomali tespiti iyilesiyor mu" sorusuna simdi kesin cevap YOK -- temel
        saglamlasiyor, tespit basarisi ayri ve daha sonraki bir olcum.</li>
      </ul>
    </div>
  `;
}

function renderLstm() {
  const colors = { c1: "var(--c1)", c2: "var(--c2)", c3: "var(--c3)", c4: "var(--c4)", c5: "var(--c5)" };
  const palette = [colors.c1, colors.c2, colors.c3, colors.c4, colors.c5];
  const series = DATA.lstm_series.map((s, i) => ({
    color: palette[i % palette.length],
    label: `${s.channel} / ${s.profile}`,
    target: s.target_per_hour_at_pareto_1x,
    points: s.points.map(p => ({
      x: p.alpha, y: p.rate,
      tooltip: `alfa=${p.alpha.toExponential(2)} -> ${p.rate.toFixed(5)} alarm/saat (${p.n_episodes} episode, ucus-orani ${(p.flight_fraction * 100).toFixed(1)}%)`,
    })),
  }));
  const allRates = series.flatMap(s => s.points.map(p => Math.max(p.y, 1e-5)));
  const allAlphas = DATA.lstm_series[0].points.map(p => p.alpha);
  const yMin = Math.pow(10, Math.floor(Math.log10(Math.min(...allRates))));
  const yMax = Math.pow(10, Math.ceil(Math.log10(Math.max(...allRates))));

  drawLogLogChart(document.getElementById("lstm-chart"), {
    series, xDomain: [Math.min(...allAlphas), Math.max(...allAlphas)], yDomain: [yMin, yMax],
    xLabel: "alfa (esik gevsekligi)", yLabel: "alarm / saat",
    refLines: series.map(s => ({ value: s.target, color: s.color, label: `${s.label} hedefi: ${s.target.toFixed(5)}/saat` })),
  });

  document.getElementById("lstm-legend").innerHTML = series.map(s => `
    <div class="legend-item">
      <span class="legend-swatch" style="background:${s.color}"></span>
      ${s.label} <span style="opacity:0.7">(hedef ${s.target.toFixed(5)}/saat)</span>
    </div>`).join("");
}

function renderSensitivity() {
  // Ortak bir alfa noktasinda (izgaranin ortasina yakin) 5 profili karsilastir.
  const fixedAlphaIndex = 3; // ~1.91e-4
  const alpha = DATA.lstm_series[0].points[fixedAlphaIndex].alpha;
  document.getElementById("sensitivity-sub").textContent =
    `Sabit alfa=${alpha.toExponential(2)} secildi; ayni esikte sinyaller arasi oran ne kadar farkli.`;

  const palette = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"];
  const rows = DATA.lstm_series.map((s, i) => ({
    label: `${s.channel} / ${s.profile}`,
    rate: s.points[fixedAlphaIndex].rate,
    color: palette[i % palette.length],
  })).sort((a, b) => b.rate - a.rate);

  const width = 960, height = 46 * rows.length + 20;
  const margin = { top: 10, right: 90, bottom: 10, left: 230 };
  const plotW = width - margin.left - margin.right;
  const container = document.getElementById("sensitivity-chart");
  const svg = el("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}` }, container);
  const maxRate = Math.max(...rows.map(r => r.rate), 1e-6);
  rows.forEach((r, i) => {
    const y = margin.top + i * 46;
    const barW = Math.max(2, (r.rate / maxRate) * plotW);
    el("text", { x: 8, y: y + 24, class: "axis-label", "font-size": "12" }, svg).textContent = r.label;
    el("rect", {
      x: margin.left, y: y + 8, width: barW, height: 20, rx: 4, fill: r.color,
    }, svg);
    el("text", {
      x: margin.left + barW + 8, y: y + 23, class: "axis-label", "font-size": "12",
    }, svg).textContent = `${r.rate.toFixed(5)}/saat`;
  });
}

function renderCusum() {
  const cal = DATA.cusum.calibration_points;
  const dev = DATA.cusum.development_points;
  const mkPoints = (arr) => arr.map(p => ({
    x: p.h, y: p.rate,
    tooltip: `h=${p.h.toFixed(2)} -> ${p.rate.toFixed(6)} alarm/saat (${p.n_episodes} episode, ucus-orani ${(p.flight_fraction * 100).toFixed(2)}%)`,
  }));
  const series = [
    { color: "var(--cal)", label: "Kalibrasyon gunu", dashed: true, points: mkPoints(cal) },
    { color: "var(--dev)", label: "Gelisim gunu (baska gun)", dashed: false, points: mkPoints(dev) },
  ];
  const allH = cal.map(p => p.h);
  const allRates = [...cal, ...dev].map(p => Math.max(p.rate, 1e-4));
  const yMin = Math.pow(10, Math.floor(Math.log10(Math.min(...allRates))));
  const yMax = Math.pow(10, Math.ceil(Math.log10(Math.max(...allRates))));

  drawLogLogChart(document.getElementById("cusum-chart"), {
    series, xDomain: [Math.min(...allH), Math.max(...allH)], yDomain: [yMin, yMax],
    xLabel: "threshold_h (esik -- buyuk = siki/az alarm)", yLabel: "alarm / saat",
  });
  document.getElementById("cusum-legend").innerHTML = series.map(s => `
    <div class="legend-item">
      <span class="legend-swatch" style="background:${s.color}"></span>${s.label}
    </div>`).join("");

  const markers = DATA.cusum.pareto_markers.sort((a, b) => a.v - b.v);
  const rowsHtml = markers.map(m => {
    const unreliable = m.n_episodes_at_h_calibration <= 1;
    return `<tr>
      <td>Pareto V=${m.v}</td>
      <td class="num">${m.h.toFixed(2)}</td>
      <td class="num">${m.target_per_hour.toFixed(5)}</td>
      <td class="num">${m.n_episodes_at_h_calibration}</td>
      <td>${unreliable ? "<b>GUVENILMEZ</b> (tek episode'a dayaniyor)" : "kabul edilebilir destek"}</td>
    </tr>`;
  }).join("");
  document.getElementById("cusum-callout").innerHTML = `
    <b>Pareto hedeflerinden turetilen esikler (yalniz kalibrasyon gununden, gelisim gunune hic
    bakilmadan sabitlendi):</b>
    <table class="freq">
      <thead><tr><th>Hedef</th><th>h</th><th>Hedef/saat</th><th>Kalibrasyonda episode sayisi</th><th>Guvenilirlik</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>`;
}

function renderProvenance() {
  const p = DATA.provenance;
  document.getElementById("provenance").innerHTML = `
    LSTM: kalibrasyon <code>${p.lstm_calibration_day}</code> (${p.lstm_n_calibration_parts} parca),
    gelisim <code>${p.lstm_development_day}</code> (${p.lstm_n_development_parts} parca) &middot;
    CUSUM: fit ${p.cusum_n_fit_parts} parca, kalibrasyon ${p.cusum_n_calibration_parts} parca,
    gelisim ${p.cusum_n_development_parts} parca &middot; ADR-041, tam-hacim degil (216/237
    parcanin bir alt-kumesi).
  `;
}

renderStatus();
renderLstm();
renderSensitivity();
renderCusum();
renderProvenance();
</script>
"""


def main() -> None:
    payload = load_payload()
    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Panel yazildi: {OUT_PATH}")


if __name__ == "__main__":
    main()
