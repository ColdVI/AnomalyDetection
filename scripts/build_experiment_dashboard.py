"""Deney kayit paneli uretici.

artifacts/experiment_registry_adrs.json (docs/decisions.md'den cikarilmis, ADR-008+)
ve artifacts/experiment_registry_adsb.json (ADS-B contextual_physics_v1 hatti, elle
tutulan) dosyalarini birlestirir, docs/experiment_dashboard.html olarak tek dosyalik,
veri gomulmus, filtrelenebilir bir panel uretir.

Yeni bir deney/ADR eklendiginde: ilgili JSON kayit dosyasina bir obje eklenir, bu
script yeniden calistirilir. Agir bir sistem degil -- iki duz JSON dosyasi + bir
statik HTML uretici.

Kullanim:
    python scripts/build_experiment_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REGISTRY_FILES = [
    ROOT / "artifacts" / "experiment_registry_adrs.json",
    ROOT / "artifacts" / "experiment_registry_adsb.json",
]
OUT_PATH = ROOT / "docs" / "experiment_dashboard.html"


def load_registry() -> list[dict]:
    # Ayni ADR id'si iki dosyada da olabilir (ornek: ADR-037..040 hem ADR-genel
    # cikarimda hem elle tutulan ADS-B kaydinda) -- ADS-B dosyasi (REGISTRY_FILES'ta
    # SONRA yuklenen) daha hassas/kaynagindan dogrulanmis oldugu icin ustyazar.
    by_id: dict[str, dict] = {}
    for path in REGISTRY_FILES:
        if not path.exists():
            print(f"  UYARI: {path} bulunamadi, atlaniyor")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {path.name}: {len(data)} kayit")
        for entry in data:
            by_id[entry["id"]] = entry
    entries = list(by_id.values())
    entries.sort(key=lambda e: (e.get("date") or "", e.get("id") or ""), reverse=True)
    return entries


TEMPLATE = """<title>Deney Kayit Paneli</title>
<style>
:root {
  --bg: #f5f3ee;
  --bg-elevated: #ffffff;
  --ink: #1b1f24;
  --ink-dim: #565f68;
  --border: #ddd8cc;
  --accent: #1f7d76;
  --accent-soft: #e2efee;
  --status-accepted: #3f8f5c;
  --status-accepted-soft: #e4f3e8;
  --status-rejected: #b5453c;
  --status-rejected-soft: #f7e6e4;
  --status-mixed: #a3781f;
  --status-mixed-soft: #f3ecd9;
  --status-smoke: #5b6b8c;
  --status-smoke-soft: #e7eaf2;
  --mono: "Cascadia Mono", "Consolas", ui-monospace, "SFMono-Regular", monospace;
  --sans: "Segoe UI", ui-sans-serif, system-ui, -apple-system, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #10141a;
    --bg-elevated: #171c24;
    --ink: #e8e6de;
    --ink-dim: #96a0ab;
    --border: #2a3138;
    --accent: #5ec7bd;
    --accent-soft: #1a2c2c;
    --status-accepted: #6cc78e;
    --status-accepted-soft: #16281d;
    --status-rejected: #e08079;
    --status-rejected-soft: #2c1917;
    --status-mixed: #d9b95e;
    --status-mixed-soft: #2b2413;
    --status-smoke: #93a3c9;
    --status-smoke-soft: #1b2130;
  }
}
:root[data-theme="dark"] {
  --bg: #10141a;
  --bg-elevated: #171c24;
  --ink: #e8e6de;
  --ink-dim: #96a0ab;
  --border: #2a3138;
  --accent: #5ec7bd;
  --accent-soft: #1a2c2c;
  --status-accepted: #6cc78e;
  --status-accepted-soft: #16281d;
  --status-rejected: #e08079;
  --status-rejected-soft: #2c1917;
  --status-mixed: #d9b95e;
  --status-mixed-soft: #2b2413;
  --status-smoke: #93a3c9;
  --status-smoke-soft: #1b2130;
}
:root[data-theme="light"] {
  --bg: #f5f3ee;
  --bg-elevated: #ffffff;
  --ink: #1b1f24;
  --ink-dim: #565f68;
  --border: #ddd8cc;
  --accent: #1f7d76;
  --accent-soft: #e2efee;
  --status-accepted: #3f8f5c;
  --status-accepted-soft: #e4f3e8;
  --status-rejected: #b5453c;
  --status-rejected-soft: #f7e6e4;
  --status-mixed: #a3781f;
  --status-mixed-soft: #f3ecd9;
  --status-smoke: #5b6b8c;
  --status-smoke-soft: #e7eaf2;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.45;
}
.page { max-width: 980px; margin: 0 auto; padding: 28px 20px 80px; }
header.top { display: flex; flex-direction: column; gap: 4px; margin-bottom: 22px; }
header.top h1 { font-size: 1.55rem; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
header.top p { margin: 0; color: var(--ink-dim); font-size: 0.92rem; }

.stats { display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 22px; }
.stat {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 14px; min-width: 108px;
}
.stat .n { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.3rem; font-weight: 600; }
.stat .l { font-size: 0.74rem; color: var(--ink-dim); text-transform: uppercase; letter-spacing: 0.04em; }

.controls {
  position: sticky; top: 0; z-index: 5; background: var(--bg);
  padding: 10px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 18px;
  display: flex; flex-direction: column; gap: 10px;
}
.controls input[type="search"] {
  width: 100%; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-elevated); color: var(--ink); font-size: 0.92rem; font-family: var(--sans);
}
.controls input[type="search"]:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.filter-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.filter-row select {
  padding: 6px 10px; border-radius: 7px; border: 1px solid var(--border);
  background: var(--bg-elevated); color: var(--ink); font-size: 0.84rem; font-family: var(--sans);
}
.filter-row button.reset {
  padding: 6px 10px; border-radius: 7px; border: 1px solid var(--border); background: transparent;
  color: var(--ink-dim); font-size: 0.8rem; cursor: pointer;
}
.filter-row button.reset:hover { color: var(--ink); border-color: var(--accent); }
.count-line { font-size: 0.8rem; color: var(--ink-dim); }

.card {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px; margin-bottom: 12px;
}
.card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 10px; }
.pill {
  font-size: 0.7rem; font-weight: 600; padding: 2px 9px; border-radius: 999px;
  letter-spacing: 0.02em; white-space: nowrap;
}
.pill.accepted { background: var(--status-accepted-soft); color: var(--status-accepted); }
.pill.rejected { background: var(--status-rejected-soft); color: var(--status-rejected); }
.pill.mixed { background: var(--status-mixed-soft); color: var(--status-mixed); }
.pill.smoke { background: var(--status-smoke-soft); color: var(--status-smoke); }
.card-id { font-family: var(--mono); font-size: 0.82rem; color: var(--ink-dim); }
.card-date { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-dim); font-variant-numeric: tabular-nums; }
.card-track { font-weight: 600; font-size: 1rem; flex-basis: 100%; margin-top: 2px; }
.card-sub { font-size: 0.84rem; color: var(--ink-dim); flex-basis: 100%; }
.card-sub b { color: var(--ink); font-weight: 600; }
.card-finding { margin-top: 8px; font-size: 0.92rem; }

.card-details { margin-top: 10px; border-top: 1px dashed var(--border); padding-top: 10px; display: none; }
.card.open .card-details { display: block; }
.card-details h4 { margin: 10px 0 4px; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  font-family: var(--mono); font-size: 0.76rem; background: var(--accent-soft); color: var(--accent);
  padding: 2px 8px; border-radius: 6px;
}
.gate-chip { font-family: var(--sans); font-size: 0.78rem; padding: 2px 9px; border-radius: 6px; border: 1px solid var(--border); }

table.metrics { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 4px; }
table.metrics th, table.metrics td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }
table.metrics th { color: var(--ink-dim); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
table.metrics td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
.table-wrap { overflow-x: auto; }

.notes { font-size: 0.86rem; color: var(--ink-dim); }
.toggle-btn {
  margin-top: 10px; background: none; border: none; color: var(--accent); font-size: 0.82rem;
  cursor: pointer; padding: 0; font-family: var(--sans); font-weight: 600;
}
.toggle-btn:hover { text-decoration: underline; }
.empty { color: var(--ink-dim); font-size: 0.9rem; padding: 30px 0; text-align: center; }
</style>
<div class="page">
  <header class="top">
    <h1>Deney Kayit Paneli</h1>
    <p>ADR gunlugunden ve ADS-B contextual_physics_v1 hattindan cikarilan tum deney/model sonuclari — filtrelenebilir tek liste.</p>
  </header>

  <div class="stats" id="stats"></div>

  <div class="controls">
    <input type="search" id="search" placeholder="Ara: yontem, veri seti, kolon, bulgu..." />
    <div class="filter-row">
      <select id="f-track"><option value="">Tum hatlar</option></select>
      <select id="f-method"><option value="">Tum yontemler</option></select>
      <select id="f-dataset"><option value="">Tum veri setleri</option></select>
      <select id="f-status"><option value="">Tum durumlar</option></select>
      <button class="reset" id="reset">Filtreleri temizle</button>
    </div>
    <div class="count-line" id="count-line"></div>
  </div>

  <div id="list"></div>
</div>

<script>
const DATA = __DATA_JSON__;

const STATUS_LABEL = {
  accepted: "Kabul edildi",
  rejected: "Reddedildi",
  development_accepted_operational_rejected: "Gelistirmede kabul / Operasyonel red",
  smoke_only: "On-deneme",
  analysis_only: "Analiz / on-kayit",
};
const STATUS_CLASS = {
  accepted: "accepted",
  rejected: "rejected",
  development_accepted_operational_rejected: "mixed",
  smoke_only: "smoke",
  analysis_only: "smoke",
};

function statusLabel(s) { return STATUS_LABEL[s] || (s || "Bilinmiyor"); }
function statusClass(s) { return STATUS_CLASS[s] || "smoke"; }

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b, "tr"));
}

function populateSelect(id, values) {
  const sel = document.getElementById(id);
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    sel.appendChild(opt);
  }
}

function renderStats() {
  const total = DATA.length;
  const counts = {};
  for (const e of DATA) {
    const k = e.status || "unknown";
    counts[k] = (counts[k] || 0) + 1;
  }
  const items = [["Toplam kayit", total, null]];
  for (const k of Object.keys(STATUS_LABEL)) {
    if (counts[k]) items.push([statusLabel(k), counts[k], statusClass(k)]);
  }
  const el = document.getElementById("stats");
  el.innerHTML = items.map(([label, n, cls]) => `
    <div class="stat">
      <div class="n" style="${cls ? `color:var(--status-${cls})` : ""}">${n}</div>
      <div class="l">${label}</div>
    </div>`).join("");
}

function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  return typeof n === "number" ? n.toString() : n;
}

function renderMetricsTable(rows) {
  if (!rows || rows.length === 0) return "";
  const hasRho = rows.some(r => r.rho !== undefined && r.rho !== null);
  return `<h4>Sonuc tablosu</h4><div class="table-wrap"><table class="metrics">
    <thead><tr>
      <th>Skor / modul</th><th>Karar politikasi</th><th>Butce / seviye</th>
      <th>Recall</th><th>FA / saat</th>${hasRho ? "<th>rho</th>" : ""}
    </tr></thead>
    <tbody>
      ${rows.map(r => `<tr>
        <td>${r.score_or_module ?? "—"}</td>
        <td>${r.decision_policy ?? "—"}</td>
        <td>${r.budget_tier ?? "—"}</td>
        <td class="num">${fmtNum(r.recall)}</td>
        <td class="num">${fmtNum(r.fa_per_hour)}</td>
        ${hasRho ? `<td class="num">${fmtNum(r.rho)}</td>` : ""}
      </tr>`).join("")}
    </tbody>
  </table></div>`;
}

function renderGates(gates) {
  if (!gates || Object.keys(gates).length === 0) return "";
  const chips = Object.entries(gates).map(([k, v]) => {
    const vs = String(v).toUpperCase();
    let color = "var(--ink-dim)";
    if (vs.includes("PASS") || vs.includes("GEC")) color = "var(--status-accepted)";
    else if (vs.includes("FAIL") || vs.includes("KALD")) color = "var(--status-rejected)";
    return `<span class="gate-chip" style="color:${color};border-color:${color}">${k}: ${v}</span>`;
  }).join(" ");
  return `<h4>Gate sonuclari</h4><div class="chips">${chips}</div>`;
}

function renderCard(e, idx) {
  const cols = (e.columns_or_channels || []).map(c => `<span class="chip">${c}</span>`).join("");
  const metricsHtml = renderMetricsTable(e.metrics_table);
  const gatesHtml = renderGates(e.gate_results);
  const hasDetails = cols || metricsHtml || gatesHtml || e.thresholds_or_targets || e.notes;
  return `
    <div class="card" id="card-${idx}">
      <div class="card-head">
        <span class="pill ${statusClass(e.status)}">${statusLabel(e.status)}</span>
        <span class="card-id">${e.id || ""}</span>
        <span class="card-date">${e.date || ""}</span>
        <div class="card-track">${e.track || ""}</div>
        <div class="card-sub"><b>${e.method || "—"}</b> · ${e.dataset || "—"}</div>
      </div>
      ${e.key_finding ? `<div class="card-finding">${e.key_finding}</div>` : ""}
      ${hasDetails ? `<button class="toggle-btn" onclick="document.getElementById('card-${idx}').classList.toggle('open')">Detaylari goster / gizle</button>` : ""}
      <div class="card-details">
        ${cols ? `<h4>Kolonlar / kanallar</h4><div class="chips">${cols}</div>` : ""}
        ${e.thresholds_or_targets ? `<h4>Esik / hedef</h4><div class="notes">${e.thresholds_or_targets}</div>` : ""}
        ${gatesHtml}
        ${metricsHtml}
        ${e.notes ? `<h4>Notlar</h4><div class="notes">${e.notes}</div>` : ""}
      </div>
    </div>`;
}

function applyFilters() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const track = document.getElementById("f-track").value;
  const method = document.getElementById("f-method").value;
  const dataset = document.getElementById("f-dataset").value;
  const status = document.getElementById("f-status").value;

  const filtered = DATA.filter(e => {
    if (track && e.track !== track) return false;
    if (method && e.method !== method) return false;
    if (dataset && e.dataset !== dataset) return false;
    if (status && e.status !== status) return false;
    if (q) {
      const hay = [
        e.id, e.track, e.method, e.dataset, e.key_finding, e.notes,
        ...(e.columns_or_channels || []),
      ].join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  document.getElementById("list").innerHTML =
    filtered.length ? filtered.map((e, i) => renderCard(e, i)).join("")
                     : `<div class="empty">Bu filtrelerle eslesen kayit yok.</div>`;
  document.getElementById("count-line").textContent =
    `${filtered.length} / ${DATA.length} kayit gosteriliyor`;
}

populateSelect("f-track", uniqueSorted(DATA.map(e => e.track)));
populateSelect("f-method", uniqueSorted(DATA.map(e => e.method)));
populateSelect("f-dataset", uniqueSorted(DATA.map(e => e.dataset)));
populateSelect("f-status", Object.keys(STATUS_LABEL).filter(k => DATA.some(e => e.status === k)));
document.getElementById("f-status").querySelectorAll("option:not(:first-child)").forEach(o => {
  o.textContent = statusLabel(o.value);
});

for (const id of ["f-track", "f-method", "f-dataset", "f-status"]) {
  document.getElementById(id).addEventListener("change", applyFilters);
}
document.getElementById("search").addEventListener("input", applyFilters);
document.getElementById("reset").addEventListener("click", () => {
  document.getElementById("search").value = "";
  for (const id of ["f-track", "f-method", "f-dataset", "f-status"]) {
    document.getElementById(id).value = "";
  }
  applyFilters();
});

renderStats();
applyFilters();
</script>
"""


def main() -> None:
    print("Kayit dosyalari yukleniyor...")
    entries = load_registry()
    print(f"Toplam {len(entries)} kayit birlestirildi.")

    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(entries, ensure_ascii=False))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Panel yazildi: {OUT_PATH}")


if __name__ == "__main__":
    main()
