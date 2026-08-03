"""Render the 16 previously-missing final-presentation visuals (round 2).

Scope discipline, same as render_presentation_assets.py: this script only reads
existing reports/markdown for numbers that are already audited in
`09_number_consistency_audit.md`, `01_project_timeline.md` and the v3.1 run
JSONs under artifacts/four_dataset_probabilistic_v31/. It never trains or
scores a model, never changes a threshold, and never touches the sealed final
fault test. Where a panel is illustrative rather than data-driven (e.g. the
observability comparison in NG10), it is labeled "kavramsal gosterim" on the
figure itself so it cannot be mistaken for a real timeline.

Every numeric label here traces to one of:
  - docs/final_presentation/09_number_consistency_audit.md
  - docs/final_presentation/01_project_timeline.md
  - artifacts/four_dataset_probabilistic_v31/runs/*/training_report.json
  - docs/PROJE_SUREC_VE_SONUC.md (UAV GNSS Integrity v1 table)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/final_presentation/presentation_assets/generated_v2"

NAVY = "#16324F"
BLUE = "#2F6BFF"
CYAN = "#27A9C7"
GREEN = "#2E8B57"
AMBER = "#F0A202"
RED = "#D64545"
PURPLE = "#6C5CE7"
INK = "#172033"
MUTED = "#617086"
LIGHT = "#F4F7FB"
GRID = "#D7DFEA"
WHITE = "#FFFFFF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 25,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "svg.fonttype": "none",
        }
    )


def canvas(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.text(0.7, 8.35, title, fontsize=27, weight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(0.72, 7.82, subtitle, fontsize=14, color=MUTED, va="top")
    ax.plot([0.7, 15.3], [7.55, 7.55], color=GRID, lw=1.2)
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=BLUE, fill=WHITE, title_size=16, body_size=13, align="left"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12", linewidth=2, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    if align == "center":
        ax.text(x + w / 2, y + h - 0.30, title, fontsize=title_size, weight="bold", color=color, va="top", ha="center")
        if body:
            ax.text(x + w / 2, y + h - 0.80, body, fontsize=body_size, color=INK, va="top", ha="center", linespacing=1.35)
    else:
        ax.text(x + 0.22, y + h - 0.28, title, fontsize=title_size, weight="bold", color=color, va="top")
        if body:
            ax.text(x + 0.22, y + h - 0.80, body, fontsize=body_size, color=INK, va="top", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, lw=lw, color=color, linestyle=ls))


def footnote(ax, text: str) -> None:
    ax.text(15.3, 0.35, text, fontsize=10.5, color=MUTED, ha="right")


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=180, facecolor=WHITE, bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / f"{stem}.svg", facecolor=WHITE, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def tr(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# NG01 - Slayt 2: Veri yolculugu haritasi
# ---------------------------------------------------------------------------
def render_ng01() -> None:
    fig, ax = canvas("Veri yolculuğu: her durak bir varsayımı sınadı", "Tematik akış — bazı aşamalar zaman içinde örtüşür, kesin takvim değildir")
    nodes = [
        ("OpenSky", "başlangıç\nkeşfi", MUTED, "Dış gözlem\nyeterli değil"),
        ("ALFA", "", BLUE, "Veri çeşitliliği\nkritik"),
        ("UAV\nAttack", "", BLUE, "Etiket ≠\nölçülebilir iz"),
        ("UAV-\nSEAD", "", BLUE, "Ölçek ≠\nbağımsızlık"),
        ("RflyMAD", "", BLUE, "Doğru truth\nzorunlu"),
        ("ADS-B\n(geniş)", "", AMBER, "Sentetik başarı\nyanıltabilir"),
        ("ADS-B\n(dar/fizik)", "", CYAN, "Basit fizik kuralı\ngüçlü kaldı"),
        ("Ortak\nmodel", "", PURPLE, "Yakalama + alarm\nbirlikte okunur"),
    ]
    n = len(nodes)
    x0, x1 = 0.9, 15.1
    xs = np.linspace(x0, x1, n)
    y = 4.6
    ax.plot([x0, x1], [y, y], color=GRID, lw=4, solid_capstyle="round", zorder=1)
    for x, (title, sub, color, lesson) in zip(xs, nodes):
        ax.scatter([x], [y], s=520, color=color, edgecolor=WHITE, linewidth=2.5, zorder=3)
        label = title + ("\n" + sub if sub else "")
        ax.text(x, y + 1.05, label, ha="center", va="bottom", fontsize=12.5, weight="bold", color=NAVY, linespacing=1.1)
        ax.text(x, y - 0.75, lesson, ha="center", va="top", fontsize=10.8, color=color, linespacing=1.25, weight="bold")
    footnote(ax, "Kaynak: 01_project_timeline.md; ADR-001 (2026-06-29)")
    save(fig, "NG01_veri_yolculugu_haritasi")


# ---------------------------------------------------------------------------
# NG02 - Slayt 3: Problem semasi
# ---------------------------------------------------------------------------
def render_ng02() -> None:
    fig, ax = canvas("Aranan şey: gerçek arıza aralığında, düşük yükle bir alarm", "Kavramsal gösterim — gerçek bir uçuşun telemetrisi değildir")
    t = np.linspace(0, 10, 600)
    rng = np.random.default_rng(7)
    channels = [
        ("İrtifa", 5.9, BLUE, 0.35),
        ("Hava hızı", 4.85, CYAN, 0.22),
        ("Motor komutu", 3.8, PURPLE, 0.30),
    ]
    fault_lo, fault_hi = 6.2, 8.0
    x0, xw = 2.5, 12.0
    ax.axvspan(x0 + fault_lo / 10 * xw, x0 + fault_hi / 10 * xw, ymin=0.28, ymax=0.78, color="#F6C5C5", alpha=0.55, zorder=0)
    for name, base, color, amp in channels:
        sig = base + amp * np.sin(t * 1.3 + rng.uniform(0, 5)) * 0.4
        sig += np.where(t > fault_lo, amp * 1.8 * (1 - np.exp(-(t - fault_lo))), 0.0) * (1 if color != CYAN else -1)
        sig += rng.normal(0, amp * 0.06, size=t.shape)
        x_plot = x0 + t / 10 * xw
        ax.plot(x_plot, sig, color=color, lw=2.2)
        ax.text(x0 - 0.15, base, name, fontsize=12.5, color=color, ha="right", va="center", weight="bold")
    ax.text(x0 + (fault_lo + fault_hi) / 2 / 10 * xw, 6.65, "gerçek arıza aralığı", ha="center", fontsize=12.5, color="#B23B3B", weight="bold")
    ax.plot([x0, x0 + xw], [2.75, 2.75], color=GRID, lw=1)
    ax.text(x0, 2.55, "0", fontsize=11, color=MUTED)
    ax.text(x0 + xw, 2.55, "zaman", fontsize=11, color=MUTED, ha="right")

    box(ax, 0.9, 0.55, 6.6, 1.55, "İstenen", "Alarm, arıza aralığının içinde başlar", color=GREEN, fill="#EEF8F2", title_size=15, body_size=12.5)
    box(ax, 8.0, 0.55, 6.6, 1.55, "İstenmeyen", "Arıza aralığı dışında (normal uçuşta) alarm", color=RED, fill="#FCEEEE", title_size=15, body_size=12.5)
    footnote(ax, "Kaynak: docs/final_rapor_ml_fizibilite_2026-07-16.md")
    save(fig, "NG02_problem_semasi")


# ---------------------------------------------------------------------------
# NG03 - Slayt 6: Normal-only egitim semasi
# ---------------------------------------------------------------------------
def render_ng03() -> None:
    fig, ax = canvas("Yarı-denetimli çerçeve: yalnız normal uçuşla eğitim", "Etiketler yalnız eşik kalibrasyonu ve değerlendirmede kullanılır")
    box(ax, 2.0, 5.35, 12.0, 1.85, "Normal uçuşlar", "Model eğitimi ve iç doğrulama\nyalnız bu havuzdan öğrenir", color=GREEN, fill="#EEF8F2", title_size=19, body_size=14, align="center")
    ax.plot([8.0, 8.0], [5.3, 4.15], color=GREEN, lw=3)
    arrow(ax, 8.0, 4.7, 8.0, 4.2, color=GREEN, lw=3)
    box(ax, 4.5, 2.35, 7.0, 1.55, "Model", "Yalnız normal davranışı öğrenir", color=NAVY, fill=LIGHT, title_size=18, body_size=13.5, align="center")

    box(ax, 2.0, 0.55, 12.0, 1.35, "Arızalı uçuşlar", "Eğitime hiç girmez — yalnız eşik kalibrasyonu ve nihai değerlendirmede kullanılır", color=AMBER, fill="#FFF7E8", title_size=17, body_size=12.5, align="center")
    arrow(ax, 8.0, 2.3, 8.0, 1.95, color=AMBER, lw=2.4, ls="--")

    ax.plot([1.6, 14.4], [3.95, 3.95], color=RED, lw=1.6, ls=":")
    ax.text(8.0, 4.02, "Etiket buradan geçmez", ha="center", fontsize=13, color="#B23B3B", weight="bold")

    ax.text(0.85, 0.15, "İstisna: LightGBM ve TCN karşılaştırmaları, açıkça ayrı işaretlenmiş denetimli kollardı; ana hat normal-only kaldı.", fontsize=11.3, color=MUTED)
    save(fig, "NG03_normal_only_egitim_semasi")


# ---------------------------------------------------------------------------
# NG04 - Slayt 8: Gozlem katmani karsilastirmasi
# ---------------------------------------------------------------------------
def render_ng04() -> None:
    fig, ax = canvas("Dışarıdan gözlem ile araç içi telemetri farklı sorulara cevap verir", "OpenSky yerine adsb.lol + ALFA/UAV Attack kararı — 2026-06-29")
    left_items = ["Konum (lat/lon)", "Barometrik irtifa", "Yer hızı ve yön", "Dikey hız", "Sinyal varlığı / kaybı"]
    right_items = ["Motor komutu ve aktüatör çıkışı", "EKF innovation / red bayrakları", "GPS kalite göstergeleri", "Tutum (attitude) setpoint'i", "Kontrol yüzeyi geri beslemesi"]
    box(ax, 0.8, 1.3, 6.9, 5.9, "ADS-B / hava trafiği yayını", "", color=BLUE, fill=LIGHT, title_size=19)
    box(ax, 8.4, 1.3, 6.9, 5.9, "Araç içi telemetri", "", color=PURPLE, fill=LIGHT, title_size=19)
    for i, item in enumerate(left_items):
        ax.text(1.15, 6.15 - i * 0.72, f"•  {item}", fontsize=14, color=INK, va="center")
    for i, item in enumerate(right_items):
        ax.text(8.75, 6.15 - i * 0.72, f"•  {item}", fontsize=14, color=INK, va="center")
    ax.plot([7.65, 7.65], [1.5, 6.9], color=GRID, lw=2, ls="--")
    note_x, note_y, note_w, note_h = 1.4, 0.35, 13.2, 1.0
    ax.add_patch(FancyBboxPatch((note_x, note_y), note_w, note_h, boxstyle="round,pad=0.025,rounding_size=0.12", linewidth=2, edgecolor=NAVY, facecolor="#EEF2FA"))
    ax.text(note_x + note_w / 2, note_y + note_h / 2, "Bu projede doğrulanmış arıza başlangıcı yalnız araç içi etiketli veri setlerinde mevcuttu",
            ha="center", va="center", fontsize=14.5, color=NAVY, weight="bold")
    save(fig, "NG04_gozlem_katmani_karsilastirmasi")


# ---------------------------------------------------------------------------
# NG05 - Slayt 12: Monolitik vs moduler (tek sayi hatasi duzeltildi)
# ---------------------------------------------------------------------------
def render_ng05() -> None:
    fig, ax = canvas("Tek modelden fiziksel modüllere geçiş", "Aynı özellik havuzu, farklı mimari — kanıt veri setine göre ayrı verilir")
    box(ax, 0.8, 4.55, 6.6, 2.6, "Monolitik", "Tüm özellikler → tek Isolation\nForest modeli", color=MUTED, fill=LIGHT, title_size=18, body_size=13.5)
    ax.text(1.1, 4.35, "ALFA: rastgele sıralamaya yakın", fontsize=12.5, color=RED, weight="bold")
    ax.text(1.1, 3.95, "UAV Attack: uçuş ROC-AUC 0,21", fontsize=12.5, color=RED, weight="bold")

    mods = ["Kontrol-\ntepki", "Rehberlik", "Navigasyon", "Sinyal\nkalitesi"]
    colors = [BLUE, CYAN, GREEN, AMBER]
    for i, (m, c) in enumerate(zip(mods, colors)):
        box(ax, 8.5 + i * 1.72, 5.6, 1.55, 1.55, m, "", color=c, fill=WHITE, title_size=11.5, align="center")
        arrow(ax, 8.5 + i * 1.72 + 0.775, 5.55, 11.5, 4.55, color=c, lw=1.4)
    box(ax, 10.6, 3.55, 3.2, 1.0, "Ortak ölçekte\nbirleştirme", "", color=NAVY, fill=LIGHT, title_size=13, align="center")
    arrow(ax, 12.2, 3.5, 12.2, 2.55, color=NAVY, lw=2.4)
    box(ax, 9.9, 1.15, 4.6, 1.4, "ALFA uçuş ROC-AUC", "0,833  (en güçlü modül — rehberlik: 0,864)", color=GREEN, fill="#EEF8F2", title_size=14, body_size=15, align="center")
    ax.text(0.85, 1.35, "Kazanım modelden değil, güçlü-ama-dar bir\nsinyalin alakasız özellikler içinde\nseyrelmesini önlemekten geldi.", fontsize=12.5, color=NAVY, linespacing=1.4)
    footnote(ax, "Kaynak: 03_method_inventory.md")
    save(fig, "NG05_monolitik_vs_moduler")


# ---------------------------------------------------------------------------
# NG06 - Slayt 13: Veri cesitliligi etkisi (ALFA 0.918 karisikligi duzeltildi)
# ---------------------------------------------------------------------------
def render_ng06() -> None:
    fig, ax = canvas("Model aynı kaldı, veri çeşitliliği arttı", "ALFA LSTM autoencoder — ham kayıtlardan çıkarılan 5 ek normal uçuş")
    ax2 = fig.add_axes([0.09, 0.30, 0.44, 0.46])
    bars = ax2.bar(["10 normal uçuş", "15 normal uçuş"], [0.731, 0.918], color=[AMBER, GREEN], width=0.55)
    ax2.axhline(0.833, color=BLUE, lw=2, ls="--")
    ax2.text(0, 0.855, "Modüler Isolation Forest: 0,833", color=BLUE, fontsize=11, ha="center", va="bottom")
    for b, v in zip(bars, [0.731, 0.918]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, tr(v), ha="center", fontsize=15, weight="bold", color=INK)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Uçuş ROC-AUC (nedensellik düzeltmesinden önce)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GRID, lw=0.8)

    box(ax, 9.6, 3.9, 5.6, 3.55, "Bu sayılar henüz final değil", "0,731 ve 0,918, nedensellik düzeltmesinden\nÖNCEKİ (non-causal) uçuş ROC-AUC'sidir.\n\nDüzeltme sonrası ALFA'nın en güçlü tekil\nsinyali cross-track error oldu: 0,751\n(ayrı slaytta).", color=RED, fill="#FCEEEE", title_size=15, body_size=13)
    ax.text(0.85, 1.35, "Mesaj: küçük veri havuzunda bir derin modelin zayıf çıkması,\nher zaman modelin kendi suçu değildir.", fontsize=13, color=NAVY, linespacing=1.4)
    footnote(ax, "Kaynak: 04_metric_evolution.md; 05_failure_and_fix_inventory.md")
    save(fig, "NG06_veri_cesitliligi_etkisi")


# ---------------------------------------------------------------------------
# NG07 - Slayt 14: Bozulma taksonomisi
# ---------------------------------------------------------------------------
def render_ng07() -> None:
    fig, ax = canvas("Altı kontrollü sentetik bozulma tipi", "Yalnız değerlendirmede kullanıldı — eğitime hiç girmedi")
    cells = [
        ("Freeze", "Sensör değerini sabitleme", "Zayıf yakalama", RED),
        ("Bias", "Sabit kayma", "Zayıf yakalama", RED),
        ("Drift", "Giderek büyüyen sapma", "CUSUM ~%75 yakalama, ~0,28 sn gecikme", GREEN),
        ("Noise", "Gürültü", "Bu turda ayrı raporlanmadı", MUTED),
        ("GPS ramp", "Yavaş, sinsi konum kayması (2 m/s)", "Zayıf yakalama", RED),
        ("Dropout", "Veri kaybı", "Bu turda ayrı raporlanmadı", MUTED),
    ]
    t = np.linspace(0, 1, 120)
    cw, ch = 4.85, 2.55
    x0, y0 = 0.75, 4.35
    for i, (name, desc, catch, color) in enumerate(cells):
        col, row = i % 3, i // 3
        x = x0 + col * (cw + 0.25)
        y = y0 - row * (ch + 0.35)
        patch = FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0.02,rounding_size=0.1", linewidth=1.8, edgecolor=GRID, facecolor=WHITE)
        ax.add_patch(patch)
        spark_x = x + 0.25 + t * (cw - 0.5)
        clean = y + ch - 0.95 + 0.14 * np.sin(t * 12)
        ax.plot(spark_x, clean, color=GRID, lw=1.6)
        rng = np.random.default_rng(i)
        if name == "Freeze":
            pert = np.where(t > 0.5, clean[np.searchsorted(t, 0.5)], clean)
        elif name == "Bias":
            pert = clean + np.where(t > 0.5, 0.35, 0.0)
        elif name == "Drift":
            pert = clean + np.where(t > 0.4, (t - 0.4) * 0.9, 0.0)
        elif name == "Noise":
            pert = clean + rng.normal(0, 0.07, size=t.shape)
        elif name == "GPS ramp":
            pert = clean + np.where(t > 0.3, (t - 0.3) * 0.28, 0.0)
        else:
            pert = np.where((t > 0.45) & (t < 0.7), np.nan, clean)
        ax.plot(spark_x, pert, color=color, lw=2.1)
        ax.text(x + 0.25, y + ch - 0.32, name, fontsize=15, weight="bold", color=NAVY)
        ax.text(x + 0.25, y + 0.62, desc, fontsize=11, color=MUTED, linespacing=1.25)
        ax.text(x + 0.25, y + 0.22, catch, fontsize=11.5, color=color, weight="bold")
    footnote(ax, "Kaynak: 03_method_inventory.md; 05_failure_and_fix_inventory.md")
    save(fig, "NG07_bozulma_taksonomisi")


# ---------------------------------------------------------------------------
# NG08 - Slayt 15: Nedensellik duzeltmesi
# ---------------------------------------------------------------------------
def render_ng08() -> None:
    fig, ax = canvas("Birinci kırılma: geleceğe bakma hatası", "CUSUM taban değerleri yalnız geçmişten hesaplanacak şekilde düzeltildi")
    ax.text(0.9, 6.85, "Düzeltme öncesi", fontsize=17, weight="bold", color=RED)
    ax.plot([3.3, 14.6], [6.5, 6.5], color=GRID, lw=10, solid_capstyle="butt")
    ax.plot([3.3, 14.6], [6.5, 6.5], color=RED, lw=10, alpha=0.35, solid_capstyle="butt")
    arrow(ax, 14.6, 6.5, 15.0, 6.9, color=RED, lw=2)
    ax.text(9.9, 6.05, "uçuşun tamamından hesaplanan taban — geleceğe bakıyor", color="#B23B3B", fontsize=12.5, ha="center")

    ax.text(0.9, 4.85, "Düzeltme sonrası", fontsize=17, weight="bold", color=GREEN)
    ax.plot([3.3, 14.6], [4.5, 4.5], color=GRID, lw=10, solid_capstyle="butt")
    for i in range(1, 12):
        xx = 3.3 + (i / 12) * 11.3
        arrow(ax, xx, 4.15, xx, 4.42, color=GREEN, lw=1.3)
    ax.text(9.9, 4.0, "her an yalnız geçmiş ve mevcut bilgi kullanılıyor (causal)", color=GREEN, fontsize=12.5, ha="center")

    box(ax, 0.9, 1.9, 6.3, 1.55, "alt_error_cusum uçuş ROC-AUC", "0,878  →  0,611", color=RED, fill="#FCEEEE", title_size=14, body_size=22, align="center")
    box(ax, 8.7, 1.9, 6.4, 1.55, "Yeni en güçlü tekil sinyal", "cross-track error: 0,751", color=GREEN, fill="#EEF8F2", title_size=14, body_size=20, align="center")
    ax.text(8.0, 1.15, "Bu bir performans kaybı değil, bir dürüstleşmedir.", ha="center", fontsize=17, color=NAVY, weight="bold")
    footnote(ax, "Kaynak: 05_failure_and_fix_inventory.md; 09_number_consistency_audit.md")
    save(fig, "NG08_nedensellik_duzeltmesi")


# ---------------------------------------------------------------------------
# NG09 - Slayt 16: Cakisma vs gercek baslangic
# ---------------------------------------------------------------------------
def render_ng09() -> None:
    fig, ax = canvas("İkinci kırılma: alarm arızaya gerçekten tepki verdi mi?", "Çakışma (overlap) yeterli değil — yeni alarm başlangıcı arızadan sonra olmalı")
    fx0, fx1 = 7.2, 11.6
    for row_y, label in [(5.6, "Senaryo A — alarm arızadan önce açılmış")]:
        ax.axvspan(fx0, fx1, ymin=(row_y - 0.55) / 9, ymax=(row_y + 0.55) / 9, color="#F6C5C5", alpha=0.55)
        ax.plot([1.0, 14.6], [row_y, row_y], color=GRID, lw=1)
        ax.plot([2.3, 9.3], [row_y, row_y], color=AMBER, lw=9, solid_capstyle="butt")
        ax.text(1.0, row_y + 0.55, label, fontsize=13, color=NAVY, weight="bold")
        ax.text(fx0, row_y - 0.75, "eski kural: YAKALANDI  •  yeni kural: YAKALANMADI", fontsize=11.5, color="#B23B3B", ha="left")
    row_y = 2.9
    ax.axvspan(fx0, fx1, ymin=(row_y - 0.55) / 9, ymax=(row_y + 0.55) / 9, color="#F6C5C5", alpha=0.55)
    ax.plot([1.0, 14.6], [row_y, row_y], color=GRID, lw=1)
    ax.plot([8.1, 13.4], [row_y, row_y], color=GREEN, lw=9, solid_capstyle="butt")
    ax.text(1.0, row_y + 0.55, "Senaryo B — alarm arıza aralığı içinde başlıyor", fontsize=13, color=NAVY, weight="bold")
    ax.text(fx0, row_y - 0.75, "her iki kuralda da: YAKALANDI", fontsize=11.5, color=GREEN, ha="left")

    box(ax, 9.4, 0.4, 5.6, 1.15, "Olay yakalama oranı", "Çakışma %59,4  →  gerçek başlangıç %19,4–22,4", color=RED, fill="#FCEEEE", title_size=13, body_size=16, align="center")
    footnote(ax, "Kaynak: 05_failure_and_fix_inventory.md")
    save(fig, "NG09_cakisma_vs_gercek_baslangic")


# ---------------------------------------------------------------------------
# NG10 - Slayt 17: Gozlenebilirlik karsilastirmasi (kavramsal)
# ---------------------------------------------------------------------------
def render_ng10() -> None:
    fig, ax = canvas("Etiketin varlığı, ölçülebilir sinyalin varlığını garanti etmez", "Kavramsal gösterim — gerçek zaman serisi değildir")
    t = np.linspace(0, 1, 300)
    ax1 = fig.add_axes([0.08, 0.20, 0.40, 0.55])
    ax2 = fig.add_axes([0.55, 0.20, 0.40, 0.55])
    for a in (ax1, ax2):
        a.set_xticks([])
        a.spines[["top", "right"]].set_visible(False)
        a.axvspan(0.35, 1.0, color="#F6C5C5", alpha=0.45)
    rng = np.random.default_rng(3)
    spoof = 0.02 * np.sin(t * 30) + np.where(t > 0.35, (t - 0.35) * 1.6, 0.0)
    ax1.plot(t, spoof, color=RED, lw=2.4)
    ax1.set_title("GPS spoofing sırasında\ngps_speed_residual", fontsize=13, color=RED)
    ax1.set_ylabel("residual", fontsize=12)

    flat = 0.02 * np.sin(t * 30) + rng.normal(0, 0.01, size=t.shape)
    ax2.plot(t, flat, color=MUTED, lw=2.4)
    ax2.set_title("Ping DoS sırasında\naynı kanal", fontsize=13, color=MUTED)

    ax.text(8.0, 1.35, "6 Ping DoS kaydının 4'ünde, saldırı etiketine rağmen\nseçilen telemetri kanallarında ölçülebilir değişim yok.",
            ha="center", fontsize=13.5, color=NAVY, weight="bold", linespacing=1.4)
    footnote(ax, "Kaynak: docs/final_rapor_ml_fizibilite_2026-07-16.md")
    save(fig, "NG10_gozlenebilirlik_karsilastirmasi")


# ---------------------------------------------------------------------------
# NG11 - Slayt 19: Yontem turu ilerlemesi
# ---------------------------------------------------------------------------
def render_ng11() -> None:
    fig, ax = canvas("Yakalama arttı; alarm yükü daha hızlı arttı", "UAV-SEAD mekanik anomali ailesi — dokuz yöntem denendi")
    ax2 = fig.add_axes([0.15, 0.32, 0.42, 0.46])
    labels = ["Başlangıç\n(EKF/PX4)", "Chronos\n(zero-shot)", "Tek özellik\n(itki komutu)"]
    vals = [0.205, 0.390, 0.459]
    colors = [MUTED, CYAN, GREEN]
    bars = ax2.barh(labels, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax2.text(v + 0.012, b.get_y() + b.get_height() / 2, tr(v), va="center", fontsize=14, weight="bold", color=INK)
    ax2.set_xlim(0, 0.62)
    ax2.set_xlabel("Mekanik anomali yakalama oranı")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", color=GRID, lw=0.8)

    box(ax, 9.6, 4.9, 5.6, 1.7, "Tek özellik dedektörünün bedeli", "38,1 yanlış alarm / saat", color=RED, fill="#FCEEEE", title_size=14, body_size=20, align="center")
    box(ax, 9.6, 2.95, 5.6, 1.65, "İki kanallı füzyon", "Yakalama arttı, yanlış alarm\n1,89–3,83 kat büyüdü", color=AMBER, fill="#FFF7E8", title_size=14, body_size=13.5, align="center")
    box(ax, 9.6, 1.15, 5.6, 1.5, "EKF innovation (tek başına)", "ROC-AUC 0,354 — ters yönlü sinyal", color=MUTED, fill=LIGHT, title_size=14, body_size=13.5, align="center")
    footnote(ax, "Kaynak: 03_method_inventory.md")
    save(fig, "NG11_yontem_turu_ilerlemesi")


# ---------------------------------------------------------------------------
# NG12 - Slayt 23: Sentetik basari tuzagi
# ---------------------------------------------------------------------------
def render_ng12() -> None:
    fig, ax = canvas("Sentetik başarı, doğal trafikte alarm tuzağına dönüştü", "İlk geniş ADS-B denemesi — reddedildi ve arşivlendi (10 Temmuz sıfırlanması)")
    box(ax, 0.9, 3.4, 6.6, 3.2, "Sentetik enjeksiyon yakalama", "%97,6", color=GREEN, fill="#EEF8F2", title_size=17, body_size=46, align="center")
    box(ax, 8.5, 3.4, 6.6, 3.2, "Doğal trafikte alarm yükü", "25,54 / saat", color=RED, fill="#FCEEEE", title_size=17, body_size=40, align="center")
    arrow(ax, 7.5, 5.0, 8.45, 5.0, color=NAVY, lw=3)
    ax.text(8.0, 5.35, "aynı sistem", ha="center", fontsize=12.5, color=MUTED)
    ax.text(8.0, 2.5, "İki sayı birlikte okunmadan hiçbir anomali sonucu değerlendirilemez.", ha="center", fontsize=15.5, color=NAVY, weight="bold")
    box(ax, 3.4, 0.55, 9.2, 1.35, "Sonuç", "Yaklaşım reddedildi; eski hat ve iki reddedilmiş ADS-B denemesi\narchive/ altına alındı, temiz bir hat yeniden açıldı.", color=NAVY, fill=LIGHT, title_size=14, body_size=12.5, align="center")
    footnote(ax, "Kaynak: AGENTS.md; 05_failure_and_fix_inventory.md")
    save(fig, "NG12_sentetik_basari_tuzagi")


# ---------------------------------------------------------------------------
# NG13 - Komut->tepki residual (Slayt 25'in bolunmesiyle yeni slayt)
# ---------------------------------------------------------------------------
def render_ng13() -> None:
    fig, ax = canvas("Komut→tepki residual: anomaliyi sapmada değil, tahmin hatasında ara", "r = y − ŷ(kontrol komutları, bağlam)")
    # Top row: command+context -> model -> expected response
    box(ax, 0.6, 5.85, 3.55, 1.55, "Kontrol komutları\n+ bağlam", "", color=BLUE, fill=LIGHT, title_size=13.5, align="center")
    arrow(ax, 4.15, 6.62, 5.15, 6.62, color=BLUE, lw=2.2)
    box(ax, 5.2, 5.85, 3.9, 1.55, "Öğrenilmiş normal\nuçuş dinamiği modeli", "", color=PURPLE, fill=LIGHT, title_size=13.5, align="center")
    arrow(ax, 9.1, 6.62, 10.1, 6.62, color=PURPLE, lw=2.2)
    box(ax, 10.15, 5.85, 3.0, 1.55, "Beklenen\ntepki ŷ", "", color=PURPLE, fill=LIGHT, title_size=13.5, align="center")

    # Second row: measured response, feeding down into residual together with ŷ
    box(ax, 10.15, 3.85, 3.0, 1.45, "Ölçülen\ntepki y", "", color=CYAN, fill=LIGHT, title_size=13.5, align="center")
    arrow(ax, 11.65, 5.8, 11.65, 5.35, color=PURPLE, lw=2.0)
    arrow(ax, 11.65, 3.8, 11.65, 3.35, color=CYAN, lw=2.0)
    box(ax, 9.9, 1.85, 3.5, 1.45, "Residual", "r = y − ŷ", color=NAVY, fill="#EEF2FA", title_size=14, body_size=16, align="center")

    ax.text(0.6, 3.55, "Kalibrasyon açığı", fontsize=13.5, weight="bold", color=NAVY)
    ax2 = fig.add_axes([0.045, 0.035, 0.28, 0.29])
    names = ["ALFA", "RflyMAD"]
    need = [11.8, 5.1]
    bars = ax2.bar(names, need, color=[AMBER, AMBER], width=0.5)
    ax2.axhline(1.0, color=GREEN, lw=2)
    ax2.text(1.35, 1.35, "mevcut maruziyet = 1×", color=GREEN, fontsize=9.5, ha="right")
    for b, v in zip(bars, need):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{tr(v,1)}×", ha="center", fontsize=12.5, weight="bold", color=INK)
    ax2.set_ylabel("gereken / mevcut\nnormal uçuş süresi", fontsize=9.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GRID, lw=0.8)

    box(ax, 5.6, 0.35, 8.4, 3.35, "Sinyal gerçekten var — ama kalibre edilemedi", "Arıza öncesi/sonrası residual dağılımları eşikten\nbağımsız olarak ayrıştı. Ama güvenilir yanlış-alarm\neşiği kurmak için bağımsız normal uçuş süresi\nyetersiz kaldı (grafik solda): gereken maruziyetin\nALFA'da yaklaşık 1/11,8'i, RflyMAD'de yaklaşık 1/5,1'i\nmevcuttu.", color=RED, fill="#FCEEEE", title_size=15, body_size=13)
    footnote(ax, "Kaynak: docs/final_rapor_ml_fizibilite_2026-07-16.md (RESIDUAL-V1)")
    save(fig, "NG13_komut_tepki_residual")


# ---------------------------------------------------------------------------
# NG14 - Slayt 28: NLL bileseni ayristirmasi
# ---------------------------------------------------------------------------
def render_ng14() -> None:
    fig, ax = canvas("Skor iki şeyi aynı anda cezalandırır", "Aynı tahmin hatası, farklı model güveninde farklı anomali skoru üretir")
    ax2 = fig.add_axes([0.08, 0.16, 0.40, 0.62])
    z = np.linspace(-3, 3, 300)
    for sigma, color, lbl in [(0.3, BLUE, "model emin (σ=0,3)"), (1.5, AMBER, "model belirsiz (σ=1,5)")]:
        nll = 0.5 * z**2 + math.log(sigma)
        ax2.plot(z, nll, color=color, lw=3, label=lbl)
    ax2.axhline(0, color=GRID, lw=1)
    ax2.axvline(1.4, color=RED, lw=1.6, ls="--")
    ax2.text(1.45, 4.4, "aynı hata (z=1,4)", color=RED, fontsize=11, rotation=90, va="top")
    ax2.set_xlabel("Standartlaştırılmış hata (z)")
    ax2.set_ylabel("NLL")
    ax2.legend(loc="upper center", frameon=False, fontsize=11.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.25)

    box(ax, 9.5, 5.2, 5.7, 1.7, "Model emin (düşük σ)", "Aynı hata → YÜKSEK NLL\n(daha anomalik sayılır)", color=RED, fill="#FCEEEE", title_size=15, body_size=14, align="center")
    box(ax, 9.5, 3.1, 5.7, 1.7, "Model belirsiz (yüksek σ)", "Aynı hata → DÜŞÜK NLL\n(daha az anomalik sayılır)", color=GREEN, fill="#EEF8F2", title_size=15, body_size=14, align="center")
    box(ax, 9.5, 1.1, 5.7, 1.55, "NLL = 0,5·((y−μ)/σ)²  +  log σ", "hata terimi (kare)      belirsizlik cezası", color=NAVY, fill=LIGHT, title_size=13.5, body_size=12, align="center")
    footnote(ax, "Kaynak: scripts/four_dataset_probabilistic_gpu_v1_runner.py")
    save(fig, "NG14_nll_bilesen_ayristirmasi")


# ---------------------------------------------------------------------------
# NG15 - GNSS butunluk pilotu (Slayt 25'in bolunmesiyle yeni slayt)
# ---------------------------------------------------------------------------
def render_ng15() -> None:
    fig, ax = canvas("GNSS bütünlük pilotu: üç yöntem, aynı disiplinli sonuç", "Geliştirme + prova rolleri ayrı tutuldu (16 Temmuz 2026)")
    ax2 = fig.add_axes([0.08, 0.30, 0.52, 0.48])
    methods = ["Uçuş kontrol\ngöstergeleri\n(PX4-native)", "Çok-kanallı\nCUSUM", "Bağlamsal\nLSTM"]
    recall = [58.8, 0.0, 47.1]
    burden = [19.58, 0.0, 20.32]
    x = np.arange(3)
    w = 0.32
    ax2b = ax2.twinx()
    b1 = ax2.bar(x - w / 2, recall, width=w, color=BLUE, label="Yakalama (%)")
    b2 = ax2b.bar(x + w / 2, burden, width=w, color=AMBER, label="Alarm / uçuş-saati")
    ax2.axhline(y=(2 / (2 + 20.32)) * 0, color=GRID, lw=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10.5)
    ax2.set_ylabel("Yakalama (%)", color=BLUE)
    ax2b.set_ylabel("Alarm / uçuş-saati", color=AMBER)
    ax2.set_ylim(0, 100)
    ax2b.set_ylim(0, 25)
    for xi, v in zip(x - w / 2, recall):
        ax2.text(xi, v + 1.5, f"%{tr(v,1)}", ha="center", fontsize=10.5, color=BLUE, weight="bold")
    for xi, v in zip(x + w / 2, burden):
        ax2b.text(xi, v + 0.5, tr(v, 2), ha="center", fontsize=10.5, color="#8a5c00", weight="bold")
    ax2.spines[["top"]].set_visible(False)
    ax2b.spines[["top"]].set_visible(False)
    ax2.set_title("Geliştirme — kritik sözleşme (bütçe: ≤2 olay/uçuş-saati)", fontsize=12.5, color=NAVY)

    box(ax, 9.6, 4.7, 5.6, 2.0, "Prova (rehearsal) LSTM", "%90 yakalama / 0 alarm-saati\n— ama geliştirmeye taşınmadı", color=AMBER, fill="#FFF7E8", title_size=14.5, body_size=13.5, align="center")
    box(ax, 9.6, 2.5, 5.6, 1.95, "Karar", "Hiçbir yöntem, kritik + danışma\nbütçesini birlikte tutturamadı.\nHedef karşılanmadı.", color=RED, fill="#FCEEEE", title_size=14.5, body_size=13, align="center")
    ax.text(0.85, 1.45, "Bu iki disiplinli çalışma da açık biçimde 'hedef karşılanmadı' ile kapandı;\nikisinde de sinyal var ama operasyonel eşik kurulamadı ayrımı belgelendi.", fontsize=11.8, color=MUTED, linespacing=1.4)
    footnote(ax, "Kaynak: docs/PROJE_SUREC_VE_SONUC.md (UAV GNSS Integrity v1, 2026-07-16)")
    save(fig, "NG15_gnss_pilotu_karsilastirmasi")


# ---------------------------------------------------------------------------
# NG16 - ALFA v3.1 group-CV sonucu (eksik kronolojik halka)
# ---------------------------------------------------------------------------
def render_ng16() -> None:
    fig, ax = canvas("ALFA'nın son sözü: grup-güvenli çapraz doğrulama", "Dört bağımsız oturum-ailesi grubu; yalnız biri kontrol eşiğini geçti")
    folds = [
        ("Grup 0", 0.444, 0.0, 0.0, True),
        ("Grup 1", 0.409, 54.5, 50.0, False),
        ("Grup 2", 0.545, 18.2, 25.0, False),
        ("Grup 3", 0.521, 100.0, 83.3, False),
    ]
    cw = 3.55
    for i, (name, auc, det, alarm, passed) in enumerate(folds):
        x = 0.75 + i * (cw + 0.25)
        gate_color = GREEN if passed else RED
        gate_text = "Kontrol eşiği: GEÇTİ" if passed else "Kontrol eşiği: KALDI"
        box(ax, x, 3.55, cw, 3.55, name, "", color=NAVY, fill=LIGHT, title_size=16, align="center")
        ax.text(x + cw / 2, 6.35, f"Uçuş ROC-AUC {tr(auc)}", ha="center", fontsize=13.5, color=INK, weight="bold")
        ax.text(x + cw / 2, 5.75, f"Yakalama  %{tr(det,1)}", ha="center", fontsize=12.5, color=INK)
        ax.text(x + cw / 2, 5.30, f"Normal alarm  %{tr(alarm,1)}", ha="center", fontsize=12.5, color=INK)
        ax.text(x + cw / 2, 4.55, gate_text, ha="center", fontsize=12.5, color=gate_color, weight="bold")
        if passed:
            ax.text(x + cw / 2, 4.05, "(ama 0 yakalama / 0 alarm — ayrım yok)", ha="center", fontsize=10, color=MUTED)
    box(ax, 2.4, 1.05, 11.2, 1.65, "Sonuç", "Dört gruptan yalnız biri genlik-baskınlığı kontrol eşiğini geçti; o grupta da\nmodel hiçbir ayrım yapmadı. Diğer üç grup kontrol eşiğinde kaldı. ALFA'da da\nnihai bilimsel karar: hedef karşılanmadı.", color=RED, fill="#FCEEEE", title_size=15, body_size=13, align="center")
    footnote(ax, "Kaynak: artifacts/four_dataset_probabilistic_v31/runs/alfa_fold_*/training_report.json")
    save(fig, "NG16_alfa_v31_group_cv_sonucu")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    render_ng01()
    render_ng02()
    render_ng03()
    render_ng04()
    render_ng05()
    render_ng06()
    render_ng07()
    render_ng08()
    render_ng09()
    render_ng10()
    render_ng11()
    render_ng12()
    render_ng13()
    render_ng14()
    render_ng15()
    render_ng16()
    print("generated=16 png+16 svg ->", OUT)


if __name__ == "__main__":
    main()
