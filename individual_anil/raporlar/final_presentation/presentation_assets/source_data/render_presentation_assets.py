"""Render the audited final-presentation visual package without model execution.

This script reads only existing reports/artifacts, copies selected plots, and
creates deterministic explanatory graphics. It never trains/scores a model,
changes a threshold, or accesses the sealed final-fault test.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "docs/final_presentation/presentation_assets"
EXISTING = PACKAGE / "existing"
GENERATED = PACKAGE / "generated"
SOURCE_DATA = PACKAGE / "source_data"

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


EXISTING_ASSETS = {
    "S05_rfly_data_composition.png": "artifacts/rfly_full/v2/visuals/01_data_composition.png",
    "S07_adsb_route_artifact.png": "docs/assets/adsb_simple_anomaly/04_route_summary.png",
    "S09_method_tradeoff.png": "docs/sunum_hafta5_gorseller/rflymad_frozen_ae_tcn_hedefli_karsilastirma.png",
    "S12_adsb_progression.png": "artifacts/adsb/plots/reporting_summary/adsb_research_progression.png",
    "S14_four_dataset_detection_summary.png": "artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/02_detection_summary.png",
    "S17_training_checkpoint.png": "artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/01_training_and_selected_checkpoint.png",
    "S19_any_window_matrix.png": "artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/03_flight_flag_matrix.png",
    "S20_event_recall_false_rate.png": "artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/01_recall_vs_false_event_rate.png",
    "S20_normal_alarm_burden.png": "artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/03_normal_test_alarm_burden.png",
    "S20_detected_motor_timeline.png": "artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/05_detected_motor_event_timeline.png",
    "S20_missed_real_sensor_timeline.png": "artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/06_missed_real_sensor_timeline.png",
}


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
    ax.text(0.7, 8.35, title, fontsize=29, weight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(0.72, 7.82, subtitle, fontsize=14.5, color=MUTED, va="top")
    ax.plot([0.7, 15.3], [7.55, 7.55], color=GRID, lw=1.2)
    return fig, ax


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    color: str = BLUE,
    fill: str = WHITE,
    title_size: float = 18,
    body_size: float = 13.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
        linewidth=2, edgecolor=color, facecolor=fill
    )
    ax.add_patch(patch)
    ax.text(x + 0.22, y + h - 0.28, title, fontsize=title_size, weight="bold", color=color, va="top")
    if body:
        ax.text(x + 0.22, y + h - 0.82, body, fontsize=body_size, color=INK, va="top", linespacing=1.35)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=2, color=color))


def save(fig: plt.Figure, stem: str) -> None:
    png = GENERATED / f"{stem}.png"
    svg = GENERATED / f"{stem}.svg"
    fig.savefig(png, dpi=180, facecolor=WHITE, bbox_inches=None, pad_inches=0)
    fig.savefig(svg, facecolor=WHITE, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def copy_existing() -> None:
    EXISTING.mkdir(parents=True, exist_ok=True)
    for target, source in EXISTING_ASSETS.items():
        src = ROOT / source
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, EXISTING / target)


def render_s01() -> None:
    fig, ax = canvas("Anomali tespitinde üç farklı ölçüm düzeyi", "Bir düzeydeki başarı diğerinin yerine geçmez")
    items = [
        ("1  Pencere", "Her zaman adımında\nmodel skoru", BLUE, "Sıralama / eşik aşımı"),
        ("2  Olay", "Ardışık alarmların\ntek olaya dönüştürülmesi", PURPLE, "Olay yakalama + yanlış olay/saat"),
        ("3  Uçuş", "Uçuş boyunca skorların\nözetlenmesi", CYAN, "Uçuş önceliklendirme"),
    ]
    xs = [0.9, 5.65, 10.4]
    for x, (title, body, color, footer) in zip(xs, items):
        box(ax, x, 3.2, 4.0, 3.25, title, body, color=color, fill=LIGHT)
        ax.text(x + 2.0, 3.58, footer, ha="center", fontsize=13, weight="bold", color=color)
    arrow(ax, 4.95, 4.8, 5.55, 4.8)
    arrow(ax, 9.7, 4.8, 10.3, 4.8)
    ax.text(8, 1.55, "Ana karar metriği", ha="center", color=MUTED, fontsize=14)
    ax.text(8, 0.95, "Gerçek olay yakalama  +  yanlış olay / normal saat", ha="center", color=NAVY, fontsize=22, weight="bold")
    save(fig, "S01_window_event_flight_concept")


def render_s03() -> None:
    fig, ax = canvas("Proje ilerlemesi: üç paralel öğrenme hattı", "Modelden önce ölçüm sözleşmesi olgunlaştı")
    lanes = [
        ("Veri ve gerçeklik", 6.35, GREEN, ["Şema denetimi", "Olay aralığı düzeltmesi", "2.712 uçuşluk parser düzeltmesi", "Kapalı nihai test"]),
        ("Model", 4.35, BLUE, ["IF / residual", "AE · TCN · Chronos", "Gaussian sonraki-adım modeli", "30. epoch seçimi"]),
        ("Değerlendirme", 2.35, PURPLE, ["Pencere / uçuş AUC", "Oturum bazlı bölünme", "Gerçek olay değerlendirmesi", "Hedef karşılanmadı"]),
    ]
    x_positions = [3.65, 7.15, 10.65, 14.15]
    for label, y, color, labels in lanes:
        ax.text(0.7, y, label, fontsize=16, weight="bold", color=color, va="center")
        ax.plot([3.1, 14.45], [y, y], color=GRID, lw=4, solid_capstyle="round")
        for x, text in zip(x_positions, labels):
            ax.scatter([x], [y], s=185, color=color, edgecolor=WHITE, linewidth=2, zorder=3)
            ax.text(x, y - 0.48, text, ha="center", va="top", fontsize=11.5, color=INK, wrap=True)
    ax.text(3.65, 1.15, "İlk turlar", ha="center", color=MUTED, fontsize=13)
    ax.text(7.15, 1.15, "Nedensellik ve bağımsızlık", ha="center", color=MUTED, fontsize=13)
    ax.text(10.65, 1.15, "Ortak olasılıksal yaklaşım", ha="center", color=MUTED, fontsize=13)
    ax.text(14.15, 1.15, "Son değerlendirme", ha="center", color=MUTED, fontsize=13)
    save(fig, "S03_project_timeline_three_lanes")


def render_s04() -> None:
    fig, ax = canvas("Veri setlerinin bilimsel fizibilitesi", "Yeşil: güçlü  •  Sarı: sınırlı  •  Kırmızı: yok / kritik eksik")
    datasets = ["ALFA", "UAV Attack", "UAV-SEAD", "RflyMAD", "ADS-B"]
    criteria = ["Olay aralığı", "Normal çeşitlilik", "Bağımsız grup", "Fiziksel gözlenebilirlik", "Doğal etiket"]
    # 2=strong, 1=limited, 0=missing. Audit-derived qualitative coding.
    values = np.array([
        [1, 0, 1, 2, 2],
        [0, 0, 0, 0, 2],
        [2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2],
        [0, 2, 2, 1, 0],
    ]).T
    colors = {0: "#F6C5C5", 1: "#FFE6A6", 2: "#BFE4CE"}
    labels = {0: "Yok", 1: "Sınırlı", 2: "Güçlü"}
    x0, y0, cw, ch = 4.25, 1.5, 2.1, 1.05
    for j, name in enumerate(datasets):
        ax.text(x0 + j * cw + cw / 2, 7.18, name, ha="center", va="center", fontsize=15, weight="bold", color=NAVY)
    for i, criterion in enumerate(criteria):
        y = y0 + (len(criteria) - 1 - i) * ch
        ax.text(3.95, y + ch / 2, criterion, ha="right", va="center", fontsize=14, color=INK)
        for j in range(len(datasets)):
            val = int(values[i, j])
            ax.add_patch(Rectangle((x0 + j * cw, y), cw - 0.05, ch - 0.05, facecolor=colors[val], edgecolor=WHITE, lw=2))
            ax.text(x0 + j * cw + (cw - 0.05) / 2, y + (ch - 0.05) / 2, labels[val], ha="center", va="center", fontsize=13, weight="bold", color=INK)
    ax.text(0.8, 0.68, "Not: ADS-B gerçek trafik içerir; doğrulanmış doğal anomali etiketi yoktur.", fontsize=13, color=MUTED)
    save(fig, "S04_dataset_feasibility_matrix")


def render_s06() -> None:
    fig, ax = canvas("Metrik merdiveni", "Her basamak farklı bir soruyu yanıtlar")
    steps = [
        ("1", "Gaussian NLL", "Normal davranıştan\nne kadar sapıldı?", BLUE),
        ("2", "Pencere sıralaması", "Skor anomalileri\nöne taşıyor mu?", CYAN),
        ("3", "Olay politikası", "Aşan pencereler\nnasıl tek alarma dönüşür?", PURPLE),
        ("4", "Operasyonel çift", "Olay yakalama  +\nyanlış olay / normal saat", RED),
    ]
    for i, (num, title, body, color) in enumerate(steps):
        x = 0.75 + i * 3.82
        y = 2.0 + i * 0.95
        box(ax, x, y, 3.25, 2.15, f"{num}  {title}", body, color=color, fill=LIGHT, title_size=16, body_size=12.5)
        if i < 3:
            arrow(ax, x + 3.25, y + 1.05, x + 3.72, y + 1.05 + 0.95)
    ax.text(0.85, 1.05, "AUC yüksek olabilir; bu, doğru zamanda az sayıda alarm üretildiğini göstermez.", fontsize=16, color=NAVY, weight="bold")
    save(fig, "S06_metric_ladder")


def render_s08() -> None:
    fig, ax = canvas("Arıza başlangıcı düzeltmesi", "Şematik gösterim — gerçek bir uçuş zaman serisi değildir")
    ax.text(1.0, 6.55, "Düzeltme öncesi", fontsize=18, weight="bold", color=RED)
    ax.plot([3.2, 14.5], [6.2, 6.2], color=GRID, lw=4)
    ax.plot([3.2, 14.5], [6.2, 6.2], color=RED, lw=10, solid_capstyle="butt")
    ax.text(3.2, 5.75, "t = 0: arıza aktif görünüyor", color=RED, fontsize=14, ha="left")
    ax.text(1.0, 3.95, "Düzeltme sonrası", fontsize=18, weight="bold", color=GREEN)
    ax.plot([3.2, 14.5], [3.6, 3.6], color=GREEN, alpha=0.35, lw=10, solid_capstyle="butt")
    onset = 10.2
    ax.plot([onset, 14.5], [3.6, 3.6], color=RED, lw=10, solid_capstyle="butt")
    ax.axvline(onset, ymin=0.34, ymax=0.49, color=NAVY, lw=2, ls="--")
    ax.text(onset, 3.05, "Gerçek başlangıç", color=NAVY, fontsize=14, ha="center")
    box(ax, 1.0, 1.15, 5.6, 1.25, "2.712 uçuş etkilendi", "6.605 uçuşluk havuz yeniden parse edildi", color=PURPLE, fill=LIGHT, title_size=20, body_size=13)
    ax.text(8.0, 1.55, "Sonuç: düzeltme öncesi olay-zamanı metrikleri kullanılmaz.", fontsize=17, color=NAVY, weight="bold")
    save(fig, "S08_truth_parser_before_after")


def render_s10(split: dict) -> None:
    fig, ax = canvas("Benzer uçuş gruplarını ayıran veri bölünmesi", "Aynı senaryo / oturum grubu birden fazla role geçmez")
    roles = [
        ("Normal eğitim", "train", GREEN),
        ("Normal doğrulama", "val", BLUE),
        ("Bağımsız normal test", "normal_test", CYAN),
        ("Anomali geliştirme", "anomaly_dev", PURPLE),
        ("Kapalı tutulan\nnihai test verisi", "final_fault_test", RED),
    ]
    for i, (label, key, color) in enumerate(roles):
        data = split["rflymad"]["roles"][key]
        x = 0.55 + i * 3.08
        box(ax, x, 2.25, 2.75, 4.35, label, f"{data['source_count']} uçuş\n{data['group_count']} bağımsız grup", color=color, fill=LIGHT, title_size=15, body_size=17)
        ax.text(x + 1.375, 2.62, "Rol sınırı", fontsize=11.5, color=MUTED, ha="center")
        if i < 4:
            ax.plot([x + 2.9, x + 2.9], [2.05, 6.8], color=GRID, lw=2, ls="--")
    ax.text(8, 1.1, "Kaynak kesişimi: 0   •   Grup kesişimi: 0", ha="center", fontsize=21, color=NAVY, weight="bold")
    save(fig, "S10_group_independent_split")


def render_s11(report: dict) -> None:
    fig, ax = canvas("Eğitilmiş–rastgele model kontrolü", "Nokta skorları saklanmadığı için scatter değil, doğrulanmış özet korelasyon kartıdır")
    rfly = report["magnitude_diagnostic"]
    cards = [
        ("UAV-SEAD", 0.964, 0.965, RED, "Çok yüksek benzerlik\nGenlik kestirmesi"),
        ("RflyMAD son değerlendirme", rfly["spearman_trained_score_vs_random_init_score"], rfly["spearman_trained_score_vs_standardized_target_magnitude"], GREEN, "Kontrol eşiğinin altında\nBu tek başına başarı değildir"),
    ]
    for i, (name, random_rho, magnitude_rho, color, verdict) in enumerate(cards):
        x = 0.95 + i * 7.45
        box(ax, x, 2.05, 6.65, 4.85, name, "", color=color, fill=LIGHT, title_size=21)
        ax.text(x + 0.45, 5.55, "Eğitilmiş ↔ rastgele", fontsize=13, color=MUTED)
        ax.text(x + 6.15, 5.48, f"ρ = {random_rho:.4f}" if i else f"ρ ≈ {random_rho:.3f}", fontsize=22, ha="right", color=color, weight="bold")
        ax.text(x + 0.45, 4.65, "Eğitilmiş ↔ hedef genliği", fontsize=13, color=MUTED)
        ax.text(x + 6.15, 4.58, f"ρ = {magnitude_rho:.4f}" if i else f"ρ ≈ {magnitude_rho:.3f}", fontsize=22, ha="right", color=color, weight="bold")
        ax.plot([x + 0.45, x + 6.2], [4.05, 4.05], color=GRID, lw=1)
        ax.text(x + 3.32, 3.15, verdict, ha="center", va="center", fontsize=17, color=NAVY, weight="bold", linespacing=1.4)
    ax.text(8, 1.25, "Kontrol eşiği: ρ = 0,80", ha="center", fontsize=16, color=MUTED)
    save(fig, "S11_trained_random_magnitude")


def render_s13() -> None:
    fig, ax = canvas("Olasılıksal yaklaşımın değerlendirme dönüşümü", "Aynı model ailesi, daha sıkı bilimsel sözleşme")
    left = ["Kaynak içi veri bölünmesi", "Tüm anomali uçuşu pozitif", "Uçuşta tek eşik aşımı", "Uçuş sıralama metriği"]
    right = ["Benzer uçuş grupları ayrı", "Gerçek arıza aralıkları", "Olay politikası + bekleme süresi", "Olay yakalama + yanlış olay/saat"]
    box(ax, 0.9, 1.5, 6.0, 5.65, "Önceki değerlendirme", "", color=AMBER, fill=LIGHT, title_size=21)
    box(ax, 9.1, 1.5, 6.0, 5.65, "Son değerlendirme", "", color=GREEN, fill=LIGHT, title_size=21)
    for i, (old, new) in enumerate(zip(left, right)):
        y = 5.85 - i * 1.05
        ax.text(1.25, y, f"•  {old}", fontsize=15, color=INK, va="center")
        ax.text(9.45, y, f"•  {new}", fontsize=15, color=INK, va="center")
        arrow(ax, 7.1, y, 8.85, y, color=GREEN)
    ax.text(8, 0.85, "Daha düşük görünen sonuç, daha güvenilir genelleme ölçümü olabilir.", ha="center", fontsize=18, color=NAVY, weight="bold")
    save(fig, "S13_probabilistic_transition")


def render_s16() -> None:
    fig, ax = canvas("Normal davranışı olasılıksal olarak öğrenen mimari", "Yalnız normal eğitim verisi • 32 adımlık nedensel geçmiş • 24 aktif kanal")
    steps = [
        ("Geçmiş pencere", "32 adım\nölçekli değer + eksik veri maskesi", BLUE, 0.55, 3.0),
        ("LSTM", "64 gizli birim\nnormal dinamik", PURPLE, 4.05, 3.0),
        ("Sonraki adım", "Kanal başına\nortalama μ + belirsizlik σ", CYAN, 7.35, 3.0),
        ("Gaussian NLL", "Tahmin hatası +\nmodel belirsizliği", AMBER, 10.65, 3.0),
        ("Olay katmanı", "Kalıcılık / K-of-N /\nCUSUM", RED, 13.35, 3.0),
    ]
    widths = [2.9, 2.6, 2.75, 2.25, 2.1]
    for i, ((title, body, color, x, y), w) in enumerate(zip(steps, widths)):
        box(ax, x, y, w, 2.8, title, body, color=color, fill=LIGHT, title_size=15.5, body_size=12)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.05, 4.4, steps[i + 1][3] - 0.1, 4.4)
    ax.text(8, 1.45, "Çıktı: pencere skoru değil, birleştirilmiş olaylar değerlendirilir.", ha="center", fontsize=19, color=NAVY, weight="bold")
    save(fig, "S16_gaussian_forecaster_architecture")


def render_s17_nll() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.09, 0.16, 0.84, 0.68])
    z = np.linspace(-3.0, 3.0, 601)
    curves = [(0.2, BLUE), (1.0, PURPLE), (2.0, RED)]
    for sigma, color in curves:
        nll = 0.5 * z**2 + math.log(sigma)
        ax.plot(z, nll, color=color, lw=3, label=f"σ = {sigma:.1f}")
    ax.axhspan(-2.0, 0.0, color="#DDECF4", alpha=0.75, zorder=0)
    ax.axhline(0, color=NAVY, lw=1.4)
    ax.axvline(0, color=GRID, lw=1)
    ax.text(-2.85, -1.72, "Negatif NLL bölgesi", color=NAVY, fontsize=15, weight="bold")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 6.2)
    ax.set_xlabel("Standartlaştırılmış tahmin hatası  z")
    ax.set_ylabel("NLL = 0.5 · z² + log(σ)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Negatif Gaussian NLL neden mümkündür?", x=0.09, y=0.965, ha="left", fontsize=29, weight="bold", color=NAVY)
    fig.text(0.09, 0.91, "σ < 1 ve tahmin hatası küçükken log(σ) terimi toplamı sıfırın altına indirebilir.", fontsize=15, color=MUTED)
    fig.text(0.91, 0.055, "Deterministik matematik görseli • deney değildir", ha="right", fontsize=12.5, color=MUTED)
    save(fig, "S17_negative_gaussian_nll")


def render_s18(report: dict) -> None:
    fig, ax = canvas("Checkpoint ve öğrenme kontrolü özeti", "Normal doğrulama verisiyle seçildi; anomali sonucundan bağımsız")
    diag = report["magnitude_diagnostic"]
    final = report["final_epoch"]
    cards = [
        ("Seçilen checkpoint", "30. epoch", BLUE),
        ("Eğitim NLL", f"{final['mean_train_gaussian_nll']:.3f}".replace(".", ","), CYAN),
        ("Doğrulama NLL", f"{final['mean_validation_gaussian_nll']:.3f}".replace(".", ","), PURPLE),
        ("Eğitilmiş ↔ rastgele", f"ρ = {diag['spearman_trained_score_vs_random_init_score']:.3f}".replace(".", ","), GREEN),
        ("Eğitilmiş ↔ genlik", f"ρ = {diag['spearman_trained_score_vs_standardized_target_magnitude']:.3f}".replace(".", ","), GREEN),
    ]
    for i, (title, value, color) in enumerate(cards):
        x = 0.6 + i * 3.08
        box(ax, x, 3.25, 2.75, 3.1, title, "", color=color, fill=LIGHT, title_size=14)
        ax.text(x + 1.375, 4.35, value, ha="center", va="center", fontsize=25, weight="bold", color=color)
    box(ax, 4.25, 1.15, 7.5, 1.25, "Kontrol sonucu", "Her iki korelasyon da ρ=0,80 eşiğinin altında; belirgin genlik kestirmesi işaretlenmedi.", color=GREEN, fill="#EEF8F2", title_size=17, body_size=13)
    save(fig, "S18_checkpoint_and_magnitude_summary")


def render_s20() -> None:
    fig, ax = canvas("Gerçek olay değerlendirmesi: final bilimsel hüküm", "Geliştirme ve bağımsız normal test sonucu • nihai test verisi açılmadı")
    box(ax, 0.8, 4.3, 4.4, 2.45, "Olay yakalama", "%43,27", color=BLUE, fill=LIGHT, title_size=17, body_size=30)
    ax.text(1.05, 4.72, "Referans olay politikasında", fontsize=12.5, color=MUTED)
    box(ax, 5.8, 4.3, 4.4, 2.45, "Yanlış olay / normal saat", "0,654", color=AMBER, fill=LIGHT, title_size=17, body_size=30)
    ax.text(6.05, 4.72, "Bağımsız normal testte", fontsize=12.5, color=MUTED)
    box(ax, 10.8, 4.3, 4.4, 2.45, "Kör noktalar", "Real  %7,81\nSensor  %5,45", color=RED, fill=LIGHT, title_size=17, body_size=21)
    box(ax, 0.8, 1.6, 9.4, 1.75, "Taramanın en yüksek yakalaması", "%58,71  @  9,153 yanlış olay / saat", color=PURPLE, fill=LIGHT, title_size=16, body_size=22)
    box(ax, 10.8, 1.6, 4.4, 1.75, "Karar", "Hedef karşılanmadı", color=RED, fill="#FCEEEE", title_size=16, body_size=20)
    ax.text(8, 0.75, "Kapalı tutulan nihai test verisi açılmadı.", ha="center", fontsize=16, color=NAVY, weight="bold")
    save(fig, "S20_final_result_card")


def write_source_data(split: dict, report: dict) -> None:
    values = {
        "provenance": {
            "timeline": "docs/final_presentation/01_project_timeline.md",
            "datasets": "docs/final_presentation/02_dataset_inventory.md",
            "metrics": "docs/final_presentation/04_metric_evolution.md",
            "failures": "docs/final_presentation/05_failure_and_fix_inventory.md",
            "v31": "docs/final_presentation/08_probabilistic_v31_deep_dive.md",
            "numbers": "docs/final_presentation/09_number_consistency_audit.md",
        },
        "split_roles": split["rflymad"]["roles"],
        "checkpoint": report["selected_checkpoint"],
        "final_epoch": report["final_epoch"],
        "magnitude_diagnostic": report["magnitude_diagnostic"],
        "final_event_result": {
            "event_recall": 0.4327,
            "false_events_per_normal_hour": 0.654,
            "real_recall": 0.0781,
            "sensor_recall": 0.0545,
            "max_sweep_recall": 0.5871,
            "max_sweep_false_events_per_hour": 9.153,
            "final_fault_test_opened": False,
            "source": "docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md",
        },
        "scatter_status": {
            "status": "BLOCKED",
            "reason": "Trained/random/magnitude point scores were not persisted; only summary Spearman correlations exist.",
        },
    }
    (SOURCE_DATA / "visual_values.json").write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    with (SOURCE_DATA / "nll_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z", "sigma_0_2", "sigma_1_0", "sigma_2_0"])
        for z in np.linspace(-3.0, 3.0, 601):
            writer.writerow([z, 0.5 * z**2 + math.log(0.2), 0.5 * z**2, 0.5 * z**2 + math.log(2.0)])


def main() -> None:
    configure_style()
    GENERATED.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    copy_existing()
    split = json.loads((ROOT / "artifacts/four_dataset_probabilistic_v31/split_report.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json").read_text(encoding="utf-8"))
    render_s01()
    render_s03()
    render_s04()
    render_s06()
    render_s08()
    render_s10(split)
    render_s11(report)
    render_s13()
    render_s16()
    render_s17_nll()
    render_s18(report)
    render_s20()
    write_source_data(split, report)
    print(f"copied={len(EXISTING_ASSETS)} generated=12 png+12 svg")


if __name__ == "__main__":
    main()
