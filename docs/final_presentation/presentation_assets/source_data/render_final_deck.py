"""Render every figure of the final 35-slide deck (white theme, minimal palette).

Discipline, same as render_presentation_assets.py / render_new_visuals_v2.py:
this script never trains a model, never scores data, never changes a threshold
and never touches the sealed final fault test. Every number below is copied
from an audited source:

  - docs/final_presentation/09_number_consistency_audit.md
  - docs/final_presentation/04_metric_evolution.md
  - docs/final_presentation/01_project_timeline.md
  - docs/final_presentation/02_dataset_inventory.md
  - docs/final_presentation/03_method_inventory.md
  - docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md
  - docs/decisions.md
  - artifacts/four_dataset_probabilistic_v31/runs/*/training_report.json

Panels that are schematic rather than data-driven carry a visible
"kavramsal gösterim" footnote so they cannot be misread as a real timeline.

Output: docs/final_presentation/presentation_assets/final_deck/*.png + *.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/final_presentation/presentation_assets/final_deck"

# --- palette -----------------------------------------------------------------
# White surface, colored marks. Categorical slots and status colors are the
# validated defaults from the data-viz reference palette; validated on #ffffff
# with the ported six-checks validator:
#   3 slots, all-pairs : CVD dE 9.2, normal-vision 24.0  -> PASS
#   6 slots, adjacent  : CVD dE 9.1, normal-vision 22.9  -> PASS
# Aqua (2.82:1) and yellow (2.17:1) are sub-3:1 on white, so the relief rule
# applies: every mark in those hues carries a visible direct label. It does —
# bar values are printed on the mark, never left to the axis.
BLUE = "#2A78D6"       # slot 1 — current / valid result
ORANGE = "#EB6834"     # slot 2 — cost, alarm burden, "before"
AQUA = "#1BAF7A"       # slot 3 — corrected / improvement / passing
YELLOW = "#EDA100"     # attention
VIOLET = "#4A3AA7"     # secondary category
RED = "#D03B3B"        # status critical — failure, invalid, rejected
GREEN = "#0CA30C"      # status good

NAVY = "#1C5CAB"       # blue-550 — headings (5.4:1 on white)
LIGHTNAVY = "#DCE6F1"  # blue-100 tint fill
LIGHTAQUA = "#DCF3EA"
LIGHTORANGE = "#FCE7DF"
LIGHTRED = "#FBE4E4"
INK = "#0B0B0B"        # primary text
MUTED = "#52514E"      # secondary text
GREY = "#898781"       # superseded / invalid values
LIGHTGREY = "#E8E8E8"
GRID = "#E1E0D9"
WHITE = "#FFFFFF"

W, H = 16.0, 9.0


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "svg.fonttype": "none",
        }
    )


def canvas(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(W, H), dpi=120)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.text(0.7, 8.4, title, fontsize=26, weight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(0.72, 7.85, subtitle, fontsize=14, color=MUTED, va="top")
    ax.plot([0.7, 15.3], [7.6, 7.6], color=GRID, lw=1.2)
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=NAVY, fill=WHITE, ts=16, bs=13, align="left", lw=1.8):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                       linewidth=lw, edgecolor=color, facecolor=fill)
    )
    if align == "center":
        ax.text(x + w / 2, y + h - 0.30, title, fontsize=ts, weight="bold", color=color, va="top", ha="center")
        if body:
            ax.text(x + w / 2, y + h - 0.82, body, fontsize=bs, color=INK, va="top", ha="center", linespacing=1.4)
    else:
        ax.text(x + 0.22, y + h - 0.28, title, fontsize=ts, weight="bold", color=color, va="top")
        if body:
            ax.text(x + 0.22, y + h - 0.80, body, fontsize=bs, color=INK, va="top", linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 lw=lw, color=color, linestyle=ls))


def foot(ax, text: str) -> None:
    ax.text(0.7, 0.32, text, fontsize=11, color=MUTED, ha="left")


def tr(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=170, facecolor=WHITE, pad_inches=0)
    fig.savefig(OUT / f"{stem}.svg", facecolor=WHITE, pad_inches=0)
    plt.close(fig)
    print(f"  {stem}")


def bars(ax, labels, values, colors, x0=1.3, x1=14.9, ybase=1.5, ytop=6.6,
         vmax=None, fmt=lambda v: tr(v, 3), sublabels=None, barw=None,
         value_labels=None):
    """Horizontal-slot vertical bars with the number printed on top.

    `value_labels` overrides the printed text per bar. Used where the audited
    source gives a qualitative statement ("rastgeleye yakın") rather than a
    number — the bar height is then illustrative and must not carry a digit.
    """
    n = len(values)
    span = x1 - x0
    slot = span / n
    bw = barw if barw else slot * 0.44
    vmax = vmax or max(values) * 1.18
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        cx = x0 + slot * (i + 0.5)
        hgt = (val / vmax) * (ytop - ybase)
        ax.add_patch(Rectangle((cx - bw / 2, ybase), bw, hgt, facecolor=col, edgecolor="none"))
        txt = value_labels[i] if value_labels and value_labels[i] else fmt(val)
        size = 21 if (not value_labels or not value_labels[i]) else 15
        ax.text(cx, ybase + hgt + 0.20, txt, fontsize=size, weight="bold", color=col, ha="center")
        ax.text(cx, ybase - 0.35, lab, fontsize=13.5, color=INK, ha="center", va="top", linespacing=1.3)
        if sublabels and sublabels[i]:
            ax.text(cx, ybase - 1.15, sublabels[i], fontsize=11.5, color=MUTED, ha="center", va="top", linespacing=1.3)
    ax.plot([x0 - 0.2, x1 + 0.2], [ybase, ybase], color=GRID, lw=1.4)


# ===========================================================================
# S03 - window / event / flight
# ===========================================================================
def s03():
    fig, ax = canvas("Aynı veriden üç farklı soru sorulabilir",
                     "Bu üçü birbirinin yerine kullanılamaz — proje boyunca en pahalı hatamız buydu")
    xL, xR = 1.6, 14.6
    ax.add_patch(Rectangle((7.3, 2.0), 3.4, 4.6, facecolor=LIGHTNAVY, edgecolor="none"))
    ax.text(9.0, 6.75, "gerçek arıza aralığı", fontsize=13, color=NAVY, ha="center", weight="bold")

    ax.text(xL - 0.75, 6.05, "window", fontsize=15, weight="bold", color=INK, ha="right", va="center")
    nw = 18
    for i in range(nw):
        cx = xL + (xR - xL) * (i + 0.5) / nw
        hot = 8 <= i <= 12
        ax.add_patch(Rectangle((cx - 0.28, 5.75), 0.56, 0.6,
                               facecolor=BLUE if hot else LIGHTGREY, edgecolor="none"))
    ax.text(xR + 0.15, 6.05, "modelin skor ürettiği kısa dilim", fontsize=12, color=MUTED, va="center")

    ax.text(xL - 0.75, 4.3, "event", fontsize=15, weight="bold", color=AQUA, ha="right", va="center")
    ax.add_patch(Rectangle((7.55, 4.0), 2.6, 0.62, facecolor=AQUA, edgecolor="none"))
    ax.text(xR + 0.15, 4.3, "ardışık alarmların birleşip\ngerçek aralıkla eşleşmesi",
            fontsize=12, color=MUTED, va="center", linespacing=1.35)

    ax.text(xL - 0.75, 2.6, "flight", fontsize=15, weight="bold", color=INK, ha="right", va="center")
    ax.add_patch(Rectangle((xL, 2.3), xR - xL, 0.62, facecolor=GREY, edgecolor="none"))
    ax.text(9.0, 2.61, "bu uçuşta en az bir alarm var mı?", fontsize=12.5, color=WHITE, ha="center", va="center")
    ax.text(xR + 0.15, 2.6, "operatöre en az bilgi veren\ndüzey", fontsize=12, color=MUTED, va="center", linespacing=1.35)

    ax.annotate("", xy=(xR, 1.55), xytext=(xL, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    ax.text(9.0, 1.15, "zaman", fontsize=12, color=MUTED, ha="center")
    foot(ax, "Kavramsal gösterim — gerçek bir uçuşun skor kaydı değildir.")
    save(fig, "F03_uc_olcum_duzeyi")


# ===========================================================================
# S04 - valid vs invalid alarm
# ===========================================================================
def s04():
    fig, ax = canvas("Geçerli alarm nedir?",
                     "Alarm arıza BAŞLADIKTAN SONRA açılmalı ve arıza aralığıyla kesişmeli")
    ax.add_patch(Rectangle((6.4, 2.0), 4.4, 4.6, facecolor=LIGHTNAVY, edgecolor="none"))
    ax.text(8.6, 6.75, "gerçek arıza aralığı", fontsize=13.5, color=NAVY, ha="center", weight="bold")

    rows = [
        (5.75, 4.2, 7.6, RED, "arıza başlamadan açıldı", "SAYILMAZ  ·  sistem uyarmadı, tesadüfen üstüne denk geldi"),
        (4.05, 7.4, 9.9, AQUA, "aralık içinde başladı", "GEÇERLİ  ·  operatöre gerçek uyarı"),
        (2.35, 11.4, 13.6, RED, "arıza bittikten sonra açıldı", "SAYILMAZ  ·  geç kalmış alarm"),
    ]
    for y, a, b, col, lab, verdict in rows:
        ax.add_patch(Rectangle((a, y), b - a, 0.55, facecolor=col, edgecolor="none"))
        ax.text(a, y + 0.78, lab, fontsize=13, color=col, weight="bold")
        ax.text(15.3, y + 0.25, verdict, fontsize=12, color=col, ha="right", va="center")

    ax.plot([6.4, 6.4], [1.7, 6.9], color=NAVY, lw=1.2, ls=":")
    ax.plot([10.8, 10.8], [1.7, 6.9], color=NAVY, lw=1.2, ls=":")
    ax.text(6.4, 1.35, "arıza başlangıcı", fontsize=11.5, color=NAVY, ha="center")
    ax.text(10.8, 1.35, "arıza sonu", fontsize=11.5, color=NAVY, ha="center")
    foot(ax, "Ölçüt çifti hiçbir slaytta ayrılmaz: event recall  +  false event / normal flight-hour")
    save(fig, "F04_gecerli_alarm")


# ===========================================================================
# S05 - four frozen rules
# ===========================================================================
def s05():
    fig, ax = canvas("Değerlendirme sözleşmesi: dört kural, en baştan dondurulmuş",
                     "Her kural bir hatanın bedeli ödendikten sonra kurala dönüştü")
    items = [
        ("Group-safe split", "Aynı kaynak / oturum / senaryo\nailesi iki role birden giremez.",
         "Akraba uçuşlar rollere dağılırsa\nmodel testi zaten eğitimde görür."),
        ("Causal scoring", "t anındaki skor yalnız x ≤ t\nkullanılarak üretilir.",
         "Uçuşun tamamından normalize etmek\ngeleceğe bakma hatasıdır."),
        ("Frozen policy", "Ölçekleme, checkpoint ve threshold\nyalnız validation'da seçilir.",
         "Test sonucuna bakıp eşik oynatmak\nsonucu geçersiz kılar."),
        ("True-onset detection", "Alarm arıza başladıktan sonra\nbaşlamalıdır.",
         "Çakışma yeterli sayılırsa yakalama\nüç kat fazla görünür."),
    ]
    xs, ys = [0.9, 8.3], [4.15, 0.85]
    hues = [BLUE, ORANGE, VIOLET, AQUA]
    for i, (t, b, why) in enumerate(items):
        x, y = xs[i % 2], ys[i // 2]
        box(ax, x, y, 6.8, 2.85, t, b, color=hues[i], ts=17, bs=13)
        ax.text(x + 0.24, y + 0.72, "Neden: " + why, fontsize=11.5, color=MUTED, va="top", linespacing=1.35)
    save(fig, "F05_degerlendirme_sozlesmesi")


# ===========================================================================
# S06 - outside observation vs onboard telemetry
# ===========================================================================
def s06():
    fig, ax = canvas("Neden dışarıdan gözlem yetmedi",
                     "Aynı uçuş, iki farklı gözlem katmanı")
    box(ax, 0.9, 1.2, 6.6, 5.9, "Dışarıdan gözlem  (ADS-B yayını)", "", color=ORANGE, fill=WHITE)
    ax.text(1.2, 6.35, "Görülebilen", fontsize=13, weight="bold", color=ORANGE)
    for i, s in enumerate(["konum", "irtifa", "yer hızı", "rota (track)"]):
        ax.text(1.4, 5.95 - i * 0.44, "•  " + s, fontsize=13, color=INK)
    ax.text(1.2, 3.95, "Görülemeyen", fontsize=13, weight="bold", color=RED)
    for i, s in enumerate(["motor gücü düşüşü", "sensör bozulması", "otopilot komutu",
                           "GPS spoofing / gerçek manevra ayrımı"]):
        ax.text(1.4, 3.55 - i * 0.44, "×  " + s, fontsize=13, color=GREY)

    box(ax, 8.5, 1.2, 6.8, 5.9, "Araç içi telemetri  (ALFA · SEAD · RflyMAD)", "", color=BLUE, fill=WHITE)
    ax.text(8.8, 6.35, "Ek olarak görülebilen", fontsize=13, weight="bold", color=BLUE)
    for i, s in enumerate(["throttle / attitude komutu", "ölçülen tepki (airspeed, roll, pitch)",
                           "EKF innovation ve test ratio", "motor / ESC kanalları",
                           "sensör tutarlılık ilişkileri", "etiketli arıza başlangıç–bitiş aralığı"]):
        ax.text(9.0, 5.95 - i * 0.52, "•  " + s, fontsize=13, color=INK)

    arrow(ax, 7.7, 4.1, 8.35, 4.1, color=BLUE, lw=2.4)
    foot(ax, "Sonuç: ADS-B atılmadı, rolü değişti — arıza yakalama verisi değil, yanlış alarm stres testi verisi oldu.")
    save(fig, "F06_gozlem_katmani")


# ===========================================================================
# S08 - column repair
# ===========================================================================
def s08():
    fig, ax = canvas("Ham kolonlardan feature'a: veri ilk hâliyle kullanılamıyordu",
                     "ALFA örneği — model kurmadan önce yapılan üç müdahale")
    steps = [
        ("1.  Boş görünen kolon", "velocity_mps kolonu %100 boş görünüyordu.",
         "Sebep veri eksikliği değil,\ntopic / kolon eşleşme hatasıydı.", "düzeltildi"),
        ("2.  Eksik uçuşlar", "İşlenmiş kayıtlar ham veriyi tam kapsamıyordu.",
         "8 eksik aday denetlendi,\n7'si yeniden parse edildi.", "normal havuz 10 → 15"),
        ("3.  Feature üretimi", "Ham kanal doğrudan anomali sinyali değil.",
         "Fiziksel residual aileleri\nüretildi.", "73 → 85 feature"),
    ]
    hues = [(BLUE, LIGHTNAVY), (ORANGE, LIGHTORANGE), (AQUA, LIGHTAQUA)]
    for i, (t, problem, action, result) in enumerate(steps):
        x = 0.9 + i * 4.95
        col, tint = hues[i]
        box(ax, x, 2.55, 4.55, 4.45, t, "", color=col, ts=16)
        ax.text(x + 0.25, 6.10, "Sorun", fontsize=12, weight="bold", color=RED)
        ax.text(x + 0.25, 5.77, problem, fontsize=12.5, color=INK, va="top", linespacing=1.4, wrap=True)
        ax.text(x + 0.25, 4.90, "İşlem", fontsize=12, weight="bold", color=MUTED)
        ax.text(x + 0.25, 4.57, action, fontsize=12.5, color=INK, va="top", linespacing=1.4)
        ax.add_patch(Rectangle((x + 0.25, 2.85), 4.05, 0.95, facecolor=tint, edgecolor="none"))
        ax.text(x + 2.28, 3.32, result, fontsize=15.5, weight="bold", color=col, ha="center", va="center")
    foot(ax, "Normal-only bir modelde normal havuzunu %50 büyütmek, mimari değiştirmekten daha etkili çıktı.")
    save(fig, "F08_kolon_onarimi")


# ===========================================================================
# S09 - command -> response feature chain
# ===========================================================================
def s09():
    fig, ax = canvas("Arıza 'motor_failure' adlı bir kolonda gelmiyor",
                     "Aranan şey ham büyüklük değil, arızanın uçuş dinamiğinde bıraktığı ikincil iz")
    nodes = [
        ("komut", "throttle\nattitude"),
        ("ölçülen tepki", "airspeed\nroll · pitch"),
        ("residual", "tepki −\nbeklenen tepki"),
        ("context", "uçuş fazı\nmesaj sıklığı"),
        ("event score", "eşik +\npersistence"),
    ]
    x = 0.9
    hues = [VIOLET, VIOLET, AQUA, ORANGE, BLUE]
    for i, (t, b) in enumerate(nodes):
        col = hues[i]
        box(ax, x, 4.55, 2.5, 1.9, t, b, color=col, ts=14, bs=12, align="center")
        if i < 4:
            arrow(ax, x + 2.55, 5.5, x + 2.95, 5.5, color=GREY)
        x += 3.0

    ax.text(0.9, 3.85, "Somut örnek — motor arızası", fontsize=15, weight="bold", color=AQUA)
    ax.text(0.9, 3.45,
            "İrtifa hemen düşmez.  Önce throttle komutu yüksek kalırken airspeed azalır,\n"
            "sonra autopilot telafi ederken attitude davranışı değişir.",
            fontsize=13.5, color=INK, va="top", linespacing=1.5)
    ax.text(0.9, 2.30, "Üretilen feature aileleri", fontsize=13, weight="bold", color=MUTED)
    ax.text(0.9, 1.95,
            "command–response residual   ·   attitude / airspeed error   ·   route deviation\n"
            "freeze counters   ·   CUSUM birikimleri   ·   GPS speed residual (UAV Attack'ta en güçlü tek feature)",
            fontsize=12.5, color=INK, va="top", linespacing=1.6)
    foot(ax, "Ham irtifaya bakan model arızayı geç görür; komut ile tepki farkına bakan model erken görür.")
    save(fig, "F09_feature_zinciri")


# ===========================================================================
# S10 - normal-only
# ===========================================================================
def s10():
    fig, ax = canvas("Neden normal-only öğrenme",
                     "Etiket eğitime girmez; yalnız threshold kalibrasyonu ve değerlendirmede kullanılır")
    box(ax, 1.0, 4.7, 4.4, 1.9, "Normal uçuşlar", "büyütülebilir kaynak\ndaha çok uçuş, daha çok saat",
        color=BLUE, align="center", ts=16, bs=12)
    box(ax, 1.0, 1.9, 4.4, 1.9, "Arızalı uçuşlar", "büyütülemez\ngerçek uçakta arıza üretmek\nriskli ve pahalı",
        color=ORANGE, align="center", ts=16, bs=12)
    box(ax, 7.9, 4.7, 3.4, 1.9, "MODEL", "yalnız normalden\nsapmayı öğrenir", color=AQUA, align="center", ts=17, bs=12)
    box(ax, 12.2, 4.7, 3.1, 1.9, "Değerlendirme", "event recall +\nyanlış alarm/saat", color=VIOLET, align="center", ts=15, bs=12)

    arrow(ax, 5.5, 5.65, 7.8, 5.65, color=BLUE, lw=2.6)
    arrow(ax, 11.4, 5.65, 12.1, 5.65, color=AQUA, lw=2.6)
    arrow(ax, 5.5, 2.85, 13.7, 2.85, color=ORANGE, lw=2.0, ls=(0, (5, 4)))
    arrow(ax, 13.75, 2.9, 13.75, 4.6, color=ORANGE, lw=2.0, ls=(0, (5, 4)))
    ax.text(9.4, 2.5, "arızalı veri modele HİÇ girmez — yalnız değerlendirmeye gider",
            fontsize=12.5, color=ORANGE, ha="center")

    ax.text(1.0, 1.35, "Gerekçe:", fontsize=13, weight="bold", color=MUTED)
    ax.text(2.15, 1.35, "arıza örneği az ve heterojen; supervised model gördüğü arıza tipine aşırı uyum sağlar, "
                        "görmediğine kör kalır.", fontsize=12.5, color=INK)
    foot(ax, "İstisna açıkça işaretlendi: LightGBM ve supervised TCN ayrı karşılaştırma kollarıydı.")
    save(fig, "F10_normal_only")


# ===========================================================================
# S11 - monolithic vs modular
# ===========================================================================
def s11():
    fig, ax = canvas("Tek modele her şeyi vermek sinyali seyreltiyor",
                     "Isolation Forest — aynı model, iki farklı feature kurulumu · flight ROC-AUC")
    ax.text(0.9, 7.05, "Isolation Forest neden seçildi?", fontsize=14, weight="bold", color=NAVY)
    ax.text(0.9, 6.72, "Etiket istemez, normal veriden aykırılık skoru üretir, hızlıdır ve hangi feature'ın "
                       "katkı verdiği okunabilir.", fontsize=12.5, color=INK, va="top")
    bars(ax,
         ["Monolitik IF\nALFA", "Monolitik IF\nUAV Attack", "Modüler IF\nALFA", "En güçlü tek modül\n(guidance)"],
         [0.50, 0.21, 0.833, 0.864],
         [ORANGE, ORANGE, BLUE, AQUA],
         x0=1.3, x1=14.2, ybase=1.9, ytop=6.1, vmax=1.0,
         value_labels=["rastgeleye yakın", None, None, None])
    ygate = 1.9 + 0.5 * 4.2
    ax.plot([1.1, 14.3], [ygate, ygate], color=GRID, lw=1.2, ls="--")
    ax.text(14.42, ygate, "0,50\nrastgele", fontsize=11, color=MUTED, va="center", linespacing=1.3)
    foot(ax, "Kaynak ALFA monolitik sonucunu sayı yerine 'rastgeleye yakın' olarak veriyor; bar yüksekliği temsilîdir.  |  "
             "İlk iki bar farklı veri setlerine aittir — monolitik kurulumun ikisinde de zayıf kaldığının kanıtıdır.")
    save(fig, "F11_monolitik_moduler")


# ===========================================================================
# S12 - data diversity
# ===========================================================================
def s12():
    fig, ax = canvas("Aynı model, daha çeşitli veri",
                     "LSTM autoencoder — mimari hiç değişmedi, yalnız normal uçuş havuzu büyüdü")
    ax.text(0.9, 7.05, "Autoencoder neden denendi?", fontsize=14, weight="bold", color=NAVY)
    ax.text(0.9, 6.72, "Isolation Forest zaman sırasını kullanmıyordu. Autoencoder normal telemetri penceresini "
                       "yeniden kurmayı öğrenir; kuramadığı yer anomali adayıdır.",
            fontsize=12.5, color=INK, va="top")
    bars(ax, ["Küçük normal havuz\n(10 normal uçuş)", "+5 normal uçuş\n(15 normal uçuş)"],
         [0.731, 0.918], [ORANGE, BLUE], x0=1.3, x1=9.6, ybase=1.9, ytop=6.0, vmax=1.0)

    box(ax, 10.3, 2.6, 4.9, 3.3, "⚠  Bu değer final değil", "", color=RED, fill=LIGHTRED, ts=15)
    ax.text(10.55, 5.05,
            "0,731 ve 0,918 nedensellik\ndüzeltmesinden ÖNCEKİ değerlerdir.\n\n"
            "Düzeltilmiş karşılığı 0,611'e düşecek\n— birkaç slayt sonra göreceksiniz.\n\n"
            "Burada gösterilen tek şey veri\nçeşitliliğinin etkisidir, başarı değil.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)
    foot(ax, "Küçük veride bir derin modelin zayıf çıkması her zaman modelin suçu değildir.")
    save(fig, "F12_veri_cesitliligi")


# ===========================================================================
# S13 - synthetic corruption taxonomy
# ===========================================================================
def s13():
    fig, ax = canvas("Modelin neye duyarlı olduğunu nasıl ölçtük",
                     "Kontrollü bozulma enjeksiyonu — yalnız değerlendirmede kullanıldı, eğitime hiç girmedi")
    ax.text(0.9, 7.05, "Neden gerekliydi?", fontsize=14, weight="bold", color=NAVY)
    ax.text(0.9, 6.72, "Gerçek arıza örneği az. 'Model çalışmıyor' demeden önce, hangi bozulma tipinde "
                       "çalışıp hangisinde çalışmadığını ayırmak gerekiyordu.",
            fontsize=12.5, color=INK, va="top")

    types = [
        ("donma (freeze)", "kanal sabit kalır", AQUA, "yakalanabilir"),
        ("sabit kayma (bias)", "sabit offset eklenir", RED, "zayıf"),
        ("büyüyen sapma (drift)", "giderek artan hata", AQUA, "CUSUM ~%75"),
        ("gürültü", "varyans artışı", AQUA, "yakalanabilir"),
        ("sinsi konum kayması", "yavaş, küçük konum hatası", RED, "zayıf"),
        ("veri kaybı (dropout)", "kanal boşluğu", AQUA, "yakalanabilir"),
    ]
    for i, (t, d, col, verdict) in enumerate(types):
        x = 0.9 + (i % 3) * 4.95
        y = 3.55 - (i // 3) * 2.35
        box(ax, x, y, 4.55, 1.95, t, d, color=col, fill=LIGHTAQUA if col == AQUA else LIGHTRED,
            ts=14.5, bs=12)
        ax.text(x + 0.24, y + 0.42, verdict, fontsize=13, weight="bold", color=col)
    foot(ax, "Ders: 'anomali' tek bir kategori değildir. Sabit kayma ve sinsi konum kayması, "
             "ölçülen bütün yöntemlerde en zor iki sınıf olarak kaldı.")
    save(fig, "F13_bozulma_taksonomisi")


# ===========================================================================
# S14 - method tour
# ===========================================================================
def s14():
    fig, ax = canvas("Yakalama arttı; alarm yükü daha hızlı arttı",
                     "UAV-SEAD üzerinde yöntem turu — her adım bir öncekinin bıraktığı boşluğu test etti")
    steps = [
        ("Isolation Forest\nbaseline", 0.205, "referans", GREY),
        ("Chronos\nzero-shot forecasting", 0.390, "hazır zaman serisi modeli\nmekanik dalda +%90", BLUE),
        ("Tek feature\nthrust command", 0.459, "en yüksek kategori\nyakalaması", AQUA),
    ]
    bars(ax, [s[0] for s in steps], [s[1] for s in steps], [s[3] for s in steps],
         x0=0.9, x1=9.4, ybase=3.30, ytop=6.35, vmax=0.55,
         sublabels=[s[2] for s in steps])

    box(ax, 10.2, 4.75, 5.1, 1.75, "Bedeli", "", color=RED, fill=LIGHTRED, ts=15)
    ax.text(12.75, 5.62, "38,1", fontsize=33, weight="bold", color=RED, ha="center", va="center")
    ax.text(12.75, 5.05, "yanlış alarm / saat", fontsize=13, color=RED, ha="center")

    box(ax, 10.2, 3.05, 5.1, 1.45, "İki kanallı fusion", "", color=ORANGE, ts=14)
    ax.text(12.75, 3.92, "kapsamı genişletme denemesi", fontsize=12, color=MUTED, ha="center")
    ax.text(12.75, 3.42, "yanlış alarm ×1,89 – ×3,83", fontsize=14, weight="bold", color=ORANGE, ha="center")

    ax.text(0.9, 1.42, "EKF innovation neden ters yönde sinyal verdi (AUC 0,354):", fontsize=13,
            weight="bold", color=VIOLET)
    ax.text(0.9, 1.05, "Anomalili ölçüm estimator tarafından reddedildiğinde innovation düzenli üretilmiyor — "
                       "en sorunlu örnekler daha 'temiz' görünüyor. Bu bir bug değil, göstergenin doğası.",
            fontsize=12, color=INK, va="top")
    foot(ax, "Bu tablo 'daha iyi model bulduk' demiyor; 'aynı duvara farklı yollardan çarptık' diyor.")
    save(fig, "F14_yontem_turu")


# ===========================================================================
# S15 - supervised alternatives
# ===========================================================================
def s15():
    fig, ax = canvas("Denetimli öğrenme neden ana hat olmadı",
                     "Reddetmeden önce iki denetimli alternatif ölçüldü")
    box(ax, 0.9, 3.9, 7.0, 3.1, "LightGBM  ·  UAV-SEAD", "", color=BLUE, ts=16)
    ax.text(1.15, 6.35, "Neden denendi: etiketler elimizdeydi; denetimli bir gradient boosting\n"
                        "modeli tabloda en güçlü baseline sayılır.", fontsize=12.5, color=INK, va="top", linespacing=1.4)
    ax.text(1.15, 5.30, "Sonuç:", fontsize=13, weight="bold", color=MUTED)
    ax.text(2.05, 5.30, "AP 0,349", fontsize=17, weight="bold", color=GREY)
    ax.text(4.10, 5.30, "<", fontsize=15, color=MUTED)
    ax.text(4.60, 5.30, "Isolation Forest 0,385", fontsize=17, weight="bold", color=AQUA)
    ax.text(1.15, 4.55, "Denetimli öğrenme, az ve heterojen etiket altında denetimsizi geçemedi.",
            fontsize=12.5, color=INK, va="top")

    box(ax, 8.3, 3.9, 7.0, 3.1, "Supervised TCN  ·  RflyMAD", "", color=VIOLET, ts=16)
    ax.text(8.55, 6.35, "Neden denendi: temporal convolution, uzun pencerelerde sıralı örüntüyü\n"
                        "yakalamada güçlüdür; rüzgâr kaynaklı yanlış alarmı bastırması umuldu.",
            fontsize=12.5, color=INK, va="top", linespacing=1.4)
    ax.text(8.55, 5.30, "5-fold development sonucu:", fontsize=12.5, weight="bold", color=MUTED)
    ax.text(8.55, 4.95, "critical  %28,87 recall  /  2,87 yanlış alarm-saat\n"
                        "advisory  %67,86 recall  /  12,54 yanlış alarm-saat\n"
                        "gerçek uçuş recall ortalaması  %7,56",
            fontsize=12.5, color=INK, va="top", linespacing=1.5)

    box(ax, 0.9, 1.2, 14.4, 2.35, "Ortak sonuç", "", color=ORANGE, fill=LIGHTORANGE, ts=15)
    ax.text(1.15, 2.85, "TCN'in en iyi durma noktası 2–5 epoch arasındaydı — daha uzun eğitim sonucu kötüleştirdi. "
                        "Bu eğitim yetersizliği değil,\nerken ezberleme işaretidir: model az sayıdaki arıza örneğini "
                        "ezberliyor, yeni arızaya genellemiyor. Etiketli arıza verisi\nbüyütülemediği için ana hat "
                        "normal-only kaldı.", fontsize=12.5, color=INK, va="top", linespacing=1.5)
    save(fig, "F15_denetimli_alternatifler")


# ===========================================================================
# S16 - ADS-B two lessons
# ===========================================================================
def s16():
    fig, ax = canvas("Etiketsiz devasa trafikte iki ders",
                     "256.150.550 satır işlendi  ·  ~497.571 skorlanabilir normal flight-hour")
    box(ax, 0.9, 3.7, 7.0, 3.4, "Ders 1 — sentetik başarı tuzağı", "", color=RED, fill=LIGHTRED, ts=16)
    ax.text(2.55, 5.75, "%97,6", fontsize=40, weight="bold", color=BLUE, ha="center", va="center")
    ax.text(2.55, 5.05, "sentetik anomali recall", fontsize=12, color=MUTED, ha="center")
    ax.text(4.40, 5.70, "ama", fontsize=15, color=MUTED, ha="center", va="center")
    ax.text(6.25, 5.75, "25,54", fontsize=40, weight="bold", color=RED, ha="center", va="center")
    ax.text(6.25, 5.05, "doğal trafikte alarm / saat", fontsize=12, color=MUTED, ha="center")
    ax.text(1.15, 4.55, "Yaklaşım reddedildi ve arşivlendi. O günden sonra yakalama oranını\n"
                        "alarm yükü olmadan raporlamak yasaklandı.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)

    box(ax, 8.3, 3.7, 7.0, 3.4, "Ders 2 — basit fizik kuralları sinir ağlarını geçti", "", color=AQUA, ts=15.5)
    vals = [("Fizik kuralları", 0.600, AQUA), ("Dense-AE", 0.572, GREY),
            ("LSTM-AE", 0.568, GREY), ("LSTM-forecaster", 0.552, GREY)]
    for i, (lab, v, col) in enumerate(vals):
        y = 6.20 - i * 0.52
        wbar = (v - 0.50) / 0.12 * 3.7
        ax.add_patch(Rectangle((11.0, y - 0.15), wbar, 0.32, facecolor=col, edgecolor="none"))
        ax.text(10.85, y, lab, fontsize=12.5, color=INK, ha="right", va="center")
        ax.text(11.05 + wbar + 0.12, y, tr(v, 3), fontsize=13, weight="bold", color=col, va="center")
    ax.text(8.55, 4.32, "Kural: dikey hız ↔ irtifa türevi, yer hızı ↔ konum türevi gibi\n"
                        "5 fiziksel tutarlılık kontrolü (pooled AUC).",
            fontsize=12, color=INK, va="top", linespacing=1.4)

    box(ax, 0.9, 1.55, 14.4, 1.65, "Dürüst not", "", color=GREY, fill="#FAFAFA", ts=14)
    ax.text(1.15, 2.55, "Karşılaştırılan üç sinir ağı, birazdan anlatacağım genlik kontrolünde zaten elenmişti ve bu AUC'ler "
                        "etiket hatalı tarihsel bir\nkoşudan geliyor. Yani bu, kuralın zaferi kadar sinir ağlarının o turdaki "
                        "geçersizliğinin de sonucudur.", fontsize=12.5, color=INK, va="top", linespacing=1.5)
    save(fig, "F16_adsb_iki_ders")


# ===========================================================================
# S17 - causality correction
# ===========================================================================
def s17():
    fig, ax = canvas("Düzeltme 1 — model geleceğe bakıyordu",
                     "CUSUM: küçük ama kalıcı sapmanın kanıtını zaman içinde biriktirir — bu yüzden eklendi")
    ax.text(0.9, 7.05, "Sorun", fontsize=13.5, weight="bold", color=RED)
    ax.text(0.9, 6.72, "CUSUM'un baseline'ı (median/MAD) uçuşun TAMAMINDAN hesaplanıyordu. Arıza sonrası değerler, "
                       "arıza öncesi skoru dolaylı etkiliyordu.", fontsize=12.5, color=INK, va="top")
    ax.text(0.9, 6.05, "Düzeltme", fontsize=13.5, weight="bold", color=AQUA)
    ax.text(0.9, 5.72, "Baseline yalnız normal training uçuşlarından hesaplandı; t anındaki skor yalnız x ≤ t ile üretildi.",
            fontsize=12.5, color=INK, va="top")

    bars(ax, ["Full-flight baseline\n(geçersiz)", "Causal düzeltme\nsonrası", "Düzeltme sonrası en güçlü\ntek sinyal: cross-track"],
         [0.878, 0.611, 0.751], [RED, BLUE, AQUA],
         x0=1.3, x1=14.9, ybase=1.75, ytop=5.05, vmax=1.0)
    arrow(ax, 4.4, 5.35, 6.6, 5.35, color=GREY, lw=2.0)
    foot(ax, "Bu bir performans kaybı değil: o 0,878 hiçbir zaman gerçek zamanda üretilebilecek bir skor değildi.")
    save(fig, "F17_nedensellik")


# ===========================================================================
# S18 - true onset
# ===========================================================================
def s18():
    fig, ax = canvas("Düzeltme 2 — arızadan önce açılmış alarm yakalama değildir",
                     "Eski değerlendirici yalnız çakışmaya bakıyordu")
    ax.add_patch(Rectangle((6.6, 4.15), 4.2, 2.9, facecolor=LIGHTNAVY, edgecolor="none"))
    ax.text(8.7, 7.20, "gerçek arıza aralığı", fontsize=12.5, color=NAVY, ha="center", weight="bold")
    ax.add_patch(Rectangle((3.5, 6.15), 5.6, 0.5, facecolor=RED, edgecolor="none"))
    ax.text(3.5, 6.90, "alarm arızadan ÖNCE açılmış, aralığa kadar sürüyor", fontsize=12.5, color=RED)
    ax.text(15.3, 6.55, "eski ölçüm:  yakalandı sayılıyordu", fontsize=12.5, color=GREY, ha="right", va="center")
    ax.text(15.3, 6.15, "yeni ölçüm:  SAYILMAZ", fontsize=13, color=RED, ha="right", va="center", weight="bold")
    ax.add_patch(Rectangle((7.4, 4.55), 2.6, 0.5, facecolor=AQUA, edgecolor="none"))
    ax.text(3.5, 5.30, "alarm arıza aralığı İÇİNDE başlıyor", fontsize=12.5, color=AQUA)
    ax.text(15.3, 4.80, "her iki ölçümde de YAKALANDI", fontsize=13, color=AQUA, ha="right", va="center", weight="bold")

    bars(ax, ["Çakışma tabanlı\nyakalama (eski)", "Gerçek başlangıç\nyakalaması (düzeltilmiş)"],
         [59.4, 21.0], [RED, BLUE], x0=1.3, x1=8.6, ybase=1.75, ytop=3.55, vmax=100,
         fmt=lambda v: "%59,4" if v > 50 else "%19,4–22,4")
    box(ax, 9.5, 1.30, 5.8, 2.5, "Operasyonel anlamı", "", color=VIOLET, ts=14)
    ax.text(9.75, 3.15, "Alarm zaten açıktı, arıza sonra başladı.\nBu alarm arızayı haber vermedi, tesadüfen\n"
                        "üstüne denk geldi. Ölçüm bunu yakalama\nsayarsa sistemin uyarı kabiliyetini üç kat\nfazla gösterir.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)
    foot(ax, "Bu düzeltmeden sonra bütün geçmiş event sonuçları geçersiz sayılıp yeniden hesaplandı.")
    save(fig, "F18_gercek_baslangic")


# ===========================================================================
# S19 - session split
# ===========================================================================
def s19():
    fig, ax = canvas("Düzeltme 3 — uçuş sayısı bağımsız deney sayısı değil",
                     "UAV-SEAD havuzu 60'tan 1.044 development uçuşuna büyütüldü — ama uçuşlar akrabaydı")
    ax.text(0.9, 7.05, "Sorun", fontsize=13.5, weight="bold", color=RED)
    ax.text(0.9, 6.72, "Aynı gün ve aynı oturumdan gelen akraba uçuşlar train ve test rollerine dağılıyordu. "
                       "Model, test uçuşunun neredeyse aynısını eğitimde görmüştü.",
            fontsize=12.5, color=INK, va="top")
    ax.text(0.9, 6.05, "Düzeltme", fontsize=13.5, weight="bold", color=NAVY)
    ax.text(0.9, 5.72, "Session-level split: akraba uçuşlar aynı role kilitlendi.", fontsize=12.5, color=INK, va="top")

    bars(ax, ["Uçuş bazlı\nsplit", "Oturum bazlı\nsplit"], [0.212, 0.012], [ORANGE, AQUA],
         x0=1.1, x1=7.4, ybase=1.6, ytop=4.7, vmax=0.26, fmt=lambda v: "±" + tr(v, 3))
    ax.text(4.25, 5.05, "Seed oynaklığı", fontsize=15, weight="bold", color=INK, ha="center")
    bars(ax, ["Uçuş bazlı\nsplit", "Oturum bazlı\nsplit"], [0.474, 0.799], [ORANGE, AQUA],
         x0=8.8, x1=15.1, ybase=1.6, ytop=4.7, vmax=1.0)
    ax.text(11.95, 5.05, "Adil satır AUC", fontsize=15, weight="bold", color=INK, ha="center")
    foot(ax, "İyileşme yeni bir modelden değil, doğru veri bölünmesinden geldi. Oynaklığın 17'ye 1 düşmesi asıl kritik "
             "olan: öncesinde hiçbir sonuca güvenilemezdi.")
    save(fig, "F19_oturum_split")


# ===========================================================================
# S20 - magnitude dominance
# ===========================================================================
def s20():
    fig, ax = canvas("Düzeltme 4 — model örüntü değil, sinyal büyüklüğü öğreniyordu",
                     "Üç farklı derin mimari neredeyse aynı sonucu verdi — bu bir başarı değil, ortak kestirme yolu işaretiydi")
    axs = fig.add_axes([0.055, 0.13, 0.36, 0.55])
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, 320)
    y = np.clip(x + rng.normal(0, 0.055, 320), 0, 1)
    axs.scatter(x, y, s=16, color=RED, alpha=0.55, edgecolors="none")
    axs.plot([0, 1], [0, 1], color=GREY, lw=1.4, ls="--")
    axs.set_xlabel("hiç eğitilmemiş rastgele model skoru", fontsize=12)
    axs.set_ylabel("eğitilmiş model skoru", fontsize=12)
    axs.set_xticks([]); axs.set_yticks([])
    for s in ("top", "right"):
        axs.spines[s].set_visible(False)
    axs.spines["left"].set_color(GRID); axs.spines["bottom"].set_color(GRID)
    axs.text(0.04, 0.93, "ρ = 0,964", fontsize=20, weight="bold", color=RED, transform=axs.transAxes)
    axs.text(0.04, 0.85, "ham sinyal büyüklüğü ile ρ = 0,965", fontsize=11.5, color=MUTED, transform=axs.transAxes)

    ax.text(7.2, 7.05, "Teşhis", fontsize=13.5, weight="bold", color=RED)
    ax.text(7.2, 6.72, "Eğitilmiş modelin skoru ile hiç eğitilmemiş bir modelin\n"
                       "skoru neredeyse mükemmel korele — yani ağırlıkların\n"
                       "katkısı yok. Model 'anomali' değil, 'büyük değer' öğrenmiş.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)
    ax.text(7.2, 5.05, "Doğrulama", fontsize=13.5, weight="bold", color=VIOLET)
    ax.text(7.2, 4.72, "Skorlar göreli hata ile yeniden hesaplandı:\n"
                       "genlik bağımlılığı 0,15–0,55'e düştü — ama yakalama da\n"
                       "%6'nın altına indi. Önceki kazancın büyük kısmı sahteydi.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)
    box(ax, 7.2, 1.05, 8.1, 2.15, "Karar", "", color=AQUA, fill=LIGHTAQUA, ts=15)
    ax.text(7.45, 2.55, "Rastgele-model kontrolü o günden sonra zorunlu güvenlik kapısı oldu.\n"
                        "Kapı: ρ < 0,80. Bu kapıdan geçmeyen hiçbir sonuç rapor edilmedi.",
            fontsize=13, color=INK, va="top", linespacing=1.5)
    foot(ax, "Agresif bir manevrada roll büyüktür; genlik takip eden bir model onu anomali sanar.")
    save(fig, "F20_genlik_baskinligi")


# ===========================================================================
# S21 - truth repair
# ===========================================================================
def s21():
    fig, ax = canvas("Düzeltme 5 — modeli değil, sorulan soruyu düzelttik",
                     "RflyMAD ground truth onarımı — modelde hiçbir değişiklik yapılmadı")
    box(ax, 0.9, 3.6, 7.0, 3.5, "Sorun 1 — etiketin kapsamı", "", color=RED, ts=16)
    ax.text(1.15, 6.35, "Arızalı uçuşun TAMAMI anomali sayılıyordu. Bu, modelin uçuşun\n"
                        "herhangi bir yerinde alarm vermesini yeterli kılıyor.",
            fontsize=12.5, color=INK, va="top", linespacing=1.4)
    bars(ax, ["Tüm-uçuş etiketi", "Gerçek aralık truth"], [0.749, 0.526], [ORANGE, BLUE],
         x0=1.3, x1=7.5, ybase=4.15, ytop=5.35, vmax=1.0)
    ax.text(4.4, 3.90, "yanında  22,28 yanlış alarm / saat", fontsize=13, weight="bold", color=RED, ha="center")

    box(ax, 8.3, 3.6, 7.0, 3.5, "Sorun 2 — parser hatası", "", color=VIOLET, ts=16)
    ax.text(8.55, 6.35, "2.712 SIL/HIL uçuşunda ayrıştırıcı t = 0'da SAHTE arıza üretiyordu —\n"
                        "arıza sanki ilk saniyeden başlıyormuş gibi görünüyordu.",
            fontsize=12.5, color=INK, va="top", linespacing=1.4)
    ax.text(11.8, 5.15, "2.712", fontsize=42, weight="bold", color=VIOLET, ha="center", va="center")
    ax.text(11.8, 4.45, "uçuş yeniden işlendi", fontsize=13, color=MUTED, ha="center")
    ax.text(11.8, 3.95, "ilk örnekten itibaren aktif görünen arıza:  1.354  →  0",
            fontsize=12.5, weight="bold", color=AQUA, ha="center")

    box(ax, 0.9, 1.0, 14.4, 2.25, "Sorulan soru değişti", "", color=AQUA, fill=LIGHTAQUA, ts=15)
    ax.text(8.1, 1.95, "\"Bu uçuş arızalı mı?\"     →     \"Gerçek arıza aralığını buldun mu?\"",
            fontsize=19, weight="bold", color=INK, ha="center", va="center")
    foot(ax, "Projedeki en büyük kazanımlardan biri yeni bir model değil, doğru bir ground truth oldu.")
    save(fig, "F21_truth_onarimi")


# ===========================================================================
# S22 - GNSS integrity pilot
# ===========================================================================
def s22():
    fig, ax = canvas("Dar çerçeve denemesi: yalnız GPS bütünlüğü",
                     "Genel dedektör yerine tek bir arıza ailesine odaklanmak sonucu değiştirir mi?")
    ax.text(0.9, 7.05, "Neden bu deneme?", fontsize=14, weight="bold", color=NAVY)
    ax.text(0.9, 6.72, "Genel amaçlı bir anomali dedektörü her arıza tipinde ortalama başarı gösteriyordu. "
                       "Soruyu daralttık: tek bir arıza ailesine\nodaklanırsak işletilebilir bir çalışma noktası "
                       "bulunur mu? Roller ayrıldı — 23 geliştirme, 15 prova, 20 mühürlü uçuş.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)

    methods = [
        ("PX4-native göstergeler", "EKF innovation ve test ratio —\nuçuş kontrolcüsünün kendi tutarlılık ölçüsü",
         "prova: %90 recall\nama 28,86 alarm / uçuş-saati"),
        ("Çok kanallı CUSUM", "birden fazla residual kanalında\nbirikimli sapma",
         "bütçe altında recall çöktü"),
        ("Bağlamsal LSTM", "uçuş fazına koşullu\nsequence prediction",
         "prova: %90 / 0 alarm-saat\ngeliştirme: %47,1 / 20,32"),
    ]
    hues = [BLUE, ORANGE, VIOLET]
    for i, (t, why, res) in enumerate(methods):
        x = 0.9 + i * 4.95
        box(ax, x, 2.2, 4.55, 3.9, t, "", color=hues[i], ts=15)
        ax.text(x + 0.24, 5.35, why, fontsize=12, color=MUTED, va="top", linespacing=1.4)
        ax.text(x + 0.24, 4.20, res, fontsize=13, weight="bold", color=INK, va="top", linespacing=1.45)
        ax.add_patch(Rectangle((x + 0.24, 2.45), 4.05, 0.6, facecolor=LIGHTRED, edgecolor=RED, lw=1.2))
        ax.text(x + 2.27, 2.75, "hedef karşılanmadı", fontsize=13, weight="bold", color=RED, ha="center", va="center")

    ax.text(0.9, 1.65, "Dondurulmuş bütçe:", fontsize=13, weight="bold", color=MUTED)
    ax.text(3.4, 1.65, "critical ≤ 2 olay/uçuş-saati (5 s persistence)   ·   advisory ≤ 12 olay/uçuş-saati (15 s persistence)",
            fontsize=12.5, color=INK)
    foot(ax, "Provada iyi görünen sonuç geliştirmeye taşınmadı — eşik ve model roller arasında kararlı genellemedi. "
             "Mühürlü set açılmadı.")
    save(fig, "F22_gnss_pilotu")


# ===========================================================================
# S23 - command-response residual study
# ===========================================================================
def s23():
    fig, ax = canvas("Genlik problemine yapısal cevap: komut → tepki residual",
                     "Anomaliyi ham büyüklükte değil, öğrenilmiş uçuş dinamiğinin tahmin hatasında ara")
    ax.text(0.9, 7.00, "r  =  ölçülen tepki  −  beklenen tepki (kontrol komutları, bağlam)",
            fontsize=19, weight="bold", color=AQUA)
    ax.text(0.9, 6.35, "Neden genlik baskınlığını yapısal olarak azaltır:", fontsize=13.5, weight="bold", color=MUTED)
    ax.text(0.9, 6.00, "Agresif manevrada roll büyük olabilir — ham skor bunu anomali sanar. Ama komuta uygun tepki varsa "
                       "residual küçük kalır.\nTersine: gaz yüksek, model airspeed'in korunmasını bekliyor, ölçülen airspeed "
                       "düşüyor — ham değer hâlâ normal aralıkta olsa bile residual hemen sapar.",
            fontsize=12.5, color=INK, va="top", linespacing=1.5)

    box(ax, 0.9, 2.7, 7.0, 2.5, "Bulgu — sinyal gerçekten var", "", color=AQUA, fill=LIGHTAQUA, ts=15.5)
    ax.text(1.15, 4.45, "ALFA motor, RflyMAD motor ve sensör sınıflarında\narıza öncesi ve sonrası residual dağılımları\n"
                        "eşikten BAĞIMSIZ olarak ayrıştı.", fontsize=12.5, color=INK, va="top", linespacing=1.45)

    box(ax, 8.3, 2.7, 7.0, 2.5, "Engel — kalibrasyon açığı", "", color=RED, fill=LIGHTRED, ts=15.5)
    ax.text(8.55, 4.45, "Güvenilir bir yanlış alarm eşiği kurmak için\ngereken bağımsız normal uçuş süresinin:",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)
    ax.text(10.4, 3.30, "ALFA  ~1/11,8", fontsize=16, weight="bold", color=RED, ha="center")
    ax.text(13.4, 3.30, "RflyMAD  ~1/5,1", fontsize=16, weight="bold", color=RED, ha="center")

    ax.text(0.9, 2.05, "Bilinen risk ve alınan önlem:", fontsize=13, weight="bold", color=MUTED)
    ax.text(0.9, 1.70, "Model y ≈ y(t−1) kopyacılığını öğrenip arızayı bir adım geriden takip edebilir. Bu yüzden tepki geçmişi "
                       "sınırlandı,\nkomut ve yavaş bağlam öne çıkarıldı.", fontsize=12.5, color=INK, va="top", linespacing=1.5)
    foot(ax, "Sonuç: sinyal var, eşik yok. Eksik olan model değil, bağımsız normal uçuş saati.")
    save(fig, "F23_komut_tepki_residual")


# ===========================================================================
# S24 - final model architecture
# ===========================================================================
def s24():
    fig, ax = canvas("Final yaklaşım: reconstruction yerine next-step forecasting",
                     "Ortak protokol — dört veri setinde birebir aynı kurulum")
    nodes = [("son 32\ncausal adım", ""), ("LSTM\n64 gizli birim", ""), ("kanal başına\nμ, σ", ""),
             ("masked\nGaussian NLL", ""), ("threshold +\npersistence", ""), ("event", "")]
    x = 0.9
    for i, (t, _) in enumerate(nodes):
        col = BLUE if i >= 3 else VIOLET
        if i == 5:
            col = AQUA
        box(ax, x, 4.7, 2.28, 1.5, t, "", color=col, ts=13.5, align="center")
        if i < 5:
            arrow(ax, x + 2.33, 5.45, x + 2.55, 5.45, color=GREY)
        x += 2.5
    ax.plot([0.9, 8.0], [6.55, 6.55], color=VIOLET, lw=1.6)
    ax.text(4.45, 6.72, "tahmin", fontsize=13, color=VIOLET, ha="center")
    ax.plot([8.4, 15.3], [6.55, 6.55], color=BLUE, lw=1.6)
    ax.text(11.85, 6.72, "karar", fontsize=13, color=BLUE, ha="center")

    ax.text(0.9, 4.15, "Neden reconstruction değil forecasting?", fontsize=14, weight="bold", color=AQUA)
    ax.text(0.9, 3.80, "Bir autoencoder girdiyi yeniden kurmaya çalışır ve büyük değerleri yeniden kurmak doğal olarak zordur — "
                       "bu yüzden skor\nbüyüklüğü takip eder. Forecasting farklı bir soru sorar: 'bir sonraki adım ne olmalı'. "
                       "Değerler büyük olsa bile komuta uygun\ntepki varsa tahmin hatası küçük kalır. Bu, bir önceki slayttaki "
                       "genlik problemine doğrudan cevaptır.",
            fontsize=12.5, color=INK, va="top", linespacing=1.5)

    specs = [("Ön işleme", "yalnız training verisinden robust ölçekleme\n(median, 1,4826×MAD), [−5, +5] clip"),
             ("Pencere kuralı", "5 saniyeden uzun boşlukta ve uçuş\nsınırında pencere kesilir"),
             ("Aktif kanal", "28 kanaldan 24 — training'de MAD'i sıfır\nçıkan kanallar kalibre edilemez sayıldı")]
    for i, (t, b) in enumerate(specs):
        box(ax, 0.9 + i * 4.95, 0.9, 4.55, 1.85, t, b, color=[BLUE, ORANGE, VIOLET][i],
            fill=WHITE, ts=13.5, bs=12)
    save(fig, "F24_final_model")


# ===========================================================================
# S25 - Gaussian NLL
# ===========================================================================
def s25():
    fig, ax = canvas("Skor: masked Gaussian NLL",
                     "Klasik reconstruction error'dan ayrıldığı yer burası")
    ax.text(8.0, 6.95, "NLL  ≈  ½ · ((x − μ)² / σ²)  +  log(σ)", fontsize=25, weight="bold", color=VIOLET, ha="center")
    ax.text(8.0, 6.40, "yalnız gözlenen kanallar üzerinden ortalanır", fontsize=12.5, color=MUTED, ha="center")

    for i, (cx, sig, lab, verdict, col) in enumerate(
            [(4.2, 0.42, "model EMİN\n(σ küçük)", "aynı hata → YÜKSEK skor", BLUE),
             (11.5, 1.30, "model EMİN DEĞİL\n(σ büyük)", "aynı hata → düşük skor", ORANGE)]):
        xs = np.linspace(-3.6, 3.6, 400)
        ys = np.exp(-0.5 * (xs / sig) ** 2)
        axi = fig.add_axes([0.135 + i * 0.455, 0.245, 0.30, 0.30])
        axi.plot(xs, ys, color=col, lw=2.4)
        axi.fill_between(xs, 0, ys, color=col, alpha=0.14)
        axi.axvline(1.05, color=INK, lw=2.0, ls="--")
        axi.set_ylim(0, 1.25); axi.set_xlim(-3.6, 3.6)
        axi.set_xticks([]); axi.set_yticks([])
        for s in ("top", "right", "left"):
            axi.spines[s].set_visible(False)
        axi.spines["bottom"].set_color(GRID)
        axi.text(1.15, 1.05, "aynı büyüklükte hata", fontsize=11, color=INK)
        ax.text(cx, 5.90, lab, fontsize=14.5, weight="bold", color=col, ha="center", linespacing=1.35)
        ax.text(cx, 1.95, verdict, fontsize=13.5, weight="bold", color=col, ha="center")

    box(ax, 0.9, 0.65, 14.4, 1.25, "", "", color=GREY, fill="#FAFAFA", lw=1.2)
    ax.text(1.2, 1.52, "Masked:", fontsize=12.5, weight="bold", color=MUTED, va="center")
    ax.text(2.35, 1.52, "eksik sensör hücreleri loss ve skor dışında tutulur — eksik veri skoru şişirmemeli.",
            fontsize=12, color=INK, va="center")
    ax.text(1.2, 1.00, "Negatif NLL:", fontsize=12.5, weight="bold", color=MUTED, va="center")
    ax.text(2.75, 1.00, "bir hata değildir — kod sabit terimi eklemez, σ < 1 iken log(σ) negatiftir. "
                        "Daha negatif validation NLL yalnız daha iyi checkpoint demektir, anomali yakalama kanıtı değil.",
            fontsize=12, color=INK, va="center")
    save(fig, "F25_gaussian_nll")


# ===========================================================================
# S26 - role card
# ===========================================================================
def s26():
    fig, ax = canvas("Bağımsızlık sözleşmesi: beş rol, sıfır kesişim",
                     "RflyMAD rol kartı — akraba uçuşlar rollere dağılmadan genelleme iddiası yapılamaz")
    roles = [("Normal\ntraining", 260, 7, "ölçekleyici ve model fit", BLUE, LIGHTNAVY),
             ("Normal\nvalidation", 90, 3, "checkpoint, threshold\nve event policy seçimi", VIOLET, "#E6E3F4"),
             ("Bağımsız\nnormal test", 91, 3, "yanlış alarm yükü\nölçümü", AQUA, LIGHTAQUA),
             ("Anomali\ndevelopment", 557, 12, "teşhis ve event\ndeğerlendirmesi", ORANGE, LIGHTORANGE),
             ("Mühürlü\nfinal test", 553, 13, "HİÇ AÇILMADI", RED, WHITE)]
    x0, slot = 0.9, 2.92
    for i, (t, n, g, use, col, tint) in enumerate(roles):
        x = x0 + i * slot
        hgt = 0.95 + (n / 557) * 2.55
        sealed = (i == 4)
        ax.add_patch(Rectangle((x, 2.9), 2.55, hgt, facecolor=tint,
                               edgecolor=col, lw=2.0, ls=(0, (4, 3)) if sealed else "-"))
        cy = 2.9 + hgt / 2
        ax.text(x + 1.27, cy + 0.22, f"{n}", fontsize=26, weight="bold", color=col, ha="center", va="center")
        ax.text(x + 1.27, cy - 0.38, "uçuş", fontsize=12, color=MUTED, ha="center", va="center")
        ax.text(x + 1.27, 2.60, f"{g} bağımsız grup", fontsize=11.5, weight="bold", color=col, ha="center", va="top")
        ax.text(x + 1.27, 6.50, t, fontsize=14, weight="bold", color=col, ha="center", va="bottom", linespacing=1.3)
        ax.text(x + 1.27, 2.05, use, fontsize=11.5,
                weight="bold" if sealed else "normal",
                color=col if sealed else MUTED, ha="center", va="top", linespacing=1.4)
    foot(ax, "Mühürlü seti açmak, sonuca bakıp karar vermek olurdu. Development sonucu geçme eşiğinin altında kaldığı "
             "için kapalı bırakıldı.")
    save(fig, "F26_rol_karti")


# ===========================================================================
# S27 - magnitude gate pass
# ===========================================================================
def s27():
    fig, ax = canvas("İlk kez genlik kontrolünden temiz geçen model",
                     "Aynı teşhis, iki farklı tur — kapı: ρ < 0,80")
    bars(ax, ["Önceki tur\n(UAV-SEAD, autoencoder ailesi)", "Bu tur\n(RflyMAD, Gaussian forecaster)"],
         [0.964, 0.329], [RED, AQUA], x0=1.3, x1=9.8, ybase=1.9, ytop=6.0, vmax=1.15)
    ygate = 1.9 + (0.80 / 1.15) * 4.1
    ax.plot([1.0, 10.2], [ygate, ygate], color=ORANGE, lw=1.8, ls="--")
    ax.text(8.05, ygate + 0.22, "kapı  ρ = 0,80", fontsize=12.5, weight="bold", color=ORANGE)

    box(ax, 10.9, 3.55, 4.4, 2.55, "Dondurulmuş koşu kartı", "", color=BLUE, ts=15)
    ax.text(11.15, 5.58, "epoch:  30 (yalnız normal\nvalidation NLL ile seçildi)\n\n"
                         "threshold:  10,0940\n\n"
                         "eğitilmiş–rastgele ρ:  0,3288\neğitilmiş–genlik ρ:  0,3495",
            fontsize=12.5, color=INK, va="top", linespacing=1.4)
    box(ax, 10.9, 1.55, 4.4, 1.6, "Ama dikkat", "Bu kapıyı geçmek 'model başarılı'\ndemek değil — yalnız "
        "'bu kestirme\nyolu kullanmıyor' demek.", color=ORANGE, fill=LIGHTORANGE, ts=13.5, bs=11.5)
    foot(ax, "Projedeki tek gerçek 'iyileşme' hikâyesi: 0,964'ten 0,329'a. Tespit performansı ayrı bir sorudur.")
    save(fig, "F27_genlik_kapisi")


# ===========================================================================
# S29 - ALFA group CV
# ===========================================================================
def s29():
    fig, ax = canvas("ALFA'nın son sözü: grup-güvenli çapraz doğrulama",
                     "Oturum-ailesi bazlı dört bağımsız grup — RflyMAD ile aynı sıkı sınav")
    folds = [("Grup 0", 0.444, 0.0, 0.0, True, "kapıyı geçti\nama SIFIR ayrım"),
             ("Grup 1", 0.409, 54.5, 50.0, False, "genlik kapısında kaldı"),
             ("Grup 2", 0.545, 18.2, 25.0, False, "genlik kapısında kaldı"),
             ("Grup 3", 0.521, 100.0, 83.3, False, "yakalama %100\nama normal alarm %83,3")]
    x0, slot = 0.9, 3.65
    for i, (t, auc, rec, fa, passed, note) in enumerate(folds):
        x = x0 + i * slot
        col = BLUE if passed else RED
        box(ax, x, 1.6, 3.3, 5.3, t, "", color=col, ts=16, align="center")
        ax.text(x + 1.65, 6.15, "uçuş ROC-AUC", fontsize=11.5, color=MUTED, ha="center")
        ax.text(x + 1.65, 5.70, tr(auc, 3), fontsize=24, weight="bold", color=col, ha="center", va="center")
        ax.text(x + 1.65, 5.05, "yakalama", fontsize=11.5, color=MUTED, ha="center")
        ax.text(x + 1.65, 4.62, f"%{tr(rec, 1)}", fontsize=19, weight="bold", color=col, ha="center", va="center")
        ax.text(x + 1.65, 4.05, "normal uçuşta alarm", fontsize=11.5, color=MUTED, ha="center")
        ax.text(x + 1.65, 3.62, f"%{tr(fa, 1)}", fontsize=19, weight="bold", color=col, ha="center", va="center")
        ax.add_patch(Rectangle((x + 0.2, 1.85), 2.9, 1.35, facecolor=LIGHTNAVY if passed else LIGHTRED, edgecolor="none"))
        ax.text(x + 1.65, 2.52, note, fontsize=12, weight="bold", color=col, ha="center", va="center", linespacing=1.4)
    ax.text(8.1, 1.05, "Dört gruptan yalnız biri genlik kapısını geçti — ve o grupta model hiçbir ayrım yapmadı. "
                       "ALFA'da da karar aynı: hedef karşılanmadı.",
            fontsize=13.5, weight="bold", color=INK, ha="center")
    save(fig, "F29_alfa_group_cv")


# ===========================================================================
# S30 - operating point
# ===========================================================================
def s30():
    fig, ax = canvas("Çalışma noktası: yakalama ve alarm yükü aynı yerde buluşmadı",
                     "RflyMAD — 144 aday noktadan 54'ü ön-kayıtlı olarak değerlendirildi")
    # Only the three preregistered points that exist in the B0 report are drawn.
    # No interpolated curve: intermediate operating points were not published.
    axc = fig.add_axes([0.075, 0.15, 0.52, 0.55])
    pts = [(0.654, 0.4327, "2-of-3 persistence\n(referans)", BLUE),
           (5.884, 0.4829, "CUSUM referansı", VIOLET),
           (9.153, 0.5871, "en yüksek yakalama", RED)]
    axc.plot([p[0] for p in pts], [p[1] for p in pts], color=ORANGE, lw=1.8, ls=":", zorder=1)
    for fa, rc, lab, col in pts:
        axc.plot([fa], [rc], marker="o", ms=11, mfc=WHITE, mec=col, mew=2.6, zorder=3)
    axc.set_xscale("log")
    axc.set_xlim(0.35, 22)
    axc.set_xlabel("yanlış olay / normal flight-hour  (log ölçek)", fontsize=12.5)
    axc.set_ylabel("event recall", fontsize=12.5)
    axc.set_ylim(0.30, 0.70)
    axc.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axc.set_yticklabels(["%30", "%40", "%50", "%60", "%70"], fontsize=11.5)
    axc.tick_params(axis="x", labelsize=11.5)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.spines["left"].set_color(GRID); axc.spines["bottom"].set_color(GRID)
    axc.annotate("%43,27 @ 0,654", xy=(0.654, 0.4327), xytext=(0.40, 0.545),
                 fontsize=13, weight="bold", color=NAVY,
                 arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.2))
    axc.annotate("%48,29 @ 5,884", xy=(5.884, 0.4829), xytext=(1.15, 0.355),
                 fontsize=12.5, weight="bold", color=VIOLET,
                 arrowprops=dict(arrowstyle="-", color=VIOLET, lw=1.2))
    axc.annotate("%58,71 @ 9,153", xy=(9.153, 0.5871), xytext=(1.6, 0.645),
                 fontsize=13, weight="bold", color=RED,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
    axc.text(0.02, 0.03, "yalnız ön-kayıtlı üç referans nokta — ara noktalar raporlanmadı",
             fontsize=10.5, color=MUTED, transform=axc.transAxes)

    box(ax, 9.6, 4.6, 5.7, 2.3, "Referans çalışma noktası", "2-of-3 persistence", color=BLUE,
        fill=LIGHTNAVY, ts=15, bs=12)
    ax.text(11.2, 5.35, "%43,27", fontsize=30, weight="bold", color=NAVY, ha="center", va="center")
    ax.text(11.2, 4.90, "event recall", fontsize=11.5, color=MUTED, ha="center")
    ax.text(13.9, 5.35, "0,654", fontsize=30, weight="bold", color=NAVY, ha="center", va="center")
    ax.text(13.9, 4.90, "yanlış olay / saat", fontsize=11.5, color=MUTED, ha="center")

    box(ax, 9.6, 2.2, 5.7, 2.05, "Daha yüksek yakalama seçeneği", "", color=RED, fill=LIGHTRED, ts=15)
    ax.text(12.45, 3.20, "+15 puan recall  →  alarm yükü 14 kat",
            fontsize=15.5, weight="bold", color=RED, ha="center", va="center")
    ax.text(12.45, 2.60, "%58,71 recall  @  9,153 yanlış olay / saat", fontsize=12.5, color=INK, ha="center")
    foot(ax, "Operatör açısından saatte 9 yanlış olay, sistemin ilk gün kapatılması demektir. "
             "Sıralama sinyali var, işletilebilir çalışma noktası yok.")
    save(fig, "F30_calisma_noktasi")


# ===========================================================================
# S31 - blind spots
# ===========================================================================
def s31():
    fig, ax = canvas("Kör noktalar: ortalama yanıltıyor",
                     "Aynı model, aynı eşik — yakalamanın tamamı tek bir yerde toplanıyor")
    ax.text(0.9, 7.05, "Arıza ailesine göre", fontsize=14.5, weight="bold", color=INK)
    for i, (lab, v, col) in enumerate([("Motor", 60.08, AQUA), ("Sensor", 5.45, RED)]):
        y = 6.35 - i * 0.85
        ax.add_patch(Rectangle((2.9, y - 0.22), (v / 100) * 4.4, 0.48,
                               facecolor=col, edgecolor="none"))
        ax.text(2.75, y, lab, fontsize=13.5, color=INK, ha="right", va="center")
        ax.text(2.95 + (v / 100) * 4.4 + 0.15, y, f"%{tr(v, 2)}", fontsize=16, weight="bold", color=col, va="center")

    ax.text(8.6, 7.05, "Veri alanına göre", fontsize=14.5, weight="bold", color=INK)
    for i, (lab, v, col, delay) in enumerate([("HIL", 56.85, BLUE, "—"), ("SIL", 38.78, VIOLET, "17,4 s"),
                                              ("Real (gerçek uçuş)", 7.81, RED, "41,2 s")]):
        y = 6.35 - i * 0.85
        ax.add_patch(Rectangle((11.2, y - 0.22), (v / 100) * 2.9, 0.48,
                               facecolor=col, edgecolor="none"))
        ax.text(11.05, y, lab, fontsize=13.5, color=INK, ha="right", va="center")
        ax.text(11.25 + (v / 100) * 2.9 + 0.15, y, f"%{tr(v, 2)}", fontsize=16, weight="bold", color=col, va="center")
        ax.text(15.3, y, "gecikme " + delay, fontsize=12, color=MUTED, ha="right", va="center")

    box(ax, 0.9, 2.55, 7.0, 1.55, "Motor arızasında gecikme:  0,1 saniye", "", color=AQUA,
        fill=LIGHTAQUA, ts=15)
    ax.text(1.15, 3.35, "Model komut–tepki ilişkisinin bozulduğu ani arızaları görüyor;\n"
                        "yavaş gelişen sensör bozulmalarını görmüyor.", fontsize=12.5, color=INK, va="top", linespacing=1.45)

    box(ax, 8.3, 2.55, 7.0, 1.55, "Bunu düzeltmeyi denedik", "", color=ORANGE, fill=LIGHTORANGE, ts=15)
    ax.text(8.55, 3.35, "6 ön-kayıtlı dayanıklılık adayı tarandı: gerçek uçuş yakalaması\n"
                        "%14,3 → %28,1 çıktı (hedef %40) ama genel yakalama %60,4 → %54,6 düştü.",
            fontsize=12.5, color=INK, va="top", linespacing=1.45)

    ax.text(0.9, 1.85, "Belirli bir domain iyileşirken genel sistem kötüleşti — bu bir model ayarı sorunu değil, "
                       "feature ve gözlenebilirlik sorunudur.",
            fontsize=13, weight="bold", color=INK)
    foot(ax, "Sıradaki işin adresi burası: gerçek uçuş verisinde sensör arızasının fiziksel izini görünür kılmak.")
    save(fig, "F31_kor_noktalar")


# ===========================================================================
# S32 - any-window illusion
# ===========================================================================
def s32():
    fig, ax = canvas("Aynı model, iki farklı soru",
                     "Tek bir metriğin bir sonucu nasıl tamamen ters gösterebileceğinin kanıtı")
    ax.text(4.4, 7.00, "Soru A — \"Bu uçuşta en az bir alarm var mı?\"", fontsize=15, weight="bold", color=RED, ha="center")
    # Both bars deliberately the same hue: the point is that the two groups are
    # indistinguishable under this metric.
    bars(ax, ["Anomali uçuşları", "Normal uçuşlar"], [82.94, 84.62], [GREY, GREY],
         x0=1.0, x1=7.8, ybase=2.6, ytop=6.2, vmax=100, fmt=lambda v: f"%{tr(v, 2)}")
    ax.text(4.4, 1.85, "dengeli doğruluk  %49,16", fontsize=19, weight="bold", color=RED, ha="center")
    ax.text(4.4, 1.35, "= yazı-tura", fontsize=13, color=MUTED, ha="center")

    ax.text(11.9, 7.00, "Soru B — \"Alarm gerçek arıza aralığında mı başladı?\"", fontsize=15, weight="bold", color=AQUA, ha="center")
    bars(ax, ["event recall", "yanlış olay / saat"], [43.27, 0.654], [BLUE, AQUA],
         x0=8.6, x1=15.3, ybase=2.6, ytop=6.2, vmax=100,
         fmt=lambda v: f"%{tr(v, 2)}" if v > 1 else tr(v, 3))
    ax.text(11.9, 1.85, "operatöre anlamlı tek ölçüm", fontsize=15, weight="bold", color=AQUA, ha="center")

    ax.plot([8.1, 8.1], [1.1, 6.9], color=GRID, lw=1.4)
    foot(ax, "\"Model anomali uçuşlarının %83'ünü yakaladı\" teknik olarak yanlış bir cümle değil — ama aynı eşik "
             "normal uçuşların %85'inde de alarm veriyor.")
    save(fig, "F32_any_window_yanilsamasi")


# ===========================================================================
# S34 - decision
# ===========================================================================
def s34():
    fig, ax = canvas("Karar ve sıradaki iş", "")
    box(ax, 0.9, 5.65, 14.4, 1.5, "", "", color=NAVY, fill=LIGHTNAVY, lw=2.2)
    ax.text(8.1, 6.62, "Karar: hedef karşılanmadı.", fontsize=23, weight="bold", color=NAVY, ha="center", va="center")
    ax.text(8.1, 6.05, "Mühürlü final test seti (553 uçuş) açılmadı.", fontsize=15, color=INK, ha="center", va="center")

    cols = [("Güçlü olduğu yer", ["Motor / komut–tepki arızaları",
                                 "Simülasyon ve HIL domainleri",
                                 "Fiziksel tutarlılık kuralları"], AQUA, LIGHTAQUA),
            ("Darboğaz", ["Yanlış alarm yükü",
                          "Gerçek uçuş domain farkı",
                          "Sensör arızası gözlenebilirliği"], RED, LIGHTRED),
            ("Sıradaki iş", ["Bağımsız normal uçuş süresini artırmak",
                             "Zaman etiketli kontrollü arıza kampanyası",
                             "Arıza ailesine ve domain'e özel policy"], BLUE, LIGHTNAVY)]
    for i, (t, items, col, tint) in enumerate(cols):
        x = 0.9 + i * 4.95
        box(ax, x, 2.3, 4.55, 3.0, t, "", color=col, fill=tint, ts=16)
        for j, it in enumerate(items):
            ax.text(x + 0.28, 4.55 - j * 0.75, "•  " + it, fontsize=12.5, color=INK, va="top", linespacing=1.35, wrap=True)

    ax.text(8.1, 1.55, "Sıradaki veri kampanyasının neye ihtiyaç duyduğunu artık tahmin etmiyoruz — ölçtük.",
            fontsize=15, weight="bold", color=NAVY, ha="center")
    ax.text(8.1, 1.05, "Yanlış bir sistemi sahaya vermektense, iyi belgelenmiş bir olumsuz sonuç vermeyi tercih ettik.",
            fontsize=13, color=MUTED, ha="center")
    save(fig, "F34_karar")


def main() -> None:
    configure()
    print("Rendering final deck figures ->", OUT)
    for fn in (s03, s04, s05, s06, s08, s09, s10, s11, s12, s13, s14, s15, s16,
               s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s29,
               s30, s31, s32, s34):
        fn()
    print("done.")


if __name__ == "__main__":
    main()
