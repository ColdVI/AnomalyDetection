"""Veri olmadan calistirilabilen uctan uca demo.

Bu script HICBIR algoritmayi yeniden yazmaz -- sadece bu klasordeki gercek
proje moduellerini (adsb.*, gecmis_calismalar.anomaly_core.*) sentetik,
fiziksel olarak tutarli ucus verisiyle cagirir. Kendi kodu yalniz (1) sentetik
ucus uretimi ve (2) ekrana yazdirma/ozetlemedir.

Onemli: importlar icin `from adsb.evaluation import ...` scikit-learn'e
(zaten requirements.txt'te olan bir bagimlilik) transitif olarak ihtiyac
duyar -- cunku bu script event-recall/false-alarm hesaplamasini KENDI
YENIDEN YAZMAK yerine projenin gercek degerlendirme moduelunu cagiriyor.
Bkz. demo/README.md.

Calistirma: `python demo/demo_calistir.py` (herhangi bir dizinden).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows konsolunun varsayilan kod sayfasi (cp1252) Turkce'ye ozgu i/s/g
# harflerini (i, s, g) iceremiyor -- stdout'u acikca UTF-8'e zorla ki hangi
# terminalden calistirilirsa calistirilsin cikti hatasiz kalsin.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_DEMO_DIR = Path(__file__).resolve().parent
_INDIVIDUAL_ANIL_ROOT = _DEMO_DIR.parent
if str(_INDIVIDUAL_ANIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_INDIVIDUAL_ANIL_ROOT))

import numpy as np
import pandas as pd

from adsb.features import build_feature_table
from adsb.rules import ResidualRuleScorer, RULE_CHANNELS
from adsb.synthetic import inject_bias, inject_freeze, inject_position_ramp
from adsb.truth import attach_clean_truth_v2
from adsb.evaluation import (
    EpisodeContract,
    event_detection_metrics,
    event_observability_denominators,
    natural_alert_burden,
    truth_event_table,
)
from adsb.diagnostics import magnitude_domination_check, magnitude_only_score
from gecmis_calismalar.anomaly_core.sequential import MultiChannelPageCUSUM, PageCUSUMConfig

SEED = 20260807
DT_S = 1.0
M_PER_DEG_LAT = 111_320.0
N_TRAIN_NORMAL = 40
N_TEST_NORMAL = 20
N_TEST_EVENT = 18
CUSUM_WINDOW_ROWS = 20  # bolum 6'daki genlik-denetimi pencereleri icin


def _section(n: int, title: str) -> None:
    print()
    print(f"===== [{n}/6] {title} =====")


# --------------------------------------------------------------------- 1 ---
def make_flight(flight_id: str, n_rows: int, rng: np.random.Generator,
                 start_lat: float, start_lon: float, start_alt: float,
                 cruise_speed: float) -> pd.DataFrame:
    """Fiziksel olarak tutarli tek bir sentetik ucus uretir.

    "Tutarli" demek: bildirilen ground_speed_ms/track_deg/vertical_rate_ms,
    lat/lon/alt dizisinin turevleriyle (kucuk sensor gurultusu disinda)
    birbirini tutuyor -- normal'i tanimlayan sey bu, adsb.features'daki
    residual formulleri tam olcup dogruluyor.
    """
    t = np.arange(n_rows, dtype=float) * DT_S

    true_track = (rng.uniform(0, 360) + np.cumsum(rng.normal(0, 0.05, n_rows))) % 360.0
    true_speed = np.clip(cruise_speed + np.cumsum(rng.normal(0, 0.05, n_rows)), 80.0, 260.0)
    reported_vrate = 0.4 * np.sin(2 * np.pi * t / 240.0)

    eps_speed = rng.normal(0, 0.15, n_rows)
    eps_track = rng.normal(0, 0.15, n_rows)
    eps_vrate = rng.normal(0, 0.05, n_rows)

    lat = np.empty(n_rows)
    lon = np.empty(n_rows)
    alt = np.empty(n_rows)
    lat[0], lon[0], alt[0] = start_lat, start_lon, start_alt
    for i in range(1, n_rows):
        dist_m = true_speed[i - 1] * DT_S
        bearing_rad = np.radians(true_track[i - 1])
        dlat_deg = (dist_m * np.cos(bearing_rad)) / M_PER_DEG_LAT
        dlon_deg = (dist_m * np.sin(bearing_rad)) / (M_PER_DEG_LAT * np.cos(np.radians(lat[i - 1])))
        lat[i] = lat[i - 1] + dlat_deg
        lon[i] = lon[i - 1] + dlon_deg
        # onceki adimin GERCEK (gurultusuz) dikey hizini entegre et; bildirilen
        # deger asagida kucuk gurultuyle raporlanir -- residual bu farktan dogar.
        alt[i] = alt[i - 1] + (reported_vrate[i - 1] - eps_vrate[i - 1]) * DT_S

    # yavas degisen jeoit/basinc farki -- baro (alt) ve jeometrik (alt_geom_m)
    # irtifa arasindaki dogal, sabit-OLMAYAN fark. altitude_source_residual bu
    # farkin ZAMAN TUREVI oldugu icin (bkz. adsb/features.py), sabit bir fark
    # MAD=0 uretir ve kanal elenir -- bu yuzden burada kasitli olarak yavas
    # bir sinus + kucuk gurultu var.
    geoid_offset = 3.0 * np.sin(2 * np.pi * t / 900.0) + rng.normal(0, 0.1, n_rows)

    return pd.DataFrame({
        "flight_id": flight_id,
        "timestamp_utc": t,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "alt_geom_m": alt + geoid_offset,
        "ground_speed_ms": true_speed + eps_speed,
        "track_deg": (true_track + eps_track) % 360.0,
        "vertical_rate_ms": reported_vrate,
    })


def _make_pool(prefix: str, n_flights: int, rng: np.random.Generator) -> list[pd.DataFrame]:
    flights = []
    for i in range(n_flights):
        n_rows = int(rng.integers(150, 260))
        flights.append(make_flight(
            f"{prefix}_{i:03d}", n_rows, rng,
            start_lat=rng.uniform(38.0, 41.0), start_lon=rng.uniform(28.0, 33.0),
            start_alt=rng.uniform(3000.0, 9000.0), cruise_speed=rng.uniform(140.0, 220.0),
        ))
    return flights


def generate_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_normal = pd.concat(
        [attach_clean_truth_v2(f) for f in _make_pool("train", N_TRAIN_NORMAL, rng)],
        ignore_index=True,
    )
    test_normal = pd.concat(
        [attach_clean_truth_v2(f) for f in _make_pool("testnorm", N_TEST_NORMAL, rng)],
        ignore_index=True,
    )

    event_types = ["freeze", "bias", "position_ramp"]
    injected = []
    for i, clean in enumerate(_make_pool("testevt", N_TEST_EVENT, rng)):
        scenario = event_types[i % len(event_types)]
        onset_frac = float(rng.uniform(0.4, 0.6))
        if scenario == "freeze":
            # sensor donmasi: bildirilen dikey hiz onset'ten sonra sabit kalir.
            injected.append(inject_freeze(clean, "vertical_rate_ms", onset_frac=onset_frac))
        elif scenario == "bias":
            # ani sapma: bildirilen yer hizina +4 sigma eklenir.
            injected.append(inject_bias(clean, "ground_speed_ms", sigma_mult=4.0, onset_frac=onset_frac))
        else:
            # sinsi (stealthy) konum kaymasi: bildirilen hiz/rota DEGISMEZ,
            # yalniz konum kayar -- speed/heading residual'in yakalamasi gereken durum.
            injected.append(inject_position_ramp(
                clean, meters_per_s=2.0, bearing_deg=float(rng.uniform(0, 360)), onset_frac=onset_frac,
            ))
    test_event = pd.concat(injected, ignore_index=True)
    return train_normal, test_normal, test_event


def section_1_data(rng: np.random.Generator):
    _section(1, "Veri uretimi (sentetik, fiziksel olarak tutarli ucuslar)")
    train_normal, test_normal, test_event = generate_data(rng)
    print(f"yalniz-normal egitim havuzu : {train_normal.flight_id.nunique():>3} ucus, {len(train_normal):>5} satir")
    print(f"normal test havuzu          : {test_normal.flight_id.nunique():>3} ucus, {len(test_normal):>5} satir  (yanlis alarm yuku buradan olculur)")
    print(f"olayli test havuzu          : {test_event.flight_id.nunique():>3} ucus, {len(test_event):>5} satir  (freeze/bias/position_ramp, 6+6+6)")
    print("ONEMLI: enjeksiyon YALNIZ test havuzuna uygulandi -- egitim (train_normal)")
    print("        hicbir enjekte edilmis satir/pencere gormedi (novelty-detection sozlesmesi).")
    return train_normal, test_normal, test_event


# --------------------------------------------------------------------- 2 ---
def section_2_features(train_normal, test_normal, test_event):
    _section(2, "Feature -- adsb.features.build_feature_table")
    train_feat = build_feature_table(train_normal)
    test_normal_feat = build_feature_table(test_normal)
    test_event_feat = build_feature_table(test_event)
    print("kanal basina medyan |residual| (yalniz-normal egitim havuzu):")
    for channel in RULE_CHANNELS:
        print(f"  {channel:<28} {np.nanmedian(np.abs(train_feat[channel])):.4f}")
    print("residual'lar OGRENILMIS DEGIL -- aritmetik ozdeslik (bildirilen deger eksi")
    print("konum/irtifa turevinden hesaplanan beklenen deger). Residual sifira yakinsa")
    print("kanal tutarlidir; residual'in BUYUKLUGU tek basina hicbir sey ifade etmez.")
    return train_feat, test_normal_feat, test_event_feat


# --------------------------------------------------------------------- 3 ---
def section_3_rule_scorer(train_feat):
    _section(3, "Kural skorlayici -- adsb.rules.ResidualRuleScorer")
    scorer = ResidualRuleScorer()
    scorer.fit(train_feat)
    print("kalibrasyon (yalniz-normal egitim havuzundan, kanal basina median/MAD):")
    for channel, cal in scorer.calibration_.items():
        print(f"  {channel:<28} median={cal['median']:+.4f}  MAD={cal['mad']:.4f}")
    if scorer.excluded_channels_:
        print(f"  haric tutulan kanallar (MAD=0): {scorer.excluded_channels_}")
    print("Etiketler bu adimda KULLANILMADI -- fit() yalniz yalniz-normal train satirlarini")
    print("goruyor, hangi satirin 'olayli' oldugunu hic bilmiyor.")
    return scorer


# --------------------------------------------------------------------- 4 ---
def section_4_cusum(train_feat, test_normal_feat, test_event_feat):
    _section(4, "Ardisik karar -- gecmis_calismalar.anomaly_core.sequential.MultiChannelPageCUSUM")
    for feat in (train_feat, test_normal_feat, test_event_feat):
        feat["evaluable"] = feat[RULE_CHANNELS].notna().all(axis=1)

    config = PageCUSUMConfig(
        channels=tuple(RULE_CHANNELS),
        reference_shift_z=4.0,
        z_clip=8.0,
        max_gap_s=10.0,
        time_col="timestamp_utc",  # ADS-B tarafinda varsayilan "timestamp_s" DEGIL
        evaluable_col="evaluable",  # bu kolonu biz uretiyoruz (kutuphane uretmiyor)
    )
    cusum = MultiChannelPageCUSUM(config)
    cusum.fit(train_feat.loc[train_feat["evaluable"]])

    train_scores = cusum.score(train_feat)
    threshold = float(np.quantile(
        train_scores.loc[train_scores["cusum_evaluable"], "cusum_score"], 0.999,
    ))
    print(f"esik = yalniz-normal egitim skorlarinin %99.9 yuzdeligi = {threshold:.3f}")
    print("Bu esik test/olayli sonuclar hic gorulmeden donduruldu (asagidaki bolum 5,")
    print("bu esigin degistirilmedigi TEK bir degerlendirme calistirmasidir).")

    test_normal_scores = cusum.score(test_normal_feat)
    test_event_scores = cusum.score(test_event_feat)
    return cusum, threshold, test_normal_scores, test_event_scores


# --------------------------------------------------------------------- 5 ---
def section_5_evaluation(test_normal_feat, test_event_feat, threshold,
                          test_normal_scores, test_event_scores):
    _section(5, "Degerlendirme -- adsb.evaluation (aynı dondurulmus esikte)")

    alarm_normal = ((test_normal_scores["cusum_score"] > threshold)
                     & test_normal_scores["cusum_evaluable"]).to_numpy()
    alarm_event = ((test_event_scores["cusum_score"] > threshold)
                    & test_event_scores["cusum_evaluable"]).to_numpy()

    meta_normal = test_normal_feat[["flight_id", "timestamp_utc"]].copy()
    meta_normal["t_start"] = meta_normal["timestamp_utc"]
    meta_normal["t_end"] = meta_normal["timestamp_utc"] + DT_S
    burden = natural_alert_burden(meta_normal, alarm_normal, contract=EpisodeContract())

    print("yanlis alarm yuku (normal test havuzu, ayni dondurulmus esik):")
    print(f"  alarm veren ucus orani     : {burden['alerted_flight_fraction']:.3f}"
          f"  ({burden['n_alerted_flights']}/{burden['n_scoreable_flights']} ucus)")
    print(f"  alarm episode / ucus-saati : {burden['alert_episodes_per_scoreable_flight_hour']:.3f}"
          f"  ({burden['n_alert_episodes']} episode / {burden['scoreable_flight_hours']:.2f} saat)")

    events_all = truth_event_table(test_event_feat)
    denom = event_observability_denominators(events_all)
    events_observable = events_all.loc[events_all["observable_eligible"]].reset_index(drop=True)

    meta_event = test_event_feat[["flight_id", "timestamp_utc"]].copy()
    meta_event["t_start"] = meta_event["timestamp_utc"]
    meta_event["t_end"] = meta_event["timestamp_utc"] + DT_S
    detection = event_detection_metrics(events_observable, meta_event, alarm_event)

    print("\nolay yakalama (recall), gozlenebilirligi dogrulanmis olaylar uzerinden:")
    print(f"  gozlenebilir olay sayisi    : {denom['n_observable_eligible_events']} / {denom['n_declared_events']} enjekte edilen")
    print(f"  event_recall                : {detection['event_recall']:.3f}"
          f"  ({detection['n_detected_events']}/{detection['n_events']})")
    print(f"  medyan ilk-alarm gecikmesi  : {detection['first_alarm_delay_s']['median']:.1f} s"
          f"  (p95={detection['first_alarm_delay_s']['p95']:.1f} s, n={detection['first_alarm_delay_s']['n_detected']})")

    per_event = pd.DataFrame(detection["per_event"]).merge(
        events_observable[["event_id", "event_type"]], on="event_id", how="left",
    )
    print("\nsenaryo kirilimi:")
    for event_type, group in per_event.groupby("event_type"):
        print(f"  {event_type:<15} {int(group['detected'].sum())}/{len(group)} yakalandi")

    print("\nUYARI: tek basina recall anlamsizdir -- esik dusurulerek recall istenen")
    print("       seviyeye cikarilabilir, ama bu ayni oranda yanlis alarmi da yukseltir.")
    print("       Bu ikisi HER ZAMAN birlikte raporlanmali (bkz. README.md kural #2).")


# --------------------------------------------------------------------- 6 ---
def section_6_magnitude_audit(scorer, test_normal_feat, test_event_feat):
    _section(6, "Genlik-baskinligi denetimi -- adsb.diagnostics.magnitude_domination_check")

    combined = pd.concat([test_normal_feat, test_event_feat], ignore_index=True)
    windows, masks, window_penalty_means = [], [], []
    penalties = scorer.row_penalties(combined)
    for _, group in combined.groupby("flight_id", sort=False):
        values = group[RULE_CHANNELS].to_numpy(dtype=float)
        pen = penalties.loc[group.index].to_numpy()
        n_windows = len(values) // CUSUM_WINDOW_ROWS
        for i in range(n_windows):
            block = values[i * CUSUM_WINDOW_ROWS:(i + 1) * CUSUM_WINDOW_ROWS]
            windows.append(np.nan_to_num(block, nan=0.0))
            masks.append(np.isfinite(block))
            window_penalty_means.append(
                pen[i * CUSUM_WINDOW_ROWS:(i + 1) * CUSUM_WINDOW_ROWS].mean()
            )
    X = np.stack(windows)
    M = np.stack(masks)
    rule_window_scores = np.array(window_penalty_means)

    rng = np.random.default_rng(1)
    random_baseline = rng.normal(size=len(X))

    print(f"({len(X)} pencere, {CUSUM_WINDOW_ROWS} satir/pencere, normal+olayli test havuzlarindan)\n")

    # (a) genlige duyarli bir skorlayici -- KASITLI OLARAK ham genligin kendisi.
    amplitude_scores = magnitude_only_score(X, M)
    report_a = magnitude_domination_check(amplitude_scores, random_baseline, X, M)
    verdict_a = "ISARETLENDI" if report_a["magnitude_domination_flagged"] else "gecti"
    print(f"(a) genlige duyarli skorlayici : rho(egitilmis, ham-genlik) = "
          f"{report_a['rho_trained_vs_magnitude']:.3f}  -> {verdict_a}")

    # (b) kural skorlayicisinin pencere skoru (z-skor esik-asimi, kirpili -- ham genlik degil).
    report_b = magnitude_domination_check(rule_window_scores, random_baseline, X, M)
    verdict_b = "ISARETLENDI" if report_b["magnitude_domination_flagged"] else "gecti"
    print(f"(b) kural skorlayicisi (pencere): rho(egitilmis, ham-genlik) = "
          f"{report_b['rho_trained_vs_magnitude']:.3f}  -> {verdict_b}")

    print("\nGercek deneylerde (bu demoda degil) UAV-SEAD veri kumesinde LSTM-AE, Dense-AE")
    print("ve USAD ucunun de (a) tarafinda ciktigini not dusuyoruz -- bkz. BULGU_KOD_ESLEMESI.md.")


def main() -> None:
    start = time.monotonic()
    rng = np.random.default_rng(SEED)

    print("individual_anil/demo/demo_calistir.py -- veri gerektirmeyen uctan uca demo")
    print("(bkz. demo/README.md -- bu demo gercek proje moduellerini sentetik veriyle calistirir)")

    train_normal, test_normal, test_event = section_1_data(rng)
    train_feat, test_normal_feat, test_event_feat = section_2_features(
        train_normal, test_normal, test_event,
    )
    scorer = section_3_rule_scorer(train_feat)
    cusum, threshold, test_normal_scores, test_event_scores = section_4_cusum(
        train_feat, test_normal_feat, test_event_feat,
    )
    section_5_evaluation(
        test_normal_feat, test_event_feat, threshold, test_normal_scores, test_event_scores,
    )
    section_6_magnitude_audit(scorer, test_normal_feat, test_event_feat)

    elapsed = time.monotonic() - start
    print()
    print("=" * 70)
    print(f"Demo tamamlandi ({elapsed:.1f} s).")
    print("BU SAYILAR SENTETIK VERIYE AITTIR, CALISMANIN BULGUSU DEGILDIR.")
    print("Gercek veri kumeleri, deneyler ve sonuclar icin: ../BULGU_KOD_ESLEMESI.md,")
    print("../raporlar/decisions.md ve ../raporlar/ altindaki raporlar.")
    print("=" * 70)


if __name__ == "__main__":
    main()
