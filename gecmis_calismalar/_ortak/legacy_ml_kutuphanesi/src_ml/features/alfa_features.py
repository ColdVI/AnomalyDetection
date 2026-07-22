"""ALFA Silver -> ML feature tablosu (ML-0 fazi).

Girdi: parse_alfa.py'nin urettigi Silver tablo (artik nav_info-errors,
path_dev, vfr_hud ve velocity kolonlarini da tasiyor). Cikti: ucus-ici
hesaplanan, sizinti-suz (leakage-free) feature'lar + kimlik kolonlari
(source_id, label, t_rel_s) -- kimlik kolonlari split/degerlendirme icindir,
FEATURE_COLUMNS'a girmez.

Feature katmanlari (FableChat karari):
  K1 Hazir residual'lar : alt_error, aspd_error, xtrack_error, path_dev_*
     (otopilotun KENDI hesapladigi hata sinyalleri -- en az gurultulu kaynak)
  K2 Hesaplanan residual'lar: command-tracking hatalari (wrap-aware yaw),
     koordineli donus residual'i, enerji tutarliligi, GPS kinematik residual
  K3 Zamansal: rate'ler, gecmise-bakan rolling RMS/std/max, CUSUM, EWMA sapmasi,
     spektral bant enerjisi (osilasyon imzasi), donma sayaclari
  K4 Baglam: t_rel_s, airspeed_available (ALFA'nin bilinen airspeed=0 artifact'i)

Mutlak lat/lon/timestamp ASLA feature degildir (shortcut learning: model
"lon=-79 -> ALFA" ogrenir). Yerine adim mesafesi / hiz / kalkis noktasina uzaklik.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ml.features.temporal import (
    angular_error_deg,
    consecutive_unchanged,
    cusum,
    cusum_kwargs,
    ewma_deviation,
    haversine_m,
    rate_per_s,
    rolling_stats,
    spectral_band_energy,
)

logger = logging.getLogger(__name__)

G_MPS2 = 9.80665

# ALFA ~4 Hz: 2 sn ~= 8 satir, 5 sn ~= 20 satir.
WIN_2S = 8
WIN_5S = 20

# Kimlik/degerlendirme kolonlari -- modele ASLA verilmez.
ID_COLUMNS = ["source_id", "label", "t_rel_s"]
CUSUM_SOURCE_COLUMNS = ["roll_error", "pitch_error", "alt_error", "xtrack_error"]


def _flight_features(g: pd.DataFrame, *, cusum_baselines: dict | None = None) -> pd.DataFrame:
    """Tek bir ucusun (source_id) feature'larini hesaplar. g zaman-siralidir."""
    out = pd.DataFrame(index=g.index)
    t_s = g["ts_ns"].astype(float) / 1e9
    out["t_rel_s"] = t_s - t_s.iloc[0]

    # --- K2: command-tracking hatalari ---
    out["roll_error"] = g["roll_measured"] - g["roll_commanded"]
    out["pitch_error"] = g["pitch_measured"] - g["pitch_commanded"]
    out["airspeed_error"] = g["airspeed_measured"] - g["airspeed_commanded"]
    out["yaw_error"] = angular_error_deg(g["yaw_measured"], g["yaw_commanded"])
    if "velocity_measured" in g.columns and "velocity_commanded" in g.columns:
        out["velocity_error"] = g["velocity_measured"] - g["velocity_commanded"]
    for c in ["roll_error", "pitch_error", "yaw_error", "airspeed_error"]:
        out[f"abs_{c}"] = out[c].abs()

    # --- K1: otopilotun hazir residual'lari ---
    for c in ["alt_error", "aspd_error", "xtrack_error", "wp_dist"]:
        if c in g.columns:
            out[c] = g[c]
    if all(c in g.columns for c in ["path_dev_x", "path_dev_y", "path_dev_z"]):
        out["path_dev_mag"] = np.sqrt(g["path_dev_x"] ** 2 + g["path_dev_y"] ** 2 + g["path_dev_z"] ** 2)

    # --- K3: rate'ler (ucus-ici birinci fark / dt) ---
    out["roll_rate"] = rate_per_s(g["roll_measured"], t_s)
    out["pitch_rate"] = rate_per_s(g["pitch_measured"], t_s)
    out["yaw_rate"] = rate_per_s(g["yaw_measured"], t_s, angular=True)
    out["altitude_rate"] = rate_per_s(g["alt"], t_s)
    out["airspeed_rate"] = rate_per_s(g["airspeed_measured"], t_s)
    out["roll_error_rate"] = rate_per_s(out["roll_error"], t_s)
    out["pitch_error_rate"] = rate_per_s(out["pitch_error"], t_s)
    out["yaw_error_rate"] = rate_per_s(out["yaw_error"], t_s, angular=True)
    # Komut rate'leri: keskin komut degisimi sirasindaki tracking hatasi normal,
    # sabit komutta buyuyen hata anormaldir -- modele bu baglami verir.
    out["roll_command_rate"] = rate_per_s(g["roll_commanded"], t_s)
    out["pitch_command_rate"] = rate_per_s(g["pitch_commanded"], t_s)
    out["yaw_command_rate"] = rate_per_s(g["yaw_commanded"], t_s, angular=True)

    # --- K2: GPS kinematigi (ham lat/lon yerine turevler) ---
    step_m = pd.Series(
        haversine_m(g["lat"].shift(), g["lon"].shift(), g["lat"], g["lon"]), index=g.index)
    out["gps_step_m"] = step_m
    dt = t_s.diff()
    out["gps_speed_calc_mps"] = (step_m / dt).where(dt > 0)
    out["gps_accel_mps2"] = rate_per_s(out["gps_speed_calc_mps"], t_s)
    out["dist_from_start_m"] = pd.Series(
        haversine_m(g["lat"].iloc[0], g["lon"].iloc[0], g["lat"], g["lon"]), index=g.index)
    # GPS'ten hesaplanan hiz ile otopilotun ground_speed'i tutarli mi?
    if "ground_speed_ms" in g.columns:
        out["gps_speed_residual"] = out["gps_speed_calc_mps"] - g["ground_speed_ms"]

    # --- K2: koordineli donus residual'i (sabit kanat fizigi) ---
    # Beklenen donus hizi: psi_dot = g * tan(roll) / V. Ucak yatiyorsa donmek
    # ZORUNDADIR; yatip donmuyorsa sensor yalan soyluyor ya da yuzey arizali.
    # Sadece V > 5 m/s iken anlamli (yerde/stall'da fizik gecerli degil).
    v = g["ground_speed_ms"] if "ground_speed_ms" in g.columns else g["airspeed_measured"]
    expected_yaw_rate = np.degrees(G_MPS2 * np.tan(np.radians(g["roll_measured"])) / v.where(v > 5.0))
    out["turn_residual"] = out["yaw_rate"] - expected_yaw_rate

    # --- K2: enerji tutarliligi ---
    # Ozgul mekanik enerji E = g*h + V^2/2; seyirde surekli dusuyorsa ve throttle
    # yuksekse aciklanamayan enerji kaybi (motor arizasi SUPHE skoru, kanit degil).
    out["specific_energy"] = G_MPS2 * g["alt"] + 0.5 * v.astype(float) ** 2
    out["energy_rate"] = rate_per_s(out["specific_energy"], t_s)
    if "throttle" in g.columns:
        out["throttle"] = g["throttle"]
        out["energy_rate_x_throttle"] = out["energy_rate"] * g["throttle"]
    if "climb_rate_ms" in g.columns:
        out["climb_rate_ms"] = g["climb_rate_ms"]
        # baro/GPS dikey tutarliligi: hesaplanan irtifa turevi vs hud climb
        out["climb_residual"] = out["altitude_rate"] - g["climb_rate_ms"]

    # --- K3: gecmise-bakan rolling istatistikler ---
    for col, wins in [("roll_error", (WIN_2S, WIN_5S)), ("pitch_error", (WIN_2S, WIN_5S)),
                      ("yaw_error", (WIN_5S,)), ("airspeed_error", (WIN_5S,)),
                      ("xtrack_error", (WIN_5S,)), ("turn_residual", (WIN_5S,))]:
        if col not in out.columns:
            continue
        for w in wins:
            sec = 2 if w == WIN_2S else 5
            out = pd.concat([out, rolling_stats(out[col], w, f"{col}_{sec}s", stats=("mean", "std", "max", "rms"))], axis=1)

    # --- K3: CUSUM (yavas/isrararli sapma dedektoru) ---
    for col in ["roll_error", "pitch_error", "alt_error", "xtrack_error"]:
        if col in out.columns:
            cs = cusum(out[col], **cusum_kwargs(cusum_baselines, col))
            out[f"{col}_cusum_pos"] = cs["cusum_pos"]
            out[f"{col}_cusum_neg"] = cs["cusum_neg"]

    # --- K3: EWMA sapmasi + spektral osilasyon imzasi ---
    out["roll_ewma_dev"] = ewma_deviation(g["roll_measured"])
    out["pitch_ewma_dev"] = ewma_deviation(g["pitch_measured"])
    out["roll_spec_energy_5s"] = spectral_band_energy(g["roll_measured"], WIN_5S)
    out["pitch_spec_energy_5s"] = spectral_band_energy(g["pitch_measured"], WIN_5S)

    # --- K4: donma sayaclari + airspeed artifact bayragi ---
    out["airspeed_frozen_count"] = consecutive_unchanged(g["airspeed_measured"])
    out["gps_frozen_count"] = consecutive_unchanged(g["lat"].astype(str) + "," + g["lon"].astype(str))
    # ALFA bilinen artifact: bircok logda airspeed tamamen 0 (sensor yok) --
    # "arıza" degil "sensor mevcudiyeti". Ablation icin ayri bayrak (docs/AGENTS.md).
    out["airspeed_available"] = (g["airspeed_measured"].astype(float) != 0).astype(int)

    # K4 baglam: ayni deger yerde normal / seyirde anormal olabilir. Tam
    # flight_phase yerine basit in_air proxy'si (hiz esigi) -- FableChat Katman 4.
    out["in_air"] = (v.astype(float) > 3.0).astype(int)

    return out


def build_alfa_features(silver: pd.DataFrame, *, cusum_baselines: dict | None = None) -> pd.DataFrame:
    """ALFA Silver -> feature tablosu. Ucus basina bagimsiz hesap, sonra UNION."""
    frames = []
    for source_id, g in silver.sort_values("ts_ns").groupby("source_id", sort=False):
        feats = _flight_features(g.reset_index(drop=True), cusum_baselines=cusum_baselines)
        feats["source_id"] = source_id
        feats["label"] = g["label"].reset_index(drop=True)
        frames.append(feats)
    out = pd.concat(frames, ignore_index=True)
    logger.info("ALFA features: %d satir, %d kolon (%d feature + %d kimlik)",
                len(out), out.shape[1], len(feature_columns(out)), len(ID_COLUMNS))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Modele girecek kolonlar: kimlik kolonlari haric her sey."""
    return [c for c in df.columns if c not in ID_COLUMNS]
