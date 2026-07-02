"""UAV Attack Silver -> ML feature tablosu (ML-0 fazi).

Girdi: parse_uav_attack.py Silver'i (artik vel_m_s / vel_n/e/d / cog_rad /
fix_type kolonlarini da tasiyor). Ayni feature semantigi UAV-SEAD gibi diger
PX4 kaynaklarina da uygulanabilir (build_px4_features'i dogrudan kullanirlar) --
"ortak residual semantigi, ayri model instance'lari" karari.

Feature katmanlari:
  K2 GPS kinematigi   : gps_step_m, hesaplanan hiz/ivme, log1p hiz, rota degisimi,
                        GPS-hiz residual'i (hesaplanan vs receiver vel_m_s)
  K2 GPS sagligi      : eph/epv/hdop/vdop/satellites/jamming/noise + delta'lari
  K2 Batarya          : guc (V*I), remaining tuketim tutarliligi
  K3 Zamansal         : gecmise-bakan rolling, CUSUM (stealthy spoofing), donma
  K4 Missingness      : attitude/battery/gps-health eksiklik bayraklari + stale
                        sayaclari. DIKKAT: Ping DoS satirlarinin ~%36'sinda
                        attitude eksik -- gercek saldiri imzasi ile merge
                        artifact'ini ayirt etmek icin ablation SART (Model A
                        dahil / Model B haric). Bkz. docs/AGENTS.md.

Mutlak lat/lon/timestamp feature degildir. Zaman ekseni: PX4 "timestamp"
(acilistan beri us, monotonik) -> t_rel_s. timestamp_utc GUVENILMEZ
(cogu logda sifir/tekrarli).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ml.features.temporal import (
    consecutive_unchanged,
    cusum,
    haversine_m,
    rate_per_s,
    rolling_stats,
    wrap_angle_deg,
)

logger = logging.getLogger(__name__)

# UAV Attack ~5 Hz: 1 sn ~= 5 satir, 5 sn ~= 25 satir.
WIN_1S = 5
WIN_5S = 25

ID_COLUMNS = ["source_id", "label", "t_rel_s"]

_GPS_HEALTH_COLS = ["eph", "epv", "raw_gps_eph", "raw_gps_epv", "hdop", "vdop",
                    "satellites_used", "s_variance_m_s", "jamming_indicator", "noise_per_ms"]
_ATTITUDE_COLS = ["roll_deg", "pitch_deg", "yaw_deg"]
_BATTERY_COLS = ["voltage_v", "remaining", "current_a"]


def _flight_features(g: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    t_s = g[time_col].astype(float) / 1e6  # PX4 timestamp: us -> s
    out["t_rel_s"] = t_s - t_s.iloc[0]

    # --- K2: konum kinematigi ---
    # Dis mekan (GPS): lat/lon + haversine. Ic mekan (UAV-SEAD External
    # Position ucuslari, mocap): pos_x_m/pos_y_m + oklid. Ayni feature adi,
    # ayni semantik -- "adim mesafesi/hiz", mutlak konum degil.
    has_latlon = "lat" in g.columns and g["lat"].notna().any()
    if has_latlon:
        step_m = pd.Series(
            haversine_m(g["lat"].shift(), g["lon"].shift(), g["lat"], g["lon"]), index=g.index)
        dist_start = pd.Series(
            haversine_m(g["lat"].iloc[0], g["lon"].iloc[0], g["lat"], g["lon"]), index=g.index)
    elif "pos_x_m" in g.columns and g["pos_x_m"].notna().any():
        dx, dy = g["pos_x_m"].astype(float).diff(), g["pos_y_m"].astype(float).diff()
        step_m = np.sqrt(dx ** 2 + dy ** 2)
        dist_start = np.sqrt((g["pos_x_m"] - g["pos_x_m"].iloc[0]) ** 2
                             + (g["pos_y_m"] - g["pos_y_m"].iloc[0]) ** 2)
    else:
        step_m = pd.Series(np.nan, index=g.index)
        dist_start = pd.Series(np.nan, index=g.index)
    out["gps_step_m"] = step_m
    dt = t_s.diff()
    out["gps_speed_calc_mps"] = (step_m / dt).where(dt > 0)
    # Spoofing sicramalari yuz binlerce m/s "hiz" uretir -- SILINMEZ (aradigimiz
    # anomali), ama olcek patlamasin diye log1p versiyonu da verilir.
    out["log_gps_speed"] = np.log1p(out["gps_speed_calc_mps"].clip(lower=0))
    out["gps_accel_mps2"] = rate_per_s(out["gps_speed_calc_mps"], t_s)
    out["vertical_rate_calc"] = rate_per_s(g["alt"], t_s)
    out["dist_from_start_m"] = dist_start

    # Analytical redundancy: konumdan hesaplanan hiz ile receiver'in kendi
    # bildirdigi hiz tutarli mi? Kaba spoofing'de konum sicrar ama vel_m_s sicramaz.
    if "vel_m_s" in g.columns:
        out["gps_speed_residual"] = (out["gps_speed_calc_mps"] - g["vel_m_s"]).abs()
        out["vel_m_s"] = g["vel_m_s"]
    if "vertical_rate_mps" in g.columns:
        out["vertical_rate_residual"] = (out["vertical_rate_calc"] - g["vertical_rate_mps"]).abs()
    if "cog_rad" in g.columns:
        cog_deg = np.degrees(g["cog_rad"].astype(float))
        out["course_change_deg"] = pd.Series(wrap_angle_deg(cog_deg.diff()), index=g.index)

    # --- K2: attitude dinamigi ---
    for c in _ATTITUDE_COLS:
        if c in g.columns:
            out[c] = g[c]
            out[f"{c.replace('_deg','')}_rate"] = rate_per_s(g[c], t_s, angular=(c == "yaw_deg"))

    # --- K2: GPS sagligi + delta'lar ---
    for c in _GPS_HEALTH_COLS:
        if c in g.columns:
            out[c] = g[c]
            out[f"{c}_delta"] = g[c].astype(float).diff()

    # --- K2: batarya tutarliligi ---
    if all(c in g.columns for c in _BATTERY_COLS):
        # current_a=-1 PX4'te "olcum yok" sentinel'i -- guce katilmaz.
        current = g["current_a"].where(g["current_a"] >= 0)
        out["battery_power_w"] = g["voltage_v"] * current
        out["voltage_delta"] = g["voltage_v"].astype(float).diff()
        out["remaining_delta"] = g["remaining"].astype(float).diff()

    # --- K3: rolling (gecmise-bakan) ---
    if "jamming_indicator" in g.columns:
        out = pd.concat([out, rolling_stats(g["jamming_indicator"], WIN_1S, "jamming_1s", stats=("mean",)),
                         rolling_stats(g["jamming_indicator"], WIN_5S, "jamming_5s", stats=("max",))], axis=1)
    if "noise_per_ms" in g.columns:
        out = pd.concat([out, rolling_stats(g["noise_per_ms"], WIN_5S, "noise_5s", stats=("std", "mean"))], axis=1)
    if "hdop" in g.columns:
        out = pd.concat([out, rolling_stats(g["hdop"], WIN_5S, "hdop_5s", stats=("max",))], axis=1)
    if "satellites_used" in g.columns:
        out = pd.concat([out, rolling_stats(g["satellites_used"], WIN_5S, "sats_5s", stats=("min",))], axis=1)
    out = pd.concat([out, rolling_stats(out["gps_step_m"], WIN_5S, "gps_step_5s", stats=("max", "rms"))], axis=1)

    # --- K3: CUSUM -- yavas drift (stealthy spoofing) dedektoru ---
    for col in ["gps_speed_residual", "hdop", "noise_per_ms"]:
        if col in out.columns:
            src = out[col]
        elif col in g.columns:
            src = g[col]
        else:
            continue
        cs = cusum(src)
        out[f"{col}_cusum_pos"] = cs["cusum_pos"]

    # --- K3/K4: donma + missingness ---
    if has_latlon:
        pos_key = g["lat"].astype(str) + "," + g["lon"].astype(str)
    elif "pos_x_m" in g.columns and g["pos_x_m"].notna().any():
        pos_key = g["pos_x_m"].astype(str) + "," + g["pos_y_m"].astype(str)
    else:
        pos_key = pd.Series("", index=g.index)
    out["gps_frozen_count"] = consecutive_unchanged(pos_key)
    att_missing = g[[c for c in _ATTITUDE_COLS if c in g.columns]].isnull().any(axis=1)
    bat_missing = g[[c for c in _BATTERY_COLS if c in g.columns]].isnull().any(axis=1)
    gps_missing = g[[c for c in _GPS_HEALTH_COLS if c in g.columns]].isnull().any(axis=1)
    out["attitude_missing"] = att_missing.astype(int)
    out["battery_missing"] = bat_missing.astype(int)
    out["gps_health_missing"] = gps_missing.astype(int)
    out["num_missing_groups"] = out[["attitude_missing", "battery_missing", "gps_health_missing"]].sum(axis=1)
    # stale sayaci: kac ardisik ornek boyunca attitude gelmedi (+ saniye cinsi)
    out["attitude_stale_count"] = att_missing.groupby((~att_missing).cumsum()).cumcount()
    dt_med = float(dt.median()) if np.isfinite(dt.median()) else 0.2
    out["attitude_stale_s"] = out["attitude_stale_count"] * dt_med

    # K4 baglam: basit in_air/is_moving proxy'si (FableChat Katman 4).
    speed_src = g["vel_m_s"] if "vel_m_s" in g.columns else out["gps_speed_calc_mps"]
    out["is_moving"] = (speed_src.astype(float).fillna(0) > 1.0).astype(int)

    return out


# Missingness feature'lari: ablation Model B'de dusurulecek kolonlar.
MISSINGNESS_COLUMNS = ["attitude_missing", "battery_missing", "gps_health_missing",
                       "num_missing_groups", "attitude_stale_count", "attitude_stale_s"]


def build_px4_features(silver: pd.DataFrame, *, time_col: str = "timestamp") -> pd.DataFrame:
    """PX4-tabanli Silver (UAV Attack veya UAV-SEAD) -> feature tablosu."""
    frames = []
    for source_id, g in silver.sort_values(time_col).groupby("source_id", sort=False):
        feats = _flight_features(g.reset_index(drop=True), time_col)
        feats["source_id"] = source_id
        feats["label"] = g["label"].reset_index(drop=True)
        frames.append(feats)
    out = pd.concat(frames, ignore_index=True)
    logger.info("PX4 features: %d satir, %d kolon", len(out), out.shape[1])
    return out


def build_uav_attack_features(silver: pd.DataFrame) -> pd.DataFrame:
    return build_px4_features(silver, time_col="timestamp")


def feature_columns(df: pd.DataFrame, *, include_missingness: bool = True) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLUMNS]
    if not include_missingness:
        cols = [c for c in cols if c not in MISSINGNESS_COLUMNS]
    return cols
