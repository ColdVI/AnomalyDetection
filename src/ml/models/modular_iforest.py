"""Aciklanabilir, feature-ailesi bazli Isolation Forest dedektoru."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ALFA_MODULES = {
    "kontrol_tepki": [
        "abs_roll_error", "abs_pitch_error", "abs_yaw_error", "roll_error_rate",
        "pitch_error_rate", "yaw_error_rate", "roll_error_2s_rms", "roll_error_5s_rms",
        "pitch_error_2s_rms", "pitch_error_5s_rms", "yaw_error_5s_rms",
        "roll_spec_energy_5s", "pitch_spec_energy_5s", "roll_error_cusum_pos",
        "pitch_error_cusum_pos", "turn_residual", "turn_residual_5s_rms",
    ],
    "rehberlik": [
        "alt_error", "aspd_error", "xtrack_error", "path_dev_mag", "wp_dist",
        "alt_error_cusum_pos", "alt_error_cusum_neg", "xtrack_error_cusum_pos",
        "climb_residual", "energy_rate", "altitude_rate", "airspeed_error",
        "abs_airspeed_error", "xtrack_error_5s_rms",
    ],
}


PX4_BASE_MODULES = {
    "nav_butunlugu": [
        "gps_step_m", "log_gps_speed", "gps_accel_mps2", "vertical_rate_calc",
        "gps_speed_residual", "vertical_rate_residual", "course_change_deg",
        "gps_step_5s_max", "gps_step_5s_rms", "gps_speed_residual_cusum_pos",
        "gps_frozen_count",
    ],
    "sinyal_kalitesi": [
        "jamming_indicator", "noise_per_ms", "hdop", "vdop", "satellites_used",
        "s_variance_m_s", "eph", "epv", "jamming_1s_mean", "jamming_5s_max",
        "noise_5s_std", "noise_5s_mean", "hdop_5s_max", "sats_5s_min",
        "hdop_cusum_pos", "noise_per_ms_cusum_pos",
    ],
    "veri_kalitesi": [
        "attitude_missing", "battery_missing", "gps_health_missing",
        "num_missing_groups", "attitude_stale_count", "attitude_stale_s",
    ],
}


# ML-7 aday modulleri: default'a ancak event-onset/FA butcesini development'ta
# gectikten sonra alinacak. UAV Attack'ta olmayan kolonlar otomatik elenir.
PX4_ML7_CANDIDATE_MODULES = {
    **PX4_BASE_MODULES,
    "irtifa_tutarliligi": [
        "alt_baro_residual", "alt_local_residual", "vertical_rate_local_residual",
        "alt_baro_residual_5s_max", "alt_baro_residual_5s_rms",
        "alt_local_residual_5s_max", "alt_local_residual_5s_rms",
        "alt_baro_residual_cusum_pos", "alt_local_residual_cusum_pos",
        "hgt_test_ratio", "hgt_test_ratio_5s_max", "pos_vert_accuracy",
    ],
    "kontrol_cevabi": [
        "roll_setpoint_error", "pitch_setpoint_error", "yaw_setpoint_error",
        "roll_rate_error", "pitch_rate_error", "yaw_rate_error",
        "attitude_error_mag", "attitude_error_mag_5s_max",
        "attitude_error_mag_5s_rms", "attitude_error_mag_cusum_pos",
        "actuator_effort", "actuator_thrust_cmd", "control_strain",
        "vibe_0", "vibe_1", "vibe_2",
    ],
    "ekf_redleri": [
        "innovation_check_flags_active", "innovation_check_flags_bit_count",
        "gps_check_fail_flags_active", "gps_check_fail_flags_bit_count",
        "filter_fault_flags_active", "filter_fault_flags_bit_count",
        "timeout_flags_active", "timeout_flags_bit_count", "pre_flt_fail",
        "local_xy_reset_counter_delta", "local_z_reset_counter_delta",
        "local_vxy_reset_counter_delta", "local_vz_reset_counter_delta",
        "local_xy_valid", "local_z_valid", "local_v_xy_valid", "local_v_z_valid",
    ],
}


# ML-9 kategori-eslesmeli adaylari. Bunlar development Gate B/C gecmeden
# PX4_BASE_MODULES'a veya paketlenmis production varsayimina alinmaz.
PX4_ML9_CANDIDATE_MODULES = {
    **PX4_ML7_CANDIDATE_MODULES,
    "dikey_tutarlilik": [
        "ekf_alt_innov", "ekf_vertical_vel_innov",
        "ekf_alt_innov_5s_max", "ekf_vertical_vel_innov_5s_max",
        "ekf_alt_innov_cusum_pos",
    ],
    "motor_simetrisi": [
        "actuator_output_std", "actuator_output_range",
        "actuator_output_std_5s_max", "actuator_output_range_5s_max",
        "actuator_output_std_cusum_pos",
    ],
}

# Gate B'nin dikey ayrisma karsilastirmasi icin mevcut pooled innovation
# referansi; default modul ailesine eklenmez.
PX4_ML9_POOLED_EKF_REFERENCE = {
    "pooled_ekf": ["ekf_pos_innov_mag", "ekf_vel_innov_mag"],
}

# ML-12 ince-modul adaylari (docs/ML12_INCE_MODUL_PLAN.md, ON-KAYIT).
# Hipotez: guclu tekil sinyal genis modulde seyreliyor (ML-11 H26).
# Listeler plana sabitlendi; Gate B/C gecmeden default'a alinmaz.
PX4_ML12_THIN_MODULES = {
    "itki_komutu": ["actuator_thrust_cmd"],
    "itki_kontrol_ince": [
        "actuator_thrust_cmd", "attitude_error_mag", "control_strain",
    ],
}


def anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    return -model.score_samples(X)


def fit_modular_iforest(scaled: pd.DataFrame, split: dict,
                        modules: dict[str, list[str]], *, seed: int,
                        n_jobs: int = 1) -> dict:
    """Normal train'de modulleri fit et, normal val'de iki esigi kalibre et."""
    train = scaled[scaled["source_id"].isin(split["train"])]
    val = scaled[scaled["source_id"].isin(split["val"])]
    fitted = {}
    for name, requested_cols in modules.items():
        cols = [c for c in requested_cols if c in scaled.columns]
        if not cols:
            continue
        model = IsolationForest(
            n_estimators=300, max_samples=256, random_state=seed, n_jobs=n_jobs).fit(train[cols])
        val_scores = anomaly_scores(model, val[cols])
        val_flights = val.assign(_score=val_scores).groupby("source_id")["_score"].max()
        fitted[name] = {
            "model": model,
            "feature_columns": cols,
            "row_threshold_q99": float(np.quantile(val_scores, 0.99)),
            "flight_threshold_max": float(val_flights.max()),
        }
    return fitted


def score_flights(fitted: dict, scaled: pd.DataFrame) -> pd.DataFrame:
    """Modul esik oranlari, fusion skoru ve baskin modulu ucus bazinda dondur."""
    ratios = {}
    for name, item in fitted.items():
        scores = anomaly_scores(item["model"], scaled[item["feature_columns"]])
        flight = scaled.assign(_score=scores).groupby("source_id")["_score"].max()
        tau = max(float(item["flight_threshold_max"]), np.finfo(float).eps)
        ratios[name] = flight / tau
    result = pd.DataFrame(ratios)
    if result.empty:
        return result
    result["fusion"] = result.max(axis=1)
    result["dominant_module"] = result[list(ratios)].idxmax(axis=1)
    return result
