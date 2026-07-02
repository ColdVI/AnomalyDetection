import numpy as np
import pandas as pd

from src.ml.features.alfa_features import ID_COLUMNS as ALFA_ID_COLUMNS
from src.ml.features.alfa_features import build_alfa_features, feature_columns
from src.ml.features.temporal import (
    angular_error_deg,
    consecutive_unchanged,
    cusum,
    haversine_m,
    rate_per_s,
    rolling_stats,
    wrap_angle_deg,
)
from src.ml.features.uav_attack_features import MISSINGNESS_COLUMNS, build_uav_attack_features
from src.ml.features.uav_attack_features import feature_columns as px4_feature_columns


def test_wrap_angle_deg_handles_wraparound():
    # 179 - (-179) naive fark 358 verir; gercek acisal hata -2'dir.
    assert angular_error_deg(179.0, -179.0) == -2.0
    assert angular_error_deg(-179.0, 179.0) == 2.0
    assert wrap_angle_deg(360.0) == 0.0
    assert wrap_angle_deg(-180.0) == -180.0


def test_rate_per_s_angular_crosses_wrap_without_spike():
    s = pd.Series([359.0, 1.0])
    t = pd.Series([0.0, 1.0])
    rate = rate_per_s(s, t, angular=True)
    assert rate.iloc[1] == 2.0  # -358 degil


def test_rate_per_s_duplicate_timestamp_gives_nan_not_inf():
    s = pd.Series([1.0, 2.0])
    t = pd.Series([5.0, 5.0])
    rate = rate_per_s(s, t)
    assert np.isnan(rate.iloc[1])


def test_haversine_known_distance():
    # 1 derece boylam farki, ekvatorda ~111.19 km
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert abs(d - 111_195) < 200


def test_rolling_stats_is_past_only():
    s = pd.Series([0.0, 0.0, 0.0, 100.0])
    out = rolling_stats(s, window_rows=2, prefix="x", stats=("max",))
    # 100 gelmeden onceki ornekte max hala 0 olmali (gelecege bakmiyor)
    assert out["x_max"].iloc[2] == 0.0
    assert out["x_max"].iloc[3] == 100.0


def test_cusum_accumulates_persistent_shift_not_noise():
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 1, 200))
    shifted = pd.Series(np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)]))
    cs_noise = cusum(noise)
    cs_shift = cusum(shifted)
    assert cs_shift["cusum_pos"].iloc[-1] > 10 * max(cs_noise["cusum_pos"].iloc[-1], 1e-9)


def test_consecutive_unchanged_counts_freeze():
    s = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0])
    out = consecutive_unchanged(s)
    assert list(out) == [0, 1, 2, 0, 1]


def _tiny_alfa_silver() -> pd.DataFrame:
    n = 30
    t = np.arange(n) * 250_000_000  # 4 Hz, ns
    return pd.DataFrame({
        "ts_ns": np.concatenate([t, t]),
        "lat": np.concatenate([40 + np.arange(n) * 1e-5, 41 + np.arange(n) * 1e-5]),
        "lon": np.concatenate([29 + np.arange(n) * 1e-5, 30 + np.arange(n) * 1e-5]),
        "alt": 100.0,
        "roll_measured": 1.0, "roll_commanded": 0.5,
        "pitch_measured": 2.0, "pitch_commanded": 2.0,
        "yaw_measured": 179.0, "yaw_commanded": -179.0,
        "airspeed_measured": 15.0, "airspeed_commanded": 14.0,
        "velocity_measured": 15.0, "velocity_commanded": 15.0,
        "ground_speed_ms": 15.0, "throttle": 60.0, "climb_rate_ms": 0.0,
        "alt_error": 0.1, "aspd_error": 0.2, "xtrack_error": 0.3, "wp_dist": 50.0,
        "path_dev_x": 0.1, "path_dev_y": 0.1, "path_dev_z": 0.1,
        "label": ["normal"] * n + ["engine_fault"] * n,
        "source_id": ["seq_normal"] * n + ["seq_fault"] * n,
    })


def test_build_alfa_features_no_leakage_columns():
    feats = build_alfa_features(_tiny_alfa_silver())
    cols = feature_columns(feats)
    # kimlik kolonlari feature listesinde olamaz
    assert not set(ALFA_ID_COLUMNS) & set(cols)
    # mutlak konum/zaman sizintisi yok
    for banned in ["lat", "lon", "ts_ns", "timestamp_utc", "source_id", "label"]:
        assert banned not in cols
    # yaw hatasi wrap-aware: 179 vs -179 -> -2
    assert (feats["yaw_error"] == -2.0).all()


def test_build_alfa_features_t_rel_starts_at_zero_per_flight():
    feats = build_alfa_features(_tiny_alfa_silver())
    for _, g in feats.groupby("source_id"):
        assert g["t_rel_s"].iloc[0] == 0.0


def _tiny_uav_silver() -> pd.DataFrame:
    n = 40
    t = np.arange(n) * 200_000  # 5 Hz, us
    return pd.DataFrame({
        "timestamp": np.concatenate([t, t]),
        "lat": np.concatenate([43 + np.arange(n) * 1e-5, 43.5 + np.arange(n) * 1e-5]),
        "lon": np.concatenate([-78 + np.arange(n) * 1e-5, -78.5 + np.arange(n) * 1e-5]),
        "alt": 50.0,
        "roll_deg": 1.0, "pitch_deg": 1.0, "yaw_deg": 90.0,
        "eph": 1.0, "epv": 1.0, "hdop": 0.8, "vdop": 1.1,
        "satellites_used": 10.0, "s_variance_m_s": 0.25,
        "jamming_indicator": 0.0, "noise_per_ms": 0.0,
        "voltage_v": 12.0, "remaining": 0.9, "current_a": -1.0,
        "vel_m_s": 5.0, "vel_n_m_s": 5.0, "vel_e_m_s": 0.0, "vel_d_m_s": 0.0,
        "vertical_rate_mps": 0.0, "cog_rad": 1.57, "fix_type": 3.0,
        "label": ["benign"] * n + ["gps_spoofing"] * n,
        "source_id": ["log_benign"] * n + ["log_spoof"] * n,
    })


def test_build_uav_attack_features_has_missingness_and_residuals():
    feats = build_uav_attack_features(_tiny_uav_silver())
    for col in MISSINGNESS_COLUMNS:
        assert col in feats.columns
    assert "gps_speed_residual" in feats.columns
    # current_a=-1 sentinel: guce katilmamali -> battery_power_w NaN
    assert feats["battery_power_w"].isna().all()
    # missingness haric feature listesi ablation icin kucultulebilmeli
    with_m = px4_feature_columns(feats, include_missingness=True)
    without_m = px4_feature_columns(feats, include_missingness=False)
    assert set(with_m) - set(without_m) == set(MISSINGNESS_COLUMNS)
