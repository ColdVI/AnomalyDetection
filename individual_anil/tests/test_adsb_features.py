"""ADS-B feature/olcekleme/segmentasyon testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from __future__ import annotations

import numpy as np

import pandas as pd

import pytest

from adsb.features import (
    EARTH_RADIUS_M,
    altitude_source_residual,
    build_feature_table,
    heading_residual,
    speed_residual,
    turn_bank_residual,
    velocity_component_residuals,
    vertical_rate_residual,
)

from adsb.scaling import ClippedRobustScaler

from adsb.segmentation import assign_flight_ids, flight_summary, new_leg_agreement, segment_flights



# ===== kaynak: test_adsb_features =====

_M_PER_DEG_LAT = EARTH_RADIUS_M * np.pi / 180.0


def _straight_flight(n=5, dt=10.0, speed_mps=100.0, climb_mps=5.0) -> pd.DataFrame:
    t = np.arange(n) * dt
    dlat_per_step = (speed_mps * dt) / _M_PER_DEG_LAT
    lat = 40.0 + np.arange(n) * dlat_per_step
    lon = np.full(n, 29.0)
    alt = 1000.0 + np.arange(n) * climb_mps * dt
    return pd.DataFrame({
        "flight_id": "F1", "timestamp_utc": t, "lat": lat, "lon": lon, "alt": alt,
        "ground_speed_ms": speed_mps, "track_deg": 0.0, "vertical_rate_ms": climb_mps,
    })


def test_vertical_rate_residual_zero_for_consistent_climb():
    df = _straight_flight()
    res = vertical_rate_residual(df)
    assert np.isnan(res.iloc[0])
    assert res.iloc[1:].abs().max() < 1e-6


def test_speed_residual_zero_for_consistent_flight():
    df = _straight_flight()
    res = speed_residual(df)
    assert res.iloc[1:].abs().max() < 1e-6


def test_velocity_component_residuals_zero_for_consistent_northbound_flight():
    df = _straight_flight()
    residuals = velocity_component_residuals(df)
    assert residuals.iloc[0].isna().all()
    assert residuals.iloc[1:].abs().to_numpy().max() < 1e-6


def test_velocity_component_residual_preserves_signed_north_shift():
    df = _straight_flight(speed_mps=100.0)
    dt = float(df["timestamp_utc"].iloc[1] - df["timestamp_utc"].iloc[0])
    df["lat"] = 40.0 + np.arange(len(df)) * (102.0 * dt / _M_PER_DEG_LAT)
    residuals = velocity_component_residuals(df)
    assert residuals["east_velocity_residual"].iloc[1:].abs().max() < 1e-6
    assert residuals["north_velocity_residual"].iloc[1:].to_numpy() == pytest.approx(-2.0)


def test_velocity_component_residual_skips_nonpositive_dt():
    df = _straight_flight(n=3)
    df["timestamp_utc"] = [0.0, 0.0, -1.0]
    residuals = velocity_component_residuals(df)
    assert residuals.isna().all().all()


def test_velocity_component_residual_is_prefix_invariant():
    df = _straight_flight(n=6)
    full = velocity_component_residuals(df)
    prefix = velocity_component_residuals(df.iloc[:4])
    pd.testing.assert_frame_equal(full.iloc[:4], prefix)


def test_heading_residual_zero_for_northbound_flight():
    df = _straight_flight()
    res = heading_residual(df)
    assert res.iloc[1:].abs().max() < 0.1


def test_turn_bank_residual_zero_for_coordinated_turn():
    n, dt, speed, roll = 5, 10.0, 100.0, 10.0
    g = 9.80665
    rate_deg_s = np.degrees(g * np.tan(np.radians(roll)) / speed)
    track = np.arange(n) * rate_deg_s * dt
    df = pd.DataFrame({
        "flight_id": "F1", "timestamp_utc": np.arange(n) * dt,
        "track_deg": track, "roll_deg": roll, "ground_speed_ms": speed,
    })
    res = turn_bank_residual(df)
    assert res.iloc[1:].abs().max() < 1e-6


def test_build_feature_table_missing_roll_gives_nan_column():
    df = _straight_flight()
    out = build_feature_table(df)
    assert out["turn_bank_residual"].isna().all()


def test_build_feature_table_missing_alt_geom_gives_nan_column():
    df = _straight_flight()
    out = build_feature_table(df)
    assert out["altitude_source_residual"].isna().all()


def test_altitude_source_residual_zero_for_constant_gap():
    df = _straight_flight()
    df["alt_geom_m"] = df["alt"] + 50.0  # sabit ofset -- jeoit sapmasi gibi, ZAMANLA degismiyor
    res = altitude_source_residual(df)
    assert res.iloc[1:].abs().max() < 1e-6


def test_altitude_source_residual_nonzero_when_gap_drifts():
    df = _straight_flight()
    df["alt_geom_m"] = df["alt"] + 50.0
    df.loc[3:, "alt_geom_m"] += 200.0  # bir kaynak aniden sapiyor (sahte/hatali)
    res = altitude_source_residual(df)
    assert abs(res.iloc[3]) > 1.0


def test_build_feature_table_does_not_leak_across_flight_boundary():
    a = _straight_flight(n=3)
    a["alt_geom_m"] = a["alt"] + 50.0
    b = _straight_flight(n=3)
    b["flight_id"] = "F2"
    b["timestamp_utc"] = b["timestamp_utc"] + 10_000
    b["alt_geom_m"] = b["alt"] + 80.0  # farkli ofset -- sinir-otesi diff olsaydi yakalardik
    df = pd.concat([a, b], ignore_index=True)
    out = build_feature_table(df)
    first_of_b = out[out["flight_id"] == "F2"].iloc[0]
    assert np.isnan(first_of_b["vertical_rate_residual"])
    assert np.isnan(first_of_b["speed_residual"])
    assert np.isnan(first_of_b["heading_residual"])
    assert np.isnan(first_of_b["east_velocity_residual"])
    assert np.isnan(first_of_b["north_velocity_residual"])
    assert np.isnan(first_of_b["altitude_source_residual"])



# ===== kaynak: test_adsb_scaling =====

def test_fit_transform_centers_and_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1000.0, scale=50.0, size=(200, 5, 1))  # buyuk-genlikli tek kanal
    M = np.ones_like(X)

    scaler = ClippedRobustScaler(clip=5.0)
    scaled = scaler.fit_transform(X, M)

    assert abs(np.median(scaled)) < 0.5  # medyan sifira yakin
    assert scaled.min() >= -5.0 and scaled.max() <= 5.0  # kirpma calisti


def test_extreme_outlier_is_clipped_not_dominant():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 5, 1))
    M = np.ones_like(X)
    X[0, 0, 0] = 1_000_000.0  # asiri deger (sensor hatasi benzeri)

    scaler = ClippedRobustScaler(clip=5.0)
    scaled = scaler.fit_transform(X, M)

    assert scaled.max() <= 5.0  # kirpilmamis olsaydi milyonlarca olurdu


def test_masked_values_excluded_from_fit_statistics():
    X = np.array([[[1.0], [2.0], [3.0], [1_000_000.0]]])  # (1, 4, 1)
    M = np.array([[[1.0], [1.0], [1.0], [0.0]]])  # son deger maskeli (gecersiz)

    scaler = ClippedRobustScaler(clip=5.0)
    scaler.fit(X, M)

    # medyan 1000000'dan etkilenmemis olmali (yalniz 1,2,3 kullanildi)
    assert scaler.median_[0] == pytest.approx(2.0)


def test_masked_positions_remain_zero_after_transform():
    X = np.array([[[1.0, 5.0], [2.0, 999.0]]])
    M = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    scaler = ClippedRobustScaler(clip=5.0).fit(X, M)
    scaled = scaler.transform(X, M)
    assert scaled[0, 0, 1] == 0.0
    assert scaled[0, 1, 1] == 0.0


def test_constant_channel_does_not_divide_by_zero():
    X = np.full((10, 3, 1), 42.0)
    M = np.ones_like(X)
    scaler = ClippedRobustScaler(clip=5.0)
    scaled = scaler.fit_transform(X, M)  # IQR=0 -> fallback 1.0, patlamamali
    assert np.isfinite(scaled).all()


def test_transform_before_fit_raises():
    scaler = ClippedRobustScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((1, 2, 1)), np.ones((1, 2, 1)))


def test_fit_on_train_applies_consistently_to_val():
    rng = np.random.default_rng(2)
    X_train = rng.normal(loc=100.0, scale=10.0, size=(50, 4, 2))
    M_train = np.ones_like(X_train)
    X_val = rng.normal(loc=100.0, scale=10.0, size=(10, 4, 2))
    M_val = np.ones_like(X_val)

    scaler = ClippedRobustScaler(clip=5.0).fit(X_train, M_train)
    scaled_val = scaler.transform(X_val, M_val)
    # ayni median/iqr val'e de uygulanmis (val'in kendi istatistigi kullanilmamis)
    manual = np.clip((X_val - scaler.median_) / (scaler.iqr_ + 1e-6), -5.0, 5.0)
    np.testing.assert_allclose(scaled_val, manual)



# ===== kaynak: test_adsb_segmentation =====

def _traces(icao: str, timestamps: list[float], new_leg_at: set[float] | None = None) -> pd.DataFrame:
    new_leg_at = new_leg_at or set()
    return pd.DataFrame({
        "source_id": icao,
        "timestamp_utc": timestamps,
        "lat": np.linspace(40.0, 41.0, len(timestamps)),
        "lon": np.linspace(29.0, 30.0, len(timestamps)),
        "flags_new_leg": [t in new_leg_at for t in timestamps],
    })


def test_assign_flight_ids_splits_on_gap():
    a = _traces("A", [0, 10, 20, 30])  # boşluksuz -> 1 uçuş
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])  # 2 saatlik boşluk -> 2 uçuş
    df = pd.concat([a, b], ignore_index=True)

    flight_id = assign_flight_ids(df, gap_s=1800.0)

    assert flight_id[df["source_id"] == "A"].nunique() == 1
    assert flight_id[df["source_id"] == "B"].nunique() == 2
    b_ids = flight_id[df["source_id"] == "B"].tolist()
    assert b_ids[0] == b_ids[1]
    assert b_ids[2] == b_ids[3]
    assert b_ids[0] != b_ids[2]


def test_assign_flight_ids_no_gap_is_single_flight():
    df = _traces("A", [0, 60, 120, 180, 240])
    assert assign_flight_ids(df, gap_s=1800.0).nunique() == 1


def test_assign_flight_ids_unsorted_input_matches_sorted():
    a = _traces("A", [0, 10, 20, 30])
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    df = pd.concat([a, b], ignore_index=True)
    shuffled = df.sample(frac=1.0, random_state=0)

    sorted_result = assign_flight_ids(df, gap_s=1800.0)
    shuffled_result = assign_flight_ids(shuffled, gap_s=1800.0)

    pd.testing.assert_series_equal(
        shuffled_result.sort_index(), sorted_result.sort_index(), check_names=False
    )


def test_assign_flight_ids_empty_df():
    df = _traces("A", [])
    assert len(assign_flight_ids(df, gap_s=1800.0)) == 0


def test_segment_flights_output_sorted_with_flight_id_column():
    a = _traces("A", [30, 0, 20, 10])  # kasten sırasız
    out = segment_flights(a, gap_s=1800.0)
    assert out["timestamp_utc"].is_monotonic_increasing
    assert "flight_id" in out.columns
    assert out["flight_id"].nunique() == 1


def test_segment_flights_preserves_source_order_for_equal_timestamps():
    frame = _traces("A", [10, 0, 10, 10])
    frame["source_order"] = ["first-at-10", "at-0", "second-at-10", "third-at-10"]

    out = segment_flights(frame, gap_s=1800.0)

    assert out.loc[out["timestamp_utc"].eq(10), "source_order"].tolist() == [
        "first-at-10",
        "second-at-10",
        "third-at-10",
    ]


def test_new_leg_agreement_full():
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30], new_leg_at={7200 + 20})
    seg = segment_flights(b, gap_s=1800.0)
    assert new_leg_agreement(seg) == pytest.approx(1.0)


def test_new_leg_agreement_zero_when_flag_never_set():
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    seg = segment_flights(b, gap_s=1800.0)
    assert new_leg_agreement(seg) == pytest.approx(0.0)


def test_new_leg_agreement_nan_when_no_boundaries():
    a = _traces("A", [0, 10, 20, 30])
    seg = segment_flights(a, gap_s=1800.0)
    assert np.isnan(new_leg_agreement(seg))


def test_flight_summary_counts_rows_and_duration():
    a = _traces("A", [0, 10, 20, 30])
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    df = pd.concat([a, b], ignore_index=True)
    seg = segment_flights(df, gap_s=1800.0)

    summary = flight_summary(seg)
    assert set(summary.columns) == {"flight_id", "n_rows", "duration_s", "start_time"}
    assert summary.set_index("flight_id").loc["A_000", "n_rows"] == 4
    assert summary.set_index("flight_id").loc["A_000", "duration_s"] == pytest.approx(30.0)

