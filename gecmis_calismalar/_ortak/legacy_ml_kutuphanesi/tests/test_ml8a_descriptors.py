import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features.window_descriptors import (
    DESCRIPTORS,
    available_channels,
    build_descriptor_schema,
    build_window_descriptors,
    descriptor_schema_sha256,
    guard_band_label,
    interval_overlap_fraction,
    label_windows_from_intervals,
    write_descriptor_schema,
)


CHANNELS = ["gps_speed_residual", "alt_baro_residual"]


def _flight(duration_s: int = 30, source_id: str = "flight_a") -> pd.DataFrame:
    t = np.arange(duration_s + 1, dtype=float)
    return pd.DataFrame(
        {
            "source_id": source_id,
            "t_rel_s": t,
            "gps_speed_residual": np.sin(t / 4.0),
            "alt_baro_residual": np.where(t == 7.0, np.nan, t / 10.0),
        }
    )


def test_descriptor_prefix_invariance():
    full = _flight(30)
    prefix = full[full["t_rel_s"] <= 18.0].copy()

    expected = build_window_descriptors(prefix, CHANNELS)
    actual = build_window_descriptors(full, CHANNELS)
    actual = actual[actual["t_rel_s"] <= 18.0].reset_index(drop=True)

    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual)


def test_descriptor_no_future_leak():
    original = _flight(30)
    changed_future = original.copy()
    future = changed_future["t_rel_s"] > 15.0
    changed_future.loc[future, CHANNELS] = 1_000_000.0

    expected = build_window_descriptors(original, CHANNELS)
    actual = build_window_descriptors(changed_future, CHANNELS)
    descriptor_cols = [c for c in expected if "__" in c]

    pd.testing.assert_frame_equal(
        expected.loc[expected["t_rel_s"] <= 15.0, descriptor_cols].reset_index(drop=True),
        actual.loc[actual["t_rel_s"] <= 15.0, descriptor_cols].reset_index(drop=True),
    )


def test_descriptor_never_crosses_flight_boundary():
    first = _flight(12, "first")
    second = _flight(12, "second")
    second[CHANNELS] = 999.0
    out = build_window_descriptors(pd.concat([first, second], ignore_index=True), CHANNELS)

    second_start = out[(out["source_id"] == "second") & (out["t_rel_s"] == 0.0)].iloc[0]
    assert second_start["gps_speed_residual__mean"] == 999.0
    assert second_start["window_row_count"] == 1


def test_limited_ffill_is_causal_and_staleness_is_explicit():
    frame = pd.DataFrame(
        {
            "source_id": "flight",
            "t_rel_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "gps_speed_residual": [5.0, np.nan, np.nan, np.nan, 9.0],
        }
    )
    out = build_window_descriptors(frame, ["gps_speed_residual"])
    at_three = out[out["t_rel_s"] == 3.0].iloc[0]
    at_four = out[out["t_rel_s"] == 4.0].iloc[0]

    assert at_three["gps_speed_residual__last"] == 5.0
    assert at_three["gps_speed_residual__stale_fraction"] == 0.25
    assert at_four["gps_speed_residual__last"] == 9.0
    assert at_four["gps_speed_residual__missing_fraction"] == 3 / 5


def test_descriptor_does_not_invent_empty_windows_inside_large_time_gap():
    frame = pd.DataFrame(
        {
            "source_id": "flight",
            "t_rel_s": [0.0, 0.5, 1.0, 1_000.0, 1_000.5, 1_001.0],
            "gps_speed_residual": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    out = build_window_descriptors(frame, ["gps_speed_residual"])
    assert out["t_rel_s"].tolist() == [0.0, 1.0, 1_000.0, 1_001.0]
    assert (out["window_row_count"] > 0).all()


def test_schema_contains_only_observed_source_channels_and_has_stable_hash():
    alfa = pd.DataFrame({"xtrack_error": [1.0], "alt_baro_residual": [2.0]})
    sead = pd.DataFrame(
        {
            "gps_speed_residual": [1.0],
            "alt_baro_residual": [2.0],
            "innovation_check_flags_bit_count": [0],
            "attitude_missing": [0],
        }
    )
    schema = build_descriptor_schema({"alfa": alfa, "uav_sead": sead})

    assert available_channels(alfa, "alfa") == ["xtrack_error"]
    assert schema["sources"]["alfa"]["channels"] == ["xtrack_error"]
    assert schema["sources"]["uav_sead"]["channels"] == [
        "gps_speed_residual",
        "alt_baro_residual",
        "innovation_check_flags_bit_count",
        "attitude_missing",
    ]
    assert schema["descriptors"] == list(DESCRIPTORS)

    path = Path("artifacts/ml8a/.descriptor_schema_test.json")
    try:
        returned_hash = write_descriptor_schema(schema, path)
        assert json.loads(path.read_text(encoding="utf-8")) == schema
        assert returned_hash == descriptor_schema_sha256(path)
        assert returned_hash == descriptor_schema_sha256(schema)
    finally:
        path.unlink(missing_ok=True)


def test_guard_band_labeling():
    assert guard_band_label(interval_overlap_fraction(0.0, 10.0, [])) == "negative"
    assert guard_band_label(interval_overlap_fraction(0.0, 10.0, [(2.0, 5.0)])) == "guard_band"
    assert guard_band_label(interval_overlap_fraction(0.0, 10.0, [(2.0, 8.0)])) == "positive"

    windows = pd.DataFrame(
        {
            "source_id": ["flight"] * 3,
            "window_start_s": [0.0, 10.0, 20.0],
            "window_end_s": [10.0, 20.0, 30.0],
        }
    )
    labeled = label_windows_from_intervals(
        windows,
        {"flight": [(12_000_000.0, 15_000_000.0), (22_000_000.0, 28_000_000.0)]},
        t0_by_source={"flight": 0.0},
        interval_unit="absolute_us",
    )
    assert labeled["train_label"].tolist() == ["negative", "guard_band", "positive"]
