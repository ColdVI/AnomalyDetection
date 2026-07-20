"""RESIDUAL-V1 feature/hizalama/sema testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

import numpy as np

import pandas as pd

from residual_v1.features.align import align_to_clock, observed_tolerances

from residual_v1.features.build import build_xy

from residual_v1.features.spec import ResidualChannelSpec

import json

import pytest

from residual_v1.features.physics import coordinated_turn_yaw_rate, finite_difference

from residual_v1.features.spec import (
    ALFA_SPECS,
    RFLY_SPECS,
    ResidualChannelSpec,
    descriptor_schema_payload,
    descriptor_schema_sha256,
)

from residual_v1.features.phases import label_phases

from residual_v1.ingest.alfa_channels import CHANNELS as ALFA_CHANNELS

from residual_v1.ingest.rfly_channels import CHANNELS as RFLY_CHANNELS

from residual_v1.schema import ChannelSpec

from pathlib import Path



# ===== kaynak: test_residual_v1_align =====

def test_backward_tolerance_staleness_and_stale_mask():
    flight = {
        "clock": pd.DataFrame({"t": [1.0, 1.2, 2.0], "roll": [0.0, 0.1, 0.2]}),
        "slow": pd.DataFrame({"t": [0.9, 1.1, 1.7, 2.1], "ground_speed": [9, 11, 17, 21]}),
    }
    result = align_to_clock(
        flight,
        "clock",
        {"slow": 0.25},
        stale={"ground_speed": [{"start_s": 1.15, "end_s": 1.25}]},
    )
    # t=1.0 takes 0.9, never the future 1.1 sample.
    assert result.loc[0, "ground_speed"] == 9
    assert np.isclose(result.loc[0, "ground_speed_staleness_ms"], 100.0)
    # The causally matched 1.1 value is then masked by the stale segment.
    assert np.isnan(result.loc[1, "ground_speed"])
    assert np.isinf(result.loc[1, "ground_speed_staleness_ms"])
    # 1.7 is 300 ms old at t=2.0 and therefore outside tolerance.
    assert np.isnan(result.loc[2, "ground_speed"])
    assert np.isinf(result.loc[2, "ground_speed_staleness_ms"])


def test_observed_topic_rate_can_expand_a_decimated_tolerance():
    flight = {"imu": pd.DataFrame({"t": [0.0, 0.4, 0.8], "x": [1, 2, 3]})}
    tolerance = observed_tolerances(flight, {"imu": 0.04})
    assert np.isclose(tolerance["imu"], 0.6)



# ===== kaynak: test_residual_v1_build_features =====

def test_horizon_guard_band_phase_exclusion_and_no_response_feature():
    time = np.arange(0.0, 25.1, 0.1)
    flight = pd.DataFrame(
        {
            "t": time,
            "aileron_cmd": np.sin(time),
            "airspeed": np.full(len(time), 20.0),
            "roll_rate": time,
        }
    )
    phases = pd.DataFrame(
        {
            "t": time,
            "phase": np.where(time < 1.0, "ground", "cruise"),
            "phase_boundary": np.isclose(time, 2.0),
        }
    )
    spec = ResidualChannelSpec(
        "R1_test", ("aileron_cmd",), "roll_rate", ("airspeed_context",)
    )
    X, y, meta = build_xy(
        flight,
        spec,
        phases,
        events=[{"fault_class": "engine", "onset_s": 20.0, "end_s": 25.0}],
        flight_id="f1",
    )
    assert not any(column == "roll_rate" or column.startswith("roll_rate__") for column in X)
    assert "airspeed_context__last" in X
    assert not any(
        column.startswith("airspeed_context__tri_")
        or column == "airspeed_context__delta_1s"
        for column in X
    )
    assert "aileron_cmd__delta_1s" in X
    assert not (meta["row_meta"]["phase"] == "ground").any()
    assert 2.0 not in meta["row_meta"]["t"].tolist()
    assert meta["row_meta"].loc[meta["row_meta"]["t"] < 10.0, "train_eligible"].all()
    assert not meta["row_meta"].loc[meta["row_meta"]["t"] >= 10.0, "train_eligible"].any()

    # At t=1.0, future samples 1.1..1.5 average to 1.3.
    row_index = meta["row_meta"].index[np.isclose(meta["row_meta"]["t"], 1.0)][0]
    assert np.isclose(y.loc[row_index], 1.3)


def test_nan_rows_are_dropped_and_reported():
    time = np.arange(0.0, 3.0, 0.1)
    command = np.sin(time)
    command[15] = np.nan
    flight = pd.DataFrame(
        {"t": time, "aileron_cmd": command, "airspeed": 20.0, "roll_rate": time}
    )
    phases = pd.DataFrame({"t": time, "phase": "cruise", "phase_boundary": False})
    spec = ResidualChannelSpec(
        "R1_test", ("aileron_cmd",), "roll_rate", ("airspeed_context",)
    )
    X, _, meta = build_xy(flight, spec, phases)
    assert X.notna().all().all()
    assert meta["nan_drop_count"] > 0



# ===== kaynak: test_residual_v1_feature_spec =====

@pytest.mark.parametrize("feature", ["roll_rate", "roll_rate_lag1", "lag_2_roll_rate", "roll_rate_history"])
def test_ar_leakage_guard_rejects_response_and_history(feature):
    with pytest.raises(ValueError, match="response"):
        ResidualChannelSpec("bad", ("aileron_cmd", feature), "roll_rate")


def test_response_history_is_also_rejected_as_context():
    with pytest.raises(ValueError, match="response or response history"):
        ResidualChannelSpec("bad", ("aileron_cmd",), "roll_rate", ("roll_rate_lag1",))


def test_registered_specs_and_descriptor_hash_are_stable():
    assert [spec.name.split("_")[0] for spec in ALFA_SPECS] == [f"R{i}" for i in range(1, 7)]
    assert [spec.name.split("_")[0] for spec in RFLY_SPECS] == [f"Q{i}" for i in range(1, 5)]
    payload = descriptor_schema_payload()
    assert payload["schema"] == "descriptor_schema_residual_v1"
    assert len(descriptor_schema_sha256()) == 64
    json.dumps(payload)
    assert ALFA_SPECS[-1].boundary_masks == ("waypoint",)
    assert all(not spec.boundary_masks for spec in (*ALFA_SPECS[:-1], *RFLY_SPECS))


def test_unknown_boundary_mask_is_rejected():
    with pytest.raises(ValueError, match="unsupported boundary masks"):
        ResidualChannelSpec("bad", (), "response", boundary_masks=("unknown",))


def test_physics_helpers_use_observed_samples():
    time = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    values = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0], name="speed")
    derivative = finite_difference(time, values)
    assert np.allclose(derivative, 2.0)
    yaw_rate = coordinated_turn_yaw_rate(pd.Series([0.0, np.pi / 4]), pd.Series([10.0, 10.0]))
    assert np.isclose(yaw_rate.iloc[0], 0.0)
    assert np.isclose(yaw_rate.iloc[1], 9.80665 / 10.0)



# ===== kaynak: test_residual_v1_phases =====

CONFIG = {
    "ground_speed_max_mps": 3.0,
    "ground_climb_abs_max_mps": 0.3,
    "maneuver_roll_abs_min_deg": 25.0,
    "maneuver_roll_rate_abs_min_deg_s": 15.0,
    "takeoff_climb_min_mps": 0.8,
    "landing_climb_max_mps": -0.5,
    "transition_altitude_agl_max_m": 60.0,
    "boundary_buffer_s": 1.0,
}


def test_phase_sequence_and_time_based_boundary_mask():
    frame = pd.DataFrame(
        {
            "t": np.arange(0.0, 8.0),
            "ground_speed": [0, 0, 8, 12, 15, 15, 8, 0],
            "climb_rate": [0, 0, 2, 1, 0, 0, -2, 0],
            "altitude": [100, 100, 102, 110, 130, 140, 120, 100],
            "roll": np.deg2rad([0, 0, 0, 0, 35, 0, 0, 0]),
            "roll_rate": np.zeros(8),
        }
    )
    result = label_phases(frame, config=CONFIG)
    assert result["phase"].tolist() == [
        "ground",
        "ground",
        "takeoff",
        "takeoff",
        "maneuver",
        "cruise",
        "landing",
        "ground",
    ]
    # Every transition timestamp and its immediate +/- 1 s neighbours are masked.
    assert result.loc[[1, 2, 3], "phase_boundary"].all()
    assert result.loc[[6, 7], "phase_boundary"].all()



# ===== kaynak: test_residual_v1_schema =====

def test_channel_spec_validates_bounds_and_frequency():
    with pytest.raises(ValueError, match="valid_min"):
        ChannelSpec("bad", "topic", "u", 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="nominal_hz"):
        ChannelSpec("bad", "topic", "u", 0.0, 1.0, 0.0)


@pytest.mark.parametrize("channels", [ALFA_CHANNELS, RFLY_CHANNELS])
def test_channel_inventory_names_are_unique_and_valid(channels):
    names = [channel.name for channel in channels]
    assert len(names) == len(set(names))
    assert all(channel.valid_min < channel.valid_max for channel in channels)
    assert all(channel.nominal_hz > 0 for channel in channels)
    assert all(channel.role in {"response", "command", "context"} for channel in channels)


def test_battery_is_context_only():
    battery = next(channel for channel in RFLY_CHANNELS if channel.name == "battery_voltage")
    assert battery.role == "context"



# ===== kaynak: test_residual_v1_no_interpolation_lint =====

def test_residual_v1_contains_no_forbidden_interpolation_patterns():
    forbidden = ("interpolate" + "(", ".resample" + "(", "fillna" + "(method=")
    hits = []
    for path in Path("residual_v1").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in source:
                hits.append(f"{path}: {pattern}")
    assert not hits, "forbidden Silver interpolation patterns:\n" + "\n".join(hits)

