import numpy as np
import pandas as pd

from gecmis_calismalar.residual_v1.features.build import build_xy
from gecmis_calismalar.residual_v1.features.spec import ResidualChannelSpec
from gecmis_calismalar.residual_v1.features.waypoints import label_waypoint_boundaries


CONFIG = {
    "maximum_turn_distance_m": 25.0,
    "trend_window_s": 2.0,
    "minimum_approach_excursion_m": 10.0,
    "minimum_departure_excursion_m": 10.0,
    "minimum_event_separation_s": 5.0,
    "mask_buffer_s": 2.0,
}


def _v_turn(time: np.ndarray, center: float) -> np.ndarray:
    return 5.0 + 10.0 * np.abs(time - center)


def test_v_turn_is_detected_and_mask_has_exact_time_buffer():
    time = np.arange(0.0, 10.1, 0.5)
    result = label_waypoint_boundaries(
        pd.DataFrame({"t": time, "waypoint_distance": _v_turn(time, 5.0)}),
        config=CONFIG,
    )
    assert result.loc[result["waypoint_event"], "t"].tolist() == [5.0]
    expected = np.abs(time - 5.0) <= 2.0
    assert np.array_equal(result["waypoint_boundary"].to_numpy(), expected)


def test_small_observed_oscillation_does_not_trigger():
    time = np.arange(0.0, 10.1, 0.5)
    distance = 20.0 + 3.0 * np.sin(time)
    result = label_waypoint_boundaries(
        pd.DataFrame({"t": time, "waypoint_distance": distance}),
        config=CONFIG,
    )
    assert not result["waypoint_event"].any()
    assert not result["waypoint_boundary"].any()


def test_flight_start_reset_is_not_a_two_sided_v_turn():
    time = np.arange(0.0, 6.1, 0.1)
    distance = np.full(len(time), 40.0)
    distance[1] = 200.0
    distance[2:] = 5.0 + 20.0 * (time[2:] - time[2])
    result = label_waypoint_boundaries(
        pd.DataFrame({"t": time, "waypoint_distance": distance}),
        config=CONFIG,
    )
    assert not result["waypoint_event"].any()


def test_nearby_v_turn_candidates_are_merged():
    time = np.arange(0.0, 14.1, 0.5)
    first = _v_turn(time, 5.0)
    second = _v_turn(time, 9.0)
    distance = np.minimum(first, second)
    result = label_waypoint_boundaries(
        pd.DataFrame({"t": time, "waypoint_distance": distance}),
        config=CONFIG,
    )
    assert int(result["waypoint_event"].sum()) == 1


def test_build_xy_applies_waypoint_mask_only_when_declared():
    time = np.arange(0.0, 10.1, 0.5)
    flight = pd.DataFrame(
        {
            "t": time,
            "waypoint_distance": _v_turn(time, 5.0),
            "xtrack_error": np.sin(time),
        }
    )
    phases = pd.DataFrame({"t": time, "phase": "cruise", "phase_boundary": False})
    masked = ResidualChannelSpec(
        "R6_test",
        (),
        "xtrack_error",
        boundary_masks=("waypoint",),
    )
    unmasked = ResidualChannelSpec("control", (), "xtrack_error")
    _, _, masked_meta = build_xy(
        flight,
        masked,
        phases,
        boundary_configs={"waypoint": CONFIG},
    )
    _, _, control_meta = build_xy(flight, unmasked, phases)
    assert masked_meta["boundary_masks"]["waypoint"]["event_count"] == 1
    assert not masked_meta["row_meta"]["t"].between(3.0, 7.0).any()
    assert control_meta["row_meta"]["t"].between(3.0, 7.0).any()
    assert masked_meta["output_rows"] < control_meta["output_rows"]


def test_non_r6_command_channel_is_unchanged_by_waypoint_column():
    time = np.arange(0.0, 10.1, 0.5)
    base = pd.DataFrame(
        {"t": time, "aileron_cmd": np.sin(time), "roll_rate": np.cos(time)}
    )
    phases = pd.DataFrame({"t": time, "phase": "cruise", "phase_boundary": False})
    spec = ResidualChannelSpec("R1_test", ("aileron_cmd",), "roll_rate")
    X_base, y_base, meta_base = build_xy(base, spec, phases)
    with_waypoint = base.assign(waypoint_distance=_v_turn(time, 5.0))
    X_waypoint, y_waypoint, meta_waypoint = build_xy(with_waypoint, spec, phases)
    pd.testing.assert_frame_equal(X_base, X_waypoint)
    pd.testing.assert_series_equal(y_base, y_waypoint)
    assert meta_base["output_rows"] == meta_waypoint["output_rows"]
