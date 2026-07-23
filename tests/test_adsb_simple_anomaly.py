from __future__ import annotations

import numpy as np
import pandas as pd

from adsb.simple_anomaly import (
    detect_altitude_deviation_events,
    detect_route_deviation_events,
    flight_phase,
)


def _three_phase_flight() -> pd.DataFrame:
    takeoff_n, cruise_n, landing_n = 12, 12, 12
    takeoff_alt = np.linspace(0.0, 1_100.0, takeoff_n)
    cruise_alt = np.full(cruise_n, 1_100.0)
    landing_alt = np.linspace(1_100.0, 0.0, landing_n)
    return pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": np.arange(takeoff_n + cruise_n + landing_n) * 10.0,
        "alt": np.concatenate([takeoff_alt, cruise_alt, landing_alt]),
        "vertical_rate_ms": np.concatenate([
            np.full(takeoff_n, 5.0),
            np.zeros(cruise_n),
            np.full(landing_n, -4.0),
        ]),
    })


def test_flight_phase_labels_hand_built_three_phase_flight():
    frame = _three_phase_flight()
    phase = flight_phase(frame)
    assert phase.iloc[:12].eq("takeoff").all()
    assert phase.iloc[12:24].eq("cruise").all()
    assert phase.iloc[24:].eq("landing").all()


def test_flight_phase_marks_incomplete_trace_uncertain():
    frame = _three_phase_flight().iloc[:24].copy()
    assert flight_phase(frame).eq("uncertain").all()


def test_flight_phase_preserves_unsorted_input_index():
    frame = _three_phase_flight().sample(frac=1.0, random_state=7)
    phase = flight_phase(frame)
    ordered = frame.assign(phase=phase).sort_values("timestamp_utc")
    assert ordered["phase"].iloc[:12].eq("takeoff").all()
    assert ordered["phase"].iloc[12:24].eq("cruise").all()
    assert ordered["phase"].iloc[24:].eq("landing").all()


def _altitude_event_frame(duration_samples: int) -> pd.DataFrame:
    n = 50
    alt = np.full(n, 1_000.0)
    alt[10:10 + duration_samples] = 1_200.0
    source_residual = np.zeros(n)
    source_residual[12] = 6.0
    return pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": np.arange(n) * 10.0,
        "alt": alt,
        "flight_phase": "cruise",
        "altitude_source_residual": source_residual,
    })


def test_altitude_deviation_requires_more_than_two_minutes():
    long = detect_altitude_deviation_events(_altitude_event_frame(14))
    assert len(long) == 1
    assert long.iloc[0]["duration_s"] == 130.0
    assert bool(long.iloc[0]["data_quality_suspect"])

    exactly_two_minutes = detect_altitude_deviation_events(
        _altitude_event_frame(13)
    )
    assert exactly_two_minutes.empty


def test_altitude_deviation_ignores_noncruise_rows():
    frame = _altitude_event_frame(14)
    frame.loc[10:23, "flight_phase"] = "landing"
    assert detect_altitude_deviation_events(frame).empty


def _route_frame(residual: list[float], timestamps: list[float] | None = None) -> pd.DataFrame:
    n = len(residual)
    return pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": timestamps if timestamps is not None else np.arange(n) * 10.0,
        "flight_phase": "cruise",
        "heading_residual": residual,
        "ground_speed_ms": np.full(n, 100.0),
        "east_velocity_residual": np.zeros(n),
        "north_velocity_residual": np.zeros(n),
    })


def test_route_deviation_requires_four_consecutive_samples():
    frame = _route_frame([0.0, 25.0, -30.0, 22.0, 21.0, 0.0])
    events = detect_route_deviation_events(frame)
    assert len(events) == 1
    assert events.iloc[0]["n_samples"] == 4

    assert detect_route_deviation_events(
        _route_frame([0.0, 25.0, 30.0, 22.0, 0.0])
    ).empty


def test_route_deviation_breaks_on_more_than_thirty_second_gap():
    frame = _route_frame(
        [25.0, 25.0, 25.0, 25.0], timestamps=[0.0, 10.0, 50.1, 60.1],
    )
    assert detect_route_deviation_events(frame).empty
