"""Event-schema compatibility contracts for contextual-v2 alarms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adsb.contextual_events import COMMON_EVENT_COLUMNS, contextual_alarm_events
from adsb.simple_anomaly import ALTITUDE_EVENT_COLUMNS, ROUTE_EVENT_COLUMNS


def test_contextual_events_share_simple_anomaly_prefix_and_merge_semantics() -> None:
    meta = pd.DataFrame(
        {
            "flight_id": ["f", "f", "f", "f"],
            "t_end": [10.0, 40.0, 101.0, 105.0],
        }
    )
    events = contextual_alarm_events(
        meta,
        np.array([True, True, True, False]),
        detector="model",
        channel="speed_residual",
        profile="bias",
        pareto_v=25.0,
        threshold_parameter="threshold_h",
        threshold_value=3.5,
        recipe="ground_speed_biased",
    )

    assert list(events.columns[: len(COMMON_EVENT_COLUMNS)]) == list(COMMON_EVENT_COLUMNS)
    assert list(ALTITUDE_EVENT_COLUMNS[:6]) == list(COMMON_EVENT_COLUMNS)
    assert list(ROUTE_EVENT_COLUMNS[:6]) == list(COMMON_EVENT_COLUMNS)
    assert len(events) == 2
    assert events.loc[0, "n_samples"] == 2
    assert events.loc[0, "duration_s"] == 30.0
    assert events.loc[1, "duration_s"] == 0.0
    assert events["event_id"].is_unique


def test_contextual_events_return_typed_empty_schema() -> None:
    events = contextual_alarm_events(
        pd.DataFrame({"flight_id": ["f"], "t_end": [1.0]}),
        np.array([False]),
        detector="cusum",
        channel="east_north_velocity_residual",
        profile="accumulation",
        pareto_v=0.1,
        threshold_parameter="threshold_h",
        threshold_value=1.0,
        recipe="position_ramp_stealthy",
    )

    assert events.empty
    assert list(events.columns[:6]) == list(COMMON_EVENT_COLUMNS)
