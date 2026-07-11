from __future__ import annotations

import hashlib

import pandas as pd

from src.adsb_behavioral.injection import INJECTION_TYPES, inject_flight
from tests.adsb_behavioral.test_physics_residuals import _straight_flight


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def test_all_injections_leave_source_unchanged_and_mark_event():
    source = _straight_flight(50)
    source["source_id"] = "abc123"
    before = _digest(source)
    for injection_type in INJECTION_TYPES:
        injected = inject_flight(source, injection_type=injection_type, severity="easy")
        assert _digest(source) == before
        assert injected["is_injected_anomaly"].any()
        assert injected["event_start_utc"].nunique() == 1
        assert injected["event_end_utc"].nunique() == 1
        assert injected["flight_id"].iloc[0] != source["flight_id"].iloc[0]


def test_injection_is_deterministic():
    source = _straight_flight(50)
    first = inject_flight(source, injection_type="position_drift", severity="medium", seed=9)
    second = inject_flight(source, injection_type="position_drift", severity="medium", seed=9)
    pd.testing.assert_frame_equal(first, second)
