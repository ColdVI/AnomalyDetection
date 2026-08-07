"""Tests for adsb.models.contextual_persistence_v2 (contextual_physics_v2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.models.contextual_persistence_v2 import (
    NULL_MEAN_SURPRISE,
    CumulativeConformalPersistence,
    PersistenceV2Config,
)


def _config(**overrides) -> PersistenceV2Config:
    values = {
        "reference_shift_multiplier": 1.5,
        "threshold_h": 2.0,
        "max_gap_s": 60.0,
        "missing_reset_s": 3.0,
        "surprise_clip": 10.0,
    }
    values.update(overrides)
    return PersistenceV2Config(**values)


def _frame(
    p_values,
    times,
    *,
    channel="vertical_rate_residual",
    flights=None,
    on_ground=None,
) -> pd.DataFrame:
    n = len(p_values)
    return pd.DataFrame(
        {
            "flight_id": flights if flights is not None else ["F1"] * n,
            "timestamp_utc": times,
            "on_ground": on_ground if on_ground is not None else [False] * n,
            "channel": [channel] * n,
            "conformal_p_value": p_values,
        }
    )


def test_config_rejects_multiplier_at_or_below_one() -> None:
    with pytest.raises(ValueError):
        _config(reference_shift_multiplier=1.0)


def test_sustained_low_p_value_accumulates_and_alarms() -> None:
    # -log10(0.01) = 2.0, reference = 1.5 * 0.4343 = 0.6514 -> step gain ~1.35
    frame = _frame([0.01] * 10, list(range(0, 100, 10)))
    detector = CumulativeConformalPersistence(_config())
    result = detector.score(frame)
    # Row 0 of any flight is never evaluable (no previous timestamp to derive
    # dt from) -- identical to VectorPageCUSUM's first-row behavior.
    assert not result["persistence_v2_evaluable"].iloc[0]
    assert result["persistence_v2_evaluable"].iloc[1:].all()
    assert result["persistence_v2_state"].iloc[-1] > result["persistence_v2_state"].iloc[1]
    assert result["persistence_v2_alarm"].any()


def test_pure_null_noise_never_alarms() -> None:
    rng = np.random.default_rng(20260723)
    p_values = rng.uniform(0.05, 0.95, size=200)
    times = np.arange(0, 200 * 5, 5)
    frame = _frame(p_values, times)
    detector = CumulativeConformalPersistence(_config(reference_shift_multiplier=1.5))
    result = detector.score(frame)
    # Under the null, mean surprise ~= NULL_MEAN_SURPRISE < reference (1.5x) --
    # accumulator should drift down, not sustain an alarm.
    assert not result["persistence_v2_alarm"].any()


def test_flight_boundary_resets_state() -> None:
    frame = _frame(
        [0.01, 0.01, 0.01, 0.01],
        [0, 10, 0, 10],
        flights=["F1", "F1", "F2", "F2"],
    )
    detector = CumulativeConformalPersistence(_config())
    result = detector.score(frame)
    assert result["persistence_v2_reset_reason"].iloc[2] == "flight_start"
    assert result["persistence_v2_state"].iloc[2] == 0.0


def test_ground_transition_resets_state() -> None:
    frame = _frame(
        [0.01, 0.01, 0.01, 0.01],
        [0, 10, 20, 30],
        on_ground=[False, False, True, False],
    )
    detector = CumulativeConformalPersistence(_config())
    result = detector.score(frame)
    assert result["persistence_v2_reset_reason"].iloc[2] == "on_ground"
    assert result["persistence_v2_reset_reason"].iloc[3] == "ground_transition"
    assert result["persistence_v2_state"].iloc[3] == 0.0


def test_missing_p_value_carries_then_resets() -> None:
    # row0: flight_start, not eligible. row1: eligible, p=0.01 -> state accumulates.
    # rows 2-4: missing p, dt=1,1,3 -- missing_reset_s=3.0, elapsed reaches 3 at row4.
    frame = _frame(
        [0.01, 0.01, np.nan, np.nan, np.nan],
        [0, 10, 11, 12, 15],
    )
    detector = CumulativeConformalPersistence(_config(missing_reset_s=3.0))
    result = detector.score(frame)
    assert result["persistence_v2_state"].iloc[1] > 0.0
    # state carries through the missing rows below the reset threshold
    assert result["persistence_v2_state"].iloc[2] == result["persistence_v2_state"].iloc[1]
    assert result["persistence_v2_state"].iloc[3] == result["persistence_v2_state"].iloc[1]
    # elapsed missing reaches missing_reset_s (1+1+3=5 >= 3) by row 4 -> reset
    assert result["persistence_v2_state"].iloc[4] == 0.0


def test_long_gap_resets_state() -> None:
    frame = _frame([0.01, 0.01], [0, 1000])
    detector = CumulativeConformalPersistence(_config(max_gap_s=60.0))
    result = detector.score(frame)
    assert result["persistence_v2_reset_reason"].iloc[1] == "long_gap"


def test_channels_do_not_interfere() -> None:
    n = 6
    frame = pd.concat(
        [
            _frame([0.01] * n, list(range(0, n * 10, 10)), channel="a"),
            _frame([0.5] * n, list(range(0, n * 10, 10)), channel="b"),
        ],
        ignore_index=True,
    )
    detector = CumulativeConformalPersistence(_config())
    result = detector.score(frame)
    a_final = result["persistence_v2_state"].iloc[n - 1]
    b_final = result["persistence_v2_state"].iloc[2 * n - 1]
    assert a_final > b_final


def test_to_dict_from_dict_roundtrip() -> None:
    detector = CumulativeConformalPersistence(_config())
    payload = detector.to_dict()
    assert payload["reference_shift"] == pytest.approx(1.5 * NULL_MEAN_SURPRISE)
    restored = CumulativeConformalPersistence.from_dict(payload)
    assert restored.config == detector.config


def test_missing_required_column_raises() -> None:
    frame = _frame([0.01], [0]).drop(columns=["on_ground"])
    detector = CumulativeConformalPersistence(_config())
    with pytest.raises(KeyError):
        detector.score(frame)
