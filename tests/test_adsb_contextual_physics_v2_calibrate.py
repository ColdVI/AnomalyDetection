"""Unit contracts for contextual_physics_v2 natural calibration helpers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from adsb.models.contextual_persistence_v2 import NULL_MEAN_SURPRISE
from scripts.adsb_contextual_physics_v2_calibrate import (
    _audit_on_ground_conflicts,
    _derive_reference_multiplier,
    _select_nearest,
)


def test_reference_multiplier_follows_frozen_formula() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["a", "a", "b", "b"],
            "conformal_p_value": [0.1, 0.1, 0.01, 0.01],
        }
    )
    multiplier, means = _derive_reference_multiplier(frame)

    expected = np.ceil(100 * max(1.01, 1.10 * means["b"] / NULL_MEAN_SURPRISE)) / 100
    assert multiplier == expected
    assert means["b"] > means["a"]


def test_nearest_burden_uses_conservative_threshold_tie_break() -> None:
    curve = [
        {"threshold_h": 1.0, "alert_episodes_per_scoreable_flight_hour": 0.5},
        {"threshold_h": 2.0, "alert_episodes_per_scoreable_flight_hour": 1.5},
    ]
    chosen = _select_nearest(
        curve, target_per_hour=1.0, parameter="threshold_h", higher_is_conservative=True
    )

    assert chosen["threshold_h"] == 2.0


def test_nearest_burden_uses_lower_alpha_tie_break() -> None:
    curve = [
        {"alpha": 0.01, "alert_episodes_per_scoreable_flight_hour": 0.5},
        {"alpha": 0.02, "alert_episodes_per_scoreable_flight_hour": 1.5},
    ]
    chosen = _select_nearest(
        curve, target_per_hour=1.0, parameter="alpha", higher_is_conservative=False
    )

    assert chosen["alpha"] == 0.01


def test_ground_conflict_audit_quarantines_entire_affected_flight() -> None:
    class Source:
        path = Path("part.parquet")

        @staticmethod
        def load() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "flight_id": ["bad", "bad", "bad", "good"],
                    "timestamp_utc": [10.0, 10.0, 11.0, 10.0],
                    "on_ground": [True, False, False, False],
                }
            )

    conflicts, quarantined, counts = _audit_on_ground_conflicts([Source()])

    assert quarantined == {"bad"}
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["flight_id"] == "bad"
    assert conflicts.iloc[0]["observed_on_ground_values"] == '["False","True"]'
    assert counts == {
        "audited_parts": 1,
        "audited_feature_rows": 4,
    }
