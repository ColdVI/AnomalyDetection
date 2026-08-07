"""Focused contracts for the contextual_physics_v2 truth-v2 evaluator."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from scripts.adsb_contextual_physics_v2_truth_v2_eval import (
    _audit_truth_v2_ground_conflicts,
    _persistence_detector,
    _with_exposure_bounds,
)


def test_exposure_bounds_reset_at_channel_boundary() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["a", "a", "b", "b"],
            "flight_id": ["f", "f", "f", "f"],
            "timestamp_utc": [10.0, 12.0, 10.0, 12.0],
        }
    )

    bounded = _with_exposure_bounds(frame, "timestamp_utc")

    assert bounded["t_start"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert bounded["t_end"].tolist() == [10.0, 12.0, 10.0, 12.0]


def test_persistence_detector_uses_frozen_calibration_multiplier() -> None:
    detector = _persistence_detector(
        {"persistence_v2": {"reference_shift_multiplier": 1.37}}
    )

    assert detector.config.reference_shift_multiplier == 1.37
    assert detector.config.max_gap_s == 30.0
    assert detector.config.missing_reset_s == 60.0
    assert detector.config.surprise_clip == 6.0


def test_truth_ground_audit_builds_paired_flight_quarantine(tmp_path: Path) -> None:
    bad = pd.DataFrame(
        {
            "flight_id": ["bad", "bad", "good"],
            "timestamp_utc": [10.0, 10.0, 10.0],
            "on_ground": [True, False, False],
        }
    )
    clean = tmp_path / "clean.parquet"
    injected = tmp_path / "injected.parquet"
    bad.to_parquet(clean, index=False)
    bad.to_parquet(injected, index=False)

    conflicts, quarantined, counts = _audit_truth_v2_ground_conflicts(
        tmp_path, ("clean.parquet", "injected.parquet")
    )

    assert quarantined == {"bad"}
    assert len(conflicts) == 2
    assert set(conflicts["corpus_file"]) == {
        "clean.parquet",
        "injected.parquet",
    }
    assert counts == {
        "audited_files": 2,
        "audited_rows": 6,
        "conflict_records_across_files": 2,
        "unique_conflict_keys": 1,
        "clean_flights_before_quarantine": 2,
    }
