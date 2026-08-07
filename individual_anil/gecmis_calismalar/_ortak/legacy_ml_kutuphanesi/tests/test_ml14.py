"""ML-14 refresh discipline tests."""

from __future__ import annotations

import copy

import pandas as pd

from src.ml.data.splits import (
    assert_no_flight_overlap,
    build_split_manifest,
    session_of,
)


def _rows(source_id: str, label: str) -> list[dict[str, str]]:
    return [{"source_id": source_id, "label": label}] * 2


def _ml14_refresh_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for i in range(8):
        rows += _rows(f"old-clean-{i}/normal", "normal")
    rows += _rows("old-final/normal", "normal")
    rows += _rows("old-final/fault", "external_position")
    rows += _rows("old-dev/normal", "normal")
    rows += _rows("old-dev/fault", "external_position")
    for i in range(5):
        rows += _rows(f"new-anom-{i}/normal", "normal")
        rows += _rows(f"new-anom-{i}/fault", "external_position")
    for i in range(5):
        rows += _rows(f"new-clean-{i}/normal", "normal")
    return pd.DataFrame(rows)


def _previous_source_config() -> dict:
    return {
        "flight_labels": {
            **{f"old-clean-{i}/normal": "normal" for i in range(8)},
            "old-final/normal": "normal",
            "old-final/fault": "external_position",
            "old-dev/normal": "normal",
            "old-dev/fault": "external_position",
        },
        "splits": {
            "split_00": {
                "seed": 0,
                "train": [f"old-clean-{i}/normal" for i in range(4)],
                "val": [f"old-clean-{i}/normal" for i in range(4, 6)],
                "test": [
                    "old-clean-6/normal",
                    "old-clean-7/normal",
                    "old-dev/normal",
                    "old-dev/fault",
                ],
                "final_holdout": ["old-final/normal", "old-final/fault"],
                "exploration": [],
            }
        },
    }


def _strip_created(manifest: dict) -> dict:
    out = copy.deepcopy(manifest)
    out.pop("created_utc", None)
    return out


def test_ml14_frozen_holdout_preserves_old_final_and_blocks_old_dev_leak():
    manifest = build_split_manifest(
        {"uav_sead": _ml14_refresh_frame()},
        quotas={"uav_sead": (2, 2)},
        frozen_holdout={"uav_sead": _previous_source_config()},
    )
    split = manifest["sources"]["uav_sead"]["splits"]["split_00"]
    final = set(split["final_holdout"])
    development = set(split["train"]) | set(split["val"]) | set(split["test"])

    assert {"old-final/normal", "old-final/fault"} <= final
    assert not {"old-dev/normal", "old-dev/fault"} & final
    assert not final & development
    assert_no_flight_overlap(split)


def test_ml14_frozen_holdout_adds_only_new_anomaly_sessions_by_fixed_fraction():
    manifest = build_split_manifest(
        {"uav_sead": _ml14_refresh_frame()},
        quotas={"uav_sead": (2, 2)},
        frozen_holdout={"uav_sead": _previous_source_config()},
    )
    split = manifest["sources"]["uav_sead"]["splits"]["split_00"]
    final_sessions = {session_of(f) for f in split["final_holdout"]}
    old_sessions = {session_of(f) for f in _previous_source_config()["flight_labels"]}
    new_final_sessions = sorted(s for s in final_sessions if s not in old_sessions)

    assert "old-final" in final_sessions
    assert "old-dev" not in final_sessions
    assert len(new_final_sessions) == round(5 * 0.30)
    for session in new_final_sessions:
        assert f"{session}/normal" in split["final_holdout"]
        assert f"{session}/fault" in split["final_holdout"]


def test_ml14_frozen_holdout_is_fixed_across_experiment_seeds():
    manifest = build_split_manifest(
        {"uav_sead": _ml14_refresh_frame()},
        quotas={"uav_sead": (2, 2)},
        frozen_holdout={"uav_sead": _previous_source_config()},
    )
    splits = manifest["sources"]["uav_sead"]["splits"].values()
    holdouts = {tuple(split["final_holdout"]) for split in splits}
    assert len(holdouts) == 1


def test_build_split_manifest_without_frozen_holdout_preserves_existing_behavior():
    kwargs = {"quotas": {"uav_sead": (2, 2)}}
    plain = build_split_manifest({"uav_sead": _ml14_refresh_frame()}, **kwargs)
    explicit_none = build_split_manifest(
        {"uav_sead": _ml14_refresh_frame()},
        frozen_holdout=None,
        **kwargs,
    )
    assert _strip_created(plain) == _strip_created(explicit_none)
