import pandas as pd
import pytest
import json
from pathlib import Path

from scripts.build_four_dataset_probabilistic_v3_group_splits import (
    GroupSplitContractError,
    alfa_outer_fold_assignments,
    group_table,
    rfly_domain_normal_roles,
    validate_protocol,
    weighted_normal_group_roles,
)


def _registry(rows):
    return pd.DataFrame(rows)


def test_weighted_roles_keep_whole_normal_groups_and_populate_three_roles():
    groups = pd.DataFrame(
        {
            "group_id": [f"g{i}" for i in range(8)],
            "source_count": [9, 7, 5, 4, 3, 2, 1, 1],
            "composition": ["normal_only"] * 8,
        }
    )
    first = weighted_normal_group_roles(groups, "uav_sead")
    second = weighted_normal_group_roles(groups, "uav_sead")
    assert first == second
    assert set(first) == set(groups["group_id"])
    assert set(first.values()) == {"train", "val", "test"}


def test_rfly_domain_roles_hold_out_one_group_per_domain_for_val_and_test():
    rows = []
    for domain, count in (("HIL", 5), ("SIL", 5), ("Real", 3)):
        for index in range(count):
            rows.append(
                {
                    "group_id": f"{domain}-{index}",
                    "composition": "normal_only",
                    "domain": domain,
                    "domain_count": 1,
                }
            )
    roles = rfly_domain_normal_roles(pd.DataFrame(rows))
    for domain, count in (("HIL", 5), ("SIL", 5), ("Real", 3)):
        selected = {key: value for key, value in roles.items() if key.startswith(domain)}
        assert list(selected.values()).count("val") == 1
        assert list(selected.values()).count("test") == 1
        assert list(selected.values()).count("train") == count - 2


def test_alfa_outer_folds_place_two_pure_normal_groups_in_each_fold():
    groups = pd.DataFrame(
        {
            "group_id": [f"n{i}" for i in range(8)] + [f"a{i}" for i in range(8)],
            "composition": ["normal_only"] * 8 + ["anomaly_only"] * 8,
            "label_signature": ["normal"] * 8 + ["engine_fault"] * 8,
        }
    )
    assignments = alfa_outer_fold_assignments(groups)
    for fold in range(4):
        assert sum(assignments[f"n{i}"] == fold for i in range(8)) == 2
        assert sum(assignments[f"a{i}"] == fold for i in range(8)) == 2


def test_protocol_validator_rejects_anomaly_in_validation():
    frame = _registry(
        [
            {
                "dataset": "x",
                "group_id": "g1",
                "source_id": "normal-train",
                "flight_label": "normal",
                "is_normal": True,
            },
            {
                "dataset": "x",
                "group_id": "g2",
                "source_id": "anomaly-val",
                "flight_label": "fault",
                "is_normal": False,
            },
            {
                "dataset": "x",
                "group_id": "g3",
                "source_id": "normal-test",
                "flight_label": "normal",
                "is_normal": True,
            },
            {
                "dataset": "x",
                "group_id": "g4",
                "source_id": "anomaly-test",
                "flight_label": "fault",
                "is_normal": False,
            },
        ]
    )
    materialised = {
        "group_ids": {
            "train": ["g1"],
            "val": ["g2"],
            "test": ["g3", "g4"],
            "stress_test": [],
        },
        "source_ids": {
            "train": ["normal-train"],
            "val": ["anomaly-val"],
            "test": ["normal-test", "anomaly-test"],
            "stress_test": [],
        },
    }
    with pytest.raises(GroupSplitContractError, match="Anomaly leaked"):
        validate_protocol(frame, materialised)


def test_group_table_distinguishes_normal_mixed_and_anomaly_groups():
    frame = _registry(
        [
            {"dataset": "x", "group_id": "n", "source_id": "n1", "flight_label": "normal", "is_normal": True},
            {"dataset": "x", "group_id": "m", "source_id": "m1", "flight_label": "normal", "is_normal": True},
            {"dataset": "x", "group_id": "m", "source_id": "m2", "flight_label": "fault", "is_normal": False},
            {"dataset": "x", "group_id": "a", "source_id": "a1", "flight_label": "fault", "is_normal": False},
        ]
    )
    composition = group_table(frame).set_index("group_id")["composition"].to_dict()
    assert composition == {"a": "anomaly_only", "m": "mixed", "n": "normal_only"}


def test_generated_v3_manifest_records_passed_contracts_and_attack_no_go():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/four_dataset_probabilistic_v3_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (
            root
            / "artifacts/four_dataset_probabilistic_v3/group_split_v1_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["all_contracts_passed"] is True
    assert report["training_started"] is False
    for dataset in ("rflymad", "uav_sead"):
        assert report["datasets"][dataset]["group_overlap_count"] == 0
        assert report["datasets"][dataset]["source_overlap_count"] == 0
        assert report["datasets"][dataset]["roles"]["train"]["anomaly_source_count"] == 0
        assert report["datasets"][dataset]["roles"]["val"]["anomaly_source_count"] == 0
    attack = manifest["datasets"]["uav_attack"]
    assert attack["status"] == "no_go_group_safe_split_not_feasible"
    assert attack["physics_v3_training_authorized"] is False
    assert len(attack["source_ids_preserved_unassigned"]) == 19
