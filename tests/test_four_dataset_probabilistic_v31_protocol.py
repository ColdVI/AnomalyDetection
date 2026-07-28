import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_four_dataset_probabilistic_v31_protocol import (
    V31ContractError,
    canonical_sha,
    resolve_sources,
    split_anomaly_groups,
)
from scripts.four_dataset_probabilistic_v31_runner import allocate_equal


ROOT = Path(__file__).resolve().parents[1]


def test_anomaly_partition_is_deterministic_group_level_and_populates_both_roles():
    groups = pd.DataFrame(
        {
            "group_id": ["a", "b", "c", "d", "e"],
            "source_count": [80, 70, 30, 20, 5],
            "domain": ["SIL", "SIL", "HIL", "HIL", "Real"],
            "fault_family": ["Motor", "Motor", "Sensor", "Sensor", "Motor"],
        }
    )
    first = split_anomaly_groups(groups)
    assert first == split_anomaly_groups(groups)
    assert set(first) == set(groups["group_id"])
    assert set(first.values()) == {"anomaly_dev", "final_fault_test"}
    assert first["a"] != first["b"]
    assert first["c"] != first["d"]


def test_final_fault_role_requires_explicit_one_shot_access():
    manifest = {
        "datasets": {
            "rflymad": {
                "source_ids": {
                    "train": ["n1"],
                    "val": ["n2"],
                    "normal_test": ["n3"],
                    "anomaly_dev": ["a1"],
                    "final_fault_test": ["a2"],
                }
            }
        }
    }
    development = resolve_sources(manifest, "development")
    assert "final_fault_test" not in development
    with pytest.raises(V31ContractError, match="open_final"):
        resolve_sources(manifest, "final_evaluation")
    final = resolve_sources(manifest, "final_evaluation", open_final=True)
    assert final == {"normal_test": ("n3",), "final_fault_test": ("a2",)}


def test_generated_v31_contract_preserves_v3_and_seals_final_fault_test():
    v3_path = ROOT / "configs/four_dataset_probabilistic_v3_split_manifest.json"
    v31 = json.loads(
        (ROOT / "configs/four_dataset_probabilistic_v31_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (ROOT / "artifacts/four_dataset_probabilistic_v31/split_report.json").read_text(
            encoding="utf-8"
        )
    )
    roles = v31["datasets"]["rflymad"]["source_ids"]
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    assert roles["train"] == v3["datasets"]["rflymad"]["source_ids"]["train"]
    assert roles["val"] == v3["datasets"]["rflymad"]["source_ids"]["val"]
    assert set(roles["normal_test"] + roles["anomaly_dev"] + roles["final_fault_test"]) == set(
        v3["datasets"]["rflymad"]["source_ids"]["test"]
    )
    assert not set(roles["anomaly_dev"]) & set(roles["final_fault_test"])
    seal = canonical_sha(
        {
            "group_ids": v31["datasets"]["rflymad"]["group_ids"]["final_fault_test"],
            "source_ids": roles["final_fault_test"],
        }
    )
    assert seal == v31["datasets"]["rflymad"]["final_fault_test_seal_sha256"]
    assert report["all_contracts_passed"] is True
    assert report["rflymad"]["group_overlap_count"] == 0
    assert report["rflymad"]["source_overlap_count"] == 0
    assert v31["datasets"]["uav_sead"]["training_authorized"] is False
    assert v31["datasets"]["uav_attack"]["main_training_authorized"] is False


def test_preflight_artifacts_enforce_train_only_fit_and_keep_sead_on_hold():
    evaluation = json.loads(
        (ROOT / "configs/four_dataset_probabilistic_v31_evaluation_contract.json").read_text(
            encoding="utf-8"
        )
    )
    normal_audit = json.loads(
        (
            ROOT
            / "artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json"
        ).read_text(encoding="utf-8")
    )
    sead_audit = json.loads(
        (
            ROOT
            / "artifacts/four_dataset_probabilistic_v31/uav_sead_session_key_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert evaluation["training"]["optimizer_roles"] == ["train"]
    assert evaluation["training"]["preprocessing_fit_roles"] == ["train"]
    assert evaluation["development"]["final_fault_test_visible"] is False
    assert evaluation["uncertainty"]["row_bootstrap_allowed"] is False
    assert normal_audit["contract_passed"] is True
    assert normal_audit["test_data_read"] is False
    assert normal_audit["hard_checks"]["realized_fault_ranges_absent"] == "pass"
    assert normal_audit["coverage"]["source_count"] == 260
    assert normal_audit["coverage"]["feature_finite_fraction"]["battery_voltage"] == 0.0
    assert sead_audit["scope"]["audit_complete"] is True
    assert sead_audit["training_authorized"] is False
    assert sead_audit["decision"] == "strict_parent_retained_no_defensible_refined_key_yet"


def test_balanced_quota_preserves_total_and_differs_by_at_most_one():
    quota = allocate_equal(101, ["g3", "g1", "g2"], 20260728)
    assert sum(quota.values()) == 101
    assert max(quota.values()) - min(quota.values()) == 1
    assert quota == allocate_equal(101, ["g2", "g3", "g1"], 20260728)


def test_runner_roles_exclude_final_fault_and_alfa_smoke_uses_best_validation():
    roles = json.loads(
        (ROOT / "configs/four_dataset_probabilistic_v31_runner_roles.json").read_text(
            encoding="utf-8"
        )
    )
    v31 = json.loads(
        (ROOT / "configs/four_dataset_probabilistic_v31_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    development = set().union(*map(set, roles["sources"]["rflymad"]["roles"].values()))
    final_fault = set(v31["datasets"]["rflymad"]["source_ids"]["final_fault_test"])
    assert development.isdisjoint(final_fault)
    assert len(roles["sources"]["rflymad"]["roles"]["test"]) == 648
    flags = []
    for fold in range(4):
        report = json.loads(
            (
                ROOT
                / f"artifacts/four_dataset_probabilistic_v31/runs/alfa_fold_{fold}/training_report.json"
            ).read_text(encoding="utf-8")
        )
        history = json.loads(
            (
                ROOT
                / f"artifacts/four_dataset_probabilistic_v31/runs/alfa_fold_{fold}/training_history.json"
            ).read_text(encoding="utf-8")
        )
        selected = min(
            history,
            key=lambda row: (row["mean_validation_gaussian_nll"], row["epoch"]),
        )
        assert report["selected_checkpoint"]["epoch"] == selected["epoch"]
        assert report["completed_epochs"] == 30
        flags.append(report["magnitude_domination_flagged_at_0_8"])
    assert flags == [False, True, True, True]
