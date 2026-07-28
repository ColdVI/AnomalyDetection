import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "configs/four_dataset_probabilistic_gpu_v2_split_manifest.json"
CONFIG_PATH = ROOT / "configs/four_dataset_probabilistic_gpu_v2.json"


def _payload():
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def test_balanced_v2_roles_are_disjoint_and_normal_only_before_test():
    payload = _payload()
    for dataset, source in payload["sources"].items():
        labels = source["flight_labels"]
        normal = source["normal_label"]
        roles = source["roles"]
        role_sets = {role: set(values) for role, values in roles.items()}
        names = list(role_sets)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                assert role_sets[left].isdisjoint(role_sets[right]), (
                    dataset,
                    left,
                    right,
                )
        assert all(labels[source_id] == normal for source_id in roles["train"])
        assert all(labels[source_id] == normal for source_id in roles["val"])
        assert any(labels[source_id] == normal for source_id in roles["test"])
        assert any(labels[source_id] != normal for source_id in roles["test"])


def test_balanced_v2_normal_ratios_and_primary_test_policy():
    payload = _payload()
    for dataset, source in payload["sources"].items():
        counts = source["counts"]
        normal_total = counts["normal_sources"]
        assert abs(counts["train"] - normal_total * 0.70) <= 1.0
        assert abs(counts["val"] - normal_total * 0.15) <= 1.0
        clean_test = counts["primary_test_normal"]
        assert abs(clean_test - normal_total * 0.15) <= 1.0
        anomaly_labels = {
            label
            for label in source["flight_labels"].values()
            if label != source["normal_label"]
        }
        primary_labels = set(source["primary_test_labels"]) - {source["normal_label"]}
        assert primary_labels == anomaly_labels
        if clean_test >= len(anomaly_labels):
            assert counts["primary_test_anomaly"] == clean_test


def test_rflymad_v2_matches_reusable_bundle_scope_and_preserves_domain_coverage():
    payload = _payload()["sources"]["rflymad"]
    manifest = pd.read_parquet(
        ROOT / "artifacts/rfly_full/v2/dataset_manifest.parquet"
    )
    development = manifest.loc[manifest["split"].astype(str).eq("development")].copy()
    development["fault_family"] = development["fault_family"].astype(str)
    development["cv_fold"] = pd.to_numeric(
        development["cv_fold"], errors="raise"
    ).astype(int)
    reusable = development.loc[
        development["fault_family"].eq("NoFault")
        | development["cv_fold"].eq(1)
    ]
    reusable_ids = set(reusable["canonical_case_id"].astype(str))
    primary_ids = set().union(
        *(
            set(payload["roles"][role])
            for role in ("train", "val", "test", "stress_test")
        )
    )
    assert primary_ids == reusable_ids
    normal = development.loc[development["fault_family"].astype(str).eq("NoFault")]
    domain = normal.set_index(normal["canonical_case_id"].astype(str))["domain"].astype(str)
    for role in ("train", "val", "test"):
        normal_role = [
            source_id
            for source_id in payload["roles"][role]
            if payload["flight_labels"][source_id] == "NoFault"
        ]
        assert set(domain.loc[normal_role]) == {"SIL", "HIL", "Real"}


def test_v2_config_points_to_new_contract_without_changing_v1_epoch_grid():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["candidate_namespace"] == "four_dataset_probabilistic_gpu_v2"
    assert config["split_name"] == "balanced_v2"
    assert config["common"]["epochs"] == 30
    assert config["common"]["history_rows"] == 32
    assert config["common"]["validation_alarm_quantile"] == 0.995
    assert config["datasets"]["rflymad"]["explicit_split_manifest_path"] == (
        "configs/four_dataset_probabilistic_gpu_v2_split_manifest.json"
    )
