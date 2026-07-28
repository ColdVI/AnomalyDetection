"""Build compatibility roles/configs for the v3.1 training runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V31 = ROOT / "configs/four_dataset_probabilistic_v31_split_manifest.json"
V2_CONFIG = ROOT / "configs/four_dataset_probabilistic_gpu_v2.json"
REGISTRY = ROOT / "artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet"
OUTPUT_ROLES = ROOT / "configs/four_dataset_probabilistic_v31_runner_roles.json"


class RunnerContractError(RuntimeError):
    """Raised when v3.1 cannot be projected without changing membership."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(v31: dict[str, Any], registry: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = {
        dataset: dict(
            zip(
                part["source_id"].astype(str),
                part["flight_label"].astype(str),
            )
        )
        for dataset, part in registry.groupby("dataset", sort=True)
    }
    rfly = v31["datasets"]["rflymad"]["source_ids"]
    rfly_test = sorted(rfly["normal_test"] + rfly["anomaly_dev"])
    if set(rfly_test) & set(rfly["final_fault_test"]):
        raise RunnerContractError("Final fault sources leaked into development test")
    sources: dict[str, Any] = {
        "rflymad": {
            "normal_label": "NoFault",
            "roles": {
                "train": rfly["train"],
                "val": rfly["val"],
                "test": rfly_test,
            },
            "flight_labels": labels["rflymad"],
            "final_fault_test_excluded": True,
            "final_fault_test_seal_sha256": v31["datasets"]["rflymad"][
                "final_fault_test_seal_sha256"
            ],
        }
    }
    base_v3 = json.loads(
        (ROOT / v31["inputs"]["base_v3_manifest_path"]).read_text(encoding="utf-8")
    )
    folds: dict[str, Any] = {}
    for fold in base_v3["datasets"]["alfa"]["folds"]:
        name = f"grouped_cv_4_fold_{int(fold['fold'])}"
        folds[name] = {
            "seed": int(v31["seed"]),
            "train": fold["source_ids"]["train"],
            "val": fold["source_ids"]["val"],
            "test": fold["source_ids"]["test"],
        }
    sources["alfa"] = {
        "normal_label": "normal",
        "roles": folds["grouped_cv_4_fold_0"],
        "splits": folds,
        "flight_labels": labels["alfa"],
    }
    roles = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "created_date": "2026-07-28",
        "source_protocol": V31.relative_to(ROOT).as_posix(),
        "source_protocol_sha256": _sha256(V31),
        "development_scope_excludes_final_fault_test": True,
        "sources": sources,
    }

    v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    common = copy.deepcopy(v2["common"])
    common["select_best_validation_checkpoint"] = True
    common["training_sampler"] = "group_then_source_balanced_epoch_quota"
    common["epochs"] = 30
    base_config = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "contract_date": "2026-07-28",
        "split_manifest": OUTPUT_ROLES.relative_to(ROOT).as_posix(),
        "objective": (
            "normal-only causal Gaussian forecasting with group/source-balanced "
            "optimizer exposure and sealed final fault test"
        ),
        "common": common,
        "prohibitions": {
            "no_row_random_split": True,
            "no_anomaly_rows_in_optimizer": True,
            "no_validation_or_test_preprocessing_fit": True,
            "no_final_fault_test_during_development": True,
            "no_test_threshold_selection": True,
        },
    }
    configs: dict[str, Any] = {}
    for fold in range(4):
        spec = copy.deepcopy(v2["datasets"]["alfa"])
        spec["evaluation_status"] = "four-fold group-safe CV smoke; no single-split claim"
        spec["group_registry_path"] = REGISTRY.relative_to(ROOT).as_posix()
        cfg = copy.deepcopy(base_config)
        cfg["split_name"] = f"grouped_cv_4_fold_{fold}"
        cfg["datasets"] = {"alfa": spec}
        configs[f"alfa_fold_{fold}"] = cfg
    rfly_spec = copy.deepcopy(v2["datasets"]["rflymad"])
    rfly_spec["explicit_split_manifest_path"] = OUTPUT_ROLES.relative_to(ROOT).as_posix()
    rfly_spec["evaluation_status"] = (
        "v3.1 development: normal test plus anomaly-dev; final fault test sealed"
    )
    rfly_spec["group_registry_path"] = REGISTRY.relative_to(ROOT).as_posix()
    rfly_cfg = copy.deepcopy(base_config)
    rfly_cfg["split_name"] = "rflymad_v31_development"
    rfly_cfg["datasets"] = {"rflymad": rfly_spec}
    configs["rflymad"] = rfly_cfg
    return roles, configs


def _write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    v31 = json.loads(V31.read_text(encoding="utf-8"))
    registry = pd.read_parquet(REGISTRY)
    roles, configs = build(v31, registry)
    _write(OUTPUT_ROLES, roles, args.overwrite)
    for name, config in configs.items():
        _write(
            ROOT / f"configs/four_dataset_probabilistic_v31_{name}.json",
            config,
            args.overwrite,
        )
    print(
        json.dumps(
            {
                "roles": OUTPUT_ROLES.relative_to(ROOT).as_posix(),
                "configs": sorted(configs),
                "rflymad_role_counts": {
                    role: len(values)
                    for role, values in roles["sources"]["rflymad"]["roles"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
