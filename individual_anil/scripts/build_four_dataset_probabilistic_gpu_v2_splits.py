"""Build deterministic balanced-source v2 contracts for four UAV datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs/four_dataset_probabilistic_gpu_v1.json"
GOLD_SPLIT = ROOT / "data/gold/ml_features/split_manifest.json"
RFLY_MANIFEST = ROOT / "artifacts/rfly_full/v2/dataset_manifest.parquet"
V2_SPLIT = ROOT / "configs/four_dataset_probabilistic_gpu_v2_split_manifest.json"
V2_CONFIG = ROOT / "configs/four_dataset_probabilistic_gpu_v2.json"
SEED = 20260727
NORMAL_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class SplitContractError(RuntimeError):
    pass


def _key(dataset: str, role: str, source_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}:{dataset}:{role}:{source_id}".encode("utf-8")
    ).hexdigest()


def _counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise SplitContractError("At least three normal sources are required")
    train = max(1, round(total * NORMAL_RATIOS["train"]))
    val = max(1, round(total * NORMAL_RATIOS["val"]))
    test = total - train - val
    if test < 1:
        train -= 1 - test
        test = 1
    if train < 1:
        raise SplitContractError("Normal train role became empty")
    return train, val, test


def _split_normal(
    dataset: str,
    source_ids: list[str],
    strata: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for source_id in source_ids:
        stratum = strata[source_id] if strata is not None else "all"
        buckets.setdefault(str(stratum), []).append(source_id)
    result = {"train": [], "val": [], "test": []}
    for stratum, members in sorted(buckets.items()):
        ordered = sorted(members, key=lambda value: _key(dataset, f"normal:{stratum}", value))
        train_n, val_n, _ = _counts(len(ordered))
        result["train"].extend(ordered[:train_n])
        result["val"].extend(ordered[train_n : train_n + val_n])
        result["test"].extend(ordered[train_n + val_n :])
    for role in result:
        result[role] = sorted(
            result[role], key=lambda value: _key(dataset, f"role:{role}", value)
        )
    return result


def _balanced_anomaly_sample(
    dataset: str,
    anomaly_ids: list[str],
    labels: dict[str, str],
    requested: int,
) -> tuple[list[str], list[str]]:
    by_label: dict[str, list[str]] = {}
    for source_id in anomaly_ids:
        by_label.setdefault(labels[source_id], []).append(source_id)
    target = min(len(anomaly_ids), max(requested, len(by_label)))
    allocation = {label: 1 for label in by_label}
    remaining = target - len(allocation)
    if remaining > 0:
        population = sum(len(values) for values in by_label.values())
        raw = {
            label: remaining * len(values) / population
            for label, values in by_label.items()
        }
        for label in allocation:
            allocation[label] += min(
                len(by_label[label]) - 1, int(math.floor(raw[label]))
            )
        left = target - sum(allocation.values())
        candidates = sorted(
            by_label,
            key=lambda label: (
                -(raw[label] - math.floor(raw[label])),
                _key(dataset, "label", label),
            ),
        )
        while left:
            progressed = False
            for label in candidates:
                if allocation[label] < len(by_label[label]):
                    allocation[label] += 1
                    left -= 1
                    progressed = True
                    if left == 0:
                        break
            if not progressed:
                raise SplitContractError("Could not allocate anomaly test sources")
    selected: list[str] = []
    for label, members in sorted(by_label.items()):
        ordered = sorted(
            members, key=lambda value: _key(dataset, f"anomaly:{label}", value)
        )
        selected.extend(ordered[: allocation[label]])
    chosen = set(selected)
    stress = [source_id for source_id in anomaly_ids if source_id not in chosen]
    return (
        sorted(selected, key=lambda value: _key(dataset, "test-anomaly", value)),
        sorted(stress, key=lambda value: _key(dataset, "stress", value)),
    )


def _gold_sources(
    dataset: str, v1: dict[str, Any], frozen: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str] | None]:
    path = ROOT / v1["datasets"][dataset]["data_path"]
    available = set(
        pd.read_parquet(path, columns=["source_id"])["source_id"].astype(str).unique()
    )
    normal_label = str(v1["datasets"][dataset]["normal_label"])
    labels = {
        str(source_id): normal_label if str(label) == "normal" else str(label)
        for source_id, label in frozen["sources"][dataset]["flight_labels"].items()
        if str(source_id) in available
    }
    if labels.keys() != available:
        missing = sorted(available - labels.keys())
        raise SplitContractError(f"{dataset}: labels missing for {missing[:5]}")
    return labels, None


def _rfly_sources() -> tuple[dict[str, str], dict[str, str]]:
    manifest = pd.read_parquet(RFLY_MANIFEST)
    development = manifest.loc[manifest["split"].astype(str).eq("development")].copy()
    development["fault_family"] = development["fault_family"].astype(str)
    development["cv_fold"] = pd.to_numeric(
        development["cv_fold"], errors="raise"
    ).astype(int)
    # Reuse the already transferred v1 RflyMAD ZIP exactly: it contains every
    # development NoFault source plus every fold-1 primary-evaluation source.
    # Other anomaly folds are deliberately outside the v2 transfer contract.
    reusable = development.loc[
        development["fault_family"].eq("NoFault")
        | development["cv_fold"].eq(1)
    ].copy()
    ids = reusable["canonical_case_id"].astype(str)
    if ids.duplicated().any():
        raise SplitContractError("RflyMAD development source ids are not unique")
    labels = dict(zip(ids, reusable["fault_family"]))
    strata = dict(zip(ids, reusable["domain"].astype(str)))
    return labels, strata


def _build_dataset(
    dataset: str,
    labels: dict[str, str],
    normal_label: str,
    strata: dict[str, str] | None,
) -> dict[str, Any]:
    normal_ids = sorted(source_id for source_id, label in labels.items() if label == normal_label)
    anomaly_ids = sorted(source_id for source_id, label in labels.items() if label != normal_label)
    normal_roles = _split_normal(dataset, normal_ids, strata)
    primary_anomaly, stress = _balanced_anomaly_sample(
        dataset, anomaly_ids, labels, len(normal_roles["test"])
    )
    roles = {
        "train": normal_roles["train"],
        "val": normal_roles["val"],
        "test": sorted(
            normal_roles["test"] + primary_anomaly,
            key=lambda value: _key(dataset, "primary-test", value),
        ),
        "stress_test": stress,
    }
    for left, left_ids in roles.items():
        for right, right_ids in roles.items():
            if left < right and set(left_ids) & set(right_ids):
                raise SplitContractError(f"{dataset}: overlap between {left} and {right}")
    if any(labels[source_id] != normal_label for source_id in roles["train"] + roles["val"]):
        raise SplitContractError(f"{dataset}: anomaly leaked into optimizer/calibration")
    test_counts = Counter(labels[source_id] for source_id in roles["test"])
    return {
        "normal_label": normal_label,
        "roles": roles,
        "splits": {
            "balanced_v2": {
                "seed": SEED,
                "train": roles["train"],
                "val": roles["val"],
                "test": roles["test"],
            }
        },
        "flight_labels": dict(sorted(labels.items())),
        "counts": {
            "all_sources": len(labels),
            "normal_sources": len(normal_ids),
            "anomaly_sources": len(anomaly_ids),
            "train": len(roles["train"]),
            "val": len(roles["val"]),
            "primary_test": len(roles["test"]),
            "primary_test_normal": sum(
                labels[source_id] == normal_label for source_id in roles["test"]
            ),
            "primary_test_anomaly": sum(
                labels[source_id] != normal_label for source_id in roles["test"]
            ),
            "stress_test_anomaly": len(stress),
        },
        "primary_test_labels": dict(sorted(test_counts.items())),
    }


def _write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    v1 = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(GOLD_SPLIT.read_text(encoding="utf-8"))
    sources: dict[str, Any] = {}
    for dataset in ("alfa", "uav_attack", "uav_sead"):
        labels, strata = _gold_sources(dataset, v1, frozen)
        sources[dataset] = _build_dataset(
            dataset, labels, str(v1["datasets"][dataset]["normal_label"]), strata
        )
    rfly_labels, rfly_strata = _rfly_sources()
    sources["rflymad"] = _build_dataset(
        "rflymad", rfly_labels, "NoFault", rfly_strata
    )
    split_payload = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_gpu_v2",
        "seed": SEED,
        "split_unit": "source/flight; never row-random",
        "normal_split_policy": NORMAL_RATIOS,
        "primary_test_policy": (
            "all held-out normal sources plus a deterministic label-stratified "
            "anomaly sample; at least one source per anomaly label where possible"
        ),
        "stress_test_policy": "remaining anomaly sources; not used by the primary v2 report",
        "sources": sources,
    }
    v2 = json.loads(json.dumps(v1))
    v2["candidate_namespace"] = "four_dataset_probabilistic_gpu_v2"
    v2["split_manifest"] = V2_SPLIT.relative_to(ROOT).as_posix()
    v2["split_name"] = "balanced_v2"
    v2["objective"] = (
        "normal-only causal multivariate one-step forecasting with learned "
        "per-channel Gaussian scale; balanced source-level primary evaluation"
    )
    rfly = v2["datasets"]["rflymad"]
    for key in ("train_folds", "validation_fold", "test_fold"):
        rfly.pop(key, None)
    rfly["explicit_split_manifest_path"] = V2_SPLIT.relative_to(ROOT).as_posix()
    rfly["evaluation_status"] = (
        "development-only balanced primary test; Real/Wind transfer gates remain failed"
    )
    v2["prohibitions"]["no_v1_artifact_reinterpretation"] = True
    v2["prohibitions"]["no_stress_test_threshold_selection"] = True
    _write(V2_SPLIT, split_payload, args.overwrite)
    _write(V2_CONFIG, v2, args.overwrite)
    print(
        json.dumps(
            {dataset: payload["counts"] for dataset, payload in sources.items()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
