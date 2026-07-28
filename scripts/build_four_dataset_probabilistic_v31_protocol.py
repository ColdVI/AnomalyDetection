"""Freeze the v3.1 development/blind-test protocol without changing v3.

RflyMAD is the only dataset whose anomaly corpus is large enough to support a
group-level anomaly-development split.  Model weights and preprocessing remain
normal-only; anomaly_dev may only be used for model/event-rule selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "configs/four_dataset_probabilistic_v3_split_manifest.json"
REGISTRY = ROOT / "artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet"
OUTPUT_MANIFEST = ROOT / "configs/four_dataset_probabilistic_v31_split_manifest.json"
OUTPUT_REPORT = ROOT / "artifacts/four_dataset_probabilistic_v31/split_report.json"
SEED = 20260728
ROLES = ("train", "val", "normal_test", "anomaly_dev", "final_fault_test")


class V31ContractError(RuntimeError):
    """Raised when the v3.1 isolation or immutable-input contract fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tie(purpose: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}:{purpose}:{value}".encode()).hexdigest()


def split_anomaly_groups(groups: pd.DataFrame) -> dict[str, str]:
    """Deterministically balance whole groups within domain/fault strata."""

    required = {"group_id", "source_count", "domain", "fault_family"}
    missing = required - set(groups.columns)
    if missing:
        raise V31ContractError(f"Anomaly group fields missing: {sorted(missing)}")
    if groups.empty:
        raise V31ContractError("RflyMAD anomaly group set is empty")

    assignments: dict[str, str] = {}
    totals = {"anomaly_dev": 0, "final_fault_test": 0}
    for (domain, family), part in groups.groupby(
        ["domain", "fault_family"], sort=True, dropna=False
    ):
        ordered = part.assign(
            tie=part["group_id"].map(
                lambda value: _tie(f"{domain}:{family}", str(value))
            )
        ).sort_values(["source_count", "tie"], ascending=[False, True], kind="stable")
        previous: str | None = None
        for index, row in enumerate(ordered.itertuples(index=False)):
            if index == 0:
                role = min(
                    totals,
                    key=lambda candidate: (
                        totals[candidate],
                        _tie("role", f"{domain}:{family}:{candidate}"),
                    ),
                )
            else:
                role = (
                    "final_fault_test" if previous == "anomaly_dev" else "anomaly_dev"
                )
            assignments[str(row.group_id)] = role
            totals[role] += int(row.source_count)
            previous = role
    if set(assignments.values()) != {"anomaly_dev", "final_fault_test"}:
        raise V31ContractError("Both anomaly roles must be populated")
    return assignments


def _group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, part in frame.groupby("group_id", sort=True):
        domains = sorted(set(part["domain"].dropna().astype(str)))
        families = sorted(set(part["fault_family"].dropna().astype(str)))
        rows.append(
            {
                "group_id": str(group_id),
                "source_count": int(len(part)),
                "normal_count": int(part["is_normal"].astype(bool).sum()),
                "domain": domains[0] if len(domains) == 1 else "mixed",
                "fault_family": families[0] if len(families) == 1 else "mixed",
            }
        )
    return pd.DataFrame(rows)


def _role_counts(frame: pd.DataFrame, roles: dict[str, list[str]]) -> dict[str, Any]:
    indexed = frame.set_index("source_id")
    result: dict[str, Any] = {}
    for role in ROLES:
        sources = roles[role]
        part = indexed.loc[sources] if sources else indexed.iloc[0:0]
        result[role] = {
            "source_count": len(sources),
            "group_count": int(part["group_id"].nunique()),
            "normal_source_count": int(part["is_normal"].astype(bool).sum()),
            "anomaly_source_count": int((~part["is_normal"].astype(bool)).sum()),
            "domain_counts": dict(sorted(Counter(part["domain"].astype(str)).items())),
            "fault_family_counts": dict(
                sorted(Counter(part["fault_family"].astype(str)).items())
            ),
        }
    return result


def validate_contract(
    registry: pd.DataFrame,
    base: dict[str, Any],
    group_roles: dict[str, list[str]],
    source_roles: dict[str, list[str]],
) -> dict[str, Any]:
    frame = registry.loc[registry["dataset"].eq("rflymad")].copy()
    all_groups = [value for role in ROLES for value in group_roles[role]]
    all_sources = [value for role in ROLES for value in source_roles[role]]
    if len(all_groups) != len(set(all_groups)):
        raise V31ContractError("Group overlap detected")
    if len(all_sources) != len(set(all_sources)):
        raise V31ContractError("Source overlap detected")
    if set(all_groups) != set(frame["group_id"].astype(str)):
        raise V31ContractError("Group roles are not exhaustive")
    if set(all_sources) != set(frame["source_id"].astype(str)):
        raise V31ContractError("Source roles are not exhaustive")

    indexed = frame.set_index("source_id")
    for role in ("train", "val", "normal_test"):
        if not indexed.loc[source_roles[role], "is_normal"].astype(bool).all():
            raise V31ContractError(f"{role} is not normal-only")
    for role in ("anomaly_dev", "final_fault_test"):
        if indexed.loc[source_roles[role], "is_normal"].astype(bool).any():
            raise V31ContractError(f"Normal source leaked into {role}")

    base_roles = base["datasets"]["rflymad"]
    if source_roles["train"] != base_roles["source_ids"]["train"]:
        raise V31ContractError("v3 train role changed")
    if source_roles["val"] != base_roles["source_ids"]["val"]:
        raise V31ContractError("v3 validation role changed")
    base_test = set(base_roles["source_ids"]["test"])
    proposed_test = set(source_roles["normal_test"]) | set(source_roles["anomaly_dev"]) | set(
        source_roles["final_fault_test"]
    )
    if proposed_test != base_test:
        raise V31ContractError("v3 test membership changed rather than being partitioned")
    return {
        "contract_passed": True,
        "source_overlap_count": 0,
        "group_overlap_count": 0,
        "v3_train_preserved": True,
        "v3_validation_preserved": True,
        "v3_test_membership_preserved": True,
        "roles": _role_counts(frame, source_roles),
    }


def build_protocol(
    registry: pd.DataFrame, base: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = registry.loc[registry["dataset"].eq("rflymad")].copy()
    groups = _group_summary(frame)
    base_rfly = base["datasets"]["rflymad"]
    group_roles = {role: [] for role in ROLES}
    group_roles["train"] = list(base_rfly["group_ids"]["train"])
    group_roles["val"] = list(base_rfly["group_ids"]["val"])

    base_test_groups = set(base_rfly["group_ids"]["test"])
    test_groups = groups.loc[groups["group_id"].isin(base_test_groups)].copy()
    normal = test_groups.loc[test_groups["normal_count"].eq(test_groups["source_count"])]
    anomaly = test_groups.loc[test_groups["normal_count"].eq(0)]
    if len(normal) + len(anomaly) != len(test_groups):
        raise V31ContractError("Mixed RflyMAD v3 test group encountered")
    group_roles["normal_test"] = sorted(normal["group_id"].astype(str))
    anomaly_assignments = split_anomaly_groups(anomaly)
    for role in ("anomaly_dev", "final_fault_test"):
        group_roles[role] = sorted(
            group_id for group_id, value in anomaly_assignments.items() if value == role
        )

    source_roles: dict[str, list[str]] = {}
    for role in ROLES:
        source_roles[role] = sorted(
            frame.loc[frame["group_id"].isin(group_roles[role]), "source_id"].astype(str)
        )
    diagnostic = validate_contract(registry, base, group_roles, source_roles)
    final_seal = {
        "group_ids": group_roles["final_fault_test"],
        "source_ids": source_roles["final_fault_test"],
    }
    protocol = {
        "status": "frozen_group_safe_v31",
        "training_paradigm": "normal-only optimizer; semi-supervised model selection",
        "role_semantics": {
            "train": "optimizer and train-only preprocessing fit",
            "val": "normal-only calibration and false-alarm thresholding",
            "normal_test": "normal false-alarm evaluation; never threshold selection",
            "anomaly_dev": "model/event-rule comparison only; never optimizer or preprocessing fit",
            "final_fault_test": "sealed until one-shot final evaluation",
        },
        "group_ids": group_roles,
        "source_ids": source_roles,
        "final_fault_test_seal_sha256": canonical_sha(final_seal),
    }
    return protocol, diagnostic


def resolve_sources(
    manifest: dict[str, Any], scope: str, *, open_final: bool = False
) -> dict[str, tuple[str, ...]]:
    """Contract-aware loader used by future runners and tests."""

    roles = manifest["datasets"]["rflymad"]["source_ids"]
    if scope == "training":
        selected = ("train", "val")
    elif scope == "development":
        selected = ("train", "val", "normal_test", "anomaly_dev")
    elif scope == "final_evaluation":
        if not open_final:
            raise V31ContractError("final_fault_test requires explicit open_final=True")
        selected = ("normal_test", "final_fault_test")
    else:
        raise V31ContractError(f"Unknown access scope: {scope}")
    return {role: tuple(map(str, roles[role])) for role in selected}


def _write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=BASE_MANIFEST)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_path = args.base_manifest.resolve()
    registry_path = args.registry.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    registry = pd.read_parquet(registry_path)
    if base.get("candidate_namespace") != "four_dataset_probabilistic_v3":
        raise V31ContractError("Unexpected base namespace")
    if base["inputs"]["group_registry_sha256"] != sha256(registry_path):
        raise V31ContractError("Registry hash differs from frozen v3 input")
    protocol, diagnostic = build_protocol(registry, base)
    inputs = {
        "base_v3_manifest_path": base_path.relative_to(ROOT).as_posix(),
        "base_v3_manifest_sha256": sha256(base_path),
        "group_registry_path": registry_path.relative_to(ROOT).as_posix(),
        "group_registry_sha256": sha256(registry_path),
    }
    manifest = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "created_date": "2026-07-28",
        "seed": SEED,
        "inputs": inputs,
        "global_rules": {
            "v3_manifest_modified": False,
            "optimizer_anomaly_access": False,
            "preprocessing_fit_roles": ["train"],
            "threshold_fit_roles": ["val"],
            "development_default_excludes_final_fault_test": True,
            "fault_test_open_count_allowed": 1,
            "row_or_window_split_allowed": False,
            "result_dependent_repartition_allowed": False,
        },
        "datasets": {
            "rflymad": protocol,
            "uav_sead": {
                "status": "hold_session_key_audit_required",
                "strict_parent_v3_preserved": True,
                "training_authorized": False,
            },
            "alfa": {
                "status": "four_fold_group_safe_cv_authorized",
                "protocol_source": "base v3 manifest",
            },
            "uav_attack": {
                "status": "external_stress_or_exploratory_only",
                "main_training_authorized": False,
            },
        },
    }
    report = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "artifact_role": "pre-training contract verification",
        "created_date": "2026-07-28",
        "training_started": False,
        "inputs": inputs,
        "all_contracts_passed": True,
        "rflymad": diagnostic,
    }
    _write(args.output_manifest.resolve(), manifest, args.overwrite)
    _write(args.output_report.resolve(), report, args.overwrite)
    print(json.dumps(report["rflymad"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
