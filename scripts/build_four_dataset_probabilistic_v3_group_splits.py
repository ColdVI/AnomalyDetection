"""Freeze group-disjoint v3 evaluation roles for the feasible UAV datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet"
REGISTRY_REPORT = (
    ROOT / "artifacts/four_dataset_probabilistic_v3/group_registry_v1_report.json"
)
OUTPUT_MANIFEST = ROOT / "configs/four_dataset_probabilistic_v3_split_manifest.json"
OUTPUT_REPORT = (
    ROOT / "artifacts/four_dataset_probabilistic_v3/group_split_v1_report.json"
)
SEED = 20260728
ROLE_ORDER = ("train", "val", "test", "stress_test")


class GroupSplitContractError(RuntimeError):
    """Raised when a proposed v3 role assignment breaks group isolation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(dataset: str, purpose: str, value: str) -> str:
    return hashlib.sha256(
        f"{SEED}:{dataset}:{purpose}:{value}".encode("utf-8")
    ).hexdigest()


def group_table(dataset_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a registry partition to one row per group."""

    required = {"dataset", "group_id", "source_id", "flight_label", "is_normal"}
    missing = required - set(dataset_frame.columns)
    if missing:
        raise GroupSplitContractError(f"Registry columns missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for group_id, part in dataset_frame.groupby("group_id", sort=True):
        normal_count = int(part["is_normal"].sum())
        source_count = int(len(part))
        if normal_count == source_count:
            composition = "normal_only"
        elif normal_count == 0:
            composition = "anomaly_only"
        else:
            composition = "mixed"
        domains = sorted(
            {
                str(value)
                for value in part.get("domain", pd.Series(dtype=object)).dropna()
                if str(value)
            }
        )
        rows.append(
            {
                "group_id": str(group_id),
                "source_count": source_count,
                "normal_source_count": normal_count,
                "anomaly_source_count": source_count - normal_count,
                "composition": composition,
                "label_signature": "|".join(
                    sorted(set(part["flight_label"].astype(str)))
                ),
                "domain": domains[0] if len(domains) == 1 else None,
                "domain_count": len(domains),
            }
        )
    return pd.DataFrame(rows)


def weighted_normal_group_roles(
    groups: pd.DataFrame,
    dataset: str,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Assign whole normal groups while approximating source-count ratios."""

    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    if set(ratios) != {"train", "val", "test"} or abs(sum(ratios.values()) - 1) > 1e-9:
        raise GroupSplitContractError("Normal role ratios must be train/val/test and sum to one")
    if not groups["composition"].eq("normal_only").all():
        raise GroupSplitContractError("Weighted normal assignment received a non-normal group")
    if len(groups) < 3:
        raise GroupSplitContractError("At least three pure-normal groups are required")
    total = float(groups["source_count"].sum())
    targets = {role: total * ratio for role, ratio in ratios.items()}
    counts = {role: 0 for role in ratios}
    assignments: dict[str, str] = {}
    ordered = groups.assign(
        tie=groups["group_id"].map(lambda value: _key(dataset, "normal-role", value))
    ).sort_values(["source_count", "tie"], ascending=[False, True], kind="stable")
    for row in ordered.itertuples(index=False):
        deficits = {
            role: (targets[role] - counts[role]) / targets[role] for role in ratios
        }
        role = max(ratios, key=lambda candidate: (deficits[candidate], -ROLE_ORDER.index(candidate)))
        assignments[str(row.group_id)] = role
        counts[role] += int(row.source_count)
    if set(assignments.values()) != set(ratios):
        raise GroupSplitContractError("Weighted assignment left a normal role empty")
    return assignments


def rfly_domain_normal_roles(groups: pd.DataFrame) -> dict[str, str]:
    """Keep at least one normal group per domain in validation and test."""

    assignments: dict[str, str] = {}
    normal = groups.loc[groups["composition"].eq("normal_only")]
    if normal["domain"].isna().any() or normal["domain_count"].ne(1).any():
        raise GroupSplitContractError("RflyMAD normal groups must have one domain")
    for domain, part in normal.groupby("domain", sort=True):
        ordered = sorted(
            part["group_id"].astype(str),
            key=lambda value: _key("rflymad", f"domain:{domain}", value),
        )
        if len(ordered) < 3:
            raise GroupSplitContractError(
                f"RflyMAD domain {domain} has fewer than three normal groups"
            )
        assignments[ordered[0]] = "val"
        assignments[ordered[1]] = "test"
        for group_id in ordered[2:]:
            assignments[group_id] = "train"
    return assignments


def alfa_outer_fold_assignments(groups: pd.DataFrame, folds: int = 4) -> dict[str, int]:
    """Assign ALFA groups to deterministic outer folds without splitting sessions."""

    assignments: dict[str, int] = {}
    normal = groups.loc[groups["composition"].eq("normal_only")]
    if len(normal) < folds * 2:
        raise GroupSplitContractError("ALFA requires at least two pure-normal groups per fold")
    ordered_normal = sorted(
        normal["group_id"].astype(str),
        key=lambda value: _key("alfa", "normal-outer-fold", value),
    )
    for index, group_id in enumerate(ordered_normal):
        assignments[group_id] = index % folds

    other = groups.loc[groups["composition"].ne("normal_only")]
    for signature, part in other.groupby("label_signature", sort=True):
        ordered = sorted(
            part["group_id"].astype(str),
            key=lambda value: _key("alfa", f"truth-fold:{signature}", value),
        )
        offset = int(_key("alfa", "signature-offset", str(signature))[:8], 16) % folds
        for index, group_id in enumerate(ordered):
            assignments[group_id] = (offset + index) % folds
    if set(assignments) != set(groups["group_id"].astype(str)):
        raise GroupSplitContractError("ALFA outer-fold coverage is incomplete")
    return assignments


def _materialise_roles(
    frame: pd.DataFrame, group_roles: dict[str, str]
) -> dict[str, dict[str, list[str]]]:
    missing = set(frame["group_id"].astype(str)) - set(group_roles)
    if missing:
        raise GroupSplitContractError(f"Group role coverage missing: {sorted(missing)[:5]}")
    role_groups = {role: [] for role in ROLE_ORDER}
    role_sources = {role: [] for role in ROLE_ORDER}
    for role in ROLE_ORDER:
        groups = sorted(group_id for group_id, assigned in group_roles.items() if assigned == role)
        sources = sorted(
            frame.loc[frame["group_id"].isin(groups), "source_id"].astype(str)
        )
        role_groups[role] = groups
        role_sources[role] = sources
    return {"group_ids": role_groups, "source_ids": role_sources}


def validate_protocol(
    frame: pd.DataFrame,
    materialised: dict[str, dict[str, list[str]]],
    require_test_classes: bool = True,
) -> dict[str, Any]:
    """Validate exhaustive, group-disjoint roles and return count diagnostics."""

    group_roles = materialised["group_ids"]
    source_roles = materialised["source_ids"]
    all_groups: list[str] = []
    all_sources: list[str] = []
    counts: dict[str, Any] = {}
    indexed = frame.set_index("source_id")
    for role in ROLE_ORDER:
        groups = group_roles[role]
        sources = source_roles[role]
        all_groups.extend(groups)
        all_sources.extend(sources)
        labels = indexed.loc[sources, "flight_label"].astype(str) if sources else pd.Series(dtype=str)
        is_normal = indexed.loc[sources, "is_normal"].astype(bool) if sources else pd.Series(dtype=bool)
        counts[role] = {
            "group_count": len(groups),
            "source_count": len(sources),
            "normal_source_count": int(is_normal.sum()),
            "anomaly_source_count": int((~is_normal).sum()),
            "label_counts": dict(sorted(Counter(labels).items())),
        }
    expected_groups = set(frame["group_id"].astype(str))
    expected_sources = set(frame["source_id"].astype(str))
    if len(all_groups) != len(set(all_groups)) or set(all_groups) != expected_groups:
        raise GroupSplitContractError("Protocol group roles are not disjoint/exhaustive")
    if len(all_sources) != len(set(all_sources)) or set(all_sources) != expected_sources:
        raise GroupSplitContractError("Protocol source roles are not disjoint/exhaustive")
    for role in ("train", "val"):
        if counts[role]["source_count"] == 0:
            raise GroupSplitContractError(f"Protocol {role} role is empty")
        if counts[role]["anomaly_source_count"]:
            raise GroupSplitContractError(f"Anomaly leaked into {role}")
    if require_test_classes:
        if not counts["test"]["normal_source_count"]:
            raise GroupSplitContractError("Test has no normal source")
        if not counts["test"]["anomaly_source_count"]:
            raise GroupSplitContractError("Test has no anomaly source")
    return {
        "contract_passed": True,
        "group_overlap_count": 0,
        "source_overlap_count": 0,
        "roles": counts,
    }


def _build_alfa(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = group_table(frame)
    outer = alfa_outer_fold_assignments(groups)
    folds: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for fold in range(4):
        roles: dict[str, str] = {}
        for row in groups.itertuples(index=False):
            group_id = str(row.group_id)
            group_fold = outer[group_id]
            if group_fold == fold:
                roles[group_id] = "test"
            elif row.composition == "normal_only" and group_fold == (fold + 1) % 4:
                roles[group_id] = "val"
            elif row.composition == "normal_only":
                roles[group_id] = "train"
            else:
                roles[group_id] = "stress_test"
        materialised = _materialise_roles(frame, roles)
        diagnostic = validate_protocol(frame, materialised)
        folds.append(
            {
                "fold": fold,
                "validation_fold_rule": "next outer fold modulo 4 among pure-normal groups",
                **materialised,
            }
        )
        reports.append({"fold": fold, **diagnostic})
    payload = {
        "status": "grouped_cv_smoke_only",
        "primary_protocol": "grouped_cv_4",
        "operational_claim_eligible": False,
        "ineligibility_reason": (
            "Only eight pure-normal session groups; each fold has two validation "
            "and two normal-test sources."
        ),
        "folds": folds,
    }
    report = {"status": payload["status"], "folds": reports}
    return payload, report


def _build_fixed(
    frame: pd.DataFrame, dataset: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = group_table(frame)
    normal = groups.loc[groups["composition"].eq("normal_only")]
    if dataset == "rflymad":
        roles = rfly_domain_normal_roles(groups)
        assignment_rule = (
            "Within each of SIL/HIL/Real: one hash-selected pure-normal group to "
            "validation, one to normal test, remaining pure-normal groups to train."
        )
    else:
        roles = weighted_normal_group_roles(normal, dataset)
        assignment_rule = (
            "Pure-normal parent groups assigned by deterministic weighted 70/15/15 "
            "source-count balancing."
        )
    for row in groups.loc[groups["composition"].ne("normal_only")].itertuples(index=False):
        roles[str(row.group_id)] = "test"
    materialised = _materialise_roles(frame, roles)
    diagnostic = validate_protocol(frame, materialised)
    payload = {
        "status": "group_safe_fixed_split",
        "primary_protocol": "group_safe_in_domain_v1",
        "assignment_rule": assignment_rule,
        "all_mixed_and_anomaly_groups_role": "test",
        "group_ids": materialised["group_ids"],
        "source_ids": materialised["source_ids"],
    }
    if dataset == "rflymad":
        payload["transfer_protocol_status"] = (
            "separate SIL/HIL/Real transfer protocols remain required and are not "
            "reinterpreted from the in-domain split"
        )
    return payload, diagnostic


def build_payloads(registry: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for dataset in ("alfa", "rflymad", "uav_sead"):
        frame = registry.loc[registry["dataset"].eq(dataset)].copy()
        if dataset == "alfa":
            datasets[dataset], reports[dataset] = _build_alfa(frame)
        else:
            datasets[dataset], reports[dataset] = _build_fixed(frame, dataset)
    attack = registry.loc[registry["dataset"].eq("uav_attack")]
    attack_groups = group_table(attack)
    pure_normal = attack_groups.loc[attack_groups["composition"].eq("normal_only")]
    if len(pure_normal) != 1:
        raise GroupSplitContractError(
            "Expected the preregistered UAV Attack one-pure-normal-group NO-GO"
        )
    datasets["uav_attack"] = {
        "status": "no_go_group_safe_split_not_feasible",
        "primary_protocol": None,
        "reason": (
            "Only one pure-normal mode/platform/campaign group exists; disjoint "
            "normal train, validation and test roles cannot all be populated."
        ),
        "pure_normal_group_ids": sorted(pure_normal["group_id"].astype(str)),
        "source_ids_preserved_unassigned": sorted(attack["source_id"].astype(str)),
        "new_normal_campaigns_required": True,
        "physics_v3_training_authorized": False,
    }
    reports["uav_attack"] = {
        "status": datasets["uav_attack"]["status"],
        "contract_passed": True,
        "training_not_started": True,
        "source_count_preserved": int(len(attack)),
        "pure_normal_group_count": int(len(pure_normal)),
    }
    return datasets, reports


def _write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--registry-report", type=Path, default=REGISTRY_REPORT)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    registry_report_path = args.registry_report.resolve()
    registry = pd.read_parquet(registry_path)
    registry_report = json.loads(registry_report_path.read_text(encoding="utf-8"))
    if len(registry) != registry_report["totals"]["source_count"]:
        raise GroupSplitContractError("Registry Parquet/report source count mismatch")
    if registry["source_id"].duplicated().any():
        raise GroupSplitContractError("Registry source_id is not unique")
    datasets, reports = build_payloads(registry)
    inputs = {
        "group_registry_path": registry_path.relative_to(ROOT).as_posix(),
        "group_registry_sha256": _sha256(registry_path),
        "group_registry_report_path": registry_report_path.relative_to(ROOT).as_posix(),
        "group_registry_report_sha256": _sha256(registry_report_path),
    }
    manifest = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v3",
        "seed": SEED,
        "created_date": "2026-07-28",
        "split_unit": "dataset-specific group; never row, window or source-random",
        "inputs": inputs,
        "normal_role_target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "global_rules": {
            "train_and_validation_normal_only": True,
            "group_overlap_allowed": False,
            "mixed_group_may_be_split_by_label": False,
            "test_metric_may_select_split": False,
            "v2_artifacts_modified": False,
        },
        "datasets": datasets,
    }
    report = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v3",
        "artifact_role": "group-safe split contract verification",
        "created_date": "2026-07-28",
        "inputs": inputs,
        "all_contracts_passed": True,
        "training_started": False,
        "datasets": reports,
    }
    _write(args.output_manifest.resolve(), manifest, args.overwrite)
    _write(args.output_report.resolve(), report, args.overwrite)
    print(
        json.dumps(
            {
                "manifest": args.output_manifest.resolve().relative_to(ROOT).as_posix(),
                "report": args.output_report.resolve().relative_to(ROOT).as_posix(),
                "dataset_status": {
                    dataset: payload["status"] for dataset, payload in datasets.items()
                },
                "role_counts": {
                    dataset: (
                        [fold["roles"] for fold in payload["folds"]]
                        if dataset == "alfa"
                        else (
                            payload["roles"]
                            if "roles" in payload
                            else {
                                "status": payload["status"],
                                "training_not_started": payload.get(
                                    "training_not_started", False
                                ),
                            }
                        )
                    )
                    for dataset, payload in reports.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
