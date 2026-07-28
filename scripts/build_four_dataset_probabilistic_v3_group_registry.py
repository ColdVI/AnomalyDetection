"""Build and audit the dataset-specific group registry required by UAV v3.

This command does not assign new train/validation/test roles.  It records the
best currently available grouping key for every source in the frozen v2 split
and measures how often a v2 source-level split crossed those groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from zipfile import ZipFile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_SPLIT = ROOT / "configs/four_dataset_probabilistic_gpu_v2_split_manifest.json"
V2_CONFIG = ROOT / "configs/four_dataset_probabilistic_gpu_v2.json"
RFLY_MANIFEST = ROOT / "artifacts/rfly_full/v2/dataset_manifest.parquet"
UAV_ATTACK_ARCHIVE = ROOT / "data/objectstore/bronze/uav_attack/UAVAttackData.zip"
UAV_SEAD_LABELS = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
OUTPUT_DIR = ROOT / "artifacts/four_dataset_probabilistic_v3"
REGISTRY_NAME = "group_registry_v1.parquet"
REPORT_NAME = "group_registry_v1_report.json"
DATASET_ORDER = ("alfa", "rflymad", "uav_sead", "uav_attack")
ROLE_ORDER = ("train", "val", "test", "stress_test")

ALFA_SESSION = re.compile(
    r"^(carbonZ_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_|$)"
)
ATTACK_ACE_OR_LOG = re.compile(
    r"^(?P<prefix>ace|log)(?:[-_].*?)?_(?P<year>\d{4})-"
    r"(?P<month>\d{1,2})-(?P<day>\d{1,2})(?:-|$)"
)
ATTACK_NUMERIC = re.compile(
    r"^\d+-(?P<year>\d{4})-(?P<month>\d{1,2})-"
    r"(?P<day>\d{1,2})(?:-|$)"
)


class GroupRegistryContractError(RuntimeError):
    """Raised when a source cannot be mapped without violating the contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alfa_group(source_id: str) -> tuple[str, str, str]:
    """Return group key, evidence basis and registry status for an ALFA source."""

    match = ALFA_SESSION.match(str(source_id))
    if match is None:
        raise GroupRegistryContractError(
            f"ALFA source does not match frozen session grammar: {source_id}"
        )
    return (
        f"alfa:{match.group(1)}",
        "source_id carbonZ timestamp/session family",
        "deterministic_source_grammar",
    )


def _normalised_date(match: re.Match[str]) -> str:
    return (
        f"{int(match.group('year')):04d}-"
        f"{int(match.group('month')):02d}-"
        f"{int(match.group('day')):02d}"
    )


def _uav_attack_date(source_id: str) -> str:
    """Extract and normalise the campaign date encoded in a UAV Attack source."""

    value = str(source_id)
    match = ATTACK_ACE_OR_LOG.match(value)
    if match is not None:
        pass
    else:
        match = ATTACK_NUMERIC.match(value)
    if match is None:
        raise GroupRegistryContractError(
            f"UAV Attack source does not match campaign grammar: {source_id}"
        )
    return _normalised_date(match)


def uav_attack_group(
    source_id: str, campaign_mode: str, platform: str
) -> tuple[str, str, str]:
    """Return an archive-backed mode/platform/date campaign group."""

    date = _uav_attack_date(source_id)
    if not campaign_mode or not platform:
        raise GroupRegistryContractError(
            f"UAV Attack archive metadata is incomplete for {source_id}"
        )
    return (
        f"uav_attack:{campaign_mode}:{platform}:{date}",
        "archive directory mode/platform plus source_id campaign date",
        "authoritative_archive_path_metadata",
    )


def uav_sead_group(source_id: str) -> tuple[str, str, str]:
    """Return a conservative parent-path proxy for UAV-SEAD mission grouping."""

    value = str(source_id).replace("\\", "/").strip("/")
    if not value:
        raise GroupRegistryContractError("UAV-SEAD source_id is empty")
    path = PurePosixPath(value)
    if len(path.parts) == 1:
        # Root-level names do not carry evidence that they belong together.
        key = f"root_source:{value}"
        basis = "root-level source kept as its own provisional group"
    else:
        key = path.parent.as_posix()
        basis = "source_id parent path used as conservative mission/session proxy"
    return (
        f"uav_sead:{key}",
        basis,
        "frozen_conservative_proxy_active_metadata_exhausted",
    )


def _source_roles(dataset_payload: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {str(key): str(value) for key, value in dataset_payload["flight_labels"].items()}
    normal_label = str(dataset_payload["normal_label"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role in ROLE_ORDER:
        if role not in dataset_payload["roles"]:
            raise GroupRegistryContractError(f"Missing v2 role: {role}")
        for raw_source_id in dataset_payload["roles"][role]:
            source_id = str(raw_source_id)
            if source_id in seen:
                raise GroupRegistryContractError(
                    f"Source appears in more than one v2 role: {source_id}"
                )
            if source_id not in labels:
                raise GroupRegistryContractError(f"Missing flight label: {source_id}")
            seen.add(source_id)
            rows.append(
                {
                    "source_id": source_id,
                    "v2_role": role,
                    "flight_label": labels[source_id],
                    "normal_label": normal_label,
                    "is_normal": labels[source_id] == normal_label,
                }
            )
    if seen != set(labels):
        missing = sorted(set(labels) - seen)
        extra = sorted(seen - set(labels))
        raise GroupRegistryContractError(
            f"Role/label coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return rows


def _load_rfly_metadata(path: Path, source_ids: set[str]) -> pd.DataFrame:
    columns = [
        "canonical_case_id",
        "split_group_id",
        "domain",
        "platform",
        "flight_status",
        "fault_family",
        "fault_subtype",
        "representation",
    ]
    metadata = pd.read_parquet(path, columns=columns).copy()
    metadata["canonical_case_id"] = metadata["canonical_case_id"].astype(str)
    selected = metadata.loc[metadata["canonical_case_id"].isin(source_ids)].copy()
    if selected["canonical_case_id"].duplicated().any():
        duplicate = selected.loc[
            selected["canonical_case_id"].duplicated(False), "canonical_case_id"
        ].iloc[0]
        raise GroupRegistryContractError(f"Duplicate RflyMAD metadata: {duplicate}")
    actual = set(selected["canonical_case_id"])
    if actual != source_ids:
        raise GroupRegistryContractError(
            f"RflyMAD metadata coverage mismatch; missing={sorted(source_ids - actual)[:5]}"
        )
    if selected["split_group_id"].isna().any():
        raise GroupRegistryContractError("RflyMAD selected metadata has null split_group_id")
    selected = selected.set_index("canonical_case_id")
    if not selected.index.is_unique:
        raise GroupRegistryContractError("RflyMAD selected metadata index is not unique")
    return selected


def _load_uav_attack_metadata(path: Path, source_ids: set[str]) -> pd.DataFrame:
    """Map every selected source to its unique raw ZIP directory metadata."""

    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    rows: list[dict[str, str]] = []
    for source_id in sorted(source_ids):
        directories: set[PurePosixPath] = set()
        for name in names:
            member = PurePosixPath(name)
            basename = member.name
            if basename.startswith(f"{source_id}_") or basename.startswith(
                f"{source_id}."
            ):
                directories.add(member.parent)
        if len(directories) != 1:
            raise GroupRegistryContractError(
                f"UAV Attack source must map to one archive directory: "
                f"{source_id} -> {sorted(map(str, directories))}"
            )
        directory = next(iter(directories))
        parts = directory.parts
        if parts[0] == "Live GPS Spoofing and Jamming" and len(parts) == 2:
            campaign_mode = "live"
            platform = "live_platform_unspecified"
            archive_class = parts[1]
        elif parts[0] == "Simulated - OTU Survey" and len(parts) == 3:
            campaign_mode = "simulated"
            platform = parts[1]
            archive_class = parts[2]
        else:
            raise GroupRegistryContractError(
                f"Unexpected UAV Attack archive directory: {directory.as_posix()}"
            )
        rows.append(
            {
                "source_id": source_id,
                "campaign_mode": campaign_mode,
                "platform": platform,
                "archive_class": archive_class,
                "archive_member_directory": directory.as_posix(),
            }
        )
    metadata = pd.DataFrame(rows).set_index("source_id")
    if set(metadata.index) != source_ids or not metadata.index.is_unique:
        raise GroupRegistryContractError("UAV Attack archive metadata coverage failed")
    return metadata


def _validate_gold_coverage(
    root: Path,
    config: dict[str, Any],
    dataset: str,
    expected_ids: set[str],
) -> dict[str, Any]:
    data_path = root / str(config["datasets"][dataset]["data_path"])
    available = set(
        pd.read_parquet(data_path, columns=["source_id"])["source_id"]
        .astype(str)
        .unique()
    )
    if available != expected_ids:
        raise GroupRegistryContractError(
            f"{dataset}: Gold/v2 source coverage mismatch; "
            f"missing={sorted(expected_ids - available)[:5]}, "
            f"unexpected={sorted(available - expected_ids)[:5]}"
        )
    return {
        "path": data_path.relative_to(root).as_posix(),
        "available_source_count": len(available),
        "coverage_verified": True,
    }


def _annotate_group_audit(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(["dataset", "group_id"], sort=False)
    result["group_source_count"] = grouped["source_id"].transform("size").astype(int)
    result["group_role_count"] = grouped["v2_role"].transform("nunique").astype(int)
    role_map = grouped["v2_role"].agg(
        lambda values: "|".join(role for role in ROLE_ORDER if role in set(values))
    )
    role_lookup = role_map.to_dict()
    result["group_v2_roles"] = [
        role_lookup[(dataset, group_id)]
        for dataset, group_id in zip(result["dataset"], result["group_id"])
    ]
    result["v2_cross_role_group"] = result["group_role_count"].gt(1)
    return result


def build_registry(
    split: dict[str, Any],
    rfly_metadata: pd.DataFrame,
    attack_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row-per-source registry from a loaded v2 split payload."""

    all_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        if dataset not in split["sources"]:
            raise GroupRegistryContractError(f"Dataset missing from v2 split: {dataset}")
        rows = _source_roles(split["sources"][dataset])
        for row in rows:
            source_id = row["source_id"]
            metadata: dict[str, Any] = {
                "domain": None,
                "platform": None,
                "flight_status": None,
                "fault_family": None,
                "fault_subtype": None,
                "representation": None,
                "campaign_mode": None,
                "archive_class": None,
                "archive_member_directory": None,
            }
            if dataset == "alfa":
                group_id, basis, status = alfa_group(source_id)
            elif dataset == "uav_attack":
                if source_id not in attack_metadata.index:
                    raise GroupRegistryContractError(
                        f"UAV Attack archive metadata missing source: {source_id}"
                    )
                attack_record = attack_metadata.loc[source_id]
                group_id, basis, status = uav_attack_group(
                    source_id,
                    str(attack_record["campaign_mode"]),
                    str(attack_record["platform"]),
                )
                metadata.update(
                    {
                        "domain": str(attack_record["campaign_mode"]),
                        "platform": str(attack_record["platform"]),
                        "representation": "raw archive ULog/CSV bundle",
                        "campaign_mode": str(attack_record["campaign_mode"]),
                        "archive_class": str(attack_record["archive_class"]),
                        "archive_member_directory": str(
                            attack_record["archive_member_directory"]
                        ),
                    }
                )
            elif dataset == "uav_sead":
                group_id, basis, status = uav_sead_group(source_id)
            else:
                if source_id not in rfly_metadata.index:
                    raise GroupRegistryContractError(
                        f"RflyMAD metadata missing source: {source_id}"
                    )
                record = rfly_metadata.loc[source_id]
                group_id = f"rflymad:{record['split_group_id']}"
                basis = "dataset_manifest.split_group_id"
                status = "authoritative_dataset_metadata"
                metadata = {
                    key: (
                        None
                        if key not in record.index or pd.isna(record[key])
                        else str(record[key])
                    )
                    for key in metadata
                }
                if str(record["fault_family"]) != row["flight_label"]:
                    raise GroupRegistryContractError(
                        f"RflyMAD label mismatch for {source_id}: "
                        f"{row['flight_label']} != {record['fault_family']}"
                    )
            all_rows.append(
                {
                    "dataset": dataset,
                    **row,
                    "group_id": group_id,
                    "group_basis": basis,
                    "group_status": status,
                    **metadata,
                }
            )
    registry = _annotate_group_audit(pd.DataFrame(all_rows))
    if registry["source_id"].duplicated().any():
        raise GroupRegistryContractError("Registry source_id is not globally unique")
    return registry.sort_values(
        ["dataset", "group_id", "v2_role", "source_id"], kind="stable"
    ).reset_index(drop=True)


def build_report(
    registry: pd.DataFrame,
    split_path: Path,
    rfly_path: Path,
    attack_path: Path,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        part = registry.loc[registry["dataset"].eq(dataset)]
        cross = part.loc[part["v2_cross_role_group"]]
        composition = (
            part.groupby("group_id", sort=True)
            .agg(source_count=("source_id", "size"), normal_sources=("is_normal", "sum"))
            .reset_index()
        )
        composition["composition"] = "mixed"
        composition.loc[composition["normal_sources"].eq(0), "composition"] = (
            "anomaly_only"
        )
        composition.loc[
            composition["normal_sources"].eq(composition["source_count"]),
            "composition",
        ] = "normal_only"
        composition_counts: dict[str, Any] = {}
        for kind in ("normal_only", "mixed", "anomaly_only"):
            selected = composition.loc[composition["composition"].eq(kind)]
            composition_counts[kind] = {
                "group_count": int(len(selected)),
                "source_count": int(selected["source_count"].sum()),
                "normal_source_count": int(selected["normal_sources"].sum()),
            }
        pure_normal_groups = composition_counts["normal_only"]["group_count"]
        pure_normal_sources = composition_counts["normal_only"]["source_count"]
        group_details = (
            cross.groupby("group_id", sort=True)
            .agg(
                source_count=("source_id", "size"),
                roles=("group_v2_roles", "first"),
                normal_sources=("is_normal", "sum"),
            )
            .reset_index()
        )
        datasets[dataset] = {
            "source_count": int(len(part)),
            "group_count": int(part["group_id"].nunique()),
            "cross_role_group_count": int(cross["group_id"].nunique()),
            "sources_in_cross_role_groups": int(len(cross)),
            "cross_role_source_fraction": float(len(cross) / len(part)),
            "group_status_counts": {
                str(key): int(value)
                for key, value in part["group_status"].value_counts().sort_index().items()
            },
            "v2_role_source_counts": {
                role: int(part["v2_role"].eq(role).sum()) for role in ROLE_ORDER
            },
            "group_composition": composition_counts,
            "group_safe_normal_role_feasibility": {
                "pure_normal_group_count": pure_normal_groups,
                "pure_normal_source_count": pure_normal_sources,
                "minimum_three_way_group_gate_passed": pure_normal_groups >= 3,
                "status": (
                    "not_feasible_only_one_pure_normal_group"
                    if dataset == "uav_attack"
                    else "feasible_subject_to_frozen_split_protocol"
                ),
            },
            "cross_role_groups": group_details.to_dict(orient="records"),
        }
    return {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v3",
        "artifact_role": "pre-split group registry and frozen-v2 overlap audit",
        "created_date": "2026-07-28",
        "does_not_assign_v3_roles": True,
        "inputs": {
            "v2_split_manifest": {
                "path": split_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(split_path),
            },
            "rfly_manifest": {
                "path": rfly_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(rfly_path),
            },
            "uav_attack_archive": {
                "path": attack_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(attack_path),
            },
            "gold_source_coverage": coverage,
        },
        "totals": {
            "source_count": int(len(registry)),
            "dataset_count": int(registry["dataset"].nunique()),
            "group_count_within_dataset": int(
                registry[["dataset", "group_id"]].drop_duplicates().shape[0]
            ),
        },
        "datasets": datasets,
        "interpretation_contract": {
            "cross_role_group": (
                "A grouping key present in more than one frozen v2 role; it is an "
                "audit finding, not proof that every member is a duplicate flight."
            ),
            "conservative_group": (
                "The UAV-SEAD parent-path fallback was frozen after the active Gold "
                "and Bronze labels metadata were shown to contain no mission/session "
                "field. It remains an explicit limitation and may be superseded only "
                "by separately versioned stronger metadata. UAV Attack mode/platform "
                "metadata comes from the raw archive path."
            ),
            "training_gate": (
                "No v3 training may start until chosen group keys have zero overlap "
                "between new train, validation and test roles."
            ),
        },
    }


def _write_outputs(
    output_dir: Path,
    registry: pd.DataFrame,
    report: dict[str, Any],
    overwrite: bool,
) -> tuple[Path, Path]:
    registry_path = output_dir / REGISTRY_NAME
    report_path = output_dir / REPORT_NAME
    existing = [path for path in (registry_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists; pass --overwrite only after reviewing the prior artifact: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(registry_path, index=False, engine="pyarrow")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=V2_SPLIT)
    parser.add_argument("--v2-config", type=Path, default=V2_CONFIG)
    parser.add_argument("--rfly-manifest", type=Path, default=RFLY_MANIFEST)
    parser.add_argument("--uav-attack-archive", type=Path, default=UAV_ATTACK_ARCHIVE)
    parser.add_argument("--uav-sead-labels", type=Path, default=UAV_SEAD_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_path = args.split_manifest.resolve()
    config_path = args.v2_config.resolve()
    rfly_path = args.rfly_manifest.resolve()
    attack_path = args.uav_attack_archive.resolve()
    sead_labels_path = args.uav_sead_labels.resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    coverage: dict[str, Any] = {}
    for dataset in ("alfa", "uav_sead", "uav_attack"):
        expected = set(split["sources"][dataset]["flight_labels"])
        coverage[dataset] = _validate_gold_coverage(ROOT, config, dataset, expected)

    sead_labels = json.loads(sead_labels_path.read_text(encoding="utf-8"))
    sead_expected = set(split["sources"]["uav_sead"]["flight_labels"])
    if not isinstance(sead_labels, dict) or not sead_expected.issubset(sead_labels):
        raise GroupRegistryContractError("UAV-SEAD labels.json coverage mismatch")
    sead_schema = sorted(
        {str(field) for record in sead_labels.values() for field in record}
    )
    coverage["uav_sead"].update(
        {
            "labels_path": sead_labels_path.relative_to(ROOT).as_posix(),
            "labels_sha256": _sha256(sead_labels_path),
            "label_record_count": len(sead_labels),
            "selected_label_coverage_verified": True,
            "label_schema_fields": sead_schema,
            "mission_or_session_field_present": any(
                field in {"mission", "mission_id", "session", "session_id"}
                for field in sead_schema
            ),
        }
    )

    rfly_ids = set(split["sources"]["rflymad"]["flight_labels"])
    rfly_metadata = _load_rfly_metadata(rfly_path, rfly_ids)
    coverage["rflymad"] = {
        "path": rfly_path.relative_to(ROOT).as_posix(),
        "available_source_count": int(len(rfly_metadata)),
        "coverage_verified": True,
    }

    attack_ids = set(split["sources"]["uav_attack"]["flight_labels"])
    attack_metadata = _load_uav_attack_metadata(attack_path, attack_ids)
    coverage["uav_attack"].update(
        {
            "archive_path": attack_path.relative_to(ROOT).as_posix(),
            "archive_source_count": int(len(attack_metadata)),
            "archive_coverage_verified": True,
        }
    )

    registry = build_registry(split, rfly_metadata, attack_metadata)
    report = build_report(registry, split_path, rfly_path, attack_path, coverage)
    registry_path, report_path = _write_outputs(
        args.output_dir.resolve(), registry, report, args.overwrite
    )
    print(
        json.dumps(
            {
                "registry": registry_path.relative_to(ROOT).as_posix(),
                "report": report_path.relative_to(ROOT).as_posix(),
                "totals": report["totals"],
                "datasets": {
                    dataset: {
                        key: value
                        for key, value in payload.items()
                        if key
                        in {
                            "source_count",
                            "group_count",
                            "cross_role_group_count",
                            "sources_in_cross_role_groups",
                        }
                    }
                    for dataset, payload in report["datasets"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
