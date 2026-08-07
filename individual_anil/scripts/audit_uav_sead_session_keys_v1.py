"""Extract read-only ULog header evidence for a refined UAV-SEAD session key.

This audit never assigns train/validation/test roles.  The conservative v3
parent-key split remains authoritative until a separately frozen registry v2 is
supported by the evidence emitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pyulog import ULog


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
REGISTRY = ROOT / "artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet"
BRONZE = ROOT / "data/objectstore/bronze/uav_sead"
OUTPUT_ROWS = ROOT / "artifacts/four_dataset_probabilistic_v31/uav_sead_session_metadata_v1.parquet"
OUTPUT_REPORT = ROOT / "artifacts/four_dataset_probabilistic_v31/uav_sead_session_key_audit_v1.json"


class SessionAuditError(RuntimeError):
    """Raised when source identity, files, or frozen registry inputs disagree."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _normalised_member(record: dict[str, Any]) -> str:
    value = str(record["object_name"]).replace("\\", "/")
    prefix = "uav_sead/"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    return value


def _metadata(path: Path) -> dict[str, Any]:
    ulog = ULog(str(path), message_name_filter_list=[], parse_header_only=True)
    info = dict(ulog.msg_info_dict)
    parameters = dict(ulog.initial_parameters)
    identity_parameters = {
        key: parameters.get(key)
        for key in (
            "MAV_SYS_ID",
            "MAV_COMP_ID",
            "SYS_AUTOSTART",
            "SYS_AUTOCONFIG",
            "SYS_MC_EST_GROUP",
            "SYS_VEHICLE_RESP",
        )
        if key in parameters
    }
    return {
        "ulog_start_timestamp_us": int(ulog.start_timestamp),
        "system_uuid": str(info.get("sys_uuid", "")),
        "system_name": str(info.get("sys_name", "")),
        "hardware": str(info.get("ver_hw", "")),
        "firmware_hash": str(info.get("ver_sw", "")),
        "os_hash": str(info.get("sys_os_ver", "")),
        "time_ref_utc": str(info.get("time_ref_utc", "")),
        "parameter_count": len(parameters),
        "parameter_hash": _canonical_sha(parameters),
        "identity_parameter_hash": _canonical_sha(identity_parameters),
        "identity_parameters_json": json.dumps(
            identity_parameters, sort_keys=True, separators=(",", ":"), default=str
        ),
    }


def _key_diagnostic(rows: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    keys = rows.loc[:, columns].fillna("").astype(str).agg("|".join, axis=1)
    table = rows.assign(candidate_key=keys).groupby("candidate_key", sort=False).agg(
        sources=("source_id", "size"),
        labels=("label", "nunique"),
        strict_parent_groups=("strict_parent_group_id", "nunique"),
    )
    return {
        "columns": columns,
        "key_count": int(len(table)),
        "singleton_key_count": int(table["sources"].eq(1).sum()),
        "cross_label_key_count": int(table["labels"].gt(1).sum()),
        "cross_strict_parent_key_count": int(table["strict_parent_groups"].gt(1).sum()),
        "largest_key_source_count": int(table["sources"].max()),
    }


def audit(labels_path: Path, registry_path: Path, bronze: Path, limit: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    registry = pd.read_parquet(registry_path)
    registry = registry.loc[registry["dataset"].eq("uav_sead")].copy()
    registry_index = registry.set_index("source_id")
    selected_ids = set(registry_index.index.astype(str))
    label_ids = set(map(str, labels))
    if not selected_ids <= label_ids:
        raise SessionAuditError("Registry sources are missing from labels.json")
    ordered = sorted(selected_ids)
    if limit:
        ordered = ordered[:limit]
    rows: list[dict[str, Any]] = []
    bytes_read_scope = 0
    for index, source_id in enumerate(ordered, start=1):
        record = labels[source_id]
        member = _normalised_member(record)
        path = bronze / Path(member)
        if not path.is_file():
            raise SessionAuditError(f"ULog missing: {member}")
        declared_bytes = record.get("size_bytes")
        expected_bytes = int(declared_bytes) if declared_bytes is not None else path.stat().st_size
        if declared_bytes is not None and path.stat().st_size != expected_bytes:
            raise SessionAuditError(f"ULog byte-size mismatch: {member}")
        evidence = _metadata(path)
        registry_row = registry_index.loc[source_id]
        rows.append(
            {
                "source_id": source_id,
                "label": str(record["label"]),
                "is_normal": bool(registry_row["is_normal"]),
                "object_name": member,
                "strict_parent_group_id": str(registry_row["group_id"]),
                "bytes": expected_bytes,
                "size_bytes_declared": declared_bytes is not None,
                **evidence,
            }
        )
        bytes_read_scope += expected_bytes
        if index == 1 or index % 100 == 0 or index == len(ordered):
            print(f"uav_sead_header_audit={index}/{len(ordered)}", flush=True)
    frame = pd.DataFrame(rows)
    diagnostics = {
        "uuid": _key_diagnostic(frame, ["system_uuid"]),
        "uuid_firmware": _key_diagnostic(frame, ["system_uuid", "firmware_hash"]),
        "uuid_firmware_parameters": _key_diagnostic(
            frame, ["system_uuid", "firmware_hash", "parameter_hash"]
        ),
        "uuid_firmware_identity_parameters": _key_diagnostic(
            frame, ["system_uuid", "firmware_hash", "identity_parameter_hash"]
        ),
        "parent_uuid_firmware_parameters": _key_diagnostic(
            frame,
            [
                "strict_parent_group_id",
                "system_uuid",
                "firmware_hash",
                "parameter_hash",
            ],
        ),
    }
    report = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "artifact_role": "UAV-SEAD session-key metadata audit; no split assignment",
        "created_date": "2026-07-28",
        "strict_parent_v3_modified": False,
        "training_authorized": False,
        "inputs": {
            "labels_path": labels_path.relative_to(ROOT).as_posix(),
            "labels_sha256": _sha256(labels_path),
            "registry_path": registry_path.relative_to(ROOT).as_posix(),
            "registry_sha256": _sha256(registry_path),
            "bronze_root": bronze.relative_to(ROOT).as_posix(),
        },
        "scope": {
            "selected_source_count": len(selected_ids),
            "audited_source_count": len(frame),
            "audit_complete": len(frame) == len(selected_ids),
            "raw_file_bytes_in_scope": bytes_read_scope,
            "header_only_parse": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "metadata_coverage": {
            column: int(frame[column].astype(str).ne("").sum())
            for column in (
                "system_uuid",
                "hardware",
                "firmware_hash",
                "os_hash",
                "parameter_hash",
            )
        },
        "label_counts": dict(sorted(Counter(frame["label"]).items())),
        "candidate_key_diagnostics": diagnostics,
        "observed_key_interpretation": {
            "system_uuid": "vehicle identity is too coarse and crosses labels/parents",
            "full_parameter_hash": "nearly source-unique and therefore too fine for session grouping",
            "ulog_start_timestamp": "boot-relative timestamp; not a wall-clock session key",
            "strict_parent": "retained pending mission/waypoint or independently justified temporal-session evidence",
        },
        "decision": (
            "strict_parent_retained_no_defensible_refined_key_yet"
            if len(frame) == len(selected_ids)
            else "sample_only_full_audit_required"
        ),
        "next_gate": (
            "A refined registry v2 requires documented semantic review of keys, "
            "timestamp overlap/mission evidence where available, then a new immutable split."
        ),
    }
    return frame, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--bronze", type=Path, default=BRONZE)
    parser.add_argument("--output-rows", type=Path, default=OUTPUT_ROWS)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows_path = args.output_rows.resolve()
    report_path = args.output_report.resolve()
    if (rows_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite deliberately")
    rows, report = audit(
        args.labels.resolve(), args.registry.resolve(), args.bronze.resolve(), args.limit
    )
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(rows_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
