"""RflyMAD real-flight discovery, truth validation, and frozen role assignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyulog import ULog

GPS_FAULT_ID = 123456
MAG_FAULT_ID = 123455
IDLE_SENTINEL = 1500
FAULT_MODE_NAMES = {3: "noise", 4: "scale_factor"}
FLIGHT_MODES = ("acce", "circling", "hover", "velocity", "waypoint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flight_mode(path: Path) -> str:
    for part in path.parts:
        token = part.lower()
        for mode in FLIGHT_MODES:
            if token == mode or token.startswith(mode + "-"):
                return mode
    return "unknown"


def _fault_truth(path: Path) -> dict[str, Any]:
    ulog = ULog(str(path), ["rfly_ctrl_lxl"])
    data = ulog.get_dataset("rfly_ctrl_lxl").data
    timestamps = np.asarray(data["timestamp"], dtype=float) / 1e6
    ids = np.asarray(data["id"])
    modes = np.asarray(data["mode"])
    active = (ids != IDLE_SENTINEL) | (modes != IDLE_SENTINEL)
    if not active.any():
        return {"fault_id": None, "fault_mode": None, "fault_onset_s": None, "fault_end_s": None}
    index = np.flatnonzero(active)
    return {
        "fault_id": int(ids[index[0]]),
        "fault_mode": int(modes[index[0]]),
        "fault_onset_s": float(timestamps[index[0]]),
        "fault_end_s": float(timestamps[index[-1]]),
    }


def discover_cases(bronze_root: str | Path) -> list[dict[str, Any]]:
    root = Path(bronze_root)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "Real-No_Fault").rglob("*.ulg")):
        rows.append(
            {
                "flight_id": path.relative_to(root).as_posix(),
                "path": path.as_posix(),
                "domain": "real",
                "class": "normal",
                "flight_mode": _flight_mode(path),
                "fault_id": None,
                "fault_mode": None,
                "fault_mode_name": None,
                "fault_onset_s": None,
                "fault_end_s": None,
            }
        )
    for path in sorted((root / "Real-Sensors").rglob("*.ulg")):
        if "-gps" not in path.as_posix().lower():
            continue
        truth = _fault_truth(path)
        fault_id = truth["fault_id"]
        case_class = (
            "gps_fault"
            if fault_id == GPS_FAULT_ID
            else "quarantine_magnetometer"
            if fault_id == MAG_FAULT_ID
            else "quarantine_other"
        )
        rows.append(
            {
                "flight_id": path.relative_to(root).as_posix(),
                "path": path.as_posix(),
                "domain": "real",
                "class": case_class,
                "flight_mode": _flight_mode(path),
                **truth,
                "fault_mode_name": FAULT_MODE_NAMES.get(truth["fault_mode"], "unknown"),
            }
        )
    return rows


def _assign_balanced(rows: list[dict[str, Any]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    normal_by_mode = {
        mode: sorted(
            [row for row in rows if row["class"] == "normal" and row["flight_mode"] == mode],
            key=lambda row: row["flight_id"],
        )
        for mode in FLIGHT_MODES
    }
    for mode, group in normal_by_mode.items():
        if len(group) < 10:
            raise ValueError(f"{mode}: at least 10 normal flights required, found {len(group)}")
        role_counts = (("fit", 4), ("calibration", 2), ("rehearsal", 1), ("holdout", 2))
        used = 0
        for role, count in role_counts:
            for row in group[used : used + count]:
                assignments[row["flight_id"]] = role
            used += count
        for row in group[used:]:
            assignments[row["flight_id"]] = "development"

    gps_rows = [row for row in rows if row["class"] == "gps_fault"]
    for mode in FLIGHT_MODES:
        for fault_mode in (3, 4):
            group = sorted(
                [
                    row
                    for row in gps_rows
                    if row["flight_mode"] == mode and row["fault_mode"] == fault_mode
                ],
                key=lambda row: row["flight_id"],
            )
            if len(group) < 2:
                raise ValueError(
                    f"{mode}/fault_mode={fault_mode}: at least two GPS faults required"
                )
            assignments[group[0]["flight_id"]] = "rehearsal"
            assignments[group[1]["flight_id"]] = "holdout"
            for row in group[2:]:
                assignments[row["flight_id"]] = "development"
    for row in rows:
        if row["class"].startswith("quarantine"):
            assignments[row["flight_id"]] = "quarantine"
    return assignments


def build_role_manifest(bronze_root: str | Path, config_sha256: str) -> dict[str, Any]:
    rows = discover_cases(bronze_root)
    assignments = _assign_balanced(rows)
    public_rows = []
    holdout_ids = []
    for row in rows:
        role = assignments[row["flight_id"]]
        record = {
            **row,
            "role": role,
            "file_sha256": sha256_file(Path(row["path"])),
        }
        if role == "holdout":
            holdout_ids.append(record["flight_id"])
            public_rows.append(
                {
                    "role": "holdout",
                    "class": record["class"],
                    "flight_mode": record["flight_mode"],
                    "fault_mode": record["fault_mode"],
                    "sealed_id_sha256": hashlib.sha256(
                        record["flight_id"].encode("utf-8")
                    ).hexdigest(),
                    "file_sha256": record["file_sha256"],
                }
            )
        else:
            public_rows.append(record)
    role_counts: dict[str, int] = {}
    for role in assignments.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    sealed_digest = hashlib.sha256(
        json.dumps(sorted(holdout_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "candidate_namespace": "uav_gnss_integrity_v1",
        "config_sha256": config_sha256,
        "role_counts": role_counts,
        "holdout_unsealed": False,
        "sealed_holdout_id_digest": sealed_digest,
        "cases": public_rows,
    }


def cases_for_role(
    bronze_root: str | Path,
    role: str,
    *,
    allow_holdout: bool = False,
) -> list[dict[str, Any]]:
    if role == "holdout" and not allow_holdout:
        raise PermissionError("holdout role requires an explicit unseal record")
    rows = discover_cases(bronze_root)
    assignments = _assign_balanced(rows)
    return [{**row, "role": assignments[row["flight_id"]]} for row in rows if assignments[row["flight_id"]] == role]

