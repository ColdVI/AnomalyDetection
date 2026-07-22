"""Natural-rate RflyMAD-Real ULog ingestion."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from residual_v1.ingest.common import (
    drop_non_monotonic_timestamps,
    fix_quaternion_sign_continuity,
    write_json,
)

DEFAULT_BRONZE_ROOT = Path("data/objectstore/bronze/rflymad")
DEFAULT_OUTPUT_ROOT = Path("artifacts/residual_v1/silver/rfly")
DEFAULT_EXCLUSIONS = Path("configs/residual_v1_rfly_exclusions.json")
REAL_SUBSETS = {"Real-NoFault", "Real-No_Fault", "Real-Motor", "Real-Sensors"}
RFLY_IDLE_SENTINEL = 1500

_TOPIC_FIELDS: dict[str, dict[str, str]] = {
    "vehicle_attitude": {
        "q[0]": "attitude_qw",
        "q[1]": "attitude_qx",
        "q[2]": "attitude_qy",
        "q[3]": "attitude_qz",
    },
    "vehicle_attitude_setpoint": {
        "roll_body": "roll_sp",
        "pitch_body": "pitch_sp",
        "yaw_body": "yaw_sp",
        "thrust_body[2]": "thrust_sp",
    },
    "vehicle_rates_setpoint": {
        "roll": "roll_rate_sp",
        "pitch": "pitch_rate_sp",
        "yaw": "yaw_rate_sp",
    },
    "vehicle_angular_velocity": {
        "xyz[0]": "roll_rate",
        "xyz[1]": "pitch_rate",
        "xyz[2]": "yaw_rate",
    },
    "sensor_combined": {
        "accelerometer_m_s2[0]": "accel_x",
        "accelerometer_m_s2[1]": "accel_y",
        "accelerometer_m_s2[2]": "accel_z",
    },
    "vehicle_local_position": {
        "x": "local_x",
        "y": "local_y",
        "z": "local_z",
        "vx": "local_vx",
        "vy": "local_vy",
        "vz": "local_vz",
        "ax": "local_ax",
        "ay": "local_ay",
        "az": "local_az",
    },
    "vehicle_local_position_setpoint": {
        "x": "position_sp_x",
        "y": "position_sp_y",
        "z": "position_sp_z",
        "vx": "velocity_sp_x",
        "vy": "velocity_sp_y",
        "vz": "velocity_sp_z",
    },
    "actuator_outputs": {
        "output[0]": "motor_pwm_0",
        "output[1]": "motor_pwm_1",
        "output[2]": "motor_pwm_2",
        "output[3]": "motor_pwm_3",
    },
    "battery_status": {"voltage_v": "battery_voltage"},
    "rfly_ctrl_lxl": {"id": "fault_id", "mode": "fault_mode"},
}


def _posix_parts(path: str | Path) -> tuple[str, ...]:
    return PurePosixPath(str(path).replace("\\", "/")).parts


def case_id_from_path(path: str | Path, root: str | Path) -> str:
    relative = Path(path).relative_to(root).as_posix()
    parts = _posix_parts(relative)
    for index, part in enumerate(parts):
        if part.startswith("log_"):
            return "/".join(parts[: index + 1])
    return "/".join(parts[:-1])


def infer_fault_class(case_id: str) -> str:
    parts = _posix_parts(case_id)
    subset = parts[0] if parts else ""
    if subset in {"Real-NoFault", "Real-No_Fault"}:
        return "normal"
    if subset == "Real-Motor":
        return "motor"
    if subset == "Real-Sensors":
        return "sensor"
    return "unknown"


def _session_from_case(case_id: str, metadata: Mapping[str, Any] | None = None) -> str:
    if metadata:
        explicit = metadata.get("test_session") or metadata.get("session")
        if explicit:
            return str(explicit)
    match = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", case_id)
    if match:
        year, month, day = (int(value) for value in match.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"
    return case_id.split("/")[0]


def load_exclusions(path: str | Path = DEFAULT_EXCLUSIONS) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {flight_id: str(payload["reason"]) for flight_id in payload["flight_ids"]}


def _select_datasets(ulog: Any) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    ordered = sorted(
        ulog.data_list,
        key=lambda dataset: (getattr(dataset, "name", ""), int(getattr(dataset, "multi_id", 0))),
    )
    for dataset in ordered:
        name = getattr(dataset, "name", "")
        if name not in _TOPIC_FIELDS or name in selected:
            continue
        selected[name] = dataset
    return selected


def read_ulog_topics(
    path: str | Path,
    *,
    ulog_factory: Callable[[str, list[str]], Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read only pre-registered ULog topics, retaining their natural clocks."""

    if ulog_factory is None:
        from pyulog import ULog

        ulog = ULog(str(path), list(_TOPIC_FIELDS))
    else:
        ulog = ulog_factory(str(path), list(_TOPIC_FIELDS))
    result: dict[str, pd.DataFrame] = {}
    for topic, dataset in _select_datasets(ulog).items():
        data = dataset.data
        if "timestamp" not in data:
            continue
        mapping = _TOPIC_FIELDS[topic]
        source_columns = [column for column in mapping if column in data]
        payload: dict[str, Any] = {"timestamp": data["timestamp"]}
        payload.update({mapping[column]: data[column] for column in source_columns})
        result[topic] = pd.DataFrame(payload)
    return result


def normalise_rfly_topics(
    topic_frames: Mapping[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, int], float]:
    cleaned: dict[str, pd.DataFrame] = {}
    dropped: dict[str, int] = {}
    for topic, frame in topic_frames.items():
        output, count = drop_non_monotonic_timestamps(frame, "timestamp")
        cleaned[topic] = output
        dropped[topic] = count
    timestamps = [
        pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
        for frame in cleaned.values()
        if not frame.empty
    ]
    if not timestamps:
        raise ValueError("ULog has no usable registered timestamp")
    t0_us = float(min(values.min() for values in timestamps))
    normalised: dict[str, pd.DataFrame] = {}
    for topic, frame in cleaned.items():
        output = frame.copy()
        output.insert(
            0,
            "t",
            (pd.to_numeric(output.pop("timestamp"), errors="coerce") - t0_us) / 1e6,
        )
        if topic == "vehicle_attitude":
            output = fix_quaternion_sign_continuity(
                output, ("attitude_qw", "attitude_qx", "attitude_qy", "attitude_qz")
            )
        normalised[topic] = output.reset_index(drop=True)
    return normalised, dropped, t0_us


def extract_rfly_events(
    normalised_topics: Mapping[str, pd.DataFrame], fault_class: str
) -> list[dict[str, float | str]]:
    if fault_class == "normal":
        return []
    control = normalised_topics.get("rfly_ctrl_lxl")
    if control is None or not {"fault_id", "fault_mode"}.issubset(control):
        return []
    active = (
        pd.to_numeric(control["fault_id"], errors="coerce") != RFLY_IDLE_SENTINEL
    ) | (
        pd.to_numeric(control["fault_mode"], errors="coerce") != RFLY_IDLE_SENTINEL
    )
    if not bool(active.any()):
        return []
    times = pd.to_numeric(control.loc[active, "t"], errors="coerce")
    return [
        {
            "fault_class": fault_class,
            "onset_s": float(times.min()),
            "end_s": float(times.max()),
        }
    ]


def ingest_rfly_flight(
    case_id: str,
    topic_frames: Mapping[str, pd.DataFrame],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    input_file: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict:
    destination = Path(output_root) / Path(case_id)
    if destination.exists():
        raise FileExistsError(destination)
    normalised, dropped, t0_us = normalise_rfly_topics(topic_frames)
    destination.mkdir(parents=True)
    for topic, frame in normalised.items():
        frame.to_parquet(destination / f"{topic}.parquet", index=False)
    fault_class = infer_fault_class(case_id)
    events = extract_rfly_events(normalised, fault_class)
    write_json(destination / "events.json", events)
    flight = {
        "dataset": "rfly",
        "flight_id": case_id,
        "session": _session_from_case(case_id, metadata),
        "fault_class": fault_class,
        "is_anomalous": fault_class != "normal",
        "t0_us": int(t0_us),
    }
    write_json(destination / "flight.json", flight)
    report = {
        **flight,
        "input_file": input_file,
        "dropped_non_monotonic": dropped,
        "topic_rows": {topic: int(len(frame)) for topic, frame in normalised.items()},
        "event_count": len(events),
    }
    write_json(destination / "ingest_report.json", report)
    return report


def discover_real_ulogs(root: str | Path = DEFAULT_BRONZE_ROOT) -> Iterator[tuple[str, Path]]:
    base = Path(root)
    for subset in sorted(REAL_SUBSETS):
        subset_root = base / subset
        if not subset_root.exists():
            continue
        for path in sorted(subset_root.rglob("*.ulg")):
            yield case_id_from_path(path, base), path


def ingest_rfly(
    bronze_root: str | Path = DEFAULT_BRONZE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    limit: int | None = None,
    flight_ids: set[str] | None = None,
    ulog_factory: Callable[[str, list[str]], Any] | None = None,
) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    exclusions = load_exclusions()
    parsed: list[dict] = []
    excluded: dict[str, str] = {}
    seen_cases: set[str] = set()
    for case_id, path in discover_real_ulogs(bronze_root):
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        if flight_ids is not None and case_id not in flight_ids:
            continue
        if case_id in exclusions:
            excluded[case_id] = exclusions[case_id]
            continue
        frames = read_ulog_topics(path, ulog_factory=ulog_factory)
        parsed.append(ingest_rfly_flight(case_id, frames, root, input_file=str(path)))
        if limit is not None and len(parsed) >= limit:
            break
    summary = {
        "dataset": "rfly",
        "flight_count": len(parsed),
        "excluded": excluded,
        "excluded_count": len(excluded),
        "event_count": sum(report["event_count"] for report in parsed),
    }
    write_json(root / "dataset_report.json", summary)
    return summary


def reconcile_rfly_fault_classes(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    """Reconcile already-materialised Real-Sensors metadata to headline class."""

    root = Path(output_root)
    changed_flights: list[str] = []
    for flight_path in sorted((root / "Real-Sensors").rglob("flight.json")):
        flight = json.loads(flight_path.read_text(encoding="utf-8"))
        if flight.get("fault_class") == "sensor":
            continue
        flight["fault_class"] = "sensor"
        write_json(flight_path, flight)
        report_path = flight_path.parent / "ingest_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["fault_class"] = "sensor"
        write_json(report_path, report)
        events_path = flight_path.parent / "events.json"
        events = json.loads(events_path.read_text(encoding="utf-8"))
        for event in events:
            event["fault_class"] = "sensor"
        write_json(events_path, events)
        changed_flights.append(str(flight["flight_id"]))
    return {
        "changed_count": len(changed_flights),
        "fault_class": "sensor",
        "changed_flights": changed_flights,
    }
