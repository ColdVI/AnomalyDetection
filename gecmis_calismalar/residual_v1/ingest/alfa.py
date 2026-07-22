"""Natural-rate ALFA processed-CSV ingestion.

The output deliberately retains one parquet per source topic. Cross-topic
alignment belongs to the feature layer and is never performed here.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator, Mapping
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from gecmis_calismalar.residual_v1.ingest.common import (
    drop_non_monotonic_timestamps,
    fix_quaternion_sign_continuity,
    wrap_radians,
    write_json,
)

DEFAULT_OUTPUT_ROOT = Path("artifacts/residual_v1/silver/alfa")
TIMESTAMP_COLUMN = "%time"
_DATE = re.compile(r"carbonZ_(\d{4}-\d{2}-\d{2})")

_COLUMN_MAP: dict[str, dict[str, str]] = {
    "mavros-nav_info-roll": {"field.commanded": "roll_cmd", "field.measured": "roll"},
    "mavros-nav_info-pitch": {"field.commanded": "pitch_cmd", "field.measured": "pitch"},
    "mavros-nav_info-yaw": {"field.commanded": "yaw_cmd", "field.measured": "yaw"},
    "mavros-nav_info-airspeed": {
        "field.commanded": "airspeed_cmd",
        "field.measured": "airspeed",
    },
    "mavros-imu-data": {
        "field.orientation.x": "quat_x",
        "field.orientation.y": "quat_y",
        "field.orientation.z": "quat_z",
        "field.orientation.w": "quat_w",
        "field.angular_velocity.x": "roll_rate",
        "field.angular_velocity.y": "pitch_rate",
        "field.angular_velocity.z": "yaw_rate",
        "field.linear_acceleration.x": "accel_x",
        "field.linear_acceleration.y": "accel_y",
        "field.linear_acceleration.z": "accel_z",
    },
    "mavros-rc-out": {
        # The Carbon-Z airframe logs the two aileron servo outputs on
        # MAVLink SERVO5/SERVO6 (zero-based RCOut channels4/channels5).
        # channels0 is constant at 1500 in every one of the 47 processed
        # flights, so treating it as aileron input silently kills R1.
        "field.channels4": "aileron_left_cmd",
        "field.channels5": "aileron_right_cmd",
        "field.channels1": "elevator_cmd",
        "field.channels2": "throttle_pwm",
        "field.channels3": "rudder_cmd",
    },
    "mavros-vfr_hud": {
        "field.airspeed": "airspeed_hud",
        "field.groundspeed": "ground_speed",
        "field.throttle": "throttle_cmd",
        "field.altitude": "altitude_hud",
        "field.climb": "climb_rate",
    },
    "mavros-global_position-global": {
        "field.latitude": "latitude",
        "field.longitude": "longitude",
        "field.altitude": "altitude",
    },
    "mavros-local_position-velocity": {
        "field.twist.linear.x": "local_vx",
        "field.twist.linear.y": "local_vy",
        "field.twist.linear.z": "local_vz",
        "field.twist.angular.x": "roll_rate_local",
        "field.twist.angular.y": "pitch_rate_local",
        "field.twist.angular.z": "yaw_rate_local",
    },
    "mavros-nav_info-errors": {
        "field.alt_error": "altitude_error",
        "field.aspd_error": "airspeed_error",
        "field.xtrack_error": "xtrack_error",
        "field.wp_dist": "waypoint_distance",
    },
    "mavctrl-path_dev": {
        "field.x": "path_dev_x",
        "field.y": "path_dev_y",
        "field.z": "path_dev_z",
    },
}

_ANGLE_COLUMNS = {"roll_cmd", "roll", "pitch_cmd", "pitch", "yaw_cmd", "yaw"}
_PWM_DELTA_COLUMNS = {
    "aileron_cmd",
    "aileron_left_cmd",
    "aileron_right_cmd",
    "elevator_cmd",
    "rudder_cmd",
}
_FAILURE_PREFIX = "failure_status-"


def infer_fault_class(flight_id: str) -> str:
    lower = flight_id.lower()
    if "no_failure" in lower:
        return "normal"
    if "engine" in lower:
        return "engine"
    if "elevator" in lower:
        return "elevator"
    if "aileron" in lower and "rudder" in lower:
        return "aileron_rudder"
    if "aileron" in lower:
        return "aileron"
    if "rudder" in lower:
        return "rudder"
    return "unknown"


def _topic_from_filename(flight_id: str, filename: str) -> str | None:
    stem = Path(filename).stem
    prefix = f"{flight_id}-"
    return stem[len(prefix) :] if stem.startswith(prefix) else None


def _read_zip_flights(path: Path) -> Iterator[tuple[str, dict[str, pd.DataFrame], list[str]]]:
    with zipfile.ZipFile(path) as archive:
        grouped: dict[str, list[zipfile.ZipInfo]] = {}
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".csv"):
                continue
            flight_id = Path(member.filename).parent.name
            grouped.setdefault(flight_id, []).append(member)
        for flight_id in sorted(grouped):
            frames: dict[str, pd.DataFrame] = {}
            inputs: list[str] = []
            for member in grouped[flight_id]:
                topic = _topic_from_filename(flight_id, member.filename)
                if topic is None or (topic not in _COLUMN_MAP and not topic.startswith(_FAILURE_PREFIX)):
                    continue
                frames[topic] = pd.read_csv(BytesIO(archive.read(member)))
                inputs.append(member.filename)
            if frames:
                yield flight_id, frames, inputs


def _read_directory_flights(path: Path) -> Iterator[tuple[str, dict[str, pd.DataFrame], list[str]]]:
    roots = sorted(directory for directory in path.iterdir() if directory.is_dir())
    if not roots and list(path.glob("*.csv")):
        roots = [path]
    for root in roots:
        flight_id = root.name
        frames: dict[str, pd.DataFrame] = {}
        inputs: list[str] = []
        for csv_path in sorted(root.glob("*.csv")):
            topic = _topic_from_filename(flight_id, csv_path.name)
            if topic is None:
                topic = csv_path.stem
            if topic not in _COLUMN_MAP and not topic.startswith(_FAILURE_PREFIX):
                continue
            frames[topic] = pd.read_csv(csv_path)
            inputs.append(str(csv_path))
        if frames:
            yield flight_id, frames, inputs


def iter_alfa_flights(source: str | Path) -> Iterator[tuple[str, dict[str, pd.DataFrame], list[str]]]:
    path = Path(source)
    if path.suffix.lower() == ".zip":
        yield from _read_zip_flights(path)
    elif path.is_dir():
        yield from _read_directory_flights(path)
    else:
        raise ValueError(f"ALFA source must be a processed zip or directory: {path}")


def _clean_source_frames(
    topic_frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    cleaned: dict[str, pd.DataFrame] = {}
    dropped: dict[str, int] = {}
    for topic, frame in topic_frames.items():
        output, count = drop_non_monotonic_timestamps(frame, TIMESTAMP_COLUMN)
        cleaned[topic] = output
        dropped[topic] = count
    return cleaned, dropped


def normalise_alfa_topic(
    frame: pd.DataFrame,
    topic: str,
    t0_ns: float,
    *,
    trim_before_s: float | None = None,
) -> pd.DataFrame:
    """Select and normalise one ALFA topic without cross-topic alignment."""

    if TIMESTAMP_COLUMN not in frame:
        raise ValueError(f"{topic}: missing {TIMESTAMP_COLUMN}")
    mapping = _COLUMN_MAP.get(topic, {})
    columns = [TIMESTAMP_COLUMN, *[column for column in mapping if column in frame]]
    if topic.startswith(_FAILURE_PREFIX) and "field.data" in frame:
        columns.append("field.data")
    result = frame.loc[:, list(dict.fromkeys(columns))].rename(columns=mapping).copy()
    result.insert(0, "t", (pd.to_numeric(result.pop(TIMESTAMP_COLUMN), errors="coerce") - t0_ns) / 1e9)
    for column in result.columns.difference(["t"]):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if {"aileron_left_cmd", "aileron_right_cmd"}.issubset(result.columns):
        # A single R1 command represents net symmetric roll authority. Keeping
        # both natural-rate servo columns alongside the mean preserves the
        # left/right fault evidence for audit without widening the v1 model.
        result["aileron_cmd"] = result[["aileron_left_cmd", "aileron_right_cmd"]].mean(
            axis=1
        )
    for column in _ANGLE_COLUMNS.intersection(result.columns):
        result[column] = wrap_radians(np.deg2rad(result[column]))
    for column in _PWM_DELTA_COLUMNS.intersection(result.columns):
        values = pd.to_numeric(result[column], errors="coerce")
        baseline_values = values
        if trim_before_s is not None:
            baseline_values = values.loc[pd.to_numeric(result["t"], errors="coerce") < trim_before_s]
        baseline = baseline_values.median()
        if pd.isna(baseline):
            raise ValueError(f"{topic}: no finite PWM trim samples before fault onset")
        result[column] = values - float(baseline)
    if topic == "mavros-imu-data":
        result = fix_quaternion_sign_continuity(
            result, ("quat_x", "quat_y", "quat_z", "quat_w")
        )
    return result.reset_index(drop=True)


def extract_failure_events(
    topic_frames: Mapping[str, pd.DataFrame], t0_ns: float
) -> list[dict[str, float | str]]:
    events: list[dict[str, float | str]] = []
    for topic, frame in sorted(topic_frames.items()):
        if not topic.startswith(_FAILURE_PREFIX) or "field.data" not in frame:
            continue
        active = frame.loc[pd.to_numeric(frame["field.data"], errors="coerce").fillna(0) != 0]
        if active.empty:
            continue
        times = (pd.to_numeric(active[TIMESTAMP_COLUMN], errors="coerce") - t0_ns) / 1e9
        fault_class = topic[len(_FAILURE_PREFIX) :].removesuffix("s")
        events.append(
            {
                "fault_class": fault_class,
                "onset_s": float(times.min()),
                "end_s": float(times.max()),
            }
        )
    return events


def fault_class_from_events(events: list[dict[str, float | str]], flight_id: str) -> str:
    classes = sorted({str(event["fault_class"]) for event in events})
    return "_".join(classes) if classes else infer_fault_class(flight_id)


def ingest_alfa_flight(
    flight_id: str,
    topic_frames: Mapping[str, pd.DataFrame],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    input_files: list[str] | None = None,
) -> dict:
    """Write one ALFA flight as natural-rate topic parquets."""

    destination = Path(output_root) / flight_id
    if destination.exists():
        raise FileExistsError(destination)
    cleaned, dropped = _clean_source_frames(topic_frames)
    nonempty_times = [
        pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce").dropna()
        for frame in cleaned.values()
        if TIMESTAMP_COLUMN in frame and not frame.empty
    ]
    if not nonempty_times:
        raise ValueError(f"{flight_id}: no usable timestamp")
    t0_ns = float(min(series.min() for series in nonempty_times))
    events = extract_failure_events(cleaned, t0_ns)
    first_onset_s = min(
        (float(event["onset_s"]) for event in events),
        default=None,
    )
    destination.mkdir(parents=True)
    row_counts: dict[str, int] = {}
    for topic, frame in cleaned.items():
        normalised = normalise_alfa_topic(
            frame,
            topic,
            t0_ns,
            trim_before_s=first_onset_s,
        )
        if normalised.empty:
            continue
        normalised.to_parquet(destination / f"{topic}.parquet", index=False)
        row_counts[topic] = int(len(normalised))
    write_json(destination / "events.json", events)
    session_match = _DATE.search(flight_id)
    metadata = {
        "dataset": "alfa",
        "flight_id": flight_id,
        "session": session_match.group(1) if session_match else "unknown",
        "fault_class": fault_class_from_events(events, flight_id),
        "is_anomalous": bool(events),
        "t0_ns": int(t0_ns),
    }
    write_json(destination / "flight.json", metadata)
    report = {
        **metadata,
        "dropped_non_monotonic": dropped,
        "input_files": input_files or [],
        "topic_rows": row_counts,
    }
    write_json(destination / "ingest_report.json", report)
    return report


def ingest_alfa(
    source: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    flight_ids: set[str] | None = None,
) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    reports = []
    for flight_id, frames, inputs in iter_alfa_flights(source):
        if flight_ids is not None and flight_id not in flight_ids:
            continue
        reports.append(ingest_alfa_flight(flight_id, frames, root, input_files=inputs))
    summary = {
        "dataset": "alfa",
        "source": str(source),
        "flight_count": len(reports),
        "event_count": sum(int(report["is_anomalous"]) for report in reports),
        "dropped_non_monotonic": sum(
            sum(report["dropped_non_monotonic"].values()) for report in reports
        ),
    }
    write_json(root / "dataset_report.json", summary)
    return summary


def load_alfa_flight(path: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(path)
    return {file.stem: pd.read_parquet(file) for file in sorted(root.glob("*.parquet"))}
