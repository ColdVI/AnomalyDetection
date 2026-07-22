"""Causal 10 Hz, multi-rate RflyMAD parser for the v2 experiment track.

The existing 1 Hz parsed batches remain immutable historical baselines.  V2 is
written to a separate directory so that acquisition can resume without mixing
feature semantics inside one parquet collection.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.parquet import read_schema

from rfly_full.contract import (
    DATASET_MANIFEST, V2_ROOT, _source_records, build_manifest, domain_of, taxonomy,
)
from rfly_full.pipeline import FEATURES, _atomic_json, _euler, _test_info_interval, _topic

PARSED_10HZ_ROOT = V2_ROOT / "parsed_10hz"
PARSE_STATE = V2_ROOT / "parse_10hz_state.json"
SAMPLE_HZ = 10
FEATURE_SCHEMA_VERSION = 1
TRUTH_SCHEMA_VERSION = 2
CROSSCHECK_SCHEMA_VERSION = 2
CROSSCHECK_ONSET_TOLERANCE_S = 16.0
CROSSCHECK_COLUMNS = (
    "truth_crosscheck_eligible_v2",
    "truth_crosscheck_onset_delta_s",
    "truth_crosscheck_offset_delta_s",
    "truth_crosscheck_overlap_s",
    "truth_crosscheck_disagreement_v2",
    "truth_crosscheck_schema_version",
)
HF_FEATURES = (
    "imu_accel_mag_mean", "imu_accel_mag_std", "imu_accel_mag_rms",
    "imu_accel_mag_ptp", "imu_accel_diff_rms",
    "imu_gyro_mag_mean", "imu_gyro_mag_std", "imu_gyro_mag_rms",
    "imu_gyro_mag_ptp", "imu_gyro_diff_rms",
    "output_hf_mean", "output_hf_std", "output_hf_ptp", "output_diff_rms",
)
V2_FEATURES = (*FEATURES, *HF_FEATURES)


def _control_domain(package: str) -> str:
    return domain_of(str(package)).upper()


def _truth_crosscheck_metrics(
    t_rel_s: np.ndarray,
    control_active: np.ndarray | None,
    planned_active: np.ndarray | None,
) -> dict[str, bool | float | int]:
    """Compare control/TestInfo intervals without requiring samplewise identity.

    TestInfo and the control topic use different time origins in SIL, and their
    offset semantics differ by fault family.  The v2 decision therefore checks
    a bounded onset delta plus actual interval overlap.  The signed offset delta
    remains visible as a diagnostic instead of silently widening a tolerance.
    """
    result: dict[str, bool | float | int] = {
        "truth_crosscheck_eligible_v2": False,
        "truth_crosscheck_onset_delta_s": float("nan"),
        "truth_crosscheck_offset_delta_s": float("nan"),
        "truth_crosscheck_overlap_s": 0.0,
        "truth_crosscheck_disagreement_v2": False,
        "truth_crosscheck_schema_version": CROSSCHECK_SCHEMA_VERSION,
    }
    if control_active is None or planned_active is None:
        return result
    t = np.asarray(t_rel_s, dtype=float)
    control = np.asarray(control_active, dtype=bool)
    planned = np.asarray(planned_active, dtype=bool)
    if len(t) != len(control) or len(t) != len(planned):
        raise ValueError("crosscheck arrays must have identical lengths")
    if not (control.any() or planned.any()):
        return result
    result["truth_crosscheck_eligible_v2"] = True
    if not control.any() or not planned.any():
        result["truth_crosscheck_disagreement_v2"] = True
        return result

    control_start, control_end = float(t[control].min()), float(t[control].max())
    planned_start, planned_end = float(t[planned].min()), float(t[planned].max())
    onset_delta = control_start - planned_start
    offset_delta = control_end - planned_end
    overlap = control & planned
    step_s = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    result.update({
        "truth_crosscheck_onset_delta_s": onset_delta,
        "truth_crosscheck_offset_delta_s": offset_delta,
        "truth_crosscheck_overlap_s": float(overlap.sum() * step_s),
        "truth_crosscheck_disagreement_v2": bool(
            abs(onset_delta) > CROSSCHECK_ONSET_TOLERANCE_S or not overlap.any()
        ),
    })
    return result


def _migrate_truth_state(state: dict, manifest: pd.DataFrame) -> set[str]:
    if state.get('truth_schema_version') == TRUTH_SCHEMA_VERSION:
        return set()
    packages = manifest['case_id'].astype(str).str.split('/', n=1).str[0]
    affected = set(manifest.loc[
        packages.str.match(r'^(?:SIL|HIL)_'), 'canonical_case_id'
    ].astype(str))
    completed = set(map(str, state.get('completed', [])))
    state['completed'] = sorted(completed - affected)
    state['failed'] = {
        str(key): value for key, value in state.get('failed', {}).items()
        if str(key) not in affected
    }
    state['truth_schema_version'] = TRUTH_SCHEMA_VERSION
    state['truth_reparse_invalidated'] = len(completed & affected)
    return affected


def _merge_causal(base: pd.DataFrame, extra: pd.DataFrame | None, tolerance_us: int) -> pd.DataFrame:
    if extra is None:
        return base
    return pd.merge_asof(
        base.sort_values("timestamp"), extra.sort_values("timestamp"),
        on="timestamp", direction="backward", tolerance=tolerance_us,
    )


def _binned_stats(
    timestamp: np.ndarray, values: np.ndarray, *, step_us: int, prefix: str
) -> pd.DataFrame:
    timestamp = np.asarray(timestamp, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return pd.DataFrame(columns=["timestamp"])
    timestamp, values = timestamp[finite], values[finite]
    endpoint = ((timestamp + step_us - 1) // step_us) * step_us
    data = pd.DataFrame({"timestamp": endpoint, "value": values})
    data["difference"] = data["value"].diff()
    grouped = data.groupby("timestamp", sort=True)
    result = grouped["value"].agg(["mean", "std", "min", "max"]).reset_index()
    result[f"{prefix}_mean"] = result.pop("mean")
    result[f"{prefix}_std"] = result.pop("std").fillna(0.0)
    result[f"{prefix}_rms"] = np.sqrt(grouped["value"].apply(lambda value: np.mean(np.square(value))).to_numpy())
    result[f"{prefix}_ptp"] = result.pop("max") - result.pop("min")
    diff = grouped["difference"].apply(
        lambda value: float(np.sqrt(np.nanmean(np.square(value)))) if value.notna().any() else 0.0
    )
    result[f"{prefix}_diff_rms"] = diff.to_numpy()
    return result


def _imu_summaries(ulog, step_us: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        data = ulog.get_dataset("sensor_combined").data
    except (KeyError, IndexError, StopIteration):
        return None, None
    timestamp = np.asarray(data.get("timestamp", []), dtype=np.int64)
    accel_fields = [f"accelerometer_m_s2[{index}]" for index in range(3)]
    gyro_fields = [f"gyro_rad[{index}]" for index in range(3)]
    accel = None
    gyro = None
    if len(timestamp) and all(field in data for field in accel_fields):
        magnitude = np.linalg.norm(np.column_stack([data[field] for field in accel_fields]), axis=1)
        accel = _binned_stats(timestamp, magnitude, step_us=step_us, prefix="imu_accel_mag")
        accel = accel.rename(columns=dict(imu_accel_mag_diff_rms='imu_accel_diff_rms'))
    if len(timestamp) and all(field in data for field in gyro_fields):
        magnitude = np.linalg.norm(np.column_stack([data[field] for field in gyro_fields]), axis=1)
        gyro = _binned_stats(timestamp, magnitude, step_us=step_us, prefix="imu_gyro_mag")
        gyro = gyro.rename(columns=dict(imu_gyro_mag_diff_rms='imu_gyro_diff_rms'))
    return accel, gyro


def _output_summaries(ulog, step_us: int) -> pd.DataFrame | None:
    outputs = _topic(ulog, "actuator_outputs", tuple(f"output[{index}]" for index in range(16)))
    if outputs is None:
        return None
    columns = [column for column in outputs if column.startswith("output[")]
    values = outputs[columns].replace(0, np.nan)
    per_sample = values.mean(axis=1).to_numpy(dtype=float)
    result = _binned_stats(
        outputs["timestamp"].to_numpy(), per_sample, step_us=step_us, prefix="output_hf"
    )
    # Keep the public feature name concise and symmetric with output_diff_rms.
    return result.rename(columns={"output_hf_diff_rms": "output_diff_rms"})


def _active_control(base: pd.DataFrame, ulog, domain: str) -> np.ndarray | None:
    fault = _topic(ulog, "rfly_ctrl_lxl", ("id", "mode"), "ctrl_")
    if fault is None or "ctrl_id" not in fault or "ctrl_mode" not in fault:
        return None
    aligned = _merge_causal(base[["timestamp"]], fault, tolerance_us=2_000_000)
    sentinel = 0 if domain in {"SIL", "HIL"} else 1500
    present = aligned[["ctrl_id", "ctrl_mode"]].notna().all(axis=1)
    active = present & (
        aligned["ctrl_id"].ne(sentinel) | aligned["ctrl_mode"].ne(sentinel)
    )
    return active.to_numpy(dtype=bool)


def parse_ulg_v2(
    path: Path, object_name: str, package: str, *, sample_hz: int = SAMPLE_HZ
) -> pd.DataFrame:
    from pyulog import ULog

    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    step_us = int(round(1_000_000 / sample_hz))
    ulog = ULog(str(path))
    local = _topic(
        ulog, "vehicle_local_position",
        ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"), "local_",
    )
    if local is None or len(local) < 2:
        raise ValueError("vehicle_local_position unavailable")
    start, end = int(local.timestamp.min()), int(local.timestamp.max())
    timestamps = np.arange(math.ceil(start / step_us) * step_us, end + 1, step_us, dtype=np.int64)
    base = _merge_causal(pd.DataFrame({"timestamp": timestamps}), local, max(step_us * 2, 200_000))

    attitude = _topic(ulog, "vehicle_attitude", tuple(f"q[{index}]" for index in range(4)))
    if attitude is not None and all(f"q[{index}]" in attitude for index in range(4)):
        base = _merge_causal(base, _euler(attitude), max(step_us * 2, 200_000))
    controls = _topic(ulog, "actuator_controls_0", tuple(f"control[{index}]" for index in range(4)))
    if controls is not None:
        controls = controls.rename(columns={
            f"control[{index}]": name for index, name in enumerate(
                ("act_roll", "act_pitch", "act_yaw", "act_thrust")
            )
        })
    base = _merge_causal(base, controls, max(step_us * 2, 200_000))
    outputs = _topic(ulog, "actuator_outputs", tuple(f"output[{index}]" for index in range(16)))
    if outputs is not None:
        columns = [column for column in outputs if column.startswith("output[")]
        values = outputs[columns].replace(0, np.nan)
        outputs = outputs[["timestamp"]].assign(
            output_mean=values.mean(axis=1), output_std=values.std(axis=1),
            output_range=values.max(axis=1) - values.min(axis=1),
        )
    base = _merge_causal(base, outputs, max(step_us * 2, 200_000))
    base = _merge_causal(base, _topic(ulog, "battery_status", ("voltage_v", "current_a", "remaining"), "battery_"), 2_000_000)
    base = _merge_causal(base, _topic(ulog, "estimator_status", ("vel_test_ratio", "pos_test_ratio", "hgt_test_ratio", "mag_test_ratio")), 2_000_000)
    base = _merge_causal(base, _topic(ulog, "vehicle_gps_position", ("eph", "epv", "satellites_used"), "gps_"), 2_000_000)
    accel, gyro = _imu_summaries(ulog, step_us)
    base = _merge_causal(base, accel, step_us)
    base = _merge_causal(base, gyro, step_us)
    base = _merge_causal(base, _output_summaries(ulog, step_us), step_us)

    base["t_rel_s"] = (base["timestamp"] - base["timestamp"].iloc[0]) / 1e6
    tax = taxonomy(package, object_name)
    domain = domain_of(str(package))
    control_active = _active_control(base, ulog, _control_domain(package))
    published = _test_info_interval(path)
    planned = (
        base["t_rel_s"].between(*published, inclusive="both").to_numpy()
        if published is not None else np.zeros(len(base), dtype=bool)
    )
    if tax["fault_family"] == "NoFault":
        condition_active = np.zeros(len(base), dtype=bool)
        truth_source = "normal_no_fault"
    elif control_active is not None and control_active.any():
        condition_active = control_active
        truth_source = "rfly_ctrl_lxl"
    elif published is not None:
        condition_active = planned
        truth_source = "test_info_fallback"
    else:
        condition_active = np.zeros(len(base), dtype=bool)
        truth_source = "missing"

    base["case_id"] = object_name.rsplit("/Log/", 1)[0] if "/Log/" in object_name else str(Path(object_name).parent).replace("\\", "/")
    base["object_name"] = object_name
    base["package"] = package
    base["domain"] = domain
    base["fault_family"] = tax["fault_family"]
    base["fault_subtype"] = tax["fault_subtype"]
    base["environment_condition"] = tax["environment_condition"]
    base["system_fault"] = bool(tax["system_fault"])
    base["condition_active"] = condition_active
    base["fault_active"] = condition_active & bool(tax["system_fault"])
    base["environment_active"] = condition_active & (tax["environment_condition"] != "None")
    base["truth_source"] = truth_source
    base["truth_crosscheck_disagreement"] = bool(
        control_active is not None and control_active.any() and published is not None
        and np.mean(control_active != planned) > 0.01
    )
    crosscheck = _truth_crosscheck_metrics(
        base["t_rel_s"].to_numpy(dtype=float),
        control_active,
        planned if published is not None else None,
    )
    for key, value in crosscheck.items():
        base[key] = value
    for feature in V2_FEATURES:
        if feature not in base:
            base[feature] = np.nan
        base[feature] = pd.to_numeric(base[feature], errors="coerce").astype(np.float32)
    metadata = [
        "timestamp", "t_rel_s", "case_id", "object_name", "package", "domain",
        "fault_family", "fault_subtype", "environment_condition", "system_fault",
        "condition_active", "fault_active", "environment_active", "truth_source",
        "truth_crosscheck_disagreement", *CROSSCHECK_COLUMNS,
    ]
    return base[[*metadata, *V2_FEATURES]]


def _raw_paths() -> dict[str, Path]:
    ulogs, _ = _source_records()
    return {
        name: Path(record["path"]) for name, record in ulogs.items()
        if record.get("path")
    }


def postprocess_crosscheck_metrics(*, split: str = "development") -> dict:
    """Add v2 cross-check metadata without reparsing ULogs or reading other splits."""
    if split != "development":
        raise ValueError("cross-check post-processing is restricted to development")
    manifest = pd.read_parquet(
        DATASET_MANIFEST,
        columns=[
            "canonical_case_id", "domain", "split",
            "planned_fault_start_s", "planned_fault_end_s",
        ],
    ).drop_duplicates("canonical_case_id")
    manifest = manifest.loc[manifest["split"].eq(split)].copy()
    completed: list[str] = []
    failed: dict[str, str] = {}
    eligible = 0
    disagreement = 0
    for row in manifest.itertuples(index=False):
        canonical = str(row.canonical_case_id)
        target = PARSED_10HZ_ROOT / str(row.domain) / f"{canonical}.parquet"
        try:
            frame = pd.read_parquet(target)
            t = frame["t_rel_s"].to_numpy(dtype=float)
            has_published = bool(
                pd.notna(row.planned_fault_start_s)
                and pd.notna(row.planned_fault_end_s)
            )
            planned = (
                frame["t_rel_s"].between(
                    float(row.planned_fault_start_s),
                    float(row.planned_fault_end_s),
                    inclusive="both",
                ).to_numpy(dtype=bool)
                if has_published else None
            )
            control = (
                frame["condition_active"].to_numpy(dtype=bool)
                if str(frame["truth_source"].iloc[0]) == "rfly_ctrl_lxl" else None
            )
            metrics = _truth_crosscheck_metrics(t, control, planned)
            for key, value in metrics.items():
                frame[key] = value
            temporary = target.with_suffix(".crosscheck.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
        except Exception as exc:
            failed[canonical] = str(exc)
            continue
        completed.append(canonical)
        eligible += int(bool(metrics["truth_crosscheck_eligible_v2"]))
        disagreement += int(bool(metrics["truth_crosscheck_disagreement_v2"]))
    state = {
        "status": "complete" if not failed and len(completed) == len(manifest) else "incomplete",
        "split": split,
        "crosscheck_schema_version": CROSSCHECK_SCHEMA_VERSION,
        "onset_tolerance_s": CROSSCHECK_ONSET_TOLERANCE_S,
        "flights": int(len(manifest)),
        "completed": len(completed),
        "failed": failed,
        "eligible": eligible,
        "disagreement_v2": disagreement,
        "locked_test_features_read": False,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _atomic_json(V2_ROOT / "crosscheck_v2_development_state.json", state)
    return state


def run(
    *, max_flights: int | None = None, deadline: datetime | None = None,
    priority_normal: bool = False,
) -> dict:
    if not DATASET_MANIFEST.exists():
        build_manifest()
    manifest = pd.read_parquet(DATASET_MANIFEST)
    manifest = manifest.drop_duplicates("canonical_case_id")
    if priority_normal:
        manifest = (
            manifest.assign(
                _parse_priority=~manifest["evaluation_role"].eq("normal_reference"),
                _split_priority=~manifest["split"].eq("development"),
            )
            .sort_values(["_parse_priority", "_split_priority", "canonical_case_id"])
            .drop(columns=["_parse_priority", "_split_priority"])
        )
    else:
        manifest = manifest.sort_values("canonical_case_id")
    paths = _raw_paths()
    state = json.loads(PARSE_STATE.read_text(encoding="utf-8")) if PARSE_STATE.exists() else {
        "schema_version": 2, "sample_hz": SAMPLE_HZ, "completed": [], "failed": {},
    }
    if state.get('feature_schema_version') != FEATURE_SCHEMA_VERSION:
        state['completed'] = []
        state['failed'] = {}
        state['feature_schema_version'] = FEATURE_SCHEMA_VERSION
    _migrate_truth_state(state, manifest)
    completed = set(state["completed"])
    processed_now = 0
    for row in manifest.itertuples(index=False):
        canonical = str(row.canonical_case_id)
        if canonical in completed:
            target = PARSED_10HZ_ROOT / str(row.domain) / f'{canonical}.parquet'
            if target.exists() and set(V2_FEATURES).issubset(read_schema(target).names):
                continue
            completed.remove(canonical)
        if max_flights is not None and processed_now >= max_flights:
            break
        if deadline is not None and datetime.now().astimezone() >= deadline:
            state["stop_reason"] = "deadline"
            break
        raw = paths.get(str(row.object_name))
        if raw is None or not raw.exists():
            state["failed"][canonical] = "raw ULog path unavailable"
            continue
        try:
            frame = parse_ulg_v2(raw, str(row.object_name), str(row.case_id).split("/", 1)[0])
            for key in ("canonical_case_id", "split", "cv_fold", "evaluation_role"):
                frame[key] = getattr(row, key)
            target = PARSED_10HZ_ROOT / str(row.domain) / f"{canonical}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
        except Exception as exc:
            state["failed"][canonical] = str(exc)
            state["updated_at"] = datetime.now().astimezone().isoformat()
            _atomic_json(PARSE_STATE, state)
            continue
        completed.add(canonical)
        state["completed"] = sorted(completed)
        state["failed"].pop(canonical, None)
        state["updated_at"] = datetime.now().astimezone().isoformat()
        state["stop_reason"] = "running"
        _atomic_json(PARSE_STATE, state)
        processed_now += 1
    remaining = int(len(manifest) - len(completed))
    state["remaining"] = remaining
    state["canonical_flights"] = int(len(manifest))
    if remaining == 0:
        state["stop_reason"] = "complete"
    elif state.get("stop_reason") == "running":
        state["stop_reason"] = "max_flights" if max_flights is not None else "incomplete"
    _atomic_json(PARSE_STATE, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-flights", type=int)
    parser.add_argument("--deadline")
    parser.add_argument(
        "--priority-normal", action="store_true",
        help="Parse development/locked NoFault flights before other roles.",
    )
    args = parser.parse_args()
    state = run(
        max_flights=args.max_flights,
        deadline=datetime.fromisoformat(args.deadline) if args.deadline else None,
        priority_normal=args.priority_normal,
    )
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
