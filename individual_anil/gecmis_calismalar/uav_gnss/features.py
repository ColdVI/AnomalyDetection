"""Causal PX4 ULog feature extraction for GNSS-integrity scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyulog import ULog

Z_CHANNELS = (
    "z_gps_hpos_n",
    "z_gps_hpos_e",
    "z_gps_hvel_n",
    "z_gps_hvel_e",
    "z_gps_vvel",
)
RATIO_CHANNELS = (
    "ratio_gps_hpos",
    "ratio_gps_hvel",
    "ratio_gps_vvel",
)
MODEL_INPUT_CHANNELS = Z_CHANNELS + RATIO_CHANNELS + (
    "gps_eph",
    "gps_epv",
    "gps_hdop",
    "gps_vdop",
    "gps_satellites_used",
    "gps_noise_per_ms",
    "gps_jamming_indicator",
    "local_vx",
    "local_vy",
    "local_vz",
)


def _topic_frame(
    ulog: ULog,
    topic: str,
    fields: dict[str, str],
) -> pd.DataFrame:
    try:
        data = ulog.get_dataset(topic).data
    except Exception:
        return pd.DataFrame(columns=["timestamp_s", *fields.values()])
    result = {"timestamp_s": np.asarray(data["timestamp"], dtype=float) / 1e6}
    for source, target in fields.items():
        values = data.get(source)
        result[target] = (
            np.asarray(values, dtype=float)
            if values is not None
            else np.full(len(result["timestamp_s"]), np.nan)
        )
    return pd.DataFrame(result).sort_values("timestamp_s")


def _merge_causal(base: pd.DataFrame, other: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    if other.empty:
        for column in other.columns:
            if column != "timestamp_s":
                base[column] = np.nan
        return base
    return pd.merge_asof(
        base.sort_values("timestamp_s"),
        other.sort_values("timestamp_s"),
        on="timestamp_s",
        direction="backward",
        tolerance=tolerance_s,
    )


def extract_flight_features(case: dict[str, Any], *, max_gap_s: float) -> pd.DataFrame:
    path = Path(case["path"])
    ulog = ULog(str(path))
    innovations = _topic_frame(
        ulog,
        "estimator_innovations",
        {
            "gps_hpos[0]": "innov_gps_hpos_n",
            "gps_hpos[1]": "innov_gps_hpos_e",
            "gps_hvel[0]": "innov_gps_hvel_n",
            "gps_hvel[1]": "innov_gps_hvel_e",
            "gps_vvel": "innov_gps_vvel",
        },
    )
    variances = _topic_frame(
        ulog,
        "estimator_innovation_variances",
        {
            "gps_hpos[0]": "var_gps_hpos_n",
            "gps_hpos[1]": "var_gps_hpos_e",
            "gps_hvel[0]": "var_gps_hvel_n",
            "gps_hvel[1]": "var_gps_hvel_e",
            "gps_vvel": "var_gps_vvel",
        },
    )
    ratios = _topic_frame(
        ulog,
        "estimator_innovation_test_ratios",
        {
            "gps_hpos[0]": "ratio_gps_hpos",
            "gps_hvel[0]": "ratio_gps_hvel",
            "gps_vvel": "ratio_gps_vvel",
        },
    )
    status = _topic_frame(
        ulog,
        "estimator_status",
        {
            "innovation_check_flags": "innovation_check_flags",
            "filter_fault_flags": "filter_fault_flags",
            "solution_status_flags": "solution_status_flags",
            "gps_check_fail_flags": "gps_check_fail_flags",
            "reset_count_pos_ne": "reset_count_pos_ne",
            "reset_count_vel_ne": "reset_count_vel_ne",
            "reset_count_vel_d": "reset_count_vel_d",
        },
    )
    gps = _topic_frame(
        ulog,
        "vehicle_gps_position",
        {
            "fix_type": "gps_fix_type",
            "eph": "gps_eph",
            "epv": "gps_epv",
            "hdop": "gps_hdop",
            "vdop": "gps_vdop",
            "satellites_used": "gps_satellites_used",
            "noise_per_ms": "gps_noise_per_ms",
            "jamming_indicator": "gps_jamming_indicator",
        },
    )
    land = _topic_frame(ulog, "vehicle_land_detected", {"landed": "landed"})
    vehicle = _topic_frame(ulog, "vehicle_status", {"nav_state": "nav_state"})
    local = _topic_frame(
        ulog,
        "vehicle_local_position",
        {"vx": "local_vx", "vy": "local_vy", "vz": "local_vz"},
    )
    if innovations.empty:
        return pd.DataFrame()
    frame = innovations
    for other, tolerance in (
        (variances, 0.6),
        (ratios, 0.6),
        (status, 0.6),
        (gps, 0.6),
        (land, 2.0),
        (vehicle, 1.0),
        (local, 0.6),
    ):
        frame = _merge_causal(frame, other, tolerance)
    mapping = (
        ("gps_hpos_n", "z_gps_hpos_n"),
        ("gps_hpos_e", "z_gps_hpos_e"),
        ("gps_hvel_n", "z_gps_hvel_n"),
        ("gps_hvel_e", "z_gps_hvel_e"),
        ("gps_vvel", "z_gps_vvel"),
    )
    for source, target in mapping:
        innovation = pd.to_numeric(frame[f"innov_{source}"], errors="coerce")
        variance = pd.to_numeric(frame[f"var_{source}"], errors="coerce")
        valid = variance.gt(0) & np.isfinite(innovation) & np.isfinite(variance)
        frame[target] = np.where(valid, innovation / np.sqrt(variance), np.nan)
    frame["flight_id"] = case["flight_id"]
    frame["flight_mode"] = case["flight_mode"]
    frame["class"] = case["class"]
    frame["role"] = case["role"]
    frame["fault_id"] = case.get("fault_id")
    frame["fault_mode"] = case.get("fault_mode")
    frame["fault_mode_name"] = case.get("fault_mode_name")
    frame["fault_onset_s"] = case.get("fault_onset_s")
    frame["fault_end_s"] = case.get("fault_end_s")
    frame["dt_s"] = frame["timestamp_s"].diff()
    has_z = frame.loc[:, Z_CHANNELS].notna().any(axis=1)
    airborne = pd.to_numeric(frame["landed"], errors="coerce").eq(0)
    valid_time = frame["dt_s"].gt(0) & frame["dt_s"].le(max_gap_s)
    frame["evaluable"] = has_z & airborne & valid_time
    frame["not_evaluable_reason"] = np.select(
        [
            ~airborne,
            ~has_z,
            ~valid_time,
        ],
        ["landed_or_unknown", "missing_innovation_or_variance", "initial_or_time_gap"],
        default="",
    )
    frame["context_phase"] = frame["flight_mode"]
    frame["context_cadence"] = pd.cut(
        frame["dt_s"],
        bins=[-np.inf, 0.55, 1.0, np.inf],
        labels=["nominal_2hz", "slow", "gap"],
    ).astype("string").fillna("initial")
    ratio_values = frame.loc[:, RATIO_CHANNELS].apply(pd.to_numeric, errors="coerce")
    frame["px4_native_score"] = ratio_values.max(axis=1, skipna=True).fillna(0.0)
    flags = pd.to_numeric(frame["innovation_check_flags"], errors="coerce").fillna(0)
    frame["px4_native_alarm"] = frame["evaluable"] & (
        frame["px4_native_score"].gt(1.0) | flags.ne(0)
    )
    return frame


def load_role_features(
    cases: list[dict[str, Any]],
    *,
    max_gap_s: float,
) -> pd.DataFrame:
    frames = [extract_flight_features(case, max_gap_s=max_gap_s) for case in cases]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def airborne_exposure_seconds(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    dt = pd.to_numeric(frame["dt_s"], errors="coerce").fillna(0.0)
    airborne = pd.to_numeric(frame["landed"], errors="coerce").eq(0)
    return float(dt.where(airborne & dt.gt(0), 0.0).sum())


def scoreable_exposure_seconds(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    dt = pd.to_numeric(frame["dt_s"], errors="coerce").fillna(0.0)
    return float(dt.where(frame["evaluable"] & dt.gt(0), 0.0).sum())

