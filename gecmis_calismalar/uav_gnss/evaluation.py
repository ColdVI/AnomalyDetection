"""Explicit episode, burden, event deadline, and confidence metrics."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from uav_gnss.features import airborne_exposure_seconds, scoreable_exposure_seconds


def alarm_episodes(
    frame: pd.DataFrame,
    alarms: np.ndarray | pd.Series,
    *,
    merge_gap_s: float,
) -> pd.DataFrame:
    flags = np.asarray(alarms, dtype=bool)
    rows: list[dict] = []
    active = frame.loc[flags, ["flight_id", "timestamp_s"]]
    for flight_id, group in active.groupby("flight_id", sort=False):
        times = np.sort(group["timestamp_s"].to_numpy(float))
        if not len(times):
            continue
        start = end = float(times[0])
        for timestamp in times[1:]:
            timestamp = float(timestamp)
            if timestamp - end <= merge_gap_s:
                end = timestamp
            else:
                rows.append({"flight_id": flight_id, "episode_start": start, "episode_end": end})
                start = end = timestamp
        rows.append({"flight_id": flight_id, "episode_start": start, "episode_end": end})
    return pd.DataFrame(rows, columns=["flight_id", "episode_start", "episode_end"])


def natural_burden(
    frame: pd.DataFrame,
    alarms: np.ndarray | pd.Series,
    *,
    merge_gap_s: float,
) -> dict:
    episodes = alarm_episodes(frame, alarms, merge_gap_s=merge_gap_s)
    hours = scoreable_exposure_seconds(frame) / 3600.0
    airborne_hours = airborne_exposure_seconds(frame) / 3600.0
    return {
        "n_alert_episodes": int(len(episodes)),
        "n_flights": int(frame["flight_id"].nunique()),
        "n_alerted_flights": int(episodes["flight_id"].nunique()) if len(episodes) else 0,
        "scoreable_flight_hours": hours,
        "airborne_flight_hours": airborne_hours,
        "episodes_per_scoreable_flight_hour": len(episodes) / hours if hours > 0 else None,
        "scoreable_coverage": hours / airborne_hours if airborne_hours > 0 else None,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def deadline_event_metrics(
    fault_frame: pd.DataFrame,
    alarms: np.ndarray | pd.Series,
    *,
    deadline_s: float,
) -> dict:
    flags = np.asarray(alarms, dtype=bool)
    per_event: list[dict] = []
    for flight_id, group in fault_frame.groupby("flight_id", sort=False):
        onset = float(group["fault_onset_s"].iloc[0])
        end = float(group["fault_end_s"].iloc[0])
        deadline = min(end, onset + deadline_s)
        active = group.loc[
            flags[group.index.to_numpy()]
            & group["timestamp_s"].between(onset, deadline, inclusive="both")
        ]
        hit = not active.empty
        delay = float(active["timestamp_s"].iloc[0] - onset) if hit else None
        event_rows = group["timestamp_s"].between(onset, end, inclusive="both")
        evaluable_rows = event_rows & group["evaluable"]
        per_event.append(
            {
                "flight_id": flight_id,
                "flight_mode": group["flight_mode"].iloc[0],
                "fault_mode": int(group["fault_mode"].iloc[0]),
                "fault_mode_name": group["fault_mode_name"].iloc[0],
                "detected": hit,
                "first_alarm_delay_s": delay,
                "event_evaluable_fraction": (
                    float(evaluable_rows.sum() / event_rows.sum()) if event_rows.sum() else None
                ),
            }
        )
    total = len(per_event)
    detected = sum(row["detected"] for row in per_event)
    lower, upper = wilson_interval(detected, total)
    by_mode: dict[str, dict] = {}
    for fault_mode in (3, 4):
        subset = [row for row in per_event if row["fault_mode"] == fault_mode]
        hits = sum(row["detected"] for row in subset)
        by_mode[str(fault_mode)] = {
            "name": "noise" if fault_mode == 3 else "scale_factor",
            "n_events": len(subset),
            "n_detected": hits,
            "recall": hits / len(subset) if subset else None,
        }
    fractions = [
        row["event_evaluable_fraction"]
        for row in per_event
        if row["event_evaluable_fraction"] is not None
    ]
    delays = [row["first_alarm_delay_s"] for row in per_event if row["detected"]]
    return {
        "deadline_s": deadline_s,
        "n_events": total,
        "n_detected": detected,
        "recall": detected / total if total else None,
        "wilson_95": {"lower": lower, "upper": upper},
        "by_fault_mode": by_mode,
        "event_evaluable_coverage_macro": float(np.mean(fractions)) if fractions else None,
        "delay_s": {
            "median": float(np.median(delays)) if delays else None,
            "p95": float(np.quantile(delays, 0.95)) if delays else None,
        },
        "per_event": per_event,
    }

