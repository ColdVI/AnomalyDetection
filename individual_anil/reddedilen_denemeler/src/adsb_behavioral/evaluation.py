"""Event and flight-level evaluation for injected ADS-B anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.adsb_behavioral.decision import alarm_onsets, alarm_states, exposure_hours


def evaluate_detector(
    normal_test: pd.DataFrame,
    injected_test: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    k: int = 2,
    n: int = 3,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    normal = normal_test.copy()
    injected = injected_test.copy()
    normal["alarm_onset"] = alarm_onsets(normal, score_col=score_col, threshold=threshold, k=k, n=n)
    injected["alarm_onset"] = alarm_onsets(injected, score_col=score_col, threshold=threshold, k=k, n=n)
    injected["alarm_active"] = alarm_states(
        injected, score_col=score_col, threshold=threshold, k=k, n=n
    )

    normal_hours = exposure_hours(normal)
    false_events = int(normal["alarm_onset"].sum())
    fa_per_hour = float(false_events / normal_hours) if normal_hours > 0 else float("inf")

    event_rows: list[dict] = []
    for flight_id, group in injected.groupby("flight_id", sort=True):
        start = float(group["event_start_utc"].iloc[0])
        end = float(group["event_end_utc"].iloc[0])
        alarms = group.loc[
            group["alarm_active"] & group["timestamp_utc"].between(start, end), "timestamp_utc"
        ]
        detected = bool(len(alarms))
        event_rows.append(
            {
                "flight_id": flight_id,
                "source_flight_id": group["source_flight_id"].iloc[0],
                "injection_type": group["injection_type"].iloc[0],
                "severity": group["severity"].iloc[0],
                "event_start_utc": start,
                "event_end_utc": end,
                "detected": detected,
                "delay_s": float(alarms.iloc[0] - start) if detected else np.nan,
            }
        )
    events = pd.DataFrame(event_rows)

    normal_flights = normal.groupby("flight_id")["alarm_onset"].any()
    injected_flights = events.set_index("flight_id")["detected"] if len(events) else pd.Series(dtype=bool)
    tp = int(injected_flights.sum())
    fn = int((~injected_flights).sum())
    fp = int(normal_flights.sum())
    tn = int((~normal_flights).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    breakdown = (
        events.groupby(["severity", "injection_type"], as_index=False)
        .agg(events=("detected", "size"), detected=("detected", "sum"), event_recall=("detected", "mean"), median_delay_s=("delay_s", "median"))
        if len(events)
        else pd.DataFrame()
    )
    overall = {
        "score_col": score_col,
        "threshold": float(threshold),
        "normal_test_hours": normal_hours,
        "false_events": false_events,
        "false_events_per_hour": fa_per_hour,
        "event_recall": float(events["detected"].mean()) if len(events) else 0.0,
        "median_detection_delay_s": float(events["delay_s"].median()) if events["detected"].any() else None,
        "flight_tp": tp,
        "flight_fp": fp,
        "flight_tn": tn,
        "flight_fn": fn,
        "flight_precision": precision,
        "flight_recall": recall,
        "flight_f1": f1,
    }
    return overall, events, breakdown
