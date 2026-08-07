"""Explicit event, flight and window metrics for the RflyMAD DL track."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _truth_mask(
    source_id: str,
    times: np.ndarray,
    label: str,
    intervals: dict[str, tuple[float, float]],
) -> np.ndarray:
    if label == "normal":
        return np.zeros(len(times), dtype=bool)
    onset, end = intervals[source_id]
    return (times >= onset) & (times <= end)


def evaluate_policy(
    frame: pd.DataFrame,
    score_name: str,
    policy,
    *,
    intervals: dict[str, tuple[float, float]],
) -> tuple[dict, dict, list[dict], list[dict]]:
    detected = 0
    events = 0
    false_alarms = 0
    normal_hours = 0.0
    delays: list[float] = []
    flight_tp = flight_fn = flight_fp = flight_tn = 0
    category = {
        "motor": {"events": 0, "detected": 0},
        "sensor": {"events": 0, "detected": 0},
    }
    flight_rows: list[dict] = []

    for source_id, group in frame.groupby("source_id", sort=False):
        group = group.sort_values("t_rel_s")
        times = group["t_rel_s"].to_numpy(dtype=float)
        label = str(group["label"].iloc[0])
        truth = _truth_mask(source_id, times, label, intervals)
        onsets = policy.apply(group[score_name].to_numpy(dtype=float))
        false_count = int((onsets & ~truth).sum())
        false_alarms += false_count

        if len(times) > 1:
            differences = np.diff(times)
            exposure = (~truth[:-1]) & (differences <= 2.0)
            normal_hours += float(differences[exposure].sum() / 3600.0)

        if label == "normal":
            predicted = bool(onsets.any())
            flight_fp += int(predicted)
            flight_tn += int(not predicted)
            hit = False
            delay = None
        else:
            events += 1
            inside = np.flatnonzero(onsets & truth)
            hit = bool(len(inside))
            detected += int(hit)
            if hit:
                onset = intervals[source_id][0]
                delay = float(times[inside[0]] - onset)
                delays.append(delay)
            else:
                delay = None
            flight_tp += int(hit)
            flight_fn += int(not hit)
            key = "motor" if label == "motor_fault" else "sensor"
            category[key]["events"] += 1
            category[key]["detected"] += int(hit)

        flight_rows.append(
            {
                "source_id": source_id,
                "label": label,
                "event_detected": hit,
                "first_alarm_delay_s": delay,
                "false_alarm_events": false_count,
            }
        )

    metrics = {
        "n_events": events,
        "detected_events": detected,
        "event_onset_recall": detected / events if events else np.nan,
        "false_alarm_events": false_alarms,
        "normal_hours": normal_hours,
        "false_alarms_per_hour": (
            false_alarms / normal_hours if normal_hours else np.nan
        ),
        "median_detection_delay_s": (
            float(np.median(delays)) if delays else np.nan
        ),
        "p95_detection_delay_s": (
            float(np.quantile(delays, 0.95)) if delays else np.nan
        ),
    }
    confusion = {
        "tp": flight_tp,
        "fn": flight_fn,
        "fp": flight_fp,
        "tn": flight_tn,
    }
    categories = [
        {
            "fault_group": name,
            "n_events": values["events"],
            "detected_events": values["detected"],
            "event_onset_recall": (
                values["detected"] / values["events"]
                if values["events"]
                else np.nan
            ),
        }
        for name, values in category.items()
    ]
    return metrics, confusion, [{**row} for row in categories], flight_rows


def window_diagnostics(
    meta: pd.DataFrame,
    scores: np.ndarray,
    labels: dict[str, str],
    intervals: dict[str, tuple[float, float]],
) -> dict:
    truth = np.zeros(len(meta), dtype=bool)
    for index, row in enumerate(meta.itertuples(index=False)):
        source_id = str(row.flight_id)
        if labels[source_id] != "normal":
            onset, end = intervals[source_id]
            truth[index] = onset <= float(row.t_end) <= end
    finite = np.isfinite(scores)
    usable_truth = truth[finite]
    usable_scores = np.asarray(scores, dtype=float)[finite]
    positives = int(usable_truth.sum())
    negatives = int(len(usable_truth) - positives)
    return {
        "n_windows": int(len(scores)),
        "n_positive_windows": positives,
        "n_negative_windows": negatives,
        "auroc": (
            float(roc_auc_score(usable_truth, usable_scores))
            if positives and negatives
            else np.nan
        ),
        "auprc": (
            float(average_precision_score(usable_truth, usable_scores))
            if positives and negatives
            else np.nan
        ),
        "positive_prevalence": (
            positives / len(usable_truth) if len(usable_truth) else np.nan
        ),
    }


def magnitude_diagnostics(
    trained_scores: np.ndarray,
    untrained_scores: np.ndarray,
    x: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.8,
) -> dict:
    denominator = mask.sum(axis=(1, 2)).clip(min=1.0)
    magnitude = (np.square(x) * mask).sum(axis=(1, 2)) / denominator
    rho_random = float(
        spearmanr(trained_scores, untrained_scores, nan_policy="omit").statistic
    )
    rho_magnitude = float(
        spearmanr(trained_scores, magnitude, nan_policy="omit").statistic
    )
    return {
        "rho_trained_vs_untrained": rho_random,
        "rho_trained_vs_magnitude": rho_magnitude,
        "rho_threshold": threshold,
        "magnitude_domination_flagged": bool(
            rho_random >= threshold or rho_magnitude >= threshold
        ),
    }
