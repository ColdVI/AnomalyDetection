"""Evaluate the frozen causal RflyMAD v3.1 B0 event-policy grid.

Calibration uses only ``val_normal``. ``normal_test`` and ``anomaly_dev`` are
report-only, and the sealed final-fault role is neither resolved nor opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID_PATH = Path("configs/rflymad_probabilistic_v31_event_grid.json")
PREREG_PATH = Path("docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_B0_EVENT_PREREG_20260728.md")
DEFAULT_INPUT = Path("artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval")
EXPECTED_NAMESPACE = "rflymad_probabilistic_v31_b0_event_v1"


class EventEvalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float
    alarm_s: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EventEvalError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EventEvalError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _intervals(
    times: np.ndarray,
    active: np.ndarray,
    max_gap_s: float,
    minimum_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> list[Interval]:
    times = np.asarray(times, dtype=float)
    active = np.asarray(active, dtype=bool)
    if len(times) != len(active) or np.any(np.diff(times) < 0) or not np.all(np.isfinite(times)):
        raise EventEvalError("Invalid causal time stream")
    positions = np.flatnonzero(active)
    raw: list[tuple[int, int]] = []
    if len(positions):
        start = previous = int(positions[0])
        for position_value in positions[1:]:
            position = int(position_value)
            if position == previous + 1 and times[position] - times[previous] <= max_gap_s:
                previous = position
            else:
                raw.append((start, previous))
                start = previous = position
        raw.append((start, previous))
    qualified: list[Interval] = []
    for start, end in raw:
        if times[end] - times[start] + 1e-12 < minimum_duration_s:
            continue
        alarm_target = times[start] + minimum_duration_s
        alarm_index = min(end, start + int(np.searchsorted(times[start:end + 1], alarm_target, side="left")))
        qualified.append(Interval(float(times[start]), float(times[end]), float(times[alarm_index])))
    merged: list[Interval] = []
    for event in qualified:
        if merged and event.start_s - merged[-1].end_s <= merge_gap_s:
            prior = merged[-1]
            merged[-1] = Interval(prior.start_s, max(prior.end_s, event.end_s), prior.alarm_s)
        else:
            merged.append(event)
    return merged


def _refractory(events: list[Interval], seconds: float) -> list[Interval]:
    kept: list[Interval] = []
    next_allowed = -math.inf
    for event in events:
        if event.alarm_s + 1e-12 < next_allowed:
            continue
        kept.append(event)
        next_allowed = event.alarm_s + seconds
    return kept


def _exposure(times: np.ndarray, max_gap_s: float) -> float:
    delta = np.diff(np.asarray(times, dtype=float))
    return float(delta[(delta >= 0.0) & (delta <= max_gap_s)].sum())


def _persistence_events(times: np.ndarray, z: np.ndarray, threshold: float, duration: float, common: dict[str, Any], refractory: float) -> list[Interval]:
    return _refractory(_intervals(times, z >= threshold, float(common["max_gap_seconds"]), duration, float(common["merge_gap_seconds"])), refractory)


def _kofn_events(times: np.ndarray, z: np.ndarray, threshold: float, horizon: float, minimum: int, common: dict[str, Any], refractory: float) -> list[Interval]:
    active = np.zeros(len(times), dtype=bool)
    recent: deque[float] = deque()
    previous = -math.inf
    max_gap = float(common["max_gap_seconds"])
    for index, (time_s, score) in enumerate(zip(times, z)):
        if time_s - previous > max_gap:
            recent.clear()
        previous = float(time_s)
        while recent and recent[0] < time_s - horizon:
            recent.popleft()
        if score >= threshold:
            recent.append(float(time_s))
        active[index] = len(recent) >= minimum
    return _refractory(_intervals(times, active, max_gap, 0.0, float(common["merge_gap_seconds"])), refractory)


def _cusum_trace(times: np.ndarray, z: np.ndarray, allowance: float, max_gap_s: float) -> np.ndarray:
    trace = np.zeros(len(times), dtype=float)
    state = 0.0
    previous = -math.inf
    for index, (time_s, score) in enumerate(zip(times, z)):
        if time_s - previous > max_gap_s:
            state = 0.0
        previous = float(time_s)
        state = max(0.0, state + float(score) - allowance)
        trace[index] = state
    return trace


def _cusum_events(times: np.ndarray, z: np.ndarray, allowance: float, h: float, common: dict[str, Any], refractory: float) -> list[Interval]:
    events: list[Interval] = []
    state = 0.0
    previous = -math.inf
    next_allowed = -math.inf
    max_gap = float(common["max_gap_seconds"])
    for time_s, score in zip(times, z):
        if time_s - previous > max_gap:
            state = 0.0
        previous = float(time_s)
        if time_s < next_allowed:
            continue
        state = max(0.0, state + float(score) - allowance)
        if state >= h:
            events.append(Interval(float(time_s), float(time_s), float(time_s)))
            state = 0.0
            next_allowed = float(time_s) + refractory
    return events


def _validation_rate(validation: pd.DataFrame, eventizer: Callable[[np.ndarray, np.ndarray], list[Interval]], max_gap: float) -> tuple[float | None, int, float]:
    count = 0
    exposure_s = 0.0
    for _, frame in validation.groupby("source_id", sort=True):
        ordered = frame.sort_values("t_rel_s", kind="mergesort")
        times = ordered["t_rel_s"].to_numpy(float)
        count += len(eventizer(times, ordered["standardized_nll"].to_numpy(float)))
        exposure_s += _exposure(times, max_gap)
    hours = exposure_s / 3600.0
    return ((count / hours) if hours else None), count, hours


def _candidate_points(validation: pd.DataFrame, grid: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    common = grid["common"]
    z = validation["standardized_nll"].to_numpy(float)
    quantiles = list(map(float, common["threshold_candidate_quantiles"]))
    thresholds = [(q, float(np.quantile(z, q))) for q in quantiles]
    candidates: list[dict[str, Any]] = []
    for refractory in map(float, common["refractory_seconds"]):
        for duration in map(float, grid["families"]["threshold_persistence"]["minimum_duration_seconds"]):
            for quantile, threshold in thresholds:
                func = lambda t, s, th=threshold, d=duration, r=refractory: _persistence_events(t, s, th, d, common, r)
                rate, count, hours = _validation_rate(validation, func, float(common["max_gap_seconds"]))
                candidates.append({"family": "threshold_persistence", "policy": f"duration_{duration:g}s", "refractory_s": refractory, "candidate_quantile": quantile, "threshold": threshold, "allowance": None, "validation_false_events": count, "validation_exposure_hours": hours, "validation_false_events_per_hour": rate})
        for policy in grid["families"]["k_of_n_time"]["policies"]:
            for quantile, threshold in thresholds:
                func = lambda t, s, th=threshold, p=policy, r=refractory: _kofn_events(t, s, th, float(p["horizon_seconds"]), int(p["minimum_exceedances"]), common, r)
                rate, count, hours = _validation_rate(validation, func, float(common["max_gap_seconds"]))
                candidates.append({"family": "k_of_n_time", "policy": str(policy["name"]), "refractory_s": refractory, "candidate_quantile": quantile, "threshold": threshold, "allowance": None, "validation_false_events": count, "validation_exposure_hours": hours, "validation_false_events_per_hour": rate})
        for allowance in map(float, grid["families"]["standardized_nll_cusum"]["allowance_candidates"]):
            traces = []
            for _, frame in validation.groupby("source_id", sort=True):
                ordered = frame.sort_values("t_rel_s", kind="mergesort")
                traces.append(_cusum_trace(ordered["t_rel_s"].to_numpy(float), ordered["standardized_nll"].to_numpy(float), allowance, float(common["max_gap_seconds"])))
            pooled = np.concatenate(traces)
            for quantile in map(float, grid["families"]["standardized_nll_cusum"]["h_candidate_quantiles"]):
                h = float(np.quantile(pooled, quantile))
                func = lambda t, s, a=allowance, level=h, r=refractory: _cusum_events(t, s, a, level, common, r)
                rate, count, hours = _validation_rate(validation, func, float(common["max_gap_seconds"]))
                candidates.append({"family": "standardized_nll_cusum", "policy": f"allowance_{allowance:g}", "refractory_s": refractory, "candidate_quantile": quantile, "threshold": h, "allowance": allowance, "validation_false_events": count, "validation_exposure_hours": hours, "validation_false_events_per_hour": rate})
    candidate_frame = pd.DataFrame(candidates)
    selected: list[dict[str, Any]] = []
    keys = ["family", "policy", "refractory_s"]
    for values, frame in candidate_frame.groupby(keys, sort=True):
        for budget in map(float, common["false_event_budgets_per_normal_hour"]):
            eligible = frame[frame["validation_false_events_per_hour"] <= budget]
            choice = eligible.sort_values(["threshold", "candidate_quantile"], ascending=[True, False]).iloc[0] if len(eligible) else None
            row = dict(zip(keys, values))
            row["budget_false_events_per_hour"] = budget
            row["available"] = choice is not None
            for column in ("candidate_quantile", "threshold", "allowance", "validation_false_events", "validation_exposure_hours", "validation_false_events_per_hour"):
                row[column] = None if choice is None else choice[column]
            selected.append(row)
    return candidate_frame, selected


def _eventizer(point: dict[str, Any], grid: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], list[Interval]]:
    common = grid["common"]
    threshold = float(point["threshold"])
    refractory = float(point["refractory_s"])
    if point["family"] == "threshold_persistence":
        duration = float(str(point["policy"]).split("_")[1][:-1])
        return lambda t, z: _persistence_events(t, z, threshold, duration, common, refractory)
    if point["family"] == "k_of_n_time":
        policy = next(item for item in grid["families"]["k_of_n_time"]["policies"] if item["name"] == point["policy"])
        return lambda t, z: _kofn_events(t, z, threshold, float(policy["horizon_seconds"]), int(policy["minimum_exceedances"]), common, refractory)
    allowance = float(point["allowance"])
    return lambda t, z: _cusum_events(t, z, allowance, threshold, common, refractory)


def _overlap(predicted: list[Interval], truth: list[Interval]) -> float:
    total = 0.0
    for alarm in predicted:
        for event in truth:
            total += max(0.0, min(alarm.end_s, event.end_s) - max(alarm.alarm_s, event.start_s))
    return total


def _source_records(frame: pd.DataFrame, eventizer: Callable[[np.ndarray, np.ndarray], list[Interval]], max_gap: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_id, source in frame.groupby("source_id", sort=True):
        source = source.sort_values("t_rel_s", kind="mergesort")
        times = source["t_rel_s"].to_numpy(float)
        predicted = eventizer(times, source["standardized_nll"].to_numpy(float))
        available = source["truth_available"].to_numpy(bool)
        truth = _intervals(times, source["truth_active"].to_numpy(bool) & available, max_gap)
        delays = []
        detected = 0
        for event in truth:
            alarms = [alarm.alarm_s for alarm in predicted if event.start_s <= alarm.alarm_s <= event.end_s]
            if alarms:
                detected += 1
                delays.append(min(alarms) - event.start_s)
        predicted_duration = sum(max(0.0, item.end_s - item.alarm_s) for item in predicted)
        truth_duration = sum(max(0.0, item.end_s - item.start_s) for item in truth)
        row0 = source.iloc[0]
        records.append({
            "source_id": str(source_id), "group_id": str(row0["group_id"]), "domain": str(row0["domain"]),
            "fault_family": str(row0["fault_family"]), "is_anomaly_flight": bool(row0["is_anomaly_flight"]),
            "exposure_s": _exposure(times, max_gap), "predicted_events": len(predicted),
            "event_count": len(truth), "detected_events": detected, "detection_delays_s": delays,
            "predicted_duration_s": predicted_duration, "truth_duration_s": truth_duration,
            "overlap_duration_s": _overlap(predicted, truth), "truth_available_fraction": float(available.mean()),
        })
    return records


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    normal = [row for row in records if not row["is_anomaly_flight"]]
    anomaly = [row for row in records if row["is_anomaly_flight"]]
    normal_hours = sum(row["exposure_s"] for row in normal) / 3600.0
    false_events = sum(row["predicted_events"] for row in normal)
    event_count = sum(row["event_count"] for row in anomaly)
    detected = sum(row["detected_events"] for row in anomaly)
    delays = [delay for row in anomaly for delay in row["detection_delays_s"]]
    pred_duration = sum(row["predicted_duration_s"] for row in records)
    truth_duration = sum(row["truth_duration_s"] for row in anomaly)
    overlap = sum(row["overlap_duration_s"] for row in anomaly)
    exposure = sum(row["exposure_s"] for row in records)
    return {
        "normal_sources": len(normal), "anomaly_sources": len(anomaly), "normal_exposure_hours": normal_hours,
        "false_events": false_events, "false_events_per_normal_hour": false_events / normal_hours if normal_hours else None,
        "false_events_per_normal_flight": false_events / len(normal) if normal else None,
        "normal_flights_with_any_false_event_fraction": sum(row["predicted_events"] > 0 for row in normal) / len(normal) if normal else None,
        "event_count": event_count, "detected_events": detected, "event_recall": detected / event_count if event_count else None,
        "median_detection_delay_s": float(np.median(delays)) if delays else None,
        "p90_detection_delay_s": float(np.quantile(delays, 0.9)) if delays else None,
        "range_precision_time_weighted": overlap / pred_duration if pred_duration else None,
        "range_recall_time_weighted": overlap / truth_duration if truth_duration else None,
        "alarm_time_fraction": pred_duration / exposure if exposure else None,
    }


def _bootstrap(records: list[dict[str, Any]], config: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["seed"]) + seed_offset)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = ("anomaly" if row["is_anomaly_flight"] else "normal", row["domain"], row["fault_family"] if row["is_anomaly_flight"] else "normal")
        strata.setdefault(key, []).append(row)
    names = ("false_events_per_normal_hour", "event_recall", "median_detection_delay_s", "range_precision_time_weighted", "range_recall_time_weighted")
    values = {name: [] for name in names}
    for _ in range(int(config["replicates"])):
        sample: list[dict[str, Any]] = []
        for rows in strata.values():
            sample.extend(rows[index] for index in rng.integers(0, len(rows), len(rows)))
        aggregate = _aggregate(sample)
        for name in names:
            value = aggregate[name]
            if value is not None and math.isfinite(value):
                values[name].append(value)
    tail = (1.0 - float(config["confidence_level"])) / 2.0
    return {name: ([float(x) for x in np.quantile(data, [tail, 1.0 - tail])] if data else None) for name, data in values.items()}


def evaluate(root: Path, output_dir: Path) -> dict[str, Any]:
    grid = _read_json(root / GRID_PATH)
    manifest = _read_json(output_dir / "score_export_manifest.json")
    ledger_path = output_dir / "score_streams.parquet"
    checks = {
        "namespace": grid.get("candidate_namespace") == EXPECTED_NAMESPACE == manifest.get("candidate_namespace"),
        "grid_hash": manifest["artifacts"]["event_grid"]["sha256"] == _sha256(root / GRID_PATH),
        "prereg_hash": manifest["artifacts"]["preregistration"]["sha256"] == _sha256(root / PREREG_PATH),
        "ledger_hash": manifest["artifacts"]["score_streams"]["sha256"] == _sha256(ledger_path),
        "final_not_accessed": manifest.get("final_fault_test_accessed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise EventEvalError(f"CONTRACT ERROR: {failed}")
    ledger = pd.read_parquet(ledger_path)
    if set(ledger["role"].astype(str).unique()) != {"val_normal", "normal_test", "anomaly_dev"}:
        raise EventEvalError("Unexpected ledger role set")
    validation = ledger[ledger["role"] == "val_normal"].copy()
    normal_test = ledger[ledger["role"] == "normal_test"].copy()
    anomaly_dev = ledger[ledger["role"] == "anomaly_dev"].copy()
    if validation["is_anomaly_flight"].any() or normal_test["is_anomaly_flight"].any() or not anomaly_dev["is_anomaly_flight"].all():
        raise EventEvalError("Role/label contract violation")

    candidates, selected = _candidate_points(validation, grid)
    candidates.to_csv(output_dir / "calibration_candidates.csv", index=False)
    pd.DataFrame(selected).to_csv(output_dir / "operating_grid.csv", index=False)
    metric_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    source_output: list[dict[str, Any]] = []
    max_gap = float(grid["common"]["max_gap_seconds"])
    for index, point in enumerate(selected):
        point_id = f"P{index + 1:03d}"
        if not point["available"]:
            metric_rows.append({"point_id": point_id, **point})
            continue
        eventizer = _eventizer(point, grid)
        records = _source_records(pd.concat([normal_test, anomaly_dev], ignore_index=True), eventizer, max_gap)
        aggregate = _aggregate(records)
        metric_rows.append({"point_id": point_id, **point, **aggregate})
        bootstrap[point_id] = _bootstrap(records, grid["bootstrap"], index * 1009)
        for row in records:
            source_output.append({"point_id": point_id, **{k: v for k, v in row.items() if k != "detection_delays_s"}, "detection_delays_s": json.dumps(row["detection_delays_s"])})
        for dimension in ("domain", "fault_family", "group_id"):
            values = sorted(set(row[dimension] for row in records))
            for value in values:
                subset = [row for row in records if row[dimension] == value]
                breakdown_rows.append({"point_id": point_id, "dimension": dimension, "value": value, **_aggregate(subset)})
        print(f"evaluated {point_id}/{len(selected)} {point['family']} {point['policy']}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "event_metrics.csv", index=False)
    pd.DataFrame(breakdown_rows).to_csv(output_dir / "domain_family_breakdown.csv", index=False)
    pd.DataFrame(source_output).to_parquet(output_dir / "source_event_metrics.parquet", index=False)
    _write_json(output_dir / "bootstrap_intervals.json", bootstrap)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    available = metrics[metrics["available"] == True].copy()  # noqa: E712
    fig, ax = plt.subplots(figsize=(10, 6))
    for family, frame in available.groupby("family"):
        ax.scatter(frame["false_events_per_normal_hour"], frame["event_recall"], label=family, alpha=0.75)
    ax.set(xlabel="Normal-test false events / hour", ylabel="Anomaly-dev real-event recall", title="RflyMAD v3.1 B0 frozen event grid")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / "01_recall_vs_false_event_rate.png", dpi=180); plt.close(fig)
    reference = available[(available["budget_false_events_per_hour"] == 1.0) & (available["refractory_s"] == 30.0)].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = reference["family"].str.replace("standardized_nll_", "", regex=False) + "\n" + reference["policy"]
    ax.bar(np.arange(len(reference)), reference["event_recall"], color="#2878B5")
    ax.set_xticks(np.arange(len(reference)), labels, rotation=35, ha="right"); ax.set(ylabel="Real-event recall", title="Frozen 1 event/h budget, 30 s refractory")
    ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(plots / "02_reference_budget_event_recall.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(reference)), reference["normal_flights_with_any_false_event_fraction"], color="#E07B39")
    ax.set_xticks(np.arange(len(reference)), labels, rotation=35, ha="right"); ax.set(ylabel="Normal flights with >=1 false event", title="Independent normal-test burden")
    ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(plots / "03_normal_test_alarm_burden.png", dpi=180); plt.close(fig)
    report = {
        "schema_version": 1, "candidate_namespace": EXPECTED_NAMESPACE, "status": "complete",
        "base_model_changed": False, "calibration_role": "val_normal", "test_selection_performed": False,
        "final_fault_test_accessed": False, "contract_checks": checks,
        "ledger": {"rows": len(ledger), "role_sources": {role: int(frame["source_id"].nunique()) for role, frame in ledger.groupby("role")}},
        "grid_points": len(selected), "available_grid_points": int(sum(bool(row["available"]) for row in selected)),
        "interpretation": "All frozen points are reported; no test winner is selected. CUSUM alarms are point events: range precision is undefined and range recall is zero under this representation, so neither is comparable to interval-producing families.",
        "artifacts": {name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size} for name in ("operating_grid.csv", "event_metrics.csv", "domain_family_breakdown.csv", "bootstrap_intervals.json")},
    }
    _write_json(output_dir / "evaluation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        report = evaluate(root, output.resolve())
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return 0
    except (EventEvalError, KeyError, ValueError, OSError) as exc:
        print(f"CONTRACT ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
