"""Create report-ready RflyMAD v3.1 B0 score/truth/alarm timelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from evaluate_rflymad_probabilistic_v31_events import _eventizer, _read_json
except ModuleNotFoundError:  # pragma: no cover
    from scripts.evaluate_rflymad_probabilistic_v31_events import _eventizer, _read_json


GRID = Path("configs/rflymad_probabilistic_v31_event_grid.json")
OUTPUT = Path("artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval")


def _shade(ax: plt.Axes, times: np.ndarray, active: np.ndarray, color: str, label: str) -> None:
    shown = False
    positions = np.flatnonzero(active)
    if not len(positions):
        return
    start = previous = int(positions[0])
    spans: list[tuple[int, int]] = []
    for value in positions[1:]:
        position = int(value)
        if position == previous + 1 and times[position] - times[previous] <= 5.0:
            previous = position
        else:
            spans.append((start, previous)); start = previous = position
    spans.append((start, previous))
    for start, end in spans:
        ax.axvspan(times[start], times[end], color=color, alpha=0.16, label=label if not shown else None)
        shown = True


def _plot(frame: pd.DataFrame, point: dict, grid: dict, path: Path, subtitle: str) -> None:
    frame = frame.sort_values("t_rel_s", kind="mergesort")
    times = frame["t_rel_s"].to_numpy(float)
    z = frame["standardized_nll"].to_numpy(float)
    alarms = _eventizer(point, grid)(times, z)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.0]})
    axes[0].plot(times, z, color="#1F4E79", linewidth=1.0, label="Validation-standardized Gaussian NLL")
    axes[0].axhline(float(point["threshold"]), color="#D1495B", linestyle="--", linewidth=1.2, label="Frozen K-of-N score threshold")
    for alarm_index, alarm in enumerate(alarms):
        axes[0].axvline(alarm.alarm_s, color="#B00020", linewidth=1.4, label="Alarm" if alarm_index == 0 else None)
    _shade(axes[0], times, frame["fault_active"].to_numpy(bool), "#D62728", "Fault active")
    _shade(axes[0], times, frame["condition_active"].to_numpy(bool), "#FFB000", "Condition active")
    axes[0].set_ylabel("Standardized NLL"); axes[0].legend(loc="upper right", ncol=2); axes[0].grid(alpha=0.2)
    axes[1].plot(times, frame["gaussian_nll_score"], color="#2878B5", linewidth=0.9)
    axes[1].set_ylabel("Gaussian NLL"); axes[1].grid(alpha=0.2)
    axes[2].plot(times, frame["target_magnitude"], color="#4C956C", linewidth=0.9)
    axes[2].set(ylabel="Target magnitude", xlabel="Flight-relative time (s)"); axes[2].grid(alpha=0.2)
    row = frame.iloc[0]
    fig.suptitle(f"RflyMAD v3.1 B0 — {subtitle}\n{row['source_id']} | {row['domain']} | {row['fault_family']} | alarms={len(alarms)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    grid = _read_json(root / GRID)
    ledger = pd.read_parquet(output / "score_streams.parquet")
    points = pd.read_csv(output / "operating_grid.csv")
    point = points[(points["family"] == "k_of_n_time") & (points["policy"] == "2_of_3_approx_10hz") & (points["budget_false_events_per_hour"] == 1.0) & (points["refractory_s"] == 30.0)].iloc[0].to_dict()
    eventize = _eventizer(point, grid)
    groups = {source_id: frame for source_id, frame in ledger.groupby("source_id", sort=True)}
    normal_candidates = []
    detected_candidates = []
    missed_candidates = []
    for source_id, frame in groups.items():
        if frame["role"].iloc[0] == "val_normal":
            continue
        times = frame["t_rel_s"].to_numpy(float)
        alarms = eventize(times, frame["standardized_nll"].to_numpy(float))
        truth_times = times[frame["truth_active"].to_numpy(bool)]
        detected = bool(len(truth_times) and any(truth_times.min() <= item.alarm_s <= truth_times.max() for item in alarms))
        if frame["role"].iloc[0] == "normal_test" and alarms:
            normal_candidates.append((source_id, float(frame["standardized_nll"].max())))
        elif detected and str(frame["fault_family"].iloc[0]) == "Motor":
            detected_candidates.append((source_id, abs(alarms[0].alarm_s - float(np.median(truth_times)))))
        elif not detected and str(frame["domain"].iloc[0]) == "Real" and str(frame["fault_family"].iloc[0]) == "Sensor":
            missed_candidates.append((source_id, float(frame["standardized_nll"].max())))
    plot_dir = output / "plots"; plot_dir.mkdir(exist_ok=True)
    chosen = [
        (max(normal_candidates, key=lambda item: item[1])[0], "04_normal_false_event_timeline.png", "Independent normal-test false event"),
        (min(detected_candidates, key=lambda item: item[1])[0], "05_detected_motor_event_timeline.png", "Detected Motor event"),
        (max(missed_candidates, key=lambda item: item[1])[0], "06_missed_real_sensor_timeline.png", "Missed Real/Sensor event"),
    ]
    for source_id, filename, subtitle in chosen:
        _plot(groups[source_id], point, grid, plot_dir / filename, subtitle)
        print(filename, source_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
