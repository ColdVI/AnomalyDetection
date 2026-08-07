"""Create the frozen four-dataset v2 professional result figures.

The script is reporting-only: it reads accepted checkpoints' histories and the
event-evaluation score ledgers.  It never trains, changes a threshold, or picks
an epoch.  The displayed reference point is the preregistered 1.0 s persistence
and 1 false-event/hour validation budget cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from four_dataset_probabilistic_event_eval_v1 import _eventize_source
except ModuleNotFoundError:  # pragma: no cover
    from scripts.four_dataset_probabilistic_event_eval_v1 import _eventize_source


DATASETS = ("alfa", "rflymad", "uav_sead", "uav_attack")
DISPLAY = {
    "alfa": "ALFA",
    "rflymad": "RflyMAD",
    "uav_sead": "UAV-SEAD",
    "uav_attack": "UAV Attack",
}
COLORS = {
    "alfa": "#2A9D8F",
    "rflymad": "#264653",
    "uav_sead": "#F4A261",
    "uav_attack": "#E76F51",
}
REFERENCE_PERSISTENCE_S = 1.0
REFERENCE_BUDGET_PER_HOUR = 1.0
MAX_GAP_S = 5.0
MERGE_GAP_S = 2.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference(report: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in report["selected_budget_points"]
        if float(row["minimum_above_threshold_seconds"]) == REFERENCE_PERSISTENCE_S
        and float(row["budget_false_events_per_hour"]) == REFERENCE_BUDGET_PER_HOUR
    ]
    if len(matches) != 1 or not matches[0]["available"]:
        raise ValueError(f"Reference point unavailable for {report['dataset']}")
    return matches[0]


def _prediction_mask(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    ordered = frame.sort_values("target_time_s", kind="mergesort")
    times = ordered["target_time_s"].to_numpy(float)
    predicted = np.zeros(len(ordered), dtype=bool)
    for interval in _eventize_source(
        ordered,
        threshold,
        REFERENCE_PERSISTENCE_S,
        MERGE_GAP_S,
        MAX_GAP_S,
    ):
        predicted |= (times >= interval.alarm_s) & (times <= interval.end_s)
    result = pd.Series(predicted, index=ordered.index).reindex(frame.index)
    return result.to_numpy(bool)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _training_plot(run_root: Path, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for axis, dataset in zip(axes.flat, DATASETS, strict=True):
        history = pd.read_csv(run_root / f"{dataset}_gpu_v2" / "training_history.csv")
        axis.plot(history["epoch"], history["mean_train_gaussian_nll"], label="train", lw=2)
        axis.plot(
            history["epoch"],
            history["mean_validation_gaussian_nll"],
            label="normal validation",
            lw=2,
        )
        best = history.loc[history["mean_validation_gaussian_nll"].idxmin()]
        axis.scatter([best["epoch"]], [best["mean_validation_gaussian_nll"]], s=45, zorder=3)
        elapsed = float(history["elapsed_seconds"].sum())
        windows = int(history["windows"].sum())
        axis.set_title(
            f"{DISPLAY[dataset]} | frozen epoch 30\n"
            f"GPU epoch time sum {elapsed:.1f}s; {windows/1e6:.2f}M window-epochs"
        )
        axis.set_ylabel("masked Gaussian NLL")
        axis.grid(alpha=0.22)
        axis.text(
            0.02,
            0.04,
            f"lowest observed val: epoch {int(best['epoch'])} (descriptive only)",
            transform=axis.transAxes,
            fontsize=8,
            color="#444444",
        )
    axes[0, 0].legend(loc="best")
    for axis in axes[-1, :]:
        axis.set_xlabel("epoch")
    fig.suptitle(
        "Four-dataset probabilistic v2 — training diagnostics\n"
        "Epoch/checkpoint was frozen in advance; curves are not used for reselection",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir / "01_training_diagnostics.png")


def _summary_plot(reports: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(DATASETS))
    labels = [DISPLAY[value] for value in DATASETS]
    colors = [COLORS[value] for value in DATASETS]

    axis = axes[0, 0]
    auc = [reports[d]["interval_truth_window_metrics"]["roc_auc"] for d in DATASETS]
    plotted = [np.nan if value is None else value for value in auc]
    bars = axis.bar(x, plotted, color=colors)
    axis.axhline(0.5, color="#555555", ls="--", lw=1, label="random ranking")
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("ROC-AUC")
    axis.set_title("A. Interval-truth window ranking")
    for bar, value in zip(bars, auc, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            0.03 if value is None else value + 0.025,
            "N/A" if value is None else f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    axis.legend(loc="upper left", fontsize=8)

    axis = axes[0, 1]
    width = 0.35
    maximum = [reports[d]["flight_metrics"]["max"]["roc_auc"] for d in DATASETS]
    mean = [reports[d]["flight_metrics"]["mean"]["roc_auc"] for d in DATASETS]
    axis.bar(x - width / 2, maximum, width, label="max score", color="#8ECAE6")
    axis.bar(x + width / 2, mean, width, label="mean score", color="#023047")
    axis.axhline(0.5, color="#777777", ls="--", lw=1)
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("flight ROC-AUC")
    axis.set_title("B. Exploratory flight triage (not model selection)")
    axis.legend(fontsize=8)

    reference = {dataset: _reference(report) for dataset, report in reports.items()}
    axis = axes[1, 0]
    recall = []
    for dataset in DATASETS:
        evaluation = reference[dataset]["evaluation"]
        recall.append(evaluation["event_recall"] if evaluation else None)
    bars = axis.bar(x, [np.nan if v is None else v for v in recall], color=colors)
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("event recall")
    axis.set_title("C. Event localization — 1 s / 1 event·h⁻¹ reference")
    for bar, value in zip(bars, recall, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            0.03 if value is None else value + 0.025,
            "N/A" if value is None else f"{100*value:.1f}%",
            ha="center",
            fontsize=9,
        )

    axis = axes[1, 1]
    false_rate = [
        reference[d]["evaluation"]["false_events_per_normal_hour"] for d in DATASETS
    ]
    bars = axis.bar(x, false_rate, color=colors)
    axis.axhline(1.0, color="#AA0000", ls="--", lw=1.4, label="reference budget")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("test false events / normal hour")
    axis.set_title("D. Normal-flight operational burden")
    axis.legend(fontsize=8)
    for bar, value in zip(bars, false_rate, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.03, value * 1.12),
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Four-dataset probabilistic v2 — frozen-checkpoint result summary",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir / "02_detection_summary.png")


def _operating_grid_plot(reports: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    datasets = ("alfa", "rflymad", "uav_sead")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    image = None
    for axis, dataset in zip(axes, datasets, strict=True):
        report = reports[dataset]
        rows = report["selected_budget_points"]
        persistence = sorted({float(row["minimum_above_threshold_seconds"]) for row in rows})
        budgets = sorted({float(row["budget_false_events_per_hour"]) for row in rows})
        matrix = np.full((len(persistence), len(budgets)), np.nan)
        for row in rows:
            evaluation = row["evaluation"]
            if evaluation and evaluation["event_recall"] is not None:
                i = persistence.index(float(row["minimum_above_threshold_seconds"]))
                j = budgets.index(float(row["budget_false_events_per_hour"]))
                matrix[i, j] = float(evaluation["event_recall"])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(persistence)):
            for j in range(len(budgets)):
                text = "—" if np.isnan(matrix[i, j]) else f"{100*matrix[i, j]:.0f}%"
                axis.text(j, i, text, ha="center", va="center", color="white" if matrix[i, j] < 0.55 else "black", fontsize=8)
        eligible = "claim-eligible" if report["claim_eligible_normal_validation_count"] else "insufficient val flights"
        axis.set_title(f"{DISPLAY[dataset]}\n{eligible}")
        axis.set_xticks(range(len(budgets)), [f"{value:g}" for value in budgets])
        axis.set_yticks(range(len(persistence)), [f"{value:g}" for value in persistence])
        axis.set_xlabel("validation budget (false events/hour)")
        axis.set_ylabel("persistence (seconds)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="test event recall")
    fig.suptitle(
        "Frozen operating grid — every cell reported, no test-set retuning",
        fontsize=15,
        weight="bold",
    )
    return _save(fig, out_dir / "03_event_operating_grid.png")


def _confusion_plot(
    reports: dict[str, dict[str, Any]], ledger_root: Path, out_dir: Path
) -> Path:
    datasets = ("alfa", "rflymad", "uav_sead")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    image = None
    for axis, dataset in zip(axes, datasets, strict=True):
        ledger = pd.read_parquet(ledger_root / dataset / "test_window_scores.parquet")
        threshold = float(_reference(reports[dataset])["threshold"])
        predicted = np.zeros(len(ledger), dtype=bool)
        for _, indexes in ledger.groupby("source_id", sort=False).groups.items():
            positions = np.asarray(indexes, dtype=np.int64)
            predicted[positions] = _prediction_mask(ledger.loc[positions], threshold)
        eligible = ledger["truth_available"].to_numpy(bool)
        truth = ledger["truth_active"].to_numpy(bool)[eligible]
        pred = predicted[eligible]
        counts = np.array(
            [
                [np.sum(~truth & ~pred), np.sum(~truth & pred)],
                [np.sum(truth & ~pred), np.sum(truth & pred)],
            ],
            dtype=float,
        )
        row_total = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, row_total, out=np.zeros_like(counts), where=row_total > 0)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    f"{int(counts[i, j]):,}\n{100*normalized[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.55 else "black",
                )
        axis.set_xticks([0, 1], ["normal", "alarm"])
        axis.set_yticks([0, 1], ["normal", "anomaly"])
        axis.set_xlabel("predicted state")
        axis.set_ylabel("interval truth")
        axis.set_title(DISPLAY[dataset])
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="row-normalized share")
    fig.suptitle(
        "Reference window-state confusion matrices\n"
        "Causal 1 s persistence; unavailable interval truth excluded",
        fontsize=14,
        weight="bold",
    )
    return _save(fig, out_dir / "04_reference_confusion_matrices.png")


def _mask_spans(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    if not np.any(mask):
        return []
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(float(times[a]), float(times[b])) for a, b in zip(starts, ends, strict=True)]


def _representative_source(ledger: pd.DataFrame, anomaly: bool, require_truth: bool) -> str:
    rows = []
    for source_id, frame in ledger.groupby("source_id", sort=True):
        if bool(frame["is_anomaly_flight"].iloc[0]) != anomaly:
            continue
        if require_truth and not bool(frame["truth_available"].all()):
            continue
        duration = float(frame["target_time_s"].max() - frame["target_time_s"].min())
        rows.append((str(source_id), duration))
    if not rows:
        raise ValueError("No representative source candidate")
    median = float(np.median([duration for _, duration in rows]))
    return min(rows, key=lambda item: (abs(item[1] - median), item[0]))[0]


def _timeline_plot(
    dataset: str,
    report: dict[str, Any],
    ledger_root: Path,
    out_dir: Path,
) -> Path:
    ledger = pd.read_parquet(ledger_root / dataset / "test_window_scores.parquet")
    reference = _reference(report)
    threshold = float(reference["threshold"])
    normal_id = _representative_source(ledger, anomaly=False, require_truth=True)
    anomaly_id = _representative_source(
        ledger, anomaly=True, require_truth=dataset != "uav_attack"
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    for axis, source_id, kind in zip(
        axes, (normal_id, anomaly_id), ("normal flight", "anomaly-labelled flight"), strict=True
    ):
        frame = ledger[ledger["source_id"].astype(str) == source_id].sort_values("target_time_s")
        times = frame["target_time_s"].to_numpy(float)
        scores = frame["score"].to_numpy(float)
        predicted = _prediction_mask(frame, threshold)
        axis.plot(times, scores, color=COLORS[dataset], lw=1.15, label="model anomaly score")
        axis.axhline(threshold, color="#9B2226", ls="--", lw=1.3, label="validation threshold")
        for start, end in _mask_spans(times, predicted):
            axis.axvspan(start, end, color="#E63946", alpha=0.2, label="causal alarm" if start == _mask_spans(times, predicted)[0][0] else None)
        truth_available = frame["truth_available"].to_numpy(bool)
        truth = frame["truth_active"].to_numpy(bool) & truth_available
        truth_spans = _mask_spans(times, truth)
        for index, (start, end) in enumerate(truth_spans):
            axis.axvspan(start, end, color="#2A9D8F", alpha=0.25, label="true anomaly interval" if index == 0 else None)
        axis.set_yscale("symlog", linthresh=0.25)
        axis.grid(alpha=0.2)
        label = str(frame["flight_label"].iloc[0])
        suffix = "interval truth unavailable" if kind.startswith("anomaly") and not truth_available.all() else f"truth windows={int(truth.sum())}, alarm windows={int(predicted.sum())}"
        axis.set_title(f"{kind}: {source_id}\nlabel={label}; {suffix}", fontsize=10)
        axis.set_ylabel("score (symlog)")
        axis.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time since Gold flight start (seconds)")
    fig.suptitle(
        f"{DISPLAY[dataset]} reference timelines — deterministic median-duration selection\n"
        "1 s persistence / 1 false-event·h⁻¹ validation-budget cell",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir / f"05_{dataset}_reference_timelines.png")


def run(ledger_root: Path, run_root: Path, out_dir: Path) -> list[Path]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    reports = {
        dataset: _read_json(ledger_root / dataset / "event_evaluation_report.json")
        for dataset in DATASETS
    }
    outputs = [
        _training_plot(run_root, out_dir),
        _summary_plot(reports, out_dir),
        _operating_grid_plot(reports, out_dir),
        _confusion_plot(reports, ledger_root, out_dir),
    ]
    outputs.extend(
        _timeline_plot(dataset, reports[dataset], ledger_root, out_dir)
        for dataset in DATASETS
    )
    manifest = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_event_eval_v1",
        "reporting_only": True,
        "reference_point": {
            "minimum_above_threshold_seconds": REFERENCE_PERSISTENCE_S,
            "validation_false_event_budget_per_hour": REFERENCE_BUDGET_PER_HOUR,
            "selection_note": "fixed presentation cell; not chosen from test performance",
        },
        "timeline_selection": "closest-to-median test duration within normal/anomaly class; source_id lexical tie-break; score-independent",
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in outputs
        ],
    }
    manifest_path = out_dir / "results_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=Path("artifacts/four_dataset_probabilistic_event_eval_v1/20260728"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "artifacts/four_dataset_probabilistic_event_eval_v1/accepted_runs/20260728T071250Z"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run(args.ledger_root.resolve(), args.run_root.resolve(), args.out_dir.resolve())
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
