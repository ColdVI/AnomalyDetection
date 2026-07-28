"""Create reporting-only figures for the accepted RflyMAD v3.1 development run."""

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


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728"
OUT_DIR = ROOT / "artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    if report.get("candidate_namespace") != "four_dataset_probabilistic_v31":
        raise ValueError("Unexpected run namespace")
    if report.get("magnitude_domination_flagged_at_0_8") is not False:
        raise ValueError("Magnitude gate did not pass")
    for name, record in report["artifact_hashes"].items():
        path = run_dir / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    history = pd.read_csv(run_dir / "training_history.csv")
    flights = pd.read_csv(run_dir / "flight_metrics.csv")
    return report, history, flights


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_plots(run_dir: Path, out_dir: Path) -> list[Path]:
    report, history, flights = _load(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = report["selected_checkpoint"]
    threshold = float(report["evaluation"]["threshold"])
    outputs: list[Path] = []

    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.plot(history["epoch"], history["mean_train_gaussian_nll"], lw=2, label="train")
    axis.plot(history["epoch"], history["mean_validation_gaussian_nll"], lw=2, label="normal validation")
    axis.scatter([selected["epoch"]], [selected["mean_validation_gaussian_nll"]], s=70, color="#D62828", zorder=3, label="selected checkpoint")
    axis.set(title="RflyMAD v3.1 — group-safe training", xlabel="Epoch", ylabel="Gaussian NLL")
    axis.grid(alpha=0.22)
    axis.legend()
    path = out_dir / "01_training_and_selected_checkpoint.png"
    _save(fig, path); outputs.append(path)

    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    normal = flights.loc[flights["is_anomaly"].eq(0), "max_score"].clip(lower=1e-6)
    anomaly = flights.loc[flights["is_anomaly"].eq(1), "max_score"].clip(lower=1e-6)
    low = float(min(normal.min(), anomaly.min(), threshold))
    high = float(max(normal.max(), anomaly.max(), threshold))
    bins = np.geomspace(max(1e-6, low), high, 36)
    axis.hist(normal, bins=bins, alpha=0.65, label=f"normal test (n={len(normal)})", color="#2A9D8F")
    axis.hist(anomaly, bins=bins, alpha=0.55, label=f"anomaly-dev (n={len(anomaly)})", color="#E76F51")
    axis.axvline(threshold, color="#111111", ls="--", lw=2, label=f"val q=0.995 threshold ({threshold:.2f})")
    axis.set_xscale("log")
    axis.set(title="Flight maximum-score distributions", xlabel="Maximum window score (log scale)", ylabel="Flights")
    axis.grid(alpha=0.18)
    axis.legend()
    path = out_dir / "02_flight_max_score_distribution.png"
    _save(fig, path); outputs.append(path)

    truth = flights["is_anomaly"].astype(bool).to_numpy()
    prediction = flights["detected"].astype(bool).to_numpy()
    matrix = np.array([
        [int((~truth & ~prediction).sum()), int((~truth & prediction).sum())],
        [int((truth & ~prediction).sum()), int((truth & prediction).sum())],
    ])
    fig, axis = plt.subplots(figsize=(6.8, 5.8))
    image = axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=17, weight="bold")
    axis.set_xticks([0, 1], ["No flight flag", "Any-window flag"])
    axis.set_yticks([0, 1], ["Normal test", "Anomaly-dev"])
    axis.set_title("Flight-level flag matrix\nDevelopment threshold; not an event confusion matrix")
    fig.colorbar(image, ax=axis, shrink=0.8)
    path = out_dir / "03_flight_flag_matrix.png"
    _save(fig, path); outputs.append(path)

    by_label = flights.groupby("label", sort=True).agg(
        flights=("source_id", "size"),
        detection_rate=("detected", "mean"),
        median_max_score=("max_score", "median"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    colors = ["#2A9D8F" if label == "NoFault" else "#E76F51" for label in by_label["label"]]
    axes[0].bar(by_label["label"], by_label["detection_rate"], color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set(title="Any-window flight flag rate", ylabel="Fraction of flights")
    axes[1].bar(by_label["label"], by_label["median_max_score"], color=colors)
    axes[1].axhline(threshold, color="#111111", ls="--", lw=1.5, label="validation threshold")
    axes[1].set_yscale("log")
    axes[1].set(title="Median flight maximum score", ylabel="Score (log scale)")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("RflyMAD v3.1 — label-level development diagnostics", fontsize=15, weight="bold")
    fig.tight_layout()
    path = out_dir / "04_label_level_diagnostics.png"
    _save(fig, path); outputs.append(path)

    manifest = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "source_run": run_dir.relative_to(ROOT).as_posix(),
        "training_report_sha256": _sha256(run_dir / "training_report.json"),
        "threshold_reselected": False,
        "final_fault_test_accessed": False,
        "plots": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in outputs
        ],
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    outputs = make_plots(args.run_dir.resolve(), args.out_dir.resolve())
    print(json.dumps([path.relative_to(ROOT).as_posix() for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
