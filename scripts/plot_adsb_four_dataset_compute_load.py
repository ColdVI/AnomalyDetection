"""Plot ADS-B versus four-dataset compute load and runtime planning ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["ADS-B", "ALFA", "UAV-Attack", "UAV-SEAD", "RflyMAD"]
TRAIN_ROWS = np.asarray([275_468_643, 13_026, 17_656, 911_940, 234_529], dtype=float)
TRAIN_WINDOWS = np.asarray([236_129_563, 13_026, 17_656, 911_940, 234_529], dtype=float)
EPOCHS = np.asarray([8, 30, 30, 30, 30], dtype=float)
WINDOW_EPOCHS = TRAIN_WINDOWS * EPOCHS
INPUT_GIB = np.asarray([17.6739455, 18_784_062 / 2**30, 14_090_281 / 2**30, 655_990_007 / 2**30, 338_932_957 / 2**30])
TEST_EXPOSURE = [26_802_690, 4_941, 18_087, 331_724, 114_022]
RUNTIME_LABELS = ["ADS-B training\nL4 artifact session", "ADS-B truth-v2\nCPU measured", "ALFA\nL4 epoch loop", "UAV-Attack\nL4 epoch loop", "UAV-SEAD\nL4 epoch loop", "RflyMAD\nL4 epoch loop"]
RUNTIME_LOW_MIN = np.asarray([2539.8835 / 60.0, 7762.0 / 60.0, 6.305738218 / 60.0, 7.776712619 / 60.0, 457.717206897 / 60.0, 189.373313505 / 60.0])
RUNTIME_HIGH_MIN = RUNTIME_LOW_MIN.copy()


def _labels(axis, bars, values, formatter) -> None:
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.16, formatter(value), ha="center", va="bottom", fontsize=8)


def run(out_dir: Path) -> list[Path]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    colors = ["#264653", "#2A9D8F", "#4C956C", "#F4A261", "#E76F51"]
    fig = plt.figure(figsize=(17, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.38, wspace=0.25)

    axis = fig.add_subplot(grid[0, 0])
    bars = axis.bar(DATASETS, WINDOW_EPOCHS, color=colors)
    axis.set_yscale("log")
    axis.set_ylabel("training window-epochs (log scale)")
    axis.set_title("A. Optimizer exposure")
    axis.grid(axis="y", which="both", alpha=0.2)
    _labels(axis, bars, WINDOW_EPOCHS, lambda v: f"{v/1e6:.2f}M" if v < 1e9 else f"{v/1e9:.3f}B")
    ratio = WINDOW_EPOCHS[0] / WINDOW_EPOCHS[1:].sum()
    axis.text(0.02, 0.03, f"ADS-B / four-dataset total = {ratio:.1f}x", transform=axis.transAxes, fontsize=10, weight="bold")

    axis = fig.add_subplot(grid[0, 1])
    bars = axis.bar(DATASETS, INPUT_GIB, color=colors)
    axis.set_yscale("log")
    axis.set_ylabel("frozen training/transfer input (GiB, log scale)")
    axis.set_title("B. Data footprint")
    axis.grid(axis="y", which="both", alpha=0.2)
    _labels(axis, bars, INPUT_GIB, lambda v: f"{v:.3f} GiB" if v < 1 else f"{v:.2f} GiB")
    axis.text(0.02, 0.03, "Four-dataset Drive bundle total: 980.87 MiB", transform=axis.transAxes, fontsize=9)

    axis = fig.add_subplot(grid[1, :])
    x = np.arange(len(RUNTIME_LABELS))
    midpoint = (RUNTIME_LOW_MIN + RUNTIME_HIGH_MIN) / 2.0
    error = np.vstack([midpoint - RUNTIME_LOW_MIN, RUNTIME_HIGH_MIN - midpoint])
    point_colors = ["#264653", "#6C757D", "#2A9D8F", "#4C956C", "#F4A261", "#E76F51"]
    for index, color in enumerate(point_colors):
        axis.errorbar(
            x[index],
            midpoint[index],
            yerr=error[:, index : index + 1],
            fmt="none",
            ecolor=color,
            elinewidth=8,
            capsize=8,
            alpha=0.82,
        )
    axis.scatter(x, midpoint, c=point_colors, s=80, zorder=4, edgecolor="white")
    axis.set_xticks(x, RUNTIME_LABELS)
    axis.set_ylabel("wall-clock minutes")
    axis.set_title("C. Runtime: measured compute-loop values from accepted artifacts")
    axis.grid(axis="y", alpha=0.25)
    for index, (low, high) in enumerate(zip(RUNTIME_LOW_MIN, RUNTIME_HIGH_MIN, strict=True)):
        label = f"{low:.1f} min" if low == high else f"{low:.0f}-{high:.0f} min"
        axis.text(index, high + 5, label, ha="center", fontsize=9, weight="bold")
    axis.text(
        0.01,
        0.96,
        "All four accepted v2 runs used NVIDIA L4. GPU-loop totals exclude Drive transfer, hash verification, "
        "Parquet loading and final score export; those stages are mainly CPU/I/O bound.",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
    )
    axis.text(
        0.01,
        0.86,
        "Measured basis: training_history.csv, 30 frozen epochs, batch=1024. The raw archives are not streamed into "
        "the optimizer; Colab consumed the curated Gold/parsed transfer bundles.",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        color="#444444",
    )

    fig.suptitle("UAV anomaly studies - compute load and execution planning (2026-07-27)", fontsize=16, weight="bold")
    output = out_dir / "adsb_vs_four_dataset_compute_load.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "schema_version": 1,
        "status": "artifact_measured_training_loop_not_end_to_end_notebook_benchmark",
        "adsb": {
            "physical_training_rows": int(TRAIN_ROWS[0]),
            "training_windows_per_epoch": int(TRAIN_WINDOWS[0]),
            "epochs": int(EPOCHS[0]),
            "training_window_epochs": int(WINDOW_EPOCHS[0]),
            "training_input_gib": float(INPUT_GIB[0]),
            "l4_training_elapsed_seconds_this_reported_session": 2539.8835,
            "truth_v2_cpu_elapsed_seconds_observed": 7762.0,
            "truth_v2_rows_audited": TEST_EXPOSURE[0],
        },
        "four_dataset": [
            {
                "dataset": DATASETS[index],
                "optimizer_windows_per_epoch": int(TRAIN_WINDOWS[index]),
                "epochs": int(EPOCHS[index]),
                "training_window_epochs": int(WINDOW_EPOCHS[index]),
                "input_gib": float(INPUT_GIB[index]),
                "test_windows": TEST_EXPOSURE[index],
                "measured_l4_training_loop_minutes": float(RUNTIME_LOW_MIN[index + 1]),
            }
            for index in range(1, 5)
        ],
        "sequential_four_dataset_l4_training_loop_minutes": float(RUNTIME_LOW_MIN[2:].sum()),
        "parallel_four_training_loop_lower_bound_minutes": float(RUNTIME_LOW_MIN[2:].max()),
        "measurement_note": "ADS-B elapsed is session-scoped. Four-dataset values are exact sums of per-epoch elapsed_seconds in the accepted L4 histories; end-to-end notebook time is longer because transfer, verification, Parquet load and scoring are excluded.",
    }
    json_path = out_dir / "adsb_vs_four_dataset_compute_load.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return [output, json_path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adsb/plots/compute_load_comparison"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps({"outputs": [str(path) for path in run(args.out_dir.resolve())]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
