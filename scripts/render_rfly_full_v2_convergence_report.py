import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPERIMENT_ROOT = (
    ROOT / "artifacts" / "rfly_full" / "v2" / "normal_temporal_ae"
    / "robustness" / "approved_20260722_nested_v1"
)
OUTPUT = EXPERIMENT_ROOT / "candidates" / "R4"
COLORS = {"R3": "#4c78a8", "R4": "#e45756"}


def _save(fig, name: str) -> Path:
    path = OUTPUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def render_mean_comparison(comparison: pd.DataFrame) -> Path:
    selected = comparison.loc[
        comparison["candidate"].isin(["R3", "R4"])
        & comparison["policy"].eq("critical")
    ].set_index("candidate")
    recall_columns = [
        ("event_recall_mean", "Genel recall"),
        ("real_motor_recall_mean", "Real Motor"),
        ("real_sensor_recall_mean", "Real Sensor"),
        ("real_macro_recall_mean", "Real macro"),
    ]
    fa_columns = [
        ("all_nonfault_fa_per_hour_mean", "Tüm nonfault"),
        ("real_normal_fa_per_hour_mean", "Real normal"),
        ("wind_fa_per_hour_mean", "Wind"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.36
    for axis, columns, title, ylabel in (
        (axes[0], recall_columns, "Critical recall: R3 ve convergence R4", "Recall (%)"),
        (axes[1], fa_columns, "Critical false-alarm yükü", "Alarm / saat"),
    ):
        x = np.arange(len(columns))
        for offset, candidate in ((-width / 2, "R3"), (width / 2, "R4")):
            values = [float(selected.loc[candidate, column]) for column, _ in columns]
            if "Recall" in ylabel:
                values = [value * 100 for value in values]
            bars = axis.bar(
                x + offset, values, width, label=candidate,
                color=COLORS[candidate],
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axis.set_xticks(x, [label for _, label in columns])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    return _save(fig, "04_R3_R4_metric_comparison.png")


def render_rotation_stability(metrics: pd.DataFrame) -> Path:
    critical = metrics.loc[metrics["policy"].eq("critical")].sort_values("rotation")
    x = critical["rotation"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(x, critical["event_recall"] * 100, marker="o", label="Genel")
    axes[0].plot(x, critical["real_macro_recall"] * 100, marker="o", label="Real macro")
    axes[0].axhline(40, color="#54a24b", linestyle="--", linewidth=1, label="Real hedef %40")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_title("Recall rotasyon kararlılığı")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        x, critical["all_nonfault_fa_per_hour"], marker="o", label="Tüm nonfault"
    )
    axes[1].plot(
        x, critical["real_normal_fa_per_hour"], marker="o", label="Real normal"
    )
    axes[1].axhline(2, color="#54a24b", linestyle="--", linewidth=1, label="Genel hedef 2/s")
    axes[1].axhline(4, color="#f2cf5b", linestyle="--", linewidth=1, label="Real hedef 4/s")
    axes[1].set_ylabel("Alarm / saat")
    axes[1].set_title("Normal false-alarm kararlılığı")
    axes[1].legend(fontsize=8)

    axes[2].bar(x, critical["wind_fa_per_hour"], color="#72b7b2")
    axes[2].axhline(15, color="#54a24b", linestyle="--", linewidth=1, label="Ara hedef 15/s")
    axes[2].set_ylabel("Wind alarm / saat")
    axes[2].set_title("Wind rotasyon kararlılığı")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel("Outer rotasyon")
        axis.set_xticks(x)
        axis.grid(axis="y", alpha=0.25)
    return _save(fig, "05_R4_rotation_stability.png")


def render_tradeoff(comparison: pd.DataFrame) -> Path:
    critical = comparison.loc[comparison["policy"].eq("critical")].copy()
    fig, axis = plt.subplots(figsize=(8.5, 6))
    axis.fill_betweenx(
        [40, 100], 0, 4, color="#54a24b", alpha=0.10,
        label="Başarı bölgesi",
    )
    for _, row in critical.iterrows():
        candidate = str(row["candidate"])
        x = float(row["real_normal_fa_per_hour_mean"])
        y = float(row["real_macro_recall_mean"]) * 100
        axis.scatter(x, y, s=75, zorder=3)
        axis.annotate(candidate, (x, y), xytext=(5, 5), textcoords="offset points")
    axis.axvline(4, color="#e45756", linestyle="--", linewidth=1, label="Real FA sınırı 4/s")
    axis.axhline(40, color="#4c78a8", linestyle="--", linewidth=1, label="Real macro hedef %40")
    axis.set_xlabel("Held-out Real-NoFault FA / saat (ortalama)")
    axis.set_ylabel("Real macro recall (%)")
    axis.set_title("Real transfer trade-off: hiçbir aday hedef bölgesinde değil")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save(fig, "06_real_recall_fa_tradeoff.png")


def render_convergence_reduction() -> Path:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rotation_convergence"]
    x = np.arange(len(rows))
    initial = [float(row["initial_validation_loss"]) for row in rows]
    best = [float(row["best_validation_loss"]) for row in rows]
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    first = axis.bar(x - width / 2, initial, width, label="Epoch 0", color="#bab0ac")
    second = axis.bar(x + width / 2, best, width, label="En iyi checkpoint", color="#54a24b")
    axis.bar_label(first, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(second, fmt="%.2f", padding=3, fontsize=8)
    axis.set_xticks(x, [str(index) for index in range(len(rows))])
    axis.set_xlabel("Rotasyon")
    axis.set_ylabel("Inner Real validation loss")
    axis.set_title("Convergence fine-tune validation-loss değişimi")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save(fig, "07_R4_validation_loss_reduction.png")


def main() -> None:
    comparison = pd.read_csv(EXPERIMENT_ROOT / "candidate_comparison_by_policy.csv")
    metrics = pd.read_csv(OUTPUT / "all_rotation_metrics.csv")
    paths = [
        render_mean_comparison(comparison),
        render_rotation_stability(metrics),
        render_tradeoff(comparison),
        render_convergence_reduction(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
