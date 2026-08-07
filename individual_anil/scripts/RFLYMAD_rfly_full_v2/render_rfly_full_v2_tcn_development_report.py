"""Render the frozen development-only TCN sweep report figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWEEP_ROOT = (
    ROOT / "artifacts" / "rfly_full" / "v2" / "supervised_tcn"
    / "development_5fold_20260722_v1"
)
AE_ROOT = (
    ROOT / "artifacts" / "rfly_full" / "v2" / "normal_temporal_ae"
    / "sweep_20260722_093049"
)
ROBUSTNESS_ROOT = (
    ROOT / "artifacts" / "rfly_full" / "v2" / "normal_temporal_ae"
    / "robustness" / "approved_20260722_nested_v1"
)
OUTPUT = ROOT / "docs" / "assets" / "rflymad_v2_tcn_development"
POLICY_COLORS = {"critical": "#e45756", "advisory": "#4c78a8"}


def _save(fig: plt.Figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def render_epoch_curves(history: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for outer_fold, frame in history.groupby("outer_fold"):
        frame = frame.sort_values("epoch")
        axes[0].plot(
            frame["epoch"], frame["train_loss"], marker=".",
            label=f"fold {outer_fold}",
        )
        axes[1].plot(
            frame["epoch"], frame["validation_loss"], marker=".",
            label=f"fold {outer_fold}",
        )
    axes[0].set_title("TCN eğitim kaybı — epoch bazında")
    axes[1].set_title("TCN validation kaybı — epoch bazında")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Kayıp")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    return _save(fig, "01_epoch_loss_curves.png")


def render_best_epochs(history: pd.DataFrame) -> Path:
    rows = []
    for outer_fold, frame in history.groupby("outer_fold"):
        best = frame.loc[frame["validation_loss"].astype(float).idxmin()]
        rows.append({
            "outer_fold": int(outer_fold),
            "best_epoch": int(best["epoch"]),
            "epoch_cap": int(frame["epoch_cap"].max()),
            "best_validation_loss": float(best["validation_loss"]),
        })
    table = pd.DataFrame(rows).sort_values("outer_fold")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = axes[0].bar(
        table["outer_fold"], table["best_epoch"], color="#4c78a8",
    )
    axes[0].bar_label(bars, fmt="%d", padding=3)
    axes[0].plot(
        table["outer_fold"], table["epoch_cap"], "o--",
        color="#e45756", label="epoch tavanı",
    )
    axes[0].set_title("Validation ile seçilen en iyi epoch")
    axes[0].set_xlabel("Outer fold")
    axes[0].set_ylabel("Epoch")
    axes[0].legend()
    loss_bars = axes[1].bar(
        table["outer_fold"], table["best_validation_loss"], color="#72b7b2",
    )
    axes[1].bar_label(loss_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1].set_title("En iyi validation loss")
    axes[1].set_xlabel("Outer fold")
    axes[1].set_ylabel("Validation loss")
    for axis in axes:
        axis.set_xticks(table["outer_fold"])
        axis.grid(axis="y", alpha=0.25)
    return _save(fig, "02_best_epochs.png")


def render_fold_metrics(outer: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for policy, frame in outer.groupby("policy", sort=False):
        frame = frame.sort_values("outer_fold")
        color = POLICY_COLORS[policy]
        axes[0, 0].plot(
            frame["outer_fold"], frame["event_recall"] * 100,
            "o-", color=color, label=policy,
        )
        axes[0, 1].plot(
            frame["outer_fold"], frame["all_nonfault_fa_per_hour"],
            "o-", color=color, label=policy,
        )
    critical = outer.loc[outer["policy"].eq("critical")].sort_values("outer_fold")
    x = critical["outer_fold"]
    axes[1, 0].plot(x, critical["real_motor_recall"] * 100, "o-", label="Motor")
    axes[1, 0].plot(x, critical["real_sensor_recall"] * 100, "o-", label="Sensor")
    axes[1, 0].plot(x, critical["real_macro_recall"] * 100, "o--", label="Macro")
    axes[1, 0].axhline(40, color="#54a24b", linestyle="--", label="macro hedef %40")
    axes[1, 1].plot(x, critical["wind_fa_per_hour"], "o-", label="Wind")
    axes[1, 1].plot(x, critical["real_normal_fa_per_hour"], "o-", label="Real normal")
    axes[1, 1].plot(x, critical["all_nonfault_fa_per_hour"], "o-", label="Tüm nonfault")
    axes[1, 1].axhline(15, color="#54a24b", linestyle="--", label="Wind ara hedef 15/s")
    axes[0, 0].set_title("Event recall kararlılığı")
    axes[0, 0].set_ylabel("Recall (%)")
    axes[0, 1].set_title("Nonfault false-alarm kararlılığı")
    axes[0, 1].set_ylabel("Alarm / saat")
    axes[1, 0].set_title("Critical Real fault recall")
    axes[1, 0].set_ylabel("Recall (%)")
    axes[1, 1].set_title("Critical domain false-alarm yükü")
    axes[1, 1].set_ylabel("Alarm / saat")
    for axis in axes.ravel():
        axis.set_xlabel("Outer fold")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return _save(fig, "03_fold_metric_stability.png")


def render_ae_tcn_comparison(aggregate: pd.DataFrame) -> Path:
    ae = pd.read_csv(AE_ROOT / "report_summary.csv").set_index("policy")
    tcn = aggregate.set_index("policy")
    policies = ["critical", "advisory"]
    x = np.arange(len(policies))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        ("recall_mean", "event_recall_mean", 100, "Event recall (%)"),
        ("fa_mean", "all_nonfault_fa_per_hour_mean", 1, "Nonfault alarm / saat"),
        ("wind_fa_mean", "wind_fa_per_hour_mean", 1, "Wind alarm / saat"),
    ]
    for axis, (ae_col, tcn_col, scale, title) in zip(axes, panels):
        ae_values = [float(ae.loc[p, ae_col]) * scale for p in policies]
        tcn_values = [float(tcn.loc[p, tcn_col]) * scale for p in policies]
        first = axis.bar(x - width / 2, ae_values, width, label="AE", color="#9d9da1")
        second = axis.bar(x + width / 2, tcn_values, width, label="TCN", color="#4c78a8")
        axis.bar_label(first, fmt="%.1f", padding=3, fontsize=8)
        axis.bar_label(second, fmt="%.1f", padding=3, fontsize=8)
        axis.set_xticks(x, policies)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Aynı development 5-fold kapsamındaki AE–TCN tanımlayıcı karşılaştırması")
    return _save(fig, "04_ae_tcn_comparison.png")


def render_real_tradeoff(aggregate: pd.DataFrame) -> Path:
    baseline = pd.read_csv(
        ROBUSTNESS_ROOT / "candidate_comparison_by_policy.csv"
    )
    ae = baseline.loc[
        baseline["candidate"].eq("frozen_baseline")
        & baseline["policy"].eq("critical")
    ].iloc[0]
    tcn = aggregate.loc[aggregate["policy"].eq("critical")].iloc[0]
    points = [
        ("AE", float(ae["real_normal_fa_per_hour_mean"]),
         float(ae["real_macro_recall_mean"]) * 100, "#9d9da1"),
        ("TCN", float(tcn["real_normal_fa_per_hour_mean"]),
         float(tcn["real_macro_recall_mean"]) * 100, "#4c78a8"),
    ]
    fig, axis = plt.subplots(figsize=(8.5, 6))
    axis.axvspan(0, 4, color="#54a24b", alpha=0.08)
    axis.axhspan(40, 100, color="#54a24b", alpha=0.08)
    axis.axvline(4, color="#e45756", linestyle="--", label="Real FA sınırı 4/s")
    axis.axhline(40, color="#54a24b", linestyle="--", label="Real macro hedef %40")
    for label, x_value, y_value, color in points:
        axis.scatter(x_value, y_value, s=110, color=color, zorder=3)
        axis.annotate(label, (x_value, y_value), xytext=(7, 7), textcoords="offset points")
    axis.set_xlim(left=0)
    axis.set_ylim(0, max(50, max(point[2] for point in points) * 1.25))
    axis.set_xlabel("Real-NoFault alarm / saat")
    axis.set_ylabel("Real macro recall (%)")
    axis.set_title("Critical Real transfer: recall–false-alarm trade-off")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save(fig, "05_real_tradeoff.png")


def main() -> None:
    history = pd.read_csv(SWEEP_ROOT / "training_history.csv")
    outer = pd.read_csv(SWEEP_ROOT / "outer_fold_metrics.csv")
    aggregate = pd.read_csv(SWEEP_ROOT / "aggregate_metrics.csv")
    paths = [
        render_epoch_curves(history),
        render_best_epochs(history),
        render_fold_metrics(outer),
        render_ae_tcn_comparison(aggregate),
        render_real_tradeoff(aggregate),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
