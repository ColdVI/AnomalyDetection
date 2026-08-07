"""Visual reporting for normal-only RflyMAD v2 validation rotations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"critical": "#d62828", "advisory": "#277da1"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _heatmap(ax: plt.Axes, values: np.ndarray, rows: list[str], columns: list[str], title: str) -> None:
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_title(title, fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def render(sweep: Path) -> None:
    progress = json.loads((sweep / "progress.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(sweep / "all_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    for policy, group in metrics.groupby("policy"):
        ax.scatter(
            group["all_nonfault_fa_per_hour"], group["event_recall"],
            s=75, color=COLORS[policy], label=policy,
        )
        for row in group.itertuples():
            ax.annotate(f"r{row.validation_rotation}", (row.all_nonfault_fa_per_hour, row.event_recall), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(2, color="#d62828", linestyle="--", alpha=0.6)
    ax.axhline(0.30, color="#d62828", linestyle=":", alpha=0.6)
    ax.axvline(12, color="#277da1", linestyle="--", alpha=0.6)
    ax.axhline(0.50, color="#277da1", linestyle=":", alpha=0.6)
    ax.set_xlabel("Yanlış alarm / saat — NoFault validation + arıza pre/post")
    ax.set_ylabel("Arıza-uçuş recall")
    ax.set_title("Normal-only temporal AE — validation rotasyonları", fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, sweep / "01_recall_vs_false_alarm_rotations.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    columns = [
        ("event_recall", "Arıza recall"),
        ("all_nonfault_fa_per_hour", "Normal maruziyet FA/saat"),
        ("environment_fa_per_hour", "Wind FA/saat"),
    ]
    for ax, (column, title) in zip(axes, columns):
        values = [metrics.loc[metrics.policy.eq(policy), column] for policy in ("critical", "advisory")]
        boxes = ax.boxplot(
            values, tick_labels=["critical", "advisory"], patch_artist=True
        )
        for box, policy in zip(boxes["boxes"], ("critical", "advisory")):
            box.set_facecolor(COLORS[policy])
            box.set_alpha(0.6)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    _save(fig, sweep / "02_metric_stability.png")

    family_rows = []
    for item in progress["completed"]:
        table = pd.read_csv(Path(item["output"]) / "domain_family_metrics.csv")
        table["validation_rotation"] = item["rotation"]
        family_rows.append(table)
    families = pd.concat(family_rows, ignore_index=True)
    labels = sorted({f"{row.domain} / {row.fault_family}" for row in families.itertuples()})
    rotations = sorted(metrics["validation_rotation"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, policy in zip(axes, ("critical", "advisory")):
        subset = families.loc[families.policy.eq(policy)].copy()
        subset["label"] = subset["domain"] + " / " + subset["fault_family"]
        table = subset.pivot(index="label", columns="validation_rotation", values="recall").reindex(index=labels, columns=rotations)
        _heatmap(ax, table.to_numpy(float), labels, [f"r{x}" for x in rotations], f"{policy.title()} aile recall")
    _save(fig, sweep / "03_family_recall_heatmap.png")

    fig, axes = plt.subplots(2, len(rotations), figsize=(3.2 * len(rotations), 6.5))
    for column, rotation in enumerate(rotations):
        for row, policy in enumerate(("critical", "advisory")):
            item = metrics.loc[
                metrics.validation_rotation.eq(rotation) & metrics.policy.eq(policy)
            ].iloc[0]
            matrix = np.asarray([[item.flight_tn, item.flight_fp], [item.flight_fn, item.flight_tp]], dtype=int)
            axes[row, column].imshow(matrix, cmap="Blues", vmin=0, vmax=metrics[["flight_tn", "flight_tp"]].to_numpy().max())
            for i in range(2):
                for j in range(2):
                    axes[row, column].text(j, i, str(matrix[i, j]), ha="center", va="center")
            axes[row, column].set_xticks([0, 1], ["Normal", "Alarm"])
            axes[row, column].set_yticks([0, 1], ["Normal", "Arıza"])
            axes[row, column].set_title(f"{policy} / r{rotation}")
    _save(fig, sweep / "04_confusion_matrices_by_rotation.png")

    aggregate = metrics.groupby("policy").agg(
        recall_mean=("event_recall", "mean"), recall_std=("event_recall", "std"),
        fa_mean=("all_nonfault_fa_per_hour", "mean"), fa_std=("all_nonfault_fa_per_hour", "std"),
        wind_fa_mean=("environment_fa_per_hour", "mean"),
    )
    aggregate.to_csv(sweep / "report_summary.csv")
