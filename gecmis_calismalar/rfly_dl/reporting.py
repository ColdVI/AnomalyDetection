"""Additional reproducible visual diagnostics for RflyMAD DL artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gecmis_calismalar.rfly_dl.config import BUDGETS, MIN_RECALL

MODEL_ORDER = ("lstm_ae", "dense_ae", "usad")
MODEL_LABELS = {
    "lstm_ae": "LSTM-AE",
    "dense_ae": "Dense-AE",
    "usad": "USAD",
}
MODEL_COLORS = {
    "lstm_ae": "#2563eb",
    "dense_ae": "#f59e0b",
    "usad": "#dc2626",
}
ADDITIONAL_PLOTS = (
    "decision_heatmaps.png",
    "split_stability.png",
    "fault_group_recall.png",
    "detection_delay_distributions.png",
    "normal_alarm_burden.png",
    "window_metrics_by_split.png",
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_decision_heatmaps(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        ("critical", "threshold", "Kritik\nEşik"),
        ("critical", "k_of_n", "Kritik\nK/N"),
        ("critical", "cusum", "Kritik\nCUSUM"),
        ("advisory", "threshold", "Uyarı\nEşik"),
        ("advisory", "k_of_n", "Uyarı\nK/N"),
        ("advisory", "cusum", "Uyarı\nCUSUM"),
    ]
    recall = np.full((len(MODEL_ORDER), len(columns)), np.nan)
    false_alarm = np.full_like(recall, np.nan)
    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, (budget, decision, _label) in enumerate(columns):
            match = summary[
                summary["model"].eq(model)
                & summary["budget"].eq(budget)
                & summary["decision"].eq(decision)
            ]
            if not match.empty:
                recall[row_index, column_index] = float(
                    match.iloc[0]["mean_event_onset_recall"]
                )
                false_alarm[row_index, column_index] = float(
                    match.iloc[0]["mean_false_alarms_per_hour"]
                )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8))
    images = [
        axes[0].imshow(recall, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto"),
        axes[1].imshow(
            false_alarm, cmap="YlOrRd", vmin=0.0,
            vmax=max(150.0, float(np.nanmax(false_alarm))), aspect="auto",
        ),
    ]
    titles = (
        "Olay başlangıcı recall — yüksek değer daha iyi",
        "Yanlış alarm / saat — düşük değer daha iyi (kritik ≤2, uyarı ≤12)",
    )
    values = (recall, false_alarm)
    formats = (".3f", ".1f")
    for axis, image, title, matrix, fmt in zip(
        axes, images, titles, values, formats
    ):
        axis.set_xticks(
            np.arange(len(columns)), [column[2] for column in columns]
        )
        axis.set_yticks(
            np.arange(len(MODEL_ORDER)),
            [MODEL_LABELS[model] for model in MODEL_ORDER],
        )
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index, row_index, format(value, fmt),
                        ha="center", va="center", fontsize=9, color="#111827",
                    )
        fig.colorbar(image, ax=axis, shrink=0.78, pad=0.02)
    fig.suptitle("Model × karar mekanizması × bütçe ısı haritası", y=1.01)
    _save(fig, path)


def _plot_split_stability(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics[
        metrics["decision"].eq("cusum") & metrics["budget"].eq("advisory")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for model in MODEL_ORDER:
        group = subset[subset["model"].eq(model)].sort_values("seed")
        axes[0].plot(
            group["seed"], group["event_onset_recall"], marker="o",
            linewidth=2, label=MODEL_LABELS[model], color=MODEL_COLORS[model],
        )
        axes[1].plot(
            group["seed"], group["false_alarms_per_hour"], marker="o",
            linewidth=2, label=MODEL_LABELS[model], color=MODEL_COLORS[model],
        )
    axes[0].axhline(
        MIN_RECALL["advisory"], linestyle="--", color="#111827",
        label="recall hedefi",
    )
    axes[1].axhline(
        BUDGETS["advisory"], linestyle="--", color="#111827",
        label="FA bütçesi",
    )
    axes[0].set(title="Split bazında recall", xlabel="Split tohumu", ylabel="Recall")
    axes[1].set(
        title="Split bazında yanlış alarm yükü",
        xlabel="Split tohumu", ylabel="Yanlış alarm / saat",
    )
    for axis in axes:
        axis.set_xticks(sorted(subset["seed"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("CUSUM / uyarı çalışma noktasının split kararlılığı")
    _save(fig, path)


def _plot_fault_group_recall(categories: pd.DataFrame, path: Path) -> None:
    subset = categories[
        categories["decision"].eq("cusum")
        & categories["budget"].eq("advisory")
    ]
    aggregate = (
        subset.groupby(["model", "fault_group"])["event_onset_recall"]
        .agg(["mean", "std"])
    )
    groups = ("motor", "sensor")
    x = np.arange(len(groups))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for model_index, model in enumerate(MODEL_ORDER):
        means = [
            aggregate.loc[(model, group), "mean"]
            if (model, group) in aggregate.index else np.nan
            for group in groups
        ]
        errors = [
            aggregate.loc[(model, group), "std"]
            if (model, group) in aggregate.index else np.nan
            for group in groups
        ]
        positions = x + (model_index - 1) * width
        bars = axis.bar(
            positions, means, width, yerr=errors, capsize=4,
            label=MODEL_LABELS[model], color=MODEL_COLORS[model], alpha=0.88,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.axhline(
        MIN_RECALL["advisory"], linestyle="--", color="#111827",
        label="uyarı recall hedefi",
    )
    axis.set_xticks(x, ["Motor arızaları", "Sensör arızaları"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Olay başlangıcı recall")
    axis.set_title("Arıza ailesine göre yakalama — CUSUM / uyarı")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, path)


def _plot_detection_delays(flights: pd.DataFrame, path: Path) -> None:
    detected = flights["event_detected"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    subset = flights[
        flights["decision"].eq("cusum")
        & flights["budget"].eq("advisory")
        & detected
        & ~flights["label"].eq("normal")
    ].copy()
    subset["fault_group"] = np.where(
        subset["label"].str.contains("motor", case=False, na=False),
        "motor", "sensor",
    )
    subset["first_alarm_delay_s"] = pd.to_numeric(
        subset["first_alarm_delay_s"], errors="coerce"
    )
    data, labels, colors = [], [], []
    for model in MODEL_ORDER:
        for group in ("motor", "sensor"):
            values = subset[
                subset["model"].eq(model)
                & subset["fault_group"].eq(group)
            ]["first_alarm_delay_s"].dropna().to_numpy()
            data.append(values)
            labels.append(f"{MODEL_LABELS[model]}\n{group}")
            colors.append(MODEL_COLORS[model])
    fig, axis = plt.subplots(figsize=(11, 4.8))
    boxes = axis.boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for index, values in enumerate(data, start=1):
        axis.text(
            index, 0.98, f"n={len(values)}", ha="center", va="top",
            transform=axis.get_xaxis_transform(), fontsize=8,
        )
    axis.set_ylabel("İlk alarm gecikmesi (s)")
    axis.set_title(
        "Yakalanan olaylarda gecikme dağılımı — CUSUM / uyarı"
    )
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def _plot_normal_alarm_burden(flights: pd.DataFrame, path: Path) -> None:
    subset = flights[
        flights["decision"].eq("cusum")
        & flights["budget"].eq("advisory")
        & flights["label"].eq("normal")
    ].copy()
    subset["false_alarm_events"] = pd.to_numeric(
        subset["false_alarm_events"], errors="coerce"
    ).fillna(0)
    proportions, distributions = [], []
    for model in MODEL_ORDER:
        values = subset[subset["model"].eq(model)]["false_alarm_events"].to_numpy()
        proportions.append(float(np.mean(values > 0)) if len(values) else np.nan)
        distributions.append(values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(MODEL_ORDER))
    bars = axes[0].bar(
        x, proportions,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
    )
    axes[0].bar_label(
        bars, labels=[f"%{value * 100:.1f}" for value in proportions], padding=3
    )
    axes[0].set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("En az bir alarm alan normal değerlendirme oranı")
    axes[0].set_title("Normal uçuş düzeyi yanlış pozitif")
    axes[0].grid(axis="y", alpha=0.25)
    boxes = axes[1].boxplot(
        distributions,
        tick_labels=[MODEL_LABELS[model] for model in MODEL_ORDER],
        patch_artist=True, showfliers=True,
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch, model in zip(boxes["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.55)
    axes[1].set_ylabel("Normal uçuş başına alarm olayı")
    axes[1].set_title("Alarm sayısı dağılımı")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("CUSUM / uyarı — normal uçuş alarm yükü")
    _save(fig, path)


def _plot_window_metrics(diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for model in MODEL_ORDER:
        group = diagnostics[diagnostics["model"].eq(model)].sort_values("seed")
        axes[0].plot(
            group["seed"], group["auroc"], marker="o",
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
        axes[1].plot(
            group["seed"], group["auprc"], marker="o",
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
    prevalence = (
        diagnostics.groupby("seed")["positive_prevalence"].mean().sort_index()
    )
    axes[1].plot(
        prevalence.index, prevalence.values, color="#111827",
        linestyle="--", marker=".", label="AUPRC rastgele tabanı",
    )
    axes[0].axhline(0.5, color="#111827", linestyle="--", label="rastgele AUROC")
    axes[0].set(title="Pencere AUROC", xlabel="Split tohumu", ylabel="AUROC")
    axes[1].set(title="Pencere AUPRC", xlabel="Split tohumu", ylabel="AUPRC")
    for axis in axes:
        axis.set_ylim(0.35, 0.75)
        axis.set_xticks(sorted(diagnostics["seed"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Eşiksiz pencere ayrışmasının split bazında görünümü")
    _save(fig, path)


def render_additional_plots(artifact_dir: Path | str) -> tuple[Path, ...]:
    """Render plots from persisted tables without retraining any model."""
    root = Path(artifact_dir)
    summary = pd.read_csv(root / "summary_table.csv")
    metrics = pd.read_csv(root / "metrics.csv")
    categories = pd.read_csv(root / "fault_group_metrics.csv")
    flights = pd.read_csv(root / "per_flight_metrics.csv")
    diagnostics = pd.read_csv(root / "window_diagnostics.csv")
    plotters = (
        (_plot_decision_heatmaps, summary),
        (_plot_split_stability, metrics),
        (_plot_fault_group_recall, categories),
        (_plot_detection_delays, flights),
        (_plot_normal_alarm_burden, flights),
        (_plot_window_metrics, diagnostics),
    )
    outputs = []
    for filename, (plotter, table) in zip(ADDITIONAL_PLOTS, plotters):
        output = root / filename
        plotter(table, output)
        outputs.append(output)
    return tuple(outputs)


def refresh_manifest_hashes(artifact_dir: Path | str) -> None:
    """Refresh the file hash inventory after deterministic report rendering."""
    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest["additional_plots"] = list(ADDITIONAL_PLOTS)
    manifest["files"] = {
        str(path.relative_to(root)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
