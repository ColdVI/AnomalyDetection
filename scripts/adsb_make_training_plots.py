"""ADSB-2 egitim raporunu (artifacts/adsb/models/baseline_training_report.json)
gorsellestirir: loss egrileri, model x senaryo ROC egrileri, AUC heatmap, guven-
skoru dagilimlari. Salt-okunur -- rapor dosyasini degistirmez, egitim calistirmaz.

Kullanim:
    python scripts/adsb_make_training_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_PATH = Path("artifacts/adsb/models/baseline_training_report.json")
OUT_DIR = Path("artifacts/adsb/plots")

MODEL_LABELS = {"dense_ae": "Dense-AE", "lstm_ae": "LSTM-AE", "lstm_forecaster": "LSTM-forecaster"}
MODEL_COLORS = {"dense_ae": "#2E86AB", "lstm_ae": "#A23B72", "lstm_forecaster": "#F18F01"}


def plot_loss_curves(report: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in MODEL_LABELS.items():
        if key not in report:
            continue
        hist = report[key]["train_loss_history"]
        ax.plot(range(1, len(hist) + 1), hist, marker="o", ms=3, label=label, color=MODEL_COLORS[key])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Egitim loss (maskeli MSE)")
    ax.set_title("ADS-B egitim loss egrileri")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "loss_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_roc_curves(report: dict, out_dir: Path) -> list[Path]:
    paths = []
    recipe_names = None
    for key in MODEL_LABELS:
        if key in report:
            recipe_names = list(report[key]["per_recipe"].keys())
            break
    if not recipe_names:
        return paths

    n = len(recipe_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, recipe in zip(axes, recipe_names):
        for key, label in MODEL_LABELS.items():
            if key not in report or recipe not in report[key]["per_recipe"]:
                continue
            r = report[key]["per_recipe"][recipe]
            auc = r["auc"]
            ax.plot(r["roc_fpr"], r["roc_tpr"], label=f"{label} (AUC={auc:.3f})", color=MODEL_COLORS[key])
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_title(recipe, fontsize=10)
        ax.set_xlabel("FPR")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TPR")
    fig.suptitle("Model x senaryo ROC egrileri (temiz vs. sentetik-bozuk, kalici korpus)")
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_auc_heatmap(report: dict, out_dir: Path) -> Path:
    model_keys = [k for k in MODEL_LABELS if k in report]
    recipe_names = list(report[model_keys[0]]["per_recipe"].keys())
    auc_matrix = np.array([
        [report[m]["per_recipe"][r]["auc"] for r in recipe_names] for m in model_keys
    ])

    fig, ax = plt.subplots(figsize=(1.6 * len(recipe_names) + 2, 1.2 * len(model_keys) + 1.5))
    im = ax.imshow(auc_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(recipe_names)))
    ax.set_xticklabels(recipe_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in model_keys])
    for i in range(len(model_keys)):
        for j in range(len(recipe_names)):
            ax.text(j, i, f"{auc_matrix[i, j]:.3f}", ha="center", va="center",
                     color="black", fontsize=9)
    ax.set_title("AUC heatmap: model x sentetik-bozulma senaryosu\n(0.5=rastgele, 1.0=mukemmel ayrim)")
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    path = out_dir / "auc_heatmap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_confusion_matrices(report: dict, out_dir: Path) -> Path:
    model_keys = [k for k in MODEL_LABELS if k in report]
    recipe_names = list(report[model_keys[0]]["per_recipe"].keys())
    threshold = report.get("confidence_threshold", 0.95)

    fig, axes = plt.subplots(len(model_keys), len(recipe_names),
                              figsize=(2.4 * len(recipe_names), 2.4 * len(model_keys)))
    if len(model_keys) == 1:
        axes = axes.reshape(1, -1)
    for i, m in enumerate(model_keys):
        for j, r in enumerate(recipe_names):
            cm = np.array(report[m]["per_recipe"][r]["confusion_matrix_conf0.95"])
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["temiz", "bozuk"], fontsize=7)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["temiz", "bozuk"], fontsize=7)
            if i == 0:
                ax.set_title(r, fontsize=8)
            if j == 0:
                ax.set_ylabel(MODEL_LABELS[m], fontsize=9)
    fig.suptitle(f"Confusion matrix (guven-skoru esigi={threshold}) -- satir=gercek, sutun=tahmin")
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_score_distributions(report: dict, out_dir: Path) -> Path:
    model_keys = [k for k in MODEL_LABELS if k in report]
    recipe_names = list(report[model_keys[0]]["per_recipe"].keys())

    fig, axes = plt.subplots(len(model_keys), 1, figsize=(8, 2.8 * len(model_keys)), sharex=False)
    if len(model_keys) == 1:
        axes = [axes]
    for ax, m in zip(axes, model_keys):
        means_clean = [report[m]["per_recipe"][r]["mean_clean_score"] for r in recipe_names]
        means_corrupt = [report[m]["per_recipe"][r]["mean_corrupt_score"] for r in recipe_names]
        x = np.arange(len(recipe_names))
        width = 0.35
        ax.bar(x - width / 2, means_clean, width, label="temiz", color="#4C956C")
        ax.bar(x + width / 2, means_corrupt, width, label="bozuk", color="#C1121F")
        ax.set_xticks(x)
        ax.set_xticklabels(recipe_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("ort. skor")
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Ortalama rekonstruksiyon/tahmin skoru: temiz vs. sentetik-bozuk")
    fig.tight_layout()
    path = out_dir / "score_distributions.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_loss_curves(report, OUT_DIR),
        *plot_roc_curves(report, OUT_DIR),
        plot_auc_heatmap(report, OUT_DIR),
        plot_confusion_matrices(report, OUT_DIR),
        plot_score_distributions(report, OUT_DIR),
    ]
    print(f"{len(paths)} grafik yazildi -> {OUT_DIR}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
