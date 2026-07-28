"""Plot truth-v2 recall, natural burden, and event outcome matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPRESENTATIVE_BUDGETS = (0.1, 5.0, 50.0, 500.0)


class EvaluationPlotContractError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("candidate_namespace") != "contextual_physics_v2":
        raise EvaluationPlotContractError("Unexpected evaluation namespace")
    if report.get("threshold_selection_performed_on_truth_v2") is not False:
        raise EvaluationPlotContractError("Truth-v2 threshold selection must be false")
    return report


def _rows(report: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    for recipe, result in report["results"].items():
        if "profiles" in result:
            for profile, payload in result["profiles"].items():
                rows.append((f"{recipe}\n{profile}", payload["points"]))
        else:
            rows.append((f"{recipe}\nCUSUM", result["points"]))
    return rows


def _matrix(
    rows: list[tuple[str, list[dict[str, Any]]]],
    budgets: list[float],
    field: str,
) -> np.ndarray:
    values = []
    for _, points in rows:
        by_budget = {float(point["pareto_v"]): point for point in points}
        values.append([float(by_budget[float(v)][field]) for v in budgets])
    return np.asarray(values, dtype=float)


def _heatmap(
    matrix: np.ndarray,
    labels: list[str],
    budgets: list[float],
    *,
    title: str,
    colorbar_label: str,
    output: Path,
    burden: bool = False,
) -> None:
    display = np.log10(matrix + 1e-4) if burden else matrix
    fig, axis = plt.subplots(figsize=(15, 7))
    image = axis.imshow(display, aspect="auto", cmap="magma" if burden else "viridis")
    axis.set_xticks(range(len(budgets)), [f"V={v:g}" for v in budgets])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("frozen budget grid (episodes / 100 scoreable flight-hours)")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.3f}" if burden else f"{100.0 * value:.1f}%"
            axis.text(column, row, text, ha="center", va="center", fontsize=7, color="white")
    bar = fig.colorbar(image, ax=axis, shrink=0.85)
    bar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _event_outcomes(
    rows: list[tuple[str, list[dict[str, Any]]]],
    output: Path,
) -> None:
    labels = [label for label, _ in rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    for axis, budget in zip(axes.flat, REPRESENTATIVE_BUDGETS, strict=True):
        tp: list[int] = []
        fn: list[int] = []
        for _, points in rows:
            point = next(p for p in points if float(p["pareto_v"]) == budget)
            detected = int(point["n_detected_events"])
            total = int(point["n_events"])
            tp.append(detected)
            fn.append(total - detected)
        y = np.arange(len(labels))
        axis.barh(y, tp, label="detected (TP)", color="#2A9D8F")
        axis.barh(y, fn, left=tp, label="missed (FN)", color="#E76F51")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(f"Injected observable events - V={budget:g}")
        axis.set_xlabel("event count")
        axis.grid(axis="x", alpha=0.25)
    axes.flat[0].legend(loc="lower right")
    fig.suptitle(
        "Event-level outcome matrices (TP/FN only)\n"
        "Paired-clean data are reported separately as natural episodes/hour; "
        "they are not relabelled as FP/TN.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(report_path: Path, out_dir: Path) -> list[Path]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    report = _load(report_path)
    budgets = [float(value) for value in report["budget_grid"]]
    rows = _rows(report)
    labels = [label for label, _ in rows]
    recall = _matrix(rows, budgets, "event_recall")
    burden = _matrix(rows, budgets, "paired_clean_natural_burden_per_hour")
    out_dir.mkdir(parents=True, exist_ok=False)
    outputs = [
        out_dir / "event_recall_heatmap.png",
        out_dir / "paired_clean_natural_burden_heatmap.png",
        out_dir / "event_detection_outcome_matrices.png",
    ]
    _heatmap(
        recall,
        labels,
        budgets,
        title="contextual_physics_v2 - observable-event recall",
        colorbar_label="event recall",
        output=outputs[0],
    )
    _heatmap(
        burden,
        labels,
        budgets,
        title="contextual_physics_v2 - paired-clean natural alert burden",
        colorbar_label="log10(episodes/hour + 1e-4)",
        output=outputs[1],
        burden=True,
    )
    _event_outcomes(rows, outputs[2])
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=Path(
            "artifacts/adsb/runs/20260724_contextual_physics_v2_truth_v2_eval_v1/"
            "truth_v2_eval_report.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/adsb/plots/contextual_v2_evaluation"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = run(args.eval_report.resolve(), args.out_dir.resolve())
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
