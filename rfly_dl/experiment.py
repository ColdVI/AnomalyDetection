"""End-to-end five-split direct DL experiment for RflyMAD."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rfly_dl.config import (
    BATCH_SIZE, BUDGETS, CUSUM_BOOTSTRAP_HOURS, FEATURE_COLUMNS,
    LEARNING_RATE, MAX_EPOCHS, MAX_GAP_S, MIN_RECALL, MODEL_NAMES,
    PATIENCE, RHO_THRESHOLD, SCALE_CLIP, WINDOW, WINDOW_STRIDE,
)
from rfly_dl.data import (
    GOLD_PATH, SILVER_PATH, SPLIT_PATH, align_window_scores,
    apply_robust_scaler, empirical_probability, feature_completeness,
    fit_robust_scaler, ids_sha256, load_contract, load_development,
    make_windows, sha256,
)
from rfly_dl.decision import fit_policies
from rfly_dl.evaluation import (
    evaluate_policy, magnitude_diagnostics, window_diagnostics,
)
from rfly_dl.models import make_model, reconstruction_scores, train_model
from rfly_dl.reporting import render_additional_plots

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/rfly_dl"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _streams(frame: pd.DataFrame, score_name: str) -> list[np.ndarray]:
    return [
        group.sort_values("t_rel_s")[score_name].to_numpy(dtype=float)
        for _, group in frame.groupby("source_id", sort=False)
    ]


def _summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["model", "decision", "budget"], sort=True):
        model, decision, budget = keys
        recall = float(group["event_onset_recall"].mean())
        fa = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "model": model,
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": recall,
            "mean_false_alarms_per_hour": fa,
            "mean_median_detection_delay_s": float(
                group["median_detection_delay_s"].mean()
            ),
            "seed_count": int(group["seed"].nunique()),
            "passed_operational_gate": bool(
                len(group) >= 5
                and recall >= MIN_RECALL[budget]
                and fa <= BUDGETS[budget]
            ),
        })
    return pd.DataFrame(rows)


def _best_rows(summary: pd.DataFrame) -> list[dict]:
    result = []
    for model, group in summary.groupby("model", sort=True):
        feasible = group[
            group["mean_false_alarms_per_hour"] <= group["budget"].map(BUDGETS)
        ]
        candidates = feasible if len(feasible) else group
        chosen = candidates.sort_values(
            ["mean_event_onset_recall", "mean_false_alarms_per_hour"],
            ascending=[False, True],
        ).iloc[0]
        result.append(_jsonable(chosen.to_dict()))
    return result


def _plot_recall_fa(summary: pd.DataFrame, path: Path) -> None:
    colors = {"lstm_ae": "#2563eb", "dense_ae": "#f59e0b", "usad": "#dc2626"}
    markers = {"threshold": "o", "k_of_n": "s", "cusum": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, budget in zip(axes, ("critical", "advisory")):
        subset = summary[summary["budget"].eq(budget)]
        for row in subset.itertuples(index=False):
            axis.scatter(
                row.mean_false_alarms_per_hour,
                row.mean_event_onset_recall,
                color=colors[row.model],
                marker=markers[row.decision],
                s=75,
            )
            axis.annotate(
                f"{row.model.replace('_ae', '').upper()}\n{row.decision}",
                (row.mean_false_alarms_per_hour, row.mean_event_onset_recall),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        axis.axvline(BUDGETS[budget], color="#111827", linestyle="--", linewidth=1)
        axis.axhline(MIN_RECALL[budget], color="#111827", linestyle=":", linewidth=1)
        axis.set_title(f"{budget.title()} hedefi")
        axis.set_xlabel("Yanlış alarm / saat")
        axis.set_xlim(left=0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Arıza olayı recall")
    fig.suptitle("RflyMAD doğrudan DL — beş split ortalaması")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_experiment(
    run_name: str,
    *,
    selected_splits: tuple[str, ...] = (
        "split_00", "split_01", "split_02", "split_03", "split_04",
    ),
    selected_models: tuple[str, ...] = MODEL_NAMES,
    max_epochs: int = MAX_EPOCHS,
    device: str = "cpu",
    torch_threads: int = 4,
) -> Path:
    unknown_models = set(selected_models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    output = ARTIFACT_ROOT / run_name
    if output.exists():
        raise FileExistsError(f"Run output already exists: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(max(1, torch_threads))

    folds, development, holdout = load_contract(selected_splits)
    raw, intervals, invalid = load_development(folds, development, holdout)
    labels = (
        raw[["source_id", "label"]]
        .drop_duplicates("source_id")
        .set_index("source_id")["label"]
        .astype(str)
        .to_dict()
    )
    feature_completeness(raw).to_csv(
        output / "feature_completeness.csv", index=False
    )

    metric_rows, confusion_rows, category_rows = [], [], []
    diagnostic_rows, flight_rows, coverage_rows = [], [], []
    histories: list[pd.DataFrame] = []

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if not raw[raw["source_id"].isin(parts["train"])]["label"].eq("normal").all():
            raise AssertionError("Anomalous flight entered DL training")
        if not raw[raw["source_id"].isin(parts["val"])]["label"].eq("normal").all():
            raise AssertionError("Anomalous flight entered DL validation")

        split_dir = output / split_name
        split_dir.mkdir()
        train_raw = raw[raw["source_id"].isin(parts["train"])]
        scaler = fit_robust_scaler(train_raw)
        (split_dir / "scaler.json").write_text(
            json.dumps(scaler, indent=2), encoding="utf-8"
        )
        scaled = apply_robust_scaler(raw, scaler)
        windows = {}
        for part in ("train", "val", "test"):
            subset = scaled[scaled["source_id"].isin(parts[part])]
            windows[part] = make_windows(subset)
            x, mask, _meta = windows[part]
            coverage_rows.append({
                "split": split_name,
                "part": part,
                "n_flights": int(subset["source_id"].nunique()),
                "n_rows": int(len(subset)),
                "n_windows": int(len(x)),
                "observed_cell_fraction": float(mask.mean()) if len(mask) else np.nan,
            })
        x_train, m_train, _ = windows["train"]
        x_val, m_val, meta_val = windows["val"]
        x_test, m_test, meta_test = windows["test"]
        base_val = scaled[scaled["source_id"].isin(parts["val"])]
        base_test = scaled[scaled["source_id"].isin(parts["test"])]

        split_policies: dict[str, dict] = {}
        for model_index, model_name in enumerate(selected_models):
            model_seed = seed + model_index * 1000
            model = make_model(
                model_name, window=WINDOW,
                n_features=len(FEATURE_COLUMNS), seed=model_seed,
            )
            training = train_model(
                model_name, model, x_train, m_train, x_val, m_val,
                seed=model_seed, max_epochs=max_epochs, patience=PATIENCE,
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=device,
            )
            model_dir = split_dir / model_name
            model_dir.mkdir()
            torch.save({
                "model_name": model_name,
                "window": WINDOW,
                "n_features": len(FEATURE_COLUMNS),
                "feature_columns": list(FEATURE_COLUMNS),
                "state_dict": training.model.state_dict(),
            }, model_dir / "model.pt")
            history = pd.DataFrame(training.history)
            history["split"] = split_name
            history["model"] = model_name
            history.to_csv(model_dir / "training_history.csv", index=False)
            histories.append(history)

            val_scores = reconstruction_scores(
                model_name, training.model, x_val, m_val, device=device
            )
            test_scores = reconstruction_scores(
                model_name, training.model, x_test, m_test, device=device
            )
            audit_count = min(5000, len(x_test))
            audit_index = np.linspace(0, len(x_test) - 1, audit_count, dtype=int)
            untrained = make_model(
                model_name, window=WINDOW, n_features=len(FEATURE_COLUMNS),
                seed=model_seed + 100_000,
            )
            untrained_scores = reconstruction_scores(
                model_name, untrained, x_test[audit_index], m_test[audit_index],
                device=device,
            )
            magnitude = magnitude_diagnostics(
                test_scores[audit_index], untrained_scores,
                x_test[audit_index], m_test[audit_index], threshold=RHO_THRESHOLD,
            )
            window_metrics = window_diagnostics(
                meta_test, test_scores, labels, intervals
            )
            diagnostic_rows.append({
                "split": split_name,
                "seed": seed,
                "model": model_name,
                "best_epoch": training.best_epoch,
                "best_val_loss": training.best_val_loss,
                "parameter_count": training.parameter_count,
                "audit_windows": audit_count,
                **window_metrics,
                **magnitude,
            })

            val_probability = empirical_probability(val_scores, val_scores)
            test_probability = empirical_probability(val_scores, test_scores)
            score_name = "anomaly_probability"
            val_stream_frame = align_window_scores(
                base_val, meta_val, val_probability, score_name
            )
            test_stream_frame = align_window_scores(
                base_test, meta_test, test_probability, score_name
            )
            val_streams = _streams(val_stream_frame, score_name)

            model_policies: dict[str, dict] = {}
            for budget_name, budget in BUDGETS.items():
                policies = fit_policies(val_streams, budget, seed=model_seed)
                for decision_name, policy in policies.items():
                    model_policies[
                        f"{budget_name}:{decision_name}"
                    ] = policy.to_dict()
                    metrics, confusion, categories, per_flight = evaluate_policy(
                        test_stream_frame, score_name, policy, intervals=intervals
                    )
                    common = {
                        "split": split_name, "seed": seed, "model": model_name,
                        "decision": decision_name, "budget": budget_name,
                    }
                    metric_rows.append({**common, **metrics})
                    confusion_rows.append({**common, **confusion})
                    category_rows.extend({**common, **row} for row in categories)
                    flight_rows.extend({**common, **row} for row in per_flight)
            split_policies[model_name] = model_policies
            del training, model, untrained, val_scores, test_scores
            gc.collect()

        (split_dir / "policies.json").write_text(
            json.dumps(_jsonable(split_policies), indent=2), encoding="utf-8"
        )
        del scaled, windows, x_train, m_train, x_val, m_val, x_test, m_test
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    confusions = pd.DataFrame(confusion_rows)
    categories = pd.DataFrame(category_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    flights = pd.DataFrame(flight_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary_table = _summary_table(metrics)
    metrics.to_csv(output / "metrics.csv", index=False)
    confusions.to_csv(output / "flight_level_confusions.csv", index=False)
    categories.to_csv(output / "fault_group_metrics.csv", index=False)
    diagnostics.to_csv(output / "window_diagnostics.csv", index=False)
    flights.to_csv(output / "per_flight_metrics.csv", index=False)
    coverage.to_csv(output / "window_coverage.csv", index=False)
    summary_table.to_csv(output / "summary_table.csv", index=False)

    _plot_recall_fa(summary_table, output / "recall_vs_false_alarm.png")
    _plot_confusions(summary_table, confusions, output / "confusion_matrices.png")
    _plot_diagnostics(diagnostics, output / "score_diagnostics.png")
    _plot_training(histories, output / "training_curves.png")
    render_additional_plots(output)

    full_run = len(selected_splits) == 5
    gate_passed = bool(
        full_run and len(selected_models) == len(MODEL_NAMES)
        and summary_table["passed_operational_gate"].any()
    )
    summary = {
        "status": "complete_five_split" if full_run else "smoke_only",
        "models": list(selected_models),
        "evaluated_splits": list(selected_splits),
        "operational_gate_status": (
            "passed" if gate_passed else ("failed" if full_run else "smoke_only")
        ),
        "operational_rule": (
            "any model x decision: critical recall>=0.30 at FA/h<=2 "
            "or advisory recall>=0.50 at FA/h<=12, five splits"
        ),
        "best_by_model": _best_rows(summary_table),
        "magnitude_flagged_model_splits": int(
            diagnostics["magnitude_domination_flagged"].sum()
        ),
        "magnitude_audits": int(len(diagnostics)),
        "historical_holdout": {
            "excluded_from_runner": True,
            "runner_rows_read": False,
            "n_flights": len(holdout),
            "blindness_claim_for_new_track": False,
            "reason": (
                "aggregate full-parquet schema/count inspection occurred on 2026-07-20 "
                "before the runner; holdout remains excluded but is not claimed unread"
            ),
        },
        "invalid_no_active_fault_exclusions": len(invalid),
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files = [
        path for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "RflyMAD direct deep-learning development evaluation",
        "pre_registration": "docs/RFLY_DL_DOGRUDAN_DEGERLENDIRME_PLANI.md",
        "models": list(selected_models),
        "feature_columns": list(FEATURE_COLUMNS),
        "window": WINDOW,
        "window_stride": WINDOW_STRIDE,
        "max_gap_s": MAX_GAP_S,
        "train_only_robust_scaler_clip": SCALE_CLIP,
        "max_epochs": max_epochs,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "cusum_bootstrap_hours": CUSUM_BOOTSTRAP_HOURS,
        "evaluated_splits": list(selected_splits),
        "development_ids_sha256": ids_sha256(set(raw["source_id"].unique())),
        "holdout_excluded": True,
        "holdout_blindness_claim": False,
        "input_hashes": {
            "gold": sha256(GOLD_PATH),
            "silver": sha256(SILVER_PATH),
            "split_manifest": sha256(SPLIT_PATH),
        },
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output
def _plot_confusions(
    summary: pd.DataFrame, confusions: pd.DataFrame, path: Path
) -> None:
    best = {row["model"]: row for row in _best_rows(summary)}
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        chosen = best[model]
        subset = confusions[
            confusions["model"].eq(model)
            & confusions["decision"].eq(chosen["decision"])
            & confusions["budget"].eq(chosen["budget"])
        ]
        matrix = np.array([
            [int(subset["tn"].sum()), int(subset["fp"].sum())],
            [int(subset["fn"].sum()), int(subset["tp"].sum())],
        ])
        axis.imshow(matrix, cmap="Blues")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, f"{value:,}", ha="center", va="center")
        axis.set_xticks([0, 1], ["Alarm yok", "Alarm"])
        axis.set_yticks([0, 1], ["Normal uçuş", "Arızalı uçuş"])
        axis.set_title(
            f"{model.upper()}\n{chosen['decision']} / {chosen['budget']}"
        )
    fig.suptitle(
        "Uçuş düzeyi matris — split toplamı (aynı uçuşlar seed başına tekrar sayılır)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    aggregate = diagnostics.groupby("model", sort=True).agg(
        auroc=("auroc", "mean"),
        auprc=("auprc", "mean"),
        prevalence=("positive_prevalence", "mean"),
        rho_random=("rho_trained_vs_untrained", "mean"),
        rho_magnitude=("rho_trained_vs_magnitude", "mean"),
    )
    models = list(aggregate.index)
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.23
    axes[0].bar(x - width, aggregate["auroc"], width, label="AUROC")
    axes[0].bar(x, aggregate["auprc"], width, label="AUPRC")
    axes[0].bar(x + width, aggregate["prevalence"], width, label="AUPRC tabanı")
    axes[0].set_xticks(x, models)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Pencere sıralama tanısı")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        x - width / 2, aggregate["rho_random"], width, label="eğitilmiş-random"
    )
    axes[1].bar(
        x + width / 2, aggregate["rho_magnitude"], width, label="eğitilmiş-genlik"
    )
    axes[1].axhline(RHO_THRESHOLD, color="#dc2626", linestyle="--", label="uyarı")
    axes[1].set_xticks(x, models)
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Öğrenme / genlik baskınlığı denetimi")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_training(histories: list[pd.DataFrame], path: Path) -> None:
    models = sorted(set().union(*(set(frame["model"]) for frame in histories)))
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        for history in histories:
            subset = history[history["model"].eq(model)]
            if subset.empty:
                continue
            axis.plot(
                subset["epoch"], subset["val_loss"], alpha=0.7,
                label=str(subset["split"].iloc[0]),
            )
        axis.set_title(model.upper())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Normal validation loss")
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
