"""ML-16 Kol D: SEAD Dense (non-recurrent) Autoencoder wired into the official
ml14/ml15 fusion + decision-layer evaluation harness (docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md).

Reuses, unmodified:
  - src/ml/models/dense_autoencoder.py (DenseAutoencoder, train_dense_autoencoder,
    reconstruction_scores/masked_mse re-exported from lstm_autoencoder.py)
  - src/ml/models/lstm_autoencoder.py (AE_FEATURES/WINDOW/STRIDE -- SAME columns/window/
    stride as the LSTM-AE family, for a fair comparison; no new windowing config)
  - src/ml/data/windowing.py::build_windows (no new windowing code)
  - scripts/run_ml8a_temporal_boosting.py::_align_score (the established, already-in-repo
    window-score -> arbitrary-timestamp causal alignment convention, merge_asof
    direction="backward")
  - scripts/run_ml9_category_evaluation.py (_evaluate, _fit_policies, _jsonable,
    _score_modules, _streams, BUDGETS, MIN_RECALL)
  - src/ml/evaluation/score_fusion.py (empirical_probability, last_causal_per_bucket,
    max_score_fusion)
  - src/ml/decision/decision_layers.py (via _fit_policies: threshold/k_of_n/cusum)
  - src/ml/models/modular_iforest.py (fit_modular_iforest, PX4_ML7_CANDIDATE_MODULES,
    PX4_ML12_THIN_MODULES) to reproduce the existing_fusion/itki_komutu/ml14_fusion
    columns needed as fusion partners for (b)/(c).

This file mirrors scripts/run_ml_lstm_sead_evaluation.py's structure closely (same evaluation
harness, different model family) -- intentional, not accidental duplication: each model family
owns its own runner script in this codebase's existing convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ml8a_temporal_boosting import _align_score
from scripts.run_ml9_category_evaluation import (
    BUDGETS,
    MIN_RECALL,
    _evaluate,
    _fit_policies,
    _jsonable,
    _score_modules,
    _streams,
)
from src.ml.artifacts import save_torch_checkpoint
from src.ml.data.scaling import apply_scaler_params, fit_scaler_params
from src.ml.data.windowing import build_windows
from src.ml.evaluation.events import load_uav_sead_ranges, load_uav_sead_ranges_by_category
from src.ml.evaluation.score_fusion import (
    empirical_probability,
    last_causal_per_bucket,
    max_score_fusion,
)
from src.ml.features.uav_attack_features import feature_columns
from src.ml.models.dense_autoencoder import (
    DenseAutoencoder,
    reconstruction_scores,
    train_dense_autoencoder,
)
from src.ml.models.lstm_autoencoder import AE_FEATURES, STRIDE, WINDOW
from src.ml.models.modular_iforest import (
    PX4_ML12_THIN_MODULES,
    PX4_ML7_CANDIDATE_MODULES,
    fit_modular_iforest,
)
from src.ml.training_log import write_training_log

SOURCE = "uav_sead"
FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
ML14_REFERENCE = ROOT / "artifacts/ml14/uav_sead/full_matrix"

AE_COLS = AE_FEATURES[SOURCE]
DENSE_WINDOW = WINDOW[SOURCE]
DENSE_STRIDE = STRIDE[SOURCE]
MAX_GAP_S = 2.0

# (a)/(b)/(c) -- the only three variants registered in docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md SS1.
SCORE_SOURCES = ("dense_ae_recon", "dense_ae_ml14_fusion", "dense_ae_itki_fusion")
# Intermediate columns recomputed with unchanged ml14 code, only to prove determinism
# against the frozen artifacts/ml14 run (Gate A) and to build (b)/(c). Never evaluated
# as extra reported score-source rows themselves.
DETERMINISM_SOURCES = ("existing_fusion", "itki_komutu", "ml14_fusion")
DECISION_STRIDE_S = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_second_streams(scored: pd.DataFrame) -> pd.DataFrame:
    # Includes DETERMINISM_SOURCES too -- needed for the Gate A determinism check loop,
    # which fits/evaluates existing_fusion/itki_komutu/ml14_fusion the same way ml14 does.
    return last_causal_per_bucket(
        scored,
        stride_seconds=DECISION_STRIDE_S,
        columns=["source_id", "t_rel_s", "label", *SCORE_SOURCES, *DETERMINISM_SOURCES],
    )


def _dense_sequence(scaled: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Scaled AE feature columns with NaNs restored (masked_mse handles missingness)."""
    sequence = scaled[["source_id", "t_rel_s", "label"]].copy()
    for col in AE_COLS:
        sequence[col] = scaled[col].where(raw[col].notna())
    return sequence


def _train_dense_ae(sequence: pd.DataFrame, train_ids: set[str], val_ids: set[str], seed: int):
    x_train, m_train, meta_train = build_windows(
        sequence[sequence["source_id"].isin(train_ids)], AE_COLS,
        window=DENSE_WINDOW, stride=DENSE_STRIDE, max_gap_s=MAX_GAP_S,
    )
    x_val, m_val, meta_val = build_windows(
        sequence[sequence["source_id"].isin(val_ids)], AE_COLS,
        window=DENSE_WINDOW, stride=DENSE_STRIDE, max_gap_s=MAX_GAP_S,
    )
    if not len(x_train) or not len(x_val):
        raise RuntimeError("SEAD Dense-AE train/val pencereleri bos (train veya val ucus seti cok kisa)")
    model, training = train_dense_autoencoder(
        DenseAutoencoder(DENSE_WINDOW, len(AE_COLS)), x_train, m_train, x_val, m_val, seed=seed,
    )
    return model, training, {
        "train_windows": int(len(x_train)), "val_windows": int(len(x_val)),
        "train_flights": len(train_ids), "val_flights": len(val_ids),
    }


def _score_dense_ae(model, sequence: pd.DataFrame, score_ids: set[str]) -> pd.DataFrame:
    x, mask, meta = build_windows(
        sequence[sequence["source_id"].isin(score_ids)], AE_COLS,
        window=DENSE_WINDOW, stride=DENSE_STRIDE, max_gap_s=MAX_GAP_S,
    )
    if not len(x):
        raise RuntimeError("SEAD Dense-AE skorlama pencereleri bos")
    raw_scores = reconstruction_scores(model, x, mask)
    return pd.DataFrame({
        "source_id": meta["source_id"].to_numpy(),
        "t_rel_s": meta["t_end"].to_numpy(dtype=float),
        "dense_ae_recon_raw": raw_scores,
    })


def _gate_a(metrics_determinism: pd.DataFrame, window_coverage: list[dict]) -> dict:
    if not ML14_REFERENCE.exists():
        return {"status": "unknown", "reason": "artifacts/ml14 reference missing"}
    frozen = pd.read_csv(ML14_REFERENCE / "metrics.csv")
    key = ["split", "seed", "score_source", "decision", "budget"]
    left = metrics_determinism.set_index(key)[["event_onset_recall", "false_alarms_per_hour"]].sort_index()
    right = (
        frozen[frozen["score_source"].isin(DETERMINISM_SOURCES) & frozen["split"].isin(metrics_determinism["split"].unique())]
        .set_index(key)[["event_onset_recall", "false_alarms_per_hour"]].sort_index()
    )
    common = left.index.intersection(right.index)
    matched = len(common)
    if matched:
        diff = (left.loc[common] - right.loc[common]).abs()
        max_abs_diff = float(diff.to_numpy().max())
    else:
        max_abs_diff = float("nan")
    identical = matched > 0 and max_abs_diff < 1e-9
    return {
        "status": "passed" if identical else "failed",
        "rule": "existing_fusion/itki_komutu/ml14_fusion recomputed with unchanged ml14 code "
                "must exactly reproduce frozen artifacts/ml14/uav_sead/full_matrix rows",
        "matched_rows": int(matched),
        "expected_rows": int(len(left)),
        "max_abs_diff": max_abs_diff,
        "window_coverage": window_coverage,
    }


def _gate_b(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    for (source, decision, budget), group in metrics.groupby(["score_source", "decision", "budget"]):
        recall = float(group["event_onset_recall"].mean())
        fa = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "score_source": source,
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": recall,
            "mean_false_alarms_per_hour": fa,
            "passed": recall >= MIN_RECALL[budget] and fa <= BUDGETS[budget],
        })
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "rule": "any of {dense_ae_recon, dense_ae_ml14_fusion, dense_ae_itki_fusion} x "
                "{threshold, k_of_n, cusum}: critical recall >=0.30 @ FA<=2 "
                "or advisory recall >=0.50 @ FA<=12",
        "evaluated_seed_count": seed_count,
        "candidates": rows,
    }


def run(run_name: str, split_names: tuple[str, ...] | None = None) -> Path:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"][SOURCE]
    folds = config["splits"]
    if split_names is not None:
        unknown = set(split_names) - set(folds)
        if unknown:
            raise ValueError(f"Unknown split names: {sorted(unknown)}")
        folds = {name: folds[name] for name in split_names}

    holdout = set(config["splits"]["split_00"]["final_holdout"])
    development = set().union(*(
        set(split[part]) for split in folds.values() for part in ("train", "val", "test")
    ))
    if development & holdout:
        raise AssertionError("Blind holdout entered the ML-16 Kol D development request")

    raw = pd.read_parquet(
        FEATURE_PATH, filters=[("source_id", "in", sorted(development))],
    ).reset_index(drop=True)
    silver_time = pd.read_parquet(
        SILVER_PATH, columns=["source_id", "timestamp"],
        filters=[("source_id", "in", sorted(development))],
    )
    for name, frame in (("feature", raw), ("silver", silver_time)):
        if set(frame["source_id"].unique()) & holdout:
            raise AssertionError(f"Blind holdout rows were read from {name} table")
    missing_ae_cols = [c for c in AE_COLS if c not in raw.columns]
    if missing_ae_cols:
        raise AssertionError(f"AE_FEATURES[uav_sead] columns missing from gold table: {missing_ae_cols}")

    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(LABEL_PATH)
    categories = load_uav_sead_ranges_by_category(LABEL_PATH)
    output = ROOT / "artifacts/ml_dense_ae_sead/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []
    all_determinism: list[dict] = []
    window_coverage: list[dict] = []
    module_definitions = {**PX4_ML7_CANDIDATE_MODULES, "itki_komutu": PX4_ML12_THIN_MODULES["itki_komutu"]}

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")

        scaler = fit_scaler_params(raw[raw["source_id"].isin(parts["train"])], feature_columns(raw))
        scaled = apply_scaler_params(raw, scaler)
        fitted = fit_modular_iforest(scaled, split, module_definitions, seed=seed, n_jobs=1)
        scored = _score_modules(fitted, scaled, parts["val"])
        scored["ml14_fusion"] = max_score_fusion(scored, ["existing_fusion", "itki_komutu"])
        scored = scored.reset_index(drop=True)

        sequence = _dense_sequence(scaled, raw)
        model, training, window_counts = _train_dense_ae(sequence, parts["train"], parts["val"], seed)
        history = training.pop("history")
        write_training_log(history, SOURCE, "ml16_kol_d_dense_ae_sead", f"{run_id}_{split_name}")

        score_ids = parts["val"] | parts["test"]
        window_scores = _score_dense_ae(model, sequence, score_ids)

        endpoints = scored[["source_id", "t_rel_s"]].reset_index(drop=True)
        scored["dense_ae_recon_raw"] = _align_score(endpoints, window_scores, "dense_ae_recon_raw")

        val_mask = scored["source_id"].isin(parts["val"]).to_numpy()
        scored["dense_ae_recon"] = empirical_probability(
            scored.loc[val_mask, "dense_ae_recon_raw"].to_numpy(dtype=float),
            scored["dense_ae_recon_raw"].to_numpy(dtype=float),
        )
        scored["dense_ae_ml14_fusion"] = max_score_fusion(scored, ["dense_ae_recon", "ml14_fusion"])
        scored["dense_ae_itki_fusion"] = max_score_fusion(scored, ["dense_ae_recon", "itki_komutu"])

        test_mask = scored["source_id"].isin(parts["test"]).to_numpy()
        coverage = {
            "split": split_name,
            "val_row_finite_fraction": float(np.isfinite(
                scored.loc[val_mask, "dense_ae_recon_raw"].to_numpy(dtype=float)).mean()),
            "test_row_finite_fraction": float(np.isfinite(
                scored.loc[test_mask, "dense_ae_recon_raw"].to_numpy(dtype=float)).mean()),
            **window_counts,
        }
        window_coverage.append(coverage)

        streams = _one_second_streams(scored)

        split_dir = output / split_name
        models_dir = split_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, item in fitted.items():
            joblib.dump(item["model"], models_dir / f"{name}.joblib")
        (split_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
        (split_dir / "calibration.json").write_text(json.dumps({
            name: {key: value for key, value in item.items() if key != "model"}
            for name, item in fitted.items()
        }, indent=2), encoding="utf-8")
        save_torch_checkpoint(model, models_dir / "dense_ae.pt", metadata={
            "source": SOURCE,
            "feature_columns": AE_COLS,
            "window": DENSE_WINDOW,
            "stride": DENSE_STRIDE,
            "max_gap_s": MAX_GAP_S,
            "seed": seed,
            "training": training,
            "window_counts": window_counts,
            "train_flights": sorted(parts["train"]),
            "val_flights": sorted(parts["val"]),
            "blind_holdout_used": False,
        })
        (split_dir / "dense_ae_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

        policies_json = {}
        for score_source in SCORE_SOURCES:
            val_streams = _streams(streams, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                for decision, policy in _fit_policies(val_streams, budget, seed).items():
                    policies_json[f"{score_source}:{budget_name}:{decision}"] = policy.to_dict()
                    overall, by_label, by_category = _evaluate(
                        streams, parts["test"], score_source, policy,
                        t0=t0, ranges=ranges, ranges_by_category=categories,
                        flight_labels=config["flight_labels"],
                    )
                    common = {
                        "split": split_name, "seed": seed, "score_source": score_source,
                        "decision": decision, "budget": budget_name,
                    }
                    all_metrics.append({**common, **overall})
                    all_labels.extend({**common, **row} for row in by_label)
                    all_categories.extend({**common, **row} for row in by_category)

        # Gate-A determinism proof only (never reported as an evaluated variant).
        for score_source in DETERMINISM_SOURCES:
            val_streams = _streams(streams, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                for decision, policy in _fit_policies(val_streams, budget, seed).items():
                    overall, _, _ = _evaluate(
                        streams, parts["test"], score_source, policy,
                        t0=t0, ranges=ranges, ranges_by_category=categories,
                        flight_labels=config["flight_labels"],
                    )
                    all_determinism.append({
                        "split": split_name, "seed": seed, "score_source": score_source,
                        "decision": decision, "budget": budget_name, **overall,
                    })
        (split_dir / "policies.json").write_text(json.dumps(policies_json, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(all_metrics)
    label_metrics = pd.DataFrame(all_labels)
    category_metrics = pd.DataFrame(all_categories)
    determinism = pd.DataFrame(all_determinism)
    metrics.to_csv(output / "metrics.csv", index=False)
    label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
    category_metrics.to_csv(output / "category_metrics.csv", index=False)
    determinism.to_csv(output / "determinism_check.csv", index=False)

    gates = {
        "gate_a": _gate_a(determinism, window_coverage),
        "gate_b": _gate_b(metrics),
    }
    (output / "gates.json").write_text(json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8")

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-16 Kol D SEAD Dense-AE + official fusion/decision evaluation",
        "source": SOURCE,
        "plan": "docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md",
        "score_sources_evaluated": list(SCORE_SOURCES),
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "development_flights": int(raw["source_id"].nunique()),
        "development_source_ids_sha256": hashlib.sha256(
            "\n".join(sorted(development)).encode("utf-8")).hexdigest(),
        "evaluated_splits": sorted(folds),
        "ae_feature_columns": AE_COLS,
        "dense_ae_window": DENSE_WINDOW,
        "dense_ae_stride": DENSE_STRIDE,
        "gps_health_column_coverage_note": (
            "course_change_deg/jamming_indicator/noise_per_ms/hdop/vdop/satellites_used/"
            "s_variance_m_s are row-complete in only ~6% of development rows (~12% of "
            "flights have any value) -- structural GPS-health topic gap, not imputed; "
            "masked_mse/build_windows mask channel absorbs it (see docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md sec2)."
        ),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "gate_status": {name: value["status"] for name, value in gates.items()},
        "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_matrix")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()
    output = run(args.run_name, tuple(args.splits) if args.splits else None)
    print(f"ML-16 Kol D artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
