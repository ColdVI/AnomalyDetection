"""ML-16 Kol N: magnitude-normalized post-hoc reconstruction scoring, computed from the
ALREADY-TRAINED, frozen SEAD LSTM-AE/Dense-AE/USAD checkpoints (ADR-016/017/018) --
docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md.

NO retraining. NO new model architecture. NO fusion with ml14_fusion/itki_komutu this
round (recon-alone only, for a clean 6-cell comparison against the ADR-016/017/018
recon-alone baselines). Every checkpoint/scaler is loaded from the existing frozen
`artifacts/ml_{lstm,dense_ae,usad}_sead/uav_sead/full_matrix/split_NN/` run directories
and used strictly for inference (forward pass only, no `.backward()`/optimizer step
anywhere in this file).

Reuses, unmodified:
  - src/ml/models/lstm_autoencoder.py (LSTMAutoencoder, masked_mse_per_channel,
    AE_FEATURES/WINDOW/STRIDE)
  - src/ml/models/dense_autoencoder.py (DenseAutoencoder)
  - src/ml/models/usad.py (USAD)
  - src/ml/evaluation/magnitude_normalized_scoring.py (the two SS1 score formulas +
    per-architecture channel-error adapters -- new this phase, but shared/tested there,
    not reimplemented here)
  - src/ml/data/windowing.py::build_windows (no new windowing code)
  - src/ml/data/scaling.py::apply_scaler_params (scaler is REUSED from the frozen
    ADR-016/017/018 run's scaler.json -- fit_scaler_params is never called here)
  - scripts/run_ml8a_temporal_boosting.py::_align_score (same causal alignment
    convention already used for lstm_recon/dense_ae_recon/usad_score)
  - scripts/run_ml9_category_evaluation.py (_evaluate, _fit_policies, _jsonable,
    _streams, BUDGETS, MIN_RECALL)
  - src/ml/evaluation/score_fusion.py (empirical_probability, last_causal_per_bucket)
  - scripts/run_ml_lstm_sead_evaluation.py::_lstm_sequence,
    scripts/run_ml_dense_ae_sead_evaluation.py::_dense_sequence,
    scripts/run_ml_usad_sead_evaluation.py::_usad_sequence (the scaled/NaN-restored
    per-architecture row table builders -- byte-identical bodies across the three
    modules; imported per architecture rather than re-implemented once, so a future
    change to any one of them is picked up here automatically instead of silently
    diverging)

One script for all three architectures (unlike Kol L/D/U's one-file-per-architecture
convention) because this phase does not contain any architecture-specific TRAINING code
-- only a small "how do I call this frozen model's forward pass" adapter per family
(docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md SS4).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as ss

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
    _streams,
)
from scripts.run_ml_dense_ae_sead_evaluation import _dense_sequence
from scripts.run_ml_lstm_sead_evaluation import _lstm_sequence
from scripts.run_ml_usad_sead_evaluation import _usad_sequence
from src.ml.data.scaling import apply_scaler_params
from src.ml.data.windowing import build_windows
from src.ml.evaluation.events import load_uav_sead_ranges, load_uav_sead_ranges_by_category
from src.ml.evaluation.magnitude_normalized_scoring import (
    RELATIVE_ERROR_EPS,
    average_available_channels,
    compute_channel_errors_single_recon,
    compute_channel_errors_usad,
    compute_channel_errors_zero_baseline,
    per_channel_rank_normalized_scores,
)
from src.ml.evaluation.score_fusion import empirical_probability, last_causal_per_bucket
from src.ml.models.dense_autoencoder import DenseAutoencoder
from src.ml.models.lstm_autoencoder import AE_FEATURES, STRIDE, WINDOW, LSTMAutoencoder
from src.ml.models.usad import USAD

SOURCE = "uav_sead"
FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"

AE_COLS = AE_FEATURES[SOURCE]
AE_WINDOW = WINDOW[SOURCE]
AE_STRIDE = STRIDE[SOURCE]
MAX_GAP_S = 2.0
DECISION_STRIDE_S = 1.0
VARIANTS = ("relerr", "rankpct")
UNTRAINED_SEED = 999  # same convention as scripts/diagnose_ml_lstm_sead_magnitude_domination.py

ARCHITECTURES = {
    "lstm": {
        "score_prefix": "lstm",
        "frozen_run_dir": ROOT / "artifacts/ml_lstm_sead/uav_sead/full_matrix",
        "checkpoint_name": "lstm_ae.pt",
        "model_ctor": lambda n_features: LSTMAutoencoder(n_features),
        "sequence_fn": _lstm_sequence,
        "channel_error_fn": compute_channel_errors_single_recon,
    },
    "dense_ae": {
        "score_prefix": "dense_ae",
        "frozen_run_dir": ROOT / "artifacts/ml_dense_ae_sead/uav_sead/full_matrix",
        "checkpoint_name": "dense_ae.pt",
        "model_ctor": lambda n_features: DenseAutoencoder(AE_WINDOW, n_features),
        "sequence_fn": _dense_sequence,
        "channel_error_fn": compute_channel_errors_single_recon,
    },
    "usad": {
        "score_prefix": "usad",
        "frozen_run_dir": ROOT / "artifacts/ml_usad_sead/uav_sead/full_matrix",
        "checkpoint_name": "usad.pt",
        "model_ctor": lambda n_features: USAD(AE_WINDOW, n_features),
        "sequence_fn": _usad_sequence,
        "channel_error_fn": compute_channel_errors_usad,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spearman(a: np.ndarray, b: np.ndarray) -> dict:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return {"spearman": None, "n_windows": int(finite.sum())}
    correlation = ss.spearmanr(a[finite], b[finite]).correlation
    return {"spearman": float(correlation) if np.isfinite(correlation) else None,
            "n_windows": int(finite.sum())}


def _load_frozen_model(arch: dict, n_features: int, checkpoint_dir: Path):
    model = arch["model_ctor"](n_features)
    checkpoint = torch.load(
        checkpoint_dir / "models" / arch["checkpoint_name"], map_location="cpu", weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _score_variant_columns(arch_name: str) -> dict[str, str]:
    return {variant: f"{arch_name}_{variant}" for variant in VARIANTS}


def _correlation_diagnostic(
    arch: dict, n_features: int, x_train: np.ndarray, m_train: np.ndarray,
    x_score: np.ndarray, m_score: np.ndarray, test_window_mask: np.ndarray,
    trained_relerr_test: np.ndarray, trained_rankpct_test: np.ndarray,
) -> dict:
    """SS3: recompute trained-vs-untrained-random-init and trained-vs-magnitude-only
    Spearman correlations using the NEW scoring formulas (relerr/rankpct) instead of raw
    masked_mse -- the required, non-informational diagnostic. Both the untrained
    random-init network and the magnitude-only (reconstruction=0) null are scored with
    the SAME formula, each calibrated against its OWN train-window reference (fair,
    self-consistent comparison; see plan SS3).

    Runs on the FULL val+test window set (x_score/m_score, already built once by the
    caller for the main scoring pipeline) and slices down to the test-flight subset only
    AFTER the (small, n x n_features) per-window score arrays are computed -- not by
    copying a separate x_test/m_test window array up front, which would double memory
    on SEAD's ~250k-window splits for no benefit (every reduction here is row-wise
    independent, so slicing before or after the per-window reduction is equivalent).
    """
    channel_error_fn = arch["channel_error_fn"]

    torch.manual_seed(UNTRAINED_SEED)
    untrained_model = arch["model_ctor"](n_features)
    untrained_model.eval()
    u_mse_train, u_valid_train, _ = channel_error_fn(untrained_model, x_train, m_train)
    u_mse_score, u_valid_score, u_rel_score = channel_error_fn(untrained_model, x_score, m_score)
    untrained_relerr_test = average_available_channels(u_rel_score, u_valid_score)[test_window_mask]
    untrained_rankpct_test = per_channel_rank_normalized_scores(
        u_mse_train, u_valid_train, u_mse_score, u_valid_score)[test_window_mask]

    mag_mse_train, mag_valid_train, _ = compute_channel_errors_zero_baseline(x_train, m_train)
    mag_mse_score, mag_valid_score, mag_rel_score = compute_channel_errors_zero_baseline(x_score, m_score)
    magnitude_relerr_test = average_available_channels(mag_rel_score, mag_valid_score)[test_window_mask]
    magnitude_rankpct_test = per_channel_rank_normalized_scores(
        mag_mse_train, mag_valid_train, mag_mse_score, mag_valid_score)[test_window_mask]

    result = {}
    for variant, trained_arr, untrained_arr, magnitude_arr in (
        ("relerr", trained_relerr_test, untrained_relerr_test, magnitude_relerr_test),
        ("rankpct", trained_rankpct_test, untrained_rankpct_test, magnitude_rankpct_test),
    ):
        result[variant] = {
            "trained_vs_untrained_random_init_spearman": _spearman(trained_arr, untrained_arr),
            "trained_vs_magnitude_only_spearman": _spearman(trained_arr, magnitude_arr),
            "untrained_vs_magnitude_only_spearman": _spearman(untrained_arr, magnitude_arr),
        }
    return result


def _gate_a(holdout: set[str]) -> dict:
    return {
        "status": "passed",
        "rule": "blind holdout never read (development-set assertion below); no "
                ".fit(/train_lstm_autoencoder/train_dense_autoencoder/train_usad/"
                "fit_modular_iforest/fit_scaler_params call anywhere in this scoring "
                "path (verified statically in "
                "tests/test_ml16_kol_n_magnitude_normalized_scoring.py); "
                "masked_mse_per_channel sums to masked_mse exactly (same test file). "
                "No frozen-CSV byte-identity determinism check this round -- these are "
                "NEW scores, nothing existing to reproduce.",
        "blind_holdout_flights": len(holdout),
        "blind_holdout_read": False,
    }


def _gate_b(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique()) if len(metrics) else 0
    rows = []
    for (arch, source, decision, budget), group in metrics.groupby(
        ["architecture", "score_source", "decision", "budget"],
    ):
        recall = float(group["event_onset_recall"].mean())
        fa = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "architecture": arch, "score_source": source, "decision": decision, "budget": budget,
            "mean_event_onset_recall": recall, "mean_false_alarms_per_hour": fa,
            "passed": recall >= MIN_RECALL[budget] and fa <= BUDGETS[budget],
        })
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "rule": "any of {lstm,dense_ae,usad}_{relerr,rankpct} x {threshold,k_of_n,cusum}: "
                "critical recall >=0.30 @ FA<=2 or advisory recall >=0.50 @ FA<=12",
        "evaluated_seed_count": seed_count,
        "candidates": rows,
    }


def run(run_name: str = "full_matrix", split_names: tuple[str, ...] | None = None,
        architectures: tuple[str, ...] | None = None) -> dict[str, Path]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"][SOURCE]
    folds = config["splits"]
    if split_names is not None:
        unknown = set(split_names) - set(folds)
        if unknown:
            raise ValueError(f"Unknown split names: {sorted(unknown)}")
        folds = {name: folds[name] for name in split_names}
    arch_names = architectures if architectures is not None else tuple(ARCHITECTURES)
    unknown_arch = set(arch_names) - set(ARCHITECTURES)
    if unknown_arch:
        raise ValueError(f"Unknown architectures: {sorted(unknown_arch)}")

    holdout = set(config["splits"]["split_00"]["final_holdout"])
    development = set().union(*(
        set(split[part]) for split in folds.values() for part in ("train", "val", "test")
    ))
    if development & holdout:
        raise AssertionError("Blind holdout entered the ML-16 Kol N development request")

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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    outputs: dict[str, Path] = {}
    summary_rows: list[dict] = []

    for arch_name in arch_names:
        arch = ARCHITECTURES[arch_name]
        score_cols = list(_score_variant_columns(arch_name).values())
        output = ROOT / "artifacts/ml16_kol_n" / arch_name / run_name
        output.mkdir(parents=True, exist_ok=True)

        all_metrics: list[dict] = []
        all_labels: list[dict] = []
        all_categories: list[dict] = []
        all_correlations: list[dict] = []
        window_coverage: list[dict] = []

        for split_name, split in folds.items():
            seed = int(split["seed"])
            parts = {name: set(split[name]) for name in ("train", "val", "test")}
            if set().union(*parts.values()) & holdout:
                raise AssertionError(f"{arch_name}/{split_name}: holdout entered train/val/test")

            checkpoint_dir = arch["frozen_run_dir"] / split_name
            scaler = json.loads((checkpoint_dir / "scaler.json").read_text(encoding="utf-8"))
            scaled = apply_scaler_params(raw, scaler)
            sequence = arch["sequence_fn"](scaled, raw)

            x_train, m_train, meta_train = build_windows(
                sequence[sequence["source_id"].isin(parts["train"])], AE_COLS,
                window=AE_WINDOW, stride=AE_STRIDE, max_gap_s=MAX_GAP_S,
            )
            score_ids = parts["val"] | parts["test"]
            x_score, m_score, meta_score = build_windows(
                sequence[sequence["source_id"].isin(score_ids)], AE_COLS,
                window=AE_WINDOW, stride=AE_STRIDE, max_gap_s=MAX_GAP_S,
            )
            if not len(x_train) or not len(x_score):
                raise RuntimeError(
                    f"{arch_name}/{split_name}: ML-16 Kol N train/score pencereleri bos")

            model = _load_frozen_model(arch, len(AE_COLS), checkpoint_dir)
            channel_error_fn = arch["channel_error_fn"]
            channel_mse_train, channel_valid_train, _ = channel_error_fn(model, x_train, m_train)
            channel_mse_score, channel_valid_score, channel_rel_score = channel_error_fn(
                model, x_score, m_score)

            relerr_raw_score = average_available_channels(channel_rel_score, channel_valid_score)
            rankpct_raw_score = per_channel_rank_normalized_scores(
                channel_mse_train, channel_valid_train, channel_mse_score, channel_valid_score)
            raw_scores = {"relerr": relerr_raw_score, "rankpct": rankpct_raw_score}

            window_frame = pd.DataFrame({
                "source_id": meta_score["source_id"].to_numpy(),
                "t_rel_s": meta_score["t_end"].to_numpy(dtype=float),
                **{f"{arch_name}_{variant}_raw": values for variant, values in raw_scores.items()},
            })

            endpoints = (
                sequence[sequence["source_id"].isin(score_ids)][["source_id", "t_rel_s", "label"]]
                .reset_index(drop=True)
            )
            for variant in VARIANTS:
                raw_col = f"{arch_name}_{variant}_raw"
                endpoints[raw_col] = _align_score(
                    endpoints[["source_id", "t_rel_s"]].reset_index(drop=True), window_frame, raw_col)

            val_mask = endpoints["source_id"].isin(parts["val"]).to_numpy()
            for variant in VARIANTS:
                raw_col = f"{arch_name}_{variant}_raw"
                final_col = f"{arch_name}_{variant}"
                endpoints[final_col] = empirical_probability(
                    endpoints.loc[val_mask, raw_col].to_numpy(dtype=float),
                    endpoints[raw_col].to_numpy(dtype=float),
                )

            test_mask = endpoints["source_id"].isin(parts["test"]).to_numpy()
            coverage = {"split": split_name}
            for variant in VARIANTS:
                raw_col = f"{arch_name}_{variant}_raw"
                coverage[f"{variant}_val_row_finite_fraction"] = float(np.isfinite(
                    endpoints.loc[val_mask, raw_col].to_numpy(dtype=float)).mean())
                coverage[f"{variant}_test_row_finite_fraction"] = float(np.isfinite(
                    endpoints.loc[test_mask, raw_col].to_numpy(dtype=float)).mean())
            coverage.update({
                "train_windows": int(len(x_train)), "score_windows": int(len(x_score)),
                "train_flights": len(parts["train"]), "val_flights": len(parts["val"]),
                "test_flights": len(parts["test"]),
            })
            window_coverage.append(coverage)

            streams = last_causal_per_bucket(
                endpoints, stride_seconds=DECISION_STRIDE_S,
                columns=["source_id", "t_rel_s", "label", *score_cols],
            )

            split_dir = output / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            policies_json = {}
            for score_source in score_cols:
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
                            "split": split_name, "seed": seed, "architecture": arch_name,
                            "score_source": score_source, "decision": decision, "budget": budget_name,
                        }
                        all_metrics.append({**common, **overall})
                        all_labels.extend({**common, **row} for row in by_label)
                        all_categories.extend({**common, **row} for row in by_category)
            (split_dir / "policies.json").write_text(json.dumps(policies_json, indent=2), encoding="utf-8")

            test_window_mask = meta_score["source_id"].isin(parts["test"]).to_numpy()
            trained_relerr_test = relerr_raw_score[test_window_mask]
            trained_rankpct_test = rankpct_raw_score[test_window_mask]
            correlations = _correlation_diagnostic(
                arch, len(AE_COLS), x_train, m_train, x_score, m_score, test_window_mask,
                trained_relerr_test, trained_rankpct_test,
            )
            correlation_record = {
                "split": split_name, "seed": seed, "architecture": arch_name,
                "n_test_windows": int(test_window_mask.sum()), "correlations": correlations,
            }
            all_correlations.append(correlation_record)
            (split_dir / "correlation_diagnostic.json").write_text(
                json.dumps(_jsonable(correlation_record), indent=2, allow_nan=True), encoding="utf-8")

            for variant in VARIANTS:
                summary_rows.append({
                    "architecture": arch_name, "variant": variant, "split": split_name, "seed": seed,
                    "trained_vs_untrained_spearman":
                        correlations[variant]["trained_vs_untrained_random_init_spearman"]["spearman"],
                    "trained_vs_magnitude_spearman":
                        correlations[variant]["trained_vs_magnitude_only_spearman"]["spearman"],
                })

            # Explicit cleanup: SEAD splits carry ~200k-260k windows x 50 x 22 -- drop the
            # big per-split arrays now rather than waiting for the next loop iteration's
            # reassignment, so peak memory stays bounded to ~one split at a time instead of
            # drifting upward across the 5-split loop.
            del (x_train, m_train, meta_train, x_score, m_score, meta_score, scaled, sequence,
                 channel_mse_train, channel_valid_train, channel_mse_score, channel_valid_score,
                 channel_rel_score, window_frame, endpoints, streams, model)
            gc.collect()

        metrics = pd.DataFrame(all_metrics)
        label_metrics = pd.DataFrame(all_labels)
        category_metrics = pd.DataFrame(all_categories)
        metrics.to_csv(output / "metrics.csv", index=False)
        label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
        category_metrics.to_csv(output / "category_metrics.csv", index=False)
        (output / "correlation_diagnostics.json").write_text(
            json.dumps(_jsonable(all_correlations), indent=2, allow_nan=True), encoding="utf-8")

        gates = {"gate_a": _gate_a(holdout), "gate_b": _gate_b(metrics)}
        (output / "gates.json").write_text(
            json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8")

        files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
        artifact_manifest = {
            "artifact_schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": f"ML-16 Kol N SEAD {arch_name} magnitude-normalized post-hoc scoring "
                     "(frozen checkpoint, no retraining)",
            "source": SOURCE,
            "architecture": arch_name,
            "plan": "docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md",
            "score_sources_evaluated": score_cols,
            "relative_error_eps": RELATIVE_ERROR_EPS,
            "frozen_checkpoint_source": str(arch["frozen_run_dir"].relative_to(ROOT)).replace("\\", "/"),
            "blind_holdout_read": False,
            "blind_holdout_flights": len(holdout),
            "development_flights": int(raw["source_id"].nunique()),
            "development_source_ids_sha256": hashlib.sha256(
                "\n".join(sorted(development)).encode("utf-8")).hexdigest(),
            "evaluated_splits": sorted(folds),
            "ae_feature_columns": AE_COLS,
            "window": AE_WINDOW,
            "stride": AE_STRIDE,
            "window_coverage": window_coverage,
            "split_manifest_sha256": _sha256(SPLIT_PATH),
            "feature_table_sha256": _sha256(FEATURE_PATH),
            "silver_table_sha256": _sha256(SILVER_PATH),
            "gate_status": {name: value["status"] for name, value in gates.items()},
            "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files},
        }
        (output / "manifest.json").write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
        outputs[arch_name] = output

    summary_path = ROOT / "artifacts/ml16_kol_n/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": "docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md",
        "architectures_evaluated": list(arch_names),
        "splits_evaluated": sorted(folds),
        "correlation_summary_rows": summary_rows,
        "outputs": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in outputs.items()},
    }), indent=2, allow_nan=True), encoding="utf-8")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_matrix")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--architectures", nargs="+", default=None, choices=list(ARCHITECTURES))
    args = parser.parse_args()
    outputs = run(
        args.run_name,
        tuple(args.splits) if args.splits else None,
        tuple(args.architectures) if args.architectures else None,
    )
    for name, path in outputs.items():
        print(f"ML-16 Kol N [{name}] artifact: {path}")
        print((path / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
