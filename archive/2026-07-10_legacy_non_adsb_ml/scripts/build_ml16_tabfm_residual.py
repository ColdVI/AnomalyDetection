"""ML-16 TabFM cross-feature residual preflight and precompute scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.evaluation.score_fusion import last_causal_per_bucket
from src.ml.features.uav_attack_features import feature_columns

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
ARTIFACT_DIR = ROOT / "artifacts/ml16/uav_sead"
PREFLIGHT_PATH = ARTIFACT_DIR / "tabfm_preflight.json"

TABFM_PACKAGE = "tabfm==1.0.0"
TABFM_MODEL_ID = "google/tabfm-1.0.0-pytorch"
TARGET_CHANNELS = ("actuator_output_std", "hgt_test_ratio")
N_CONTEXT = 4096
PREFLIGHT_ROWS = 1000
DECISION_STRIDE_S = 1.0
RUNTIME_LIMIT_HOURS = 3.0
NON_FEATURE_COLUMNS = {
    "source_id", "t_rel_s", "label", "timestamp", "flight_label",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def target_family_prefix(target: str) -> str:
    parts = target.split("_")
    if len(parts) <= 1:
        return target
    return "_".join(parts[:-1]) + "_"


def excluded_family_columns(columns: list[str] | tuple[str, ...], target: str) -> list[str]:
    prefix = target_family_prefix(target)
    return sorted(column for column in columns if column == target or column.startswith(prefix))


def predictor_columns(columns: list[str] | tuple[str, ...], target: str) -> list[str]:
    excluded = set(excluded_family_columns(columns, target)) | NON_FEATURE_COLUMNS
    return [column for column in columns if column not in excluded]


def deterministic_context_indices(n_rows: int, n_context: int, seed: int) -> np.ndarray:
    if n_rows < 0 or n_context <= 0:
        raise ValueError("n_rows must be non-negative and n_context positive")
    if n_rows <= n_context:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n_context, replace=False).astype(np.int64))


def _manifest_scope() -> tuple[dict, set[str], set[str]]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    split = config["splits"]["split_00"]
    development = set(split["train"] + split["val"] + split["test"])
    holdout = set(split["final_holdout"])
    if development & holdout:
        raise AssertionError("Blind holdout overlaps ML-16 development scope")
    return config, development, holdout


def _load_preflight_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict, set[str]]:
    config, development, holdout = _manifest_scope()
    split = config["splits"]["split_00"]
    columns = sorted(set(["source_id", "t_rel_s", "label", *TARGET_CHANNELS]))
    probe = pd.read_parquet(FEATURE_PATH, columns=None)
    all_features = feature_columns(probe)
    columns = sorted(set(columns + all_features))
    frame = pd.read_parquet(
        FEATURE_PATH,
        columns=columns,
        filters=[("source_id", "in", sorted(development))],
    )
    if set(frame["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows entered TabFM preflight")
    train = frame[frame["source_id"].isin(split["train"])].copy()
    decisions = last_causal_per_bucket(
        frame,
        stride_seconds=DECISION_STRIDE_S,
        columns=columns,
    )
    return train, decisions, config, holdout


def _prepared_matrix(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    *,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    selected = frame[predictors + [target]].replace([np.inf, -np.inf], np.nan)
    selected = selected.dropna(axis=0, subset=[target])
    x = selected[predictors]
    if fill_values is None:
        fill_values = x.median(numeric_only=True).reindex(predictors).fillna(0.0)
    x = x.fillna(fill_values).fillna(0.0)
    return x, selected[target]


def _load_tabfm_regressor(seed: int):
    from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

    model = tabfm_v1_0_0_pytorch.load(model_type="regression", device="cpu")
    return TabFMRegressor(
        model,
        max_num_rows=N_CONTEXT,
        random_state=seed,
        batch_size=1,
        verbose=False,
    )


def run_preflight(seed: int = 0) -> Path:
    import importlib.metadata as metadata

    started = time.perf_counter()
    train, decisions, config, holdout = _load_preflight_frames()
    package_version = metadata.version("tabfm")
    target = TARGET_CHANNELS[0]
    predictors = predictor_columns(feature_columns(train), target)
    context_x, context_y = _prepared_matrix(train, target, predictors)
    fill_values = context_x.median(numeric_only=True).reindex(predictors).fillna(0.0)
    eval_x, eval_y = _prepared_matrix(
        decisions, target, predictors, fill_values=fill_values,
    )
    context_idx = deterministic_context_indices(len(context_x), N_CONTEXT, seed)
    eval_count = min(PREFLIGHT_ROWS, len(eval_x))

    context_x = context_x.iloc[context_idx]
    context_y = context_y.iloc[context_idx]
    eval_x = eval_x.iloc[:eval_count]

    load_started = time.perf_counter()
    status = "passed"
    error: str | None = None
    model_load_seconds = np.nan
    fit_seconds = np.nan
    predict_seconds = np.nan
    prediction_shape: list[int] = []
    try:
        if len(context_idx) < N_CONTEXT:
            raise RuntimeError("Not enough finite train-normal target rows for TabFM preflight")
        if eval_count < PREFLIGHT_ROWS:
            raise RuntimeError("Not enough finite one-second target rows for TabFM preflight")
        regressor = _load_tabfm_regressor(seed)
        model_load_seconds = time.perf_counter() - load_started
        fit_started = time.perf_counter()
        regressor.fit(context_x, context_y)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = regressor.predict(eval_x)
        predict_seconds = time.perf_counter() - predict_started
        prediction_shape = list(np.asarray(predictions).shape)
    except Exception as exc:  # pragma: no cover - exercised by environment failures.
        status = "failed"
        error = repr(exc)

    decision_rows = int(len(decisions))
    target_count = len(TARGET_CHANNELS)
    if np.isfinite(predict_seconds):
        projected_seconds = float((fit_seconds + predict_seconds) * target_count * decision_rows / eval_count)
    else:
        projected_seconds = np.nan
    projected_hours = float(projected_seconds / 3600.0) if np.isfinite(projected_seconds) else np.nan
    runtime_passed = bool(np.isfinite(projected_hours) and projected_hours < RUNTIME_LIMIT_HOURS)
    if status == "passed" and not runtime_passed:
        status = "failed"

    result = {
        "stage": "ML-16 TabFM preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "package": TABFM_PACKAGE,
        "package_version": package_version,
        "model_id": TABFM_MODEL_ID,
        "target_channels": list(TARGET_CHANNELS),
        "timed_target": target,
        "context_rows": int(len(context_x)),
        "preflight_rows": int(eval_count),
        "n_context": N_CONTEXT,
        "seed": int(seed),
        "family_exclusion": {
            channel: excluded_family_columns(feature_columns(train), channel)
            for channel in TARGET_CHANNELS
        },
        "predictor_count": len(predictors),
        "predictor_imputation": "train-normal median, then 0 for all-missing predictors",
        "model_load_seconds": float(model_load_seconds) if np.isfinite(model_load_seconds) else None,
        "fit_seconds": float(fit_seconds) if np.isfinite(fit_seconds) else None,
        "predict_seconds": float(predict_seconds) if np.isfinite(predict_seconds) else None,
        "prediction_shape": prediction_shape,
        "development_flights": int(len(set(config["flight_labels"]) - holdout)),
        "blind_holdout_read": False,
        "blind_holdout_flights": int(len(holdout)),
        "development_decision_rows": decision_rows,
        "projection": {
            "target_count": target_count,
            "projected_full_runtime_seconds": projected_seconds if np.isfinite(projected_seconds) else None,
            "projected_full_runtime_hours": projected_hours if np.isfinite(projected_hours) else None,
            "runtime_limit_hours": RUNTIME_LIMIT_HOURS,
            "under_3h_rule_passed": runtime_passed,
            "decision": "tabfm_enabled" if runtime_passed else "tabfm_not_feasible_in_this_environment",
        },
        "wall_seconds": float(time.perf_counter() - started),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
    }
    _write_json(result, PREFLIGHT_PATH)
    return PREFLIGHT_PATH


def run_full() -> Path:
    raise RuntimeError("ML-16 TabFM full precompute is locked until ML-15 completes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--preflight", action="store_true")
    stage.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.preflight:
        output = run_preflight(seed=args.seed)
    else:
        output = run_full()
    print(output)
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
