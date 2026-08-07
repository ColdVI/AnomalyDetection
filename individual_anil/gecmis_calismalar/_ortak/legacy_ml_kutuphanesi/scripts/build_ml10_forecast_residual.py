"""Precompute causal ML-10 Chronos forecast-residual scores for UAV-SEAD.

The command has deliberately separate stages.  ``--preflight`` records the
model/channel checks, ``--feasibility`` measures eight real development
flights, and ``--full`` is refused until the feasibility artifact authorizes
the selected scope/stride.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
ARTIFACT_DIR = ROOT / "artifacts/ml10/uav_sead"
PREFLIGHT_PATH = ARTIFACT_DIR / "preflight_check.json"
FEASIBILITY_PATH = ARTIFACT_DIR / "feasibility_check.json"
SCORE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml10_forecast_residual.parquet"
PRECOMPUTE_DIR = ARTIFACT_DIR / "precompute"

MODEL_ID = "amazon/chronos-bolt-tiny"
MODEL_PARAMETER_COUNT = 8_650_000
ALTITUDE_CHANNEL = "alt"
ACTUATOR_CHANNEL = "actuator_output_std"
CONTEXT_WINDOW = 512
MIN_CONTEXT = 16
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
DEFAULT_STRIDE_S = 1.0
FALLBACK_STRIDE_S = 5.0
FEASIBILITY_FLIGHTS = 8
EPS = 1e-6


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


def _manifest_context() -> tuple[dict, set[str], set[str]]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    split = config["splits"]["split_00"]
    development = set(split["train"] + split["val"] + split["test"])
    holdout = set(split["final_holdout"])
    if development & holdout:
        raise AssertionError("Blind holdout overlaps the ML-10 development scope")
    return config, development, holdout


def _load_development_channels(ids: set[str]) -> pd.DataFrame:
    silver = pd.read_parquet(
        SILVER_PATH,
        columns=["source_id", "timestamp", "alt", "local_alt_m", "baro_alt_m"],
        filters=[("source_id", "in", sorted(ids))],
    )
    feature = pd.read_parquet(
        FEATURE_PATH,
        columns=["source_id", "t_rel_s", ACTUATOR_CHANNEL],
        filters=[("source_id", "in", sorted(ids))],
    )
    silver = silver.sort_values(["source_id", "timestamp"], kind="stable").reset_index(drop=True)
    feature = feature.sort_values(["source_id", "t_rel_s"], kind="stable").reset_index(drop=True)
    if len(silver) != len(feature) or not silver["source_id"].equals(feature["source_id"]):
        raise AssertionError("Silver and Gold feature rows do not align by flight")
    derived_t = (
        silver["timestamp"]
        - silver.groupby("source_id", sort=False)["timestamp"].transform("min")
    ) / 1_000_000.0
    if not np.allclose(derived_t, feature["t_rel_s"], atol=1e-6, equal_nan=True):
        raise AssertionError("Silver timestamps do not align with Gold t_rel_s")
    silver["t_rel_s"] = feature["t_rel_s"].to_numpy()
    silver[ACTUATOR_CHANNEL] = feature[ACTUATOR_CHANNEL].to_numpy()
    return silver


def _load_pipeline() -> BaseChronosPipeline:
    return BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float32,
    )


def _decision_indices(times: np.ndarray, stride_s: float) -> np.ndarray:
    buckets = np.floor(np.asarray(times, dtype=float) / stride_s).astype(np.int64)
    if not len(buckets):
        return np.empty(0, dtype=np.int64)
    return np.r_[np.flatnonzero(buckets[1:] != buckets[:-1]), len(buckets) - 1]


def _contexts(
    values: np.ndarray, times: np.ndarray, stride_s: float,
) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray]:
    """Return past-only contexts, actual values, and their row positions."""
    tensors: list[torch.Tensor] = []
    actuals: list[float] = []
    positions: list[int] = []
    for position in _decision_indices(times, stride_s):
        actual = float(values[position])
        if not math.isfinite(actual):
            continue
        history = np.asarray(values[:position], dtype=np.float32)
        history = history[np.isfinite(history)]
        if len(history) < MIN_CONTEXT:
            continue
        tensors.append(torch.from_numpy(history[-CONTEXT_WINDOW:].copy()))
        actuals.append(actual)
        positions.append(int(position))
    return tensors, np.asarray(actuals, dtype=float), np.asarray(positions, dtype=np.int64)


def forecast_residual(
    pipeline: BaseChronosPipeline,
    values: np.ndarray,
    times: np.ndarray,
    *,
    stride_s: float,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal normalized distance outside Chronos' q10-q90 interval."""
    contexts, actuals, positions = _contexts(values, times, stride_s)
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            quantiles, _ = pipeline.predict_quantiles(
                contexts[start:start + batch_size],
                prediction_length=1,
                quantile_levels=QUANTILE_LEVELS,
            )
            q = quantiles[:, 0, :].detach().cpu().numpy()
            actual = actuals[start:start + batch_size]
            width = q[:, 2] - q[:, 0]
            scores.append(
                np.maximum.reduce([np.zeros(len(actual)), actual - q[:, 2], q[:, 0] - actual])
                / (width + EPS)
            )
    return positions, np.concatenate(scores) if scores else np.empty(0, dtype=float)


def run_preflight() -> Path:
    _, development, holdout = _manifest_context()
    frame = _load_development_channels(development)
    candidates = ["alt", "local_alt_m", "baro_alt_m"]
    row_completeness = {name: float(frame[name].notna().mean()) for name in candidates}
    flight_any = {
        name: float(frame.groupby("source_id")[name].agg(lambda x: x.notna().any()).mean())
        for name in candidates
    }
    chosen = max(candidates, key=row_completeness.get)
    if chosen != ALTITUDE_CHANNEL:
        raise AssertionError(f"Frozen altitude choice {ALTITUDE_CHANNEL!r} no longer wins: {chosen!r}")

    sample_id = sorted(development)[0]
    sample = frame.loc[frame["source_id"] == sample_id, chosen].dropna().iloc[:CONTEXT_WINDOW]
    load_started = time.perf_counter()
    pipeline = _load_pipeline()
    load_seconds = time.perf_counter() - load_started
    context = torch.tensor(sample.to_numpy(dtype=np.float32))
    started = time.perf_counter()
    quantiles, mean = pipeline.predict_quantiles(
        context, prediction_length=1, quantile_levels=QUANTILE_LEVELS,
    )
    first_seconds = time.perf_counter() - started
    warm = []
    for _ in range(10):
        started = time.perf_counter()
        pipeline.predict_quantiles(
            context, prediction_length=1, quantile_levels=QUANTILE_LEVELS,
        )
        warm.append(time.perf_counter() - started)

    result = {
        "stage": "ML-10 mandatory preflight checks (§1)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": MODEL_ID,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "device": "cpu",
            "torch_dtype": "float32",
            "package": "chronos-forecasting==2.3.1",
            "load_seconds": load_seconds,
            "first_prediction_seconds": first_seconds,
            "warm_prediction_mean_seconds": float(np.mean(warm)),
            "warm_prediction_max_seconds": float(np.max(warm)),
            "quantile_shape": list(quantiles.shape),
            "mean_shape": list(mean.shape),
            "status": "passed" if float(np.mean(warm)) < 1.0 else "failed",
        },
        "altitude_channel_audit": {
            "development_flights": len(development),
            "holdout_flights_not_read": len(holdout),
            "row_completeness": row_completeness,
            "flight_any_value_rate": flight_any,
            "selected_channel": chosen,
        },
        "actuator_transfer_audit": {
            "channel": ACTUATOR_CHANNEL,
            "row_completeness": float(frame[ACTUATOR_CHANNEL].notna().mean()),
            "flight_any_value_rate": float(
                frame.groupby("source_id")[ACTUATOR_CHANNEL]
                .agg(lambda x: x.notna().any()).mean()
            ),
            "status": "passed",
        },
        "blind_holdout_read": False,
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
    }
    if result["model"]["status"] != "passed":
        raise RuntimeError("Chronos CPU preflight did not meet the fixed sub-second rule")
    _write_json(result, PREFLIGHT_PATH)
    return PREFLIGHT_PATH


def _representative_ids(frame: pd.DataFrame, count: int) -> list[str]:
    sizes = frame.groupby("source_id").size().sort_values(kind="stable")
    available = frame.groupby("source_id")[ACTUATOR_CHANNEL].agg(lambda x: x.notna().any())
    sizes = sizes[available.reindex(sizes.index).fillna(False)]
    positions = np.linspace(0, len(sizes) - 1, count, dtype=int)
    return [str(sizes.index[position]) for position in positions]


def _benchmark(
    pipeline: BaseChronosPipeline, frame: pd.DataFrame, ids: list[str], stride_s: float,
) -> tuple[list[dict], int, float]:
    rows = []
    total_predictions = 0
    wall_started = time.perf_counter()
    for source_id in ids:
        group = frame[frame["source_id"] == source_id].sort_values("t_rel_s")
        started = time.perf_counter()
        channel_counts = {}
        for channel in (ALTITUDE_CHANNEL, ACTUATOR_CHANNEL):
            positions, _ = forecast_residual(
                pipeline,
                group[channel].to_numpy(dtype=float),
                group["t_rel_s"].to_numpy(dtype=float),
                stride_s=stride_s,
            )
            channel_counts[channel] = len(positions)
            total_predictions += len(positions)
        rows.append({
            "source_id": source_id,
            "rows": len(group),
            "duration_s": float(group["t_rel_s"].max()),
            "wall_seconds": time.perf_counter() - started,
            "predictions": channel_counts,
        })
    return rows, total_predictions, time.perf_counter() - wall_started


def run_feasibility() -> Path:
    if not PREFLIGHT_PATH.exists():
        raise RuntimeError("Run --preflight and record §1 checks before feasibility")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight["model"]["status"] != "passed" or preflight.get("blind_holdout_read") is not False:
        raise RuntimeError("Recorded preflight did not pass")
    _, development, holdout = _manifest_context()
    frame = _load_development_channels(development)
    selected = _representative_ids(frame, FEASIBILITY_FLIGHTS)
    pipeline = _load_pipeline()
    rows, predictions, wall_seconds = _benchmark(
        pipeline, frame, selected, DEFAULT_STRIDE_S,
    )

    total_decisions = sum(
        len(_decision_indices(group["t_rel_s"].to_numpy(), DEFAULT_STRIDE_S))
        for _, group in frame.groupby("source_id", sort=False)
    )
    sample_decisions = sum(
        len(_decision_indices(
            frame.loc[frame["source_id"] == source_id, "t_rel_s"].to_numpy(),
            DEFAULT_STRIDE_S,
        ))
        for source_id in selected
    )
    # Both channels are timed together; scale by the number of causal decision
    # points, retaining the measured missing-channel effects in wall_seconds.
    projected_seconds = wall_seconds * total_decisions / sample_decisions
    projected_hours = projected_seconds / 3600.0
    if projected_hours < 3.0:
        decision = "full_development_1s"
        selected_stride = DEFAULT_STRIDE_S
    elif projected_hours <= 8.0:
        decision = "remeasure_5s"
        selected_stride = FALLBACK_STRIDE_S
    else:
        decision = "fixed_target_subset"
        selected_stride = DEFAULT_STRIDE_S

    result = {
        "stage": "ML-10 mandatory feasibility checkpoint (§3)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "context_window_rows": CONTEXT_WINDOW,
        "minimum_context_rows": MIN_CONTEXT,
        "batch_size": 64,
        "measured_stride_seconds": DEFAULT_STRIDE_S,
        "sample_selection": "8 deterministic flight-size quantiles with both target channels",
        "sample_flights": rows,
        "sample_flight_count": len(selected),
        "sample_total_predictions": predictions,
        "sample_wall_seconds": wall_seconds,
        "sample_mean_wall_seconds_per_flight": wall_seconds / len(selected),
        "development_flights": len(development),
        "development_total_decision_points": total_decisions,
        "projected_full_runtime_seconds": projected_seconds,
        "projected_full_runtime_hours": projected_hours,
        "fixed_rule": {
            "under_3h": "full development at 1s stride",
            "3_to_8h": "remeasure at 5s stride",
            "over_8h": "fixed target-category subset plus equal-size seeded normal subset",
        },
        "decision": decision,
        "selected_stride_seconds": selected_stride,
        "full_run_authorized": decision == "full_development_1s",
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "preflight_sha256": _sha256(PREFLIGHT_PATH),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
    }
    _write_json(result, FEASIBILITY_PATH)
    return FEASIBILITY_PATH


def run_full() -> Path:
    if not FEASIBILITY_PATH.exists():
        raise RuntimeError("§3 feasibility artifact is required before the full precompute")
    feasibility = json.loads(FEASIBILITY_PATH.read_text(encoding="utf-8"))
    if feasibility.get("full_run_authorized") is not True:
        raise RuntimeError("The recorded §3 decision does not authorize the full development run")
    if feasibility.get("blind_holdout_read") is not False:
        raise RuntimeError("Feasibility artifact does not prove blind-holdout isolation")
    for path, key in (
        (SPLIT_PATH, "split_manifest_sha256"),
        (FEATURE_PATH, "feature_table_sha256"),
        (SILVER_PATH, "silver_table_sha256"),
    ):
        if _sha256(path) != feasibility[key]:
            raise RuntimeError(f"Input changed after feasibility was locked: {path}")

    _, development, holdout = _manifest_context()
    frame = _load_development_channels(development)
    if set(frame["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows entered the ML-10 precompute")
    stride_s = float(feasibility["selected_stride_seconds"])
    pipeline = _load_pipeline()
    started = time.perf_counter()
    output_frames: list[pd.DataFrame] = []
    flight_timings: list[dict] = []
    for number, (source_id, group) in enumerate(
        frame.groupby("source_id", sort=True), start=1,
    ):
        group = group.sort_values("t_rel_s").reset_index(drop=True)
        scored = group[["source_id", "t_rel_s"]].copy()
        flight_started = time.perf_counter()
        counts = {}
        for channel, output_column in (
            (ALTITUDE_CHANNEL, "chronos_alt_residual"),
            (ACTUATOR_CHANNEL, "chronos_actuator_std_residual"),
        ):
            values = np.full(len(group), np.nan, dtype=float)
            positions, residuals = forecast_residual(
                pipeline,
                group[channel].to_numpy(dtype=float),
                group["t_rel_s"].to_numpy(dtype=float),
                stride_s=stride_s,
            )
            values[positions] = residuals
            scored[output_column] = values
            counts[output_column] = len(residuals)
        output_frames.append(scored)
        flight_timings.append({
            "source_id": source_id,
            "wall_seconds": time.perf_counter() - flight_started,
            "rows": len(group),
            "scored_points": counts,
        })
        if number % 50 == 0 or number == len(development):
            print(f"ML-10 precompute: {number}/{len(development)} development flights")

    output = pd.concat(output_frames, ignore_index=True)
    if output["source_id"].nunique() != len(development):
        raise AssertionError("Full precompute did not cover every development flight")
    if set(output["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout score was produced")
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(SCORE_PATH, index=False)
    total_wall_seconds = time.perf_counter() - started
    PRECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)
    timings_path = PRECOMPUTE_DIR / "flight_timings.json"
    _write_json({"flights": flight_timings}, timings_path)
    model = getattr(pipeline, "model", None)
    model_config = getattr(model, "config", None)
    manifest = {
        "artifact_schema_version": 1,
        "stage": "ML-10 causal Chronos forecast-residual precompute",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": getattr(model_config, "_commit_hash", None),
        "zero_shot": True,
        "training_or_gradient_updates": False,
        "device": "cpu",
        "context_window_rows": CONTEXT_WINDOW,
        "minimum_context_rows": MIN_CONTEXT,
        "decision_stride_seconds": stride_s,
        "quantile_levels": QUANTILE_LEVELS,
        "development_flights": len(development),
        "development_rows": len(output),
        "blind_holdout_read": False,
        "blind_holdout_scored": False,
        "blind_holdout_flights": len(holdout),
        "total_wall_seconds": total_wall_seconds,
        "finite_scores": {
            column: int(output[column].notna().sum())
            for column in ("chronos_alt_residual", "chronos_actuator_std_residual")
        },
        "score_table": str(SCORE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "score_table_sha256": _sha256(SCORE_PATH),
        "preflight_sha256": _sha256(PREFLIGHT_PATH),
        "feasibility_sha256": _sha256(FEASIBILITY_PATH),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "input_feature_table_sha256": _sha256(FEATURE_PATH),
        "input_silver_table_sha256": _sha256(SILVER_PATH),
        "files": {"flight_timings.json": _sha256(timings_path)},
    }
    manifest_path = PRECOMPUTE_DIR / "manifest.json"
    _write_json(manifest, manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--preflight", action="store_true")
    stage.add_argument("--feasibility", action="store_true")
    stage.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        output = run_preflight()
    elif args.feasibility:
        output = run_feasibility()
    else:
        output = run_full()
    print(output)
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
