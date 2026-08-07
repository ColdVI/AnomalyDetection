"""ML-16 Chronos residual-channel expansion scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
ARTIFACT_DIR = ROOT / "artifacts/ml16/uav_sead"
PREFLIGHT_PATH = ARTIFACT_DIR / "chronos_channel_preflight.json"

MODEL_ID = "amazon/chronos-bolt-tiny"
MODEL_REVISION_POLICY = "reuse ML-10 pinned model/revision"
DECISION_STRIDE_S = 1.0
CONTEXT_WINDOW = 512
MIN_CONTEXT = 16
COMPLETENESS_FLOOR = 0.99

CHANNEL_CANDIDATES = {
    "velocity_position_xy": ("gps_speed_calc_mps", "vel_m_s"),
    "position_z": ("hgt_test_ratio",),
    "actuator_reference": ("actuator_output_std",),
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


def choose_channel_by_completeness(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    completeness_floor: float = COMPLETENESS_FLOOR,
) -> dict:
    rows = []
    for column in candidates:
        if column not in frame.columns:
            rows.append({"channel": column, "row_completeness": 0.0, "available": False})
            continue
        rows.append({
            "channel": column,
            "row_completeness": float(frame[column].notna().mean()),
            "available": True,
        })
    best = max(rows, key=lambda row: (row["row_completeness"], row["channel"]))
    return {
        "candidates": rows,
        "selected_channel": best["channel"] if best["row_completeness"] >= completeness_floor else None,
        "selected_row_completeness": best["row_completeness"],
        "passed": bool(best["row_completeness"] >= completeness_floor),
    }


def past_context_positions(
    values: np.ndarray,
    times: np.ndarray,
    *,
    stride_s: float = DECISION_STRIDE_S,
    min_context: int = MIN_CONTEXT,
) -> np.ndarray:
    buckets = np.floor(np.asarray(times, dtype=float) / stride_s).astype(np.int64)
    if not len(buckets):
        return np.empty(0, dtype=np.int64)
    decision_positions = np.r_[np.flatnonzero(buckets[1:] != buckets[:-1]), len(buckets) - 1]
    usable = []
    values = np.asarray(values, dtype=float)
    for position in decision_positions:
        history = values[:position]
        if np.isfinite(history).sum() >= min_context and np.isfinite(values[position]):
            usable.append(int(position))
    return np.asarray(usable, dtype=np.int64)


def _manifest_scope() -> tuple[dict, set[str], set[str]]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    split = config["splits"]["split_00"]
    development = set(split["train"] + split["val"] + split["test"])
    holdout = set(split["final_holdout"])
    if development & holdout:
        raise AssertionError("Blind holdout overlaps ML-16 Chronos development scope")
    return config, development, holdout


def run_preflight() -> Path:
    config, development, holdout = _manifest_scope()
    wanted = sorted({column for values in CHANNEL_CANDIDATES.values() for column in values})
    columns = ["source_id", "t_rel_s", *wanted]
    available_columns = set(pd.read_parquet(FEATURE_PATH, columns=None).columns)
    columns = [column for column in columns if column in available_columns]
    frame = pd.read_parquet(
        FEATURE_PATH,
        columns=columns,
        filters=[("source_id", "in", sorted(development))],
    )
    if set(frame["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows entered ML-16 Chronos preflight")

    selections = {
        name: choose_channel_by_completeness(frame, candidates)
        for name, candidates in CHANNEL_CANDIDATES.items()
    }
    result = {
        "stage": "ML-16 Chronos channel preflight scaffold",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(item["passed"] for item in selections.values()) else "failed",
        "model_id": MODEL_ID,
        "model_revision_policy": MODEL_REVISION_POLICY,
        "decision_stride_seconds": DECISION_STRIDE_S,
        "context_window": CONTEXT_WINDOW,
        "minimum_context": MIN_CONTEXT,
        "completeness_floor": COMPLETENESS_FLOOR,
        "channel_selections": selections,
        "development_flights": int(len(set(config["flight_labels"]) - holdout)),
        "blind_holdout_read": False,
        "blind_holdout_flights": int(len(holdout)),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
    }
    _write_json(result, PREFLIGHT_PATH)
    return PREFLIGHT_PATH


def run_feasibility() -> Path:
    raise RuntimeError("ML-16 Chronos feasibility is locked until ML-14 rebuild completes")


def run_full() -> Path:
    raise RuntimeError("ML-16 Chronos full precompute is locked until ML-15 completes")


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
