"""Leakage-controlled RflyMAD loading, scaling, windowing and score alignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adsb.windowing import build_windows
from rfly_dl.config import FEATURE_COLUMNS, MAX_GAP_S, SCALE_CLIP, WINDOW, WINDOW_STRIDE

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/gold/ml_features/rflymad/rflymad_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/rflymad_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def load_contract(
    selected_splits: tuple[str, ...],
) -> tuple[dict[str, dict], set[str], set[str]]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    available = manifest["sources"]["rflymad"]["splits"]
    unknown = set(selected_splits) - set(available)
    if unknown:
        raise ValueError(f"Unknown RflyMAD splits: {sorted(unknown)}")
    folds = {name: available[name] for name in selected_splits}
    development = set().union(
        *(
            set(split[part])
            for split in folds.values()
            for part in ("train", "val", "test")
        )
    )
    holdout = set().union(
        *(set(split.get("final_holdout", [])) for split in folds.values())
    )
    if development & holdout:
        raise AssertionError("Historical holdout overlaps development")
    return folds, development, holdout


def load_development(
    folds: dict[str, dict], development: set[str], holdout: set[str]
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], set[str]]:
    silver_columns = [
        "source_id",
        "fault_onset_s",
        "fault_end_s",
        "fault_interval_source",
    ]
    silver = pd.read_parquet(
        SILVER_PATH,
        filters=[("source_id", "in", sorted(development))],
        columns=silver_columns,
    )
    if set(silver["source_id"].unique()) & holdout:
        raise AssertionError("Holdout rows entered the Rfly DL Silver read")
    invalid = set(
        silver.loc[
            silver["fault_interval_source"].eq("rfly_ctrl_lxl_no_active_fault"),
            "source_id",
        ].unique()
    )
    valid_development = development - invalid

    for split in folds.values():
        for part in ("train", "val", "test"):
            split[part] = sorted(set(split[part]) - invalid)

    columns = ["t_rel_s", "source_id", "label", *FEATURE_COLUMNS]
    raw = pd.read_parquet(
        GOLD_PATH,
        filters=[("source_id", "in", sorted(valid_development))],
        columns=columns,
    )
    if set(raw["source_id"].unique()) & holdout:
        raise AssertionError("Holdout rows entered the Rfly DL Gold read")
    if set(raw["source_id"].unique()) & invalid:
        raise AssertionError("Invalid no-active-fault rows entered evaluation")
    missing = set(FEATURE_COLUMNS) - set(raw.columns)
    if missing:
        raise KeyError(f"Missing frozen Rfly DL features: {sorted(missing)}")

    intervals: dict[str, tuple[float, float]] = {}
    for source_id, group in silver[
        silver["source_id"].isin(valid_development)
    ].groupby("source_id", sort=False):
        first = group.iloc[0]
        if first["fault_interval_source"] == "rfly_ctrl_lxl":
            onset, end = float(first["fault_onset_s"]), float(first["fault_end_s"])
            if not np.isfinite(onset) or not np.isfinite(end) or end < onset:
                raise ValueError(f"Invalid fault interval for {source_id}")
            intervals[str(source_id)] = (onset, end)
    anomaly_ids = set(raw.loc[raw["label"].ne("normal"), "source_id"].unique())
    missing_truth = anomaly_ids - set(intervals)
    if missing_truth:
        raise AssertionError(
            f"Anomalous development flights lack interval truth: {len(missing_truth)}"
        )
    return raw.sort_values(["source_id", "t_rel_s"]).reset_index(drop=True), intervals, invalid


def fit_robust_scaler(train: pd.DataFrame) -> dict:
    parameters: dict[str, dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"Training feature is entirely missing: {column}")
        center = float(np.median(finite))
        q1, q3 = np.quantile(finite, [0.25, 0.75])
        scale = float(q3 - q1)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        parameters[column] = {"center": center, "scale": scale}
    return {
        "kind": "train_normal_robust_scaler",
        "clip": SCALE_CLIP,
        "columns": parameters,
    }


def apply_robust_scaler(raw: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    scaled = raw.copy()
    for column, parameters in scaler["columns"].items():
        values = pd.to_numeric(scaled[column], errors="coerce").astype(float)
        values = (values - parameters["center"]) / parameters["scale"]
        finite = values.notna()
        values.loc[finite] = values.loc[finite].clip(-SCALE_CLIP, SCALE_CLIP)
        scaled[column] = values.astype(np.float32)
    return scaled


def make_windows(frame: pd.DataFrame):
    return build_windows(
        frame,
        list(FEATURE_COLUMNS),
        window=WINDOW,
        stride=WINDOW_STRIDE,
        max_gap_s=MAX_GAP_S,
        flight_id_col="source_id",
        time_col="t_rel_s",
    )


def empirical_probability(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if not len(finite):
        raise ValueError("Validation score reference is empty")
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    result[valid] = (
        np.searchsorted(finite, values[valid], side="right") + 0.5
    ) / (len(finite) + 1.0)
    return result


def align_window_scores(
    base: pd.DataFrame, meta: pd.DataFrame, scores: np.ndarray, score_name: str
) -> pd.DataFrame:
    if len(meta) != len(scores):
        raise ValueError("Window metadata and scores have different lengths")
    score_frame = meta[["flight_id", "t_end"]].copy()
    score_frame[score_name] = np.asarray(scores, dtype=float)
    frames: list[pd.DataFrame] = []
    for source_id, group in base.groupby("source_id", sort=False):
        target = group[["source_id", "t_rel_s", "label"]].sort_values("t_rel_s").copy()
        window_group = score_frame[score_frame["flight_id"].eq(source_id)].sort_values(
            "t_end"
        )
        aligned = pd.merge_asof(
            target,
            window_group[["t_end", score_name]],
            left_on="t_rel_s",
            right_on="t_end",
            direction="backward",
            allow_exact_matches=True,
        )
        frames.append(aligned.drop(columns=["t_end"]))
    return pd.concat(frames, ignore_index=True)


def feature_completeness(raw: pd.DataFrame) -> pd.DataFrame:
    group = np.where(raw["label"].eq("normal"), "normal", "anomaly")
    rows = []
    for column in FEATURE_COLUMNS:
        rows.append(
            {
                "feature": column,
                "all_row_completeness": float(raw[column].notna().mean()),
                "normal_row_completeness": float(
                    raw.loc[group == "normal", column].notna().mean()
                ),
                "anomaly_row_completeness": float(
                    raw.loc[group == "anomaly", column].notna().mean()
                ),
            }
        )
    return pd.DataFrame(rows)
