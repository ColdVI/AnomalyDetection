"""Yeni SEAD fiziksel-residual modullerini development deneyi icin paketle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.artifacts import save_modular_iforest_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.models.modular_iforest import PX4_ML7_CANDIDATE_MODULES, fit_modular_iforest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    feature_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feature_root / "split_manifest.json").read_text(encoding="utf-8"))
    split = manifest["sources"]["uav_sead"]["splits"]["split_00"]
    scaler = json.loads((ROOT / "artifacts/scalers/uav_sead_robust_scaler.json").read_text())
    cusum = json.loads((ROOT / "artifacts/cusum/uav_sead_cusum_baseline.json").read_text())
    features = pd.read_parquet(
        feature_root / "uav_sead/uav_sead_ml_features.parquet")
    scaled = apply_scaler_params(features, scaler)
    fitted = fit_modular_iforest(
        scaled, split, PX4_ML7_CANDIDATE_MODULES, seed=split["seed"])
    path = save_modular_iforest_bundle(
        fitted, ROOT / "artifacts/models/uav_sead/ml7_candidate_iforest",
        scaler_params=scaler, cusum_baselines=cusum,
        metadata={
            "source": "uav_sead",
            "model_version": "ml7_candidate_iforest",
            "feature_version": "physical_residuals_v2",
            "split_id": "split_00",
            "train_flights": split["train"],
            "validation_flights": split["val"],
            "blind_holdout_used": False,
            "candidate_only": True,
        })
    print(path)


if __name__ == "__main__":
    main()
