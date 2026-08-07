"""split_00 normal train ile modular IF bundle'lari uret (blind holdout'a dokunmaz)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.artifacts import save_modular_iforest_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.models.modular_iforest import ALFA_MODULES, PX4_BASE_MODULES, fit_modular_iforest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    feature_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feature_root / "split_manifest.json").read_text(encoding="utf-8"))
    configs = {"alfa": ALFA_MODULES, "uav_attack": PX4_BASE_MODULES, "uav_sead": PX4_BASE_MODULES}
    for source, modules in configs.items():
        feature_path = feature_root / source / f"{source}_ml_features.parquet"
        if not feature_path.exists():
            continue
        scaler = json.loads((ROOT / "artifacts/scalers" / f"{source}_robust_scaler.json").read_text())
        cusum = json.loads((ROOT / "artifacts/cusum" / f"{source}_cusum_baseline.json").read_text())
        df = apply_scaler_params(pd.read_parquet(feature_path), scaler)
        split = manifest["sources"][source]["splits"]["split_00"]
        fitted = fit_modular_iforest(df, split, modules, seed=split["seed"])
        path = save_modular_iforest_bundle(
            fitted, ROOT / "artifacts/models" / source / "ml6_modular_iforest",
            scaler_params=scaler, cusum_baselines=cusum,
            metadata={
                "source": source,
                "model_version": "ml6_modular_iforest",
                "feature_version": "causal_cusum_v1",
                "split_id": "split_00",
                "train_flights": split["train"],
                "validation_flights": split["val"],
                "blind_holdout_used": False,
            })
        print(f"{source}: {path}")


if __name__ == "__main__":
    main()
