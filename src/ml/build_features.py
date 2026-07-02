"""ML-0 orkestratoru: Silver -> feature tablolari + split manifest + scaler.

Cikti yerlesimi (FableChat/PIPELINE_PLAN karari -- Silver'a ASLA yazilmaz,
feature mantigi degisirse bu tablolar yeniden uretilir):

    data/gold/ml_features/alfa/alfa_ml_features.parquet
    data/gold/ml_features/uav_attack/uav_attack_ml_features.parquet
    data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet   (varsa)
    data/gold/ml_features/split_manifest.json
    artifacts/scalers/<source>_robust_scaler.json   (split_00 train'inde fit)

Kullanim:
    python -m src.ml.build_features [--skip-uav-sead]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.ml.data.scaling import fit_scaler_params, write_scaler_params
from src.ml.data.splits import assert_no_flight_overlap, build_split_manifest, write_manifest
from src.ml.features.alfa_features import build_alfa_features
from src.ml.features.alfa_features import feature_columns as alfa_feature_columns
from src.ml.features.uav_attack_features import build_px4_features, build_uav_attack_features
from src.ml.features.uav_attack_features import feature_columns as px4_feature_columns

logger = logging.getLogger(__name__)

SILVER_DIR = Path("data/silver")
OUT_DIR = Path("data/gold/ml_features")
SCALER_DIR = Path("artifacts/scalers")


def _write(df: pd.DataFrame, source: str) -> Path:
    out = OUT_DIR / source / f"{source}_ml_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("%s: %d satir x %d kolon -> %s", source, len(df), df.shape[1], out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver -> ML feature tablolari (ML-0)")
    parser.add_argument("--skip-uav-sead", action="store_true",
                        help="UAV-SEAD Silver'i yoksa/istenmiyorsa atla")
    args = parser.parse_args()

    tables: dict[str, pd.DataFrame] = {}
    col_fns: dict[str, callable] = {}

    alfa_silver = pd.read_parquet(SILVER_DIR / "alfa_silver.parquet")
    tables["alfa"] = build_alfa_features(alfa_silver)
    col_fns["alfa"] = alfa_feature_columns

    attack_silver = pd.read_parquet(SILVER_DIR / "uav_attack_silver.parquet")
    tables["uav_attack"] = build_uav_attack_features(attack_silver)
    col_fns["uav_attack"] = px4_feature_columns

    sead_path = SILVER_DIR / "uav_sead_silver.parquet"
    if not args.skip_uav_sead and sead_path.exists():
        sead_silver = pd.read_parquet(sead_path)
        tables["uav_sead"] = build_px4_features(sead_silver)
        col_fns["uav_sead"] = px4_feature_columns
    else:
        logger.warning("UAV-SEAD Silver yok/atlandi (%s) -- leave-dataset-out icin sonra eklenebilir", sead_path)

    for source, df in tables.items():
        _write(df, source)

    manifest = build_split_manifest(tables)
    for source, entry in manifest["sources"].items():
        for split in entry["splits"].values():
            assert_no_flight_overlap(split)
    write_manifest(manifest, OUT_DIR / "split_manifest.json")

    # Scaler: her kaynagin split_00 train (normal) ucuslarinda fit edilir.
    for source, df in tables.items():
        split0 = manifest["sources"][source]["splits"]["split_00"]
        train_df = df[df["source_id"].isin(split0["train"])]
        params = fit_scaler_params(train_df, col_fns[source](df))
        write_scaler_params(params, SCALER_DIR / f"{source}_robust_scaler.json")

    logger.info("ML-0 tamam: %d kaynak, manifest + scaler'lar hazir", len(tables))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
