"""ML-0 orkestratoru: Silver -> feature tablolari + split manifest + scaler.

Cikti yerlesimi (FableChat/PIPELINE_PLAN karari -- Silver'a ASLA yazilmaz,
feature mantigi degisirse bu tablolar yeniden uretilir):

    data/gold/ml_features/alfa/alfa_ml_features.parquet
    data/gold/ml_features/uav_attack/uav_attack_ml_features.parquet
    data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet   (varsa)
    data/gold/ml_features/split_manifest.json
    artifacts/scalers/<source>_robust_scaler.json   (split_00 train'inde fit)
    artifacts/cusum/<source>_cusum_baseline.json    (split_00 train'inde fit)

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
from src.ml.features.alfa_features import CUSUM_SOURCE_COLUMNS as ALFA_CUSUM_COLUMNS
from src.ml.features.alfa_features import build_alfa_features
from src.ml.features.alfa_features import feature_columns as alfa_feature_columns
from src.ml.features.temporal import fit_cusum_baselines, write_cusum_baselines
from src.ml.features.uav_attack_features import CUSUM_SOURCE_COLUMNS as PX4_CUSUM_COLUMNS
from src.ml.features.uav_attack_features import build_px4_features, build_uav_attack_features
from src.ml.features.uav_attack_features import feature_columns as px4_feature_columns

logger = logging.getLogger(__name__)

SILVER_DIR = Path("data/silver")
OUT_DIR = Path("data/gold/ml_features")
SCALER_DIR = Path("artifacts/scalers")
CUSUM_DIR = Path("artifacts/cusum")


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

    # CUSUM baseline'i train-normal veriden ogrenildigi icin iki gecis gerekir:
    # (1) nedensel default ile etiket/source_id tablosu ve split, (2) train'de
    # fit edilen sabit baseline'larla nihai feature tablolari.
    silver_tables: dict[str, pd.DataFrame] = {}
    tables: dict[str, pd.DataFrame] = {}
    col_fns: dict[str, callable] = {}
    cusum_cols: dict[str, list[str]] = {}

    alfa_silver = pd.read_parquet(SILVER_DIR / "alfa_silver.parquet")
    rosbag_path = SILVER_DIR / "alfa_rosbag_silver.parquet"
    if rosbag_path.exists():
        # ML-4: processed'e girmemis raw rosbag ucuslari (parse_alfa_rosbag.py) --
        # ayni Silver semasi, ek ucuslar (5 normal + 2 engine_fault).
        rosbag_silver = pd.read_parquet(rosbag_path)
        alfa_silver = pd.concat([alfa_silver, rosbag_silver], ignore_index=True, sort=False)
        logger.info("ALFA: +%d rosbag satiri (%d ek ucus) eklendi",
                    len(rosbag_silver), rosbag_silver["source_id"].nunique())
    silver_tables["alfa"] = alfa_silver
    tables["alfa"] = build_alfa_features(alfa_silver)
    col_fns["alfa"] = alfa_feature_columns
    cusum_cols["alfa"] = ALFA_CUSUM_COLUMNS

    attack_silver = pd.read_parquet(SILVER_DIR / "uav_attack_silver.parquet")
    silver_tables["uav_attack"] = attack_silver
    tables["uav_attack"] = build_uav_attack_features(attack_silver)
    col_fns["uav_attack"] = px4_feature_columns
    cusum_cols["uav_attack"] = PX4_CUSUM_COLUMNS

    sead_path = SILVER_DIR / "uav_sead_silver.parquet"
    if not args.skip_uav_sead and sead_path.exists():
        sead_silver = pd.read_parquet(sead_path)
        silver_tables["uav_sead"] = sead_silver
        tables["uav_sead"] = build_px4_features(sead_silver)
        col_fns["uav_sead"] = px4_feature_columns
        cusum_cols["uav_sead"] = PX4_CUSUM_COLUMNS
    else:
        logger.warning("UAV-SEAD Silver yok/atlandi (%s) -- leave-dataset-out icin sonra eklenebilir", sead_path)

    # Provisional manifest yalnizca ucus kimligi/etiketi icin kullanilir.
    manifest = build_split_manifest(tables)

    baselines: dict[str, dict] = {}
    for source, df in tables.items():
        split0 = manifest["sources"][source]["splits"]["split_00"]
        train_df = df[df["source_id"].isin(split0["train"])]
        params = fit_cusum_baselines(train_df, cusum_cols[source])
        baselines[source] = params
        write_cusum_baselines(params, CUSUM_DIR / f"{source}_cusum_baseline.json")

    # Nihai feature tablolarini sabit train-normal CUSUM baseline'lariyla kur.
    tables["alfa"] = build_alfa_features(
        silver_tables["alfa"], cusum_baselines=baselines["alfa"])
    tables["uav_attack"] = build_uav_attack_features(
        silver_tables["uav_attack"], cusum_baselines=baselines["uav_attack"])
    if "uav_sead" in silver_tables:
        tables["uav_sead"] = build_px4_features(
            silver_tables["uav_sead"], cusum_baselines=baselines["uav_sead"])

    for source, df in tables.items():
        _write(df, source)

    # Etiket/split degismemeli; nihai artifact manifestini final tablodan yaz.
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

    logger.info("ML-0 tamam: %d kaynak, manifest + scaler + causal CUSUM baseline hazir",
                len(tables))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
