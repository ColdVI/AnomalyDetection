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
import json
import logging
from pathlib import Path

import pandas as pd

from src.ml.data.scaling import fit_scaler_params, write_scaler_params
from src.ml.data.splits import (
    SPLIT_QUOTAS,
    assert_no_flight_overlap,
    build_split_manifest,
    write_manifest,
)
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


def rebuild_uav_sead_with_frozen_manifest(manifest_path: Path) -> None:
    """SEAD feature/artifact'lerini mevcut split'i degistirmeden yeniden kur.

    ML-9 yeni split uretmeyi yasaklar. Bu yol manifesti yalniz train kimliklerini
    secmek icin okur; dosyayi yazmaz ve final_holdout'u fit/skor akisina sokmaz.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    split0 = config["splits"]["split_00"]
    train_ids = set(split0["train"])
    holdout_ids = set(split0["final_holdout"])
    if train_ids & holdout_ids:
        raise AssertionError("Frozen manifestte train/final_holdout kesisiyor")

    silver = pd.read_parquet(SILVER_DIR / "uav_sead_silver.parquet")
    available_ids = set(silver["source_id"].unique())
    missing_train = train_ids - available_ids
    if missing_train:
        raise ValueError(f"Frozen manifest train ucuslari Silver'da eksik: {sorted(missing_train)}")

    provisional = build_px4_features(silver)
    train = provisional[provisional["source_id"].isin(train_ids)]
    baselines = fit_cusum_baselines(train, PX4_CUSUM_COLUMNS)
    write_cusum_baselines(baselines, CUSUM_DIR / "uav_sead_cusum_baseline.json")

    features = build_px4_features(silver, cusum_baselines=baselines)
    _write(features, "uav_sead")
    scaler = fit_scaler_params(
        features[features["source_id"].isin(train_ids)], px4_feature_columns(features))
    write_scaler_params(scaler, SCALER_DIR / "uav_sead_robust_scaler.json")
    logger.info(
        "UAV-SEAD frozen-manifest rebuild tamam: %d train fit, %d holdout kapali; manifest yazilmadi",
        len(train_ids), len(holdout_ids),
    )


def rebuild_only_uav_sead_with_frozen_holdout(previous_manifest_path: Path) -> dict:
    """ML-14 refresh path: rewrite only SEAD-derived outputs.

    ALFA/UAV Attack feature parquet files are read only so their split content can
    remain in the single global manifest, but their bytes and scalers are not
    rewritten.
    """
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    previous_sources = previous_manifest["sources"]
    old_sead = previous_sources["uav_sead"]

    tables: dict[str, pd.DataFrame] = {}
    for source in ("alfa", "uav_attack"):
        path = OUT_DIR / source / f"{source}_ml_features.parquet"
        if path.exists() and source in previous_sources:
            tables[source] = pd.read_parquet(path)

    silver = pd.read_parquet(SILVER_DIR / "uav_sead_silver.parquet")
    provisional = build_px4_features(silver)
    provisional_tables = {**tables, "uav_sead": provisional}
    provisional_manifest = build_split_manifest(
        provisional_tables,
        frozen_holdout={"uav_sead": old_sead},
    )
    provisional_split = provisional_manifest["sources"]["uav_sead"]["splits"]["split_00"]
    labels = provisional_manifest["sources"]["uav_sead"]["flight_labels"]
    development = (
        set(provisional_split["train"])
        | set(provisional_split["val"])
        | set(provisional_split["test"])
    )
    development_normal = [
        source_id for source_id in development
        if labels.get(source_id) == "normal"
    ]
    ml14_quota = max(30, round(0.15 * len(development_normal)))
    quotas = dict(SPLIT_QUOTAS)
    quotas["uav_sead"] = (ml14_quota, ml14_quota)

    final_manifest = build_split_manifest(
        provisional_tables,
        quotas=quotas,
        frozen_holdout={"uav_sead": old_sead},
    )
    final_split = final_manifest["sources"]["uav_sead"]["splits"]["split_00"]
    train_ids = set(final_split["train"])
    train = provisional[provisional["source_id"].isin(train_ids)]
    baselines = fit_cusum_baselines(train, PX4_CUSUM_COLUMNS)
    write_cusum_baselines(baselines, CUSUM_DIR / "uav_sead_cusum_baseline.json")

    features = build_px4_features(silver, cusum_baselines=baselines)
    _write(features, "uav_sead")
    final_tables = {**tables, "uav_sead": features}
    final_manifest = build_split_manifest(
        final_tables,
        quotas=quotas,
        frozen_holdout={"uav_sead": old_sead},
    )
    for source, entry in final_manifest["sources"].items():
        for split in entry["splits"].values():
            assert_no_flight_overlap(split)
    write_manifest(final_manifest, OUT_DIR / "split_manifest.json")

    final_split = final_manifest["sources"]["uav_sead"]["splits"]["split_00"]
    scaler = fit_scaler_params(
        features[features["source_id"].isin(final_split["train"])],
        px4_feature_columns(features),
    )
    write_scaler_params(scaler, SCALER_DIR / "uav_sead_robust_scaler.json")

    return {
        "ml14_split_quota": ml14_quota,
        "development_normal_flights": len(development_normal),
        "uav_sead_feature_rows": int(len(features)),
        "uav_sead_feature_flights": int(features["source_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver -> ML feature tablolari (ML-0)")
    parser.add_argument("--skip-uav-sead", action="store_true",
                        help="UAV-SEAD Silver'i yoksa/istenmiyorsa atla")
    parser.add_argument(
        "--uav-sead-only-frozen-manifest", action="store_true",
        help="Yalniz SEAD'i mevcut split_manifest ile yeniden kur; split uretme/yazma",
    )
    parser.add_argument(
        "--only-uav-sead", action="store_true",
        help="ML-14: yalniz SEAD Silver/feature/split/scaler/CUSUM yenile",
    )
    parser.add_argument(
        "--frozen-holdout-manifest",
        default="artifacts/ml14/uav_sead/previous_split_manifest.json",
        help="ML-14 eski split manifest kopyasi",
    )
    args = parser.parse_args()

    if args.uav_sead_only_frozen_manifest:
        rebuild_uav_sead_with_frozen_manifest(OUT_DIR / "split_manifest.json")
        return
    if args.only_uav_sead:
        summary = rebuild_only_uav_sead_with_frozen_holdout(
            Path(args.frozen_holdout_manifest)
        )
        logger.info("ML-14 only-uav-sead rebuild summary: %s", summary)
        return

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

    rfly_path = SILVER_DIR / "rflymad_silver.parquet"
    if rfly_path.exists():
        rfly_silver = pd.read_parquet(rfly_path)
        silver_tables["rflymad"] = rfly_silver
        tables["rflymad"] = build_px4_features(rfly_silver)
        col_fns["rflymad"] = px4_feature_columns
        cusum_cols["rflymad"] = PX4_CUSUM_COLUMNS
    else:
        logger.info("RflyMAD Silver yok (%s) -- RFLY-0 indirme/parse sonrasi eklenir", rfly_path)

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
