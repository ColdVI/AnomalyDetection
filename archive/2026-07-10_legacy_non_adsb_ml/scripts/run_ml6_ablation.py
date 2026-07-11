"""ML-6: veri buyutme ve session-split etkisini 2x2 ayir.

Kosullar: 179/349 ucus x flight/session split. Blind final holdout hicbir
metrige girmez; yalniz development anomaly seti skorlanir. Her hucre kendi
train scaler'ini ve causal CUSUM baseline'ini yeniden fit eder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ml.data.scaling import apply_scaler_params, fit_scaler_params
from src.ml.data.splits import flight_label_table, make_group_split
from src.ml.features.temporal import cusum, cusum_kwargs, fit_cusum_baselines
from src.ml.features.uav_attack_features import CUSUM_SOURCE_COLUMNS, feature_columns
from src.ml.models.modular_iforest import PX4_BASE_MODULES, fit_modular_iforest, score_flights

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
OUT_PATH = ROOT / "data/gold/ml6/sead_2x2_ablation.csv"


def old_179_ids(df: pd.DataFrame) -> list[str]:
    labels = flight_label_table(df).sort_values("source_id")
    quotas = {
        "normal": 59,
        "external_position_anomaly": 30,
        "mechanical_fault": 30,
        "altitude_anomaly": 30,
        "global_position_anomaly": 30,
    }
    ids = []
    for label, quota in quotas.items():
        ids += labels[labels["flight_label"] == label]["source_id"].head(quota).tolist()
    if len(ids) != 179:
        raise ValueError(f"179 alt kumesi kurulamadi: {len(ids)}")
    return ids


def refit_cusum(df: pd.DataFrame, train_ids: list[str]) -> pd.DataFrame:
    out = df.copy()
    params = fit_cusum_baselines(out[out["source_id"].isin(train_ids)], CUSUM_SOURCE_COLUMNS)
    mapping = {
        "gps_speed_residual": "gps_speed_residual_cusum_pos",
        "hdop": "hdop_cusum_pos",
        "noise_per_ms": "noise_per_ms_cusum_pos",
    }
    for source, target in mapping.items():
        if source not in out or target not in out:
            continue
        pieces = []
        for _, g in out.groupby("source_id", sort=False):
            pieces.append(cusum(g[source], **cusum_kwargs(params, source))["cusum_pos"])
        out[target] = pd.concat(pieces).sort_index()
    return out


def main() -> None:
    full = pd.read_parquet(FEATURE_PATH)
    subsets = {"179": set(old_179_ids(full)), "349": set(full["source_id"].unique())}
    rows = []
    for size_name, ids in subsets.items():
        df0 = full[full["source_id"].isin(ids)].copy()
        flights = flight_label_table(df0)
        for split_name, by_session in [("flight", False), ("session", True)]:
            for seed in range(5):
                split = make_group_split(
                    flights, seed=seed, n_val=min(10, max(1, int((flights.flight_label == 'normal').sum() * .15))),
                    n_test_normal=min(10, max(1, int((flights.flight_label == 'normal').sum() * .15))),
                    by_session=by_session, final_holdout_fraction=0.30)
                df = refit_cusum(df0, split["train"])
                cols = feature_columns(df)
                scaler = fit_scaler_params(df[df["source_id"].isin(split["train"])], cols)
                scaled = apply_scaler_params(df, scaler)
                fitted = fit_modular_iforest(scaled, split, PX4_BASE_MODULES, seed=seed)
                test = scaled[scaled["source_id"].isin(split["test"])]
                scored = score_flights(fitted, test)
                labels = flights.set_index("source_id")["flight_label"]
                y = np.array([0 if labels[f] == "normal" else 1 for f in scored.index])
                rows.append({
                    "n_flights": int(size_name), "split_unit": split_name, "seed": seed,
                    "n_train": len(split["train"]), "n_val": len(split["val"]),
                    "n_dev_test": len(split["test"]), "n_blind_holdout": len(split["final_holdout"]),
                    "flight_roc": roc_auc_score(y, scored["fusion"]),
                    "detection_at_1": float((scored.loc[y == 1, "fusion"] > 1).mean()),
                    "false_alarm_at_1": float((scored.loc[y == 0, "fusion"] > 1).mean()),
                })
    result = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(result.groupby(["n_flights", "split_unit"])[
        ["flight_roc", "detection_at_1", "false_alarm_at_1"]].agg(["mean", "std"]).round(3))
    print(f"\nYazildi: {OUT_PATH}")


if __name__ == "__main__":
    main()
