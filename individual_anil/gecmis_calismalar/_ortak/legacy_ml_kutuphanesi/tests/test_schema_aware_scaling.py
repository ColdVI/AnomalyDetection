"""Schema-aware scaling tests for pooled PX4 sources."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_ml9_category_evaluation import _score_modules
from src.ml.data.scaling import (
    apply_scaler_params,
    fit_scaler_params,
    infer_source_column_presence,
    infer_source_schema_groups,
)
from src.ml.models.modular_iforest import fit_modular_iforest


def _pooled_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": [
            "sead_train",
            "sead_train",
            "sead_val",
            "sead_test",
            "Real-No_Fault/hover/TestCase01/log",
            "Real-No_Fault/hover/TestCase01/log",
            "Real-No_Fault/hover/TestCase02/log",
            "Real-Motor/hover/TestCase03/log",
        ],
        "label": ["normal"] * 7 + ["motor_fault"],
        "t_rel_s": np.arange(8, dtype=float),
        "common_feature": [1.0, np.nan, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        "sead_only_feature": [20.0, 21.0, 22.0, 23.0, np.nan, np.nan, np.nan, np.nan],
        "rfly_only_feature": [np.nan, np.nan, np.nan, np.nan, 30.0, 31.0, 32.0, 33.0],
    })


def test_structural_missing_columns_remain_nan_but_in_source_gaps_impute():
    frame = _pooled_frame()
    feature_cols = ["common_feature", "sead_only_feature", "rfly_only_feature"]
    groups = infer_source_schema_groups(frame)
    presence = infer_source_column_presence(frame, feature_cols, groups)

    assert set(presence["default"]) == {"common_feature", "sead_only_feature"}
    assert set(presence["rflymad"]) == {"common_feature", "rfly_only_feature"}

    train_mask = frame["source_id"].isin([
        "sead_train",
        "Real-No_Fault/hover/TestCase01/log",
    ])
    params = fit_scaler_params(
        frame[train_mask],
        feature_cols,
        source_groups=groups.loc[train_mask],
        source_presence=presence,
    )
    scaled = apply_scaler_params(frame, params, source_groups=groups)

    rfly_rows = groups == "rflymad"
    sead_rows = groups == "default"
    assert scaled.loc[rfly_rows, "sead_only_feature"].isna().all()
    assert scaled.loc[sead_rows, "rfly_only_feature"].isna().all()
    assert scaled.loc[rfly_rows, "common_feature"].notna().all()
    assert scaled.loc[sead_rows, "common_feature"].notna().all()
    assert np.isfinite(scaled.loc[1, "common_feature"])


def test_structural_missing_module_scores_are_nan_and_fusion_can_skip_them():
    frame = _pooled_frame()
    feature_cols = ["common_feature", "sead_only_feature", "rfly_only_feature"]
    groups = infer_source_schema_groups(frame)
    presence = infer_source_column_presence(frame, feature_cols, groups)
    params = fit_scaler_params(
        frame[frame["label"] == "normal"],
        feature_cols,
        source_groups=groups.loc[frame["label"] == "normal"],
        source_presence=presence,
    )
    scaled = apply_scaler_params(frame, params, source_groups=groups)
    split = {
        "train": ["sead_train", "Real-No_Fault/hover/TestCase01/log"],
        "val": ["sead_val", "Real-No_Fault/hover/TestCase02/log"],
        "test": ["sead_test", "Real-Motor/hover/TestCase03/log"],
    }
    fitted = fit_modular_iforest(
        scaled,
        split,
        {
            "dikey_tutarlilik": ["sead_only_feature"],
            "nav_butunlugu": ["common_feature"],
        },
        seed=0,
        n_jobs=1,
    )
    scored = _score_modules(fitted, scaled, {"sead_val", "Real-No_Fault/hover/TestCase02/log"})

    rfly_rows = scored["source_id"].str.startswith("Real-")
    assert scored.loc[rfly_rows, "dikey_tutarlilik"].isna().all()
    assert scored.loc[rfly_rows, "nav_butunlugu"].notna().all()
    assert scored.loc[rfly_rows, "ml9_fusion"].notna().all()
