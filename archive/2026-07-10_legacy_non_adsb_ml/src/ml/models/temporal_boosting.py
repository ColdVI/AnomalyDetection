"""Class-balanced LightGBM scorer for ML-8A temporal descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ml.data.scaling import apply_scaler_params, fit_scaler_params


@dataclass
class TemporalBoostingFit:
    model: lgb.LGBMClassifier
    scaler: dict
    feature_columns: list[str]
    validation_auprc: float


def descriptor_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return flattened channel/descriptor columns in stable table order."""

    return [column for column in df.columns if "__" in column]


def fit_temporal_boosting(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target_col: str = "target",
    seed: int = 0,
) -> TemporalBoostingFit:
    """Fit the frozen ML-8A recipe with train-only scaling/imputation."""

    feature_cols = descriptor_feature_columns(train)
    if not feature_cols:
        raise ValueError("No descriptor feature columns found")
    if set(train[target_col].unique()) != {0, 1}:
        raise ValueError("Training data must contain both binary classes")
    if validation[target_col].nunique() < 2:
        raise ValueError("Validation data must contain both binary classes")

    scaler = fit_scaler_params(train, feature_cols)
    x_train = apply_scaler_params(train[feature_cols], scaler)[feature_cols]
    x_val = apply_scaler_params(validation[feature_cols], scaler)[feature_cols]
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        train[target_col].astype(int),
        eval_set=[(x_val, validation[target_col].astype(int))],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_score = model.predict_proba(x_val)[:, 1]
    return TemporalBoostingFit(
        model=model,
        scaler=scaler,
        feature_columns=feature_cols,
        validation_auprc=float(average_precision_score(validation[target_col], val_score)),
    )


def predict_temporal_boosting(fit: TemporalBoostingFit, frame: pd.DataFrame) -> np.ndarray:
    scaled = apply_scaler_params(frame[fit.feature_columns], fit.scaler)
    return fit.model.predict_proba(scaled[fit.feature_columns])[:, 1]


def binary_window_metrics(y_true, scores) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    if len(np.unique(y)) < 2:
        raise ValueError("Window metrics require both classes")
    return {
        "window_auroc": float(roc_auc_score(y, score)),
        "window_auprc": float(average_precision_score(y, score)),
        "prevalence": float(np.mean(y)),
    }


def gain_importance(fit: TemporalBoostingFit) -> list[dict[str, float | str]]:
    gains = fit.model.booster_.feature_importance(importance_type="gain")
    rows = [
        {"feature": feature, "gain": float(gain)}
        for feature, gain in zip(fit.feature_columns, gains)
    ]
    return sorted(rows, key=lambda row: float(row["gain"]), reverse=True)
