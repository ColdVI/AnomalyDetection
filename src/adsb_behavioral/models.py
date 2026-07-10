"""Normal-only robust-rule and Isolation Forest ADS-B baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.adsb_behavioral.physics_residuals import MODEL_FEATURES


EPSILON = 1e-6


@dataclass
class RobustPhysicsModel:
    medians: dict[str, dict[str, float]]
    scales: dict[str, dict[str, float]]
    global_medians: dict[str, float]
    global_scales: dict[str, float]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "RobustPhysicsModel":
        clean = train[train["quality_good"]].copy()
        global_medians: dict[str, float] = {}
        global_scales: dict[str, float] = {}
        for feature in MODEL_FEATURES:
            values = clean[feature].replace([np.inf, -np.inf], np.nan).dropna()
            median = float(values.median()) if len(values) else 0.0
            mad = float((values - median).abs().median()) if len(values) else 1.0
            global_medians[feature] = median
            global_scales[feature] = max(1.4826 * mad, EPSILON)

        medians: dict[str, dict[str, float]] = {}
        scales: dict[str, dict[str, float]] = {}
        for phase, group in clean.groupby("phase"):
            medians[str(phase)] = {}
            scales[str(phase)] = {}
            for feature in MODEL_FEATURES:
                values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) < 50:
                    medians[str(phase)][feature] = global_medians[feature]
                    scales[str(phase)][feature] = global_scales[feature]
                    continue
                median = float(values.median())
                mad = float((values - median).abs().median())
                medians[str(phase)][feature] = median
                scales[str(phase)][feature] = max(1.4826 * mad, EPSILON)
        return cls(medians, scales, global_medians, global_scales)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        z_values = np.full((len(frame), len(MODEL_FEATURES)), np.nan, dtype=float)
        for row_index, (_, row) in enumerate(frame.iterrows()):
            phase = str(row.get("phase", ""))
            phase_medians = self.medians.get(phase, self.global_medians)
            phase_scales = self.scales.get(phase, self.global_scales)
            for feature_index, feature in enumerate(MODEL_FEATURES):
                value = row.get(feature)
                if pd.notna(value) and np.isfinite(value):
                    z_values[row_index, feature_index] = max(
                        0.0, (float(value) - phase_medians[feature]) / phase_scales[feature]
                    )
        available = np.sum(np.isfinite(z_values), axis=1)
        filled = np.where(np.isfinite(z_values), z_values, -np.inf)
        sorted_values = np.sort(filled, axis=1)
        largest = sorted_values[:, -1]
        second = sorted_values[:, -2] if len(MODEL_FEATURES) > 1 else largest
        score = np.where(available >= 2, (largest + second) / 2.0, largest)
        score[available == 0] = np.nan
        reason_index = np.argmax(filled, axis=1)
        reasons = np.array(MODEL_FEATURES, dtype=object)[reason_index]
        reasons[available == 0] = None
        result = frame.copy()
        result["rule_score"] = score
        result["rule_reason"] = reasons
        return result


@dataclass
class IsolationForestPhysicsModel:
    imputer: SimpleImputer
    scaler: RobustScaler
    model: IsolationForest

    @classmethod
    def fit(cls, train: pd.DataFrame, *, seed: int = 20260710) -> "IsolationForestPhysicsModel":
        clean = train[train["quality_good"]][MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
        if len(clean) < 100:
            raise ValueError("At least 100 quality-good train rows are required")
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        values = scaler.fit_transform(imputer.fit_transform(clean))
        model = IsolationForest(
            n_estimators=200,
            max_samples=min(2048, len(values)),
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(values)
        return cls(imputer, scaler, model)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
        transformed = self.scaler.transform(self.imputer.transform(values))
        result = frame.copy()
        result["iforest_score"] = -self.model.score_samples(transformed)
        result.loc[~result["quality_good"], "iforest_score"] = np.nan
        return result
