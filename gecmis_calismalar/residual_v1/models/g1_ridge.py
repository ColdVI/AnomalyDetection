"""Development-only, session-grouped G1 ridge residual models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
META_COLUMNS = frozenset({"flight_id", "t", "phase", "train_eligible"})
EXPECTED_POSITIVE_SIGNS: dict[str, tuple[str, ...]] = {
    "R1_aileron_roll_rate": ("aileron_cmd__last",),
    "R2_elevator_pitch_rate": ("elevator_cmd__last",),
    "R3_rudder_coordinated_yaw_rate": ("rudder_cmd__last",),
    "R4_throttle_airspeed_derivative": ("throttle_cmd__last",),
    "R5_pitch_throttle_climb_rate": ("throttle_cmd__last",),
}


class InsufficientSessionCoverage(ValueError):
    """Raised instead of silently replacing session CV with row/flight CV."""


@dataclass
class G1ChannelFit:
    model: Ridge
    residuals: pd.DataFrame
    report: dict
    coefficients: dict[str, float]
    coefficient_sanity: dict


def _feature_columns(matrix: pd.DataFrame, response: str) -> list[str]:
    columns = [
        column
        for column in matrix.columns
        if column not in META_COLUMNS and column != response
    ]
    forbidden = [
        column
        for column in columns
        if column == response or column.startswith(f"{response}__")
    ]
    if forbidden:
        raise ValueError(f"response leakage reached G1 matrix: {forbidden}")
    if not columns:
        raise ValueError("G1 requires at least one feature")
    return columns


def _session_groups(
    matrix: pd.DataFrame,
    session_by_flight: Mapping[str, str],
) -> pd.Series:
    groups = matrix["flight_id"].astype(str).map(session_by_flight)
    if groups.isna().any():
        missing = sorted(matrix.loc[groups.isna(), "flight_id"].astype(str).unique())
        raise ValueError(f"missing session metadata for flights: {missing}")
    return groups.astype(str)


def grouped_session_splits(
    groups: Sequence[str],
    *,
    max_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return group-disjoint folds, capped at the available session count."""

    values = np.asarray(groups, dtype=object)
    unique = np.unique(values)
    if len(unique) < 2:
        raise InsufficientSessionCoverage(
            f"session-grouped CV requires at least 2 sessions; observed {len(unique)}"
        )
    splitter = GroupKFold(n_splits=min(max_folds, len(unique)))
    placeholder = np.zeros((len(values), 1), dtype=float)
    return list(splitter.split(placeholder, groups=values))


def _coefficient_sanity(
    channel: str,
    coefficients: Mapping[str, float],
    *,
    expected_positive_signs: Mapping[str, Sequence[str]],
) -> dict:
    expected = tuple(expected_positive_signs.get(channel, ()))
    if not expected:
        return {
            "channel": channel,
            "status": "not_pre_registered",
            "checks": [],
        }
    checks = []
    for feature in expected:
        coefficient = coefficients.get(feature)
        checks.append(
            {
                "feature": feature,
                "expected_sign": "positive",
                "coefficient": coefficient,
                "passed": coefficient is not None and coefficient > 0.0,
            }
        )
    return {
        "channel": channel,
        "status": "passed" if all(check["passed"] for check in checks) else "flagged",
        "checks": checks,
    }


def fit_g1_channel(
    matrix: pd.DataFrame,
    *,
    channel: str,
    response: str,
    session_by_flight: Mapping[str, str],
    alpha_grid: Sequence[float] = ALPHA_GRID,
    max_folds: int = 5,
    expected_positive_signs: Mapping[str, Sequence[str]] = EXPECTED_POSITIVE_SIGNS,
) -> G1ChannelFit:
    """Select alpha with development-only session folds, then fit the train mask."""

    required = {"flight_id", "t", "phase", "train_eligible", response}
    missing = sorted(required - set(matrix))
    if missing:
        raise ValueError(f"{channel}: missing columns {missing}")
    if matrix.empty:
        raise ValueError(f"{channel}: no feature rows")
    if not alpha_grid or any(float(alpha) <= 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain positive values")

    feature_columns = _feature_columns(matrix, response)
    train_mask = matrix["train_eligible"].astype(bool)
    finite = (
        matrix[feature_columns].notna().all(axis=1)
        & pd.to_numeric(matrix[response], errors="coerce").notna()
    )
    train_mask &= finite
    if not train_mask.any():
        raise ValueError(f"{channel}: no finite train-eligible rows")

    train = matrix.loc[train_mask].reset_index(drop=True)
    groups = _session_groups(train, session_by_flight)
    splits = grouped_session_splits(groups, max_folds=max_folds)
    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = pd.to_numeric(train[response], errors="raise").to_numpy(dtype=float)

    candidates = []
    for alpha in (float(value) for value in alpha_grid):
        predictions = np.full(len(train), np.nan, dtype=float)
        for fit_index, validation_index in splits:
            model = Ridge(alpha=alpha)
            model.fit(X_train[fit_index], y_train[fit_index])
            predictions[validation_index] = model.predict(X_train[validation_index])
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"{channel}: incomplete grouped CV predictions")
        candidates.append(
            {
                "alpha": alpha,
                "cv_mse": float(mean_squared_error(y_train, predictions)),
                "cv_r2": float(r2_score(y_train, predictions)),
            }
        )
    selected = min(candidates, key=lambda item: (item["cv_mse"], item["alpha"]))

    model = Ridge(alpha=float(selected["alpha"]))
    model.fit(X_train, y_train)
    coefficients = {
        feature: float(value)
        for feature, value in zip(feature_columns, model.coef_, strict=True)
    }
    sanity = _coefficient_sanity(
        channel,
        coefficients,
        expected_positive_signs=expected_positive_signs,
    )

    scoring_mask = finite
    scoring = matrix.loc[scoring_mask].copy()
    X_scoring = scoring[feature_columns].to_numpy(dtype=float)
    y_scoring = pd.to_numeric(scoring[response], errors="raise").to_numpy(dtype=float)
    y_hat = model.predict(X_scoring)
    residuals = scoring.loc[
        :, ["flight_id", "t", "phase", "train_eligible"]
    ].reset_index(drop=True)
    residuals["channel"] = channel
    residuals["y"] = y_scoring
    residuals["y_hat"] = y_hat
    residuals["r"] = y_scoring - y_hat

    train_prediction = model.predict(X_train)
    train_sessions = sorted(groups.unique().tolist())
    fold_sessions = []
    group_values = groups.to_numpy()
    for fold_index, (fit_index, validation_index) in enumerate(splits):
        fit_sessions = sorted(np.unique(group_values[fit_index]).tolist())
        validation_sessions = sorted(np.unique(group_values[validation_index]).tolist())
        if set(fit_sessions) & set(validation_sessions):
            raise RuntimeError("session leakage detected in grouped CV")
        fold_sessions.append(
            {
                "fold": fold_index,
                "fit_sessions": fit_sessions,
                "validation_sessions": validation_sessions,
            }
        )

    report = {
        "channel": channel,
        "response": response,
        "status": "trained",
        "selected_alpha": float(selected["alpha"]),
        "cv_mse": float(selected["cv_mse"]),
        "cv_r2": float(selected["cv_r2"]),
        "train_r2": float(r2_score(y_train, train_prediction)),
        "intercept": float(model.intercept_),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "coverage": {
            "matrix_rows": int(len(matrix)),
            "scored_rows": int(scoring_mask.sum()),
            "train_rows": int(train_mask.sum()),
            "matrix_flights": int(matrix["flight_id"].astype(str).nunique()),
            "train_flights": int(train["flight_id"].astype(str).nunique()),
            "train_sessions": len(train_sessions),
            "session_ids": train_sessions,
            "cv_folds": len(splits),
        },
        "alpha_candidates": candidates,
        "fold_sessions": fold_sessions,
    }
    return G1ChannelFit(
        model=model,
        residuals=residuals,
        report=report,
        coefficients=coefficients,
        coefficient_sanity=sanity,
    )
