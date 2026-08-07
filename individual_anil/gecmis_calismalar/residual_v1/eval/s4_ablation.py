"""S-4 command-removal ablation for trained G1 ridge channels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from gecmis_calismalar.residual_v1.features.spec import ResidualChannelSpec
from gecmis_calismalar.residual_v1.models.g1_ridge import META_COLUMNS

S4_MIN_VARIANCE_RATIO = 1.15


def _is_command_feature(column: str, command: str) -> bool:
    return column.startswith(f"{command}__")


def command_ablation_report(
    matrix: pd.DataFrame,
    *,
    spec: ResidualChannelSpec,
    selected_alpha: float,
    full_feature_columns: Sequence[str],
    minimum_variance_ratio: float = S4_MIN_VARIANCE_RATIO,
) -> dict:
    """Refit the selected G1 ridge with all declared command features removed."""

    if not spec.command_inputs:
        raise ValueError(f"{spec.name}: S-4 requires declared command inputs")
    if selected_alpha <= 0.0 or minimum_variance_ratio <= 1.0:
        raise ValueError("selected_alpha must be positive and S-4 ratio must exceed one")
    missing = sorted(
        {"flight_id", "train_eligible", spec.response, *full_feature_columns} - set(matrix)
    )
    if missing:
        raise ValueError(f"{spec.name}: missing S-4 columns {missing}")
    forbidden = [
        column
        for column in full_feature_columns
        if column in META_COLUMNS
        or column == spec.response
        or column.startswith(f"{spec.response}__")
    ]
    if forbidden:
        raise ValueError(f"{spec.name}: invalid full feature columns {forbidden}")

    removed = sorted(
        column
        for column in full_feature_columns
        if any(_is_command_feature(column, command) for command in spec.command_inputs)
    )
    for command in spec.command_inputs:
        if not any(_is_command_feature(column, command) for column in removed):
            raise ValueError(f"{spec.name}: no feature columns found for command {command}")
    retained = [column for column in full_feature_columns if column not in removed]

    numeric_response = pd.to_numeric(matrix[spec.response], errors="coerce")
    finite = matrix[list(full_feature_columns)].notna().all(axis=1) & numeric_response.notna()
    train_mask = matrix["train_eligible"].astype(bool) & finite
    train = matrix.loc[train_mask]
    if train.empty:
        raise ValueError(f"{spec.name}: no finite train-eligible S-4 rows")
    y = numeric_response.loc[train_mask].to_numpy(float)

    full_model = Ridge(alpha=float(selected_alpha))
    full_model.fit(train[list(full_feature_columns)].to_numpy(float), y)
    full_residual = y - full_model.predict(train[list(full_feature_columns)].to_numpy(float))

    if retained:
        crippled_model = Ridge(alpha=float(selected_alpha))
        crippled_model.fit(train[retained].to_numpy(float), y)
        crippled_prediction = crippled_model.predict(train[retained].to_numpy(float))
        crippled_model_kind = "ridge"
    else:
        crippled_prediction = np.full(len(y), float(np.mean(y)))
        crippled_model_kind = "intercept_only"
    crippled_residual = y - crippled_prediction

    full_variance = float(np.var(full_residual))
    crippled_variance = float(np.var(crippled_residual))
    if full_variance == 0.0:
        ratio = float("inf") if crippled_variance > 0.0 else 1.0
    else:
        ratio = crippled_variance / full_variance
    flagged = bool(ratio < minimum_variance_ratio)
    return {
        "gate": "S-4",
        "channel": spec.name,
        "status": "flagged" if flagged else "passed",
        "flagged": flagged,
        "criterion": "var_crippled_over_var_full < minimum_variance_ratio",
        "minimum_variance_ratio": float(minimum_variance_ratio),
        "variance_ratio": float(ratio),
        "full_residual_variance": full_variance,
        "crippled_residual_variance": crippled_variance,
        "selected_alpha": float(selected_alpha),
        "train_rows": int(len(train)),
        "train_flights": int(train["flight_id"].astype(str).nunique()),
        "full_feature_count": len(full_feature_columns),
        "crippled_feature_count": len(retained),
        "removed_command_features": removed,
        "retained_context_features": retained,
        "crippled_model_kind": crippled_model_kind,
    }
