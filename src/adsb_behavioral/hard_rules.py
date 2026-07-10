"""Immediate, explicit physical-consistency checks for obvious ADS-B corruption."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Deliberately interpretable engineering limits. These are not learned thresholds.
HARD_LIMITS = {
    "abs_speed_residual_mps": 20.0,
    "abs_track_residual_deg": 20.0,
    "abs_vrate_residual_mps": 5.0,
    "abs_horizontal_accel_mps2": 8.0,
    "abs_horizontal_jerk_mps3": 4.0,
    "abs_baro_geom_delta_rate_mps": 5.0,
    "duplicate_position_moving": 0.5,
}

REASON_CODES = {
    "abs_speed_residual_mps": "speed_position_mismatch",
    "abs_track_residual_deg": "track_bearing_mismatch",
    "abs_vrate_residual_mps": "altitude_vertical_rate_mismatch",
    "abs_horizontal_accel_mps2": "impossible_horizontal_acceleration",
    "abs_horizontal_jerk_mps3": "impossible_horizontal_jerk",
    "abs_baro_geom_delta_rate_mps": "baro_geom_discontinuity",
    "duplicate_position_moving": "position_freeze_while_moving",
}


def add_hard_rule_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Score obvious violations; score >=1 means at least one hard rule fired.

    Hard rules are instantaneous by design. They must not be routed through a persistence
    filter that would suppress a one-sample 10,000-ft jump.
    """

    ratios = pd.DataFrame(index=frame.index)
    for feature, limit in HARD_LIMITS.items():
        ratios[feature] = pd.to_numeric(frame[feature], errors="coerce") / limit

    # Track is undefined/noisy while stopped; only enforce it at meaningful speed.
    slow = frame["ground_speed_ms"].fillna(0.0).lt(30.0)
    ratios.loc[slow, "abs_track_residual_deg"] = np.nan

    values = ratios.to_numpy(dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf)
    reason_index = np.argmax(filled, axis=1)
    score = np.max(filled, axis=1)
    score[~finite.any(axis=1)] = np.nan
    features = np.array(list(ratios.columns), dtype=object)
    reasons = np.array([REASON_CODES[feature] for feature in features], dtype=object)[reason_index]
    reasons[~finite.any(axis=1)] = None

    out = frame.copy()
    out["hard_rule_score"] = score
    out["hard_rule_violation"] = score >= 1.0
    out["hard_rule_reason"] = reasons
    return out
