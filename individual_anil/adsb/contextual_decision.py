"""Explicit per-channel alert budgets and time-aware anomaly decision profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from adsb.conditional_calibration import validate_alert_alpha


TemporalMode = Literal["instant", "persistence", "accumulation"]


@dataclass(frozen=True)
class ChannelAlertBudget:
    """Pre-registered allocation of one total false-alert alpha."""

    total_alpha: float
    channel_alpha: dict[str, float]

    def __post_init__(self) -> None:
        total = validate_alert_alpha(self.total_alpha)
        if not self.channel_alpha:
            raise ValueError("at least one channel alpha allocation is required")
        allocations = {name: validate_alert_alpha(value) for name, value in self.channel_alpha.items()}
        if sum(allocations.values()) > total + np.finfo(float).eps:
            raise ValueError("channel alpha allocations exceed total_alpha")


@dataclass(frozen=True)
class DetectorProfile:
    """An anomaly-family decision contract over one calibrated score channel."""

    anomaly_type: str
    channel: str
    mode: TemporalMode
    max_gap_s: float
    persistence_s: float | None = None
    reference_surprisal: float | None = None
    accumulation_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.anomaly_type or not self.channel:
            raise ValueError("anomaly_type and channel are required")
        if not np.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and > 0")
        if self.mode == "instant":
            if any(value is not None for value in (
                self.persistence_s, self.reference_surprisal, self.accumulation_threshold
            )):
                raise ValueError("instant mode cannot carry temporal thresholds")
        elif self.mode == "persistence":
            if self.persistence_s is None or not np.isfinite(self.persistence_s) or self.persistence_s <= 0:
                raise ValueError("persistence mode requires persistence_s > 0")
            if self.reference_surprisal is not None or self.accumulation_threshold is not None:
                raise ValueError("persistence mode cannot carry accumulation parameters")
        elif self.mode == "accumulation":
            values = (self.reference_surprisal, self.accumulation_threshold)
            if any(value is None or not np.isfinite(value) or value <= 0 for value in values):
                raise ValueError("accumulation mode requires positive finite parameters")
            if self.persistence_s is not None:
                raise ValueError("accumulation mode cannot carry persistence_s")
        else:
            raise ValueError(f"unsupported temporal mode: {self.mode}")


def apply_detector_profile(
    calibrated: pd.DataFrame,
    *,
    profile: DetectorProfile,
    budget: ChannelAlertBudget,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Apply an explicit channel budget with second-based temporal evidence.

    The input must already contain conformal p-values.  Rows of other channels
    are rejected instead of silently fused.
    """

    required = {flight_id_col, time_col, "channel", "conformal_p_value"}
    missing = required.difference(calibrated.columns)
    if missing:
        raise KeyError(f"Missing decision columns: {sorted(missing)}")
    if profile.channel not in budget.channel_alpha:
        raise ValueError(f"No alert allocation exists for channel {profile.channel!r}")
    if not calibrated["channel"].eq(profile.channel).all():
        raise ValueError("A detector profile accepts exactly one score channel")
    alpha = budget.channel_alpha[profile.channel]

    output = pd.DataFrame(index=calibrated.index)
    output["anomaly_type"] = profile.anomaly_type
    output["channel"] = profile.channel
    output["alert_alpha"] = alpha
    output["temporal_evidence"] = 0.0
    output["alarm"] = False
    output["reset_reason"] = ""

    for flight_id, group in calibrated.groupby(flight_id_col, sort=False):
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy(float)
        p_values = pd.to_numeric(group["conformal_p_value"], errors="coerce").to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError(f"{flight_id}: decision timestamps must be finite and sorted")
        if not np.isfinite(p_values).all() or np.any((p_values <= 0) | (p_values > 1)):
            raise ValueError(f"{flight_id}: conformal p-values must be in (0, 1]")

        evidence = 0.0
        previous_time: float | None = None
        for index, timestamp, p_value in zip(group.index, times, p_values):
            if previous_time is None:
                output.loc[index, "reset_reason"] = "flight_start"
            else:
                dt_s = float(timestamp - previous_time)
                if dt_s <= 0 or dt_s > profile.max_gap_s:
                    evidence = 0.0
                    output.loc[index, "reset_reason"] = "invalid_or_large_gap"
                elif profile.mode == "persistence":
                    evidence = evidence + dt_s if p_value <= alpha else 0.0
                elif profile.mode == "accumulation":
                    surprise = -np.log(p_value)
                    increment = dt_s * (surprise - float(profile.reference_surprisal))
                    evidence = max(0.0, evidence + increment)

            if profile.mode == "instant":
                alarm = p_value <= alpha
            elif profile.mode == "persistence":
                alarm = evidence >= float(profile.persistence_s)
            else:
                alarm = evidence >= float(profile.accumulation_threshold)
            output.loc[index, "temporal_evidence"] = evidence
            output.loc[index, "alarm"] = bool(alarm)
            previous_time = float(timestamp)
    return output
