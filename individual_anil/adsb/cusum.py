"""Causal two-sided Page CUSUM for signed ADS-B velocity residuals.

This module deliberately does not select an alarm threshold from synthetic
recall.  The physical shift target and the already calibrated natural-burden
threshold are mandatory :class:`CusumConfig` inputs and are serializable with
the train-only robust calibration.

For each east/north residual channel the detector keeps positive and negative
Page states.  Their joint maximum is the single alarm statistic; the four
states are not calibrated as four independent alert budgets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from adsb.features import VECTOR_RESIDUAL_FEATURES

ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class CusumConfig:
    """Pre-registered CUSUM contract; all numeric choices are mandatory.

    ``target_vector_shift_mps`` is converted to a direction-agnostic axis
    lower bound: a 2-D vector of that magnitude has at least one component of
    magnitude ``target/sqrt(2)``.  Channel-specific reference values are then
    ``k_c = axis_target / (2 * train_mad_c)`` in robust-z units.

    ``threshold_h`` must come from a separately frozen normal calibration and
    natural alarm-burden budget.  This class intentionally has no threshold
    search or synthetic-aware tuning method.
    """

    target_vector_shift_mps: float
    threshold_h: float
    max_gap_s: float
    missing_reset_s: float
    z_clip: float
    channels: tuple[str, str] = tuple(VECTOR_RESIDUAL_FEATURES)
    flight_id_col: str = "flight_id"
    time_col: str = "timestamp_utc"
    on_ground_col: str = "on_ground"

    def __post_init__(self) -> None:
        if len(self.channels) != 2 or len(set(self.channels)) != 2:
            raise ValueError("Exactly two distinct residual channels are required")
        for name in ("target_vector_shift_mps", "threshold_h", "max_gap_s", "z_clip"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not np.isfinite(self.missing_reset_s) or self.missing_reset_s < 0.0:
            raise ValueError("missing_reset_s must be finite and >= 0")

    @property
    def minimum_axis_shift_mps(self) -> float:
        return self.target_vector_shift_mps / sqrt(2.0)


def _state_column(channel: str, direction: str) -> str:
    return f"{channel}_cusum_{direction}"


def _z_column(channel: str) -> str:
    return f"{channel}_robust_z"


def _observed_column(channel: str) -> str:
    return f"{channel}_observed"


def _missing_reset_column(channel: str) -> str:
    return f"{channel}_missing_reset"


def _ground_status(value: Any) -> bool | None:
    """Return True/False, or None when ground state itself is unknown."""
    if pd.isna(value):
        return None
    return bool(value)


class VectorPageCUSUM:
    """Train-normal calibrated causal Page CUSUM with a four-state joint alarm."""

    def __init__(self, config: CusumConfig):
        self.config = config
        self.calibration_: dict[str, dict[str, float]] | None = None
        self.excluded_channels_: dict[str, str] = {}

    def _require_columns(self, df: pd.DataFrame) -> None:
        required = {
            *self.config.channels,
            self.config.flight_id_col,
            self.config.time_col,
            self.config.on_ground_col,
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"Missing CUSUM columns: {missing}")
        if df[self.config.flight_id_col].isna().any():
            raise ValueError("flight_id cannot be missing for causal CUSUM")

    def _eligible_transition_mask(self, df: pd.DataFrame) -> pd.Series:
        """Mask matching the global reset/skip rules used by ``score_rows``."""
        flight = df[self.config.flight_id_col]
        same_flight = flight.eq(flight.shift(1))
        time = pd.to_numeric(df[self.config.time_col], errors="coerce")
        dt = time - time.shift(1)
        ground = df[self.config.on_ground_col].astype("boolean")
        previous_ground = ground.shift(1)
        return (
            same_flight
            & dt.gt(0.0)
            & dt.le(self.config.max_gap_s)
            & ground.eq(False)
            & previous_ground.eq(False)
        ).fillna(False)

    def fit(self, normal_train_features: pd.DataFrame) -> "VectorPageCUSUM":
        """Fit median/MAD on an externally guaranteed normal-train partition.

        This method never substitutes a floor.  A channel with no eligible
        finite values or exact zero MAD is excluded and remains visible in the
        serialized calibration and score schema.
        """
        self._require_columns(normal_train_features)
        eligible = self._eligible_transition_mask(normal_train_features).to_numpy(dtype=bool)
        calibration: dict[str, dict[str, float]] = {}
        excluded: dict[str, str] = {}

        for channel in self.config.channels:
            values = normal_train_features[channel].to_numpy(dtype=float, na_value=np.nan)
            values = values[eligible & np.isfinite(values)]
            if values.size == 0:
                excluded[channel] = "no_eligible_finite_train_values"
                continue

            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) * ROBUST_MAD_SCALE
            if not np.isfinite(mad) or mad == 0.0:
                excluded[channel] = "mad_zero"
                continue

            calibration[channel] = {
                "median": median,
                "mad": mad,
                "k": self.config.minimum_axis_shift_mps / (2.0 * mad),
            }

        if not calibration:
            raise ValueError("No CUSUM channel is calibratable; all channels were excluded")

        self.calibration_ = calibration
        self.excluded_channels_ = excluded
        return self

    def _reset_all(
        self,
        positive: dict[str, float],
        negative: dict[str, float],
        missing_elapsed_s: dict[str, float],
    ) -> None:
        for channel in self.config.channels:
            positive[channel] = 0.0
            negative[channel] = 0.0
            missing_elapsed_s[channel] = 0.0

    def score_rows(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score rows in their given stream order without looking ahead.

        Global state resets at flight starts, current or previous ground rows,
        unknown ground status, invalid/backward time, and gaps above
        ``max_gap_s``.  Equal timestamps are skipped without changing state.

        A finite channel observation updates its two signed Page states.  A
        missing channel carries state for less than ``missing_reset_s`` elapsed
        seconds, then resets only that channel.  A row with no observed active
        channel is not evaluable and cannot emit a fresh alarm.
        """
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before score_rows()")
        self._require_columns(features)

        n_rows = len(features)
        output: dict[str, np.ndarray] = {
            "cusum_dt_s": np.full(n_rows, np.nan, dtype=float),
            "cusum_joint_score": np.zeros(n_rows, dtype=float),
            "cusum_joint_alarm": np.zeros(n_rows, dtype=bool),
            "cusum_evaluable": np.zeros(n_rows, dtype=bool),
            "cusum_observed_channels": np.zeros(n_rows, dtype=np.int8),
            "cusum_reset_reason": np.full(n_rows, "none", dtype=object),
        }
        for channel in self.config.channels:
            fill = 0.0 if channel in self.calibration_ else np.nan
            output[_state_column(channel, "positive")] = np.full(n_rows, fill, dtype=float)
            output[_state_column(channel, "negative")] = np.full(n_rows, fill, dtype=float)
            output[_z_column(channel)] = np.full(n_rows, np.nan, dtype=float)
            output[_observed_column(channel)] = np.zeros(n_rows, dtype=bool)
            output[_missing_reset_column(channel)] = np.zeros(n_rows, dtype=bool)

        flights = features[self.config.flight_id_col].to_numpy(dtype=object)
        times = pd.to_numeric(features[self.config.time_col], errors="coerce").to_numpy(dtype=float)
        ground_values = features[self.config.on_ground_col].to_numpy(dtype=object)
        channel_values = {
            channel: features[channel].to_numpy(dtype=float, na_value=np.nan)
            for channel in self.config.channels
        }

        positive = {channel: 0.0 for channel in self.config.channels}
        negative = {channel: 0.0 for channel in self.config.channels}
        missing_elapsed_s = {channel: 0.0 for channel in self.config.channels}
        previous_flight: Any = None
        previous_time = float("nan")
        previous_ground: bool | None = None
        initialized = False

        for row_number in range(n_rows):
            flight = flights[row_number]
            timestamp = times[row_number]
            ground = _ground_status(ground_values[row_number])
            eligible = False
            dt = float("nan")
            reason = "none"

            if not initialized or flight != previous_flight:
                self._reset_all(positive, negative, missing_elapsed_s)
                reason = "flight_start"
            elif ground is None or previous_ground is None:
                self._reset_all(positive, negative, missing_elapsed_s)
                reason = "unknown_ground_status"
            elif ground:
                self._reset_all(positive, negative, missing_elapsed_s)
                reason = "on_ground"
            elif previous_ground:
                self._reset_all(positive, negative, missing_elapsed_s)
                reason = "ground_transition"
            elif not np.isfinite(timestamp) or not np.isfinite(previous_time):
                self._reset_all(positive, negative, missing_elapsed_s)
                reason = "invalid_time"
            else:
                dt = timestamp - previous_time
                output["cusum_dt_s"][row_number] = dt
                if dt < 0.0:
                    self._reset_all(positive, negative, missing_elapsed_s)
                    reason = "negative_dt"
                elif dt > self.config.max_gap_s:
                    self._reset_all(positive, negative, missing_elapsed_s)
                    reason = "long_gap"
                elif dt == 0.0:
                    reason = "zero_dt"
                else:
                    eligible = True

            observed_count = 0
            if eligible:
                for channel in self.config.channels:
                    if channel not in self.calibration_:
                        continue
                    value = channel_values[channel][row_number]
                    if np.isfinite(value):
                        calibration = self.calibration_[channel]
                        z = (value - calibration["median"]) / calibration["mad"]
                        clipped_z = float(np.clip(z, -self.config.z_clip, self.config.z_clip))
                        k = calibration["k"]
                        positive[channel] = max(0.0, positive[channel] + clipped_z - k)
                        negative[channel] = max(0.0, negative[channel] - clipped_z - k)
                        missing_elapsed_s[channel] = 0.0
                        output[_z_column(channel)][row_number] = z
                        output[_observed_column(channel)][row_number] = True
                        observed_count += 1
                    else:
                        missing_elapsed_s[channel] += dt
                        if missing_elapsed_s[channel] >= self.config.missing_reset_s:
                            positive[channel] = 0.0
                            negative[channel] = 0.0
                            output[_missing_reset_column(channel)][row_number] = True

            active_states: list[float] = []
            for channel in self.config.channels:
                if channel not in self.calibration_:
                    continue
                output[_state_column(channel, "positive")][row_number] = positive[channel]
                output[_state_column(channel, "negative")][row_number] = negative[channel]
                active_states.extend((positive[channel], negative[channel]))

            joint_score = max(active_states, default=0.0)
            evaluable = observed_count > 0
            output["cusum_joint_score"][row_number] = joint_score
            output["cusum_evaluable"][row_number] = evaluable
            output["cusum_observed_channels"][row_number] = observed_count
            output["cusum_joint_alarm"][row_number] = (
                evaluable and joint_score > self.config.threshold_h
            )
            output["cusum_reset_reason"][row_number] = reason

            previous_flight = flight
            previous_time = timestamp
            previous_ground = ground
            initialized = True

        return pd.DataFrame(output, index=features.index)

    def to_dict(self) -> dict[str, Any]:
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before serialization")
        config_dict = asdict(self.config)
        config_dict["channels"] = list(self.config.channels)
        return {
            "schema_version": 1,
            "config": config_dict,
            "calibration": self.calibration_,
            "excluded_channels": self.excluded_channels_,
            # state_count describes the states that can actually update.
            # Keeping the configured count separate prevents a one-axis,
            # MAD=0-degraded detector from being reported as four-state.
            "state_count": 2 * len(self.calibration_),
            "configured_state_count": 2 * len(self.config.channels),
            "axis_coverage_status": (
                "complete_two_axis"
                if len(self.calibration_) == len(self.config.channels)
                else "degraded_axis_coverage"
            ),
            "joint_statistic": "max",
            "alarm_comparator": ">",
            "normalization": "median_and_1.4826_times_mad",
            "mad_zero_policy": "exclude",
            "missing_policy": "carry_then_elapsed_time_reset",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorPageCUSUM":
        config_dict = dict(payload["config"])
        config_dict["channels"] = tuple(config_dict["channels"])
        detector = cls(CusumConfig(**config_dict))
        detector.calibration_ = {
            channel: {key: float(value) for key, value in calibration.items()}
            for channel, calibration in payload["calibration"].items()
        }
        detector.excluded_channels_ = dict(payload.get("excluded_channels", {}))
        return detector
