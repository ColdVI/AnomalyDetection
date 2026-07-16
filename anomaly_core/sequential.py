"""Causal multi-channel Page CUSUM score generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class PageCUSUMConfig:
    channels: tuple[str, ...]
    reference_shift_z: float
    z_clip: float
    max_gap_s: float
    flight_id_col: str = "flight_id"
    time_col: str = "timestamp_s"
    evaluable_col: str = "evaluable"

    def __post_init__(self) -> None:
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("distinct channels are required")
        if min(self.reference_shift_z, self.z_clip, self.max_gap_s) <= 0:
            raise ValueError("CUSUM numeric parameters must be positive")


class MultiChannelPageCUSUM:
    def __init__(self, config: PageCUSUMConfig):
        self.config = config
        self.calibration_: dict[str, dict[str, float]] | None = None

    def fit(self, normal: pd.DataFrame) -> "MultiChannelPageCUSUM":
        calibration: dict[str, dict[str, float]] = {}
        eligible = normal[self.config.evaluable_col].fillna(False).to_numpy(dtype=bool)
        for channel in self.config.channels:
            values = pd.to_numeric(normal[channel], errors="coerce").to_numpy(float)
            values = values[eligible & np.isfinite(values)]
            if len(values) == 0:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)) * ROBUST_MAD_SCALE)
            if mad > 0 and np.isfinite(mad):
                calibration[channel] = {"median": median, "mad": mad}
        if not calibration:
            raise ValueError("no CUSUM channel is calibratable")
        self.calibration_ = calibration
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before score()")
        output = pd.DataFrame(index=frame.index)
        output["cusum_score"] = 0.0
        output["cusum_evaluable"] = False
        output["cusum_reset_reason"] = ""
        k = self.config.reference_shift_z / 2.0
        for _, group in frame.groupby(self.config.flight_id_col, sort=False):
            state_pos = {channel: 0.0 for channel in self.calibration_}
            state_neg = {channel: 0.0 for channel in self.calibration_}
            previous_time: float | None = None
            for index, row in group.iterrows():
                timestamp = float(row[self.config.time_col])
                evaluable = bool(row[self.config.evaluable_col])
                reason = ""
                if previous_time is None:
                    reason = "flight_start"
                elif timestamp <= previous_time or timestamp - previous_time > self.config.max_gap_s:
                    reason = "invalid_or_large_gap"
                elif not evaluable:
                    reason = "not_evaluable"
                if reason:
                    state_pos = {channel: 0.0 for channel in self.calibration_}
                    state_neg = {channel: 0.0 for channel in self.calibration_}
                observed = 0
                if evaluable and not reason:
                    for channel, calibration in self.calibration_.items():
                        value = float(row[channel])
                        if not np.isfinite(value):
                            continue
                        z = np.clip(
                            (value - calibration["median"]) / calibration["mad"],
                            -self.config.z_clip,
                            self.config.z_clip,
                        )
                        state_pos[channel] = max(0.0, state_pos[channel] + z - k)
                        state_neg[channel] = max(0.0, state_neg[channel] - z - k)
                        observed += 1
                output.at[index, "cusum_score"] = max(
                    [*state_pos.values(), *state_neg.values()], default=0.0
                )
                output.at[index, "cusum_evaluable"] = observed > 0
                output.at[index, "cusum_reset_reason"] = reason
                previous_time = timestamp
        return output

    def to_dict(self) -> dict:
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before serialization")
        return {
            "schema_version": 1,
            "config": {
                "channels": list(self.config.channels),
                "reference_shift_z": self.config.reference_shift_z,
                "z_clip": self.config.z_clip,
                "max_gap_s": self.config.max_gap_s,
            },
            "calibration": self.calibration_,
        }

