"""Strict normal-only robust scaling for contextual model channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NATURAL_FIT_ROLE = "natural_clean_fit"
ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class StrictScalingConfig:
    clip: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("clip must be finite and > 0")


class StrictNaturalRobustScaler:
    """Median/MAD scaler that excludes zero-MAD channels without a floor."""

    def __init__(self, config: StrictScalingConfig):
        self.config = config
        self.calibration_: dict[str, dict[str, float]] | None = None
        self.excluded_channels_: tuple[str, ...] = ()

    def fit(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        *,
        data_role: str,
        contains_synthetic: bool,
    ) -> "StrictNaturalRobustScaler":
        if data_role != NATURAL_FIT_ROLE:
            raise ValueError("Only natural_clean_fit may fit the strict scaler")
        if contains_synthetic:
            raise ValueError("Synthetic data cannot enter strict scaling fit")
        missing = set(columns).difference(frame.columns)
        if missing:
            raise KeyError(f"Missing scaling columns: {sorted(missing)}")

        calibration: dict[str, dict[str, float]] = {}
        excluded: list[str] = []
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                excluded.append(column)
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)) * ROBUST_MAD_SCALE)
            if mad == 0.0:
                excluded.append(column)
                continue
            calibration[column] = {"median": median, "mad": mad}
        if not calibration:
            raise ValueError("All requested channels have no data or MAD=0")
        self.calibration_ = calibration
        self.excluded_channels_ = tuple(excluded)
        return self

    @property
    def active_channels(self) -> tuple[str, ...]:
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before active_channels")
        return tuple(self.calibration_)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before transform()")
        result = pd.DataFrame(index=frame.index)
        for column, calibration in self.calibration_.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            scaled = (values - calibration["median"]) / calibration["mad"]
            result[column] = scaled.clip(-self.config.clip, self.config.clip)
        return result

    def to_dict(self) -> dict[str, object]:
        if self.calibration_ is None:
            raise RuntimeError("fit() must be called before serialization")
        return {
            "clip": self.config.clip,
            "calibration": self.calibration_,
            "excluded_channels": list(self.excluded_channels_),
            "mad_zero_policy": "exclude_without_floor",
            "fit_role": NATURAL_FIT_ROLE,
        }
