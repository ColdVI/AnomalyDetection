"""Natural-only hierarchical conformal calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NATURAL_CALIBRATION_ROLE = "natural_clean_calibration"
LEVELS = (
    ("channel", "context_phase", "context_cadence"),
    ("channel", "context_phase"),
    ("channel",),
)


@dataclass(frozen=True)
class ConditionalCalibrationConfig:
    min_group_size: int

    def __post_init__(self) -> None:
        if self.min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")


class HierarchicalConformalCalibrator:
    """Empirical upper-tail p-values with deterministic context fallback."""

    def __init__(self, config: ConditionalCalibrationConfig):
        self.config = config
        self.groups_: dict[tuple[str, tuple[Any, ...]], np.ndarray] | None = None

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        required = {"channel", "context_phase", "context_cadence", "score"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"missing conformal columns: {sorted(missing)}")
        scores = pd.to_numeric(frame["score"], errors="coerce")
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("conformal scores must be finite and non-negative")

    def fit(
        self,
        calibration: pd.DataFrame,
        *,
        data_role: str,
        contains_synthetic: bool,
    ) -> "HierarchicalConformalCalibrator":
        if data_role != NATURAL_CALIBRATION_ROLE:
            raise ValueError("only natural_clean_calibration may fit conformal tails")
        if contains_synthetic:
            raise ValueError("synthetic data cannot enter conformal calibration")
        self._validate(calibration)
        groups: dict[tuple[str, tuple[Any, ...]], np.ndarray] = {}
        for columns in LEVELS:
            level = "+".join(columns)
            grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
            for key, group in calibration.groupby(grouper, sort=True, dropna=False):
                key_tuple = key if isinstance(key, tuple) else (key,)
                values = np.sort(group["score"].to_numpy(dtype=float))
                if len(values) >= self.config.min_group_size:
                    groups[(level, key_tuple)] = values
        if not any(level == "channel" for level, _ in groups):
            raise ValueError("no channel has enough natural calibration samples")
        self.groups_ = groups
        return self

    def transform(self, scored: pd.DataFrame) -> pd.DataFrame:
        self._validate(scored)
        if self.groups_ is None:
            raise RuntimeError("fit() must be called before transform()")
        rows: list[dict[str, Any]] = []
        for _, row in scored.iterrows():
            selected: tuple[str, np.ndarray] | None = None
            for columns in LEVELS:
                level = "+".join(columns)
                key = tuple(row[column] for column in columns)
                values = self.groups_.get((level, key))
                if values is not None:
                    selected = (level, values)
                    break
            if selected is None:
                raise ValueError(f"no calibration fallback for channel {row['channel']!r}")
            level, values = selected
            first_ge = int(np.searchsorted(values, float(row["score"]), side="left"))
            tail_count = len(values) - first_ge
            rows.append(
                {
                    "conformal_p_value": (1.0 + tail_count) / (len(values) + 1.0),
                    "calibration_level": level,
                    "calibration_n": int(len(values)),
                }
            )
        return pd.DataFrame.from_records(rows, index=scored.index)

    def to_dict(self) -> dict[str, Any]:
        if self.groups_ is None:
            raise RuntimeError("fit() must be called before serialization")
        groups = []
        for (level, key), values in sorted(
            self.groups_.items(), key=lambda item: (item[0][0], repr(item[0][1]))
        ):
            groups.append(
                {"level": level, "key": list(key), "values": values.astype(float).tolist()}
            )
        return {
            "schema_version": 1,
            "min_group_size": self.config.min_group_size,
            "groups": groups,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HierarchicalConformalCalibrator":
        calibrator = cls(
            ConditionalCalibrationConfig(min_group_size=int(payload["min_group_size"]))
        )
        calibrator.groups_ = {
            (row["level"], tuple(row["key"])): np.asarray(row["values"], dtype=float)
            for row in payload["groups"]
        }
        return calibrator

