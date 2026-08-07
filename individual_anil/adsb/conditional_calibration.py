"""Hierarchical conditional conformal calibration for channel scores.

Calibration accepts natural-clean calibration data only.  Synthetic rows,
rehearsal rows, and evaluation rows are rejected by the public fit contract.
No alert alpha or threshold has a default here: an operational budget must be
supplied by the caller after it has been pre-registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


NATURAL_CALIBRATION_ROLE = "natural_clean_calibration"
_LEVELS = (
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


def validate_alert_alpha(alpha: float | None) -> float:
    """Require an explicit pre-registered per-channel alert allocation."""

    if alpha is None:
        raise ValueError("alert alpha is mandatory; no implicit threshold is allowed")
    value = float(alpha)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("alert alpha must be finite and between 0 and 1")
    return value


class HierarchicalConformalCalibrator:
    """Empirical upper-tail p-values with deterministic context fallback."""

    def __init__(self, config: ConditionalCalibrationConfig):
        self.config = config
        self._groups: dict[tuple[str, tuple[Any, ...]], np.ndarray] | None = None

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        required = {"channel", "context_phase", "context_cadence", "score"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing conformal columns: {sorted(missing)}")
        scores = pd.to_numeric(frame["score"], errors="coerce")
        if not np.isfinite(scores).all():
            raise ValueError("Calibration scores must all be finite")
        if (scores < 0).any():
            raise ValueError("Calibration scores must be non-negative")

    def fit(
        self,
        calibration: pd.DataFrame,
        *,
        data_role: str,
        contains_synthetic: bool,
    ) -> "HierarchicalConformalCalibrator":
        if data_role != NATURAL_CALIBRATION_ROLE:
            raise ValueError("Only natural_clean_calibration may fit conformal tails")
        if contains_synthetic:
            raise ValueError("Synthetic data cannot enter conformal calibration")
        self._validate_frame(calibration)

        groups: dict[tuple[str, tuple[Any, ...]], np.ndarray] = {}
        for columns in _LEVELS:
            level_name = "+".join(columns)
            grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
            for key, group in calibration.groupby(grouper, sort=True, dropna=False):
                key_tuple = key if isinstance(key, tuple) else (key,)
                values = np.sort(group["score"].to_numpy(dtype=float))
                if len(values) >= self.config.min_group_size:
                    groups[(level_name, key_tuple)] = values
        if not any(level == "channel" for level, _ in groups):
            raise ValueError("No channel has enough natural calibration samples")
        self._groups = groups
        return self

    def _lookup(self, row: pd.Series) -> tuple[str, np.ndarray]:
        if self._groups is None:
            raise RuntimeError("fit() must be called before transform()")
        for columns in _LEVELS:
            level_name = "+".join(columns)
            key = tuple(row[column] for column in columns)
            values = self._groups.get((level_name, key))
            if values is not None:
                return level_name, values
        raise ValueError(f"No calibration fallback exists for channel {row['channel']!r}")

    def transform(self, scored: pd.DataFrame) -> pd.DataFrame:
        self._validate_frame(scored)
        records: list[dict[str, Any]] = []
        for _, row in scored.iterrows():
            level, values = self._lookup(row)
            score = float(row["score"])
            first_ge = int(np.searchsorted(values, score, side="left"))
            tail_count = len(values) - first_ge
            records.append(
                {
                    "conformal_p_value": (1.0 + tail_count) / (len(values) + 1.0),
                    "calibration_level": level,
                    "calibration_n": int(len(values)),
                }
            )
        return pd.DataFrame.from_records(records, index=scored.index)

    def alarms(self, scored: pd.DataFrame, *, alpha: float | None) -> pd.DataFrame:
        explicit_alpha = validate_alert_alpha(alpha)
        result = self.transform(scored)
        result["alert_alpha"] = explicit_alpha
        result["alarm"] = result["conformal_p_value"].le(explicit_alpha)
        return result
