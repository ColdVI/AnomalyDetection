"""Cumulative, whole-flight persistence over conformal p-values (contextual_physics_v2).

ADR-042 observed that ``adsb/cusum.py``'s ``east_north_cusum`` channel (V=5.0:
%49.7 recall) generalized far better than contextual_physics_v1's LSTM channels
(V=5.0: <%6 recall on 4 of 5 profiles), and attributed this to a structural
difference in evidence accumulation: CUSUM accumulates in raw robust-z space
without resetting for the whole flight, while the LSTM's "persistence" mode was
a fixed 30-second sliding window -- far less time to build up evidence.

This module gives the LSTM's ``conformal_p_value`` stream the same
whole-flight, never-time-windowed accumulation CUSUM already has, using the
IDENTICAL causal reset-boundary rules as :class:`adsb.cusum.VectorPageCUSUM`
(flight start, unknown/on-ground status, ground transition, invalid or overlong
time gap) so the two detectors stay directly comparable. ``adsb/cusum.py``
itself is not modified.

Design note: a conformal p-value is uniform on (0, 1) under the null by
construction, so ``-log10(p)`` has a known, closed-form null mean
(``1 / ln(10)``) -- unlike CUSUM's raw residual channels, no empirical
median/MAD fit is required to find a reference shift; ``fit()`` only needs the
sign/eligibility of that reference, supplied as a mandatory multiplier so the
accumulator drifts down (not up) on pure noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Any

import numpy as np
import pandas as pd

NULL_MEAN_SURPRISE = 1.0 / log(10.0)  # E[-log10(U)] for U ~ Uniform(0, 1)


@dataclass(frozen=True)
class PersistenceV2Config:
    """Pre-registered contract; no threshold search, mirrors CusumConfig's discipline.

    ``reference_shift_multiplier`` must be > 1.0: the reference subtracted each
    step is ``reference_shift_multiplier * NULL_MEAN_SURPRISE``, i.e. strictly
    above the null-noise mean so the accumulator decays under pure noise and
    only grows under sustained, genuinely low p-values.
    """

    reference_shift_multiplier: float
    threshold_h: float
    max_gap_s: float
    missing_reset_s: float
    surprise_clip: float
    channel_col: str = "channel"
    flight_id_col: str = "flight_id"
    time_col: str = "timestamp_utc"
    on_ground_col: str = "on_ground"
    p_value_col: str = "conformal_p_value"

    def __post_init__(self) -> None:
        if self.reference_shift_multiplier <= 1.0:
            raise ValueError("reference_shift_multiplier must be > 1.0")
        for name in ("threshold_h", "max_gap_s", "surprise_clip"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not np.isfinite(self.missing_reset_s) or self.missing_reset_s < 0.0:
            raise ValueError("missing_reset_s must be finite and >= 0")

    @property
    def reference_shift(self) -> float:
        return self.reference_shift_multiplier * NULL_MEAN_SURPRISE


def _ground_status(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    return bool(value)


class CumulativeConformalPersistence:
    """Causal, whole-flight, per-channel cumulative surprise over p-values.

    Expects a long-form frame with one row per (flight, timestamp, channel)
    observation -- the same shape :class:`adsb.conditional_calibration.
    HierarchicalConformalCalibrator` scores, joined back with ``timestamp_utc``/
    ``on_ground`` from the source feature table (this module does not compute
    conformal p-values itself).
    """

    def __init__(self, config: PersistenceV2Config):
        self.config = config

    def _require_columns(self, df: pd.DataFrame) -> None:
        required = {
            self.config.channel_col,
            self.config.flight_id_col,
            self.config.time_col,
            self.config.on_ground_col,
            self.config.p_value_col,
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"Missing persistence_v2 columns: {missing}")

    def score(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Score rows in their given per-channel stream order without looking ahead.

        Reset boundaries are identical to ``VectorPageCUSUM.score_rows``: flight
        start, unknown/on-ground status, ground transition, invalid or backward
        time, and gaps above ``max_gap_s``. Equal timestamps are skipped without
        changing state. A missing/NaN p-value carries state for less than
        ``missing_reset_s`` elapsed seconds, then resets only that channel.
        """
        self._require_columns(scored)
        reference = self.config.reference_shift
        out_state = np.zeros(len(scored), dtype=float)
        out_alarm = np.zeros(len(scored), dtype=bool)
        out_evaluable = np.zeros(len(scored), dtype=bool)
        out_reason = np.full(len(scored), "none", dtype=object)

        for _, group in scored.groupby(self.config.channel_col, sort=False):
            order = group.index.to_numpy()
            times = pd.to_numeric(group[self.config.time_col], errors="coerce").to_numpy(float)
            grounds = group[self.config.on_ground_col].to_numpy(dtype=object)
            flights = group[self.config.flight_id_col].to_numpy(dtype=object)
            p_values = pd.to_numeric(group[self.config.p_value_col], errors="coerce").to_numpy(float)

            state = 0.0
            missing_elapsed_s = 0.0
            previous_flight: Any = None
            previous_time = float("nan")
            previous_ground: bool | None = None
            initialized = False

            for position, row_index in enumerate(order):
                flight = flights[position]
                timestamp = times[position]
                ground = _ground_status(grounds[position])
                eligible = False
                dt = float("nan")
                reason = "none"

                if not initialized or flight != previous_flight:
                    state, missing_elapsed_s = 0.0, 0.0
                    reason = "flight_start"
                elif ground is None or previous_ground is None:
                    state, missing_elapsed_s = 0.0, 0.0
                    reason = "unknown_ground_status"
                elif ground:
                    state, missing_elapsed_s = 0.0, 0.0
                    reason = "on_ground"
                elif previous_ground:
                    state, missing_elapsed_s = 0.0, 0.0
                    reason = "ground_transition"
                elif not np.isfinite(timestamp) or not np.isfinite(previous_time):
                    state, missing_elapsed_s = 0.0, 0.0
                    reason = "invalid_time"
                else:
                    dt = timestamp - previous_time
                    if dt < 0.0:
                        state, missing_elapsed_s = 0.0, 0.0
                        reason = "negative_dt"
                    elif dt > self.config.max_gap_s:
                        state, missing_elapsed_s = 0.0, 0.0
                        reason = "long_gap"
                    elif dt == 0.0:
                        reason = "zero_dt"
                    else:
                        eligible = True

                evaluable = False
                if eligible:
                    p_value = p_values[position]
                    if np.isfinite(p_value) and p_value > 0.0:
                        surprise = min(-np.log10(p_value), self.config.surprise_clip)
                        state = max(0.0, state + surprise - reference)
                        missing_elapsed_s = 0.0
                        evaluable = True
                    else:
                        missing_elapsed_s += dt
                        if missing_elapsed_s >= self.config.missing_reset_s:
                            state = 0.0

                out_state[row_index] = state
                out_evaluable[row_index] = evaluable
                out_alarm[row_index] = evaluable and state > self.config.threshold_h
                out_reason[row_index] = reason

                previous_flight = flight
                previous_time = timestamp
                previous_ground = ground
                initialized = True

        return pd.DataFrame(
            {
                "persistence_v2_state": out_state,
                "persistence_v2_alarm": out_alarm,
                "persistence_v2_evaluable": out_evaluable,
                "persistence_v2_reset_reason": out_reason,
            },
            index=scored.index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "reference_shift": self.config.reference_shift,
            "null_mean_surprise": NULL_MEAN_SURPRISE,
            "joint_statistic": "per_channel_cumulative_surprise",
            "alarm_comparator": ">",
            "missing_policy": "carry_then_elapsed_time_reset",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CumulativeConformalPersistence":
        return cls(PersistenceV2Config(**payload["config"]))
