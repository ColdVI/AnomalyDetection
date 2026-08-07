"""Vectorized-write performance twin of adsb.contextual_decision.apply_detector_profile.

adsb/contextual_decision.py::apply_detector_profile is frozen (its recurrence
semantics are the tested source of truth) and MUST NOT be edited to add a
performance path -- ADR-040 recorded that its per-row ``.loc[index, col] =
value`` writes (four pandas-aligned scalar assignments per row) make an
alpha-grid x profile sweep impractical at real development-day volume
(millions of rows).

This module reproduces the exact same recurrence, row by row, using plain
numpy arrays inside the loop and writing to the output DataFrame ONCE per
profile call instead of once per row. Nothing about the math changes --
tests/test_adsb_contextual_decision_fast.py asserts the two implementations
agree exactly on multiple synthetic scenarios covering every temporal mode,
gap resets, and multi-flight inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adsb.contextual_decision import ChannelAlertBudget, DetectorProfile


def apply_detector_profile_fast(
    calibrated: pd.DataFrame,
    *,
    profile: DetectorProfile,
    budget: ChannelAlertBudget,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Same contract and output schema as apply_detector_profile, vectorized writes."""

    required = {flight_id_col, time_col, "channel", "conformal_p_value"}
    missing = required.difference(calibrated.columns)
    if missing:
        raise KeyError(f"Missing decision columns: {sorted(missing)}")
    if profile.channel not in budget.channel_alpha:
        raise ValueError(f"No alert allocation exists for channel {profile.channel!r}")
    if not calibrated["channel"].eq(profile.channel).all():
        raise ValueError("A detector profile accepts exactly one score channel")
    alpha = budget.channel_alpha[profile.channel]

    index_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    alarm_parts: list[np.ndarray] = []
    reason_parts: list[np.ndarray] = []

    for flight_id, group in calibrated.groupby(flight_id_col, sort=False):
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy(float)
        p_values = pd.to_numeric(group["conformal_p_value"], errors="coerce").to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError(f"{flight_id}: decision timestamps must be finite and sorted")
        if not np.isfinite(p_values).all() or np.any((p_values <= 0) | (p_values > 1)):
            raise ValueError(f"{flight_id}: conformal p-values must be in (0, 1]")

        n = len(times)
        evidence = np.zeros(n, dtype=float)
        alarm = np.zeros(n, dtype=bool)
        reason = np.full(n, "", dtype=object)
        reason[0] = "flight_start"

        if profile.mode == "instant":
            # No cross-row state feeds the alarm decision itself in this mode
            # (see apply_detector_profile: only persistence/accumulation ever
            # mutate `evidence`), so the whole profile is embarrassingly
            # parallel across rows -- no python loop needed at all.
            alarm = p_values <= alpha
            if n > 1:
                dt = np.diff(times)
                gap_bad = (dt <= 0) | (dt > profile.max_gap_s)
                tail = reason[1:]
                tail[gap_bad] = "invalid_or_large_gap"
                reason[1:] = tail
        else:
            ev = 0.0
            prev_time = times[0]
            persistence_s = profile.persistence_s
            reference_surprisal = profile.reference_surprisal
            accumulation_threshold = profile.accumulation_threshold
            mode_is_persistence = profile.mode == "persistence"
            max_gap_s = profile.max_gap_s
            for i in range(n):
                if i > 0:
                    dt_s = times[i] - prev_time
                    if dt_s <= 0 or dt_s > max_gap_s:
                        ev = 0.0
                        reason[i] = "invalid_or_large_gap"
                    elif mode_is_persistence:
                        ev = ev + dt_s if p_values[i] <= alpha else 0.0
                    else:
                        surprise = -np.log(p_values[i])
                        ev = max(0.0, ev + dt_s * (surprise - reference_surprisal))
                evidence[i] = ev
                alarm[i] = ev >= (persistence_s if mode_is_persistence else accumulation_threshold)
                prev_time = times[i]

        index_parts.append(group.index.to_numpy())
        evidence_parts.append(evidence)
        alarm_parts.append(alarm)
        reason_parts.append(reason)

    combined_index = pd.Index(np.concatenate(index_parts)) if index_parts else calibrated.index
    combined = pd.DataFrame(
        {
            "temporal_evidence": np.concatenate(evidence_parts) if evidence_parts else np.array([], dtype=float),
            "alarm": np.concatenate(alarm_parts) if alarm_parts else np.array([], dtype=bool),
            "reset_reason": np.concatenate(reason_parts) if reason_parts else np.array([], dtype=object),
        },
        index=combined_index,
    ).reindex(calibrated.index)

    output = pd.DataFrame(index=calibrated.index)
    output["anomaly_type"] = profile.anomaly_type
    output["channel"] = profile.channel
    output["alert_alpha"] = alpha
    output["temporal_evidence"] = combined["temporal_evidence"].to_numpy(dtype=float)
    output["alarm"] = combined["alarm"].to_numpy(dtype=bool)
    output["reset_reason"] = combined["reset_reason"].to_numpy(dtype=object)
    return output
