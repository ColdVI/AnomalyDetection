"""RESIDUAL-V1 wrapper around the shared two-sided Page CUSUM core."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_core.sequential import MultiChannelPageCUSUM, PageCUSUMConfig


def score_cusum_channel(
    frame: pd.DataFrame,
    *,
    channel: str,
    k: float = 1.0,
    z_clip: float = 8.0,
    max_gap_s: float = 1.0,
) -> pd.DataFrame:
    """Score one already-robust-scaled channel with the shared two-sided core."""

    required = {"flight_id", "t", "z"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{channel}: missing CUSUM columns {missing}")
    if min(k, z_clip, max_gap_s) <= 0.0:
        raise ValueError("CUSUM parameters must be positive")
    work = pd.DataFrame(
        {
            "flight_id": frame["flight_id"].astype(str),
            "timestamp_s": pd.to_numeric(frame["t"], errors="coerce"),
            channel: pd.to_numeric(frame["z"], errors="coerce"),
        },
        index=frame.index,
    )
    work["evaluable"] = np.isfinite(work["timestamp_s"]) & np.isfinite(work[channel])
    detector = MultiChannelPageCUSUM(
        PageCUSUMConfig(
            channels=(channel,),
            # The shared core internally uses reference_shift_z / 2 as k.
            reference_shift_z=2.0 * k,
            z_clip=z_clip,
            max_gap_s=max_gap_s,
        )
    )
    # Scaling is already frozen from train-normal median/MAD. Identity
    # calibration prevents an accidental second robust scaling in the core.
    detector.calibration_ = {channel: {"median": 0.0, "mad": 1.0}}
    scored = detector.score(work)
    output = frame.loc[:, [column for column in ("flight_id", "t", "z") if column in frame]].copy()
    output["channel"] = channel
    output["cusum_score"] = scored["cusum_score"].to_numpy(float)
    output["cusum_evaluable"] = scored["cusum_evaluable"].to_numpy(bool)
    output["cusum_reset_reason"] = scored["cusum_reset_reason"].to_numpy(str)
    return output


def threshold_crossing_alarms(
    scores: pd.DataFrame,
    *,
    threshold: float,
    refractory_s: float = 60.0,
) -> pd.DataFrame:
    """Emit rising-threshold crossings, with a per-flight refractory guard."""

    if threshold <= 0.0 or refractory_s <= 0.0:
        raise ValueError("threshold and refractory_s must be positive")
    required = {"flight_id", "t", "channel", "cusum_score", "cusum_reset_reason"}
    missing = sorted(required - set(scores))
    if missing:
        raise ValueError(f"missing alarm columns {missing}")
    events = []
    for flight_id, flight in scores.groupby("flight_id", sort=False):
        previous_score = 0.0
        last_alarm = -float("inf")
        for row in flight.itertuples(index=False):
            timestamp = float(row.t)
            score = float(row.cusum_score)
            if row.cusum_reset_reason:
                previous_score = 0.0
            crossing = previous_score < threshold <= score
            if crossing and timestamp - last_alarm >= refractory_s:
                events.append(
                    {
                        "flight_id": str(flight_id),
                        "t_alarm": timestamp,
                        "channel": str(row.channel),
                        "channel_contribution": score,
                    }
                )
                last_alarm = timestamp
            previous_score = score
    return pd.DataFrame(
        events,
        columns=("flight_id", "t_alarm", "channel", "channel_contribution"),
    )
