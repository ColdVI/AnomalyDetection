"""60-second block-bootstrap CUSUM threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from residual_v1.decision.cusum import score_cusum_channel, threshold_crossing_alarms


@dataclass(frozen=True)
class CalibrationConfig:
    k: float = 1.0
    z_clip: float = 8.0
    max_gap_s: float = 1.0
    refractory_s: float = 60.0
    block_s: float = 60.0
    repetitions: int = 500
    seed: int = 11


class InsufficientCalibrationExposure(ValueError):
    """Raised when the target rate is below the data's one-alarm resolution."""


def normal_block_exposure(frame: pd.DataFrame, *, block_s: float = 60.0) -> dict:
    durations = []
    for _, flight in frame.groupby("flight_id", sort=False):
        flight = flight.sort_values("t", kind="stable")
        start = float(flight["t"].iloc[0])
        block_index = np.floor((pd.to_numeric(flight["t"]) - start) / block_s).astype(int)
        for _, block in flight.groupby(block_index, sort=True):
            if len(block) < 2:
                continue
            duration = float(block["t"].iloc[-1] - block["t"].iloc[0])
            if duration > 0.0:
                durations.append(duration)
    return {
        "block_count": len(durations),
        "exposure_hours": float(sum(durations) / 3600.0),
    }


def _score_blocks(frame: pd.DataFrame, *, channel: str, config: CalibrationConfig) -> list[pd.DataFrame]:
    blocks = []
    for flight_id, flight in frame.groupby("flight_id", sort=False):
        flight = flight.sort_values("t", kind="stable").copy()
        start = float(flight["t"].iloc[0])
        block_index = np.floor((pd.to_numeric(flight["t"]) - start) / config.block_s).astype(int)
        for index, block in flight.groupby(block_index, sort=True):
            if len(block) < 2:
                continue
            duration = float(block["t"].iloc[-1] - block["t"].iloc[0])
            if duration <= 0.0:
                continue
            synthetic = block.copy()
            synthetic["flight_id"] = f"{flight_id}::block{index}"
            scored = score_cusum_channel(
                synthetic,
                channel=channel,
                k=config.k,
                z_clip=config.z_clip,
                max_gap_s=config.max_gap_s,
            )
            scored.attrs["duration_s"] = duration
            blocks.append(scored)
    if not blocks:
        raise ValueError(f"{channel}: no normal bootstrap blocks")
    return blocks


def calibrate_channel_threshold(
    normal: pd.DataFrame,
    *,
    channel: str,
    target_alarms_per_hour: float,
    config: CalibrationConfig = CalibrationConfig(),
) -> dict:
    if target_alarms_per_hour <= 0.0:
        raise ValueError("target_alarms_per_hour must be positive")
    blocks = _score_blocks(normal, channel=channel, config=config)
    durations = np.asarray([block.attrs["duration_s"] for block in blocks], dtype=float)
    exposure_hours = float(durations.sum() / 3600.0)
    minimum_hours = 1.0 / target_alarms_per_hour
    if exposure_hours < minimum_hours:
        raise InsufficientCalibrationExposure(
            f"{channel}: {exposure_hours:.6f} normal hours cannot resolve "
            f"{target_alarms_per_hour:.6f} alarms/hour; minimum one-alarm exposure is "
            f"{minimum_hours:.6f} hours"
        )
    rng = np.random.default_rng(config.seed)
    samples = rng.integers(0, len(blocks), size=(config.repetitions, len(blocks)))
    sampled_hours = durations[samples].sum(axis=1) / 3600.0

    def bootstrap_rates(threshold: float) -> np.ndarray:
        counts = np.asarray(
            [
                len(
                    threshold_crossing_alarms(
                        block,
                        threshold=threshold,
                        refractory_s=config.refractory_s,
                    )
                )
                for block in blocks
            ],
            dtype=float,
        )
        return counts[samples].sum(axis=1) / sampled_hours

    max_score = max(float(block["cusum_score"].max()) for block in blocks)
    lower = np.finfo(float).eps
    lower_rates = bootstrap_rates(lower)
    if float(lower_rates.mean()) <= target_alarms_per_hour:
        threshold = lower
        rates = lower_rates
    else:
        upper = max(max_score + 1.0, 1.0)
        for _ in range(50):
            midpoint = (lower + upper) / 2.0
            candidate = bootstrap_rates(midpoint)
            if float(candidate.mean()) <= target_alarms_per_hour:
                upper = midpoint
            else:
                lower = midpoint
        threshold = upper
        rates = bootstrap_rates(threshold)
    observed_alarms = sum(
        len(
            threshold_crossing_alarms(
                block,
                threshold=threshold,
                refractory_s=config.refractory_s,
            )
        )
        for block in blocks
    )
    return {
        "channel": channel,
        "threshold_h": float(threshold),
        "target_alarms_per_flight_hour": float(target_alarms_per_hour),
        "selection_rule": "smallest_binary_searched_h_with_bootstrap_mean_rate_at_or_below_target",
        "observed_normal_alarms": int(observed_alarms),
        "observed_normal_hours": exposure_hours,
        "minimum_one_alarm_resolution_hours": minimum_hours,
        "bootstrap_block_count": len(blocks),
        "bootstrap_repetitions": config.repetitions,
        "bootstrap_rate_mean": float(rates.mean()),
        "bootstrap_rate_median": float(np.median(rates)),
        "bootstrap_rate_p95": float(np.quantile(rates, 0.95)),
        "bootstrap_rates": rates.tolist(),
        "config": {
            "k": config.k,
            "z_clip": config.z_clip,
            "max_gap_s": config.max_gap_s,
            "refractory_s": config.refractory_s,
            "block_s": config.block_s,
            "seed": config.seed,
        },
    }
