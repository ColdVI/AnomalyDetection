"""Observed-sample waypoint V-turn detection for the ALFA R6 mask."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path("configs/residual_v1_waypoint_mask.json")
CONFIG_KEYS = (
    "maximum_turn_distance_m",
    "trend_window_s",
    "minimum_approach_excursion_m",
    "minimum_departure_excursion_m",
    "minimum_event_separation_s",
    "mask_buffer_s",
)


def validate_waypoint_config(config: Mapping[str, float]) -> dict[str, float]:
    missing = sorted(set(CONFIG_KEYS) - set(config))
    extra = sorted(set(config) - set(CONFIG_KEYS))
    if missing or extra:
        raise ValueError(f"waypoint config keys differ; missing={missing}, extra={extra}")
    result = {key: float(config[key]) for key in CONFIG_KEYS}
    if not all(np.isfinite(value) and value > 0.0 for value in result.values()):
        raise ValueError("all waypoint mask parameters must be finite and positive")
    return result


def load_waypoint_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, float]:
    return validate_waypoint_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _turn_candidates(
    time: np.ndarray,
    distance: np.ndarray,
    config: Mapping[str, float],
) -> list[int]:
    window_s = config["trend_window_s"]
    candidates: list[int] = []
    for index, (current_t, current_distance) in enumerate(zip(time, distance, strict=True)):
        if not np.isfinite(current_distance) or current_distance > config["maximum_turn_distance_m"]:
            continue
        # A telemetry initialisation/reset at either flight edge is not a
        # two-sided V-turn. Both frozen trend legs must be fully observable.
        if current_t - time[0] < window_s or time[-1] - current_t < window_s:
            continue
        past_start = int(np.searchsorted(time, current_t - window_s, side="left"))
        future_end = int(np.searchsorted(time, current_t + window_s, side="right"))
        past = distance[past_start:index]
        future = distance[index + 1 : future_end]
        finite_past = past[np.isfinite(past)]
        finite_future = future[np.isfinite(future)]
        if not len(finite_past) or not len(finite_future):
            continue
        approach = float(finite_past.max() - current_distance)
        departure = float(finite_future.max() - current_distance)
        if (
            approach >= config["minimum_approach_excursion_m"]
            and departure >= config["minimum_departure_excursion_m"]
        ):
            candidates.append(index)
    return candidates


def _merge_candidates(
    candidates: list[int],
    time: np.ndarray,
    distance: np.ndarray,
    separation_s: float,
) -> list[int]:
    if not candidates:
        return []
    clusters: list[list[int]] = [[candidates[0]]]
    for index in candidates[1:]:
        if time[index] - time[clusters[-1][-1]] <= separation_s:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    events: list[int] = []
    for cluster in clusters:
        minimum = min(distance[index] for index in cluster)
        tied = [index for index in cluster if distance[index] == minimum]
        events.append(tied[len(tied) // 2])
    return events


def label_waypoint_boundaries(
    flight: pd.DataFrame,
    *,
    config: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return V-turn event points and their frozen time-based exclusion mask.

    The detector uses only observed/aligned samples. It never interpolates,
    resamples, or fills missing waypoint-distance values.
    """

    if "t" not in flight or "waypoint_distance" not in flight:
        raise ValueError("waypoint mask requires t and waypoint_distance")
    cfg = validate_waypoint_config(config or load_waypoint_config())
    time = pd.to_numeric(flight["t"], errors="coerce").to_numpy(float)
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("t must be finite and strictly increasing")
    distance = pd.to_numeric(flight["waypoint_distance"], errors="coerce").to_numpy(float)
    candidates = _turn_candidates(time, distance, cfg)
    events = _merge_candidates(
        candidates,
        time,
        distance,
        cfg["minimum_event_separation_s"],
    )
    event_mask = np.zeros(len(time), dtype=bool)
    boundary = np.zeros(len(time), dtype=bool)
    for index in events:
        event_mask[index] = True
        boundary |= np.abs(time - time[index]) <= cfg["mask_buffer_s"]
    return pd.DataFrame(
        {"t": time, "waypoint_event": event_mask, "waypoint_boundary": boundary},
        index=flight.index,
    )
