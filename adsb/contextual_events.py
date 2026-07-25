"""Adapter from contextual detector alarm emissions to simple-anomaly events."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from adsb.evaluation import EpisodeContract, alarm_episodes


COMMON_EVENT_COLUMNS = (
    "event_id",
    "flight_id",
    "start_time",
    "end_time",
    "duration_s",
    "n_samples",
)

CONTEXTUAL_EVENT_COLUMNS = (
    *COMMON_EVENT_COLUMNS,
    "detector",
    "channel",
    "profile",
    "pareto_v",
    "threshold_parameter",
    "threshold_value",
    "recipe",
)


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text or "none"


def contextual_alarm_events(
    meta: pd.DataFrame,
    alarm: np.ndarray,
    *,
    detector: str,
    channel: str,
    profile: str,
    pareto_v: float,
    threshold_parameter: str,
    threshold_value: float,
    recipe: str,
    contract: EpisodeContract = EpisodeContract(),
) -> pd.DataFrame:
    """Collapse alarm rows and emit the common ``simple_anomaly`` event prefix.

    ``start_time`` and ``end_time`` are alarm-emission bounds, not injected truth
    bounds.  Detector metadata follows the six common columns rather than
    changing their meanings.
    """

    episodes = alarm_episodes(meta, alarm, contract=contract)
    if episodes.empty:
        return pd.DataFrame(columns=CONTEXTUAL_EVENT_COLUMNS)
    rows: list[dict[str, object]] = []
    prefix = "_".join(
        (
            _slug(detector),
            _slug(channel),
            _slug(profile),
            _slug(f"V{float(pareto_v):g}"),
        )
    )
    for flight_id, group in episodes.groupby("flight_id", sort=False, dropna=False):
        ordered = group.sort_values(["episode_start", "episode_end"], kind="mergesort")
        for number, episode in enumerate(ordered.itertuples(index=False), start=1):
            start = float(episode.episode_start)
            end = float(episode.episode_end)
            rows.append(
                {
                    "event_id": f"{flight_id}_{prefix}_{number:03d}",
                    "flight_id": flight_id,
                    "start_time": start,
                    "end_time": end,
                    "duration_s": end - start,
                    "n_samples": int(episode.n_emissions),
                    "detector": detector,
                    "channel": channel,
                    "profile": profile,
                    "pareto_v": float(pareto_v),
                    "threshold_parameter": threshold_parameter,
                    "threshold_value": float(threshold_value),
                    "recipe": recipe,
                }
            )
    return pd.DataFrame(rows, columns=CONTEXTUAL_EVENT_COLUMNS)
