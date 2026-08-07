"""G0 non-learned command/response physics baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from gecmis_calismalar.residual_v1.features.build import augment_physics

DEFAULT_CONFIG = Path("configs/residual_v1_g0.json")
SCORE_COLUMNS = ("flight_id", "t", "channel", "z")


def load_g0_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(SCORE_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"ScoreFrame missing columns: {sorted(missing)}")
    return frame.loc[:, SCORE_COLUMNS]


def command_no_response_score(
    frame: pd.DataFrame,
    *,
    command: str,
    response: str,
    command_threshold: float,
    response_stationary_threshold: float,
    window_s: float,
) -> pd.Series:
    time = pd.to_numeric(frame["t"], errors="coerce").to_numpy(float)
    command_values = pd.to_numeric(frame[command], errors="coerce").to_numpy(float)
    response_values = pd.to_numeric(frame[response], errors="coerce").to_numpy(float)
    score = np.full(len(frame), np.nan, dtype=float)
    for index, current in enumerate(time):
        start = int(np.searchsorted(time, current - window_s, side="left"))
        commands = command_values[start : index + 1]
        responses = response_values[start : index + 1]
        if len(commands) < 2 or not np.isfinite(commands).all() or not np.isfinite(responses).all():
            continue
        command_span = float(np.ptp(commands))
        response_span = float(np.ptp(responses))
        command_excess = command_span / command_threshold
        response_motion = response_span / response_stationary_threshold
        score[index] = max(0.0, command_excess - response_motion)
    return pd.Series(score, index=frame.index, name=f"{command}_{response}_no_response")


def _long_score(flight_id: str, time: pd.Series, channel: str, score: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": flight_id,
            "t": pd.to_numeric(time, errors="coerce"),
            "channel": channel,
            "z": pd.to_numeric(score, errors="coerce"),
        }
    )


def score_g0(
    flight: pd.DataFrame,
    *,
    flight_id: str,
    config: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    cfg = dict(config or load_g0_config())
    frame = augment_physics(flight)
    scores: list[pd.DataFrame] = []
    pairs = (
        ("aileron_cmd", "roll_rate"),
        ("elevator_cmd", "pitch_rate"),
        ("rudder_cmd", "yaw_rate"),
    )
    pair_scores = []
    for command, response in pairs:
        if command not in frame or response not in frame:
            continue
        pair_scores.append(
            command_no_response_score(
                frame,
                command=command,
                response=response,
                command_threshold=float(cfg["command_delta_thresholds"][command]),
                response_stationary_threshold=float(
                    cfg["response_stationary_thresholds"][response]
                ),
                window_s=float(cfg["response_window_s"]),
            )
        )
    if pair_scores:
        aggregate = pd.concat(pair_scores, axis=1).max(axis=1, skipna=True)
        aggregate = aggregate.mask(pd.concat(pair_scores, axis=1).isna().all(axis=1))
        scores.append(_long_score(flight_id, frame["t"], "G0_command_no_response", aggregate))

    if {"yaw_rate", "coordinated_turn_term"}.issubset(frame):
        error = (
            pd.to_numeric(frame["yaw_rate"], errors="coerce")
            - pd.to_numeric(frame["coordinated_turn_term"], errors="coerce")
        ).abs() / float(cfg["coordinated_turn_error_band_rad_s"])
        scores.append(_long_score(flight_id, frame["t"], "G0_coordinated_turn", error))

    if {"throttle_cmd", "airspeed_derivative"}.issubset(frame):
        throttle = pd.to_numeric(frame["throttle_cmd"], errors="coerce")
        acceleration = pd.to_numeric(frame["airspeed_derivative"], errors="coerce")
        deficit = (
            float(cfg["minimum_expected_airspeed_accel_m_s2"]) - acceleration
        ) / float(cfg["airspeed_accel_error_band_m_s2"])
        thrust_score = deficit.clip(lower=0.0).where(
            throttle >= float(cfg["throttle_high_ratio"]), 0.0
        )
        scores.append(_long_score(flight_id, frame["t"], "G0_thrust_speed", thrust_score))
    if not scores:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return validate_score_frame(pd.concat(scores, ignore_index=True))

