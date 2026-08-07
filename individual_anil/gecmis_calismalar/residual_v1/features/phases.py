"""Rule-based, non-learned flight phase segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path("configs/residual_v1_phases.json")
PHASES = ("ground", "takeoff", "cruise", "maneuver", "landing")


def load_phase_config(
    dataset: str, path: str | Path = DEFAULT_CONFIG
) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset not in payload:
        raise ValueError(f"missing phase config for {dataset}")
    return {key: float(value) for key, value in payload[dataset].items()}


def _numeric(frame: pd.DataFrame, candidates: tuple[str, ...], default: float = np.nan) -> pd.Series:
    for candidate in candidates:
        if candidate in frame:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _altitude_agl(frame: pd.DataFrame, ground: pd.Series) -> pd.Series:
    if "altitude" in frame:
        altitude = pd.to_numeric(frame["altitude"], errors="coerce")
    elif "altitude_hud" in frame:
        altitude = pd.to_numeric(frame["altitude_hud"], errors="coerce")
    elif "local_z" in frame:
        altitude = -pd.to_numeric(frame["local_z"], errors="coerce")
    else:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    ground_values = altitude.loc[ground & altitude.notna()]
    reference = float(ground_values.median()) if not ground_values.empty else float(altitude.min())
    return altitude - reference


def label_phases(
    flight: pd.DataFrame,
    *,
    config: Mapping[str, float] | None = None,
    dataset: str = "alfa",
) -> pd.DataFrame:
    """Return ``t``, categorical ``phase``, and the transition mask."""

    if "t" not in flight:
        raise ValueError("flight must contain t")
    cfg = dict(config or load_phase_config(dataset))
    time = pd.to_numeric(flight["t"], errors="coerce")
    if time.isna().any() or not bool((time.diff().dropna() > 0).all()):
        raise ValueError("t must be finite and strictly increasing")
    ground_speed = _numeric(flight, ("ground_speed", "horizontal_speed", "local_speed"))
    if ground_speed.isna().all() and {"local_vx", "local_vy"}.issubset(flight):
        ground_speed = np.sqrt(
            pd.to_numeric(flight["local_vx"], errors="coerce") ** 2
            + pd.to_numeric(flight["local_vy"], errors="coerce") ** 2
        )
    climb = _numeric(flight, ("climb_rate", "vertical_rate"))
    if climb.isna().all() and "local_vz" in flight:
        climb = -pd.to_numeric(flight["local_vz"], errors="coerce")
    roll = _numeric(flight, ("roll", "roll_rad"))
    roll_rate = _numeric(flight, ("roll_rate",))

    ground = (
        (ground_speed < cfg["ground_speed_max_mps"])
        & (climb.abs() < cfg["ground_climb_abs_max_mps"])
    ).fillna(False)
    altitude_agl = _altitude_agl(flight, ground)
    low_altitude = (altitude_agl <= cfg["transition_altitude_agl_max_m"]).fillna(False)
    maneuver = (
        (np.rad2deg(roll).abs() > cfg["maneuver_roll_abs_min_deg"])
        | (np.rad2deg(roll_rate).abs() > cfg["maneuver_roll_rate_abs_min_deg_s"])
    ).fillna(False)
    takeoff = ((climb > cfg["takeoff_climb_min_mps"]) & low_altitude).fillna(False)
    landing = ((climb < cfg["landing_climb_max_mps"]) & low_altitude).fillna(False)

    phase = np.full(len(flight), "cruise", dtype=object)
    phase[landing.to_numpy()] = "landing"
    phase[takeoff.to_numpy()] = "takeoff"
    phase[maneuver.to_numpy()] = "maneuver"
    phase[ground.to_numpy()] = "ground"

    transition_indices = np.flatnonzero(np.r_[False, phase[1:] != phase[:-1]])
    boundary = np.zeros(len(flight), dtype=bool)
    buffer_s = cfg["boundary_buffer_s"]
    t = time.to_numpy(float)
    for index in transition_indices:
        boundary |= np.abs(t - t[index]) <= buffer_s
    return pd.DataFrame({"t": t, "phase": phase, "phase_boundary": boundary}, index=flight.index)

