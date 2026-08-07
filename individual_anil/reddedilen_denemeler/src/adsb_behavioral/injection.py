"""Deterministic behavioral anomaly injection on copied ADS-B flights."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


INJECTION_TYPES = (
    "position_jump",
    "position_drift",
    "altitude_bias",
    "speed_bias",
    "track_bias",
    "vertical_rate_bias",
    "freeze",
    "coherent_route_drift",
)

SEVERITIES = {
    "easy": {
        "position_jump_m": 5_000.0,
        "position_drift_m": 8_000.0,
        "altitude_bias_m": 1_000.0,
        "speed_bias_mps": 80.0,
        "track_bias_deg": 90.0,
        "vertical_rate_bias_mps": 20.0,
        "coherent_drift_m": 12_000.0,
    },
    "medium": {
        "position_jump_m": 1_000.0,
        "position_drift_m": 2_000.0,
        "altitude_bias_m": 300.0,
        "speed_bias_mps": 30.0,
        "track_bias_deg": 30.0,
        "vertical_rate_bias_mps": 8.0,
        "coherent_drift_m": 3_000.0,
    },
}


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def assigned_injection_type(flight_id: str) -> str:
    return INJECTION_TYPES[_stable_int(flight_id) % len(INJECTION_TYPES)]


def _event_bounds(length: int, seed_value: int) -> tuple[int, int]:
    if length < 20:
        raise ValueError("Injection requires at least 20 rows")
    rng = np.random.default_rng(seed_value)
    event_length = max(8, min(length // 4, 40))
    low = max(3, length // 4)
    high = max(low + 1, length - event_length - 3)
    start = int(rng.integers(low, high))
    return start, min(length - 2, start + event_length)


def inject_flight(
    flight: pd.DataFrame,
    *,
    injection_type: str,
    severity: str,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Return an injected copy; never mutate the source flight."""

    if injection_type not in INJECTION_TYPES:
        raise ValueError(f"Unknown injection type: {injection_type}")
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")
    if flight["flight_id"].nunique() != 1:
        raise ValueError("inject_flight expects exactly one flight")

    original_id = str(flight["flight_id"].iloc[0])
    out = flight.sort_values("timestamp_utc").copy(deep=True).reset_index(drop=True)
    start, end = _event_bounds(len(out), seed + _stable_int(f"{original_id}:{severity}"))
    event = np.arange(start, end + 1)
    fraction = np.linspace(0.0, 1.0, len(event))
    params = SEVERITIES[severity]

    latitude = np.clip(out.loc[event, "lat"].to_numpy(dtype=float), -89.0, 89.0)
    metres_per_lon_degree = np.maximum(111_320.0 * np.cos(np.radians(latitude)), 1_000.0)

    if injection_type == "position_jump":
        out.loc[event, "lat"] += params["position_jump_m"] / 111_320.0
    elif injection_type == "position_drift":
        out.loc[event, "lat"] += fraction * params["position_drift_m"] / 111_320.0
        out.loc[event, "lon"] += fraction * params["position_drift_m"] / metres_per_lon_degree
    elif injection_type == "altitude_bias":
        out.loc[event, "alt"] += params["altitude_bias_m"]
    elif injection_type == "speed_bias":
        out.loc[event, "ground_speed_ms"] += params["speed_bias_mps"]
    elif injection_type == "track_bias":
        out.loc[event, "track_deg"] = (
            out.loc[event, "track_deg"] + params["track_bias_deg"]
        ) % 360.0
    elif injection_type == "vertical_rate_bias":
        out.loc[event, "vertical_rate_ms"] += params["vertical_rate_bias_mps"]
    elif injection_type == "freeze":
        out.loc[event, "lat"] = float(out.loc[start, "lat"])
        out.loc[event, "lon"] = float(out.loc[start, "lon"])
        out.loc[event, "alt"] = float(out.loc[start, "alt"])
    elif injection_type == "coherent_route_drift":
        north = fraction * params["coherent_drift_m"]
        east = fraction * params["coherent_drift_m"] * 0.6
        out.loc[event, "lat"] += north / 111_320.0
        out.loc[event, "lon"] += east / metres_per_lon_degree
        # Keep reported speed and track approximately coherent with the altered path.
        out.loc[event, "track_deg"] = (out.loc[event, "track_deg"] + 15.0 * fraction) % 360.0

    injected_id = f"{original_id}:inj:{injection_type}:{severity}"
    out["source_flight_id"] = original_id
    out["flight_id"] = injected_id
    out["is_injected_anomaly"] = False
    out.loc[event, "is_injected_anomaly"] = True
    out["injection_type"] = injection_type
    out["severity"] = severity
    out["event_start_utc"] = float(out.loc[start, "timestamp_utc"])
    out["event_end_utc"] = float(out.loc[end, "timestamp_utc"])
    return out


def build_injected_copies(
    test_rows: pd.DataFrame,
    *,
    severities: tuple[str, ...] = ("easy", "medium"),
    seed: int = 20260710,
) -> pd.DataFrame:
    copies: list[pd.DataFrame] = []
    for flight_id, flight in test_rows.groupby("flight_id", sort=True):
        if len(flight) < 20:
            continue
        injection_type = assigned_injection_type(str(flight_id))
        for severity in severities:
            copies.append(
                inject_flight(
                    flight,
                    injection_type=injection_type,
                    severity=severity,
                    seed=seed,
                )
            )
    return pd.concat(copies, ignore_index=True) if copies else pd.DataFrame()
