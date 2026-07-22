"""Small, explicit physics transformations used by residual channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_GRAVITY_M_S2 = 9.80665


def finite_difference(
    time: pd.Series,
    values: pd.Series,
    *,
    window_s: float = 0.5,
) -> pd.Series:
    """Observed-sample 0.5 s difference with one-sided flight edges."""

    if window_s <= 0:
        raise ValueError("window_s must be positive")
    t = pd.to_numeric(time, errors="coerce").to_numpy(float)
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if len(t) != len(x) or len(t) == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    result = np.full(len(t), np.nan, dtype=float)
    half = window_s / 2.0
    for index, current in enumerate(t):
        left = int(np.searchsorted(t, current - half, side="left"))
        right = int(np.searchsorted(t, current + half, side="right") - 1)
        if left == right:
            if index == 0 and len(t) > 1:
                right = 1
            elif index == len(t) - 1 and len(t) > 1:
                left = len(t) - 2
            elif 0 < index < len(t) - 1:
                left, right = index - 1, index + 1
        dt = t[right] - t[left]
        if dt > 0 and np.isfinite(x[left]) and np.isfinite(x[right]):
            result[index] = (x[right] - x[left]) / dt
    return pd.Series(result, index=values.index, name=f"{values.name}_derivative")


def coordinated_turn_yaw_rate(
    roll_rad: pd.Series,
    airspeed_m_s: pd.Series,
    *,
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
) -> pd.Series:
    roll = pd.to_numeric(roll_rad, errors="coerce")
    airspeed = pd.to_numeric(airspeed_m_s, errors="coerce")
    result = gravity_m_s2 * np.tan(roll) / airspeed
    return result.mask(airspeed <= 0).rename("coordinated_turn_term")


def vector_norm(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> pd.Series:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return np.sqrt((values**2).sum(axis=1, min_count=len(columns))).rename(name)


def motor_pwm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = tuple(f"motor_pwm_{index}" for index in range(4))
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    output = pd.DataFrame(index=frame.index)
    output["motor_pwm_total"] = values.sum(axis=1, min_count=len(columns))
    output["motor_pwm_asymmetry"] = values.std(axis=1, ddof=0).where(values.notna().all(axis=1))
    return output

