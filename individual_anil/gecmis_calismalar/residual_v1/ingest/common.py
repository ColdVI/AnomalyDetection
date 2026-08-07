"""Shared, model-free ingestion primitives."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def wrap_radians(values: pd.Series) -> pd.Series:
    """Wrap radians to (-pi, pi], preserving missing values."""

    numeric = pd.to_numeric(values, errors="coerce")
    wrapped = (numeric + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped.mask(np.isclose(wrapped, -math.pi), math.pi)


def drop_non_monotonic_timestamps(
    frame: pd.DataFrame, timestamp_col: str
) -> tuple[pd.DataFrame, int]:
    """Drop every row whose timestamp does not advance in source order."""

    if timestamp_col not in frame:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    result = frame.copy()
    result[timestamp_col] = pd.to_numeric(result[timestamp_col], errors="coerce")
    finite = np.isfinite(result[timestamp_col].to_numpy(dtype=float))
    result = result.loc[finite].copy()
    timestamp = result[timestamp_col].to_numpy(dtype=float)
    if len(timestamp) == 0:
        return result, int(len(frame))
    advancing = np.r_[True, np.diff(timestamp) > 0]
    dropped = int(len(frame) - int(advancing.sum()))
    return result.loc[advancing].reset_index(drop=True), dropped


def fix_quaternion_sign_continuity(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Choose the q/-q representation closest to the previous sample."""

    columns = tuple(columns)
    if not columns or any(column not in frame for column in columns):
        return frame
    result = frame.copy()
    values = result.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float).copy()
    previous: np.ndarray | None = None
    for index in range(len(values)):
        current = values[index]
        if not np.isfinite(current).all():
            continue
        if previous is not None and float(np.dot(previous, current)) < 0.0:
            current = -current
            values[index] = current
        previous = current
    result.loc[:, columns] = values
    return result


def write_json(path: str | Path, payload: object, *, fail_if_exists: bool = False) -> None:
    target = Path(path)
    if fail_if_exists and target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
