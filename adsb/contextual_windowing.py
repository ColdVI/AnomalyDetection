"""Leakage-safe next-row windows for the contextual residual forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adsb.context import CausalContextConfig, build_causal_context


PHASES = ("ground", "climb", "level", "descent", "unknown")


@dataclass(frozen=True)
class ContextualForecastBatch:
    X: np.ndarray
    X_mask: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    meta: pd.DataFrame
    input_features: tuple[str, ...]
    target_channels: tuple[str, ...]


def _context_numeric_matrix(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    signal_columns: tuple[str, ...],
    config: CausalContextConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    missing = set(signal_columns).difference(frame.columns)
    if missing:
        raise KeyError(f"Missing contextual signal columns: {sorted(missing)}")

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    names: list[str] = []
    for column in signal_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(numeric)
        values.append(np.where(valid, numeric, 0.0))
        masks.append(valid.astype(float))
        names.append(column)

    for column in ("context_log1p_dt_s", "track_sin", "track_cos"):
        numeric = context[column].to_numpy(float)
        valid = np.isfinite(numeric)
        values.append(np.where(valid, numeric, 0.0))
        masks.append(valid.astype(float))
        names.append(column)

    cadence_categories = tuple(
        [f"cadence_{number}" for number in range(len(config.cadence_edges_s) + 1)]
        + ["gap", "initial_or_invalid"]
    )
    for prefix, categories, source in (
        ("phase", PHASES, context["context_phase"]),
        ("cadence", cadence_categories, context["context_cadence"]),
    ):
        for category in categories:
            values.append(source.eq(category).to_numpy(dtype=float))
            masks.append(np.ones(len(frame), dtype=float))
            names.append(f"{prefix}={category}")

    return np.column_stack(values), np.column_stack(masks), tuple(names)


def build_contextual_forecast_windows(
    frame: pd.DataFrame,
    *,
    signal_columns: tuple[str, ...],
    target_channels: tuple[str, ...],
    history_rows: int,
    context_config: CausalContextConfig,
) -> ContextualForecastBatch:
    """Predict row ``t`` from rows ``[t-history_rows, t)`` within one flight."""

    if history_rows < 1:
        raise ValueError("history_rows must be >= 1")
    missing_targets = set(target_channels).difference(frame.columns)
    if missing_targets:
        raise KeyError(f"Missing target channels: {sorted(missing_targets)}")
    context = build_causal_context(frame, context_config)
    matrix, matrix_mask, input_names = _context_numeric_matrix(
        frame, context, signal_columns=signal_columns, config=context_config
    )
    target = frame.loc[:, target_channels].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target_mask = np.isfinite(target)

    windows: list[np.ndarray] = []
    window_masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for flight_id, group in frame.groupby(context_config.flight_id_col, sort=False):
        pos = positions.loc[group.index].to_numpy(dtype=int)
        times = pd.to_numeric(group[context_config.time_col], errors="coerce").to_numpy(float)
        for local_target in range(history_rows, len(pos)):
            local_start = local_target - history_rows
            interval = times[local_start : local_target + 1]
            if np.any(np.diff(interval) <= 0) or np.any(np.diff(interval) > context_config.max_gap_s):
                continue
            history_position = pos[local_start:local_target]
            target_position = pos[local_target]
            windows.append(matrix[history_position])
            window_masks.append(matrix_mask[history_position])
            targets.append(np.where(target_mask[target_position], target[target_position], 0.0))
            target_masks.append(target_mask[target_position].astype(float))
            metadata.append(
                {
                    "flight_id": flight_id,
                    "target_timestamp_utc": float(times[local_target]),
                    "context_phase": context.iloc[target_position]["context_phase"],
                    "context_cadence": context.iloc[target_position]["context_cadence"],
                }
            )

    feature_count = len(input_names)
    target_count = len(target_channels)
    if not windows:
        return ContextualForecastBatch(
            X=np.zeros((0, history_rows, feature_count), dtype=np.float32),
            X_mask=np.zeros((0, history_rows, feature_count), dtype=np.float32),
            y=np.zeros((0, target_count), dtype=np.float32),
            y_mask=np.zeros((0, target_count), dtype=np.float32),
            meta=pd.DataFrame(
                columns=["flight_id", "target_timestamp_utc", "context_phase", "context_cadence"]
            ),
            input_features=input_names,
            target_channels=target_channels,
        )
    return ContextualForecastBatch(
        X=np.asarray(windows, dtype=np.float32),
        X_mask=np.asarray(window_masks, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        y_mask=np.asarray(target_masks, dtype=np.float32),
        meta=pd.DataFrame.from_records(metadata),
        input_features=input_names,
        target_channels=target_channels,
    )
