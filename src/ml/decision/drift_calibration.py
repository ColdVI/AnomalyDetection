"""Drift-aware false-alarm calibration wrapper for existing decision layers."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np


DEFAULT_QUANTILE = 0.75
DEFAULT_FLOOR = 1.0
DEFAULT_CAP = 5.0
DEFAULT_MIN_SESSIONS = 4


def _as_session_streams(
    val_streams_by_session: Mapping[str, Iterable[np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    sessions: dict[str, list[np.ndarray]] = {}
    for session, streams in val_streams_by_session.items():
        values = [np.asarray(stream, dtype=float) for stream in streams]
        if not values:
            raise ValueError(f"Validation session has no streams: {session!r}")
        if any(stream.ndim != 1 for stream in values):
            raise ValueError(f"Validation session has a non-1D stream: {session!r}")
        sessions[str(session)] = values
    if not sessions:
        raise ValueError("At least one validation session is required")
    return sessions


def _flatten(sessions: Mapping[str, list[np.ndarray]]) -> list[np.ndarray]:
    return [stream for session in sorted(sessions) for stream in sessions[session]]


def _fit_policy(
    decision_fit_fn: Callable[..., Any],
    streams: list[np.ndarray],
    budget: float,
    *,
    seed: int,
    stride_seconds: float,
    decision_kwargs: Mapping[str, Any] | None,
):
    kwargs: dict[str, Any] = dict(decision_kwargs or {})
    signature = inspect.signature(decision_fit_fn)
    if "stride_seconds" in signature.parameters:
        kwargs.setdefault("stride_seconds", stride_seconds)
    if "seed" in signature.parameters:
        kwargs.setdefault("seed", seed)
    return decision_fit_fn(streams, budget, **kwargs)


def _fa_per_hour(policy, streams: list[np.ndarray], stride_seconds: float) -> float:
    seconds = sum(len(stream) * stride_seconds for stream in streams)
    if seconds <= 0:
        return np.inf
    alarms = sum(int(np.asarray(policy.apply(stream), dtype=bool).sum()) for stream in streams)
    return float(alarms / (seconds / 3600.0))


def _clamp_multiplier(value: float, *, floor: float, cap: float) -> float:
    if not np.isfinite(value):
        value = cap
    return float(min(cap, max(floor, value)))


def fit_drift_corrected_policy(
    val_streams_by_session: Mapping[str, Iterable[np.ndarray]],
    budget: float,
    decision_fit_fn: Callable[..., Any],
    seed: int,
    *,
    stride_seconds: float = 1.0,
    quantile: float = DEFAULT_QUANTILE,
    floor: float = DEFAULT_FLOOR,
    cap: float = DEFAULT_CAP,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    fallback_drift_multiplier: float | None = None,
    decision_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, dict]:
    """Fit an existing decision policy with a jackknife FA-drift correction.

    The returned policy is the object produced by ``decision_fit_fn``; this
    module only chooses the effective FA budget.  If validation has too few
    sessions, callers must provide the preregistered ML-14 fallback multiplier.
    """

    if budget <= 0:
        raise ValueError("FA budget must be positive")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if floor <= 0 or cap < floor:
        raise ValueError("Require 0 < floor <= cap")
    if min_sessions < 1:
        raise ValueError("min_sessions must be positive")

    sessions = _as_session_streams(val_streams_by_session)
    session_count = len(sessions)
    session_ratios: list[dict[str, float | str | int]] = []
    fallback_used = session_count < min_sessions

    if fallback_used:
        if fallback_drift_multiplier is None:
            raise ValueError(
                "fallback_drift_multiplier is required when validation has "
                f"fewer than {min_sessions} sessions"
            )
        drift_multiplier = _clamp_multiplier(
            float(fallback_drift_multiplier), floor=floor, cap=cap,
        )
        fallback_source = "provided_ml14_median_shift"
    else:
        ratios: list[float] = []
        for held_session in sorted(sessions):
            fit_sessions = {
                name: streams for name, streams in sessions.items()
                if name != held_session
            }
            fit_streams = _flatten(fit_sessions)
            held_streams = sessions[held_session]
            policy = _fit_policy(
                decision_fit_fn,
                fit_streams,
                budget,
                seed=seed,
                stride_seconds=stride_seconds,
                decision_kwargs=decision_kwargs,
            )
            fa = _fa_per_hour(policy, held_streams, stride_seconds)
            ratio = float(fa / budget)
            ratios.append(ratio)
            session_ratios.append({
                "session": held_session,
                "heldout_streams": len(held_streams),
                "heldout_hours": float(
                    sum(len(stream) * stride_seconds for stream in held_streams) / 3600.0
                ),
                "false_alarms_per_hour": fa,
                "ratio": ratio,
            })
        drift_multiplier = _clamp_multiplier(
            float(np.quantile(np.asarray(ratios, dtype=float), quantile)),
            floor=floor,
            cap=cap,
        )
        fallback_source = None

    effective_budget = float(budget / drift_multiplier)
    policy = _fit_policy(
        decision_fit_fn,
        _flatten(sessions),
        effective_budget,
        seed=seed,
        stride_seconds=stride_seconds,
        decision_kwargs=decision_kwargs,
    )
    report = {
        "method": "session_jackknife_drift_corrected_policy",
        "nominal_budget_per_hour": float(budget),
        "effective_budget_per_hour": effective_budget,
        "drift_multiplier": drift_multiplier,
        "quantile": float(quantile),
        "floor": float(floor),
        "cap": float(cap),
        "min_sessions": int(min_sessions),
        "session_count": int(session_count),
        "fallback_used": bool(fallback_used),
        "fallback_source": fallback_source,
        "session_ratios": session_ratios,
    }
    return policy, report
