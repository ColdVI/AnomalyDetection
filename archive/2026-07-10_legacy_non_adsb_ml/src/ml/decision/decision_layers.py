"""Frozen causal decision layers for ML-8A score streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from src.ml.evaluation.events import k_of_n_alarm


def _as_streams(streams: Iterable[np.ndarray]) -> list[np.ndarray]:
    result = [np.asarray(stream, dtype=float) for stream in streams]
    if not result or any(stream.ndim != 1 for stream in result):
        raise ValueError("At least one one-dimensional score stream is required")
    return result


def _onsets(mask) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~np.r_[False, mask[:-1]]


def _fa_per_hour(onsets: list[np.ndarray], streams: list[np.ndarray], stride_seconds: float) -> float:
    hours = sum(len(stream) * stride_seconds for stream in streams) / 3600.0
    return float(sum(int(value.sum()) for value in onsets) / hours) if hours > 0 else np.inf


def _candidate_thresholds(streams: list[np.ndarray]) -> np.ndarray:
    finite = np.concatenate([stream[np.isfinite(stream)] for stream in streams])
    if not len(finite):
        raise ValueError("Calibration streams contain no finite scores")
    unique = np.unique(finite)
    if len(unique) > 4096:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 4096)))
    return np.r_[unique, np.inf]


@dataclass(frozen=True)
class ThresholdPolicy:
    threshold: float
    fa_budget_per_hour: float
    calibration_fa_per_hour: float

    def apply(self, score_stream) -> np.ndarray:
        scores = np.asarray(score_stream, dtype=float)
        return _onsets(np.isfinite(scores) & (scores > self.threshold))

    def to_dict(self) -> dict:
        return {"kind": "threshold", **asdict(self)}


def fit_threshold_policy(
    val_normal_score_streams: Iterable[np.ndarray],
    fa_budget_per_hour: float,
    *,
    stride_seconds: float = 1.0,
) -> ThresholdPolicy:
    streams = _as_streams(val_normal_score_streams)
    for threshold in _candidate_thresholds(streams):
        fa = _fa_per_hour(
            [_onsets(np.isfinite(stream) & (stream > threshold)) for stream in streams],
            streams,
            stride_seconds,
        )
        if fa <= fa_budget_per_hour:
            return ThresholdPolicy(float(threshold), float(fa_budget_per_hour), fa)
    raise RuntimeError("No threshold satisfies the FA budget")


@dataclass(frozen=True)
class KOfNPolicy:
    k: int
    n: int
    threshold: float
    fa_budget_per_hour: float
    calibration_fa_per_hour: float

    def apply(self, score_stream) -> np.ndarray:
        return _onsets(k_of_n_alarm(score_stream, self.threshold, k=self.k, n=self.n))

    def to_dict(self) -> dict:
        return {"kind": "k_of_n", **asdict(self)}


def fit_k_of_n_policy(
    val_normal_score_streams: Iterable[np.ndarray],
    fa_budget_per_hour: float,
    *,
    candidates: tuple[tuple[int, int], ...] = ((2, 3), (3, 5)),
    stride_seconds: float = 1.0,
) -> KOfNPolicy:
    streams = _as_streams(val_normal_score_streams)
    feasible: list[KOfNPolicy] = []
    for k, n in candidates:
        for threshold in _candidate_thresholds(streams):
            fa = _fa_per_hour(
                [_onsets(k_of_n_alarm(stream, threshold, k=k, n=n)) for stream in streams],
                streams,
                stride_seconds,
            )
            if fa <= fa_budget_per_hour:
                feasible.append(KOfNPolicy(k, n, float(threshold), float(fa_budget_per_hour), fa))
                break
    if not feasible:
        raise RuntimeError("No K-of-N policy satisfies the FA budget")
    return min(feasible, key=lambda policy: (policy.threshold, policy.n, policy.k))


def _standardized_logit(scores: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return (logits - mu) / sigma


def cusum_alarm_onsets(
    score_stream,
    *,
    mu_normal: float,
    sigma_normal: float,
    k: float,
    h: float,
    refractory_steps: int = 30,
) -> np.ndarray:
    """Apply one-sided causal CUSUM, resetting after every alarm."""

    standardized = _standardized_logit(score_stream, mu_normal, sigma_normal)
    onsets = np.zeros(len(standardized), dtype=bool)
    state = 0.0
    refractory = 0
    for index, value in enumerate(standardized):
        if refractory > 0:
            refractory -= 1
            state = 0.0
            continue
        state = max(0.0, state + (float(value) if np.isfinite(value) else 0.0) - k)
        if state > h:
            onsets[index] = True
            state = 0.0
            refractory = refractory_steps
    return onsets


def _moving_block_bootstrap(
    streams: list[np.ndarray],
    *,
    output_samples: int,
    block_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = [stream for stream in streams if len(stream) >= block_samples]
    circular = False
    if not eligible:
        # Kısa ALFA validation uçuşlarında 60 s'lik doğrusal blok yoktur.
        # Uçuşları birbirine yapıştırmak yerine aynı uçuş içinde circular MBB
        # uygula; blok uzunluğu ve uçuş izolasyonu korunur.
        eligible = [stream for stream in streams if len(stream)]
        circular = True
    if not eligible:
        raise ValueError("Validation streams are empty")
    blocks: list[np.ndarray] = []
    generated = 0
    while generated < output_samples:
        stream = eligible[int(rng.integers(len(eligible)))]
        if circular:
            start = int(rng.integers(0, len(stream)))
            block = stream[(start + np.arange(block_samples)) % len(stream)]
        else:
            start = int(rng.integers(0, len(stream) - block_samples + 1))
            block = stream[start:start + block_samples]
        blocks.append(block)
        generated += len(block)
    return np.concatenate(blocks)[:output_samples]


@dataclass(frozen=True)
class CusumPolicy:
    mu_normal: float
    sigma_normal: float
    k: float
    h: float
    refractory_seconds: float
    stride_seconds: float
    bootstrap_hours: float
    fa_budget_per_hour: float
    calibration_fa_per_hour: float

    def apply(self, score_stream) -> np.ndarray:
        steps = max(0, round(self.refractory_seconds / self.stride_seconds))
        return cusum_alarm_onsets(
            score_stream,
            mu_normal=self.mu_normal,
            sigma_normal=self.sigma_normal,
            k=self.k,
            h=self.h,
            refractory_steps=steps,
        )

    def to_dict(self) -> dict:
        return {"kind": "cusum", **asdict(self)}


def policy_from_dict(value: dict):
    """Rehydrate a checksum-stored policy for resumable matrix runs."""

    data = dict(value)
    kind = data.pop("kind")
    if kind == "threshold":
        return ThresholdPolicy(**data)
    if kind == "k_of_n":
        return KOfNPolicy(**data)
    if kind == "cusum":
        return CusumPolicy(**data)
    raise ValueError(f"Unknown decision policy kind: {kind!r}")


def fit_cusum_policy(
    val_normal_score_streams: Iterable[np.ndarray],
    fa_budget_per_hour: float,
    *,
    stride_seconds: float = 1.0,
    block_seconds: float = 60.0,
    bootstrap_hours: float = 200.0,
    k: float = 0.5,
    refractory_seconds: float = 30.0,
    seed: int = 0,
) -> CusumPolicy:
    streams = _as_streams(val_normal_score_streams)
    logits = [
        np.log(np.clip(stream, 1e-6, 1 - 1e-6) / (1 - np.clip(stream, 1e-6, 1 - 1e-6)))
        for stream in streams
    ]
    pooled = np.concatenate([value[np.isfinite(value)] for value in logits])
    mu = float(np.mean(pooled))
    sigma = float(np.std(pooled, ddof=0))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    standardized = [(value - mu) / sigma for value in logits]
    synthetic = _moving_block_bootstrap(
        standardized,
        output_samples=max(1, round(bootstrap_hours * 3600.0 / stride_seconds)),
        block_samples=max(1, round(block_seconds / stride_seconds)),
        seed=seed,
    )
    synthetic_scores = 1.0 / (1.0 + np.exp(-(synthetic * sigma + mu)))
    refractory_steps = max(0, round(refractory_seconds / stride_seconds))

    def fa_at(h: float) -> float:
        count = int(cusum_alarm_onsets(
            synthetic_scores,
            mu_normal=mu,
            sigma_normal=sigma,
            k=k,
            h=h,
            refractory_steps=refractory_steps,
        ).sum())
        return float(count / bootstrap_hours)

    low, high = 0.0, 1.0
    while fa_at(high) > fa_budget_per_hour and high < 1e6:
        high *= 2.0
    for _ in range(12):
        middle = (low + high) / 2.0
        if fa_at(middle) <= fa_budget_per_hour:
            high = middle
        else:
            low = middle
    fa = fa_at(high)
    return CusumPolicy(
        mu_normal=mu,
        sigma_normal=sigma,
        k=float(k),
        h=float(high),
        refractory_seconds=float(refractory_seconds),
        stride_seconds=float(stride_seconds),
        bootstrap_hours=float(bootstrap_hours),
        fa_budget_per_hour=float(fa_budget_per_hour),
        calibration_fa_per_hour=fa,
    )
