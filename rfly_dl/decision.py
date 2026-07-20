"""Causal alarm policies calibrated only on normal validation flights."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rfly_dl.config import (
    CUSUM_BLOCK_SECONDS,
    CUSUM_BOOTSTRAP_HOURS,
    CUSUM_K,
    CUSUM_REFRACTORY_SECONDS,
    DECISION_STRIDE_S,
)


def alarm_onsets(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=bool)
    return state & ~np.r_[False, state[:-1]]


def k_of_n_state(
    scores: np.ndarray, threshold: float, *, k: int, n: int
) -> np.ndarray:
    finite = np.isfinite(scores) & (np.asarray(scores, dtype=float) > threshold)
    counts = np.convolve(finite.astype(np.int16), np.ones(n, dtype=np.int16), mode="full")
    return counts[: len(finite)] >= k


def _false_alarms_per_hour(
    onsets: list[np.ndarray], streams: list[np.ndarray], stride_seconds: float
) -> float:
    hours = sum(len(stream) * stride_seconds for stream in streams) / 3600.0
    return (
        float(sum(int(value.sum()) for value in onsets) / hours)
        if hours > 0
        else float("inf")
    )


def _candidate_thresholds(streams: list[np.ndarray]) -> np.ndarray:
    finite_parts = [stream[np.isfinite(stream)] for stream in streams]
    finite = np.concatenate(finite_parts) if finite_parts else np.empty(0)
    if not len(finite):
        raise ValueError("Validation streams contain no finite score")
    unique = np.unique(finite)
    if len(unique) > 4096:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 4096)))
    return np.r_[unique, np.inf]


@dataclass(frozen=True)
class ThresholdPolicy:
    threshold: float
    fa_budget_per_hour: float
    calibration_fa_per_hour: float

    def apply(self, scores: np.ndarray) -> np.ndarray:
        state = np.isfinite(scores) & (np.asarray(scores, dtype=float) > self.threshold)
        return alarm_onsets(state)

    def to_dict(self) -> dict:
        return {"kind": "threshold", **asdict(self)}


@dataclass(frozen=True)
class KOfNPolicy:
    k: int
    n: int
    threshold: float
    fa_budget_per_hour: float
    calibration_fa_per_hour: float

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return alarm_onsets(k_of_n_state(scores, self.threshold, k=self.k, n=self.n))

    def to_dict(self) -> dict:
        return {"kind": "k_of_n", **asdict(self)}


def _standardized_logit(scores: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return (logits - mu) / sigma


def cusum_alarm_onsets(
    scores: np.ndarray,
    *,
    mu_normal: float,
    sigma_normal: float,
    k: float,
    h: float,
    refractory_steps: int,
) -> np.ndarray:
    standardized = _standardized_logit(scores, mu_normal, sigma_normal)
    onsets = np.zeros(len(standardized), dtype=bool)
    state = 0.0
    refractory = 0
    for index, value in enumerate(standardized):
        if refractory:
            refractory -= 1
            state = 0.0
            continue
        state = max(0.0, state + (float(value) if np.isfinite(value) else 0.0) - k)
        if state > h:
            onsets[index] = True
            state = 0.0
            refractory = refractory_steps
    return onsets


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

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return cusum_alarm_onsets(
            scores,
            mu_normal=self.mu_normal,
            sigma_normal=self.sigma_normal,
            k=self.k,
            h=self.h,
            refractory_steps=max(
                0, round(self.refractory_seconds / self.stride_seconds)
            ),
        )

    def to_dict(self) -> dict:
        return {"kind": "cusum", **asdict(self)}


def fit_threshold_policy(
    streams: list[np.ndarray], fa_budget_per_hour: float
) -> ThresholdPolicy:
    for threshold in _candidate_thresholds(streams):
        fa = _false_alarms_per_hour(
            [
                alarm_onsets(
                    np.isfinite(stream)
                    & (np.asarray(stream, dtype=float) > float(threshold))
                )
                for stream in streams
            ],
            streams,
            DECISION_STRIDE_S,
        )
        if fa <= fa_budget_per_hour:
            return ThresholdPolicy(float(threshold), float(fa_budget_per_hour), fa)
    raise RuntimeError("No threshold policy meets validation FA budget")


def fit_k_of_n_policy(
    streams: list[np.ndarray], fa_budget_per_hour: float
) -> KOfNPolicy:
    feasible: list[KOfNPolicy] = []
    for k, n in ((2, 3), (3, 5)):
        for threshold in _candidate_thresholds(streams):
            fa = _false_alarms_per_hour(
                [
                    alarm_onsets(
                        k_of_n_state(stream, float(threshold), k=k, n=n)
                    )
                    for stream in streams
                ],
                streams,
                DECISION_STRIDE_S,
            )
            if fa <= fa_budget_per_hour:
                feasible.append(
                    KOfNPolicy(
                        k, n, float(threshold), float(fa_budget_per_hour), fa
                    )
                )
                break
    if not feasible:
        raise RuntimeError("No K-of-N policy meets validation FA budget")
    return min(feasible, key=lambda policy: (policy.threshold, policy.n, policy.k))


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
            block = stream[start : start + block_samples]
        blocks.append(block)
        generated += len(block)
    return np.concatenate(blocks)[:output_samples]


def fit_cusum_policy(
    streams: list[np.ndarray], fa_budget_per_hour: float, *, seed: int
) -> CusumPolicy:
    logits = []
    for stream in streams:
        clipped = np.clip(np.asarray(stream, dtype=float), 1e-6, 1.0 - 1e-6)
        logits.append(np.log(clipped / (1.0 - clipped)))
    pooled_parts = [value[np.isfinite(value)] for value in logits]
    pooled = np.concatenate(pooled_parts)
    mu = float(pooled.mean())
    sigma = float(pooled.std(ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    standardized = [(value - mu) / sigma for value in logits]
    synthetic_z = _moving_block_bootstrap(
        standardized,
        output_samples=max(
            1, round(CUSUM_BOOTSTRAP_HOURS * 3600.0 / DECISION_STRIDE_S)
        ),
        block_samples=max(1, round(CUSUM_BLOCK_SECONDS / DECISION_STRIDE_S)),
        seed=seed,
    )
    synthetic_scores = 1.0 / (
        1.0 + np.exp(-np.clip(synthetic_z * sigma + mu, -40.0, 40.0))
    )
    refractory_steps = max(
        0, round(CUSUM_REFRACTORY_SECONDS / DECISION_STRIDE_S)
    )

    def fa_at(h: float) -> float:
        count = int(
            cusum_alarm_onsets(
                synthetic_scores,
                mu_normal=mu,
                sigma_normal=sigma,
                k=CUSUM_K,
                h=h,
                refractory_steps=refractory_steps,
            ).sum()
        )
        return count / CUSUM_BOOTSTRAP_HOURS

    low, high = 0.0, 1.0
    while fa_at(high) > fa_budget_per_hour and high < 1e6:
        high *= 2.0
    for _ in range(12):
        middle = (low + high) / 2.0
        if fa_at(middle) <= fa_budget_per_hour:
            high = middle
        else:
            low = middle
    return CusumPolicy(
        mu_normal=mu,
        sigma_normal=sigma,
        k=CUSUM_K,
        h=float(high),
        refractory_seconds=CUSUM_REFRACTORY_SECONDS,
        stride_seconds=DECISION_STRIDE_S,
        bootstrap_hours=CUSUM_BOOTSTRAP_HOURS,
        fa_budget_per_hour=float(fa_budget_per_hour),
        calibration_fa_per_hour=float(fa_at(high)),
    )


def fit_policies(
    streams: list[np.ndarray], fa_budget_per_hour: float, *, seed: int
) -> dict[str, ThresholdPolicy | KOfNPolicy | CusumPolicy]:
    return {
        "threshold": fit_threshold_policy(streams, fa_budget_per_hour),
        "k_of_n": fit_k_of_n_policy(streams, fa_budget_per_hour),
        "cusum": fit_cusum_policy(streams, fa_budget_per_hour, seed=seed),
    }
