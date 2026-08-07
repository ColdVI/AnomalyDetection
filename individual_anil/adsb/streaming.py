"""Bounded-memory helpers for full-day ADS-B baseline evaluation.

No helper in this module discovers or opens the sealed raw holdout pool.  Input
files are always supplied explicitly by the calling run manifest.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class BoundedPrioritySampler:
    """Deterministic uniform finite-value sample with a hard memory capacity."""

    capacity: int
    values: np.ndarray | None = None
    priorities: np.ndarray | None = None
    finite_seen: int = 0
    _stream_offsets: dict[tuple[int, str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.values is None:
            self.values = np.empty(0, dtype=float)
        if self.priorities is None:
            self.priorities = np.empty(0, dtype=float)

    def add(
        self,
        values: np.ndarray,
        *,
        probability: float,
        seed: int,
        file_key: str,
        purpose: str,
    ) -> None:
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        self.finite_seen += int(len(array))
        if not len(array):
            return
        stream_key = (int(seed), str(file_key), str(purpose))
        offset = self._stream_offsets.get(stream_key, 0)
        priority = _stream_uniform(
            seed=stream_key[0],
            file_key=stream_key[1],
            purpose=stream_key[2],
            suffix="priority",
            offset=offset,
            size=len(array),
        )
        self._stream_offsets[stream_key] = offset + len(array)
        keep = priority < probability
        if not np.any(keep):
            return
        new_values = array[keep]
        new_priorities = priority[keep] / probability
        values_all = np.concatenate([self.values, new_values])
        priorities_all = np.concatenate([self.priorities, new_priorities])
        if len(values_all) > self.capacity:
            indices = np.argpartition(priorities_all, self.capacity - 1)[: self.capacity]
            values_all = values_all[indices]
            priorities_all = priorities_all[indices]
        order = np.argsort(priorities_all, kind="stable")
        self.values = values_all[order]
        self.priorities = priorities_all[order]


def _stream_uniform(
    *,
    seed: int,
    file_key: str,
    purpose: str,
    suffix: str,
    offset: int,
    size: int,
) -> np.ndarray:
    """Return a deterministic slice of a context-keyed PCG64 stream.

    Advancing by the number of prior rows makes repeated chunked calls exactly
    equivalent to one call over the concatenated rows. Different file/purpose
    streams remain independent, so processing-file order cannot change bottom-k
    membership.
    """

    if offset < 0 or size < 0:
        raise ValueError("offset and size must be non-negative")
    local_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}\0{file_key}\0{purpose}\0{suffix}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    bit_generator = np.random.PCG64(local_seed)
    if offset:
        bit_generator.advance(offset)
    return np.random.Generator(bit_generator).random(size)


@dataclass
class BoundedFramePrioritySampler:
    """Deterministic bottom-k sampler for rows of identically shaped frames.

    rows_seen counts all input rows while frame can never exceed capacity. The
    default seed is the pre-registered project sampling seed; callers should
    still record it in their run configuration.
    """

    capacity: int
    seed: int = 20260713
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows_seen: int = 0
    _stream_offsets: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _data_columns: tuple[object, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def add(self, frame: pd.DataFrame, *, file_key: str, purpose: str) -> None:
        if "_sample_priority" in frame.columns:
            raise ValueError("input frame uses reserved column '_sample_priority'")
        if frame.empty:
            return
        columns = tuple(frame.columns)
        if self._data_columns is None:
            self._data_columns = columns
        elif columns != self._data_columns:
            raise ValueError("all sampled frames must have identical ordered columns")

        stream_key = (str(file_key), str(purpose))
        offset = self._stream_offsets.get(stream_key, 0)
        incoming = frame.copy()
        incoming["_sample_priority"] = _stream_uniform(
            seed=int(self.seed),
            file_key=stream_key[0],
            purpose=stream_key[1],
            suffix="frame_priority",
            offset=offset,
            size=len(incoming),
        )
        self._stream_offsets[stream_key] = offset + len(incoming)
        self.rows_seen += int(len(incoming))

        combined = pd.concat([self.frame, incoming], ignore_index=True)
        if len(combined) > self.capacity:
            combined = combined.nsmallest(
                self.capacity, "_sample_priority", keep="first"
            )
        self.frame = combined.sort_values(
            "_sample_priority", kind="stable"
        ).reset_index(drop=True)


def prefixed_flight_id(day: str, flight_id: object) -> str:
    """Make flight identity unambiguous across days."""

    return f"{day}:{flight_id}"


def stable_fit_role(flight_id: object, *, seed: int = 20260713, fit_fraction: float = 0.8) -> str:
    """Order-independent SHA-256 split into normal fit vs normal calibration."""

    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    digest = hashlib.sha256(f"{seed}\0{flight_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest, "big") / float(1 << 256)
    return "fit" if u < fit_fraction else "calibration"


def deterministic_file_sample(
    values: np.ndarray,
    *,
    probability: float,
    seed: int,
    file_key: str,
    purpose: str,
) -> np.ndarray:
    """Uniform Bernoulli row sample reproducible per file and purpose.

    The caller must keep file ordering and ``file_key`` values in the manifest.
    Sampling decisions do not depend on the numeric values being sampled.
    """

    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    array = np.asarray(values)
    local_seed = int.from_bytes(
        hashlib.sha256(f"{seed}\0{file_key}\0{purpose}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(local_seed)
    keep = rng.random(len(array)) < probability
    return array[keep]


def dkw_quantile_error_bound(n: int, *, alpha: float = 0.05) -> float | None:
    """Distribution-free DKW CDF error bound for a uniform row sample."""

    if n <= 0:
        return None
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def robust_sample_calibration(samples: dict[str, Sequence[float]]) -> dict:
    """Median and 1.4826*MAD; exact zero-MAD channels are excluded, never floored."""

    included: dict[str, dict[str, float | int]] = {}
    excluded: list[str] = []
    for channel in sorted(samples):
        values = np.asarray(samples[channel], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            excluded.append(channel)
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        if mad == 0.0:
            excluded.append(channel)
            continue
        included[channel] = {
            "median": median,
            "mad": mad,
            "sample_n": int(len(values)),
            "dkw_cdf_error_95": dkw_quantile_error_bound(len(values)),
        }
    return {"calibration": included, "excluded_channels": excluded, "mad_zero_policy": "exclude"}


def count_alarm_episodes(times: np.ndarray, alarms: np.ndarray, *, merge_gap_s: float) -> int:
    """Count causal alarm emissions after flight-local debounce/merge."""

    t = np.asarray(times, dtype=float)
    a = np.asarray(alarms, dtype=bool)
    if t.shape != a.shape:
        raise ValueError("times and alarms must have identical shapes")
    active = np.sort(t[a & np.isfinite(t)])
    if not len(active):
        return 0
    return 1 + int(np.count_nonzero(np.diff(active) > merge_gap_s))


def scoreable_row_exposure_seconds(
    times: np.ndarray,
    evaluable: np.ndarray,
    *,
    max_gap_s: float,
) -> float:
    """Sum causal intervals ending in an evaluable row, excluding gaps/resets."""

    t = np.asarray(times, dtype=float)
    e = np.asarray(evaluable, dtype=bool)
    if t.shape != e.shape:
        raise ValueError("times and evaluable must have identical shapes")
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    valid = e[1:] & np.isfinite(dt) & (dt > 0.0) & (dt <= max_gap_s)
    return float(dt[valid].sum())


@dataclass(frozen=True)
class CusumBurdenCalibration:
    """Pre-registered natural-only CUSUM threshold calibration contract."""

    candidate_h: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
        750.0,
        1000.0,
    )
    advisory_budget_episodes_per_hour: float = 12.0
    bootstrap_repetitions: int = 500
    bootstrap_batch_size: int = 4
    bootstrap_seed: int = 20260713
    upper_quantile: float = 0.95
    merge_gap_s: float = 60.0
    moving_block_s: float = 300.0
    moving_block_stride_s: float = 150.0

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.candidate_h))) != self.candidate_h:
            raise ValueError("candidate_h must be unique and strictly increasing")
        if any(value <= 0 for value in self.candidate_h):
            raise ValueError("candidate_h values must be positive")
        if not math.isfinite(self.advisory_budget_episodes_per_hour) or (
            self.advisory_budget_episodes_per_hour < 0.0
        ):
            raise ValueError("advisory budget must be finite and non-negative")
        if self.bootstrap_repetitions <= 0:
            raise ValueError("bootstrap_repetitions must be positive")
        if self.bootstrap_batch_size <= 0:
            raise ValueError("bootstrap_batch_size must be positive")
        if not 0.0 < self.upper_quantile < 1.0:
            raise ValueError("upper_quantile must lie in (0, 1)")
        if self.merge_gap_s < 0.0:
            raise ValueError("merge_gap_s must be non-negative")
        if self.moving_block_s <= 0.0 or self.moving_block_stride_s <= 0.0:
            raise ValueError("moving block duration and stride must be positive")
        if self.moving_block_stride_s > self.moving_block_s:
            raise ValueError("moving block stride cannot exceed its duration")


def moving_block_burden_rows(
    flight_id: object,
    times: np.ndarray,
    joint_scores: np.ndarray,
    evaluable: np.ndarray,
    *,
    contract: CusumBurdenCalibration,
    max_gap_s: float,
) -> list[dict]:
    """Overlapping time-block summaries preserving within-block serial structure.

    Episode onsets are derived once from each complete flight.  A continuing
    alarm therefore does not become a new episode merely because another block
    begins.  Blocks are half-open except for the final block, which includes the
    flight's last timestamp; a final anchored block is appended when the regular
    stride would otherwise omit the tail.
    """

    t = np.asarray(times, dtype=float)
    score = np.asarray(joint_scores, dtype=float)
    ok = np.asarray(evaluable, dtype=bool)
    if not (t.shape == score.shape == ok.shape):
        raise ValueError("times, joint_scores and evaluable must have identical shapes")
    finite_t = t[np.isfinite(t)]
    if not len(finite_t):
        return []
    if max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be positive")
    first, last = float(finite_t.min()), float(finite_t.max())
    latest_start = max(first, last - contract.moving_block_s)
    starts = np.arange(
        first,
        latest_start + np.finfo(float).eps * max(1.0, abs(latest_start)),
        contract.moving_block_stride_s,
        dtype=float,
    )
    if not len(starts):
        starts = np.array([first], dtype=float)
    tail_tolerance = np.finfo(float).eps * max(1.0, abs(latest_start)) * 8.0
    if starts[-1] < latest_start - tail_tolerance:
        starts = np.append(starts, latest_start)
    else:
        starts[-1] = latest_start

    # Attribute each valid causal interval to its ending row.  Applying the
    # half-open block mask to endpoints avoids losing the interval that ends
    # exactly at a non-final block boundary.
    row_exposure_s = np.zeros(len(t), dtype=float)
    if len(t) > 1:
        dt = np.diff(t)
        valid_interval = (
            ok[1:]
            & np.isfinite(dt)
            & (dt > 0.0)
            & (dt <= max_gap_s)
        )
        row_exposure_s[1:] = np.where(valid_interval, dt, 0.0)

    onset_times_by_h: dict[float, np.ndarray] = {}
    for h in contract.candidate_h:
        alarm = ok & np.isfinite(score) & (score > h)
        active_times = np.sort(t[alarm & np.isfinite(t)])
        if len(active_times):
            is_onset = np.r_[True, np.diff(active_times) > contract.merge_gap_s]
            onset_times_by_h[h] = active_times[is_onset]
        else:
            onset_times_by_h[h] = np.empty(0, dtype=float)

    rows: list[dict] = []
    for block_index, start in enumerate(starts):
        end = start + contract.moving_block_s
        is_final = block_index == len(starts) - 1
        if is_final:
            select = (t >= start) & (t <= end)
        else:
            select = (t >= start) & (t < end)
        exposure_s = float(row_exposure_s[select].sum())
        row = {
            "flight_id": str(flight_id),
            "block_index": int(block_index),
            "block_start": float(start),
            "exposure_s": exposure_s,
        }
        for h in contract.candidate_h:
            onset_times = onset_times_by_h[h]
            if is_final:
                onset_in_block = (onset_times >= start) & (onset_times <= end)
            else:
                onset_in_block = (onset_times >= start) & (onset_times < end)
            row[f"h_{h:g}"] = int(np.count_nonzero(onset_in_block))
        rows.append(row)
    return rows


def select_cusum_threshold(
    block_rows: pd.DataFrame,
    *,
    contract: CusumBurdenCalibration,
    observed_exposure_s: float | None = None,
    observed_episodes_by_h: Mapping[float, int] | None = None,
) -> dict:
    """Choose the smallest h whose conservative natural-only upper meets budget.

    When full-flight counters are supplied, they alone define the observed
    point burden.  Moving blocks are then used only for bootstrap uncertainty.
    The reported conservative upper is the maximum of the full-flight observed
    rate and the raw moving-block bootstrap quantile.  This invariant prevents
    a bootstrap quantile from being described or used as an upper bound when it
    happens to fall below the observed point estimate.
    Both observed arguments must be supplied together.  The no-argument
    fallback is retained for older callers and is labelled explicitly.
    """

    if block_rows.empty:
        raise ValueError("block_rows cannot be empty")
    if "exposure_s" not in block_rows.columns:
        raise ValueError("block_rows lacks exposure_s")
    exposure = block_rows["exposure_s"].to_numpy(dtype=float)
    usable = np.isfinite(exposure) & (exposure > 0.0)
    if not np.any(usable):
        raise ValueError("no positive scoreable exposure in calibration blocks")
    frame = block_rows.loc[usable].reset_index(drop=True)
    exposure = frame["exposure_s"].to_numpy(dtype=float)
    h_columns = [f"h_{h:g}" for h in contract.candidate_h]
    missing_columns = [column for column in h_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"block_rows lacks candidate columns: {missing_columns}")
    episode_matrix = frame[h_columns].to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(episode_matrix))
        or np.any(episode_matrix < 0.0)
        or np.any(episode_matrix != np.floor(episode_matrix))
    ):
        raise ValueError(
            "block episode counts must be finite non-negative integers"
        )

    supplied_exposure = observed_exposure_s is not None
    supplied_episodes = observed_episodes_by_h is not None
    if supplied_exposure != supplied_episodes:
        raise ValueError(
            "observed_exposure_s and observed_episodes_by_h must be supplied together"
        )
    if supplied_exposure:
        assert observed_exposure_s is not None
        assert observed_episodes_by_h is not None
        if not math.isfinite(observed_exposure_s) or observed_exposure_s <= 0.0:
            raise ValueError("observed_exposure_s must be finite and positive")
        normalized_observed: dict[float, int] = {}
        for raw_h, raw_count in observed_episodes_by_h.items():
            h = float(raw_h)
            if isinstance(raw_count, (bool, np.bool_)) or int(raw_count) != raw_count:
                raise ValueError("full-flight observed episode counts must be integers")
            count = int(raw_count)
            if count < 0:
                raise ValueError("full-flight observed episode counts cannot be negative")
            normalized_observed[h] = count
        expected_h = set(contract.candidate_h)
        if set(normalized_observed) != expected_h:
            raise ValueError(
                "observed_episodes_by_h keys must exactly match candidate_h"
            )
        observed_hours = float(observed_exposure_s / 3600.0)
        observed_counts = normalized_observed
        observed_source = "full_flight_counters"
    else:
        observed_hours = float(exposure.sum() / 3600.0)
        observed_counts = {
            h: int(episode_matrix[:, column_index].sum())
            for column_index, h in enumerate(contract.candidate_h)
        }
        observed_source = "moving_blocks_legacy_fallback"

    rng = np.random.default_rng(contract.bootstrap_seed)
    repetitions = contract.bootstrap_repetitions
    boot_hours = np.empty(repetitions, dtype=float)
    boot_episode_sums = np.empty((repetitions, len(h_columns)), dtype=float)
    for start in range(0, repetitions, contract.bootstrap_batch_size):
        end = min(repetitions, start + contract.bootstrap_batch_size)
        draws = rng.integers(0, len(frame), size=(end - start, len(frame)))
        boot_hours[start:end] = exposure[draws].sum(axis=1) / 3600.0
        for column_index in range(len(h_columns)):
            boot_episode_sums[start:end, column_index] = (
                episode_matrix[:, column_index][draws].sum(axis=1)
            )

    candidates: list[dict] = []
    selected: float | None = None
    for column_index, h in enumerate(contract.candidate_h):
        observed_count = observed_counts[h]
        observed_rate = float(observed_count / observed_hours)
        boot_rate = boot_episode_sums[:, column_index] / boot_hours
        raw_bootstrap_quantile = float(
            np.quantile(boot_rate, contract.upper_quantile)
        )
        conservative_upper = max(observed_rate, raw_bootstrap_quantile)
        meets = conservative_upper <= contract.advisory_budget_episodes_per_hour
        candidates.append(
            {
                "h": h,
                "observed_episode_count": observed_count,
                "observed_episodes_per_hour": observed_rate,
                "bootstrap_raw_quantile_95_episodes_per_hour": raw_bootstrap_quantile,
                "conservative_upper_95_episodes_per_hour": conservative_upper,
                "meets_advisory_budget": bool(meets),
            }
        )
        if selected is None and meets:
            selected = h
    return {
        "selected_h": selected,
        "selection_rule": (
            "smallest candidate with max(full-flight observed episodes/hour, "
            "moving-block bootstrap p95) <= advisory budget"
        ),
        "upper_bound_contract": (
            "conservative upper = max(full-flight observed episodes/hour, "
            "raw moving-block bootstrap quantile)"
        ),
        "budget_episodes_per_hour": contract.advisory_budget_episodes_per_hour,
        "n_blocks": int(len(frame)),
        "observed_burden_source": observed_source,
        "observed_exposure_hours": observed_hours,
        "bootstrap_repetitions": contract.bootstrap_repetitions,
        "bootstrap_batch_size": contract.bootstrap_batch_size,
        "bootstrap_upper_quantile": contract.upper_quantile,
        "candidates": candidates,
    }
