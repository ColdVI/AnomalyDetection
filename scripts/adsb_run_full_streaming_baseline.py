"""Run the pre-registered, bounded-memory three-day ADS-B baseline.

This is the Step 5 execution entry point approved on 2026-07-13.  It only
accepts the already-open Silver Parquet corpus and explicit synthetic-clean
*reference* files used to exclude source flights.  It has no raw-tar or
Downloads discovery path.

The execution order is part of the contract:

1. inventory exactly the open Silver Parquet files and derive exact,
   day-prefixed flight IDs;
2. exclude every source flight present in the synthetic-clean references and
   make the SHA-256 flight split;
3. create the fail-if-exists immutable run manifest;
4. fit bounded deterministic robust samples on 2026-02-28 normal-fit flights;
5. select diagnostic rule/CUSUM thresholds using only 2026-02-28 normal
   calibration burden; and
6. report natural burden for fit, calibration, held-out synthetic-source
   validation, development, and frozen
   rehearsal without changing any setting after seeing later days.

No day is concatenated and no full-day window tensor is materialized.  One
Parquet part is loaded at a time with only the columns required by the two
detectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.cusum import CusumConfig, VectorPageCUSUM  # noqa: E402
from adsb.features import VECTOR_RESIDUAL_FEATURES, build_feature_table  # noqa: E402
from adsb.rules import CAP, RULE_CHANNELS, Z0, ResidualRuleScorer  # noqa: E402
from adsb.run_manifest import (  # noqa: E402
    InputSpec,
    create_immutable_run_manifest,
    sha256_json,
)
from adsb.segmentation import segment_flights  # noqa: E402
from adsb.streaming import (  # noqa: E402
    BoundedFramePrioritySampler,
    BoundedPrioritySampler,
    CusumBurdenCalibration,
    count_alarm_episodes,
    moving_block_burden_rows,
    prefixed_flight_id,
    robust_sample_calibration,
    scoreable_row_exposure_seconds,
    select_cusum_threshold,
    stable_fit_role,
)


FIT_DAY = "2026-02-28"
DEVELOPMENT_DAY = "2026-03-01"
REHEARSAL_DAY = "2026-03-16"
OPEN_DAYS = (FIT_DAY, DEVELOPMENT_DAY, REHEARSAL_DAY)
EXPECTED_PARTS_BY_DAY = {
    FIT_DAY: 237,
    DEVELOPMENT_DAY: 216,
    REHEARSAL_DAY: 185,
}
EXPECTED_TOTAL_PARTS = sum(EXPECTED_PARTS_BY_DAY.values())
EXPECTED_ROWS_BY_DAY = {
    FIT_DAY: 88_762_032,
    DEVELOPMENT_DAY: 85_991_023,
    REHEARSAL_DAY: 81_401_954,
}

DAY_INPUT_ROLE = {
    # The exact split contract further divides this date into fit/calibration.
    FIT_DAY: "fit",
    DEVELOPMENT_DAY: "development",
    REHEARSAL_DAY: "rehearsal",
}

SILVER_COLUMNS = (
    "_source_file",
    "source_id",
    "timestamp_utc",
    "lat",
    "lon",
    "alt",
    "alt_geom_m",
    "on_ground",
    "ground_speed_ms",
    "track_deg",
    "vertical_rate_ms",
)
PREFLIGHT_COLUMNS = ("_source_file", "source_id", "timestamp_utc")
ALL_RESIDUAL_CHANNELS = tuple(RULE_CHANNELS) + tuple(VECTOR_RESIDUAL_FEATURES)
SILVER_COLUMN_TYPE_FAMILIES = {
    "_source_file": "string",
    "source_id": "string",
    "timestamp_utc": "floating",
    "lat": "floating",
    "lon": "floating",
    "alt": "floating",
    "alt_geom_m": "floating",
    "on_ground": "boolean",
    "ground_speed_ms": "floating",
    "track_deg": "floating",
    "vertical_rate_ms": "floating",
}

FROZEN_CODE_FILES = (
    "scripts/adsb_run_full_streaming_baseline.py",
    "adsb/streaming.py",
    "adsb/cusum.py",
    "adsb/features.py",
    "adsb/rules.py",
    "adsb/segmentation.py",
    "adsb/run_manifest.py",
)

SPLIT_SEED = 20260713
FIT_FRACTION = 0.8
SEGMENT_GAP_S = 1800.0
WINDOW_SIZE = 12
WINDOW_STRIDE = 6
MAX_GAP_S = 60.0
FIT_SAMPLE_PROBABILITY = 0.02
BASELINE_SCORE_SAMPLE_PROBABILITY = 0.05
FIT_SAMPLE_CAP_PER_CHANNEL = 1_000_000
RULE_SCORE_SAMPLE_CAP = 1_000_000
DISTRIBUTION_SAMPLE_CAP = 200_000
CUSUM_BLOCK_SAMPLE_CAP = 200_000
SAMPLE_SEED = 20260713
DKW_ALPHA = 0.05
RULE_DIAGNOSTIC_EMPIRICAL_QUANTILE = 0.95
CUSUM_TARGET_VECTOR_SHIFT_MPS = 2.0
CUSUM_Z_CLIP = 3.0
CUSUM_MISSING_RESET_S = 60.0
EPISODE_MERGE_GAP_S = 60.0
CADENCE_STRATA_S = (2.0, 5.0, 15.0)

SOURCE_DAY_RE = re.compile(r"v(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})")


class StreamingContractError(ValueError):
    """An input or execution detail violates the frozen Step 5 contract."""


@dataclass(frozen=True)
class PartInventory:
    path: Path
    day: str
    source_file: str
    footer_rows: int
    source_id_count: int
    flight_id_count: int

    def report_record(self, repo_root: Path) -> dict[str, Any]:
        try:
            display_path = self.path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            display_path = self.path.resolve().as_posix()
        return {
            "path": display_path,
            "day": self.day,
            "source_file": self.source_file,
            "footer_rows": self.footer_rows,
            "source_id_count": self.source_id_count,
            "flight_id_count": self.flight_id_count,
        }


@dataclass(frozen=True)
class PreflightInventory:
    parts: tuple[PartInventory, ...]
    flight_ids_by_day: Mapping[str, tuple[str, ...]]
    part_counts_by_day: Mapping[str, int]
    footer_rows_by_day: Mapping[str, int]
    selected_schema: Mapping[str, str]


@dataclass
class DistributionAccumulator:
    """Exact finite moments plus a bounded deterministic quantile sample."""

    finite_n: int = 0
    missing_n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    sampler: BoundedPrioritySampler = field(
        default_factory=lambda: BoundedPrioritySampler(DISTRIBUTION_SAMPLE_CAP)
    )

    def add(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        sample_probability: float,
        seed: int,
        file_key: str,
        purpose: str,
    ) -> None:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        self.missing_n += int(len(array) - len(finite))
        if len(finite):
            self.finite_n += int(len(finite))
            self.total += float(finite.sum(dtype=np.float64))
            self.total_sq += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
            current_min = float(finite.min())
            current_max = float(finite.max())
            self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
            self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)
        self.sampler.add(
            array,
            probability=sample_probability,
            seed=seed,
            file_key=file_key,
            purpose=purpose,
        )

    def report(self) -> dict[str, Any]:
        sample = self.sampler.values
        if self.finite_n:
            mean = self.total / self.finite_n
            variance = max(0.0, self.total_sq / self.finite_n - mean * mean)
        else:
            mean = variance = None
        return {
            "finite_n": self.finite_n,
            "missing_n": self.missing_n,
            "mean": mean,
            "std_population": math.sqrt(variance) if variance is not None else None,
            "min": self.minimum,
            "max": self.maximum,
            "quantile_sample_n": int(len(sample)),
            "quantile_sample_capacity": self.sampler.capacity,
            "quantile_finite_rows_seen": self.sampler.finite_seen,
            "quantile_sample_probability": BASELINE_SCORE_SAMPLE_PROBABILITY,
            "dkw_cdf_error_95": (
                math.sqrt(math.log(2.0 / DKW_ALPHA) / (2.0 * len(sample)))
                if len(sample)
                else None
            ),
            "sample_quantiles": {
                "p01": float(np.quantile(sample, 0.01)) if len(sample) else None,
                "p05": float(np.quantile(sample, 0.05)) if len(sample) else None,
                "p50": float(np.quantile(sample, 0.50)) if len(sample) else None,
                "p95": float(np.quantile(sample, 0.95)) if len(sample) else None,
                "p99": float(np.quantile(sample, 0.99)) if len(sample) else None,
            },
        }


@dataclass
class RoleBurdenAccumulator:
    day: str
    role: str
    row_count: int = 0
    flight_count: int = 0
    rule_scoreable_window_count: int = 0
    rule_alert_episodes: int = 0
    rule_scoreable_exposure_s: float = 0.0
    rule_scoreable_flights: set[str] = field(default_factory=set)
    rule_alerted_flights: set[str] = field(default_factory=set)
    cusum_evaluable_rows: int = 0
    cusum_alert_rows: int = 0
    cusum_alert_episodes: int = 0
    cusum_scoreable_exposure_s: float = 0.0
    cusum_scoreable_flights: set[str] = field(default_factory=set)
    cusum_alerted_flights: set[str] = field(default_factory=set)
    cusum_cadence_exposure_s: Counter[str] = field(default_factory=Counter)
    cusum_cadence_alert_episodes: Counter[str] = field(default_factory=Counter)
    cusum_cadence_scoreable_flights: dict[str, set[str]] = field(
        default_factory=lambda: {name: set() for name in ("le_2s", "2_to_5s", "5_to_15s", "gt_15s")}
    )
    cusum_cadence_alerted_flights: dict[str, set[str]] = field(
        default_factory=lambda: {name: set() for name in ("le_2s", "2_to_5s", "5_to_15s", "gt_15s")}
    )
    reset_reasons: Counter[str] = field(default_factory=Counter)
    observed_channel_counts: Counter[int] = field(default_factory=Counter)
    channel_observed_rows: Counter[str] = field(default_factory=Counter)
    distributions: dict[str, DistributionAccumulator] = field(
        default_factory=lambda: {
            channel: DistributionAccumulator()
            for channel in (
                *ALL_RESIDUAL_CHANNELS,
                "rule_window_score",
                "rule_observed_support_q",
                "cusum_joint_score",
            )
        }
    )

    def report(self, *, rule_threshold: float, cusum_threshold: float | None) -> dict[str, Any]:
        rule_hours = self.rule_scoreable_exposure_s / 3600.0
        cusum_hours = self.cusum_scoreable_exposure_s / 3600.0
        return {
            "day": self.day,
            "role": self.role,
            "row_unit": {
                "n_rows": self.row_count,
                "cusum_evaluable_rows": self.cusum_evaluable_rows,
                "cusum_unevaluable_rows": self.row_count - self.cusum_evaluable_rows,
                "cusum_scoreability_contract": (
                    "airborne same-flight transition with 0<dt<=60s and at least one finite "
                    "fit-active signed velocity-residual axis"
                ),
                "cusum_joint_score_primary_distribution_support": "cusum_evaluable == true only",
                "cusum_alert_rows": self.cusum_alert_rows if cusum_threshold is not None else None,
                "cusum_reset_reason_counts": dict(sorted(self.reset_reasons.items())),
                "cusum_observed_channel_count_distribution": {
                    str(key): value for key, value in sorted(self.observed_channel_counts.items())
                },
                "cusum_channel_observed_rows": dict(sorted(self.channel_observed_rows.items())),
            },
            "rule_window_unit": {
                "window_size_rows": WINDOW_SIZE,
                "stride_rows": WINDOW_STRIDE,
                "max_within_window_gap_s": MAX_GAP_S,
                "n_scoreable_windows": self.rule_scoreable_window_count,
                "diagnostic_threshold": rule_threshold,
                "scoreability_contract": (
                    "at least one finite cell among fit-active rule channels on the full "
                    "12-row causal <=60s window support"
                ),
                "observed_support_q": (
                    "finite active-channel cells / (window rows * active rule channels)"
                ),
                "missing_cell_penalty_contribution": 0.0,
            },
            "rule_event_unit": {
                "n_alert_episodes": self.rule_alert_episodes,
                "merge_gap_s": EPISODE_MERGE_GAP_S,
                "emission_time": "window_t_end",
                "episodes_per_scoreable_flight_hour": (
                    self.rule_alert_episodes / rule_hours if rule_hours > 0.0 else None
                ),
            },
            "rule_flight_unit": {
                "n_input_flights": self.flight_count,
                "n_scoreable_flights": len(self.rule_scoreable_flights),
                "n_alerted_flights": len(self.rule_alerted_flights),
                "alerted_flight_fraction": (
                    len(self.rule_alerted_flights) / len(self.rule_scoreable_flights)
                    if self.rule_scoreable_flights
                    else None
                ),
                "scoreable_exposure_s": self.rule_scoreable_exposure_s,
                "scoreable_flight_hours": rule_hours,
            },
            "cusum_event_unit": {
                "selected_h": cusum_threshold,
                "n_alert_episodes": self.cusum_alert_episodes if cusum_threshold is not None else None,
                "merge_gap_s": EPISODE_MERGE_GAP_S,
                "emission_time": "row_timestamp_utc",
                "episodes_per_scoreable_flight_hour": (
                    self.cusum_alert_episodes / cusum_hours
                    if cusum_threshold is not None and cusum_hours > 0.0
                    else None
                ),
            },
            "cusum_flight_unit": {
                "n_input_flights": self.flight_count,
                "n_scoreable_flights": len(self.cusum_scoreable_flights),
                "n_alerted_flights": (
                    len(self.cusum_alerted_flights) if cusum_threshold is not None else None
                ),
                "alerted_flight_fraction": (
                    len(self.cusum_alerted_flights) / len(self.cusum_scoreable_flights)
                    if cusum_threshold is not None and self.cusum_scoreable_flights
                    else None
                ),
                "scoreable_exposure_s": self.cusum_scoreable_exposure_s,
                "scoreable_flight_hours": cusum_hours,
            },
            "cusum_cadence_strata": {
                stratum: {
                    "definition": "flight median positive evaluable dt; bins are (0,2], (2,5], (5,15], (15,60] seconds",
                    "n_scoreable_flights": len(self.cusum_cadence_scoreable_flights[stratum]),
                    "n_alerted_flights": len(self.cusum_cadence_alerted_flights[stratum]),
                    "scoreable_flight_hours": self.cusum_cadence_exposure_s[stratum] / 3600.0,
                    "n_alert_episodes": (
                        self.cusum_cadence_alert_episodes[stratum]
                        if cusum_threshold is not None else None
                    ),
                    "episodes_per_scoreable_flight_hour": (
                        self.cusum_cadence_alert_episodes[stratum]
                        / (self.cusum_cadence_exposure_s[stratum] / 3600.0)
                        if cusum_threshold is not None
                        and self.cusum_cadence_exposure_s[stratum] > 0.0
                        else None
                    ),
                }
                for stratum in self.cusum_cadence_scoreable_flights
            },
            "channel_and_score_distributions": {
                name: accumulator.report()
                for name, accumulator in sorted(self.distributions.items())
            },
        }


def _reject_sealed_or_raw_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    lowered_parts = {part.lower() for part in resolved.parts}
    forbidden_parts = lowered_parts & {"archive", "downloads", "raw"}
    if forbidden_parts:
        names = ", ".join(sorted(forbidden_parts))
        raise StreamingContractError(
            f"{label} cannot be under sealed/history/raw path component(s) {names}: {resolved}"
        )
    if resolved.suffix.lower() in {".tar", ".gz", ".tgz", ".zip", ".7z"}:
        raise StreamingContractError(f"{label} must not be a raw archive: {resolved}")
    return resolved


def stable_part_sample_key(part: PartInventory) -> str:
    """Location-independent key used by every deterministic row sampler."""

    return f"{part.day}/{part.path.name}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_code_hashes(repo_root: Path) -> dict[str, str]:
    """Hash dirty-worktree code bytes that define this run before the manifest."""

    root = repo_root.resolve(strict=True)
    hashes: dict[str, str] = {}
    for relative in FROZEN_CODE_FILES:
        candidate = _reject_sealed_or_raw_path(root / relative, label=f"frozen code {relative}")
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise StreamingContractError(f"Frozen code escapes repo root: {candidate}") from exc
        if not candidate.is_file():
            raise StreamingContractError(f"Frozen code is not a file: {candidate}")
        hashes[relative] = _sha256_file(candidate)
    return hashes


def _selected_schema_signature(schema: pa.Schema, *, path: Path) -> dict[str, str]:
    missing = sorted(set(SILVER_COLUMNS) - set(schema.names))
    if missing:
        raise StreamingContractError(f"Silver part {path} lacks columns: {missing}")

    signature: dict[str, str] = {}
    for column in SILVER_COLUMNS:
        dtype = schema.field(column).type
        family = SILVER_COLUMN_TYPE_FAMILIES[column]
        valid = {
            "string": pa.types.is_string(dtype) or pa.types.is_large_string(dtype),
            "floating": pa.types.is_floating(dtype),
            "boolean": pa.types.is_boolean(dtype),
        }[family]
        if not valid:
            raise StreamingContractError(
                f"Silver part {path} column {column!r} has {dtype}, expected {family}"
            )
        signature[column] = str(dtype)
    return signature


def _require_synthetic_path(path: Path) -> Path:
    resolved = _reject_sealed_or_raw_path(path, label="synthetic clean reference")
    if not any("synthetic" in part.lower() for part in resolved.parts):
        raise StreamingContractError(
            f"Synthetic clean reference lacks an explicit synthetic path marker: {resolved}"
        )
    if resolved.suffix.lower() != ".parquet" or not resolved.is_file():
        raise StreamingContractError(f"Synthetic clean reference must be a Parquet file: {resolved}")
    return resolved


def source_day(source_file: object) -> str:
    """Parse and validate one of the three open dates from provenance."""

    value = str(source_file)
    match = SOURCE_DAY_RE.search(value)
    if match is None:
        raise StreamingContractError(f"Cannot derive source day from _source_file={value!r}")
    day = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    if day not in OPEN_DAYS:
        raise StreamingContractError(f"Out-of-contract Silver day {day!r} in {value!r}")
    return day


def _single_source_file(frame: pd.DataFrame, *, path: Path) -> tuple[str, str]:
    values = frame["_source_file"].dropna().astype(str).unique().tolist()
    if len(values) != 1 or frame["_source_file"].isna().any():
        raise StreamingContractError(
            f"Each Silver part must contain exactly one non-null _source_file; {path} has {values}"
        )
    return values[0], source_day(values[0])


def _segment_prefixed(frame: pd.DataFrame, *, day: str) -> pd.DataFrame:
    if frame["source_id"].isna().any():
        raise StreamingContractError("source_id cannot be null during flight preflight")
    segmented = segment_flights(frame, gap_s=SEGMENT_GAP_S)
    segmented["flight_id"] = segmented["flight_id"].map(
        lambda flight_id: prefixed_flight_id(day, flight_id)
    )
    return segmented


def inventory_open_silver(
    silver_dir: Path,
    *,
    expected_total_parts: int = EXPECTED_TOTAL_PARTS,
    expected_parts_by_day: Mapping[str, int] | None = EXPECTED_PARTS_BY_DAY,
    expected_rows_by_day: Mapping[str, int] | None = EXPECTED_ROWS_BY_DAY,
) -> PreflightInventory:
    """Inventory every explicit open-Silver part and derive exact flight IDs.

    The `(day, source_id)` single-part assertion is important: part-local
    feature construction is only causal if one aircraft trace cannot continue
    silently in another part.
    """

    root = _reject_sealed_or_raw_path(Path(silver_dir), label="Silver directory")
    if not root.is_dir():
        raise StreamingContractError(f"Silver path is not a directory: {root}")
    if any("synthetic" in part.lower() for part in root.parts):
        raise StreamingContractError(f"Silver directory cannot be synthetic: {root}")

    paths = tuple(sorted(root.glob("*.parquet")))
    if len(paths) != expected_total_parts:
        raise StreamingContractError(
            f"Expected exactly {expected_total_parts} open Silver Parquet parts, found {len(paths)}"
        )

    owners: dict[tuple[str, str], Path] = {}
    parts: list[PartInventory] = []
    flights_by_day: dict[str, set[str]] = {day: set() for day in OPEN_DAYS}
    part_counts: Counter[str] = Counter()
    footer_rows: Counter[str] = Counter()
    selected_schema: dict[str, str] | None = None
    sample_keys: set[str] = set()

    for path in paths:
        resolved = _reject_sealed_or_raw_path(path, label="Silver input")
        parquet = pq.ParquetFile(resolved)
        current_schema = _selected_schema_signature(parquet.schema_arrow, path=resolved)
        if selected_schema is None:
            selected_schema = current_schema
        elif current_schema != selected_schema:
            differences = {
                column: {"first": selected_schema[column], "current": current_schema[column]}
                for column in SILVER_COLUMNS
                if current_schema[column] != selected_schema[column]
            }
            raise StreamingContractError(
                f"Selected Silver field types differ across parts at {resolved}: {differences}"
            )
        frame = pd.read_parquet(resolved, columns=list(PREFLIGHT_COLUMNS))
        if len(frame) != parquet.metadata.num_rows:
            raise StreamingContractError(f"Footer/read row mismatch in {resolved}")
        raw_source_file, day = _single_source_file(frame, path=resolved)
        source_ids = sorted(frame["source_id"].astype(str).unique().tolist())
        for source_id in source_ids:
            key = (day, source_id)
            previous = owners.get(key)
            if previous is not None and previous != resolved:
                raise StreamingContractError(
                    f"(day, source_id) spans parts: {key} in {previous.name} and {resolved.name}"
                )
            owners[key] = resolved
        segmented = _segment_prefixed(frame, day=day)
        flight_ids = set(segmented["flight_id"].astype(str).unique().tolist())
        flights_by_day[day].update(flight_ids)
        record = PartInventory(
            path=resolved,
            day=day,
            source_file=raw_source_file,
            footer_rows=int(parquet.metadata.num_rows),
            source_id_count=len(source_ids),
            flight_id_count=len(flight_ids),
        )
        sample_key = stable_part_sample_key(record)
        if sample_key in sample_keys:
            raise StreamingContractError(f"Stable part sample key is not unique: {sample_key}")
        sample_keys.add(sample_key)
        parts.append(record)
        part_counts[day] += 1
        footer_rows[day] += record.footer_rows
        del segmented, frame

    if set(part_counts) != set(OPEN_DAYS):
        raise StreamingContractError(f"Open Silver days are incomplete: {dict(part_counts)}")
    if expected_parts_by_day is not None and dict(part_counts) != dict(expected_parts_by_day):
        raise StreamingContractError(
            f"Per-day part counts differ from the frozen inventory: {dict(part_counts)}"
        )
    if expected_rows_by_day is not None and dict(footer_rows) != dict(expected_rows_by_day):
        raise StreamingContractError(
            f"Per-day footer rows differ from the frozen inventory: {dict(footer_rows)}"
        )

    return PreflightInventory(
        parts=tuple(parts),
        flight_ids_by_day={day: tuple(sorted(flights_by_day[day])) for day in OPEN_DAYS},
        part_counts_by_day=dict(part_counts),
        footer_rows_by_day=dict(footer_rows),
        selected_schema=dict(selected_schema or {}),
    )


def load_synthetic_exclusion_ids(
    paths: Sequence[Path],
    *,
    fit_day: str = FIT_DAY,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Read only `flight_id` from explicit clean synthetic references."""

    if not paths:
        raise StreamingContractError("At least one synthetic-clean reference is required")
    exclusions: set[str] = set()
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _require_synthetic_path(raw_path)
        parquet = pq.ParquetFile(path)
        if "flight_id" not in parquet.schema_arrow.names:
            raise StreamingContractError(f"Synthetic reference lacks flight_id: {path}")
        series = pd.read_parquet(path, columns=["flight_id"])["flight_id"]
        if series.isna().any():
            raise StreamingContractError(f"Synthetic reference has null flight_id: {path}")
        raw_ids = sorted(series.astype(str).unique().tolist())
        normalized = {
            flight_id if ":" in flight_id else prefixed_flight_id(fit_day, flight_id)
            for flight_id in raw_ids
        }
        exclusions.update(normalized)
        records.append(
            {
                "path": path.as_posix(),
                "footer_rows": int(parquet.metadata.num_rows),
                "raw_unique_flight_ids": len(raw_ids),
                "normalized_unique_flight_ids": len(normalized),
                "use": "exclusion_reference_only",
            }
        )
    return exclusions, records


def derive_exact_splits(
    flight_ids_by_day: Mapping[str, Sequence[str]],
    *,
    synthetic_exclusion_ids: Iterable[str],
    seed: int = SPLIT_SEED,
    fit_fraction: float = FIT_FRACTION,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Create exact roles and record attempted/matched synthetic exclusions."""

    excluded = set(map(str, synthetic_exclusion_ids))
    if not excluded:
        raise StreamingContractError("Synthetic exclusion set cannot be empty")
    fit_day_ids = set(map(str, flight_ids_by_day[FIT_DAY]))
    matched = sorted(fit_day_ids & excluded)
    unmatched = sorted(excluded - fit_day_ids)
    if unmatched or len(matched) != len(excluded):
        raise StreamingContractError(
            "Synthetic exclusion IDs must match the open fit day exactly; "
            f"supplied={len(excluded)} matched={len(matched)} unmatched={len(unmatched)}"
        )
    remaining = sorted(fit_day_ids - excluded)
    splits: dict[str, list[str]] = {
        "fit": [],
        "calibration": [],
        # These are unmodified real flights used as sources for the paired
        # synthetic corpus.  They never enter fit/threshold calibration, but
        # remain a legitimate separately reported natural-burden reference.
        "validation": matched,
        "development": sorted(map(str, flight_ids_by_day[DEVELOPMENT_DAY])),
        "rehearsal": sorted(map(str, flight_ids_by_day[REHEARSAL_DAY])),
    }
    for flight_id in remaining:
        role = stable_fit_role(flight_id, seed=seed, fit_fraction=fit_fraction)
        splits[role].append(flight_id)
    for ids in splits.values():
        ids.sort()
    audit = {
        "split_algorithm": "sha256(seed + NUL + day_prefixed_flight_id) uniform cutoff",
        "seed": seed,
        "fit_fraction": fit_fraction,
        "synthetic_exclusion_ids_supplied": len(excluded),
        "synthetic_exclusion_ids_matched_on_fit_day": len(matched),
        "synthetic_exclusion_ids_not_present_in_open_fit_day": 0,
        "synthetic_exclusion_match_contract": "all supplied IDs matched exactly before split",
        "excluded_matched_ids_sha256": sha256_json(matched),
        "role_counts": {role: len(ids) for role, ids in splits.items()},
    }
    return splits, audit


def _role_owner(splits: Mapping[str, Sequence[str]]) -> dict[str, str]:
    owner: dict[str, str] = {}
    for role, ids in splits.items():
        for flight_id in ids:
            if flight_id in owner:
                raise StreamingContractError(
                    f"Flight {flight_id!r} occurs in both {owner[flight_id]!r} and {role!r}"
                )
            owner[flight_id] = role
    return owner


def _read_feature_part(part: PartInventory) -> pd.DataFrame:
    frame = pd.read_parquet(part.path, columns=list(SILVER_COLUMNS))
    source_file, day = _single_source_file(frame, path=part.path)
    if day != part.day or source_file != part.source_file:
        raise StreamingContractError(f"Silver provenance changed after preflight: {part.path}")
    segmented = _segment_prefixed(frame, day=day)
    features = build_feature_table(segmented)
    time = pd.to_numeric(features["timestamp_utc"], errors="coerce")
    dt = time - time.shift(1)
    transition_valid = (
        features["flight_id"].eq(features["flight_id"].shift(1))
        & dt.gt(0.0)
        & dt.le(MAX_GAP_S)
    ).fillna(False)
    # Every residual is a transition quantity.  Segmentation permits gaps up
    # to 30 minutes, but the detector contract permits only causal <=60 s
    # transitions.  Values outside that support are unscoreable, never fitted
    # or silently treated as ordinary zero-penalty observations.
    features.loc[~transition_valid, list(ALL_RESIDUAL_CHANNELS)] = np.nan
    features["residual_transition_valid"] = transition_valid.to_numpy(dtype=bool)
    return features


def _append_sample(
    destination: dict[str, BoundedPrioritySampler],
    *,
    channel: str,
    values: np.ndarray,
    probability: float,
    file_key: str,
    purpose: str,
) -> None:
    destination[channel].add(
        values,
        probability=probability,
        seed=SAMPLE_SEED,
        file_key=file_key,
        purpose=purpose,
    )


def fit_sampled_robust_calibration(
    parts: Sequence[PartInventory],
    splits: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Fit rule/vector medians on the bounded 2% normal-fit row sample."""

    fit_ids = set(splits["fit"])
    chunks: dict[str, BoundedPrioritySampler] = {
        channel: BoundedPrioritySampler(FIT_SAMPLE_CAP_PER_CHANNEL)
        for channel in ALL_RESIDUAL_CHANNELS
    }
    fit_rows_seen = 0
    vector_fit_eligible_rows_seen = 0
    for part in parts:
        if part.day != FIT_DAY:
            continue
        features = _read_feature_part(part)
        selected = features["flight_id"].isin(fit_ids)
        fit_rows_seen += int(selected.sum())
        time = pd.to_numeric(features["timestamp_utc"], errors="coerce")
        ground = features["on_ground"].astype("boolean")
        vector_eligible = (
            selected
            & features["flight_id"].eq(features["flight_id"].shift(1))
            & time.sub(time.shift(1)).gt(0.0)
            & time.sub(time.shift(1)).le(MAX_GAP_S)
            & ground.eq(False)
            & ground.shift(1).eq(False)
        ).fillna(False)
        vector_fit_eligible_rows_seen += int(vector_eligible.sum())
        file_key = stable_part_sample_key(part)
        for channel in ALL_RESIDUAL_CHANNELS:
            channel_selected = (
                vector_eligible if channel in VECTOR_RESIDUAL_FEATURES else selected
            )
            _append_sample(
                chunks,
                channel=channel,
                values=features.loc[channel_selected, channel].to_numpy(dtype=float),
                probability=FIT_SAMPLE_PROBABILITY,
                file_key=file_key,
                purpose=f"normal_fit_robust_calibration:{channel}",
            )
        del features

    arrays = {channel: sampler.values for channel, sampler in chunks.items()}
    robust = robust_sample_calibration(arrays)
    robust.update(
        {
            "fit_rows_seen": fit_rows_seen,
            "vector_fit_eligible_rows_seen": vector_fit_eligible_rows_seen,
            "row_sample_probability": FIT_SAMPLE_PROBABILITY,
            "row_sample_capacity_per_channel": FIT_SAMPLE_CAP_PER_CHANNEL,
            "finite_rows_seen_by_channel": {
                channel: sampler.finite_seen for channel, sampler in chunks.items()
            },
            "sample_seed": SAMPLE_SEED,
            "sample_contract": (
                "independent deterministic Bernoulli per stable file/channel/purpose, "
                "then deterministic bottom-k hard capacity"
            ),
            "dkw_alpha": DKW_ALPHA,
        }
    )
    return robust


def build_rule_scorer(robust: Mapping[str, Any]) -> ResidualRuleScorer:
    calibration = {
        channel: {
            "median": float(robust["calibration"][channel]["median"]),
            "mad": float(robust["calibration"][channel]["mad"]),
        }
        for channel in RULE_CHANNELS
        if channel in robust["calibration"]
    }
    if not calibration:
        raise StreamingContractError("Every rule channel was excluded from robust calibration")
    scorer = ResidualRuleScorer(
        channels=list(RULE_CHANNELS),
        weights={channel: 1.0 for channel in RULE_CHANNELS},
    )
    scorer.calibration_ = calibration
    scorer.excluded_channels_ = [channel for channel in RULE_CHANNELS if channel not in calibration]
    return scorer


def build_cusum_detector(
    robust: Mapping[str, Any],
    *,
    threshold_h: float,
) -> VectorPageCUSUM:
    config = CusumConfig(
        target_vector_shift_mps=CUSUM_TARGET_VECTOR_SHIFT_MPS,
        threshold_h=threshold_h,
        max_gap_s=MAX_GAP_S,
        missing_reset_s=CUSUM_MISSING_RESET_S,
        z_clip=CUSUM_Z_CLIP,
    )
    detector = VectorPageCUSUM(config)
    calibration: dict[str, dict[str, float]] = {}
    for channel in VECTOR_RESIDUAL_FEATURES:
        if channel not in robust["calibration"]:
            continue
        median = float(robust["calibration"][channel]["median"])
        mad = float(robust["calibration"][channel]["mad"])
        calibration[channel] = {
            "median": median,
            "mad": mad,
            "k": config.minimum_axis_shift_mps / (2.0 * mad),
        }
    detector.calibration_ = calibration
    detector.excluded_channels_ = {
        channel: "mad_zero_or_no_finite_fit_sample"
        for channel in VECTOR_RESIDUAL_FEATURES
        if channel not in calibration
    }
    return detector


def cusum_axis_coverage(detector: VectorPageCUSUM) -> dict[str, Any]:
    active = [channel for channel in VECTOR_RESIDUAL_FEATURES if channel in detector.calibration_]
    excluded = [channel for channel in VECTOR_RESIDUAL_FEATURES if channel not in detector.calibration_]
    complete = len(active) == len(VECTOR_RESIDUAL_FEATURES)
    return {
        "required_axes": list(VECTOR_RESIDUAL_FEATURES),
        "active_axes": active,
        "excluded_axes": excluded,
        "active_axis_count": len(active),
        "required_axis_count": len(VECTOR_RESIDUAL_FEATURES),
        "status": "complete" if complete else "degraded_axis_coverage",
        "threshold_selection_allowed": complete,
        "gate_eligible": complete,
    }


def cusum_gate_decision(
    axis_coverage: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(axis_coverage.get("gate_eligible", False)):
        return {
            "gate_eligible": False,
            "gate_status": "fail_degraded_axis_coverage",
        }
    if selection.get("selected_h") is None:
        return {
            "gate_eligible": False,
            "gate_status": "fail_no_admissible_cusum_threshold",
        }
    return {
        "gate_eligible": True,
        "gate_status": "eligible_for_step_7_review",
    }


def causal_window_scores(
    flight: pd.DataFrame,
    row_penalties: np.ndarray,
    *,
    active_rule_channels: Sequence[str],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    max_gap_s: float = MAX_GAP_S,
) -> pd.DataFrame:
    """Score only windows wholly inside a strictly causal <=max-gap run."""

    if len(flight) != len(row_penalties):
        raise ValueError("flight and row_penalties must have identical lengths")
    ordered = flight.assign(_row_penalty=np.asarray(row_penalties, dtype=float)).sort_values(
        "timestamp_utc", kind="mergesort"
    )
    times = pd.to_numeric(ordered["timestamp_utc"], errors="coerce").to_numpy(dtype=float)
    penalties = ordered["_row_penalty"].to_numpy(dtype=float)
    if not active_rule_channels:
        raise ValueError("active_rule_channels cannot be empty")
    observed_cells = np.column_stack(
        [
            np.isfinite(ordered[channel].to_numpy(dtype=float))
            for channel in active_rule_channels
        ]
    )
    observable = observed_cells.any(axis=1)

    rows: list[dict[str, Any]] = []
    if not len(ordered):
        return pd.DataFrame(
            columns=[
                "t_start",
                "t_end",
                "score",
                "observed_active_channel_cells",
                "possible_active_channel_cells",
                "observed_rows",
                "observed_support_q",
            ]
        )
    dt = np.diff(times)
    breaks = np.flatnonzero(~np.isfinite(dt) | (dt <= 0.0) | (dt > max_gap_s)) + 1
    boundaries = np.concatenate(([0], breaks, [len(ordered)]))
    for run_start, run_end in zip(boundaries[:-1], boundaries[1:]):
        length = int(run_end - run_start)
        if length < window_size:
            continue
        for local_start in range(0, length - window_size + 1, stride):
            start = int(run_start + local_start)
            end = start + window_size
            if not observable[start:end].any():
                continue
            support = observed_cells[start:end]
            observed_count = int(support.sum())
            possible_count = int(support.size)
            rows.append(
                {
                    "t_start": float(times[start]),
                    "t_end": float(times[end - 1]),
                    "score": float(np.mean(penalties[start:end])),
                    "observed_active_channel_cells": observed_count,
                    "possible_active_channel_cells": possible_count,
                    "observed_rows": int(support.any(axis=1).sum()),
                    "observed_support_q": observed_count / possible_count,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "t_start",
            "t_end",
            "score",
            "observed_active_channel_cells",
            "possible_active_channel_cells",
            "observed_rows",
            "observed_support_q",
        ],
    )


def interval_union_exposure_seconds(windows: pd.DataFrame) -> float:
    """Union overlapping `[t_start,t_end]` window supports."""

    if windows.empty:
        return 0.0
    intervals = sorted(
        (float(start), float(end))
        for start, end in windows[["t_start", "t_end"]].itertuples(index=False, name=None)
        if np.isfinite(start) and np.isfinite(end) and end >= start
    )
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return float(total + current_end - current_start)


def calibrate_normal_burden(
    parts: Sequence[PartInventory],
    splits: Mapping[str, Sequence[str]],
    *,
    rule_scorer: ResidualRuleScorer,
    robust: Mapping[str, Any],
    contract: CusumBurdenCalibration,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use only February normal-calibration flights to select diagnostics."""

    calibration_ids = set(splits["calibration"])
    rule_score_sampler = BoundedPrioritySampler(RULE_SCORE_SAMPLE_CAP)
    rule_support_distribution = DistributionAccumulator()
    block_sampler = BoundedFramePrioritySampler(
        CUSUM_BLOCK_SAMPLE_CAP,
        seed=SAMPLE_SEED,
    )
    # A legal positive placeholder; alarm flags are ignored while selecting h.
    detector = build_cusum_detector(robust, threshold_h=contract.candidate_h[-1])
    axis_coverage = cusum_axis_coverage(detector)
    full_exposure_s = 0.0
    full_episodes_by_h: Counter[float] = Counter()

    for part in parts:
        if part.day != FIT_DAY:
            continue
        features = _read_feature_part(part)
        selected = features["flight_id"].isin(calibration_ids)
        role_features = features.loc[selected].copy()
        if role_features.empty:
            del features, role_features
            continue
        penalties = rule_scorer.row_penalties(role_features).to_numpy(dtype=float)
        part_window_scores: list[np.ndarray] = []
        part_block_rows: list[dict[str, Any]] = []
        cusum_rows = detector.score_rows(role_features)
        for flight_id, positions in role_features.groupby("flight_id", sort=False).indices.items():
            index = np.asarray(positions, dtype=int)
            flight = role_features.iloc[index]
            windows = causal_window_scores(
                flight,
                penalties[index],
                active_rule_channels=tuple(rule_scorer.calibration_),
            )
            if not windows.empty:
                part_window_scores.append(windows["score"].to_numpy(dtype=float))
                rule_support_distribution.add(
                    windows["observed_support_q"].to_numpy(dtype=float),
                    sample_probability=BASELINE_SCORE_SAMPLE_PROBABILITY,
                    seed=SAMPLE_SEED,
                    file_key=f"{stable_part_sample_key(part)}:{flight_id}",
                    purpose="normal_calibration_rule_observed_support_q",
                )

            scored = cusum_rows.iloc[index]
            times = flight["timestamp_utc"].to_numpy(dtype=float)
            scores = scored["cusum_joint_score"].to_numpy(dtype=float)
            evaluable = scored["cusum_evaluable"].to_numpy(dtype=bool)
            full_exposure_s += scoreable_row_exposure_seconds(
                times, evaluable, max_gap_s=MAX_GAP_S
            )
            for candidate_h in contract.candidate_h:
                full_episodes_by_h[candidate_h] += count_alarm_episodes(
                    times,
                    evaluable & np.isfinite(scores) & (scores > candidate_h),
                    merge_gap_s=contract.merge_gap_s,
                )
            part_block_rows.extend(
                moving_block_burden_rows(
                    flight_id,
                    times,
                    scores,
                    evaluable,
                    contract=contract,
                    max_gap_s=MAX_GAP_S,
                )
            )
        if part_block_rows:
            block_sampler.add(
                pd.DataFrame(part_block_rows),
                file_key=stable_part_sample_key(part),
                purpose="normal_calibration_cusum_moving_blocks",
            )
        del part_block_rows
        if part_window_scores:
            combined = np.concatenate(part_window_scores)
            rule_score_sampler.add(
                combined,
                probability=BASELINE_SCORE_SAMPLE_PROBABILITY,
                seed=SAMPLE_SEED,
                file_key=stable_part_sample_key(part),
                purpose="normal_calibration_rule_window_score",
            )
        del features, role_features, penalties, cusum_rows

    score_sample = rule_score_sampler.values
    if not len(score_sample):
        raise StreamingContractError("No finite rule-window sample in normal calibration")
    rule_threshold = float(np.quantile(score_sample, RULE_DIAGNOSTIC_EMPIRICAL_QUANTILE))

    if block_sampler.frame.empty and axis_coverage["threshold_selection_allowed"]:
        raise StreamingContractError("No CUSUM calibration blocks were produced")
    block_frame = block_sampler.frame.drop(columns=["_sample_priority"], errors="ignore")
    if block_frame.empty and axis_coverage["threshold_selection_allowed"]:
        raise StreamingContractError("No CUSUM calibration blocks were produced")
    if axis_coverage["threshold_selection_allowed"]:
        cusum_selection = select_cusum_threshold(
            block_frame,
            contract=contract,
            observed_exposure_s=full_exposure_s,
            observed_episodes_by_h=dict(full_episodes_by_h),
        )
    else:
        cusum_selection = {
            "status": "degraded_axis_coverage",
            "gate_eligible": False,
            "selected_h": None,
            "selection_performed": False,
            "reason": "both signed velocity residual axes must be active",
            "axis_coverage": axis_coverage,
            "observed_exposure_s": full_exposure_s,
            "observed_episodes_by_h": {
                str(h): int(full_episodes_by_h[h]) for h in contract.candidate_h
            },
        }
    gate_decision = cusum_gate_decision(axis_coverage, cusum_selection)
    report = {
        "source_role": "2026-02-28 normal calibration only",
        "synthetic_used": False,
        "rule_diagnostic": {
            "empirical_quantile": RULE_DIAGNOSTIC_EMPIRICAL_QUANTILE,
            "calibration_semantics": "empirical normal-calibration score quantile; not confidence or probability",
            "comparator": ">",
            "selected_threshold": rule_threshold,
            "score_sample_probability": BASELINE_SCORE_SAMPLE_PROBABILITY,
            "score_sample_seed": SAMPLE_SEED,
            "score_sample_n": int(len(score_sample)),
            "score_sample_capacity": RULE_SCORE_SAMPLE_CAP,
            "score_finite_windows_seen": rule_score_sampler.finite_seen,
            "dkw_cdf_error_95": math.sqrt(
                math.log(2.0 / DKW_ALPHA) / (2.0 * len(score_sample))
            ),
            "status": "diagnostic_only_not_operational_gate",
            "scoreability_contract": (
                "at least one finite fit-active rule-channel cell in a complete causal window"
            ),
            "observed_support_q_distribution": rule_support_distribution.report(),
        },
        "cusum_natural_burden_selection": cusum_selection,
        "cusum_axis_coverage": axis_coverage,
        "cusum_gate_eligible": gate_decision["gate_eligible"],
        "cusum_gate_status": gate_decision["gate_status"],
        "cusum_block_sampling": {
            "rows_seen": block_sampler.rows_seen,
            "retained_rows": len(block_frame),
            "capacity": CUSUM_BLOCK_SAMPLE_CAP,
            "algorithm": "deterministic bottom-k priority sample",
        },
    }
    return report, block_frame


def _update_role_burden(
    accumulator: RoleBurdenAccumulator,
    *,
    part: PartInventory,
    features: pd.DataFrame,
    rule_penalties: np.ndarray,
    rule_scorer: ResidualRuleScorer,
    rule_threshold: float,
    cusum_rows: pd.DataFrame,
    cusum_threshold: float | None,
) -> None:
    accumulator.row_count += len(features)
    accumulator.flight_count += int(features["flight_id"].nunique())
    file_key = f"{stable_part_sample_key(part)}:{accumulator.role}"
    for channel in ALL_RESIDUAL_CHANNELS:
        accumulator.distributions[channel].add(
            features[channel].to_numpy(dtype=float),
            sample_probability=BASELINE_SCORE_SAMPLE_PROBABILITY,
            seed=SAMPLE_SEED,
            file_key=file_key,
            purpose=f"natural_distribution:{channel}",
        )

    accumulator.cusum_evaluable_rows += int(cusum_rows["cusum_evaluable"].sum())
    accumulator.reset_reasons.update(cusum_rows["cusum_reset_reason"].astype(str).tolist())
    accumulator.observed_channel_counts.update(
        map(int, cusum_rows["cusum_observed_channels"].to_numpy(dtype=int))
    )
    for channel in VECTOR_RESIDUAL_FEATURES:
        observed_col = f"{channel}_observed"
        if observed_col in cusum_rows:
            accumulator.channel_observed_rows[channel] += int(cusum_rows[observed_col].sum())
    accumulator.distributions["cusum_joint_score"].add(
        cusum_rows.loc[
            cusum_rows["cusum_evaluable"].to_numpy(dtype=bool), "cusum_joint_score"
        ].to_numpy(dtype=float),
        sample_probability=BASELINE_SCORE_SAMPLE_PROBABILITY,
        seed=SAMPLE_SEED,
        file_key=file_key,
        purpose="natural_distribution:cusum_joint_score",
    )

    for flight_id, positions in features.groupby("flight_id", sort=False).indices.items():
        index = np.asarray(positions, dtype=int)
        flight = features.iloc[index]
        windows = causal_window_scores(
            flight,
            rule_penalties[index],
            active_rule_channels=tuple(rule_scorer.calibration_),
        )
        if not windows.empty:
            scores = windows["score"].to_numpy(dtype=float)
            alarms = scores > rule_threshold
            accumulator.rule_scoreable_window_count += len(windows)
            accumulator.rule_scoreable_flights.add(str(flight_id))
            accumulator.rule_scoreable_exposure_s += interval_union_exposure_seconds(windows)
            n_episodes = count_alarm_episodes(
                windows["t_end"].to_numpy(dtype=float),
                alarms,
                merge_gap_s=EPISODE_MERGE_GAP_S,
            )
            accumulator.rule_alert_episodes += n_episodes
            if n_episodes:
                accumulator.rule_alerted_flights.add(str(flight_id))
            accumulator.distributions["rule_window_score"].add(
                scores,
                sample_probability=BASELINE_SCORE_SAMPLE_PROBABILITY,
                seed=SAMPLE_SEED,
                file_key=f"{file_key}:{flight_id}",
                purpose="natural_distribution:rule_window_score",
            )
            accumulator.distributions["rule_observed_support_q"].add(
                windows["observed_support_q"].to_numpy(dtype=float),
                sample_probability=BASELINE_SCORE_SAMPLE_PROBABILITY,
                seed=SAMPLE_SEED,
                file_key=f"{file_key}:{flight_id}",
                purpose="natural_distribution:rule_observed_support_q",
            )

        scored = cusum_rows.iloc[index]
        times = flight["timestamp_utc"].to_numpy(dtype=float)
        evaluable = scored["cusum_evaluable"].to_numpy(dtype=bool)
        exposure_s = scoreable_row_exposure_seconds(times, evaluable, max_gap_s=MAX_GAP_S)
        accumulator.cusum_scoreable_exposure_s += exposure_s
        dt = np.diff(times)
        valid_dt = dt[
            evaluable[1:] & np.isfinite(dt) & (dt > 0.0) & (dt <= MAX_GAP_S)
        ]
        median_dt = float(np.median(valid_dt)) if len(valid_dt) else None
        if median_dt is None:
            cadence_stratum = None
        elif median_dt <= CADENCE_STRATA_S[0]:
            cadence_stratum = "le_2s"
        elif median_dt <= CADENCE_STRATA_S[1]:
            cadence_stratum = "2_to_5s"
        elif median_dt <= CADENCE_STRATA_S[2]:
            cadence_stratum = "5_to_15s"
        else:
            cadence_stratum = "gt_15s"
        if exposure_s > 0.0:
            accumulator.cusum_scoreable_flights.add(str(flight_id))
            accumulator.cusum_cadence_exposure_s[cadence_stratum] += exposure_s
            accumulator.cusum_cadence_scoreable_flights[cadence_stratum].add(str(flight_id))
        if cusum_threshold is not None:
            alarms = evaluable & (
                scored["cusum_joint_score"].to_numpy(dtype=float) > cusum_threshold
            )
            accumulator.cusum_alert_rows += int(alarms.sum())
            n_episodes = count_alarm_episodes(
                times, alarms, merge_gap_s=EPISODE_MERGE_GAP_S
            )
            accumulator.cusum_alert_episodes += n_episodes
            if cadence_stratum is not None:
                accumulator.cusum_cadence_alert_episodes[cadence_stratum] += n_episodes
            if n_episodes:
                accumulator.cusum_alerted_flights.add(str(flight_id))
                if cadence_stratum is not None:
                    accumulator.cusum_cadence_alerted_flights[cadence_stratum].add(str(flight_id))


def evaluate_natural_days(
    parts: Sequence[PartInventory],
    splits: Mapping[str, Sequence[str]],
    *,
    rule_scorer: ResidualRuleScorer,
    robust: Mapping[str, Any],
    rule_threshold: float,
    cusum_threshold: float | None,
) -> dict[str, Any]:
    """Evaluate fixed settings part-locally; rehearsal never feeds back."""

    owner = _role_owner(splits)
    detector_h = cusum_threshold if cusum_threshold is not None else 1.0
    detector = build_cusum_detector(robust, threshold_h=detector_h)
    accumulators = {
        role: RoleBurdenAccumulator(
            day=FIT_DAY if role in {"fit", "calibration", "validation"} else (
                DEVELOPMENT_DAY if role == "development" else REHEARSAL_DAY
            ),
            role=role,
        )
        for role in ("fit", "calibration", "validation", "development", "rehearsal")
    }

    for part in parts:
        features = _read_feature_part(part)
        features["_evaluation_role"] = features["flight_id"].map(owner)
        if features["_evaluation_role"].isna().any():
            missing = sorted(
                features.loc[features["_evaluation_role"].isna(), "flight_id"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise StreamingContractError(
                f"Every inventoried flight must have an exact evaluation role; missing={missing[:10]}"
            )
        # Synthetic-source flights have the explicit natural-only validation role.
        for role, role_features in features.groupby("_evaluation_role", sort=False, dropna=True):
            if role not in accumulators:
                raise StreamingContractError(f"Unexpected evaluation role {role!r}")
            role_features = role_features.drop(columns=["_evaluation_role"]).reset_index(drop=True)
            penalties = rule_scorer.row_penalties(role_features).to_numpy(dtype=float)
            cusum_rows = detector.score_rows(role_features)
            _update_role_burden(
                accumulators[role],
                part=part,
                features=role_features,
                rule_penalties=penalties,
                rule_scorer=rule_scorer,
                rule_threshold=rule_threshold,
                cusum_rows=cusum_rows,
                cusum_threshold=cusum_threshold,
            )
            del role_features, penalties, cusum_rows
        del features

    return {
        "settings_frozen_before_development_and_rehearsal": True,
        "rehearsal_feedback_into_settings": False,
        "units_are_separate": ["row", "window", "event", "flight"],
        "roles": {
            role: accumulator.report(
                rule_threshold=rule_threshold,
                cusum_threshold=cusum_threshold,
            )
            for role, accumulator in accumulators.items()
        },
    }


def frozen_config(
    contract: CusumBurdenCalibration,
    *,
    selected_schema: Mapping[str, str],
    code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """All constants hashed into the immutable manifest before scoring."""

    return {
        "step": 5,
        "contract_date": "2026-07-13",
        "open_days": {
            FIT_DAY: "normal_fit_normal_calibration_and_held_out_source_validation",
            DEVELOPMENT_DAY: "development",
            REHEARSAL_DAY: "frozen_rehearsal",
        },
        "expected_inventory": {
            "total_parts": EXPECTED_TOTAL_PARTS,
            "parts_by_day": EXPECTED_PARTS_BY_DAY,
            "footer_rows_by_day": EXPECTED_ROWS_BY_DAY,
        },
        "silver_columns": list(SILVER_COLUMNS),
        "silver_selected_schema": dict(selected_schema),
        "silver_type_families": dict(SILVER_COLUMN_TYPE_FAMILIES),
        "forbidden_input_path_components": ["archive", "downloads", "raw"],
        "frozen_code_sha256": dict(code_sha256),
        "part_local_contract": "(day, source_id) must occur in exactly one Parquet part",
        "source_order_contract": (
            "stable mergesort by source_id,timestamp_utc; equal timestamps retain Parquet source order; "
            "CUSUM dt=0 rows skip state update"
        ),
        "flight_identity": "day:source_id_segmentSequence",
        "segmentation_gap_s": SEGMENT_GAP_S,
        "split": {
            "algorithm": "sha256(seed + NUL + day_prefixed_flight_id) uniform cutoff",
            "seed": SPLIT_SEED,
            "fit_fraction": FIT_FRACTION,
        },
        "synthetic_policy": "clean references provide exclusion IDs only; no synthetic fit/calibration",
        "fit_robust_sampling": {
            "probability": FIT_SAMPLE_PROBABILITY,
            "seed": SAMPLE_SEED,
            "dkw_alpha": DKW_ALPHA,
            "mad_zero_policy": "exclude",
            "stable_part_key": "day/Parquet-basename (repo-location independent)",
            "hard_capacity_per_channel": FIT_SAMPLE_CAP_PER_CHANNEL,
        },
        "rule": {
            "channels": list(RULE_CHANNELS),
            "weights": {channel: 1.0 for channel in RULE_CHANNELS},
            "z0": Z0,
            "cap": CAP,
            "row_penalty": "sum weighted clipped robust-z excess",
            "window_reduction": "mean",
            "window_size_rows": WINDOW_SIZE,
            "stride_rows": WINDOW_STRIDE,
            "max_gap_s": MAX_GAP_S,
            "baseline_score_sample_probability": BASELINE_SCORE_SAMPLE_PROBABILITY,
            "diagnostic_empirical_quantile": RULE_DIAGNOSTIC_EMPIRICAL_QUANTILE,
            "empirical_quantile_semantics": "normal-calibration distribution quantile; not confidence or probability",
            "threshold_status": "diagnostic_only",
            "scoreability": (
                "complete causal 12-row window, <=60s transitions, and at least one finite "
                "fit-active rule-channel cell"
            ),
            "observed_support_q": (
                "finite fit-active rule-channel cells / (12 * number of active channels)"
            ),
            "missing_cell_penalty_contribution": 0.0,
        },
        "cusum": {
            "channels": list(VECTOR_RESIDUAL_FEATURES),
            "target_vector_shift_mps": CUSUM_TARGET_VECTOR_SHIFT_MPS,
            "z_clip": CUSUM_Z_CLIP,
            "max_gap_s": MAX_GAP_S,
            "missing_reset_s": CUSUM_MISSING_RESET_S,
            "state_count": 4,
            "joint_statistic": "max",
            "burden_calibration": asdict(contract),
            "selection_data": "2026-02-28 normal calibration only",
            "synthetic_selection_forbidden": True,
            "axis_coverage_gate": (
                "both east and north residual axes must be active; otherwise status is "
                "degraded_axis_coverage, selected_h is null, and gate fails"
            ),
            "scoreability": (
                "airborne same-flight 0<dt<=60s transition with at least one finite active axis"
            ),
            "primary_score_distribution_support": "cusum_evaluable == true only",
        },
        "episode_merge_gap_s": EPISODE_MERGE_GAP_S,
        "natural_distribution_sample_probability": BASELINE_SCORE_SAMPLE_PROBABILITY,
        "bounded_sample_capacities": {
            "fit_per_channel": FIT_SAMPLE_CAP_PER_CHANNEL,
            "rule_score": RULE_SCORE_SAMPLE_CAP,
            "natural_distribution_per_role_channel": DISTRIBUTION_SAMPLE_CAP,
            "cusum_calibration_blocks": CUSUM_BLOCK_SAMPLE_CAP,
        },
        "cadence_strata_flight_median_dt_s": [0.0, *CADENCE_STRATA_S, MAX_GAP_S],
        "sample_key_contract": "all deterministic samplers use day/Parquet-basename, never absolute paths",
        "later_day_parameter_updates_forbidden": True,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("JSON artifact contains a non-finite number; use explicit None")
        return numeric
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    safe_payload = _json_safe(dict(payload))
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            safe_payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(value)
        handle.write("\n")


def write_artifact_checksums(run_dir: Path) -> Path:
    """Write the last run artifact; the index intentionally cannot hash itself."""

    target = run_dir / "artifact_checksums.json"
    if target.exists():
        raise FileExistsError(target)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == target:
            continue
        relative = path.relative_to(run_dir).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "self_excluded": True,
        "files": files,
    }
    write_json_exclusive(target, payload)
    return target


def run(
    *,
    repo_root: Path,
    silver_dir: Path,
    synthetic_clean_references: Sequence[Path],
    run_dir: Path,
) -> dict[str, Any]:
    """Execute the full immutable run and return the in-memory final report."""

    root = repo_root.resolve(strict=True)
    destination = run_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists and is immutable: {destination}")
    forbidden_destination_parts = {
        part.lower() for part in destination.parts
    } & {"archive", "downloads", "raw"}
    if forbidden_destination_parts:
        raise StreamingContractError(
            f"Run directory uses forbidden path components: {sorted(forbidden_destination_parts)}"
        )

    inventory = inventory_open_silver(silver_dir)
    exclusion_ids, exclusion_records = load_synthetic_exclusion_ids(
        synthetic_clean_references
    )
    splits, split_audit = derive_exact_splits(
        inventory.flight_ids_by_day,
        synthetic_exclusion_ids=exclusion_ids,
    )
    contract = CusumBurdenCalibration(
        advisory_budget_episodes_per_hour=12.0,
        bootstrap_repetitions=500,
        bootstrap_seed=20260713,
        upper_quantile=0.95,
        merge_gap_s=EPISODE_MERGE_GAP_S,
        moving_block_s=300.0,
        moving_block_stride_s=150.0,
    )
    code_sha256 = frozen_code_hashes(root)
    config = frozen_config(
        contract,
        selected_schema=inventory.selected_schema,
        code_sha256=code_sha256,
    )
    base_config_sha256 = sha256_json(config)

    inputs = [InputSpec(part.path, DAY_INPUT_ROLE[part.day]) for part in inventory.parts]
    inputs.extend(
        InputSpec(_require_synthetic_path(path), "reference")
        for path in synthetic_clean_references
    )
    manifest_path = create_immutable_run_manifest(
        run_dir=destination,
        repo_root=root,
        inputs=inputs,
        splits=splits,
        split_algorithm="sha256_uniform_cutoff_v1",
        split_seed=SPLIT_SEED,
        synthetic_flight_ids=exclusion_ids,
        config=config,
    )

    preflight_report = {
        "manifest": manifest_path.name,
        "inventory": {
            "part_count": len(inventory.parts),
            "part_counts_by_day": inventory.part_counts_by_day,
            "footer_rows_by_day": inventory.footer_rows_by_day,
            "selected_schema": inventory.selected_schema,
            "parts": [part.report_record(root) for part in inventory.parts],
        },
        "synthetic_clean_references": exclusion_records,
        "split_audit": split_audit,
    }
    write_json_exclusive(destination / "preflight_inventory.json", preflight_report)

    robust = fit_sampled_robust_calibration(inventory.parts, splits)
    rule_scorer = build_rule_scorer(robust)
    robust_report = {
        **robust,
        "rule_scorer": rule_scorer.to_dict(),
        "cusum_calibration_preview": build_cusum_detector(
            robust, threshold_h=contract.candidate_h[-1]
        ).to_dict(),
    }
    write_json_exclusive(destination / "fit_robust_calibration.json", robust_report)

    threshold_report, block_frame = calibrate_normal_burden(
        inventory.parts,
        splits,
        rule_scorer=rule_scorer,
        robust=robust,
        contract=contract,
    )
    block_path = destination / "cusum_normal_calibration_blocks.parquet"
    if block_path.exists():
        raise FileExistsError(block_path)
    block_frame.to_parquet(block_path, index=False)
    write_json_exclusive(destination / "normal_burden_calibration.json", threshold_report)
    del block_frame

    rule_threshold = float(threshold_report["rule_diagnostic"]["selected_threshold"])
    selected_h = threshold_report["cusum_natural_burden_selection"]["selected_h"]
    cusum_threshold = float(selected_h) if selected_h is not None else None
    selected_detector = (
        build_cusum_detector(robust, threshold_h=cusum_threshold).to_dict()
        if cusum_threshold is not None
        else None
    )
    derived_config = {
        "base_config_sha256": base_config_sha256,
        "fit_robust_calibration_file_sha256": _sha256_file(
            destination / "fit_robust_calibration.json"
        ),
        "normal_burden_calibration_file_sha256": _sha256_file(
            destination / "normal_burden_calibration.json"
        ),
        "rule_scorer": rule_scorer.to_dict(),
        "rule_diagnostic": {
            "selected_threshold": rule_threshold,
            "comparator": ">",
            "empirical_quantile": RULE_DIAGNOSTIC_EMPIRICAL_QUANTILE,
            "status": "diagnostic_only_not_operational_gate",
        },
        "cusum": {
            "axis_coverage": threshold_report["cusum_axis_coverage"],
            "selected_h": cusum_threshold,
            "selection": threshold_report["cusum_natural_burden_selection"],
            "selected_detector": selected_detector,
            "gate_eligible": threshold_report["cusum_gate_eligible"],
            "gate_status": threshold_report["cusum_gate_status"],
        },
        "frozen_before_roles": ["development", "rehearsal"],
        "later_role_feedback_forbidden": True,
    }
    derived_record = {
        "schema_version": 1,
        "derived_config": derived_config,
        "payload_sha256": sha256_json(derived_config),
    }
    derived_path = destination / "derived_frozen_config.json"
    write_json_exclusive(derived_path, derived_record)
    derived_file_sha256 = _sha256_file(derived_path)
    write_text_exclusive(
        destination / "derived_frozen_config.sha256",
        derived_file_sha256,
    )

    natural = evaluate_natural_days(
        inventory.parts,
        splits,
        rule_scorer=rule_scorer,
        robust=robust,
        rule_threshold=rule_threshold,
        cusum_threshold=cusum_threshold,
    )
    write_json_exclusive(destination / "natural_burden_by_role.json", natural)
    if sha256_json(config) != base_config_sha256:
        raise StreamingContractError("Frozen base config changed during evaluation")
    if frozen_code_hashes(root) != code_sha256:
        raise StreamingContractError("Frozen code bytes changed during evaluation")

    final_report = {
        "run_id": destination.name,
        "manifest": manifest_path.name,
        "config_sha256": base_config_sha256,
        "config_and_code_unchanged_through_evaluation": True,
        "derived_frozen_config": {
            "path": derived_path.name,
            "payload_sha256": derived_record["payload_sha256"],
            "file_sha256": derived_file_sha256,
            "sidecar": "derived_frozen_config.sha256",
        },
        "artifact_checksum_index": "artifact_checksums.json",
        "input_scope": "638 already-open Silver Parquet parts; no raw/sealed input",
        "synthetic_training_rows": 0,
        "synthetic_reference_use": "flight-ID exclusion only",
        "fit_calibration": robust_report,
        "normal_threshold_calibration": threshold_report,
        "natural_burden": natural,
        "rehearsal_changed_parameters": False,
        "gate_status": (
            "evidence_only_pending_step_7_review"
            if threshold_report["cusum_gate_eligible"]
            else threshold_report["cusum_gate_status"]
        ),
    }
    write_json_exclusive(destination / "streaming_baseline_report.json", final_report)
    write_artifact_checksums(destination)
    return final_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/objectstore/silver/adsblol_historical"),
    )
    parser.add_argument(
        "--synthetic-clean-reference",
        action="append",
        type=Path,
        required=True,
        help="Explicit v1/v2 clean Parquet used only to exclude its source flight IDs",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        repo_root=args.repo_root,
        silver_dir=args.silver_dir,
        synthetic_clean_references=args.synthetic_clean_reference,
        run_dir=args.run_dir,
    )
    print(json.dumps({"run_id": report["run_id"], "gate_status": report["gate_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
