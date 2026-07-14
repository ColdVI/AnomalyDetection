"""Fail-closed truth-v2 evaluation for the frozen Step-5 vector CUSUM.

This module is evaluation-only.  It cannot fit a detector, calibrate a channel,
search a threshold, or fuse scores.  The detector is loaded byte-for-byte from
the completed Step-5 artifact chain and emits at the current row timestamp.

Truth is also row-local and causal: a positive row must be observable-changed,
truth-evaluable, and CUSUM-evaluable.  Unchanged rows from a corrupt file are
timeline diagnostics only; they are never recycled as AUC negatives.  The sole
negative source is the paired, unmodified clean reference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from adsb.cusum import VectorPageCUSUM
from adsb.evaluation import event_observability_denominators, truth_event_table
from adsb.features import VECTOR_RESIDUAL_FEATURES, build_feature_table
from adsb.run_manifest import (
    InputSpec,
    create_immutable_run_manifest,
    inspect_input_file,
    sha256_file,
    sha256_json,
)
from adsb.synthetic import PHYSICS_BREAK_RECIPES


STEP5_REQUIRED_FILES = (
    "run_manifest.json",
    "derived_frozen_config.json",
    "derived_frozen_config.sha256",
    "streaming_baseline_report.json",
    "artifact_checksums.json",
    "natural_burden_by_role.json",
)
FROZEN_SCORING_CODE = ("adsb/cusum.py", "adsb/features.py")
DEFAULT_SAMPLE_CAPACITY_PER_CLASS = 250_000
DEFAULT_SAMPLE_SEED = 20260713
DEFAULT_EPISODE_MERGE_GAP_S = 60.0

PAIR_COLUMNS = ("flight_id", "timestamp_utc")
SCORING_COLUMNS = (
    "flight_id",
    "timestamp_utc",
    "lat",
    "lon",
    "alt",
    "alt_geom_m",
    "ground_speed_ms",
    "track_deg",
    "vertical_rate_ms",
    "roll_deg",
    "on_ground",
    "event_id",
    "event_type",
    "attack_onset",
    "observable_onset",
    "event_end",
    "injection_active",
    "observable_changed",
    "evaluable_truth",
)


class CusumTruthV2ContractError(ValueError):
    """An input or provenance relation is inconsistent with the frozen run."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CusumTruthV2ContractError(f"{path} must contain a JSON object")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Non-finite floats are forbidden in evaluation JSON")
        return result
    if value is pd.NA:
        return None
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_safe(value),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _assert_unsealed_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    forbidden = {part.lower() for part in resolved.parts} & {
        "archive",
        "downloads",
        "raw",
    }
    if forbidden:
        raise CusumTruthV2ContractError(
            f"{label} uses forbidden path components: {sorted(forbidden)}"
        )
    return resolved


def _verify_indexed_file(
    directory: Path,
    checksum_index: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    files = checksum_index.get("files")
    if not isinstance(files, dict) or name not in files:
        raise CusumTruthV2ContractError(f"Step-5 checksum index lacks {name}")
    record = files[name]
    path = directory / name
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise CusumTruthV2ContractError(f"Step-5 byte count changed: {name}")
    actual = sha256_file(path)
    if record.get("sha256") != actual:
        raise CusumTruthV2ContractError(f"Step-5 SHA-256 changed: {name}")
    return {"path": name, "bytes": path.stat().st_size, "sha256": actual}


def _natural_burden_panel(
    natural: Mapping[str, Any], *, selected_h: float
) -> dict[str, Any]:
    if natural.get("settings_frozen_before_development_and_rehearsal") is not True:
        raise CusumTruthV2ContractError("Step-5 settings were not frozen before later roles")
    if natural.get("rehearsal_feedback_into_settings") is not False:
        raise CusumTruthV2ContractError("Step-5 rehearsal feedback guard is not false")
    roles = natural.get("roles")
    if not isinstance(roles, dict):
        raise CusumTruthV2ContractError("Step-5 natural burden lacks roles")

    panel: dict[str, Any] = {}
    for role in ("validation", "development", "rehearsal"):
        record = roles.get(role)
        if not isinstance(record, dict) or record.get("role") != role:
            raise CusumTruthV2ContractError(f"Step-5 burden lacks role {role}")
        event = record.get("cusum_event_unit")
        flight = record.get("cusum_flight_unit")
        if not isinstance(event, dict) or not isinstance(flight, dict):
            raise CusumTruthV2ContractError(f"Step-5 {role} CUSUM burden is incomplete")
        if float(event.get("selected_h", np.nan)) != selected_h:
            raise CusumTruthV2ContractError(f"Step-5 {role} burden uses another h")
        if event.get("emission_time") != "row_timestamp_utc":
            raise CusumTruthV2ContractError(f"Step-5 {role} emission unit changed")
        if float(event.get("merge_gap_s", np.nan)) != DEFAULT_EPISODE_MERGE_GAP_S:
            raise CusumTruthV2ContractError(f"Step-5 {role} episode contract changed")
        panel[role] = {
            "day": record.get("day"),
            "role": role,
            "cusum_event_unit": event,
            "cusum_flight_unit": flight,
            "cusum_cadence_strata": record.get("cusum_cadence_strata"),
        }
    return {
        "unit": "natural_cusum_burden_from_completed_step5",
        "primary_role": "validation",
        "selected_h": selected_h,
        "validation": panel["validation"],
        "temporal_stability_panel": {
            "development": panel["development"],
            "rehearsal": panel["rehearsal"],
        },
        "synthetic_rows_used_for_burden": 0,
    }


@dataclass(frozen=True)
class FrozenStep5Bundle:
    detector: VectorPageCUSUM
    selected_detector: dict[str, Any]
    selected_h: float
    burden_panel: dict[str, Any]
    input_records: tuple[dict[str, Any], ...]
    base_config_sha256: str
    derived_payload_sha256: str
    validation_day: str
    validation_flight_ids: tuple[str, ...]


def load_frozen_step5_bundle(
    step5_dir: Path,
    *,
    repo_root: Path,
) -> FrozenStep5Bundle:
    """Verify the completed Step-5 hash/config chain and load its CUSUM.

    Only scoring dependencies are required to remain byte-identical today.
    Other Step-5 code hashes remain historical evidence in its immutable
    manifest but are not imported by this evaluator.
    """

    directory = _assert_unsealed_path(step5_dir, label="Step-5 run")
    root = repo_root.resolve(strict=True)
    paths = {name: (directory / name) for name in STEP5_REQUIRED_FILES}
    for name, path in paths.items():
        if not path.is_file():
            raise CusumTruthV2ContractError(f"Step-5 run is incomplete: {name}")

    index = _load_json(paths["artifact_checksums.json"])
    if (
        index.get("schema_version") != 1
        or index.get("algorithm") != "sha256"
        or index.get("self_excluded") is not True
    ):
        raise CusumTruthV2ContractError("Unsupported Step-5 checksum index")
    indexed_names = (
        "run_manifest.json",
        "derived_frozen_config.json",
        "derived_frozen_config.sha256",
        "streaming_baseline_report.json",
        "natural_burden_by_role.json",
    )
    input_records = tuple(
        _verify_indexed_file(directory, index, name) for name in indexed_names
    )

    manifest = _load_json(paths["run_manifest.json"])
    if manifest.get("manifest_schema_version") != 1:
        raise CusumTruthV2ContractError("Unsupported Step-5 manifest schema")
    config_record = manifest.get("config")
    if not isinstance(config_record, dict):
        raise CusumTruthV2ContractError("Step-5 manifest lacks config")
    base_config = config_record.get("value")
    base_hash = config_record.get("sha256")
    if not isinstance(base_config, dict) or base_hash != sha256_json(base_config):
        raise CusumTruthV2ContractError("Step-5 base config hash mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or manifest.get("input_contract_sha256") != sha256_json(inputs):
        raise CusumTruthV2ContractError("Step-5 input contract hash mismatch")
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, dict) or not isinstance(split_contract.get("splits"), dict):
        raise CusumTruthV2ContractError("Step-5 split contract is missing")
    normalized_splits: dict[str, list[str]] = {}
    for role, split_record in split_contract["splits"].items():
        if not isinstance(split_record, dict) or not isinstance(split_record.get("flight_ids"), list):
            raise CusumTruthV2ContractError(f"Step-5 split {role} is malformed")
        ids = [str(value) for value in split_record["flight_ids"]]
        if (
            ids != sorted(ids)
            or len(ids) != len(set(ids))
            or split_record.get("flight_id_count") != len(ids)
            or split_record.get("flight_ids_sha256") != sha256_json(ids)
        ):
            raise CusumTruthV2ContractError(f"Step-5 split {role} hash/count mismatch")
        normalized_splits[str(role)] = ids
    split_payload = {
        "algorithm": split_contract.get("algorithm"),
        "seed": split_contract.get("seed"),
        "splits": normalized_splits,
    }
    if split_contract.get("contract_sha256") != sha256_json(split_payload):
        raise CusumTruthV2ContractError("Step-5 split-contract hash mismatch")
    validation_ids = tuple(normalized_splits.get("validation", ()))
    if not validation_ids:
        raise CusumTruthV2ContractError("Step-5 validation split is empty")

    derived_record = _load_json(paths["derived_frozen_config.json"])
    derived = derived_record.get("derived_config")
    derived_hash = derived_record.get("payload_sha256")
    if (
        derived_record.get("schema_version") != 1
        or not isinstance(derived, dict)
        or derived_hash != sha256_json(derived)
        or derived.get("base_config_sha256") != base_hash
    ):
        raise CusumTruthV2ContractError("Step-5 derived config chain mismatch")
    derived_file_hash = sha256_file(paths["derived_frozen_config.json"])
    sidecar = paths["derived_frozen_config.sha256"].read_text(encoding="utf-8").strip()
    if sidecar != derived_file_hash:
        raise CusumTruthV2ContractError("Step-5 derived-config sidecar mismatch")

    report = _load_json(paths["streaming_baseline_report.json"])
    report_derived = report.get("derived_frozen_config")
    if (
        report.get("manifest") != "run_manifest.json"
        or report.get("artifact_checksum_index") != "artifact_checksums.json"
        or report.get("config_sha256") != base_hash
        or report.get("config_and_code_unchanged_through_evaluation") is not True
        or report.get("rehearsal_changed_parameters") is not False
        or report.get("synthetic_training_rows") != 0
        or report.get("gate_status") != "evidence_only_pending_step_7_review"
        or not isinstance(report_derived, dict)
        or report_derived.get("file_sha256") != derived_file_hash
        or report_derived.get("payload_sha256") != derived_hash
    ):
        raise CusumTruthV2ContractError("Step-5 completion/report chain mismatch")

    natural = _load_json(paths["natural_burden_by_role.json"])
    if report.get("natural_burden") != natural:
        raise CusumTruthV2ContractError("Step-5 natural-burden report copy differs")

    cusum = derived.get("cusum")
    if not isinstance(cusum, dict):
        raise CusumTruthV2ContractError("Step-5 derived config lacks CUSUM")
    selected_h = float(cusum.get("selected_h", np.nan))
    selected = cusum.get("selected_detector")
    axis = cusum.get("axis_coverage")
    if (
        selected_h != 1.0
        or cusum.get("gate_eligible") is not True
        or not isinstance(selected, dict)
        or not isinstance(axis, dict)
        or axis.get("status") != "complete"
        or axis.get("active_axis_count") != 2
    ):
        raise CusumTruthV2ContractError("Frozen Step-5 CUSUM is not the approved h=1 four-state detector")
    if (
        base_config.get("cusum", {}).get("synthetic_selection_forbidden") is not True
        or derived.get("later_role_feedback_forbidden") is not True
        or derived.get("frozen_before_roles") != ["development", "rehearsal"]
    ):
        raise CusumTruthV2ContractError("Step-5 no-post-hoc/no-synthetic-selection guards differ")
    if (
        selected.get("mad_zero_policy") != "exclude"
        or selected.get("excluded_channels") != {}
        or selected.get("state_count") != 4
        or selected.get("axis_coverage_status") != "complete_two_axis"
        or selected.get("alarm_comparator") != ">"
        or selected.get("config", {}).get("channels") != list(VECTOR_RESIDUAL_FEATURES)
        or float(selected.get("config", {}).get("threshold_h", np.nan)) != selected_h
    ):
        raise CusumTruthV2ContractError("Frozen Step-5 detector policy/config mismatch")
    calibration = selected.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != set(VECTOR_RESIDUAL_FEATURES):
        raise CusumTruthV2ContractError("Frozen Step-5 detector axes differ")
    for channel, values in calibration.items():
        mad = float(values.get("mad", np.nan))
        if not np.isfinite(mad) or mad <= 0.0:
            raise CusumTruthV2ContractError(f"Frozen channel {channel} has invalid MAD")

    frozen_code = base_config.get("frozen_code_sha256")
    if not isinstance(frozen_code, dict):
        raise CusumTruthV2ContractError("Step-5 base config lacks code hashes")
    for relative in FROZEN_SCORING_CODE:
        current = sha256_file(root / relative)
        if frozen_code.get(relative) != current:
            raise CusumTruthV2ContractError(
                f"Frozen scoring dependency bytes changed since Step 5: {relative}"
            )

    detector = VectorPageCUSUM.from_dict(selected)
    if detector.to_dict() != selected:
        raise CusumTruthV2ContractError("VectorPageCUSUM serialization is not an exact round trip")
    burden_panel = _natural_burden_panel(natural, selected_h=selected_h)
    validation_day = str(burden_panel["validation"]["day"])
    validation_flights_reported = burden_panel["validation"]["cusum_flight_unit"].get(
        "n_input_flights"
    )
    if validation_flights_reported != len(validation_ids):
        raise CusumTruthV2ContractError("Step-5 validation split and burden flight counts differ")
    return FrozenStep5Bundle(
        detector=detector,
        selected_detector=selected,
        selected_h=selected_h,
        burden_panel=burden_panel,
        input_records=input_records,
        base_config_sha256=str(base_hash),
        derived_payload_sha256=str(derived_hash),
        validation_day=validation_day,
        validation_flight_ids=validation_ids,
    )


def _required_columns(path: Path, columns: Sequence[str]) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(columns) - available
    if missing:
        raise CusumTruthV2ContractError(f"{path}: missing columns {sorted(missing)}")
    return list(columns)


def iter_parquet_flights(path: Path, columns: Sequence[str]) -> Iterator[pd.DataFrame]:
    """Yield complete flights retaining at most one row group plus one flight."""

    parquet = pq.ParquetFile(path)
    selected = _required_columns(path, columns)
    pending: pd.DataFrame | None = None
    seen: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=selected).to_pandas()
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
            pending = None
        if frame.empty:
            continue
        if frame["flight_id"].isna().any():
            raise CusumTruthV2ContractError(f"{path}: null flight_id")
        block = frame["flight_id"].ne(frame["flight_id"].shift()).cumsum()
        groups = list(frame.groupby(block, sort=False))
        pending = groups[-1][1].reset_index(drop=True)
        for _, flight in groups[:-1]:
            key = str(flight["flight_id"].iloc[0])
            if key in seen:
                raise CusumTruthV2ContractError(f"{path}: non-contiguous flight {key}")
            seen.add(key)
            yield flight.reset_index(drop=True)
    if pending is not None:
        key = str(pending["flight_id"].iloc[0])
        if key in seen:
            raise CusumTruthV2ContractError(f"{path}: repeated flight at EOF {key}")
        yield pending.reset_index(drop=True)


def _validate_flight_order(flight: pd.DataFrame, *, path: Path) -> str:
    ids = flight["flight_id"].astype(str).unique()
    if len(ids) != 1:
        raise CusumTruthV2ContractError(f"{path}: yielded flight has multiple IDs")
    times = pd.to_numeric(flight["timestamp_utc"], errors="coerce").to_numpy(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0.0):
        raise CusumTruthV2ContractError(f"{path}: timestamps are invalid or out of source order")
    return str(ids[0])


def iter_exact_paired_flights(
    clean_path: Path,
    corrupt_path: Path,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    clean_iter = iter_parquet_flights(clean_path, PAIR_COLUMNS)
    corrupt_iter = iter_parquet_flights(corrupt_path, SCORING_COLUMNS)
    marker = object()
    for clean, corrupt in zip_longest(clean_iter, corrupt_iter, fillvalue=marker):
        if clean is marker or corrupt is marker:
            raise CusumTruthV2ContractError("Paired files have different flight counts")
        assert isinstance(clean, pd.DataFrame) and isinstance(corrupt, pd.DataFrame)
        clean_id = _validate_flight_order(clean, path=clean_path)
        corrupt_id = _validate_flight_order(corrupt, path=corrupt_path)
        if clean_id != corrupt_id or len(clean) != len(corrupt):
            raise CusumTruthV2ContractError("Paired flight ID/row count differs")
        clean_time = clean["timestamp_utc"].to_numpy(dtype=float)
        corrupt_time = corrupt["timestamp_utc"].to_numpy(dtype=float)
        if not np.array_equal(clean_time, corrupt_time):
            raise CusumTruthV2ContractError(f"Paired timestamps differ for {clean_id}")
        yield clean, corrupt


def _score_flight(detector: VectorPageCUSUM, flight: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_table(flight)
    time = pd.to_numeric(features["timestamp_utc"], errors="coerce")
    dt = time - time.shift(1)
    transition_valid = (
        features["flight_id"].eq(features["flight_id"].shift(1))
        & dt.gt(0.0)
        & dt.le(detector.config.max_gap_s)
    ).fillna(False)
    features.loc[~transition_valid, list(VECTOR_RESIDUAL_FEATURES)] = np.nan
    return detector.score_rows(features)


@dataclass
class BoundedScoreReservoir:
    """Deterministic bottom-k uniform score sample with a hard capacity."""

    capacity: int
    seed: int
    purpose: str
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    priorities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    finite_seen: int = 0
    _offsets: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("sample capacity must be positive")

    def add(self, values: Iterable[float], *, source_key: str) -> None:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        self.finite_seen += int(len(array))
        if not len(array):
            return
        offset = self._offsets.get(source_key, 0)
        local_seed = int.from_bytes(
            hashlib.sha256(
                f"{self.seed}\0{self.purpose}\0{source_key}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        generator = np.random.PCG64(local_seed)
        if offset:
            generator.advance(offset)
        priority = np.random.Generator(generator).random(len(array))
        self._offsets[source_key] = offset + len(array)

        if len(self.priorities) == self.capacity:
            keep = priority < float(self.priorities.max())
            array, priority = array[keep], priority[keep]
            if not len(array):
                return
        scores = np.concatenate([self.scores, array])
        priorities = np.concatenate([self.priorities, priority])
        if len(scores) > self.capacity:
            indices = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            scores, priorities = scores[indices], priorities[indices]
        order = np.argsort(priorities, kind="stable")
        self.scores, self.priorities = scores[order], priorities[order]


def sampled_binary_diagnostics(
    negative: BoundedScoreReservoir,
    positive: BoundedScoreReservoir,
) -> dict[str, Any]:
    """Class-stratified sampled AUROC/AUPRC with population weighting."""

    n_neg, n_pos = negative.finite_seen, positive.finite_seen
    sample_neg, sample_pos = len(negative.scores), len(positive.scores)
    result: dict[str, Any] = {
        "unit": "row",
        "status": "diagnostic_bounded_uniform_class_stratified_sample",
        "never_used_for_threshold_or_selection": True,
        "negative_source": "single_unmodified_clean_reference_cusum_evaluable_rows",
        "positive_source": "corrupt_observable_changed_and_truth_evaluable_and_cusum_evaluable_rows",
        "corrupt_unchanged_rows_used_as_negatives": False,
        "population_n_negative": n_neg,
        "population_n_positive": n_pos,
        "sample_n_negative": sample_neg,
        "sample_n_positive": sample_pos,
        "sample_capacity_per_class": negative.capacity,
        "sample_seed": negative.seed,
        "auroc_estimate": None,
        "auprc_estimate_population_weighted": None,
    }
    if not n_neg or not n_pos or not sample_neg or not sample_pos:
        return result
    labels = np.concatenate(
        [np.zeros(sample_neg, dtype=np.int8), np.ones(sample_pos, dtype=np.int8)]
    )
    scores = np.concatenate([negative.scores, positive.scores])
    weights = np.concatenate(
        [
            np.full(sample_neg, n_neg / sample_neg, dtype=float),
            np.full(sample_pos, n_pos / sample_pos, dtype=float),
        ]
    )
    result["auroc_estimate"] = float(roc_auc_score(labels, scores, sample_weight=weights))
    result["auprc_estimate_population_weighted"] = float(
        average_precision_score(labels, scores, sample_weight=weights)
    )
    return result


def _validate_clean_truth(flight: pd.DataFrame, *, path: Path) -> None:
    if (
        flight["event_id"].notna().any()
        or flight["injection_active"].fillna(False).astype(bool).any()
        or flight["observable_changed"].fillna(False).astype(bool).any()
        or not flight["evaluable_truth"].fillna(False).astype(bool).all()
    ):
        raise CusumTruthV2ContractError(f"{path}: clean truth-v2 contract violated")


def score_clean_reference(
    clean_path: Path,
    detector: VectorPageCUSUM,
    reservoir: BoundedScoreReservoir,
) -> tuple[dict[str, Any], list[str]]:
    counts = {"n_rows": 0, "n_flights": 0, "n_cusum_evaluable_rows": 0, "n_alarm_rows": 0}
    flight_ids: list[str] = []
    for flight in iter_parquet_flights(clean_path, SCORING_COLUMNS):
        flight_id = _validate_flight_order(flight, path=clean_path)
        _validate_clean_truth(flight, path=clean_path)
        scored = _score_flight(detector, flight)
        evaluable = scored["cusum_evaluable"].to_numpy(dtype=bool)
        alarm = scored["cusum_joint_alarm"].to_numpy(dtype=bool)
        score = scored["cusum_joint_score"].to_numpy(dtype=float)
        reservoir.add(score[evaluable], source_key="clean")
        counts["n_rows"] += len(flight)
        counts["n_flights"] += 1
        counts["n_cusum_evaluable_rows"] += int(evaluable.sum())
        counts["n_alarm_rows"] += int(alarm.sum())
        flight_ids.append(flight_id)
    if len(flight_ids) != len(set(flight_ids)):
        raise CusumTruthV2ContractError("Clean reference has duplicate flight IDs")
    if reservoir.finite_seen != counts["n_cusum_evaluable_rows"]:
        raise CusumTruthV2ContractError("Clean score reservoir denominator differs")
    counts["alarm_row_fraction_among_cusum_evaluable"] = (
        counts["n_alarm_rows"] / counts["n_cusum_evaluable_rows"]
        if counts["n_cusum_evaluable_rows"]
        else None
    )
    return counts, flight_ids


def _event_rows_for_flight(
    recipe: str,
    flight: pd.DataFrame,
    scored: pd.DataFrame,
) -> list[dict[str, Any]]:
    events = truth_event_table(flight)
    if events.empty:
        raise CusumTruthV2ContractError(f"{recipe}: corrupt flight lacks an event")
    times = flight["timestamp_utc"].to_numpy(dtype=float)
    alarms = scored["cusum_joint_alarm"].to_numpy(dtype=bool)
    cusum_evaluable = scored["cusum_evaluable"].to_numpy(dtype=bool)
    changed = flight["observable_changed"].fillna(False).to_numpy(dtype=bool)
    truth_evaluable = flight["evaluable_truth"].fillna(False).to_numpy(dtype=bool)
    event_ids = flight["event_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for event in events.to_dict(orient="records"):
        event_id = str(event["event_id"])
        if str(event["event_type"]) != recipe:
            raise CusumTruthV2ContractError(f"{recipe}: event_type differs from recipe")
        event_mask = event_ids == event_id
        positive = event_mask & changed & truth_evaluable & cusum_evaluable
        n_positive = int(positive.sum())
        n_alerted_positive = int((positive & alarms).sum())
        detected: bool | None = None
        delay: float | None = None
        if bool(event["observable_eligible"]):
            onset, end = float(event["observable_onset"]), float(event["event_end"])
            hits = times[alarms & (times >= onset) & (times <= end)]
            detected = bool(len(hits))
            delay = float(hits[0] - onset) if len(hits) else None
        rows.append(
            {
                "recipe": recipe,
                **event,
                "cusum_scoreable_observable_changed_rows": n_positive,
                "cusum_alerted_observable_changed_rows": n_alerted_positive,
                "active_row_coverage": (
                    n_alerted_positive / n_positive if n_positive else None
                ),
                "detected": detected,
                "first_alarm_delay_s": delay,
                "alarm_emission_time": "row_timestamp_utc",
            }
        )
    return rows


def summarize_event_rows(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in event_rows if bool(row["observable_eligible"])]
    detected = [row for row in eligible if row["detected"] is True]
    delays = [float(row["first_alarm_delay_s"]) for row in detected]
    coverage_values = [
        float(row["active_row_coverage"])
        for row in eligible
        if row["active_row_coverage"] is not None
    ]
    n_active = sum(int(row["cusum_scoreable_observable_changed_rows"]) for row in eligible)
    n_alerted = sum(int(row["cusum_alerted_observable_changed_rows"]) for row in eligible)
    return {
        "event_detection": {
            "unit": "observable_eligible_event",
            "alarm_emission_time": "row_timestamp_utc",
            "n_events": len(eligible),
            "n_detected_events": len(detected),
            "event_recall": len(detected) / len(eligible) if eligible else None,
            "first_alarm_delay_s": {
                "n_detected": len(delays),
                "median": float(np.median(delays)) if delays else None,
                "p95": float(np.quantile(delays, 0.95)) if delays else None,
            },
            "point_adjustment": False,
        },
        "active_row_coverage": {
            "unit": "observable_changed_truth_evaluable_cusum_evaluable_row",
            "n_events_with_scoreable_changed_rows": len(coverage_values),
            "n_scoreable_changed_rows": n_active,
            "n_alerted_changed_rows": n_alerted,
            "micro_fraction": n_alerted / n_active if n_active else None,
            "macro_mean_fraction": float(np.mean(coverage_values)) if coverage_values else None,
            "point_adjustment": False,
        },
    }


def evaluate_corrupt_recipe(
    *,
    recipe: str,
    clean_path: Path,
    corrupt_path: Path,
    detector: VectorPageCUSUM,
    positive_reservoir: BoundedScoreReservoir,
    pooled_positive_reservoir: BoundedScoreReservoir | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = {
        "n_rows": 0,
        "n_flights": 0,
        "n_truth_evaluable_observable_changed_rows": 0,
        "n_cusum_scoreable_positive_rows": 0,
        "n_alerted_positive_rows": 0,
        "n_corrupt_q0_timeline_rows": 0,
        "n_corrupt_q0_timeline_alarm_rows": 0,
    }
    event_rows: list[dict[str, Any]] = []
    for _, flight in iter_exact_paired_flights(clean_path, corrupt_path):
        scored = _score_flight(detector, flight)
        truth_evaluable = flight["evaluable_truth"].fillna(False).to_numpy(dtype=bool)
        changed = flight["observable_changed"].fillna(False).to_numpy(dtype=bool)
        cusum_evaluable = scored["cusum_evaluable"].to_numpy(dtype=bool)
        alarms = scored["cusum_joint_alarm"].to_numpy(dtype=bool)
        scores = scored["cusum_joint_score"].to_numpy(dtype=float)
        observable = truth_evaluable & changed
        positive = observable & cusum_evaluable
        corrupt_q0 = truth_evaluable & ~changed & cusum_evaluable
        positive_reservoir.add(scores[positive], source_key=recipe)
        if pooled_positive_reservoir is not None:
            pooled_positive_reservoir.add(scores[positive], source_key=recipe)
        counts["n_rows"] += len(flight)
        counts["n_flights"] += 1
        counts["n_truth_evaluable_observable_changed_rows"] += int(observable.sum())
        counts["n_cusum_scoreable_positive_rows"] += int(positive.sum())
        counts["n_alerted_positive_rows"] += int((positive & alarms).sum())
        counts["n_corrupt_q0_timeline_rows"] += int(corrupt_q0.sum())
        counts["n_corrupt_q0_timeline_alarm_rows"] += int((corrupt_q0 & alarms).sum())
        event_rows.extend(_event_rows_for_flight(recipe, flight, scored))
    if positive_reservoir.finite_seen != counts["n_cusum_scoreable_positive_rows"]:
        raise CusumTruthV2ContractError(f"{recipe}: positive score denominator differs")

    event_frame = pd.DataFrame(event_rows)
    denominators = event_observability_denominators(event_frame)
    event_metrics = summarize_event_rows(event_rows)
    positive_n = counts["n_cusum_scoreable_positive_rows"]
    q0_n = counts["n_corrupt_q0_timeline_rows"]
    return {
        "positive_row_recall": {
            "unit": "observable_changed_truth_evaluable_cusum_evaluable_row",
            "n_positive_rows": positive_n,
            "n_alerted_positive_rows": counts["n_alerted_positive_rows"],
            "recall": counts["n_alerted_positive_rows"] / positive_n if positive_n else None,
        },
        "corrupt_q0_timeline_sanity": {
            "unit": "corrupt_truth_evaluable_unchanged_cusum_evaluable_row",
            "included_as_auc_negative": False,
            "n_rows": q0_n,
            "n_alarm_rows": counts["n_corrupt_q0_timeline_alarm_rows"],
            "alarm_fraction": counts["n_corrupt_q0_timeline_alarm_rows"] / q0_n if q0_n else None,
        },
        "event_observability_denominators": denominators,
        **event_metrics,
        "audit_counts": counts,
    }, event_rows


def validate_corpus_manifest(
    corpus_dir: Path,
    recipes: Sequence[str],
) -> tuple[Path, dict[str, Path], list[dict[str, Any]]]:
    directory = _assert_unsealed_path(corpus_dir, label="truth-v2 corpus")
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != "adsb_synthetic_truth_v2"
        or manifest.get("synthetic_never_training") is not True
    ):
        raise CusumTruthV2ContractError("Corpus is not protected truth-v2 evaluation data")
    unknown = sorted(set(recipes) - set(PHYSICS_BREAK_RECIPES))
    if unknown or not recipes:
        raise CusumTruthV2ContractError(f"Unknown or empty recipes: {unknown}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise CusumTruthV2ContractError("Corpus manifest lacks outputs")
    by_recipe = {str(record.get("recipe")): record for record in outputs}
    selected = ["clean", *recipes]
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for recipe in selected:
        record = by_recipe.get(recipe)
        if not isinstance(record, dict):
            raise CusumTruthV2ContractError(f"Corpus manifest lacks {recipe}")
        path = (directory / str(record.get("path"))).resolve(strict=True)
        if path.parent != directory or path.suffix.lower() != ".parquet":
            raise CusumTruthV2ContractError(f"Corpus output path escapes directory: {recipe}")
        parquet = pq.ParquetFile(path)
        if (
            int(record.get("bytes", -1)) != path.stat().st_size
            or int(record.get("footer_rows", -1)) != parquet.metadata.num_rows
            or int(record.get("n_rows", -1)) != parquet.metadata.num_rows
            or record.get("sha256") != sha256_file(path)
        ):
            raise CusumTruthV2ContractError(f"Corpus output hash/footer changed: {recipe}")
        paths[recipe] = path
        records.append(
            {
                "recipe": recipe,
                "path": path.name,
                "bytes": path.stat().st_size,
                "footer_rows": parquet.metadata.num_rows,
                "sha256": record["sha256"],
            }
        )
    return manifest_path, paths, records


def _write_output_checksums(run_dir: Path) -> Path:
    names = ("run_manifest.json", "summary.json", "event_table.parquet")
    records = {
        name: {"bytes": (run_dir / name).stat().st_size, "sha256": sha256_file(run_dir / name)}
        for name in names
    }
    path = run_dir / "artifact_checksums.json"
    write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "status": "complete",
            "algorithm": "sha256",
            "self_excluded": True,
            "files": records,
        },
    )
    return path


def _assert_inputs_and_code_unchanged(
    manifest_path: Path,
    *,
    repo_root: Path,
    code_sha256: Mapping[str, str],
) -> None:
    """Re-measure every input and scoring byte after the last evaluation read."""

    manifest = _load_json(manifest_path)
    for frozen in manifest.get("inputs", []):
        recorded_path = Path(str(frozen["path"]))
        path = recorded_path if recorded_path.is_absolute() else repo_root / recorded_path
        current = inspect_input_file(path, role=str(frozen["role"]), repo_root=repo_root)
        if current != frozen:
            raise CusumTruthV2ContractError(
                f"Evaluation input changed after manifest creation: {frozen['path']}"
            )
    for relative, expected in code_sha256.items():
        if sha256_file(repo_root / relative) != expected:
            raise CusumTruthV2ContractError(
                f"Evaluation/scoring code changed during run: {relative}"
            )


def run_evaluation(
    *,
    repo_root: Path,
    step5_dir: Path,
    corpus_dir: Path,
    run_dir: Path,
    recipes: Sequence[str] = tuple(PHYSICS_BREAK_RECIPES),
    sample_capacity_per_class: int = DEFAULT_SAMPLE_CAPACITY_PER_CLASS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> Path:
    """Create an immutable corrected CUSUM evaluation run.

    The caller controls only output location and diagnostic sample capacity.
    Detector parameters, h, calibration, fusion, and threshold search are not
    function arguments and therefore cannot be changed here.
    """

    root = repo_root.resolve(strict=True)
    destination = run_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Evaluation run already exists: {destination}")
    forbidden_destination = {part.lower() for part in destination.parts} & {
        "archive",
        "downloads",
        "raw",
    }
    if forbidden_destination:
        raise CusumTruthV2ContractError(
            f"Evaluation output uses forbidden path components: {sorted(forbidden_destination)}"
        )
    bundle = load_frozen_step5_bundle(step5_dir, repo_root=root)
    corpus_manifest, corpus_paths, corpus_records = validate_corpus_manifest(
        corpus_dir, recipes
    )
    clean_path = corpus_paths["clean"]

    # Exact IDs are read before manifest creation; synthetic IDs can only have
    # the protected test role in this evaluation.
    flight_ids = [
        _validate_flight_order(flight, path=clean_path)
        for flight in iter_parquet_flights(clean_path, PAIR_COLUMNS)
    ]
    if len(flight_ids) != len(set(flight_ids)):
        raise CusumTruthV2ContractError("Clean corpus flight IDs are not unique")
    expected_validation_ids = tuple(
        sorted(f"{bundle.validation_day}:{flight_id}" for flight_id in flight_ids)
    )
    if expected_validation_ids != bundle.validation_flight_ids:
        raise CusumTruthV2ContractError(
            "Truth-v2 clean flights do not exactly match Step-5 natural validation flights"
        )

    step5 = Path(step5_dir).resolve(strict=True)
    inputs = [
        *[InputSpec(step5 / name, "reference") for name in STEP5_REQUIRED_FILES],
        InputSpec(corpus_manifest, "reference"),
        InputSpec(clean_path, "natural_evaluation"),
        *[InputSpec(corpus_paths[recipe], "synthetic_evaluation") for recipe in recipes],
    ]
    evaluation_code = {
        relative: sha256_file(root / relative)
        for relative in (
            "adsb/cusum_truth_v2_eval.py",
            "scripts/adsb_evaluate_cusum_truth_v2.py",
            *FROZEN_SCORING_CODE,
        )
    }
    config = {
        "step": "frozen_step5_cusum_corrected_truth_v2_evaluation",
        "detector_source": "Step-5 derived_frozen_config selected_detector",
        "detector_load": "VectorPageCUSUM.from_dict exact round_trip",
        "fit_performed": False,
        "calibration_performed": False,
        "threshold_sweep_performed": False,
        "fusion_performed": False,
        "selected_h": bundle.selected_h,
        "alarm_comparator": ">",
        "alarm_emission_time": "row_timestamp_utc",
        "truth_contract": {
            "architecture": "row_level_causal_emission",
            "positive": "observable_changed AND evaluable_truth AND cusum_evaluable",
            "negative": "single unmodified clean reference AND cusum_evaluable",
            "corrupt_unchanged_role": "timeline_sanity_only_not_auc_negative",
            "event_denominator": "observable_eligible_event",
            "active_coverage": "observable_changed truth-evaluable CUSUM-evaluable rows",
        },
        "diagnostic_auc_sampling": {
            "algorithm": "deterministic_uniform_bottom_k_separate_by_class",
            "capacity_per_class": sample_capacity_per_class,
            "seed": sample_seed,
            "auprc_population_weighting": True,
            "never_selection": True,
        },
        "synthetic_usage": "evaluation_only_never_fit_calibration_or_threshold_selection",
        "recipes": list(recipes),
        "stealthy_ramp_focus": "position_ramp_stealthy",
        "frozen_step5": {
            "base_config_sha256": bundle.base_config_sha256,
            "derived_payload_sha256": bundle.derived_payload_sha256,
            "selected_detector": bundle.selected_detector,
            "validation_day": bundle.validation_day,
            "validation_flight_ids_sha256": sha256_json(list(bundle.validation_flight_ids)),
            "validation_matches_truth_v2_clean_exactly": True,
        },
        "corpus_outputs": corpus_records,
        "evaluation_code_sha256": evaluation_code,
    }
    manifest_path = create_immutable_run_manifest(
        run_dir=destination,
        repo_root=root,
        inputs=inputs,
        splits={"test": flight_ids},
        split_algorithm="paired_truth_v2_all_flights_test_only_v1",
        split_seed=None,
        synthetic_flight_ids=flight_ids,
        config=config,
    )
    destination = manifest_path.parent

    negative = BoundedScoreReservoir(
        sample_capacity_per_class, sample_seed, "single_clean_negative"
    )
    clean_summary, scored_clean_ids = score_clean_reference(
        clean_path, bundle.detector, negative
    )
    if scored_clean_ids != flight_ids:
        raise CusumTruthV2ContractError("Clean scoring order changed after manifest")

    pooled_positive = BoundedScoreReservoir(
        sample_capacity_per_class, sample_seed, "pooled_corrupt_positive"
    )
    per_recipe: dict[str, Any] = {}
    all_event_rows: list[dict[str, Any]] = []
    pooled_positive_rows = pooled_alerted_positive_rows = 0
    for recipe in recipes:
        positive = BoundedScoreReservoir(
            sample_capacity_per_class, sample_seed, f"{recipe}_positive"
        )
        synthetic, event_rows = evaluate_corrupt_recipe(
            recipe=recipe,
            clean_path=clean_path,
            corrupt_path=corpus_paths[recipe],
            detector=bundle.detector,
            positive_reservoir=positive,
            pooled_positive_reservoir=pooled_positive,
        )
        pooled_positive_rows += int(synthetic["positive_row_recall"]["n_positive_rows"])
        pooled_alerted_positive_rows += int(
            synthetic["positive_row_recall"]["n_alerted_positive_rows"]
        )
        per_recipe[recipe] = {
            "synthetic_detection": synthetic,
            "diagnostic_sampled_auc_auprc": sampled_binary_diagnostics(negative, positive),
            "paired_natural_cusum_burden": bundle.burden_panel,
        }
        all_event_rows.extend(event_rows)

    if pooled_positive.finite_seen != pooled_positive_rows:
        raise CusumTruthV2ContractError("Pooled positive score denominator differs")
    pooled_event = summarize_event_rows(all_event_rows)
    pooled = {
        "synthetic_detection": {
            "positive_row_recall": {
                "unit": "observable_changed_truth_evaluable_cusum_evaluable_row",
                "n_positive_rows": pooled_positive_rows,
                "n_alerted_positive_rows": pooled_alerted_positive_rows,
                "recall": (
                    pooled_alerted_positive_rows / pooled_positive_rows
                    if pooled_positive_rows
                    else None
                ),
            },
            **pooled_event,
        },
        "diagnostic_sampled_auc_auprc": sampled_binary_diagnostics(
            negative, pooled_positive
        ),
        "negative_pool_contract": "single clean pool, never duplicated by recipe",
        "paired_natural_cusum_burden": bundle.burden_panel,
    }

    _assert_inputs_and_code_unchanged(
        manifest_path,
        repo_root=root,
        code_sha256=evaluation_code,
    )

    event_frame = pd.DataFrame(all_event_rows)
    for column in ("recipe", "event_id", "event_type", "flight_id"):
        event_frame[column] = event_frame[column].astype("string")
    event_frame["detected"] = event_frame["detected"].astype("boolean")
    event_path = destination / "event_table.parquet"
    if event_path.exists():
        raise FileExistsError(event_path)
    event_frame.to_parquet(event_path, index=False)

    report = {
        "schema_version": "adsb_frozen_step5_cusum_truth_v2_eval_v1",
        "status": "complete_evaluation_only",
        "run_manifest": manifest_path.name,
        "frozen_detector": {
            "selected_h": bundle.selected_h,
            "alarm_comparator": ">",
            "fit_performed_in_this_run": False,
            "calibration_performed_in_this_run": False,
            "threshold_sweep_performed_in_this_run": False,
            "fusion_performed_in_this_run": False,
            "serialized_detector": bundle.selected_detector,
        },
        "truth_architecture": config["truth_contract"],
        "clean_reference": clean_summary,
        "per_recipe": per_recipe,
        "pooled_all_recipes_vs_single_clean_reference": pooled,
        "stealthy_ramp_focus": {
            "recipe": "position_ramp_stealthy",
            "available": "position_ramp_stealthy" in per_recipe,
            "result": per_recipe.get("position_ramp_stealthy"),
        },
        "natural_burden_pairing_contract": (
            "Every synthetic detection block carries Step-5 validation burden "
            "plus development/rehearsal temporal panel"
        ),
        "event_table": {
            "path": event_path.name,
            "footer_rows": len(event_frame),
            "sha256": sha256_file(event_path),
            "unit": "one row per declared flight event",
        },
    }
    summary_path = destination / "summary.json"
    write_json_exclusive(summary_path, report)
    _write_output_checksums(destination)
    return summary_path
