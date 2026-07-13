"""Report S2 natural reason/episode burden over the three already-open days.

The command consumes the immutable Step-5 manifest as the transitive input
contract.  It does not discover raw archives, Downloads, or sealed holdouts and
does not reparse historical data.  Existing Silver v1 rows lack update-age
metadata and are therefore reported as ``freshness_unknown`` by design.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.run_manifest import (  # noqa: E402
    InputSpec,
    arrow_schema_sha256,
    create_immutable_run_manifest,
    sha256_file,
    sha256_json,
)
from adsb.s2_streaming import (  # noqa: E402
    FRESHNESS_COLUMNS,
    merge_s2_summaries,
    summarize_s2_part,
)
from adsb.segmentation import segment_flights  # noqa: E402

FIT_DAY = "2026-02-28"
DEVELOPMENT_DAY = "2026-03-01"
REHEARSAL_DAY = "2026-03-16"
OPEN_DAYS = (FIT_DAY, DEVELOPMENT_DAY, REHEARSAL_DAY)
EXPECTED_PARTS_BY_DAY = {
    FIT_DAY: 237,
    DEVELOPMENT_DAY: 216,
    REHEARSAL_DAY: 185,
}
EXPECTED_ROWS_BY_DAY = {
    FIT_DAY: 88_762_032,
    DEVELOPMENT_DAY: 85_991_023,
    REHEARSAL_DAY: 81_401_954,
}
DAY_INPUT_ROLE = {
    FIT_DAY: "fit",
    DEVELOPMENT_DAY: "development",
    REHEARSAL_DAY: "rehearsal",
}
SEGMENT_GAP_S = 1800.0
SCOREABLE_MAX_GAP_S = 60.0
MESSAGE_GAP_THRESHOLD_S = 60.0
FRESHNESS_MAX_AGE_S = 60.0
SILVER_RELATIVE_DIR = Path("data/objectstore/silver/adsblol_historical")
SOURCE_DAY_RE = re.compile(r"v(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})")
STEP5_COMPLETION_FILES = (
    "streaming_baseline_report.json",
    "derived_frozen_config.json",
    "derived_frozen_config.sha256",
    "artifact_checksums.json",
)
STEP6_IMPLEMENTATION_PATHS = (
    Path("adsb/run_manifest.py"),
    Path("adsb/s2.py"),
    Path("adsb/s2_streaming.py"),
    Path("adsb/segmentation.py"),
    Path("src/silver/parse_adsblol_historical.py"),
    Path("scripts/adsb_report_s2_natural_burden.py"),
)

S2_COLUMNS = (
    "_source_file", "source_id", "timestamp_utc", "squawk", "emergency", "nic",
    "nac_p", "sil", "adsb_version", "alt", "alt_geom_m", "on_ground",
)


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_safe_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _implementation_sha256(repo_root: Path) -> dict[str, str]:
    """Hash every Python module executed by the Step-6 reporting path."""

    return {
        path.as_posix(): sha256_file(repo_root / path)
        for path in STEP6_IMPLEMENTATION_PATHS
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_repo_file(path: Path, repo_root: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    lowered_candidate = {part.casefold() for part in candidate.parts}
    if "downloads" in lowered_candidate or "archive" in lowered_candidate:
        raise ValueError(f"{label} cannot be under Downloads/archive: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not _is_within(resolved, repo_root):
        raise ValueError(f"{label} must be a regular file inside the repository: {resolved}")
    lowered = {part.casefold() for part in resolved.parts}
    if "downloads" in lowered or "archive" in lowered:
        raise ValueError(f"{label} cannot be under Downloads/archive: {resolved}")
    if resolved.suffix.casefold() in {".tar", ".tgz", ".gz"}:
        raise ValueError(f"{label} cannot be a raw archive: {resolved}")
    return resolved


def source_day(source_file: object) -> str:
    value = str(source_file)
    match = SOURCE_DAY_RE.search(value)
    if match is None:
        raise ValueError(f"Cannot derive an open day from _source_file={value!r}")
    day = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    if day not in OPEN_DAYS:
        raise ValueError(f"Out-of-contract Silver day {day!r}")
    return day


def _validate_base_contract(base: Mapping[str, Any]) -> None:
    if base.get("manifest_schema_version") != 1:
        raise ValueError("Unsupported baseline manifest schema")
    config = base.get("config")
    if not isinstance(config, dict) or config.get("value", {}).get("step") != 5:
        raise ValueError("S2 report requires a Step-5 input contract")
    if config.get("sha256") != sha256_json(config["value"]):
        raise ValueError("Step-5 config hash does not match its payload")
    inputs = base.get("inputs")
    if not isinstance(inputs, list) or base.get("input_contract_sha256") != sha256_json(inputs):
        raise ValueError("Step-5 input contract hash does not match its records")

    split_contract = base.get("split_contract")
    if not isinstance(split_contract, dict) or not isinstance(split_contract.get("splits"), dict):
        raise ValueError("Step-5 split contract is missing")
    normalized_splits: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role, record in sorted(split_contract["splits"].items()):
        ids = record.get("flight_ids")
        if not isinstance(ids, list) or ids != sorted(set(ids)):
            raise ValueError(f"Step-5 split {role!r} is not a canonical unique list")
        if record.get("flight_id_count") != len(ids) or record.get(
            "flight_ids_sha256"
        ) != sha256_json(ids):
            raise ValueError(f"Step-5 split record hash/count mismatch: {role}")
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Step-5 split roles overlap: {role}")
        seen.update(ids)
        normalized_splits[role] = ids
    expected_roles = {"fit", "calibration", "validation", "development", "rehearsal"}
    if set(normalized_splits) != expected_roles:
        raise ValueError(
            f"Step-5 split roles differ from the frozen contract: {sorted(normalized_splits)}"
        )
    split_payload = {
        "algorithm": split_contract.get("algorithm"),
        "seed": split_contract.get("seed"),
        "splits": normalized_splits,
    }
    if split_contract.get("contract_sha256") != sha256_json(split_payload):
        raise ValueError("Step-5 split contract hash mismatch")

    guard = base.get("synthetic_guard")
    if not isinstance(guard, dict) or guard.get("status") != "passed":
        raise ValueError("Step-5 synthetic guard did not pass")
    excluded = guard.get("excluded_flight_ids")
    if not isinstance(excluded, list) or excluded != sorted(set(excluded)):
        raise ValueError("Step-5 synthetic exclusion IDs are not canonical")
    if guard.get("excluded_flight_id_count") != len(excluded) or guard.get(
        "excluded_flight_ids_sha256"
    ) != sha256_json(excluded):
        raise ValueError("Step-5 synthetic exclusion hash/count mismatch")
    protected = set(normalized_splits.get("fit", ())) | set(
        normalized_splits.get("calibration", ())
    )
    if protected.intersection(excluded) or guard.get("overlap_count") != 0:
        raise ValueError("Synthetic-source flights overlap Step-5 fit/calibration")
    if set(normalized_splits["validation"]) != set(excluded):
        raise ValueError("Step-5 validation split does not exactly contain synthetic-source exclusions")


def _validate_step5_completion(
    manifest_path: Path,
    base: Mapping[str, Any],
) -> tuple[list[Path], dict[str, str]]:
    run_dir = manifest_path.parent
    paths = [run_dir / name for name in STEP5_COMPLETION_FILES]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Step-5 is incomplete; missing {path.name}")

    final_report = _load_json_object(run_dir / "streaming_baseline_report.json")
    if final_report.get("run_id") != base.get("run_id"):
        raise ValueError("Step-5 completion report run_id mismatch")
    if final_report.get("manifest") != manifest_path.name:
        raise ValueError("Step-5 completion report manifest reference mismatch")
    if final_report.get("config_sha256") != base["config"]["sha256"]:
        raise ValueError("Step-5 completion report config hash mismatch")
    if final_report.get("synthetic_training_rows") != 0:
        raise ValueError("Step-5 completion report does not prove zero synthetic training rows")
    if final_report.get("rehearsal_changed_parameters") is not False:
        raise ValueError("Step-5 completion report does not prove a frozen rehearsal")
    if final_report.get("config_and_code_unchanged_through_evaluation") is not True:
        raise ValueError("Step-5 completion report does not prove frozen code/config")
    if final_report.get("synthetic_reference_use") != "flight-ID exclusion only":
        raise ValueError("Unexpected Step-5 synthetic reference use")
    if final_report.get("gate_status") not in {
        "evidence_only_pending_step_7_review",
        "fail_degraded_axis_coverage",
        "fail_no_admissible_cusum_threshold",
    }:
        raise ValueError("Unexpected Step-5 gate status")
    if final_report.get("artifact_checksum_index") != "artifact_checksums.json":
        raise ValueError("Step-5 completion report lacks the checksum-manifest reference")

    derived_path = run_dir / "derived_frozen_config.json"
    sidecar_path = run_dir / "derived_frozen_config.sha256"
    sidecar = sidecar_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", sidecar) or sidecar != sha256_file(derived_path):
        raise ValueError("Step-5 derived config SHA-256 sidecar mismatch")
    derived = _load_json_object(derived_path)
    if derived.get("schema_version") != 1:
        raise ValueError("Unsupported Step-5 derived config schema")
    derived_config = derived.get("derived_config")
    if not isinstance(derived_config, dict) or set(derived) != {
        "schema_version",
        "derived_config",
        "payload_sha256",
    }:
        raise ValueError("Invalid Step-5 derived config envelope")
    if derived.get("payload_sha256") != sha256_json(derived_config):
        raise ValueError("Step-5 derived config payload hash mismatch")
    if derived_config.get("base_config_sha256") != base["config"]["sha256"]:
        raise ValueError("Step-5 derived config base hash mismatch")
    derived_reference = final_report.get("derived_frozen_config")
    if not isinstance(derived_reference, dict) or derived_reference != {
        "path": derived_path.name,
        "payload_sha256": derived["payload_sha256"],
        "file_sha256": sidecar,
        "sidecar": sidecar_path.name,
    }:
        raise ValueError("Step-5 completion report derived-config reference mismatch")

    checksum_path = run_dir / "artifact_checksums.json"
    checksum_manifest = _load_json_object(checksum_path)
    if (
        checksum_manifest.get("schema_version") != 1
        or checksum_manifest.get("algorithm") != "sha256"
        or checksum_manifest.get("self_excluded") is not True
        or not isinstance(checksum_manifest.get("files"), dict)
    ):
        raise ValueError("Invalid Step-5 artifact checksum manifest")
    expected_files = set(checksum_manifest["files"])
    actual_files = {
        item.relative_to(run_dir).as_posix()
        for item in run_dir.rglob("*")
        if item.is_file() and item.resolve() != checksum_path.resolve()
    }
    if expected_files != actual_files:
        raise ValueError("Step-5 artifact checksum coverage is not exact")
    for relative, record in checksum_manifest["files"].items():
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"Unsafe Step-5 artifact path: {relative!r}")
        candidate = (run_dir / raw).resolve(strict=True)
        if not _is_within(candidate, run_dir.resolve()):
            raise ValueError(f"Step-5 artifact escapes its run directory: {relative!r}")
        stat = candidate.stat()
        if not isinstance(record, dict) or record.get("bytes") != stat.st_size:
            raise ValueError(f"Step-5 artifact byte mismatch: {relative}")
        if record.get("sha256") != sha256_file(candidate):
            raise ValueError(f"Step-5 artifact SHA-256 mismatch: {relative}")

    hashes = {path.name: sha256_file(path) for path in paths}
    return [path.resolve() for path in paths], hashes


def _verify_silver_inputs(
    base: Mapping[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], str, str, dict[str, dict[str, Any]]]:
    silver_root = (repo_root / SILVER_RELATIVE_DIR).resolve(strict=True)
    if not silver_root.is_dir() or not _is_within(silver_root, repo_root):
        raise ValueError("The exact repository Silver directory is unavailable")

    selected: list[dict[str, Any]] = []
    for record in base["inputs"]:
        raw = Path(record.get("path", ""))
        candidate = raw if raw.is_absolute() else repo_root / raw
        lowered_candidate = {part.casefold() for part in candidate.parts}
        if "downloads" in lowered_candidate or "archive" in lowered_candidate:
            raise ValueError(f"Step-5 input cannot be under Downloads/archive: {candidate}")
        resolved = candidate.resolve(strict=True)
        if resolved.parent == silver_root:
            selected.append({"path": resolved, "record": record})
    if len(selected) != sum(EXPECTED_PARTS_BY_DAY.values()):
        raise ValueError(
            f"Expected 638 exact Silver inputs in Step-5 manifest, found {len(selected)}"
        )
    if len({item["path"] for item in selected}) != len(selected):
        raise ValueError("Step-5 Silver input paths are duplicated")

    schema_modes: set[str] = set()
    selected_schemas: list[dict[str, dict[str, Any]]] = []
    part_counts: Counter[str] = Counter()
    footer_rows: Counter[str] = Counter()
    base_columns = set(S2_COLUMNS)
    freshness_columns = set(FRESHNESS_COLUMNS)
    for item in selected:
        path = item["path"]
        record = item["record"]
        if record.get("format") != "parquet" or path.suffix.casefold() != ".parquet":
            raise ValueError(f"Silver input is not Parquet: {path}")
        stat = path.stat()
        if record.get("bytes") != stat.st_size or record.get("mtime_ns") != stat.st_mtime_ns:
            raise ValueError(f"Silver stat changed after Step 5: {path}")
        parquet = pq.ParquetFile(path)
        if record.get("footer_rows") != parquet.metadata.num_rows:
            raise ValueError(f"Silver footer row count changed after Step 5: {path}")
        if record.get("schema_sha256") != arrow_schema_sha256(parquet.schema_arrow):
            raise ValueError(f"Silver Arrow schema changed after Step 5: {path}")
        columns = set(parquet.schema_arrow.names)
        missing = base_columns - columns
        if missing:
            raise ValueError(f"Silver input lacks S2 columns {sorted(missing)}: {path}")
        present_freshness = freshness_columns & columns
        if not present_freshness:
            mode = "silver_v1_legacy_freshness_unknown"
        elif present_freshness == freshness_columns:
            mode = "silver_v2_update_age_available"
        else:
            raise ValueError(f"Silver input has a partial freshness schema: {path}")
        schema_modes.add(mode)
        selected_names = list(S2_COLUMNS)
        if mode == "silver_v2_update_age_available":
            selected_names.extend(FRESHNESS_COLUMNS)
        selected_schemas.append(
            {
                name: {
                    "type": str(parquet.schema_arrow.field(name).type),
                    "nullable": parquet.schema_arrow.field(name).nullable,
                }
                for name in selected_names
            }
        )
        role = record.get("role")
        if role not in set(DAY_INPUT_ROLE.values()):
            raise ValueError(f"Unexpected Step-5 role for Silver input: {role!r}")
        day = next(day for day, expected_role in DAY_INPUT_ROLE.items() if expected_role == role)
        item["day_from_role"] = day
        item["schema_mode"] = mode
        part_counts[day] += 1
        footer_rows[day] += int(parquet.metadata.num_rows)
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"Silver SHA-256 changed after Step 5: {path}")

    if len(schema_modes) != 1:
        raise ValueError(f"Mixed Silver freshness schemas are forbidden: {sorted(schema_modes)}")
    if any(schema != selected_schemas[0] for schema in selected_schemas[1:]):
        raise ValueError("Selected S2 Arrow field types/nullability differ across Silver parts")
    if dict(part_counts) != EXPECTED_PARTS_BY_DAY or dict(footer_rows) != EXPECTED_ROWS_BY_DAY:
        raise ValueError(
            f"Step-5 Silver inventory mismatch: parts={dict(part_counts)}, rows={dict(footer_rows)}"
        )
    silver_contract_sha256 = sha256_json(
        [
            item["record"]
            for item in sorted(selected, key=lambda value: str(value["record"]["path"]))
        ]
    )
    return (
        sorted(selected, key=lambda value: str(value["record"]["path"])),
        next(iter(schema_modes)),
        silver_contract_sha256,
        selected_schemas[0],
    )


def _load_base_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    manifest_path = _require_repo_file(path, repo_root, label="Step-5 manifest")
    if manifest_path.name != "run_manifest.json":
        raise ValueError("Step-5 baseline input must be named run_manifest.json")
    base = _load_json_object(manifest_path)
    _validate_base_contract(base)
    completion_paths, completion_hashes = _validate_step5_completion(manifest_path, base)
    silver_inputs, schema_mode, silver_contract_sha256, selected_schema = (
        _verify_silver_inputs(base, repo_root)
    )
    return {
        "manifest_path": manifest_path,
        "base": base,
        "completion_paths": completion_paths,
        "completion_hashes": completion_hashes,
        "silver_inputs": silver_inputs,
        "schema_mode": schema_mode,
        "selected_schema": selected_schema,
        "silver_contract_sha256": silver_contract_sha256,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Non-finite values are forbidden in S2 JSON artifacts")
        return numeric
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    safe = _json_safe(dict(payload))
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(safe, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(*, repo_root: Path, baseline_manifest: Path, run_dir: Path) -> dict:
    root = repo_root.resolve(strict=True)
    destination = run_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(destination)
    if not _is_within(destination, root):
        raise ValueError("Step-6 run directory must stay inside the repository")
    lowered_destination = {part.casefold() for part in destination.parts}
    if "downloads" in lowered_destination or "archive" in lowered_destination:
        raise ValueError("Step-6 run directory cannot be under Downloads/archive")

    contract = _load_base_manifest(baseline_manifest, root)
    base = contract["base"]
    silver_inputs = contract["silver_inputs"]
    split_records = base["split_contract"]["splits"]
    splits = {role: record["flight_ids"] for role, record in split_records.items()}
    synthetic_ids = base["synthetic_guard"]["excluded_flight_ids"]
    implementation_sha256 = _implementation_sha256(root)
    read_columns = list(S2_COLUMNS)
    if contract["schema_mode"] == "silver_v2_update_age_available":
        read_columns.extend(FRESHNESS_COLUMNS)
    config = {
        "step": 6,
        "contract_date": "2026-07-13",
        "base_step5_manifest_sha256": sha256_file(contract["manifest_path"]),
        "base_step5_input_contract_sha256": base["input_contract_sha256"],
        "base_step5_completion_sha256": contract["completion_hashes"],
        "transitive_silver_contract_sha256": contract["silver_contract_sha256"],
        "transitive_silver_verification": (
            "all 638 files matched Step-5 bytes, mtime_ns, SHA-256, footer rows, "
            "and Arrow schema hash before manifest creation"
        ),
        "implementation_sha256": implementation_sha256,
        "read_columns": read_columns,
        "freshness_schema_mode": contract["schema_mode"],
        "selected_s2_arrow_schema": contract["selected_schema"],
        "segmentation_gap_s": SEGMENT_GAP_S,
        "s2": {
            "freshness_max_age_s": FRESHNESS_MAX_AGE_S,
            "message_gap_threshold_s": MESSAGE_GAP_THRESHOLD_S,
            "scoreable_max_gap_s": SCOREABLE_MAX_GAP_S,
            "state_episode_semantics": "rising edge or new flight; cadence gap alone does not split",
            "message_gap_episode_semantics": "one point event per post-gap row",
            "faa_reference_scope": "not asserted in historical corpus",
            "residual_penalty_combination": "forbidden",
            "attack_ground_truth_claim": "none",
        },
        "legacy_silver_policy": (
            "all update metadata absent => freshness_unknown; partial schema fails closed"
        ),
        "memory_contract": (
            "one Parquet part plus O(unique source_id) ownership and O(part count) summaries"
        ),
        "holdout_access": "forbidden",
    }
    config_sha256 = sha256_json(config)
    manifest_inputs = [InputSpec(contract["manifest_path"], "reference")]
    manifest_inputs.extend(InputSpec(path, "reference") for path in contract["completion_paths"])
    manifest_path = create_immutable_run_manifest(
        run_dir=destination,
        repo_root=root,
        inputs=manifest_inputs,
        splits=splits,
        split_algorithm=base["split_contract"]["algorithm"],
        split_seed=base["split_contract"]["seed"],
        synthetic_flight_ids=synthetic_ids,
        config=config,
    )

    summaries_by_day: dict[str, list[dict]] = {day: [] for day in OPEN_DAYS}
    owners: dict[tuple[str, str], str] = {}
    observed_part_counts: Counter[str] = Counter()
    for part_number, item in enumerate(silver_inputs):
        path = item["path"]
        expected = item["record"]
        stat_before = path.stat()
        frame = pd.read_parquet(path, columns=read_columns)
        if len(frame) != expected["footer_rows"]:
            raise ValueError(f"Silver read/footer row mismatch: {path}")
        source_files = frame["_source_file"].dropna().astype(str).unique()
        if len(source_files) != 1 or frame["_source_file"].isna().any():
            raise ValueError(f"Invalid part provenance: {path}")
        day = source_day(source_files[0])
        if day != item["day_from_role"]:
            raise ValueError(f"Step-5 role/source-day mismatch: {path}")
        if frame["source_id"].isna().any():
            raise ValueError(f"source_id cannot be null: {path}")
        for source_id in frame["source_id"].astype(str).unique():
            key = (day, source_id)
            if key in owners:
                raise ValueError(f"(day, source_id) spans parts: {key}")
            owners[key] = path.name
        segmented = segment_flights(frame, gap_s=SEGMENT_GAP_S)
        segmented["flight_id"] = day + ":" + segmented["flight_id"].astype(str)
        summaries_by_day[day].append(
            summarize_s2_part(
                segmented,
                scoreable_max_gap_s=SCOREABLE_MAX_GAP_S,
                message_gap_threshold_s=MESSAGE_GAP_THRESHOLD_S,
                freshness_max_age_s=FRESHNESS_MAX_AGE_S,
            )
        )
        observed_part_counts[day] += 1
        stat_after = path.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
            or stat_after.st_size != expected["bytes"]
            or stat_after.st_mtime_ns != expected["mtime_ns"]
        ):
            raise ValueError(f"Silver input changed during Step-6 read: {path}")
        del frame, segmented
        if (part_number + 1) % 50 == 0:
            print(f"S2 progress: {part_number + 1}/{len(silver_inputs)}", flush=True)

    if dict(observed_part_counts) != EXPECTED_PARTS_BY_DAY:
        raise ValueError(f"S2 part totals changed: {dict(observed_part_counts)}")
    for item in silver_inputs:
        stat = item["path"].stat()
        if stat.st_size != item["record"]["bytes"] or stat.st_mtime_ns != item["record"]["mtime_ns"]:
            raise ValueError(f"Silver input changed before Step-6 completion: {item['path']}")
    if _implementation_sha256(root) != implementation_sha256:
        raise ValueError("Step-6 implementation bytes changed during evaluation")
    if sha256_json(config) != config_sha256:
        raise ValueError("Step-6 config changed during evaluation")

    day_reports = {
        day: merge_s2_summaries(summaries_by_day[day]) for day in OPEN_DAYS
    }
    observed_rows = {day: report["n_rows"] for day, report in day_reports.items()}
    if observed_rows != EXPECTED_ROWS_BY_DAY:
        raise ValueError(f"S2 row totals changed: {observed_rows}")
    pooled = merge_s2_summaries(
        [item for day in OPEN_DAYS for item in summaries_by_day[day]]
    )
    report = {
        "run_id": destination.name,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha256,
        "config_and_code_unchanged_through_evaluation": True,
        "step5_provenance": {
            "run_id": base["run_id"],
            "manifest_sha256": config["base_step5_manifest_sha256"],
            "completion_sha256": contract["completion_hashes"],
            "transitive_silver_contract_sha256": contract["silver_contract_sha256"],
            "silver_file_count": len(silver_inputs),
            "silver_rows": sum(EXPECTED_ROWS_BY_DAY.values()),
        },
        "units_are_separate": [
            "row",
            "state_rising_edge_episode",
            "message_gap_point_event",
            "flight_segment",
            "scoreable_flight_hour",
        ],
        "semantics": {
            "declared_status": "operational declaration; not attack ground truth",
            "position_quality": "data-quality status; FAA advisory disabled without asserted scope",
            "altitude_availability": "separate from residual penalty",
            "message_gap": "one interval point-event, separate from row missingness",
            "natural_burden_name": "nominal reason burden, never false-positive rate",
        },
        "freshness_schema": {
            "observed_mode": contract["schema_mode"],
            "legacy": "all update metadata absent => freshness_unknown",
            "future_parser": "silver_v2 emits updated/update_timestamp_utc/update_age_s",
            "partial_schema": "fail_closed",
        },
        "by_day": day_reports,
        "pooled": pooled,
        "residual_penalty_combined": False,
        "holdout_accessed": False,
        "artifact_checksum_index": "artifact_checksums.json",
        "gate_status": "evidence_only_pending_step_7_review",
    }
    report_path = destination / "s2_natural_burden_report.json"
    write_json_exclusive(report_path, report)
    artifact_records = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (manifest_path, report_path)
    }
    write_json_exclusive(
        destination / "artifact_checksums.json",
        {
            "schema_version": 1,
            "algorithm": "sha256",
            "self_excluded": True,
            "files": artifact_records,
        },
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(
        repo_root=args.repo_root,
        baseline_manifest=args.baseline_manifest,
        run_dir=args.run_dir,
    )
    print(args.run_dir / "s2_natural_burden_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
