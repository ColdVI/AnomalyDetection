"""Audit the Step-5 v1 CUSUM bootstrap-upper semantics without rerunning science.

This is a post-result correctness audit.  It reads only the immutable Step-5
JSON/checksum chain, distinguishes the stored raw moving-block bootstrap p95
from a conservative upper bound, and recomputes budget eligibility as::

    conservative upper = max(full-flight observed rate, raw bootstrap p95)

It does not read Silver, synthetic observations, raw archives, Downloads, or
bootstrap blocks; it does not score, sample, fit, tune, or change candidates.
The source run remains byte-for-byte preserved.  Only its ambiguous bootstrap
"upper" semantics are superseded by the audit report in a new run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.run_manifest import (  # noqa: E402
    InputSpec,
    create_immutable_run_manifest,
    sha256_file,
    sha256_json,
)


SOURCE_FILENAMES = (
    "run_manifest.json",
    "normal_burden_calibration.json",
    "derived_frozen_config.json",
    "derived_frozen_config.sha256",
    "streaming_baseline_report.json",
    "artifact_checksums.json",
)
INDEXED_SOURCE_FILENAMES = SOURCE_FILENAMES[:-1]
OLD_RAW_FIELD = "bootstrap_upper_95_episodes_per_hour"
NEW_RAW_FIELD = "bootstrap_raw_quantile_95_episodes_per_hour"
NEW_CONSERVATIVE_FIELD = "conservative_upper_95_episodes_per_hour"
FORBIDDEN_PATH_COMPONENTS = frozenset({"archive", "downloads", "raw"})


class AuditContractError(ValueError):
    """The immutable source chain or audit contract is inconsistent."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"Cannot read canonical JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditContractError(f"Expected a JSON object: {path}")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise AuditContractError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditContractError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise AuditContractError(f"{label} must be finite and non-negative")
    return result


def _indexed_file_record(
    source_dir: Path,
    checksum_index: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    files = checksum_index.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(name), dict):
        raise AuditContractError(f"Final checksum index lacks {name!r}")
    expected = files[name]
    path = source_dir / name
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if expected.get("bytes") != actual_bytes or expected.get("sha256") != actual_sha256:
        raise AuditContractError(f"Final checksum mismatch for {name}")
    return {"bytes": actual_bytes, "sha256": actual_sha256}


def verify_source_chain(source_run_dir: Path) -> dict[str, Any]:
    """Verify the v1 manifest -> calibration -> derived -> final checksum chain."""

    source = source_run_dir.resolve(strict=True)
    forbidden = {part.lower() for part in source.parts} & FORBIDDEN_PATH_COMPONENTS
    if forbidden:
        raise AuditContractError(
            f"Source path uses forbidden components: {sorted(forbidden)}"
        )
    missing = [name for name in SOURCE_FILENAMES if not (source / name).is_file()]
    if missing:
        raise AuditContractError(f"Source run lacks required chain files: {missing}")

    checksum_index = _load_object(source / "artifact_checksums.json")
    if (
        checksum_index.get("schema_version") != 1
        or checksum_index.get("algorithm") != "sha256"
        or checksum_index.get("self_excluded") is not True
    ):
        raise AuditContractError("Unsupported or non-self-excluding final checksum index")
    verified_files = {
        name: _indexed_file_record(source, checksum_index, name)
        for name in INDEXED_SOURCE_FILENAMES
    }
    verified_files["artifact_checksums.json"] = {
        "bytes": (source / "artifact_checksums.json").stat().st_size,
        "sha256": sha256_file(source / "artifact_checksums.json"),
        "self_excluded_from_source_index": True,
    }

    source_manifest = _load_object(source / "run_manifest.json")
    normal_report = _load_object(source / "normal_burden_calibration.json")
    derived_record = _load_object(source / "derived_frozen_config.json")
    final_report = _load_object(source / "streaming_baseline_report.json")
    if source_manifest.get("run_id") != source.name or final_report.get("run_id") != source.name:
        raise AuditContractError("Source run identity differs across manifest/final report")

    derived_config = derived_record.get("derived_config")
    if not isinstance(derived_config, dict):
        raise AuditContractError("derived_frozen_config.json lacks derived_config")
    payload_sha256 = sha256_json(derived_config)
    if derived_record.get("payload_sha256") != payload_sha256:
        raise AuditContractError("Derived-config payload hash mismatch")
    derived_file_sha256 = verified_files["derived_frozen_config.json"]["sha256"]
    sidecar = (source / "derived_frozen_config.sha256").read_text(
        encoding="ascii"
    ).strip()
    if sidecar != derived_file_sha256:
        raise AuditContractError("Derived-config SHA-256 sidecar mismatch")

    normal_sha256 = verified_files["normal_burden_calibration.json"]["sha256"]
    if derived_config.get("normal_burden_calibration_file_sha256") != normal_sha256:
        raise AuditContractError("Derived config does not bind the normal-burden report")
    final_derived = final_report.get("derived_frozen_config")
    if not isinstance(final_derived, dict) or (
        final_derived.get("file_sha256") != derived_file_sha256
        or final_derived.get("payload_sha256") != payload_sha256
        or final_derived.get("sidecar") != "derived_frozen_config.sha256"
    ):
        raise AuditContractError("Final report does not bind the derived-config chain")

    normal_selection = normal_report.get("cusum_natural_burden_selection")
    derived_cusum = derived_config.get("cusum")
    final_threshold = final_report.get("normal_threshold_calibration")
    if not isinstance(normal_selection, dict):
        raise AuditContractError("Normal-burden report lacks CUSUM selection")
    if not isinstance(derived_cusum, dict) or derived_cusum.get("selection") != normal_selection:
        raise AuditContractError("Derived config CUSUM selection differs from calibration report")
    if not isinstance(final_threshold, dict) or (
        final_threshold.get("cusum_natural_burden_selection") != normal_selection
    ):
        raise AuditContractError("Final report CUSUM selection differs from calibration report")

    base_config = source_manifest.get("config", {}).get("value")
    try:
        burden_contract = base_config["cusum"]["burden_calibration"]
    except (KeyError, TypeError) as exc:
        raise AuditContractError("Source manifest lacks CUSUM burden contract") from exc
    candidates = normal_selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AuditContractError("CUSUM selection has no candidates")
    candidate_h = [candidate.get("h") for candidate in candidates if isinstance(candidate, dict)]
    if len(candidate_h) != len(candidates) or candidate_h != burden_contract.get("candidate_h"):
        raise AuditContractError("Candidate list differs from the immutable source contract")
    contract_pairs = (
        ("budget_episodes_per_hour", "advisory_budget_episodes_per_hour"),
        ("bootstrap_repetitions", "bootstrap_repetitions"),
        ("bootstrap_batch_size", "bootstrap_batch_size"),
        ("bootstrap_upper_quantile", "upper_quantile"),
    )
    for selection_key, contract_key in contract_pairs:
        if normal_selection.get(selection_key) != burden_contract.get(contract_key):
            raise AuditContractError(
                f"Selection {selection_key} differs from immutable source contract"
            )
    if normal_selection.get("bootstrap_upper_quantile") != 0.95:
        raise AuditContractError("v1 field audit requires the frozen 0.95 quantile")

    return {
        "source_run_dir": source,
        "source_manifest": source_manifest,
        "selection": normal_selection,
        "burden_contract": burden_contract,
        "verified_files": verified_files,
        "derived_payload_sha256": payload_sha256,
    }


def corrected_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Reinterpret stored v1 values only; perform no resampling or rerun."""

    budget = _finite_nonnegative(
        selection.get("budget_episodes_per_hour"),
        label="budget_episodes_per_hour",
    )
    raw_candidates = selection.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise AuditContractError("CUSUM selection has no candidates")

    comparisons: list[dict[str, Any]] = []
    previous_h = -math.inf
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise AuditContractError(f"Candidate {index} is not an object")
        h = _finite_nonnegative(candidate.get("h"), label=f"candidate[{index}].h")
        if h <= previous_h:
            raise AuditContractError("Candidate h values must remain strictly increasing")
        previous_h = h
        observed = _finite_nonnegative(
            candidate.get("observed_episodes_per_hour"),
            label=f"candidate[{index}].observed_episodes_per_hour",
        )
        raw_quantile = _finite_nonnegative(
            candidate.get(OLD_RAW_FIELD),
            label=f"candidate[{index}].{OLD_RAW_FIELD}",
        )
        old_meets = candidate.get("meets_advisory_budget")
        if not isinstance(old_meets, bool):
            raise AuditContractError(f"Candidate {h:g} old meets flag is not boolean")
        recomputed_old_meets = raw_quantile <= budget
        if old_meets != recomputed_old_meets:
            raise AuditContractError(
                f"Candidate {h:g} old meets flag disagrees with stored raw quantile"
            )
        conservative_upper = max(observed, raw_quantile)
        new_meets = conservative_upper <= budget
        comparisons.append(
            {
                "h": h,
                "observed_episodes_per_hour": observed,
                "old_raw_field_name": OLD_RAW_FIELD,
                "old_raw_bootstrap_quantile_95_episodes_per_hour": raw_quantile,
                NEW_RAW_FIELD: raw_quantile,
                NEW_CONSERVATIVE_FIELD: conservative_upper,
                "old_meets_advisory_budget": old_meets,
                "new_meets_advisory_budget": new_meets,
            }
        )

    old_selected_from_flags = next(
        (row["h"] for row in comparisons if row["old_meets_advisory_budget"]),
        None,
    )
    old_selected_h = selection.get("selected_h")
    if old_selected_h is not None:
        old_selected_h = _finite_nonnegative(old_selected_h, label="selected_h")
    if old_selected_h != old_selected_from_flags:
        raise AuditContractError("Stored selected_h differs from stored candidate flags")
    new_selected_h = next(
        (row["h"] for row in comparisons if row["new_meets_advisory_budget"]),
        None,
    )
    return {
        "budget_episodes_per_hour": budget,
        "old_selected_h": old_selected_h,
        "new_selected_h": new_selected_h,
        "selected_h_unchanged": old_selected_h == new_selected_h,
        "candidates": comparisons,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            dict(payload),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _write_output_checksums(run_dir: Path) -> Path:
    target = run_dir / "artifact_checksums.json"
    if target.exists():
        raise FileExistsError(target)
    files = {
        path.relative_to(run_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path != target
    }
    _write_json_exclusive(
        target,
        {
            "schema_version": 1,
            "algorithm": "sha256",
            "self_excluded": True,
            "files": files,
        },
    )
    return target


def run(*, repo_root: Path, source_run_dir: Path, run_dir: Path) -> dict[str, Any]:
    """Create one immutable audit run and return its report."""

    root = repo_root.resolve(strict=True)
    destination = run_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Audit run directory already exists: {destination}")
    forbidden = {part.lower() for part in destination.parts} & FORBIDDEN_PATH_COMPONENTS
    if forbidden:
        raise AuditContractError(
            f"Audit destination uses forbidden components: {sorted(forbidden)}"
        )

    verified = verify_source_chain(source_run_dir)
    source = verified["source_run_dir"]
    if destination == source or source in destination.parents:
        raise AuditContractError("Audit destination cannot equal or nest under source run")
    correction = corrected_selection(verified["selection"])
    source_inputs = [InputSpec(source / name, "reference") for name in SOURCE_FILENAMES]
    audit_config = {
        "audit_schema_version": 1,
        "audit_type": "post_result_correctness_audit_not_scientific_rerun",
        "source_run_id": source.name,
        "source_artifact_policy": "preserved_read_only",
        "superseded_scope": "bootstrap_upper_semantics_only",
        "old_field_semantics": f"{OLD_RAW_FIELD} is a raw bootstrap p95, not an upper bound",
        "corrected_invariant": (
            "conservative upper = max(full-flight observed episodes/hour, "
            "raw moving-block bootstrap p95)"
        ),
        "selection_rule": (
            "smallest unchanged candidate h whose conservative upper is within "
            "the unchanged advisory budget"
        ),
        "scientific_recomputation": False,
        "scores_read": False,
        "bootstrap_blocks_read": False,
        "candidate_h": verified["burden_contract"]["candidate_h"],
        "advisory_budget_episodes_per_hour": verified["burden_contract"][
            "advisory_budget_episodes_per_hour"
        ],
        "bootstrap_repetitions": verified["burden_contract"]["bootstrap_repetitions"],
        "bootstrap_seed": verified["burden_contract"]["bootstrap_seed"],
        "upper_quantile": verified["burden_contract"]["upper_quantile"],
        "audit_code_sha256": sha256_file(Path(__file__).resolve()),
        "corrected_streaming_code_sha256": sha256_file(root / "adsb" / "streaming.py"),
    }
    manifest_path = create_immutable_run_manifest(
        run_dir=destination,
        repo_root=root,
        inputs=source_inputs,
        splits={"reference": []},
        split_algorithm="post_result_correctness_audit_no_flight_split_v1",
        split_seed=None,
        synthetic_flight_ids=[],
        config=audit_config,
    )

    unchanged = correction["selected_h_unchanged"]
    report = {
        "schema_version": 1,
        "run_id": destination.name,
        "manifest": manifest_path.name,
        "audit_type": "post_result_correctness_audit_not_scientific_rerun",
        "status": (
            "passed_selected_h_unchanged"
            if unchanged
            else "failed_selected_h_changed"
        ),
        "source": {
            "run_id": source.name,
            "artifacts_preserved": True,
            "old_report_disposition": "preserved_superseded_for_upper_semantics_only",
            "verified_input_files": verified["verified_files"],
            "derived_payload_sha256": verified["derived_payload_sha256"],
        },
        "scope": {
            "scientific_rerun": False,
            "score_recomputation": False,
            "bootstrap_resampling": False,
            "candidate_or_budget_change": False,
            "semantic_correction_only": True,
        },
        "correction": {
            "old_field": OLD_RAW_FIELD,
            "old_field_correct_interpretation": "raw moving-block bootstrap p95",
            "new_raw_field": NEW_RAW_FIELD,
            "new_conservative_field": NEW_CONSERVATIVE_FIELD,
            "invariant": (
                "conservative upper = max(full-flight observed episodes/hour, "
                "raw moving-block bootstrap p95)"
            ),
        },
        "selection_comparison": correction,
        "failure_action_if_selected_h_changed": (
            "report nonzero exit; do not silently replace v1 selected_h"
        ),
        "artifact_checksum_index": "artifact_checksums.json",
    }
    _write_json_exclusive(destination / "cusum_bootstrap_upper_audit.json", report)
    _write_output_checksums(destination)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        repo_root=args.repo_root,
        source_run_dir=args.source_run_dir,
        run_dir=args.run_dir,
    )
    comparison = report["selection_comparison"]
    print(json.dumps({
        "run_id": report["run_id"],
        "status": report["status"],
        "old_selected_h": comparison["old_selected_h"],
        "new_selected_h": comparison["new_selected_h"],
    }, sort_keys=True))
    return 0 if comparison["selected_h_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
