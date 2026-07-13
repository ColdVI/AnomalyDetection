from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.adsb_report_s2_natural_burden as reporter
from adsb.run_manifest import inspect_input_file, sha256_file, sha256_json
from adsb.s2_streaming import (
    FRESHNESS_FIELDS,
    merge_s2_summaries,
    summarize_s2_part,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": ["f1"] * 5,
            "timestamp_utc": [0.0, 10.0, 20.0, 100.0, 110.0],
            "squawk": [None, "7700", "7700", None, None],
            "emergency": [None, "general", "general", None, None],
            "nic": [8, 8, 0, 12, None],
            "nac_p": [9] * 5,
            "sil": [3] * 5,
            "adsb_version": [2] * 5,
            "alt": [1000.0, 1000.0, None, None, 1000.0],
            "alt_geom_m": [1010.0] * 5,
            "on_ground": [False] * 5,
        }
    )


def _with_freshness(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field in FRESHNESS_FIELDS:
        result[f"{field}_updated"] = True
        result[f"{field}_update_timestamp_utc"] = result["timestamp_utc"]
        result[f"{field}_update_age_s"] = 0.0
    return result


def test_legacy_freshness_is_unknown_and_matching_is_not_corroborated():
    result = summarize_s2_part(_frame())
    assert result["freshness"]["squawk"] == {"freshness_unknown": 5}
    assert result["declared_status"]["legacy_matching_rows_not_corroboration"] == 2
    assert result["reason_burden"]["DECLARED_STATUS_NOT_CORROBORATED"]["episodes"] == 1


def test_availability_message_gap_and_quality_have_separate_burden():
    result = summarize_s2_part(_frame())
    assert result["altitude_availability"]["row_counts"]["BARO_ALT_DROPOUT"] == 2
    assert result["reason_burden"]["MESSAGE_GAP"]["episodes"] == 1
    assert result["position_quality"]["status_counts"]["nic"]["schema_invalid"] == 1
    assert result["position_quality"]["status_counts"]["nic"]["reported_unknown_or_unavailable"] == 1
    assert result["scoreable_flight_hours"] == pytest.approx(30.0 / 3600.0)


def test_merge_keeps_rows_events_flights_and_exposure_units_explicit():
    part = summarize_s2_part(_frame())
    merged = merge_s2_summaries([part, part])
    assert merged["n_rows"] == 10
    assert merged["n_flights"] == 2
    reason = merged["reason_burden"]["MESSAGE_GAP"]
    assert reason["episodes"] == 2
    assert reason["flight_fraction"] == pytest.approx(1.0)
    assert reason["episodes_per_scoreable_flight_hour"] == pytest.approx(120.0)
    assert merged["burden_units"]["rate"] == "per scoreable flight-hour"


def test_fresh_matching_and_conflicting_values_have_distinct_consistency_reasons():
    frame = _with_freshness(_frame().iloc[:3].copy())
    frame["squawk"] = ["7700", "7600", "7500"]
    frame["emergency"] = ["general", "general", "unlawful"]

    result = summarize_s2_part(frame)

    counts = result["declared_status"]["consistency_row_counts"]
    assert counts == {"corroborated": 2, "contradictory": 1, "not_corroborated": 0}
    assert result["reason_burden"]["DECLARED_STATUS_CORROBORATED"]["episodes"] == 2
    assert result["reason_burden"]["DECLARED_STATUS_CONTRADICTORY"]["episodes"] == 1
    assert result["declared_status"]["legacy_matching_rows_not_corroboration"] == 0


def test_sparse_cadence_does_not_turn_one_state_into_repeated_episodes():
    frame = _frame().iloc[:3].copy()
    frame["timestamp_utc"] = [0.0, 100.0, 200.0]
    frame["squawk"] = ["7700"] * 3
    frame["emergency"] = [None] * 3

    result = summarize_s2_part(frame)

    assert result["reason_burden"]["DECLARED_SQUAWK_GENERAL"]["episodes"] == 1
    assert result["reason_burden"]["MESSAGE_GAP"]["episodes"] == 2
    assert result["scoreable_flight_hours"] == 0.0


def test_partial_freshness_schema_and_nonfinite_time_fail_closed():
    partial = _frame()
    partial["squawk_updated"] = True
    with pytest.raises(KeyError, match="Partial freshness schema"):
        summarize_s2_part(partial)

    invalid_time = _frame()
    invalid_time.loc[2, "timestamp_utc"] = np.inf
    with pytest.raises(ValueError, match="timestamp_utc must be finite"):
        summarize_s2_part(invalid_time)


def test_merge_rejects_different_burden_contracts():
    first = summarize_s2_part(_frame(), scoreable_max_gap_s=60.0)
    second = summarize_s2_part(_frame(), scoreable_max_gap_s=30.0)
    with pytest.raises(ValueError, match="different contracts"):
        merge_s2_summaries([first, second])


def test_strict_json_writer_rejects_nonfinite_before_creating_file(tmp_path: Path):
    target = tmp_path / "report.json"
    with pytest.raises(ValueError, match="Non-finite"):
        reporter.write_json_exclusive(target, {"bad": np.inf})
    assert not target.exists()


def test_step6_implementation_guard_includes_run_manifest_and_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_manifest_path = Path("adsb/run_manifest.py")
    assert run_manifest_path in reporter.STEP6_IMPLEMENTATION_PATHS

    target = tmp_path / run_manifest_path
    target.parent.mkdir(parents=True)
    target.write_text("first version\n", encoding="utf-8")
    monkeypatch.setattr(reporter, "STEP6_IMPLEMENTATION_PATHS", (run_manifest_path,))
    before = reporter._implementation_sha256(tmp_path)

    target.write_text("second version\n", encoding="utf-8")

    assert reporter._implementation_sha256(tmp_path) != before


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_step5_completion_requires_exact_checksum_coverage(tmp_path: Path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    base_config = {"step": 5}
    base = {
        "run_id": tmp_path.name,
        "config": {"value": base_config, "sha256": sha256_json(base_config)},
    }
    derived_config = {
        "base_config_sha256": base["config"]["sha256"],
        "rule_threshold": 2.0,
        "cusum_h": 10.0,
    }
    derived = {
        "schema_version": 1,
        "derived_config": derived_config,
        "payload_sha256": sha256_json(derived_config),
    }
    derived_path = tmp_path / "derived_frozen_config.json"
    _write_json(derived_path, derived)
    sidecar_path = tmp_path / "derived_frozen_config.sha256"
    sidecar_path.write_text(sha256_file(derived_path) + "\n", encoding="utf-8")
    final = {
        "run_id": tmp_path.name,
        "manifest": manifest.name,
        "config_sha256": base["config"]["sha256"],
        "synthetic_training_rows": 0,
        "synthetic_reference_use": "flight-ID exclusion only",
        "rehearsal_changed_parameters": False,
        "config_and_code_unchanged_through_evaluation": True,
        "gate_status": "evidence_only_pending_step_7_review",
        "artifact_checksum_index": "artifact_checksums.json",
        "derived_frozen_config": {
            "path": derived_path.name,
            "payload_sha256": derived["payload_sha256"],
            "file_sha256": sha256_file(derived_path),
            "sidecar": sidecar_path.name,
        },
    }
    final_path = tmp_path / "streaming_baseline_report.json"
    _write_json(final_path, final)
    artifact_path = tmp_path / "artifact_checksums.json"
    covered = [manifest, final_path, derived_path, sidecar_path]
    _write_json(
        artifact_path,
        {
            "schema_version": 1,
            "algorithm": "sha256",
            "self_excluded": True,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in covered
            },
        },
    )

    paths, hashes = reporter._validate_step5_completion(manifest, base)
    assert {path.name for path in paths} == set(reporter.STEP5_COMPLETION_FILES)
    assert hashes["streaming_baseline_report.json"] == sha256_file(final_path)

    final_path.write_text(final_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="byte mismatch"):
        reporter._validate_step5_completion(manifest, base)


def _silver_frame(source_day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_source_file": [f"v{source_day.replace('-', '.')}-planes.tar"],
            "source_id": ["abc"],
            "timestamp_utc": [1.0],
            "squawk": [None],
            "emergency": [None],
            "nic": [8],
            "nac_p": [9],
            "sil": [3],
            "adsb_version": [2],
            "alt": [1000.0],
            "alt_geom_m": [1010.0],
            "on_ground": [False],
        }
    )


def test_transitive_silver_verification_accepts_legacy_and_rejects_partial_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    silver = tmp_path / reporter.SILVER_RELATIVE_DIR
    silver.mkdir(parents=True)
    inputs = []
    for day in reporter.OPEN_DAYS:
        path = silver / f"{day}.parquet"
        _silver_frame(day).to_parquet(path, index=False)
        inputs.append(
            inspect_input_file(path, role=reporter.DAY_INPUT_ROLE[day], repo_root=tmp_path)
        )
    monkeypatch.setattr(
        reporter, "EXPECTED_PARTS_BY_DAY", {day: 1 for day in reporter.OPEN_DAYS}
    )
    monkeypatch.setattr(
        reporter, "EXPECTED_ROWS_BY_DAY", {day: 1 for day in reporter.OPEN_DAYS}
    )

    verified, mode, contract_hash, selected_schema = reporter._verify_silver_inputs(
        {"inputs": inputs}, tmp_path.resolve()
    )
    assert len(verified) == 3
    assert mode == "silver_v1_legacy_freshness_unknown"
    assert len(contract_hash) == 64
    assert set(selected_schema) == set(reporter.S2_COLUMNS)

    partial_path = silver / f"{reporter.FIT_DAY}.parquet"
    partial = _silver_frame(reporter.FIT_DAY)
    partial["squawk_updated"] = True
    partial.to_parquet(partial_path, index=False)
    inputs[0] = inspect_input_file(
        partial_path, role=reporter.DAY_INPUT_ROLE[reporter.FIT_DAY], repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="partial freshness schema"):
        reporter._verify_silver_inputs({"inputs": inputs}, tmp_path.resolve())
