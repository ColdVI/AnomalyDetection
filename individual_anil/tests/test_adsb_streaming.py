"""ADS-B S2/streaming testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from __future__ import annotations

import numpy as np

import pandas as pd

import pytest

from adsb.s2 import (
    S2Config,
    classify_s2_rows,
    explode_s2_reasons,
    field_freshness,
    reason_episodes,
)

import json

from pathlib import Path

import scripts.adsb_report_s2_natural_burden as reporter

from adsb.run_manifest import inspect_input_file, sha256_file, sha256_json

from adsb.s2_streaming import (
    FRESHNESS_FIELDS,
    merge_s2_summaries,
    summarize_s2_part,
)

from adsb.streaming import (
    BoundedFramePrioritySampler,
    BoundedPrioritySampler,
    CusumBurdenCalibration,
    count_alarm_episodes,
    deterministic_file_sample,
    dkw_quantile_error_bound,
    moving_block_burden_rows,
    prefixed_flight_id,
    robust_sample_calibration,
    scoreable_row_exposure_seconds,
    select_cusum_threshold,
    stable_fit_role,
)

import copy

import scripts.adsb_run_full_streaming_baseline as runner

from adsb.streaming import CusumBurdenCalibration



# ===== kaynak: test_adsb_s2 =====

FRESH_FIELDS = ("squawk", "emergency", "nic", "nac_p", "sil", "adsb_version")


def _with_fresh_metadata(frame: pd.DataFrame, *, updated: bool = True, age_s: float = 0.0):
    result = frame.copy()
    for field in FRESH_FIELDS:
        result[f"{field}_updated"] = updated
        result[f"{field}_update_timestamp_utc"] = 1000.0 if updated or age_s >= 0 else np.nan
        result[f"{field}_update_age_s"] = age_s
    return result


def _base_row(**overrides):
    row = {
        "flight_id": "f1",
        "source_id": "abc123",
        "timestamp_utc": 1000.0,
        "on_ground": False,
        "alt": 1000.0,
        "alt_geom_m": 1010.0,
        "ads_source_type": "adsb_icao",
        "squawk": "1200",
        "emergency": "none",
        "nic": 8,
        "nac_p": 9,
        "sil": 3,
        "adsb_version": 2,
        "faa_reference_scope": True,
    }
    row.update(overrides)
    return row


def test_matching_fresh_critical_declarations_are_corroborated_and_independent():
    frame = _with_fresh_metadata(pd.DataFrame([_base_row(squawk="7700", emergency="general")]))

    result = classify_s2_rows(frame)

    assert result.loc[0, "declared_status_squawk_type"] == "general"
    assert result.loc[0, "declared_status_emergency_type"] == "general"
    assert result.loc[0, "declared_status_consistency"] == "corroborated"
    assert result.loc[0, "declared_status_reason_codes"] == (
        "DECLARED_SQUAWK_GENERAL",
        "DECLARED_EMERGENCY_GENERAL",
        "DECLARED_STATUS_CORROBORATED",
    )
    assert bool(result.loc[0, "declared_status_active"]) is True


def test_only_two_fresh_conflicting_critical_states_are_contradictory():
    rows = [
        _base_row(squawk="7500", emergency="nordo"),
        _base_row(squawk="7700", emergency="lifeguard", timestamp_utc=1010.0),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result.loc[0, "declared_status_consistency"] == "contradictory"
    assert "DECLARED_STATUS_CONTRADICTORY" in result.loc[0, "declared_status_reason_codes"]
    # Lifeguard is a separate declaration, not an explicitly conflicting
    # member of the three critical squawk/emergency mappings.
    assert result.loc[1, "declared_status_consistency"] == "not_corroborated"
    assert "DECLARED_EMERGENCY_LIFEGUARD" in result.loc[1, "declared_status_reason_codes"]


@pytest.mark.parametrize("emergency", ["lifeguard", "minfuel", "downed", "reserved"])
def test_standalone_emergency_types_remain_separate_declarations(emergency):
    frame = _with_fresh_metadata(
        pd.DataFrame([_base_row(squawk="1200", emergency=emergency)])
    )

    result = classify_s2_rows(frame)

    assert result.loc[0, "declared_status_emergency_type"] == emergency
    assert f"DECLARED_EMERGENCY_{emergency.upper()}" in result.loc[
        0, "declared_status_reason_codes"
    ]
    assert result.loc[0, "declared_status_consistency"] == "not_corroborated"


def test_stale_or_legacy_freshness_cannot_corroborate():
    stale = _with_fresh_metadata(
        pd.DataFrame([_base_row(squawk="7600", emergency="nordo")]),
        updated=False,
        age_s=61.0,
    )
    stale_result = classify_s2_rows(stale, config=S2Config(freshness_max_age_s=60.0))
    assert stale_result.loc[0, "squawk_freshness"] == "stale"
    assert stale_result.loc[0, "emergency_freshness"] == "stale"
    assert stale_result.loc[0, "declared_status_consistency"] == "not_corroborated"

    # Existing Silver v1 has values but no update columns.  It is never silently
    # promoted to fresh/corroborated.
    legacy = pd.DataFrame([_base_row(squawk="7600", emergency="nordo")])
    legacy_result = classify_s2_rows(legacy)
    assert legacy_result.loc[0, "squawk_freshness"] == "freshness_unknown"
    assert legacy_result.loc[0, "emergency_freshness"] == "freshness_unknown"
    assert legacy_result.loc[0, "declared_status_consistency"] == "not_corroborated"


def test_field_freshness_distinguishes_never_observed_and_invalid_age():
    frame = pd.DataFrame(
        {
            "squawk_updated": [False, False, True],
            "squawk_update_timestamp_utc": [np.nan, 1000.0, np.nan],
            "squawk_update_age_s": [np.nan, -1.0, np.nan],
        }
    )

    states = field_freshness(frame, "squawk", max_age_s=60.0)

    assert states.tolist() == ["never_observed", "invalid_update_metadata", "fresh"]


def test_position_quality_schema_missing_zero_and_scoped_advisory_are_distinct():
    rows = [
        _base_row(nic=6, nac_p=7, sil=2),
        _base_row(nic=6, nac_p=7, sil=2, faa_reference_scope=False, timestamp_utc=1010.0),
        _base_row(nic=12, nac_p=12, sil=4, adsb_version=3, timestamp_utc=1020.0),
        _base_row(nic=0, nac_p=0, sil=0, timestamp_utc=1030.0),
        _base_row(nic=None, nac_p=None, sil=None, timestamp_utc=1040.0),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result.loc[0, "position_quality_scope"] == "eligible"
    assert result.loc[0, "position_quality_nic_status"] == "below_faa_reference"
    assert result.loc[0, "position_quality_nac_p_status"] == "below_faa_reference"
    assert result.loc[0, "position_quality_sil_status"] == "below_faa_reference"
    assert bool(result.loc[0, "position_quality_advisory"]) is True

    assert result.loc[1, "position_quality_scope"] == "outside_asserted_scope"
    assert result.loc[1, "position_quality_nic_status"] == (
        "reported_valid_outside_faa_reference_scope"
    )
    assert bool(result.loc[1, "position_quality_advisory"]) is False

    assert result.loc[2, "position_quality_scope"] == "version_schema_invalid"
    assert result.loc[2, "position_quality_adsb_version_status"] == "schema_invalid"
    assert result.loc[2, "position_quality_nic_status"] == "schema_invalid"
    assert result.loc[2, "position_quality_nac_p_status"] == "schema_invalid"
    assert result.loc[2, "position_quality_sil_status"] == "schema_invalid"
    assert "POSITION_QUALITY_ADSB_VERSION_SCHEMA_INVALID" in result.loc[
        2, "position_quality_reason_codes"
    ]

    for field in ("nic", "nac_p", "sil"):
        assert result.loc[3, f"position_quality_{field}_status"] == (
            "reported_unknown_or_unavailable"
        )
        assert result.loc[4, f"position_quality_{field}_status"] == "missing"


def test_faa_reference_requires_airborne_adsb_version_and_explicit_scope():
    rows = [
        _base_row(nic=6, nac_p=7, sil=2, on_ground=True),
        _base_row(nic=6, nac_p=7, sil=2, ads_source_type="mlat", timestamp_utc=1010.0),
        _base_row(nic=6, nac_p=7, sil=2, adsb_version=1, timestamp_utc=1020.0),
    ]
    without_scope = _base_row(nic=6, nac_p=7, sil=2, timestamp_utc=1030.0)
    without_scope.pop("faa_reference_scope")
    rows.append(without_scope)
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result["position_quality_scope"].tolist() == [
        "ground_not_applicable",
        "not_adsb_source",
        "version_ineligible",
        "scope_unknown",  # column exists due to other rows, but this value is null
    ]
    assert not result["position_quality_advisory"].any()


def test_altitude_availability_and_message_gap_are_separate_channels():
    rows = [
        _base_row(timestamp_utc=0.0, on_ground=True, alt=None, alt_geom_m=None),
        _base_row(timestamp_utc=10.0, on_ground=False, alt=None, alt_geom_m=1000.0),
        _base_row(timestamp_utc=80.0, on_ground=False, alt=None, alt_geom_m=None),
        _base_row(timestamp_utc=90.0, on_ground=False, alt=900.0, alt_geom_m=None),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result["altitude_availability"].tolist() == [
        "GROUND_ALT_NOT_APPLICABLE",
        "BARO_ALT_DROPOUT",
        "ALL_ALTITUDE_UNAVAILABLE",
        "AVAILABLE",
    ]
    assert result["message_gap"].tolist() == [False, False, True, False]
    assert result.loc[2, "message_gap_s"] == 70.0
    assert result.loc[2, "message_gap_interval_start"] == 10.0
    assert result.loc[2, "message_gap_interval_end"] == 80.0
    assert result.loc[2, "message_gap_reason_code"] == "MESSAGE_GAP"
    assert result.loc[2, "altitude_availability"] == "ALL_ALTITUDE_UNAVAILABLE"


def test_message_gap_state_resets_per_entity_and_marks_nonmonotonic_time():
    rows = [
        _base_row(flight_id="a", timestamp_utc=0.0),
        _base_row(flight_id="b", timestamp_utc=1000.0),
        _base_row(flight_id="a", timestamp_utc=70.0),
        _base_row(flight_id="a", timestamp_utc=60.0),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame, entity_col="flight_id")

    assert result["message_interval_status"].tolist() == [
        "FIRST_OBSERVATION",
        "FIRST_OBSERVATION",
        "MESSAGE_GAP",
        "NON_MONOTONIC_TIMESTAMP",
    ]


def test_reason_rows_can_be_eventized_without_anomaly_or_false_positive_labels():
    rows = [
        _base_row(timestamp_utc=0.0, emergency="general"),
        _base_row(timestamp_utc=30.0, emergency="general"),
        _base_row(timestamp_utc=100.0, emergency="general"),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))
    classified = classify_s2_rows(frame)
    long = explode_s2_reasons(classified)
    emergency = long.loc[long["reason_code"] == "DECLARED_EMERGENCY_GENERAL"]

    episodes = reason_episodes(emergency, entity_col="flight_id", merge_gap_s=60.0)

    assert len(emergency) == 3
    assert episodes[["episode_start", "episode_end", "n_rows"]].to_dict("records") == [
        {"episode_start": 0.0, "episode_end": 100.0, "n_rows": 3},
    ]
    assert not {"anomaly", "false_positive", "label"}.intersection(long.columns)


def test_reason_episodes_split_on_inactive_row_not_sparse_cadence():
    rows = [
        _base_row(timestamp_utc=0.0, emergency="general"),
        _base_row(timestamp_utc=10.0, emergency=None),
        _base_row(timestamp_utc=20.0, emergency="general"),
    ]
    classified = classify_s2_rows(_with_fresh_metadata(pd.DataFrame(rows)))
    emergency = explode_s2_reasons(classified).loc[
        lambda frame: frame["reason_code"] == "DECLARED_EMERGENCY_GENERAL"
    ]

    episodes = reason_episodes(emergency, entity_col="flight_id", merge_gap_s=10_000.0)

    assert episodes[["episode_start", "episode_end", "n_rows"]].to_dict("records") == [
        {"episode_start": 0.0, "episode_end": 0.0, "n_rows": 1},
        {"episode_start": 20.0, "episode_end": 20.0, "n_rows": 1},
    ]


def test_reason_episodes_treat_each_message_gap_row_as_a_point_event():
    rows = [
        _base_row(timestamp_utc=0.0),
        _base_row(timestamp_utc=100.0),
        _base_row(timestamp_utc=200.0),
    ]
    classified = classify_s2_rows(_with_fresh_metadata(pd.DataFrame(rows)))
    gaps = explode_s2_reasons(classified).loc[
        lambda frame: frame["reason_code"] == "MESSAGE_GAP"
    ]

    episodes = reason_episodes(gaps, entity_col="flight_id", merge_gap_s=10_000.0)

    assert episodes[["episode_start", "episode_end", "n_rows"]].to_dict("records") == [
        {"episode_start": 100.0, "episode_end": 100.0, "n_rows": 1},
        {"episode_start": 200.0, "episode_end": 200.0, "n_rows": 1},
    ]


def test_reason_episodes_require_original_row_positions():
    reason_rows = pd.DataFrame(
        {
            "flight_id": ["f1"],
            "timestamp_utc": [0.0],
            "channel": ["declared_status"],
            "reason_code": ["DECLARED_EMERGENCY_GENERAL"],
        }
    )

    with pytest.raises(KeyError, match="row_position"):
        reason_episodes(reason_rows, entity_col="flight_id")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"freshness_max_age_s": -1.0},
        {"message_gap_threshold_s": 0.0},
        {"faa_reference_scope_col": ""},
        {"faa_eligible_versions": ()},
    ],
)
def test_invalid_s2_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        S2Config(**kwargs)



# ===== kaynak: test_adsb_s2_streaming =====

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


def _worker_item(path: Path, day: str) -> dict:
    stat = path.stat()
    return {
        "path": path,
        "day_from_role": day,
        "record": {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "footer_rows": 1,
        },
    }


def test_parallel_part_processing_matches_single_worker_in_input_order(tmp_path: Path):
    paths = []
    for index, day in enumerate((reporter.FIT_DAY, reporter.DEVELOPMENT_DAY)):
        path = tmp_path / f"part-{index}.parquet"
        frame = _silver_frame(day)
        frame["source_id"] = f"source-{index}"
        frame.to_parquet(path, index=False)
        paths.append(_worker_item(path, day))

    kwargs = {
        "silver_inputs": paths,
        "read_columns": tuple(reporter.S2_COLUMNS),
    }
    single = list(reporter._iter_silver_part_results(**kwargs, n_jobs=1))
    parallel = list(reporter._iter_silver_part_results(**kwargs, n_jobs=2))

    assert parallel == single
    assert [item["part_number"] for item in parallel] == [0, 1]
    assert [item["day"] for item in parallel] == [
        reporter.FIT_DAY,
        reporter.DEVELOPMENT_DAY,
    ]


@pytest.mark.parametrize("n_jobs", [0, reporter.MAX_N_JOBS + 1, True, 1.5])
def test_parallel_worker_count_is_bounded_and_typed(n_jobs):
    with pytest.raises((TypeError, ValueError)):
        list(
            reporter._iter_silver_part_results(
                [],
                read_columns=tuple(reporter.S2_COLUMNS),
                n_jobs=n_jobs,
            )
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



# ===== kaynak: test_adsb_streaming =====

def test_day_prefix_and_hash_split_are_order_independent():
    assert prefixed_flight_id("2026-02-28", "abc_000") == "2026-02-28:abc_000"
    assert stable_fit_role("abc_000", seed=7) == stable_fit_role("abc_000", seed=7)


def test_file_sample_is_reproducible_and_value_independent():
    values = np.arange(1000)
    a = deterministic_file_sample(values, probability=0.2, seed=1, file_key="p", purpose="x")
    b_mask_values = deterministic_file_sample(values + 10_000, probability=0.2, seed=1, file_key="p", purpose="x")
    assert np.array_equal(a + 10_000, b_mask_values)


def test_robust_calibration_excludes_exact_zero_mad_without_floor():
    result = robust_sample_calibration({"good": [0.0, 1.0, 2.0], "zero": [4.0, 4.0, 4.0]})
    assert result["excluded_channels"] == ["zero"]
    assert result["calibration"]["good"]["mad"] == pytest.approx(1.4826)
    assert dkw_quantile_error_bound(1000) > 0


def test_episode_and_exposure_contracts_are_separate():
    times = np.array([0.0, 10.0, 20.0, 100.0])
    assert count_alarm_episodes(times, np.array([False, True, True, True]), merge_gap_s=60) == 2
    assert scoreable_row_exposure_seconds(times, np.ones(4, bool), max_gap_s=60) == pytest.approx(20.0)


def test_moving_blocks_and_threshold_selection_use_natural_budget_only():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        advisory_budget_episodes_per_hour=0.1,
        bootstrap_repetitions=20,
        moving_block_s=300.0,
        moving_block_stride_s=150.0,
    )
    times = np.arange(0.0, 1201.0, 10.0)
    scores = np.full(len(times), 5.0)
    rows = moving_block_burden_rows(
        "f", times, scores, np.ones(len(times), bool), contract=contract, max_gap_s=60.0
    )
    frame = pd.DataFrame(rows)
    result = select_cusum_threshold(frame, contract=contract)
    assert result["selected_h"] == 10.0
    assert result["candidates"][0]["meets_advisory_budget"] is False
    assert result["candidates"][1]["meets_advisory_budget"] is True


def test_moving_blocks_count_continuing_alarm_episode_onset_only_once():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0,), bootstrap_repetitions=5,
        moving_block_s=300.0, moving_block_stride_s=150.0,
    )
    times = np.arange(0.0, 1201.0, 10.0)
    rows = moving_block_burden_rows(
        "f", times, np.full(len(times), 5.0), np.ones(len(times), bool),
        contract=contract, max_gap_s=60.0,
    )
    assert sum(row["h_1"] for row in rows) == 1


def test_moving_blocks_anchor_tail_and_use_half_open_nonfinal_boundary():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0,),
        bootstrap_repetitions=5,
        moving_block_s=300.0,
        moving_block_stride_s=300.0,
    )
    times = np.array([0.0, 300.0, 350.0, 650.0])
    scores = np.array([0.0, 5.0, 0.0, 5.0])
    rows = moving_block_burden_rows(
        "f",
        times,
        scores,
        np.ones(len(times), bool),
        contract=contract,
        max_gap_s=400.0,
    )
    assert [row["block_start"] for row in rows] == [0.0, 300.0, 350.0]
    # t=300 is excluded from [0,300); t=650 is included by the final block.
    assert [row["h_1"] for row in rows] == [0, 1, 1]
    # The valid interval ending exactly at the final block's left boundary is
    # retained by endpoint attribution.
    assert rows[-1]["exposure_s"] == pytest.approx(350.0)


def test_full_flight_counters_define_observed_burden_not_overlapping_blocks():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        advisory_budget_episodes_per_hour=100.0,
        bootstrap_repetitions=10,
    )
    blocks = pd.DataFrame(
        {
            "exposure_s": [1800.0, 1800.0, 1800.0],
            "h_1": [4, 4, 4],
            "h_10": [0, 0, 0],
        }
    )
    result = select_cusum_threshold(
        blocks,
        contract=contract,
        observed_exposure_s=3600.0,
        observed_episodes_by_h={1.0: 1, 10.0: 0},
    )
    assert result["observed_burden_source"] == "full_flight_counters"
    assert result["observed_exposure_hours"] == pytest.approx(1.0)
    assert result["candidates"][0]["observed_episode_count"] == 1
    assert result["candidates"][0]["observed_episodes_per_hour"] == pytest.approx(1.0)
    assert result["candidates"][1]["observed_episodes_per_hour"] == pytest.approx(0.0)


def test_conservative_upper_cannot_fall_below_full_flight_observed_rate():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        advisory_budget_episodes_per_hour=0.5,
        bootstrap_repetitions=10,
    )
    # The bootstrap sample has no h=1 episode, while the authoritative
    # full-flight counter has one episode/hour.  The raw quantile therefore
    # cannot be used by itself as an upper bound or for threshold selection.
    blocks = pd.DataFrame(
        {
            "exposure_s": [1800.0, 1800.0],
            "h_1": [0, 0],
            "h_10": [0, 0],
        }
    )
    result = select_cusum_threshold(
        blocks,
        contract=contract,
        observed_exposure_s=3600.0,
        observed_episodes_by_h={1.0: 1, 10.0: 0},
    )

    low_h = result["candidates"][0]
    assert low_h["bootstrap_raw_quantile_95_episodes_per_hour"] == 0.0
    assert low_h["observed_episodes_per_hour"] == pytest.approx(1.0)
    assert low_h["conservative_upper_95_episodes_per_hour"] == pytest.approx(1.0)
    assert low_h["meets_advisory_budget"] is False
    assert result["selected_h"] == 10.0
    assert "bootstrap_upper_95_episodes_per_hour" not in low_h


def test_conservative_upper_uses_raw_bootstrap_quantile_when_it_is_larger():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0,),
        advisory_budget_episodes_per_hour=100.0,
        bootstrap_repetitions=10,
    )
    blocks = pd.DataFrame({"exposure_s": [3600.0], "h_1": [3]})
    result = select_cusum_threshold(
        blocks,
        contract=contract,
        observed_exposure_s=3600.0,
        observed_episodes_by_h={1.0: 1},
    )
    candidate = result["candidates"][0]
    assert candidate["bootstrap_raw_quantile_95_episodes_per_hour"] == pytest.approx(3.0)
    assert candidate["conservative_upper_95_episodes_per_hour"] == pytest.approx(3.0)


def test_full_flight_observed_arguments_are_all_or_nothing_and_exact():
    contract = CusumBurdenCalibration(candidate_h=(1.0,), bootstrap_repetitions=2)
    blocks = pd.DataFrame({"exposure_s": [60.0], "h_1": [0]})
    with pytest.raises(ValueError, match="supplied together"):
        select_cusum_threshold(blocks, contract=contract, observed_exposure_s=60.0)
    with pytest.raises(ValueError, match="exactly match"):
        select_cusum_threshold(
            blocks,
            contract=contract,
            observed_exposure_s=60.0,
            observed_episodes_by_h={},
        )


def _add_scalar_streams(sampler):
    sampler.add(
        np.r_[np.arange(400.0), np.nan],
        probability=0.37,
        seed=7,
        file_key="a",
        purpose="x",
    )
    sampler.add(
        np.arange(400.0, 1000.0),
        probability=0.37,
        seed=7,
        file_key="a",
        purpose="x",
    )
    sampler.add(
        np.arange(10_000.0, 10_500.0),
        probability=0.37,
        seed=7,
        file_key="b",
        purpose="x",
    )


def test_bounded_priority_sampler_is_chunk_exact_deterministic_and_hard_capped():
    chunked = BoundedPrioritySampler(capacity=25)
    _add_scalar_streams(chunked)

    one_shot = BoundedPrioritySampler(capacity=25)
    one_shot.add(
        np.arange(1000.0),
        probability=0.37,
        seed=7,
        file_key="a",
        purpose="x",
    )
    one_shot.add(
        np.arange(10_000.0, 10_500.0),
        probability=0.37,
        seed=7,
        file_key="b",
        purpose="x",
    )
    assert len(chunked.values) == 25
    assert chunked.finite_seen == 1500
    assert np.array_equal(chunked.values, one_shot.values)
    assert np.array_equal(chunked.priorities, one_shot.priorities)

    unbounded = BoundedPrioritySampler(capacity=2000)
    _add_scalar_streams(unbounded)
    assert np.array_equal(chunked.values, unbounded.values[:25])
    assert np.array_equal(chunked.priorities, unbounded.priorities[:25])


def test_bounded_priority_sampler_is_independent_of_file_processing_order():
    forward = BoundedPrioritySampler(capacity=40)
    reverse = BoundedPrioritySampler(capacity=40)
    streams = [("a", np.arange(500.0)), ("b", np.arange(1000.0, 1500.0))]
    for key, values in streams:
        forward.add(values, probability=1.0, seed=11, file_key=key, purpose="p")
        assert len(forward.values) <= forward.capacity
    for key, values in reversed(streams):
        reverse.add(values, probability=1.0, seed=11, file_key=key, purpose="p")
        assert len(reverse.values) <= reverse.capacity
    assert np.array_equal(forward.values, reverse.values)
    assert np.array_equal(forward.priorities, reverse.priorities)


def test_bounded_frame_sampler_is_chunk_exact_order_independent_and_hard_capped():
    source_a = pd.DataFrame({"row_id": np.arange(100), "value": np.arange(100) * 2})
    source_b = pd.DataFrame(
        {"row_id": np.arange(100, 180), "value": np.arange(100, 180) * 2}
    )

    chunked = BoundedFramePrioritySampler(capacity=30, seed=19)
    chunked.add(source_a.iloc[:35], file_key="a", purpose="blocks")
    chunked.add(source_a.iloc[35:], file_key="a", purpose="blocks")
    chunked.add(source_b, file_key="b", purpose="blocks")

    reordered = BoundedFramePrioritySampler(capacity=30, seed=19)
    reordered.add(source_b, file_key="b", purpose="blocks")
    reordered.add(source_a, file_key="a", purpose="blocks")

    assert chunked.rows_seen == reordered.rows_seen == 180
    assert len(chunked.frame) == len(reordered.frame) == 30
    pd.testing.assert_frame_equal(chunked.frame, reordered.frame)

    unbounded = BoundedFramePrioritySampler(capacity=180, seed=19)
    unbounded.add(source_a, file_key="a", purpose="blocks")
    unbounded.add(source_b, file_key="b", purpose="blocks")
    pd.testing.assert_frame_equal(
        chunked.frame,
        unbounded.frame.iloc[:30].reset_index(drop=True),
    )


def test_bounded_frame_sampler_rejects_schema_drift_and_reserved_column():
    sampler = BoundedFramePrioritySampler(capacity=2)
    sampler.add(pd.DataFrame({"a": [1]}), file_key="x", purpose="p")
    with pytest.raises(ValueError, match="identical ordered columns"):
        sampler.add(pd.DataFrame({"b": [2]}), file_key="y", purpose="p")
    with pytest.raises(ValueError, match="reserved column"):
        sampler.add(
            pd.DataFrame({"a": [2], "_sample_priority": [0.1]}),
            file_key="z",
            purpose="p",
        )



# ===== kaynak: test_adsb_full_streaming_baseline =====

def _preflight_part(path: Path, day: str, source_id: str, times: list[float]) -> Path:
    n = len(times)
    frame = pd.DataFrame(
        {
            "_source_file": [f"v{day.replace('-', '.')}-planes-readsb-prod-0.tar"] * n,
            "source_id": [source_id] * n,
            "timestamp_utc": times,
            "lat": np.full(n, 40.0),
            "lon": np.full(n, 29.0),
            "alt": np.full(n, 1_000.0),
            "alt_geom_m": np.full(n, 1_012.0),
            "on_ground": np.zeros(n, dtype=bool),
            "ground_speed_ms": np.full(n, 100.0),
            "track_deg": np.full(n, 90.0),
            "vertical_rate_ms": np.zeros(n),
        }
    )
    frame.to_parquet(path, index=False)
    return path


def _full_fit_part(path: Path) -> runner.PartInventory:
    rows: list[pd.DataFrame] = []
    for source_number, source_id in enumerate(("fit", "cal")):
        n = 36
        i = np.arange(n, dtype=float)
        t = 1_772_300_000.0 + source_number * 10_000.0 + i * 10.0
        # Deliberate deterministic curvature/jitter keeps several train MADs nonzero.
        lat = 40.0 + source_number * 0.1 + np.cumsum(0.0007 + 0.00005 * np.sin(i / 2.0))
        lon = 29.0 + source_number * 0.1 + np.cumsum(0.0008 + 0.00004 * np.cos(i / 3.0))
        alt = 1_000.0 + 2.0 * i + 0.2 * np.sin(i)
        rows.append(
            pd.DataFrame(
                {
                    "_source_file": ["v2026.02.28-planes-readsb-prod-0.tar"] * n,
                    "source_id": [source_id] * n,
                    "timestamp_utc": t,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "alt_geom_m": alt + 12.0 + 0.5 * np.cos(i / 4.0),
                    "on_ground": [False] * n,
                    "ground_speed_ms": 12.0 + 0.5 * np.sin(i / 3.0),
                    "track_deg": 35.0 + 2.0 * np.cos(i / 5.0),
                    "vertical_rate_ms": 0.2 + 0.05 * np.sin(i / 2.0),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    frame.to_parquet(path, index=False)
    return runner.PartInventory(
        path=path.resolve(),
        day=runner.FIT_DAY,
        source_file="v2026.02.28-planes-readsb-prod-0.tar",
        footer_rows=len(frame),
        source_id_count=2,
        flight_id_count=2,
    )


def test_small_parquet_inventory_derives_days_prefixes_and_exact_counts(tmp_path: Path):
    silver = tmp_path / "silver_fixture"
    silver.mkdir()
    _preflight_part(silver / "a.parquet", "2026-02-28", "a", [0.0, 10.0, 2000.0])
    _preflight_part(silver / "b.parquet", "2026-03-01", "b", [0.0, 10.0])
    _preflight_part(silver / "c.parquet", "2026-03-16", "c", [0.0, 10.0])

    inventory = runner.inventory_open_silver(
        silver,
        expected_total_parts=3,
        expected_parts_by_day={day: 1 for day in runner.OPEN_DAYS},
        expected_rows_by_day=None,
    )

    assert inventory.part_counts_by_day == {day: 1 for day in runner.OPEN_DAYS}
    assert inventory.flight_ids_by_day[runner.FIT_DAY] == (
        "2026-02-28:a_000",
        "2026-02-28:a_001",
    )
    assert set(inventory.selected_schema) == set(runner.SILVER_COLUMNS)


def test_inventory_rejects_one_day_source_id_spanning_parts(tmp_path: Path):
    silver = tmp_path / "silver_fixture"
    silver.mkdir()
    _preflight_part(silver / "a1.parquet", "2026-02-28", "same", [0.0])
    _preflight_part(silver / "a2.parquet", "2026-02-28", "same", [10.0])
    _preflight_part(silver / "b.parquet", "2026-03-01", "b", [0.0])
    _preflight_part(silver / "c.parquet", "2026-03-16", "c", [0.0])

    with pytest.raises(runner.StreamingContractError, match="spans parts"):
        runner.inventory_open_silver(
            silver,
            expected_total_parts=4,
            expected_parts_by_day={
                runner.FIT_DAY: 2,
                runner.DEVELOPMENT_DAY: 1,
                runner.REHEARSAL_DAY: 1,
            },
            expected_rows_by_day=None,
        )


def test_inventory_preflight_requires_full_selected_schema_and_types(tmp_path: Path):
    silver = tmp_path / "silver_fixture"
    silver.mkdir()
    first = _preflight_part(silver / "a.parquet", "2026-02-28", "a", [0.0])
    second = _preflight_part(silver / "b.parquet", "2026-03-01", "b", [0.0])
    third = _preflight_part(silver / "c.parquet", "2026-03-16", "c", [0.0])
    broken = pd.read_parquet(second)
    broken["on_ground"] = "false"
    broken.to_parquet(second, index=False)

    with pytest.raises(runner.StreamingContractError, match="on_ground.*expected boolean"):
        runner.inventory_open_silver(
            silver,
            expected_total_parts=3,
            expected_parts_by_day={day: 1 for day in runner.OPEN_DAYS},
            expected_rows_by_day=None,
        )

    assert first.exists() and third.exists()


def test_synthetic_clean_is_reference_only_and_normalized_to_fit_day(tmp_path: Path):
    synthetic_dir = tmp_path / "synthetic_fixture"
    synthetic_dir.mkdir()
    clean = synthetic_dir / "clean.parquet"
    pd.DataFrame(
        {"flight_id": ["a_000", "2026-02-28:b_000", "a_000"], "x": [1, 2, 3]}
    ).to_parquet(clean, index=False)

    excluded, records = runner.load_synthetic_exclusion_ids([clean])

    assert excluded == {"2026-02-28:a_000", "2026-02-28:b_000"}
    assert records[0]["use"] == "exclusion_reference_only"

    ordinary = tmp_path.parent / "ordinary_clean_reference.parquet"
    pd.DataFrame({"flight_id": ["a_000"]}).to_parquet(ordinary, index=False)
    with pytest.raises(runner.StreamingContractError, match="synthetic path marker"):
        runner.load_synthetic_exclusion_ids([ordinary])


def test_exact_split_excludes_synthetic_sources_before_sha256_assignment():
    days = {
        runner.FIT_DAY: tuple(f"2026-02-28:f{i}_000" for i in range(20)),
        runner.DEVELOPMENT_DAY: ("2026-03-01:d_000",),
        runner.REHEARSAL_DAY: ("2026-03-16:r_000",),
    }
    excluded = {"2026-02-28:f3_000"}
    first, audit = runner.derive_exact_splits(days, synthetic_exclusion_ids=excluded)
    second, _ = runner.derive_exact_splits(
        {key: tuple(reversed(value)) for key, value in days.items()},
        synthetic_exclusion_ids=excluded,
    )

    assert first == second
    assert not any(excluded & set(first[role]) for role in ("fit", "calibration"))
    assert set(first["validation"]) == excluded
    assert audit["synthetic_exclusion_ids_matched_on_fit_day"] == 1
    assert set(first) == {"fit", "calibration", "validation", "development", "rehearsal"}


def test_exact_split_fails_closed_when_any_synthetic_source_is_unmatched():
    days = {
        runner.FIT_DAY: ("2026-02-28:f0_000",),
        runner.DEVELOPMENT_DAY: (),
        runner.REHEARSAL_DAY: (),
    }
    with pytest.raises(runner.StreamingContractError, match="match the open fit day exactly"):
        runner.derive_exact_splits(
            days,
            synthetic_exclusion_ids={"2026-02-28:not_present_000"},
        )


def test_sample_key_is_repo_location_independent():
    common = dict(
        day=runner.FIT_DAY,
        source_file="v2026.02.28-source.tar",
        footer_rows=1,
        source_id_count=1,
        flight_id_count=1,
    )
    left = runner.PartInventory(path=Path("C:/repo_a/silver/part-001.parquet"), **common)
    right = runner.PartInventory(path=Path("D:/repo_b/silver/part-001.parquet"), **common)
    assert runner.stable_part_sample_key(left) == runner.stable_part_sample_key(right)
    assert ":/repo" not in runner.stable_part_sample_key(left).lower()


@pytest.mark.parametrize("component", ["archive", "Downloads", "raw"])
def test_sealed_history_and_raw_path_components_are_rejected(tmp_path: Path, component: str):
    forbidden = tmp_path / component
    forbidden.mkdir()
    candidate = forbidden / "input.parquet"
    candidate.touch()
    with pytest.raises(runner.StreamingContractError, match="cannot be under"):
        runner._reject_sealed_or_raw_path(candidate, label="test input")


def test_causal_windows_never_cross_a_sixty_second_gap():
    times = np.r_[np.arange(13) * 10.0, 1_000.0 + np.arange(13) * 10.0]
    flight = pd.DataFrame(
        {
            "timestamp_utc": times,
            "vertical_rate_residual": np.ones(len(times)),
        }
    )
    windows = runner.causal_window_scores(
        flight,
        np.arange(len(times), dtype=float),
        active_rule_channels=("vertical_rate_residual",),
    )

    assert len(windows) == 2
    assert (windows["t_end"] - windows["t_start"] <= 110.0).all()
    assert runner.interval_union_exposure_seconds(windows) == pytest.approx(220.0)
    assert windows["observed_support_q"].eq(1.0).all()


def test_causal_window_reports_observed_channel_support_q():
    n = runner.WINDOW_SIZE
    flight = pd.DataFrame(
        {
            "timestamp_utc": np.arange(n, dtype=float),
            "vertical_rate_residual": [1.0] + [np.nan] * (n - 1),
            "speed_residual": [np.nan] * n,
        }
    )
    windows = runner.causal_window_scores(
        flight,
        np.zeros(n),
        active_rule_channels=("vertical_rate_residual", "speed_residual"),
    )
    assert len(windows) == 1
    assert windows.loc[0, "observed_active_channel_cells"] == 1
    assert windows.loc[0, "possible_active_channel_cells"] == 2 * n
    assert windows.loc[0, "observed_support_q"] == pytest.approx(1 / (2 * n))


def test_feature_part_nulls_every_residual_outside_sixty_second_transition(tmp_path: Path):
    times = np.array([0.0, 100.0, 110.0])
    frame = pd.DataFrame(
        {
            "_source_file": ["v2026.02.28-planes-readsb-prod-0.tar"] * 3,
            "source_id": ["a"] * 3,
            "timestamp_utc": times,
            "lat": [40.0, 40.001, 40.002],
            "lon": [29.0, 29.001, 29.002],
            "alt": [1_000.0, 1_010.0, 1_020.0],
            "alt_geom_m": [1_012.0, 1_022.5, 1_033.0],
            "on_ground": [False] * 3,
            "ground_speed_ms": [100.0, 101.0, 102.0],
            "track_deg": [45.0, 46.0, 47.0],
            "vertical_rate_ms": [1.0, 1.0, 1.0],
        }
    )
    path = tmp_path / "part.parquet"
    frame.to_parquet(path, index=False)
    part = runner.PartInventory(
        path=path,
        day=runner.FIT_DAY,
        source_file="v2026.02.28-planes-readsb-prod-0.tar",
        footer_rows=3,
        source_id_count=1,
        flight_id_count=1,
    )
    features = runner._read_feature_part(part)
    assert features["residual_transition_valid"].tolist() == [False, False, True]
    assert features.loc[:1, list(runner.ALL_RESIDUAL_CHANNELS)].isna().all().all()
    assert features.loc[2, list(runner.ALL_RESIDUAL_CHANNELS)].notna().all()


def test_small_part_smoke_fits_bounded_calibration_and_normal_burden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    part = _full_fit_part(tmp_path / "fit.parquet")
    splits = {
        "fit": ["2026-02-28:fit_000"],
        "calibration": ["2026-02-28:cal_000"],
        "development": [],
        "rehearsal": [],
    }
    monkeypatch.setattr(runner, "FIT_SAMPLE_PROBABILITY", 1.0)
    monkeypatch.setattr(runner, "BASELINE_SCORE_SAMPLE_PROBABILITY", 1.0)

    robust = runner.fit_sampled_robust_calibration([part], splits)
    scorer = runner.build_rule_scorer(robust)
    detector = runner.build_cusum_detector(robust, threshold_h=10.0)

    assert robust["fit_rows_seen"] == 36
    assert set(detector.calibration_) <= set(runner.VECTOR_RESIDUAL_FEATURES)
    assert detector.to_dict()["mad_zero_policy"] == "exclude"

    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 1000.0),
        advisory_budget_episodes_per_hour=1_000_000.0,
        bootstrap_repetitions=10,
        moving_block_s=300.0,
        moving_block_stride_s=150.0,
    )
    report, blocks = runner.calibrate_normal_burden(
        [part],
        splits,
        rule_scorer=scorer,
        robust=robust,
        contract=contract,
    )

    assert report["synthetic_used"] is False
    assert report["rule_diagnostic"]["score_sample_n"] > 0
    assert report["rule_diagnostic"]["empirical_quantile"] == pytest.approx(0.95)
    assert "confidence" not in report["rule_diagnostic"]
    assert report["cusum_natural_burden_selection"]["selected_h"] == 1.0
    assert not blocks.empty


def test_incomplete_cusum_axis_coverage_has_no_h_selection_or_gate(tmp_path: Path, monkeypatch):
    part = _full_fit_part(tmp_path / "fit.parquet")
    splits = {
        "fit": ["2026-02-28:fit_000"],
        "calibration": ["2026-02-28:cal_000"],
        "validation": [],
        "development": [],
        "rehearsal": [],
    }
    monkeypatch.setattr(runner, "FIT_SAMPLE_PROBABILITY", 1.0)
    monkeypatch.setattr(runner, "BASELINE_SCORE_SAMPLE_PROBABILITY", 1.0)
    robust = runner.fit_sampled_robust_calibration([part], splits)
    scorer = runner.build_rule_scorer(robust)
    degraded = copy.deepcopy(robust)
    degraded["calibration"].pop(runner.VECTOR_RESIDUAL_FEATURES[1], None)
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        bootstrap_repetitions=2,
    )

    report, _ = runner.calibrate_normal_burden(
        [part], splits, rule_scorer=scorer, robust=degraded, contract=contract
    )

    assert report["cusum_axis_coverage"]["status"] == "degraded_axis_coverage"
    assert report["cusum_natural_burden_selection"]["selected_h"] is None
    assert report["cusum_gate_status"] == "fail_degraded_axis_coverage"


def test_complete_axes_without_admissible_h_fail_gate_explicitly():
    decision = runner.cusum_gate_decision(
        {"gate_eligible": True, "status": "complete"},
        {"selected_h": None},
    )
    assert decision == {
        "gate_eligible": False,
        "gate_status": "fail_no_admissible_cusum_threshold",
    }


def test_primary_cusum_distribution_uses_only_evaluable_rows_and_cadence_is_reported():
    n = runner.WINDOW_SIZE
    features = pd.DataFrame(
        {
            "flight_id": ["F"] * n,
            "timestamp_utc": np.arange(n, dtype=float),
            **{channel: np.ones(n) for channel in runner.ALL_RESIDUAL_CHANNELS},
        }
    )
    scorer = runner.ResidualRuleScorer()
    scorer.calibration_ = {
        runner.RULE_CHANNELS[0]: {"median": 0.0, "mad": 1.0},
    }
    scorer.excluded_channels_ = list(runner.RULE_CHANNELS[1:])
    penalties = scorer.row_penalties(features).to_numpy(dtype=float)
    evaluable = np.r_[False, np.ones(n - 1, dtype=bool)]
    cusum_rows = pd.DataFrame(
        {
            "cusum_evaluable": evaluable,
            "cusum_joint_score": np.r_[999.0, np.ones(n - 1)],
            "cusum_reset_reason": ["flight_start"] + ["none"] * (n - 1),
            "cusum_observed_channels": np.r_[0, np.full(n - 1, 2)],
            **{
                f"{channel}_observed": evaluable
                for channel in runner.VECTOR_RESIDUAL_FEATURES
            },
        }
    )
    accumulator = runner.RoleBurdenAccumulator(day=runner.FIT_DAY, role="fit")
    part = runner.PartInventory(
        path=Path("part.parquet"),
        day=runner.FIT_DAY,
        source_file="source.tar",
        footer_rows=n,
        source_id_count=1,
        flight_id_count=1,
    )
    runner._update_role_burden(
        accumulator,
        part=part,
        features=features,
        rule_penalties=penalties,
        rule_scorer=scorer,
        rule_threshold=1.0,
        cusum_rows=cusum_rows,
        cusum_threshold=10.0,
    )
    report = accumulator.report(rule_threshold=1.0, cusum_threshold=10.0)
    distribution = report["channel_and_score_distributions"]["cusum_joint_score"]
    assert distribution["finite_n"] == n - 1
    assert distribution["max"] == 1.0
    assert report["cusum_cadence_strata"]["le_2s"]["n_scoreable_flights"] == 1


def test_json_writer_rejects_nonfinite_values_and_refuses_reuse(tmp_path: Path):
    target = tmp_path / "result.json"
    with pytest.raises(ValueError, match="non-finite"):
        runner.write_json_exclusive(target, {"nan": float("nan")})
    assert not target.exists()
    runner.write_json_exclusive(target, {"ok": True})
    with pytest.raises(FileExistsError):
        runner.write_json_exclusive(target, {"ok": True})


def test_frozen_code_hashes_cover_every_registered_runtime_file():
    repo_root = Path(runner.__file__).resolve().parent.parent
    hashes = runner.frozen_code_hashes(repo_root)
    assert tuple(hashes) == runner.FROZEN_CODE_FILES
    assert all(len(value) == 64 for value in hashes.values())


def test_artifact_checksum_index_is_last_and_self_excluded(tmp_path: Path):
    (tmp_path / "one.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("evidence\n", encoding="utf-8")
    target = runner.write_artifact_checksums(tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload["files"]) == {"one.json", "two.txt"}
    assert payload["self_excluded"] is True
    with pytest.raises(FileExistsError):
        runner.write_artifact_checksums(tmp_path)


def test_run_writes_derived_freeze_before_evaluation_and_final_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_path = tmp_path / "part.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(input_path, index=False)
    synthetic_dir = tmp_path / "synthetic_reference"
    synthetic_dir.mkdir()
    synthetic_path = synthetic_dir / "clean.parquet"
    pd.DataFrame({"flight_id": ["fitid"]}).to_parquet(synthetic_path, index=False)
    part = runner.PartInventory(
        path=input_path,
        day=runner.FIT_DAY,
        source_file="v2026.02.28-source.tar",
        footer_rows=1,
        source_id_count=1,
        flight_id_count=1,
    )
    inventory = runner.PreflightInventory(
        parts=(part,),
        flight_ids_by_day={
            runner.FIT_DAY: ("fitid",),
            runner.DEVELOPMENT_DAY: (),
            runner.REHEARSAL_DAY: (),
        },
        part_counts_by_day={runner.FIT_DAY: 1},
        footer_rows_by_day={runner.FIT_DAY: 1},
        selected_schema={column: "double" for column in runner.SILVER_COLUMNS},
    )
    inventory.selected_schema["_source_file"] = "string"
    inventory.selected_schema["source_id"] = "string"
    inventory.selected_schema["on_ground"] = "bool"
    splits = {
        "fit": [],
        "calibration": [],
        "validation": ["fitid"],
        "development": [],
        "rehearsal": [],
    }
    robust = {
        "calibration": {
            channel: {"median": 0.0, "mad": 1.0, "sample_n": 10}
            for channel in runner.ALL_RESIDUAL_CHANNELS
        },
        "excluded_channels": [],
        "mad_zero_policy": "exclude",
    }
    threshold = {
        "rule_diagnostic": {
            "selected_threshold": 2.5,
            "empirical_quantile": runner.RULE_DIAGNOSTIC_EMPIRICAL_QUANTILE,
        },
        "cusum_axis_coverage": {
            "active_axes": list(runner.VECTOR_RESIDUAL_FEATURES),
            "excluded_axes": [],
            "gate_eligible": True,
            "status": "complete",
        },
        "cusum_natural_burden_selection": {
            "selected_h": 5.0,
            "observed_burden_source": "full_flight_counters",
        },
        "cusum_gate_eligible": True,
        "cusum_gate_status": "eligible_for_step_7_review",
    }
    frozen_hashes = {relative: "0" * 64 for relative in runner.FROZEN_CODE_FILES}

    monkeypatch.setattr(runner, "inventory_open_silver", lambda _: inventory)
    monkeypatch.setattr(
        runner,
        "load_synthetic_exclusion_ids",
        lambda _: ({"fitid"}, [{"path": synthetic_path.as_posix()}]),
    )
    monkeypatch.setattr(
        runner,
        "derive_exact_splits",
        lambda *_args, **_kwargs: (splits, {"match": "exact"}),
    )
    monkeypatch.setattr(runner, "frozen_code_hashes", lambda _root: frozen_hashes)

    def fake_manifest(*, run_dir, **_kwargs):
        run_dir.mkdir()
        path = run_dir / "run_manifest.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "create_immutable_run_manifest", fake_manifest)
    monkeypatch.setattr(runner, "fit_sampled_robust_calibration", lambda *_: robust)
    monkeypatch.setattr(
        runner,
        "calibrate_normal_burden",
        lambda *_args, **_kwargs: (
            threshold,
            pd.DataFrame({"exposure_s": [1.0], "h_1": [0]}),
        ),
    )

    def fake_evaluation(*_args, **_kwargs):
        destination = tmp_path / "run"
        assert (destination / "derived_frozen_config.json").is_file()
        assert (destination / "derived_frozen_config.sha256").is_file()
        return {"settings_frozen_before_development_and_rehearsal": True, "roles": {}}

    monkeypatch.setattr(runner, "evaluate_natural_days", fake_evaluation)
    destination = tmp_path / "run"
    report = runner.run(
        repo_root=Path(runner.__file__).resolve().parent.parent,
        silver_dir=tmp_path,
        synthetic_clean_references=[synthetic_path],
        run_dir=destination,
    )

    derived = json.loads((destination / "derived_frozen_config.json").read_text("utf-8"))
    assert set(derived) == {"schema_version", "derived_config", "payload_sha256"}
    assert derived["payload_sha256"] == runner.sha256_json(derived["derived_config"])
    assert report["gate_status"] == "evidence_only_pending_step_7_review"
    checksums = json.loads((destination / "artifact_checksums.json").read_text("utf-8"))
    assert "streaming_baseline_report.json" in checksums["files"]
    assert "derived_frozen_config.sha256" in checksums["files"]

