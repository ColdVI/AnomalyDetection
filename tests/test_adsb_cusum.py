"""ADS-B CUSUM ve truth-v2 degerlendirme testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from __future__ import annotations

from math import sqrt

import numpy as np

import pandas as pd

import pytest

from adsb.cusum import ROBUST_MAD_SCALE, CusumConfig, VectorPageCUSUM

from adsb.features import VECTOR_RESIDUAL_FEATURES

import json

from pathlib import Path

import scripts.adsb_audit_cusum_bootstrap_upper as audit

from adsb.run_manifest import sha256_file, sha256_json

from adsb.cusum_truth_v2_eval import (
    BoundedScoreReservoir,
    CusumTruthV2ContractError,
    iter_exact_paired_flights,
    load_frozen_step5_bundle,
    run_evaluation,
    sampled_binary_diagnostics,
)

from adsb.synthetic import inject_position_ramp

from adsb.truth import attach_clean_truth_v2



# ===== kaynak: test_adsb_cusum =====

EAST, NORTH = VECTOR_RESIDUAL_FEATURES


def _config(**overrides) -> CusumConfig:
    values = {
        "target_vector_shift_mps": 2.0,
        "threshold_h": 1.0,
        "max_gap_s": 60.0,
        "missing_reset_s": 3.0,
        "z_clip": 3.0,
    }
    values.update(overrides)
    return CusumConfig(**values)


def _features(
    times,
    east,
    north,
    *,
    flights=None,
    on_ground=None,
) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "flight_id": flights if flights is not None else ["F1"] * n,
            "timestamp_utc": times,
            "on_ground": on_ground if on_ground is not None else [False] * n,
            EAST: east,
            NORTH: north,
        }
    )


def _normal_train() -> pd.DataFrame:
    # The first value is a deliberately extreme flight-start row and is not an
    # eligible transition.  The remaining ten values have median 0 and raw
    # MAD 2, so robust MAD is exactly 2 * 1.4826.
    values = [99.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    return _features(np.arange(len(values), dtype=float), values, values)


def _fit_detector(**config_overrides) -> VectorPageCUSUM:
    return VectorPageCUSUM(_config(**config_overrides)).fit(_normal_train())


def test_fit_uses_train_only_robust_median_mad_and_physical_k():
    detector = _fit_detector()
    expected_mad = 2.0 * ROBUST_MAD_SCALE
    for channel in (EAST, NORTH):
        calibration = detector.calibration_[channel]
        assert calibration["median"] == 0.0
        assert calibration["mad"] == pytest.approx(expected_mad)
        assert calibration["k"] == pytest.approx((2.0 / sqrt(2.0)) / (2.0 * expected_mad))


def test_zero_mad_channel_is_excluded_without_floor():
    train = _normal_train()
    train[NORTH] = 0.0
    detector = VectorPageCUSUM(_config()).fit(train)
    assert detector.excluded_channels_ == {NORTH: "mad_zero"}
    assert NORTH not in detector.calibration_

    scored = detector.score_rows(_features([0.0, 1.0], [0.0, 4.0], [0.0, 1_000.0]))
    assert scored[f"{NORTH}_cusum_positive"].isna().all()
    assert scored[f"{NORTH}_cusum_negative"].isna().all()
    assert scored["cusum_observed_channels"].tolist() == [0, 1]

    serialized = detector.to_dict()
    assert serialized["state_count"] == 2
    assert serialized["configured_state_count"] == 4
    assert serialized["axis_coverage_status"] == "degraded_axis_coverage"


def test_signed_page_updates_match_formula_and_joint_alarm_uses_four_states():
    detector = _fit_detector(z_clip=10.0, threshold_h=0.5)
    calibration = detector.calibration_[EAST]
    mad = calibration["mad"]
    k = calibration["k"]
    test = _features(
        [0.0, 1.0, 2.0],
        [0.0, 2.0 * mad, -2.0 * mad],
        [0.0, 0.0, 0.0],
    )
    scored = detector.score_rows(test)

    east_positive = f"{EAST}_cusum_positive"
    east_negative = f"{EAST}_cusum_negative"
    state_columns = {
        f"{EAST}_cusum_positive",
        f"{EAST}_cusum_negative",
        f"{NORTH}_cusum_positive",
        f"{NORTH}_cusum_negative",
    }
    assert state_columns.issubset(scored.columns)
    assert scored.loc[1, east_positive] == pytest.approx(2.0 - k)
    assert scored.loc[1, east_negative] == 0.0
    assert scored.loc[2, east_positive] == 0.0
    assert scored.loc[2, east_negative] == pytest.approx(2.0 - k)
    assert scored.loc[1, "cusum_joint_score"] == scored.loc[1, east_positive]
    assert scored.loc[1, "cusum_joint_alarm"]
    assert scored.loc[2, "cusum_joint_alarm"]


def test_joint_alarm_uses_strictly_greater_than_frozen_h():
    provisional = _fit_detector(z_clip=10.0)
    mad = provisional.calibration_[EAST]["mad"]
    exact_state = 2.0 - provisional.calibration_[EAST]["k"]
    detector = _fit_detector(z_clip=10.0, threshold_h=exact_state)
    scored = detector.score_rows(
        _features([0.0, 1.0], [0.0, 2.0 * mad], [0.0, 0.0])
    )
    assert scored.loc[1, "cusum_joint_score"] == pytest.approx(exact_state)
    assert not scored.loc[1, "cusum_joint_alarm"]


def test_global_resets_cover_ground_time_and_flight_boundaries():
    detector = _fit_detector(z_clip=10.0)
    high = 4.0 * detector.calibration_[EAST]["mad"]
    test = _features(
        [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 99.0, 99.0, 200.0],
        [0.0, high, high, high, high, high, high, high, high],
        [0.0] * 9,
        flights=["F1"] * 8 + ["F2"],
        on_ground=[False, False, True, False, False, False, False, False, False],
    )
    scored = detector.score_rows(test)
    east_positive = f"{EAST}_cusum_positive"

    assert scored["cusum_reset_reason"].tolist() == [
        "flight_start",
        "none",
        "on_ground",
        "ground_transition",
        "none",
        "long_gap",
        "negative_dt",
        "zero_dt",
        "flight_start",
    ]
    assert scored.loc[1, east_positive] > 0.0
    assert (scored.loc[[2, 3, 5, 6, 8], east_positive] == 0.0).all()
    assert not scored.loc[[0, 2, 3, 5, 6, 7, 8], "cusum_evaluable"].any()


def test_zero_dt_skips_update_without_resetting_accumulated_state():
    detector = _fit_detector(z_clip=10.0)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features([0.0, 1.0, 1.0, 2.0], [0.0, high, 100.0 * high, high], [0.0] * 4)
    )
    state = f"{EAST}_cusum_positive"
    assert scored.loc[2, "cusum_reset_reason"] == "zero_dt"
    assert not scored.loc[2, "cusum_evaluable"]
    assert scored.loc[2, state] == scored.loc[1, state]
    assert scored.loc[3, state] > scored.loc[2, state]


def test_missing_state_is_carried_then_channel_reset_by_elapsed_time():
    detector = _fit_detector(z_clip=10.0, missing_reset_s=2.5, threshold_h=0.5)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, high, np.nan, np.nan, np.nan],
            [0.0] * 5,
        )
    )
    state = f"{EAST}_cusum_positive"
    missing_reset = f"{EAST}_missing_reset"

    assert scored.loc[2, state] == scored.loc[1, state]
    assert scored.loc[3, state] == scored.loc[1, state]
    assert not scored.loc[2:3, missing_reset].any()
    assert scored.loc[4, missing_reset]
    assert scored.loc[4, state] == 0.0


def test_all_channels_missing_is_not_evaluable_and_cannot_emit_fresh_alarm():
    detector = _fit_detector(z_clip=10.0, missing_reset_s=30.0, threshold_h=0.5)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features([0.0, 1.0, 2.0], [0.0, high, np.nan], [0.0, 0.0, np.nan])
    )
    assert scored.loc[2, "cusum_joint_score"] > 0.0
    assert not scored.loc[2, "cusum_evaluable"]
    assert not scored.loc[2, "cusum_joint_alarm"]


def test_unknown_ground_status_resets_and_is_not_evaluable():
    detector = _fit_detector(z_clip=10.0)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features([0.0, 1.0, 2.0], [0.0, high, high], [0.0] * 3, on_ground=[False, None, False])
    )
    assert scored["cusum_reset_reason"].tolist() == [
        "flight_start",
        "unknown_ground_status",
        "unknown_ground_status",
    ]
    assert not scored["cusum_evaluable"].any()
    assert scored["cusum_joint_score"].eq(0.0).all()


def test_scoring_is_prefix_invariant():
    detector = _fit_detector(z_clip=10.0, missing_reset_s=2.5)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    test = _features(
        [0.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        [0.0, high, np.nan, 100.0 * high, -high, -high],
        [0.0, 0.0, 0.0, 100.0 * high, 0.0, 0.0],
    )
    full = detector.score_rows(test)
    prefix = detector.score_rows(test.iloc[:5])
    pd.testing.assert_frame_equal(full.iloc[:5], prefix)


def test_roundtrip_preserves_config_calibration_and_scores():
    detector = _fit_detector(z_clip=10.0)
    clone = VectorPageCUSUM.from_dict(detector.to_dict())
    test = _features([0.0, 1.0, 2.0], [0.0, 5.0, -5.0], [0.0, 1.0, -1.0])
    pd.testing.assert_frame_equal(detector.score_rows(test), clone.score_rows(test))
    assert clone.to_dict() == detector.to_dict()
    assert clone.to_dict()["axis_coverage_status"] == "complete_two_axis"


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_vector_shift_mps": 0.0},
        {"threshold_h": 0.0},
        {"max_gap_s": -1.0},
        {"missing_reset_s": -1.0},
        {"z_clip": 0.0},
        {"channels": (EAST, EAST)},
    ],
)
def test_config_rejects_invalid_or_non_four_state_contract(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)



# ===== kaynak: test_adsb_cusum_bootstrap_upper_audit =====

REPO_ROOT = Path(__file__).parent.parent


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selection(*, observed_h1: float = 0.6) -> dict:
    return {
        "selected_h": 1.0,
        "selection_rule": "legacy raw bootstrap p95",
        "budget_episodes_per_hour": 0.5,
        "n_blocks": 2,
        "observed_burden_source": "full_flight_counters",
        "observed_exposure_hours": 10.0,
        "bootstrap_repetitions": 500,
        "bootstrap_batch_size": 4,
        "bootstrap_upper_quantile": 0.95,
        "candidates": [
            {
                "h": 1.0,
                "observed_episode_count": 6,
                "observed_episodes_per_hour": observed_h1,
                "bootstrap_upper_95_episodes_per_hour": 0.4,
                "meets_advisory_budget": True,
            },
            {
                "h": 2.0,
                "observed_episode_count": 1,
                "observed_episodes_per_hour": 0.1,
                "bootstrap_upper_95_episodes_per_hour": 0.2,
                "meets_advisory_budget": True,
            },
        ],
    }


def _source_run(path: Path, *, observed_h1: float = 0.4) -> Path:
    path.mkdir()
    selection = _selection(observed_h1=observed_h1)
    contract = {
        "candidate_h": [1.0, 2.0],
        "advisory_budget_episodes_per_hour": 0.5,
        "bootstrap_repetitions": 500,
        "bootstrap_batch_size": 4,
        "bootstrap_seed": 20260713,
        "upper_quantile": 0.95,
    }
    _write_json(
        path / "run_manifest.json",
        {
            "run_id": path.name,
            "config": {"value": {"cusum": {"burden_calibration": contract}}},
        },
    )
    _write_json(
        path / "normal_burden_calibration.json",
        {"cusum_natural_burden_selection": selection},
    )
    normal_sha256 = sha256_file(path / "normal_burden_calibration.json")
    derived_config = {
        "normal_burden_calibration_file_sha256": normal_sha256,
        "cusum": {"selection": selection},
    }
    _write_json(
        path / "derived_frozen_config.json",
        {
            "schema_version": 1,
            "derived_config": derived_config,
            "payload_sha256": sha256_json(derived_config),
        },
    )
    derived_sha256 = sha256_file(path / "derived_frozen_config.json")
    (path / "derived_frozen_config.sha256").write_text(
        derived_sha256 + "\n", encoding="ascii"
    )
    _write_json(
        path / "streaming_baseline_report.json",
        {
            "run_id": path.name,
            "derived_frozen_config": {
                "file_sha256": derived_sha256,
                "payload_sha256": sha256_json(derived_config),
                "sidecar": "derived_frozen_config.sha256",
            },
            "normal_threshold_calibration": {
                "cusum_natural_burden_selection": selection
            },
        },
    )
    indexed = {}
    for name in audit.INDEXED_SOURCE_FILENAMES:
        indexed[name] = {
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
    _write_json(
        path / "artifact_checksums.json",
        {
            "schema_version": 1,
            "algorithm": "sha256",
            "self_excluded": True,
            "files": indexed,
        },
    )
    return path


def test_corrected_selection_uses_max_and_can_change_selected_h():
    result = audit.corrected_selection(_selection(observed_h1=0.6))

    assert result["old_selected_h"] == 1.0
    assert result["new_selected_h"] == 2.0
    assert result["selected_h_unchanged"] is False
    first = result["candidates"][0]
    assert first["old_raw_bootstrap_quantile_95_episodes_per_hour"] == 0.4
    assert first["conservative_upper_95_episodes_per_hour"] == 0.6
    assert first["old_meets_advisory_budget"] is True
    assert first["new_meets_advisory_budget"] is False


def test_audit_run_manifests_chain_is_fail_if_exists_and_preserves_selection(tmp_path: Path):
    source = _source_run(tmp_path / "source-v1", observed_h1=0.4)
    destination = tmp_path / "audit-v1"

    report = audit.run(
        repo_root=REPO_ROOT,
        source_run_dir=source,
        run_dir=destination,
    )

    assert report["status"] == "passed_selected_h_unchanged"
    assert report["scope"]["scientific_rerun"] is False
    assert report["source"]["artifacts_preserved"] is True
    assert report["selection_comparison"]["old_selected_h"] == 1.0
    assert report["selection_comparison"]["new_selected_h"] == 1.0
    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert {entry["path"].rsplit("/", 1)[-1] for entry in manifest["inputs"]} == set(
        audit.SOURCE_FILENAMES
    )
    assert (destination / "artifact_checksums.json").is_file()

    with pytest.raises(FileExistsError):
        audit.run(
            repo_root=REPO_ROOT,
            source_run_dir=source,
            run_dir=destination,
        )


def test_source_chain_tamper_is_rejected_before_audit_directory_creation(tmp_path: Path):
    source = _source_run(tmp_path / "source-v1")
    (source / "normal_burden_calibration.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "audit-v1"

    with pytest.raises(audit.AuditContractError, match="checksum mismatch"):
        audit.run(
            repo_root=REPO_ROOT,
            source_run_dir=source,
            run_dir=destination,
        )
    assert not destination.exists()



# ===== kaynak: test_adsb_cusum_truth_v2_evaluation =====

# (REPO_ROOT: adsb_cusum_bootstrap_upper_audit ile ayni/esdeger, tekrar atlandi)

STEP5 = REPO_ROOT / "artifacts/adsb/runs/20260713_step5_full_streaming_v1"


def _clean_flight(flight_id: str, offset: float) -> pd.DataFrame:
    n = 24
    timestamp = offset + np.arange(n, dtype=float)
    # Approximately northbound motion with internally consistent speed/track.
    lat = 40.0 + np.arange(n, dtype=float) * (50.0 / 111_320.0)
    return pd.DataFrame(
        {
            "flight_id": flight_id,
            "timestamp_utc": timestamp,
            "lat": lat,
            "lon": np.full(n, 29.0),
            "alt": np.full(n, 1000.0),
            "alt_geom_m": np.full(n, 1010.0),
            "ground_speed_ms": np.full(n, 50.0),
            "track_deg": np.zeros(n),
            "vertical_rate_ms": np.zeros(n),
            "roll_deg": np.zeros(n),
            "on_ground": np.zeros(n, dtype=bool),
        }
    )


def _write_corpus(root: Path, *, corrupt_timestamp: bool = False) -> Path:
    corpus = root / "synthetic_fixture" / "adsb_v2"
    corpus.mkdir(parents=True)
    clean_raw = pd.concat(
        [_clean_flight("f1", 0.0), _clean_flight("f2", 100.0)], ignore_index=True
    )
    clean = pd.concat(
        [attach_clean_truth_v2(group.reset_index(drop=True)) for _, group in clean_raw.groupby("flight_id", sort=False)],
        ignore_index=True,
    )
    corrupt = pd.concat(
        [
            inject_position_ramp(
                group.reset_index(drop=True),
                meters_per_s=2.0,
                event_type="position_ramp_stealthy",
            )
            for _, group in clean_raw.groupby("flight_id", sort=False)
        ],
        ignore_index=True,
    )
    if corrupt_timestamp:
        corrupt.loc[3, "timestamp_utc"] += 0.25
    clean_path = corpus / "clean.parquet"
    corrupt_path = corpus / "position_ramp_stealthy.parquet"
    clean.to_parquet(clean_path, index=False)
    corrupt.to_parquet(corrupt_path, index=False)

    outputs = []
    for recipe, path in (
        ("clean", clean_path),
        ("position_ramp_stealthy", corrupt_path),
    ):
        outputs.append(
            {
                "recipe": recipe,
                "path": path.name,
                "n_rows": len(clean),
                "footer_rows": len(clean),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "adsb_synthetic_truth_v2",
                "synthetic_never_training": True,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return corpus


def _write_step5_fixture(root: Path, flight_ids: list[str]) -> Path:
    directory = root / "step5_fixture"
    directory.mkdir(parents=True)
    actual_derived = json.loads((STEP5 / "derived_frozen_config.json").read_text())
    selected = actual_derived["derived_config"]["cusum"]["selected_detector"]
    base_config = {
        "cusum": {"synthetic_selection_forbidden": True},
        "frozen_code_sha256": {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in ("adsb/cusum.py", "adsb/features.py")
        },
    }
    base_hash = sha256_json(base_config)
    validation_ids = sorted(f"2026-02-28:{value}" for value in flight_ids)
    normalized = {"validation": validation_ids}
    split_payload = {"algorithm": "fixture", "seed": None, "splits": normalized}
    manifest = {
        "manifest_schema_version": 1,
        "config": {"sha256": base_hash, "value": base_config},
        "inputs": [],
        "input_contract_sha256": sha256_json([]),
        "split_contract": {
            "algorithm": "fixture",
            "seed": None,
            "contract_sha256": sha256_json(split_payload),
            "splits": {
                "validation": {
                    "flight_id_count": len(validation_ids),
                    "flight_ids_sha256": sha256_json(validation_ids),
                    "flight_ids": validation_ids,
                }
            },
        },
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    derived = {
        "base_config_sha256": base_hash,
        "cusum": {
            "axis_coverage": {"status": "complete", "active_axis_count": 2},
            "selected_h": 1.0,
            "selected_detector": selected,
            "gate_eligible": True,
        },
        "later_role_feedback_forbidden": True,
        "frozen_before_roles": ["development", "rehearsal"],
    }
    derived_record = {
        "schema_version": 1,
        "derived_config": derived,
        "payload_sha256": sha256_json(derived),
    }
    derived_path = directory / "derived_frozen_config.json"
    derived_path.write_text(json.dumps(derived_record), encoding="utf-8")
    derived_file_hash = sha256_file(derived_path)
    (directory / "derived_frozen_config.sha256").write_text(
        derived_file_hash + "\n", encoding="utf-8"
    )

    roles = {}
    for role, day, n_flights in (
        ("validation", "2026-02-28", len(validation_ids)),
        ("development", "2026-03-01", 3),
        ("rehearsal", "2026-03-16", 3),
    ):
        roles[role] = {
            "role": role,
            "day": day,
            "cusum_event_unit": {
                "selected_h": 1.0,
                "emission_time": "row_timestamp_utc",
                "merge_gap_s": 60.0,
                "n_alert_episodes": 1,
                "episodes_per_scoreable_flight_hour": 1.0,
            },
            "cusum_flight_unit": {
                "n_input_flights": n_flights,
                "n_scoreable_flights": n_flights,
                "n_alerted_flights": 1,
                "scoreable_flight_hours": 1.0,
            },
            "cusum_cadence_strata": {},
        }
    natural = {
        "settings_frozen_before_development_and_rehearsal": True,
        "rehearsal_feedback_into_settings": False,
        "roles": roles,
    }
    (directory / "natural_burden_by_role.json").write_text(
        json.dumps(natural), encoding="utf-8"
    )
    report = {
        "manifest": "run_manifest.json",
        "artifact_checksum_index": "artifact_checksums.json",
        "config_sha256": base_hash,
        "config_and_code_unchanged_through_evaluation": True,
        "rehearsal_changed_parameters": False,
        "synthetic_training_rows": 0,
        "gate_status": "evidence_only_pending_step_7_review",
        "derived_frozen_config": {
            "file_sha256": derived_file_hash,
            "payload_sha256": derived_record["payload_sha256"],
        },
        "natural_burden": natural,
    }
    (directory / "streaming_baseline_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    indexed = {}
    for name in (
        "run_manifest.json",
        "derived_frozen_config.json",
        "derived_frozen_config.sha256",
        "streaming_baseline_report.json",
        "natural_burden_by_role.json",
    ):
        path = directory / name
        indexed[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (directory / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "self_excluded": True,
                "files": indexed,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_real_step5_bundle_loads_exact_frozen_h1_detector_and_burden_panel():
    bundle = load_frozen_step5_bundle(STEP5, repo_root=REPO_ROOT)
    assert bundle.selected_h == 1.0
    assert bundle.detector.config.threshold_h == 1.0
    assert bundle.selected_detector["mad_zero_policy"] == "exclude"
    assert bundle.selected_detector["excluded_channels"] == {}
    assert len(bundle.validation_flight_ids) == 8910
    assert bundle.burden_panel["primary_role"] == "validation"
    assert set(bundle.burden_panel["temporal_stability_panel"]) == {
        "development",
        "rehearsal",
    }


def test_step5_chain_fails_closed_when_an_indexed_artifact_is_tampered(tmp_path: Path):
    copied = _write_step5_fixture(tmp_path, ["f1", "f2"])
    with (copied / "natural_burden_by_role.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(CusumTruthV2ContractError, match="byte count changed"):
        load_frozen_step5_bundle(copied, repo_root=REPO_ROOT)


def test_bounded_diagnostic_sample_has_hard_cap_and_population_weighting():
    negative = BoundedScoreReservoir(25, 7, "negative")
    positive = BoundedScoreReservoir(25, 7, "positive")
    negative.add(np.arange(100, dtype=float), source_key="clean")
    positive.add(np.arange(100, 200, dtype=float), source_key="corrupt")
    result = sampled_binary_diagnostics(negative, positive)
    assert len(negative.scores) == 25
    assert len(positive.scores) == 25
    assert result["population_n_negative"] == 100
    assert result["population_n_positive"] == 100
    assert result["auroc_estimate"] == 1.0
    assert result["auprc_estimate_population_weighted"] == 1.0
    assert result["never_used_for_threshold_or_selection"] is True


def test_exact_pairing_rejects_timestamp_drift(tmp_path: Path):
    corpus = _write_corpus(tmp_path, corrupt_timestamp=True)
    with pytest.raises(CusumTruthV2ContractError, match="timestamps differ"):
        list(
            iter_exact_paired_flights(
                corpus / "clean.parquet",
                corpus / "position_ramp_stealthy.parquet",
            )
        )


def test_small_frozen_cusum_truth_v2_run_is_immutable_and_pairs_burden(tmp_path: Path):
    corpus = _write_corpus(tmp_path)
    step5 = _write_step5_fixture(tmp_path, ["f1", "f2"])
    run_dir = tmp_path / "run"
    summary_path = run_evaluation(
        repo_root=REPO_ROOT,
        step5_dir=step5,
        corpus_dir=corpus,
        run_dir=run_dir,
        recipes=("position_ramp_stealthy",),
        sample_capacity_per_class=100,
        sample_seed=11,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recipe = summary["per_recipe"]["position_ramp_stealthy"]
    assert summary["status"] == "complete_evaluation_only"
    assert summary["frozen_detector"]["selected_h"] == 1.0
    assert summary["frozen_detector"]["fit_performed_in_this_run"] is False
    assert recipe["synthetic_detection"]["corrupt_q0_timeline_sanity"][
        "included_as_auc_negative"
    ] is False
    assert recipe["diagnostic_sampled_auc_auprc"][
        "negative_source"
    ].startswith("single_unmodified_clean_reference")
    assert recipe["paired_natural_cusum_burden"]["primary_role"] == "validation"
    assert set(recipe["paired_natural_cusum_burden"]["temporal_stability_panel"]) == {
        "development",
        "rehearsal",
    }
    assert summary["stealthy_ramp_focus"]["available"] is True
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "event_table.parquet").is_file()
    checksum = json.loads((run_dir / "artifact_checksums.json").read_text())
    assert checksum["status"] == "complete"
    assert checksum["files"]["event_table.parquet"]["sha256"] == sha256_file(
        run_dir / "event_table.parquet"
    )

    with pytest.raises(FileExistsError):
        run_evaluation(
            repo_root=REPO_ROOT,
            step5_dir=step5,
            corpus_dir=corpus,
            run_dir=run_dir,
            recipes=("position_ramp_stealthy",),
            sample_capacity_per_class=100,
            sample_seed=11,
        )

