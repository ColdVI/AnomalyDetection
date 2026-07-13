from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.adsb_run_full_streaming_baseline as runner
from adsb.streaming import CusumBurdenCalibration


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
