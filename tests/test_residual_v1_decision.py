"""RESIDUAL-V1 karar/olcekleme/sanity testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

import numpy as np

import pandas as pd

from residual_v1.decision.calibrate import (
    CalibrationConfig,
    InsufficientCalibrationExposure,
    calibrate_channel_threshold,
)

import pytest

from residual_v1.decision.cusum import score_cusum_channel, threshold_crossing_alarms

from residual_v1.decision.scaling import ZeroMADChannel, fit_robust_scaler, robust_z

from scripts.residual_v1_sanity_plots import _spearman_or_none

from residual_v1.eval.sanity_gates import (
    GateError,
    require_s3_pass,
    s1_magnitude_gate,
    s3_event_separation_gate,
)

from residual_v1.eval.s4_ablation import command_ablation_report

from residual_v1.features.spec import ResidualChannelSpec

from residual_v1.viz.handout import chunked, downsample, flight_slug, split_scope



# ===== kaynak: test_residual_v1_cusum =====

def test_wrapper_reuses_two_sided_core_for_positive_and_negative_shifts():
    frame = pd.DataFrame(
        {
            "flight_id": ["positive"] * 4 + ["negative"] * 4,
            "t": list(range(4)) * 2,
            "z": [0.0, 3.0, 3.0, 3.0, 0.0, -3.0, -3.0, -3.0],
        }
    )
    scored = score_cusum_channel(frame, channel="c", k=1.0, max_gap_s=2.0)
    assert np.allclose(scored.loc[scored["flight_id"] == "positive", "cusum_score"], [0, 2, 4, 6])
    assert np.allclose(scored.loc[scored["flight_id"] == "negative", "cusum_score"], [0, 2, 4, 6])


def test_alarm_is_a_rising_crossing_not_a_repeated_above_threshold_row():
    frame = pd.DataFrame({"flight_id": "f", "t": np.arange(5.0), "z": [0, 3, 3, 3, 3]})
    scored = score_cusum_channel(frame, channel="c", max_gap_s=2.0)
    alarms = threshold_crossing_alarms(scored, threshold=3.0, refractory_s=60.0)
    assert len(alarms) == 1
    assert alarms.iloc[0]["t_alarm"] == 2.0


def test_bootstrap_calibration_returns_rate_at_or_below_target():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "flight_id": np.repeat(["f1", "f2", "f3"], 121),
            "t": np.tile(np.arange(121.0), 3),
            "z": rng.normal(size=363),
        }
    )
    report = calibrate_channel_threshold(
        frame,
        channel="c",
        target_alarms_per_hour=12.0,
        config=CalibrationConfig(repetitions=20, seed=3),
    )
    assert report["threshold_h"] > 0.0
    assert report["bootstrap_rate_mean"] <= 12.0 + 1e-9
    assert len(report["bootstrap_rates"]) == 20


def test_calibration_refuses_target_below_one_alarm_resolution():
    frame = pd.DataFrame(
        {"flight_id": "f", "t": np.arange(61.0), "z": np.zeros(61)}
    )
    with pytest.raises(InsufficientCalibrationExposure, match="minimum one-alarm exposure"):
        calibrate_channel_threshold(
            frame,
            channel="c",
            target_alarms_per_hour=0.5,
            config=CalibrationConfig(repetitions=5),
        )



# ===== kaynak: test_residual_v1_scaling =====

def test_scaler_fits_only_train_normal_and_clips_at_eight():
    residual = pd.Series([-2.0, 0.0, 2.0, 1000.0])
    train = pd.Series([True, True, True, False])
    params = fit_robust_scaler(residual, train, channel="c")
    assert params.median == 0.0
    assert params.mad == 2.0
    assert params.fit_rows == 3
    assert np.allclose(robust_z(residual, params), [-1.0, 0.0, 1.0, 8.0])


def test_zero_mad_is_an_explicit_exclusion():
    with pytest.raises(ZeroMADChannel, match="MAD is zero"):
        fit_robust_scaler(pd.Series([1.0, 1.0, 9.0]), pd.Series([True, True, False]), channel="c")


def test_non_train_outlier_does_not_change_parameters():
    base = fit_robust_scaler(pd.Series([0.0, 1.0, 2.0]), pd.Series([True, True, True]), channel="c")
    extended = fit_robust_scaler(
        pd.Series([0.0, 1.0, 2.0, 1e9]),
        pd.Series([True, True, True, False]),
        channel="c",
    )
    assert base == extended



# ===== kaynak: test_residual_v1_sanity =====

def test_sanity_spearman_marks_constant_input_without_nan():
    statistic, status = _spearman_or_none(
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series([1.0, 2.0, 3.0]),
    )
    assert statistic is None
    assert status == "constant_input"



# ===== kaynak: test_residual_v1_sanity_gates =====

def _flight_frame(scores: list[float], magnitudes: list[float]) -> pd.DataFrame:
    parts = []
    for index, (score, magnitude) in enumerate(zip(scores, magnitudes, strict=True)):
        parts.append(
            pd.DataFrame(
                {
                    "flight_id": f"f{index}",
                    "z": [score, -score],
                    "input_magnitude": [magnitude, magnitude],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_s1_flags_magnitude_ordering_at_threshold():
    result = s1_magnitude_gate(
        _flight_frame([1, 2, 3, 4], [10, 20, 30, 40]),
        dataset="d",
        channel="c",
    )
    assert result.status == "flagged"
    assert np.isclose(result.metrics["spearman_rho"], 1.0)


def test_s1_passes_independent_flight_ordering():
    result = s1_magnitude_gate(
        _flight_frame([1, 2, 3, 4, 5], [30, 10, 50, 20, 40]),
        dataset="d",
        channel="c",
    )
    assert result.status == "passed"
    assert result.metrics["spearman_rho"] < 0.5


def test_s1_returns_not_evaluable_for_constant_proxy():
    result = s1_magnitude_gate(
        _flight_frame([1, 2, 3], [1, 1, 1]),
        dataset="d",
        channel="c",
    )
    assert result.status == "not_evaluable"


def test_s3_passes_separated_pre_and_post_distributions():
    time = np.arange(0.0, 120.0, 0.5)
    z = np.zeros(len(time))
    z[(time >= 100.0) & (time <= 115.0)] = 5.0
    frame = pd.DataFrame({"flight_id": "f", "t": time, "z": z})
    result = s3_event_separation_gate(
        frame,
        dataset="d",
        channel="c",
        fault_class="fault",
        events=[{"flight_id": "f", "onset_s": 100.0}],
    )
    assert result.status == "passed"
    assert result.metrics["ks_pvalue"] < 0.01


def test_s3_fails_when_frozen_windows_do_not_separate():
    time = np.arange(0.0, 120.0, 0.5)
    frame = pd.DataFrame({"flight_id": "f", "t": time, "z": np.sin(time)})
    result = s3_event_separation_gate(
        frame,
        dataset="d",
        channel="c",
        fault_class="fault",
        events=[{"flight_id": "f", "onset_s": 100.0}],
    )
    assert result.status == "failed"


def test_calibration_lock_requires_explicit_pass_for_every_class():
    with pytest.raises(GateError, match="S-3 PASS required"):
        require_s3_pass({"engine": {"status": "failed"}}, ["engine"])
    require_s3_pass({"motor": {"status": "passed"}}, ["motor"])
    require_s3_pass({"sensor": "passed"}, ["sensor"])



# ===== kaynak: test_residual_v1_s4_ablation =====

def _matrix(*, context_drives_response: bool) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = 400
    command = rng.normal(size=rows)
    context = rng.normal(size=rows)
    noise = rng.normal(scale=0.05, size=rows)
    response = (context if context_drives_response else command) + noise
    return pd.DataFrame(
        {
            "flight_id": np.where(np.arange(rows) < rows / 2, "f1", "f2"),
            "t": np.arange(rows, dtype=float),
            "phase": "cruise",
            "train_eligible": True,
            "command__last": command,
            "command__delta_1s": command,
            "context__last": context,
            "phase_cruise": 1.0,
            "response": response,
        }
    )


FEATURES = ["command__last", "command__delta_1s", "context__last", "phase_cruise"]


SPEC = ResidualChannelSpec("channel", ("command",), "response", ("context",))


def test_s4_passes_when_command_removal_destroys_fit():
    report = command_ablation_report(
        _matrix(context_drives_response=False),
        spec=SPEC,
        selected_alpha=0.1,
        full_feature_columns=FEATURES,
    )
    assert report["status"] == "passed"
    assert report["variance_ratio"] > 1.15
    assert report["removed_command_features"] == ["command__delta_1s", "command__last"]


def test_s4_flags_when_context_alone_matches_full_model():
    report = command_ablation_report(
        _matrix(context_drives_response=True),
        spec=SPEC,
        selected_alpha=0.1,
        full_feature_columns=FEATURES,
    )
    assert report["status"] == "flagged"
    assert report["variance_ratio"] < 1.15


def test_s4_fails_closed_when_declared_command_columns_are_absent():
    matrix = _matrix(context_drives_response=False).drop(columns=["command__last", "command__delta_1s"])
    try:
        command_ablation_report(
            matrix,
            spec=SPEC,
            selected_alpha=0.1,
            full_feature_columns=["context__last", "phase_cruise"],
        )
    except ValueError as error:
        assert "no feature columns found for command" in str(error)
    else:
        raise AssertionError("S-4 accepted a missing declared command")



# ===== kaynak: test_residual_v1_handout =====

def test_handout_scope_keeps_holdout_separate():
    visible, sealed = split_scope(
        {
            "partitions": {
                "development": {"flight_ids": ["dev/a"]},
                "test": {"flight_ids": ["test/b"]},
                "holdout": {"flight_ids": ["secret/c"]},
            }
        }
    )
    assert visible == {"dev/a": "development", "test/b": "test"}
    assert sealed == ["secret/c"]
    assert "secret/c" not in visible


def test_handout_scope_rejects_role_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_scope(
            {
                "partitions": {
                    "development": {"flight_ids": ["same"]},
                    "test": {"flight_ids": ["same"]},
                    "holdout": {"flight_ids": []},
                }
            }
        )


def test_downsample_uses_observed_rows_and_keeps_endpoints():
    frame = pd.DataFrame({"t": range(100), "value": range(100)})
    result = downsample(frame, max_points=11)
    assert len(result) == 11
    assert result.iloc[0].to_dict() == {"t": 0, "value": 0}
    assert result.iloc[-1].to_dict() == {"t": 99, "value": 99}
    assert set(result["t"]).issubset(set(frame["t"]))


def test_claude_pdf_chunks_stay_below_one_hundred_pages():
    values = [{"flight_id": str(index)} for index in range(348)]
    parts = chunked(values, 87)
    assert [len(part) for part in parts] == [87, 87, 87, 87]
    assert all(len(part) < 100 for part in parts)


def test_flight_slug_is_stable_and_path_safe():
    first = flight_slug("Real-Motor/acce/406_1/log")
    assert first == flight_slug("Real-Motor/acce/406_1/log")
    assert "/" not in first
    assert "\\" not in first

