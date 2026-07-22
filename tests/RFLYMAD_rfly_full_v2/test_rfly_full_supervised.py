import numpy as np
import pandas as pd
import pytest
import torch

import gecmis_calismalar.rfly_full.supervised as supervised_module
from gecmis_calismalar.rfly_full.supervised import (
    SAMPLE_HZ,
    TemporalFaultClassifier,
    _build_training_windows,
    development_run_status,
    _flight_windows,
    _reservoir_update,
    _score_streaming,
    alarm_onsets,
    training_window_label,
)

SCALER = {
    "features": ["local_x", "local_y"],
    "medians": {"local_x": 0.5, "local_y": 1.5},
    "scales": {"local_x": 1.0, "local_y": 1.0},
    "clip": 10.0,
}


def test_supervised_training_excludes_transitions_and_prefault_fault_flights():
    assert training_window_label(np.zeros(5, bool), "normal_reference") == 0
    assert training_window_label(np.ones(5, bool), "fault_detection") == 1
    assert training_window_label(np.array([0, 0, 1, 1], bool), "fault_detection") is None
    assert training_window_label(np.zeros(5, bool), "fault_detection") is None
    assert training_window_label(np.ones(5, bool), "environment_robustness") is None


def test_alarm_policy_is_time_based_and_refractory():
    scores = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1], float)
    alarms = alarm_onsets(scores, 0.5, k=4, n_seconds=6, refractory_seconds=30)
    assert np.flatnonzero(alarms).tolist() == [5]


def test_development_smoke_fold_must_differ_from_validation(monkeypatch):
    monkeypatch.setattr(
        supervised_module,
        "_available_manifest",
        lambda: pd.DataFrame([{
            "split": "development", "cv_fold": 0,
            "canonical_case_id": "normal", "domain": "HIL",
        }]),
    )
    with pytest.raises(ValueError, match="must differ"):
        supervised_module.run(
            validation_fold=0, development_smoke_fold=0, epochs=1,
        )


def test_development_run_status_never_promotes_locked_test_unread_runs():
    assert development_run_status(
        epochs=3, parse_complete=True, development_smoke_fold=1,
    ) == "smoke_only"
    assert development_run_status(
        epochs=12, parse_complete=True, development_smoke_fold=1,
    ) == "development_only"
    assert development_run_status(
        epochs=12, parse_complete=True, development_smoke_fold=None,
    ) == "complete"
    assert development_run_status(
        epochs=12, parse_complete=False, development_smoke_fold=1,
    ) == "smoke_only"


def _fake_flight(
    *, length: int, fault_active: np.ndarray, fault_family: str = "Motor",
    evaluation_role: str = "fault_detection",
) -> pd.DataFrame:
    return pd.DataFrame({
        "t_rel_s": np.arange(length) / SAMPLE_HZ,
        "fault_active": fault_active,
        "fault_family": [fault_family] * length,
        "fault_subtype": ["Motor_1"] * length,
        "evaluation_role": [evaluation_role] * length,
        "local_x": np.linspace(0, 1, length),
        "local_y": np.linspace(1, 2, length),
    })


class _Row:
    def __init__(self, canonical_case_id: str, domain: str):
        self.canonical_case_id = canonical_case_id
        self.domain = domain


def test_reservoir_update_respects_capacity_and_is_deterministic():
    def sample(seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        reservoir: list[int] = []
        seen = 0
        for item in range(1000):
            seen = _reservoir_update(reservoir, seen, 10, item, rng)
        return reservoir

    first = sample(42)
    second = sample(42)
    assert len(first) == 10
    assert first == second
    assert sample(43) != first


def test_flight_windows_yields_correct_slices(monkeypatch):
    flight = _fake_flight(length=250, fault_active=np.r_[np.zeros(120, bool), np.ones(130, bool)])
    monkeypatch.setattr(supervised_module, "_load_flight", lambda canonical, domain, columns: flight)
    row = _Row("f", "SIL")
    columns = ["t_rel_s", "fault_active", "fault_family", "fault_subtype", "evaluation_role", *SCALER["features"]]
    windows = list(_flight_windows(row, SCALER, columns, length=200, stride=50))
    assert len(windows) == (250 - 200) // 50 + 1
    first_window, t_end_s, active_at_end, active_slice = windows[0]
    assert first_window.shape == (200, len(SCALER["features"]) * 2)
    assert t_end_s == flight["t_rel_s"].iloc[199]
    assert active_at_end == bool(flight["fault_active"].iloc[199])
    assert len(active_slice) == 200


def test_build_training_windows_respects_cap_and_is_deterministic(monkeypatch):
    length = 400
    flights = {
        "normal_a": _fake_flight(length=length, fault_active=np.zeros(length, bool), fault_family="NoFault", evaluation_role="normal_reference"),
        "normal_b": _fake_flight(length=length, fault_active=np.zeros(length, bool), fault_family="NoFault", evaluation_role="normal_reference"),
        "fault_a": _fake_flight(length=length, fault_active=np.ones(length, bool), fault_family="Motor", evaluation_role="fault_detection"),
    }
    monkeypatch.setattr(supervised_module, "_load_flight", lambda canonical, domain, columns: flights[canonical])

    manifest = pd.DataFrame([
        {"canonical_case_id": "normal_a", "domain": "SIL", "fault_family": "NoFault", "fault_subtype": "NoFault", "evaluation_role": "normal_reference"},
        {"canonical_case_id": "normal_b", "domain": "SIL", "fault_family": "NoFault", "fault_subtype": "NoFault", "evaluation_role": "normal_reference"},
        {"canonical_case_id": "fault_a", "domain": "SIL", "fault_family": "Motor", "fault_subtype": "Motor_1", "evaluation_role": "fault_detection"},
    ])
    family_index = {"Motor": 0}

    result = _build_training_windows(manifest, SCALER, family_index, stride_seconds=5, max_windows=4, seed=1)
    assert len(result.x) <= 4
    assert (result.binary == 0).sum() <= 2
    assert (result.binary == 1).sum() <= 2
    assert set(np.unique(result.binary).tolist()).issubset({0, 1})

    repeat = _build_training_windows(manifest, SCALER, family_index, stride_seconds=5, max_windows=4, seed=1)
    np.testing.assert_array_equal(result.x, repeat.x)
    np.testing.assert_array_equal(result.binary, repeat.binary)
    np.testing.assert_array_equal(result.family, repeat.family)


def test_score_streaming_is_batch_size_invariant(monkeypatch):
    length = 250
    flights = {
        "a": _fake_flight(length=length, fault_active=np.r_[np.zeros(100, bool), np.ones(150, bool)], fault_family="Motor", evaluation_role="fault_detection"),
        "b": _fake_flight(length=length, fault_active=np.zeros(length, bool), fault_family="NoFault", evaluation_role="normal_reference"),
    }
    monkeypatch.setattr(supervised_module, "_load_flight", lambda canonical, domain, columns: flights[canonical])

    manifest = pd.DataFrame([
        {"canonical_case_id": "a", "domain": "SIL", "fault_family": "Motor", "fault_subtype": "Motor_1", "evaluation_role": "fault_detection"},
        {"canonical_case_id": "b", "domain": "SIL", "fault_family": "NoFault", "fault_subtype": "NoFault", "evaluation_role": "normal_reference"},
    ])
    torch.manual_seed(0)
    model = TemporalFaultClassifier(channels_in=len(SCALER["features"]) * 2, families=1)
    model.eval()

    meta_small, binary_small, family_small = _score_streaming(manifest, model, SCALER, stride_seconds=1, batch_size=3)
    meta_big, binary_big, family_big = _score_streaming(manifest, model, SCALER, stride_seconds=1, batch_size=10_000)

    assert len(meta_small) == len(meta_big) > 0
    np.testing.assert_allclose(binary_small, binary_big, atol=1e-5)
    np.testing.assert_allclose(family_small, family_big, atol=1e-5)
    assert meta_small["fault_active"].tolist() == meta_big["fault_active"].tolist()
