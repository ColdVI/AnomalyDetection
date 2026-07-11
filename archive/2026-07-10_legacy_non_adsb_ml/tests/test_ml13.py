"""ML-13 channel architecture discipline tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_ml10_forecast_evaluation as ml10_runner
from scripts import run_ml13_channel_evaluation as ml13_runner
from scripts import run_ml9_category_evaluation as ml9_runner
from src.ml.decision import decision_layers
from src.ml.decision.channel_union import union_onsets
from src.ml.evaluation import events, score_fusion

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
RUN_DIR = ROOT / "artifacts/ml13/uav_sead/full_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_channel_union_dedupes_same_bucket():
    system = np.array([False, True, False, True])
    mechanical = np.array([False, True, True, False])
    combined = union_onsets([system, mechanical])
    assert combined.tolist() == [False, True, True, True]
    assert int(combined.sum()) == 3


def test_channel_union_requires_aligned_masks():
    with pytest.raises(ValueError):
        union_onsets([np.array([True]), np.array([True, False])])


def test_budget_allocations_match_preregistered_plan():
    assert ml13_runner.BUDGET_ALLOCATIONS == {
        "agirlikli_sistem": {
            "advisory": {"sistem": 10.0, "mekanik": 2.0},
            "critical": {"sistem": 1.67, "mekanik": 0.33},
        },
        "dengeli": {
            "advisory": {"sistem": 8.0, "mekanik": 4.0},
            "critical": {"sistem": 1.33, "mekanik": 0.67},
        },
        "esit": {
            "advisory": {"sistem": 6.0, "mekanik": 6.0},
            "critical": {"sistem": 1.0, "mekanik": 1.0},
        },
    }


def test_ml13_runner_has_no_scorer_model_training_call():
    source = inspect.getsource(ml13_runner)
    assert "fit_modular_iforest" not in source
    assert "IsolationForest" not in source
    assert ".fit(" not in source


def test_ml13_reuses_shared_decision_and_event_helpers():
    assert ml13_runner._fit_policies is ml10_runner._fit_policies
    assert ml13_runner._evaluate is ml9_runner._evaluate
    assert ml13_runner.event_metrics is events.event_metrics
    assert ml13_runner.last_causal_per_bucket is score_fusion.last_causal_per_bucket
    assert ml13_runner.fit_cusum_policy is decision_layers.fit_cusum_policy
    assert ml13_runner.fit_threshold_policy is decision_layers.fit_threshold_policy
    assert ml13_runner.fit_k_of_n_policy is decision_layers.fit_k_of_n_policy


def test_ml13_artifact_holdout_isolation_and_checksums():
    manifest_path = RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("ML-13 full_matrix kosusu henuz yapilmadi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["blind_holdout_read"] is False
    assert manifest["blind_holdout_flights"] == 131
    assert manifest["development_flights"] == 480
    assert manifest["score_channels"]["sistem"]["score_source"] == "existing_fusion"
    assert manifest["score_channels"]["mekanik"]["score_source"] == "itki_komutu"

    for relative, expected in manifest["files"].items():
        assert _sha256(RUN_DIR / relative) == expected, relative

    if manifest.get("split_manifest_sha256") != _sha256(SPLIT_PATH):
        pytest.skip("eski veri donemi artifact'i")

    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = split_manifest["sources"]["uav_sead"]
    holdout = set(config["splits"]["split_00"]["final_holdout"])
    expected_dev = sorted(set(config["flight_labels"]) - holdout)
    expected_hash = hashlib.sha256(
        "\n".join(expected_dev).encode("utf-8")
    ).hexdigest()
    assert manifest["development_source_ids_sha256"] == expected_hash


def test_ml13_baseline_rows_match_frozen_csvs():
    metrics_path = RUN_DIR / "baseline_metrics.csv"
    if not metrics_path.exists():
        pytest.skip("ML-13 full_matrix kosusu henuz yapilmadi")
    baseline = pd.read_csv(metrics_path)
    key = ["split", "seed", "score_source", "decision", "budget"]

    ml9 = pd.read_csv(ml13_runner.ML9_DIR / "metrics.csv")
    left = (
        baseline[baseline["score_source"] == "existing_fusion"]
        .set_index(key)
        .sort_index()
    )
    right = (
        ml9[ml9["score_source"] == "existing_fusion"]
        .set_index(key)
        .sort_index()
    )
    pd.testing.assert_frame_equal(left, right, check_like=True)

    ml12 = pd.read_csv(ml13_runner.ML12_DIR / "metrics.csv")
    left = (
        baseline[baseline["score_source"] == "ml12_fusion_itki"]
        .set_index(key)
        .sort_index()
    )
    right = (
        ml12[ml12["score_source"] == "ml12_fusion_itki"]
        .set_index(key)
        .sort_index()
    )
    pd.testing.assert_frame_equal(left, right, check_like=True)


def test_ml13_gate_c2_does_not_replace_c1():
    gates_path = RUN_DIR / "gates.json"
    if not gates_path.exists():
        pytest.skip("ML-13 full_matrix kosusu henuz yapilmadi")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    assert "gate_c1" in gates
    assert "gate_c2" in gates
    assert gates["gate_c2"]["rule"].startswith("mechanical channel only")
