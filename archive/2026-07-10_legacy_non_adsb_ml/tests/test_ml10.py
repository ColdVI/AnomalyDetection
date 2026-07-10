import inspect
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import build_ml10_forecast_residual as precompute
from scripts import run_ml10_forecast_evaluation as ml10_runner
from scripts import run_ml9_category_evaluation as ml9_runner
from src.ml.decision import decision_layers
from src.ml.evaluation import score_fusion


class _LastValuePipeline:
    """Tiny deterministic stand-in with the same quantile API as Chronos."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        contexts = inputs if isinstance(inputs, list) else [inputs]
        rows = []
        for context in contexts:
            last = float(context[-1])
            rows.append([[last - 0.1, last, last + 0.1]])
        quantiles = torch.tensor(rows, dtype=torch.float32)
        return quantiles, quantiles[:, :, 1]


def test_chronos_forecast_residual_no_future_leak():
    times = np.arange(120, dtype=float)
    original = np.sin(times / 8.0)
    changed_future = original.copy()
    changed_future[80:] = 1_000_000.0
    pipeline = _LastValuePipeline()

    original_positions, original_scores = precompute.forecast_residual(
        pipeline, original, times, stride_s=1.0, batch_size=13,
    )
    changed_positions, changed_scores = precompute.forecast_residual(
        pipeline, changed_future, times, stride_s=1.0, batch_size=7,
    )
    original_prefix = original_positions < 80
    changed_prefix = changed_positions < 80
    np.testing.assert_array_equal(
        original_positions[original_prefix], changed_positions[changed_prefix],
    )
    np.testing.assert_allclose(
        original_scores[original_prefix], changed_scores[changed_prefix], rtol=0, atol=0,
    )


def test_chronos_zero_shot_no_training_step():
    source = inspect.getsource(precompute)
    forbidden = (".backward(", ".train(", "optimizer.step(", "loss.backward(")
    assert all(token not in source for token in forbidden)
    assert "torch.inference_mode()" in source
    assert "training_or_gradient_updates\": False" in source


def test_ml10_decision_layers_reused_not_reimplemented():
    assert ml10_runner.fit_threshold_policy is decision_layers.fit_threshold_policy
    assert ml10_runner.fit_k_of_n_policy is decision_layers.fit_k_of_n_policy
    assert ml10_runner.fit_cusum_policy is decision_layers.fit_cusum_policy


def test_ml10_score_fusion_not_duplicated():
    assert ml9_runner._empirical_probability is score_fusion.empirical_probability
    assert ml10_runner._empirical_probability is score_fusion.empirical_probability
    assert ml9_runner.max_score_fusion is score_fusion.max_score_fusion
    assert ml10_runner.max_score_fusion is score_fusion.max_score_fusion


def test_ml10_precompute_contains_development_only():
    root = Path(__file__).resolve().parents[1]
    score_path = root / "data/gold/ml_features/uav_sead/uav_sead_ml10_forecast_residual.parquet"
    split_path = root / "data/gold/ml_features/split_manifest.json"
    manifest_path = root / "artifacts/ml10/uav_sead/full_matrix/manifest.json"
    if not score_path.exists():
        return
    if manifest_path.exists():
        artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
        if artifact_manifest.get("split_manifest_sha256") != current_sha:
            pytest.skip("eski veri donemi artifact'i")
    config = json.loads(split_path.read_text(encoding="utf-8"))["sources"]["uav_sead"]
    split = config["splits"]["split_00"]
    development = set(split["train"] + split["val"] + split["test"])
    holdout = set(split["final_holdout"])
    scores = pd.read_parquet(score_path, columns=["source_id"])
    observed = set(scores["source_id"].unique())
    assert observed == development
    assert not observed & holdout
