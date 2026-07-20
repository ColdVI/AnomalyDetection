"""ADS-B contextual torch testleri (torch gerektirir)

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

import numpy as np

import pytest

import torch

from adsb.models.contextual_residual_forecaster import (
    NATURAL_FIT_ROLE,
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    channelwise_gaussian_nll,
    contextual_channel_scores,
    train_contextual_residual_forecaster,
    weighted_masked_channel_loss,
)

import json

from pathlib import Path

import pandas as pd

from adsb.contextual_scaling import (
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)

from scripts import adsb_train_contextual_physics_v1 as runner



# ===== kaynak: test_adsb_contextual_forecaster =====

def _config() -> ContextualForecasterConfig:
    return ContextualForecasterConfig(
        input_features=4,
        target_channels=2,
        hidden_size=6,
        num_layers=1,
        min_scale=0.1,
        max_scale=3.0,
    )


def test_forecaster_shapes_and_bounded_positive_scale():
    model = ContextualResidualForecaster(_config())
    location, scale = model(torch.zeros(5, 3, 4), torch.ones(5, 3, 4))
    assert location.shape == (5, 2)
    assert scale.shape == (5, 2)
    assert torch.all(scale >= 0.1)
    assert torch.all(scale <= 3.0)


def test_channel_loss_keeps_mask_and_explicit_weights():
    target = torch.tensor([[2.0, 100.0]])
    location = torch.zeros_like(target)
    scale = torch.ones_like(target)
    mask = torch.tensor([[1.0, 0.0]])
    nll, surprise = channelwise_gaussian_nll(target, location, scale, mask)
    assert surprise.tolist() == [[2.0, 0.0]]
    assert weighted_masked_channel_loss(nll, mask, torch.tensor([4.0, 1.0])).item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="one explicit weight"):
        weighted_masked_channel_loss(nll, mask, torch.tensor([1.0]))


def test_training_contract_rejects_synthetic_and_smoke_is_finite():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3, 4)).astype(np.float32)
    mask = np.ones_like(X)
    y = np.column_stack((X[:, -1, 0], X[:, -1, 1])).astype(np.float32)
    y_mask = np.ones_like(y)
    kwargs = dict(
        config=_config(),
        channel_weights=(1.0, 1.0),
        data_role=NATURAL_FIT_ROLE,
        epochs=2,
        batch_size=5,
        learning_rate=1e-2,
        seed=3,
    )
    with pytest.raises(ValueError, match="Synthetic"):
        train_contextual_residual_forecaster(
            X, mask, y, y_mask, contains_synthetic=True, **kwargs
        )
    model, history = train_contextual_residual_forecaster(
        X, mask, y, y_mask, contains_synthetic=False, **kwargs
    )
    assert len(history) == 2
    assert np.isfinite(history).all()
    scores, location, scale = contextual_channel_scores(model, X, mask, y, y_mask)
    assert scores.shape == location.shape == scale.shape == y.shape
    assert np.isfinite(scores).all()



# ===== kaynak: test_adsb_contextual_training_runner =====

def _feature_frame(flight_id: str = "2026-02-28:abc123_000") -> pd.DataFrame:
    n = 40
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "flight_id": [flight_id] * n,
            "timestamp_utc": t,
            "on_ground": [False] * n,
            "vertical_rate_ms": np.sin(t / 5),
            "track_deg": np.mod(350 + t, 360),
            "vertical_rate_residual": np.sin(t / 4),
            "speed_residual": np.cos(t / 6),
            "heading_residual": np.sin(t / 7),
            "altitude_source_residual": np.cos(t / 8),
            "east_velocity_residual": np.sin(t / 9),
            "north_velocity_residual": np.cos(t / 10),
        }
    )


def _small_config() -> dict:
    config = json.loads(Path("configs/adsb_contextual_physics_v1_train.json").read_text())
    config["window"]["history_rows"] = 3
    config["model"].update({"hidden_size": 4, "min_scale": 0.1, "max_scale": 3.0})
    config["training"].update({"epochs": 1, "batch_size": 8, "learning_rate": 0.01})
    return config


def test_flight_sampling_is_deterministic_and_requires_probability():
    ids = [f"2026-02-28:abc{i:03d}_000" for i in range(100)]
    first = runner._sample_flights(ids, probability=0.2, seed=7, purpose="fit")
    second = runner._sample_flights(reversed(ids), probability=0.2, seed=7, purpose="fit")
    assert first == second
    assert 0 < len(first) < len(ids)
    with pytest.raises(runner.ContextualTrainingContractError):
        runner._sample_flights(ids, probability=0.0, seed=7, purpose="fit")


def test_selected_features_filters_sources_and_exact_flights(tmp_path):
    raw = pd.DataFrame(
        {
            "_source_file": ["v2026.02.28.tar"] * 8,
            "source_id": ["abc123"] * 4 + ["def456"] * 4,
            "timestamp_utc": [0.0, 1.0, 2.0, 3.0] * 2,
            "lat": [0.0, 0.001, 0.002, 0.003] * 2,
            "lon": [0.0] * 8,
            "alt": np.arange(8, dtype=float),
            "alt_geom_m": np.arange(8, dtype=float) + 1,
            "on_ground": [False] * 8,
            "ground_speed_ms": [100.0] * 8,
            "track_deg": [0.0] * 8,
            "vertical_rate_ms": [1.0] * 8,
        }
    )
    path = tmp_path / "part.parquet"
    raw.to_parquet(path, index=False)
    selected = {"2026-02-28:abc123_000"}
    result = runner._selected_features(path, selected, {"abc123"})
    assert set(result["flight_id"]) == selected
    assert len(result) == 4


def test_one_epoch_streaming_training_and_natural_diagnostic(monkeypatch, tmp_path):
    frame = _feature_frame()
    channels = tuple(_small_config()["channels"])
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=5.0)).fit(
        frame,
        channels,
        data_role=NATURAL_FIT_ROLE,
        contains_synthetic=False,
    )

    def fake_iter(paths, selected_flights):
        yield Path("fixture.parquet"), frame.copy()

    monkeypatch.setattr(runner, "_iter_selected_features", fake_iter)
    model, report = runner._train(
        [Path("fixture.parquet")],
        ("2026-02-28:abc123_000",),
        scaler=scaler,
        config=_small_config(),
        run_dir=tmp_path,
    )
    assert report["epochs"][0]["windows"] == 37
    assert np.isfinite(report["epochs"][0]["mean_weighted_gaussian_nll"])
    diagnostic = runner._natural_diagnostics(
        model,
        [Path("fixture.parquet")],
        ("2026-02-28:abc123_000",),
        scaler=scaler,
        config=_small_config(),
    )
    assert diagnostic["windows"] == 37
    assert diagnostic["never_used_for_optimizer_or_threshold"] is True
    assert set(diagnostic["per_channel_standardized_surprise"]) == set(channels)

