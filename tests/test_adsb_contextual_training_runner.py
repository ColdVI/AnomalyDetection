import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adsb.contextual_scaling import (
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)
from scripts import adsb_train_contextual_physics_v1 as runner


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
