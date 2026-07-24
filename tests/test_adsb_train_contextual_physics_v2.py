"""Regression tests for contextual_physics_v2 training batch construction."""

from __future__ import annotations

import numpy as np
import torch

from adsb.models.contextual_residual_forecaster import (
    ContextualForecasterConfig,
    ContextualResidualForecaster,
)
from scripts.adsb_train_contextual_physics_v2 import (
    _numpy_batch_indices,
    _write_epoch_checkpoint_atomic,
)


def test_single_row_remainder_preserves_batch_axis() -> None:
    windows = np.zeros((1, 12, 20), dtype=np.float32)
    index = _numpy_batch_indices(torch.tensor([0]), start=0, batch_size=512)

    assert index.shape == (1,)
    assert windows[index].shape == (1, 12, 20)


def test_multi_row_batch_preserves_permutation_order() -> None:
    windows = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    permutation = torch.tensor([3, 1, 0, 2])
    index = _numpy_batch_indices(permutation, start=1, batch_size=2)

    assert index.tolist() == [1, 0]
    assert np.array_equal(windows[index], windows[[1, 0]])


def test_epoch_checkpoint_is_atomic_and_records_completed_history(tmp_path) -> None:
    model = ContextualResidualForecaster(
        ContextualForecasterConfig(
            input_features=3,
            target_channels=2,
            hidden_size=4,
            num_layers=1,
            min_scale=0.01,
            max_scale=100.0,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = tmp_path / "training_epoch_checkpoint.pt"
    history = [{"epoch": 1, "windows": 17, "batches": 2}]

    _write_epoch_checkpoint_atomic(
        checkpoint,
        model=model,
        optimizer=optimizer,
        history=history,
    )
    payload = torch.load(checkpoint, weights_only=False)

    assert payload["artifact_role"] == "incomplete_training_recovery_only"
    assert payload["completed_epochs"] == 1
    assert payload["history"] == history
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert not checkpoint.with_name(f"{checkpoint.name}.tmp").exists()
