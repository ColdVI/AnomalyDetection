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
