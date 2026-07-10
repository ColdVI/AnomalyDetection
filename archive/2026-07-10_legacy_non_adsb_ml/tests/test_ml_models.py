import numpy as np
import torch

from src.ml.evaluation.thresholds import pot_threshold
from src.ml.models.lstm_autoencoder import LSTMAutoencoder, masked_mse


def test_lstm_autoencoder_preserves_sequence_shape():
    model = LSTMAutoencoder(5)
    x = torch.zeros(3, 12, 5)
    assert model(x).shape == x.shape


def test_masked_mse_ignores_missing_cells():
    x = torch.tensor([[[1.0, 100.0]]])
    reconstruction = torch.tensor([[[0.0, 0.0]]])
    mask = torch.tensor([[[1.0, 0.0]]])
    assert float(masked_mse(x, reconstruction, mask)) == 1.0


def test_pot_threshold_falls_back_to_max_with_too_few_exceedances():
    values = np.array([1.0, 2.0, 3.0])
    assert pot_threshold(values, min_exceedances=10) == 3.0
