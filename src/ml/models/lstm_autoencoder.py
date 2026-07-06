"""Normal-only, maskeli LSTM autoencoder (notebook 03/05'in paketlenmis hali)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


AE_FEATURES = {
    "alfa": [
        "abs_roll_error", "abs_pitch_error", "abs_yaw_error", "roll_error_rate",
        "pitch_error_rate", "yaw_error_rate", "roll_error_2s_rms", "roll_error_5s_rms",
        "pitch_error_2s_rms", "pitch_error_5s_rms", "yaw_error_5s_rms",
        "roll_spec_energy_5s", "pitch_spec_energy_5s", "turn_residual", "alt_error",
        "aspd_error", "xtrack_error", "path_dev_mag", "climb_residual", "energy_rate",
        "altitude_rate", "abs_airspeed_error", "gps_speed_calc_mps",
    ],
    "uav_attack": [
        "gps_step_m", "log_gps_speed", "gps_accel_mps2", "vertical_rate_calc",
        "gps_speed_residual", "vertical_rate_residual", "course_change_deg",
        "gps_frozen_count", "jamming_indicator", "noise_per_ms", "hdop", "vdop",
        "satellites_used", "s_variance_m_s", "eph", "epv", "roll_rate",
        "pitch_rate", "yaw_rate", "attitude_missing", "num_missing_groups", "attitude_stale_s",
    ],
    # SEAD ayni PX4 omurgasini kullanir. Bu model ML-8A sirasinda yeniden
    # egitilmis ayri bir karsilastirma satiridir; mevcut ML-6 baseline degildir.
    "uav_sead": [
        "gps_step_m", "log_gps_speed", "gps_accel_mps2", "vertical_rate_calc",
        "gps_speed_residual", "vertical_rate_residual", "course_change_deg",
        "gps_frozen_count", "jamming_indicator", "noise_per_ms", "hdop", "vdop",
        "satellites_used", "s_variance_m_s", "eph", "epv", "roll_rate",
        "pitch_rate", "yaw_rate", "attitude_missing", "num_missing_groups", "attitude_stale_s",
    ],
}
WINDOW = {"alfa": 40, "uav_attack": 50, "uav_sead": 50}
STRIDE = {"alfa": 4, "uav_attack": 5, "uav_sead": 5}


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, latent: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        repeated = self.from_latent(z).unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.head(decoded)


def masked_mse(x, reconstruction, mask, *, per_sample: bool = False):
    error = ((x - reconstruction) ** 2) * mask
    if per_sample:
        return error.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
    return error.sum() / mask.sum().clamp(min=1.0)


def train_lstm_autoencoder(model, x_train, m_train, x_val, m_val, *,
                           seed: int, epochs: int = 40, batch_size: int = 64,
                           learning_rate: float = 1e-3, patience: int = 5):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    xt, mt = torch.tensor(x_train), torch.tensor(m_train)
    xv, mv = torch.tensor(x_val), torch.tensor(m_val)
    best_loss, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(xt))
        for start in range(0, len(permutation), batch_size):
            idx = permutation[start:start + batch_size]
            loss = masked_mse(xt[idx], model(xt[idx]), mt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(xv, model(xv), mv))
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("LSTM-AE training en iyi state uretmedi")
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "epochs": epoch + 1}


def reconstruction_scores(model, x, mask, *, batch_size: int = 512) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size])
            mb = torch.tensor(mask[start:start + batch_size])
            output.append(masked_mse(xb, model(xb), mb, per_sample=True).numpy())
    return np.concatenate(output) if output else np.array([])
