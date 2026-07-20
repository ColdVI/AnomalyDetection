from __future__ import annotations

import numpy as np
import pandas as pd

from rfly_dl.config import FEATURE_COLUMNS
from rfly_dl.data import align_window_scores, apply_robust_scaler, fit_robust_scaler
from rfly_dl.decision import fit_k_of_n_policy, fit_threshold_policy
from rfly_dl.models import make_model, reconstruction_scores, train_model


def test_train_only_scaler_preserves_missing_mask_and_clips():
    train = pd.DataFrame({column: [0.0, 1.0, np.nan] for column in FEATURE_COLUMNS})
    scaler = fit_robust_scaler(train)
    test = train.copy()
    test.loc[0, FEATURE_COLUMNS[0]] = 1e9
    scaled = apply_robust_scaler(test, scaler)
    assert np.isnan(scaled.loc[2, FEATURE_COLUMNS[0]])
    assert scaled.loc[0, FEATURE_COLUMNS[0]] == 10.0


def test_window_alignment_never_uses_future_score():
    base = pd.DataFrame(
        {
            "source_id": ["f"] * 5,
            "t_rel_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "label": ["normal"] * 5,
        }
    )
    meta = pd.DataFrame(
        {"flight_id": ["f", "f"], "t_start": [0.0, 1.0], "t_end": [2.0, 4.0]}
    )
    aligned = align_window_scores(base, meta, np.array([0.2, 0.9]), "score")
    assert aligned["score"].iloc[:2].isna().all()
    assert aligned["score"].iloc[2:4].tolist() == [0.2, 0.2]
    assert aligned["score"].iloc[4] == 0.9


def test_validation_fitted_policies_respect_requested_budget():
    streams = [
        np.array([0.0, 0.1, 0.2, 0.9, 0.1, 0.2] * 100, dtype=float),
        np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3] * 100, dtype=float),
    ]
    threshold = fit_threshold_policy(streams, 12.0)
    k_of_n = fit_k_of_n_policy(streams, 12.0)
    assert threshold.calibration_fa_per_hour <= 12.0
    assert k_of_n.calibration_fa_per_hour <= 12.0


def test_all_deep_models_train_and_score_finite():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(12, 6, 3)).astype(np.float32)
    m_train = np.ones_like(x_train)
    x_val = rng.normal(size=(5, 6, 3)).astype(np.float32)
    m_val = np.ones_like(x_val)
    for index, name in enumerate(("lstm_ae", "dense_ae", "usad")):
        model = make_model(name, window=6, n_features=3, seed=10 + index)
        trained = train_model(
            name,
            model,
            x_train,
            m_train,
            x_val,
            m_val,
            seed=10 + index,
            max_epochs=2,
            patience=2,
        )
        scores = reconstruction_scores(name, trained.model, x_val, m_val)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
