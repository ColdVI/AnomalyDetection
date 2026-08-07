"""ML-16 Kol N discipline tests (docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md).

Covers: frozen-model-not-retrained static assertions, reuse-not-reimplemented identity
checks, the two SS1 scoring formulas matching their spec on hand-verified toy examples
(via an independent, plain-Python reference computation -- not the vectorized
implementation under test), correct exclusion (never silent imputation) of masked/NaN/
no-reference channels, the `masked_mse_per_channel` <-> `masked_mse` recombination
invariant, and artifact-level checks once a real run exists.
"""

from __future__ import annotations

import hashlib
import io
import json
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import run_ml8a_temporal_boosting as ml8a_runner
from scripts import run_ml9_category_evaluation as ml9_runner
from scripts import run_ml16_kol_n_magnitude_normalized_scoring as kol_n_runner
from scripts import run_ml_dense_ae_sead_evaluation as dense_runner
from scripts import run_ml_lstm_sead_evaluation as lstm_runner
from scripts import run_ml_usad_sead_evaluation as usad_runner
from src.ml.data import scaling, windowing
from src.ml.evaluation import magnitude_normalized_scoring as scoring
from src.ml.evaluation import score_fusion
from src.ml.models.lstm_autoencoder import masked_mse, masked_mse_per_channel

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_names(source: str) -> set[str]:
    """NAME tokens appearing in actual code -- excludes string literals/docstrings and
    comments, so a prose sentence like "fit_scaler_params is never called here" inside a
    module docstring does not trip a naive substring search."""
    names: set[str] = set()
    for tok_type, tok_string, *_ in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok_type == tokenize.NAME:
            names.add(tok_string)
    return names


# ---------------------------------------------------------------------------
# Frozen-model / no-training static assertions
# ---------------------------------------------------------------------------

def test_scoring_script_never_trains_or_fits_a_model():
    forbidden = {
        "fit", "train_lstm_autoencoder", "train_dense_autoencoder", "train_usad",
        "fit_modular_iforest", "fit_scaler_params", "IsolationForest", "backward",
    }
    for path in (
        Path("scripts/run_ml16_kol_n_magnitude_normalized_scoring.py"),
        Path("src/ml/evaluation/magnitude_normalized_scoring.py"),
    ):
        names = _code_names(path.read_text(encoding="utf-8"))
        overlap = names & forbidden
        assert not overlap, f"{path}: forbidden identifiers used as code: {overlap}"


def test_scoring_script_loads_checkpoints_with_weights_only():
    source = Path("scripts/run_ml16_kol_n_magnitude_normalized_scoring.py").read_text(encoding="utf-8")
    assert "weights_only=True" in source
    assert "model.eval()" in source


def test_scoring_script_reuses_frozen_scaler_not_a_freshly_fit_one():
    source = Path("scripts/run_ml16_kol_n_magnitude_normalized_scoring.py").read_text(encoding="utf-8")
    assert "scaler.json" in source
    assert "apply_scaler_params" in source
    assert "fit_scaler_params" not in _code_names(source)


# ---------------------------------------------------------------------------
# Reuse-not-reimplemented identity checks
# ---------------------------------------------------------------------------

def test_kol_n_reuses_decision_and_fusion_helpers_not_reimplemented():
    assert kol_n_runner._fit_policies is ml9_runner._fit_policies
    assert kol_n_runner._evaluate is ml9_runner._evaluate
    assert kol_n_runner._streams is ml9_runner._streams
    assert kol_n_runner.BUDGETS is ml9_runner.BUDGETS
    assert kol_n_runner.MIN_RECALL is ml9_runner.MIN_RECALL
    assert kol_n_runner._align_score is ml8a_runner._align_score
    assert kol_n_runner.build_windows is windowing.build_windows
    assert kol_n_runner.apply_scaler_params is scaling.apply_scaler_params
    assert kol_n_runner.empirical_probability is score_fusion.empirical_probability
    assert kol_n_runner.last_causal_per_bucket is score_fusion.last_causal_per_bucket
    assert kol_n_runner._lstm_sequence is lstm_runner._lstm_sequence
    assert kol_n_runner._dense_sequence is dense_runner._dense_sequence
    assert kol_n_runner._usad_sequence is usad_runner._usad_sequence


def test_kol_n_score_sources_and_eps_match_preregistered_plan():
    assert kol_n_runner.VARIANTS == ("relerr", "rankpct")
    assert scoring.RELATIVE_ERROR_EPS == pytest.approx(0.1)
    assert kol_n_runner.RELATIVE_ERROR_EPS is scoring.RELATIVE_ERROR_EPS
    assert set(kol_n_runner.ARCHITECTURES) == {"lstm", "dense_ae", "usad"}
    for arch_name, prefix in (("lstm", "lstm"), ("dense_ae", "dense_ae"), ("usad", "usad")):
        cols = kol_n_runner._score_variant_columns(arch_name)
        assert cols == {"relerr": f"{prefix}_relerr", "rankpct": f"{prefix}_rankpct"}


def test_kol_n_no_fusion_variants_this_round():
    # docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md SS1: recon-alone only, no
    # ml14_fusion/itki_komutu fused variant this round.
    source = Path("scripts/run_ml16_kol_n_magnitude_normalized_scoring.py").read_text(encoding="utf-8")
    names = _code_names(source)
    assert "ml14_fusion" not in names
    assert "itki_komutu" not in names
    assert "max_score_fusion" not in names


# ---------------------------------------------------------------------------
# masked_mse_per_channel <-> masked_mse recombination invariant
# ---------------------------------------------------------------------------

def test_masked_mse_per_channel_recombines_to_masked_mse_per_sample():
    rng = np.random.default_rng(0)
    n, t, f = 5, 7, 4
    x = torch.tensor(rng.normal(size=(n, t, f)), dtype=torch.float32)
    recon = torch.tensor(rng.normal(size=(n, t, f)), dtype=torch.float32)
    mask = torch.tensor(rng.integers(0, 2, size=(n, t, f)), dtype=torch.float32)
    # Ensure at least one window has an entirely-masked-out channel (edge case).
    mask[0, :, 0] = 0.0

    expected = masked_mse(x, recon, mask, per_sample=True)
    numerator, denominator = masked_mse_per_channel(x, recon, mask, per_sample=True)
    assert numerator.shape == (n, f)
    assert denominator.shape == (n, f)
    recombined = numerator.sum(dim=-1) / denominator.sum(dim=-1).clamp(min=1.0)
    assert torch.allclose(recombined, expected, atol=1e-6)


def test_masked_mse_per_channel_recombines_to_masked_mse_pooled():
    rng = np.random.default_rng(1)
    n, t, f = 6, 5, 3
    x = torch.tensor(rng.normal(size=(n, t, f)), dtype=torch.float32)
    recon = torch.tensor(rng.normal(size=(n, t, f)), dtype=torch.float32)
    mask = torch.tensor(rng.integers(0, 2, size=(n, t, f)), dtype=torch.float32)

    expected = masked_mse(x, recon, mask, per_sample=False)
    numerator, denominator = masked_mse_per_channel(x, recon, mask, per_sample=False)
    assert numerator.shape == (f,)
    assert denominator.shape == (f,)
    recombined = numerator.sum() / denominator.sum().clamp(min=1.0)
    assert torch.allclose(recombined, expected, atol=1e-6)


def test_masked_mse_signature_and_behavior_unchanged():
    # ADR-016/017/018 reproducibility guard: masked_mse itself must be untouched.
    rng = np.random.default_rng(2)
    x = torch.tensor(rng.normal(size=(3, 4, 2)), dtype=torch.float32)
    recon = torch.tensor(rng.normal(size=(3, 4, 2)), dtype=torch.float32)
    mask = torch.ones(3, 4, 2)
    error = ((x - recon) ** 2) * mask
    expected_per_sample = error.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
    assert torch.allclose(masked_mse(x, recon, mask, per_sample=True), expected_per_sample)


# ---------------------------------------------------------------------------
# SS1(a) relative error: hand-verified toy example (independent reference loop)
# ---------------------------------------------------------------------------

def _reference_relative_error(x, recon, mask, eps):
    """Plain-Python, unvectorized reference implementation of SS1(a) -- deliberately
    independent of `channel_relative_error`'s numpy code path."""
    n, t, f = x.shape
    channel_rel = np.full((n, f), np.nan)
    channel_valid = np.zeros((n, f), dtype=bool)
    for i in range(n):
        for c in range(f):
            total, count = 0.0, 0
            for step in range(t):
                if mask[i, step, c] == 1:
                    total += abs(x[i, step, c] - recon[i, step, c]) / (abs(x[i, step, c]) + eps)
                    count += 1
            if count > 0:
                channel_rel[i, c] = total / count
                channel_valid[i, c] = True
    return channel_rel, channel_valid


def test_relative_error_matches_hand_verified_toy_example():
    eps = 0.1
    # 1 window, 3 timesteps, 2 channels; channel 2's 3rd timestep is masked out.
    x = np.array([[[1.0, 10.0], [2.0, 10.0], [3.0, 0.0]]])
    recon = np.array([[[1.5, 9.0], [2.5, 8.0], [100.0, 999.0]]])
    mask = np.array([[[1, 1], [1, 1], [1, 0]]], dtype=np.float32)

    expected_channel_rel, expected_channel_valid = _reference_relative_error(x, recon, mask, eps)
    channel_rel, channel_valid = scoring.channel_relative_error(x, recon, mask, eps=eps)
    # channel_relative_error deliberately computes in float32 (see its docstring -- this
    # roughly halves peak memory on SEAD's ~250k-window splits), so match float32
    # precision here rather than the float64 reference loop's exact rtol.
    np.testing.assert_allclose(channel_rel, expected_channel_rel, rtol=1e-6)
    np.testing.assert_array_equal(channel_valid, expected_channel_valid)

    # Manual channel-average (both channels available in this toy window):
    manual_average = expected_channel_rel[0].mean()
    window_score = scoring.relative_error_window_scores(x, recon, mask, eps=eps)
    assert window_score[0] == pytest.approx(manual_average, rel=1e-6)


def test_relative_error_window_with_zero_available_channels_is_nan_not_zero():
    x = np.array([[[1.0, 2.0], [1.0, 2.0]]])
    recon = np.array([[[5.0, 5.0], [5.0, 5.0]]])
    mask = np.zeros((1, 2, 2), dtype=np.float32)  # nothing valid anywhere
    channel_rel, channel_valid = scoring.channel_relative_error(x, recon, mask)
    assert not channel_valid.any()
    score = scoring.relative_error_window_scores(x, recon, mask)
    assert np.isnan(score[0])


def test_relative_error_channel_entirely_masked_is_excluded_not_zero_filled():
    # 1 window, 2 timesteps, 2 channels; channel 1 fully masked out, channel 0 has huge
    # error. If the masked channel were silently treated as 0 error, the average would
    # be dragged toward 0 -- instead it must equal channel 0's own error exactly.
    x = np.array([[[1.0, 999.0], [1.0, 999.0]]])
    recon = np.array([[[100.0, 999.0], [100.0, 999.0]]])
    mask = np.array([[[1, 0], [1, 0]]], dtype=np.float32)
    channel_rel, channel_valid = scoring.channel_relative_error(x, recon, mask)
    assert channel_valid.tolist() == [[True, False]]
    score = scoring.relative_error_window_scores(x, recon, mask)
    assert score[0] == pytest.approx(channel_rel[0, 0])


# ---------------------------------------------------------------------------
# SS1(b) rank-normalized error: hand-verified toy example
# ---------------------------------------------------------------------------

def test_rank_normalized_matches_hand_verified_toy_example():
    # 2 channels. Channel 0 reference (train-normal) squared errors: [1, 2, 3, 4, 5].
    # Channel 1 reference is EMPTY (no valid train windows) -> must be excluded, not
    # defaulted to a neutral 0.5.
    reference_mse = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    reference_valid = np.array([[True, False]] * 5)

    query_mse = np.array([[2.5, 999.0], [10.0, -5.0]])
    query_valid = np.array([[True, True], [True, True]])

    scores = scoring.per_channel_rank_normalized_scores(
        reference_mse, reference_valid, query_mse, query_valid)

    # Independently recompute channel 0's percentile with the SAME (unchanged, already
    # tested elsewhere) empirical_probability function used by the implementation --
    # this test is about correct per-channel dispatch/averaging/exclusion, not about
    # re-deriving empirical_probability's own formula.
    expected_channel0 = score_fusion.empirical_probability(
        reference_mse[:, 0], query_mse[:, 0])
    # Channel 1 has no reference -> excluded -> final score equals channel 0 alone.
    np.testing.assert_allclose(scores, expected_channel0, rtol=1e-9)


def test_rank_normalized_query_channel_invalid_in_this_window_is_excluded():
    reference_mse = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    reference_valid = np.array([[True, True]] * 3)
    query_mse = np.array([[2.0, 999.0]])
    query_valid = np.array([[True, False]])  # channel 1 not available in THIS window

    scores = scoring.per_channel_rank_normalized_scores(
        reference_mse, reference_valid, query_mse, query_valid)
    expected = score_fusion.empirical_probability(reference_mse[:, 0], query_mse[:, 0])
    np.testing.assert_allclose(scores, expected, rtol=1e-9)


def test_rank_normalized_window_with_no_available_channels_is_nan():
    reference_mse = np.array([[1.0], [2.0]])
    reference_valid = np.array([[True], [True]])
    query_mse = np.array([[5.0]])
    query_valid = np.array([[False]])
    scores = scoring.per_channel_rank_normalized_scores(
        reference_mse, reference_valid, query_mse, query_valid)
    assert np.isnan(scores[0])


# ---------------------------------------------------------------------------
# Architecture adapters: shape/consistency smoke tests on tiny synthetic windows
# ---------------------------------------------------------------------------

def test_single_recon_adapter_matches_manual_masked_mse_per_channel():
    from src.ml.models.lstm_autoencoder import LSTMAutoencoder

    torch.manual_seed(0)
    n_features = 3
    model = LSTMAutoencoder(n_features)
    model.eval()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 6, n_features)).astype(np.float32)
    mask = rng.integers(0, 2, size=(4, 6, n_features)).astype(np.float32)

    channel_mse, channel_valid, channel_rel = scoring.compute_channel_errors_single_recon(
        model, x, mask, batch_size=2)
    with torch.no_grad():
        recon = model(torch.as_tensor(x)).numpy()
    expected_mse, expected_valid = scoring.channel_squared_error(x, recon, mask)
    expected_rel, _ = scoring.channel_relative_error(x, recon, mask)
    np.testing.assert_allclose(channel_mse, expected_mse, atol=1e-5)
    np.testing.assert_array_equal(channel_valid, expected_valid)
    np.testing.assert_allclose(channel_rel, expected_rel, atol=1e-5)


def test_usad_adapter_matches_manual_alpha_beta_combination():
    from src.ml.models.usad import USAD

    torch.manual_seed(0)
    n_features = 3
    window = 6
    model = USAD(window, n_features)
    model.eval()
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, window, n_features)).astype(np.float32)
    mask = rng.integers(0, 2, size=(4, window, n_features)).astype(np.float32)

    channel_mse, channel_valid, channel_rel = scoring.compute_channel_errors_usad(
        model, x, mask, batch_size=2)
    with torch.no_grad():
        w1 = model.ae1(torch.as_tensor(x))
        w3 = model.ae2(w1)
        w1, w3 = w1.numpy(), w3.numpy()
    mse1, valid = scoring.channel_squared_error(x, w1, mask)
    mse3, _ = scoring.channel_squared_error(x, w3, mask)
    rel1, _ = scoring.channel_relative_error(x, w1, mask)
    rel3, _ = scoring.channel_relative_error(x, w3, mask)
    np.testing.assert_allclose(channel_mse, 0.5 * mse1 + 0.5 * mse3, atol=1e-5)
    np.testing.assert_array_equal(channel_valid, valid)
    np.testing.assert_allclose(channel_rel, 0.5 * rel1 + 0.5 * rel3, atol=1e-5)


# ---------------------------------------------------------------------------
# Artifact-level checks (skip until a real run exists)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name", ["lstm", "dense_ae", "usad"])
def test_kol_n_artifact_holdout_isolation_and_checksums(arch_name):
    run_dir = ROOT / "artifacts/ml16_kol_n" / arch_name / "full_matrix"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("ML-16 Kol N full_matrix kosusu henuz yapilmadi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["blind_holdout_read"] is False
    assert manifest["architecture"] == arch_name
    assert set(manifest["score_sources_evaluated"]) == {f"{arch_name}_relerr", f"{arch_name}_rankpct"}
    for relative, expected in manifest["files"].items():
        assert _sha256(run_dir / relative) == expected, relative


def test_kol_n_gate_a_passes_if_run_exists():
    for arch_name in ("lstm", "dense_ae", "usad"):
        gates_path = ROOT / "artifacts/ml16_kol_n" / arch_name / "full_matrix" / "gates.json"
        if not gates_path.exists():
            pytest.skip("ML-16 Kol N full_matrix kosusu henuz yapilmadi")
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        assert gates["gate_a"]["status"] == "passed", gates["gate_a"]


def test_kol_n_correlation_diagnostics_recorded_for_every_cell_if_run_exists():
    for arch_name in ("lstm", "dense_ae", "usad"):
        diag_path = ROOT / "artifacts/ml16_kol_n" / arch_name / "full_matrix" / "correlation_diagnostics.json"
        if not diag_path.exists():
            pytest.skip("ML-16 Kol N full_matrix kosusu henuz yapilmadi")
        records = json.loads(diag_path.read_text(encoding="utf-8"))
        assert len(records) == 5  # one per split
        for record in records:
            for variant in ("relerr", "rankpct"):
                cell = record["correlations"][variant]
                assert "trained_vs_untrained_random_init_spearman" in cell
                assert "trained_vs_magnitude_only_spearman" in cell
