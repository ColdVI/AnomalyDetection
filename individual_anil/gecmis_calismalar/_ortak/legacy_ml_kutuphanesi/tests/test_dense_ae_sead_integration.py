"""ML-16 Kol D SEAD Dense-AE discipline tests (docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md).

Covers: reuse-not-reimplemented identity checks (Gate A prerequisites), causal
no-future-leak scoring, train/val-only windowing (no test-flight leakage),
merge_asof-backward alignment correctness, val-only calibration, architecture
sanity (parameter count vs LSTM-AE, shape roundtrip), and artifact-level
holdout isolation once a real run exists.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import run_ml8a_temporal_boosting as ml8a_runner
from scripts import run_ml9_category_evaluation as ml9_runner
from scripts import run_ml_dense_ae_sead_evaluation as dense_runner
from src.ml.data import windowing
from src.ml.evaluation import score_fusion
from src.ml.models import dense_autoencoder, lstm_autoencoder

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
RUN_DIR = ROOT / "artifacts/ml_dense_ae_sead/uav_sead/full_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_sequence(n_flights: int = 4, rows_per_flight: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = dense_runner.AE_COLS
    frames = []
    for i in range(n_flights):
        t = np.arange(rows_per_flight, dtype=float) * 0.1
        data = {"source_id": f"flight-{i}", "t_rel_s": t, "label": "normal"}
        for col in cols:
            data[col] = rng.normal(size=rows_per_flight)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Reuse-not-reimplemented identity tests (Gate A prerequisites)
# ---------------------------------------------------------------------------

def test_dense_ae_reuses_decision_and_fusion_helpers_not_reimplemented():
    assert dense_runner._fit_policies is ml9_runner._fit_policies
    assert dense_runner._evaluate is ml9_runner._evaluate
    assert dense_runner._score_modules is ml9_runner._score_modules
    assert dense_runner._streams is ml9_runner._streams
    assert dense_runner.max_score_fusion is score_fusion.max_score_fusion
    assert dense_runner.last_causal_per_bucket is score_fusion.last_causal_per_bucket
    assert dense_runner.empirical_probability is score_fusion.empirical_probability
    assert dense_runner._align_score is ml8a_runner._align_score
    assert dense_runner.build_windows is windowing.build_windows
    assert dense_runner.train_dense_autoencoder is dense_autoencoder.train_dense_autoencoder
    assert dense_runner.reconstruction_scores is dense_autoencoder.reconstruction_scores
    assert dense_runner.DenseAutoencoder is dense_autoencoder.DenseAutoencoder
    # Dense AE re-exports (not reimplements) the LSTM family's mask-aware loss/scoring helpers.
    assert dense_autoencoder.masked_mse is lstm_autoencoder.masked_mse
    assert dense_autoencoder.reconstruction_scores is lstm_autoencoder.reconstruction_scores


def test_dense_ae_budgets_and_targets_match_project_wide_frozen_values():
    assert ml9_runner.BUDGETS == {"critical": 2.0, "advisory": 12.0}
    assert ml9_runner.MIN_RECALL == {"critical": 0.30, "advisory": 0.50}
    assert dense_runner.BUDGETS is ml9_runner.BUDGETS
    assert dense_runner.MIN_RECALL is ml9_runner.MIN_RECALL


def test_dense_ae_score_sources_match_preregistered_plan():
    # docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md SS1: exactly (a)/(b)/(c), no ad hoc variants.
    assert dense_runner.SCORE_SOURCES == (
        "dense_ae_recon", "dense_ae_ml14_fusion", "dense_ae_itki_fusion",
    )


def test_dense_ae_features_window_stride_match_existing_uav_sead_config():
    # docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md SS2: same columns/window/stride as LSTM-AE family.
    assert dense_runner.AE_COLS == lstm_autoencoder.AE_FEATURES["uav_sead"]
    assert dense_runner.DENSE_WINDOW == lstm_autoencoder.WINDOW["uav_sead"]
    assert dense_runner.DENSE_STRIDE == lstm_autoencoder.STRIDE["uav_sead"]


# ---------------------------------------------------------------------------
# Architecture sanity
# ---------------------------------------------------------------------------

def test_dense_ae_parameter_count_roughly_matches_lstm_ae():
    """docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md SS2: capacity chosen to be close to
    LSTMAutoencoder's so a performance gap reflects architecture, not capacity."""
    lstm_params = sum(p.numel() for p in lstm_autoencoder.LSTMAutoencoder(22).parameters())
    dense_params = sum(
        p.numel() for p in dense_autoencoder.DenseAutoencoder(50, 22).parameters())
    assert lstm_params == 17414
    assert dense_params == 16574
    ratio = dense_params / lstm_params
    assert 0.9 <= ratio <= 1.1


def test_dense_ae_forward_shape_roundtrip():
    model = dense_autoencoder.DenseAutoencoder(50, 22)
    x = torch.randn(5, 50, 22)
    out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# No-leak / determinism / alignment behavioral tests on synthetic data
# ---------------------------------------------------------------------------

def test_dense_ae_training_windows_unaffected_by_other_flight_corruption():
    """Corrupting test-only flights must not change the train/val windows fed to fitting."""
    sequence = _synthetic_sequence(n_flights=6)
    train_ids = {"flight-0", "flight-1"}
    val_ids = {"flight-2", "flight-3"}
    test_ids = {"flight-4", "flight-5"}

    corrupted = sequence.copy()
    mask = corrupted["source_id"].isin(test_ids)
    for col in dense_runner.AE_COLS:
        corrupted.loc[mask, col] = corrupted.loc[mask, col] + 1_000_000.0

    for ids in (train_ids, val_ids):
        x1, m1, meta1 = windowing.build_windows(
            sequence[sequence["source_id"].isin(ids)], dense_runner.AE_COLS,
            window=dense_runner.DENSE_WINDOW, stride=dense_runner.DENSE_STRIDE,
            max_gap_s=dense_runner.MAX_GAP_S,
        )
        x2, m2, meta2 = windowing.build_windows(
            corrupted[corrupted["source_id"].isin(ids)], dense_runner.AE_COLS,
            window=dense_runner.DENSE_WINDOW, stride=dense_runner.DENSE_STRIDE,
            max_gap_s=dense_runner.MAX_GAP_S,
        )
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(m1, m2)
        pd.testing.assert_frame_equal(meta1, meta2)


def test_dense_ae_training_deterministic_given_reset_rng_state():
    sequence = _synthetic_sequence(n_flights=2, rows_per_flight=250)
    train_ids = {"flight-0"}
    val_ids = {"flight-1"}

    torch.manual_seed(7)
    model_a, training_a, _ = dense_runner._train_dense_ae(sequence, train_ids, val_ids, seed=7)
    torch.manual_seed(7)
    model_b, training_b, _ = dense_runner._train_dense_ae(sequence, train_ids, val_ids, seed=7)

    assert training_a["best_val_loss"] == pytest.approx(training_b["best_val_loss"])
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b)


def test_dense_ae_scoring_is_causal_no_future_leak():
    sequence = _synthetic_sequence(n_flights=3, rows_per_flight=300)
    train_ids = {"flight-0"}
    val_ids = {"flight-1"}
    score_ids = {"flight-2"}
    model, _, _ = dense_runner._train_dense_ae(sequence, train_ids, val_ids, seed=1)

    original = dense_runner._score_dense_ae(model, sequence, score_ids)
    changed = sequence.copy()
    tail = changed["source_id"].eq("flight-2") & (changed["t_rel_s"] >= 20.0)
    for col in dense_runner.AE_COLS:
        changed.loc[tail, col] = changed.loc[tail, col] + 500.0
    changed_scores = dense_runner._score_dense_ae(model, changed, score_ids)

    prefix = original["t_rel_s"] < 20.0
    pd.testing.assert_frame_equal(
        original.loc[prefix].reset_index(drop=True),
        changed_scores.loc[changed_scores["t_rel_s"] < 20.0].reset_index(drop=True),
    )
    assert int(prefix.sum()) > 0


def test_align_score_causal_carry_forward_matches_merge_asof_and_shape():
    sequence = _synthetic_sequence(n_flights=2, rows_per_flight=200)
    train_ids = {"flight-0"}
    val_ids = {"flight-1"}
    model, _, _ = dense_runner._train_dense_ae(sequence, train_ids, val_ids, seed=2)
    window_scores = dense_runner._score_dense_ae(model, sequence, val_ids)

    endpoints = (
        sequence[sequence["source_id"].isin(val_ids)][["source_id", "t_rel_s"]]
        .sort_values("t_rel_s").reset_index(drop=True)
    )
    aligned = dense_runner._align_score(endpoints, window_scores, "dense_ae_recon_raw")

    assert len(aligned) == len(endpoints)
    first_window_end = window_scores["t_rel_s"].min()
    before = endpoints["t_rel_s"].to_numpy() < first_window_end
    assert before.any() and (~before).any()
    assert np.isnan(aligned[before]).all()
    assert np.isfinite(aligned[~before]).all()

    expected = pd.merge_asof(
        endpoints, window_scores.sort_values("t_rel_s"),
        on="t_rel_s", direction="backward",
    )["dense_ae_recon_raw"].to_numpy()
    np.testing.assert_array_equal(aligned, expected)


def test_dense_ae_recon_calibration_uses_only_val_normal_reference():
    source = inspect.getsource(dense_runner.run)
    assert 'scored.loc[val_mask, "dense_ae_recon_raw"]' in source
    assert 'val_mask = scored["source_id"].isin(parts["val"])' in source


def test_dense_ae_fusion_variants_are_max_of_registered_partners():
    source = inspect.getsource(dense_runner.run)
    assert 'max_score_fusion(scored, ["dense_ae_recon", "ml14_fusion"])' in source
    assert 'max_score_fusion(scored, ["dense_ae_recon", "itki_komutu"])' in source


# ---------------------------------------------------------------------------
# Artifact-level checks (skip until a real run exists)
# ---------------------------------------------------------------------------

def test_dense_ae_artifact_holdout_isolation_and_checksums():
    manifest_path = RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("ML-16 Kol D full_matrix kosusu henuz yapilmadi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["blind_holdout_read"] is False
    assert manifest["score_sources_evaluated"] == [
        "dense_ae_recon", "dense_ae_ml14_fusion", "dense_ae_itki_fusion",
    ]
    for relative, expected in manifest["files"].items():
        assert _sha256(RUN_DIR / relative) == expected, relative

    if manifest.get("split_manifest_sha256") != _sha256(SPLIT_PATH):
        pytest.skip("eski veri donemi artifact'i")
    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = split_manifest["sources"]["uav_sead"]
    holdout = set(config["splits"]["split_00"]["final_holdout"])
    expected_dev = sorted(set(config["flight_labels"]) - holdout)
    expected_hash = hashlib.sha256("\n".join(expected_dev).encode("utf-8")).hexdigest()
    assert manifest["development_source_ids_sha256"] == expected_hash


def test_dense_ae_gate_a_determinism_passes_if_run_exists():
    gates_path = RUN_DIR / "gates.json"
    if not gates_path.exists():
        pytest.skip("ML-16 Kol D full_matrix kosusu henuz yapilmadi")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    assert gates["gate_a"]["status"] == "passed", gates["gate_a"]
