"""ML-16 Kol U SEAD USAD discipline tests (docs/ML16_KOL_U_USAD_SEAD_PLAN.md).

Covers: reuse-not-reimplemented identity checks (Gate A prerequisites), causal
no-future-leak scoring, train/val-only windowing (no test-flight leakage),
merge_asof-backward alignment correctness, val-only calibration, USAD-specific
architecture sanity (phase-2 adversarial term changes gradients vs phase-1-only,
output shapes match input), and artifact-level holdout isolation once a real
run exists.
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
from scripts import run_ml_usad_sead_evaluation as usad_runner
from src.ml.data import windowing
from src.ml.evaluation import score_fusion
from src.ml.models import lstm_autoencoder, usad as usad_module

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
RUN_DIR = ROOT / "artifacts/ml_usad_sead/uav_sead/full_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_sequence(n_flights: int = 4, rows_per_flight: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = usad_runner.AE_COLS
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

def test_usad_reuses_decision_and_fusion_helpers_not_reimplemented():
    assert usad_runner._fit_policies is ml9_runner._fit_policies
    assert usad_runner._evaluate is ml9_runner._evaluate
    assert usad_runner._score_modules is ml9_runner._score_modules
    assert usad_runner._streams is ml9_runner._streams
    assert usad_runner.max_score_fusion is score_fusion.max_score_fusion
    assert usad_runner.last_causal_per_bucket is score_fusion.last_causal_per_bucket
    assert usad_runner.empirical_probability is score_fusion.empirical_probability
    assert usad_runner._align_score is ml8a_runner._align_score
    assert usad_runner.build_windows is windowing.build_windows
    assert usad_runner.train_usad is usad_module.train_usad
    assert usad_runner.usad_reconstruction_scores is usad_module.usad_reconstruction_scores
    assert usad_runner.USAD is usad_module.USAD
    # USAD re-exports (not reimplements) the LSTM family's mask-aware loss helper.
    assert usad_module.masked_mse is lstm_autoencoder.masked_mse


def test_usad_budgets_and_targets_match_project_wide_frozen_values():
    assert ml9_runner.BUDGETS == {"critical": 2.0, "advisory": 12.0}
    assert ml9_runner.MIN_RECALL == {"critical": 0.30, "advisory": 0.50}
    assert usad_runner.BUDGETS is ml9_runner.BUDGETS
    assert usad_runner.MIN_RECALL is ml9_runner.MIN_RECALL


def test_usad_score_sources_match_preregistered_plan():
    # docs/ML16_KOL_U_USAD_SEAD_PLAN.md SS1: exactly (a)/(b)/(c), no ad hoc variants.
    assert usad_runner.SCORE_SOURCES == (
        "usad_score", "usad_ml14_fusion", "usad_itki_fusion",
    )


def test_usad_features_window_stride_match_existing_uav_sead_config():
    # docs/ML16_KOL_U_USAD_SEAD_PLAN.md SS2: same columns/window/stride as LSTM-AE family.
    assert usad_runner.AE_COLS == lstm_autoencoder.AE_FEATURES["uav_sead"]
    assert usad_runner.USAD_WINDOW == lstm_autoencoder.WINDOW["uav_sead"]
    assert usad_runner.USAD_STRIDE == lstm_autoencoder.STRIDE["uav_sead"]


# ---------------------------------------------------------------------------
# USAD-specific architecture sanity
# ---------------------------------------------------------------------------

def test_usad_forward_shapes_match_input():
    model = usad_module.USAD(50, 22)
    x = torch.randn(5, 50, 22)
    assert model.ae1(x).shape == x.shape
    assert model.ae2(x).shape == x.shape
    assert model.forward(x).shape == x.shape


def test_usad_phase2_adversarial_term_changes_gradients_vs_phase1_only():
    """docs/ML16_KOL_U_USAD_SEAD_PLAN.md SS3 Gate A: the adversarial term in
    L_AE2 (-(1-1/n)*||w-AE2(AE1(w))||^2) must actually move the encoder's
    gradients relative to training AE2 on a plain (non-adversarial, direct-
    reconstruction-only) loss -- otherwise "phase 2" would be a no-op copy of
    phase 1 and the implementation would not actually be USAD."""
    torch.manual_seed(0)
    model = usad_module.USAD(50, 22)
    x = torch.randn(4, 50, 22)
    mask = torch.ones(4, 50, 22)

    model.zero_grad()
    _, loss2 = model.training_losses(x, mask, epoch_n=5)
    loss2.backward()
    grad_with_adversarial = model.encoder[0].weight.grad.clone()

    model.zero_grad()
    direct_only = lstm_autoencoder.masked_mse(x, model.ae2(x), mask)
    direct_only.backward()
    grad_direct_only = model.encoder[0].weight.grad.clone()

    assert (grad_with_adversarial - grad_direct_only).abs().max().item() > 1e-6


def test_usad_alpha_beta_schedule_matches_paper_1_over_n():
    model = usad_module.USAD(50, 22)
    x = torch.randn(3, 50, 22)
    mask = torch.ones(3, 50, 22)
    with torch.no_grad():
        w1 = model.ae1(x)
        w2 = model.ae2(x)
        w3 = model.ae2(w1)
        term1 = lstm_autoencoder.masked_mse(x, w1, mask)
        term2 = lstm_autoencoder.masked_mse(x, w2, mask)
        term_adv = lstm_autoencoder.masked_mse(x, w3, mask)
        for n in (1, 2, 10):
            loss1, loss2 = model.training_losses(x, mask, epoch_n=n)
            inv_n = 1.0 / n
            expected1 = inv_n * term1 + (1 - inv_n) * term_adv
            expected2 = inv_n * term2 - (1 - inv_n) * term_adv
            assert loss1.item() == pytest.approx(expected1.item(), abs=1e-6)
            assert loss2.item() == pytest.approx(expected2.item(), abs=1e-6)
    with pytest.raises(ValueError):
        model.training_losses(x, mask, epoch_n=0)


def test_usad_score_matches_paper_alpha_beta_combination():
    torch.manual_seed(1)
    model = usad_module.USAD(50, 22)
    x = np.random.default_rng(1).normal(size=(6, 50, 22)).astype(np.float32)
    mask = np.ones_like(x, dtype=np.float32)
    scores = usad_module.usad_reconstruction_scores(model, x, mask, alpha=0.5, beta=0.5)
    assert scores.shape == (6,)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x)
        w1 = model.ae1(xt)
        w3 = model.ae2(w1)
        term1 = lstm_autoencoder.masked_mse(xt, w1, torch.tensor(mask), per_sample=True)
        term3 = lstm_autoencoder.masked_mse(xt, w3, torch.tensor(mask), per_sample=True)
        expected = (0.5 * term1 + 0.5 * term3).numpy()
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# No-leak / determinism / alignment behavioral tests on synthetic data
# ---------------------------------------------------------------------------

def test_usad_training_windows_unaffected_by_other_flight_corruption():
    """Corrupting test-only flights must not change the train/val windows fed to fitting."""
    sequence = _synthetic_sequence(n_flights=6)
    train_ids = {"flight-0", "flight-1"}
    val_ids = {"flight-2", "flight-3"}
    test_ids = {"flight-4", "flight-5"}

    corrupted = sequence.copy()
    mask = corrupted["source_id"].isin(test_ids)
    for col in usad_runner.AE_COLS:
        corrupted.loc[mask, col] = corrupted.loc[mask, col] + 1_000_000.0

    for ids in (train_ids, val_ids):
        x1, m1, meta1 = windowing.build_windows(
            sequence[sequence["source_id"].isin(ids)], usad_runner.AE_COLS,
            window=usad_runner.USAD_WINDOW, stride=usad_runner.USAD_STRIDE,
            max_gap_s=usad_runner.MAX_GAP_S,
        )
        x2, m2, meta2 = windowing.build_windows(
            corrupted[corrupted["source_id"].isin(ids)], usad_runner.AE_COLS,
            window=usad_runner.USAD_WINDOW, stride=usad_runner.USAD_STRIDE,
            max_gap_s=usad_runner.MAX_GAP_S,
        )
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(m1, m2)
        pd.testing.assert_frame_equal(meta1, meta2)


def test_usad_training_deterministic_given_reset_rng_state():
    sequence = _synthetic_sequence(n_flights=2, rows_per_flight=250)
    train_ids = {"flight-0"}
    val_ids = {"flight-1"}

    torch.manual_seed(7)
    model_a, training_a, _ = usad_runner._train_usad(sequence, train_ids, val_ids, seed=7)
    torch.manual_seed(7)
    model_b, training_b, _ = usad_runner._train_usad(sequence, train_ids, val_ids, seed=7)

    assert training_a["best_val_loss"] == pytest.approx(training_b["best_val_loss"])
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b)


def test_usad_scoring_is_causal_no_future_leak():
    sequence = _synthetic_sequence(n_flights=3, rows_per_flight=300)
    train_ids = {"flight-0"}
    val_ids = {"flight-1"}
    score_ids = {"flight-2"}
    model, _, _ = usad_runner._train_usad(sequence, train_ids, val_ids, seed=1)

    original = usad_runner._score_usad(model, sequence, score_ids)
    changed = sequence.copy()
    tail = changed["source_id"].eq("flight-2") & (changed["t_rel_s"] >= 20.0)
    for col in usad_runner.AE_COLS:
        changed.loc[tail, col] = changed.loc[tail, col] + 500.0
    changed_scores = usad_runner._score_usad(model, changed, score_ids)

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
    model, _, _ = usad_runner._train_usad(sequence, train_ids, val_ids, seed=2)
    window_scores = usad_runner._score_usad(model, sequence, val_ids)

    endpoints = (
        sequence[sequence["source_id"].isin(val_ids)][["source_id", "t_rel_s"]]
        .sort_values("t_rel_s").reset_index(drop=True)
    )
    aligned = usad_runner._align_score(endpoints, window_scores, "usad_score_raw")

    assert len(aligned) == len(endpoints)
    first_window_end = window_scores["t_rel_s"].min()
    before = endpoints["t_rel_s"].to_numpy() < first_window_end
    assert before.any() and (~before).any()
    assert np.isnan(aligned[before]).all()
    assert np.isfinite(aligned[~before]).all()

    expected = pd.merge_asof(
        endpoints, window_scores.sort_values("t_rel_s"),
        on="t_rel_s", direction="backward",
    )["usad_score_raw"].to_numpy()
    np.testing.assert_array_equal(aligned, expected)


def test_usad_score_calibration_uses_only_val_normal_reference():
    source = inspect.getsource(usad_runner.run)
    assert 'scored.loc[val_mask, "usad_score_raw"]' in source
    assert 'val_mask = scored["source_id"].isin(parts["val"])' in source


def test_usad_fusion_variants_are_max_of_registered_partners():
    source = inspect.getsource(usad_runner.run)
    assert 'max_score_fusion(scored, ["usad_score", "ml14_fusion"])' in source
    assert 'max_score_fusion(scored, ["usad_score", "itki_komutu"])' in source


# ---------------------------------------------------------------------------
# Artifact-level checks (skip until a real run exists)
# ---------------------------------------------------------------------------

def test_usad_artifact_holdout_isolation_and_checksums():
    manifest_path = RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("ML-16 Kol U full_matrix kosusu henuz yapilmadi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["blind_holdout_read"] is False
    assert manifest["score_sources_evaluated"] == [
        "usad_score", "usad_ml14_fusion", "usad_itki_fusion",
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


def test_usad_gate_a_determinism_passes_if_run_exists():
    gates_path = RUN_DIR / "gates.json"
    if not gates_path.exists():
        pytest.skip("ML-16 Kol U full_matrix kosusu henuz yapilmadi")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    assert gates["gate_a"]["status"] == "passed", gates["gate_a"]
