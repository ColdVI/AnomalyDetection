"""ML-11 gorselleme fazi disiplin testleri.

(a) subsample deterministik, (b) blind holdout hicbir gorselde kullanilmadi,
(c) viz manifest checksum'lari dosyalarla eslesiyor, (d) egitim izi altyapisi
(epoch history + loss.csv/png) calisiyor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
VIZ_ROOT = ROOT / "artifacts/viz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_deterministic_subsample_is_repeatable_and_capped():
    from scripts.make_visualizations import deterministic_subsample_index

    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "source_id": np.repeat([f"f{i}" for i in range(5)], 400),
        "value": rng.normal(size=2000),
    })
    first = deterministic_subsample_index(frame, per_flight_cap=100, max_points=300)
    second = deterministic_subsample_index(frame, per_flight_cap=100, max_points=300)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 300
    per_flight = frame.loc[first].groupby("source_id").size()
    assert per_flight.max() <= 100


def test_load_development_never_returns_holdout_flights():
    from scripts.make_visualizations import load_development

    frame, dev_ids, holdout_ids, flight_labels, _ = load_development(
        "uav_sead", columns=["source_id"])
    assert len(holdout_ids) == 131
    assert not set(dev_ids) & set(holdout_ids)
    assert not set(frame["source_id"].unique()) & set(holdout_ids)
    assert set(dev_ids) | set(holdout_ids) == set(flight_labels)


@pytest.mark.parametrize("source", ["alfa", "uav_attack", "uav_sead"])
def test_viz_manifest_checksums_and_holdout_isolation(source):
    manifest_path = VIZ_ROOT / source / "viz_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"{source} icin viz kosusu henuz yapilmadi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # checksum'lar dosyalarla eslesmeli
    assert manifest["files"], "viz manifest bos"
    for relative, expected in manifest["files"].items():
        assert _sha256(VIZ_ROOT / source / relative) == expected, relative

    # holdout izolasyonu: kosuda kullanilan development id kumesi, split
    # manifest'ten bagimsiz yeniden turetilenle ayni olmali (holdout haric).
    assert manifest["blind_holdout_read"] is False
    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = split_manifest["sources"][source]
    holdout = set(config["splits"]["split_00"].get("final_holdout", []))
    expected_dev = sorted(set(config["flight_labels"]) - holdout)
    expected_hash = hashlib.sha256(
        "\n".join(expected_dev).encode("utf-8")).hexdigest()
    assert manifest["development_source_ids_sha256"] == expected_hash
    assert manifest["development_flights"] == len(expected_dev)
    assert manifest["blind_holdout_flights"] == len(holdout)
    if source == "uav_sead":
        assert manifest["blind_holdout_flights"] == 131
        assert manifest["development_flights"] == 480


def test_lstm_training_returns_epoch_history():
    torch = pytest.importorskip("torch")
    from src.ml.models.lstm_autoencoder import (
        LSTMAutoencoder, train_lstm_autoencoder)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 8, 3)).astype(np.float32)
    m = np.ones_like(x, dtype=np.float32)
    model, info = train_lstm_autoencoder(
        LSTMAutoencoder(3), x[:8], m[:8], x[8:], m[8:], seed=0, epochs=3,
        batch_size=4, patience=5)
    history = info["history"]
    assert len(history) == info["epochs"]
    assert {"epoch", "train_loss", "val_loss"} <= set(history[0])
    assert info["best_val_loss"] == min(h["val_loss"] for h in history)


def test_write_training_log_produces_csv_and_png(tmp_path):
    from src.ml.training_log import write_training_log

    history = [{"epoch": i + 1, "train_loss": 1.0 / (i + 1),
                "val_loss": 1.2 / (i + 1)} for i in range(4)]
    out = write_training_log(history, "alfa", "ml6_lstm_ae", "run_test",
                             root=tmp_path)
    assert out == tmp_path / "alfa" / "ml6_lstm_ae" / "run_test"
    frame = pd.read_csv(out / "loss.csv")
    assert list(frame.columns) == ["epoch", "train_loss", "val_loss"]
    assert len(frame) == 4
    assert (out / "loss.png").exists()
    with pytest.raises(ValueError):
        write_training_log([], "alfa", "ml6_lstm_ae", "bos", root=tmp_path)
