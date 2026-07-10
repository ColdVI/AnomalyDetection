"""RFLY-1 frozen severity-sweep discipline tests."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from scripts.run_rfly1_severity_sweep import frozen_model_inventory, sha256_file


def test_severity_sweep_source_does_not_train_models():
    source = Path("scripts/run_rfly1_severity_sweep.py").read_text(encoding="utf-8")
    assert "IsolationForest" not in source
    assert "fit_modular_iforest" not in source


def test_frozen_model_inventory_is_checksum_only():
    root = Path(".tmp_test_rfly1_severity") / uuid.uuid4().hex
    try:
        model_dir = root / "models"
        model_dir.mkdir(parents=True)
        model = model_dir / "itki_komutu.joblib"
        model.write_bytes(b"frozen-model-bytes")
        before = sha256_file(model)

        inventory = frozen_model_inventory(model_dir)

        assert inventory["exists"] is True
        assert inventory["model_count"] == 1
        assert inventory["models"] == {"itki_komutu.joblib": before}
        assert sha256_file(model) == before
    finally:
        shutil.rmtree(root, ignore_errors=True)
