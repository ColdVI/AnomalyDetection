"""Deploy edilebilir model artifact bundle yazma/yukleme yardimcilari."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib


def _write_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_modular_iforest_bundle(fitted: dict, output_dir: str | Path, *,
                                scaler_params: dict, cusum_baselines: dict,
                                metadata: dict) -> Path:
    """IF modelleri + feature/esik/scaler/CUSUM/surum manifestini birlikte yaz."""
    out = Path(output_dir)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    calibration = {"modules": {}}
    written: list[Path] = []
    for name, item in fitted.items():
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(item["model"], model_path)
        written.append(model_path)
        calibration["modules"][name] = {
            "feature_columns": item["feature_columns"],
            "row_threshold_q99": item["row_threshold_q99"],
            "flight_threshold_max": item["flight_threshold_max"],
        }
    for filename, value in [
        ("calibration.json", calibration),
        ("scaler.json", scaler_params),
        ("cusum_baseline.json", cusum_baselines),
    ]:
        path = out / filename
        _write_json(value, path)
        written.append(path)

    manifest = {
        "artifact_schema_version": 1,
        "model_type": "modular_isolation_forest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "files": {str(p.relative_to(out)).replace("\\", "/"): _sha256(p) for p in written},
    }
    manifest_path = out / "manifest.json"
    _write_json(manifest, manifest_path)
    return manifest_path


def load_modular_iforest_bundle(output_dir: str | Path) -> tuple[dict, dict]:
    """Bundle'i yukle ve dosya hash'lerini dogrula."""
    out = Path(output_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if _sha256(out / rel) != expected:
            raise ValueError(f"Artifact checksum uyusmazligi: {rel}")
    calibration = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    fitted = {}
    for name, params in calibration["modules"].items():
        fitted[name] = {
            "model": joblib.load(out / "models" / f"{name}.joblib"),
            **params,
        }
    return fitted, manifest


def save_torch_checkpoint(model, output_path: str | Path, *, metadata: dict) -> Path:
    """LSTM/TCN gibi torch modellerini state_dict + metadata ile kaydet."""
    import torch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return path


def save_lstm_bundle(model, output_dir: str | Path, *, scaler_params: dict,
                     calibration: dict, metadata: dict) -> Path:
    """LSTM-AE checkpoint + scaler + threshold + checksum manifesti yaz."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = save_torch_checkpoint(model, out / "model.pt", metadata=metadata)
    scaler_path, calibration_path = out / "scaler.json", out / "calibration.json"
    _write_json(scaler_params, scaler_path)
    _write_json(calibration, calibration_path)
    written = [checkpoint, scaler_path, calibration_path]
    manifest = {
        "artifact_schema_version": 1,
        "model_type": "lstm_autoencoder",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "files": {p.name: _sha256(p) for p in written},
    }
    path = out / "manifest.json"
    _write_json(manifest, path)
    return path


def load_lstm_bundle(output_dir: str | Path):
    """Load an LSTM-AE bundle after verifying every recorded checksum."""

    import torch
    from src.ml.models.lstm_autoencoder import LSTMAutoencoder

    out = Path(output_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if _sha256(out / rel) != expected:
            raise ValueError(f"Artifact checksum uyusmazligi: {rel}")
    checkpoint = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model = LSTMAutoencoder(len(manifest["feature_columns"]))
    model.load_state_dict(checkpoint["state_dict"])
    scaler = json.loads((out / "scaler.json").read_text(encoding="utf-8"))
    calibration = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    return model, scaler, calibration, manifest
