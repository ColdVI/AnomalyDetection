"""RFLY-1 severity sweep scaffold for frozen ML-14 artifacts.

The script inventories frozen split_00 model files and simulation TestInfo
severity values. It does not train or update model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.silver.parse_rflymad import DEFAULT_BRONZE_DIR, SIM_SUBSETS  # noqa: E402

OUT_DIR = ROOT / "artifacts/rfly1/severity_sweep"
DEFAULT_MODEL_DIR = ROOT / "artifacts/ml14/uav_sead/full_matrix/split_00/models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_model_inventory(model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    files = sorted(model_dir.glob("*.joblib")) if model_dir.exists() else []
    return {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "model_count": len(files),
        "models": {path.name: sha256_file(path) for path in files},
    }


def _simulation_testinfo_files(bronze_dir: Path = DEFAULT_BRONZE_DIR) -> list[Path]:
    files: list[Path] = []
    for subset in SIM_SUBSETS:
        root = bronze_dir / subset
        if root.exists():
            files.extend(sorted(root.rglob("TestInfo.csv")))
            files.extend(sorted(root.rglob("TestInfo_*.xlsx")))
    return files


def extract_fault_parameters_from_csv(path: Path) -> list[str]:
    values: list[str] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key, value in row.items():
                    if key and key.strip().lower() == "fault parameter" and value not in {None, ""}:
                        values.append(str(value).strip())
    except UnicodeDecodeError:
        with path.open(encoding="latin-1", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key, value in row.items():
                    if key and key.strip().lower() == "fault parameter" and value not in {None, ""}:
                        values.append(str(value).strip())
    return values


def severity_preflight(bronze_dir: Path = DEFAULT_BRONZE_DIR, model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    info_files = _simulation_testinfo_files(bronze_dir)
    params: list[str] = []
    for path in info_files:
        if path.suffix.lower() == ".csv":
            params.extend(extract_fault_parameters_from_csv(path))
    inventory = frozen_model_inventory(model_dir)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "RFLY-1 frozen ML-14 severity sweep preflight",
        "status": "ready" if info_files else "blocked_no_simulation_testinfo",
        "frozen_ml14_model_inventory": inventory,
        "model_files_modified": False,
        "simulation_testinfo_count": len(info_files),
        "fault_parameter_count": len(params),
        "fault_parameter_values": dict(sorted(Counter(params).items())),
        "severity_bucket_policy": "not frozen until TestInfo Fault Parameter distribution is available",
    }


def write_preflight(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = severity_preflight()
    path = out_dir / "preflight.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(write_preflight())
        return
    report_path = write_preflight()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "ready":
        raise SystemExit(f"severity sweep blocked: {report['status']} ({report_path})")
    raise SystemExit("severity sweep scoring is held until severity buckets are frozen from TestInfo distribution")


if __name__ == "__main__":
    main()
