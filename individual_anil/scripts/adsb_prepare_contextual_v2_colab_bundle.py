"""Prepare a hash-locked, sharded Colab transfer bundle for ADS-B v2 training.

The bundle contains only the frozen fit inputs (237 Step-5 fit parts plus the
547 fit-expansion parts), the already-computed natural-fit scaler, and the
minimal code/contract files needed by the Colab runner.  Parquet files are
stored without recompression because they are already compressed; shards make
Google Drive uploads resumable and avoid one fragile ~18 GiB browser upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


STEP5_MANIFEST = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json")
EXPANSION_MANIFEST = Path(
    "artifacts/adsb/runs/20260723_step5_v2_fit_expansion/run_manifest.json"
)
TRAIN_CONFIG = Path("configs/adsb_contextual_physics_v2_train.json")
PREREG = Path("docs/adsb_contextual_physics_v2_prereg_20260723.md")
SCALER = Path("artifacts/adsb/runs/20260724_contextual_physics_v2_train_v4/fit_scaler.json")
EXPECTED_DAYS = ("2024-09-01", "2025-02-15", "2025-06-15")
EXPECTED_STEP5_PARTS = 237
EXPECTED_EXPANSION_PARTS = 547


class ColabBundleError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ColabBundleError(f"Expected JSON object: {path}")
    return value


def _safe_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ColabBundleError(f"Source escapes repository root: {relative}") from exc
    forbidden = {part.lower() for part in path.parts} & {"archive", "downloads", "raw"}
    if forbidden:
        raise ColabBundleError(f"Forbidden source path: {relative}")
    return path


def _collect_data_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    step_path = (root / STEP5_MANIFEST).resolve(strict=True)
    expansion_path = (root / EXPANSION_MANIFEST).resolve(strict=True)
    config = _load_json((root / TRAIN_CONFIG).resolve(strict=True))
    step = _load_json(step_path)
    expansion = _load_json(expansion_path)

    if config.get("candidate_namespace") != "contextual_physics_v2":
        raise ColabBundleError("Unexpected candidate namespace")
    observed_step_hash = _sha256_file(step_path)
    if observed_step_hash != config.get("source_step5_manifest_sha256"):
        raise ColabBundleError("Frozen Step-5 manifest SHA-256 mismatch")
    days = tuple(record.get("source_day") for record in expansion.get("days", []))
    if days != EXPECTED_DAYS:
        raise ColabBundleError(f"Unexpected fit-expansion day set/order: {days}")
    if expansion.get("base_step5_manifest_sha256") != observed_step_hash:
        raise ColabBundleError("Fit expansion was built against another Step-5 manifest")

    step_records = [record for record in step.get("inputs", []) if record.get("role") == "fit"]
    if len(step_records) != EXPECTED_STEP5_PARTS:
        raise ColabBundleError(f"Expected {EXPECTED_STEP5_PARTS} Step-5 fit parts")

    records: list[dict[str, Any]] = []
    for record in step_records:
        records.append(
            {
                "role": "fit_step5",
                "source_day": "2026-02-28",
                "path": str(record["path"]).replace("\\", "/"),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
        )
    expansion_count = 0
    for day_record in expansion["days"]:
        for record in day_record["files"]:
            records.append(
                {
                    "role": "fit_expansion",
                    "source_day": str(day_record["source_day"]),
                    "path": str(record["path"]).replace("\\", "/"),
                    "bytes": int(record["bytes"]),
                    "sha256": str(record["sha256"]),
                }
            )
            expansion_count += 1
    if expansion_count != EXPECTED_EXPANSION_PARTS:
        raise ColabBundleError(f"Expected {EXPECTED_EXPANSION_PARTS} expansion parts")
    if len({record["path"] for record in records}) != len(records):
        raise ColabBundleError("Duplicate data path in frozen fit inputs")
    return records, step, expansion


def _support_paths(root: Path) -> list[Path]:
    explicit = [
        TRAIN_CONFIG,
        PREREG,
        STEP5_MANIFEST,
        EXPANSION_MANIFEST,
        SCALER,
        Path("requirements.txt"),
        Path("scripts/adsb_train_contextual_physics_v2.py"),
        Path("scripts/adsb_contextual_v2_colab_runner.py"),
    ]
    paths = [(root / path).resolve(strict=True) for path in explicit]
    paths.extend(sorted((root / "adsb").rglob("*.py")))
    unique: dict[str, Path] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        unique[relative] = path
    return [unique[key] for key in sorted(unique)]


def _partition(records: Iterable[dict[str, Any]], maximum_bytes: int) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for record in records:
        size = int(record["bytes"])
        if size > maximum_bytes:
            raise ColabBundleError(
                f"Single input exceeds shard size ({size} > {maximum_bytes}): {record['path']}"
            )
        if current and current_bytes + size > maximum_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def _write_zip_atomic(path: Path, root: Path, relatives: Iterable[str]) -> dict[str, Any]:
    temporary = path.with_name(f"{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for relative in relatives:
            archive.write(root / relative, arcname=relative)
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_bundle(
    *,
    root: Path,
    output_dir: Path,
    shard_size_gib: float,
    build_archives: bool,
    verify_source_hashes: bool,
) -> dict[str, Any]:
    repo_root = root.resolve(strict=True)
    destination = output_dir.resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    records, step, expansion = _collect_data_records(repo_root)

    totals_by_day: dict[str, dict[str, int]] = {}
    for number, record in enumerate(records, start=1):
        path = _safe_source(repo_root, record["path"])
        if path.stat().st_size != record["bytes"]:
            raise ColabBundleError(f"Byte-size mismatch: {record['path']}")
        if verify_source_hashes and _sha256_file(path) != record["sha256"]:
            raise ColabBundleError(f"SHA-256 mismatch: {record['path']}")
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        record["parquet_rows"] = rows
        summary = totals_by_day.setdefault(record["source_day"], {"parts": 0, "bytes": 0, "rows": 0})
        summary["parts"] += 1
        summary["bytes"] += record["bytes"]
        summary["rows"] += rows
        if number % 50 == 0 or number == len(records):
            print(f"manifest_progress={number}/{len(records)}", flush=True)

    support = []
    for path in _support_paths(repo_root):
        support.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    scaler_payload = _load_json((repo_root / SCALER).resolve(strict=True))
    if scaler_payload.get("scaler", {}).get("fit_role") != "natural_clean_fit":
        raise ColabBundleError("Bundled scaler is not natural_clean_fit")

    manifest = {
        "schema_version": 1,
        "bundle_role": "contextual_physics_v2_colab_transfer",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_namespace": "contextual_physics_v2",
        "frozen_training_parameters_changed": False,
        "source_step5_manifest_sha256": _sha256_file(repo_root / STEP5_MANIFEST),
        "fit_expansion_manifest_sha256": expansion["fit_expansion_sha256"],
        "fit_expansion_days": list(EXPECTED_DAYS),
        "data": records,
        "support_files": support,
        "totals": {
            "parts": len(records),
            "bytes": sum(record["bytes"] for record in records),
            "parquet_rows": sum(record["parquet_rows"] for record in records),
            "by_day": totals_by_day,
        },
        "source_hashes_verified_during_build": verify_source_hashes,
        "notes": [
            "Parquet is already compressed; transfer ZIP shards use ZIP_STORED.",
            "The Colab runner must verify every extracted source before training.",
            "The bundled scaler is reused exactly; no threshold or model hyperparameter is changed.",
        ],
    }
    _write_json_atomic(destination / "bundle_manifest.json", manifest)

    transfer: dict[str, Any] = {
        "schema_version": 1,
        "bundle_manifest": {
            "path": "bundle_manifest.json",
            "bytes": (destination / "bundle_manifest.json").stat().st_size,
            "sha256": _sha256_file(destination / "bundle_manifest.json"),
        },
        "archives_built": build_archives,
        "archives": [],
    }
    if build_archives:
        code_relative = [record["path"] for record in support]
        code_info = _write_zip_atomic(destination / "contextual_v2_code_and_contract.zip", repo_root, code_relative)
        code_info.update({"kind": "code_and_contract", "files": len(code_relative)})
        transfer["archives"].append(code_info)

        maximum = int(shard_size_gib * (1024**3))
        shards = _partition(records, maximum)
        for index, shard in enumerate(shards, start=1):
            name = f"contextual_v2_data_{index:03d}_of_{len(shards):03d}.zip"
            print(
                f"archive_start={index}/{len(shards)} files={len(shard)} "
                f"source_gib={sum(r['bytes'] for r in shard) / 2**30:.3f}",
                flush=True,
            )
            info = _write_zip_atomic(destination / name, repo_root, (record["path"] for record in shard))
            info.update(
                {
                    "kind": "data",
                    "files": len(shard),
                    "source_bytes": sum(record["bytes"] for record in shard),
                }
            )
            transfer["archives"].append(info)
            _write_json_atomic(destination / "transfer_index.json", transfer)
            print(f"archive_done={index}/{len(shards)} sha256={info['sha256']}", flush=True)
    _write_json_atomic(destination / "transfer_index.json", transfer)
    instructions = destination / "UPLOAD_TO_GOOGLE_DRIVE.txt"
    instructions.write_text(
        "Upload every file in this directory to one Google Drive folder.\n"
        "Open notebooks/adsb_contextual_physics_v2_colab.ipynb in Colab,\n"
        "set BUNDLE_DIR to that Drive folder, then run cells from top to bottom.\n"
        "Do not rename individual ZIP shards.\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "transfer": transfer, "output_dir": str(destination)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/adsb/colab/contextual_v2_transfer"),
    )
    parser.add_argument("--shard-size-gib", type=float, default=1.8)
    parser.add_argument("--build-archives", action="store_true")
    parser.add_argument("--verify-source-hashes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (0.25 <= args.shard_size_gib <= 4.0):
        raise ColabBundleError("--shard-size-gib must be in [0.25, 4.0]")
    result = build_bundle(
        root=args.repo_root,
        output_dir=args.output_dir,
        shard_size_gib=args.shard_size_gib,
        build_archives=args.build_archives,
        verify_source_hashes=args.verify_source_hashes,
    )
    totals = result["manifest"]["totals"]
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": result["output_dir"],
                "parts": totals["parts"],
                "rows": totals["parquet_rows"],
                "gib": round(totals["bytes"] / 2**30, 3),
                "archives_built": result["transfer"]["archives_built"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
