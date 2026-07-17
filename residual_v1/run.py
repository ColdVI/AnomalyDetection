"""Immutable run-directory and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from residual_v1.ingest.common import write_json

DEFAULT_RUN_ROOT = Path("artifacts/residual_v1/runs")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def create_run_dir(
    task: str,
    *,
    seed: int,
    config_paths: Iterable[str | Path] = (),
    input_paths: Iterable[str | Path] = (),
    run_root: str | Path = DEFAULT_RUN_ROOT,
    timestamp: datetime | None = None,
) -> tuple[Path, dict]:
    now = timestamp or datetime.now(timezone.utc)
    safe_task = "".join(character if character.isalnum() or character in "-_" else "_" for character in task)
    destination = Path(run_root) / f"{now:%Y%m%d_%H%M%S}_{safe_task}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    configs = {str(Path(path)): sha256_file(path) for path in config_paths}
    inputs = {str(Path(path)): sha256_file(path) for path in input_paths if Path(path).is_file()}
    manifest = {
        "schema_version": 1,
        "created_utc": now.isoformat(),
        "task": task,
        "seed": int(seed),
        "git_sha": git_sha(),
        "config_hash_sha256": sha256_payload(configs),
        "config_files": configs,
        "input_files": inputs,
    }
    write_json(destination / "manifest.json", manifest, fail_if_exists=True)
    return destination, manifest


def update_manifest(run_dir: str | Path, **fields: object) -> dict:
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(fields)
    write_json(path.with_suffix(".json.next"), manifest, fail_if_exists=True)
    path.with_suffix(".json.next").replace(path)
    return manifest

