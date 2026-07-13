"""Tests for immutable ADS-B run provenance manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from adsb.run_manifest import (
    InputSpec,
    ManifestError,
    build_split_contract,
    create_immutable_run_manifest,
    inspect_input_file,
    make_deterministic_split,
    sha256_json,
    silver_row_count_provenance,
)


REPO_ROOT = Path(__file__).parent.parent


def _parquet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "flight_id": ["f1", "f1", "f2"],
            "timestamp_utc": [1.0, 2.0, 3.0],
            "alt": [1000.0, None, 1200.0],
        }
    ).to_parquet(path, index=False)
    return path


def _create(run_dir: Path, input_path: Path) -> Path:
    return create_immutable_run_manifest(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        inputs=[InputSpec(input_path, "fit")],
        splits={"fit": ["f2", "f1"], "development": ["f3"]},
        split_algorithm="test_explicit_v1",
        split_seed=7,
        synthetic_flight_ids=["s2", "s1"],
        config={"window": 12, "features": ["a", "b"]},
    )


def test_inspect_parquet_records_bytes_hash_footer_rows_and_schema(tmp_path: Path):
    input_path = _parquet(tmp_path / "input.parquet")
    record = inspect_input_file(input_path, role="fit", repo_root=REPO_ROOT)

    assert record["bytes"] == input_path.stat().st_size
    assert record["sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert record["footer_rows"] == 3
    assert record["format"] == "parquet"
    assert len(record["schema_sha256"]) == 64
    assert record["schema_hash_contract"].endswith("v1")


def test_deterministic_split_is_input_and_weight_order_invariant():
    ids = [f"f{i}" for i in range(11)]
    first = make_deterministic_split(
        ids,
        {"fit": 0.8, "development": 0.2},
        seed=19,
        excluded_flight_ids=["f10"],
    )
    second = make_deterministic_split(
        reversed(ids),
        {"development": 2, "fit": 8},
        seed=19,
        excluded_flight_ids=["f10"],
    )

    assert first == second
    assert len(first["fit"]) == 8
    assert len(first["development"]) == 2
    assert "f10" not in first["fit"] + first["development"]


def test_split_contract_stores_sorted_ids_and_order_invariant_hash():
    first = build_split_contract(
        {"fit": ["f2", "f1"], "development": ["d1"]},
        algorithm="explicit_v1",
        seed=0,
    )
    second = build_split_contract(
        {"development": ["d1"], "fit": ["f1", "f2"]},
        algorithm="explicit_v1",
        seed=0,
    )

    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["splits"]["fit"]["flight_ids"] == ["f1", "f2"]
    assert first["splits"]["fit"]["flight_ids_sha256"] == sha256_json(["f1", "f2"])


def test_split_contract_rejects_flight_id_overlap():
    with pytest.raises(ManifestError, match="occurs in both"):
        build_split_contract(
            {"fit": ["same"], "development": ["same"]},
            algorithm="explicit_v1",
            seed=0,
        )


def test_manifest_is_complete_and_run_directory_is_immutable(tmp_path: Path):
    input_path = _parquet(tmp_path / "silver" / "input.parquet")
    run_dir = tmp_path / "runs" / "run-001"
    manifest_path = _create(run_dir, input_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == run_dir / "run_manifest.json"
    assert manifest["run_id"] == "run-001"
    assert manifest["immutability"]["fail_if_exists"] is True
    assert len(manifest["git"]["commit"]) == 40
    assert isinstance(manifest["git"]["dirty"], bool)
    assert manifest["config"]["sha256"] == sha256_json(manifest["config"]["value"])
    assert manifest["split_contract"]["splits"]["fit"]["flight_ids"] == ["f1", "f2"]
    assert manifest["synthetic_guard"]["status"] == "passed"
    assert manifest["synthetic_guard"]["excluded_flight_ids"] == ["s1", "s2"]
    assert manifest["synthetic_guard"]["overlap_count"] == 0

    with pytest.raises(FileExistsError, match="immutable"):
        _create(run_dir, input_path)


def test_manifest_rejects_synthetic_path_before_opening_it(tmp_path: Path):
    # The file intentionally does not exist: the path-role guard must fail first,
    # proving that a protected synthetic path is not opened or hashed.
    synthetic_path = tmp_path / "synthetic" / "missing.parquet"
    with pytest.raises(ManifestError, match="Synthetic path is forbidden"):
        create_immutable_run_manifest(
            run_dir=tmp_path / "run",
            repo_root=REPO_ROOT,
            inputs=[InputSpec(synthetic_path, "fit")],
            splits={"fit": ["f1"]},
            split_algorithm="explicit_v1",
            split_seed=0,
            synthetic_flight_ids=[],
            config={},
        )
    assert not (tmp_path / "run").exists()


def test_manifest_rejects_synthetic_source_id_in_fit(tmp_path: Path):
    input_path = _parquet(tmp_path / "silver" / "input.parquet")
    with pytest.raises(ManifestError, match="overlap protected splits"):
        create_immutable_run_manifest(
            run_dir=tmp_path / "run",
            repo_root=REPO_ROOT,
            inputs=[InputSpec(input_path, "fit")],
            splits={"fit": ["f1"]},
            split_algorithm="explicit_v1",
            split_seed=0,
            synthetic_flight_ids=["f1"],
            config={},
        )
    assert not (tmp_path / "run").exists()


def test_silver_row_count_discrepancy_is_explicit_and_unresolved():
    provenance = silver_row_count_provenance()
    assert provenance["footer_observed_rows"] == 256_155_009
    assert provenance["documented_rows"] == 256_150_550
    assert provenance["delta_rows"] == 4_459
    assert provenance["status"] == "unresolved_do_not_silently_correct"


def test_cli_creates_manifest_without_input_discovery(tmp_path: Path):
    input_path = _parquet(tmp_path / "silver" / "input.parquet")
    config_path = tmp_path / "config.json"
    split_path = tmp_path / "splits.json"
    exclusions_path = tmp_path / "excluded.json"
    config_path.write_text('{"window": 12}', encoding="utf-8")
    split_path.write_text(
        json.dumps(
            {
                "algorithm": "explicit_cli_v1",
                "seed": 3,
                "splits": {"fit": ["f1"], "development": ["f2"]},
            }
        ),
        encoding="utf-8",
    )
    exclusions_path.write_text('["s1"]', encoding="utf-8")
    run_dir = tmp_path / "runs" / "cli-run"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "adsb_create_run_manifest.py"),
            "--run-dir",
            str(run_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--input",
            f"fit={input_path}",
            "--config",
            str(config_path),
            "--splits",
            str(split_path),
            "--synthetic-flight-ids",
            str(exclusions_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = Path(completed.stdout.strip())
    assert manifest_path == run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_contract"]["algorithm"] == "explicit_cli_v1"
    assert manifest["inputs"][0]["footer_rows"] == 3

