from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.adsb_audit_cusum_bootstrap_upper as audit
from adsb.run_manifest import sha256_file, sha256_json


REPO_ROOT = Path(__file__).parent.parent


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selection(*, observed_h1: float = 0.6) -> dict:
    return {
        "selected_h": 1.0,
        "selection_rule": "legacy raw bootstrap p95",
        "budget_episodes_per_hour": 0.5,
        "n_blocks": 2,
        "observed_burden_source": "full_flight_counters",
        "observed_exposure_hours": 10.0,
        "bootstrap_repetitions": 500,
        "bootstrap_batch_size": 4,
        "bootstrap_upper_quantile": 0.95,
        "candidates": [
            {
                "h": 1.0,
                "observed_episode_count": 6,
                "observed_episodes_per_hour": observed_h1,
                "bootstrap_upper_95_episodes_per_hour": 0.4,
                "meets_advisory_budget": True,
            },
            {
                "h": 2.0,
                "observed_episode_count": 1,
                "observed_episodes_per_hour": 0.1,
                "bootstrap_upper_95_episodes_per_hour": 0.2,
                "meets_advisory_budget": True,
            },
        ],
    }


def _source_run(path: Path, *, observed_h1: float = 0.4) -> Path:
    path.mkdir()
    selection = _selection(observed_h1=observed_h1)
    contract = {
        "candidate_h": [1.0, 2.0],
        "advisory_budget_episodes_per_hour": 0.5,
        "bootstrap_repetitions": 500,
        "bootstrap_batch_size": 4,
        "bootstrap_seed": 20260713,
        "upper_quantile": 0.95,
    }
    _write_json(
        path / "run_manifest.json",
        {
            "run_id": path.name,
            "config": {"value": {"cusum": {"burden_calibration": contract}}},
        },
    )
    _write_json(
        path / "normal_burden_calibration.json",
        {"cusum_natural_burden_selection": selection},
    )
    normal_sha256 = sha256_file(path / "normal_burden_calibration.json")
    derived_config = {
        "normal_burden_calibration_file_sha256": normal_sha256,
        "cusum": {"selection": selection},
    }
    _write_json(
        path / "derived_frozen_config.json",
        {
            "schema_version": 1,
            "derived_config": derived_config,
            "payload_sha256": sha256_json(derived_config),
        },
    )
    derived_sha256 = sha256_file(path / "derived_frozen_config.json")
    (path / "derived_frozen_config.sha256").write_text(
        derived_sha256 + "\n", encoding="ascii"
    )
    _write_json(
        path / "streaming_baseline_report.json",
        {
            "run_id": path.name,
            "derived_frozen_config": {
                "file_sha256": derived_sha256,
                "payload_sha256": sha256_json(derived_config),
                "sidecar": "derived_frozen_config.sha256",
            },
            "normal_threshold_calibration": {
                "cusum_natural_burden_selection": selection
            },
        },
    )
    indexed = {}
    for name in audit.INDEXED_SOURCE_FILENAMES:
        indexed[name] = {
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
    _write_json(
        path / "artifact_checksums.json",
        {
            "schema_version": 1,
            "algorithm": "sha256",
            "self_excluded": True,
            "files": indexed,
        },
    )
    return path


def test_corrected_selection_uses_max_and_can_change_selected_h():
    result = audit.corrected_selection(_selection(observed_h1=0.6))

    assert result["old_selected_h"] == 1.0
    assert result["new_selected_h"] == 2.0
    assert result["selected_h_unchanged"] is False
    first = result["candidates"][0]
    assert first["old_raw_bootstrap_quantile_95_episodes_per_hour"] == 0.4
    assert first["conservative_upper_95_episodes_per_hour"] == 0.6
    assert first["old_meets_advisory_budget"] is True
    assert first["new_meets_advisory_budget"] is False


def test_audit_run_manifests_chain_is_fail_if_exists_and_preserves_selection(tmp_path: Path):
    source = _source_run(tmp_path / "source-v1", observed_h1=0.4)
    destination = tmp_path / "audit-v1"

    report = audit.run(
        repo_root=REPO_ROOT,
        source_run_dir=source,
        run_dir=destination,
    )

    assert report["status"] == "passed_selected_h_unchanged"
    assert report["scope"]["scientific_rerun"] is False
    assert report["source"]["artifacts_preserved"] is True
    assert report["selection_comparison"]["old_selected_h"] == 1.0
    assert report["selection_comparison"]["new_selected_h"] == 1.0
    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert {entry["path"].rsplit("/", 1)[-1] for entry in manifest["inputs"]} == set(
        audit.SOURCE_FILENAMES
    )
    assert (destination / "artifact_checksums.json").is_file()

    with pytest.raises(FileExistsError):
        audit.run(
            repo_root=REPO_ROOT,
            source_run_dir=source,
            run_dir=destination,
        )


def test_source_chain_tamper_is_rejected_before_audit_directory_creation(tmp_path: Path):
    source = _source_run(tmp_path / "source-v1")
    (source / "normal_burden_calibration.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "audit-v1"

    with pytest.raises(audit.AuditContractError, match="checksum mismatch"):
        audit.run(
            repo_root=REPO_ROOT,
            source_run_dir=source,
            run_dir=destination,
        )
    assert not destination.exists()
