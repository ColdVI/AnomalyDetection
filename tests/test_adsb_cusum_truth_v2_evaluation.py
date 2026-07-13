from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adsb.cusum_truth_v2_eval import (
    BoundedScoreReservoir,
    CusumTruthV2ContractError,
    iter_exact_paired_flights,
    load_frozen_step5_bundle,
    run_evaluation,
    sampled_binary_diagnostics,
)
from adsb.run_manifest import sha256_file, sha256_json
from adsb.synthetic import inject_position_ramp
from adsb.truth import attach_clean_truth_v2


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP5 = REPO_ROOT / "artifacts/adsb/runs/20260713_step5_full_streaming_v1"


def _clean_flight(flight_id: str, offset: float) -> pd.DataFrame:
    n = 24
    timestamp = offset + np.arange(n, dtype=float)
    # Approximately northbound motion with internally consistent speed/track.
    lat = 40.0 + np.arange(n, dtype=float) * (50.0 / 111_320.0)
    return pd.DataFrame(
        {
            "flight_id": flight_id,
            "timestamp_utc": timestamp,
            "lat": lat,
            "lon": np.full(n, 29.0),
            "alt": np.full(n, 1000.0),
            "alt_geom_m": np.full(n, 1010.0),
            "ground_speed_ms": np.full(n, 50.0),
            "track_deg": np.zeros(n),
            "vertical_rate_ms": np.zeros(n),
            "roll_deg": np.zeros(n),
            "on_ground": np.zeros(n, dtype=bool),
        }
    )


def _write_corpus(root: Path, *, corrupt_timestamp: bool = False) -> Path:
    corpus = root / "synthetic_fixture" / "adsb_v2"
    corpus.mkdir(parents=True)
    clean_raw = pd.concat(
        [_clean_flight("f1", 0.0), _clean_flight("f2", 100.0)], ignore_index=True
    )
    clean = pd.concat(
        [attach_clean_truth_v2(group.reset_index(drop=True)) for _, group in clean_raw.groupby("flight_id", sort=False)],
        ignore_index=True,
    )
    corrupt = pd.concat(
        [
            inject_position_ramp(
                group.reset_index(drop=True),
                meters_per_s=2.0,
                event_type="position_ramp_stealthy",
            )
            for _, group in clean_raw.groupby("flight_id", sort=False)
        ],
        ignore_index=True,
    )
    if corrupt_timestamp:
        corrupt.loc[3, "timestamp_utc"] += 0.25
    clean_path = corpus / "clean.parquet"
    corrupt_path = corpus / "position_ramp_stealthy.parquet"
    clean.to_parquet(clean_path, index=False)
    corrupt.to_parquet(corrupt_path, index=False)

    outputs = []
    for recipe, path in (
        ("clean", clean_path),
        ("position_ramp_stealthy", corrupt_path),
    ):
        outputs.append(
            {
                "recipe": recipe,
                "path": path.name,
                "n_rows": len(clean),
                "footer_rows": len(clean),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "adsb_synthetic_truth_v2",
                "synthetic_never_training": True,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return corpus


def _write_step5_fixture(root: Path, flight_ids: list[str]) -> Path:
    directory = root / "step5_fixture"
    directory.mkdir(parents=True)
    actual_derived = json.loads((STEP5 / "derived_frozen_config.json").read_text())
    selected = actual_derived["derived_config"]["cusum"]["selected_detector"]
    base_config = {
        "cusum": {"synthetic_selection_forbidden": True},
        "frozen_code_sha256": {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in ("adsb/cusum.py", "adsb/features.py")
        },
    }
    base_hash = sha256_json(base_config)
    validation_ids = sorted(f"2026-02-28:{value}" for value in flight_ids)
    normalized = {"validation": validation_ids}
    split_payload = {"algorithm": "fixture", "seed": None, "splits": normalized}
    manifest = {
        "manifest_schema_version": 1,
        "config": {"sha256": base_hash, "value": base_config},
        "inputs": [],
        "input_contract_sha256": sha256_json([]),
        "split_contract": {
            "algorithm": "fixture",
            "seed": None,
            "contract_sha256": sha256_json(split_payload),
            "splits": {
                "validation": {
                    "flight_id_count": len(validation_ids),
                    "flight_ids_sha256": sha256_json(validation_ids),
                    "flight_ids": validation_ids,
                }
            },
        },
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    derived = {
        "base_config_sha256": base_hash,
        "cusum": {
            "axis_coverage": {"status": "complete", "active_axis_count": 2},
            "selected_h": 1.0,
            "selected_detector": selected,
            "gate_eligible": True,
        },
        "later_role_feedback_forbidden": True,
        "frozen_before_roles": ["development", "rehearsal"],
    }
    derived_record = {
        "schema_version": 1,
        "derived_config": derived,
        "payload_sha256": sha256_json(derived),
    }
    derived_path = directory / "derived_frozen_config.json"
    derived_path.write_text(json.dumps(derived_record), encoding="utf-8")
    derived_file_hash = sha256_file(derived_path)
    (directory / "derived_frozen_config.sha256").write_text(
        derived_file_hash + "\n", encoding="utf-8"
    )

    roles = {}
    for role, day, n_flights in (
        ("validation", "2026-02-28", len(validation_ids)),
        ("development", "2026-03-01", 3),
        ("rehearsal", "2026-03-16", 3),
    ):
        roles[role] = {
            "role": role,
            "day": day,
            "cusum_event_unit": {
                "selected_h": 1.0,
                "emission_time": "row_timestamp_utc",
                "merge_gap_s": 60.0,
                "n_alert_episodes": 1,
                "episodes_per_scoreable_flight_hour": 1.0,
            },
            "cusum_flight_unit": {
                "n_input_flights": n_flights,
                "n_scoreable_flights": n_flights,
                "n_alerted_flights": 1,
                "scoreable_flight_hours": 1.0,
            },
            "cusum_cadence_strata": {},
        }
    natural = {
        "settings_frozen_before_development_and_rehearsal": True,
        "rehearsal_feedback_into_settings": False,
        "roles": roles,
    }
    (directory / "natural_burden_by_role.json").write_text(
        json.dumps(natural), encoding="utf-8"
    )
    report = {
        "manifest": "run_manifest.json",
        "artifact_checksum_index": "artifact_checksums.json",
        "config_sha256": base_hash,
        "config_and_code_unchanged_through_evaluation": True,
        "rehearsal_changed_parameters": False,
        "synthetic_training_rows": 0,
        "gate_status": "evidence_only_pending_step_7_review",
        "derived_frozen_config": {
            "file_sha256": derived_file_hash,
            "payload_sha256": derived_record["payload_sha256"],
        },
        "natural_burden": natural,
    }
    (directory / "streaming_baseline_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    indexed = {}
    for name in (
        "run_manifest.json",
        "derived_frozen_config.json",
        "derived_frozen_config.sha256",
        "streaming_baseline_report.json",
        "natural_burden_by_role.json",
    ):
        path = directory / name
        indexed[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (directory / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "self_excluded": True,
                "files": indexed,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_real_step5_bundle_loads_exact_frozen_h1_detector_and_burden_panel():
    bundle = load_frozen_step5_bundle(STEP5, repo_root=REPO_ROOT)
    assert bundle.selected_h == 1.0
    assert bundle.detector.config.threshold_h == 1.0
    assert bundle.selected_detector["mad_zero_policy"] == "exclude"
    assert bundle.selected_detector["excluded_channels"] == {}
    assert len(bundle.validation_flight_ids) == 8910
    assert bundle.burden_panel["primary_role"] == "validation"
    assert set(bundle.burden_panel["temporal_stability_panel"]) == {
        "development",
        "rehearsal",
    }


def test_step5_chain_fails_closed_when_an_indexed_artifact_is_tampered(tmp_path: Path):
    copied = _write_step5_fixture(tmp_path, ["f1", "f2"])
    with (copied / "natural_burden_by_role.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(CusumTruthV2ContractError, match="byte count changed"):
        load_frozen_step5_bundle(copied, repo_root=REPO_ROOT)


def test_bounded_diagnostic_sample_has_hard_cap_and_population_weighting():
    negative = BoundedScoreReservoir(25, 7, "negative")
    positive = BoundedScoreReservoir(25, 7, "positive")
    negative.add(np.arange(100, dtype=float), source_key="clean")
    positive.add(np.arange(100, 200, dtype=float), source_key="corrupt")
    result = sampled_binary_diagnostics(negative, positive)
    assert len(negative.scores) == 25
    assert len(positive.scores) == 25
    assert result["population_n_negative"] == 100
    assert result["population_n_positive"] == 100
    assert result["auroc_estimate"] == 1.0
    assert result["auprc_estimate_population_weighted"] == 1.0
    assert result["never_used_for_threshold_or_selection"] is True


def test_exact_pairing_rejects_timestamp_drift(tmp_path: Path):
    corpus = _write_corpus(tmp_path, corrupt_timestamp=True)
    with pytest.raises(CusumTruthV2ContractError, match="timestamps differ"):
        list(
            iter_exact_paired_flights(
                corpus / "clean.parquet",
                corpus / "position_ramp_stealthy.parquet",
            )
        )


def test_small_frozen_cusum_truth_v2_run_is_immutable_and_pairs_burden(tmp_path: Path):
    corpus = _write_corpus(tmp_path)
    step5 = _write_step5_fixture(tmp_path, ["f1", "f2"])
    run_dir = tmp_path / "run"
    summary_path = run_evaluation(
        repo_root=REPO_ROOT,
        step5_dir=step5,
        corpus_dir=corpus,
        run_dir=run_dir,
        recipes=("position_ramp_stealthy",),
        sample_capacity_per_class=100,
        sample_seed=11,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recipe = summary["per_recipe"]["position_ramp_stealthy"]
    assert summary["status"] == "complete_evaluation_only"
    assert summary["frozen_detector"]["selected_h"] == 1.0
    assert summary["frozen_detector"]["fit_performed_in_this_run"] is False
    assert recipe["synthetic_detection"]["corrupt_q0_timeline_sanity"][
        "included_as_auc_negative"
    ] is False
    assert recipe["diagnostic_sampled_auc_auprc"][
        "negative_source"
    ].startswith("single_unmodified_clean_reference")
    assert recipe["paired_natural_cusum_burden"]["primary_role"] == "validation"
    assert set(recipe["paired_natural_cusum_burden"]["temporal_stability_panel"]) == {
        "development",
        "rehearsal",
    }
    assert summary["stealthy_ramp_focus"]["available"] is True
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "event_table.parquet").is_file()
    checksum = json.loads((run_dir / "artifact_checksums.json").read_text())
    assert checksum["status"] == "complete"
    assert checksum["files"]["event_table.parquet"]["sha256"] == sha256_file(
        run_dir / "event_table.parquet"
    )

    with pytest.raises(FileExistsError):
        run_evaluation(
            repo_root=REPO_ROOT,
            step5_dir=step5,
            corpus_dir=corpus,
            run_dir=run_dir,
            recipes=("position_ramp_stealthy",),
            sample_capacity_per_class=100,
            sample_seed=11,
        )
