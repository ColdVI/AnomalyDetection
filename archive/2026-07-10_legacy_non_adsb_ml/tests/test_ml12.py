"""ML-12 ince-modul fazi disiplin testleri (docs/ML12_INCE_MODUL_PLAN.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_ml10_forecast_evaluation as ml10_runner
from scripts import run_ml12_thin_module_evaluation as ml12_runner
from scripts import run_ml9_category_evaluation as ml9_runner
from src.ml.decision import decision_layers
from src.ml.evaluation import score_fusion
from src.ml.models.modular_iforest import (
    PX4_ML7_CANDIDATE_MODULES,
    PX4_ML9_CANDIDATE_MODULES,
    PX4_ML12_THIN_MODULES,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
RUN_DIR = ROOT / "artifacts/ml12/uav_sead/full_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_thin_module_definitions_match_preregistered_plan():
    # Listeler docs/ML12_INCE_MODUL_PLAN.md §1'e sabitlendi; sonuc gorulup
    # degistirilmedigini bu test garanti eder.
    assert PX4_ML12_THIN_MODULES == {
        "itki_komutu": ["actuator_thrust_cmd"],
        "itki_kontrol_ince": [
            "actuator_thrust_cmd", "attitude_error_mag", "control_strain",
        ],
    }
    # Ince moduller mevcut default/candidate ailelerine sizmamali.
    assert not set(PX4_ML12_THIN_MODULES) & set(PX4_ML9_CANDIDATE_MODULES)
    assert not set(PX4_ML12_THIN_MODULES) & set(PX4_ML7_CANDIDATE_MODULES)


def test_ml12_decision_layers_and_fusion_reused_not_reimplemented():
    assert ml12_runner._fit_policies is ml10_runner._fit_policies
    assert ml12_runner._evaluate is ml9_runner._evaluate
    assert ml12_runner._score_modules is ml9_runner._score_modules
    assert ml12_runner.max_score_fusion is score_fusion.max_score_fusion
    assert ml12_runner.last_causal_per_bucket is score_fusion.last_causal_per_bucket
    # Karar katmani fonksiyonlari ml10 uzerinden degismeden geliyor.
    assert ml10_runner.fit_cusum_policy is decision_layers.fit_cusum_policy
    assert ml10_runner.fit_threshold_policy is decision_layers.fit_threshold_policy
    assert ml10_runner.fit_k_of_n_policy is decision_layers.fit_k_of_n_policy


def test_ml12_artifact_holdout_isolation_and_checksums():
    manifest_path = RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("ML-12 full_matrix kosusu henuz yapilmadi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["blind_holdout_read"] is False
    assert manifest["blind_holdout_flights"] == 131
    assert manifest["development_flights"] == 480

    for relative, expected in manifest["files"].items():
        assert _sha256(RUN_DIR / relative) == expected, relative

    if manifest.get("split_manifest_sha256") != _sha256(SPLIT_PATH):
        pytest.skip("eski veri donemi artifact'i")

    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = split_manifest["sources"]["uav_sead"]
    holdout = set(config["splits"]["split_00"]["final_holdout"])
    expected_dev = sorted(set(config["flight_labels"]) - holdout)
    expected_hash = hashlib.sha256(
        "\n".join(expected_dev).encode("utf-8")).hexdigest()
    assert manifest["development_source_ids_sha256"] == expected_hash


def test_ml12_gate_b_status_derived_only_from_b1_arm():
    gates_path = RUN_DIR / "gates.json"
    if not gates_path.exists():
        pytest.skip("ML-12 full_matrix kosusu henuz yapilmadi")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    rows = gates["gate_b"]["comparisons"]
    assert rows, "Gate B karsilastirmalari bos"
    b1 = [row for row in rows if row["arm"] == "b1_gate"]
    b2 = [row for row in rows if row["arm"] == "b2_informative"]
    assert all(row["baseline"] == "motor_simetrisi" for row in b1)
    assert all(row["baseline"] == "chronos_motor" for row in b2)
    expected = "passed" if any(row["meaningful_gain"] for row in b1) else "failed"
    if gates["gate_b"]["evaluated_seed_count"] >= 5:
        assert gates["gate_b"]["status"] == expected


def test_ml12_reused_baseline_rows_match_frozen_ml9_csv():
    metrics_path = RUN_DIR / "category_metrics.csv"
    if not metrics_path.exists():
        pytest.skip("ML-12 full_matrix kosusu henuz yapilmadi")
    ml12 = pd.read_csv(metrics_path)
    ml9 = pd.read_csv(ml12_runner.ML9_DIR / "category_metrics.csv")
    key = ["split", "seed", "score_source", "decision", "budget",
           "annotation_category"]
    for source in ("motor_simetrisi", "existing_fusion"):
        left = (ml12[ml12["score_source"] == source]
                .set_index(key)["event_onset_recall"].sort_index())
        right = (ml9[ml9["score_source"] == source]
                 .set_index(key)["event_onset_recall"].sort_index())
        pd.testing.assert_series_equal(left, right, check_names=False)
