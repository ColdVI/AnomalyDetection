"""UAV GNSS pipeline testleri (torch gerektirir)

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from uav_gnss.frozen_runner import require_prior_role

import json

import pytest

from uav_gnss.pipeline import PilotRunner



# ===== kaynak: test_uav_gnss_frozen_runner =====

def _report(current_pass: bool):
    methods = {
        method: {
            "passes_gate": current_pass,
            "events": {"recall": 1.0 if method == "lstm" else 0.0},
        }
        for method in ("px4_native", "cusum", "lstm")
    }
    return {
        "contracts": {
            "critical": {"methods": {key: dict(value) for key, value in methods.items()}},
            "advisory": {"methods": {key: dict(value) for key, value in methods.items()}},
        }
    }


def test_rehearsal_cannot_resurrect_failed_development():
    result = require_prior_role(_report(True), _report(False))
    assert result["selected_critical_method"] is None
    assert result["preliminary_status"] == "no_go_on_current_role"
    assert not any(
        method["passes_gate"]
        for method in result["contracts"]["critical"]["methods"].values()
    )



# ===== kaynak: test_uav_gnss_holdout =====

def test_holdout_requires_exact_config_bound_unseal(tmp_path):
    config = {
        "data": {"bronze_root": str(tmp_path / "data")},
        "output_dir": str(tmp_path / "out"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runner = PilotRunner(config_path)
    with pytest.raises(PermissionError, match="sealed"):
        runner._assert_unsealed()
    unseal = runner.out / "HOLDOUT_UNSEAL.json"
    unseal.write_text(
        json.dumps(
            {
                "candidate_namespace": "uav_gnss_integrity_v1",
                "config_sha256": runner.config_sha256,
                "approval": "UNSEAL HOLDOUT",
            }
        ),
        encoding="utf-8",
    )
    runner._assert_unsealed()

