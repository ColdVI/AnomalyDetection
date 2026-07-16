import json

import pytest

from uav_gnss.pipeline import PilotRunner


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

