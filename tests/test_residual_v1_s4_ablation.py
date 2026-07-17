import numpy as np
import pandas as pd

from residual_v1.eval.s4_ablation import command_ablation_report
from residual_v1.features.spec import ResidualChannelSpec


def _matrix(*, context_drives_response: bool) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = 400
    command = rng.normal(size=rows)
    context = rng.normal(size=rows)
    noise = rng.normal(scale=0.05, size=rows)
    response = (context if context_drives_response else command) + noise
    return pd.DataFrame(
        {
            "flight_id": np.where(np.arange(rows) < rows / 2, "f1", "f2"),
            "t": np.arange(rows, dtype=float),
            "phase": "cruise",
            "train_eligible": True,
            "command__last": command,
            "command__delta_1s": command,
            "context__last": context,
            "phase_cruise": 1.0,
            "response": response,
        }
    )


FEATURES = ["command__last", "command__delta_1s", "context__last", "phase_cruise"]
SPEC = ResidualChannelSpec("channel", ("command",), "response", ("context",))


def test_s4_passes_when_command_removal_destroys_fit():
    report = command_ablation_report(
        _matrix(context_drives_response=False),
        spec=SPEC,
        selected_alpha=0.1,
        full_feature_columns=FEATURES,
    )
    assert report["status"] == "passed"
    assert report["variance_ratio"] > 1.15
    assert report["removed_command_features"] == ["command__delta_1s", "command__last"]


def test_s4_flags_when_context_alone_matches_full_model():
    report = command_ablation_report(
        _matrix(context_drives_response=True),
        spec=SPEC,
        selected_alpha=0.1,
        full_feature_columns=FEATURES,
    )
    assert report["status"] == "flagged"
    assert report["variance_ratio"] < 1.15


def test_s4_fails_closed_when_declared_command_columns_are_absent():
    matrix = _matrix(context_drives_response=False).drop(columns=["command__last", "command__delta_1s"])
    try:
        command_ablation_report(
            matrix,
            spec=SPEC,
            selected_alpha=0.1,
            full_feature_columns=["context__last", "phase_cruise"],
        )
    except ValueError as error:
        assert "no feature columns found for command" in str(error)
    else:
        raise AssertionError("S-4 accepted a missing declared command")
