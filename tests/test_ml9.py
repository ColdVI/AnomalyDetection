import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.decision import decision_layers
from src.ml.evaluation.events import load_uav_sead_ranges_by_category
from src.ml.features.uav_attack_features import actuator_output_imbalance


def test_category_ranges_preserve_category_label():
    payload = json.dumps({
        "flight": {
            "ranges": [
                [["Position.Z", [[10, 20], [30, 40]]]],
                [["Actuator Outputs", [[50, 60]]]],
            ]
        }
    })
    with patch("src.ml.evaluation.events.Path.read_text", return_value=payload):
        result = load_uav_sead_ranges_by_category("labels.json")
    assert result == {
        "flight": {
            "Position.Z": [(10.0, 20.0), (30.0, 40.0)],
            "Actuator Outputs": [(50.0, 60.0)],
        }
    }


def test_actuator_output_active_channel_detection():
    frame = pd.DataFrame({
        "actuator_output_0": [1000.0, 1010.0, 1020.0],
        "actuator_output_1": [1000.0, 990.0, 980.0],
        "actuator_output_2": [0.0, 0.0, 0.0],
    })
    result = actuator_output_imbalance(frame)
    assert result["actuator_active_channels"].iloc[-1] == 2


def test_actuator_output_imbalance_flags_stuck_motor():
    moving = np.r_[np.linspace(1000, 1100, 10), np.linspace(1100, 1200, 10)]
    frame = pd.DataFrame({f"actuator_output_{i}": moving.copy() for i in range(4)})
    frame.loc[10:, "actuator_output_0"] = 1100.0
    result = actuator_output_imbalance(frame)
    assert result["actuator_output_range"].iloc[-1] == 100.0


def test_ml9_decision_layers_reused_not_reimplemented():
    from scripts import run_ml9_category_evaluation as runner

    assert runner.fit_threshold_policy is decision_layers.fit_threshold_policy
    assert runner.fit_k_of_n_policy is decision_layers.fit_k_of_n_policy
    assert runner.fit_cusum_policy is decision_layers.fit_cusum_policy
