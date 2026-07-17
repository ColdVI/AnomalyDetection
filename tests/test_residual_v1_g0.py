import numpy as np
import pandas as pd

from residual_v1.models.g0_rules import command_no_response_score, score_g0


def test_command_present_response_absent_rule_fires():
    time = np.arange(0.0, 4.0, 0.1)
    command = np.zeros(len(time))
    command[time >= 2.0] = 150.0
    frame = pd.DataFrame(
        {
            "t": time,
            "aileron_cmd": command,
            "roll_rate": np.zeros(len(time)),
            "airspeed": 20.0,
        }
    )
    score = command_no_response_score(
        frame,
        command="aileron_cmd",
        response="roll_rate",
        command_threshold=75.0,
        response_stationary_threshold=0.1,
        window_s=1.0,
    )
    assert score.loc[time < 2.0].fillna(0.0).max() == 0.0
    assert score.loc[(time >= 2.0) & (time <= 3.0)].max() >= 2.0

    long = score_g0(frame, flight_id="f1")
    assert list(long.columns) == ["flight_id", "t", "channel", "z"]
    rule = long[long["channel"] == "G0_command_no_response"]
    assert rule["z"].max() >= 2.0
