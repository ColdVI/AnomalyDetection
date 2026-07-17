import numpy as np
import pandas as pd

from residual_v1.features.phases import label_phases


CONFIG = {
    "ground_speed_max_mps": 3.0,
    "ground_climb_abs_max_mps": 0.3,
    "maneuver_roll_abs_min_deg": 25.0,
    "maneuver_roll_rate_abs_min_deg_s": 15.0,
    "takeoff_climb_min_mps": 0.8,
    "landing_climb_max_mps": -0.5,
    "transition_altitude_agl_max_m": 60.0,
    "boundary_buffer_s": 1.0,
}


def test_phase_sequence_and_time_based_boundary_mask():
    frame = pd.DataFrame(
        {
            "t": np.arange(0.0, 8.0),
            "ground_speed": [0, 0, 8, 12, 15, 15, 8, 0],
            "climb_rate": [0, 0, 2, 1, 0, 0, -2, 0],
            "altitude": [100, 100, 102, 110, 130, 140, 120, 100],
            "roll": np.deg2rad([0, 0, 0, 0, 35, 0, 0, 0]),
            "roll_rate": np.zeros(8),
        }
    )
    result = label_phases(frame, config=CONFIG)
    assert result["phase"].tolist() == [
        "ground",
        "ground",
        "takeoff",
        "takeoff",
        "maneuver",
        "cruise",
        "landing",
        "ground",
    ]
    # Every transition timestamp and its immediate +/- 1 s neighbours are masked.
    assert result.loc[[1, 2, 3], "phase_boundary"].all()
    assert result.loc[[6, 7], "phase_boundary"].all()
