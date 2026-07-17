import numpy as np
import pandas as pd

from residual_v1.features.build import build_xy
from residual_v1.features.spec import ResidualChannelSpec


def test_horizon_guard_band_phase_exclusion_and_no_response_feature():
    time = np.arange(0.0, 25.1, 0.1)
    flight = pd.DataFrame(
        {
            "t": time,
            "aileron_cmd": np.sin(time),
            "airspeed": np.full(len(time), 20.0),
            "roll_rate": time,
        }
    )
    phases = pd.DataFrame(
        {
            "t": time,
            "phase": np.where(time < 1.0, "ground", "cruise"),
            "phase_boundary": np.isclose(time, 2.0),
        }
    )
    spec = ResidualChannelSpec(
        "R1_test", ("aileron_cmd",), "roll_rate", ("airspeed_context",)
    )
    X, y, meta = build_xy(
        flight,
        spec,
        phases,
        events=[{"fault_class": "engine", "onset_s": 20.0, "end_s": 25.0}],
        flight_id="f1",
    )
    assert not any(column == "roll_rate" or column.startswith("roll_rate__") for column in X)
    assert "airspeed_context__last" in X
    assert not any(
        column.startswith("airspeed_context__tri_")
        or column == "airspeed_context__delta_1s"
        for column in X
    )
    assert "aileron_cmd__delta_1s" in X
    assert not (meta["row_meta"]["phase"] == "ground").any()
    assert 2.0 not in meta["row_meta"]["t"].tolist()
    assert meta["row_meta"].loc[meta["row_meta"]["t"] < 10.0, "train_eligible"].all()
    assert not meta["row_meta"].loc[meta["row_meta"]["t"] >= 10.0, "train_eligible"].any()

    # At t=1.0, future samples 1.1..1.5 average to 1.3.
    row_index = meta["row_meta"].index[np.isclose(meta["row_meta"]["t"], 1.0)][0]
    assert np.isclose(y.loc[row_index], 1.3)


def test_nan_rows_are_dropped_and_reported():
    time = np.arange(0.0, 3.0, 0.1)
    command = np.sin(time)
    command[15] = np.nan
    flight = pd.DataFrame(
        {"t": time, "aileron_cmd": command, "airspeed": 20.0, "roll_rate": time}
    )
    phases = pd.DataFrame({"t": time, "phase": "cruise", "phase_boundary": False})
    spec = ResidualChannelSpec(
        "R1_test", ("aileron_cmd",), "roll_rate", ("airspeed_context",)
    )
    X, _, meta = build_xy(flight, spec, phases)
    assert X.notna().all().all()
    assert meta["nan_drop_count"] > 0
