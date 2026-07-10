import numpy as np
import pandas as pd

from src.silver.parse_alfa_rosbag import FIXED_TOPICS, PAIR_TOPICS, POSITION_TOPIC, assemble_flight


def _topic_dfs(failure_active: bool = False):
    ts = np.arange(5) * 250_000_000
    dfs = {
        POSITION_TOPIC: pd.DataFrame({"ts_ns": ts, "latitude": 40.0, "longitude": 29.0, "altitude": 100.0}),
        "/mavros/nav_info/roll": pd.DataFrame({"ts_ns": ts, "measured": 1.0, "commanded": 0.5}),
        "/mavros/nav_info/errors": pd.DataFrame({"ts_ns": ts, "alt_error": 0.1, "aspd_error": 0.2,
                                                  "xtrack_error": 0.3, "wp_dist": 50.0}),
        "/mavros/vfr_hud": pd.DataFrame({"ts_ns": ts, "airspeed": 15.0, "groundspeed": 14.0,
                                          "throttle": 60.0, "climb": 0.5}),
        "/mavros/nav_info/velocity": pd.DataFrame({"ts_ns": ts, "meas_x": 3.0, "meas_y": 4.0, "meas_z": 0.0,
                                                    "des_x": 3.0, "des_y": 4.0, "des_z": 0.0}),
        "/failure_status/engines": pd.DataFrame({
            "ts_ns": [ts[3]], "data": [failure_active]}),
    }
    return dfs


def test_assemble_flight_matches_parse_alfa_schema():
    out = assemble_flight(_topic_dfs(), "carbonZ_test")
    for col in ["ts_ns", "lat", "lon", "alt", "roll_measured", "roll_commanded",
                "alt_error", "xtrack_error", "ground_speed_ms", "throttle", "climb_rate_ms",
                "velocity_measured", "velocity_commanded", "label", "source_id", "timestamp_utc"]:
        assert col in out.columns, col
    assert out["velocity_measured"].iloc[0] == 5.0  # sqrt(3^2+4^2)
    assert (out["label"] == "normal").all()  # failure hic aktiflesmedi
    assert (out["source_type"] == "alfa").all()


def test_assemble_flight_labels_after_failure_onset():
    out = assemble_flight(_topic_dfs(failure_active=True), "carbonZ_test")
    # onset ts[3]'ten itibaren engine_fault (engines -> engine normalizasyonu)
    assert (out[out["ts_ns"] >= 750_000_000]["label"] == "engine_fault").all()
    assert (out[out["ts_ns"] < 750_000_000]["label"] == "normal").all()


def test_assemble_flight_without_position_returns_none():
    dfs = _topic_dfs()
    del dfs[POSITION_TOPIC]
    assert assemble_flight(dfs, "x") is None


def test_assemble_flight_tolerates_missing_optional_topics():
    dfs = {POSITION_TOPIC: _topic_dfs()[POSITION_TOPIC]}
    out = assemble_flight(dfs, "carbonZ_minimal")
    assert len(out) == 5 and (out["label"] == "normal").all()
    assert "roll_measured" not in out.columns  # nav_info yoksa kolon da yok (NaN uydurulmaz)
