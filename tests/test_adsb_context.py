import numpy as np
import pandas as pd
import pytest

from adsb.context import CausalContextConfig, build_causal_context


def _config() -> CausalContextConfig:
    return CausalContextConfig(
        phase_history_rows=2,
        level_rate_threshold_mps=1.0,
        cadence_edges_s=(1.0, 5.0),
        max_gap_s=60.0,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": ["A"] * 5,
            "timestamp_utc": [0.0, 1.0, 3.0, 6.0, 70.0],
            "on_ground": [False] * 5,
            "vertical_rate_ms": [0.0, 0.0, 20.0, 0.0, 0.0],
            "track_deg": [359.0, 1.0, 90.0, np.nan, 180.0],
        }
    )


def test_context_is_lagged_causal_and_track_is_circular():
    context = build_causal_context(_frame(), _config())

    assert context.loc[0, "context_phase"] == "unknown"
    assert context.loc[2, "context_phase"] == "level"  # current 20 m/s cannot route itself
    assert context.loc[3, "context_phase"] == "climb"
    assert context.loc[1, "context_cadence"] == "cadence_0"
    assert context.loc[2, "context_cadence"] == "cadence_1"
    assert context.loc[4, "context_cadence"] == "gap"
    assert context.loc[0, "track_cos"] == pytest.approx(context.loc[1, "track_cos"], abs=1e-3)
    assert context.loc[0, "track_sin"] == pytest.approx(-context.loc[1, "track_sin"], abs=1e-3)


def test_future_changes_do_not_change_context_prefix():
    original = _frame()
    changed = original.copy()
    changed.loc[4, ["vertical_rate_ms", "track_deg"]] = [999.0, 270.0]
    left = build_causal_context(original, _config()).iloc[:4]
    right = build_causal_context(changed, _config()).iloc[:4]
    pd.testing.assert_frame_equal(left, right)


def test_context_rejects_unsorted_flight_rows():
    frame = _frame()
    frame.loc[2, "timestamp_utc"] = 0.5
    with pytest.raises(ValueError, match="sorted"):
        build_causal_context(frame, _config())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_history_rows": 0},
        {"level_rate_threshold_mps": 0.0},
        {"cadence_edges_s": (5.0, 1.0)},
        {"max_gap_s": 5.0},
    ],
)
def test_context_config_fails_closed(kwargs):
    values = {
        "phase_history_rows": 2,
        "level_rate_threshold_mps": 1.0,
        "cadence_edges_s": (1.0, 5.0),
        "max_gap_s": 60.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        CausalContextConfig(**values)
