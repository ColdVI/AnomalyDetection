import numpy as np
import pandas as pd

from adsb.context import CausalContextConfig
from adsb.contextual_windowing import build_contextual_forecast_windows


def test_windows_are_next_row_causal_and_never_cross_flights():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 4 + ["B"] * 4,
            "timestamp_utc": [0.0, 1.0, 2.0, 3.0, 10.0, 12.0, 14.0, 16.0],
            "on_ground": [False] * 8,
            "vertical_rate_ms": [0.0] * 8,
            "track_deg": [359.0, 1.0, 2.0, 3.0, 90.0, 90.0, 90.0, 90.0],
            "signal": np.arange(8, dtype=float),
            "target": np.arange(10, 18, dtype=float),
        }
    )
    config = CausalContextConfig(
        phase_history_rows=2,
        level_rate_threshold_mps=1.0,
        cadence_edges_s=(1.0, 5.0),
        max_gap_s=60.0,
    )
    batch = build_contextual_forecast_windows(
        frame,
        signal_columns=("signal",),
        target_channels=("target",),
        history_rows=2,
        context_config=config,
    )
    assert len(batch.X) == 4
    assert batch.meta["flight_id"].tolist() == ["A", "A", "B", "B"]
    assert batch.y[:, 0].tolist() == [12.0, 13.0, 16.0, 17.0]
    assert batch.X[0, :, 0].tolist() == [0.0, 1.0]
    assert batch.X[2, :, 0].tolist() == [4.0, 5.0]
    assert "track_sin" in batch.input_features
    assert "phase=level" in batch.input_features
