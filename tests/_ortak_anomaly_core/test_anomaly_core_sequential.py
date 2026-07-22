import pandas as pd

from anomaly_core.sequential import MultiChannelPageCUSUM, PageCUSUMConfig


def test_cusum_is_causal_and_resets_on_gap():
    fit = pd.DataFrame(
        {
            "flight_id": ["fit"] * 5,
            "timestamp_s": [0, 1, 2, 3, 4],
            "evaluable": [False, True, True, True, True],
            "x": [0, -1, 0, 1, 0],
        }
    )
    detector = MultiChannelPageCUSUM(
        PageCUSUMConfig(
            channels=("x",),
            reference_shift_z=1.0,
            z_clip=8.0,
            max_gap_s=2.0,
        )
    ).fit(fit)
    scored = detector.score(
        pd.DataFrame(
            {
                "flight_id": ["A"] * 5,
                "timestamp_s": [0, 1, 2, 10, 11],
                "evaluable": [False, True, True, True, True],
                "x": [0, 5, 5, 5, 5],
            }
        )
    )
    assert scored.loc[2, "cusum_score"] > scored.loc[1, "cusum_score"]
    assert scored.loc[3, "cusum_score"] == 0.0
    assert scored.loc[3, "cusum_reset_reason"] == "invalid_or_large_gap"

