import numpy as np
import pandas as pd
import pytest

from adsb.contextual_scaling import (
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)


def test_zero_mad_is_excluded_without_floor_and_transform_is_clipped():
    frame = pd.DataFrame({"active": [0.0, 1.0, 2.0, 100.0], "constant": [4.0] * 4})
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=3.0)).fit(
        frame,
        ("active", "constant"),
        data_role=NATURAL_FIT_ROLE,
        contains_synthetic=False,
    )
    assert scaler.active_channels == ("active",)
    assert scaler.excluded_channels_ == ("constant",)
    assert scaler.to_dict()["mad_zero_policy"] == "exclude_without_floor"
    assert scaler.transform(frame)["active"].max() == 3.0


def test_scaler_rejects_synthetic_fit():
    with pytest.raises(ValueError, match="Synthetic"):
        StrictNaturalRobustScaler(StrictScalingConfig(clip=3.0)).fit(
            pd.DataFrame({"x": [0.0, 1.0]}),
            ("x",),
            data_role=NATURAL_FIT_ROLE,
            contains_synthetic=True,
        )


def test_all_zero_mad_channels_fail_closed():
    with pytest.raises(ValueError, match="MAD=0"):
        StrictNaturalRobustScaler(StrictScalingConfig(clip=3.0)).fit(
            pd.DataFrame({"x": [1.0, 1.0]}),
            ("x",),
            data_role=NATURAL_FIT_ROLE,
            contains_synthetic=False,
        )
