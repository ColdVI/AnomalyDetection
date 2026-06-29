import math

import pytest

from src.common.bbox import in_turkey


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(39.0, 35.0), (36.0, 26.0), (42.0, 45.0), ("41.01", "29.0")],
)
def test_in_turkey_accepts_points_inside_or_on_boundary(lat, lon):
    assert in_turkey(lat, lon) is True


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (35.999, 35.0),
        (42.001, 35.0),
        (39.0, 25.999),
        (39.0, 45.001),
        (None, 35.0),
        (39.0, None),
        (math.nan, 35.0),
        (39.0, math.inf),
        ("not-a-number", 35.0),
    ],
)
def test_in_turkey_rejects_invalid_or_outside_points(lat, lon):
    assert in_turkey(lat, lon) is False
