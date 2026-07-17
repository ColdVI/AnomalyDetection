import pandas as pd
import pytest

from residual_v1.viz.handout import chunked, downsample, flight_slug, split_scope


def test_handout_scope_keeps_holdout_separate():
    visible, sealed = split_scope(
        {
            "partitions": {
                "development": {"flight_ids": ["dev/a"]},
                "test": {"flight_ids": ["test/b"]},
                "holdout": {"flight_ids": ["secret/c"]},
            }
        }
    )
    assert visible == {"dev/a": "development", "test/b": "test"}
    assert sealed == ["secret/c"]
    assert "secret/c" not in visible


def test_handout_scope_rejects_role_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_scope(
            {
                "partitions": {
                    "development": {"flight_ids": ["same"]},
                    "test": {"flight_ids": ["same"]},
                    "holdout": {"flight_ids": []},
                }
            }
        )


def test_downsample_uses_observed_rows_and_keeps_endpoints():
    frame = pd.DataFrame({"t": range(100), "value": range(100)})
    result = downsample(frame, max_points=11)
    assert len(result) == 11
    assert result.iloc[0].to_dict() == {"t": 0, "value": 0}
    assert result.iloc[-1].to_dict() == {"t": 99, "value": 99}
    assert set(result["t"]).issubset(set(frame["t"]))


def test_claude_pdf_chunks_stay_below_one_hundred_pages():
    values = [{"flight_id": str(index)} for index in range(348)]
    parts = chunked(values, 87)
    assert [len(part) for part in parts] == [87, 87, 87, 87]
    assert all(len(part) < 100 for part in parts)


def test_flight_slug_is_stable_and_path_safe():
    first = flight_slug("Real-Motor/acce/406_1/log")
    assert first == flight_slug("Real-Motor/acce/406_1/log")
    assert "/" not in first
    assert "\\" not in first
