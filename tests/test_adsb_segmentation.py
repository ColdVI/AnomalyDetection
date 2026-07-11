"""Faz 0.3 segmentasyon testleri -- sentetik (gercek veri gerektirmez)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.segmentation import assign_flight_ids, flight_summary, new_leg_agreement, segment_flights


def _traces(icao: str, timestamps: list[float], new_leg_at: set[float] | None = None) -> pd.DataFrame:
    new_leg_at = new_leg_at or set()
    return pd.DataFrame({
        "source_id": icao,
        "timestamp_utc": timestamps,
        "lat": np.linspace(40.0, 41.0, len(timestamps)),
        "lon": np.linspace(29.0, 30.0, len(timestamps)),
        "flags_new_leg": [t in new_leg_at for t in timestamps],
    })


def test_assign_flight_ids_splits_on_gap():
    a = _traces("A", [0, 10, 20, 30])  # boşluksuz -> 1 uçuş
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])  # 2 saatlik boşluk -> 2 uçuş
    df = pd.concat([a, b], ignore_index=True)

    flight_id = assign_flight_ids(df, gap_s=1800.0)

    assert flight_id[df["source_id"] == "A"].nunique() == 1
    assert flight_id[df["source_id"] == "B"].nunique() == 2
    b_ids = flight_id[df["source_id"] == "B"].tolist()
    assert b_ids[0] == b_ids[1]
    assert b_ids[2] == b_ids[3]
    assert b_ids[0] != b_ids[2]


def test_assign_flight_ids_no_gap_is_single_flight():
    df = _traces("A", [0, 60, 120, 180, 240])
    assert assign_flight_ids(df, gap_s=1800.0).nunique() == 1


def test_assign_flight_ids_unsorted_input_matches_sorted():
    a = _traces("A", [0, 10, 20, 30])
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    df = pd.concat([a, b], ignore_index=True)
    shuffled = df.sample(frac=1.0, random_state=0)

    sorted_result = assign_flight_ids(df, gap_s=1800.0)
    shuffled_result = assign_flight_ids(shuffled, gap_s=1800.0)

    pd.testing.assert_series_equal(
        shuffled_result.sort_index(), sorted_result.sort_index(), check_names=False
    )


def test_assign_flight_ids_empty_df():
    df = _traces("A", [])
    assert len(assign_flight_ids(df, gap_s=1800.0)) == 0


def test_segment_flights_output_sorted_with_flight_id_column():
    a = _traces("A", [30, 0, 20, 10])  # kasten sırasız
    out = segment_flights(a, gap_s=1800.0)
    assert out["timestamp_utc"].is_monotonic_increasing
    assert "flight_id" in out.columns
    assert out["flight_id"].nunique() == 1


def test_new_leg_agreement_full():
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30], new_leg_at={7200 + 20})
    seg = segment_flights(b, gap_s=1800.0)
    assert new_leg_agreement(seg) == pytest.approx(1.0)


def test_new_leg_agreement_zero_when_flag_never_set():
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    seg = segment_flights(b, gap_s=1800.0)
    assert new_leg_agreement(seg) == pytest.approx(0.0)


def test_new_leg_agreement_nan_when_no_boundaries():
    a = _traces("A", [0, 10, 20, 30])
    seg = segment_flights(a, gap_s=1800.0)
    assert np.isnan(new_leg_agreement(seg))


def test_flight_summary_counts_rows_and_duration():
    a = _traces("A", [0, 10, 20, 30])
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    df = pd.concat([a, b], ignore_index=True)
    seg = segment_flights(df, gap_s=1800.0)

    summary = flight_summary(seg)
    assert set(summary.columns) == {"flight_id", "n_rows", "duration_s", "start_time"}
    assert summary.set_index("flight_id").loc["A_000", "n_rows"] == 4
    assert summary.set_index("flight_id").loc["A_000", "duration_s"] == pytest.approx(30.0)
