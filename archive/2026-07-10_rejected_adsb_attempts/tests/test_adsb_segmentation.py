"""ADSB-0 segmentasyon testleri -- sentetik (gercek veri gerektirmez)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adsb.segmentation import assign_flight_ids, new_leg_agreement, segment_flights


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
    # aircraft A: continuous (1 flight); aircraft B: one 2h gap in the middle (2 flights)
    a = _traces("A", [0, 10, 20, 30])
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])
    df = pd.concat([a, b], ignore_index=True)

    flight_id = assign_flight_ids(df, gap_s=1800.0)

    assert flight_id[df["source_id"] == "A"].nunique() == 1
    assert flight_id[df["source_id"] == "B"].nunique() == 2
    # B's first two rows share a flight, last two share a different one
    b_ids = flight_id[df["source_id"] == "B"].tolist()
    assert b_ids[0] == b_ids[1]
    assert b_ids[2] == b_ids[3]
    assert b_ids[0] != b_ids[2]


def test_assign_flight_ids_no_gap_is_single_flight():
    df = _traces("A", [0, 60, 120, 180, 240])
    flight_id = assign_flight_ids(df, gap_s=1800.0)
    assert flight_id.nunique() == 1


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
    result = assign_flight_ids(df, gap_s=1800.0)
    assert len(result) == 0


def test_segment_flights_output_sorted_with_flight_id_column():
    a = _traces("A", [30, 0, 20, 10])  # deliberately unsorted
    out = segment_flights(a, gap_s=1800.0)
    assert out["timestamp_utc"].is_monotonic_increasing
    assert "flight_id" in out.columns
    assert out["flight_id"].nunique() == 1


def test_new_leg_agreement_full():
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30], new_leg_at={7200 + 20})
    seg = segment_flights(b, gap_s=1800.0)
    agreement = new_leg_agreement(seg)
    assert agreement == pytest.approx(1.0)


def test_new_leg_agreement_zero_when_flag_never_set():
    b = _traces("B", [0, 10, 7200 + 20, 7200 + 30])  # flags_new_leg all False
    seg = segment_flights(b, gap_s=1800.0)
    agreement = new_leg_agreement(seg)
    assert agreement == pytest.approx(0.0)


def test_new_leg_agreement_nan_when_no_boundaries():
    a = _traces("A", [0, 10, 20, 30])  # single flight, no boundary rows
    seg = segment_flights(a, gap_s=1800.0)
    agreement = new_leg_agreement(seg)
    assert np.isnan(agreement)
