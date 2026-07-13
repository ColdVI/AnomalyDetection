from __future__ import annotations

import numpy as np
import pandas as pd

# Script functions are pure and intentionally imported for exact legacy replay.
from scripts.adsb_build_synthetic_truth_v2_corpus import (
    _active_mask,
    _annotate_recipe_vectorized,
)


def test_non_dropout_active_range_is_onset_to_end():
    assert _active_mask(10, "track_frozen").tolist() == [False] * 5 + [True] * 5


def test_dropout_active_range_is_exact_legacy_random_block_not_onset_to_end():
    active = _active_mask(10, "altitude_dropout")
    positions = np.flatnonzero(active)
    assert len(positions) == int((10 - 5) * 0.3)
    assert positions[0] >= 5
    assert positions[-1] < 10
    assert not active[-1] or len(positions) == 5


def test_vectorized_pair_annotation_keeps_exact_block_per_flight():
    clean = pd.DataFrame(
        {
            "flight_id": np.repeat(["a", "b"], 10),
            "timestamp_utc": np.tile(np.arange(10, dtype=float), 2),
            "alt": 1000.0,
            "label": None,
        }
    )
    corrupt = clean.copy()
    expected = np.tile(_active_mask(10, "altitude_dropout"), 2)
    corrupt.loc[expected, "alt"] = np.nan
    truth = _annotate_recipe_vectorized(clean, corrupt, "altitude_dropout")
    assert truth["injection_active"].tolist() == expected.tolist()
    assert truth["observable_changed"].tolist() == expected.tolist()
    assert truth.groupby("flight_id")["event_id"].nunique().eq(1).all()
