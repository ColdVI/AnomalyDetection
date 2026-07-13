"""adsb/rules.py testleri -- elle hesaplanmis beklenen degerlerle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.rules import CAP, RULE_CHANNELS, Z0, ResidualRuleScorer


def _normal_feat(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(0.0, 1.0, n) for c in RULE_CHANNELS})


def test_fit_computes_median_and_mad():
    df = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0, 5.0] for c in RULE_CHANNELS})
    scorer = ResidualRuleScorer().fit(df)
    cal = scorer.calibration_["speed_residual"]
    assert cal["median"] == 3.0
    # MAD = median(|x-3|) = median([2,1,0,1,2]) = 1.0 -> *1.4826
    assert cal["mad"] == pytest.approx(1.4826)


def test_zero_mad_channel_is_excluded_not_hair_trigger():
    """Zero-MAD dersi (arsivlenen deneme + bu projenin 1. kural turu): kuantize/
    sabit kanal floor'la 'kil tetik' yapilmamali, kalibre-edilemez sayilip
    skordan haric tutulmali."""
    df = _normal_feat()
    df["heading_residual"] = 0.0  # sabit -> MAD=0
    scorer = ResidualRuleScorer().fit(df)
    assert "heading_residual" in scorer.excluded_channels_
    assert "heading_residual" not in scorer.calibration_
    bozuk = df.copy()
    bozuk.loc[0, "heading_residual"] = 0.001  # ufacik sapma penalty URETMEMELI
    pen = scorer.row_penalties(bozuk)
    assert pen.iloc[0] == 0.0
    assert np.isfinite(pen).all()


def test_clean_rows_get_near_zero_penalty():
    df = _normal_feat()
    scorer = ResidualRuleScorer().fit(df)
    pen = scorer.row_penalties(df)
    # z<3 icin penalty 0 -- normal dagilimda satirlarin buyuk cogunlugu
    assert (pen == 0.0).mean() > 0.9


def test_violation_gets_positive_penalty_proportional_to_magnitude():
    df = _normal_feat()
    scorer = ResidualRuleScorer().fit(df)
    bozuk = df.copy()
    bozuk.loc[0, "speed_residual"] = 8.0   # ~8 sigma
    bozuk.loc[1, "speed_residual"] = 100.0  # asiri -- CAP'lenmeli
    pen = scorer.row_penalties(bozuk)
    assert pen.iloc[0] > 0
    assert pen.iloc[1] > pen.iloc[0]
    assert pen.iloc[1] <= CAP * sum(scorer.weights.values())


def test_penalty_matches_manual_formula():
    df = pd.DataFrame({c: [0.0] * 100 for c in RULE_CHANNELS})
    df.loc[:, "speed_residual"] = np.linspace(-1, 1, 100)  # median~0
    scorer = ResidualRuleScorer().fit(df)
    cal = scorer.calibration_["speed_residual"]
    test_row = df.iloc[[0]].copy()
    test_row["speed_residual"] = cal["median"] + (Z0 + 2.5) * cal["mad"]
    pen = scorer.row_penalties(test_row)
    assert pen.iloc[0] == pytest.approx(2.5, abs=1e-9)


def test_nan_channel_contributes_zero_not_nan():
    df = _normal_feat()
    scorer = ResidualRuleScorer().fit(df)
    bozuk = df.copy()
    bozuk.loc[0, "vertical_rate_residual"] = np.nan
    pen = scorer.row_penalties(bozuk)
    assert np.isfinite(pen.iloc[0])


def test_row_penalties_before_fit_raises():
    with pytest.raises(RuntimeError):
        ResidualRuleScorer().row_penalties(_normal_feat())


def test_roundtrip_to_from_dict():
    df = _normal_feat()
    scorer = ResidualRuleScorer().fit(df)
    clone = ResidualRuleScorer.from_dict(scorer.to_dict())
    pd.testing.assert_series_equal(scorer.row_penalties(df), clone.row_penalties(df))
