"""ADS-B kural/degerlendirme/diagnostik testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from __future__ import annotations

import numpy as np

import pandas as pd

import pytest

from adsb.rules import CAP, RULE_CHANNELS, Z0, ResidualRuleScorer

from adsb.evaluation import (
    EpisodeContract,
    alarm_episodes,
    diagnostic_window_metrics,
    event_detection_metrics,
    natural_alert_burden,
    scoreable_exposure,
)

from adsb.diagnostics import (
    fit_score_baseline,
    magnitude_domination_check,
    magnitude_only_score,
    z_score_confidence,
)



# ===== kaynak: test_adsb_rules =====

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



# ===== kaynak: test_adsb_evaluation =====

def test_window_metrics_keep_mixed_q_out_of_steady_state():
    result = diagnostic_window_metrics(
        np.array([0.0, 0.25, 1.0, np.nan]), np.array([0.0, 0.5, 1.0, 9.0])
    )
    assert result["n_unscoreable"] == 1
    assert result["q_strata"] == {"q_eq_0": 1, "q_mixed": 1, "q_eq_1": 1}
    assert result["primary_y_any"]["n_positive"] == 2
    assert result["secondary_steady_state"]["n"] == 2


def test_exposure_uses_interval_union_not_window_sum():
    meta = pd.DataFrame(
        {"flight_id": ["a", "a", "b"], "t_start": [0.0, 5.0, 0.0], "t_end": [10.0, 15.0, 5.0]}
    )
    result = scoreable_exposure(meta)
    assert result["scoreable_seconds"] == pytest.approx(20.0)
    assert result["n_scoreable_flights"] == 2


def test_alarm_episode_merge_is_flight_local_and_uses_t_end():
    meta = pd.DataFrame(
        {"flight_id": ["a", "a", "a", "b"], "t_start": [0, 0, 0, 0], "t_end": [10, 65, 130, 50]}
    )
    episodes = alarm_episodes(meta, np.ones(4, dtype=bool), contract=EpisodeContract(merge_gap_s=60))
    assert len(episodes) == 3
    assert episodes.loc[episodes.flight_id == "a", "n_emissions"].tolist() == [2, 1]


def test_natural_burden_reports_episode_rate_and_flight_fraction_separately():
    meta = pd.DataFrame(
        {"flight_id": ["a", "a", "b"], "t_start": [0.0, 1800.0, 0.0], "t_end": [1800.0, 3600.0, 3600.0]}
    )
    result = natural_alert_burden(meta, np.array([True, True, False]))
    assert result["n_alert_episodes"] == 2
    assert result["scoreable_flight_hours"] == pytest.approx(2.0)
    assert result["alert_episodes_per_scoreable_flight_hour"] == pytest.approx(1.0)
    assert result["alerted_flight_fraction"] == pytest.approx(0.5)


def test_event_metric_does_not_point_adjust_whole_event():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "flight_id": ["a", "b"],
            "observable_onset": [100.0, 200.0],
            "event_end": [160.0, 260.0],
        }
    )
    meta = pd.DataFrame({"flight_id": ["a", "a", "b"], "t_end": [90.0, 130.0, 270.0]})
    result = event_detection_metrics(events, meta, np.array([True, True, True]))
    assert result["event_recall"] == pytest.approx(0.5)
    assert result["first_alarm_delay_s"]["median"] == pytest.approx(30.0)



# ===== kaynak: test_adsb_diagnostics =====

def test_magnitude_only_score_matches_manual_computation():
    X = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    M = np.array([[[1.0, 1.0], [1.0, 0.0]]])  # son eleman maskeli
    # 1^2+2^2+3^2 (4. eleman maskeli, disarida) = 1+4+9 = 14
    result = magnitude_only_score(X, M)
    np.testing.assert_allclose(result, [14.0])


def test_flags_when_trained_score_is_pure_magnitude():
    """SEAD'in kendi bulgusunu tekrar-uretir: skor == genlik ise (ogrenilmis sinyal
    yok) kontrol bunu YAKALAMALI."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5, 3))
    M = np.ones_like(X)

    magnitude_scores = magnitude_only_score(X, M)
    trained_scores = magnitude_scores.copy()  # egitilmis skor TAM OLARAK genlik
    untrained_scores = rng.normal(size=200)  # ilgisiz rastgele init skoru

    report = magnitude_domination_check(trained_scores, untrained_scores, X, M)
    assert report["magnitude_domination_flagged"] is True
    assert report["rho_trained_vs_magnitude"] > 0.99


def test_flags_when_trained_score_matches_untrained_score():
    """SEAD'in ASIL bulgusu: egitilmis == rastgele-init (ogrenme hicbir sey
    degistirmemis)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5, 3))
    M = np.ones_like(X)
    untrained_scores = rng.normal(size=200)
    trained_scores = untrained_scores.copy()  # egitim skoru degistirmemis

    report = magnitude_domination_check(trained_scores, untrained_scores, X, M)
    assert report["magnitude_domination_flagged"] is True
    assert report["rho_trained_vs_untrained"] > 0.99


def test_does_not_flag_when_scores_are_independent():
    """Genlikten ve rastgele-init'ten BAGIMSIZ bir egitilmis skor (gercek ogrenme
    senaryosunun proxy'si) yanlislikla isaretlenmemeli."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5, 3))
    M = np.ones_like(X)
    untrained_scores = rng.normal(size=300)
    # trained_scores, ne genlikle ne de untrained ile iliskili -- tamamen ayri rng
    trained_scores = np.random.default_rng(99).normal(size=300)

    report = magnitude_domination_check(trained_scores, untrained_scores, X, M)
    assert report["magnitude_domination_flagged"] is False
    assert abs(report["rho_trained_vs_untrained"]) < 0.3
    assert abs(report["rho_trained_vs_magnitude"]) < 0.3


def test_threshold_is_configurable():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 5, 3))
    M = np.ones_like(X)
    magnitude_scores = magnitude_only_score(X, M)
    untrained_scores = rng.normal(size=100)
    # magnitude ile orta duzeyde korele (gurultu eklenmis) bir skor
    trained_scores = magnitude_scores + rng.normal(scale=magnitude_scores.std() * 2, size=100)

    lenient = magnitude_domination_check(trained_scores, untrained_scores, X, M, rho_threshold=0.99)
    strict = magnitude_domination_check(trained_scores, untrained_scores, X, M, rho_threshold=0.05)
    assert lenient["magnitude_domination_flagged"] is False
    assert strict["magnitude_domination_flagged"] is True


def test_fit_score_baseline_matches_manual_median_mad():
    train_scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # 100.0 aykiri-deger
    baseline = fit_score_baseline(train_scores)
    assert baseline["median"] == 3.0
    # MAD = median(|x - 3|) = median([2,1,0,1,97]) = 1.0, *1.4826
    assert abs(baseline["mad"] - 1.4826) < 1e-4


def test_z_score_confidence_is_half_at_the_median():
    baseline = {"median": 10.0, "mad": 2.0}
    conf = z_score_confidence(np.array([10.0]), baseline)
    assert abs(conf[0] - 0.5) < 1e-9


def test_z_score_confidence_higher_for_scores_further_above_median():
    baseline = fit_score_baseline(np.random.default_rng(0).normal(loc=5.0, scale=1.0, size=500))
    near = z_score_confidence(np.array([baseline["median"] + baseline["mad"]]), baseline)
    far = z_score_confidence(np.array([baseline["median"] + 5 * baseline["mad"]]), baseline)
    assert far[0] > near[0]
    assert 0.0 <= near[0] <= 1.0 and 0.0 <= far[0] <= 1.0


def test_z_score_confidence_robust_to_outlier_in_baseline_fit():
    """SEAD dersi: aykiri-deger baseline'i bozmamali (ortalama/std kullansaydik bozardi)."""
    clean_train = np.random.default_rng(1).normal(loc=5.0, scale=1.0, size=200)
    with_outlier = np.concatenate([clean_train, [10_000.0]])
    b_clean = fit_score_baseline(clean_train)
    b_outlier = fit_score_baseline(with_outlier)
    assert abs(b_clean["median"] - b_outlier["median"]) < 0.5
    assert abs(b_clean["mad"] - b_outlier["mad"]) < 0.5

