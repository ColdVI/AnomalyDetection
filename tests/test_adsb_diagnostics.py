"""adsb/diagnostics.py testleri -- SEAD'deki genlik-baskinligi bulgusunu (ADR-016)
tekrar-uretme/yakalama yetenegini dogrudan sinar."""

from __future__ import annotations

import numpy as np

from adsb.diagnostics import magnitude_domination_check, magnitude_only_score


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
