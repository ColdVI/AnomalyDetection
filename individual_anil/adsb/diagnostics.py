"""ZORUNLU egitim-sonrasi tani: model gercekten ogreniyor mu, yoksa yalniz
genlige mi bakiyor?

SEAD dersi (archive/2026-07-10_legacy_non_adsb_ml/docs/decisions.md ADR-016..019):
LSTM-AE/Dense-AE/USAD skorlari kirpilmamis olcekleme yuzunden birkac asiri-genlik
kanaldan suruklendi -- egitilmis model skoruyla RASTGELE-BASLATILMIS (hic
egitilmemis) ayni mimarinin skoru arasindaki Spearman korelasyonu ~0.96 cikti,
yani "egitilmis" model fiilen rastgele init'ten ayirt edilemiyordu. Bu modul o
teshisi, HER egitimden sonra calistirilmasi gereken standart bir fonksiyona
donusturur -- ADS-B modelleri icin bu kontrol baslangictan (ilk egitimden) itibaren
zorunlu, SEAD'de oldugu gibi sonradan kesfedilen bir bulgu degil.

Kullanim orunegi (herhangi bir mimari icin, model-bagimsiz):
    trained_model, _ = train_dense_autoencoder(X_train, M_train, ...)
    untrained_model = DenseAutoencoder(window, n_features, hidden_dims)  # egitilmemis
    trained_scores = reconstruction_scores(trained_model, X_eval, M_eval)
    untrained_scores = reconstruction_scores(untrained_model, X_eval, M_eval)
    report = magnitude_domination_check(trained_scores, untrained_scores, X_eval, M_eval)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr

DEFAULT_RHO_THRESHOLD = 0.8


def magnitude_only_score(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Model-bagimsiz taban: maskeli ||x||^2, pencere basina."""
    return (X ** 2 * M).sum(axis=tuple(range(1, X.ndim)))


def magnitude_domination_check(
    trained_scores: np.ndarray,
    untrained_scores: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
) -> dict:
    """Egitilmis skor, rastgele-init skoru VEYA ham genlik ile asiri korele mi?

    rho >= rho_threshold ise (varsayilan 0.8) egitilmis modelin buyuk ihtimalle
    ogrenilmis bir orunutten degil, kirpilmamis genlikten skor uretiyor olabilecegi
    isaretlenir -- bu KESIN kanit degil, SEAD'deki gibi daha derin inceleme
    (kanal-bazinda residual analizi) gerektirir, ama erken uyari olarak zorunludur.
    """
    mag_scores = magnitude_only_score(X, M)

    rho_trained_untrained, _ = spearmanr(trained_scores, untrained_scores)
    rho_trained_magnitude, _ = spearmanr(trained_scores, mag_scores)

    flagged = (rho_trained_untrained >= rho_threshold) or (rho_trained_magnitude >= rho_threshold)

    return {
        "rho_trained_vs_untrained": float(rho_trained_untrained),
        "rho_trained_vs_magnitude": float(rho_trained_magnitude),
        "rho_threshold": rho_threshold,
        "magnitude_domination_flagged": bool(flagged),
    }


def fit_score_baseline(train_scores: np.ndarray) -> dict:
    """TRAIN (yalniz-normal) skor dagiliminin medyan/MAD'ini dondurur -- z-score/
    guven hesaplamasinin referans alacagi taban. Medyan/MAD kullanilir (ortalama/std
    degil): kirpilmamis olcekleme SEAD'de birkac asiri-genlik pencerenin skoru
    surukledigini gostermisti (ADR-016), medyan/MAD aykiri-degerlere ortalama/std'den
    çok daha az duyarli.
    """
    median = float(np.median(train_scores))
    mad = float(np.median(np.abs(np.asarray(train_scores) - median))) * 1.4826  # normal-tutarli olcek
    return {"median": median, "mad": mad or 1e-6}


def z_score_confidence(scores: np.ndarray, baseline: dict) -> np.ndarray:
    """Skorlari TRAIN tabanina gore z-score'a, sonra standart normal CDF ile
    (0,1) araliginda bir "guven" degerine cevirir -- skor taban dagilimdan ne
    kadar uzaktaysa guven o kadar 1'e yaklasir (orn. 0.95 = tek-tarafli z~1.645).

    NOT: bu istatistiksel bir p-degeri DEGIL -- rekonstruksiyon skorlari normal
    dagilmiyor (sag-carpik, sifirda sinirli); CDF sadece yorumlanabilir, tek
    yonlu bir "ne kadar uc-deger" olcegi olarak kullanilir.
    """
    z = (np.asarray(scores) - baseline["median"]) / baseline["mad"]
    return norm.cdf(z)
