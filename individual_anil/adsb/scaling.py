"""Kirpili robust olcekleme -- SEAD dersinin (archive/.../decisions.md ADR-016)
DOGRUDAN uygulanmasi: kirpilmamis RobustScaler, birkac asiri-genlik kanalin (orn.
irtifa binlerce, dikey hiz tek haneli) skoru surüklemesine yol acmisti. Burada
kirpma ZORUNLU -- olceklenmis deger [-clip, +clip] araligina sikistirilir.

Fit SADECE train (normal) pencerelerinden yapilir; val/test'e aynen uygulanir
(leakage yok). Maskeli (eksik) degerler istatistige girmez.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CLIP = 5.0
_EPS = 1e-6


class ClippedRobustScaler:
    def __init__(self, clip: float = DEFAULT_CLIP):
        self.clip = clip
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, X: np.ndarray, M: np.ndarray) -> "ClippedRobustScaler":
        """X, M: (n, window, features). Yalniz M=1 olan degerler istatistige girer."""
        n_features = X.shape[-1]
        median = np.zeros(n_features)
        iqr = np.ones(n_features)
        for f in range(n_features):
            valid = X[..., f][M[..., f] > 0]
            if len(valid) == 0:
                continue
            q1, med, q3 = np.percentile(valid, [25, 50, 75])
            median[f] = med
            iqr[f] = (q3 - q1) or 1.0
        self.median_ = median
        self.iqr_ = iqr
        return self

    def transform(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("fit() once cagrilmadan transform() cagrilamaz")
        scaled = (X - self.median_) / (self.iqr_ + _EPS)
        scaled = np.clip(scaled, -self.clip, self.clip)
        return scaled * M  # maskeli (eksik) pozisyonlar 0'da kalir

    def fit_transform(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        return self.fit(X, M).transform(X, M)

    def to_dict(self) -> dict:
        return {"clip": self.clip, "median": self.median_.tolist(), "iqr": self.iqr_.tolist()}
