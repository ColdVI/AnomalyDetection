"""Kural-bazli, formul-tabanli fizik-residual penalty skorlayicisi.

ADR-024'un kok-neden teshisi: esit-agirlikli NN loss'u residual kanallarini
gormezden geliyor (uc mimari de magnitude-domination'da isaretlendi, pooled AUC
0.55-0.57). Bu modul TERS yaklasim: ogrenme YOK, residual'lar zaten aritmetik
ozdeslik oldugu icin dogrudan istatistiksel esik + penalty toplami kullanilir.
ML-12 dersiyle ayni ruh: tek domain-secilmis sinyal, 16-feature ogrenilmis
modeli gecmisti (0.205 -> 0.459 recall).

Matematik (tamamen seffaf, on-kayitli, sonuc gorulup ayarlanmadi):
    1. Kalibrasyon (yalniz-normal train satirlarindan, kanal basina):
           median_c, MAD_c = median(|x - median_c|) * 1.4826
       MAD_c == 0 ise kanal skordan tamamen haric tutulur; floor uygulanmaz.
    2. Satir penalty'si (kanal basina robust z-skoru esik asimi):
           z_c   = |x_c - median_c| / MAD_c
           pen_c = min(max(0, z_c - Z0), CAP)          Z0=3.0, CAP=10.0
           penalty(satir) = sum_c w_c * pen_c           w_c = 1.0 (uniform)
       NaN kanal -> katki 0 (yargilanamaz; missingness AYRI bir S2 veri-kalite
       kanalinin isi, fizik kuralinin degil -- altitude_dropout bu skorlayicinin
       KAPSAMI DISINDA, durustce beyan edilir).
    3. Pencere skoru: pencere icindeki satir-penalty'lerinin ortalamasi
       (NN'lerin pencere-MSE'siyle ayni degerlendirme birimi -- karsilastirma
       adil olsun diye).

Sabitler (Z0, CAP, agirliklar) SONUC GORULMEDEN secilmistir ve bu turda
degistirilemez -- proje disiplini (post-hoc ayar yasagi).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE_CHANNELS = [
    "vertical_rate_residual",
    "speed_residual",
    "heading_residual",
    "altitude_source_residual",
]
Z0 = 3.0        # penalty baslama esigi (robust z)
CAP = 10.0      # tek kanalin toplam skoru ele gecirmesini onler
MAD_FLOOR = 1e-6


class ResidualRuleScorer:
    """Yalniz-normal train verisinden kanal-bazina median/MAD kalibre eder,
    sonra herhangi bir feature-tablosuna satir-bazi penalty skoru verir."""

    def __init__(self, channels: list[str] | None = None,
                 weights: dict[str, float] | None = None):
        self.channels = channels or list(RULE_CHANNELS)
        self.weights = weights or {c: 1.0 for c in self.channels}
        self.calibration_: dict[str, dict[str, float]] | None = None

    def fit(self, feat_df: pd.DataFrame) -> "ResidualRuleScorer":
        calibration = {}
        excluded = []
        for c in self.channels:
            vals = feat_df[c].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                raise ValueError(f"'{c}' kanalinda hic sonlu deger yok -- kalibre edilemez")
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median))) * 1.4826
            if mad == 0.0:
                # Kuantize/ayrik veri (orn. irtifa 25ft adimlarla raporlanir ->
                # baro-jeo fark turevi cogunlukla TAM 0): MAD=0 cikan kanal robust
                # kalibre EDILEMEZ. Floor'la "kil tetik" yapmak yerine skordan
                # tamamen haric tutulur (1. turda floor denendi: normal pencerelerin
                # %93.8'ine gurultu-penalty'si basti, tum senaryolari zehirledi).
                excluded.append(c)
                continue
            calibration[c] = {"median": median, "mad": max(mad, MAD_FLOOR)}
        self.calibration_ = calibration
        self.excluded_channels_ = excluded
        return self

    def row_penalties(self, feat_df: pd.DataFrame) -> pd.Series:
        """Satir basina toplam penalty (>= 0). NaN kanallar katki yapmaz."""
        if self.calibration_ is None:
            raise RuntimeError("Once fit() cagrilmali")
        total = np.zeros(len(feat_df), dtype=float)
        for c in self.channels:
            if c not in self.calibration_:  # MAD=0 nedeniyle haric tutulan kanal
                continue
            cal = self.calibration_[c]
            z = np.abs(feat_df[c].to_numpy(dtype=float) - cal["median"]) / cal["mad"]
            pen = np.clip(z - Z0, 0.0, CAP)
            total += self.weights[c] * np.nan_to_num(pen, nan=0.0)
        return pd.Series(total, index=feat_df.index, name="rule_penalty")

    def to_dict(self) -> dict:
        return {
            "channels": self.channels,
            "weights": self.weights,
            "z0": Z0,
            "cap": CAP,
            "calibration": self.calibration_,
            "excluded_channels": getattr(self, "excluded_channels_", []),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualRuleScorer":
        scorer = cls(channels=list(d["channels"]), weights=dict(d["weights"]))
        scorer.calibration_ = {k: dict(v) for k, v in d["calibration"].items()}
        scorer.excluded_channels_ = list(d.get("excluded_channels", []))
        return scorer
