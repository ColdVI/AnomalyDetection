# ML-16 Kol D: SEAD Duz (Recurrent-Olmayan) Autoencoder Degerlendirmesi

Durum: ON-KAYIT (2026-07-09, sonuclar gorulmeden sabitlendi, sonra kosulacak).
Ust plan: `docs/ML14_MASTER_IYILESTIRME_PLANI.md`. Kardes dokuman:
`docs/ML16_KOL_U_USAD_SEAD_PLAN.md` (Kol U, USAD). Bu ikisi ve
`docs/ML16_KOL_L_LSTM_SEAD_PLAN.md` (Kol L, paralel calisan baska bir ajanin
isi -- bu faz ONA DOKUNMUYOR, sadece referans/karsilastirma noktasi) ayni
ML-16 fazinin bagimsiz kollaridir. **ONEMLI isim ayrimi:** "ML-17" bu depoda
ZATEN farkli, spesifik bir anlama sahip -- blind holdout'un kullanici
onayiyla bir kez acildigi endgame fazi
(`docs/ML14_MASTER_IYILESTIRME_PLANI.md`, `docs/ML15_KALIBRASYON_PLAN.md`).
Bu faz o degildir; holdout burada da ACILMAZ (SS3 Gate A).

## SS0 Gerekce ve durustce beklenti

`docs/ML1_BULGULAR_VE_HATALAR.md` ML-2 bolumu (2026-07-02,
`notebooks/03_autoencoder_lstm_ae_egitim.ipynb`): ALFA'da yalniz ~10 dk normal
ucus veriyle Dense AE ucus-ROC **0.622**, LSTM-AE **0.731**, IF-fuzyon
**0.833** -- Dense AE en zayif model. Bu mimari hicbir zaman SEAD icin
egitilmedi ve hicbir zaman bu depodaki yeniden-kullanilabilir `src/ml/`
modulu olarak paketlenmedi -- yalniz not defteri kodu olarak var oldu.

ML-14 veri zenginlestirmesi SEAD normal-ucus egitim havuzunu 324'ten 899'a
cikardi -- Kol L'nin (LSTM-AE) tam olarak ayni veri artisiyla IF-fuzyonuna
karsi kaybetmekten (0.731) kesin bicimde onune gecmesine (0.918, ALFA'da)
neden olan degisim. Dense AE'nin eski kaybinin de dusuk-veri artifakti olup
olmadigini gormek adil ve ucuz bir sorudur.

**Durust on-kayitli beklenti (sonuc gorulmeden yazildi):** LSTM-AE'nin
avantaji kismen mimaridir -- pencereyi sirali sequence olarak modelleyip
zamansal baglami (roll/pitch/yaw hizi, gps hizinin zaman ici degisimi gibi
sinyallerin SIRASI) kullanir; Dense AE penceredeki 50 zaman-adimini TEK duz
vektore flatten ettigi icin bu sirayi kaybeder (feature'lar hala var, ama
"hangi deger once/sonra geldi" bilgisi modele hic girmiyor). Bu nedenle
**Dense AE'nin bu turda da LSTM-AE'ye (ve muhtemelen mevcut en iyi
ml14_fusion'a) kaybetmesi daha olasi sonuctur** -- bu deney veri hacmi
degil mimari farkini kapatmayi test ediyor, yuksek-guven bir "kazanir"
bahsi degil. **Sonuc ne olursa olsun -- "yine kaybetti, bu mimari bir
dezavantaj, veri hacmi artifakti degil" dahil -- durustce raporlanacak;**
sonuc gorulduktan sonra hicbir post-hoc parametre degisikligi yapilmayacak
(ADR-012/013 disiplini).

## SS1 Degerlendirilecek skor varyantlari (SABIT -- uc ve yalnizca uc)

1. **(a) `dense_ae_recon`** -- SEAD-normal-ucuslarinda-egitilmis Dense
   Autoencoder'in maskeli-MSE reconstruction skorunun tek basina, val-normal
   ampirik CDF'ine kalibre edilmis hali.
2. **(b) `dense_ae_ml14_fusion` = max(dense_ae_recon, ml14_fusion)** -- mevcut
   en iyi genel fuzyonla max-birlesim.
3. **(c) `dense_ae_itki_fusion` = max(dense_ae_recon, itki_komutu)** -- mevcut
   en iyi kategori-uzmani ince modulle max-birlesim.

Baska hicbir ad hoc varyant (farkli fusion kombinasyonu, farkli kalibrasyon,
farkli mimari/hiperparametre, farkli pencere/stride) degerlendirilmeyecek.
`ml14_fusion`/`itki_komutu`'nun KENDI satirlari bu kosuda yeniden
raporlanmaz (zaten donmus `artifacts/ml14/uav_sead/full_matrix` CSV'sinden
alinir); bu kosuda yalniz ara sutun olarak (b/c'yi uretmek icin) yeniden
hesaplanirlar -- ayni kod (`fit_modular_iforest`, `_score_modules`,
`max_score_fusion`) ile, bu yuzden donmus ml14 kosusuyla sayisal olarak
ortusmesi beklenir (determinism testiyle dogrulanir, Kol L'nin Gate A'siyla
ayni kalip).

## SS2 Model, pencereleme ve karar-donusu hizalama protokolu

**Mimari (yeni, `src/ml/models/dense_autoencoder.py`):** `DenseAutoencoder`
-- pencereyi `(window, n_features)` -> duz `window*n_features` vektor ->
simetrik fully-connected encoder (`Linear(flat,7)+ReLU` -> `Linear(7,4)`
latent) / decoder (`Linear(4,7)+ReLU` -> `Linear(7,flat)`) -> tekrar
`(window, n_features)`. Parametre sayisi **16574**, `LSTMAutoencoder(22,
hidden=32, latent=16)`'nin **17414** parametresine kasitli olarak kabaca
esitlendi (%4.8 daha az) -- performans farki kapasiteden degil mimariden
gelsin diye. `masked_mse`/`reconstruction_scores`
`src/ml/models/lstm_autoencoder.py`'den DEGISTIRILMEDEN import edilir (ikisi
de yalniz tensor sekline/`model(x)` cagrisina bagli, LSTM'e ozgu degil).

`AE_FEATURES["uav_sead"]` (22 sutun), `WINDOW["uav_sead"]=50`,
`STRIDE["uav_sead"]=5` -- Kol L (ve LSTM-AE'nin kendisi) ile BIREBIR AYNI,
adil karsilastirma icin kasitli olarak degistirilmedi. Pencereleme
`src/ml/data/windowing.py::build_windows` ile uretilir, yeni pencereleme
kodu YAZILMAZ.

**Split protokolu:** `data/gold/ml_features/split_manifest.json`'daki GUNCEL
`sources.uav_sead.splits` (split_00..split_04, ayni ml14/ml15/Kol L 5-seed
protokolu). Her split icin: aynı split'in `fit_scaler_params`/
`apply_scaler_params` (train-only RobustScaler, IF modulleriyle PAYLASILAN
tek scaler) ile olceklenen tabloda, `AE_FEATURES["uav_sead"]` sutunlari NaN
korunarak ayrilir ve `build_windows(..., window=50, stride=5, max_gap_s=2.0)`
ile pencerelenir. Egitim yalniz split'in `train` (normal) ucuslarindan,
erken-durdurma `val` (normal) ucuslarindan; `test` ucuslari egitime hic
girmez. Her split kendi Dense AE'sini egitir (5 ayri model).

**Kapsam/eksik-veri notu:** Kol L'nin SS2'sinde olcculen ayni yapisal GPS-
sagligi topic bosluğu (development satir doluluğu `course_change_deg` vb.
sutunlarda ~%6.0) burada da gecerli -- ayni ham feature tablosu kullaniliyor.
Bu sutunlar SESSIZCE cross-flight/cross-source medyanla DOLDURULMAYACAK:
`build_windows`'in maske kanali (NaN->0 + is_missing maskesi) ve
`masked_mse` bunu zaten dogru bicimde ele aliyor (docs/ML_YETERSIZLIKLER_KAYDI.md
B.1/B.2 ile ayni yapisal eksiklik ailesi, "hayalet imputation" YAPILMIYOR).

**Pencere-skorundan karar-anina (1 Hz) hizalama:**
`scripts/run_ml8a_temporal_boosting.py::_align_score` DOGRUDAN import edilip
kullanilir (Kol L ile ayni, yeni bir hizalama semasi ICAT EDILMEZ). Her ham
satirda (kaynak, t_rel_s) icin "en son tamamlanmis pencerenin reconstruction
skoru" (ilk pencere tamamlanmadan onceki satirlar NaN -- nedensel, gelecek
sizintisi yok). Bu ham skor `src/ml/evaluation/score_fusion.py::
empirical_probability` ile split'in val-normal referansina kalibre edilir,
sonra `last_causal_per_bucket` ile 1 saniyelik karar kovasina indirgenir.

**Karar katmanlari/butce/fusion -- DEGISTIRILMEDEN import edilir:**
`src/ml/decision/decision_layers.py` (Threshold/K-of-N/CUSUM),
`src/ml/evaluation/score_fusion.py::max_score_fusion`/`last_causal_per_bucket`,
`scripts/run_ml9_category_evaluation.py::_evaluate`/`_streams`/
`_score_modules`/`_fit_policies`/`_jsonable`/`BUDGETS`/`MIN_RECALL`. Butceler:
**critical <=2 FA/saat, advisory <=12 FA/saat** (proje capinda donmus,
degistirilmedi). Uc karar tipi: **threshold, k_of_n, cusum**.

## SS3 Gate tanimlari (SABIT)

- **Gate A (zorunlu, guvenlik):** 200 ucusluk final holdout hicbir asamada
  okunmadi (assert); karar katmanlari/score-fusion/evaluate fonksiyonlari
  identity testiyle "degistirilmeden import edildi" kanitlanir;
  `dense_ae_ml14_fusion`/`dense_ae_itki_fusion`'i uretmek icin yeniden
  hesaplanan `existing_fusion`/`itki_komutu`/`ml14_fusion` ara sutunlari
  donmus `artifacts/ml14/uav_sead/full_matrix` CSV'siyle satir-satir ayni
  cikar (determinism testi). Gecmezse dur.
- **Gate B (operasyonel hedef, ml9/ml10/ml14/Kol L ile AYNI kural):** uc skor
  kaynagindan (`dense_ae_recon`, `dense_ae_ml14_fusion`, `dense_ae_itki_fusion`)
  HERHANGI biri x uc karar tipinden HERHANGI biri x iki butceden birinde
  **critical >=0.30 recall @ <=2 FA/saat** VEYA **advisory >=0.50 recall @
  <=12 FA/saat** saglarsa gecer.

Sonuc gorulduktan sonra pencere/stride/mimari/fusion/butce/karar-tipi listesi
DEGISTIRILMEZ. Gate B gecerse bile tek bir development kosusu blind holdout
acmaya yetmez (proje capinda ayri bir karar); gecmezse de holdout acilmaz.
**Sonuc ne olursa olsun durustce raporlanacak; post-hoc parametre degisikligi
YAPILMAYACAK.**

## SS4 Dosyalar

| Dosya | Is |
|---|---|
| `src/ml/models/dense_autoencoder.py` (yeni) | `DenseAutoencoder`, `train_dense_autoencoder`, `masked_mse`/`reconstruction_scores` import |
| `scripts/run_ml_dense_ae_sead_evaluation.py` (yeni) | SS2 protokolu; `--splits split_00` smoke destegi |
| `artifacts/ml_dense_ae_sead/uav_sead/<run>/` | `metrics.csv`, `flight_label_metrics.csv`, `category_metrics.csv`, `gates.json`, checksum'li `manifest.json`, split-basi model checkpoint, training_log referansi |
| `tests/test_dense_ae_sead_integration.py` (yeni) | veri sizintisi yok, determinism, shape/index hizalamasi, mimari/parametre-sayisi sanity, Gate A identity |
| `docs/ML_YETERSIZLIKLER_KAYDI.md` | gercek sonucla guncellenir |
| `docs/decisions.md` | ADR-017 eklenir (ADR-016 paralel Kol L calismasi tarafindan rezerve olabilir -- gerekirse insan yeniden numaralandirir) |

Kabul: `--splits split_00` smoke -> tam 5-split kosu; tam `pytest -q` yesil
(bilinen 4 MinIO haric).
