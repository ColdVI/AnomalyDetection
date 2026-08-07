# ML-16 Kol U: SEAD USAD (Adversarial Twin-AE) Degerlendirmesi

Durum: ON-KAYIT (2026-07-09, sonuclar gorulmeden sabitlendi, sonra kosulacak).
Ust plan: `docs/ML14_MASTER_IYILESTIRME_PLANI.md`. Kardes dokuman:
`docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md` (Kol D, Dense AE). `docs/ML16_KOL_L_LSTM_SEAD_PLAN.md`
(Kol L, LSTM-AE) paralel calisan baska bir ajanin isi -- bu faz ONA
DOKUNMUYOR, yalniz referans/karsilastirma noktasi. **ONEMLI isim ayrimi:**
"ML-17" bu depoda ZATEN farkli, spesifik bir anlama sahip -- blind holdout'un
kullanici onayiyla bir kez acildigi endgame fazi. Bu faz o degildir; holdout
burada da ACILMAZ (SS3 Gate A).

## SS0 Gerekce ve durustce beklenti

`docs/ML1_BULGULAR_VE_HATALAR.md` ML-3 bolumu (2026-07-02,
`notebooks/04_ablation_enjeksiyon_usad.ipynb`): USAD ALFA'da **0.450**, UAV
Attack'ta **0.531** ucus-ROC -- ikisi de o zamanki LSTM-AE'nin (ALFA 0.731)
altinda. Kayitli yorum: "az-veri rejiminde adversarial egitim kararsiz
(beklenen)". USAD hicbir zaman SEAD icin egitilmedi ve hicbir zaman bu
depodaki yeniden-kullanilabilir `src/ml/` modulu olarak paketlenmedi.

ML-14 veri zenginlestirmesi SEAD normal-ucus egitim havuzunu 324'ten 899'a
cikardi. USAD'in onceki kaybi icin verilen gerekce ("az veri -> adversarial
egitim kararsizligi") Dense AE'nin gerekcesinden (mimari/zamansal-baglam
kaybi) NITELIKSEL OLARAK FARKLI -- bu, USAD'in daha fazla veriyle gercekten
duzelebilecegini iddia eden, dogasi geregi daha belirsiz bir hipotez.

**Durust on-kayitli beklenti (sonuc gorulmeden yazildi):** USAD, Dense AE'nin
aksine, "zamansal modellemede yapisal olarak dezavantajli" degil -- ayni
flatten-pencere girdisini kullaniyor (bu faz de Dense AE ile ayni flatten
yaklasimini secti, SS2), asil fark egitim rejiminde (adversarial iki-fazli
kayip). Dolayisiyla bu kolun sonucu Dense AE'ninkinden daha az tahmin
edilebilir: onceki kayip gercekten veri-hacmi artifaktiyse 899 ucusla
duzelebilir; degilse (adversarial egitimin kendi ic dinamigi -- ornegin
encoder'in iki decoder arasinda salinip kararli bir denge bulamamasi -- daha
fazla veriyle otomatik cozulmeyen bir ozellikse) yine kaybedebilir. **Her iki
sonuc da bu on-kayitta acikca kabul edilebilir bilgilendirici sonuclardir**
("USAD da yine kaybetti, veri hacmi tek basina yeterli degilmis" DAHIL).
Sonuc gorulduktan sonra hicbir post-hoc parametre degisikligi yapilmayacak.

## SS1 Degerlendirilecek skor varyantlari (SABIT -- uc ve yalnizca uc)

1. **(a) `usad_score`** -- SEAD-normal-ucuslarinda-egitilmis USAD'in paper'in
   standart alpha=0.5/beta=0.5 kombinasyon skorunun (SS2), val-normal ampirik
   CDF'ine kalibre edilmis hali.
2. **(b) `usad_ml14_fusion` = max(usad_score, ml14_fusion)** -- mevcut en iyi
   genel fuzyonla max-birlesim.
3. **(c) `usad_itki_fusion` = max(usad_score, itki_komutu)** -- mevcut en iyi
   kategori-uzmani ince modulle max-birlesim.

Baska hicbir ad hoc varyant degerlendirilmeyecek (ozellikle: alpha/beta
paper'in 0.5/0.5 varsayilanindan BASKA bir degere sonuc gorulduktan sonra
ayarlanmayacak). `ml14_fusion`/`itki_komutu`'nun KENDI satirlari bu kosuda
yeniden raporlanmaz, yalniz ara sutun olarak yeniden hesaplanirlar (Gate A
determinism testiyle dogrulanir).

## SS2 Model, egitim ve karar-donusu hizalama protokolu

**Mimari (yeni, `src/ml/models/usad.py`):** `USAD` sinifi -- Audibert ve
digerleri (2020), "USAD: UnSupervised Anomaly Detection on Multivariate Time
Series" (KDD 2020) formulasyonu. Paylasilan encoder `E` (flatten edilmis
pencere `window*n_features` -> latent), iki bagimsiz decoder `D1`/`D2`
(latent -> flatten pencere). `AE1(w)=D1(E(w))`, `AE2(w)=D2(E(w))`.

**Egitim (paper'in Bolum 3.3/Algorithm 1'i, kanonik referans
uygulamasinin -- manigalati/usad GitHub, USAD'in fiili referans kodu --
per-batch guncelleme deseniyle: her optimizer adimi icin forward pass
YENIDEN hesaplanir, tek bir grafikte iki kez backward COGRULMAZ):**
epoch `n` (1-indeksli) icin

```
L_AE1 = (1/n) * ||w - AE1(w)||^2      + (1 - 1/n) * ||w - AE2(AE1(w))||^2
L_AE2 = (1/n) * ||w - AE2(w)||^2      - (1 - 1/n) * ||w - AE2(AE1(w))||^2
```

`L_AE1` (encoder+D1'i gunceller) w'yu dogrudan yeniden insa etmeyi VE AE2'yi
kandirmayi (AE2'nin AE1(w)'yi gercekmis gibi iyi insa etmesini) hedefler;
`L_AE2` (encoder+D2'i gunceller) w'yu dogrudan yeniden insa etmeyi VE (eksi
isaretle) AE1'in ciktisini gercekten AYIRT ETMEYI (o rekonstrüksiyonu KOTU
yeniden insa etmeyi) hedefler -- paper'in adversarial ciftinin tam karsiligi.
Erken/gec epoch agirliklandirmasi (alpha=1/n erken epoch'ta yuksek/gec
epoch'ta dusuk, beta=1-1/n tam tersi) paper'in kendi semasi, burada
DEGISTIRILMEDI.

**Erken-durdurma kriteri (bu faza ozgu, dokumante edilen ek tasarim karari --
paper sabit epoch sayisi kullaniyor, bir val kriteri onermiyor):**
`L_AE2`'nin isaretli/adversarial terimi nedeniyle "kucuk=iyi" monoton bir
kalite olcusu degil; bunun yerine iki decoder'in DOGRUDAN (agirliksiz,
adversarial-terimsiz) maskeli-MSE toplami val-normal penceresinde izlenir
(`USAD.validation_reconstruction_quality`) ve en iyi epoch bununla secilir.

**Skor (paper'in Bolum 3.4'u, standart kombinasyon):**
`score = alpha*||w-AE1(w)||^2 + beta*||w-AE2(AE1(w))||^2`, `alpha=beta=0.5`
(paper'in/referans kodun varsayilani, esit agirlik; sonuc gorulduktan sonra
ayarlanmayacak).

**Maskeli girdi isleme:** `masked_mse` `src/ml/models/lstm_autoencoder.py`'den
DEGISTIRILMEDEN import edilir -- USAD'in tum terimleri (L_AE1, L_AE2,
inference skoru) bu tek fonksiyon uzerinden hesaplanir, boylece Dense AE ve
LSTM-AE ile birebir ayni maskeli-kayip semantigi (eksik kanal ne "dogru
tahmin edildi" diye odullenir ne de baska ucus/kaynagin istatistigiyle
DOLDURULUR -- docs/ML_YETERSIZLIKLER_KAYDI.md B.1/B.2 ile ayni disiplin).

`AE_FEATURES["uav_sead"]` (22 sutun), `WINDOW["uav_sead"]=50`,
`STRIDE["uav_sead"]=5`, pencereleme (`src/ml/data/windowing.py::build_windows`),
split protokolu (split_00..split_04, train-only scaler/egitim, val-only erken
durdurma+kalibrasyon, test hic egitime girmez), pencere-skorundan karar-anina
hizalama (`scripts/run_ml8a_temporal_boosting.py::_align_score`,
`pd.merge_asof(..., direction="backward")`), karar katmanlari/butce/fusion
(threshold/k_of_n/cusum, critical<=2 FA/saat, advisory<=12 FA/saat) -- Kol D
ve Kol L ile BIREBIR AYNI, degistirilmeden import edilir/uygulanir (SS2
detaylari icin `docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md` SS2'ye bakiniz, burada
tekrarlanmaz).

**USAD parametre sayisi:** 25409 (encoder 7739 + decoder1 8835 + decoder2
8835), `DenseAutoencoder`'in 16574'unden ve `LSTMAutoencoder`'in 17414'unden
daha buyuk -- bu USAD'in kendi mimarisinin (paylasilan tek encoder + IKI
bagimsiz decoder) dogal sonucu, kasitli bir kapasite avantaji DEGIL (Dense AE
icin gecerli olan "kabaca esit kapasite" kosulu USAD icin talep edilmedi,
gorev tanimi geregi).

## SS3 Gate tanimlari (SABIT)

- **Gate A (zorunlu, guvenlik):** 200 ucusluk final holdout hicbir asamada
  okunmadi; karar katmanlari/score-fusion/evaluate fonksiyonlari identity
  testiyle "degistirilmeden import edildi" kanitlanir; `usad_ml14_fusion`/
  `usad_itki_fusion`'i uretmek icin yeniden hesaplanan
  `existing_fusion`/`itki_komutu`/`ml14_fusion` ara sutunlari donmus
  `artifacts/ml14/uav_sead/full_matrix` CSV'siyle satir-satir ayni cikar
  (determinism testi). Ayrica USAD'a ozgu bir mimari-saglik testi: faz-2
  adversarial teriminin gradyanlari gercekten degistirdigi (yalniz-faz-1
  egitimden farkli oldugu) birim testle kanitlanir. Gecmezse dur.
- **Gate B (operasyonel hedef, ml9/ml10/ml14/Kol L/Kol D ile AYNI kural):** uc
  skor kaynagindan (`usad_score`, `usad_ml14_fusion`, `usad_itki_fusion`)
  HERHANGI biri x uc karar tipinden HERHANGI biri x iki butceden birinde
  **critical >=0.30 recall @ <=2 FA/saat** VEYA **advisory >=0.50 recall @
  <=12 FA/saat** saglarsa gecer.

Sonuc gorulduktan sonra egitim semasi/alpha-beta/pencere/stride/mimari/
fusion/butce/karar-tipi listesi DEGISTIRILMEZ. Gate B gecerse bile tek bir
development kosusu blind holdout acmaya yetmez; gecmezse de holdout acilmaz.
**Sonuc ne olursa olsun durustce raporlanacak; post-hoc parametre degisikligi
YAPILMAYACAK.**

## SS4 Dosyalar

| Dosya | Is |
|---|---|
| `src/ml/models/usad.py` (yeni) | `USAD`, `train_usad`, `usad_reconstruction_scores`, `masked_mse` import |
| `scripts/run_ml_usad_sead_evaluation.py` (yeni) | SS2 protokolu; `--splits split_00` smoke destegi |
| `artifacts/ml_usad_sead/uav_sead/<run>/` | `metrics.csv`, `flight_label_metrics.csv`, `category_metrics.csv`, `gates.json`, checksum'li `manifest.json`, split-basi model checkpoint, training_log referansi |
| `tests/test_usad_sead_integration.py` (yeni) | veri sizintisi yok, determinism, shape/index hizalamasi, faz-2 adversarial-gradyan sanity, Gate A identity |
| `docs/ML_YETERSIZLIKLER_KAYDI.md` | gercek sonucla guncellenir (C.3'un devami) |
| `docs/decisions.md` | ADR-018 eklenir (ADR-016/017 paralel calismalar tarafindan rezerve olabilir -- gerekirse insan yeniden numaralandirir) |

Kabul: `--splits split_00` smoke -> tam 5-split kosu; tam `pytest -q` yesil
(bilinen 4 MinIO haric).
