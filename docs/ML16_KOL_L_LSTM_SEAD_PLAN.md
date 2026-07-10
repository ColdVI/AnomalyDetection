# ML-16 Kol L: SEAD LSTM-AE Yeniden Eğitimi ve Resmi Değerlendirme Hattına Kablolanması

Durum: UYGULANDI (2026-07-09; plan sonuç görülmeden yazılıp sabitlendi, sonra koşuldu).
Sonuç: Gate A GEÇTİ (determinizm max_abs_diff=7.1e-15), Gate B KALDI (hiçbir hücre
operasyonel hedefi karşılamadı). AYRICA: koşum sonrası bir çapraz-model tutarlılık kontrolü,
reconstruction skorlarının büyük ölçüde ham `RobustScaler` genlik-aykırı-değerine hâkim
olduğunu (öğrenilmiş zamansal örüntüye değil) ortaya çıkardı — bkz. `docs/decisions.md`
ADR-016, `docs/ML_YETERSIZLIKLER_KAYDI.md` B.5,
`artifacts/ml_lstm_sead/uav_sead/full_matrix/magnitude_domination_diagnostic.json`.
Artifact: `artifacts/ml_lstm_sead/uav_sead/full_matrix/`.
Üst plan: `docs/ML16_OZ_KOSULLU_DEDEKTOR_PLAN.md` (Kol F/Chronos genişletme, Kol T/TabFM
pilotu) — bu doküman AYNI ML-16 fazının üçüncü, bağımsız kolu: **Kol L (LSTM-AE)**. Diğer
iki koldan farklı olarak Kol L, ML-15'in kayma-düzeltmeli kalibrasyon sarmalayıcısına
BAĞIMLI DEĞİL — kendi Gate A/B tanımıyla, ml14/ml15 hattının değişmeyen fusion/karar
kodunu doğrudan kullanır (bkz. §2/§3). **ÖNEMLİ isim ayrımı:** "ML-17" bu depoda ZATEN
farklı, spesifik bir anlama sahip — blind holdout'un kullanıcı onayıyla bir kez açıldığı
endgame fazı (`docs/ML14_MASTER_IYILESTIRME_PLANI.md`, `docs/ML15_KALIBRASYON_PLAN.md`,
`docs/RFLY1_ETIKET_DUZELTME_VE_SIMULASYON_PLANI.md`). Bu faz o değildir; holdout burada da
AÇILMAZ (bkz. §3 Gate A). Kapatılan kayıt: `docs/ML_YETERSIZLIKLER_KAYDI.md` C.1 son cümlesi
— "ML-8A/başka model ailesi (planlanmadı)" artık planlanıp koşuluyor.

## §0 Gerekçe ve kapsam

`src/ml/models/lstm_autoencoder.py` (LSTMAutoencoder, maskeli MSE) ALFA'da ROC 0.918 ile
IF'i (0.832) geçen, projenin çalışan ikinci model ailesi. SEAD için `AE_FEATURES["uav_sead"]`
ML-8A sırasında BİR KEZ, yan karşılaştırma olarak eğitilmişti
(`scripts/package_ml8a_sead_lstm.py`, `artifacts/models/uav_sead/ml8a_retrained_lstm_ae/`) ve
window-AUPRC 0.395 ile IF-füzyonu (0.385) hafifçe geçmişti (H17) — ama bu sayı resmi
`ml14`/`ml15` karar-katmanı+FA-bütçesi hattından hiç geçmedi ve ML-14 veri
zenginleştirmesinden (SEAD normal uçuş sayısı 324→899, `split_manifest.json` değişti) ÖNCEYE
ait. Bu faz LSTM-AE'yi GÜNCEL split'lerde yeniden eğitip aynı resmi hatta (aynı fusion/karar
kodu, aynı bütçe, aynı hedef) sokuyor.

**Mevcut en iyi (ml14, 5-seed, development, `artifacts/ml14/uav_sead/full_matrix/metrics.csv`
üzerinden doğrulandı — bu sayılar zaten hesaplanmış, ÖNCEDEN VAR, bu fazın kendi sonucu
değil):**

| Skor | Karar/Bütçe | Recall | FA/saat |
|---|---|---|---|
| `ml14_fusion` | CUSUM / advisory | 0.1257 | 9.95 |
| `ml14_fusion` | CUSUM / critical | 0.0433 | 1.60 |
| `ml14_fusion` | K-of-N / advisory | 0.0560 | 2.28 |
| `existing_fusion` | CUSUM / advisory | 0.1186 | 9.86 |
| `itki_komutu` (genel, kategori-dışı) | CUSUM / advisory | 0.1445 | 35.67 (bütçe aşıyor) |

Hedefler değişmiyor: **critical ≥0.30 recall @ ≤2 FA/saat**, **advisory ≥0.50 recall @ ≤12
FA/saat** (proje çapında donmuş).

## §1 Değerlendirilecek skor varyantları (SABİT — üç ve yalnızca üç)

1. **(a) `lstm_recon`** — SEAD-üzerinde-yeniden-eğitilmiş LSTM-AE'nin maskeli-MSE
   reconstruction skorunun tek başına, val-normal ampirik CDF'ine kalibre edilmiş hâli.
2. **(b) `lstm_ml14_fusion` = max(lstm_recon, ml14_fusion)** — mevcut en iyi genel füzyonla
   max-birleşim.
3. **(c) `lstm_itki_fusion` = max(lstm_recon, itki_komutu)** — mevcut en iyi kategori-uzmanı
   ince modülle max-birleşim.

Başka hiçbir ad hoc varyant (farklı fusion kombinasyonu, farklı kalibrasyon, farklı mimari,
farklı pencere/stride) değerlendirilmeyecek. `existing_fusion`/`ml14_fusion`/`itki_komutu`'nun
KENDİ satırları bu koşuda yeniden raporlanmaz (zaten yukarıda donmuş `artifacts/ml14` CSV'sinden
alınıyor); bu koşuda yalnızca ara sütun olarak (a/b/c'yi üretmek için) yeniden hesaplanırlar —
aynı kod (`fit_modular_iforest`, `_score_modules`, `max_score_fusion`) ile, bu yüzden
sayısal olarak donmuş ml14 koşusuyla örtüşmesi beklenir (determinism testiyle doğrulanır).

## §2 Model, pencereleme ve karar-döngüsü hizalama protokolü

**Mimari/eğitim (DEĞİŞTİRİLMEDEN yeniden kullanılır):** `LSTMAutoencoder`,
`train_lstm_autoencoder`, `masked_mse`, `reconstruction_scores` —
`src/ml/models/lstm_autoencoder.py`'den olduğu gibi import edilir. `AE_FEATURES["uav_sead"]`
(22 sütun), `WINDOW["uav_sead"]=50`, `STRIDE["uav_sead"]=5` sabit kalır (H17'deki mimariyle
aynı — yalnızca veri/split güncel). Pencereleme `src/ml/data/windowing.py::build_windows`
ile üretilir, yeni pencereleme kodu YAZILMAZ.

**Split protokolü:** `data/gold/ml_features/split_manifest.json`'daki GÜNCEL
`sources.uav_sead.splits` (split_00..split_04, aynı `ml14`/`ml15` 5-seed protokolü). Her split
için: aynı split'in `fit_scaler_params`/`apply_scaler_params` (train-only RobustScaler, IF
modülleriyle PAYLAŞILAN tek scaler — ayrı bir LSTM-özel scaler icat edilmez) kullanılarak
ölçeklenen tabloda, `AE_FEATURES["uav_sead"]` sütunları NaN'ları koruyarak (`raw[col].notna()`
maskesiyle) ayrılır ve `build_windows(..., window=50, stride=5, max_gap_s=2.0)` ile pencerelenir.
Eğitim yalnızca split'in `train` (normal) uçuşlarından, erken-durdurma `val` (normal)
uçuşlarından; `test` uçuşları eğitime hiç girmez. Her split kendi modelini eğitir (5 ayrı LSTM).

**Kapsam denetimi (ön-kayıt, sonuç görülmeden ölçüldü):** development satır/uçuş doluluğu
`gps_step_m`/`vertical_rate_calc`/`roll_rate` vb. hareket sütunlarında ≥99.7%, ama
`course_change_deg`, `jamming_indicator`, `noise_per_ms`, `hdop`, `vdop`, `satellites_used`,
`s_variance_m_s` sütunları satır bazında yalnız **%6.0**, uçuş bazında yalnız **%11.9**
dolu (GPS-sağlığı topic'i SEAD loglarının çoğunda hiç yok — `docs/ML_YETERSIZLIKLER_KAYDI.md`
B.1/B.2 ile aynı yapısal eksiklik ailesi). Bu sütunlar SESSİZCE cross-flight/cross-source
medyanla DOLDURULMAYACAK: `build_windows`'ın maske kanalı (NaN→0 + `is_missing` maskesi) ve
`masked_mse` bunu zaten doğru biçimde ele alıyor (kayıp kanal ne "doğru tahmin edildi" diye
ödüllendiriliyor ne de hayalet istatistikle dolduruluyor). Sonuç yorumlanırken bu yapısal
kısıtlama açıkça belirtilecek.

**Pencere-skorundan karar-anına (1 Hz) hizalama — mevcut kod-tabanındaki YERLEŞİK
konvansiyonun tekrar kullanımı, yeni bir hizalama şeması İCAT EDİLMEZ:**
`scripts/run_ml8a_temporal_boosting.py::_align_score` (LSTM pencere-sonu skorunu
`pd.merge_asof(..., direction="backward")` ile nedensel olarak "en son tamamlanmış pencere"
değerine taşıyan, bu depoda LSTM için ZATEN kullanılan fonksiyon) DOĞRUDAN import edilip
kullanılır. Alternatif/daha genel emsal olarak ML-10 Chronos residual kanalı incelendi
(`scripts/build_ml10_forecast_residual.py`): o da "karar-anı = kovanın son satırı" + yalnız-
geçmiş bağlam ile aynı nedensel-taşıma etkisini üretiyor, ama kendi pencereleme yerine
doğrudan karar-indeksi üzerinde çalışıyor. LSTM için zaten var olan `_align_score` yolu
seçildi çünkü (i) bu tam olarak LSTM pencere-skoru için yazılmış emsal, (ii) `build_windows`'ı
değiştirmeden pencere sonu zaman damgasını (`t_end`) doğrudan kullanıyor. Sonuç: her ham
satırda (kaynak, t_rel_s) için "en son tamamlanmış pencerenin reconstruction skoru" (ilk
pencere tamamlanmadan önceki satırlar NaN — nedensel, gelecek sızıntısı yok). Bu ham skor
`src/ml/evaluation/score_fusion.py::empirical_probability` ile split'in val-normal referansına
kalibre edilir (diğer tüm skor kaynaklarıyla aynı kalibrasyon fonksiyonu), sonra
`last_causal_per_bucket` ile ml14/ml15 ile AYNI 1 saniyelik karar kovasına indirgenir ve
modül skorlarıyla aynı (`source_id`,`t_rel_s`) satır ızgarasında birleştirilir.

**Karar katmanları/bütçe/fusion — DEĞİŞTİRİLMEDEN import edilir:**
`src/ml/decision/decision_layers.py` (Threshold/K-of-N/CUSUM, ml9/ml14 ile birebir aynı
`_fit_policies`), `src/ml/evaluation/score_fusion.py::max_score_fusion`/
`last_causal_per_bucket`, `scripts/run_ml9_category_evaluation.py::_evaluate`/`_streams`/
`_score_modules`/`_jsonable`. Bütçeler: **critical ≤2 FA/saat, advisory ≤12 FA/saat** (proje
çapında donmuş, değiştirilmedi). Üç karar tipi: **threshold, k_of_n, cusum** (ml14/ml15 ile
birebir aynı liste, başka karar tipi eklenmez).

## §3 Gate tanımları (SABİT)

- **Gate A (zorunlu, güvenlik):** 131+69=200 uçuşluk final holdout hiçbir aşamada
  okunmadı (assert); karar katmanları/score-fusion/evaluate fonksiyonları identity testiyle
  "değiştirilmeden import edildi" kanıtlanır; `lstm_ml14_fusion`/`lstm_itki_fusion`'ı üretmek
  için yeniden hesaplanan `existing_fusion`/`itki_komutu`/`ml14_fusion` ara sütunları donmuş
  `artifacts/ml14/uav_sead/full_matrix` CSV'siyle satır-satır aynı çıkar (determinism testi).
  Geçmezse dur.
- **Gate B (operasyonel hedef, ml9/ml10/ml14 ile AYNI kural):** üç skor kaynağından
  (`lstm_recon`, `lstm_ml14_fusion`, `lstm_itki_fusion`) HERHANGİ biri × üç karar tipinden
  HERHANGİ biri × iki bütçeden birinde **critical ≥0.30 recall @ ≤2 FA/saat** VEYA
  **advisory ≥0.50 recall @ ≤12 FA/saat** sağlarsa geçer.

Sonuç görüldükten sonra pencere/stride/mimari/fusion/bütçe/karar-tipi listesi
DEĞİŞTİRİLMEZ. Gate B geçerse bile, tek bir development koşusu blind holdout açmaya yetmez
(proje çapında ayrı bir karar); geçmezse de holdout açılmaz. **Sonuç ne olursa olsun — mevcut
en iyiyi geçse de geçmese de — dürüstçe raporlanacak; ADR-012/013 emsalindeki gibi post-hoc
parametre değişikliği YAPILMAYACAK.**

## §4 Dosyalar

| Dosya | İş |
|---|---|
| `scripts/run_ml_lstm_sead_evaluation.py` (yeni) | §2 protokolü; `--splits split_00` smoke desteği |
| `artifacts/ml_lstm_sead/uav_sead/<run>/` | `metrics.csv`, `flight_label_metrics.csv`, `category_metrics.csv`, `gates.json`, checksum'lı `manifest.json`, split-başı `models/lstm_ae/` (state_dict), `training_log` referansı |
| `tests/test_lstm_sead_integration.py` (yeni) | veri sızıntısı yok (train-only fit), determinism (aynı seed→aynı skor), shape/index hizalaması (fusion'dan önce), Gate A identity testleri |
| `docs/ML_YETERSIZLIKLER_KAYDI.md` C.1 | gerçek sonuçla güncellenir |
| `docs/decisions.md` | ADR-016 eklenir |

Kabul: `--splits split_00` smoke → tam 5-split koşu; tam `pytest -q` yeşil (bilinen 4 MinIO
hariç).
