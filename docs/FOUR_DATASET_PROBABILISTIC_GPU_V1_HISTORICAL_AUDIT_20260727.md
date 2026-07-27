# Four-Dataset Probabilistic GPU v1 — tarihsel denetim

Tarih: 2026-07-27  
Kapsam: RflyMAD, UAV-SEAD, ALFA ve UAV-Attack  
Durum: salt-okunur tarihsel denetim; bu belge yeni bir model sonucu veya çalışma izni değildir.

## 1. Yönetici özeti

Kesin cevap şudur:

| Veri kümesi | Geçmişte yerel `mu/sigma + Gaussian NLL` var mı? | Yeni Gaussian forecaster'ın yeniliği |
|---|---|---|
| RflyMAD | **Evet.** Dar GNSS bütünlük pilotunda LSTM iki başlıkla konum ve ölçek üretmiş, maskeli/ağırlıklı Gaussian NLL ile eğitilmiştir. | Kayıp ve temel probabilistik fikir yeni değildir. Ancak tüm RflyFull kapsamına, başka hedeflere veya yeni fizik bağlamına uygulanması yeni bir temsil deneyi olabilir. |
| UAV-SEAD | **Hayır.** Yerel eğitimler maskeli rekonstrüksiyon MSE'dir. Chronos forecast-residual denenmiştir fakat zero-shot'tır; fine-tuning/optimizer yoktur. | Yerel heteroscedastic Gaussian forecaster yenidir. Dört veri kümesi içinde veri ölçeği ve korunan blind holdout nedeniyle en güçlü adaydır. |
| ALFA | **Hayır.** Dense/LSTM-AE ve USAD maskeli MSE kullanmıştır. Sonraki residual çalışmasında öğrenilmiş forecaster oturum sızıntısı nedeniyle hiç eğitilmemiştir. | Mimari ve loss yenidir; sonuç ancak exploratory sayılabilir. Bağımsız normal oturum ve kalibrasyon maruziyeti açığını çözmez. |
| UAV-Attack | **Hayır.** Dense/LSTM-AE ve USAD maskeli MSE kullanmıştır. | Mimari ve loss yenidir; fakat 6 normal log, tek jamming örneği ve gözlenemeyen Ping DoS vakaları nedeniyle exploratory kalır. |

Tarihsel kaynakların `.py`, `.md`, `.ipynb` ve `.json` dosyalarında yapılan exact-formül/terim taramasında `scale_head` ve `log(safe_scale) + 0.5*((y-mu)/safe_scale)^2` uygulaması yalnız `gecmis_calismalar/anomaly_core/forecaster.py:30-53,56-100` içinde bulundu. Dolayısıyla “dört veri kümesinde de daha önce Gaussian NLL yaptık” denemez.

Metrik seviyeleri birbirine çevrilmemelidir: ALFA/UAV-Attack eski sayıları çoğunlukla **uçuş ROC-AUC**, RflyMAD/UAV-SEAD son çalışmalarının sayıları ise **event recall + false alarm/saat** değerleridir. Aşağıdaki sonuçlar tarihsel durum göstermek içindir; veri kümeleri arasında skor sıralaması değildir.

## 2. Veri ölçeği ve değerlendirme sözleşmeleri

| Veri kümesi | Güncel denetlenen ölçek | Bağımsızlık/split durumu | Sonuç statüsü |
|---|---|---|---|
| RflyMAD Full v2 | 6.605 canonical uçuş; 1.225 locked-test. Real bölümünde yalnız 51 NoFault uçuşu vardır. Dar GNSS pilot fit'i 20 temiz uçuş, 4.044 satır ve 3.671 penceredir. | Full v2 test'i domain/family/environment katmanlarında grouped %20 locked-test; development deterministic 5-fold'dur. | Locked test açılmadan development sonuçları; mevcut temsil Real transferini göstermedi. |
| UAV-SEAD | 1.244 uçuş; güncel Gold 1.664.490 satır x 147 kolon, 655.990.007 bayt. | Split birimi session; 1.044 development, bunun 899'u normal; 200 uçuş blind-final-holdout. | Blind holdout açılmadı; raporlanan sonuçlar development'tır. |
| ALFA | ML-feature hattında 54 uçuş/15 normal; Gold 40.364 satır x 88 kolon, 18.784.062 bayt. | Source-id split; `split_00` 11 normal train, 2 normal validation ve 40 test uçuşu içerir; final holdout yok, `development-only`. Sonraki RESIDUAL-V1 sözleşmesi daha dar 47 resmî uçuş/11 normal corpus kullanmıştır. | Küçük-n ve oturum bağımlılığı nedeniyle exploratory. |
| UAV-Attack | 19 log: 6 normal, 6 Ping DoS, 6 GPS spoofing, 1 GPS jamming; Gold 79.646 satır x 61 kolon, 14.090.281 bayt. | Source-id split; örneğin `split_00` 4 normal train, 1 normal validation, 1 normal + 13 anomaly test; final holdout yok. | Manifest açıkça `development-only; anomaly data too scarce` der. |

Kanıtlar:

- RflyMAD ölçeği ve sözleşmesi: `artifacts/rfly_full/v2/dataset_manifest_summary.json:4-11,13-98,100-108`; GNSS pilot fit ölçeği: `artifacts/uav_gnss_integrity_v1/fit_result.json:2-6`.
- UAV-SEAD uçuş/split statüsü: `data/gold/ml_features/split_manifest.json:2327-2331`; güncel development/holdout sayıları: `gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/decisions.md:466-480`.
- ALFA 54-uçuş manifesti ve statüsü: `data/gold/ml_features/split_manifest.json:7-10`; `split_00` train/validation listeleri: aynı dosya `:67-87`; 54/15 büyütme kaydı: `gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML1_BULGULAR_VE_HATALAR.md:159-167`.
- UAV-Attack sayıları, etiketleri ve statüsü: `data/gold/ml_features/split_manifest.json:1792-1816`; `split_00` ve boş final holdout: aynı dosya `:1817-1881`.
- Parquet satır/kolon/bayt değerleri 2026-07-27'de ilgili binary dosyaların footer'ı ve dosya metadata'sı salt-okunur okunarak doğrulandı: `data/gold/ml_features/alfa/alfa_ml_features.parquet`, `data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet`, `data/gold/ml_features/uav_attack/uav_attack_ml_features.parquet`. Binary dosyalarda satır numarası yoktur. UAV-SEAD satır sayısı ayrıca `gecmis_calismalar/UAV_SEAD/egitilmis_modeller/ml14/uav_sead/rebuild_report.json:40-41` ile doğrulanır.

## 3. RflyMAD

### 3.1 Geçmiş modeller, loss ve epoch

RflyMAD için Gaussian forecaster daha önce gerçekten uygulanmıştır. `ResidualForecaster`, maskeyi girişe ekleyen bir LSTM'den iki ayrı başlıkla `location` ve sınırlandırılmış `scale` üretir (`gecmis_calismalar/anomaly_core/forecaster.py:30-53`). Eğitim yalnız `natural_clean_fit` rolündeki sentetiksiz veriye izin verir; loss:

`log(scale) + 0.5 * ((y - location) / scale)^2`

şeklindeki kanal-ağırlıklı, hedef-maskeli Gaussian NLL'dir (`gecmis_calismalar/anomaly_core/forecaster.py:56-100`). Pilot config'i 8 epoch, batch 256, Adam `lr=1e-3`, hidden 32 ve tek LSTM katmanıdır (`configs/uav_gnss_integrity_v1.json:27-43`). Loss 8 epochta `1.0509 -> 0.4130` düşmüştür (`artifacts/uav_gnss_integrity_v1/fit_result.json:85-108`).

RflyFull v2'nin daha geniş deneyleri Gaussian değildir:

- Normal temporal convolutional AE, maskeli rekonstrüksiyon MSE ve AdamW kullanır (`gecmis_calismalar/rfly_full/normal_ae.py:137-198`). R4 beş rotasyonda 13/792/229/13/640 epoch tamamlamış; en iyi epoch'lar 1/780/217/1/628 olmuştur (`artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/final_summary.json:122-186`).
- Supervised TCN, binary BCE-with-logits ile fault-family weighted cross-entropy toplamını kullanır (`gecmis_calismalar/rfly_full/supervised.py:340-380`). Beş fold 12 epoch cap ile çalışmış, en iyi epoch'lar 5/5/3/2/2'dir (`artifacts/rfly_full/v2/supervised_tcn/development_5fold_20260722_v1/summary.json:13-91`).
- Eski direct-DL hattı Dense-AE, LSTM-AE ve USAD'ı 40 epoch cap/patience 5 ile çalıştırmıştır (`gecmis_calismalar/rfly_dl/config.py:53-56`). Bu hat masked-MSE ailesidir; Gaussian değildir.

### 3.2 GPU durumu

GNSS Gaussian pilotu ve RflyFull v2 AE/TCN akışları CPU tensorleri oluşturur; device/CUDA taşıması yapmaz (`gecmis_calismalar/anomaly_core/forecaster.py:75-100`, `gecmis_calismalar/rfly_full/normal_ae.py:142-185`, `gecmis_calismalar/rfly_full/supervised.py:340-377`).

Eski direct-DL hattı ise teknik olarak CUDA-ready'dir: model ve bütün tensorler verilen `device`'a taşınır (`gecmis_calismalar/rfly_dl/models.py:151-173`), CLI `--device` kabul eder fakat varsayılanı `cpu`dur (`scripts/RFLYMAD_rfly_full_v2/run_rfly_dl_evaluation.py:20-37`). Dolayısıyla “RflyMAD'de hiç GPU desteği yoktu” da doğru değildir; doğru ifade, **Gaussian pilotun ve son Full-v2 ana hatlarının GPU kullanmadığı, eski direct-DL altyapısının GPU'ya geçirilebilir olduğu**dur.

### 3.3 Sonuç ve sınır

- GNSS Gaussian LSTM, development critical noktada %47,1 recall ve 20,32 FA/saat; advisory'de %82,4 recall ve 101,60 FA/saat üretmiş, iki kapıyı da geçememiştir (`gecmis_calismalar/RFLYMAD/raporlar/uav_gnss_integrity_v1_final_no_go_report.tex:94-99,119-129`). Trained-random Spearman 0,678 ve trained-magnitude 0,690 sınırı geçse de operasyonel sonuç `NO-GO`dur.
- Direct-DL'nin üç mimarisi de operasyonel kapıda kalmış; 15/15 model-split magnitude audit tarafından işaretlenmiştir (`artifacts/rfly_dl/direct_v1_5split_20260720/summary.json:1-50`).
- Full-v2 TCN hiçbir critical/advisory/Real/Wind kapısını geçmemiştir (`artifacts/rfly_full/v2/supervised_tcn/development_5fold_20260722_v1/summary.json:95-137`). Uzun R4 eğitimi de mevcut temsilin Real transferini göstermemiştir (`artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/final_summary.json:191-205`).
- Tarihsel sözleşme yeni epoch/LR/threshold kombinasyonu avını açıkça yasaklar; kök neden temsil/aşırı uyum ve bağımsız Real-NoFault session azlığıdır (`gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_SONRAKI_ADIMLAR_20260722.md:42-49,73-92,107-118`). Yeni çalışma ancak yeni, yazılı bir temsil sözleşmesi olarak açılmalıdır.

**Yenilik kararı:** Gaussian NLL açısından yeni değil. Yeni v1 ancak hedefler, fizik bağlamı, split sözleşmesi ve magnitude/random kontrolleri bakımından yeni bir deney olarak savunulabilir.

## 4. UAV-SEAD

### 4.1 Geçmiş modeller, loss ve epoch

SEAD'de modüler Isolation Forest/CUSUM, supervised LightGBM, zero-shot Chronos ve üç derin rekonstrüksiyon ailesi denenmiştir. Yeni Gaussian forecaster'a en yakın tarihsel yöntem Chronos'tur; fakat `amazon/chronos-bolt-tiny` CPU'da zero-shot çalışmış, fine-tuning/gradient/optimizer adımı uygulanmamıştır (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/decisions.md:268-284`). Bu nedenle yerel Gaussian-NLL eğitiminin öncülü sayılmaz.

ML-16'da güncel 5-seed SEAD splitlerinde eğitilen modeller:

- LSTM-AE: 22 feature, window 50/stride 5, hidden 32/latent 16; maskeli rekonstrüksiyon MSE, Adam, 40 epoch cap, patience 5 (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/src_ml/models/lstm_autoencoder.py:12-63,93-131`). Gerçek beş split 7/26/35/9/8 epochta durmuştur; örneğin split01'in son kaydı epoch 26'dır (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/artifacts/training_logs/uav_sead/ml16_kol_l_lstm_sead/20260709T164537Z_split_01/loss.csv:27`).
- Dense-AE: aynı maskeli MSE ve aynı epoch/batch/patience sözleşmesi (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/src_ml/models/dense_autoencoder.py:33-106`). Beş split 7/21/16/6/6 epochta durmuştur; split01 son kaydı epoch 21'dir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/artifacts/training_logs/uav_sead/ml16_kol_d_dense_ae_sead/20260709T170703Z_split_01/loss.csv:22`).
- USAD: ortak encoder + iki decoder, iki optimizer'lı adversarial eğitim; bütün terimler yine maskeli MSE'dir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/src_ml/models/usad.py:18-35,96-180`). Beş split 6/11/8/8/8 epochta durmuştur; split01 son kaydı epoch 11'dir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/artifacts/training_logs/uav_sead/ml16_kol_u_usad_sead/20260709T170713Z_split_01/loss.csv:12`).

### 4.2 GPU durumu

Bu üç tarihsel SEAD trainer'ı model/tensorleri doğrudan CPU'da kurar ve skorları doğrudan `.numpy()` ile çıkarır; `device`, `.to(device)`, CUDA veya AMP yolu yoktur (`lstm_autoencoder.py:93-141`, `dense_autoencoder.py:67-106`, `usad.py:127-180,185-201`, aynı tarihsel dizin). Chronos deneyi de açıkça CPU'dur (`decisions.md:273-278`, aynı tarihsel dizin).

Bu veri 1,66 milyon satırla dört küme içinde yeni GPU eğitiminin en anlamlı olduğu adaydır. Yine de epoch sayısı değil 5 session splitinin tamamı, kalibrasyon ve kör holdout izolasyonu deney birimidir.

### 4.3 Sonuç ve sınır

- Üç derin modelin en iyi ham `threshold/critical` noktaları sırasıyla LSTM `0,219 recall / 2,92 FA-saat`, Dense `0,215 / 2,77`, USAD `0,217 / 2,87` olmuş ve hiçbir model Gate B'yi geçmemiştir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/decisions.md:483-513,567-600,626-657`).
- Kök neden yalnız mimari zayıflığı değildir: trained-LSTM skor sırası random-init eşine `rho=0,964`, model-free ham büyüklüğe `rho=0,965` koreledir. Kırpmasız scaler ve sentinel/aşırı-genlik pencereleri üç mimarinin skoruna hakim olmuştur (`decisions.md:515-538`, aynı tarihsel dizin). Aynı ölçeklemeyle üç modeli tekrar koşmanın bilgi kazandırmayacağı tarihsel karara yazılmıştır (`decisions.md:664-669`).
- Chronos mechanical kategori kolunda Gate B'yi geçmiştir (`0,205 -> 0,390`, 4/5 seed), fakat bütün-system fusion `0,213 recall / 23,92 FA-saat` ile Gate C'de kalmıştır (`decisions.md:286-295`, aynı tarihsel dizin).
- Topic/feature kapsama sınırları sürer: örneğin bazı irtifa residual kanalları anomaly uçuşlarında %0 veya yaklaşık %7 doludur (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML_YETERSIZLIKLER_KAYDI.md:99-111`). Gaussian başlık mevcut olmayan topic'i oluşturamaz.

**Yenilik kararı:** Yerel, mask-aware heteroscedastic next-step Gaussian forecaster yenidir. Çalışma development-only yürütülmeli; 200-uçuş blind holdout Gate geçmeden açılmamalı ve trained-vs-random, trained-vs-magnitude testleri ön-kayıtlı zorunlu kapı olmalıdır.

## 5. ALFA

### 5.1 Geçmiş modeller, loss ve epoch

ALFA'da monolitik IF, modüler IF+CUSUM, Dense-AE, LSTM-AE, USAD ve daha sonra fiziksel residual/ridge planı denenmiştir:

- İlk IF satır ROC'u `0,497+/-0,026`; modüler füzyon uçuş ROC'u `0,833+/-0,172`, yalnız rehberlik modülü `0,864+/-0,081` olmuştur (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML1_BULGULAR_VE_HATALAR.md:7-31`).
- İlk 5-seed AE turunda Dense-AE 35,8+/-9,4 epoch ve `0,622+/-0,230` uçuş ROC; LSTM-AE 23,4+/-13,5 epoch ve `0,731+/-0,153` uçuş ROC üretmiştir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/notebooks/03_autoencoder_lstm_ae_egitim.ipynb:319-365`). Loss maskeli MSE, cap 40/patience 5'tir.
- Normal uçuşlar 10'dan 15'e çıkarıldığında aynı LSTM-AE `0,918+/-0,104` uçuş ROC'a yükselmiştir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/notebooks/05_veri_buyutme_yeniden_olcum.ipynb:189-210`). Kullanılan trainer maskeli MSE, Adam, 40 epoch cap ve patience 5'tir (`05_veri_buyutme_yeniden_olcum.ipynb:214-258`).
- USAD `0,450` uçuş ROC ile LSTM-AE'nin altında kalmıştır (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML1_BULGULAR_VE_HATALAR.md:148-157`). USAD'ın kaybı da adversarial ağırlıklı maskeli MSE terimleridir; Gaussian değildir.

### 5.2 GPU durumu

ALFA notebookları ve ortak tarihsel modeller tensorleri CPU'da oluşturur; device/CUDA/AMP yolu yoktur (`03_autoencoder_lstm_ae_egitim.ipynb:220-261`; ortak `lstm_autoencoder.py:93-141`). Tarihsel süreler ilk turda seed başına yaklaşık 0,88 s Dense ve 1,32 s LSTM'dir (`03_autoencoder_lstm_ae_egitim.ipynb:339-358`). Eski modeli L4 üzerinde saatlerce epoch döndürmek veri ölçeğine uygun değildir; GPU zamanı fold/seed/loss ablation'ına ayrılmalıdır.

### 5.3 Sonuç ve sınır

Sonraki RESIDUAL-V1 çalışması faz segmentasyonu ve altı fiziksel kanal tanımlamıştır (`gecmis_calismalar/_ortak/raporlar/RESIDUAL_V1.md:144-176`), fakat ALFA R1-R5 normal geliştirme uçuşlarının tamamı tek 2018-07-18 oturumuna ait olduğu için session-CV sızıntısı yaratmadan learned forecaster/ridge modeli eğitilememiştir (`RESIDUAL_V1.md:517-533`, aynı dizin).

Ek yapısal sınırlar:

- Rudder 4, elevator 2, aileron-rudder 1 uçuş; mevcut corpus resmî külliyatı zaten kapsar (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML_YETERSIZLIKLER_KAYDI.md:22-33`).
- RESIDUAL-V1'in korumalı kalibrasyonunda yalnız `0,168846` normal uçuş-saat vardır; 2 saat hedefi için `11,845x` eksiktir. Bootstrap yeni bağımsız maruziyet oluşturamaz (`gecmis_calismalar/_ortak/raporlar/RESIDUAL_V1.md:967-980`). Nihai karar mevcut development-normal exposure ile `NO-GO`dur (`RESIDUAL_V1.md:1009-1023`, aynı dizin).
- ML-feature hattındaki 54/15 ile daha sonraki RESIDUAL-V1'in 47/11 corpus sayıları farklı sözleşmelere aittir; birleştirilerek örnek sayısı şişirilmemelidir (`data/gold/ml_features/split_manifest.json:7-10`; `RESIDUAL_V1.md:982-992`, aynı dizin).

**Yenilik kararı:** Gaussian forecaster yeni olur, fakat istatistiksel iddia yeni değildir. Tek-session normal dinamik, az fault-family örneği ve yetersiz normal maruziyet daha fazla epoch veya learned variance ile çözülemez.

## 6. UAV-Attack

### 6.1 Geçmiş modeller, loss ve epoch

Geçmiş ana yöntemler monolitik IF, modüler IF+CUSUM, Dense-AE, LSTM-AE ve USAD'dır:

- Monolitik IF satır ROC'u `0,209+/-0,138`; modüler füzyon uçuş ROC'u `0,600+/-0,212` olmuştur (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML1_BULGULAR_VE_HATALAR.md:7-31`).
- Dense-AE 5 seed ortalamasında 30,6+/-8,1 epoch, 4,9 s ve `0,677+/-0,167` uçuş ROC; LSTM-AE 24,8+/-16,2 epoch, 13,22 s ve `0,677+/-0,126` uçuş ROC üretmiştir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/notebooks/03_autoencoder_lstm_ae_egitim.ipynb:367-395`). İki modelin kaybı maskeli MSE'dir (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/src_ml/models/lstm_autoencoder.py:59-63,93-131`).
- Paketlenmiş split00 LSTM artefaktı 22 feature, window 50/stride 5 kullanmış ve 6 epochta durmuştur (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/artifacts/models/uav_attack/ml6_lstm_ae/manifest.json:3-49`).
- USAD uçuş ROC'u `0,531` olup LSTM-AE'nin altında kalmıştır (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML1_BULGULAR_VE_HATALAR.md:148-157`). Exact Gaussian/NLL geçmişi yoktur.

### 6.2 GPU durumu

Tarihsel AE kodu CPU-only'dir: model/tensorler device'a taşınmaz ve çıktı doğrudan `.numpy()` yapılır (`lstm_autoencoder.py:93-141`, aynı tarihsel dizin). Split00 yalnız 4 normal train uçuşuna dayandığı için exact eski modelin L4 üzerinde uzun eğitimi GPU kapasitesini değil bağımsız veri azlığını ölçer.

### 6.3 Sonuç ve sınır

- Parser raw ZIP'teki onlarca topic'ten yalnız `vehicle_global_position`, `vehicle_attitude`, `battery_status`, `vehicle_gps_position` topiclerini kullanır (`gecmis_calismalar/UAV_ATTACK/kaynak_kod/parse_uav_attack.py:21-25,65-78`).
- Ping DoS'un 4/6 logunda ağ saldırısı bu dört topic'e gözlenebilir bir iz bırakmaz; inter-arrival/paket metadata'sı için parser kapsamı genişlemeden hiçbir loss bunu çözemez (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML_YETERSIZLIKLER_KAYDI.md:60-66`).
- GPS jamming `n=1`, normal havuz yalnız 6 logdur (`ML_YETERSIZLIKLER_KAYDI.md:68-78`, aynı tarihsel dizin).
- Parser bütün log satırlarına tek uçuş etiketi yazar (`parse_uav_attack.py:186-188`); saldırı başlangıç/bitiş ground truth'u yoktur. Bu yüzden satır/pencere ROC yapısal olarak adaletsizdir ve ana değerlendirme uçuş/log düzeyinde kalmalıdır (`gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML1_BULGULAR_VE_HATALAR.md:7-23,109-125`).

**Yenilik kararı:** Gaussian forecaster loss/mimari olarak yenidir. Gold-only çalışma yalnız exploratory baseline olabilir. Ping DoS hedefi ciddiyse önce raw topic/inter-arrival enrichment yapılmalıdır; aksi halde uzun GPU eğitimi gözlenemeyen sinyali yaratmaz.

## 7. Gaussian forecaster'ın çözemeyeceği ortak yapısal sorunlar

1. **Eksik gözlenebilirlik:** Kaynak topic yoksa veya saldırı seçili kanallara yansımıyorsa `sigma` başlığı bilgi üretemez. UAV-Attack Ping DoS ve SEAD'in seyrek irtifa topicleri bunun açık örnekleridir.
2. **Bağımsız örnek azlığı:** Epoch ve pencere sayısı, bağımsız uçuş/oturum sayısı değildir. ALFA'nın tek-session forecaster engeli, UAV-Attack'ın 6 normal logu ve RflyMAD'in az Real-NoFault session'ı GPU ile kapanmaz.
3. **Etiket granülaritesi:** Uçuş etiketi, event onset etiketi yerine kullanılamaz. Uçuş-, event- ve pencere-düzeyi metrikler ayrı raporlanmalıdır.
4. **Magnitude/sentinel baskınlığı:** Gaussian NLL, rekonstrüksiyon MSE'deki sorunu otomatik çözmez. Model `sigma`yı büyüterek zor rejimleri gizleyebilir veya skor yine ham büyüklüğü sıralayabilir. Trained-vs-random, trained-vs-magnitude, per-channel `mu/sigma`, sigma-clamp ve variance-inflation denetimleri zorunludur.
5. **Domain ve session shift:** Öğrenilmiş belirsizlik, görülmemiş platform/oturum dağılımı için kalibre belirsizlik garantisi değildir. Split ve scaler session/flight izolasyonuna uymalıdır.
6. **Kalibrasyon maruziyeti:** Düşük FA/saat hedefi yeterli normal uçuş-saati olmadan ölçülemez. Bootstrap güven aralığı üretir; yeni maruziyet üretmez.
7. **Blind holdout yokluğu veya korunması:** UAV-SEAD'in 200 uçuşu ve RflyMAD locked test'i development kapıları geçmeden açılmamalıdır. ALFA ve UAV-Attack'ta bağımsız final holdout bulunmadığı için sonuçlar production/nihai diye adlandırılamaz.

## 8. GPU/epoch kararı

- **UAV-SEAD:** L4 için birincil aday. Beş session splitinin tamamını, çoklu seed'i ve sabit-varyans MSE ile learned-variance Gaussian NLL ablation'ını çalıştırmak anlamlıdır. Blind holdout kapalı kalmalıdır.
- **RflyMAD:** Veri ölçeği GPU'ya uygundur, fakat Gaussian loss tekrarının kendisi yenilik değildir. Mevcut “yeni hyperparameter avı yapma” kararını aşmak için önce yeni temsil/domain sözleşmesi gerekir.
- **ALFA:** GPU teknik olarak kullanılabilir; bilimsel birim çoklu session/fold'dur ve bu veri buna yetmemektedir. Uzun tek-run yerine kısa çoklu seed/LOFO ve açık exploratory etiket gerekir.
- **UAV-Attack:** Exact eski model saniyeler düzeyindedir. Gold-only uzun run gerekçesizdir; L4 zamanı raw-topic enrichment, LOFO, seed ve loss ablation'ına ayrılmalıdır.

Sonuç olarak yeni four-dataset probabilistic GPU v1 çalışması **UAV-SEAD, ALFA ve UAV-Attack için model/loss yeniliği**, **RflyMAD için ise yalnız yeni kapsam/temsil deneyi** olarak tanımlanmalıdır. Hiçbir veri kümesinde “daha çok epoch = yapısal sınırın çözülmesi” varsayımı kabul edilmemelidir.
