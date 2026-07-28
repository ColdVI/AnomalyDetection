# Four-dataset probabilistic GPU v2 — nihai teknik rapor

Tarih: 2026-07-28
Model namespace: `four_dataset_probabilistic_gpu_v2`
Değerlendirme namespace: `four_dataset_probabilistic_event_eval_v1`

## Yönetici özeti

Dört Colab koşusu da 30 epoch tamamlanmış NVIDIA L4 koşusudur. Model, scaler,
config, split ve artifact hash kontrolleri dört veri setinde de geçmiştir;
`magnitude_domination_flagged_at_0_8=false` koşulu sağlanmıştır. Model ağırlıkları,
epoch, split, eşik adayları ve persistence ızgarası sonuç görüldükten sonra
değiştirilmemiştir.

Operasyonel sonuç tek cümleyle: mevcut v2 ailesi hiçbir veri setinde doğrudan
üretim alarmı olarak kabul edilemez. RflyMAD güçlü ve tutarlı bir araştırma
baseline'ıdır; UAV-SEAD'de sıralama sinyali vardır fakat olay duyarlılığı çok
düşüktür; ALFA ve UAV Attack mevcut protokolde NO-GO'dur.

| Veri seti | Nihai statü | Esas gerekçe |
|---|---|---|
| ALFA | NO-GO; yalnız keşif | Interval ROC-AUC 0.483; 5 gerçek olaydan 1'i; yalnız 2 normal validation uçuşu; session-family çakışması |
| RflyMAD | Koşullu araştırma GO; operasyonel değil | Interval ROC-AUC 0.907; flight mean AUC 0.915; referans noktada 37/66 olay; domain-transfer kapıları ayrı ve başarısız kalıyor |
| UAV-SEAD | Sinyal var, event-detector NO-GO | Interval ROC-AUC 0.747; referans noktada yalnız 7/206 olay ve %0.264 zaman-ağırlıklı range recall |
| UAV Attack | Açık NO-GO | Bütün flight agregasyonlarında AUC 0.000; tek normal test kaynağında 32.42 false event/saat ve zamanın %75.3'ünde alarm |

## Neden eski rapor güvenilir değildi?

V2 eğitim runner'ından devralınan eski değerlendirme yolu anomalili bir uçuşun
bütün pencerelerine pozitif truth veriyor, uçuş skorunu `max(window_score)`
olarak alıyor ve tek alarm penceresiyle bütün uçuşu detected sayıyordu. Bu üç
işlem gerçek arıza/saldırı zamanını lokalize etmez ve uzun uçuşlarda çoklu-deneme
etkisi üretir.

Yeni evaluator üç düzeyi ayırır:

1. interval truth bulunan pencerelerde threshold-bağımsız sıralama;
2. kesintisiz eşik-üstü süre isteyen nedensel alarm eventleri;
3. yalnız offline triage için, keşifsel uçuş agregasyonları.

Point adjustment uygulanmamıştır. Bir gerçek olayda tek alarm noktası bulunması
bütün olay aralığını doğru saydırmaz.

## Ground-truth eşlemesi

- ALFA: `processed.zip` içindeki 37 aktif `failure_status-*` serisinin ilk aktif
  kaydı, Silver kaynağının ilk `ts_ns` değerine göre Gold `t_rel_s` eksenine
  taşındı. Testte 5 arızalı uçuş eşlendi; adı açıkça `no_ground_truth` olan bir
  uçuş metrikten dışlandı. 277 test penceresi gerçek arıza aralığına düştü.
- RflyMAD: parsed 10 Hz kaynaklardaki `fault_active` ve `condition_active`
  doğrudan kullanıldı; 66 gerçek olay vardır.
- UAV-SEAD: veri setinin `labels.json` içindeki absolute-microsecond bölgeleri,
  `absolute_us = Silver source min(timestamp) + t_rel_s × 1e6` bağıntısıyla
  eşlendi. Testte 134 anomalili uçuşun tamamı, 206 ayrı olay ve 47,957 aktif
  pencere eşlendi.
- UAV Attack: mevcut pakette interval truth yoktur. Anomalili uçuşların tümünü
  pozitif pencere saymak yasak olduğundan event recall/confusion matrix üretilmez.

Truth yenilemesi mevcut model skorlarını değiştirmeden yapıldı. Her ledger için
immutable score payload hash'i işlem öncesi ve sonrası aynı olmak zorundaydı ve
bu kapı geçti.

## Dondurulmuş referans hücresi

Aşağıdaki hücre görsel anlatım için önceden tanımlı `1.0 s persistence +
validation'da en fazla 1 false event/saat` hücresidir. Test sonucundan seçilmiş
bir kazanan değildir; bütün persistence × bütçe ızgarası ayrıca raporlanmıştır.

| Veri seti | Val quantile / eşik | Gerçek event | Yakalanan | Event recall | Test false event/saat | Medyan / p90 gecikme | Range precision / recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALFA | .990 / 4.1123 | 5 | 1 | 20.0% | 0.000 | 6.13 / 6.13 s | 100.0% / 5.50% |
| RflyMAD | .995 / 2.7784 | 66 | 37 | 56.1% | 0.768 | 6.50 / 44.06 s | 93.1% / 16.54% |
| UAV-SEAD | .990 / 2.1306 | 206 | 7 | 3.40% | 0.000 | 20.45 / 66.72 s | 96.2% / 0.264% |
| UAV Attack | .9975 / 9.5811 | N/A | N/A | N/A | 32.417 | N/A | N/A |

Ön-kayıtlı 1,000 tekrarlı, normal/anomaly tabakalı `source_id` percentile
bootstrap %95 aralıkları belirsizliğin büyüklüğünü gösterir:

- ALFA event recall: %0–60; range recall: %0–21.49;
- RflyMAD event recall: %42.42–68.18; false event/saat: 0–2.43; range recall:
  %9.86–23.92;
- UAV-SEAD event recall: %0.53–7.04; range recall: %0.006–0.556;
- UAV Attack'ta tek normal test kaynağı tekrar örneklendiği için false-event
  aralığı anlamsız biçimde noktasaldır: 32.417–32.417.

ALFA'daki sıfır test false-event değeri yalnız 0.131 normal uçuş-saatine,
UAV Attack sonucu 0.123 saate dayanır. ALFA'nın 2, Attack'ın 1 normal validation
uçuşu olduğu için ikisi de ön-kayıttaki en az 5 uçuşluk kalibrasyon iddiası
kapısını geçmez. Sıfır sayısı burada güvenilir bir “sıfır false alarm” iddiası
değildir.

## Threshold-bağımsız ve uçuş-düzeyi sonuçlar

| Veri seti | Interval ROC-AUC | Interval AP | Flight max AUC | Flight mean AUC |
|---|---:|---:|---:|---:|
| ALFA | 0.483 | 0.108 | 0.611 | 0.611 |
| RflyMAD | 0.907 | 0.805 | 0.631 | 0.915 |
| UAV-SEAD | 0.747 | 0.263 | 0.492 | 0.772 |
| UAV Attack | N/A | N/A | 0.000 | 0.000 |

Flight mean AUC değerleri test görüldükten sonra incelenen agregasyon ailesinin
parçasıdır. Bunlar yeni resmi karar eşiği veya model seçimi değildir. RflyMAD ve
UAV-SEAD'de “uçuş genel olarak alışılmadık mı?” sinyali bulunduğunu, bunun tek
başına “anomali ne zaman başladı?” sorusunu çözmediğini gösterir.

## Eğitim tanısı ve gerçek hesap yükü

| Veri seti | Train windows/epoch | 30 epoch window toplamı | L4 epoch-loop toplamı | Validation eğrisi |
|---|---:|---:|---:|---|
| ALFA | 13,026 | 390,780 | 6.31 s | epoch 30'a kadar iyileşiyor |
| UAV Attack | 17,656 | 529,680 | 7.78 s | en düşük epoch 3; sonra belirgin kötüleşme |
| UAV-SEAD | 911,940 | 27,358,200 | 457.72 s | en düşük epoch 28; epoch 30 hafif kötü |
| RflyMAD | 234,529 | 7,035,870 | 189.37 s | epoch 30'a kadar iyileşiyor |

Epoch 3/28 notları yalnız tanıdır. Ön-kayıt 30 epochu ve final checkpointi
dondurduğu için geçmişe dönük checkpoint seçimi yapılmadı. Dört koşunun GPU
eğitim döngüsü toplamı 661.17 saniye, yani yaklaşık 11.02 dakikadır. Notebook'un
Drive aktarımı, hash doğrulaması, Parquet yüklemesi ve skor exportu bu süreye
dahil değildir.

Yerel raw envanter yaklaşık 30.43 GiB olmasına rağmen optimizer raw ULog/ROS/MAT
dosyalarını doğrudan okumadı. Colab'a taşınan sözleşmeli Gold/parsed bundle toplamı
980.87 MiB idi:

| Veri seti | Yerel raw/bronze | Modele giden transfer girdisi |
|---|---:|---:|
| ALFA | 0.253 GiB | 0.0175 GiB Gold Parquet |
| RflyMAD | 18.910 GiB | 0.3157 GiB seçili parsed development bundle |
| UAV-SEAD | 10.597 GiB | 0.6109 GiB Gold Parquet |
| UAV Attack | 0.668 GiB | 0.0131 GiB Gold Parquet |

Bu nedenle “30 GB veri vardı, niçin yaklaşık 1 GB yüklendi?” sorusunun cevabı:
eğitim ham arşivi değil, önceden çıkarılmış model özelliklerini kullandı. Ham
telemetri değersiz değildir; yeni kanal eklemenin katkısını ölçen ayrı ablation
turu için saklanır.

## Uçuş modu ayrımı

Mevcut v2 dört ayrı ground/takeoff/cruise/landing modeli eğitmedi. Bu eksik,
özellikle UAV-SEAD'deki “sıralama var ama event recall yok” bulgusuna karşı
mantıklı bir v3 hipotezidir; ancak mevcut test sonucuna göre v2'ye eklenemez.

V3 için önerilen yöntem dört bağımsız modelle başlamak değil, nedensel bir
`flight_phase` context kanalıyla tek baseline kurmaktır:

- `ground`: armed/in-air değil veya kalıcı düşük irtifa ve düşük hız;
- `takeoff/climb`: in-air ve kalıcı pozitif vertical rate;
- `cruise/maneuver`: in-air, climb/descend koşulu yok;
- `descent/landing`: kalıcı negatif vertical rate; touchdown sonrası ground.

Faz yalnız geçmiş ve mevcut örnekten hesaplanmalı; fault/onset, dosya adı veya
gelecek örnek kullanılmamalıdır. Önce group-safe split yapılmalı, sonra aynı
baseline faz context'i olmadan ve faz context'iyle karşılaştırılmalıdır. Faz
başına eşik ancak normal-validation exposure yeterliyse kalibre edilmelidir;
aksi durumda global eşik korunur. Ana rapor event recall, false event/saat ve
gecikmeyi hem toplam hem faz bazında verir.

## V3 ve ham telemetri ablation kararı

Yeni eğitim sırası:

1. ALFA session-family, Attack campaign/platform, SEAD mission/session ve
   RflyMAD canonical/domain grup registry'lerini dondur;
2. aynı v2 model/features ile group-safe baseline üret;
3. nedensel flight-phase context ablation'ını çalıştır;
4. raw kanal bloklarını tek tek ekle: control → actuator → estimator → energy →
   environment/context;
5. yalnız normal validation ile event eşiği kalibre et; bütün test ızgarasını
   seçimsiz raporla.

Ham genişletme çalışma sırası kullanıcının istediği gibi ALFA → RflyMAD →
UAV-SEAD → UAV Attack'tır. ALFA hızlı bir pipeline smoke-testidir; bilimsel
başarı beklenen ana aday RflyMAD'dir. UAV-SEAD faz/context hipotezinin ana
testidir. UAV Attack ancak campaign-safe normal kalibrasyon kaynağı sağlandıktan
sonra tekrar eğitilmelidir.

Truth/leakage olarak yasaklanan girdiler: `fault_id`, `fault_mode`, injection
komutu, fault başlangıç/bitiş metadata'sı ve dosya adındaki fault kodu. PX4'in
kendi failure/health flagleri kullanılırsa fizik modelinin gizli kanalı olarak
değil, ayrı “PX4-native baseline” olarak açıkça raporlanmalıdır.

## Görsel paket

- [Eğitim tanıları](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/01_training_diagnostics.png)
- [Dört veri seti sonuç özeti](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/02_detection_summary.png)
- [Dondurulmuş event operating grid](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/03_event_operating_grid.png)
- [Interval-truth confusion matrix'leri](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/04_reference_confusion_matrices.png)
- [ALFA timeline](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/05_alfa_reference_timelines.png)
- [RflyMAD timeline](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/05_rflymad_reference_timelines.png)
- [UAV-SEAD timeline](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/05_uav_sead_reference_timelines.png)
- [UAV Attack timeline](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/05_uav_attack_reference_timelines.png)
- [ADS-B ile ölçülmüş hesap yükü karşılaştırması](../artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_compute_measured_v2/adsb_vs_four_dataset_compute_load.png)

Timeline örnekleri skora göre seçilmedi. Normal ve anomalili sınıfta test süresi
medyanına en yakın kaynak, eşitlikte alfabetik `source_id`, kullanıldı. Bu kural
“güzel görünen örnek” cherry-picking riskini azaltır.

## Yeniden üretilebilirlik

- Event sözleşmesi: `configs/four_dataset_probabilistic_event_eval_v1.json`
- Event kodu: `scripts/four_dataset_probabilistic_event_eval_v1.py`
- Görsel kodu: `scripts/plot_four_dataset_probabilistic_v2_results.py`
- Kalıcı yorum ve raporlama kaydı:
  `docs/FOUR_DATASET_V2_INTERPRETATION_V3_REPORTING_RECORD_20260728.md`
- Kabul edilen koşular: `artifacts/four_dataset_probabilistic_event_eval_v1/accepted_runs/20260728T071250Z/`
- Skor ve değerlendirme ledger'ları: `artifacts/four_dataset_probabilistic_event_eval_v1/20260728/`
- Görsel hash manifesti: `artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/results_plot_manifest.json`

İlgili metodoloji kaynakları: [AAAI — Rigorous Evaluation of Time-Series Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/20680),
[NeurIPS TSB-AD benchmark](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html),
[ALFA veri seti yayını](https://publications.ri.cmu.edu/alfa-a-dataset-for-uav-fault-and-anomaly-detection),
[RflyMAD yayını](https://journals.sagepub.com/doi/10.1177/02783649241305153).
