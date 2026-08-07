# Four-dataset probabilistic v3.1 — ALFA 4-fold CV smoke

Tarih: 2026-07-28
Statü: Pipeline smoke tamamlandı; ana performans iddiası için NO-GO.

## Amaç ve sözleşme

Bu koşu ALFA'nın sekiz saf-normal bağımsız grubunu dört fold içinde kullanarak
v3.1 split loader, train-only scaler, grup/kaynak dengeli sampler, epoch
checkpoint-resume ve normal-validation checkpoint seçimini sınamıştır. Her fold
30 epoch çalıştırılmış; test sonucu checkpoint, threshold veya feature seçmemiştir.

## Sonuç

| Fold | Seçilen epoch | En iyi normal-val NLL | Magnitude gate | Flight ROC-AUC | Normal uçuş alarm oranı | Anomalili uçuş detection |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 14 | 0,1419 | PASS | 0,4444 | 0,0% | 0,0% |
| 1 | 11 | 1,5837 | FAIL | 0,4091 | 50,0% | 54,5% |
| 2 | 4 | 1,6952 | FAIL | 0,5455 | 25,0% | 18,2% |
| 3 | 1 | 1,5552 | FAIL | 0,5208 | 83,3% | 100,0% |

Fold 1--3'te trained score ile random-init/target magnitude Spearman değerleri
0,8 kapısını aşmıştır. Bu nedenle yüksek detection görünen fold 3 başarı diye
yorumlanamaz; aynı fold normal uçuşların %83,3'ünde alarm üretmiştir. Yalnız fold
0 magnitude kapısını geçmiştir, fakat fault detection sıfırdır.

## Karar

Teknik smoke PASS'tir: dört fold da tamamlanmış, tamamlanmış-epoch checkpointleri
çalışmış, Windows atomik-yazma kilidi retry sonrasında fold 1--3 kaldığı epoch'tan
devam etmiş ve final test benzeri dış veri kullanılmamıştır.

Bilimsel ALFA baseline sonucu NO-GO'dur. Sekiz saf-normal grup ve fold başına iki
normal validation kaynağı, kararlı magnitude-independent skor kalibrasyonu için
yetersiz görünmektedir. ALFA sonucu RflyMAD eğitimi için bir stop koşulu değildir;
magnitude gate dataset/run bazında yeniden ölçülecektir. RflyMAD sonucu yalnız
kendi natural validation diagnostic'i false çıkarsa ablation aşamasına geçebilir.

Artifactlar `artifacts/four_dataset_probabilistic_v31/runs/alfa_fold_0` ile
`alfa_fold_3` arasındaki dört run klasöründedir. Bu ölçümler pencere/flight smoke
metrikleridir; event-level operasyonel metriklerle karıştırılmaz.
