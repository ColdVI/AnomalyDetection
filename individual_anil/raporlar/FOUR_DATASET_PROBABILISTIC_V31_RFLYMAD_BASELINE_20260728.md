# Four-dataset probabilistic v3.1 — RflyMAD development baseline

Tarih: 2026-07-28
Statü: Eğitim ve development değerlendirmesi tamamlandı; magnitude gate PASS,
operasyonel çalışma noktası NO-GO. Final fault test kapalıdır.

## Koşu kaynağı ve sözleşme

Koşu Google Colab üzerinde NVIDIA L4 ile 30 epoch tamamlandıktan sonra yerel
artifact deposuna değişiklik yapılmadan aktarıldı:

`artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728`

Koşu namespace'i `four_dataset_probabilistic_v31`, sözleşme SHA-256 değeri
`6c10569e2b99c885b662bffcd7f554799582c28ebcb0536a998b51a8624d4ba2`'dir.
Raporun bildirdiği artifact hashleri aktarım sonrasında doğrulandı. Değerlendirme
yalnız 91 normal-test ve 557 anomaly-development uçuşunu kullanmıştır. Ayrılmış
553 final-fault-test uçuşuna erişilmemiştir.

Model yalnız normal eğitim uçuşlarıyla eğitildi. Normal validation Gaussian NLL
değerine göre, eşitlikte daha erken epoch seçilecek biçimde dondurulan seçim
politikası epoch 30'u seçti (`-0,150592`). Alarm eşiği normal validation'ın
dondurulmuş 0,995 quantile değerinden `10,094001` olarak geldi; sonuç görüldükten
sonra yeniden seçilmedi.

## Magnitude diagnostic

| Diagnostic | Spearman rho |
|---|---:|
| Trained score / random-init score | 0,3288 |
| Trained score / standardized target magnitude | 0,3495 |

Her iki değer de dondurulmuş 0,8 kapısının altındadır;
`magnitude_domination_flagged_at_0_8 = false`. Bu nedenle koşu magnitude gate'i
geçmiştir. Bu sonuç tek başına alarm performansının kabul edildiği anlamına
gelmez.

## Development sonucu

| Düzey / ölçüm | Sonuç |
|---|---:|
| Window ROC-AUC | 0,6271 |
| Window average precision | 0,9403 |
| Flight ROC-AUC | 0,6411 |
| Flight average precision | 0,9211 |
| Normal uçuş alarm oranı | 77 / 91 = %84,62 |
| Anomalili uçuş detection oranı | 462 / 557 = %82,94 |
| Normal alarm penceresi / saat | 1.434,43 |
| Gözlenen normal süre | 1,5295 saat |

Uçuş düzeyindeki dondurulmuş-threshold bayrak matrisi şöyledir:

| | Alarm yok | Alarm var |
|---|---:|---:|
| Normal | 14 | 77 |
| Anomaly-development | 95 | 462 |

Average precision değerleri, değerlendirme kümesindeki yüksek anomaly prevalansı
(557 / 648 = %85,96) nedeniyle tek başına operasyonel başarı kanıtı değildir.
ROC-AUC değerleri sınıfları ancak mütevazı ölçüde sıralayabilen bir sinyal
olduğunu gösterirken, dondurulmuş çalışma noktası normal uçuşların %84,62'sini
alarm olarak işaretlemektedir. Bu nedenle %82,94 anomaly detection oranı kabul
edilebilir bir yanlış alarm yükünde elde edilmiş değildir.

### Fault ailesi tanısalı

| Etiket | Uçuş | Bayraklanan | Oran | Median max score |
|---|---:|---:|---:|---:|
| Environment | 69 | 68 | %98,55 | 123,61 |
| Motor | 243 | 204 | %83,95 | 114,30 |
| Propeller | 129 | 122 | %94,57 | 96,07 |
| Sensor | 110 | 65 | %59,09 | 23,76 |
| Voltage | 6 | 3 | %50,00 | 8,91 |
| NoFault | 91 | 77 | %84,62 | 42,89 |

Bu tablo yalnız development tanısalıdır. Özellikle Voltage için altı uçuş
üzerinden genelleme yapılmaz.

## Karar

Bilimsel sonuç iki parçalıdır:

- Modelin skoru salt hedef büyüklüğünün kopyası değildir ve fault/normal ayrımı
  için zayıf-orta düzeyde bir sıralama sinyali taşımaktadır.
- Dondurulmuş validation eşiğindeki alarm çalışma noktası operasyonel olarak
  **NO-GO**'dur; normal uçuş alarm oranı kabul edilemeyecek kadar yüksektir.

Bu sonuç görüldükten sonra threshold, feature, split, epoch veya magnitude kapısı
değiştirilmemiştir. Final fault test açılmayacaktır. Bir sonraki çalışma ancak
ayrı ve önceden kaydedilmiş bir development protokolüyle, normal-validation ve
anomaly-development üzerinde event oluşumu/debounce ve uçuş-fazı bağlamını
kalibre edebilir. Bunun ardından tek bir çalışma noktası kilitlenirse final fault
test bir kez açılabilir.

`normal_alarm_windows_per_hour` değeri debounce edilmiş event metriği değildir;
önceki event-level yanlış alarm sonuçlarıyla doğrudan karşılaştırılmaz.

## Görseller

Raporlama grafikleri
`artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728`
altındadır:

1. `01_training_and_selected_checkpoint.png` — train/normal-validation NLL ve
   seçilen checkpoint.
2. `02_flight_max_score_distribution.png` — normal ve anomaly-development uçuş
   max-score dağılımı, dondurulmuş eşikle birlikte.
3. `03_flight_flag_matrix.png` — uçuş bayrak matrisi; event confusion matrix
   değildir.
4. `04_label_level_diagnostics.png` — fault ailesi bazında bayrak oranı ve median
   max score.

`plot_manifest.json`, kaynak training report hashini, dört PNG hashini,
`threshold_reselected = false` ve `final_fault_test_accessed = false` kayıtlarını
içerir.
