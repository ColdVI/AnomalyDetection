# Metrik evrimi

## Neden metrik sözleşmesi değişti?

İlk turlarda satır/pencere AUC, AUPRC ve uçuş max/mean AUC baskındı. Bunlar skor sıralama gücünü gösterir; arızanın doğru zamanını bulmayı, alarm kümelerinin sayısını veya operatör yükünü göstermez. Proje bu nedenle “row/window → interval/event → flight” birimlerini açıkça ayıran sözleşmeye geçti. Kaynak: `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.

## Metrik sözlüğü

| Metrik | Birim / tanım | Doğru yorum | Yanlış yorum | Kaynak |
|---|---|---|---|---|
| ROC-AUC | Pencere, interval veya uçuş skorlarının ikili truth’u sıralama olasılığı | Eşik bağımsız sıralama sinyali | “%AUC oranında anomali yakalandı” | `scripts/plot_four_dataset_probabilistic_v2_results.py`; `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` |
| Average Precision / AUPRC | Pozitif sınıf prevalansına duyarlı precision-recall özeti | Nadir pozitiflerde sıralama özeti; prevalans mutlaka verilmeli | Farklı pozitif oranlı sürümler arasında bağlamsız kıyas | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_RESULT_AUDIT_20260728.md`; `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json` |
| Flight max/mean AUC | Her uçuşun maksimum/ortalama pencere skoruyla uçuş etiketi sıralaması | Triage için keşif metriği | Fault intervalinin yakalandığı veya alarm yükünün düşük olduğu iddiası | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` |
| Any-window flight alarm fraction | Uçuşta en az bir pencere eşik üstüyse 1 | Basit uçuş-level alarm yükü teşhisi | Event confusion matrix veya event recall | `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md` |
| Event recall | Truth eventlerden en az bir predicted event ile eşleşenlerin oranı | Arıza olayını bulma | Uçuş yakalama oranı veya zaman kapsama oranı | `scripts/evaluate_rflymad_probabilistic_v31_events.py` (`detected / event_count`) |
| False events / normal hour | Bağımsız normal testte predicted event sayısı / toplam normal uçuş saati | Operatör alarm yükünün ana ölçüsü | Alarm penceresi/saat veya normal uçuş oranıyla eşdeğer saymak | `scripts/evaluate_rflymad_probabilistic_v31_events.py` (`false_events / normal_hours`) |
| Normal flights with any false event | En az bir yanlış olay taşıyan normal uçuş oranı | Yanlış alarmın uçuşlara yayılımı | Olay frekansı | `scripts/evaluate_rflymad_probabilistic_v31_events.py` |
| Range precision | Predicted event sürelerinin truth ile zaman-ağırlıklı overlap oranı | Alarm aralıklarının ne kadarının gerçeğe denk geldiği | Kaç truth event yakalandı | `scripts/evaluate_rflymad_probabilistic_v31_events.py` |
| Range recall | Truth event süresinin predicted eventlerle zaman-ağırlıklı overlap oranı | Arıza süresinin ne kadarı kapsandı | Event recall | `scripts/evaluate_rflymad_probabilistic_v31_events.py` |
| Detection delay | Truth başlangıcından ilk eşleşen alarma süre | Zamanında uyarı | Yakalanmayan eventlerin yok sayıldığı unutulmamalı | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`; `scripts/evaluate_rflymad_probabilistic_v31_events.py` |
| Train/validation Gaussian NLL | Normal-only next-step tahmin fit metriği | Checkpoint/fit teşhisi | Anomali tespit performansı | `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json` |
| Magnitude correlations | Trained–random ve trained–target-magnitude Spearman ρ | Skor kestirme yolunu teşhis eden güvenlik kapısı | Düşük ρ’yı otomatik başarı saymak | `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `configs/four_dataset_probabilistic_v31_rflymad.json` |

## Sayısal evrim

| Aşama | Başlık sayı | Sonradan ne öğrenildi? | Bugünkü statü | Kaynak |
|---|---|---|---|---|
| ALFA erken LSTM-AE | AUPRC 0,872; raw genişletmede flight AUC 0,918 | Causal düzeltme 0,878→0,611; event overlap 0,594→0,194–0,224 | Tarihsel araştırma sonucu; final operasyonel sayı değil | `docs/PROJE_SUREC_VE_SONUC.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| SEAD uçuş split’i | Fair row AUC 0,474→0,799 | Artış model değişiminden çok veri/split değişimiydi; session split seed std ±0,212→±0,012 | Split dersi olarak kullan | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| SEAD AE ailesi | Benzer recall/score desenleri | Trained-random ρ≈0,964 ve score-magnitude ρ≈0,965 | Model öğrenmesi iddiası geçersiz; magnitude artefaktı | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| ADS-B reddedilen ilk tur | Sentetik recall %97,6 | Doğal alarm yükü 25,54/saat; doğal ground truth yok | Başarı olarak kullanılmaz | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `archive/2026-07-10_rejected_adsb_attempts/` |
| Dört-set v2 eski evaluator | ALFA/Attack/SEAD/Rfly window ve flight AUC/AP; any-window alarm | Bütün anomali uçuşu pozitif işaretlendi, max skor/any-alarm event tespiti sanıldı | Operasyonel sonuç olarak geçersiz | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_RESULT_AUDIT_20260728.md`; `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` |
| Dört-set v2 doğru event evaluator | Rfly %56,1 event recall / 0,768 false event-h; SEAD %3,4 / 0; ALFA %20 / 0; Attack 32,417 false event-h | ALFA normal exposure yalnız 0,131 h; Attack’ta interval truth yok | Yalnız Rfly koşullu research GO | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` |
| Rfly v3.1 any-window | Flight AUC 0,641; anomaly flight alarm %82,94; normal flight alarm %84,62; balanced accuracy %49,16 | Uçuş yakalama ile normal alarm neredeyse aynı; event değil | Operating point NO-GO | `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md` |
| Rfly v3.1 B0 event | 2-of-3: %43,27 event recall, 0,654 false event-h; max recall %58,71 @ 9,153/h | Real %7,81, Sensor %5,45; range recall yalnız yaklaşık %0,65–1,14 | En güncel bilimsel karar: NO-GO | `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |

## Sunumda birlikte gösterilmesi zorunlu çiftler

- Event recall **ve** false events/normal hour. Kaynak: `scripts/evaluate_rflymad_probabilistic_v31_events.py`.
- Anomali uçuş yakalama **ve** normal uçuşta any-alarm oranı. Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`.
- AUPRC **ve** pozitif prevalansı/veri rolü. Rfly v3.1’de window AP 0,940 ve flight AP 0,921, test havuzunun büyük çoğunluğu anomali uçuşudur; tek başına başlık yapılmamalıdır. Kaynak: `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`; `artifacts/four_dataset_probabilistic_v31/split_report.json`.
- Validation NLL **ve** magnitude gate; ardından ayrı event sonucu. Kaynak: `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`.

## Nihai raporlama dili

“Model anomaly uçuşların %82,9’unu yakaladı” yerine: “Development anomaly uçuşlarının %82,9’unda en az bir pencere eşiği aşıldı; bağımsız normal uçuşların %84,6’sında da en az bir aşım vardı. Gerçek olay politikasında referans nokta %43,27 event recall ve 0,654 yanlış olay/normal saat üretti; bu nedenle NO-GO verildi.” Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`; `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.
