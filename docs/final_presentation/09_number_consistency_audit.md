# Sayı tutarlılığı denetimi

## Kullanım kuralı

“Eski” her zaman yanlış demek değildir; bazen farklı veri rolü, split veya metrik birimidir. Sunumda yalnız aynı sözleşmeye ait sayılar birlikte kıyaslanmalıdır. Ana kaynak önceliği: güncel artifact JSON/CSV → aynı koşunun final raporu → yaşayan günlük → tarihsel konsolidasyon. Kaynak: `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.

## Çelişen veya bağlam isteyen sayılar

| Konu | Eski/alternatif sayı | Güncel/doğru bağlam | Karar | Kaynak |
|---|---|---|---|---|
| ALFA ölçeği | 47 işlenmiş uçuş | Raw genişletmede 54 uçuş / 15 normal | İkisi farklı sürüm; slaytta sürümü yaz | `docs/final_rapor_ml_fizibilite_2026-07-16.md` |
| ALFA performansı | LSTM-AE AUPRC 0,872; flight AUC 0,918; causal öncesi 0,878 | Causal düzeltme 0,611; event overlap 0,194–0,224; v3.1 group-CV AUC 0,409–0,545 ve NO-GO | Eski yüksek sayıyı final diye kullanma | `docs/PROJE_SUREC_VE_SONUC.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| SEAD ölçeği | Yaklaşık 1.396 toplam uçuş / 398 normal / ~64 session gibi tarihsel ara havuz | v2 development 1.044 ve 899 normal; 200 blind holdout kapalı; v3 strict-parent 1.244 ULog / 259 grup | Aşama ve rolü yaz; toplamları birbirine toplama | `docs/PROJE_SUREC_VE_SONUC.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| Rfly ölçeği | 490 Real uçuş; 1.225 locked-test | Full v2 canonical 6.605; v3.1 rol toplamları ayrı subsetler (260+90+91+557+553) | “RflyMAD” tek havuzmuş gibi sayı verme | `docs/final_rapor_ml_fizibilite_2026-07-16.md`; `artifacts/four_dataset_probabilistic_v31/split_report.json` |
| Rfly truth | 2.712 uçuşta t=0 active fault | Düzeltme sonrası active-from-first-sample 1.354→0; 6.605 complete, 0 fail | Düzeltme öncesi sonuçları geçersiz işaretle | `gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md` |
| ADS-B gün sayısı | 3 gün / 638 parça / 256.150.550 satır | Contextual v2 üç yeni günle 1.185 Silver parça | v1 ve v2 kapsamını ayır | `docs/final_rapor_ml_fizibilite_2026-07-16.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| ADS-B dropout truth | 666.739 active-injection satırı | 570.666 gözlenebilir değişmiş satır | Ground truth’ta “aktif” ile “observable”ı ayır | `docs/final_rapor_ml_fizibilite_2026-07-16.md` |
| ADS-B v2 eğitim | İlk incidentte epoch tamamlanmadı; sonraki CPU run epoch 1 sonrası öldü | Son L4 koşusu 8 epoch ve 1.889.036.504 window-epoch tamamladı; NLL 0,81609→0,80753 | Incident sayılarını final koşuyla karıştırma | `docs/ADSB_CONTEXTUAL_PHYSICS_V2_TRAINING_INCIDENT_20260723.md`; `docs/ADSB_CONTEXTUAL_PHYSICS_V2_TRAINING_INCIDENT_V4_20260725.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| Dört-set v2 Rfly | Eski window AUC 0,727/AP 0,807; flight max AUC 0,631; any-alarm %98,5 | Gerçek interval AUC 0,907/AP 0,805; flight mean 0,915; event %56,1 / 0,768 false event-h | Farklı truth/aggregation; event satırını final kullan | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_RESULT_AUDIT_20260728.md`; `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` |
| Dört-set v2 SEAD | Eski window AUC 0,729/AP 0,622; any-alarm %94,8 | Interval AUC 0,747/AP 0,263; event 7/206=%3,4; range recall %0,264 | Eski whole-flight truth’ü final kullanma | aynı |
| Dört-set v2 Attack | Eski window AUC 0,105/AP 0,728; any-alarm %100 | Interval truth yok; flight max/mean AUC 0; 32,417 false event-h ve %75,3 alarm-time | AP yüksekliği pozitif prevalans/yanlış truth artefaktı; başarı değil | aynı |
| Dört-set v2 ALFA | Event recall %20 ve 0 false event-h | Normal test exposure yalnız 0,131 h, validation yalnız 2 normal kaynak | Sıfır FA’yı güçlü kanıt diye kullanma | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` |
| Rfly v2 vs v3.1 | v2 event %56,1 / 0,768 h | v3.1 B0 %43,27 / 0,654 h | Split, grup, rol ve evaluator farklı; doğrudan model kıyası yok | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`; `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |
| Rfly v3.1 AP | Window AP 0,9403; flight AP 0,9211 | 557 anomaly vs 91 normal uçuşlu anomaly-ağırlıklı evaluation | AP’yi prevalanssız başlık yapma | `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`; `artifacts/four_dataset_probabilistic_v31/split_report.json` |
| Rfly v3.1 “recall” | Any-window anomaly flight %82,94 | B0 gerçek event recall %43,27; Real %7,81, Sensor %5,45 | %82,94 event recall değildir | `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`; `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |
| Rfly normal alarm | Any-window normal flight %84,62; alarm windows/h 1.434,429 | B0 primary false events/normal h 0,654 | Pencere, uçuş ve olay birimleri farklı | `training_report.json`; `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |
| Rfly v3.1 NLL | Train `-0,960454`, val `-0,150592` | Kod sabit Gaussian terimi eklemiyor; scale<1 iken negatif mümkün | Negatif NLL hata veya detector başarısı değil | `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `training_report.json` |
| Magnitude diagnostic | SEAD ρ≈0,964/0,965 | Rfly v3.1 ρ=0,3288/0,3495, gate=0,8, flag=false | Farklı run/dataset; Rfly PASS yalnız bu kapıya aittir | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `training_report.json`; `configs/four_dataset_probabilistic_v31_rflymad.json` |

`training_report.json` kısaltması üstte `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json` anlamındadır.

## Sunumda kullanılmaması gereken eski/geçersiz sonuçlar

1. `archive/2026-07-10_rejected_adsb_attempts/` altındaki %97,6 sentetik recall’ı güncel ADS-B başarısı gibi kullanma; 25,54 doğal alarm/saat ve ground-truth yokluğu nedeniyle yaklaşım reddedildi. Kaynak: `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.
2. RflyMAD truth-schema v2 öncesi, 2.712 uçuşluk parser hatasından etkilenebilecek interval/onset metriklerini kullanma. Kaynak: `gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md`.
3. Dört-set probabilistic v2’nin bütün anomali uçuşunu pozitif sayan eski evaluator window/AP/any-alarm sayılarını operasyonel sonuç diye kullanma. Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_RESULT_AUDIT_20260728.md`.
4. ALFA causal düzeltme öncesi 0,878 ve overlap 0,594 sayılarını nihai sonuç diye kullanma. Kaynak: `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.
5. SEAD AE sonuçlarını “üç derin model aynı şeyi öğrendi” diye olumlu sunma; random/magnitude korelasyonu bunun kestirme olduğunu gösterdi. Kaynak: `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.
6. Rfly v3.1 %82,94 any-window anomaly flight oranını event recall diye kullanma. Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`.
7. Rfly v3.1 AP 0,921/0,940 sayılarını pozitif prevalansı ve normal alarm yükünü vermeden başlık yapma. Kaynak: `training_report.json`.
8. Alarm windows/hour, normal flights with any alarm ve false events/normal hour sayılarını birbirinin yerine kullanma. Kaynak: `scripts/evaluate_rflymad_probabilistic_v31_events.py`.
9. Rfly v2 ile v3.1’i “aynı testte model geriledi” biçiminde kıyaslama; split ve roller farklı. Kaynak: `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.
10. Mühürlü 553 final fault uçuşu değerlendirilmiş gibi konuşma; B0 NO-GO nedeniyle kapalı kaldı. Kaynak: `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.

## Doğrulanan v3.1/B0 başlık seti

- Aktif özellik: 24/28; epoch: 30; threshold: 10,0940; magnitude ρ: 0,3288/0,3495, flag=false. Kaynak: `training_report.json`.
- Any-window teşhisi: flight AUC 0,6411; TP 462, FN 95, FP 77, TN 14; anomaly %82,94; normal %84,62; balanced accuracy %49,16. Kaynak: aynı; `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`.
- B0 primary: 0,654 false event/normal hour; %43,27 event recall; Real %7,81; Sensor %5,45; NO-GO. Kaynak: `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.
