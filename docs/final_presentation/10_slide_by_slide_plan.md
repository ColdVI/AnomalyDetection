# Slayt slayt final sunum planı

## Tasarım ilkesi

20 slayt, üç perde. Her sonuç slaydında veri rolü, truth birimi ve metrik birimi görünür olmalı. Her slayt tek ana cümle taşımalı; ayrıntı konuşmacı notuna bırakılmalı. Kaynak disiplini `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` raporlama sözleşmesine dayanır.

## Perde I — Problem, veri ve ölçüm (1–6)

### 1. Başlık — “İHA Telemetrisinde Anomali Tespiti: Sinyalden Güvenilir Alarma”

- **Ana mesaj:** Amaç yüksek bir AUC değil, bağımsız uçuşlarda gerçek olayı düşük alarm yüküyle bulmaktır.
- **Görsel:** Sade uçuş telemetrisi çizgisi + `window → event → flight` mini şeması; yeni vektör görsel.
- **Kaynak:** `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.

### 2. Araştırma sorusu ve başarı tanımı

- **Ana mesaj:** Normal-only model, görmediği grup/domainlerde fault intervalini yerelleştirebilir mi?
- **İçerik:** Event recall + false events/normal hour birlikte; final test yalnız gate geçilirse.
- **Görsel:** `07_missing_visuals_plan.md` P0 metrik merdiveni.
- **Kaynak:** `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.

### 3. Proje kronolojisi — üç düzeltme dalgası

- **Ana mesaj:** İlerleme model karmaşıklığından çok truth, split ve evaluator düzeltmeleriyle geldi.
- **Görsel:** P0 üç-yüzme-şerit kronoloji.
- **Kaynak:** `01_project_timeline.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.

### 4. Beş veri kaynağı, beş farklı araştırma rolü

- **Ana mesaj:** ALFA/Attack/SEAD/Rfly etiketli ama bağımsızlık ve gözlenebilirlikleri farklı; ADS-B gerçek fakat etiketsiz.
- **Görsel:** P0 fizibilite matrisi.
- **Kaynak:** `02_dataset_inventory.md`; `docs/final_rapor_ml_fizibilite_2026-07-16.md`.

### 5. Veri ölçeği ≠ bağımsız örneklem

- **Ana mesaj:** Uçuş sayısı büyürken session/scenario akrabalığı bilimsel örnek sayısını küçültür.
- **Görsel:** `artifacts/rfly_full/v2/visuals/01_data_composition.png` + küçük “6.605 uçuş / group-safe rol subsetleri” kartı.
- **Kaynak:** `artifacts/rfly_full/v2/visuals/README.md`; `artifacts/four_dataset_probabilistic_v31/split_report.json`.

### 6. Metriklerin ayrılması

- **Ana mesaj:** Window AUC, flight triage ve event alarmı üç ayrı sorudur.
- **Görsel:** `artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/02_detection_summary.png` veya P0 metrik merdiveni.
- **Kaynak:** `04_metric_evolution.md`; `scripts/plot_four_dataset_probabilistic_v2_results.py`.

## Perde II — Denemeler, başarısızlıklar ve metodolojik kazanımlar (7–13)

### 7. İlk ders: etiket her zaman ölçülebilir sinyal değildir

- **Ana mesaj:** UAV Attack’ta 6 Ping DoS kaydının 4’ü mevcut telemetride fiziksel iz bırakmadı; ADS-B rota olaylarının 24/24’ü düşük hız bearing artefaktıydı.
- **Görsel:** `docs/assets/adsb_simple_anomaly/04_route_summary.png` + `05_route_examples.png`; Attack için metin kartı.
- **Kaynak:** `docs/final_rapor_ml_fizibilite_2026-07-16.md`; `artifacts/adsb/simple_anomaly_20260722/summary.json`.

### 8. Truth hatası modeli geçersiz kılar

- **Ana mesaj:** RflyMAD’de 2.712/6.605 uçuş sahte t=0 fault ile etkilenmişti; parser düzeltildi ve sonuç hattı yeniden kuruldu.
- **Görsel:** P0 truth before/after şeması.
- **Kaynak:** `gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md`.

### 9. Yöntem turu: basitten derine aynı duvar

- **Ana mesaj:** IF, AE/USAD, LightGBM, Chronos, tek özellik, fusion, CUSUM, PX4-native ve TCN sinyal buldu; hiçbiri sürekli olarak recall ve alarm bütçesini birlikte tutturmadı.
- **Görsel:** Yöntem aileleri şeridi + `docs/sunum_hafta5_gorseller/rflymad_frozen_ae_tcn_hedefli_karsilastirma.png`.
- **Kaynak:** `03_method_inventory.md`; `docs/sunum_hafta5_gorseller/captions.md`.

### 10. Split düzeltmesi sonucu değiştirir

- **Ana mesaj:** SEAD session split seed std’yi ±0,212’den ±0,012’ye indirdi; v3.1 group-safe sonuçları bu nedenle daha güvenilir ama daha düşüktür.
- **Görsel:** `artifacts/rfly_full/v2/visuals/06_pca_tsne_family_domain.png` + P1 group split Sankey.
- **Kaynak:** `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `artifacts/rfly_full/v2/visuals/README.md`.

### 11. Karmaşık model öğrenmeyebilir: magnitude baskınlığı

- **Ana mesaj:** SEAD’de trained–random ρ≈0,964 ve score–magnitude ρ≈0,965; üç AE ailesinin ortak başarısı kestirme çıktı.
- **Görsel:** P1 trained/random/magnitude scatter; hazır değilse sayısal korelasyon kartı ve “scatter yeniden üretilecek” notu.
- **Kaynak:** `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `scripts/four_dataset_probabilistic_gpu_v1_runner.py`.

### 12. Event evaluator düzeltmesi

- **Ana mesaj:** Bütün anomaly uçuşunu pozitif sayan eski evaluator, gerçek interval/event ölçümüne geçince başarı hikâyesi değişti.
- **Görsel:** `artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/02_detection_summary.png` + `03_event_operating_grid.png`.
- **Sayılar:** Rfly %56,1 / 0,768; SEAD %3,4 / 0; Attack 32,417 false event/h; ALFA %20 / 0 ama yalnız 0,131 normal saat.
- **Kaynak:** `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`.

### 13. ADS-B sıfırlaması ve gerçek trafik dersi

- **Ana mesaj:** %97,6 sentetik recall, 25,54 doğal alarm/saat karşısında başarı değildi; proje 10 Temmuz’da sıfırlandı ve gerçek trafik üzerinde bağlamsal fizik problemine döndü.
- **Görsel:** `artifacts/adsb/plots/reporting_summary/adsb_research_progression.png` sayı/dil kontrolü sonrası; küçük `docs/assets/adsb_simple_anomaly/05_route_examples.png`.
- **Kaynak:** `AGENTS.md`; `adsb/README.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.

## Perde III — Probabilistic v3.1 ve nihai hüküm (14–20)

### 14. Ortak probabilistic yaklaşım

- **Ana mesaj:** 32-adım causal normal geçmişten bir sonraki vektör için kanal başına μ ve σ tahmin edilir; anomali skoru masked Gaussian NLL’dir.
- **Görsel:** P0 Gaussian forecaster şeması.
- **Kaynak:** `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `configs/four_dataset_probabilistic_v31_rflymad.json`.

### 15. Negatif NLL neden normaldir?

- **Ana mesaj:** Sabit Gaussian terimi yoktur; σ<1 ve hata küçükken `0.5z²+logσ` negatif olabilir. Bu fit metriğidir, detector başarısı değil.
- **Görsel:** P0 üç-σ NLL eğrisi + `artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/01_training_and_selected_checkpoint.png`.
- **Kaynak:** `08_probabilistic_v31_deep_dive.md`; `scripts/four_dataset_probabilistic_gpu_v1_runner.py`.

### 16. v3.1 bilimsel sözleşme

- **Ana mesaj:** Train-only preprocessing, scenario/domain group ayrımı, validation-only seçim, bağımsız normal test ve 553 mühürlü final fault uçuş.
- **Görsel:** P1 group-safe split Sankey veya rol kartları: 260/90/91/557/553.
- **Kaynak:** `artifacts/four_dataset_probabilistic_v31/split_report.json`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.

### 17. Model ne öğrendi? Magnitude kapısı geçti, alarm kapısı geçmedi

- **Ana mesaj:** Epoch 30; trained–random ρ=0,3288, trained–magnitude ρ=0,3495, flag=false. Ancak flight AUC yalnız 0,641.
- **Görsel:** training plot + hazırlanırsa trained/random scatter.
- **Kaynak:** `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`.

### 18. Any-window matrisi neden yanıltıcı olabilir?

- **Ana mesaj:** Anomaly uçuşların %82,94’ünde alarm var; normal uçuşların da %84,62’sinde alarm var. Balanced accuracy %49,16.
- **Görsel:** `artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/03_flight_flag_matrix.png`.
- **Altbaşlık zorunlu:** “Development threshold; event confusion matrix değildir.”
- **Kaynak:** `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`.

### 19. B0 gerçek event sonucu — final bilimsel karar

- **Ana mesaj:** Referans 2-of-3 noktası %43,27 event recall ve 0,654 yanlış olay/normal saat; en yüksek recall %58,71 için 9,153 olay/saat gerekir. NO-GO.
- **Görsel:** `artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/01_recall_vs_false_event_rate.png` + `03_normal_test_alarm_burden.png`.
- **Kaynak:** `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.

### 20. Sonuç: ne öğrendik, ne yapmıyoruz, sırada ne var?

- **Ana mesaj:** Güvenilir anomaly araştırması doğru truth, bağımsız grup, kestirme kontrolü ve event alarm yükü gerektirir; mevcut Rfly v3.1/B0 işletim noktası yoktur ve final test kapalı kalır.
- **Üç madde:** (1) model karmaşıklığı tek başına çözmedi, (2) Real %7,81 ve Sensor %5,45 ana kör noktalar, (3) yalnız ön-kayıtlı B1 phase/domain + channel NLL/scale teşhisi meşru sonraki adım.
- **Görsel:** `artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/05_detected_motor_event_timeline.png` ve `artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/06_missed_real_sensor_timeline.png` yan yana; kapanışta “NO-GO ≠ değersiz sonuç”.
- **Kaynak:** `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.

## Sunucu için yasak kısa yollar

- “%82,9 event recall” deme; bu any-window anomaly-flight oranıdır. Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`.
- “v3.1 AP %92 ile başarılı” deme; prevalans ve %84,6 normal-flight alarmı verilmeden yanıltıcıdır. Kaynak: `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json` ve `artifacts/four_dataset_probabilistic_v31/split_report.json`.
- “v2’den v3.1’e model geriledi” deme; split/rol/truth sözleşmeleri farklıdır. Kaynak: `08_probabilistic_v31_deep_dive.md`.
- “Final testte başarısız oldu” deme; final fault test açılmadı. Development/B0’da NO-GO verildi. Kaynak: `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.
- `archive/` grafiklerini güncel sonuç diye kullanma. Kaynak: `AGENTS.md`.

## 12–15 dakikaya kısaltma

Slayt 5+10, 8+12, 14+15 ve 17+18 birleştirilerek 16 slayda düşürülebilir. Bilimsel omurgadan çıkarılmaması gerekenler: veri gözlenebilirliği, truth düzeltmesi, metrik ayrımı, group-safe split, magnitude kontrolü ve B0 event sonucu. Kaynak: bu planın dayandığı `01`–`09` envanter dosyaları.
