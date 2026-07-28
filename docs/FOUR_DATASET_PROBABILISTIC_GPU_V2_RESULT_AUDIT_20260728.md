# Four-dataset probabilistic GPU v2 — sonuç ve evaluator denetimi

Tarih: 2026-07-28

Statü: Bu belge artifact kabulünden önceki denetim kaydıdır. Kabul edilmiş
koşular, interval-truth değerlendirmesi ve nihai karar için
`FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md` esas alınır.

## Karar

V2 eğitimleri tamamlanmış deneyler olarak korunacaktır. Mevcut evaluator'ın
ürettiği pencere ve uçuş alarm metrikleri operasyonel event-tespiti kanıtı
değildir. Checkpointler yeniden eğitilmeden, ayrı
`four_dataset_probabilistic_event_eval_v1` sözleşmesi altında zaman damgalı
pencere skorlarına ve event kayıtlarına dönüştürülecektir.

Bu karar model sonuçlarını gördükten sonra v2 eşiğini, epochunu, splitini veya
özelliklerini değiştirmez. V2 raporları geçmiş deney kaydı olarak olduğu gibi
kalır.

## Kodla doğrulanan evaluator sorunu

`scripts/four_dataset_probabilistic_gpu_v1_runner.py` içindeki v2 tarafından
devralınan değerlendirme yolu:

1. anomalili etiket taşıyan bir uçuşun bütün pencerelerine `window_truth=1`
   verir;
2. uçuş skorunu pencerelerin maksimumu olarak tanımlar;
3. tek bir pencere eşiği geçtiğinde bütün uçuş için `detected=1` yazar.

Dolayısıyla mevcut `window ROC-AUC`, gerçek fault/attack zaman aralığını
lokalize etme metriği değildir. Anomalili olarak etiketlenmiş uçuşlardan gelen
pencere skorlarının normal uçuşlardan gelenlere göre sıralanmasını ölçer.

Normal-validation eşiğinin 0.995 pencere quantile'ı olması ve uçuş kararının
`any alarm` olması, uzun uçuşlarda çoklu-deneme etkisi yaratır. Pencereler
bağımsız olmasa da `1 - 0.995^N` hesabı etkinin yönünü açıklar. Bu nedenle
RflyMAD, UAV-SEAD ve UAV Attack'taki çok yüksek normal-uçuş alarm oranları
operasyonel false-event oranı olarak yorumlanamaz.

## Kullanıcı tarafından sağlanan v2 sonuç özetinin statüsü

Aşağıdaki değerler Colab çıktı özetinden alınmıştır; run artifactleri henüz
yerel hash doğrulamasından geçirilmemiştir:

| Veri seti | Window ROC-AUC | Window AP / prevalence | Max-score flight ROC-AUC | Normal uçuşta any-alarm |
|---|---:|---:|---:|---:|
| ALFA | 0.307 | 0.487 / 0.599 | 0.611 | 33.3% |
| UAV Attack | 0.105 | 0.728 / 0.877 | 0.000 | 100.0% |
| UAV-SEAD | 0.729 | 0.622 / 0.440 | 0.492 | 94.8% |
| RflyMAD | 0.727 | 0.807 / 0.588 | 0.631 | 98.5% |

Bu tabloya göre RflyMAD ve UAV-SEAD'de uçuş-kaynağı düzeyinde bir skor sıralama
sinyali vardır. Bunun gerçek anomali zaman aralığını yakaladığı henüz
gösterilmemiştir. ALFA bu splitte zayıf/kararsızdır. UAV Attack'ta tek normal
test uçuşunun anomalilerden daha yüksek skorlanması ağır domain/split sorunu
gösterir.

`flight_metrics.csv` üzerinden sonradan hesaplanan mean-score ROC-AUC değerleri
(UAV-SEAD 0.772, RflyMAD 0.915) yalnız keşif bulgusudur. Test görüldükten sonra
hesaplandıkları için resmî agregasyon seçimi veya yeni başarı sonucu değildir.

## Yerel truth uygunluk denetimi

| Veri seti | Mevcut yerel durum | Event değerlendirmesi |
|---|---|---|
| ALFA | Gold'da `t_rel_s`, `in_air`, uçuş etiketi var; onset Gold'a taşınmamış | Ham `failure_status` topic'i kaynak bazında eşlenip doğrulanmadan yok |
| RflyMAD | 6,605 parsed kaynakta `fault_active`, `condition_active`; manifestte başlangıç/bitiş ve truth kaynağı var | Hazır; disagreement ve domain ayrı raporlanmalı |
| UAV-SEAD | Gold'da `t_rel_s` ve uçuş etiketi var; interval etiketi Gold'da yok | Veri setinin zaman-bölgesi etiketleri eşlenip kapsam raporu geçince var |
| UAV Attack | Gold'da yalnız uçuş/saldırı etiketi var | Mevcut paketle yok; yalnız flight/source/domain değerlendirmesi yapılabilir |

## Yeni ölçüm katmanları

- Window: ham Gaussian NLL ve yalnız gerçek interval truth bulunan satırlarda
  threshold-bağımsız metrikler.
- Event: sabit persistence ızgarası, normal-validation false-event/saat
  bütçeleri, event recall ve detection delay.
- Flight: max, mean, q99 ve top-1% mean birlikte ve keşifsel olarak; event
  sonucunun yerine geçirilmeden.

Tam dondurulmuş değerler
`configs/four_dataset_probabilistic_event_eval_v1.json` dosyasındadır.

## Artifact kabul kapısı

Her veri seti için aşağıdaki v2 dosyaları birlikte sağlanmalıdır:

- `run_manifest.json`
- `scaler.json`
- `model_state.pt`
- `training_report.json`
- `flight_metrics.csv`

Yerel çalışma, base config/split hashleri, tamamlanma durumu, dataset adı,
özellik sırası ve model boyutları doğrulanmadan skor üretmeyecektir. Eksik veya
uyuşmayan run karantinaya alınır; diğer veri setlerinin incelemesini durdurmaz.

## Araştırma dayanağı

Bu karar, point-adjustment'ın sonuçları aşırı iyimser gösterebildiğini ortaya
koyan zaman-serisi anomaly değerlendirme çalışmaları ve güvenilir ölçümün
veri/metric bütünlüğüne bağlı olduğunu gösteren TSB-AD benchmark bulgularıyla
uyumludur. Yerel kararın asıl dayanağı ise yukarıda kod ve şema üzerinden
doğrulanan proje-içi kanıttır.
