# RflyMAD probabilistic v3.1 B0 gerçek-event ön-kayıt eki

Tarih: 2026-07-28
Durum: **Sonuç görülmeden donduruldu**
Makinece okunur ızgara: `configs/rflymad_probabilistic_v31_event_grid.json`

## 1. Amaç ve değiştirilmeyen temel model

Bu ek, tamamlanmış `four_dataset_probabilistic_v31` RflyMAD B0 modelinin gerçek
zaman aralıkları üzerindeki alarm davranışını ölçer. Yeni eğitim, epoch uzatma,
checkpoint seçimi, model veya scaler değişikliği yapılmaz. Kabul edilen temel koşu
`artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728` ve
seçili checkpoint epoch 30'dur. Eğitim raporundaki magnitude-domination kapısı
`false` olarak kalmalıdır.

Bu çalışma mevcut modeli bir alarm sistemine dönüştürmenin ayrı bir değerlendirme
katmanıdır; uçuş-seviyesi `any-window` sonucu event metriği yerine kullanılamaz.

## 2. Veri rolleri ve erişim sınırı

| Rol | Kaynak | Bu ekte kullanım |
|---|---:|---|
| train | 260 | Okunmaz; model zaten tamamlandı |
| val_normal | 90 | Skor standardizasyonu, eşik/h ve bütçe kalibrasyonu |
| normal_test | 91 | Yalnız dondurulmuş noktaların false-alarm raporu |
| anomaly_dev | 557 | Yalnız dondurulmuş noktaların gerçek-event raporu |
| final_fault_test | 553 | Mühürlü; kimliği/yolu çözülmez, okunmaz, raporlanmaz |

Kalibrasyon **yalnız 90 normal validation uçuşuna** dayanır. `normal_test` veya
`anomaly_dev` sonucu hiçbir eşik, süre, CUSUM allowance/h, refractory ya da politika
seçiminde kullanılamaz. Normal-only optimizer eğitimi ile anomaly-dev üzerindeki
yarı-denetimli alarm değerlendirmesi birbirinden ayrıdır.

## 3. Dondurulmuş skor defteri

Model her görünür uçuş için sıralı pencere skorlarını tek bir Parquet defterine
aktarır. Asgari sütunlar şunlardır:

- `source_id`, `group_id`, `role`, `domain`, `fault_family`, `t_rel_s`
- `gaussian_nll_score`, `standardized_nll`, `target_magnitude`
- `truth_available`, `fault_active`, `condition_active`
- `fault_start_s`, `fault_end_s`, `truth_source`

`standardized_nll`, sadece `val_normal` skorlarından hesaplanan
`(score - median) / max(1.4826 * MAD, 1e-9)` dönüşümüdür. Test rollerinde yeniden
fit yasaktır. Model, scaler, split/contract, ön-kayıt, ızgara ve skor defteri
SHA-256 özetleri manifestte saklanır.

Gerçek fault/event truth, parse edilmiş RflyMAD zaman serisindeki
`fault_active` ve `condition_active` alanlarının hedef zamanlarına nedensel
hizalanmasından gelir. Bir anomalili uçuşun tamamını fault aralığı saymak ve
point-adjustment uygulamak yasaktır. Truth uyuşmazlığı olan satırlar unavailable
olarak raporlanır.

## 4. Dondurulmuş alarm aileleri

Makinece okunur dosyadaki bütün noktalar sonuç görülmeden dondurulmuştur:

1. Ham eşik + zamana dayalı persistence: 0.5, 1.0 ve 2.0 saniye.
2. Nedensel zamana dayalı K-of-N: yaklaşık 10 Hz için 2/3, 3/5 ve 5/10.
3. Validation-standardized NLL CUSUM: allowance 0.25, 0.5 ve 1.0.

Bütün ailelerde normal validation alarm bütçeleri 0.5, 1.0 ve 2.0 event/saat;
refractory süreleri 10 ve 30 saniyedir. Eşik ve CUSUM h aday quantile'ları,
gap/merge kuralları ve deterministik seçim kuralı JSON ızgarasındaki haliyle
uygulanır. Bütçeyi sağlayan aday yoksa sonuç `unavailable` olur; ekstrapolasyon
veya sonuç sonrası iyileştirme yapılmaz.

## 5. Raporlanacak metrikler

Normal test için false event/saat, false event/uçuş, en az bir false event içeren
uçuş oranı ve alarm-zaman oranı raporlanır. Anomaly-dev için gerçek event recall,
ilk tespit gecikmesi, zaman-ağırlıklı range precision ve range recall raporlanır.
Domain, fault-family ve group kırılımları korunur; satır, event ve uçuş düzeyleri
karıştırılmaz.

Belirsizlik 1.000 tekrarlı, seed 20260728 olan kaynak-uçuş düzeyi bootstrap ile;
normal/anomali, domain ve mümkün olduğunda fault-family tabakaları korunarak
hesaplanır. Test sonuçlarından tek bir kazanan seçilmez; dondurulmuş her nokta
raporlanır.

## 6. Karar kullanımı

Bu B0 değerlendirmesi modelde operasyonel alarm sinyali bulunup bulunmadığını
ölçer. Sonuç olumluysa sonraki B1 uçuş-modu/phase-conditioning tasarımına bağlam
sağlar; olumsuzsa aynen NO-GO/diagnostic olarak kaydedilir. Her iki durumda da
sonuç görüldükten sonra bu ek, temel model veya alarm ızgarası değiştirilmez.
