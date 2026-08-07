# Bulgu ↔ Kod Eşlemesi

Bu çalışma boyunca tekrar eden üç yapısal bulgu, tek bir veri kümesinin veya
tek bir modelin garipliği değil — farklı veri kümelerinde, farklı yöntem
ailelerinde bağımsız olarak ortaya çıktı ve her biri artık koda gömülü,
zorunlu bir denetim/sözleşme hâline getirildi. Bu belge her bulgu için: ne
olduğunu, nasıl anlaşıldığını, sonucun ne olduğunu ve **hangi dosyadaki hangi
fonksiyonda ölçüldüğünü** listeler.

---

## 1. Genlik baskınlığı (magnitude domination)

**Ne olduğu:** Bir model örüntü öğrendiğini sanıyor — aslında yalnızca
sinyalin büyüklüğünü (genliğini) ölçüyor. Eğitilmiş bir ağın skoru, hiç
eğitilmemiş (rastgele başlatılmış) aynı mimarinin skoruyla neredeyse aynıysa,
model fiilen "büyük sayı = anomali" kısayolunu bulmuş demektir — öğrenilmiş
bir örüntü değil.

**Nasıl anlaşıldığı:** Eğitilmiş model skoru, (a) rastgele-başlatılmış aynı
mimarinin skoru ve (b) ham genlik (`||x||²`, maskeli) ile Spearman sıra
korelasyonu üzerinden karşılaştırılıyor. Korelasyon eşiğin (varsayılan 0.8)
üzerindeyse işaretleniyor.

**Sonuç:** UAV-SEAD veri kümesinde LSTM-AE, Dense-AE ve USAD'ın **üçü de**
işaretlendi. Kırpılmamış ölçekleme yüzünden birkaç aşırı-genlik kanaldan
sürüklenmişlerdi; göreli hata düzeltmesinden (kırpılı robust ölçekleme)
sonra yakalama oranı çöktü — çünkü altta gerçekten öğrenilmiş bir sinyal
yoktu, yalnız genlik farkı vardı. Eşit-ağırlıklı NN loss'u residual
kanallarını görmezden geliyordu (pooled AUC 0.55–0.57 seviyesinde kaldı);
buna karşı kural-tabanlı, fizik-residual penalty skorlayıcısı (`adsb/rules.py`)
üç ayrı sinir ağını da geride bıraktı (AUROC 0.765 / AUPRC 0.883) — ama aynı
eşikle doğal trafikte kullanılamaz seviyede yanlış alarm üretti (bkz. bulgu 3).

| Kod | Ne yapıyor |
|---|---|
| [`adsb/diagnostics.py`](adsb/diagnostics.py) → `magnitude_domination_check()`, `magnitude_only_score()` | Genel-amaçlı, mimariden bağımsız denetim fonksiyonu — **her eğitimden sonra zorunlu** |
| [`gecmis_calismalar/residual_v1/eval/sanity_gates.py`](gecmis_calismalar/residual_v1/eval/sanity_gates.py) → `s1_magnitude_gate()` | RESIDUAL-V1 tarafının eşdeğer eşikten-bağımsız akıl-sağlığı kapısı |
| [`gecmis_calismalar/rfly_full/ae_diagnostics.py`](gecmis_calismalar/rfly_full/ae_diagnostics.py) | RflyMAD-Full v2 Dense-AE taban çizgisi için post-hoc genlik tanısı |
| [`scripts/adsb_isolation_forest_magnitude_check.py`](scripts/adsb_isolation_forest_magnitude_check.py) | Isolation Forest kolu için aynı denetimin çalıştırıcısı |
| `tests/test_adsb_dense_autoencoder.py`, `test_adsb_lstm_autoencoder.py`, `test_adsb_usad.py` | İlgili modeller için regresyon testleri |
| [`demo/demo_calistir.py`](demo/demo_calistir.py) bölüm 6 | `magnitude_domination_check()`'i gerçek modülden çağırıp iki karşılaştırma gösterir |

`adsb/features.py`'nin **residual-tabanlı** olmasının nedeni de bu bulgu:
residual aritmetik bir özdeşlik olduğu için (bildirilen değer eksi
konum/irtifadan türetilen beklenen değer) genlik baskınlığı orada **yapısal
olarak oluşamaz** — residual sıfıra yakınsa kanal tutarlıdır, büyüklüğün tek
başına hiçbir anlamı yoktur. Genlik baskınlığı yalnız *öğrenilmiş* skorlarda
(NN, ridge) ortaya çıkabilecek bir artefakttır.

---

## 2. Proxy-etiket şişmesi

**Ne olduğu:** Kaba (coarse) bir etiketle ölçülen performans, gerçek
performansı abartır. "Bu uçuşun/pencerenin bir yerinde arıza var" bilgisi tüm
uçuşa/pencereye yayılınca, model gerçek arıza aralığını değil, arızanın
*civarındaki* kolay-ayrışan satırları da "doğru yakaladı" sayar.

**Nasıl anlaşıldığı:** Üç veri kümesinde **bağımsız olarak** keşfedildi —
farklı şekillerde ama aynı kök nedenle:
- **ALFA:** pencere-örtüşmesinden olay-başlangıcına (bir pencere olayla
  kısmen örtüşüyorsa "pozitif" sayılıyordu → olayın gerçek başlangıç anına
  göre yeniden tanımlandı).
- **RflyMAD:** uçuş-bütününden aralığa (etiket "bu uçuş arızalı" iken artık
  "bu aralık arızalı").
- **ADS-B:** dosya-proxy'sinden (bir tar dosyası "olaylı" mı) gözlenebilirliği
  doğrulanmış olay tablosuna (bir olay yalnız gerçekten *gözlenebilir* bir
  fark ürettiyse değerlendirme paydasına giriyor — `observable_eligible`).

**Sonuç:** Düzeltme sonrasında raporlanan sayılar her üç veri kümesinde de
düştü — beklenen ve gerekli bir düzeltme, "daha kötü model" değil "daha
dürüst ölçüm" anlamına geliyor.

| Kod | Ne yapıyor |
|---|---|
| [`adsb/truth.py`](adsb/truth.py) → `attach_event_truth_v2()`, `attach_clean_truth_v2()` | Satır/pencere truth-v2 sözleşmesi — `attack_onset` ≠ `observable_onset` ayrımı |
| [`adsb/evaluation.py`](adsb/evaluation.py) → `truth_event_table()`, `active_interval_coverage()`, `event_observability_denominators()`, `event_detection_metrics()` | Olayı gözlenebilirlik durumuna göre ayırır; recall paydası yalnız `observable_eligible` olaylardır |
| [`adsb/cusum_truth_v2_eval.py`](adsb/cusum_truth_v2_eval.py) | Dondurulmuş vektör-CUSUM için fail-closed truth-v2 değerlendirmesi |
| [`gecmis_calismalar/rfly_full/truth_audit.py`](gecmis_calismalar/rfly_full/truth_audit.py) | RflyMAD-Full v2 10Hz parse için truth denetimi (uçuş-bütününden aralığa geçiş) |
| [`gecmis_calismalar/residual_v1/ingest/splits.py`](gecmis_calismalar/residual_v1/ingest/splits.py) | Oturum-düzeyinde **grup-güvenli** bölme — aynı uçuş/oturum train ve testte aynı anda bulunamaz |
| [`scripts/adsb_build_synthetic_truth_v2_corpus.py`](scripts/adsb_build_synthetic_truth_v2_corpus.py) | ADS-B truth-v2 külliyatının üretim script'i |
| [`scripts/build_four_dataset_probabilistic_v3_group_splits.py`](scripts/build_four_dataset_probabilistic_v3_group_splits.py), [`scripts/audit_uav_sead_session_keys_v1.py`](scripts/audit_uav_sead_session_keys_v1.py) | Dört-veri-kümesi grup-güvenli bölme + UAV-SEAD oturum-anahtarı denetimi |
| [`raporlar/decisions.md`](raporlar/decisions.md) | Bu düzeltmelerin ADR kayıtları |

---

## 3. Bütçe–birim uyuşmazlığı

**Ne olduğu:** Yanlış-alarm bütçesi **saat** cinsinden tanımlanıyor, ama
değerlendirme penceresi genelde **tek bir uçuş** (birkaç dakika). Kısa
uçuşlarda saatlik bir bütçe pratikte "bu uçuşta neredeyse hiç alarm verme"
anlamına geliyor — sorun modelin ayrım gücünde değil, iki farklı zaman
biriminin (saat vs. uçuş) yanlış karşılaştırılmasında.

**Nasıl anlaşıldığı:** Doğal (enjeksiyonsuz) trafikte üretilen alarm
yükünü hem uçuş-oranı hem de **skorlanabilir uçuş-saati** paydasıyla ayrı
ayrı raporlayarak — ikisi arasındaki fark, kısa uçuş süresinin bütçeyi
fiilen ne kadar sıkılaştırdığını gösterdi.

**Sonuç:** Reddedilen ilk ADS-B denemesinde bu karışıklık sentetik kolay
anomalilerde yüksek event recall (~%97.6) ile doğal veride kullanılamaz
seviyede yeni-alarm oranı (~25.5 alarm/saat) arasında görünüşte çelişkili
bir tabloya yol açmıştı (bkz. [reddedilen_denemeler/README.md](reddedilen_denemeler/README.md)) —
yüksek sentetik recall, gerçek başarı olarak yorumlanamadı.

| Kod | Ne yapıyor |
|---|---|
| [`adsb/evaluation.py`](adsb/evaluation.py) → `scoreable_exposure()` | Bütçenin paydası — çakışan skor-destek aralıklarını birleştirir, hiçbir pencereyi iki kez saymaz |
| [`adsb/evaluation.py`](adsb/evaluation.py) → `natural_alert_burden()`, `alarm_episodes()` | Doğal (enjeksiyonsuz) referans üzerindeki nominal alarm yükü — bu FP olarak etiketlenmez, ayrı bir birim |
| [`configs/adsb_contextual_physics_v1_alarm_budget.json`](configs/adsb_contextual_physics_v1_alarm_budget.json) | Dondurulmuş bütçe parametreleri |
| [`scripts/adsb_report_s2_natural_burden.py`](scripts/adsb_report_s2_natural_burden.py), [`scripts/adsb_contextual_physics_v1_cusum_burden.py`](scripts/adsb_contextual_physics_v1_cusum_burden.py) | Doğal yük raporlama script'leri |
| [`demo/demo_calistir.py`](demo/demo_calistir.py) bölüm 5 | `natural_alert_burden()`'i gerçek modülden çağırıp uçuş-oranı + uçuş-saati birimlerini ayrı ayrı yazdırır |

---

## Yöntem disiplini kodda nerede zorunlu

| Kural | Kod |
|---|---|
| Eşik yalnız normal veriden dondurulur | [`adsb/cusum.py`](adsb/cusum.py) → `CusumConfig` eşiği zorunlu girdi alır, **arama metodu yoktur** |
| Enjekte edilmiş veri eğitime giremez | [`adsb/synthetic.py`](adsb/synthetic.py) → `save_synthetic_batch()` çıktı yolunun `"synthetic"` içerdiğini zorunlu kılar |
| Aynı oturum eğitim ve teste bölünemez | [`gecmis_calismalar/residual_v1/ingest/splits.py`](gecmis_calismalar/residual_v1/ingest/splits.py), [`scripts/build_four_dataset_probabilistic_v3_group_splits.py`](scripts/build_four_dataset_probabilistic_v3_group_splits.py) |
| Kapıyı geçmeyen aday ilerlemez | [`gecmis_calismalar/residual_v1/eval/sanity_gates.py`](gecmis_calismalar/residual_v1/eval/sanity_gates.py) → `require_s3_pass()` |
| Deney sözleşmesi önceden kaydedilir | [`gecmis_calismalar/rfly_full/contract.py`](gecmis_calismalar/rfly_full/contract.py), [`gecmis_calismalar/rfly_dl/config.py`](gecmis_calismalar/rfly_dl/config.py) |
| Koşum sonradan denetlenebilir | [`adsb/run_manifest.py`](adsb/run_manifest.py), [`gecmis_calismalar/residual_v1/tracking.py`](gecmis_calismalar/residual_v1/tracking.py) |
| Yapılandırma kilitlenebilir | [`gecmis_calismalar/uav_gnss/frozen_runner.py`](gecmis_calismalar/uav_gnss/frozen_runner.py) |

Üç bulgunun hepsini **tek bir dondurulmuş koşumda, gerçek modüllerle**
görmek için: [demo/demo_calistir.py](demo/demo_calistir.py) — özellikle
bölüm 5 (bulgu 2 ve 3) ve bölüm 6 (bulgu 1). Demo'daki sayılar sentetik
veriye aittir, bu belgedeki sayılar ise yukarıda kaynağı gösterilen gerçek
raporlardan alınmıştır.
