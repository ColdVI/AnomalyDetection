# Codex İnceleme Görevi — ADS-B Anomali Tespiti Durum Değerlendirmesi (2026-07-13)

## Görevin

Aşağıdaki tarihçeyi ve işaret edilen dosyaları OKU, mevcut durumu bağımsız gözle incele ve
"bundan sonra ne yapılmalı" sorusuna öncelik sıralı, gerekçeli bir öneri raporu yaz
(`docs/codex_review_findings_2026-07-13.md` olarak). Bu turda KOD DEĞİŞİKLİĞİ YAPMA —
salt-okunur analiz (dosya okuma, küçük doğrulama scriptleri çalıştırma serbest; mevcut
artifact/rapor/veri dosyalarını DEĞİŞTİRME). Sayı uydurma: her sayıyı ya repodaki bir
dosyadan al (yolunu belirt) ya da "ölçülmedi/varsayım" diye işaretle.

## Önce oku (sırayla)

1. `docs/DURUM_TESHIS_VE_YOL_HARITASI.md` — projenin öz-teşhisi + kademeli çıta (S0-S3) sistemi
2. `docs/decisions.md` — özellikle ADR-022, ADR-023, ADR-024, ADR-025 (en alttaki dördü)
3. `adsb/README.md` — yeni hattın kuralları ve Faz 0 durumu
4. `adsb/` modülleri: `features.py`, `rules.py`, `synthetic.py`, `diagnostics.py`,
   `scaling.py`, `windowing.py`, `segmentation.py`, `models/`
5. `scripts/adsb_train_baseline_models.py`, `scripts/adsb_evaluate_rule_scorer.py`,
   `scripts/adsb_generate_synthetic_dataset.py`, `scripts/adsb_plot_injection_timelines.py`
6. Sonuç dosyaları: `artifacts/adsb/models/baseline_training_report.json`,
   `artifacts/adsb/models/rule_scorer_report.json` (+ `_round1_madfloor` varyantı),
   `artifacts/adsb/plots/` (özellikle `injection_timelines/`)

## Tarihçe (günlük)

**2026-06-29 → 2026-07-09 (eski hat, arşivde):** 4 etiketli veri setinde (ALFA, UAV Attack,
UAV-SEAD, RflyMAD) 9 yöntem denendi; hiçbiri operasyonel Gate C'yi (recall≥0.30 @ FA≤2/saat)
geçemedi. AUC/AUPRC düzeyinde literatürle uyumlu sonuçlar alındı (ALFA LSTM-AE 0.872 AUPRC).
En iyi tekil sonuç: tek domain-seçilmiş feature (`itki_komutu`) 16-feature öğrenilmiş modeli
geçti (0.205→0.459 recall). SEAD'de kritik artefakt bulundu: kırpılmamış ölçekleme yüzünden
üç derin mimari de genlik-baskınlığına düştü (eğitilmiş skor ≈ eğitilmemiş ağ, ρ≈0.96).
Tamamı `archive/2026-07-10_legacy_non_adsb_ml/` altında.

**2026-07-10:** Kullanıcı repo'yu sıfırladı: eski ML hattı + iki koordinesiz ADS-B denemesi
(Claude'un `src/adsb`'si, Codex'in `src/adsb_behavioral`'ı) arşive kaldırıldı. KURAL: yeni
`adsb/` hattı arşivden KOD KOPYALAMAZ (fikir serbest). 3 gerçek ADS-B tar günü
(2026.02.28/03.01/03.16) mevcut `src/silver/parse_adsblol_historical.py` ile parse edildi:
256.150.550 satır, 638 Silver parça, `data/objectstore/silver/adsblol_historical/`. Yeni
hatta fizik-tutarlılık residual'ları yazıldı (vertical_rate/speed/heading/turn_bank), 4 mimari
(Dense-AE, LSTM-AE, USAD, LSTM-forecaster) + testler. İlk eğitim (ölçeksiz): genlik-baskınlığı
YENİ veride de tekrar-üretildi; `ClippedRobustScaler` (clip=5, train-only) yazıldı. USAD loss'u
ölçekleme+gradient-clipping'e RAĞMEN milyarlara patlıyor — HÂLÂ ÇÖZÜLMEDİ (ADR-022).

**2026-07-12:** `docs/DURUM_TESHIS_VE_YOL_HARITASI.md` yazıldı. Ana teşhis: yöntem değil çıta
yanlıştı — literatürün raporlamadığı operasyonel FA/saat barajı araştırma-boyutlu veriyle
kanıtlanmaya çalışıldı. Çözüm: kademeli çıta S0 (pipeline doğrulama) → S1 (tek kanal eşik) →
S2 (kural-bazlı kesin sinyaller: squawk 7500/7600/7700, nic/nac_p/sil) → S3 (operasyonel çatı).

**2026-07-13 (bugün):**
- Kullanıcı yönergeleri: model-eğitimi versiyon takibi fikri (basic, not alındı ama kurulmadı —
  `baseline_training_report.json` hâlâ her koşuda üstüne yazılıyor); sentetik korpus kalıcı
  üretilsin (var olanların üstüne YAZMADAN); parser envanteri çıkarılsın.
- Parser envanteri: tarihsel tar→Silver için TEK parser var (`parse_adsblol_historical.py`);
  arşivdeki Codex denemesi bile aynı `parse_trace_bytes`'ı import ediyordu; realtime parser
  ayrı kaynak için, birim dönüşümleri tutarlı.
- Kalıcı sentetik korpus (ADR-023): 60/638 parça, SEED=0 uçuş-bazlı 80/20 bölme, val'den 8910
  uçuş × 5 senaryo + temiz = 6 parquet, 765MB, `data/objectstore/synthetic/adsb/`. İki bug
  düzeltildi: ArrowStringArray shuffle güvenilmezliği (iki scriptte), uçuş-başına-dosya
  tasarımının Windows'ta çökmesi (~59k küçük dosya → senaryo-başına tek dosya).
- Kullanıcı: "model eğitelim, z-score güven skoru (0.95 gibi), tüm grafikler, yeni kolonlar
  için literatür taraması". Literatürden `altitude_source_residual` eklendi (baro-vs-jeometrik
  irtifa tutarlılığı; NIC/NACp/SIL alanları S2 için not edildi — Silver'da mevcutlar).
  z-score güveni: train medyan/MAD tabanı + normal CDF (p-değeri İDDİASI YOK).
- Eğitim turu (ADR-024, 60 parça, 2.85M train penceresi, USAD hariç): ÜÇ MİMARİ DE
  magnitude-domination'da işaretlendi (ρ=0.84-0.94), pooled AUC 0.552-0.572. Kök neden teşhisi:
  eşit-ağırlıklı `masked_mse` kolay/büyük ham kanalları (alt, hız, track) öğreniyor,
  asıl sinyali taşıyan residual kanallarını fiilen yok sayıyor. İki altyapı bug'ı düzeltildi
  (2.85M pencereyi tek forward'ta skorlama → 20.8GB OOM → batched; batch 64→512).
- Kullanıcı: "kural bazlı matematiksel formüllü penalty/reward sistemi — destekten öte ana
  odak olsun". `adsb/rules.py::ResidualRuleScorer` yazıldı: kanal-bazı robust z (train
  medyan/MAD), pen=min(max(0, z−3), 10), uniform ağırlık (ÖN-KAYITLI), pencere = satır
  ortalaması (NN'lerle aynı birim). İki tur (ADR-025): tur 1'de `altitude_source_residual`
  MAD=0 çıktı (irtifa 25ft kuantize → fark-türevi çoğu satırda tam 0), floor kanalı kıl-tetik
  yaptı (normal pencerelerin %93.8'i penalty aldı). Tur 2 genel kuralı: MAD=0 kanal
  kalibre-edilemez, skordan hariç. SONUÇ: **pooled AUC 0.600 — üç NN'i de geçti**;
  track_frozen 0.679 (NN 0.52), ground_speed_biased 0.727 (NN 0.743 ile başabaş),
  position_ramp_stealthy 0.519 (herkes kör), altitude_dropout 0.494 (kural için beyanlı
  kapsam dışı: NaN→katkı 0).
- Zaman-çizgisi grafikleri (`injection_timelines/`) KRİTİK değerlendirme hatasını açığa
  çıkardı: bozuk dosyanın onset-ÖNCESİ pencereleri (uçuşun ilk yarısı, birebir temiz) AUC'de
  y=1 sayılıyor → mükemmel dedektör bile ~0.75'te tavan yapar. Mevcut 0.727/0.679 fiilen
  tavana yakın. (RFLY-1'deki "whole-flight proxy → interval truth" dersinin tekrarı.)
- Downloads'ta YENİ 4. tar keşfedildi: `v2025.06.15-planes-readsb-prod-0-003.tar` (2.9GB,
  henüz parse edilmedi). Eğitim şimdiye kadar 60/638 parça (tek gün: 2026.02.28) ile yapıldı.

## Dokunulmaz kısıtlar (önerilerin bunları İHLAL EDEMEZ)

1. Sentetik/enjekte veri ASLA eğitime girmez — yalnız değerlendirme.
2. Sonuç görüldükten sonra parametre/eşik değişikliği YASAK (düzeltme yalnız ön-kayıtla,
   yeni tur olarak, eskisi raporda kalarak).
3. Kör-holdout tanımlanınca dokunulmaz; şu an TANIMLANMADI — tanım önerisi getir ama açma.
4. `archive/2026-07-10_rejected_adsb_attempts/` içinden kod kopyalanmaz.
5. Her eğitimden sonra `magnitude_domination_check` zorunlu.
6. Sentetik çıktı yolu "synthetic" içermek zorunda (path guard).
7. Sentetik recall TEK BAŞINA raporlanmaz — daima doğal-veri FA oranıyla birlikte.
8. MAD=0 kanal hariç tutulur (floor'lanmaz) — ADR-025 kuralı.
9. Commit'lere Co-Authored-By eklenmez.

## Araştırmanı/görüşünü istediğim sorular (öncelik sırası ÖNERİLEN, değiştirebilirsin)

1. **Pencere-etiket düzeltmesi tasarımı:** `label` kolonundan pencere etiketi nasıl türetilmeli?
   (herhangi bir onset-sonrası satır içeren pencere y=1 mi; oran eşiği mi; onset-öncesi bozuk
   pencereler dışlanmalı mı temiz mi sayılmalı?) Literatürde standart var mı (event-based
   precision/recall, TaPR, point-adjust'un bilinen sahteliği)? Somut öneri + gerekçe.
2. **Stealthy ramp için uçuş-içi CUSUM:** nedensel, train-kalibreli k/h parametreli,
   uçuş-başı sıfırlanan CUSUM tasarımı. Eski hattın causal-CUSUM dersleri
   (`archive/.../docs/ML_YETERSIZLIKLER_KAYDI.md`) burada nasıl uygulanır? 2 m/s ramp,
   3×MAD≈5.8 m/s satır eşiğinin altında — birikimle kaç dakikada görünür olur (kabaca hesapla)?
3. **Tam hacim stratejisi:** 638 parça + yeni v2025.06.15 tar'ı. ADR-023'te not edilen
   performans tuzağı (uçuş-başına tam-tablo taraması) dahil, bellek-güvenli ölçekleme planı.
   Yeni günün (10 ay önceki bir gün!) dağılım kayması riski değerlendirilsin — train mi test mi?
4. **Kural kanalını genişletme (S2):** Silver'da hazır duran `squawk`/`emergency`/`nic`/
   `nac_p`/`sil` alanlarıyla kesin-sinyal kural kanalı. Hangi eşik/mantık, hangi FA riski?
5. **NN hattı ne olacak:** ağırlıklı-loss (residual kanallara 3-5x) deneyi hâlâ değerli mi,
   yoksa kural+CUSUM ana odakken NN'ler beklemeye mi alınmalı? USAD'ın patlaması kökten
   çözülmeye değer mi, yoksa USAD elensin mi (gerekçeli öneri)?
6. **altitude_dropout kapsamı:** missingness/veri-kalite kanalı tasarımı (kural skorlayıcıya
   değil ayrı kanala) — normal veride ~%10 alt eksikliği varken FA üretmeden nasıl?
7. **Değerlendirme birimi:** pencere-AUC yeterli mi, event-onset recall + aktif-durum recall
   ayrımı (eski hattın dersi) bu aşamada mı eklenmeli, S1'de mi?
8. **Kör-holdout tanım önerisi:** hangi gün(ler)/dilim, hangi kurallarla (açmadan!).

## Rapor formatı

- `docs/codex_review_findings_2026-07-13.md`
- Her bölüm: bulgu → kanıt (dosya yolu/satır) → öneri → tahmini efor → risk.
- En sona tek sayfalık "önerilen sıra" (1-2 haftalık somut plan).
- Ölçmediğin hiçbir sayıyı ölçülmüş gibi yazma.
