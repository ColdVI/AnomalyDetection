# Ek — AI İnceleme ve Devir Kayıtları (yalnız arsiv)

> **Bu ek, ADS-B/RESIDUAL çalışması sırasında kullanılan AI kod-inceleme
> (Codex) prompt/bulgu belgeleri ile oturum devir (handoff) notlarını tek yerde
> toplar.** Bunlar iç çalışma sürecine ait araçlardır; mentöre teslim edilen
> `main` branch'inde yer almazlar, yalnız bu `arsiv` branch'inde saklanır.
> İçerik ve sayılar değiştirilmeden korunmuştur.

## İçindekiler

1. Codex Checkpoint (2026-07-13)
2. Codex GO (2026-07-13)
3. Codex İnceleme Prompt'u (2026-07-13)
4. Codex İnceleme Prompt'u — Ek (2026-07-13)
5. Codex İnceleme Bulguları (2026-07-13)
6. Codex İnceleme Bulguları — Ek (2026-07-13)
7. Claude Devir Notu — Contextual ADS-B (2026-07-14)


---

## Codex çalışma checkpoint'i — 2026-07-13

### Durum

2026-07-14 devam turunda Adım 6 ve Adım 7 tamamlandı. Karar kayıtları docs/decisions.md
içinde ADR-026–ADR-032 olarak bulunuyor. Adım 6 tam-hacim S2 v4 koşusu PASS, Adım 7 genel
gate kararı FAIL'dir. Çalışma kullanıcı sert durma noktasındadır.

Ana konfigürasyon dondurulmadı. Adım 8 Dense-AE/USAD başlatılmadı. Adım 9 holdout freeze
başlatılmadı; Downloads/raw/archive ve üç kör holdout tarının içeriği açılmadı.

### 2026-07-14 contextual-physics v1 devamı

Kullanıcı Step-7 FAIL sonrasında anomaly-channel ve normal-bağlam özelinde ayrı threshold
mekanizmasının yeni aday olarak uygulanmasını onayladı. `contextual_physics_v1`, eski Step-5
CUSUM veya koşullu Step-8'in devamı değildir. ADR-033 ve
`docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md` yapısal sözleşmeyi kaydeder.

Nedensel lagged flight phase, gerçek delta-t/cadence, sin/cos track, MAD=0 floorsuz strict
scaling, kanal-bazlı location/scale residual forecaster, hierarchical conditional conformal
calibration ve anomaly-profile özel temporal karar katmanı uygulandı. Sentetik fit/calibration,
implicit alpha ve sessiz score fusion fail-closed'dur. Hedefli testler 21/21, geniş ADS-B/parser
regresyonu 242 geçti / 1 bilinen frozen-hash testi deselect sonucundadır.

Bilimsel config veya threshold henüz donmadı. Kullanıcı toplam operasyonel alert-alpha/burden
bütçesini ve channel paylarını sayısal olarak tanımlamadan gerçek normal fit/calibration veya
truth-v2 evaluation başlatılmaz. Holdout havuzu hâlâ açılmadı.

#### Contextual normal-only eğitim sonucu

Kullanıcının eğitim onayıyla `20260714_contextual_physics_v1_train_v1` koşusu tamamlandı.
2.929 fit uçuş / 1.267.625 satır seçildi; beş epoch'un her biri 1.180.160 pencere gördü ve loss
0,795375'ten 0,708696'ya düştü. Ayrık 770 calibration diagnostic uçuşunda 332.510 pencere için
trained-vs-untrained/magnitude rho 0,649633/0,654240; magnitude flag false oldu. Checkpoint
strict yeniden yüklendi, tüm 9.546 parametre sonlu ve checksum indexi 5/5 PASS'tir. ADR-035 tam
kanıt/hash zincirini kaydeder.

Modelin durumu `trained_not_thresholded` olarak kalır. Sentetik fit/calibration sıfır; truth-v2,
development, rehearsal ve üçlü holdout havuzu açılmadı. Sıradaki zorunlu kullanıcı kararı sayısal
operasyonel alarm bütçesi ve channel paylarıdır.

### 2026-07-14 devam sonucu

- Deterministik 4-worker ve vektörize S2 runner gerçek 365.847 satırlık parçada yaklaşık
  1,5 saniye ölçüldü; hedefli segmentation/S2/parser testleri 54/54 geçti.
- artifacts/adsb/runs/20260714_step6_s2_natural_v4 koşusu 638/638 parça ve
  256.155.009 satırı 272,2 saniyede exit 0 ile tamamladı. Checksum indexi 2/2 doğrulandı.
- ADR-031 S2 doğal reason burden'ı kaydeder; residual/CUSUM ile birleştirme ve saldırı
  ground-truth iddiası yoktur.
- ADR-032 Adım 7 gate kararını FAIL olarak kapatır. CUSUM h=1 doğal alarm doygunluğu,
  üç NN'in magnitude-domination flag'i ve kayıp frozen features.py byte snapshot'ı ana
  konfigürasyon freeze'ini engeller.
- Corrected CUSUM evaluator hash kontrolünü gevşetmedi. Mevcut features.py yeni bir aday
  olarak baştan ön-kayıtlanmadan Step-5 frozen adayının yerine geçirilemez.

### Tamamlanan kanıtlar

- Adım 1 / ADR-026: değişmez run manifesti, fail-if-exists ve giriş/split provenance.
  Açık Silver toplamı 256.155.009 satır; eski belgeli 256.150.550 toplamıyla +4.459 fark
  çözülmemiş provenance notu olarak korunuyor.
- Adım 2 / ADR-027: truth-v2 korpusu 8.910 uçuş x (clean + 5 senaryo), toplam
  26.802.690 satır ve 646.160.578 byte. Sentetik veri eğitim/fit/calibration'a girmedi.
- Adım 3 / ADR-028: donmuş kural corrected truth-v2 üzerinde yeniden ölçüldü. Pooled
  AUROC/AUPRC 0.764883/0.883313; doğal temiz burden 4.808533 episode/saat.
- Adım 4 / ADR-029: iki eksenli causal Page CUSUM ve reset/missingness/prefix sözleşmesi.
  MAD=0 kanal floor uygulanmadan hariç tutuluyor.
- Adım 5 / ADR-030: 638 parça ve 256.155.009 satırlık tam-hacim streaming kanıtı
  tamamlandı. CUSUM h=1 skorlanabilir uçuşların yaklaşık yüzde 99'unu ve evaluable
  satırların yaklaşık yüzde 78–80'ini alarma soktuğu için ana freeze reddedildi.
  Engineering-advisory 12 episode/saat sınırı kullanıcı-onaylı operasyonel gereksinim
  değildir. Bootstrap upper doğruluk denetimi seçimi değiştirmedi.
- CUSUM truth-v2 değerlendirme kodu hazır ve bağımsız incelemeden PASS aldı. Fit,
  calibration, threshold sweep veya fusion yapmıyor; tek clean negatif havuzu, corrupt q0
  dışlama ve doğal-burden eşlemesi fail-closed.
- S2 kodunda bulunan iki blocker kapatıldı: state episode'ları explicit inactive satırda
  bölünüyor fakat sparse cadence tek başına bölmüyor; MESSAGE_GAP her satırda point event.
  Step 6 kod hash zinciri artık adsb/run_manifest.py dosyasını da kapsıyor. Kök doğrulamada
  S2/parser için 39 test, CUSUM truth-v2 kapsamı için 34 test geçti.

### Tamamlanmamış Adım 6 koşusu

İlk tam-hacim deneme dizini:

artifacts/adsb/runs/20260713_step6_s2_natural_v1

Koşu kullanıcı zaman önceliği nedeniyle yaklaşık 14 dakika sonra kontrollü durduruldu.
run_manifest.json yazılmıştı fakat final S2 raporu ve checksum indexi üretilmedi. Bu namespace
yeniden kullanılmamalı ve bilimsel sonuç sayılmamalıdır; yanında INCOMPLETE_DO_NOT_USE.md
işareti bulunur.

Ölçülen çalışma davranışı: 20 mantıksal işlemciden tek çekirdek kullanıldı; süreç canlıydı,
yaklaşık 814 CPU-s tüketmiş ve yaklaşık 906 MB working set kullanmıştı. İlk 50/638 ilerleme
satırı henüz gelmemişti. Hata veya veri sözleşmesi ihlali gözlenmedi; durdurma nedeni yalnız
duvar-saatiydi.

### Devam sırası

1. Kullanıcı Adım 7 FAIL kararını ve yeni çalışma yönünü onaylamalıdır.
2. Eski frozen features.py byte snapshot'ı bulunabiliyorsa corrected CUSUM değerlendirmesi
   aynı Step-5 adayı için tamamlanabilir.
3. Snapshot bulunamıyorsa mevcut code version yalnız yeni bir aday olarak; yeni ön-kayıtlı
   operasyonel burden bütçesi, natural calibration ve yeni namespace ile baştan çalıştırılabilir.
4. Kullanıcı onayı olmadan ana konfigürasyonu dondurma, Adım 8'e veya Adım 9'a geçme.

### Değişmez kısıt hatırlatması

- Sonuç görüldükten sonra aynı run içinde parametre/eşik ayarı yok.
- Sentetik veri train/fit/calibration'a girmez.
- archive içinden kod kopyalanmaz veya import edilmez.
- MAD=0 kanal floor'lanmaz; hariç tutulur.
- Sentetik recall doğal burden yanında raporlanır.
- Satır, event, uçuş ve uçuş-saati birimleri karıştırılmaz.
- Rehearsal geri-beslemesi ve holdout seçimi yapılmaz.
- Üç holdout tarı tek havuzdur; freeze ve unseal ayrı kullanıcı kararlarıdır.
- Commit mesajına Co-Authored-By eklenmez.

### Git yayın checkpoint'i

Kaynak/test/dokümantasyon kapsamı agent/adsb-rule-cusum-checkpoint dalında
26b0225 adsb rule cusum evidence checkpoint commit'i olarak push edildi:

origin/agent/adsb-rule-cusum-checkpoint

528 MB'lık generated baseline raporu ve generated parse logu Git indexinden çıkarıldı;
.gitignore ile gelecekte yeniden stage edilmeleri engellendi, yerel dosyalar silinmedi.
Generated ADS-B run/model/plot çıktıları da ignore kapsamındadır. Kullanıcı yalnız commit+push
istediği için draft PR açılmadı. Branch, origin/main'in önceki durumundan türetilmiştir;
main'deki 10 yeni commit ile rebase/PR senkronizasyonu sonraki yayın adımıdır.


---

## Codex — Onaylandı, Uygulamaya Geç

### Durum

`docs/codex_review_findings_2026-07-13.md` ve `docs/codex_review_findings_2026-07-13_addendum.md`
kullanıcı tarafından incelendi, sayı/iddia örneklemesi bağımsızca doğrulandı (satır sayısı,
gün-farkı aritmetiği, byte toplamı), **onaylandı**. Bu tur salt-okunur analizdi
(`docs/codex_review_prompt_2026-07-13.md`'nin kısıtladığı gibi) — o kısıt artık KALKTI:
kod/veri/artifact üretmeye geç.

### Uygulanacak sıra

Ana raporun "Önerilen sıra" tablosu (`docs/codex_review_findings_2026-07-13.md:624-642`)
**birebir**, tek değişiklikle: **Adım 9, addendum'daki üçlü-havuz metniyle DEĞİŞTİ**
(`docs/codex_review_findings_2026-07-13_addendum.md:198-205` — `docs/codex_review_findings_
2026-07-13.md:638` yerine geçer). Adım 1–8 aynen:

1. Gün 1 — run manifesti, fail-if-exists, giriş/split hash'leri, 256,155,009 vs 256,150,550
   farkının provenance notu (sessiz düzeltme yok).
2. Gün 1–2 — sentetik truth v2 (`injection_active`/`observable_changed`/`evaluable_truth` +
   event aralığı), mimari-destekli `q_w`, dropout exact block; v1 korunur, testler geçmeden
   skor koşusu yok.
3. Gün 2–3 — mevcut kural aynı donmuş kalibrasyonla corrected truth'ta yeniden skorlanır; eski
   NN JSON'ları tarihsel/label-bugged referans olarak kalır, "corrected" diye sunulmaz.
4. Gün 3–5 — doğu/kuzey hız residual'ı + causal CUSUM (prefix/reset/missingness testli).
5. Gün 5–7 — 2026-02-28 fit / 2026-03-01 development / 2026-03-16 donmuş rehearsal akışı.
6. Gün 6–8 — S2 (`declared_status`, `position_quality`, freshness/update-age,
   `altitude_availability`), residual penalty'den ayrı doğal episode/burden raporu.
7. Gün 8–9 — **Gate incelemesi**: truth testleri + magnitude şartı + corrected event metrikleri
   + doğal burden + günler-arası kararlılık + provenance eksiksizse rule+CUSUM/S2 konfigürasyonu
   dondurulur. **Bu geçmeden Adım 8/9'a geçilmez.**
8. Gün 9–10, KOŞULLU — yalnız Adım 7 stabilse Dense-AE paired control (sabit 4× treatment,
   MAD=0 kanallar hariç, sweep/fusion yok). USAD yalnız bu deney başarılıysa küçük smoke test;
   aksi halde aktif kapsam dışı.
9. Gün 10 — **(addendum'daki hâliyle)** kullanıcı kararıyla `v2024.09.01`, `v2025.02.15`,
   `v2025.06.15` raw tarları içerik açılmadan aynı `pool_id` altında raw path/byte/mtime/
   SHA-256/`scope_status` ile freeze edilir. Bu iki haftada varsayılan olarak tar üyesi
   listeleme, parse veya evaluation YOK. Açılış (unseal) ayrı, sonraki tek bir gate kararı.

### Dokunulmaz kısıtlar — hâlâ geçerli

`docs/codex_review_prompt_2026-07-13.md`'deki 9 madde aynen yürürlükte (özellikle: sentetik
veri asla eğitime girmez; sonuç görüldükten sonra parametre/eşik değişikliği yok; kör-holdout
havuzu Adım 9 dışında hiçbir şekilde açılmaz; MAD=0 kanal floor'lanmaz, hariç tutulur; sentetik
recall doğal-veri FA oranından ayrı raporlanmaz; commit'lere Co-Authored-By eklenmez).

### Sert durma noktaları (bunları geçmeden bana dön)

- **Adım 7 gate incelemesi tamamlanınca** — sonuçları getir, ana konfigürasyonu dondurmadan
  önce onay bekle.
- **Adım 9'daki üç tarın unseal'i** — bu prompt YALNIZ freeze/manifest işini yetkilendiriyor;
  içerik açma/parse/evaluation için AYRI bir kullanıcı kararı gerekiyor (addendum'un kendi
  metni de bunu şart koşuyor).
- Adım 8'in ön-koşulu (Adım 7 stabil mi) sağlanmazsa NN/USAD işini atla, bunu açıkça neden
  atlandığıyla birlikte raporla — sessizce geç yapılmasın.

### Raporlama

Her adım tamamlandığında `docs/decisions.md`'ye mevcut ADR formatında (son kayıt ADR-025) kısa
bir giriş ekle: karar → kanıt/sayı (dosya yolu ile) → açık madde. Sayı uydurma; ölçmediğini
"ölçülmedi" diye işaretle — bu tarama boyunca zaten iyi uyguladığın disiplin, aynı şekilde
devam etsin.


---

## Codex İnceleme Görevi — ADS-B Anomali Tespiti Durum Değerlendirmesi (2026-07-13)

### Görevin

Aşağıdaki tarihçeyi ve işaret edilen dosyaları OKU, mevcut durumu bağımsız gözle incele ve
"bundan sonra ne yapılmalı" sorusuna öncelik sıralı, gerekçeli bir öneri raporu yaz
(`docs/codex_review_findings_2026-07-13.md` olarak). Bu turda KOD DEĞİŞİKLİĞİ YAPMA —
salt-okunur analiz (dosya okuma, küçük doğrulama scriptleri çalıştırma serbest; mevcut
artifact/rapor/veri dosyalarını DEĞİŞTİRME). Sayı uydurma: her sayıyı ya repodaki bir
dosyadan al (yolunu belirt) ya da "ölçülmedi/varsayım" diye işaretle.

### Önce oku (sırayla)

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

### Tarihçe (günlük)

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

### Dokunulmaz kısıtlar (önerilerin bunları İHLAL EDEMEZ)

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

### Araştırmanı/görüşünü istediğim sorular (öncelik sırası ÖNERİLEN, değiştirebilirsin)

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

### Rapor formatı

- `docs/codex_review_findings_2026-07-13.md`
- Her bölüm: bulgu → kanıt (dosya yolu/satır) → öneri → tahmini efor → risk.
- En sona tek sayfalık "önerilen sıra" (1-2 haftalık somut plan).
- Ölçmediğin hiçbir sayıyı ölçülmüş gibi yazma.


---

## Ek Not — 2026-07-13 (codex_review_prompt_2026-07-13.md'ye ek)

Kullanıcı `docs/codex_review_findings_2026-07-13.md`'yi onayladı: önerilen 9 adımlı sıralama
AYNEN uygulanacak. Bu dosya yalnız TEK bir değişiklik ekliyor — sıralamanın kendisine dokunmuyor.

### Değişiklik: holdout adayı artık 1 değil 3 dosya

Önceki rapor, Downloads'ta tek bir kapalı-tutulması-gereken aday tanımlamıştı
(`v2025.06.15-planes-readsb-prod-0-003.tar`). Bugün (2026-07-13) o dosyayla AYNI klasörde
**iki yeni tar daha belirdi**:

| Dosya | Byte | mtime | Not |
|---|---:|---|---|
| `v2024.09.01-planes-readsb-prod-0.tar` | 2,084,157,440 | 2026-07-13 15:13:26 | ad kalıbı diğer 3 orijinal günle (`-prod-0.tar`, shard eki yok) aynı |
| `v2025.02.15-planes-readsb-prod-0.tar` | 2,146,856,960 | 2026-07-13 15:13:03 | aynı, shard eki yok |
| `v2025.06.15-planes-readsb-prod-0-003.tar` | 3,093,094,400 | 2026-07-13 12:02:04 | önceki raporda zaten kayıtlı; `-003` shard eki VAR — tam gün mü parça mı belirsizliği önceki raporda zaten not edilmişti |

Bu üç dosyanın hiçbiri açılmadı, listelenmedi, hashlenmedi. Yukarıdaki tablo yalnız dosya
sistemi metadata'sıdır (boyut + mtime) — önceki rapordaki "açmadan" tanımıyla aynı disiplinde.

### İstenen

1. Önceki raporun **9 adımlı planı değişmeden** uygulanır (adım 1-8 bu üç dosyaya dokunmaz).
2. **Adım 9** ("holdout freeze manifesti") artık TEK dosya için değil, **üç dosyalık bir
   havuz** için tasarlanmalı. Üçü de aynı şekilde: raw yol/byte/SHA-256/mtime salt-okunur
   manifestte kilitlenir, içerik açılmaz.
3. Üç günün takvimde birbirinden çok uzak olması (2024-09, 2025-02, 2025-06 — 2026-02-28 fit
   gününden sırasıyla ~18, ~12, ~8 ay önce) kapsamlı bir zamansal-kaymayı test etme fırsatı —
   ama BUNU nasıl kullanacağını (hangisi önce açılır, hepsi mi tek seferde mi, farklı roller mi
   verilir) ÖNCEDEN, sonuç görülmeden Codex'in kendi bir sonraki raporunda ÖNERMESİ isteniyor;
   bu dosya bir seçim DAYATMIYOR.
4. Ad kalıbındaki `-003` tutarsızlığı (yalnız 06-15'te var) ilk mekanik kontrolde (raporun
   adım 4'ü, "başarısız şema/parser dış-geçerlilik sonucu olarak kaydedilir" ilkesiyle)
   açıklığa kavuşturulmalı — üçü de aynı mantıkla mı üretilmiş, yoksa 06-15 gerçekten bir
   shard mı, netleştirilmeden "tam gün" varsayılmasın.

Geri kalan her şey (dokunulmaz kısıtlar, rapor formatı, sayı-uydurmama kuralı) önceki
prompt dosyasıyla aynı şekilde geçerli.


---

## ADS-B Anomali Tespiti Bağımsız İnceleme Bulguları

**Tarih:** 2026-07-13  
**Kapsam:** `docs/codex_review_prompt_2026-07-13.md` içindeki sekiz soru, mevcut kod,
testler, rapor JSON'ları, grafikler ve Silver/sentetik veri üzerinde salt-okunur doğrulamalar.  
**Bu turda yapılan değişiklik:** Yalnız bu rapor oluşturuldu; kod, veri ve mevcut artefaktlar
değiştirilmedi. Yeni `v2025.06.15` tar içeriği açılmadı, hashlenmedi veya parse edilmedi.

### Yönetici kararı

İlk iş yeni bir model veya eşik denemek değil, **interval truth ve değerlendirme sözleşmesini
düzeltmektir**. Mevcut sentetik dosya-kimliği etiketi, gerçekten temiz olan onset-öncesi
pencereleri pozitif sayıyor; ayrıca `altitude_dropout` reçetesinin gerçek bozulma bloğu ile
etiketli aralığı farklı. Bu iki sorun düzelmeden elde edilecek yeni AUC, loss ağırlığı veya
CUSUM karşılaştırması güvenilir olmaz.

Truth düzeltmesinden sonra ana teknik hat şu olmalı:

1. şeffaf residual kural skorlayıcısını değişmeden referans olarak korumak;
2. 2 m/s konum rampı için yönlü hız residual'ı üzerinde nedensel, uçuş-içi CUSUM eklemek;
3. doğal alarm yükünü üç günün tamamında bellek-güvenli, akışkan biçimde ölçmek;
4. S2'yi residual penalty'ye karıştırmadan `declared_status`, `position_quality` ve
   `altitude_availability` durum kanalları olarak kurmak;
5. NN'leri şimdilik dondurmak; yalnız ana hat istikrar kazandıktan sonra tek, ön-kayıtlı
   Dense-AE ağırlıklı-loss falsifikasyon deneyi yapmak; USAD'ı aktif plandan çıkarmak.

Mevcut kural skoru, hatalı whole-file etiketle pooled AUC `0.599760` üretmiş ve üç NN'in
`0.552371–0.572209` aralığını geçmiş olsa da bu değerlerin hiçbiri production/gate sonucu
değildir (`artifacts/adsb/models/rule_scorer_report.json:2169`,
`artifacts/adsb/models/baseline_training_report.json:5526769,11087236,16881661`).

### Kanıt ve sayı disiplini

Bu raporda “**inceleme ölçümü**” denilen değerler, 2026-07-13 tarihinde mevcut dosyalardan
salt-okunur olarak hesaplandı; kalıcı ara çıktı yazılmadı. “**Tahmin**” veya “**örnek hesap**”
etiketli değerler ölçüm değildir. Dış kaynaklar yalnız tanım ve yöntem dayanağıdır.

`artifacts/adsb/plots/` görsel denetiminde loss/ROC/AUC heatmap/confusion-score grafikleri
JSON özetleriyle çelişmedi. `injection_timelines/` onset öncesinde clean/corrupt örtüşmesini,
ground-speed ve track senaryolarında onset sonrası ayrışmayı, position rampın zayıf ayrışmasını
ve dropout'ta penalty ayrışmamasını görsel olarak doğruladı. Bu gözlem yeni sayısal metrik
olarak kullanılmadı.

#### Çapraz-kesit bulgu: provenance ve run değişmezliği eksik

**Bulgu.** Belgelerdeki Silver toplamı ile dosyaların gerçek footer toplamı uyuşmuyor; mevcut
rapor ve sentetik yazma yolları da aynı adı sessizce yeniden kullanabiliyor. Bu, model
versiyonlamasından önce giderilmesi gereken veri/run provenance borcudur.

**Kanıt.** İnceleme ölçümünde 638 Parquet footer'ı ve üç parse logu şu toplamı verdi:

| Gün | Parça | Satır | Log kanıtı |
|---|---:|---:|---|
| 2026-02-28 | 237 | 88,762,032 | `adsb_parse_02_28.log:475-476` |
| 2026-03-01 | 216 | 85,991,023 | `adsb_parse_03_01.log:433-434` |
| 2026-03-16 | 185 | 81,401,954 | `adsb_parse_03_16.log:371-372` |
| **Toplam** | **638** | **256,155,009** | footer toplamı |

`adsb/README.md:21` ve `docs/decisions.md:844` ise `256,150,550` yazıyor; fark **4,459
satırdır**. Bu fark bu incelemede açıklanamadı ve sessizce “düzeltilmemelidir”. Baseline scripti
sabit `baseline_training_report.json` yoluna yazar
(`scripts/adsb_train_baseline_models.py:256-259`); dosyanın inceleme anındaki boyutu
**528,285,357 byte** idi. Bunun ana nedeni her ROC noktasının JSON'a yazılmasıdır
(`scripts/adsb_train_baseline_models.py:161-170`); rule raporu ROC'u örnekleyerek saklar
(`scripts/adsb_evaluate_rule_scorer.py:111-120`). Sentetik path guard gerçek veriye yazmayı
engelliyor, fakat aynı `name.parquet` varsa üstüne yazmayı engellemiyor
(`adsb/synthetic.py:124-135`). Ayrıca `adsb/rules.py:13-14` hâlâ “MAD=0 floor” derken çalışan
kod kanalı dışlıyor (`adsb/rules.py:56-76`); uygulanacak kural kodda doğrudur, docstring eskidir.

**Öneri.** Her koşu için fail-if-exists çalışan değişmez bir `run_id` dizini oluşturulsun.
Manifest en az girdi yolu/byte/SHA-256/footer satırı/şema hash'i, split ve dışlanan flight
kimlikleri, git commit'i ve dirty-state, config, seed, feature sırası, scaler/kalibrasyon,
sentetik manifest hash'i, metric sözleşmesi ve çıktı checksum'larını içersin. Tam ROC gerekirse
sıkıştırılmış sütunsal dosyaya ayrı yazılsın; JSON yalnız özet ve seyreltilmiş eğri taşısın.
Sentetik v1 korunmalı; v2 yeni namespace'e ve `exist_ok=False` eşdeğeri bir guard ile yazılmalıdır.

**Tahmini efor.** 0.5–1 iş günü; ölçülmüş süre değildir.

**Risk.** Bu yapılmazsa aynı dosya adı altında farklı veri, truth veya eşiklerin sonuçları
karışır; post-hoc değişiklik ve sentetik sızıntısı sonradan denetlenemez.

### 1. Pencere-etiket düzeltmesi

**Bulgu.** Enjeksiyon kodu satır etiketi üretse de pencereleyici etiketi taşımıyor; iki
değerlendirme scripti de bozuk Parquet'teki **bütün** pencereleri pozitif yapıyor. Ayrıca
dropout ve ramp için `label != null` her zaman fiziksel olarak aktif bozulma demek değildir.

**Kanıt.** `_mark`, onset'ten uçuş sonuna kadar etiketi dolduruyor
(`adsb/synthetic.py:28-35`). `build_windows` yalnız `flight_id/t_start/t_end` metadata'sı
döndürüyor (`adsb/windowing.py:32-54`). Rule evaluator `r_scores` dizisinin tamamına bir
etiketi veriyor (`scripts/adsb_evaluate_rule_scorer.py:100-110`); NN evaluator aynı şeyi
yapıyor (`scripts/adsb_train_baseline_models.py:150-176`). Dropout yalnız onset sonrasındaki
rastgele alt bloğa NaN yazar, fakat onset'ten sona kadar işaretler
(`adsb/synthetic.py:73-86`). Rampın onset satırında `dt=0`, dolayısıyla fiziksel perturbasyon
da sıfırdır (`adsb/synthetic.py:97-108`).

İnceleme ölçümü: mevcut `ground_speed_biased.parquet`, `WINDOW=12`, `STRIDE=6`,
`MAX_GAP_S=60` ve mevcut pencereleyiciyle salt-okunur tekrar oynatıldı. Toplam **705,787**
pencerenin **344,883**'ü tamamen onset öncesi, **15,446**'sı sınırı kesen ve **345,458**'i
tamamen aktifti. Whole-file proxy etiketi altında fiziksel durumu bilen state-oracle skorun AUC
referansı **0.7556748707** çıktı. Bu değer evrensel/matematiksel bir tavan değil;
`1-0.5×(344883/705787)` ile elde edilen, bütün normal pencerelerin skor bakımından
exchangeable ve aynı normal skora sahip olduğu **state-oracle AUC referansıdır**. Uçuş fazını
veya başka bir confound'u skorlayan bir sistem bu değeri aşabilir ve yine de daha iyi anomaly
detector olmayabilir. Ölçüm kalıcı rapor sayısı değil, mevcut
`data/objectstore/synthetic/adsb/ground_speed_biased.parquet` üzerinde bu incelemenin
tekrar oynatmasıdır. Forecaster yalnız son dört hedef satırını skorladığı için
(`adsb/models/lstm_forecaster.py:32-39,80-91`) aynı pencerelerin destek sınıfları
**344,883 normal / 4,201 karışık / 356,703 tam aktif** oldu; modelden bağımsız tek “whole
window” etiketi bu mimari için de yanlıştır.

Literatürde tek zorunlu pencere etiketi standardı yoktur. Range/event-aware precision-recall
olay varlığına, overlap/cardinality'ye ve isteğe bağlı konum ağırlığına ayrı bileşenler verir
([Tatbul ve diğerleri, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html));
TaPR detection/portion skorları ve olay-sonrası ambiguous aralık tanımlar
([Hwang ve diğerleri, CIKM 2019](https://dl.acm.org/doi/10.1145/3357384.3358118)). Tüm anomalik
aralığı tek isabetten sonra doğru sayan point-adjust, rastgele skorları bile olduğundan iyi
gösterebilir
([Kim ve diğerleri, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680)).
Benchmark uygulamalarının farklı metrik ailelerini yan yana sunması da tek standart
olmadığını gösterir
([TimeSeAD, TMLR 2023](https://openreview.net/forum?id=iMmsCI0JsS)).

**Öneri.** Sentetik truth v2'de `event_id`, `event_type`, `attack_onset`,
`observable_onset`, `event_end` ve birbirinden ayrı satır-bazı `injection_active`,
`observable_changed` ve `evaluable_truth` alanları olsun. Dropout'ta yalnız gerçek NaN bloğu
`injection_active` olur; clean `alt` zaten NaN ise enjekte komut aktif olsa da gözlenebilir
değişim yoktur. Rampta sıfırdan farklı komutun Parquet/feature çözünürlüğünde gerçekten
değiştirdiği satırlar `observable_changed` olur. Eski v1 korpus ve raporlar korunmalı;
düzeltme yeni versiyon ve yeni run olarak çalışmalıdır.

Her skor için loss/score desteği \(S_w\) açıkça tanımlansın: rule/AE için bütün pencere,
forecaster için yalnız hedef satırları. `S_w` içindeki `evaluable_truth=True` satırlarda

\[
q_w = \frac{\sum_{t\in S_w} 1[\text{observable_changed}_t]}
{\sum_{t\in S_w}1[\text{evaluable_truth}_t]}
\]

hesaplansın; payda sıfırsa pencere unscoreable truth olarak ayrılsın. Ön-kayıtlı **birincil**
sentetik pencere etiketi `y_any = 1[q_w > 0]` olsun. İkincil steady-state değerlendirme yalnız
`q∈{0,1}` altkümesinde `q=1` pozitif/`q=0` negatif çalışsın; `0<q<1` pencereleri bu ikincil
metrikte yanlış negatif yapılmadan ayrıca raporlansın.
Sonuca bakarak yüzde eşiği seçilmesin. Düzensiz örneklemede satır oranı süre oranı değildir;
aktif-süre metrikleri timestamp ile ayrıca hesaplanmalıdır. Alarm zamanı nedensel olarak
`t_end` olmalıdır. Forecaster'ın ilk sekiz history satırı receptive field, son dört satırı
score desteğidir; anomalik history/normal target pencereleri `history_contaminated` geçiş
tabakası olarak raporlanmalı, doğrudan target-positive yapılmamalıdır.

Bozuk dosyanın onset-öncesi pencereleri scenario timeline'ında temiz-negatif/FP sanity check
olarak tutulmalıdır. Headline doğal alert burden ise yalnız clean/doğal exposure'ı **bir kez**
kullanmalı; bozuk dosya kopyaları bu paydaya yeniden eklenmemelidir. Sentetik AUC'nin negatif
havuzunda da clean Parquet'te birebir bulunan aynı pencereler ikinci kez ağırlıklandırılmamalıdır.
Bu dışlama “zor örneği silmek” değil, duplike temiz gözlemi tekilleştirmektir. Testlere
satır-truth → pencere desteği → etiket → metrik
entegrasyon testi ve forecaster destek testi eklenmelidir; mevcut windowing testleri yalnız
şekil/gap/NaN davranışını kapsıyor (`tests/test_adsb_windowing.py:25-54`).

**Tahmini efor.** Truth şeması, v2 üretim ve testler 1 iş günü; metrik/timeline yeniden koşusu
0.5–1 iş günü. Bunlar planlama tahminidir.

**Risk.** `q>0` çok kısa teması pozitif yapar; bu nedenle `q` tabakaları ve event düzeyi
metrik şarttır. Oran eşiğini mevcut sonuca göre seçmek post-hoc ayar olur. V1'in üstüne yazmak
provenance'i yok eder.

### 2. Stealthy ramp için nedensel CUSUM

**Bulgu.** CUSUM mevcut eşiklenmiş `rule_penalty` üzerinde değil, fiziksel işaretini koruyan
ham residual üzerinde kurulmalıdır. Mevcut `speed_residual` skaler büyüklük farkıdır; kuzeye
sabit 2 m/s konum kaymasının etkisi uçağın başına göre projekte olduğu için birikim sinyalini
zayıflatabilir.

**Kanıt.** Rule penalty, `|z|<=3` bölgesini tam sıfır yapıyor
(`adsb/rules.py:79-90`); dolayısıyla 3×MAD altındaki drift CUSUM'a verilirse bilgi zaten
kaybolmuştur. Mevcut train kalibrasyonunda `speed_residual` medyanı `0.1337676 m/s`, robust
MAD'i `1.9296692 m/s` ve 3×MAD'i yaklaşık `5.7890 m/s`'dir
(`artifacts/adsb/models/rule_scorer_report.json:22-25`). Reçete 2 m/s ramp kullanır
(`adsb/synthetic.py:89-101`).

İnceleme ölçümü: ilk Parquet row-group'unda tamamlanmış **2,095** uçuşta, parça sınırındaki
iki eksik uçuş dışlanarak mevcut north-bearing ramp tekrar oynatıldı. Skaler
`speed_residual` için uçuş-başı post-onset imzalı ortalama değişimin mutlak değerlerinin
uçuşlar-arası medyanı yalnız **0.84449 m/s**, kuzey yönlü vektör residual medyan değişimi
**-2.0 m/s** oldu. Bu bir teşhis ölçümüdür; `k/h` seçimi değildir.

Eski hattın kullanılacak **fikri**, kodu değil: full-flight MAD geçmiş skorları geleceğe göre
değiştirmiş ve causal ROC `0.878→0.611` düşmüştü
(`archive/2026-07-10_legacy_non_adsb_ml/docs/ML1_BULGULAR_VE_HATALAR.md:204-214`). Plan daha
sonra train-normal merkez/ölçek, moving-block bootstrap, reset/refractory ve prefix checksum
önermişti (`archive/2026-07-10_legacy_non_adsb_ml/docs/ML8_PLAN.md:198-208`). Bu rapor
arşivden kod taşımayı önermiyor.

**Öneri.** Önce bildirilen ground speed/track'i ve konum türevini doğu-kuzey bileşenlerine
ayıran nedensel residual üret:

\[
v^{rep}_E=gs\sin(\chi),\quad v^{rep}_N=gs\cos(\chi),\quad
e_E=v^{rep}_E-v^{pos}_E,\quad e_N=v^{rep}_N-v^{pos}_N.
\]

`v_pos`, yalnız `t` ve `t-1` konumlarından hesaplansın. Fit-normal altkümesinden sabit
medyan ve `1.4826×MAD` ile işaretli \(z_t\) üret; MAD=0 bileşeni dışla. Ön-kayıtlı iki taraflı
Page CUSUM adayı:

\[
C_t^+=\max(0,C_{t-1}^+ + \operatorname{clip}(z_t,-3,3)-k),
\]
\[
C_t^-=\max(0,C_{t-1}^- - \operatorname{clip}(z_t,-3,3)-k),\qquad
\max(C_t^+,C_t^-)>h.
\]

Bu formül tek bileşen/tek yön gösterimidir; gerçek alarm doğu/kuzey × pozitif/negatif dört
state'in birleşik maksimumıdır. Page'in özgün sürekli denetim yaklaşımı yöntem dayanağıdır
([Page, Biometrika 1954](https://doi.org/10.1093/biomet/41.1-2.100)). `k`, önceden seçilen
asgari fiziksel kaymanın normalize değeri için `k=δ*/2` olsun. Mevcut kuzey-yönlü 2 m/s
reçetede north bileşeni için \(\delta_*=2/s_{N,train}\) kullanılabilir; kerterizi bilinmeyen
2 m/s hedefte en büyük eksen bileşeni alt sınırı `2/√2 m/s`'dir veya dönüşten-bağımsız bir
vektör CUSUM ön-kayıtlanmalıdır. `h` sentetik recall'a göre değil, 2026-02-28'in uçuş-hash ile ayrılmış yalnız
normal kalibrasyon bölümünde, önceden yazılmış doğal alarm-episode/saat bütçesini sağlayan
moving-block bootstrap kuralıyla seçilsin; blok bootstrap seri bağımlılığı korumak içindir
([Künsch, Annals of Statistics 1989](https://doi.org/10.1214/aos/1176347265)). Alarm bütçesi
önceden dondurulmadan `h` nihai seçilemez. Dört state ayrı ayrı bütçelenmemeli; birleşik max
alarmı tek toplam doğal FA/saat bütçesine kalibre edilmelidir.

Uçuş başında, `on_ground=True`, out-of-order `dt<0` veya `dt>60 s` durumunda reset;
`dt=0` duplike/eşzamanlı satırlar nedensel birleştirilmeli veya state güncellemeden atlanmalıdır.
Kısa missingness'te state'i güncellemeden taşıma, uzun missingness'te reset uygulanmalı. Her
prefix aynı geçmiş skoru üretmeli; bu prefix-invariance testi zorunludur. Flight-adaptive
full-flight merkez/MAD kullanılmamalıdır. Clean korpus inceleme ölçümünde pozitif cadence
p25/medyan/p75 **1.61/3.96/11.34 s** idi; sample-bazı CUSUM yüksek-cadence uçakları kayırabilir.
Bu nedenle sabit-zaman binleri veya timestamp-aware güncelleme ön-kayıtlanmalı; en azından
natural burden cadence tabakalarında ayrıca raporlanmalıdır.

**Kabaca gecikme — örnek hesap, ölçüm değil.** Mevcut skaler MAD yalnız ölçek örneği alınırsa
`δ=2/1.929669=1.03645`, `k=0.51823` ve ideal beklenen artış yaklaşık `0.51823`/örnek olur;
dolayısıyla `E[N]≈1.9297h` örnektir. Clean korpusta pozitif ve `dt<=60 s` aralıklarının
inceleme-medyanı **3.96 s** idi. Yalnız örnek olarak `h=5` yaklaşık **9.65 örnek / 38.2 s /
0.64 dk**, `h=10` yaklaşık **19.3 örnek / 76.4 s / 1.27 dk** verir. `h=5/10` seçilmiş eşik
değildir. `E[N]≈h/(δ-k)` yaklaşımı iid/kararlı post-change mean shift, clipping'in etkin
olmaması ve robust standardizasyonun mean/σ gibi davranması varsayımlarına dayanır; yön
geometrisi, otokorelasyon, eksiklik, cadence ve resetler gerçek gecikmeyi uzatabilir.

**Tahmini efor.** Vektör residual, nedensellik testleri, kalibrasyon ve raporlama 1–2 iş günü.

**Risk.** Thresholded penalty üstünde CUSUM kurmak stealth sinyali geri döndürülemez biçimde
siler. Sentetik ramp ile `h` seçmek değerlendirmeyi eğitime dönüştürür. Reset/freshness hatası
uzun gap'leri sahte birikime çevirebilir.

### 3. Tam hacim ve yeni gün stratejisi

**Bulgu.** “Tam hacim” 638 parçayı ve bütün örtüşen pencereleri RAM'e `concat` etmek veya
NN'e yığmak olmamalıdır. Öncelikli amaç üç gerçek günde donmuş rule/CUSUM/S2'nin doğal alarm
yükünü akışkan ölçmek; yeni 2025-06-15 verisini train'e katmak değil, kör domain-shift adayı
olarak kapalı tutmaktır.

**Kanıt.** Baseline ve rule scriptleri seçili Parquet'leri bütün kolonlarla `pd.concat` eder
(`scripts/adsb_train_baseline_models.py:67-72`,
`scripts/adsb_evaluate_rule_scorer.py:54-58`). İlk 60 parça yalnız 2026-02-28 gününden,
inceleme ölçümünde **22,520,807 satır / 692,466,521 byte** idi. Mevcut rapor
**2,849,437 train** ve **705,969 validation** penceresi kaydeder
(`artifacts/adsb/models/baseline_training_report.json:2-3`). `build_windows` ham `X/M`'yi
float32 üretir (`adsb/windowing.py:35-36,44-65`), fakat scaler medyan/MAD'i float64'tür ve
transform ölçeklenmiş `X`'i float64'e yükseltir (`adsb/scaling.py:27-28,40-45`). Fonksiyon
dönüşü sonrasındaki scaled-`X` + `M` için teorik alt sınır ilk 60 parçada
**4,095,827,712 byte**; 60→638 doğrusal izdüşümü yaklaşık **43.6 GB**'dır. Preprocessing
anında ham `X` de yaşarken alt sınır **5,461,103,616 byte**, doğrusal izdüşüm yaklaşık
**58.1 GB** olur. Bunlar **tahmindir**; scaler geçicileri, DataFrame, liste, Torch tensor,
model ve optimizer belleği dahil değildir. Batched scoring
(`scripts/adsb_train_baseline_models.py:119-128`) hazırlama/eğitim yığılmasını çözmez.

Sentetik üretici her flight için tüm segment tablosunu tekrar filtreliyor
(`scripts/adsb_generate_synthetic_dataset.py:65-73`); bu tam hacimde karesel-benzeri tarama
yüküdür. Gün/parça metadata ölçümünde üç güne ait gerçek toplam, önceki bölümde verilen
**256,155,009 satırdır**.

**Öneri.** Sabit gün rolü:

- **2026-02-28:** normal fit + uçuş-hash ile ayrılmış normal kalibrasyon; sentetik v1'in
  kaynak flight kimlikleri her iki fit kümesinden kalıcı olarak dışlanır.
- **2026-03-01:** development/generalization; merkez/ölçek/`k/h` 2026-02-28 calibration'da
  ön-kayıtlı algoritmayla dondurulmuş halde sınanır. Başarısızlık görülürse aynı run içinde
  elle ayar yok; yeni hipotez, config ve versioned development run gerekir.
- **2026-03-16:** donmuş temporal rehearsal/dev-test. Bu incelemede dağılımları okunduğu için
  artık gerçek blind holdout değildir.
- **2025-06-15:** tek-atımlık kapalı domain-shift holdout adayı; train veya development'a
  alınmaz.

Akış tasarımı: değişmez input manifesti oluştur; yalnız gerekli kolonları oku; parça/aircraft
bazında segment→feature→row score→event özeti üret; tüm `X/M`'yi tutma. Her run'da bir
`(gün, source_id)` akışının tek parçada olduğu assertion ile doğrulansın; flight anahtarı
gün/source/sequence bağlamını içersin. Rule medyanı için disk destekli deterministik dış-sıralama
ve MAD için donmuş medyanla ikinci geçiş kullanılabilir. Approximate quantile seçilirse algoritma,
seed ve hata sınırı manifestte önceden dondurulmalıdır. MAD=0 kanal yine dışlanır. NN'e daha
sonra gerekirse `IterableDataset`, sınırlı shuffle buffer ve sabit flight-hash örneklemesi
kullanılmalı; “daha çok veri” bütün pencereleri belleğe almakla eşitlenmemelidir. Sentetik
üretimde flight `groupby` tek geçiş ve stream writer kullanılmalı, v1 dosyaları korunmalıdır.

**Tahmini efor.** Manifest/split guard 0.5 gün; streaming rule/S2 1–2 gün; tam koşu ve QA
0.5–1 gün; run/holdout kilidi 0.5 gün. Ölçülmüş çalışma süresi değildir.

**Risk.** Günler rastgele karıştırılırsa zamansal genelleme ölçülemez. Sentetik-korpus kaynak
flight'larını geniş fit setine geri almak sızıntıdır. Approximate median/MAD yöntemini sonuçtan
sonra değiştirmek kalibrasyonu post-hoc yapar.

### 4. S2: squawk/emergency ve konum kalite kanalları

**Bulgu.** `squawk/emergency`, bildirilmiş operasyonel durum; `NIC/NACp/SIL`, yayıncının
bildirdiği konum doğruluk/bütünlük kalitesidir. Bunlar “kesin saldırı ground-truth'u” değildir ve
residual penalty toplamına katılmamalıdır. Özellikle NIC/NACp/SIL için naif mutlak eşikler
doğal veride büyük alarm yükü üretir.

**Kanıt.** Parser `squawk`, `emergency`, `nic`, `nac_p`, `sil` ve ADS-B version alanlarını
Silver'a yazıyor (`src/silver/parse_adsblol_historical.py:115-123`). Fakat sparse `ac_dict`
alanını `last_ac` içinde ileri taşıyor ve gap/yeni leg'de sıfırlamıyor
(`src/silver/parse_adsblol_historical.py:75-82,92-123`); ardışık aynı satırlar bağımsız yeni
beyan değildir.

İnceleme ölçümü: tüm 638 parça yalnız ilgili kolonlarla akışkan tarandı; flight sınırı
`source_id` değişimi veya `gap>1800 s`, episode aktif duruma yükselen kenar ya da aktif
durumla yeni flight'a giriş olarak sayıldı. `emergency` için non-null ve literal `none`
dışındaki değerler aktifti; state her Parquet parçası başında sıfırlandı. Gün-bazı benzersiz
`source_id` sayıları **70,970 / 64,566 / 55,213** idi ve aynı `(gün, source_id)` değerinin
birden fazla parçaya yayılması ölçülmedi. Bu tanımla **542,461 flight segmentinde**:

| Durum | Satır | Episode |
|---|---:|---:|
| squawk 7500 | 502 | 8 |
| squawk 7600 | 2,360 | 16 |
| squawk 7700 | 11,399 | 16 |
| `emergency != none` | 44,771 | 157 |

Eşleşen satırlar `7500+unlawful: 318/502`, `7600+nordo: 2,220/2,360`,
`7700+general: 11,343/11,399` idi. Gap üzerinden taşınmış state, episode başlangıçlarının
sırasıyla **3/8, 1/16, 0/16** ve tüm emergency episode'larının **32/157**'sinde görüldü.
Dolayısıyla “üç ardışık satır” debounce'u üç bağımsız ADS-B güncellemesi anlamına gelmez.

Aynı taramada NIC<7 **12,655,396 satır (%4.9405)**, NACp<8 **5,068,287 satır
(%1.9786)** ve SIL<3 **22,737,205 satır (%8.8763)** kapsadı. Null sayıları NIC için
**152,479**, NACp/SIL için ayrı ayrı **3,579,446** idi. Bunlar event/FA sayısı değil,
satır sayısıdır. ADS-B version `3–7` için **19,994**, NACp `12–15` için **104** satır da
provider şeması açısından reserved/out-of-domain veri-kalite vakasıdır.

FAA dokümanında NACp≥8, NIC≥7 ve SIL=3, belirli ABD ADS-B Out/§91.227 bağlamındaki performans
referanslarıdır; dünya geneline saldırı eşiği değildir
([FAA AC 20-165B](https://www.faa.gov/documentlibrary/media/advisory_circular/ac_20-165b.pdf),
[FAA AC 90-114C](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_90-114C.pdf)).
readsb alan sözlüğü squawk'ı dört oktal rakam, emergency'yi kategorik durum; NIC/NACp/SIL'i
kalite/bütünlük alanları olarak tanımlar
([readsb JSON dokümantasyonu](https://github.com/wiedehopf/readsb/blob/dev/README-json.md)).
FAA 7500'ü unlawful interference kodu olarak tanımlar
([FAA AIP ENR 1.13](https://www.faa.gov/air_traffic/publications/atpubs/aip_html/part2_enr_section_1.13.html)).

**Öneri.** S2 iki ana reason-code kanalı üretsin:

1. `declared_status`: her kritik squawk ve her `emergency != none` bağımsız `declared`
   reason code'udur; `7500↔unlawful`, `7600↔nordo`, `7700↔general` aynı anda ve fresh ise
   `corroborated`. Tek alan/null/stale veya eşleşmeyen fakat açıkça karşıt olmayan durum
   `not_corroborated`; ancak iki alan da fresh olup farklı kritik durumları açıkça bildiriyorsa
   `contradictory` denir. Hiçbiri alarmı bastırmaz. `lifeguard/minfuel/downed/reserved` ayrı
   beyan tipleridir. Freshness, clear/expiry ve yükselen-kenar episode semantiği parser
   düzeltmesinden sonra ön-kayıtlanmalıdır.
2. `position_quality`: reserved/out-of-domain → `schema_invalid`; null → `missing`;
   standardın kodlanmış sıfırı → `reported_unknown_or_unavailable`; airborne ve uygun ADS-B
   version/scope altında NIC<7/NACp<8/SIL<3 → yalnız
   `below_faa_reference` advisory. Temporal düşüş ancak parser gerçek alan-güncellemesi,
   update timestamp/age ve mümkünse `sil_type/sda/nac_v` bilgisini koruduktan sonra
   kullanılmalıdır.

Bu kanallar anomaly penalty'ye toplanmasın; doğal episode sıklığı ve scoreable flight-hour
başına yük ayrı raporlansın. ATC/olay ground-truth'u olmadığı için gerçek 7500/7600/7700
beyanına “false positive” denmemeli. `not_corroborated/contradictory` veri tutarlılığı ve
freshness bulgusudur, saldırı etiketi değildir.

**Tahmini efor.** Parser freshness/şema kararı 0.5–1 gün; S2 episode mantığı ve doğal-yük
raporu 1 gün. Ölçülmüş süre değildir.

**Risk.** Forward-fill'i bağımsız yayın sanmak episode sayısını şişirir. FAA değerlerini
küresel kesin-anomali eşiğine çevirmek yüksek alarm yükü ve yanlış semantik üretir. Meşru acil
durumu saldırı diye etiketlemek değerlendirme ground-truth'unu bozar.

### 5. NN hattı ve USAD kararı

**Bulgu.** Kural+CUSUM ana odakken üç çalışan NN referans olarak dondurulmalıdır. Ağırlıklı
loss fikri bilimsel olarak hâlâ ucuz bir falsifikasyon deneyi değerindedir, fakat truth ve doğal
alarm ölçümü düzelmeden yapılmamalı; 3–5× taraması yapılmamalıdır. USAD'ın tam ölçekli çözümüne
şimdi yatırım yapılmamalıdır.

**Kanıt.** Mevcut üç NN'in train-vs-untrained ve train-vs-magnitude korelasyonları eşik
`0.8` üstündedir: Dense `0.8626/0.8956`, LSTM-AE `0.8428/0.8884`, forecaster
`0.9401/0.9228`; üçü de flagged'dir
(`artifacts/adsb/models/baseline_training_report.json:34-38,5526790-5526793,11087257-11087260`).
Loss bütün sonlu kanal/hücrelerin MSE'sini eşit toplar (`adsb/windowing.py:71-75`). Kural
round-2'de pooled `0.599760`, track-frozen `0.679362` üretmiştir
(`artifacts/adsb/models/rule_scorer_report.json:896,2169`); bu karşılaştırma mevcut etiket
hatası nedeniyle yalnız yönlendirici kanıttır.

Mevcut script yalnız özet JSON'u yazar; model checkpoint/state_dict veya pencere-bazı skor/meta
saklamaz (`scripts/adsb_train_baseline_models.py:256-259`). Dolayısıyla tarihsel NN raporu
dondurulabilir, fakat corrected truth geçmiş NN skorlarına geriye dönük uygulanamaz; yeniden
karşılaştırma yeni, versioned eğitim ve skorlama gerektirir.

USAD decoder'ları sınırsız lineer çıkışlıdır (`adsb/models/usad.py:35-42`); ikinci optimizer
loss'u yeniden-yapılandırma terimini negatif işaretle içerir (`adsb/models/usad.py:105-113`).
Bu adversarial amaç yapısaldır, fakat sınırsız çıkış sayısal patlamayı kolaylaştırabilir. Özgün
USAD uygulamasının decoder sonunda sigmoid kullanması bu farkı şüpheli kılar
([resmî USAD kodu](https://raw.githubusercontent.com/manigalati/usad/master/usad.py),
[USAD makalesi](https://doi.org/10.1145/3394486.3403392)); bunun mevcut patlamanın kanıtlanmış
tek kök nedeni olduğu **ölçülmedi**. Maskelenmiş girdide AE1 çıktısının tekrar encoder'a verilmesi
de ayrıca doğrulanmalıdır.

**Öneri.** Önceki NN raporlarını değiştirmeden sakla. Truth/CUSUM/S2 hattı donduktan sonra
Dense-AE için tek, iki-kollu falsifikasyon deneyi ön-kaydet: aynı seed/split/config ile
unweighted kontrol ve fixed-weight treatment. Fit-normal MAD'i pozitif raw kanallar `1`,
fit-normal MAD'i pozitif residual kanallar `4` — `4×`, kullanıcının önerdiği 3–5× aralığının
sabit orta noktasıdır; grid/sweep yok. Train MAD'i tam sıfır olan kanal model girdisi/loss/skordan
çıkarılır ve manifestte yazılır; floor yok. Mevcut splitte bu, `altitude_source_residual`ı
dışarıda bırakıp üç residual kanalı ağırlıklandırır
(`artifacts/adsb/models/rule_scorer_report.json:31-33`).

Örnek-bazı loss formülü de dondurulsun:

\[
L=\frac{\sum_{t,c} w_c M_{t,c}(x_{t,c}-\hat{x}_{t,c})^2}
{\sum_{t,c} w_c M_{t,c}}.
\]

Kanal-bazı train/validation MSE ayrıca raporlansın. Her eğitimden sonra mevcut
`magnitude_domination_check` zorunlu olsun. Falsifikasyon ancak treatment, paired control'e
göre aynı ön-kayıtlı doğal alarm-episode/saat bütçesinde corrected event recall/gecikmesini
iyileştirir ve `magnitude_domination_flagged == false` olursa geçer; bu, iki korelasyonun da
mevcut `0.8` sınırının altında olması demektir. Synthetic AUC tek başına gate değildir. Bu
sonuç fusion izni vermez: Dense standalone karşılaştırılır; OR/max/ağırlık gibi fusion ayrı ve
sonuç görülmeden dondurulmuş deney gerektirir. Ana hatta terfi için standalone rule+CUSUM
referansına karşı aynı burden'da fayda göstermelidir. Fayda yoksa NN hattını beklemeye al;
varsa aynı sabit ayarla yalnız LSTM-AE replikasyonu değerlendir.

USAD bu fazda “unvalidated/deferred” olarak kalsın. Dense falsifikasyonu başarısızsa USAD'ı
ADSB-1 aktif aday listesinden ele. Dense başarılı olursa bile önce özgün ölçek aralığına uyumlu
bounded-output + mask davranışı için küçük, sabit smoke test; sonlu loss ve magnitude kontrolü
geçmeden tam koşu yok. Bu yeni uygulama arşivden kod kopyalamamalıdır.

**Tahmini efor.** Dense iki-kollu deney 1–2 gün; USAD yalnız koşullu smoke 0.5 gün.

**Risk.** 3/4/5× sonuçlarına bakıp en iyisini seçmek post-hoc tuning'dir. AUC yükselirken doğal
alarm yükü kötüleşebilir. USAD debug'ı ana fizik/truth hattını geciktirebilir ve “çalıştı” sonucu
operasyonel fayda göstermeyebilir.

### 6. `altitude_dropout` ve missingness

**Bulgu.** Eksikliği **fiziksel rule residual penalty'sinde** `NaN→0 katkı` olarak bırakmak
doğrudur; bu global/NN missingness politikası değildir. Ayrı availability durumları gerekir.
Normal verideki yaklaşık yüzde on eksiklik neredeyse tamamen
`on_ground` kaynaklıdır. En yüksek özgüllüklü sentetik durum “airborne baro altitude yok,
geometric altitude var”dır; “iki irtifa da yok” ise doğal veride de görülen daha zayıf bir
availability uyarısıdır.

**Kanıt.** Rule NaN katkısını sıfırlar (`adsb/rules.py:79-90`). İnceleme ölçümü, mevcut
`data/objectstore/synthetic/adsb/clean.parquet` üzerinde:

- bütün satırlarda `alt` missing oranı **%10.5621**;
- bütün satırların **%10.5105**'i `on_ground=True` ve bunların **%100**'ünde `alt` missing;
- airborne satırlarda `alt` missing oranı **%0.05766** ve bu satırların hiçbirinde
  `alt_geom_m` sonlu değildi;
- iki altitude'un da airborne missing olduğu **2,305 satır / 37 flight** vardı; detector ile
  tutarlı `gap>60 s` reset semantiğiyle **68 run**, süre medyanı **43.055 s**, p95'i
  **1,209.908 s** idi.

Bu ölçümler 8,910 uçuşlu clean korpusa aittir
(`data/objectstore/synthetic/adsb/manifest.json:2-6`), tüm günlere genellenemez. Mevcut
dropout reçetesi 8,910 flight'ın **8,235**'inde sonlu `alt` değerini gerçekten değiştirdi;
bunların **7,973**'ünde katı baro-only state görüldü. Bu, detector recall'ı değil, eşiksiz
**strict-state observability coverage = %96.82**'dir; detector recall henüz ölçülmedi. Clean
korpusta katı state sıfır episode; iki interval ucu da airborne ve `0<dt<=60 s` alınarak
hesaplanan exposure **8,049.689 saat** idi. Bağımsız Poisson varsayımı altında
“üç kuralı” üst sınırı yaklaşık **0.000373/saat** olur; bu yalnız **model-varsayımlı örnek
hesaptır**, flight kümelenmesi ve tek-gün seçimi nedeniyle operasyonel güven sınırı değildir.

**Öneri.** Residual'dan ayrı availability durumları:

- `GROUND_ALT_NOT_APPLICABLE`: `on_ground=True`; abstain, alarm yok.
- `BARO_ALT_DROPOUT`: airborne, `alt` null, `alt_geom_m` sonlu; yüksek özgüllüklü event adayı.
- `ALL_ALTITUDE_UNAVAILABLE`: airborne, ikisi de null; availability/data-quality uyarısı,
  davranış anomalisi değil.
- `MESSAGE_GAP`: gözlem sessizliği; satır-içi altitude missingness'ten ayrı interval durumu.

Episode başlangıç/bitişi timestamp ile çıkarılsın; scoreable exposure ve flight-hour ayrı
raporlansın. Persistence eşiği mevcut natural run sürelerine veya sentetik sonuca bakıp elle
seçilmemeli; ön-kayıtlı normal-kalibrasyon prosedürü ve alarm bütçesiyle belirlenmelidir.
Sentetik v2 truth, dropout'un gerçek rastgele bloğunu kullanmalıdır; onset→uçuş sonu etiketi
kullanılmamalıdır.

**Tahmini efor.** Durum üretimi, exact-truth testi ve doğal-yük raporu 0.5–1 gün.

**Risk.** `alt is null` tek başına kural yapılırsa yerdeki beklenen eksiklik alarm üretir.
İki altitude da yokken persistence tek başına kesinlik sağlamaz; clean korpusta uzun doğal
run'lar vardır. Tek gün ve sıfır gözlemden production FA garantisi çıkarılamaz.

### 7. Değerlendirme birimi ve gate'ler

**Bulgu.** Pencere-AUC S0/S1 için yararlı bir tanı metriğidir, fakat tek başına yeterli değildir.
Event-onset, aktif durum ve doğal alarm yükü **şimdi**, truth düzeltmesiyle aynı turda eklenmelidir;
S3'e ertelenmemelidir. Satır, pencere, event ve flight birimleri ayrı tutulmalıdır.

**Kanıt.** README pencereyi pragmatik varsayılan seçiyor çünkü dört model pencere üstünde
çalışıyor (`adsb/README.md:33-35`); rule ise önce satır penalty'si üretip sonra sırf NN ile
karşılaştırma için pencere ortalaması alıyor
(`scripts/adsb_evaluate_rule_scorer.py:43-51`). Mevcut JSON'larda event gecikmesi veya
alarm-episode/scoreable-hour yoktur. `confidence_threshold=0.95`, train score'un robust
z-değerine normal CDF uygulanan bir extremeness dönüşümüdür; kalibre olasılık/p-değeri değildir
(`docs/decisions.md:939-942`).

İnceleme-türevi örnek: aynı 705,787 clean pencere için `conf>=0.95` rule confusion
matrix'inde **254,687** pencereyi işaretler, yani **%36.0855**
(`artifacts/adsb/models/rule_scorer_report.json:49-56`); Dense-AE **156,204**, yani
**%22.1319** işaretler
(`artifacts/adsb/models/baseline_training_report.json:53-60`). Bunlar birbirine örtüşen
pencere oranlarıdır, FA/saat değildir ve clean korpus da yalnız seçilmiş bir güne dayanır.

**Öneri.** Yeni metric contract:

| Birim | Birincil rapor | Yorum |
|---|---|---|
| Pencere | AUROC + AUPRC; `q=0`, `0<q<1`, `q=1` tabakaları | yalnız tanı/karşılaştırma |
| Aktif aralık | zaman-ağırlıklı coverage ve precision; event-macro coverage | point-adjust yok |
| Event | event recall, ilk alarm gecikmesi, ön-kayıtlı gecikme bütçesinde recall | tek isabet tüm aralığı TP yapmaz |
| Doğal trafik | alert episode/scoreable flight-hour, clean flight'ların işaretlenen oranı | sentetik recall ile daima yan yana |
| Flight | event içeren flight recall ve flight başı alarm yükü | pencere metriğine karıştırılmaz |

Scoreability/eligibility paydası ayrıca raporlansın. Alarm episode merge/debounce/refractory
kuralı sonuç görülmeden önce dondurulsun. `conf>=0.95` mevcut raporda yalnız diagnostic isimle
korunsun; yeni operasyon eşiği normal calibration'da doğal alarm bütçesinden seçilsin. Doğal
veride olay etiketi yoksa “false alarm” yerine dürüstçe “nominal alert burden” denilsin; ancak
sentetik recall ile eşleştirilen clean referans için aynı donmuş kuralla alert burden mutlaka
verilsin. NAB gibi onset-duyarlı çerçeveler gecikme raporlamasının dayanağı olabilir
([Numenta Anomaly Benchmark](https://arxiv.org/abs/1510.03336)); tek bir literatür metriği domain
sözleşmesinin yerini almaz.

**Tahmini efor.** Metric contract, eventizer ve testler 1–2 iş günü; truth işiyle paralel
yürütülebilir.

**Risk.** Pencere bağımlılığını yok saymak güven aralıklarını aşırı dar gösterir. Point-adjust
skoru yapay yükseltir. Sentetik recall'ı doğal alarm yükü olmadan sunmak dokunulmaz kısıtı ihlal
eder.

### 8. Kör-holdout tanımı

**Bulgu.** En güçlü mevcut aday, Downloads'taki `v2025.06.15-...-003.tar` dosyasının tamamını
tek-atımlık kapalı domain-shift holdout olarak ayırmaktır. Fakat bu gün eğitimden daha eskidir;
ileri-zaman deployment drift'i değil, **geriye dönük temporal/source transfer** ölçer. Dosya adı
tek tam gün yerine bir shard'ı da gösterebilir; içerik, freeze öncesinde kontrol edilmemelidir.

**Kanıt.** Yalnız dosya metadata'sına yapılan inceleme ölçümü:
`C:\Users\PC_5812_YD26\Downloads\v2025.06.15-planes-readsb-prod-0-003.tar`,
**3,093,094,400 byte**, mtime `2026-07-13 12:02:04`. Tar açılmadı. Tarih,
2026-02-28 fit gününden **258 gün eskidir** (takvim farkı). 2026-03-16 ise bu incelemede footer
ve S2 dağılımları için okunduğundan blind sayılamaz.

**Öneri.** Kullanıcı bu dosyayı açıkça holdout seçerse, açmadan önce aşağıdaki protokol
dondurulsun:

1. Raw yol, byte, SHA-256, beklenen tarih/kaynak ve erişim günlüğü salt-okunur manifestte
   kilitlensin. SHA-256 bütün raw byte'ları okur; burada “açmamak”, tar üyelerini
   listelememek/decompress etmemek ve içerik istatistiği çıkarmamak, yalnız loglanmış hash
   geçişine izin vermek demektir.
2. Parser commit/şema, gerekli kolonlar, unit dönüşümleri, flight/window/interval truth,
   scoreability, feature ve reason-code listesi, scaler/rule kalibrasyonu, CUSUM `k/h/reset`,
   eşikler, event merge/refractory ve metric/gate'ler hashlenerek dondurulsun.
3. Holdout normal Silver dizinine yazılmasın. `parse_local_tar()` mevcut target'a append edip
   rerun'da duplicate üretebilir (`src/silver/parse_adsblol_historical.py:192-202,267-268`);
   standart `run()` ise mevcut Silver prefix'ini önce siler ve tarı belleğe indirir
   (`src/silver/parse_adsblol_historical.py:212-229`). Bu nedenle parser'a ayrı output-target
   desteği veya izole object-store/prefix gerekir. Namespace bir kez yazılır, doğrulanır,
   sonra sealed/read-only yapılır; yazım anında “salt-okunur” denmez.
4. İlk mekanik kontrolde `-003` kapsamı/timestamp/şema doğrulansın. Başarısız şema/parser
   dış-geçerlilik sonucu olarak kaydedilsin; aynı holdout'a bakarak feature/eşik/parser tamiri
   yapılıp yeniden “blind” sonuç üretilmesin.
5. Üye/satır dahil-etme ve parse-hata kuralları önceden dondurulsun. Primary sonuç bu
   sözleşmedeki bütün kapsamda raporlansın; toplam tar üyesi, başarılı/başarısız parse,
   dışlanan satır/flight ve reason code'ları zorunlu attrition tablosu olsun. İyi görünen saat
   veya aircraft seçilmesin. Development günlerinde hiç görülmemiş `source_id` altkümesi
   ön-kayıtlı secondary olabilir, primary'nin yerini alamaz.
6. Gerçek anomaly etiketi yoksa primary sonuç doğal alert/event burden ve scoreability'dir.
   Sentetik enjeksiyon yalnız donmuş reçete/truth ile secondary olarak ve aynı doğal burden'ın
   yanında raporlanabilir; sentetik asla fit/kalibrasyona girmez.
7. Bu holdout yalnız truth, rule+CUSUM/S2, doğal alarm bütçesi ve run manifesti development'ta
   donduktan; insan gate kararı kaydedildikten sonra bir kez açılır.

İleri deployment iddiası için daha sonra **2026-03-16 sonrasından** prospektif bir gün ayrıca
kilitlenmelidir; hangi gün olduğu şu an ölçülmedi/seçilmedi.

**Tahmini efor.** Freeze manifest/protokol 0.5 gün; kullanıcı gate'i sonrası tek parse/eval
0.5–1 gün. Holdout'u açma bu raporun 1–2 haftalık varsayılan planına dahil değildir.

**Risk.** Hash/freeze öncesi tar envanteri bile eşik veya şema kararını etkileyebilir. Eski günü
“gelecek drift” diye sunmak dış-geçerliliği abartır. Holdout'a özel parser düzeltmesi tek-atımlık
körlüğü tüketir.

### Dokunulmaz kısıt denetimi

**Bulgu.** Mevcut hat, sentetik train sızıntısı için runtime assertion ve sentetik path guard
taşıyor; magnitude check mevcut üç eğitimde çalışmış; çalışan rule kodu MAD=0 kanalı dışlıyor.
Eksik kalan iki temel koruma run fail-if-exists/versioning ve sentetik sonucun doğal
alert-burden ile zorunlu eşleştirilmesidir. Blind holdout henüz tanımlı değildir.

**Kanıt.** Train/synthetic flight kesişimi hata veriyor
(`scripts/adsb_train_baseline_models.py:88-100`,
`scripts/adsb_evaluate_rule_scorer.py:69-75`); path guard `synthetic` sözcüğünü zorunlu kılıyor
(`adsb/synthetic.py:124-132`); magnitude check forecaster dahil çağrılıyor
(`scripts/adsb_train_baseline_models.py:205-212,223-230,246-252`); MAD=0 `continue` ile
dışlanıyor (`adsb/rules.py:56-76`). Doğal FA/saat mevcut raporlarda yoktur. Holdout'un henüz
tanımlanmadığı `docs/decisions.md:824,883` içinde de kayıtlıdır.

**Öneri.** Dokuz kısıt run manifestinde makinece kontrol edilen gate listesi olsun: synthetic
ID/path ayrımı; fail-if-exists; result sonrası config hash değişmezliği; holdout access log;
archive import taraması; her train'de magnitude JSON; synthetic recall yanında doğal burden;
MAD=0 exclusion assertion; commit mesajında `Co-Authored-By` yokluğu. Arşivden yalnız bu raporda
belirtilen yöntem dersi alındı, kod kopyalama önerilmedi.

**Tahmini efor.** Çapraz-kesit guard/testlerin temel sürümü 0.5–1 gün.

**Risk.** İnsan hafızasına bırakılan kısıt, yeni script/run yolunda sessizce atlanır.

### Önerilen sıra — 1–2 haftalık somut plan

Aşağıdaki sürelerin tamamı **planlama tahmini**, ölçülmüş süre değildir. Holdout kapalı kalır.

| Sıra / hedef gün | Çıktı ve geçiş koşulu |
|---|---|
| **1 / Gün 1** | Run manifesti, fail-if-exists, giriş/split hashleri ve makinece kısıt gate'leri. 256,155,009 vs 256,150,550 farkı provenance notuyla kayıt altına alınır; sessiz düzeltme yok. |
| **2 / Gün 1–2** | Sentetik truth v2 (`injection_active`, `observable_changed`, `evaluable_truth` + event aralığı), mimari-destekli `q_w`, dropout exact block; v1 korunur. Unit + entegrasyon testleri geçmeden skor koşusu yok. |
| **3 / Gün 2–3** | Mevcut rule aynı donmuş kalibrasyonla corrected truth üzerinde yeniden skorlanır. NN checkpoint/skorları saklanmadığı için eski NN JSON'ları tarihsel, label-bugged referans olarak korunur ve corrected diye sunulmaz. Window AUC/AUPRC tanı; event recall/gecikme, aktif coverage ve doğal alert burden zorunlu. |
| **4 / Gün 3–5** | Doğu/kuzey hız residual'ı ve causal CUSUM; prefix/reset/missingness testleri. `k` fiziksel 2 m/s hedefinden, `h` yalnız ön-kayıtlı normal kalibrasyon + doğal alarm bütçesinden. |
| **5 / Gün 5–7** | 2026-02-28 fit/calibration akışı; 2026-03-01 development; 2026-03-16 donmuş rehearsal. Bütün süreç streaming; sentetik-kaynak flight'ları fitten dışarıda. |
| **6 / Gün 6–8** | S2 `declared_status` ve `position_quality`, parser freshness/update-age kararı; ground/baro-only/all-altitude/message-gap ayrımlı `altitude_availability`. Residual penalty'den ayrı doğal episode/burden raporu. |
| **7 / Gün 8–9** | Gate incelemesi: truth testleri, magnitude şartı, corrected event metrikleri, doğal burden, günler arası kararlılık ve provenance eksiksizse ana rule+CUSUM/S2 konfigürasyonu dondurulur. |
| **8 / Gün 9–10, koşullu** | Yalnız ana hat stabilse Dense-AE paired control + sabit `4×` treatment; train MAD=0 kanallar iki kolda da hariç, sweep/fusion yok. Magnitude flag kalkmaz veya aynı doğal burden'da treatment faydası yoksa NN beklemeye alınır. USAD yalnız bu deney başarılıysa küçük bounded-output/mask smoke; aksi halde aktif kapsamdan çıkarılır. |
| **9 / Gün 10** | Kullanıcı kararıyla 2025-06-15 raw tar için **açmadan** holdout freeze manifesti/SHA-256 hazırlanır. Bu iki haftada varsayılan olarak parse/açma yok; açılış ayrı, kayıtlı gate kararıdır. |

**İki hafta sonunda beklenen karar:** “yüksek AUC” değil; interval truth'u doğru, causal,
natural alert burden'ı ölçülmüş, günler arası tekrar oynatılabilir bir rule+CUSUM/S2 baseline.
Bu baseline yoksa NN/USAD veya blind-holdout açmak sıralama hatasıdır.


---

## Üç Dosyalı Kör-Holdout Havuzu — Ek İnceleme Bulguları

**Tarih:** 2026-07-13  
**Dayanak:** `docs/codex_review_prompt_2026-07-13.md` ve
`docs/codex_review_prompt_2026-07-13_addendum.md`  
**İlişki:** Onaylanmış `docs/codex_review_findings_2026-07-13.md` değiştirilmedi. Bu rapor
yalnız Adım 9'daki tek-dosyalı holdout tanımını üç-dosyalı havuza genişletir; Adım 1–8 ve
dokuz dokunulmaz kısıt aynen kalır.

### Yönetici kararı

Üç tarın hiçbiri eğitim, kalibrasyon, development veya parser geliştirme verisi yapılmamalıdır.
Önerilen kullanım, dosyaları sırayla açıp her sonuçtan sonra karar vermek değil, **tek mantıksal
açılışta, ara sonuçları ambargolu ortak bir holdout batch'i** olarak tüketmektir.

Üç dosya aynı `pool_id` altında, yalnız takvim uzaklığına dayalı üç zorunlu rapor stratum'udur:

- `2024-09-01`: `far_backcast`;
- `2025-02-15`: `mid_backcast`;
- `2025-06-15`: `near_backcast`, fakat `scope_status=unknown` (`-003`).

Bu adlar model sonucu veya içerik bilgisi kullanmaz; rol seçimi değildir. Üçünün gerçek rolü
aynıdır: `blind_backcast_test`. Mühendislik dosyaları seri işlemeyi gerektirirse deterministik
kronolojik sıra kullanılabilir, ancak üç job tamamlanana kadar hiçbir metrik veya mekanik sonuç
insana açılmaz; arada kod/config/eşik değişmez ve erken durdurma yapılmaz.

Birincil endpoint, üç mechanically eligible artefaktın toplam alert episode sayısının toplam
scoreable flight-hour'a oranıdır. Üç stratumun burden, scoreability ve attrition sonuçları da
zorunlu ve eksiksiz yayımlanır. En iyi gün seçilmez. Equal-date macro ortalama yalnız secondary
sensitivity'dir; üç tarih rastgele/temsili bir örnek değildir.

### 1. Metadata ve temporal kapsam

**Bulgu.** Addendum'daki üç dosya ve metadata değeri dosya sisteminde doğrulandı. İçerikleri
açılmadı, üyeleri listelenmedi, hashlenmedi veya parse edilmedi. Üç tarih de fit gününden
eskidir; bu havuz forward deployment holdout'u değil, geriye-dönük temporal/source-transfer
stres panelidir.

**Kanıt.** Hedefli `Get-Item` ile yapılan inceleme ölçümü:

| Artefakt | Byte | Yerel mtime | Fit gününden önce |
|---|---:|---|---:|
| `v2024.09.01-planes-readsb-prod-0.tar` | 2,084,157,440 | 2026-07-13 15:13:26 | 545 gün |
| `v2025.02.15-planes-readsb-prod-0.tar` | 2,146,856,960 | 2026-07-13 15:13:03 | 378 gün |
| `v2025.06.15-planes-readsb-prod-0-003.tar` | 3,093,094,400 | 2026-07-13 12:02:04 | 258 gün |

Byte ve mtime değerleri addendum ile aynıdır
(`docs/codex_review_prompt_2026-07-13_addendum.md:12-19`). Gün farkları, filename tarihi ile
2026-02-28 arasındaki bu inceleme-türevi takvim hesabıdır; yaklaşık ay ifadesi değildir. Üç
byte değerinin türetilmiş toplamı **7,324,108,800 byte**'tır. Dosya büyüklüğü kapsam veya
“tam gün” kanıtı değildir.

**Öneri.** Şimdilik yalnız bu metadata kaydı korunsun. SHA-256, ancak Adım 9 kullanıcı/gate
kararıyla başladığında, üç dosya için aynı manifest turunda hesaplansın. Hash raw byte'ları
okur ama tar üyesi listelemez/decompress etmez; erişim günlüğünde bu ayrım açıkça yazılsın.
Manifest mtime'ı hem yerel timezone bilgisiyle hem UTC olarak kaydetsin.

**Tahmini efor.** Metadata/freeze şablonu 0.5 iş günü; ölçülmüş süre değildir. Hash çalışma
süresi ölçülmedi ve burada tahmin edilmedi.

**Risk.** Filename tarihi gerçek iç timestamp kapsamını kanıtlamaz. Üç eski tarihte başarı,
2026-03-16 sonrası geleceğe genelleme kanıtı değildir; prospektif holdout ihtiyacı sürer.

### 2. Neden tek batch ve sonuç ambargosu

**Bulgu.** “Önce birini aç, sonuca göre diğerine rol ver” tasarımı ilk dosyayı fiilen
development verisine dönüştürür ve optional stopping/post-hoc dosya seçimine kapı açar. Üç
dosyanın tek mantıksal açılışı bu riski en doğrudan kapatır.

**Kanıt.** Addendum, üçlü havuzun kullanım şeklinin sonuç görülmeden önce seçilmesini istiyor
(`docs/codex_review_prompt_2026-07-13_addendum.md:23-31`). Ana rapor da parser, truth,
scoreability, feature/rule/CUSUM/eşik/event/gate sözleşmesinin açılıştan önce dondurulmasını ve
holdout'un yalnız development gate'inden sonra bir kez açılmasını şart koşuyor
(`docs/codex_review_findings_2026-07-13.md:559-591`).

**Öneri.** Açılış protokolü:

1. Tek champion sistem, parser commit/şema, scaler/rule/CUSUM/S2 artefaktları, threshold,
   eventizer, scoreability, metric ve gate hashleri önce dondurulur. Holdout üzerinde model
   seçimi veya challenger seçimi yapılmaz.
2. Runner; output isolation, restart, attrition ve rapor akışı dahil, yalnız non-blind 2026
   development raw'ı veya sentetik fixture üzerinde uçtan uca dry-run edilir.
3. Tek kullanıcı/gate kararı üç raw artefaktı birlikte `unseal` eder. Fiziksel okuma sırası
   gerekiyorsa `2024-09-01 → 2025-02-15 → 2025-06-15` olur; bu yalnız deterministik execution
   sırasıdır.
4. Job çıktıları erişim-kontrollü staging alanında tutulur. Üç job ve birleşik attrition raporu
   tamamlanmadan insan ara sonucu göremez; early-stop veya “üçüncüyü açmama” yoktur.
5. Her artefakt ayrı write-once output namespace'ine parse edilir; doğrulamadan sonra namespace
   sealed/read-only yapılır. Normal Silver target'ına yazılmaz.
6. Altyapı kesintisi yalnız aynı raw hash, container/commit, config ve checkpoint ile restart
   edilebilir. Bilimsel schema/parser uyumsuzluğu config değiştirip “blind retry” gerekçesi
   değildir.

**Tahmini efor.** Freeze/dry-run denetimi 0.5–1 iş günü; kullanıcı gate'i sonrasında üç izole
parse/evaluation ve birleşik rapor 1–2 iş günü. Bunlar planlama tahminidir.

**Risk.** Tek batch üç blind varlığı aynı anda tüketir. Bunun karşılığı, dosyaları sırayla
görerek ayarlama özgürlüğünün bilinçli olarak kapatılmasıdır. Dry-run yapılmazsa basit altyapı
hatası üç varlığı birden tüketebilir.

### 3. `-003` mekanik kapsam belirsizliği

**Bulgu.** Yalnız 2025-06-15 adındaki `-003`, shard olabileceğini düşündürür fakat kanıtlamaz.
Dosyanın daha büyük olması da tam-gün kanıtı değildir. Freeze öncesi “tam gün” varsayılmamalı;
freeze sonrası ilk erişimde kapsam mekanik olarak sınıflandırılmalıdır.

**Kanıt.** Belirsizlik addendum'da açıkça kaydedilmiştir
(`docs/codex_review_prompt_2026-07-13_addendum.md:14-16,32-35`). Ana rapor, ilk mekanik
kontrolü holdout erişimi sayar; schema/parser başarısızlığını dış-geçerlilik sonucu olarak
raporlar ve aynı holdout'a özel tamirle ikinci bir blind iddiayı yasaklar
(`docs/codex_review_findings_2026-07-13.md:569-589`).

**Öneri.** Freeze manifestinde 2025-06-15 için baştan:

- `declared_date_from_filename=2025-06-15`;
- `scope_status=unknown`;
- `full_day_claim=not_asserted`;
- `filename_suffix=-003`

kaydedilsin. İlk izinli açılışta, skordan önce ama freeze sonrasında, aynı otomatik batch içinde
önceden sınırlanmış şu mekanik kontroller çalışsın: tar integrity, member naming/index,
timestamp min/max ve saat/gap coverage, required schema/type/unit, duplicate kuralı, parse hata
ve attrition sayıları. Alarm/metrik sonucu eligibility belirleyemez.

`-003` partial capture çıkarsa post-hoc dışlanmasın veya başka dosyayla değiştirilmesin;
scoreable exposure'u saat-normalize pooled endpoint'e girsin, kapsamı zorunlu stratified
tabloda `partial_capture` diye gösterilsin ve tam-gün drift iddiası kurulmasın. Parser/schema
uyumsuzsa stratum `external_validity_failure/inconclusive` olarak tüketilmiş sayılır; diğer iki
job yine tamamlanır.

**Tahmini efor.** Mekanik scope/attrition doğrulaması üçlü evaluation süresine dahildir;
ayrı ölçülmüş süre yoktur.

**Risk.** Kapsam kontrolünü skordan sonra yorumlamak, iyi/kötü sonuca göre dosyayı dahil etme
riskini doğurur. Partial capture'ın exposure-normalizasyonu kapsam eksikliğini düzeltmez;
yalnız alarm yükü paydasını dürüstleştirir.

### 4. Endpoint, strata ve çoklu seçim kontrolü

**Bulgu.** Üç tarih daha zengin stres kapsamı sağlar, fakat üç ayrı model seçme yarışına veya
“en iyi gün” raporuna dönüştürülürse körlük avantajı kaybolur. Tek pooled primary endpoint ve
zorunlu tam stratified panel, en az seçim serbestliğiyle en çok bilgiyi korur.

**Kanıt.** Ana rapor doğal trafik için alert episode/scoreable flight-hour, clean flight alarm
oranı ve scoreability'yi ayrı ister; sentetik recall'ın doğal burden olmadan raporlanmasını
yasaklar (`docs/codex_review_findings_2026-07-13.md:499-543`). Üç tarın hiçbirinde gerçek
anomaly etiketi olduğu ölçülmedi; dolayısıyla primary outcome doğal alert burden ve
scoreability'dir, recall değildir.

**Öneri.** Sonuç sözleşmesi:

- **Tek primary endpoint:** mechanically eligible içerikte
  `Σ alert episodes / Σ scoreable flight-hours`. Episode merge/refractory ve flight-hour
  tanımı development'ta önceden dondurulur.
- **Zorunlu strata:** her artefakt için aynı burden, scoreability, toplam exposure, alarm
  episode sayısı ve eksiksiz attrition/reason-code tablosu. Dosya/saat/aircraft seçilmez.
- **Safety guardrail:** pooled ortalamanın kötü bir stratum'u gizlememesi için her eligible
  stratum aynı ön-kayıtlı burden sınırına karşı gösterilir. Per-stratum pass/fail kullanılacaksa
  aile-düzeyi tek-taraflı interval/multiplicity yöntemi development'ta sayısal olarak
  dondurulur; bu rapor ölçülmemiş bir alpha veya burden limiti uydurmaz.
- **Secondary:** equal-date macro burden ve tarih-stratified sensitivity. Üç tarih rastgele
  örnek olmadığı için “drift eğrisi” veya population trend iddiası yoktur.
- **Sentetik secondary:** yalnız donmuş recipe/truth ile tam `3 artefakt × tüm senaryolar`
  matrisi; her hücre aynı artefaktın doğal burden'ı yanında sunulur. Hücre veya senaryo seçimi
  yapılmaz.
- **Tek champion:** holdout öncesi seçilir. Rule/NN/USAD sıralaması bu havuzda yapılmaz;
  diagnostic component skorları yayınlansa bile sonraki model seçimi için bu üçlü tekrar blind
  sayılmaz.

Havuz sonucu sınıfları da önceden dondurulsun:

- `pass`: üç manifest entry'si de mechanically eligible olur; pooled endpoint ve ön-kayıtlı
  zorunlu guardrail'ler geçer;
- `performance_fail`: mechanically eligible stratum/pooled gate performans nedeniyle kalır;
- `external_validity_failure/inconclusive`: kapsam/schema/parser/scoreability sözleşmesi
  değerlendirilemez; sessizce pass veya exclusion yapılmaz.

Batch tamamlandığında üç dosya da tüketilmiş test setidir. Model/parser/eşik değişikliği yeni
versiyondur; bu havuzdaki tekrar skor “blind” diye sunulamaz. Yeni iddia yeni, tercihen
2026-03-16 sonrası prospektif holdout gerektirir.

**Tahmini efor.** Metric/guardrail/attrition sözleşmesini freeze etmek 0.5 iş günü; ölçülmüş
süre değildir.

**Risk.** Pooled oran tek başına kötü günü maskeleyebilir; post-hoc per-day gate ise
multiplicity yaratır. Tam panel ve ön-kayıtlı guardrail ikisini birlikte sınırlar. Üç gün aynı
provider ailesinden olabilir ve bağımsız/temsili örnek oldukları ölçülmedi.

### Güncellenmiş Adım 9 — Adım 1–8 değişmez

**Bulgu.** Önceki planın sırası değişmemelidir; yalnız Adım 9'un nesnesi tek tar yerine üçlü
havuzdur (`docs/codex_review_prompt_2026-07-13_addendum.md:3-4,23-26`).

**Kanıt.** Önceki Adım 9 hâlâ tek 2025-06-15 tarını adlandırır
(`docs/codex_review_findings_2026-07-13.md:624-638`); addendum bunu açıkça üçlü havuza
genişletmiştir.

**Öneri — Adım 9'un yerine geçecek metin.**

> **9 / Gün 10:** Kullanıcı kararıyla 2024-09-01, 2025-02-15 ve 2025-06-15 raw tarları,
> içerikleri açılmadan aynı `pool_id` altında raw path/byte/mtime/SHA-256 ve
> `scope_status` alanlarıyla freeze edilir. Holdout runner üçünde de dry-run edilmez; dry-run
> yalnız non-blind development/fixture üzerinde tamamlanır. Bu iki haftada varsayılan olarak
> tar üyesi listeleme, parse veya evaluation yoktur. Daha sonraki tek gate kararı üç artefaktı
> aynı mantıksal batch'te, sonuç ambargosuyla açar; üçü de test-only kalır.

**Tahmini efor.** Adım 9 freeze işi 0.5–1 iş günü; açılış/evaluation daha sonraki ayrı gate'in
1–2 iş günlük planlama tahminidir.

**Risk.** Eski tek-dosyalı satırı yeni havuzla birlikte iki alternatif plan gibi bırakmak
yorum serbestliği doğurur. Bu ek rapor, yalnız Adım 9 için normatif replacement'tır; onaylanmış
Adım 1–8'e dokunmaz.


---

## Claude handoff — ADS-B contextual anomaly detection

Tarih: 2026-07-14  
Repo: `ColdVI/AnomalyDetection`  
Aktif dal: `main`  
Aktif aday: `contextual_physics_v1`

### Bu belgenin amacı

Bu belge, kullanıcı ile Codex arasındaki son konuşmayı, tamamlanan ADS-B model eğitimini,
Codex'in teknik yorumlarını ve açık bilimsel kararları Claude'a aktarır. Claude'dan beklenen,
mevcut kanıtları bağımsız biçimde eleştirmesi ve özellikle anomaly validation'ın normal
confidence/calibration ile nasıl ilişkilendirilmesi gerektiğini değerlendirmesidir.

### Kullanıcının soruları ve yönlendirmeleri

Kullanıcı sırasıyla şunları istedi veya sorguladı:

1. Yeni contextual model için gerçek bir eğitim yapılması.
2. Eğitimin sonuçlarının çıkarılması ve overall bir rapor hazırlanması.
3. Anomaly detection sırasında sistemin neye baktığının, uçuşun kendi içindeki zaman serisini
   kullanıp kullanmadığının ve model çıktısının matematiksel anlamının açıklanması.
4. Kısaltmaların önce uzun halleriyle verilmesi ve yorumların sayılarla desteklenmesi.
5. Son olarak şu araştırma sorusu:

> Anomaly örneklerini validation'a sokmak, modelin normal hakkındaki confidence-score tahmininde
> normal kümelerini daha fazla daraltmamızı sağlar mı?

### Değişmez proje sınırları

- `archive/` salt-okunur; oradan kod kopyalanmaz veya import edilmez.
- Sentetik anomaly hiçbir train/fit/normal calibration aşamasına giremez.
- Sonuç görüldükten sonra aynı run/config içinde threshold veya hyperparameter ayarı yapılamaz.
- MAD değeri sıfır olan kanal yapay floor ile kurtarılmaz; dışlanır.
- Satır, pencere, event, uçuş ve scoreable uçuş-saati metrikleri birbirine karıştırılmaz.
- Sentetik detection sonucu doğal alarm burden ile birlikte yorumlanır.
- Rehearsal sonucu seçime geri beslenmez.
- Üç dosyalık kör holdout havuzu ayrı unseal onayı olmadan açılmaz.
- Eski Step-7 FAIL sonucu yeni aday tarafından geriye dönük değiştirilmez.

### Önce kısaltmalar

- **Automatic Dependent Surveillance–Broadcast (ADS-B):** Uçağın durum/konum yayın sistemi.
- **Long Short-Term Memory (LSTM):** Zamansal bağımlılıkları taşıyan sinir ağı.
- **Neural Network (NN):** Sinir ağı.
- **Median Absolute Deviation (MAD):** Uç değerlere dayanıklı medyan tabanlı ölçek.
- **Negative Log-Likelihood (NLL):** Olasılıksal tahmin uyumsuzluğu kaybı.
- **Area Under the Receiver Operating Characteristic Curve (AUROC):** Threshold'dan bağımsız
  pozitif-negatif sıralama metriği.
- **Area Under the Precision–Recall Curve (AUPRC):** Precision-recall eğrisinin alanı.
- **Cumulative Sum (CUSUM):** Küçük ve sürekli sapmaları zamanla biriktiren yöntem.
- **Navigation Integrity Category (NIC), Navigation Accuracy Category for Position (NACp),
  Source Integrity Level (SIL):** ADS-B bütünlük/doğruluk beyan alanları.

### Tamamlanan eğitim

Eğitim koşusu:

```text
artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1
```

Dondurulmuş config:

```text
configs/adsb_contextual_physics_v1_train.json
```

Sonuçlar:

- Durum: `trained_not_thresholded`
- Fit rolündeki 149.462 uçuştan deterministik %2 seçim: **2.929 uçuş**
- Seçili satır: **1.267.625**
- Epoch başına pencere: **1.180.160**
- Epoch başına batch: **2.417**
- Epoch: **5**
- Toplam süre: **1.702,158 saniye**
- Ayrı natural-calibration diagnostic: **770 uçuş / 332.510 pencere**
- Sentetik train satırı: **0**
- Sentetik calibration satırı: **0**
- Model parametresi: **9.546**, tamamı sonlu; strict checkpoint reload PASS
- Artefakt checksum doğrulaması: **5/5 PASS**

Weighted Gaussian NLL:

| Epoch | NLL |
|---:|---:|
| 1 | 0,795375 |
| 2 | 0,738245 |
| 3 | 0,723818 |
| 4 | 0,714755 |
| 5 | 0,708696 |

İlk-son epoch göreli düşüşü yaklaşık **%10,90**. Bu, optimizer'ın fit verisinde tahmini
iyileştirdiğini gösterir; anomaly recall kanıtı değildir.

### Time-series mekanizması

Bu model gerçek bir uçuş-içi next-step time-series modelidir:

- Aynı `flight_id` içindeki önceki **12 satır**, bir sonraki satırı tahmin etmek için kullanılır.
- Pencere başka bir uçuşa geçmez.
- Zaman ters/tekrarlıysa veya mesaj gap'i 60 saniyeyi aşarsa pencere üretilmez.
- Skorlanan satırın residual'ı giriş geçmişinde bulunmaz.
- Flight phase, mevcut satırdaki vertical-rate ile değil yalnız önceki üç satırın lagged median'ı
  ile çıkarılır; böylece anomaly kendi calibration bağlamını değiştiremez.
- Model uçuş sırasında online yeniden eğitilmez. Global normal davranış ile o uçuşun yakın geçmişini
  birlikte kullanır.

Kritik yorum: uzun süreli anomaly modelin 12 satırlık geçmişini zamanla kirletebilir. Anlık
surprise başlangıçta yüksekken model girdisi anomaly rejimine girdikçe düşebilir. Bu yüzden spike,
persistence ve accumulation/CUSUM kararları ayrı tutulmalıdır.

### Fiziksel residual kanalları

Aktif beş kanal:

```text
vertical_rate_residual
speed_residual
heading_residual
east_velocity_residual
north_velocity_residual
```

Temel anlamları:

```text
vertical-rate residual
  = bildirilen dikey hız - delta altitude / delta time

speed residual
  = bildirilen yer hızı - ardışık konumdan türetilen hız

heading residual
  = bildirilen track - ardışık konumun dairesel bearing'i

east/north velocity residual
  = bildirilen hız vektörü - konum geçişinden türetilen hız vektörü
```

`altitude_source_residual`, barometrik/geometrik irtifa farkının zaman türevidir. Fit-normal MAD
değeri tam sıfır çıktığı için floor verilmeden dışlandı.

### Model girdisi

Toplam 19 input feature:

- beş scaled residual;
- `log(1 + delta-time)`;
- `sin(track)` ve `cos(track)`;
- `ground/climb/level/descent/unknown` phase one-hot alanları;
- cadence bucket alanları;
- availability mask.

Track'in 359 derece ile 1 derece arasındaki dairesel yakınlığı sin/cos temsiliyle korunur.

### Matematik

Her residual fit-normal medyan ve MAD ile ölçeklenir:

```text
z[c,t] = clip((r[c,t] - median[c]) / MAD[c], -5, 5)
```

LSTM her kanal için bir sonraki residual'ın merkezini ve ölçeğini üretir:

```text
mu[c,t]    = beklenen merkez
sigma[c,t] = beklenen normal oynaklık
0.1 <= sigma[c,t] <= 5.0
```

Kanal-bazlı Gaussian NLL:

```text
NLL[c,t] = log(sigma[c,t])
           + 0.5 * ((z[c,t] - mu[c,t]) / sigma[c,t])^2
```

Maskeli ve açık ağırlıklı toplam loss:

```text
loss = sum(w[c] * mask[c,t] * NLL[c,t])
       / sum(w[c] * mask[c,t])
```

Detection'a taşınan kanal skoru:

```text
S[c,t] = abs(z[c,t] - mu[c,t]) / sigma[c,t]
```

Yorum:

- `S ~= 0`: model beklentisine yakın;
- `S ~= 1`: yaklaşık bir predicted-scale uzakta;
- `S ~= 2`: yaklaşık iki predicted-scale uzakta;
- yüksek S: ilgili kanal ve bağlam için daha sıra dışı.

S değeri doğrudan anomaly olasılığı veya confidence değildir. Normal calibration ile conformal
p-değerine çevrilmesi gerekir:

```text
p[c,t] = (1 + count(S_calibration >= S[c,t])) / (n + 1)
```

### Normal bağlam ve hiyerarşik calibration

Planlanan normal calibration hiyerarşisi:

```text
channel + phase + cadence
          ↓ destek yetersizse
channel + phase
          ↓ destek yetersizse
channel
          ↓ destek yoksa
unscoreable
```

Amaç, tırmanış, level flight ve farklı mesaj cadence'lerinin normal dağılımlarını tek threshold'a
zorlamamaktır.

Dar kümenin avantajı: daha homojen normal dağılım ve küçük anomaly'ye daha yüksek hassasiyet.

Dar kümenin riski: calibration örneği azalır ve p-değeri çözünürlüğü kötüleşir. En küçük mümkün
conformal p-değeri `1/(n+1)`'dir:

| Küme örneği n | En küçük p |
|---:|---:|
| 100 | yaklaşık 0,0099 |
| 1.000 | yaklaşık 0,0010 |
| 10.000 | yaklaşık 0,0001 |

Bu nedenle yalnız “daha fazla cluster” daha iyi değildir; homojenlik ile örnek desteği arasında
bias-variance/support dengesi vardır.

### Magnitude-domination sonucu

Natural diagnostic sonuçları:

- trained-vs-untrained Spearman rho: **0,649633**
- trained-vs-target-magnitude Spearman rho: **0,654240**
- önceden dondurulmuş fail sınırı: `rho >= 0,8`
- sonuç: `magnitude_domination_flagged=false`, gate **PASS**

Eski üç NN'de rho yaklaşık `0,84–0,94` idi ve magnitude-domination FLAGGED olmuştu. Yeni skor salt
genlik kopyası değildir; ancak `rho ~= 0,65` nedeniyle genlik ilişkisi tamamen ortadan kalkmış da
değildir.

Doğal diagnostic kanal p95 surprise değerleri yaklaşık `1,840–2,052` aralığındadır. Normal
pencerelerin yaklaşık %95'i kanal bazında kabaca iki predicted-scale içinde kalmıştır. Bu bir
Gaussian goodness-of-fit kanıtı değil, yalnız skor ölçeği sanity kontrolüdür.

### Eski adaylarla karşılaştırma

| Sistem | Sonuç | Yorum |
|---|---|---|
| Eski Dense-AE | tarihsel pooled AUROC 0,572; rho 0,86/0,90 | magnitude FLAGGED |
| Eski LSTM-AE | tarihsel pooled AUROC 0,568; rho 0,84/0,89 | magnitude FLAGGED |
| Eski LSTM forecaster | tarihsel pooled AUROC 0,552; rho 0,94/0,92 | magnitude FLAGGED |
| Corrected residual rule | AUROC 0,764883; AUPRC 0,883313 | ayrıştırma güçlü, doğal alarm yükü yüksek |
| Eski vector CUSUM h=1 | doğal uçuşların yaklaşık %99,1'i alarm görüyor | doygun, Step-7 FAIL |
| Yeni contextual LSTM | rho 0,649633/0,654240, magnitude PASS | detection sonucu henüz yok |

Corrected residual rule için ayrıca:

- doğal burden: **4,808533 episode / scoreable uçuş-saat**;
- alarm gören uçuş oranı: **0,892356**;
- ground-speed event recall: **0,963659**, medyan gecikme **19,31 s**;
- track event recall: **0,951804**, medyan gecikme **56,75 s**;
- stealthy ramp event recall: **0,801347**;
- stealthy ramp active-interval micro coverage: yalnız **0,183902**.

Bu, tek bir event hit ile olay boyunca faydalı coverage'ın aynı olmadığını gösterir.

### Anomaly validation hakkındaki Codex bulgusu

Anomaly validation normal confidence hesabını doğrudan kalibre etmek için kullanılmamalıdır.
Roller ayrılmalıdır:

```text
normal fit
  -> model normal next-step davranışını öğrenir

normal calibration
  -> conformal p-değeri ve doğal tail hesaplanır

normal development/rehearsal
  -> doğal alarm burden ve kararlılık ölçülür

anomaly development
  -> channel/context/temporal tasarımın anomaly sensitivity'si sınanır

dokunulmamış anomaly test
  -> nihai recall, delay, coverage ve AUROC/AUPRC ölçülür
```

Anomaly örnekleri şu amaçlarla faydalı olabilir:

- hangi normal context ayrımının detection'a gerçekten katkı verdiğini sınamak;
- hangi residual kanalının hangi anomaly ailesine tepki verdiğini görmek;
- instant/persistence/accumulation profilini değerlendirmek;
- aşırı dar kümelerin recall'ı öldürüp öldürmediğini görmek;
- persistent anomaly'nin model geçmişine karışıp skorun sönmesini ölçmek.

Anomaly örnekleri şu hesaplara karıştırılmamalıdır:

- normal MAD/scaler;
- normal conformal tail;
- normal confidence p-değeri;
- doğal alarm burden threshold'u.

Anomaly skorları normal tail'e katılırsa dağılım genişler, threshold yükselir ve küçük anomaliler
normalleşebilir.

### Önemli metodolojik sonuç

Anomaly validation'a bakarak clustering, model config veya temporal threshold değiştirilirse bu
set artık test değildir; development setidir. Sistem fiilen semi-supervised model-selection
haline gelir. Bu bilimsel olarak yapılabilir, fakat:

1. yeni config yeni namespace altında önceden kaydedilmelidir;
2. mevcut truth-v2 development rolüne düşer;
3. nihai iddia için görülmemiş truth-v3 veya eşdeğer bağımsız anomaly test gerekir;
4. üçlü kör raw holdout ayrı unseal onayı olmadan kullanılamaz.

Mevcut `contextual_physics_v1` sözleşmesinde truth-v2 feedback yasaktır. Bu nedenle anomaly
validation ile cluster/threshold tuning mevcut v1 içinde post-hoc yapılamaz.

### Henüz olmayan sonuçlar

Yeni contextual model için henüz aşağıdakiler yoktur:

- AUROC;
- AUPRC;
- event recall;
- first-alarm delay;
- active-interval coverage;
- doğal operator-facing alarm episode burden;
- alarm gören uçuş oranı;
- anomaly profile bazlı threshold.

Bu değerler hesaplanmadığı için “yeni model anomalilerin %X'ini yakalıyor” denemez. Mevcut sonuç
yalnız model eğitimi ve magnitude gate sonucudur.

### Açık operasyonel karar

Calibration/evaluation öncesinde kullanıcı tarafından şu sayı tanımlanmalıdır:

```text
100 scoreable uçuş-saatinde kabul edilen maksimum operator-facing alarm episode sayısı
```

Ardından toplam bütçenin beş residual channel ve ayrı S2 reason-code katmanları arasında nasıl
paylaşılacağı önceden dondurulmalıdır. Bu karar anomaly sonuçları görüldükten sonra verilmemelidir.

### Claude'dan istenen bağımsız değerlendirme

Lütfen aşağıdaki soruları açıkça yanıtla:

1. Anomaly validation kullanmadan normal context kümelerini daraltmak için mevcut
   `channel + phase + cadence` hiyerarşisine hangi nedensel, anomaly'den etkilenmeyen bağlamlar
   eklenebilir?
2. Minimum calibration support ve fallback seviyeleri nasıl seçilmeli? Conformal p-resolution ile
   cluster homojenliği arasındaki dengeyi öner.
3. Anomaly development kullanılması bilimsel olarak değerli mi? Değerliyse `contextual_physics_v2`
   için veri rolleri ve dokunulmamış final test nasıl kurulmalı?
4. Persistent anomaly'nin 12 satırlık history'ye karışarak surprise skorunu söndürmesi nasıl
   ölçülmeli ve önlenmeli?
5. Kanal-bazlı conformal p-değerleri instant/persistence/time-normalized accumulation profillerine
   nasıl bağlanmalı?
6. Kullanıcı operasyonel burden sayısını henüz bilmiyorsa, sonucu görmeden seçilebilecek savunulabilir
   bir burden duyarlılık protokolü nedir? Tek sayı yerine önceden dondurulmuş Pareto eğrisi kabul
   edilebilir mi?
7. Mevcut magnitude PASS kanıtı yeterli mi; hangi ek falsification testleri yapılmalı?
8. Yeni adayın eski residual rule ile aynı doğal burden seviyesinde adil karşılaştırması nasıl
   yapılmalı?

### İlgili repo belgeleri

- `docs/adsb_overall_model_report_2026-07-14.md`
- `docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md`
- `docs/decisions.md` — ADR-024, ADR-025, ADR-028, ADR-030, ADR-032–036
- `configs/adsb_contextual_physics_v1_train.json`
- `adsb/models/contextual_residual_forecaster.py`
- `adsb/contextual_windowing.py`
- `adsb/context.py`
- `adsb/conditional_calibration.py`
- `adsb/contextual_decision.py`

Son yayımlanmış ilgili commitler:

```text
3de9c9f  record contextual ADS-B training result
fd8c7ea  document contextual ADS-B model interpretation
```
