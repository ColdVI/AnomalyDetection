# Proje Skor ve Başarısızlık Defteri — tüm hatlar, tüm sayılar

Tarih: 2026-07-17 · Kapsam: projenin başından bugüne kadar **her** Gate/kapı
FAIL'i, NO-GO'su, hedefi tutturamayan sonucu — dataset, kanal/kategori, algoritma
ve gerçek metrik değerleriyle. Kaynak: `docs/decisions.md` (ADR'ler), bellek
kayıtları ve bu oturumda bağımsız doğrulanan RESIDUAL-V1/GNSS çalışması.
Başarılı/kabul edilen sonuçlar da bağlam için işaretlendi (✅), ama odak FAIL/NO-GO.

**Okuma notu:** "Gate B" = yeni yöntem eskisini belirli bir marjla geçmeli;
"Gate C" = operasyonel bütçe (recall + saatlik yanlış alarm birlikte); "S-1/S-2/S-3/S-4"
= RESIDUAL-V1'in threshold-bağımsız sanity testleri. Her satır bir ADR'ye veya bu
oturumdaki doğrulamaya bağlanabilir.

---

## 1. Legacy ML hattı — ALFA / UAV Attack / UAV-SEAD (arşivlendi 2026-07-10)

`archive/2026-07-10_legacy_non_adsb_ml/` altında. 17 fazlık ilk yaklaşım, sonunda
tamamen arşivlenip ADS-B hattına geçildi (bkz. §3).

### 1.1 Temel modeller (ML-1..ML-3)

| Dataset | Kanal/Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| ALFA | tüm satırlar | Monolitik satır-bazlı Isolation Forest | ROC | 0.50 | FAIL (rastgele) |
| UAV Attack | tüm satırlar | Monolitik satır-bazlı IF | ROC | 0.21 | FAIL (rastgeleden kötü) |
| UAV Attack | ping_dos | Modüler IF-füzyon | uçuş-recall | 4/6 log tespit edilemedi | FAIL — imza fiziğe yansımıyor |
| ALFA | genel | LSTM-AE (10dk azveri) | uçuş-ROC | 0.731 vs IF-füzyon 0.833 | FAIL — IF'i geçemedi |
| UAV-SEAD | kalibrasyonsuz transfer | IF-füzyon | normal FA | 1.00 | FAIL — tam yanlış alarm |
| ALFA | rudder_fault (n=4) | CUSUM | "1.00 tespit" | n=1'den — istatistiksel anlamsız | FAIL (istatistiksel tiyatro, sonradan reddedildi) |
| ALFA/UAV | H10 hipotezi (max-pencere skoru) | — | — | reddedildi | FAIL — oran-skoru sinyali seyreltiyor |
| ALFA/UAV | drift/bias/freeze enjeksiyonu | USAD | recall | 0.45 / 0.53 (< LSTM-AE) | FAIL — elendi |

### 1.2 Veri büyütme sonrası (ML-4..ML-5)

| Dataset | Kanal | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD | ranges (adil) | IF-füzyon | satır-ROC | 0.474 | FAIL — UAV-Attack feature'ları SEAD'i yakalamıyor (H13) |
| UAV-SEAD | EKF test-ratio | IF-füzyon | korelasyon yönü | TERS sinyal (0.354) | FAIL — arızada ölçüm reddi innovation'ı bastırıyor (H14) |
| UAV-SEAD | altitude | tip tespiti | recall | 0.27 | FAIL — zayıf |

### 1.3 Causal düzeltme sonrası gerçek sayılar (ML-6/7)

| Dataset | Kanal | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| ALFA | alt_error_cusum | CUSUM (causal) | uçuş-ROC | 0.878 → **0.611** | FAIL — "en güçlü sinyal" sanılan kanal causal ölçümde çöktü |
| ALFA | tüm kanallar | event-onset recall | recall | 0.594 → **0.194–0.224** | FAIL — eski point-adjust benzeri şişirme düzeltildi |

### 1.4 LightGBM / kategori residual / iki-kanal (ML-8A, ML-9, ML-13)

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD | genel | LightGBM window | AUPRC | 0.349 vs IF 0.385 | **Gate B kaldı** |
| UAV-SEAD | tüm policy | LightGBM+decision | kritik/advisory recall+FA | hiçbiri hedefi karşılamadı | **Gate C kaldı** |
| ALFA | sabit reçete | LightGBM | AUPRC | 0.843 < IF 0.858 < LSTM-AE 0.872 | FAIL — en zayıf model |
| UAV-SEAD | Position.Z | dikey_tutarlilik (ML-9) | recall farkı | +0.021 (4/5 seed, <0.05 baraj) | **Gate B kaldı** — büyüklük yetersiz |
| UAV-SEAD | Actuator O+C | motor_simetrisi (ML-9) | recall farkı | +0.024 (2/5 seed, <3/5 kararlılık) | **Gate B kaldı** |
| UAV-SEAD | tüm kategoriler | ML9-fusion | CUSUM/advisory recall/FA | 0.222 / 25.83 FA-saat | **Gate C kaldı** |
| UAV-SEAD | mekanik+sistem | iki-kanal (ML-13) `dengeli` | CUSUM/advisory recall/FA | 0.217→0.291 (+0.074) ama FA 23.70→**44.70** (1.89×) | **Gate B kaldı** — kazanım FA şişirerek satın alındı |
| UAV-SEAD | | | CUSUM/critical FA | 12.98→27.49 (2.12×) | aynı |
| UAV-SEAD | | | K-of-N/advisory FA | 3.29→**12.60** (3.83×) | aynı |

### 1.5 Chronos zero-shot ve ince modül (ML-10, ML-12) — Gate B geçen NADİR örnekler

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD | Actuator O+C | Chronos zero-shot (`chronos_motor`) | CUSUM/advisory recall | 0.205→**0.390** (+0.185, 4/5 seed) | ✅ Gate B geçti |
| UAV-SEAD | Position.Z | Chronos (`chronos_dikey`) | recall | 0.096→**0.023** | ❌ REDDEDİLDİ — 6/6 kombinasyon negatif |
| UAV-SEAD | tüm kategoriler | `ml10_fusion` | CUSUM/advisory recall/FA | 0.213 / 23.92 FA-saat (hedef ≥0.50/≤12) | **Gate C kaldı** |
| UAV-SEAD | Actuator O+C | ince-modül `itki_komutu` (tek feature) | CUSUM/advisory recall | 0.205→**0.459** (+0.254, en iyi kategori sonucu) | ✅ Gate B geçti (hem B1 hem B2) |
| UAV-SEAD | | `itki_komutu` | normal uçuşlarda FA | **38.1 FA-saat** | FAIL nedeni — yüksek-FA uzman kanal |
| UAV-SEAD | tüm kategoriler | `ml12_fusion_itki` | CUSUM/advisory recall/FA | 0.217 / 23.74 FA-saat | **Gate C kaldı** (yine) |

### 1.6 Veri büyütme + derin öğrenme yeniden deneme (ML-14, ML-16)

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD (899 normal) | tüm | `ml14_fusion` CUSUM/advisory | recall/FA | 0.126 / 9.95 FA-saat (eski 0.21/23.6) | FAIL — recall/precision değiş-tokuşu, hedef karşılanmadı |
| UAV-SEAD | | CUSUM/critical | recall/FA | 0.043 / 1.60 FA-saat | FAIL |
| UAV-SEAD | | K-of-N/advisory | recall/FA | 0.056 / 2.28 FA-saat | bütçe içi ama düşük recall |
| UAV-SEAD | tüm (5-seed) | LSTM-AE (Kol L) | threshold/critical recall, FA | ~0.22 ham, FA~2.8-2.9 | **Gate B kaldı** |
| UAV-SEAD | | Dense-AE (Kol D) | aynı | benzer | **Gate B kaldı** |
| UAV-SEAD | | USAD (Kol U) | aynı | benzer | **Gate B kaldı** |
| UAV-SEAD | (üçü de) | trained vs untrained-random | Spearman ρ | **0.964** | Genlik-baskınlığı artefaktı — eğitim ~hiçbir şey katmıyor |
| UAV-SEAD | (üçü de) | trained vs raw-‖x‖² | Spearman ρ | **0.965** | Aynı artefakt |
| UAV-SEAD | relerr-düzeltmeli | LSTM-AE/Dense-AE/USAD | düzeltme sonrası ρ | 0.15 / 0.23 / 0.13 (FLAGGED eşiği 0.80 altı) | ✅ genlik-bağımlılığı kırıldı |
| UAV-SEAD | relerr-düzeltmeli | aynı üçü | recall | **<0.06** | FAIL — kazancın büyük kısmı ham genlik farkıymış |

---

## 2. RflyMAD (RFLY-0 / RFLY-1)

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| RFLY-only | motor/sensör | `itki_komutu` (whole-flight proxy, **geçersiz**) | CUSUM/advisory recall/FA | 0.749 / — | ❌ GEÇERSİZ — whole-flight proxy hatası (RFLY-1 ile reddedildi) |
| RFLY-only | | threshold/critical (proxy) | recall/FA | 0.573 / 1.23 FA-saat | ❌ GEÇERSİZ (aynı neden) |
| Pooled SEAD+RFLY | | full matrix (proxy) | Gate R-C | kaldı | FAIL bile proxy ile |
| **RFLY-only (düzeltilmiş, interval-truth)** | motor/sensör | `itki_komutu` | CUSUM/advisory recall/FA | **0.526 / 22.28 FA-saat** | Gate R-A/R-B geçti, **Gate R-C kaldı** (FA hedefin ~2 katı) |
| RFLY-only (düzeltilmiş) | | `itki_komutu` critical | recall/FA | 0.442 / 9.23 FA-saat | FAIL (critical hedefi ≥0.30@≤2 karşılamadı) |
| Pooled SEAD+RFLY (düzeltilmiş) | | `itki_komutu` | CUSUM/advisory recall/FA | **0.149 / 30.00 FA-saat** | FAIL — havuzlama ciddi kötüleştiriyor |
| Pooled (düzeltilmiş) | | `rfly0_fusion` advisory | recall/FA | 0.066 / 9.39 FA-saat | bütçe içi ama çok düşük recall |
| Pooled (düzeltilmiş) | | `rfly0_fusion` critical | recall/FA | 0.018 / 1.72 FA-saat | aynı |

5 uçuşta ULog arıza-onay mesajı klasör etiketiyle çelişti → ambiguous/invalid
sayılıp sonuç görülmeden dışlandı (istatistiksel temizlik, hata değil).

---

## 3. ADS-B hattı (aktif — "3. hafta" pivotu)

### 3.1 Kural-bazlı skorlayıcı vs nöral alternatifler

| Senaryo | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|
| pooled (tüm senaryolar) | Kural-bazlı penalty scorer | AUC | **0.600** | ✅ üç NN'i de geçti |
| pooled | Dense-AE | AUC | 0.572 | FAIL — kural-bazlıyı geçemedi |
| pooled | LSTM-AE | AUC | 0.568 | FAIL |
| pooled | LSTM-forecaster | AUC | 0.552 | FAIL |
| ground_speed_biased | Dense-AE / LSTM-AE / LSTM-forecaster | AUC | 0.737 / 0.743 / 0.648 | kısmi (büyüklüğü yakaladı) |
| vertical_rate_frozen | aynı üçü | AUC | 0.600 / 0.579 / 0.552 | zayıf |
| track_frozen | aynı üçü | AUC | 0.523 / 0.521 / 0.551 | FAIL — rastgeleden farksız |
| position_ramp_stealthy | aynı üçü | AUC | 0.513 / 0.519 / 0.513 | FAIL — rastgele |
| altitude_dropout | aynı üçü | AUC | 0.489 / 0.480 / 0.498 | FAIL — rastgeleden kötü |
| genel | tüm modeller (magnitude-domination) | trained-random / trained-raw ρ | Dense 0.86/0.90, LSTM-AE 0.84/0.89, forecaster 0.94/0.92 | FLAGGED (genlik-baskınlığı) |
| pooled (proxy pencere etiketi) | — | AUC tavanı | **~0.75** | Yapısal sınır — mükemmel dedektör bile burada tavan yapar |

### 3.2 Truth-v2 düzeltilmiş kural (daha güçlü ama Adım 7'de FAIL)

| Senaryo | Metrik | Değer | Sonuç |
|---|---|---|---|
| pooled | AUROC / AUPRC | **0.764883 / 0.883313** | ✅ güçlü ayrışma |
| ground_speed | event recall (medyan gecikme) | 0.963659 (19.31 s) | ✅ |
| track | event recall (medyan gecikme) | 0.951804 (56.75 s) | kısmi (gecikme yüksek) |
| stealthy_ramp | event recall | 0.801347 | görünüşte iyi AMA aktif-aralık micro coverage yalnız **0.183902** | FAIL — kapsam çok dar |
| — | doğal temiz burden | 4.808533 episode/saat | referans |
| tam-hacim | CUSUM h=1 burden (calib/dev/rehearsal) | 6.07 / 5.74 / 5.33 episode/saat | scoreable-flight alarm oranı ~%99 — episode-merge doygunluğu gizliyor |

**ADR-032 — Adım 7 gate FAIL (2026-07-14):** Rule+CUSUM ana konfigürasyonu
dondurulamadı — üç ayrı, herhangi biri tek başına yeterli üç engel: (1) CUSUM doğal
alarm doygunluğu operasyonel gate'i geçmiyor, (2) üç NN magnitude şartını geçmiyor
(yukarıdaki ρ 0.84-0.94 FLAGGED), (3) corrected CUSUM truth-v2 ölçümü frozen
scoring-source snapshot'ı geri getirilemediği için fail-closed bloklandı (hash
uyuşmazlığı: `adsb/features.py` canonical-LF hash frozen manifestle eşleşmiyor,
735 yerel blob adayında bulunamadı). **Sonuç: Genel gate FAIL, sonraki adımlar
(Adım 8/9) başlatılmadı.**

---

## 4. UAV GNSS Bütünlük Fizibilitesi v1 — tam NO-GO (16 Temmuz 2026)

Bu oturumda tam metni okunup bağımsız doğrulandı
(`artifacts/uav_gnss_integrity_v1/uav_gnss_integrity_v1_final_no_go_report.tex`).

| Rol | Sözleşme | Yöntem | Recall | Alarm/uçuş-saati | Sonuç |
|---|---|---|---:|---:|---|
| development | critical | uçuş kontrol göstergeleri (PX4-native) | %58.8 | 19.58 | FAIL |
| development | critical | CUSUM | %0.0 | 0.00 | FAIL |
| development | critical | contextual LSTM | %47.1 | 20.32 | FAIL |
| development | advisory | PX4-native | %58.8 | 19.58 | FAIL |
| development | advisory | CUSUM | %52.9 | 0.00 | FAIL |
| development | advisory | LSTM | %82.4 | **101.60** | FAIL — alarm yükü aşırı |
| rehearsal | critical | PX4-native | %90.0 | 28.86 | FAIL — alarm yükü |
| rehearsal | critical | CUSUM | %0.0 | 0.00 | FAIL |
| rehearsal | critical | LSTM | %90.0 | 0.00 | FAIL — development'a genellemiyor |
| rehearsal | advisory | PX4-native | %90.0 | 28.86 | FAIL |
| rehearsal | advisory | CUSUM | %70.0 | 7.21 | FAIL — %90 hedefi tutturamadı |
| rehearsal | advisory | LSTM | %100.0 | 44.86 | FAIL — alarm yükü |
| — | — | LSTM trained-random Spearman | ρ | 0.678 | 0.80 altı (magnitude-domination tekrarlanmadı ama…) |
| — | — | LSTM trained-raw-magnitude | ρ | 0.690 | aynı |
| SIL-Wind | advisory | CUSUM / LSTM | alarm/saat | 8.27 / 6.88 | bütçe (12) içinde |
| HIL-Wind | advisory | CUSUM / LSTM | alarm/saat | **20.53 / 22.61** | FAIL — domain kaymasına aşırı hassas |

**Sonuç: NO-GO / not achievable with current data and instrumentation.** Hiçbir
yöntem hiçbir rolde kritik+advisory'yi birlikte tutturamadı; kör holdout hiç
açılmadı.

---

## 5. RESIDUAL-V1 — Görev 4.1 (G1 ridge) ve Faz E (bu oturum)

### 5.1 Görev 4.1 — G1 ridge, development-only

| Dataset | Kanal | CV R² | Train R² | Sonuç |
|---|---|---:|---:|---|
| ALFA | R1_aileron_roll_rate | — | — | FAIL — eğitilemedi (1 oturum, ≥2 gerekli) |
| ALFA | R2_elevator_pitch_rate | — | — | aynı |
| ALFA | R3_rudder_coordinated_yaw_rate | — | — | aynı |
| ALFA | R4_throttle_airspeed_derivative | — | — | aynı |
| ALFA | R5_pitch_throttle_climb_rate | — | — | aynı |
| RFLY | Q1_attitude_setpoint_rate_response | **0.0114** | 0.3638 | FAIL — pratikte zayıf, train-CV farkı büyük |
| RFLY | Q2_motor_pwm_distribution | **0.4564** | 0.7109 | ✅ eğitildi, en güçlü G1 sonucu |
| RFLY | Q3_total_pwm_vertical_acceleration | **0.0003** | 0.0085 | FAIL — pratikte sinyalsiz |
| RFLY | Q4_position_setpoint_velocity_response | — | — | FAIL — train-eligible satır yok |

### 5.2 Faz E — sanity kapıları (hepsi PASS) ama kalibrasyon NO-GO

| Test | Kanal | Metrik | Değer | Sonuç |
|---|---|---|---:|---|
| S-4 (komut ablasyonu) | Q1 | var(sakat)/var(tam) | 1.1992 (eşik 1.15) | ✅ PASS |
| S-4 | Q2 | aynı | 2.4972 | ✅ PASS |
| S-4 | Q3 | aynı | **1.0081** (eşik 1.15) | ❌ FLAGGED — karar hattından çıkarıldı |
| S-4 | Q4 | — | — | not_evaluable/model_unavailable |
| S-1 (büyüklük korelasyonu) | R6 | Spearman ρ | 0.4718 (eşik 0.5) | ✅ PASS |
| S-1 | Q1 | ρ | 0.1398 | ✅ PASS |
| S-1 | Q2 | ρ | 0.0179 | ✅ PASS |
| S-3 (KS ayrışması) | ALFA/engine (R6) | KS / p | 0.1646 / 5.5e-18 | ✅ PASS |
| S-3 | RFLY/motor (Q1) | KS / p | 0.1772 / ≈0 | ✅ PASS |
| S-3 | RFLY/motor (Q2) | KS / p | 0.5702 / ≈0 | ✅ PASS |
| S-3 | RFLY/sensor (Q1) | KS / p | 0.2740 / ≈0 | ✅ PASS |
| S-3 | RFLY/sensor (Q2) | KS / p | 0.0340 / 2.3e-91 | ✅ PASS (küçük etki) |
| **Kalibrasyon** | ALFA/R6 | mevcut/gereken normal saat | 0.168846 / 2.0 (**11.845× açık**) | ❌ **NO-GO** — thresholds_frozen.json yazılmadı |
| **Kalibrasyon** | RFLY/Q1,Q2 | mevcut/gereken normal saat | 0.786237 / 4.0 (**5.088× açık**) | ❌ **NO-GO** |
| Veri tavanı testi | ALFA | toplam normal uçuş (evren) | 11 / 47 sabit corpus | Ek veri yolu KAPALI |
| Veri tavanı testi | RFLY | resmî kaynak / projede / eksik üst sınır | 84 / 51 / 33 | En iyimser ingest bile 1.419h → **2.581h/~168 uçuş açık kalır** |

**Sonuç: NO-GO / mevcut development-normal maruziyetle elde edilemez** (ADR-043,
2026-07-17). Kritik ayrım: bu bir sinyal-yokluğu değil — S-3 üç sınıfta da PASS —
salt kalibrasyon için yeterli bağımsız normal uçuş-saati yok.

---

## Özet — kaç FAIL/NO-GO, kaç PASS/Gate-geçen

| Hat | Toplam denenen konfigürasyon (bu deftere göre) | Gate C / operasyonel hedefi geçen | Gate B (yöntemsel iyileşme) geçen | Tam NO-GO |
|---|---:|---:|---:|---|
| Legacy ML (ALFA/UAV/SEAD) | ~20 | **0** | 2 (Chronos ML-10, ince-modül ML-12) | — (arşivlendi) |
| RflyMAD (RFLY-0/1) | 6 | **0** | Gate R-A/B geçti, R-C hep kaldı | — |
| ADS-B | ~10 | 0 (Adım 7 FAIL) | kural-bazlı NN'leri geçti (AUC 0.60) | ADR-032 FAIL |
| UAV GNSS Integrity v1 | 12 satır (3 yöntem × 2 rol × 2 sözleşme) | **0** | — | ✅ NO-GO (ADR — GNSS raporu) |
| RESIDUAL-V1 | 9 kanal + 4 sanity kapısı + kalibrasyon | S-1/S-3 hepsi PASS, **kalibrasyon NO-GO** | Q2 tek güçlü ridge sonucu | ✅ NO-GO (ADR-043) |

**Tek satırlık gerçek:** Projenin başından beri **hiçbir konfigürasyon operasyonel
Gate C/final bütçe hedefini tam anlamıyla karşılamadı** — en yakın yaklaşımlar
RFLY-only `itki_komutu` (0.526 recall ama FA hedefin 2 katı) ve ADS-B kural-bazlı
skorlayıcı (AUC 0.60-0.88 aralığında, ama tavan ~0.75-0.88 senaryoya göre) idi.
İki en güncel çalışma (GNSS Integrity v1, RESIDUAL-V1) disiplinli NO-GO ile kapandı;
ikisinde de "sinyal var ama operasyonel eşik kurulamadı" ayrımı belgelendi.
