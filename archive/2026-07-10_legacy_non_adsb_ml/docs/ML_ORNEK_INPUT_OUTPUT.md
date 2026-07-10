# ML Sistemi — Örnek Girdi/Çıktı Rehberi

Bu dosya "sistem gerçekte ne yapıyor" sorusuna somut örneklerle cevap verir. Her örnek
gerçek feature tablolarından (`data/gold/ml_features/`) çekilmiş **gerçek sayılardır**,
uydurma değildir. Üç başarı, bir kısmi-başarı, bir öğretici karşıt-örnek ve bir
true-negative (doğru "alarm yok") örneği var — amaç sistemi olduğundan iyi göstermek
değil, nerede işe yaradığını nerede yaramadığını dürüstçe göstermek.

---

## Örnek 1 — UAV Attack: gizli GPS spoofing (BAŞARI)

Saldırgan konumu sahteliyor ama alıcının kendi bildirdiği hızı (`vel_m_s`) değiştiremiyor
— gerçekçi bir kısıt. Bu, "kaba" spoofing (78 km'lik sıçrama) değil, **gizli/yavaş**
bir sahtecilik örneği (`ace-spoofing-hackrf-log_5_...`).

| Adım | Değer |
|---|---|
| **Girdi** | `lat/lon` bir önceki örneğe göre küçük bir sıçrama yapıyor; alıcı `vel_m_s = 3.352` bildiriyor |
| **1. Konumdan hesaplanan hız** | `gps_speed_calc_mps = 0.350` (haversine mesafe / geçen süre) |
| **2. Analitik redundancy residual'i** | `gps_speed_residual = |0.350 − 3.352| = 3.00` |
| **3. Referans:** | Normal uçuşlarda bu residual ortalama **0.095 ± 0.10** — yani 3.00 değeri normalin **~29 katı** |
| **4. Isolation Forest skoru** | `nav_bütünlüğü` modülü bu satırı yüksek skorlar |
| **5. Eşiğe oran** | Normal-val Q99 eşiğinin katbekat üstünde → `ratio ≫ 1` |
| **Çıktı** | `ALARM: gps_spoofing_suspected` — küçük mutlak sayı (3 m/s) ama **istatistiksel olarak çok belirgin** |

**Neden çalışıyor:** Saldırgan tek bir kanalı (konum) sahteleyebiliyor ama fiziksel olarak
bağımsız ikinci bir kanalı (alıcının kendi hız ölçümü) değiştiremiyor — iki kanal arasındaki
tutarsızlık saldırganın erişemediği bir "kör nokta".

---

## Örnek 2 — ALFA: rudder arızası (KISMİ BAŞARI — neden zayıf olduğu da görülüyor)

Sabit-kanat fiziği: uçak yatıyorsa (`roll`) dönmek ZORUNDA (`ψ̇ = g·tan(roll)/V`). Bu ilişkiden
sapma `turn_residual` feature'ı.

| Adım | Değer |
|---|---|
| **Girdi (arızalı uçuş, t=64.8s)** | `roll_error = 44.2°`, `yaw_rate = 115.6°/s` |
| **Beklenen yaw rate (fizik)** | Bu roll/hız kombinasyonunda çok daha düşük olmalı |
| **turn_residual (bu satır)** | **86.68** — bu uçuşun gördüğü en büyük sapma |
| **Referans — normal uçuşlar** | ortalama **−2.10 ± 13.95** |
| **Referans — TÜM rudder_fault etiketli satırlar** | ortalama **−1.32 ± 15.67** |

**Buradaki dürüst bulgu:** rudder_fault'un **ortalaması** normalden neredeyse ayırt edilemiyor
(−1.32 vs −2.10 — CUSUM gibi "ısrarlı ortalama kayması" arayan yöntemler bunu kaçırır).
Arıza sinyali **seyrek ve sivri** (86.68'lik tek bir uç değer) — sürekli bir kayma değil.
Bu yüzden mevcut mean/CUSUM-ağırlıklı modüller rudder'da zayıf (0.333); ayrıca bu sınıfta
sadece **4 farklı uçuş** var, istatistiksel güç zaten düşük.

**Çıktı (bu spesifik satırda):** eşik-üstü tekil sivri değer → modülün **max-tabanlı**
istatistikleri (`turn_residual_5s_rms` gibi) bunu bazen yakalar, ortalama-tabanlı olanlar kaçırır.

---

## Örnek 3 — SEAD: altitude anomalisi (DÜRÜST ZAYIFLIK — yeni bulgu)

Bugün gerçek veride kontrol ettim: bu sınıfın zayıflığının (%0-3 onset recall) sandığımdan
daha spesifik bir sebebi var.

| Feature | Normal'de doluluk | altitude_anomaly'de doluluk | Ayrım var mı? |
|---|---|---|---|
| `alt_baro_residual` (barometre vs GPS) | %14 uçuşta var | %7 uçuşta var | Zaten çoğu uçuşta yok — kullanılamaz |
| `alt_local_residual` (yerel konum vs GPS) | %7 uçuşta var | **%0 uçuşta var** | Bu sınıf için **hiç hesaplanamıyor** |
| `hgt_test_ratio` (EKF irtifa tutarlılık) | %100 dolu, ort. **0.041** | %100 dolu, ort. **0.047** | Ortalamada neredeyse **hiç fark yok** |
| `hgt_test_ratio` — **uçuş-başı MAKSİMUM** | medyan 0.095, p90 **0.258** | medyan 0.142, p90 **0.899** | Zayıf ama **var** — üst-uçta ayrışma |

| Adım | Değer |
|---|---|
| **Girdi** | 2018-09-11/07_45_17 uçuşu, 336 satır |
| **Hesaplanan feature** | `alt_baro_residual` = TAMAMEN NaN (barometre topic'i bu uçuşta yok) |
| **Hesaplanan feature** | `alt_local_residual` = tüm satırlarda **0.0** (yerel konum referansı yok) |
| **Kalan tek sinyal** | `hgt_test_ratio` — ama ortalaması normalle aynı |
| **Çıktı** | `NO ALARM` (yanlış negatif) |

**Sonuç:** Bu bir "model zayıf" değil, **"iki feature'ın kaynak topic'i bu uçuşlarda
loglanmamış"** sorunu. `hgt_test_ratio`'nun uç-değerlerinde (p90: 0.258→0.899) hafif bir
umut var — bu, veri büyütmenin **belki** işe yarayabileceği tek nokta, ama garanti değil;
yeni indirilen uçuşların da aynı eksik-topic örüntüsünü taşıyıp taşımadığı kontrol edilmeli.

---

## Örnek 4 — SEAD: EKF'nin kendi kendini yalanlaması (ÖĞRETİCİ TERS ÖRNEK, H14)

Bu örnek "daha fazla sinyal her zaman iyidir" varsayımını çürütüyor.

| Adım | Değer |
|---|---|
| **Girdi** | Bir GPS-tutarsızlığı anomalisi sırasında EKF, gelen ölçümü **reddediyor** (`innovation_check_flags_active=1`) |
| **Beklenti** | Ölçüm reddedildiyse "innovation" (tahmin-ölçüm farkı) büyük olmalı |
| **Gerçekte olan** | Reddedilen ölçüm innovation'a hiç katkı YAPMIYOR → `vel_test_ratio`/`pos_test_ratio` **düşük kalıyor** |
| **Yalnız-EKF modülü çıktısı** | Anomali satırı **"temiz"** görünüyor — satır-ROC **0.354** (0.5 altı = rastgeleden kötü!) |
| **Çıktı (eğer bu tek başına kullanılsaydı)** | `NO ALARM` — ama gerçekte anomali VAR |

**Ders:** Bu yüzden EKF test-oranları varsayılan füzyona **alınmadı**. Reject-bayrakları
(`innovation_check_flags_active/bit_count`) ile birleşmeden kullanılamaz — "ölçüm reddedildi
mi" bilgisi olmadan "innovation küçük" bilgisi yanıltıcı.

---

## Örnek 5 — Gerçek normal uçuş (TRUE NEGATIVE — doğru "alarm yok")

Sistemin "her şeyi anomali sanmadığını" göstermek için.

| Adım | Değer |
|---|---|
| **Girdi** | Herhangi bir SEAD normal-etiketli uçuş, rastgele bir satır |
| **gps_speed_residual** | ~0.095 (normal ortalamaya yakın) |
| **hgt_test_ratio** | ~0.041 (normal ortalamaya yakın) |
| **turn_residual (ALFA benzeri uçuşta)** | normal aralıkta (~±14 std içinde) |
| **Isolation Forest skoru / eşik oranı** | `ratio < 1.0` (tüm modüllerde) |
| **Karar katmanı** | K-of-N tetiklenmiyor, CUSUM `h` eşiğini aşmıyor |
| **Çıktı** | `NO ALARM` — **doğru** (gerçek bir anomali yok) |

---

## Özet tablo — hangi örnek neyi kanıtlıyor

| # | Sınıf | Sonuç | Kanıtladığı şey |
|---|---|---|---|
| 1 | UAV Attack spoofing | Alarm ✅ (doğru) | Analitik redundancy küçük mutlak sayılarda bile işe yarıyor |
| 2 | ALFA rudder | Kısmen ✅ | Fizik-prior var ama sinyal seyrek+az veri — mean-tabanlı yöntem kaçırıyor |
| 3 | SEAD altitude | Alarm yok ❌ (yanlış negatif) | Kaynak topic eksikliği — model değil, veri sorunu |
| 4 | SEAD EKF tek-başına | Alarm yok ❌ (bilinçli dışlandı) | Bazı sinyaller reddedilme mekanizması yüzünden ters çalışır |
| 5 | SEAD normal | Alarm yok ✅ (doğru) | Sistem gürültüyü alarm saymıyor |
