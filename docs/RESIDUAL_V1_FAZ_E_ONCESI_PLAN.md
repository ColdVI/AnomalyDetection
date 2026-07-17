# RESIDUAL-V1 — Faz E Öncesi Uygulama Planı (Codex için)

Tarih: 2026-07-17 · Statü: **Uygulandı — K5/S-4/ölçekleme/S-1/S-3 tamamlandı;
kalibrasyon yetersiz development-normal maruziyeti nedeniyle STOP.**
Kaynak: Görev 4.1 sonrası kullanıcı tarafından dondurulan sıra + bu oturumda yapılan
bağımsız kod denetiminin bulguları. Bu belge `docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md`
ve `docs/RESIDUAL_V1_DENEY_TASARIMI.md`'yi günceller/tamamlar, çelişmez.

## Dondurulmuş sıra

1. K5 — waypoint ±2 s maskesi + testi
2. S-4 — komutsuz-girdi (AR/sızıntı) ablasyonu
3. Train-normal robust median/MAD ölçekleme
4. S-1 — büyüklük korelasyonu
5. S-3 — development olay ayrımı (ALFA için `not_evaluable/model_unavailable` kuralıyla)
6. Yalnız S-3 geçerse → CUSUM/kalibrasyon

Görev 5.2'deki tanıma göre **S-2 bu sırada yok** — G1 ridge için S-2 zaten atlanıyor
(`docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md` Görev 5.2: "G1 ridge için S-2 atlanır,
katsayı-sanity zaten var"), bu tutarlı.

Bu paragraf planın yazıldığı andaki başlangıç durumuydu. Güncel uygulama sonucu
`docs/RESIDUAL_V1_FAZ_E_SONUCLARI.md` dosyasındadır.

---

## 1. K5 — waypoint ±2 s maskesi

**Neden gerekli:** `docs/RESIDUAL_V1_GOREV_4_1_SARTLI_GO_KAYDI.md` K5: "R6 için waypoint
değişimi çevresindeki ±2 saniye maskesi henüz yoktur." R6 (`xtrack_error`), K6 gereği G1
ridge'e girmiyor; Görev 5.1'de doğrudan robust-z kanalı olacak. Tasarım dokümanı §3.1'de
zaten not düşülmüş: "Waypoint geçişinde yapısal sıçrama → waypoint-değişim maskesi (±2 s)."
Bu maske olmadan robust-z/CUSUM waypoint geçişlerini sahte anomali sayabilir.

**Düzeltme (Codex'in teşhisi + bu oturumda ayrıca bağımsız doğrulandı):** Önceki
sürümdeki iki iddia yanlıştı, düzeltiliyor:

- Mapping (`field.wp_dist` → `waypoint_distance`) `residual_v1/ingest/alfa.py:85`'te
  — `alfa_channels.py:85` değil (o dosyada yalnız `ChannelSpec` tuple'ları var, satır
  85 orada farklı bir kolona denk geliyor). Yanlış ankraj benim hatamdı.
- "Silver'a ulaşıyor mu" açık sorusu çözüldü: **ulaşıyor.** `residual_v1/features/align.py`
  okundu — `align_to_clock()` `flight.items()` üzerinden TÜM topic kolonlarını taşıyor;
  `ChannelSpec`/`ALFA_CHANNELS` yalnızca `default_tolerances()` içinde TOPIC-bazlı (kolon-
  bazlı değil) hizalama toleransı için kullanılıyor. `xtrack_error` zaten aynı topic'i
  (`mavros-nav_info-errors`) declare ettiği için `waypoint_distance` de otomatik olarak
  aynı toleransla taşınıyor. `ChannelSpec` declare edilmemesi yalnız `profile.py`'deki
  otomatik range/staleness hijyen denetimini devre dışı bırakıyor — kolonu düşürmüyor.
  Codex bunu hem eski hem K4-düzeltilmiş Silver kökünde 47/47 uçuşta doğruladı; ben de
  ayrıca farklı bir uçuşta (`carbonZ_2018-07-18-12-10-11_no_ground_truth`) kolonun
  var olduğunu ve tek-adım fark dağılımını bağımsız ölçtüm (aşağıya bkz.).

Yine de veri sözleşmesi/hijyen denetimi için `waypoint_distance`'ın
`alfa_channels.py`'ye `ChannelSpec("waypoint_distance", "mavros-nav_info-errors", "m",
0.0, <makul_üst_sınır>, 10.0, False, "context")` olarak eklenmesi doğru — bu artık
"gerekli mi" sorusu değil, "iyi pratik" maddesi.

**Geçiş-tespit algoritması değişti — orijinal "sıçrama/reset" varsayımı gerçek veriyle
çürütüldü.** Codex'in development-only ölçümü: pozitif tek-adım farkların medyanı +1 m,
%99.9'u +3 m, maksimum +4 m; `Δwp_dist > 5 m` aday sayısı **0**. Ben de bağımsız bir
uçuşta ölçtüm: tek-adım max fark tam +3.0 m, min −3.0 m — sıçrama yok. Gerçek örüntü
Codex'in `.tmp_pdf_reader/k5_waypoint_candidates.png` görselinde net: waypoint'e
yaklaşırken azalan, sonra artan bir **V-dönüşü**. Görseldeki 9 adayın hepsi (6 uçuş)
bu şekli gösteriyor; bazılarında `xtrack_error` de aynı noktada gerçek bir sıçrama
yapıyor (tasarım §3.1'in öngördüğü "yapısal sıçrama" xtrack'te var, ama onu bulmak için
kullanılacak bağımsız sinyal wp_dist'in sıçraması değil, V-dönüşünün kendisi).

**Dondurulacak K5 sözleşmesi (Codex'in development verisiyle türettiği, öneriliyor):**
```
maximum_turn_distance_m       = 25   # V-dönüşü sayılması için wp_dist bu değere inmeli
trend_window_s                 = 2   # trend (azalan→artan) bu pencerede ölçülür
minimum_approach_excursion_m  = 10   # yaklaşma bacağında en az bu kadar azalma
minimum_departure_excursion_m = 10   # ayrılma bacağında en az bu kadar artış
minimum_event_separation_s     = 5   # birbirine bu kadar yakın adaylar tek olay sayılır
mask_buffer_s                  = 2   # K5'in kendisi — nihai maske genişliği
```
10 m eşiği tek-adım gürültü tavanının (±3 m) ~3 katı — gürültüden yanlış tetiklenmeye
karşı savunmalı. Bu parametreler development'ta görsel+sayısal doğrulamayla türetildi
(tasarım §2 ilkesiyle uyumlu: sonuç görülmeden önce dondur); test/holdout'a hiç
bakılmadı. Kabul ediyorum, değiştirmeden Codex'e devrediyorum.

**Önerilen adımlar:**
1. `alfa_channels.py`'ye `waypoint_distance` ChannelSpec'ini ekle (yukarıdaki "iyi pratik"
   maddesi — artık ingest için zorunlu değil ama hijyen raporu için gerekli).
2. Yukarıdaki dondurulmuş sözleşmeyle V-dönüşü tespiti yaz: `trend_window_s` içinde
   önce `minimum_approach_excursion_m` azalma, ardından `minimum_departure_excursion_m`
   artış, dönüş noktasında `wp_dist ≤ maximum_turn_distance_m`; `minimum_event_separation_s`
   içindeki adaylar birleştirilir.
3. Maskeyi uygula: `residual_v1/features/phases.py::label_phases` deki `phase_boundary`
   desenine benzer ikinci bir bayrak (`waypoint_boundary`, ±`mask_buffer_s`) — ayrı config
   (`configs/residual_v1_waypoint_mask.json`, yukarıdaki 6 alan). **Yalnız R6'yı etkilemeli**;
   diğer kanallara karıştırma.
4. `residual_v1/features/build.py::build_xy`'e R6-özel maskeyi ekle — spec.py'ye
   `boundary_masks: tuple[str,...] = ()` gibi bildirimsel bir alan eklenip R6'da
   `("waypoint",)` verilmesi, `spec.name` string-karşılaştırmasından daha temiz.

**Test:** Sentetik `waypoint_distance` fixture'ı (bilinen V-dönüşü zamanlarıyla, gerçekçi
±3 m tek-adım gürültüsü enjekte edilmiş) üzerinde: (a) V-dönüşleri doğru tespit ediliyor
mu, (b) gürültü (±3 m, ≤10 m toplam) yanlış tetiklemiyor mu (negatif test), (c) maskenin
±`mask_buffer_s` genişliği doğru mu, (d) `minimum_event_separation_s` birleştirme doğru
mu, (e) R6 dışındaki kanallar ETKİLENMİYOR mu. `tests/test_residual_v1_waypoint_mask.py`,
mevcut `test_residual_v1_phases.py` deseniyle tutarlı.

---

## 2. S-4 — komutsuz-girdi ablasyonu

`docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md` Görev 4.3'te zaten tam tanımlı:
`scripts/residual_v1_s4_ablation.py` — seçilen G1 modelini komut girdileri (spec'in
`command_inputs`'u, `context_inputs` değil) çıkarılmış halde yeniden eğit;
`var(r_sakat)/var(r_tam)` < 1.15 → FLAGGED, `flags.json`'a yaz. FLAGGED kanal karar
katmanına giremez. K5'ten SONRA çalışmalı çünkü R6 hâlâ G1'e girmiyor (K6) — bu adım
yalnız RFLY Q1/Q2/Q3'ü etkiler (ALFA'da zaten eğitilmiş model yok).

## 3-4. Ölçekleme + S-1 (Görev 5.1 + 5.2'nin ilk yarısı)

`residual_v1/decision/scaling.py`: kanal başına train-normal median/MAD; MAD=0 → kanal
dışla + `excluded_channels` manifestine yaz; z clip=8. S-1: `residual_v1/eval/sanity_gates.py`
içinde, uçuş-içi ortalama |z| vs aynı pencere ham-girdi normu ‖x‖ Spearman ρ; ρ≥0.5 → FLAG.
**Not:** S-1 hem RFLY Q1/Q2/Q3 hem ALFA R6 (K5 sonrası) için koşulabilir — R1-R5 için
model yok, S-1 onlara uygulanamaz (bkz. §5).

**Yeni açık nokta (Codex): R6'da S-1 tautolojik olabilir.** R6 (K6 gereği) öğrenilmiş
bir modelin residual'ı değil, `xtrack_error`'ın doğrudan robust-z'si — yani
`z = (xtrack − median)/MAD`, ham girdinin kendisinin afin dönüşümü. S-1'in amacı
"skor aslında öğrenilmiş bir modelin ham girdi büyüklüğünü mü takip ediyor" sorusunu
sormak (temsil sızıntısı testi); R6'da öğrenilmiş bir temsil yok, dolayısıyla
`|z(xtrack)|` ile `|xtrack|` karşılaştırması bir şeyi kendisiyle karşılaştırmak olur —
median sıfıra yakınsa ρ neredeyse kesin ≥0.5 çıkar ve anlamsız bir FLAG üretir.
İki seçenek var: **(a)** R6'yı S-1'den açıkça muaf tut (basit ama S-1'in yakalamaya
çalıştığı gerçek riski — "yüksek |z(xtrack)| sadece agresif manevra mı" — hiç test
etmemiş olursun); **(b)** R6 için S-1'in "ham girdi" referansını xtrack'in kendisi
değil, bağımsız bir manevra-büyüklüğü vekiliyle (ör. roll/dönüş hızı normu veya
`build.py`'deki `context_speed`/`__speed_interaction` desenine benzer bir toplam
komut büyüklüğü) yeniden tanımla — bu, S-1'in R1-R5/Q1-Q4'teki asıl sorduğu soruyla
(skor manevra büyüklüğünü mü takip ediyor) tutarlı kalır. **Öneri: (b).** Ama K5'teki
gibi, hangi vekil sinyalin kullanılacağı development verisinde kısa bir kontrolle
seçilip dondurulmalı — şimdiden formül önermiyorum, bu Codex'in K5 sonrası ilk işi
olmalı.

## 5. S-3 — development olay ayrımı, ALFA `not_evaluable` kuralı

`residual_v1/eval/sanity_gates.py`, Görev 5.2 tanımına göre: KS testi, |z| dağılımı
`[onset, onset+15s]` vs `[onset−60, onset−10s]`, headline sınıf başına en az bir kanal
p<0.01 değilse **STOP** + `S3_FAILURE_REPORT.md`.

**Kullanıcının eklediği kritik kural:** ALFA-engine sınıfı için R1-R5'te eğitilmiş
model YOK (Görev 4.1 sonucu — tek oturum, `InsufficientSessionCoverage`). S-3 kodu
ALFA-engine'i değerlendirirken:
- R1-R5 kanalları için sonucu `not_evaluable` / `model_unavailable` olarak işaretlemeli
  (KS testi çalıştırmaya bile kalkışmamalı — model yok, z de yok).
- ALFA-engine'in TEK test edilebilir kanalı, K5 sonrası R6 (xtrack_error, doğrudan
  robust-z). ALFA-engine için S-3 PASS/FAIL kararı yalnız R6 üzerinden verilmeli.
- **RFLY'nin S-3 sonucu ALFA-engine sınıfına asla sızdırılmamalı** — iki ayrı sınıf,
  iki ayrı PASS/FAIL. Rapor şablonunda bu üç durumu (PASS / FAIL / not_evaluable)
  ayrı satırlar olarak göster, "genel PASS" gibi tek bir birleşik özet üretme.

Bu kural `S3_FAILURE_REPORT.md` şablonuna ve `sanity_gates.py`'nin dönüş tipine
(`GateResult` gibi bir yapıya `not_evaluable` durumu eklenmesi) baştan yazılmalı —
sonradan yama olarak eklenmemeli.

## 6. CUSUM + kalibrasyon (Görev 5.3-5.4) — yalnız S-3 PASS ise

`residual_v1/decision/cusum.py` (mevcut `anomaly_core.sequential.MultiChannelPageCUSUM`
sarmalayıcı, k=1.0, iki yönlü, refractory 60s) ve `decision/calibrate.py` (blok-bootstrap,
B=500, kanal FA katkısı = 0.5/aktif kanal sayısı). `thresholds_frozen.json` fail-if-exists.
Kod zaten `raise GateError` ile S-3 PASS koşuluna programatik bağlanacak şekilde
tasarlanmış (Görev 5.2 metni) — bu bağlantının gerçekten var olduğu, S-3 atlanarak
kalibrasyona geçilemeyeceği ayrı bir testle kanıtlanmalı (`GateError` fırlatma testi).

---

## Codex'in bu plana başlamadan teyit etmesi gereken açık noktalar

1. ~~`waypoint_distance` Silver'a ulaşıyor mu~~ — çözüldü, ulaşıyor (bkz. §1).
2. ~~Waypoint-geçiş tespit eşiği/pencere genişliği~~ — çözüldü, Codex'in development-
   ölçümlü V-dönüşü sözleşmesi kabul edildi (bkz. §1, dondurulmuş 6 parametre).
3. `GateResult`/rapor şemasına `not_evaluable` durumunun nasıl ekleneceği — mevcut
   `G1ChannelFit.report`/`coverage` sözleşmesiyle tutarlı bir alan adı seçilmeli.
4. ~~R6 için S-1'in ham-girdi referansı~~ — çözüldü. Dondurulan bağımsız vekil
   `sqrt((roll/rad(25°))² + (roll_rate/rad(15°/s))²)`; development S-1 sonucu
   Spearman rho=0.471774 < 0.5.

## Codex'in bu turda yaptığı ve doğrulanan iş

Citation düzeltmesi (`alfa.py:85`), ChannelSpec'in ingest filtresi olmadığı bulgusu
(47/47 uçuş, iki Silver kökü) ve V-dönüşü örüntüsü (development ölçümü + görsel) bu
oturumda ayrıca bağımsız olarak (farklı bir uçuşta, ayrı bir Python komutuyla) teyit
edildi — üçü de doğru çıktı, plana aynen işlendi.

Bu belge uygulama öncesi planı ve karar izini korur; güncel kod/sonuç ankrajları için
`docs/RESIDUAL_V1_FAZ_E_SONUCLARI.md` esas alınır.
