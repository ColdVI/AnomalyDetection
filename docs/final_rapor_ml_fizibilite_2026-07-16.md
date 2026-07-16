# İHA/Uçuş Telemetrisi Anomali Tespiti — Kapsamlı Final Raporu ve Fizibilite Hükmü

- Tarih: 2026-07-16
- Kapsam: Bu repoda yapılmış TÜM anomali-tespit çalışması — arşivlenmiş fazlar (ML-0..ML-16,
  RFLY-0/1, reddedilen ilk ADS-B denemeleri) ve aktif ADS-B hattı (ADR-021..ADR-042) dahil.
- Amaç: Üst yönetime/danışmana sunulabilir, "neyi neden denedik, ne sonuç aldık, neden
  yetmedi" sorularının tamamına sayısal kanıtla cevap veren tek belge.

---

## 0. Yönetici özeti — hüküm

**Proje, mevcut veri/etiket/operasyonel-bütçe koşulları altında hedeflenen ürünü
("düşük yanlış-alarmla gerçek anomalileri operasyonel olarak yakalayan tespit sistemi")
ulaşılabilir kılmamıştır.** Bu hüküm bir izlenim değil, ölçümdür:

- **5 veri seti** (ALFA, UAV Attack, UAV-SEAD, RflyMAD, ADS-B/adsb.lol) üzerinde,
- **17+ faz** boyunca (ML-0..ML-16, RFLY-0/1, ADS-B 8-adımlı hat + contextual aday),
- **12+ yöntem ailesi** denendi (istatistiksel, kural-bazlı/fiziksel, klasik ML,
  4 farklı derin-öğrenme mimarisi, foundation-model, konformal kalibrasyon, füzyon/kanal
  mimarileri),
- ve **hiçbir konfigürasyon, önceden ilan edilmiş operasyonel kapıyı (recall + yanlış-alarm
  bütçesi BİRLİKTE) geçemedi.**

Başarısızlığın nedenleri tahmin değil, tek tek deneyle izole edilmiş **yapısal veri/etiket
sorunlarıdır** (bkz. §6): küçük ve heterojen etiketli örneklem, proxy-etiket hataları
(bulunup düzeltildi, sayılar düştü), derin modelleri sahte başarıya götüren genlik-baskınlığı
artefaktı (bulundu, ölçüldü, giderilince altında sinyal kalmadı), ve son olarak ADS-B'de
saat-bazlı alarm bütçesi ile tek-uçuş olay penceresi arasındaki birim uyuşmazlığı.

Yol boyunca **gerçek, savunulabilir kazanımlar da üretildi** (bkz. §7): basit kural-bazlı
skorlayıcının üç sinir ağını geçtiğinin gösterilmesi, genlik-baskınlığı teşhis protokolü,
proxy-etiket düzeltme metodolojisi, ve tek başına %49.7 recall'a ulaşan CUSUM konum-sapması
dedektörü. Ama bunların hiçbiri, hedeflenen operasyonel ürün iddiasını taşımaya yetmedi.

**Projenin ulaşılabilir hale gelmesi için gerekenler** (bizim elimizde olmayanlar): gerçek,
etiketli, yeterli hacimde anomali örneği (sentetik enjeksiyon değil); ya da operasyonel
bütçe tanımının baştan farklı bir birimde konulması; ya da her ikisi. Detay §8'de.

---

## 1. Hedef ve başarı kriterleri (önceden ilan edilmiş kapılar)

Proje baştan itibaren "sonucu görüp eşik ayarlamak" hatasına karşı **önceden kayıtlı (pre-registered)
kapı disipliniyle** yürütüldü. Her faz, sonuç görülmeden yazılı kabul kriteri koydu:

| Kapı | Tanım |
|---|---|
| Gate A | Güvenlik/determinizm: veri sızıntısı yok, blind holdout kapalı, checksum bütünlüğü |
| Gate B | Yeni yöntem, mevcut en iyiyi önceden tanımlı marjla (+0.05 recall, ≥3/5 seed) geçmeli |
| Gate C | Operasyonel hedef: **critical ≥0.30 recall @ ≤2 yanlış-alarm/saat; advisory ≥0.50 @ ≤12** |
| ADS-B Pareto | 100 skorlanabilir uçuş-saatinde 0.1 / 0.5 / 1 / 2 / 5 alarm bütçe ızgarası |

Sonuç özeti: **Gate A neredeyse her fazda geçti** (altyapı sağlam), **Gate B iki kez geçti**
(ML-10 Chronos-mekanik; ML-12 tek-feature itki_komutu), **Gate C / operasyonel kapı hiçbir
fazda, hiçbir veri setinde, hiçbir yöntemle geçilmedi.**

---

## 2. Veri setleri — kaynak, ön-işleme, temizlik, enjeksiyon

### 2.1 ALFA (Carnegie Mellon; sabit-kanat İHA, gerçek uçuş arızaları)

- **İçerik:** 47 işlenmiş uçuş (makaledeki sayının tamamı — eksik indirme değil, veri setinin
  doğal boyutu); motor/kumanda-yüzeyi arızaları. Kritik sınırlama: rudder arızası **4 uçuşta**,
  aileron+rudder **1 uçuşta** — bu sınıflarda her "tespit oranı" istatistiksel olarak anlamsız.
- **Ön-işleme:** processed CSV → Bronze (ham, değiştirilmeden) → Silver parse (birim dönüşümü,
  topic birleştirme `merge_asof`) → Gold 7+3 ortak kolon. Ek olarak ham rosbag'lerden 7 uçuş
  daha parse edilerek havuz 54 uçuş / 15 normale çıkarıldı (ML-4).
- **Temizlik/düzeltmeler:** `velocity_mps` %100 null çıkıyordu — kök neden kolon adlandırması
  (`nav_info-velocity` alanları `meas_x/des_x` adında); düzeltilince null %9.3'e indi.
- **Feature:** 73→85 el yapımı feature (wrap-aware açı farkları, haversine, CUSUM, spektral,
  freeze sayaçları, otopilot residual'ları örn. `alt_error`, `xtrack_error`, fizik-prior
  `turn_residual = yaw_rate − g·tan(roll)/V`).
- **Enjeksiyon:** `src/ml/injection.py` — freeze / bias / drift / noise / gps_ramp / dropout,
  6 senaryo, birim testli.

### 2.2 UAV Attack (PX4 çok-rotorlu; GPS spoofing / jamming / DoS)

- **İçerik:** 683.9 MB zip, 767 CSV; benign/malicious etiketli.
- **Temizlik/düzeltmeler:** (1) `split_log_and_topic` regex'i gerçek dosya adlarında topic'i
  yanlış bölüyordu — bilinen 5 topic adına ankrajlı tam-eşleşmeyle düzeltildi. (2) Ping-DoS
  satırları etiketsiz kalıyordu — `infer_label_from_path`'e ping/dos kuralı eklendi.
- **Feature:** 58 feature (`build_px4_features`); en değerli türetim `gps_speed_residual`
  (konumdan hesaplanan hız vs alıcının bildirdiği hız) — gizli/gerçekçi spoofing'i yakaladı.
- **Bilinen kapsam sınırı (dürüstçe raporlandı):** Ping-DoS imzası 6 logun 4'ünde fiziğe hiç
  yansımıyor — bu topic'lerden tespit YAPILAMAZ; kapsam beyanı olarak kayıtlı.

### 2.3 UAV-SEAD (HuggingFace; dron GPS-spoofing + irtifa/mekanik/konum anomalileri)

- **İçerik ve büyütme tarihi:** 60 → 179 → 349 → 611 → **1.044 uçuş** (899 normal; 200 uçuşluk
  kör holdout hiç açılmadı). Tek-sınıf havuzları fiilen tüketildi (mechanical/global_position/
  altitude %100 indirildi).
- **Ön-işleme:** pyulog ile ULog parse; iç-mekân uçuşlarında GPS yok →
  `vehicle_local_position` öklid fallback; EKF innovation/test-ratio kolonları Silver'a eklendi.
- **Kritik split disiplini:** normal-only train, uçuş-bazlı split, 5 seed; ML-5'te **oturum-bazlı
  split** (aynı gün/oturumdaki uçuşlar aynı tarafta) — seed varyansını ±0.212'den ±0.012'ye
  indirdi ve adil satır-ROC'u 0.474→0.799'a taşıdı (kazanç modelden değil, veri temizliğinden).
- **Yapısal veri sorunu (ölçüldü):** 398-normal aşamasında havuz yalnız **64 bağımsız oturuma**
  dağılıyordu (development: 49) — gerçek bağımsız örneklem uçuş sayısının çok altında; büyütme
  oturum çeşitliliğini aynı oranda artırmadı. Literatür (Heterogeneous Normal Classes) bu durumda
  veri ARTSA BİLE performansın düşebileceğini söylüyor; ML-14'te birebir yaşandı (aşağıda).
- **Etiket sorunu:** `alt_local_residual` altitude-anomali sınıfında %0 dolulukta (topic yapısal
  olarak eksik — veri büyütmeyle düzelmedi); en az bir "normal" etiketli uçuşta donmuş-GPS/
  eph≈25000 sentineli bulundu (genlik-baskınlığı artefaktını besleyen örneklerden biri).

### 2.4 RflyMAD (gerçek + simüle dron; motor/sensör arızaları)

- **İçerik:** 490 **gerçek** uçuş (Real-Motor 242, Real-Sensors 197, Real-No_Fault 51).
  Simülasyon alt-kümeleri (SIL/HIL-Wind, 886 case) indirildi ama gerçek-normal havuzuyla
  ASLA karıştırılmadı (ayrı split sözleşmesi).
- **İki gerçek hata bulunup düzeltildi:** (1) "hayalet imputation" — bir kaynakta hiç olmayan
  kolon başka kaynağın medyanıyla dolduruluyordu; kapatıldı. (2) **whole-flight proxy etiketi** —
  ilk resmi sonuç (0.749 recall) arızalı uçuşun TAMAMINI anomali sayıyordu. ULog içindeki
  `rfly_ctrl_lxl` mesajından gerçek `(fault_onset, fault_end)` aralığı çıkarıldı; 5 uçuşta bu
  mesaj klasör etiketiyle çelişti ("arıza hiç tetiklenmedi") → sonuç görülmeden dışlandı.
- **Düzeltilmiş sonuç:** RFLY-only 0.526 recall / 22.28 FA-saat (recall hedefi geçiyor ama FA
  bütçenin ~2 katı); SEAD+RFLY havuzlaması 0.149 / 30.00 (havuzlama ciddi KÖTÜLEŞTİRDİ —
  heterojen-normal dersinin ikinci deneysel teyidi). İkisi de operasyonel kapıda kaldı.

### 2.5 ADS-B (adsb.lol; etiketsiz gerçek hava trafiği — güncel odak)

- **İçerik:** 3 tam gün (2026-02-28 / 03-01 / 03-16), ~3'er GB tar, **256.15M satır**, 638
  Silver parçası, ~542 bin uçuş segmenti, ~497.571 skorlanabilir uçuş-saati. (Provenance
  sayımında +4.459 satırlık açıklanamayan fark bulundu — sessizce düzeltilmedi,
  "çözümsüz-kayıtlı" olarak manifest'e yazıldı.)
- **Ön-işleme:** tar streaming parse (tek kanonik parser — iki bağımsız denemenin aynı parser'ı
  kullandığı ayrıca denetlendi, tutarsızlık yok) → Silver; uçuş segmentasyonu 1800 s boşluk
  kuralı; `flight_id`'ye gün öneki; roller (fit/calibration/development/rehearsal) dosya VE
  uçuş düzeyinde manifest'le kilitli.
- **Feature (öğrenilmemiş, aritmetik fizik residual'ları):** `vertical_rate_residual`
  (bildirilen dikey hız vs irtifa türevi), `speed_residual` (bildirilen yer hızı vs ardışık
  konumdan hesaplanan), `heading_residual` (bildirilen rota vs kerteriz), `east/north_velocity_residual`
  (hız vektörü vs konum türevi), `altitude_source_residual` (barometrik vs geometrik irtifa
  uyumu — literatürdeki self-consistency spoofing tespitinden motive), `turn_bank_residual`
  (yalnız roll doluysa). Veri-kalitesi sinyalleri (mesaj boşluğu, NIC, kaynak tipi) ayrı
  **S2 katmanında**, fizik kanallarına karıştırılmadan.
- **Ölçekleme kuralları:** train-normal median/MAD; MAD=0 kanal floor'lanmaz, TAMAMEN dışlanır
  ve manifest'e yazılır; robust clip=5. (Bu iki kural, SEAD'deki genlik-baskınlığı ve arşivdeki
  zero-MAD çökmesi derslerinin doğrudan sonucu.)
- **Enjeksiyon/sentetik:** `PHYSICS_BREAK_RECIPES` — 5 senaryo: `vertical_rate_frozen`,
  `ground_speed_biased`, `track_frozen`, `position_ramp_stealthy`, `altitude_dropout`.
  Kalıcı corpus: 8.910 uçuş × (temiz + 5 senaryo) = 26.8M satır, SHA-256'lı, fail-if-exists.
  **Katı kural: sentetik veri hiçbir zaman fit/ölçekleme/kalibrasyona girmez — yalnız
  değerlendirme.** (Runtime'da `data_role`/`contains_synthetic` kontrolüyle zorlanıyor,
  konvansiyonla değil.)
- **Etiket düzeltmesi (truth-v2):** İlk sentetik değerlendirme "onset'ten dosya sonuna kadar
  her satır anomali" proxy'si kullanıyordu — bu, ölçülebilir AUC'ye **~0.75 tavan** koyuyordu
  (dedektör ne kadar iyi olursa olsun). RFLY-1'dekiyle aynı hata sınıfı. truth-v2 satır düzeyinde
  `injection_active` / `observable_changed` / `evaluable_truth` ayrımı getirdi (666.739 aktif
  dropout satırının yalnız 570.666'sı gerçekten gözlemlenebilir-değişmiş — naif proxy'nin
  ne kadar fazla saydığının kanıtı).

---

## 3. Denenen yöntemler — tam envanter

### 3.1 İstatistiksel / matematiksel / kural-bazlı

| Yöntem | Nerede | Neden seçildi | Sonuç |
|---|---|---|---|
| Modüler Isolation Forest + val-normalize füzyon | ALFA/UAV/SEAD (ML-1..14) | Etiketsiz novelty tespiti; modülerlik kategori atfı verir | ALFA uçuş-ROC 0.833; ama monolitik satır-IF 0.50/0.21 (başarısız); SEAD'de operasyonel kapıyı hiç geçemedi |
| CUSUM (causal, train-normal baseline) | ALFA/SEAD karar katmanı | Kalıcı küçük kaymaları biriktirir | Dürüstlük düzeltmesi: "en güçlü sinyal" alt_error 0.878'in causal ölçümü **0.611** çıktı; gerçek en güçlü xtrack_error 0.751 |
| POT/GPD uç-değer eşiği | ALFA/SEAD | Eşiği uç-değer teorisiyle seçmek | ML-5'te ilk kez val-max'ı geçti, uçuş eşiğinde varsayılan yapıldı; kapıları tek başına açmadı |
| Robust z-score kural skorlayıcısı (`ResidualRuleScorer`) | ADS-B (ADR-025/028) | Öğrenmesiz, genlik-artefaktı yapısal olarak imkânsız | **Üç sinir ağını da geçti** (havuz AUC 0.600 vs 0.552–0.572); truth-v2 düzeltmesiyle AUROC 0.765/AUPRC 0.883; ground-speed/track olay recall 0.96/0.95 (gecikme 19 s/57 s) — ama doğal alarm yükü donmuş eşikte 4.81 alarm/saat (uçuşların %89.2'si alarmlı) → operasyonel değil |
| İki-eksenli Page-CUSUM (`VectorPageCUSUM`) | ADS-B konum-sapması | Sinsi konum kayması tek satırda görünmez, birikim şart | **Projenin en iyi tekil recall'ı: %49.7** (en gevşek bütçede), en sıkıda bile %7.1; doğal yükü de en sessiz kanal |
| Hiyerarşik konformal kalibrasyon | ADS-B contextual | Skorlar Gauss değil; "bu skor ne kadar nadir" sorusuna dağılımsız cevap | Çalıştı (16M satırlık kalibrasyon tablosu, sağlıklı kapsam) — ama tek-uçuş penceresi sınırı recall'ı boğdu (ADR-042) |
| Oturum-jackknife drift kalibrasyonu | SEAD (ML-15) | Val→test FA kayması sorununa istatistiksel düzeltme | FA'yı bütçe içine çekti ama recall açığını kapatamadı (en iyi 0.1145 @ 8.41) |

### 3.2 Klasik makine öğrenmesi

| Yöntem | Nerede | Neden | Sonuç |
|---|---|---|---|
| LightGBM (pencere-tanımlayıcı, sınıf-dengeli) | SEAD (ML-8A) | "Etiket az da olsa süpervizyon katkı verir mi?" | **Hayır**: AUPRC 0.349 < mevcut IF 0.385; literatürle tutarlı (az etiketli veride yarı-denetimli > tam-denetimli) |
| Isolation Forest (ADS-B contextual keşfi) | ADS-B (ADR-038) | LSTM'den farklı kör-nokta profili umudu | Elendi: eğitilmiş skor, ham genlikle VE kanalları karıştırılmış eğitimsiz-eşdeğerle ρ≈0.996 — hiçbir çok-değişkenli yapı öğrenmiyor |

### 3.3 Derin öğrenme

| Yöntem | Nerede | Neden | Sonuç |
|---|---|---|---|
| LSTM-Autoencoder | ALFA/UAV/SEAD/ADS-B | Zamansal örüntü + reconstruction klasik yaklaşım | ALFA'da veri büyütmeyle 0.731→**0.918** (tek gerçek DL başarısı, küçük veri hipotezinin teyidi); SEAD'de Gate B kaldı ve **genlik-baskınlığı** çıktı; ADS-B'de FLAGGED |
| Dense-Autoencoder | SEAD/ADS-B | LSTM'e parametre-eş basit kontrol | SEAD'de aynı sonuç, aynı artefakt; ADS-B'de ölçekleme sonrası FLAGGED |
| USAD (adversarial çift-decoder) | ALFA/SEAD/ADS-B | Literatürde reconstruction'dan güçlü iddiası | ALFA'da LSTM-AE'ye kaybetti (elendi); SEAD'de aynı artefakt; ADS-B'de loss sayısal patladı (23 milyar) — ρ testinin tek başına yetmediğinin kanıtı |
| Chronos-bolt-tiny (zero-shot foundation model) | SEAD (ML-10) | Eğitimsiz forecast-residual; MOMENT kurulamadı (gerçek, tekrarlanabilir kurulum hatası) | **Gate B mekanik dalda GEÇTİ** (0.205→0.390) — ama füzyonda kayboldu, Gate C kaldı; dikey kanalda 0.096→0.023 geriledi |
| LSTM next-step forecaster (`contextual_physics_v1`) | ADS-B (ADR-033..042) | Reconstruction yerine tahmin; faz/kadans bağlamı açık girdi; genlik artefaktına yapısal önlem | **Genlik kapısını geçen tek NN** (ρ=0.65 vs FLAGGED 0.84–0.99); doğal yanlış-alarm kontrol altında; AMA gerçek recall'da 5 profilin 4'ü <%6 (ADR-042, kök neden bütçe birimi) |

### 3.4 Mimari / skor-katmanı denemeleri

| Deneme | Neden | Sonuç |
|---|---|---|
| Max-füzyon (tüm modüllerin maksimumu) | Tek operatör skoru | Yüksek-FA'lı kategori uzmanlarını bütçede kullanamıyor (ML-12'de ölçüldü: itki_komutu 38.1 FA-saat bırakıyor) |
| İnce modül (tek-feature IF) | Seyrelme hipotezi: 16-feature modülde güçlü feature sulanıyor | **Gate B geçti** (0.205→0.459, bilinen en iyi kategori skoru) — Gate C yine kaldı |
| İki-kanal mimarisi (sistem + mekanik ayrı alarm) | Uzman modülü ayrı bütçeyle yaşatmak | Recall +0.074..0.105 ama FA 1.89–3.83× şişti — kazanım FA ile satın alındı, reddedildi |
| Genlik-normalize skorlar (relerr / rankpct) | Artefaktı skor katmanında gidermek (yeniden eğitimsiz) | **Belirleyici negatif sonuç**: relerr genlik bağımlılığını gerçekten kırdı (ρ 0.15–0.55) ama recall neredeyse yok oldu (<0.06) — ham kazancın altında öğrenilmiş sinyal YOKTU |
| Veri büyütme (SEAD 324→899 normal) | FA'yı doğal çeşitlilikle düşürmek | FA 23.6→9.95 düştü AMA recall 0.21→0.126 düştü — klasik değiş-tokuş; kapı yine açılmadı |

---

## 4. Kronoloji — faz faz, kapı sonuçlarıyla

| Faz | Tarih | Ne yapıldı | Kapı sonucu |
|---|---|---|---|
| ML-0..1 | 07-02 | Feature/split/scaler altyapısı; monolitik IF çöktü, modüler IF+füzyon kuruldu | Keşif (kapı yok); ALFA uçuş-ROC 0.833 |
| ML-2 | 07-02 | Dense-AE + LSTM-AE, POT eşiği | ALFA'da IF > LSTM-AE; UAV'de AE > IF |
| ML-3 | 07-02 | Enjeksiyon çerçevesi + ablasyon + USAD | Kalibre transfer tezi kanıtlandı (0.375→0.783); USAD elendi |
| ML-4 | 07-03 | Veri büyütme (ALFA rosbag) | **ALFA LSTM-AE 0.918** — tek net DL kazanımı |
| ML-5 | 07-03 | SEAD 349 uçuş + oturum-split | Satır-ROC 0.474→0.799 (veri temizliğinden); H14: EKF test-ratio TERS sinyal |
| ML-6/7 | 07-06 | Causal CUSUM + event-onset düzeltmesi + kör holdout | Şişkin sayılar düştü (0.878→0.611; overlap 0.594→onset 0.194) — dürüstlük düzeltmesi |
| ML-8A | 07-06 | LightGBM süpervizyon denemesi | Gate B **KALDI** (0.349 < 0.385) |
| ML-9 | 07-06 | EKF innovation + motor-simetri feature'ları | Gate B KALDI (kazançlar +0.02 düzeyinde) |
| ML-10 | 07-06 | Chronos zero-shot forecast-residual | Gate B mekanikte **GEÇTİ** (0.390); Gate C KALDI (füzyonda kayboldu) |
| ML-11 | 07-06 | Read-only görselleştirme/teşhis | actuator_thrust_cmd tek başına AUC 0.983 bulundu → ML-12 hipotezi |
| ML-12 | 07-07 | Tek-feature ince modül | Gate B **GEÇTİ** (0.459 kategori rekoru); Gate C KALDI (38.1 FA-saat uzman sorunu) |
| ML-13 | 07-07/08 | İki-kanal alarm mimarisi | Gate B KALDI (FA 1.89–3.83× şişti) |
| ML-14 | 07-08/09 | SEAD normal 899'a büyütme | FA düştü, recall de düştü; hedef yine ıska (0.126 @ 9.95) |
| ML-15 | 07-09/10 | Drift kalibrasyonu tam matris | Gate B/C KALDI (0.1145 @ 8.41) |
| RFLY-0/1 | 07-09 | RflyMAD + interval-truth düzeltmesi | 0.749 proxy → 0.526 gerçek / 22.28 FA; Gate R-C KALDI |
| ML-16 L/D/U | 07-09/10 | LSTM-AE / Dense-AE / USAD SEAD'de resmi hatta | Üçü de Gate B KALDI + **genlik-baskınlığı artefaktı bulundu** (ρ=0.964/0.965) |
| ML-16 N | 07-10 | Genlik-normalize skorlar | Artefakt giderilince sinyal yok (recall <0.06) — kapanış kanıtı |
| ADS-B arşiv | ≤07-10 | İki koordinesiz ilk deneme | Reddedildi: %97.6 sentetik recall ama 25.54 doğal alarm/saat; sıfırdan başlandı |
| ADSB Adım 1-4 | 07-13 | 256M satır parse, sentetik corpus, manifest, truth-v2 | Kural skorlayıcı NN'leri geçti; proxy-tavan (~0.75) bulunup düzeltildi |
| ADSB Adım 5-7 | 07-13/14 | Tam-hacim doğal yük + S2 + freeze kapısı | **Adım-7 kapısı KALDI** (CUSUM h=1 doygun %99; NN'ler FLAGGED; hash kapısı fail-closed) |
| contextual v1 | 07-14 | Faz/kadans-koşullu LSTM forecaster | Genlik kapısı **GEÇTİ** (ρ=0.65) — ilk temiz NN |
| ADR-037..041 | 07-14 | Bütçe ön-kaydı, IF keşfi (elendi), kalibrasyon, doğal yük, ölçek 20×, doğu/kuzey CUSUM | Doğal yük kontrol altında; IF ρ=0.996 ile elendi |
| ADR-042 | 07-16 | Rehearsal (3. gün) + **truth-v2 GERÇEK RECALL** | Rehearsal kararlı; **recall: 4/5 profil <%6 — reddedildi**; CUSUM %49.7 istisna |

---

## 5. Skorlama yaklaşımları — ne, neden, ne oldu

1. **Reconstruction hatası (AE'ler):** "normali yeniden üret, üretemediğin anomalidir."
   *Neden başarısız:* kırpılmamış ölçekleme + birkaç aşırı-genlikli pencere → hata, öğrenmeden
   bağımsız olarak "girdi ne kadar büyük" ölçüsüne dejenere oldu (üç mimari birebir aynı
   kategori-sayıları üretti; eğitilmiş vs eğitilmemiş ρ=0.964).
2. **Forecast-residual (Chronos, contextual v1):** "geçmişten sonrakini tahmin et, sürpriz
   anomalidir." *Durum:* genlik artefaktına yapısal olarak daha dirençli çıktı (contextual v1
   ρ=0.65 ile PASS); ama tek başına operasyonel kapıyı açamadı.
3. **Kural-bazlı ceza (robust z + kırpılmış ceza toplamı):** `pen = min(max(0, z−3), 10)`.
   *Neden:* öğrenmesiz, denetlenebilir, genlik artefaktı imkânsız. *Sonuç:* NN'leri geçti ama
   doğal yükü operasyonel seviyeye inmedi.
4. **Z-score güven katmanı → konformal p-değeri:** İlk sürüm Gauss varsayımlıydı; residual'lar
   Gauss olmadığı için hiyerarşik konformal kalibrasyona (channel+phase+cadence → channel+phase
   → channel fallback) geçildi. Matematiksel olarak doğru çalıştı.
5. **Zamansal karar modları:** instant (tek satır p<α), persistence (gerçek-saniye pencere),
   accumulation (−log p birikimi, saniye-ağırlıklı — yüksek kadansın sırf çok örnekle hızlı
   alarm üretmesi engellendi), Page-CUSUM (ham z alanında sürekli birikim).
   *ADR-042 bulgusu:* konformal-p tabanlı modlar tek-uçuş penceresiyle sınırlı kaldığı için
   boğuldu; ham-alanda sürekli biriken CUSUM aynı koşulda 7× iyi genelledi — tasarım dersi.

---

## 6. Kök nedenler — neden başarılamadı (hepsi ölçülmüş, tahmin değil)

### 6.1 Etiket kalitesi: proxy etiketler sistematik olarak şişirdi, düzeltince gerçek düşük çıktı
- ALFA event-onset düzeltmesi: 0.594 → 0.194–0.224.
- RFLY interval-truth düzeltmesi: 0.749 → 0.526 (ve 5 çelişkili-etiketli uçuş bulundu).
- ADS-B proxy-etiket AUC tavanı ~0.75 → truth-v2 ile kaldırıldı.
- Çıkarım: bu alandaki yayınlarda görülen yüksek sayıların bir kısmı muhtemelen aynı proxy
  hatasını taşıyor; biz düzelttik ve gerçek sayıların operasyonel bara yetmediğini gördük.

### 6.2 Küçük ve heterojen etiketli örneklem — yapısal, parayla/kodla çözülmedi
- ALFA: rudder n=4, aileron+rudder n=1 (veri setinin kendisi bu kadar).
- SEAD: normal havuz uçuş sayısına göre çok az bağımsız oturuma dağılıyor (398-normal
  aşamasında 64 oturum); havuzlama (SEAD+RFLY) sonucu KÖTÜLEŞTİRDİ (0.149).
- "Normal sınıfı yapay homojenleştir" kısayolu bilinçli REDDEDİLDİ — unsupervised tespitin
  varlık nedenine aykırı; bu bir çözüm değil, sorunun halının altına süpürülmesi olurdu.

### 6.3 Genlik-baskınlığı artefaktı — derin modellerin "başarısı" büyük ölçüde sahteydi
- Teşhis: eğitilmiş model sıralaması vs eğitilmemiş-rastgele aynı mimari ρ=0.964; vs saf ‖x‖²
  ρ=0.965 (SEAD); ADS-B'de NN'ler 0.84–0.99 bandı; IF 0.996.
- Kapanış kanıtı (ADR-019): genlik bağımlılığı gerçekten kaldırılınca (relerr) recall <0.06 —
  altta öğrenilmiş sinyal yoktu.
- Tek istisna: contextual_physics_v1 (ρ=0.65) — tasarımı bu artefakta karşı kuruldu ve geçti.

### 6.4 Recall–yanlış alarm değiş-tokuşu hiçbir konfigürasyonda kapanmadı
- Her Gate C tablosunda aynı örüntü: recall hedefe yaklaşan her hücre FA bütçesini aşıyor,
  bütçe içindeki her hücre recall'da çok düşük. ML-13 bunu mimariyle, ML-14 veriyle, ML-15
  kalibrasyonla zorladı — üçü de aynı duvara çarptı.

### 6.5 ADS-B'ye özgü: etiketsizlik + bütçe-birimi uyuşmazlığı (ADR-042)
- ADS-B'de gerçek etiketli anomali YOK — değerlendirme zorunlu olarak sentetik enjeksiyonla.
- Alarm bütçesi "100 uçuş-saatinde N alarm" olarak donduruldu (7/24 izleme için doğru çerçeve);
  ama her sentetik olay TEK uçuşun ~0.5–1 saatlik penceresinde kanıtlanmak zorunda. Saatte
  0.0001–0.06'lık hedef oranı o pencereye çarpılınca beklenen alarm sayısı matematiksel olarak
  yüzde-birkaçın altında kalıyor. **Model kör değil** — eşik kasıtlı gevşetilince (α=0.5)
  recall %74–92'ye çıktı, ama doğal yük ~9 alarm/saat oldu (kullanılamaz). Yani mevcut bütçe
  tanımı altında "hem sessiz hem yakalar" nokta YOK.

### 6.6 Metodoloji faturası (dürüstlük)
- Sonuç görüldükten sonra hiçbir eşik/parametre değiştirilmedi; bu yüzden "kurtarılmış" tek bir
  sayı bile yok. Bunun bedeli raporlanabilir düşük sayılar; getirisi, her sayının savunulabilir
  olması. Kör holdout'lar (SEAD 200 uçuş; ADS-B 3-günlük havuz) HİÇ açılmadı — açılmış olsaydı
  bile mevcut adaylar kapıları geçmediği için anlamı olmazdı.

---

## 7. Neler çalıştı — dürüst pozitifler

1. **Kural-bazlı fizik skorlayıcısı 3 sinir ağını geçti** (0.600 vs 0.552–0.572) — "önce basit
   ve denetlenebilir yöntem" tezinin deneysel kanıtı.
2. **Genlik-baskınlığı teşhis protokolü** (eğitilmiş-vs-eğitilmemiş / vs-ham-genlik ρ testi) —
   yeniden kullanılabilir, başka projelerin sahte-başarısını da yakalar.
3. **Proxy-etiket düzeltme metodolojisi** (event-onset, interval-truth, truth-v2) — üç veri
   setinde aynı hata sınıfı bulunup düzeltildi.
4. **Taşınabilir eşik kalibrasyonu tezi** (ML-3): model başka platforma taşınır, eşik platforma
   kalibre edilir — 0.375→0.783 ile kanıtlandı.
5. **VectorPageCUSUM konum-sapması dedektörü**: en sıkı bütçede %7.1, gevşekte %49.7 recall,
   en sessiz doğal yük — projeden çıkan tek "işe yarayabilir dedektör" adayı.
6. **contextual_physics_v1**: genlik kapısını geçen tek öğrenilmiş model; 150× hızlandırılmış
   karar katmanı (bit-bit eşdeğerlik kanıtlı); hiyerarşik konformal kalibrasyon altyapısı.
7. **Denetlenebilirlik altyapısı**: immutable run-manifest/SHA-256 zinciri, ön-kayıt disiplini,
   42 ADR'lik karar günlüğü, deney kayıt paneli — her sayı yeniden türetilebilir.

---

## 8. Fizibilite hükmü ve gerekçesi

**Hüküm: Mevcut koşullarda (eldeki veri setleri, etiket kalitesi, operasyonel alarm bütçeleri)
hedeflenen operasyonel anomali-tespit ürünü ULAŞILABİLİR DEĞİLDİR.** Bu "denedik olmadı" değil;
şu üç bağımsız kanıt hattının kesişimidir:

1. **Yöntem uzayı dürüstçe tarandı:** istatistiksel, kural-bazlı, klasik ML, 4 DL mimarisi,
   foundation-model, füzyon/kanal mimarileri, kalibrasyon katmanları — 12+ aile, 17+ faz.
   Başarısızlık tek bir yöntemin değil, yöntem SINIFLARININ ortak sonucu.
2. **Başarısızlık nedenleri izole edildi ve veri-yapısal çıktı** (§6): küçük/heterojen etiketli
   örneklem, proxy-etiket şişmesi, genlik artefaktı (giderilince altında sinyal yok), bütçe-birim
   uyuşmazlığı. Bunların hiçbiri "daha çok epoch / daha iyi hiperparametre" ile çözülecek türden
   değil — ikisi bizzat veri setlerinin doğasında, biri değerlendirme çerçevesinin tanımında.
3. **İyileştirme kaldıraçlarının hepsi çekildi:** veri büyütme (ML-4 işe yaradı, ML-14 tersine
   tepti), feature mühendisliği (ML-9: +0.02), süpervizyon (ML-8A: kaybetti), foundation model
   (ML-10: kategori kazandı, füzyonda kayboldu), mimari (ML-13: FA şişti), kalibrasyon (ML-15:
   yetmedi), yeni domain + sıfırdan disiplinli hat (ADS-B: doğal yük kontrol altında ama gerçek
   recall <%6). Çekilecek bilinen kaldıraç kalmadı.

**Hangi koşullar hükmü değiştirirdi** (projenin devamı için gerekli ama bizim temin edemediğimiz
girdiler):
- **Gerçek, etiketli, yeterli hacimde anomali verisi** — sentetik enjeksiyon değil; olay başına
  onset/end damgalı, yüzlerce bağımsız olay. (RFLY buna en yakındı ve tek başına 0.526'ya
  ulaştı — hâlâ FA bütçesi dışında ama diğer her şeyden iyi.)
- **Operasyonel bütçenin baştan farklı tanımı** — "saat başına alarm" yerine "olay penceresi
  başına tespit olasılığı + günlük operatör yükü" gibi iki-eksenli bir sözleşme. ADR-042'nin
  gevşek-eşik kontrolü, modelin %74–92 yakalayabildiğini gösteriyor; sorun modelin gözü değil,
  bütçenin ağzı.
- **Homojen ya da oturum-zengin normal havuzu** — SEAD'in 64-oturunluk yapısı yerine yüzlerce
  bağımsız oturum.
- CUSUM hattı (tek istisna) ayrıca değerlendirilebilir: konum-sapması özelinde, gevşetilmiş
  bütçeyle sınırlı-kapsamlı bir dedektör olarak devam ettirilebilir — ama bu, orijinal
  "genel anomali tespit sistemi" hedefi değildir.

---

## 9. Beklenen sorulara hazır cevaplar

**S: Neden derin öğrenme işe yaramadı? Yanlış mı kullandınız?**
C: Üç mimariyi (LSTM-AE, Dense-AE, USAD) resmi hatta, 5 seed'le, holdout-izoleli eğittik; üçü de
aynı sayıları üretti. Bunun nedenini bulduk: skorları öğrenme değil, girdi genliği belirliyordu
(eğitilmemiş ağla ρ=0.964). Artefaktı skor katmanında giderince (relerr) altında sinyal kalmadı.
Yani "yanlış kullanım" değil — bu veri/ölçekleme rejiminde öğrenilecek ayrıştırıcı sinyalin
kendisi zayıf. Bunu düzeltmek için tasarlanan contextual forecaster genlik testini geçti ama
operasyonel kapıyı o da açamadı.

**S: Basit yöntemleri denediniz mi, yoksa hep karmaşık mı gittiniz?**
C: Tam tersi sırayla: en iyi ADS-B skorlayıcımız öğrenmesiz bir robust z-score kuralı ve NN'leri
yendi. Karmaşık yöntemler basitlerin açığını kapatmak için denendi, kapatamadı.

**S: Veri az diyorsunuz; daha çok veri toplasaydınız?**
C: Topladık. SEAD normal havuzunu 324→899'a çıkardık: FA düştü ama recall da düştü (0.21→0.126) —
literatürün öngördüğü heterojen-normal etkisi. ALFA'da ise büyütme gerçekten işe yaradı
(0.731→0.918) çünkü oradaki sorun gerçekten veri azlığıydı. Yani "daha çok veri" bazen çare,
bazen değil — ikisini de deneysel olarak gösterdik. Etiketli ANOMALİ verisi ise parayla/emekle
büyütülemedi: ALFA'nın rudder sınıfı 4 uçuş, çünkü gerçek uçakta arıza üretmek riskli/pahalı.

**S: %97.6 recall gördüm eski raporda — o ne oldu?**
C: O sayı arşivlenen ilk ADS-B denemesinin sentetik-kolay senaryolardaki sayısı ve yanında
saatte 25.54 doğal alarm vardı — kullanılamaz bir dedektör. Yüksek sentetik recall tek başına
başarı değildir; bu yüzden her sonucu doğal-yanlış-alarm ile ÇİFT olarak raporlama kuralı koyduk.

**S: Eşikleri gevşetseniz recall gelmiyor mu?**
C: Geliyor — %74–92'ye kadar ölçtük. Ama bedeli saatte ~9 alarm; 500 bin uçuş-saatlik gerçek
trafikte bu, operatör başına dakikalar içinde alarm yorgunluğu demek. "Hem sessiz hem yakalar"
nokta, bu veri ve bu bütçe tanımıyla mevcut değil — sorun bir ayar değil, iki hedefin bu
koşullarda kesişmemesi.

**S: Sonuçlara güvenebilir miyiz? Cherry-picking var mı?**
C: Tersine-cherry-picking var: üç ayrı veri setinde etiket hatası bulup sayılarımızı AŞAĞI
düzelttik (0.594→0.194; 0.749→0.526; AUC tavanı 0.75'in kaldırılması). Tüm eşikler sonuç
görülmeden donduruldu; kör holdout'lar hiç açılmadı; her koşunun SHA-256'lı manifest'i var;
42 ADR'lik karar günlüğü ve filtrelenebilir deney paneli (docs/experiment_dashboard.html)
her iddianın kaynağını gösteriyor.

**S: Bundan sonra ne yapılabilir?**
C: (1) Bu rapor ve panellerle sonucu belgeleyip kapatmak; (2) yalnız CUSUM konum-sapması hattını,
yeniden tanımlanmış bir bütçeyle, sınırlı-kapsam dedektör olarak yaşatmak; (3) gerçek etiketli
olay verisi temin edilirse contextual hattı yeni bir ön-kayıtla yeniden değerlendirmek.
Üçü de ayrı karar; hiçbiri mevcut hükümle çelişmiyor.

---

## 10. Kaynak dizini

- Karar günlüğü (42 ADR): `docs/decisions.md`
- Deney kayıt paneli (34 kayıt, filtrelenebilir): `docs/experiment_dashboard.html`
- ADS-B doğal yük + gerçek recall paneli: `docs/adsb_contextual_physics_v1_burden_dashboard.html`
- Teşhis dokümanı: `docs/DURUM_TESHIS_VE_YOL_HARITASI.md`
- Yetersizlikler kaydı (30+ madde): arşivde `archive/2026-07-10_legacy_non_adsb_ml/docs/ML_YETERSIZLIKLER_KAYDI.md`
- Arşivler: `archive/2026-07-10_legacy_non_adsb_ml/` (ML-0..16, RFLY), `archive/2026-07-10_rejected_adsb_attempts/`
- Son koşu artifact'ları: `artifacts/adsb/runs/20260715_contextual_physics_v1_truth_v2_eval_v1/`,
  `.../20260715_contextual_physics_v1_rehearsal_v1/`
- Ön-kayıt sözleşmeleri: `docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md`,
  `docs/adsb_contextual_physics_v1_alarm_budget_prereg_2026-07-14.md`
