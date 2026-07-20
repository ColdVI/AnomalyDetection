# RESIDUAL-V1 — Komut→Tepki Residual Tabanlı İHA Anomali Tespiti

> **Bu dosya, RESIDUAL-V1 çalışmasının tüm tasarım, uygulama ve sonuç
> belgelerini tek yerde toplar.** Daha önce `docs/` altında ayrı duran sekiz
> RESIDUAL_V1_* dosyası buraya, kronolojik ve mantıksal sırayla birleştirildi;
> içerik ve sayılar değiştirilmeden korundu. Orijinal ayrık dosyalar `arsiv`
> branch'inde durmaya devam ediyor.

## Yönetici özeti

RESIDUAL-V1, anomaliyi ham telemetride değil, öğrenilmiş bir uçuş-dinamiği
modelinin **innovation'ında** (komut verildi → beklenen tepki − ölçülen tepki)
arayan model-tabanlı bir arıza tespit (FDI) yaklaşımıdır. Kapsam: ALFA
(sabit-kanat) ve RflyMAD-Real (çok-rotor) veri setleri.

**Nihai karar: NO-GO** — *mevcut development-normal maruziyetiyle elde edilemez.*
Bu, "dedektör çalışmadı" sonucu **değildir**. Eşikten önceki tüm yöntem zinciri
çalışmıştır:

- **K5** (waypoint V-dönüşü maskesi), **S-4** (komut ablasyonu: Q1/Q2 PASS,
  Q3 FLAGGED), **S-1** (büyüklük korelasyonu: R6, Q1, Q2 PASS) uygulandı.
- **S-3** (threshold-bağımsız ayrışma testi) ALFA/engine (R6), RFLY/motor ve
  RFLY/sensor sınıflarının **üçünde de PASS** verdi — yani sinyal kanıtlandı.
- Tıkanma noktası: dondurulmuş yanlış-alarm bütçesini güvenilir çözecek
  **yeterli bağımsız normal uçuş-saati yok.** Kalibrasyon maruziyet açığı
  ALFA/R6 için 11.845×, RFLY/Q1-Q2 için 5.088×. Bu açık ne yeniden dağıtımla
  ne de mevcut resmî kaynaktan ek ingest'le kapanıyor (her ikisi de matematiksel
  olarak elendi).

Sonuç: `thresholds_frozen.json` üretilmedi; test/holdout değerlendirmesine
geçilmedi. Kanıtlanan şey threshold-bağımsız dağılım ayrışmasıdır; operasyonel
sınır kalibre edilmemiştir.

## İçindekiler

1. Deney ve Model Tasarımı
2. İmplementasyon Planı ve Kabul Kriterleri
3. Görev 4.1 — Şartlı GO Kaydı
4. Görev 4.1 — Sonuçlar
5. Faz E — Uygulama Planı
6. Faz E — Sonuçlar
7. Kalibrasyon NO-GO Raporu
8. Ek A — Kalibrasyon Maruziyet Analizi


---

## RESIDUAL-V1 — Komut→Tepki Residual Tabanlı İHA Anomali Tespiti: Deney ve Model Tasarımı

Tarih: 2026-07-16 · Statü: Tasarım (implementasyon öncesi) · Kapsam: ALFA (sabit-kanat) + RflyMAD-Real (çok-rotor)

---

### 0. Tek cümlelik tez

Anomaliyi ham telemetride değil, **öğrenilmiş bir uçuş-dinamiği modelinin innovation'ında** (komut verildi → beklenen tepki − ölçülen tepki) ararsak, manevra genliği skordan yapısal olarak düşer, genlik-baskınlığı artefaktı doğamaz ve karar katmanındaki CUSUM tam da tasarlandığı problemi (residual ortalamasında kalıcı kayma) çözer. Bu, repodaki 17 fazın hiçbirinde denenmemiş ana yoldur ve aktüatör-arıza literatürünün standart yaklaşımıdır (model-based FDI: fault detection & isolation).

---

### 1. Başarı sözleşmesi — sonuç görülmeden, ama bu kez doğru birimde

Repodaki Gate C'nin ölümcül hatası birimdi: saat-bazlı yanlış alarm bütçesi × tek-uçuş olay penceresi matematiksel olarak kesişmiyordu (ADR-042). Yeni sözleşme **olay-bazlı** ve baştan şöyle donduruluyor:

**Birincil hedef (headline):** Etiketli arıza olayı başına, onset'ten itibaren **ortanca tespit gecikmesi ≤ 5 s** ve **p90 gecikme ≤ 15 s**; aynı eşik konfigürasyonunda normal uçuşlarda **yanlış alarm ≤ 0.5 alarm / uçuş-saati** (uçuş başına değil uçuş-saati başına, çünkü ALFA uçuşları kısa ve eşit değil). Bu iki sayı **aynı frozen eşikle, aynı anda** raporlanır — asla ayrı ayrı değil (repodaki %97.6-recall/25-FA fiyaskosunun dersi).

**Sınıf-bazlı raporlama kuralı:** n ≥ 8 olay içeren sınıflar (ALFA engine; RflyMAD motor, sensor) headline sayı alır ve bootstrap %95 güven aralığıyla verilir. n < 8 sınıflar (ALFA rudder n=4, aileron+rudder n=1) **headline'a girmez**; her biri tek tek vaka analizi olarak raporlanır ("bu uçuşta residual şöyle davrandı") ve kapsam beyanına "öğrenilmiş tespit iddiası yok, kural-bazlı kapsama var" yazılır. Repoda n=4'lük sınıfa yüzde vermek istatistiksel tiyatroydu; tekrar etmiyoruz.

**Go/no-go tanımı:** RESIDUAL-V1 başarılı sayılır ⇔ ALFA-engine ve RflyMAD-motor sınıflarının EN AZ BİRİNDE hedef gecikme+FA çifti tutturulur VE genlik-sanity kapıları (§6) geçilir. İkisi de tutmazsa hata analizi raporuyla dur — yeni veri setine ATLAMA (anti-pattern #1, §10).

---

### 2. Zihinsel simülasyon — modeli çalıştırmadan önce kafada koşmak

Bu bölüm tasarımın kalbi. Üç senaryoyu satır satır simüle ediyorum; her biri bir tasarım kararını zorluyor.

#### 2.1 Senaryo A: ALFA engine-failure uçuşu — tespit nasıl gerçekleşir

Sabit kanat, cruise, ~18 m/s hava hızı. t=0'da motor gücü kesiliyor (failure_status ilk emisyonu). Fizik zinciri şöyle akar:

t=0–1 s: Throttle komutu hâlâ yüksek (otopilot henüz fark etmedi). Hava hızı sürtünmeden ötürü saniyede ~0.3–0.5 m/s azalmaya başlar. **Ham telemetride neredeyse hiçbir şey görünmez** — pitch, roll, irtifa normal bandında. Repodaki satır-bazlı skorlayıcıların onset'te kör olmasının nedeni bu: onset anında sinyal genliği sıfıra yakın.

Bizim modelde ne olur: `thrust→airspeed` residual kanalı, "throttle %70 + bu hava hızı + bu pitch → hava hızı türevi ≈ +0.1 m/s²" tahmin eder. Ölçülen türev −0.4 m/s². Residual r_t ≈ −0.5 m/s², kanalın train-normal MAD'ı ~0.08 ise z ≈ −6. Tek satırda alarm için yetersiz ve yetmemeli (türbülans da anlık −4z üretebilir) — ama **CUSUM t=0'dan itibaren her 20 ms'de ~|z|−k birikiyor**. k=1.5, h=25 kalibrasyonuyla birikim ~4.5/z-birim/örnek × 50 Hz → eşik ~2–4 saniyede aşılır.

t=2–5 s: Otopilot hız kaybını görüp pitch-down komutu verir; sink rate artar. İkinci bağımsız kanal (`pitch_cmd→climb_rate` tutarlılığı) da kaymaya başlar. İki kanalın CUSUM'u birlikte yükselir — tek kanallı yanlış-pozitiften ayıran imza budur.

**Beklenen sonuç:** ortanca gecikme 2–5 s bandı. Bu, hedefi (≤5 s) tam sınırdan test eden gerçekçi bir senaryo; hedef keyfi değil, bu simülasyondan türedi.

#### 2.2 Senaryo B: agresif manevra, arıza yok — alarm NEDEN çalmaz

Aynı uçak, alçak irtifada 45° yatışlı keskin dönüş. Ham telemetri açısından bu uçuşun "en anormal" 10 saniyesi: roll ±45°, yaw rate 15°/s, yük faktörü 1.4 g. Repodaki AE'ler tam burada alarm üretirdi — çünkü skorları fiilen ‖x‖² idi (ρ=0.965 ölçümü).

Bizim modelde: aileron komutu −%60 verildi, model "bu komut + bu hava hızında roll rate ≈ −38°/s" der (kontrol etkinliği dinamik basınçla ölçeklenir; girdide V² etkileşimi var, §5.2). Ölçülen −36°/s. Residual ≈ 2°/s, z ≈ 0.8. **Manevra ne kadar agresif olursa olsun, komut-tepki tutarlıysa residual küçük kalır.** CUSUM birikmez. Alarm yok.

Kritik tasarım sonucu: bu ancak model **hava hızı ve uçuş fazına koşulluysa** çalışır. Koşulsuz bir model iniş yaklaşmasında (düşük V → düşük kontrol etkinliği) sistematik residual üretir ve her inişte alarm çalar. Bu yüzden faz segmentasyonu (§4.4) ve V²-etkileşimi opsiyonel değil, çekirdek gereksinim.

#### 2.3 Senaryo C: modelin kendini kandırması — autoregressive sızıntı tuzağı

En tehlikeli hata modu, ve simüle etmeden görülmez: modele girdi olarak y'nin kendi geçmişini (y_{t−1}, y_{t−2}...) verirsek, model komutu öğrenmek yerine "y_t ≈ y_{t−1}" kopyacılığını öğrenir (bir adım ilerisi için bu neredeyse her zaman en düşük MSE'dir). Arıza başladığında ne olur? **Model arızalı sinyali de bir adım geriden kusursuz takip eder** — residual hiç büyümez, arıza "normal" görünür. Kağıt üstünde düşük validation loss, sahada kör dedektör. Bu, repodaki genlik artefaktının ayna görüntüsü: orada model hiçbir şey öğrenmemişti, burada yanlış şeyi öğrenir.

Tasarım önlemleri (üçü birden zorunlu):
(a) Tepki değişkeninin kendi kısa geçmişi girdiye **girmez**; girdi = komut geçmişi + yavaş bağlam durumları (V, irtifa, faz) yalnız.
(b) Tahmin ufku tek örnek değil, **0.5 s'lik pencere ortalaması** (50 Hz'de 25 örnek) — kopyacılığın işe yaramayacağı kadar uzak.
(c) **"Arızayı-görüyor-mu" ön testi** (§6, S-3): herhangi bir eşik/kalibrasyon işine girmeden ÖNCE, development'taki etiketli olaylarda residual |z|'nin onset sonrası dağılımı onset öncesine göre KS testiyle ayrışmak zorunda. Ayrışmıyorsa mimariye dön; kalibrasyonla kurtarmaya ÇALIŞMA (repodaki ML-15 dersi: kalibrasyon sinyal üretmez, olan sinyali şekillendirir).

#### 2.4 Senaryo D (çok-rotor, RflyMAD motor arızası) — domain farkının simülasyonu

Quadrotor'da tek motor %30 etkinlik kaybederse karışım (mixer) matrisi bozulur: sabit hover için o motorun PWM komutu yükselir, çapraz motorunki düşer, gövdede küçük ama kalıcı bir roll/pitch bias'ı ve yaw drift oluşur. Buradaki en ayrıştırıcı residual sabit-kanattakinden farklıdır: `motor_pwm_asimetrisi | (thrust_toplam, attitude_cmd)` — yani "bu toplam itki ve bu duruş komutu için motorlar arası PWM dağılımı ne olmalı". Bu, aynı metodolojinin (komut→tepki residual) platforma özgü kanallarla ayrı örneklenmesi demek — domain separation ilkesi korunuyor, model ağırlığı taşınmıyor, yalnız metodoloji ve karar katmanı ortak.

---

### 3. Veri ve telemetri hijyeni — kanal kanal hâkimiyet

Modele ne beslediğini bilmeden model tasarlamak, repodaki `velocity_mps %100 null` ve "hayalet imputation" vakalarını üretti. Her kanal için üç soru: fiziksel anlamı ne, hangi hızda ve hangi saatle geliyor, hangi kirlilik modları var. Aşağıdaki envanter implementasyondan önce uçuş başına otomatik profil raporuyla (null haritası, dt histogramı, aralık ihlalleri) doğrulanacak — göz kararı değil, kabul testiyle.

#### 3.1 ALFA kanal envanteri (kullanılacak çekirdek)

| Kanal (topic) | Fiziksel anlam | ~Hz | Bilinen kirlilik ve önlem |
|---|---|---|---|
| `mavros/nav_info/*` des_/meas_ (roll, pitch, yaw, airspeed, velocity) | Otopilot komutu vs ölçüm — residual hattının ana hammaddesi | 20–25 | Kolon adlandırma tuzağı (repoda bulunan meas_x/des_x vakası); yaw'da açı sarımı → tüm açı farkları wrap-aware (atan2) hesaplanır |
| `mavros/imu/data` (gyro, accel, quaternion) | En yüksek hızlı gerçek dinamik; referans saat | 45–50 | Quaternion işaret atlaması (q ↔ −q aynı duruş) → devamlılık düzeltmesi; accel'de bias, türev alınmaz, olduğu gibi bağlam |
| `mavros/rc/out` (servo/throttle PWM) | Fiili aktüatör komutu — nav_info'dan daha ham ve daha dürüst | 20 | PWM kalibrasyonu uçaklar arası kayar → uçuş-içi normalize (trim medyanına göre delta) |
| `mavros/global_position` + `local_position` | Konum/irtifa/yer hızı | 5–10 | GPS irtifa sıçramaları; yer hızı ≠ hava hızı (rüzgâr) — ikisi ayrı kanal, asla birbirinin yerine kullanılmaz |
| `mavctrl/path_dev`, `xtrack_error` | Yörünge sapması — reponun ölçtüğü en güçlü causal kanal (0.751) | 10 | Waypoint geçişinde yapısal sıçrama → waypoint-değişim maskesi (±2 s) |
| `failure_status/*` | Etiket: yalnız arıza AKTİFKEN emisyon | olay | Onset = ilk emisyon; öncesinde 10 s guard band train'den dışlanır (0–%50 overlap dışlama kuralı korunuyor) |

#### 3.2 RflyMAD-Real kanal envanteri

ULog kaynaklı: `actuator_outputs` (motor PWM ×4), `vehicle_attitude` + `vehicle_attitude_setpoint`, `vehicle_local_position`, `battery_status` (voltaj çökmesi motor-arıza taklidi yapabilir → bağlam değişkeni olarak girer, dedektör kanalı olarak DEĞİL), `rfly_ctrl_lxl` (interval truth — repo bunu zaten çıkarmış, 5 çelişkili uçuşun dışlanması aynen devralınır).

#### 3.3 Hijyen kuralları (her iki set, uçuş başına otomatik kontrol)

Zaman damgası tekdüzeliği: dt ≤ 0 satır → at ve say; dt > 5×medyan → boşluk olarak işaretle (interpolasyon YOK, §4.1). Birim tutarlılığı: her kanala fiziksel aralık kapısı (|roll| ≤ 180°, airspeed 0–60 m/s, ...); ihlal eden uçuş karantinaya, sessiz kırpma yasak. Donmuş sensör: 2 s boyunca değişmeyen yüksek-hızlı kanal → `stale` bayrağı; bu bir VERİ KALİTESİ sinyalidir ve anomali skoruna doğrudan girmez (repodaki S2 ayrımı doğruydu, korunuyor) ama residual hesabında o pencere maskelenir — donmuş girdiyle residual hesaplamak sahte alarm üretir.

---

### 4. Silver/feature tasarımı — repodan farkı: interpolasyon yok

#### 4.1 Neden interpolasyon yok

Repo Silver'da her şeyi 50 Hz'e lineer interpole etti. Zihinsel simülasyonu: 5 Hz'lik GPS kanalını 50 Hz'e lineer interpole edersen, ardışık 10 örnek mükemmel bir doğru üzerinde yatar. Residual modeli bu yapay pürüzsüzlüğü öğrenir; gerçek örnek geldiği anda küçük bir kırılma olur ve model her 200 ms'de bir minik "sahte sürpriz" residual'ı üretir — CUSUM'a sistematik gürültü. Daha kötüsü: dropout sırasında interpolant iki uzak nokta arasında köprü kurar ve tam anomali anını pürüzsüzleştirir.

Yerine: her kanal **kendi doğal hızında** Silver'a yazılır. Hizalama feature-hesap anında `merge_asof(direction='backward', tolerance=kanal_bazlı)` ile yapılır ve her düşük-hızlı kanala bir `staleness_ms` kolonu eşlik eder. Model 20 Hz'lik nav_info saatinde koşar (residual hattının doğal hızı); IMU o saate en-yakın-geçmiş değerle bağlanır.

#### 4.2 Residual kanalları (ALFA, v1'de 6 kanal — az ve derin)

R1 `aileron_cmd → roll_rate`, R2 `elevator_cmd → pitch_rate`, R3 `rudder_cmd + roll → yaw_rate` (koordineli dönüş fiziği: beklenen yaw_rate ≈ g·tan(roll)/V terimi girdide), R4 `throttle → airspeed_türevi(0.5 s)`, R5 `pitch + throttle → climb_rate`, R6 `xtrack_error` (öğrenmesiz, doğrudan; repodaki en iyi causal kanal, aynen alınır).

Bilinçli olarak 73–85 feature'lık repodan geriye gidiyoruz: **6 fiziksel-anlamlı kanal, her biri hata analizinde tek tek açılabilir.** Genişlik değil derinlik (anti-pattern #2).

#### 4.3 RflyMAD residual kanalları (v1'de 4)

Q1 `attitude_setpoint → attitude_rate` (eksen başına), Q2 `motor_pwm_dağılımı | (toplam_itki, attitude_cmd)` — motor asimetri residual'ı, Q3 `toplam_pwm → dikey ivme`, Q4 `pozisyon setpoint → hız tepkisi`.

#### 4.4 Faz segmentasyonu

Kural-bazlı, öğrenmesiz: ground/taxi (yer hızı < 3 m/s VE irtifa değişimi ~0) → tamamen dışarı; takeoff/landing (climb_rate ve irtifa eşikleri); cruise; maneuver (|roll| > 25° veya |roll_rate| > 15°/s). Model girdisine faz one-hot + V ve V² sürekli değişken olarak girer. Faz sınırlarındaki ±1 s tampon her iki fazdan da sayılmaz (sınır belirsizliği CUSUM'u kirletmesin).

---

### 5. Model kademesi — basitten karmaşığa, her adımda çıkış kapısı

#### 5.1 G0 — fizik kuralları (öğrenmesiz taban + küçük-n sınıfların kapsayıcısı)

Üç kural: (i) komut-verildi-tepki-yok: |cmd delta| > eşik iken 1 s içinde tepki değişkeni MAD-bandında hareketsiz → ceza; (ii) koordineli dönüş residual'ı `yaw_rate − g·tan(roll)/V` (repodan devralınır); (iii) itki-hız tutarlılığı kaba bandı. G0 iki iş görür: öğrenilmiş modellerin geçmek ZORUNDA olduğu taban çizgisi (repoda kural-bazlının 3 NN'i yenmesi dersi — bu kez baştan bar olarak konuyor) ve rudder n=4 gibi öğrenme-imkânsız sınıfların tek meşru kapsayıcısı.

#### 5.2 G1 — kanal başına ridge regresyon (ana aday)

Her residual kanalı için: girdi = komutun son 1 s'lik geçmişi (20 Hz'de 20 lag, ama 5'e indirgenmiş üçgen-ağırlıklı özetle: son değer, 0–0.25 s ort., 0.25–0.5, 0.5–1.0 + delta), bağlam = [V, V², faz one-hot, V×cmd etkileşimi]; hedef = tepkinin gelecek 0.5 s ortalaması. Tepkinin kendi geçmişi girdide YOK (§2.3). Ridge seçiminin nedeni performans değil teşhis edilebilirlik: katsayılar fiziksel yorum taşır ("aileron→roll_rate kazancı 0.63°/s per %, V² ile ölçekleniyor — makul") ve hata analizi katsayı düzeyinde yapılabilir. Eğitim yalnız normal uçuşlar + arızalı uçuşların guard-band'li onset-öncesi kısmı; uçuş-bazlı değil **oturum-bazlı** split (repodan devralınan doğru ders), 5 seed.

#### 5.3 G2 — LightGBM aynı girdi/hedefle (doğrusal-olmama kontrolü)

Aynı girdiler, aynı hedef, aynı split. Amaç yeni model değil, soru cevaplamak: G2 test-residual varyansını G1'e göre >%20 düşürüyorsa dinamik anlamlı ölçüde doğrusal-dışıdır ve G2 aday olur; düşürmüyorsa G1 kalır ve derin modele hiç gidilmez. Repoda bu kıyas hiç bu netlikte kurulmadı — mimariler farklı feature setleri, farklı skorlarla yarıştı ve kıyas anlamını yitirdi.

#### 5.4 G3 — küçük GRU forecaster (yalnız kanıtla açılan kapı)

AÇILMA KOŞULU: G1/G2 sonrası test residual'larında yapısal otokorelasyon kalması (Ljung-Box p<0.01, lag ≤ 1 s) VE bu otokorelasyonun hata analizinde belirli bir dinamikten (ör. fugoid salınımı) kaynaklandığının gösterilmesi. Koşul sağlanmadan G3'e gitmek yasak — repodaki "4 DL mimarisi denendi" genişliğinin panzehiri, DL'i ancak ne için gerektiğini söyleyebildiğinde kullanmak.

---

### 6. Sanity kapıları — repodan devralınan ve yenilenen

Her aday skor, kalibrasyondan ÖNCE dört testten geçer; geçemeyen FLAGGED olur ve kalibrasyona giremez:

S-1 Genlik testi (repodan): skor vs ‖x‖ (ham girdi normu) Spearman ρ < 0.5 zorunlu. Residual tasarımı bunu yapısal olarak sağlamalı; sağlamıyorsa temsilde sızıntı var demektir.
S-2 Eğitilmemiş-eş testi (repodan): eğitilmiş skor sıralaması vs aynı mimari rastgele-ağırlık ρ < 0.7.
S-3 Arızayı-görüyor-mu testi (YENİ, §2.3'ten): development etiketli olaylarında onset-sonrası |z| dağılımı, onset-öncesine göre KS istatistiğiyle ayrışmalı (her headline sınıfta p < 0.01). Bu test EŞİKTEN BAĞIMSIZ — sinyalin varlığını ölçer. Repo bunu hiç ayrı test etmedi; sinyal-yokluğu ile eşik-yanlışlığı hep iç içe kaldı.
S-4 AR-sızıntı testi (YENİ): girdiden komut kanalları çıkarılıp yalnız bağlamla eğitilen "sakat" model, tam modele yakın performans veriyorsa model komutu değil başka bir sızıntıyı öğreniyordur → temsil incelemesi.

---

### 7. Karar katmanı — CUSUM, ama bu kez doğru kalibrasyonla

Kanal başına: r_t → robust-z (train-normal median/MAD; MAD=0 kanal dışlanır ve loglanır — repodan). İki yönlü Page-CUSUM: S⁺=max(0, S⁺+z−k), S⁻=max(0, S⁻−z−k); k=1.0 (hedef kayma ~2σ varsayımı, yarısı). Alarm: herhangi bir kanal S > h_kanal.

Kalibrasyon: her kanalın h'si, development-normal uçuşlarında **blok-bootstrap** (60 s bloklar, uçuş içi) ile "kanal başına FA katkısı = toplam bütçe / kanal sayısı" olacak şekilde çözülür. Toplam bütçe = 0.5 alarm/uçuş-saati (§1). Eşikler burada donar; test ve holdout'ta değişmez. Repodaki K-of-N grid-search yerine bu — grid search sonuca bakarak seçer, ARL-kalibrasyonu bakmadan.

Alarm sonrası: refractory 60 s (aynı olayın çoklu sayımı önlenir) + alarm anında kanal-katkı vektörü loglanır (hangi residual tetikledi — isolation/teşhis ücretsiz geliyor, sunumda güçlü demo).

---

### 8. Zayıf süpervizyon kolu (yalnız RflyMAD, yalnız residual uzayında)

RflyMAD 439 etiketli gerçek arıza uçuşuyla projedeki en büyük kullanılmayan kaynak. ML-8A'nın hatası süpervizyonu ham-feature uzayında ve tek konfigürasyonla denemekti. Burada: residual-z pencereleri (10 s, 1 s stride) üzerinde Deep SAD tarzı hedef — normal pencereler merkeze çekilir, etiketli arıza pencereleri itilir; çıktı skoru yine CUSUM'a girer (karar katmanı değişmez, yalnız z kaynağı değişir). Karşılaştırma önceden kayıtlı: unsupervised CUSUM vs weakly-supervised CUSUM, aynı split, aynı bütçe, fark ≥ +0.05 medyan-gecikme-eşdeğeri değilse basit olan kazanır. Bu kol ALFA'ya taşınmaz (etiket hacmi yetersiz) — kapsamı baştan dar tutmak, "her yerde her şey" savrulmasının önlemi.

---

### 9. Değerlendirme protokolü

Split: oturum-bazlı, 5 seed (repodan devralınır; mevcut manifest'ler uyarlanarak yeniden kullanılır). Development seti **bakmak için vardır**: her deney turunda zorunlu hata analizi — kaçırılan HER olay için komut/tepki/residual/CUSUM dörtlü grafiği çizilir ve kaçırma nedeni dört sınıftan birine atanır (sinyal-yok / sinyal-var-birikim-yavaş / maskeleme / etiket-şüpheli). Bir sonraki iterasyon bu dağılıma göre seçilir. Repodaki "ön-kayıt her şeyi kilitler" aşırılığının düzeltmesi: kilitli olan test/holdout ve eşiklerdir; development'ta bakmak, kurcalamak, anlamak serbest ve zorunludur.

Metrikler: sınıf başına olay-recall @ frozen eşik, tespit gecikmesi dağılımı (medyan, p90), normal uçuşlarda FA/uçuş-saati, hepsi bootstrap %95 CI ile. Point-adjust YOK (repo ilkesi korunur). Kör holdout: her setten ~%15 oturum, bir kez, en sonda, yalnız go kararı sonrası açılır.

---

### 10. Anti-pattern sözleşmesi — repoda düşülen hatalar ve buradaki kilit

AP-1 Veri-seti hoplama: RESIDUAL-V1 boyunca yalnız ALFA + RflyMAD-Real. ADS-B, SEAD, sentetik enjeksiyon bu deneyin kapsamı DIŞINDA. Başarısızlık halinde çıktı "yeni veri seti" değil "hata analizi raporu"dur.
AP-2 Genişlik > derinlik: aynı anda en fazla 2 model ailesi (G0+G1, sonra gerekirse G2); her aile arasında zorunlu hata-analizi turu.
AP-3 Bütçe-birim uyuşmazlığı: tüm hedefler olay/uçuş-saati biriminde ve §1'de donduruldu; hiçbir metrik farklı birimde raporlanamaz.
AP-4 Sinyal-yokluğunu kalibrasyonla örtme: S-3 kapısı geçilmeden eşik/kalibrasyon işine girilmez.
AP-5 Girdi temsili sızıntısı: tepki geçmişi girdiye giremez (S-4 ile denetlenir); interpolasyon Silver'a giremez.
AP-6 İstatistiksel tiyatro: n<8 sınıfa yüzde yok, vaka analizi var.
AP-7 Dokümantasyon/deney oranı: faz başına 1 tasarım + 1 sonuç dokümanı; ADR yalnız geri-döndürülemez kararlara.

---

### 11. Takvim (10 iş günü) ve kontrol noktaları

G1–2: Silver-v2 (doğal hız, staleness, hijyen kabul testleri) + uçuş profil raporları. Çıkış: her kanal için dt/null/aralık raporu temiz.
G3: Faz segmentasyonu + R1–R6/Q1–Q4 residual hesapları + G0 kuralları. Çıkış: Senaryo A ve B'nin gerçek uçuşlarda görsel doğrulaması (engine-fault uçuşunda R4 kayıyor mu, agresif dönüşte R1 sakin mi — zihinsel simülasyonun ampirik testi; TUTMUYORSA dur ve anla).
G4–5: G1 ridge eğitimi + S-1..S-4 kapıları + development hata analizi turu 1.
G6: CUSUM blok-bootstrap kalibrasyonu, eşik dondurma.
G7: Test değerlendirmesi (ALFA-engine, RflyMAD-motor/sensor) + hata analizi turu 2.
G8: Gerekliyse G2; RflyMAD zayıf-süpervizyon kolu.
G9: Kör holdout (yalnız kapılar geçildiyse) + rapor.
G10: Tampon + sunum materyali (alarm-anı kanal-katkı demoları).

Kontrol noktası kuralı: G3 sonundaki görsel doğrulama başarısızsa G4'e geçilmez — bu, projenin en ucuz "erken yanlışlama" fırsatı ve repodaki "8 günde 17 faz" hızının panzehiri.

---

### 12. Bu tasarımın yanlışlanabilir iddiaları (önceden yazılı)

İ-1: R4 (thrust→airspeed) residual'ı ALFA engine olaylarında onset+3 s içinde |z|>4'e ulaşır (S-3'ün somutlaşmışı).
İ-2: R1 residual'ı manevra genliğiyle korelasyonsuzdur (ρ<0.2) — genlik artefaktının yapısal ölümünün kanıtı.
İ-3: Q2 (motor asimetrisi) RflyMAD motor sınıfında tek başına, tüm-kanal füzyonunun gecikmesinin 1.5 katı içinde kalır (asimetri imzasının baskınlığı iddiası).
İ-4: G2, G1'e karşı residual varyansında <%20 iyileşme verir (dinamiğin bu rejimde yeterince doğrusal olduğu iddiası).

Bunlardan ikisi bile yanlışlanırsa tasarım revize edilir — ama bu kez neyin neden yanlış çıktığını söyleyebilecek kadar az ve keskin iddiayla yürüyoruz.


---

## RESIDUAL-V1 İmplementasyon Talimatı (Kodlama Ajanı İçin)

Referans tasarım: `RESIDUAL_V1_DENEY_TASARIMI.md` (bu talimatla birlikte repoya `docs/` altına konacak).
Hedef repo: `github.com/ColdVI/AnomalyDetection` (main).
Rol: Sen bir implementasyon ajanısın. Bu belgedeki görevleri SIRAYLA yaparsın. Her görevin kabul testi vardır; test geçmeden sonraki göreve geçemezsin. Tasarım kararlarını değiştiremezsin — belirsizlik görürsen `docs/residual_v1_questions.md` dosyasına soru yazıp o görevi atlayıp atlayamayacağını kontrol edersin (bağımlılık yoksa devam, varsa dur).

---

### 0. Genel kurallar (her görevde geçerli)

1. Yeni kod TEK pakette yaşar: repo kökünde `residual_v1/`. Mevcut `adsb/`, `src/`, `anomaly_core/` paketlerine dokunulmaz — İSTİSNA: `anomaly_core/sequential.py` içindeki `MultiChannelPageCUSUM` yeniden KULLANILIR (import edilir, kopyalanmaz).
2. Arşivdeki (`archive/2026-07-10_legacy_non_adsb_ml/`) kod READ-ONLY referanstır. ALFA/RflyMAD parser mantığı oradan OKUNARAK `residual_v1/ingest/` altına temiz yeniden yazılır; arşivden import YAPILMAZ.
3. Her çalıştırılabilir script `scripts/residual_v1_*.py` adlandırmasıyla; her config `configs/residual_v1_*.json`; her test `tests/test_residual_v1_*.py`. Mevcut pytest düzenine uyulur.
4. Tüm rastgelelik `seed` parametresiyle; varsayılan seed listesi `[11, 23, 37, 41, 53]`.
5. Çıktılar `artifacts/residual_v1/runs/<YYYYMMDD_gorevadi>/` altına; her run klasöründe `manifest.json` (git SHA, config hash SHA-256, girdi dosya hash'leri, seed). Var olan run klasörünün üstüne yazma girişimi → hata (fail-if-exists).
6. MLflow: experiment adı `residual_v1`; her koşuda config, manifest, metrikler ve grafikler artifact olarak loglanır.
7. YASAKLAR (tasarım AP-1..AP-7'nin kod karşılığı):
   - Tepki değişkeninin kendi geçmişini (lag'li y) feature olarak eklemek — feature şemasında `target_history` diye bir alan AÇILMAYACAK.
   - Silver katmanında herhangi bir interpolasyon/resample (`.interpolate`, `.resample().mean()` vb.) — lint kuralı Görev 1.4'te.
   - ADS-B, UAV-SEAD, sentetik enjeksiyon kodu çağırmak.
   - Test/holdout rolündeki uçuşları eşik seçimi, model seçimi veya hata analizinde kullanmak.
   - `point_adjust` benzeri değerlendirme gevşetmeleri.
8. Durdurma kuralları (STOP): Görev 3.5 görsel doğrulama ve Görev 5.4 S-3 kapısı "insan onayı" bekler — bu noktalarda özet rapor üretip DURURSUN, devam komutu gelmeden sonraki faza geçmezsin.

---

### 1. Faz A — Veri katmanı (Silver-v2)  [tasarım §3–4]

#### Görev 1.1 — Paket iskeleti
`residual_v1/{__init__.py, ingest/, features/, models/, decision/, eval/, viz/}` oluştur. `residual_v1/schema.py` içinde:
```python
@dataclass(frozen=True)
class ChannelSpec:
    name: str            # ör. "imu_gyro_x"
    topic: str           # kaynak topic/dosya
    unit: str            # "rad_s", "m_s", "pwm", ...
    valid_min: float
    valid_max: float
    nominal_hz: float
    is_angle: bool = False   # wrap-aware fark için
```
ALFA ve RflyMAD kanal envanterleri (tasarım §3.1–3.2 tabloları) `residual_v1/ingest/alfa_channels.py` ve `rfly_channels.py` içinde `CHANNELS: tuple[ChannelSpec, ...]` sabitleri olarak kodlanır.
Kabul: `pytest tests/test_residual_v1_schema.py` — ChannelSpec doğrulamaları (min<max, hz>0), envanterlerde isim tekilliği.

#### Görev 1.2 — ALFA ingest (doğal hızda, topic başına parquet)
`residual_v1/ingest/alfa.py`: processed-CSV kökünden uçuş başına, topic başına parquet yazar: `artifacts/residual_v1/silver/alfa/<flight_id>/<topic>.parquet`. Kurallar: timestamp'ler float saniyeye normalize (uçuş t0'ına göre); dt≤0 satırlar atılır ve sayısı `ingest_report.json`'a yazılır; İNTERPOLASYON YOK; açı kanalları radyana çevrilir ve (−π, π] aralığına sarılır; quaternion işaret devamlılığı düzeltilir (ardışık dot<0 → −q). `failure_status` topic'inden `events.json` üretilir: `{fault_class, onset_s, end_s}` (onset = ilk emisyon).
Kabul: `tests/test_residual_v1_alfa_ingest.py` — sentetik mini-CSV fixture ile: dt filtresi, wrap, quaternion düzeltmesi, onset çıkarımı birim test edilir.

#### Görev 1.3 — RflyMAD-Real ingest
`residual_v1/ingest/rfly.py`: ULog → topic parquet (pyulog). `rfly_ctrl_lxl`'den interval truth; arşiv raporundaki 5 çelişkili uçuş ID'si `configs/residual_v1_rfly_exclusions.json`'a kodlanır ve dışlanır. `battery_status.voltage_v` bağlam kanalı olarak alınır (dedektör kanalı DEĞİL — şemada `role: "context"` alanıyla işaretle; `ChannelSpec`'e `role: Literal["response","command","context"]` alanı ekle, Görev 1.1'i güncelle).
Kabul: `tests/test_residual_v1_rfly_ingest.py` (sentetik ULog fixture veya mock'lanmış pyulog).

#### Görev 1.4 — Hijyen profili + interpolasyon lint'i
`scripts/residual_v1_profile.py --dataset {alfa,rfly}`: uçuş başına JSON+HTML profil — kanal başına null oranı, dt histogramı, aralık ihlali sayısı, donmuş-sensör (`stale`) segment listesi (2 s değişimsizlik kuralı), toplam süre. Aralık ihlali >%1 olan uçuş `quarantine.json`'a düşer ve sonraki fazlarda otomatik dışlanır.
Ek: `tests/test_residual_v1_no_interpolation_lint.py` — `residual_v1/` kaynak ağacında `interpolate(`, `.resample(` ve `fillna(method=` desenlerini grep'leyen ve bulursa FAIL eden test (kaba ama etkili kilit).
Kabul: iki veri setinde profil koşusu tamamlanır; MLflow'a `phaseA_profile` run'ı olarak özet metrikler (uçuş sayısı, karantina sayısı) loglanır.

#### Görev 1.5 — Split manifesti (oturum-bazlı)
`residual_v1/ingest/splits.py`: uçuşları oturum anahtarına (ALFA: kayıt günü; RflyMAD: metadata'daki test-session alanı, yoksa gün) grupla; oturum düzeyinde 70/15/15 development/test/holdout böl, 5 seed. Arıza sınıfı stratifikasyonu oturum düzeyinde (her sınıftan her bölmeye en az 1 oturum; n<8 sınıflar yalnız development'a — vaka analizi orada yapılacak). Çıktı: `artifacts/residual_v1/splits/<dataset>_seed<k>.json`, SHA-256'ları manifest'e.
Kabul: `tests/test_residual_v1_splits.py` — aynı oturunun iki bölmeye düşmediği, seed determinizmi, stratifikasyon kuralı.

---

### 2. Faz B — Faz segmentasyonu ve hizalama  [tasarım §4.1, §4.4]

#### Görev 2.1 — Uçuş fazı segmentasyonu
`residual_v1/features/phases.py`: kural-bazlı `label_phases(flight) -> DataFrame[t, phase]`; fazlar `{ground, takeoff, cruise, maneuver, landing}`; eşikler `configs/residual_v1_phases.json`'da (ALFA: yer hızı<3 m/s & |climb|<0.3 → ground; |roll|>25° veya |roll_rate|>15°/s → maneuver; vb. — config'e yaz, koda gömme). Faz geçişlerinde ±1 s `phase_boundary=True` maskesi.
Kabul: sentetik trajektori fixture'ında beklenen faz dizisi; boundary maskesi genişliği.

#### Görev 2.2 — Referans-saat hizalama
`residual_v1/features/align.py`: `align_to_clock(flight, clock_topic, tolerances) -> DataFrame` — hedef saat ALFA'da `nav_info` (≈20 Hz), RflyMAD'de `vehicle_attitude`; diğer kanallar `merge_asof(direction="backward", tolerance=kanal_hz'e_göre)`; her düşük hızlı kanala `<kanal>_staleness_ms` kolonu. Tolerans aşımı → NaN + staleness=inf (asla ileriye taşıma yok). `stale` (Görev 1.4) segmentlerinde ilgili kanal NaN'lanır.
Kabul: `tests/test_residual_v1_align.py` — backward yönü, tolerans, staleness hesabı, stale maskeleme.

---

### 3. Faz C — Residual kanalları + G0  [tasarım §4.2–4.3, §5.1]

#### Görev 3.1 — Feature şeması (frozen)
`residual_v1/features/spec.py`: her residual kanalı için
```python
@dataclass(frozen=True)
class ResidualChannelSpec:
    name: str                     # "R1_aileron_roll_rate"
    command_inputs: tuple[str, ...]   # yalnız role∈{command,context}
    response: str                     # role=response, ASLA girdide değil
    horizon_s: float = 0.5            # hedef: gelecek 0.5 s ortalaması
    lag_summary: str = "tri4"         # son değer + 3 üçgen-ağırlıklı pencere (0–0.25, 0.25–0.5, 0.5–1.0 s)
```
ALFA R1–R6 ve RflyMAD Q1–Q4, tasarım §4.2–4.3'teki tanımlarla kodlanır. R3'te `g*tan(roll)/V` ve R4/Q3'te türev, `residual_v1/features/physics.py` yardımcılarında (türev = 0.5 s merkezi fark, uçlarda tek yönlü). Şemanın JSON dump'ının SHA-256'ı = `descriptor_schema_residual_v1` olarak her run manifest'ine.
Konstrüktör kuralı: `response` adı `command_inputs` içinde veya lag'li türevlerinde geçerse `ValueError` (AR-sızıntı kilidi, tasarım §2.3a).
Kabul: `tests/test_residual_v1_feature_spec.py` — AR-sızıntı kilidi testi dahil.

#### Görev 3.2 — Feature matrisi üretimi
`residual_v1/features/build.py`: `build_xy(flight_aligned, spec, phases) -> (X, y, meta)`; X = komut lag-özetleri + [V, V², faz one-hot, V×son_komut]; y = response'un ileri 0.5 s ortalaması; `ground` fazı ve `phase_boundary` satırları atılır; NaN içeren satır atılır ve oranı meta'ya yazılır (>%20 ise uyarı logu). Eğitim maskeleri: normal uçuşların tamamı + arızalı uçuşlarda `t < onset − 10 s` (guard band).
Kabul: birim test — guard band, faz dışlama, horizon hesabı; ve "X kolonları arasında response türevi yok" şema denetimi.

#### Görev 3.3 — G0 fizik kuralları
`residual_v1/models/g0_rules.py`: üç kural (tasarım §5.1) → kanal başına skor serisi; parametreler `configs/residual_v1_g0.json`. Çıktı formatı G1 ile aynı (aşağıdaki `ScoreFrame` sözleşmesi) ki karar katmanı ortak olsun:
`ScoreFrame = DataFrame[flight_id, t, channel, z]` (z = robust-z, scaler Görev 5.1'de).
Kabul: sentetik "komut var tepki yok" fixture'ında kural (i)'nin ateşlemesi.

#### Görev 3.4 — Residual hesap koşusu
`scripts/residual_v1_build_features.py --dataset alfa --seed 11`: tüm development uçuşları için X/y üret, parquet'e yaz, MLflow'a satır sayıları + NaN oranları.

#### Görev 3.5 — GÖRSEL DOĞRULAMA (STOP noktası, tasarım G3 checkpoint)
`scripts/residual_v1_sanity_plots.py`: (a) bir ALFA engine-fault uçuşunda R4 ham residual'ının (henüz modelsiz: y − komut-koşullu-medyan gibi kaba tahmin DEĞİL — burada yalnız y ve komutun zaman serisi + onset çizgisi) görselleştirilmesi; (b) en agresif manevralı 3 normal uçuşta R1 girdi/çıktı serileri. Çıktı: `artifacts/residual_v1/runs/<ts>_sanity/plots/*.png` + `SANITY_REPORT.md` (her grafiğe 2 cümle otomatik özet: onset sonrası airspeed türev işareti, manevra sırasında komut-tepki eşzamanlılığı).
**Burada DUR ve raporu sun.** (İnsan gözü tasarım Senaryo A/B'nin gerçek veride tuttuğunu onaylayacak.)

---

### 4. Faz D — G1/G2 modelleri  [tasarım §5.2–5.3]

#### Görev 4.1 — G1 ridge
`residual_v1/models/g1_ridge.py`: kanal başına `Ridge(alpha)` — alpha, development-içi 5-fold (OTURUM-bazlı fold!) ile `{0.1,1,10,100}` ızgarasından seçilir (bu, eşik değil model hiperparametresi; development içinde serbest). Fit yalnız train maskesinde. Çıktı: residual `r = y − ŷ` → ScoreFrame (z'leme Görev 5.1'de). Model + katsayılar + kanal başına R² MLflow'a; katsayıların fiziksel işaret kontrolü (`aileron→roll_rate` kazancı pozitif mi vb.) `coeff_sanity.json`'a.
Kabul: `tests/test_residual_v1_g1.py` — sentetik doğrusal dinamikte katsayı geri-kazanımı (bilinen kazançla üretilen veride öğrenilen kazanç ±%10).

#### Görev 4.2 — G2 LightGBM (aynı sözleşme)
`residual_v1/models/g2_lgbm.py`: aynı X/y, `LGBMRegressor` (max_depth≤6, n_estimators≤400, early stopping oturum-bazlı val ile). Karşılaştırma metriği: kanal başına test-DEĞİL, development-val residual varyans oranı `var(r_G2)/var(r_G1)`. Bu oran her headline kanalda ≥0.8 ise (yani <%20 iyileşme) G2 elenir ve Faz E'ye G1 ile gidilir — karar `g2_decision.json`'a otomatik yazılır (tasarım İ-4'ün testi).
Kabul: birim test — early stopping ve karar kuralı mantığı.

#### Görev 4.3 — S-4 AR/sızıntı ablasyonu
`scripts/residual_v1_s4_ablation.py`: seçilen modeli komut girdileri ÇIKARILMIŞ (yalnız bağlam) versiyonuyla yeniden eğit; `var(r_sakat)/var(r_tam)` < 1.15 çıkan kanal FLAGGED (model komutu kullanmıyor → sızıntı şüphesi) → `flags.json`. FLAGGED kanal karar katmanına giremez.

---

### 5. Faz E — Skorlama, sanity kapıları, CUSUM  [tasarım §6–7]

#### Görev 5.1 — Robust z + scaler sözleşmesi
`residual_v1/decision/scaling.py`: kanal başına median/MAD train-normal'den; MAD=0 → kanal dışlanır ve manifest'e `excluded_channels` yazılır; z clip=8 (CUSUM z_clip ile uyumlu). Scaler parametreleri run artifact'ı.

#### Görev 5.2 — Sanity kapıları S-1/S-2/S-3
`residual_v1/eval/sanity_gates.py`:
- S-1: kanal z'lerinin uçuş-içi |z| ortalaması vs aynı pencerenin ham-girdi normu ‖x‖ Spearman ρ; ρ≥0.5 → FLAG.
- S-2: yalnız G2/G3 için (öğrenilmiş ağaç/ağ); rastgele-init eşdeğeriyle sıralama ρ; ≥0.7 → FLAG. (G1 ridge için S-2 atlanır, katsayı-sanity zaten var.)
- S-3: development etiketli olaylarda kanal başına KS testi — |z| dağılımı `[onset, onset+15 s]` vs `[onset−60, onset−10 s]`; headline sınıf başına en az bir kanal p<0.01 değilse **STOP**: `S3_FAILURE_REPORT.md` üret (kanal başına KS istatistiği + en kötü 3 olayın grafiği) ve dur. Eşik/kalibrasyon koduna geçiş S-3 PASS koşuluna programatik olarak bağlanır (`raise GateError`).

#### Görev 5.3 — CUSUM (mevcut çekirdeğin yeniden kullanımı)
`residual_v1/decision/cusum.py`: `anomaly_core.sequential.MultiChannelPageCUSUM`'ı sar; k=1.0, İKİ YÖNLÜ (mevcut sınıf tek yönlüyse −z ile ikinci geçiş; koda bak, gerekiyorsa sarmalayıcıda çöz, çekirdeği değiştirme). Refractory 60 s. Alarm kaydı: `{flight_id, t_alarm, channel_contributions}`.

#### Görev 5.4 — Blok-bootstrap eşik kalibrasyonu — **STOP / NO-GO**
STOP — bkz. docs/RESIDUAL_V1_KALIBRASYON_NOGO_RAPORU.md.
`residual_v1/decision/calibrate.py`: development-NORMAL uçuşlarından 60 s bloklu bootstrap (B=500) ile kanal başına h çöz: hedef, kanal FA katkısı = 0.5/(aktif kanal sayısı) alarm/uçuş-saati. Çözüm monoton arama (h ↑ → FA ↓). Çıktı: `thresholds_frozen.json` + bootstrap FA dağılım grafiği. Bu dosya yazıldıktan sonra değiştirilemez (fail-if-exists) — test değerlendirmesi yalnız bunu okur.
Kabul: `tests/test_residual_v1_calibration.py` — monotonluk, blok örnekleme, determinizm (seed'li).

---

### 6. Faz F — Değerlendirme ve hata analizi  [tasarım §9]

#### Görev 6.1 — Olay-bazlı değerlendirme
`residual_v1/eval/events.py`: olay başına `detected (t_alarm ∈ [onset, min(end, onset+60 s)])`, `delay = t_alarm − onset`; uçuş-saati başına FA (normal uçuşlar + arızalı uçuşların onset-öncesi guard-dışı kısmı). Sınıf başına: olay-recall, medyan/p90 gecikme, bootstrap (olay-düzeyi, B=2000) %95 CI. n<8 sınıflar otomatik olarak `case_studies/` klasörüne yönlenir (sayı tablosuna girmez).
`scripts/residual_v1_evaluate.py --split test --seed all`: 5 seed × frozen eşik; MLflow'a tam tablo.

#### Görev 6.2 — Zorunlu hata analizi üretimi
`residual_v1/eval/postmortem.py`: kaçırılan HER olay için 4-panelli PNG (komut / tepki / kanal z'leri / CUSUM + eşik + onset) ve `miss_taxonomy.csv`'ye otomatik ön-sınıflama: onset±15 s'de max|z|<2 → "sinyal-yok"; max|z|≥3 ama CUSUM<h → "birikim-yavaş"; z yüksek ama FLAGGED kanalda → "maskeleme"; onset±5 s'de veri boşluğu/stale → "etiket-veri-şüpheli". (Nihai etiketleme insan işi; otomatik atama öneri kolonudur.)

#### Görev 6.3 — RflyMAD zayıf-süpervizyon kolu (yalnız Faz F'e kadar her şey PASS ise)
`residual_v1/models/g_sad.py`: residual-z 10 s pencereleri (stride 1 s) üzerinde Deep SAD-lite — 2 katmanlı MLP (64,32), merkez c train-normal ortalamasından; kayıp: normal pencerede ‖φ(x)−c‖², etiketli arıza penceresinde 1/‖φ(x)−c‖² (η=1). Skor → aynı z'leme → aynı CUSUM → AYNI frozen-bütçe prosedürüyle (development'ta yeniden kalibre edilen KENDİ eşiği; test'e bir kez). Önceden kayıtlı karar kuralı: medyan gecikme iyileşmesi <%20 VEYA FA bütçe aşımı → unsupervised kazanır, `g_sad_decision.json`.

#### Görev 6.4 — Kör holdout (yalnız insan onayıyla)
`scripts/residual_v1_holdout.py`: `--confirm-open` bayrağı olmadan çalışmaz; çalışınca holdout split'ini bir kez skorlar, sonucu ayrı MLflow run'ına yazar ve `HOLDOUT_OPENED.flag` bırakır (ikinci çalıştırma hata verir).

---

### 7. Rapor ve teslim

`scripts/residual_v1_final_report.py`: tüm run manifest'lerinden `docs/residual_v1_sonuc_raporu.md` derler — başarı sözleşmesi tablosu (hedef vs ölçülen, CI'larla), sanity kapı durumları, İ-1..İ-4 iddialarının tek tek doğrulanma/yanlışlanma durumu, miss taksonomisi dağılımı, kanal-katkı örnek alarm demoları. Sunum için ayrıca 3 "hero" grafiği: Senaryo-A gerçekleşmesi (engine olayında R4+CUSUM), Senaryo-B gerçekleşmesi (agresif manevrada sakin R1), miss-taksonomi pastası.

### 8. Görev sırası özeti ve STOP noktaları

1.1 → … → **3.5 STOP** → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 (**S-3 FAIL ise STOP**) → 5.3 → **5.4 STOP / NO-GO**. Mevcut durumda Faz F'ye geçilmez; yeniden açma koşulları NO-GO raporundadır.

Definition of done: tüm kabul testleri yeşil; yeterli maruziyet kapısı geçilirse `thresholds_frozen.json` tek kez yazılmış; İ-1..İ-4 durumu raporda; hiçbir yasak desende (Görev 1.4 lint + bu belgenin §0.7'si) ihlal yok. Mevcut tur bu tanıma ulaşmadı; NO-GO raporunda kayıtlıdır.


---

## RESIDUAL-V1 Görev 4.1 Şartlı GO Kaydı

Tarih: 2026-07-17  
Karar kaynağı: Görev 1.1–3.5 bağımsız denetim raporu  
Durum: Görev 4.1 için şartlar yerine getirildi; Faz E için K5 açık.

### Uygulanan şartlar

#### K1 — bağlam geçmişi sızıntısı

ResidualChannelSpec komut ve bağlam girdilerini artık ayrı rollerle taşır.
Yalnız gerçek komutlar tri4 pencereleri ile delta_1s alır; bağlam girdileri
yalnız anlık __last değeriyle matrise girer. Yanıt/yanıt-geçmişi kilidi iki
girdi rolünü de denetler.

- Eski descriptor hash:
  b6ac3412db3f6c8229cfadd37e542c6c7b29c7b89f42bd8fb5388e6032c0f93d
- K1 sonrası descriptor hash:
  86cd49485995b4779934c6b02cf85a26bf4cf303d59c987600ed00a1080c80ca

Eski feature artefaktları kanıt zinciri olarak korunur; Görev 4.1 yalnız yeni
hash ile yeniden üretilen feature matrislerini kabul eder.

#### K2 — ALFA engine/R4 beklentisinin düzeltilmesi

Sanity kanıtında engine onset anında throttle yüksek kalmıyor, sıfıra düşüyor.
Bu nedenle G1, yavaşlamanın bir bölümünü beklenen tepki olarak tahmin edebilir
ve R4 residual kayması ön-kayıttaki beklentiden küçük çıkabilir. Muhtemel sinyal
cruise + throttle=0 + sabit airspeed_cmd dağılım-dışılığıdır. Bu kayıt Görev
4.1 için başarı iddiası değildir; S-3 zayıf çıkarsa önceden kaydedilmiş açıklama
adayıdır.

#### K3 — ALFA kapsam sınırı

Önceki development feature koşusunda 32 uçuşun yalnız 5'inde R1–R5 satırı
vardı; 101.913 aday satıra karşı 32.577 satır tutuldu. Development'taki 10
engine olayının yaklaşık 4'ü R4 için kullanılabilir. ALFA test bölmesinde tek
normal uçuş bulunduğundan ALFA-özel FA/uçuş-saati tahmini savunulabilir
genişlikte değildir; doğal FA bütçesinin ana dayanağı RFLY olacaktır.

**ALFA headline iddiası tek test oturumuna dayanır ve holdout'ta R1–R5 kapsamı beklenmez.**

Bu sınır nedeniyle boş coverage başarı ya da başarısızlık olarak
yorumlanmayacak; her kanal raporunda uçuş, oturum ve satır coverage'ı ayrıca
verilecektir.

#### K4 — PWM trim merkezleme

Arızalı ALFA uçuşlarında aileron/elevator/rudder PWM deltaları artık yalnız ilk
arıza onset'inden önceki örneklerin medyanıyla merkezlenir. Onset öncesinde
sonlu trim örneği yoksa ingest fail-closed davranır. Normal uçuşlarda tüm uçuş
medyanı kullanılmaya devam eder.

#### K5 — waypoint maskesi

R6 için waypoint değişimi çevresindeki ±2 saniye maskesi henüz yoktur. R6,
Görev 4.1 G1 regresyonunun dışında tutulur. Bu maske Faz E başlamadan önce
zorunlu olarak uygulanacaktır.

#### K6 — R6 doğrudan kanal

R6 ridge ile eğitilmeyecek ve G1 metriklerine katılmayacaktır. Görev 5.1'de
öğrenmesiz biçimde doğrudan robust-z kanalına dönüştürülecektir.

### Deney disiplini

- G1 hiperparametre seçimi yalnız development içindeki oturum fold'larında
  yapılır.
- Test ve holdout feature/telemetrisi model seçimine girmez.
- Holdout açma kilidi bu aşamada kullanılmaz.
- S-3 geçmezse eşik kalibrasyonuna gidilmez; AP-1 gereği çıktı yeni veri seti
  arayışı değil hata analizi raporudur.


---

## RESIDUAL-V1 Görev 4.1 Sonuçları

Tarih: 2026-07-17  
Descriptor hash:
86cd49485995b4779934c6b02cf85a26bf4cf303d59c987600ed00a1080c80ca

### Sonuç

Görev 4.1 development-only ve oturum-bazlı CV sözleşmesiyle tamamlandı.
Test telemetrisi model seçimine girmedi; holdout telemetrisi açılmadı.

ALFA'da R1–R5 için beş feature taşıyan development uçuşunun tamamı aynı
2018-07-18 oturumunda. En az iki oturum gerektiren CV kapısı nedeniyle ALFA G1
modeli eğitilmedi. Uçuşları fold'lara bölerek aynı oturumun iki tarafa
sızdırılması reddedildi. R6, K6 gereği öğrenmesiz/doğrudan kanal olarak G1
dışında tutuldu.

RFLY'de Q1–Q3, 12 development oturumu üzerinde gerçek 5-fold session CV ile
eğitildi. Q4'ün train-eligible satırı olmadığı için model kurulmadı.

| Veri | Kanal | Train satır | Uçuş | Oturum | Alpha | CV R² | Train R² | Karar |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ALFA | R1 | 1.361 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R2 | 1.361 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R3 | 1.361 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R4 | 1.299 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R5 | 1.302 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R6 | 16.793 | 32 | 3 | — | — | — | K6: doğrudan kanal |
| RFLY | Q1 | 821.977 | 238 | 12 | 0,1 | 0,0114 | 0,3638 | eğitildi; zayıf genelleme |
| RFLY | Q2 | 821.980 | 238 | 12 | 100 | 0,4564 | 0,7109 | eğitildi; en güçlü G1 |
| RFLY | Q3 | 822.005 | 238 | 12 | 100 | 0,0003 | 0,0085 | eğitildi; pratikte sinyalsiz |
| RFLY | Q4 | 0 | 0 | 0 | — | — | — | train coverage yok |

Q1 ve özellikle Q3'ün train–CV farkı/çok düşük CV R² değeri, bu kanalların
Faz E'ye otomatik kabulü anlamına gelmez. Sonraki model adımı S-1/S-3/S-4
kapıları ve development hata analizi olmalıdır. S-3 geçmeden eşik
kalibrasyonuna gidilemez.

### Kapsam beyanı

**ALFA headline iddiası tek test oturumuna dayanır ve holdout'ta R1–R5 kapsamı beklenmez.**

ALFA için model kurulamaması başarısızlığı gizlemek üzere test oturumunun
training'e alınmasına izin vermez. Tasarım değişikliği istenirse bu ayrı insan
kararı ve yeni şema/ön-kayıt gerektirir.

### Artefaktlar

- ALFA G1 run:
  artifacts/residual_v1/runs/20260717_081305_phaseD_g1_ridge_alfa_seed11
- RFLY G1 run:
  artifacts/residual_v1/runs/20260717_090138_phaseD_g1_ridge_rfly_seed11
- Yeni ALFA Silver:
  artifacts/residual_v1/silver/alfa_preonset_trim_v2
- K1/K4 feature kökü:
  artifacts/residual_v1/features_k1k4
- Tüm uçuş görsel/rapor handout'u:
  artifacts/residual_v1/runs/20260717_065725_full_flight_handout/handout

Her eğitilmiş kanalın joblib modeli, residual parquet'i, fold oturumları,
alpha adayları, katsayıları ve coverage raporu kendi run klasöründedir.


---

## RESIDUAL-V1 — Faz E Öncesi Uygulama Planı (Codex için)

Tarih: 2026-07-17 · Statü: **Uygulandı — K5/S-4/ölçekleme/S-1/S-3 tamamlandı;
kalibrasyon yetersiz development-normal maruziyeti nedeniyle STOP.**
Kaynak: Görev 4.1 sonrası kullanıcı tarafından dondurulan sıra + bu oturumda yapılan
bağımsız kod denetiminin bulguları. Bu belge `docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md`
ve `docs/RESIDUAL_V1_DENEY_TASARIMI.md`'yi günceller/tamamlar, çelişmez.

### Dondurulmuş sıra

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

### 1. K5 — waypoint ±2 s maskesi

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

### 2. S-4 — komutsuz-girdi ablasyonu

`docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md` Görev 4.3'te zaten tam tanımlı:
`scripts/residual_v1_s4_ablation.py` — seçilen G1 modelini komut girdileri (spec'in
`command_inputs`'u, `context_inputs` değil) çıkarılmış halde yeniden eğit;
`var(r_sakat)/var(r_tam)` < 1.15 → FLAGGED, `flags.json`'a yaz. FLAGGED kanal karar
katmanına giremez. K5'ten SONRA çalışmalı çünkü R6 hâlâ G1'e girmiyor (K6) — bu adım
yalnız RFLY Q1/Q2/Q3'ü etkiler (ALFA'da zaten eğitilmiş model yok).

### 3-4. Ölçekleme + S-1 (Görev 5.1 + 5.2'nin ilk yarısı)

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

### 5. S-3 — development olay ayrımı, ALFA `not_evaluable` kuralı

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

### 6. CUSUM + kalibrasyon (Görev 5.3-5.4) — yalnız S-3 PASS ise

`residual_v1/decision/cusum.py` (mevcut `anomaly_core.sequential.MultiChannelPageCUSUM`
sarmalayıcı, k=1.0, iki yönlü, refractory 60s) ve `decision/calibrate.py` (blok-bootstrap,
B=500, kanal FA katkısı = 0.5/aktif kanal sayısı). `thresholds_frozen.json` fail-if-exists.
Kod zaten `raise GateError` ile S-3 PASS koşuluna programatik bağlanacak şekilde
tasarlanmış (Görev 5.2 metni) — bu bağlantının gerçekten var olduğu, S-3 atlanarak
kalibrasyona geçilemeyeceği ayrı bir testle kanıtlanmalı (`GateError` fırlatma testi).

---

### Codex'in bu plana başlamadan teyit etmesi gereken açık noktalar

1. ~~`waypoint_distance` Silver'a ulaşıyor mu~~ — çözüldü, ulaşıyor (bkz. §1).
2. ~~Waypoint-geçiş tespit eşiği/pencere genişliği~~ — çözüldü, Codex'in development-
   ölçümlü V-dönüşü sözleşmesi kabul edildi (bkz. §1, dondurulmuş 6 parametre).
3. `GateResult`/rapor şemasına `not_evaluable` durumunun nasıl ekleneceği — mevcut
   `G1ChannelFit.report`/`coverage` sözleşmesiyle tutarlı bir alan adı seçilmeli.
4. ~~R6 için S-1'in ham-girdi referansı~~ — çözüldü. Dondurulan bağımsız vekil
   `sqrt((roll/rad(25°))² + (roll_rate/rad(15°/s))²)`; development S-1 sonucu
   Spearman rho=0.471774 < 0.5.

### Codex'in bu turda yaptığı ve doğrulanan iş

Citation düzeltmesi (`alfa.py:85`), ChannelSpec'in ingest filtresi olmadığı bulgusu
(47/47 uçuş, iki Silver kökü) ve V-dönüşü örüntüsü (development ölçümü + görsel) bu
oturumda ayrıca bağımsız olarak (farklı bir uçuşta, ayrı bir Python komutuyla) teyit
edildi — üçü de doğru çıktı, plana aynen işlendi.

Bu belge uygulama öncesi planı ve karar izini korur; güncel kod/sonuç ankrajları için
`docs/RESIDUAL_V1_FAZ_E_SONUCLARI.md` esas alınır.


---

## RESIDUAL-V1 Faz E Sonuçları — kalibrasyonda STOP

Tarih: 2026-07-17  
Kapsam: yalnız development. Test okunmadı; sealed holdout açılmadı.

### Son karar

K5, S-4, train-normal robust ölçekleme, S-1 ve S-3 tamamlandı. S-3 ALFA-engine,
RFLY-motor ve RFLY-sensor sınıflarında ayrı ayrı PASS verdi. Buna rağmen eşik
kalibrasyonu tamamlanmış sayılmaz: development-normal uçuş saati dondurulmuş yanlış-alarm
hedeflerini çözmeye yetmiyor. Korumalı son koşu `thresholds_frozen.json` yazmadan
`GateError` ile durdu.

Bu nedenle **test veya holdout değerlendirmesine geçmek yasaktır.**

### K5 — waypoint V-dönüşü maskesi

- Mapping zaten `residual_v1/ingest/alfa.py` içindeydi; `waypoint_distance` Silver'a
  47/47 uçuşta ulaşıyordu. Kanal şimdi profil hijyeni için `alfa_channels.py` içinde
  context olarak declare edildi.
- Dondurulmuş altı parametre `configs/residual_v1_waypoint_mask.json` içindedir.
- Algoritma yalnız gözlenen/aligned örneklerde çalışır; interpolasyon/resample/fill yoktur.
- Tam iki taraflı 2 s trend penceresi zorunludur. Bu koruma, uçuşun ilk 0.3 saniyesindeki
  iki telemetri initialization/reset olayının yanlış V-dönüşü sayılmasını engelledi.
- Development sonucu: 32 uçuşun 6'sında 9 olay; toplam 697 reference-clock satırı maskeli.
- Maske yalnız `R6_xtrack_error` için bildirimsel `boundary_masks=("waypoint",)` ile uygulanır.
  R1–R5 ve Q1–Q4 etkilenmez.
- Descriptor model hash'i değişmedi; satır-uygunluk politikası ayrı waypoint config SHA-256
  ile provenance'a yazılır.

### S-4 — komut girdisi ablasyonu

Run: `artifacts/residual_v1/runs/20260717_111752_phaseE_s4_ablation_rfly_seed11`

| Kanal | Var(sakat)/Var(tam) | Eşik | Sonuç |
|---|---:|---:|---|
| Q1 | 1.1991885751 | 1.15 | PASS |
| Q2 | 2.4971561312 | 1.15 | PASS |
| Q3 | 1.0080518029 | 1.15 | FLAGGED — karar hattından çıkarıldı |
| Q4 | — | 1.15 | not_evaluable/model_unavailable |

### Ölçekleme ve S-1

Scaling run: `artifacts/residual_v1/runs/20260717_112136_phaseE_scaling_seed11`  
S-1 run: `artifacts/residual_v1/runs/20260717_112412_phaseE_s1_magnitude_seed11`

Train-eligible normal satırlardan kanal başına raw median/MAD fit edildi; z ±8'de clip edildi.
Aktif kanallar Q1, Q2 ve doğrudan R6'dır. R6, pre-K5 feature artefaktından okunmadı;
Silver'dan güncel K5 maskesiyle yeniden üretildi.

R6 için tautolojik `|z(xtrack)|` vs `|xtrack|` kullanılmadı. Development'ta fiziksel
adaylar karşılaştırıldı ve mevcut phase eşiklerini kullanan şu bağımsız yanal manevra vekili
donduruldu:

`M_R6 = sqrt((roll / rad(25°))² + (roll_rate / rad(15°/s))²)`

| Kanal | S-1 Spearman rho | Eşik | Sonuç |
|---|---:|---:|---|
| R6 | 0.4717741935 | 0.5 | PASS |
| Q1 | 0.1397776998 | 0.5 | PASS |
| Q2 | 0.0179171807 | 0.5 | PASS |

### S-3 — threshold-independent development ayrımı

Run: `artifacts/residual_v1/runs/20260717_112813_phaseE_s3_separation_seed11`

Sınıflar birleştirilmedi. ALFA R1–R5 satırları açıkça
`not_evaluable/model_unavailable`; ALFA-engine kararı yalnız R6'dan üretildi.

| Veri/sınıf | Kanal | KS | p | Pre medyan |z| | Post medyan |z| | Sonuç |
|---|---|---:|---:|---:|---:|---|
| ALFA/engine | R6 | 0.1645976552 | 5.52e-18 | 1.1285 | 1.8119 | PASS |
| RFLY/motor | Q1 | 0.1772192456 | ≈0 | 0.9841 | 1.5001 | PASS |
| RFLY/motor | Q2 | 0.5702057263 | ≈0 | 1.0976 | 5.7945 | PASS |
| RFLY/sensor | Q1 | 0.2739631940 | ≈0 | 0.8751 | 1.5857 | PASS |
| RFLY/sensor | Q2 | 0.0340301434 | 2.28e-91 | 0.9500 | 1.0228 | PASS, küçük etki |

Önemli yorum sınırı: bunlar pooled satır-düzeyi KS sonuçlarıdır. ALFA'daki 10 engine
olayının bireysel medyan kaymaları heterojendir; pooled PASS, “her olay tespit edildi”
anlamına gelmez. Handout'taki olay-düzeyi grafik bunu görünür tutar.

### CUSUM ve kalibrasyon STOP'u

İki yönlü sarmalayıcı ortak `anomaly_core.sequential.MultiChannelPageCUSUM` çekirdeğini
k=1.0, z clip=8 ve 60 s refractory ile yeniden kullanır. S-3 PASS kilidi programatiktir.

İlk kalibrasyon denemesi matematiksel maruziyet açığını görünür kıldı ve reddedildi:
`artifacts/residual_v1/runs/20260717_113330_phaseE_cusum_calibration_seed11` içindeki
eşikler kullanılmamalıdır; run'a append-only `DO_NOT_USE_THRESHOLDS.md` işareti eklendi.

Korumalı nihai koşu:
`artifacts/residual_v1/runs/20260717_113747_phaseE_cusum_calibration_seed11`

| Veri/kanal | Mevcut normal saat | Hedef alarm/saat | Tek alarmı çözmek için minimum saat |
|---|---:|---:|---:|
| ALFA/R6 | 0.168846 | 0.50 | 2.0 |
| RFLY/Q1 | 0.786237 | 0.25 | 4.0 |
| RFLY/Q2 | 0.786237 | 0.25 | 4.0 |

Bootstrap yeni bağımsız uçuş saati yaratamaz. Bu nedenle nihai run'da
`thresholds_written=false`, `calibration_locked=true`; eşik dosyası yoktur.

### Claude handout

Klasör: `artifacts/residual_v1/phase_e_handout_20260717`

- `README_FOR_CLAUDE.md`
- `SUMMARY_FOR_CLAUDE.json`
- `GALLERY.html`
- 7 numaralı PNG: K5 galerisi, S-4, S-1, S-3, olay heterojenliği ve kalibrasyon STOP'u.

Claude terminalden klasöre doğrudan erişebilir; dosya taşımaya gerek yoktur.


---

## RESIDUAL-V1 — Nihai Kalibrasyon NO-GO Raporu

Tarih: 2026-07-17  
Karar: **NO-GO — mevcut development-normal maruziyetle elde edilemez**  
Kapsam: ALFA/engine (`R6_xtrack_error`) ve RFLY/motor-sensor (`Q1`, `Q2`)

### Yönetici özeti

RESIDUAL-V1'in eşik kalibrasyonu mevcut veri ve enstrümantasyonla tamamlanamaz. Bu karar
genel bir “dedektör çalışmadı” sonucu değildir. Tam tersine, eşikten önce sınanan yöntem
zinciri çalışmıştır: K5 waypoint maskesi uygulanmış, S-4 komut ablasyonu karar hattındaki
Q1/Q2'yi doğrulamış, robust ölçekleme ve tautoloji-düzeltilmiş S-1 üç aktif kanalda geçmiş,
S-3 ise **ALFA/engine, RFLY/motor ve RFLY/sensor sınıflarının üçünde de threshold-bağımsız
ayrışma göstermiştir**.

**Bu NO-GO'nun nedeni sinyal yokluğu değil, dondurulmuş yanlış-alarm bütçesini güvenilir
biçimde çözmek için yeterli bağımsız normal uçuş-saati bulunmamasıdır.** Bu ayrım sonucu
“başarısız dedektör” diye özetlemeyi bilimsel olarak yanlış kılar: yöntem sinyali S-3 ile
kanıtlanmış, operasyonel eşik ise maruziyet çözünürlüğü kapısında fail-closed durmuştur.

GNSS-bütünlük pilotuyla karşılaştırmada RESIDUAL-V1'in güçlü farkı budur: burada üç başlık
sınıfı için önceden tanımlı, threshold-bağımsız bir sinyal kapısı açıkça PASS vermiştir.
GNSS raporu bazı telemetri ayrışmaları veya gevşek eşikte tepki bulunduğunu not etmiş olsa
da, kendi kayıtlı uçtan uca kabul kapılarından geçen operasyonel bir yöntem kuramamıştı.
Dolayısıyla bu rapor GNSS'te “literatürde hiçbir sinyal yoktu” gibi daha geniş bir iddia
kurmaz; RESIDUAL-V1 için daha güçlü ve dar kanıtı öne çıkarır: **sinyal var, kalibrasyon
maruziyeti yok.**

Sonuç olarak `thresholds_frozen.json` üretilmemiştir; Faz F test/holdout değerlendirmesine
geçilemez.

### Bu turda çalışan ve doğrulanan parçalar

#### K5 — waypoint V-dönüşü maskesi

`waypoint_distance` sinyalinin Silver'a ulaştığı doğrulandı ve altı parametreli V-dönüşü
sözleşmesi development verisinde donduruldu. K5 yalnız R6'ya uygulanır; R1–R5 ve Q1–Q4'ü
etkilemez. Development'ta 32 uçuşun 6'sında 9 olay bulundu ve 697 reference-clock satırı
maskelendi. Başlangıç resetlerinin olay sayılmaması için iki taraflı trend penceresi zorunlu
tutuldu. K5 için ayrı bir run dizini açılmadı; güncel maskeyle R6 yeniden üretiminin provenance'ı
scaling run'ı ve Faz E handout'undadır.

#### S-4 — komut ablasyonu

| Kanal | `var(r_sakat)/var(r_tam)` | Karar |
|---|---:|---|
| Q1 | 1.1991885751 | PASS |
| Q2 | 2.4971561312 | PASS |
| Q3 | 1.0080518029 | FLAGGED — karar hattından çıkarıldı |
| Q4 | — | not_evaluable/model_unavailable |

Q1 ve Q2'nin komut bilgisini gerçekten kullandığı doğrulandı. Q3'ün elenmesi gizlenmiş bir
başarı değil, S-4 kapısının amaçlandığı gibi çalıştığının kanıtıdır.

#### Robust ölçekleme ve S-1

Aktif kanallar Q1, Q2 ve R6'dır. Train-eligible normal satırlardan median/MAD fit edilmiş,
skorlar ±8'de kırpılmıştır. R6 için `xtrack_error`ı kendisiyle karşılaştıran tautolojik bir
test kullanılmamış; bağımsız yanal manevra vekili dondurulmuştur:

`M_R6 = sqrt((roll / rad(25°))² + (roll_rate / rad(15°/s))²)`

| Kanal | S-1 Spearman ρ | FLAG eşiği | Karar |
|---|---:|---:|---|
| R6 | 0.4717741935 | 0.5 | PASS |
| Q1 | 0.1397776998 | 0.5 | PASS |
| Q2 | 0.0179171807 | 0.5 | PASS |

#### S-3 — threshold-bağımsız sinyal kanıtı

Sınıflar birbirine karıştırılmamış ve veri setleri arasında sonuç taşınmamıştır. ALFA R1–R5
`not_evaluable/model_unavailable` olarak kalmış; ALFA/engine kararı yalnız R6'dan verilmiştir.

| Veri/sınıf | Kanal | KS | p | Pre medyan |z| | Post medyan |z| | Karar |
|---|---|---:|---:|---:|---:|---|
| ALFA/engine | R6 | 0.1645976552 | 5.52e-18 | 1.1285 | 1.8119 | PASS |
| RFLY/motor | Q1 | 0.1772192456 | ≈0 | 0.9841 | 1.5001 | PASS |
| RFLY/motor | Q2 | 0.5702057263 | ≈0 | 1.0976 | 5.7945 | PASS |
| RFLY/sensor | Q1 | 0.2739631940 | ≈0 | 0.8751 | 1.5857 | PASS |
| RFLY/sensor | Q2 | 0.0340301434 | 2.28e-91 | 0.9500 | 1.0228 | PASS, küçük etki |

Bu sonuçlar pooled satır-düzeyi dağılım ayrışmasıdır; “her olay yakalandı” veya operasyonel
recall kanıtı değildir. Bununla birlikte üç headline sınıfta eşikten önce ölçülebilir sinyal
bulunduğunu gösterir ve kalibrasyon kapısına geçiş için tanımlanmış S-3 koşulunu karşılar.

### Kalibrasyon maruziyet açığı

Korumalı kalibrasyon koşusu, hedef alarm oranında tek bir alarmı dahi çözebilmek için gereken
asgari süreyi development-normal maruziyetle karşılaştırmış ve eşik yazmadan durmuştur.

| Veri/kanal | Mevcut normal maruziyet (saat) | Asgari hedef (saat) | Gereken çarpan | Sonuç |
|---|---:|---:|---:|---|
| ALFA/R6 | 0.168846 | 2.0 | 11.845× | Yetersiz |
| RFLY/Q1,Q2 | 0.786237 | 4.0 | 5.088× | Yetersiz |

Bootstrap mevcut blokları yeniden örnekleyebilir; yeni bağımsız uçuş-saati yaratamaz. İlk
keşif koşusunda çıkan sıfır-alarm eşikleri bu nedenle güvenilir kalibrasyon sayılmamış ve
`DO_NOT_USE_THRESHOLDS.md` ile açıkça reddedilmiştir. Son korumalı koşunun durumu
`stopped_insufficient_calibration_exposure`, `thresholds_written=false`'dur.

### Veri tavanı testi

| Veri | Doğrulanan veri tavanı | Mevcut kullanım/split | En iyimser ekleme | En iyimser sonuç | Hedefe kalan açık |
|---|---|---|---:|---:|---:|
| ALFA | Sabit 47 uçuşluk corpus; toplam 11 normal | development 9, holdout 1, test 1 | 2 normal uçuş | 9→11 uçuş; yalnız +%22 | 11.845× gereksinimi kapanmaz |
| RFLY | Resmî kaynakta 84 `Real-No_Fault` | projede 51; development 41, holdout 10 | en çok 33 aday | ≈1.419 saat | ≈2.581 saat / ≈135 uçuş |

ALFA'da test ve holdout'taki iki normal uçuşun development'a taşınması hem rol izolasyonunu
bozardı hem de sayısal açığı kapatmazdı. Bu nedenle redistribution denenmemiştir.
ALFA'nın 47 uçuşluk corpus tavanı Keipour, Mousaei ve Scherer'in yayımlanmış
[ALFA çalışması](https://arxiv.org/abs/1907.06268) ve yerel corpus sayımıyla doğrulanmıştır.

RFLY'nin resmî veri sayfası 84 Real-No_Fault uçuş bildirmektedir. Projedeki 51 uçuşa göre
kalan en çok 33 adayın tümünün aynı ölçüde kullanılabilir olduğunu varsayan iyimser üst sınır:

`0.786237 + 33 × (0.786237 / 41) = 1.419062 saat`

Bu dahi 4.0 saat hedefinin 2.580938 saat altındadır. Development ortalaması
0.0191765 saat/uçuşla açık yaklaşık 135 ilave uçuş daha gerektirir. Resmî kaynaktaki 33 aday,
kalibrasyon için gereken toplam ek miktarın çok altındadır; bu nedenle indirme ve ingest de
başlatılmamıştır. Böylece hem redistribution hem mevcut resmî kaynaktan ek ingest yolu,
test/holdout açılmadan ve veri indirilmeden matematiksel olarak elenmiştir.

RFLY kaynak sayımı: [RflyMAD resmî dataset sayfası](https://rfly-openha.github.io/documents/4_resources/dataset.html)
(`Real Flight / No Fault = 84`). Alt-durum dağılımı resmî sayfada verilmediği için 33 sayısı
“kesin kullanılabilir uçuş” değil, **en iyimser aday tavanıdır**.

### Nihai karar ve yeniden açma koşulları

Karar **NO-GO / not achievable with current development-normal exposure** olarak dondurulmuştur.
Mevcut yanlış-alarm hedefi, sonucu gördükten sonra gevşetilmeyecek; mevcut veriyle tersine
mühendislik yapılmayacaktır.

Çalışma ancak aşağıdakilerden biri için ayrı kapsam, bütçe ve insan onayı verilirse yeniden
açılabilir:

- ALFA için mevcut 47 uçuşluk akademik corpus'un ötesinde yeni, kontrollü bir normal-uçuş
  kampanyası;
- RFLY için resmî kaynağın mevcut tavanının da ötesinde, benzer kullanılabilir maruziyet
  sağlayan yaklaşık 135 veya daha fazla yeni normal uçuş;
- ya da yanlış-alarm hedefinin ayrı bir ön-kayıtla yeniden müzakere edilmesi. Bu son seçenek
  mevcut deneyin devamı değil, yeni bir deney sözleşmesidir.

Bunlar uygulama ayrıntısı değil, yeni veri toplama veya bilimsel hedef değiştirme kararıdır;
projenin bu turdaki yetki ve kapsamının dışındadır.

### Kesinlik ve izolasyon sınırları

- Kör holdout açılmadı; açma koşulu oluşmadı.
- Test rolü kalibrasyon, hata analizi veya hedef seçimi için kullanılmadı.
- İlk keşif koşusundaki eşikler hiçbir üretim/karar hattına girmedi ve
  `DO_NOT_USE_THRESHOLDS.md` ile işaretli kaldı.
- Korumalı koşu `thresholds_frozen.json` yazmadı.
- `configs/residual_v1_cusum.json` değiştirilmedi. Rapor yazımı öncesi ve sonrası doğrulanan
  SHA-256: `627948fbfd060aa39f881f72c25cf359694642d546b2e02ee6a0a2e4d0777584`.
- Bu rapor event/uçuş recall'ı veya saha güvenilirliği iddia etmez. Kanıtlanan şey
  threshold-bağımsız dağılım ayrışmasıdır; operasyonel sınır kalibre edilmemiştir.

### Provenance ve denetim izi

Tüm yollar repo köküne göredir:

- K5 doğrulaması ve güncel R6 üretimi:
  `artifacts/residual_v1/runs/20260717_112136_phaseE_scaling_seed11`
- S-4: `artifacts/residual_v1/runs/20260717_111752_phaseE_s4_ablation_rfly_seed11`
- Scaling: `artifacts/residual_v1/runs/20260717_112136_phaseE_scaling_seed11`
- S-1: `artifacts/residual_v1/runs/20260717_112412_phaseE_s1_magnitude_seed11`
- S-3: `artifacts/residual_v1/runs/20260717_112813_phaseE_s3_separation_seed11`
- Reddedilen ilk kalibrasyon:
  `artifacts/residual_v1/runs/20260717_113330_phaseE_cusum_calibration_seed11`
- Korumalı kalibrasyon STOP'u:
  `artifacts/residual_v1/runs/20260717_113747_phaseE_cusum_calibration_seed11`
- Claude handout özeti:
  `artifacts/residual_v1/phase_e_handout_20260717/SUMMARY_FOR_CLAUDE.json`
- Görsel handout kökü: `artifacts/residual_v1/phase_e_handout_20260717`
- Maruziyet hata raporu:
  `artifacts/residual_v1/runs/20260717_113747_phaseE_cusum_calibration_seed11/CALIBRATION_COVERAGE_FAILURE.md`
- Reddedilen eşik işareti:
  `artifacts/residual_v1/runs/20260717_113330_phaseE_cusum_calibration_seed11/DO_NOT_USE_THRESHOLDS.md`
- RFLY kaynak sayımı: resmî 84, projede 51, en çok 33 ingest edilmemiş aday;
  indirme/ingest yapılmadı.

Bu rapor Görev 5.4'ün başarıyla tamamlandığını değil, kalibrasyonun fail-closed biçimde ve
sayısal gerekçeyle durduğunu kaydeder.


---

## RESIDUAL-V1 — Kalibrasyon STOP Sonrası Devam Talimatı (Codex için)

Tarih: 2026-07-17 · Statü: **Yalnız talimat — implementasyon/veri değişikliği yapılmadı.**
Kaynak: `docs/RESIDUAL_V1_FAZ_E_SONUCLARI.md`'deki kalibrasyon STOP'u + bu oturumda
yapılan bağımsız split/normal-uçuş analizi. O rapor "sonraki meşru adım daha fazla
development-normal maruziyet sağlamak veya yanlış-alarm hedef sözleşmesini yeniden
onaylamak" diyor — bu belge o iki seçeneği somut sayılarla açıyor.

### Önce: "daha fazla maruziyet" matematiksel olarak ne kadar mümkün?

Split manifest'ler + Silver `events.json` üzerinden development-normal uçuş sayıları
sayıldı (development/holdout/test roldeki normal-uçuş dağılımı):

| Veri | Development normal | Holdout normal | Test normal | Toplam normal (bilinen evren) |
|---|---:|---:|---:|---:|
| ALFA | 9 | 1 | 1 | **11** |
| RFLY | 41 | 10 | 0 | **51** |

RFLY'nin 51'i, `rflymad_parsed_pool.png`'deki bilinen Real-No_Fault sayısıyla (~51)
birebir örtüşüyor — yani development ZATEN bilinen normal-uçuş evreninin büyük
kısmını (ALFA %82, RFLY %80) elinde tutuyor.

**Kritik hesap — redistribütion (test/holdout'tan development'a kaydırma) açığı
kapatamaz:**
- ALFA: 9 normal uçuş 0.168846 saat üretiyor; hedef 2.0 saat. Elde kalan TÜM normal
  uçuşları (holdout+test'teki 2 tanesi) development'a taşısan bile development
  9→11'e çıkar (**+%22**), ama gereken çarpan **11.845×**. Matematiksel olarak
  kapanmaz.
- RFLY: 41 normal uçuş 0.786237 saat üretiyor; hedef 4.0 saat. Holdout'taki 10
  normal uçuşu da development'a taşısan development 41→51'e çıkar (**+%24**), ama
  gereken çarpan **5.088×**. Aynı şekilde kapanmaz.

**Sonuç: split içi yeniden dağıtımla (test/holdout'tan "ödünç alarak") bu açık
kapatılamaz — bu yol baştan elenir, denenmemeli.** Ayrıca test/holdout'u bu amaçla
açmak zaten Faz E raporunun açıkça yasakladığı şey.

### Codex'in yapması gerekenler (sırayla)

#### 1. Yalnız bir olgu sorusu: bilinen normal-uçuş evreni gerçekten 11/51 mi?

- **ALFA:** 47 uçuşluk academic corpus önceki bir oturumda ayrıca doğrulanmıştı
  (Keipour/Mousaei/Scherer, arXiv 1907.06268 — tam 47 işlenmiş uçuş; repodaki
  `Desktop/ALFA/processed/processed/` de tam 47 klasör). Bunun ötesinde ALFA normal
  uçuşu YOK — bu adımı atla, doğrudan §2'ye geç.
- **RFLY:** RflyMAD'ın resmî veri dokümantasyonına
  (`https://rfly-openha.github.io/documents/4_resources/dataset.html`) bak: kayıtlı
  Real-No_Fault kategorisinde şu an ingest edilen ~51'den FAZLA uçuş var mı? Varsa
  kaç tane, hangi alt-kümede. **Bu yalnız bir sayma/doğrulama işi — indirme/ingest
  YAPMA, önce sayıyı raporla.** Zaten aynı dataset'in tamamlanmamış bir köşesini
  tamamlamak AP-1'i ihlal etmez (yeni dataset değil), ama yine de aşağıdaki §3'teki
  büyüklük testini geçmeden ingest'e girişme.

#### 2. ALFA için: matematiksel olarak veri yolu kapalı — iki seçenek insana kalır

11.85× açık, 11 normal uçuşluk sabit bir evrenle kapatılamaz. Codex burada KARAR
VERMEZ, yalnız iki seçeneği net biçimde yazıp durur:
- (a) ALFA/R6 için hedef alarm bütçesini (`configs/residual_v1_cusum.json`'daki
  `total_false_alarms_per_flight_hour`, ALFA payı) yeniden müzakere etmek — bu,
  sonucu görüp eşiği gevşetmek anlamına geleceği için AYRI bir ön-kayıt gerektirir,
  şu anki dondurulmuş sözleşmenin bir parçası değildir.
  Örnekleme veri kısıtından hedefi tersine mühendislikle sızıntı olarak sayılır.
- (b) ALFA/R6 eşik kalibrasyonunu mevcut veri/enstrümantasyonla **elde edilemez**
  olarak raporlamak — `artifacts/uav_gnss_integrity_v1/uav_gnss_integrity_v1_final_no_go_report.tex`'teki
  emsal biçimde: "sinyal var (S-3 PASS), ama kabul edilebilir alarm yüküyle
  operasyonel bir karar sınırı kurulamıyor."

**Codex'in çıktısı:** bir karar değil, bu iki seçeneği ve gerekçelerini içeren kısa
bir not (`docs/RESIDUAL_V1_ALFA_KALIBRASYON_ACIK_NOKTASI.md` gibi) — kullanıcı hangi
seçeneği onaylayacağını söylemeden hiçbiri uygulanmaz.

#### 3. RFLY için: §1'in cevabına bağlı

- Eğer RflyMAD kaynağında ingest edilmemiş ek Real-No_Fault uçuş YOKSA: RFLY da
  ALFA ile aynı durumda (sabit, tükenmiş evren) — §2'deki iki seçenek RFLY/Q1/Q2
  için de aynı şekilde yazılıp insana bırakılır.
- Eğer ek uçuş VARSA: ingest etmeden önce şunu hesapla — mevcut 41 normal uçuş
  0.786237 / 41 = 0.0191765 saat/uçuş ortalaması üretiyor (development'taki
  ortalama). Gereken 4.0 saate ulaşmak için
  `(4.0 - 0.786237) / 0.0191765 = 167.588` — yani **en az 168** ek normal uçuş
  gerekir (yukarı yuvarla, kısmi uçuş olmaz) — bu, mevcut development havuzunun
  (268 uçuş) yarısından fazlası kadar yeni normal uçuş demektir. Bu sayıyı
  gerçek kaynakta bulunan ek uçuş sayısıyla karşılaştır; eğer kaynak bu kadar
  büyük değilse (muhtemel), ingest'in de açığı kapatamayacağı BAŞTAN bellidir —
  yine §2'nin iki seçeneğine düşülür. Yalnız kaynak gerçekten yeterince büyükse
  (≥168 ek uçuş), ingest planı ayrı bir onaya çıkarılır (bu da AP-2 "genişlik>derinlik"
  disipliniyle tek seferde, ölçülüp raporlanarak yapılır — sessizce büyütülmez).

### Kesin yasaklar (Faz E raporunun tekrarı, netlik için)

- Test veya sealed holdout'u bu açığı kapatmak için AÇMA.
- `configs/residual_v1_cusum.json`'daki hedefi (ya da `k`, `z_clip`, `block_s` gibi
  kalibrasyon parametrelerini) sonucu gördükten sonra sessizce değiştirip yeniden
  koşma — her değişiklik yeni bir ön-kayıt girdisi ister.
- Mevcut `20260717_113330_phaseE_cusum_calibration_seed11` altındaki
  `DO_NOT_USE_THRESHOLDS.md` işaretli eşikleri hiçbir üretim/karar kodunda kullanma.

Bu belge bir talimattır, kod veya veri değişikliği içermez.
