# RESIDUAL-V1 — Komut→Tepki Residual Tabanlı İHA Anomali Tespiti: Deney ve Model Tasarımı

Tarih: 2026-07-16 · Statü: Tasarım (implementasyon öncesi) · Kapsam: ALFA (sabit-kanat) + RflyMAD-Real (çok-rotor)

---

## 0. Tek cümlelik tez

Anomaliyi ham telemetride değil, **öğrenilmiş bir uçuş-dinamiği modelinin innovation'ında** (komut verildi → beklenen tepki − ölçülen tepki) ararsak, manevra genliği skordan yapısal olarak düşer, genlik-baskınlığı artefaktı doğamaz ve karar katmanındaki CUSUM tam da tasarlandığı problemi (residual ortalamasında kalıcı kayma) çözer. Bu, repodaki 17 fazın hiçbirinde denenmemiş ana yoldur ve aktüatör-arıza literatürünün standart yaklaşımıdır (model-based FDI: fault detection & isolation).

---

## 1. Başarı sözleşmesi — sonuç görülmeden, ama bu kez doğru birimde

Repodaki Gate C'nin ölümcül hatası birimdi: saat-bazlı yanlış alarm bütçesi × tek-uçuş olay penceresi matematiksel olarak kesişmiyordu (ADR-042). Yeni sözleşme **olay-bazlı** ve baştan şöyle donduruluyor:

**Birincil hedef (headline):** Etiketli arıza olayı başına, onset'ten itibaren **ortanca tespit gecikmesi ≤ 5 s** ve **p90 gecikme ≤ 15 s**; aynı eşik konfigürasyonunda normal uçuşlarda **yanlış alarm ≤ 0.5 alarm / uçuş-saati** (uçuş başına değil uçuş-saati başına, çünkü ALFA uçuşları kısa ve eşit değil). Bu iki sayı **aynı frozen eşikle, aynı anda** raporlanır — asla ayrı ayrı değil (repodaki %97.6-recall/25-FA fiyaskosunun dersi).

**Sınıf-bazlı raporlama kuralı:** n ≥ 8 olay içeren sınıflar (ALFA engine; RflyMAD motor, sensor) headline sayı alır ve bootstrap %95 güven aralığıyla verilir. n < 8 sınıflar (ALFA rudder n=4, aileron+rudder n=1) **headline'a girmez**; her biri tek tek vaka analizi olarak raporlanır ("bu uçuşta residual şöyle davrandı") ve kapsam beyanına "öğrenilmiş tespit iddiası yok, kural-bazlı kapsama var" yazılır. Repoda n=4'lük sınıfa yüzde vermek istatistiksel tiyatroydu; tekrar etmiyoruz.

**Go/no-go tanımı:** RESIDUAL-V1 başarılı sayılır ⇔ ALFA-engine ve RflyMAD-motor sınıflarının EN AZ BİRİNDE hedef gecikme+FA çifti tutturulur VE genlik-sanity kapıları (§6) geçilir. İkisi de tutmazsa hata analizi raporuyla dur — yeni veri setine ATLAMA (anti-pattern #1, §10).

---

## 2. Zihinsel simülasyon — modeli çalıştırmadan önce kafada koşmak

Bu bölüm tasarımın kalbi. Üç senaryoyu satır satır simüle ediyorum; her biri bir tasarım kararını zorluyor.

### 2.1 Senaryo A: ALFA engine-failure uçuşu — tespit nasıl gerçekleşir

Sabit kanat, cruise, ~18 m/s hava hızı. t=0'da motor gücü kesiliyor (failure_status ilk emisyonu). Fizik zinciri şöyle akar:

t=0–1 s: Throttle komutu hâlâ yüksek (otopilot henüz fark etmedi). Hava hızı sürtünmeden ötürü saniyede ~0.3–0.5 m/s azalmaya başlar. **Ham telemetride neredeyse hiçbir şey görünmez** — pitch, roll, irtifa normal bandında. Repodaki satır-bazlı skorlayıcıların onset'te kör olmasının nedeni bu: onset anında sinyal genliği sıfıra yakın.

Bizim modelde ne olur: `thrust→airspeed` residual kanalı, "throttle %70 + bu hava hızı + bu pitch → hava hızı türevi ≈ +0.1 m/s²" tahmin eder. Ölçülen türev −0.4 m/s². Residual r_t ≈ −0.5 m/s², kanalın train-normal MAD'ı ~0.08 ise z ≈ −6. Tek satırda alarm için yetersiz ve yetmemeli (türbülans da anlık −4z üretebilir) — ama **CUSUM t=0'dan itibaren her 20 ms'de ~|z|−k birikiyor**. k=1.5, h=25 kalibrasyonuyla birikim ~4.5/z-birim/örnek × 50 Hz → eşik ~2–4 saniyede aşılır.

t=2–5 s: Otopilot hız kaybını görüp pitch-down komutu verir; sink rate artar. İkinci bağımsız kanal (`pitch_cmd→climb_rate` tutarlılığı) da kaymaya başlar. İki kanalın CUSUM'u birlikte yükselir — tek kanallı yanlış-pozitiften ayıran imza budur.

**Beklenen sonuç:** ortanca gecikme 2–5 s bandı. Bu, hedefi (≤5 s) tam sınırdan test eden gerçekçi bir senaryo; hedef keyfi değil, bu simülasyondan türedi.

### 2.2 Senaryo B: agresif manevra, arıza yok — alarm NEDEN çalmaz

Aynı uçak, alçak irtifada 45° yatışlı keskin dönüş. Ham telemetri açısından bu uçuşun "en anormal" 10 saniyesi: roll ±45°, yaw rate 15°/s, yük faktörü 1.4 g. Repodaki AE'ler tam burada alarm üretirdi — çünkü skorları fiilen ‖x‖² idi (ρ=0.965 ölçümü).

Bizim modelde: aileron komutu −%60 verildi, model "bu komut + bu hava hızında roll rate ≈ −38°/s" der (kontrol etkinliği dinamik basınçla ölçeklenir; girdide V² etkileşimi var, §5.2). Ölçülen −36°/s. Residual ≈ 2°/s, z ≈ 0.8. **Manevra ne kadar agresif olursa olsun, komut-tepki tutarlıysa residual küçük kalır.** CUSUM birikmez. Alarm yok.

Kritik tasarım sonucu: bu ancak model **hava hızı ve uçuş fazına koşulluysa** çalışır. Koşulsuz bir model iniş yaklaşmasında (düşük V → düşük kontrol etkinliği) sistematik residual üretir ve her inişte alarm çalar. Bu yüzden faz segmentasyonu (§4.4) ve V²-etkileşimi opsiyonel değil, çekirdek gereksinim.

### 2.3 Senaryo C: modelin kendini kandırması — autoregressive sızıntı tuzağı

En tehlikeli hata modu, ve simüle etmeden görülmez: modele girdi olarak y'nin kendi geçmişini (y_{t−1}, y_{t−2}...) verirsek, model komutu öğrenmek yerine "y_t ≈ y_{t−1}" kopyacılığını öğrenir (bir adım ilerisi için bu neredeyse her zaman en düşük MSE'dir). Arıza başladığında ne olur? **Model arızalı sinyali de bir adım geriden kusursuz takip eder** — residual hiç büyümez, arıza "normal" görünür. Kağıt üstünde düşük validation loss, sahada kör dedektör. Bu, repodaki genlik artefaktının ayna görüntüsü: orada model hiçbir şey öğrenmemişti, burada yanlış şeyi öğrenir.

Tasarım önlemleri (üçü birden zorunlu):
(a) Tepki değişkeninin kendi kısa geçmişi girdiye **girmez**; girdi = komut geçmişi + yavaş bağlam durumları (V, irtifa, faz) yalnız.
(b) Tahmin ufku tek örnek değil, **0.5 s'lik pencere ortalaması** (50 Hz'de 25 örnek) — kopyacılığın işe yaramayacağı kadar uzak.
(c) **"Arızayı-görüyor-mu" ön testi** (§6, S-3): herhangi bir eşik/kalibrasyon işine girmeden ÖNCE, development'taki etiketli olaylarda residual |z|'nin onset sonrası dağılımı onset öncesine göre KS testiyle ayrışmak zorunda. Ayrışmıyorsa mimariye dön; kalibrasyonla kurtarmaya ÇALIŞMA (repodaki ML-15 dersi: kalibrasyon sinyal üretmez, olan sinyali şekillendirir).

### 2.4 Senaryo D (çok-rotor, RflyMAD motor arızası) — domain farkının simülasyonu

Quadrotor'da tek motor %30 etkinlik kaybederse karışım (mixer) matrisi bozulur: sabit hover için o motorun PWM komutu yükselir, çapraz motorunki düşer, gövdede küçük ama kalıcı bir roll/pitch bias'ı ve yaw drift oluşur. Buradaki en ayrıştırıcı residual sabit-kanattakinden farklıdır: `motor_pwm_asimetrisi | (thrust_toplam, attitude_cmd)` — yani "bu toplam itki ve bu duruş komutu için motorlar arası PWM dağılımı ne olmalı". Bu, aynı metodolojinin (komut→tepki residual) platforma özgü kanallarla ayrı örneklenmesi demek — domain separation ilkesi korunuyor, model ağırlığı taşınmıyor, yalnız metodoloji ve karar katmanı ortak.

---

## 3. Veri ve telemetri hijyeni — kanal kanal hâkimiyet

Modele ne beslediğini bilmeden model tasarlamak, repodaki `velocity_mps %100 null` ve "hayalet imputation" vakalarını üretti. Her kanal için üç soru: fiziksel anlamı ne, hangi hızda ve hangi saatle geliyor, hangi kirlilik modları var. Aşağıdaki envanter implementasyondan önce uçuş başına otomatik profil raporuyla (null haritası, dt histogramı, aralık ihlalleri) doğrulanacak — göz kararı değil, kabul testiyle.

### 3.1 ALFA kanal envanteri (kullanılacak çekirdek)

| Kanal (topic) | Fiziksel anlam | ~Hz | Bilinen kirlilik ve önlem |
|---|---|---|---|
| `mavros/nav_info/*` des_/meas_ (roll, pitch, yaw, airspeed, velocity) | Otopilot komutu vs ölçüm — residual hattının ana hammaddesi | 20–25 | Kolon adlandırma tuzağı (repoda bulunan meas_x/des_x vakası); yaw'da açı sarımı → tüm açı farkları wrap-aware (atan2) hesaplanır |
| `mavros/imu/data` (gyro, accel, quaternion) | En yüksek hızlı gerçek dinamik; referans saat | 45–50 | Quaternion işaret atlaması (q ↔ −q aynı duruş) → devamlılık düzeltmesi; accel'de bias, türev alınmaz, olduğu gibi bağlam |
| `mavros/rc/out` (servo/throttle PWM) | Fiili aktüatör komutu — nav_info'dan daha ham ve daha dürüst | 20 | PWM kalibrasyonu uçaklar arası kayar → uçuş-içi normalize (trim medyanına göre delta) |
| `mavros/global_position` + `local_position` | Konum/irtifa/yer hızı | 5–10 | GPS irtifa sıçramaları; yer hızı ≠ hava hızı (rüzgâr) — ikisi ayrı kanal, asla birbirinin yerine kullanılmaz |
| `mavctrl/path_dev`, `xtrack_error` | Yörünge sapması — reponun ölçtüğü en güçlü causal kanal (0.751) | 10 | Waypoint geçişinde yapısal sıçrama → waypoint-değişim maskesi (±2 s) |
| `failure_status/*` | Etiket: yalnız arıza AKTİFKEN emisyon | olay | Onset = ilk emisyon; öncesinde 10 s guard band train'den dışlanır (0–%50 overlap dışlama kuralı korunuyor) |

### 3.2 RflyMAD-Real kanal envanteri

ULog kaynaklı: `actuator_outputs` (motor PWM ×4), `vehicle_attitude` + `vehicle_attitude_setpoint`, `vehicle_local_position`, `battery_status` (voltaj çökmesi motor-arıza taklidi yapabilir → bağlam değişkeni olarak girer, dedektör kanalı olarak DEĞİL), `rfly_ctrl_lxl` (interval truth — repo bunu zaten çıkarmış, 5 çelişkili uçuşun dışlanması aynen devralınır).

### 3.3 Hijyen kuralları (her iki set, uçuş başına otomatik kontrol)

Zaman damgası tekdüzeliği: dt ≤ 0 satır → at ve say; dt > 5×medyan → boşluk olarak işaretle (interpolasyon YOK, §4.1). Birim tutarlılığı: her kanala fiziksel aralık kapısı (|roll| ≤ 180°, airspeed 0–60 m/s, ...); ihlal eden uçuş karantinaya, sessiz kırpma yasak. Donmuş sensör: 2 s boyunca değişmeyen yüksek-hızlı kanal → `stale` bayrağı; bu bir VERİ KALİTESİ sinyalidir ve anomali skoruna doğrudan girmez (repodaki S2 ayrımı doğruydu, korunuyor) ama residual hesabında o pencere maskelenir — donmuş girdiyle residual hesaplamak sahte alarm üretir.

---

## 4. Silver/feature tasarımı — repodan farkı: interpolasyon yok

### 4.1 Neden interpolasyon yok

Repo Silver'da her şeyi 50 Hz'e lineer interpole etti. Zihinsel simülasyonu: 5 Hz'lik GPS kanalını 50 Hz'e lineer interpole edersen, ardışık 10 örnek mükemmel bir doğru üzerinde yatar. Residual modeli bu yapay pürüzsüzlüğü öğrenir; gerçek örnek geldiği anda küçük bir kırılma olur ve model her 200 ms'de bir minik "sahte sürpriz" residual'ı üretir — CUSUM'a sistematik gürültü. Daha kötüsü: dropout sırasında interpolant iki uzak nokta arasında köprü kurar ve tam anomali anını pürüzsüzleştirir.

Yerine: her kanal **kendi doğal hızında** Silver'a yazılır. Hizalama feature-hesap anında `merge_asof(direction='backward', tolerance=kanal_bazlı)` ile yapılır ve her düşük-hızlı kanala bir `staleness_ms` kolonu eşlik eder. Model 20 Hz'lik nav_info saatinde koşar (residual hattının doğal hızı); IMU o saate en-yakın-geçmiş değerle bağlanır.

### 4.2 Residual kanalları (ALFA, v1'de 6 kanal — az ve derin)

R1 `aileron_cmd → roll_rate`, R2 `elevator_cmd → pitch_rate`, R3 `rudder_cmd + roll → yaw_rate` (koordineli dönüş fiziği: beklenen yaw_rate ≈ g·tan(roll)/V terimi girdide), R4 `throttle → airspeed_türevi(0.5 s)`, R5 `pitch + throttle → climb_rate`, R6 `xtrack_error` (öğrenmesiz, doğrudan; repodaki en iyi causal kanal, aynen alınır).

Bilinçli olarak 73–85 feature'lık repodan geriye gidiyoruz: **6 fiziksel-anlamlı kanal, her biri hata analizinde tek tek açılabilir.** Genişlik değil derinlik (anti-pattern #2).

### 4.3 RflyMAD residual kanalları (v1'de 4)

Q1 `attitude_setpoint → attitude_rate` (eksen başına), Q2 `motor_pwm_dağılımı | (toplam_itki, attitude_cmd)` — motor asimetri residual'ı, Q3 `toplam_pwm → dikey ivme`, Q4 `pozisyon setpoint → hız tepkisi`.

### 4.4 Faz segmentasyonu

Kural-bazlı, öğrenmesiz: ground/taxi (yer hızı < 3 m/s VE irtifa değişimi ~0) → tamamen dışarı; takeoff/landing (climb_rate ve irtifa eşikleri); cruise; maneuver (|roll| > 25° veya |roll_rate| > 15°/s). Model girdisine faz one-hot + V ve V² sürekli değişken olarak girer. Faz sınırlarındaki ±1 s tampon her iki fazdan da sayılmaz (sınır belirsizliği CUSUM'u kirletmesin).

---

## 5. Model kademesi — basitten karmaşığa, her adımda çıkış kapısı

### 5.1 G0 — fizik kuralları (öğrenmesiz taban + küçük-n sınıfların kapsayıcısı)

Üç kural: (i) komut-verildi-tepki-yok: |cmd delta| > eşik iken 1 s içinde tepki değişkeni MAD-bandında hareketsiz → ceza; (ii) koordineli dönüş residual'ı `yaw_rate − g·tan(roll)/V` (repodan devralınır); (iii) itki-hız tutarlılığı kaba bandı. G0 iki iş görür: öğrenilmiş modellerin geçmek ZORUNDA olduğu taban çizgisi (repoda kural-bazlının 3 NN'i yenmesi dersi — bu kez baştan bar olarak konuyor) ve rudder n=4 gibi öğrenme-imkânsız sınıfların tek meşru kapsayıcısı.

### 5.2 G1 — kanal başına ridge regresyon (ana aday)

Her residual kanalı için: girdi = komutun son 1 s'lik geçmişi (20 Hz'de 20 lag, ama 5'e indirgenmiş üçgen-ağırlıklı özetle: son değer, 0–0.25 s ort., 0.25–0.5, 0.5–1.0 + delta), bağlam = [V, V², faz one-hot, V×cmd etkileşimi]; hedef = tepkinin gelecek 0.5 s ortalaması. Tepkinin kendi geçmişi girdide YOK (§2.3). Ridge seçiminin nedeni performans değil teşhis edilebilirlik: katsayılar fiziksel yorum taşır ("aileron→roll_rate kazancı 0.63°/s per %, V² ile ölçekleniyor — makul") ve hata analizi katsayı düzeyinde yapılabilir. Eğitim yalnız normal uçuşlar + arızalı uçuşların guard-band'li onset-öncesi kısmı; uçuş-bazlı değil **oturum-bazlı** split (repodan devralınan doğru ders), 5 seed.

### 5.3 G2 — LightGBM aynı girdi/hedefle (doğrusal-olmama kontrolü)

Aynı girdiler, aynı hedef, aynı split. Amaç yeni model değil, soru cevaplamak: G2 test-residual varyansını G1'e göre >%20 düşürüyorsa dinamik anlamlı ölçüde doğrusal-dışıdır ve G2 aday olur; düşürmüyorsa G1 kalır ve derin modele hiç gidilmez. Repoda bu kıyas hiç bu netlikte kurulmadı — mimariler farklı feature setleri, farklı skorlarla yarıştı ve kıyas anlamını yitirdi.

### 5.4 G3 — küçük GRU forecaster (yalnız kanıtla açılan kapı)

AÇILMA KOŞULU: G1/G2 sonrası test residual'larında yapısal otokorelasyon kalması (Ljung-Box p<0.01, lag ≤ 1 s) VE bu otokorelasyonun hata analizinde belirli bir dinamikten (ör. fugoid salınımı) kaynaklandığının gösterilmesi. Koşul sağlanmadan G3'e gitmek yasak — repodaki "4 DL mimarisi denendi" genişliğinin panzehiri, DL'i ancak ne için gerektiğini söyleyebildiğinde kullanmak.

---

## 6. Sanity kapıları — repodan devralınan ve yenilenen

Her aday skor, kalibrasyondan ÖNCE dört testten geçer; geçemeyen FLAGGED olur ve kalibrasyona giremez:

S-1 Genlik testi (repodan): skor vs ‖x‖ (ham girdi normu) Spearman ρ < 0.5 zorunlu. Residual tasarımı bunu yapısal olarak sağlamalı; sağlamıyorsa temsilde sızıntı var demektir.
S-2 Eğitilmemiş-eş testi (repodan): eğitilmiş skor sıralaması vs aynı mimari rastgele-ağırlık ρ < 0.7.
S-3 Arızayı-görüyor-mu testi (YENİ, §2.3'ten): development etiketli olaylarında onset-sonrası |z| dağılımı, onset-öncesine göre KS istatistiğiyle ayrışmalı (her headline sınıfta p < 0.01). Bu test EŞİKTEN BAĞIMSIZ — sinyalin varlığını ölçer. Repo bunu hiç ayrı test etmedi; sinyal-yokluğu ile eşik-yanlışlığı hep iç içe kaldı.
S-4 AR-sızıntı testi (YENİ): girdiden komut kanalları çıkarılıp yalnız bağlamla eğitilen "sakat" model, tam modele yakın performans veriyorsa model komutu değil başka bir sızıntıyı öğreniyordur → temsil incelemesi.

---

## 7. Karar katmanı — CUSUM, ama bu kez doğru kalibrasyonla

Kanal başına: r_t → robust-z (train-normal median/MAD; MAD=0 kanal dışlanır ve loglanır — repodan). İki yönlü Page-CUSUM: S⁺=max(0, S⁺+z−k), S⁻=max(0, S⁻−z−k); k=1.0 (hedef kayma ~2σ varsayımı, yarısı). Alarm: herhangi bir kanal S > h_kanal.

Kalibrasyon: her kanalın h'si, development-normal uçuşlarında **blok-bootstrap** (60 s bloklar, uçuş içi) ile "kanal başına FA katkısı = toplam bütçe / kanal sayısı" olacak şekilde çözülür. Toplam bütçe = 0.5 alarm/uçuş-saati (§1). Eşikler burada donar; test ve holdout'ta değişmez. Repodaki K-of-N grid-search yerine bu — grid search sonuca bakarak seçer, ARL-kalibrasyonu bakmadan.

Alarm sonrası: refractory 60 s (aynı olayın çoklu sayımı önlenir) + alarm anında kanal-katkı vektörü loglanır (hangi residual tetikledi — isolation/teşhis ücretsiz geliyor, sunumda güçlü demo).

---

## 8. Zayıf süpervizyon kolu (yalnız RflyMAD, yalnız residual uzayında)

RflyMAD 439 etiketli gerçek arıza uçuşuyla projedeki en büyük kullanılmayan kaynak. ML-8A'nın hatası süpervizyonu ham-feature uzayında ve tek konfigürasyonla denemekti. Burada: residual-z pencereleri (10 s, 1 s stride) üzerinde Deep SAD tarzı hedef — normal pencereler merkeze çekilir, etiketli arıza pencereleri itilir; çıktı skoru yine CUSUM'a girer (karar katmanı değişmez, yalnız z kaynağı değişir). Karşılaştırma önceden kayıtlı: unsupervised CUSUM vs weakly-supervised CUSUM, aynı split, aynı bütçe, fark ≥ +0.05 medyan-gecikme-eşdeğeri değilse basit olan kazanır. Bu kol ALFA'ya taşınmaz (etiket hacmi yetersiz) — kapsamı baştan dar tutmak, "her yerde her şey" savrulmasının önlemi.

---

## 9. Değerlendirme protokolü

Split: oturum-bazlı, 5 seed (repodan devralınır; mevcut manifest'ler uyarlanarak yeniden kullanılır). Development seti **bakmak için vardır**: her deney turunda zorunlu hata analizi — kaçırılan HER olay için komut/tepki/residual/CUSUM dörtlü grafiği çizilir ve kaçırma nedeni dört sınıftan birine atanır (sinyal-yok / sinyal-var-birikim-yavaş / maskeleme / etiket-şüpheli). Bir sonraki iterasyon bu dağılıma göre seçilir. Repodaki "ön-kayıt her şeyi kilitler" aşırılığının düzeltmesi: kilitli olan test/holdout ve eşiklerdir; development'ta bakmak, kurcalamak, anlamak serbest ve zorunludur.

Metrikler: sınıf başına olay-recall @ frozen eşik, tespit gecikmesi dağılımı (medyan, p90), normal uçuşlarda FA/uçuş-saati, hepsi bootstrap %95 CI ile. Point-adjust YOK (repo ilkesi korunur). Kör holdout: her setten ~%15 oturum, bir kez, en sonda, yalnız go kararı sonrası açılır.

---

## 10. Anti-pattern sözleşmesi — repoda düşülen hatalar ve buradaki kilit

AP-1 Veri-seti hoplama: RESIDUAL-V1 boyunca yalnız ALFA + RflyMAD-Real. ADS-B, SEAD, sentetik enjeksiyon bu deneyin kapsamı DIŞINDA. Başarısızlık halinde çıktı "yeni veri seti" değil "hata analizi raporu"dur.
AP-2 Genişlik > derinlik: aynı anda en fazla 2 model ailesi (G0+G1, sonra gerekirse G2); her aile arasında zorunlu hata-analizi turu.
AP-3 Bütçe-birim uyuşmazlığı: tüm hedefler olay/uçuş-saati biriminde ve §1'de donduruldu; hiçbir metrik farklı birimde raporlanamaz.
AP-4 Sinyal-yokluğunu kalibrasyonla örtme: S-3 kapısı geçilmeden eşik/kalibrasyon işine girilmez.
AP-5 Girdi temsili sızıntısı: tepki geçmişi girdiye giremez (S-4 ile denetlenir); interpolasyon Silver'a giremez.
AP-6 İstatistiksel tiyatro: n<8 sınıfa yüzde yok, vaka analizi var.
AP-7 Dokümantasyon/deney oranı: faz başına 1 tasarım + 1 sonuç dokümanı; ADR yalnız geri-döndürülemez kararlara.

---

## 11. Takvim (10 iş günü) ve kontrol noktaları

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

## 12. Bu tasarımın yanlışlanabilir iddiaları (önceden yazılı)

İ-1: R4 (thrust→airspeed) residual'ı ALFA engine olaylarında onset+3 s içinde |z|>4'e ulaşır (S-3'ün somutlaşmışı).
İ-2: R1 residual'ı manevra genliğiyle korelasyonsuzdur (ρ<0.2) — genlik artefaktının yapısal ölümünün kanıtı.
İ-3: Q2 (motor asimetrisi) RflyMAD motor sınıfında tek başına, tüm-kanal füzyonunun gecikmesinin 1.5 katı içinde kalır (asimetri imzasının baskınlığı iddiası).
İ-4: G2, G1'e karşı residual varyansında <%20 iyileşme verir (dinamiğin bu rejimde yeterince doğrusal olduğu iddiası).

Bunlardan ikisi bile yanlışlanırsa tasarım revize edilir — ama bu kez neyin neden yanlış çıktığını söyleyebilecek kadar az ve keskin iddiayla yürüyoruz.
