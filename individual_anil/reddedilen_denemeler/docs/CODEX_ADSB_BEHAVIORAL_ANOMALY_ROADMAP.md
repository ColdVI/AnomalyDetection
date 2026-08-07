# Codex ADS-B Behavioral Anomaly Detection — Aşama 1 Yol Haritası

Durum: **AŞAMA 1 UYGULANDI — V1 geçersiz koşu ve düzeltilmiş hard-physics V3 sonuçları kaydedildi.**
Tarih: 2026-07-10
Sahiplik: Codex tarafından, mevcut ADSB0/ADSB1 planlarından bağımsız hazırlanmıştır.

## 1. Karar özeti

Bu aşamada hedef, ADS-B akışından bir uçuşun **dışarıdan gözlenebilen davranışının**
fiziksel olarak normal olup olmadığını belirlemektir.

İlk ürün iki seviyede binary sonuç verir:

1. **Olay/pencere:** Bu zaman aralığında anomali var mı? (`yes/no`)
2. **Uçuş:** Bu uçuşta en az bir doğrulanmış anomalik olay var mı? (`yes/no`)

Çıktı yalnız binary etiket olmayacaktır. Alarmın başlangıcı, süresi, skoru ve açıklanabilir
bir `reason_code` alanı bulunacaktır.

Bu çalışma:

- eski `src/ml/` deneylerinin devamı değildir;
- ALFA/UAV Attack/UAV-SEAD/RFLY modellerini değiştirmez;
- ilk aşamada derin öğrenme veya büyük foundation model kullanmaz;
- ADS-B'nin gözleyemediği motor, batarya ve kontrol yüzeyi kök nedenlerini teşhis ettiğini
  iddia etmez.

## 2. Problem tanımı

### 2.1 Tespit edilecek durum

“Behavioral anomaly”, aşağıdakilerden en az biridir:

- uçuş durum kolonlarının birbirleriyle fiziksel olarak çelişmesi;
- hareketin yakın geçmişine göre süreksiz veya dinamik olarak olanaksız olması;
- fiziksel olarak mümkün olsa bile öğrenilmiş normal uçuş davranışından güçlü sapma;
- donmuş, tekrar oynatılmış veya yavaşça kaydırılmış trajectory davranışı.

### 2.2 Ayrı tutulacak durum

Aşağıdakiler davranış anomalisiyle aynı etikete sokulmaz:

- stale mesaj;
- düzensiz örnekleme veya uzun veri boşluğu;
- düşük NIC/NACp/NACv/SIL;
- MLAT/TIS-B ile ADS-B ICAO kaynak farkı;
- eksik kolon veya bozuk parse.

Bunlar ayrı bir `data_quality_anomaly` kanalıdır. Kötü veri kalitesini “uçak anormal
uçuyor” diye raporlamak yasaktır.

## 3. Neden ADS-B ile başlamak mantıklı?

Drive envanterinde 19 tarihsel tar vardır:

- toplam: 60,123,075,072 byte (~60.12 GB);
- tarih aralığı: 2024-09-01–2026-06-28;
- tar başına yaklaşık 2.1–3.7 GB;
- parse loglarında günlük yaklaşık 97–101 milyon Silver satırı ve sıfır parse hatası
  görülmüştür.

Mevcut parser Aşama 1 için gerekli çekirdek alanları üretmektedir:

```text
timestamp_utc, lat, lon, alt, alt_geom_m, on_ground,
ground_speed_ms, track_deg, vertical_rate_ms,
indicated_airspeed_ms, roll_deg,
flags_stale, flags_new_leg, ads_source_type,
nic, nac_p, sil, rc, aircraft_type, category
```

Bu veri büyük bir normal davranış havuzu sağlar. Ancak gerçek anomali ground truth'u yoktur.
Bu nedenle kontrollü bozukluk enjeksiyonu olmadan ölçülebilir “başarı oranı” üretilemez.

## 4. İzolasyon ve dizin yapısı

Claude’un mevcut ADSB0/ADSB1 dosyalarına ve eski ML hattına dokunulmayacaktır. Codex
uygulaması ayrı namespace kullanacaktır:

```text
src/adsb_behavioral/
├── __init__.py
├── contracts.py          # giriş/çıkış şeması ve reason-code sözleşmesi
├── reader.py             # chunk/stream tabanlı Silver okuma
├── flight_segments.py    # uçuş segmentleri ve kalite raporu
├── phases.py             # ground/climb/cruise/descent/turn
├── physics_residuals.py  # fiziksel tutarlılık feature'ları
├── injection.py          # değerlendirme kopyalarına kontrollü bozukluk
├── baselines.py          # rule, MAD, IF, robust covariance
├── decisions.py          # skor -> olay -> uçuş binary kararı
├── evaluation.py         # event/flight metrikleri
└── visualization.py      # time-series ve harita çıktıları

scripts/run_adsb_behavioral_stage1.py
tests/adsb_behavioral/
notebooks/adsb_behavioral/
artifacts/adsb_behavioral_stage1/
```

## 5. Uçuş örneklemi ve veri sözleşmesi

### 5.1 İlk örneklem

İlk deney 60 GB'ın tamamında yapılmayacaktır. En yeni tek tar içinden:

- yalnız `ads_source_type == "adsb_icao"`;
- geçerli lat/lon ve monoton zaman;
- en az 10 dakika süre;
- en az 60 geçerli nokta;
- farklı aircraft type/category ve uçuş fazlarını kapsayan;
- train/validation/test'e bölünebilecek

bir pilot uçuş havuzu çıkarılır.

Başlangıç hedefi yaklaşık 1,000–5,000 uçuş segmentidir. Kesin sayı veri karnesi görülmeden
sabitlenmez; ancak seçim kuralları sonuç görülmeden sabitlenir.

### 5.2 Segmentasyon

Bir günlük ICAO trace'i doğrudan tek uçuş sayılmaz. Aday sınırlar:

- `flags_new_leg == True`;
- zaman boşluğu `>30 dakika`;
- yerde kalma + yeniden kalkış örüntüsü.

Üç sinyal ayrı raporlanır. Birbirlerini sessizce ezmezler. Nihai segmentasyon kuralı ilk
veri karnesinde, anomali sonuçlarına bakılmadan dondurulur.

### 5.3 Düzensiz zaman

Tüm fiziksel türevler gerçek `dt` ile hesaplanır. Önce sabit frekansa interpolate edip sonra
türev almak yasaktır; bu yaklaşım hayali hareket üretebilir. Sabit grid yalnız model girdisi
gerekiyorsa, residual üretiminden sonra ve gap maskesi korunarak oluşturulur.

## 6. Fiziksel feature mimarisi

Korelasyon matrisi keşif/görselleştirme amacıyla üretilecektir. Feature'ları aynı vektöre
koymanın temel nedeni korelasyon değil, fiziksel ilişki olacaktır.

### 6.1 Yatay hareket residual'ları

Ardışık iki koordinattan great-circle mesafe ve bearing hesaplanır:

```text
position_speed_mps = haversine(position[t-1], position[t]) / dt
speed_residual_mps = position_speed_mps - ground_speed_ms
track_residual_deg = circular_difference(bearing_deg, track_deg)
```

Ek alanlar:

- yatay ivme;
- track-rate;
- yatay jerk;
- aynı koordinatta reported speed > 0 durumu;
- koordinat sıçraması.

### 6.2 Dikey hareket residual'ları

```text
derived_vrate_mps = (alt[t] - alt[t-1]) / dt
vrate_residual_mps = derived_vrate_mps - vertical_rate_ms
baro_geom_delta_m = alt - alt_geom_m
```

Ek alanlar:

- baro–geometrik irtifa farkının değişim hızı;
- tırmanış ivmesi/jerk;
- `asin(vertical_rate / air_or_ground_speed)` ile yaklaşık flight-path angle;
- aynı anda ground durumu ve yüksek irtifa/hız çelişkisi.

Barometrik ve geometrik irtifanın mutlak farkı tek başına anomali sayılmaz; basınç ve referans
farkları nedeniyle değişebilir. Asıl sinyal süreksizlik ve kolonlar arası çelişkidir.

### 6.3 Dönüş residual'ları

Roll mevcut olduğunda koordineli dönüş beklentisi:

```text
expected_turn_rate ≈ g * tan(roll) / max(speed, epsilon)
turn_residual = observed_track_rate - expected_turn_rate
```

Roll eksikse residual `NaN` kalır; sıfırla doldurulmaz. Roll olmayan uçuşlar için yalnız
track-rate, hız ve yörünge eğriliği kullanılır.

### 6.4 Faz farkındalığı

Aynı dikey hız cruise ve climb fazlarında farklı anlam taşır. Bu nedenle fiziksel fazlar
feature olarak eklenir:

- ground/taxi;
- takeoff/initial climb;
- climb;
- cruise;
- descent/approach;
- turn/maneuver.

Bu, her oturuma ayrı normal model kurmak değildir. Model hâlâ ortak normal davranışı öğrenir;
yalnız hareketin fiziksel durumunu görür.

### 6.5 Veri-kalitesi feature'ları

Şunlar ayrı kalite skorunda tutulur:

- stale flag;
- `dt` ve gap büyüklüğü;
- NIC/NACp/NACv/SIL/RC;
- source type;
- duplicate timestamp/position;
- eksik alan oranı.

Davranış modeli düşük kalite noktalarını maskeler veya düşük güvenle raporlar; bunları pozitif
anomali etiketi olarak kullanmaz.

## 7. Kontrollü enjeksiyon protokolü

Orijinal dosyalar salt okunur ve hash ile korunur. Enjeksiyon yalnız validation/test uçuşlarının
bellekteki veya ayrı artifact kopyalarında yapılır.

### 7.1 Tek-kolon çelişkileri

1. lat/lon ani sıçrama;
2. lat/lon yavaş drift;
3. irtifa spike;
4. irtifa bias/ramp;
5. reported ground-speed bias;
6. track offset/flip;
7. vertical-rate bias veya işaret tersleme;
8. freeze/replay.

### 7.2 Çok-kolon senaryoları

1. Konum değişir, hız/track değiştirilmez — kolay fizik çelişkisi.
2. Konum+hız+track birlikte değiştirilir — fiziksel olarak kısmen tutarlı sahte rota.
3. İrtifa+vertical-rate birlikte değiştirilir — tek-step residual'ı atlatan yavaş drift.
4. Kalite göstergeleri normal bırakılarak trajectory spoofing.

Çok-kolon senaryoları zorunludur. Yalnız tek kolonu bozup yüksek skor almak, genel behavioral
anomaly başarısı olarak kabul edilmez.

### 7.3 Şiddetler

Her senaryo için sonuçtan önce üç seviye tanımlanır:

- `easy`: açık fizik ihlali;
- `medium`: operasyonel olarak anlamlı ama daha küçük sapma;
- `stealth`: normal dağılım sınırlarına yakın yavaş sapma.

Şiddetlerin sayısal değerleri train-normal dağılımı ve uçak performans sınırları incelendikten
sonra, test sonucu açılmadan ayrı bir manifestte dondurulur.

## 8. Model ve baseline sırası

### Baseline 0 — veri kalitesi

Önce sistem “yetersiz veri” diyebilmeli. Kötü veri üzerinde davranış kararı vermek zorunda
değildir.

### Baseline 1 — açıklanabilir fizik kuralları

Residual'lara train-normal robust quantile/MAD eşikleri uygulanır. Her alarm hangi ilişkinin
bozulduğunu açıklar.

### Baseline 2 — çok değişkenli klasik model

Fizik residual grupları üzerinde:

- Isolation Forest;
- robust covariance/Mahalanobis;
- PCA reconstruction residual.

Modeller yalnız normal train uçuşlarında fit edilir.

### Baseline 3 — causal forecast residual

Yalnız önceki baselinelar medium/stealth olaylarda yetersiz kalırsa küçük bir causal predictor
denenir. Autoencoder/LSTM/LLM ilk seçenek değildir.

Yeni model, aynı FA bütçesinde fizik baseline'ını geçmezse kabul edilmez.

## 9. Binary karar katmanı

Tek bir yüksek residual doğrudan uçuş anomalisi sayılmaz:

```text
satır residual'ları
  -> 10/20/30 saniyelik nedensel pencere
  -> K-of-N + histerezis/refractory
  -> olay başlangıcı ve bitişi
  -> uçuş anomaly yes/no
```

Örnek çıktı:

```json
{
  "flight_id": "4b1234_20260628_003",
  "anomaly": true,
  "event_start_utc": "...",
  "event_end_utc": "...",
  "score": 0.94,
  "reason_codes": ["speed_position_mismatch", "track_bearing_mismatch"],
  "data_quality": "good"
}
```

## 10. Split ve sızıntı önleme

İki ayrı genelleme testi yapılır:

1. **Zamansal genelleme:** erken tarihler train, daha ileri tarihler validation/test.
2. **Uçak genellemesi:** bazı ICAO24 kimlikleri tümüyle testte tutulur.

Aynı uçuşun pencereleri farklı splitlere dağıtılamaz. Scaler, quantile, threshold ve model yalnız
train-normal üzerinde fit edilir. Validation yalnız karar kalibrasyonu içindir. Test sonuçları
görüldükten sonra feature/eşik/şiddet değiştirilmez.

## 11. Metrikler

Class imbalance nedeniyle yalnız accuracy raporlamak yasaktır.

### Nokta/pencere

- precision, recall, F1, AUPRC;
- skor dağılımı ve reason-code doğruluğu.

### Olay

- event-onset recall;
- tespit gecikmesi;
- false events / flight-hour;
- kaçırılan olay tipi ve şiddeti.

### Uçuş binary

- flight precision/recall/F1;
- normal uçuş yanlış-pozitif oranı;
- anomaly tipi ve uçuş fazı kırılımı.

## 12. Ön-kayıtlı Gate'ler

### Gate A — veri ve güvenlik

- orijinal tar/Silver hash'leri değişmedi;
- holdout/test train fit'ine girmedi;
- bütün türevler yalnız geçmiş+mevcut veriden;
- davranış ve veri-kalitesi skorları ayrı;
- residual invariant/unit testleri geçti;
- segmentasyon veri karnesinde donduruldu.

### Gate B — kontrollü tespit

Normal uçuş bütçesi: **en fazla 1 yanlış olay / 10 uçuş-saati**.

- easy enjeksiyon event recall `>=0.90`;
- medium enjeksiyon event recall `>=0.70`;
- iki hedef de aynı FA bütçesinde;
- hiçbir anomaly ailesinin recall'ı gizlenmez, her biri ayrı raporlanır.

Stealth için ilk turda zorunlu başarı eşiği yoktur; ölçülür ve sonraki hipotezin girdisi olur.

### Gate C — model katkısı

Klasik/öğrenilmiş model, fizik-rule baseline'ını aynı veya daha düşük FA bütçesinde en az
`+0.05` macro event recall ile geçmelidir. Geçmezse production adayı fizik baseline'ıdır.

### Gate D — doğal veri sanity-check

En yüksek skorlu en az 100 doğal olay time-series ve harita üzerinde incelenir. Etiketli
ground truth olmadığı için bu incelemeden precision yüzdesi türetilmez. Sonuçlar:

- muhtemel gerçek maneuver/deviation;
- veri-kalitesi sorunu;
- açıklanamayan aday;
- bariz false positive

olarak manuel audit tablosuna yazılır.

## 13. Görselleştirme teslimleri

Her örnek uçuş için senkron paneller:

1. lat/lon rota haritası;
2. altitude ve vertical rate;
3. ground speed ve position-derived speed;
4. track, bearing ve roll;
5. fizik residual'ları;
6. veri-kalitesi sinyalleri;
7. enjeksiyon aralığı, gerçek onset ve model alarmı.

Ek özetler:

- feature korelasyon haritası;
- faz × feature doluluk tablosu;
- anomaly tipi × model recall/FA matrisi;
- normal/anomalik skor dağılımları;
- reason-code örnek galerisi.

## 14. Uygulama fazları

### Faz 0 — veri karnesi

- tek tarı stream et;
- segmentasyon karşılaştırmasını çıkar;
- uçuş süreleri, `dt`, gap, kolon doluluğu, aircraft type ve faz dağılımlarını raporla;
- manifest ve pilot örneklemi dondur.

### Faz 1 — fizik residual'ları

- yatay/dikey/dönüş residual'larını uygula;
- sentetik mikro-trajectory unit testleriyle işaret, birim ve causality doğrula;
- time-series görsellerini üret.

### Faz 2 — enjeksiyon motoru

- tek ve çok-kolon senaryolarını uygula;
- orijinal hash değişmezliği ve gerçek onset doğruluğunu test et;
- şiddet manifestini dondur.

### Faz 3 — binary fizik baseline

- train-normal MAD/quantile fit;
- K-of-N/histerezis;
- Gate A/B değerlendirmesi.

### Faz 4 — klasik çok-değişkenli modeller

- IF, robust covariance ve PCA;
- aynı split/aynı karar/aynı FA bütçesi;
- Gate C.

### Faz 5 — doğal veri audit ve ölçekleme

- top-100 doğal olayı incele;
- tek tar sonuçları sabitse 19 tar/60 GB'a chunk tabanlı ölçekle;
- model artifact, scaler, threshold ve schema hash'lerini paketle.

### Faz 6 — realtime adaptör

Historical ve realtime aynı residual sözleşmesini kullanır. Realtime adaptörü yalnız offline
Gate'ler geçtikten sonra eklenir.

## 15. Başarı ve başarısızlık yorumu

Başarı:

- kontrollü fizik bozukluklarında onset'i düşük FA ile yakalamak;
- hangi kolon ilişkisinin bozulduğunu açıklamak;
- görülmemiş uçak ve ileri tarihte performansı korumak.

Başarısızlık:

- yalnız amplitude/spike yakalamak;
- veri boşluğunu uçuş anomalisi sanmak;
- yalnız tek-kolon easy enjeksiyonlarda çalışmak;
- uçuş-seviyesi accuracy yüksekken event recall/FA bütçesini kaçırmak;
- doğal etiketsiz ADS-B'de doğrulanmamış başarı yüzdesi iddia etmek.

## 16. Sonraki genişleme

ADS-B Aşama 1 Gate'leri geçerse aynı `physics_residual -> score -> event -> binary flight`
sözleşmesi yeni telemetri kaynaklarına uygulanabilir. UAV telemetrisinde ek residual aileleri:

- command–response;
- IMU–GPS;
- motor simetrisi;
- enerji/batarya;
- estimator innovation.

RFLY ve UAV-SEAD sonraki öncelik olarak korunur. ADS-B hattı onların yerine geçmez; önce sade,
ölçülebilir ve açıklanabilir bir davranış dedektörü kurarak ortak altyapıyı kanıtlar.
