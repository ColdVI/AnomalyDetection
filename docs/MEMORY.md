# MEMORY.md — Proje Bilgi Tabanı

> Bu dosya, projenin tüm kaynak dökümanlarından (resmî staj planı, ortak proje rehberi,
> araştırma konuları, ve takım arkadaşının mimari önerisi) çıkarılmış birleşik bilgi
> tabanıdır. Codex her session'da bunu okuyup proje bağlamını öğrenmeli. Çelişki olduğunda
> önce `AGENTS.md`'deki karar kayıtlarına bak — orada hangi kaynağın geçerli olduğu yazılı.

---

## 1. Üst Bağlam — Staj Programı

- **Ekip:** 3 stajyer, veri madenciliği & büyük veri alanında.
- **Süre:** 8 hafta (1-2 ay). Hafta 1-4 = ortak proje, Hafta 5-8 = bireysel projeler.
- **Seviye:** Orta (birkaç proje deneyimi olan).
- **Alan:** İnsansız Hava Araçları (İHA/UAV), açık kaynak veri setleri.
- **Ortak proje teknolojileri (resmî plan):** Python, Pandas, Apache Kafka, Git/GitHub.
- **Ortak proje resmî veri kaynakları:** OpenSky Network API, MAVLink log örnekleri.
  (Bu repo'da bunlar **adsb.lol + ALFA + UAV Attack** ile değiştirildi — bkz. `AGENTS.md`.)

### Resmî haftalık plan (orijinal, değişmeden önce)
1. Hafta 1: OpenSky API kurulumu, ham veri inceleme, GitHub deposu oluşturma.
2. Hafta 2: Kafka producer/consumer pipeline, gerçek zamanlı veri akışı.
3. Hafta 3: MAVLink log parser, veri temizleme ve normalleştirme modülü.
4. Hafta 4: EDA, ortak rapor, kod review.

### Resmî teslimler (ortak proje)
- OpenSky'den çeken Kafka producer/consumer ETL pipeline.
- MAVLink log parse+temizleme modülü.
- Ortak GitHub deposu: kod standartları, README, veri şeması dokümantasyonu.
- EDA raporu: uçuş süresi, rota dağılımı, sensör değer aralıkları.

---

## 2. Bireysel Proje — Anomali Tespiti (bu repo'nun sahibi)

Bireysel projenin teorik araştırma konuları (proje raporuna literatür özeti olarak eklenecek):

| Konu | Odak Soru |
|---|---|
| Anomali Tespitinde Etiket Sorunu | Gözetimli/gözetimsiz anomali tespiti farkları; İHA verisinde etiketli veri kısıtları; yarı-gözetimli yaklaşımlar |
| Zaman Serisi Anomali Türleri | Nokta, bağlamsal, toplu anomali — İHA sensör verisinde örnekler, hangi algoritma hangi türe uygun |
| Isolation Forest'in Matematiksel Temeli | Mesafe-tabanlı yöntemlerden neden hızlı; rassal bölümleme; anomali skoru türetimi |
| Autoencoder ve Yeniden Yapılandırma Hatası | Reconstruction error ile anomali tespiti; eşik belirleme; LSTM-Autoencoder vs standart |
| İHA Güvenliğinde Veri Madenciliği | GPS spoofing, sensör arızası, yetkisiz müdahale tespiti — literatür karşılaştırması |

**Önerilen kaynaklar:** Chandola et al. (2009) *Anomaly Detection: A Survey*; Pang et al. (2021)
*Deep Learning for Anomaly Detection: A Review*; Kowalski & Brdys (2021) *UAV anomaly detection
using machine learning*; Liu et al. (2008) Isolation Forest; Ester et al. (1996) DBSCAN.

**Bu neden Bronze mimarisini etkiliyor:** Anomali tespiti için **etiketli** veri gerekiyor.
Resmî plandaki OpenSky+MAVLink ham trafik etiketsiz. Bu yüzden ALFA (fault ground-truth) ve
UAV Attack (benign/malicious ground-truth) dataset'leri eklendi — bireysel projenin temel
girdisi bunlar olacak.

**Hafta 5-8 için saklanan kütüphaneler:** `scikit-learn`, `pytorch`/`tensorflow`, `pyspark`,
`redis`, `geopandas`, `plotly dash`. Bronze/Silver fazında bunlara gerek yok.

---

## 3. Veri Kaynakları — Teknik Detaylar

### 3.1 OpenSky Network (resmî planda vardı, bu repo'da KULLANILMIYOR — referans için)
- REST API, ADS-B alıcı ağından canlı + geçmiş veri.
- Kredi limitleri: Anonim 400/gün (10s çözünürlük), Kayıtlı ücretsiz 4.000/gün (5s),
  Katkıcı 8.000/gün (5s).
- Basic auth Mart 2026'da kaldırıldı → OAuth2 client credentials, token 30 dk'da sona eriyor.
- Güvenli polling aralığı hesaplanmıştı: ~60 saniye (4.000 kredi / ~5 kredi-istek / 24 saat).
- **Neden bırakıldı:** auth karmaşıklığı + kredi limiti. adsb.lol auth'suz ve limitsiz.

### 3.2 adsb.lol (bu repo'da OpenSky yerine kullanılıyor)
- Lisans: ODbL 1.0. Şu an **auth gerekmiyor**, ileride API key zorunlu olabilir (kod buna
  hazır olmalı — opsiyonel header).
- Rate limit dinamik, yaklaşık 1 req/sn; 4xx hatası "yanlış şey yapıyorsun" anlamına gelir.
- Realtime endpoint: `v2/lat/{lat}/lon/{lon}/dist/{nm}` — **v3 deprecated olabilir, önce
  `api.adsb.lol/docs`'tan kontrol et.** Response: `{"ac": [...]}`, her eleman bir uçak
  (`hex`, `lat`, `lon`, `alt_baro`, `gs`, `track`, ...).
- Historical veri: `github.com/adsblol/globe_history_2026` günlük tar release'leri.
  Tar içinde gzip'li per-aircraft trace JSON'ları (`traces/...`).
  - JSON yapısı: top-level `icao`, base `timestamp`, ve `trace` dizisi.
  - Her trace elemanı: `[saniye_offset, lat, lon, alt, gs, track, flags, vert_rate, ...]`.
  - Gerçek zaman = `data["timestamp"] + offset`. `alt` bazen `"ground"` string'i olabilir,
    lat/lon `None` olabilir → null-safe filtre gerekli.
- Türkiye bbox: `lat (36, 42)`, `lon (26, 45)`.

### 3.3 MAVLink (resmî planda referans protokol — ALFA içinde de kullanılıyor)
- ArduPilot tabanlı İHA'ların iletişim protokolü. `.bin` (DataFlash) ve `.tlog` (telemetry)
  log dosyalarını `pymavlink` ile okunur.
- Önemli mesaj tipleri: `GLOBAL_POSITION_INT` (konum+irtifa+hız), `ATTITUDE` (roll/pitch/yaw),
  `GPS_RAW_INT` (ham GPS), `BATTERY_STATUS` (gerilim/akım), `HEARTBEAT` (araç tipi/uçuş modu).
- Koordinatlar `int`, 1e7 ile ölçeklenmiş (`lat=411000000` → `41.1°`). **Bu dönüşüm Silver'da
  yapılacak, Bronze'da DEĞİL.**
- `mavutil.mavlink_connection(path).recv_match(blocking=False)` döngüsünde dosya sonunda
  `None` döner — bu normal, döngü durmalı.
- **pymavlink `.bin` ve `.tlog` okur, ROS `.bag` OKUMAZ.**

### 3.4 ALFA Dataset (CMU AirLab — ground-truth etiketli, anomali projesi için kritik)
- 47 otonom uçuş, 23 tam motor arızası + 24 diğer kontrol yüzeyi (aktüatör) arıza senaryosu.
  Toplam: normal uçuşta 66 dk, arıza-sonrası 13 dk.
- 4 koleksiyon halinde gelir:
  1. **Processed** → `.bag` + `.csv` + `.mat`, **ground-truth fault tipi+zamanı dahil.**
     Dosya adı pattern'i: `carbonZ_<datetime>[_n]_<failure>.{bag,csv,mat}` (gerçek dosyalarla
     doğrulanmalı, varsayım yapılmamalı).
  2. **Raw Bag** → ham `.bag` (etiketsiz).
  3. **Telemetry** → `.tlog`, `.tlog.raw`, `mav.parm`.
  4. **Dataflash** → `.bin`, `.bin-<n>.mat`, `.gpx`, `.param`, `.kmz`, `.log`.
- **Kritik:** Ground-truth etiketli veri `.bag`/`.csv`'de. `.bag` için pymavlink yetersiz
  (rosbags/bagpy gerekir). **Bu yüzden Bronze'da birincil yol: processed `.csv`'leri
  doğrudan oku — pymavlink'e gerek yok, etiketler hazır gelir.**
- Atıf: Keipour, Mousaei, Scherer (2020), *IJRR*, DOI: 10.1177/0278364920966642.

### 3.5 UAV Attack Dataset (IEEE DataPort — benign/malicious etiketli)
- PX4 Autopilot v1.11.3, Pixhawk 4 (PX4_FMU_V5), Holybro S500 çerçeve (ana platform);
  literatürde ayrıca 3DR IRIS, Yuneec H480, DeltaQuad VTOL, Standard Tailsitter/Plane
  ile genişletilmiş varyantlar var.
- Saldırı tipleri: **GPS spoofing** ve **ping DoS**.
- Orijinal format: **ULog** (`.ulg`). CSV'ler `pyulog`'un `ulog2csv` script'i ile üretilir.
- İndirme: ücretsiz IEEE hesabı + login gerekir (`ieee-dataport.org/open-access/uav-attack-dataset`).
- Platform/etiket bilgisi dosya/klasör adından çıkarılır — **gerçek dosya adlarıyla
  doğrulanmadan `split("_")[0]` gibi varsayımlara güvenilmemeli.**

---

## 4. Ortak Şema Tasarımları (referans — Silver/Gold fazında kullanılacak, Bronze'da DEĞİL)

### 4.1 Resmî plandaki ortak şema (OpenSky + MAVLink için tasarlanmıştı)
| Kolon | OpenSky karşılığı | MAVLink karşılığı |
|---|---|---|
| `timestamp` | `time` | `_timestamp` |
| `source` | `"opensky"` | `"mavlink"` |
| `vehicle_id` | `icao24` | araç seri no |
| `lat` | doğrudan | `lat ÷ 1e7` |
| `lon` | doğrudan | `lon ÷ 1e7` |
| `altitude_m` | `baro_altitude` | `alt ÷ 1000` |
| `velocity_ms` | `velocity` | `vx ÷ 100` |
| `heading_deg` | `heading` | `hdg ÷ 100` |
| `vertrate_ms` | `vertrate` | `vz ÷ 100` |
| `on_ground` | `on_ground` | `False` |
| `extra` | callsign, squawk | mesaj tipi |

### 4.2 Arkadaşın önerdiği Gold şeması (4 kaynak için — bu repo'nun hedefi)
```
event_id, source_type, platform_id, timestamp_utc, lat, lon, altitude_m,
speed_mps, heading_deg, vertical_rate_mps, on_ground, label_available,
label_type, quality_score, extra
```
Silver dönüşüm kuralları (kaynak bazında, ileride uygulanacak):
- **adsb.lol:** timestamp unix→UTC, on_ground filtrele, null drop.
- **ALFA:** lat/lon × 1e-7, alt/1000, vx²+vy² → speed, rad→deg.
- **UAV Attack:** timestamp normalize, benign/malicious flag.

---

## 5. Kullanılacak Kütüphaneler ve Lisansları

| Kütüphane | Lisans | Amaç | Faz |
|---|---|---|---|
| `requests` | — | adsb.lol API çağrıları | Bronze |
| `pymavlink` | LGPL-3 | MAVLink/.bin/.tlog okuma | Bronze (ALFA opsiyonel yol) |
| `pyulog` | BSD | UAV Attack ULog→CSV çevrimi | Bronze |
| `confluent-kafka` | Apache-2.0 | Kafka producer/consumer | Bronze |
| `pandas` | BSD-3 | Veri işleme | Bronze, Silver |
| `pyarrow` | Apache-2.0 | Parquet yazma/okuma | Bronze |
| `folium` | MIT | Trajectory/coverage haritası | Doğrulama, Hafta 4 |
| `plotly` | MIT | İnteraktif grafikler | Hafta 4 |
| `h3` | Apache-2.0 | Coğrafi grid | Hafta 4 |
| `streamlit` | Apache-2.0 | Dashboard | Hafta 4 |

> Not: kütüphaneler referans niteliğinde, lisans/network/on-prem uyumlu alternatif
> kullanılabilir (resmî dokümanın notu).

---

## 6. Güvenlik / Gizlilik Kuralları (tüm fazlarda geçerli)
- API anahtarları `.env`'de, repoya girmez.
- Kafka sadece yerel Docker container'a bağlanır.
- Folium haritalar offline render edilir (`tiles=None` veya dış sunucuya bağlanmadan).
- Streamlit usage analytics kapatılır (`gatherUsageStats = false`).
- Tüm uçuş/log verisi `.gitignore`'da — `data/` repoya asla girmez.

---

## 7. Bilinen Sorun/Çözüm Notları (resmî dokümandan, hâlâ geçerli)
- **API 429 hatası:** kredi/rate limit dolmuş; retry-after header'ını oku, o kadar bekle.
- **MAVLink log boş geliyor:** `blocking=False` ile dosya sonunda `None` dönmesi normal.
- **Kafka consumer mesaj almıyor:** `auto.offset.reset=earliest` kontrol et; producer,
  consumer abone olmadan önce çalışmış olabilir.
- **Folium haritası boş açılıyor (tiles=None ile):** beklenen davranış, koordinatlar render
  edilir ama arka plan harita gelmez.
