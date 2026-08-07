# UAV Anomali Tespiti — Proje Genel Bakış

> Bu doküman projenin tamamını tek bir yerden anlamak için yazılmıştır.
> Her klasör, her dosya, her kişinin katkısı ve pipeline'ın nasıl çalıştığı burada.

---

## 1. Proje Nedir?

8 haftalık İHA (UAV) veri mühendisliği staj projesi. 3 stajyer birlikte çalışıyor:

| Kişi | E-posta |
|---|---|
| **Metehan** | mthnsarikaya@gmail.com |
| **Anıl** | anil04keskin@gmail.com |
| **Yusuf** | yusufkaanyildiz1453@gmail.com |

**Hedef:** 4 farklı uçuş/saldırı veri setini ortak bir pipeline'dan geçirip anomali tespiti yapabilecek temiz, etiketli veri üretmek.

**Bireysel proje (Metehan):** Bu verideki anomalileri Isolation Forest ve Autoencoder gibi modellerle tespit etmek.

---

## 2. Mimari: Bronze → Silver → Gold

```
Veri Kaynakları (ham)
    │
    ▼
┌──────────────────────────────────────────┐
│  BRONZE  (ham dosyalar, değişmeden)       │
│  MinIO bucket: "bronze"                  │
│  Format: .tar / .zip / .jsonl            │
│  Kural: Hiçbir dönüşüm, hiçbir parse    │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  SILVER  (kaynak başına parse)            │
│  MinIO bucket: "silver"                  │
│  Format: Parquet (snappy)                │
│  Kural: Unit dönüşümü + etiket +         │
│         provenance — kaynaklar ayrı      │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  GOLD  (7+3 ortak şema, tüm kaynaklar)   │
│  MinIO bucket: "gold"                    │
│  Format: Parquet (snappy)                │
│  Kural: Tüm kaynaklar aynı kolonda       │
└──────────────────────────────────────────┘
    │
    ▼
ML Modelleri (Isolation Forest, Autoencoder)
```

### Katman kuralları özeti

| Katman | Ne yapar | Ne yapmaz |
|---|---|---|
| Bronze | Ham dosyayı MinIO'ya byte-for-byte yükler | Parse, dönüşüm, filtre, provenance |
| Silver | Parse + unit dönüşümü + etiket + provenance | Kaynakları birleştirme |
| Gold | 4 kaynağı 7+3 ortak kolona hizalar | Kaynak-özel kolon üretme |

---

## 3. Veri Kaynakları

### 3.1 adsb.lol Historical (Metehan)
- **Ne:** Tüm dünyadaki uçuşların geçmiş ADS-B telemetrisi
- **Format:** `.tar` arşivi → içinde per-uçak gzip'li JSON (trace dizileri)
- **Boyut:** 7 tar, ~2-3.7 GB her biri, toplamda ~21 GB
- **Drive:** `adsb_datas/` klasörü
- **Etiket:** Yok — adsb.lol'da anomali etiketi olmaz, label = null
- **Kullanım:** Zaman serisi baseline, normal uçuş profili öğrenme

### 3.2 adsb.lol Realtime (Yusuf)
- **Ne:** Canlı adsb.lol API'sinden her 60 saniyede çekilen uçuş durumları
- **Format:** Kafka → JSONL (ham landing) → Silver Parquet
- **Etiket:** Yok
- **Kullanım:** Canlı anomali tespiti (ileride)

### 3.3 ALFA Dataset (Anıl)
- **Ne:** CMU AirLab'in fault/anomaly dataset'i, 47 otonom uçuş, mid-flight arıza enjeksiyonu
- **Format:** `processed.zip` → per-sekan klasörler → per-topic CSV'ler
- **Boyut:** ~680 MB zip
- **Drive:** `ALFA/processed/` klasörü
- **Etiket:** `engine_fault`, `aileron_fault`, `rudder_fault`, `elevator_fault`, `normal`, `unknown`
- **Kullanım:** Ground-truth arıza etiketleri ile model eğitimi/değerlendirmesi

### 3.4 UAV Attack Dataset (Anıl)
- **Ne:** IEEE DataPort GPS spoofing/jamming/Ping DoS saldırı dataset'i, PX4 platformları
- **Format:** `UAVAttackData.zip` → per-log CSV'ler (vehicle_global_position, vehicle_attitude, vb.)
- **Boyut:** ~717 MB zip, 767 CSV
- **Drive:** `UAVAtackData/` klasörü + `UAVAttackData.zip`
- **Etiket:** `benign`, `gps_spoofing`, `gps_jamming`, `ping_dos`
- **Kullanım:** Siber saldırı anomali tespiti

---

## 4. Repo Klasör Yapısı

```
AnomalyDetection/
│
├── src/
│   ├── common/              # Ortak yardımcı modüller (HERKESİN kullandığı)
│   ├── ingestion/           # Bronze: ham veri yükleme
│   ├── silver/              # Silver: parse + dönüşüm
│   ├── gold/                # Gold: birleşim
│   └── processing/          # (Referans) — Eski geniş Silver denemesi
│
├── tests/                   # Tüm birim testleri (91 test, hepsi geçiyor)
├── scripts/                 # Lokal test scriptleri (MinIO gerekmeden çalışır)
├── docs/                    # Mimari kararlar, şemalar, planlar
├── data/                    # .gitignore'da — REPOYA GİRMEZ
│   ├── bronze/
│   │   ├── adsblol_historical/_input/   ← tar'ları buraya koy
│   │   ├── adsblol_realtime/
│   │   ├── alfa/_input/                 ← processed.zip buraya
│   │   └── uav_attack/_input/           ← UAVAttackData.zip buraya
│   ├── silver/
│   └── gold/
├── docker-compose.yml       # Kafka + MinIO
├── Makefile                 # Tüm komutlar buradan
├── .env.example             # Ortam değişkenleri şablonu
└── requirements.txt
```

---

## 5. `src/common/` — Ortak Altyapı

### `minio_io.py`
Tüm katmanların (Bronze/Silver/Gold) okuma/yazma yaptığı tek yer.

| Fonksiyon | Ne yapar |
|---|---|
| `write_bronze_bytes(data, object_name)` | Ham byte'ları Bronze MinIO'ya yükler (parse yok) |
| `write_silver(df, source_type)` | DataFrame'i Silver bucket'a Parquet olarak yazar |
| `write_gold(df, source_type)` | DataFrame'i Gold bucket'a Parquet olarak yazar |
| `download_raw_bytes(client, object_name)` | Bronze'dan raw byte'ları indirir |
| `read_layer(client, bucket, source_type)` | Bir bucket'taki tüm Parquet'leri birleştirip okur |
| `list_layer_objects(client, bucket, source_type)` | Bir prefix altındaki obje listesi |
| `get_minio_client()` | `.env`'den MinIO bağlantısı kurar |
| `ensure_bucket(client, bucket)` | Bucket yoksa oluşturur |
| `ObjectStoreClient` | Protocol — gerçek MinIO veya test fake'i geçilebilir |

**Neden Protocol?** Testlerde gerçek MinIO sunucusu gerekmez. `FakeMinioClient` (aşağıda) aynı arayüzü implement eder.

### `provenance.py`
```python
add_provenance(df, source_type, source_file, schema_version="silver_v1")
```
Her Silver Parquet'e 4 standart kolon ekler:

| Kolon | Örnek |
|---|---|
| `_source_type` | `"adsblol_hist"` |
| `_ingest_ts_utc` | `"2026-07-01T12:00:00Z"` |
| `_source_file` | `"v2026.06.15.tar"` |
| `_schema_version` | `"silver_v1"` |

### `fakes.py` — `FakeMinioClient`
In-memory MinIO simülasyonu. Testlerde ve local geliştirmede Docker/MinIO başlatmadan kullanılır.

```python
fake = FakeMinioClient()
fake.buckets  # {"bronze": {"alfa/data.zip": b"..."}, "silver": {...}}
fake.put_calls  # [(bucket, object_name, content_type), ...]
```

---

## 6. `src/ingestion/` — Bronze Katmanı

### `upload_raw.py`
Ham dosyayı MinIO Bronze'a değiştirmeden yükler. Tüm kaynaklar için kullanılır.

```python
upload_raw_file(input_path, source, client=None)
# Örnek: upload_raw_file("UAVAttackData.zip", "uav_attack")
# Yazar: bronze/uav_attack/UAVAttackData.zip
```

Ayrıca `merge_tar_parts(base_path)`: `.tar.aa` + `.tar.ab` parçalı tar'ları birleştirir.

### `adsblol_producer.py` (Yusuf)
adsb.lol canlı API'yi 60 saniyede bir poll eder ve her uçağı Kafka'ya yazar.

- **Endpoint:** `https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{nm}`
- **Kafka topic:** `uav.raw.states`
- **Key:** ICAO hex kodu
- **4 sorgu noktası:** İstanbul, Doğu Anadolu, Güney kıyı, Ankara
- **Graceful shutdown:** SIGINT/SIGTERM yakalanır

### `adsblol_consumer.py` (Yusuf)
Kafka'dan okuyup ham JSONL olarak Bronze MinIO'ya yazar.

- 500 mesaj biriktir → tek JSONL dosyası olarak yaz
- **Obje adı:** `bronze/adsblol_realtime/_landing/states-YYYYMMDDTHHMMSSZ.jsonl`
- **Parquet yok** — ADR-003: Bronze = sadece ham landing

---

## 7. `src/silver/` — Silver Katmanı

### `parse_adsblol_historical.py` (Metehan)
Bronze'daki tar'ı indirip parse eder, Silver'a yazar.

**Ana fonksiyonlar:**
- `parse_trace_bytes(raw: bytes) -> pd.DataFrame`: Bir uçağın gzip JSON'ını parse eder
- `parse_local_tar(tar_path) -> list[str]`: Lokal tar dosyasını işler (MinIO'suz dev için)
- `run(bronze_prefix) -> list[str]`: MinIO Bronze'daki tüm tar'ları işler

**Unit dönüşümleri (Silver'ın görevi):**
- `alt_baro` feet → metre (`* 0.3048`)
- `"ground"` string → `on_ground=True`, `alt=None`
- `gs` knot → m/s (`* 0.5144`)
- `vrate` fpm → m/s (`* 0.00508`)

**Silver çıktı kolonları:**
`source_type, source_id (icao), timestamp_utc, lat, lon, alt, alt_geom_m, on_ground, ground_speed_ms, track_deg, vertical_rate_ms, indicated_airspeed_ms, roll_deg, flags_stale, flags_new_leg, registration, aircraft_type, flight_callsign, category, squawk, label (null), + 4 provenance kolonu`

### `parse_adsblol_realtime.py` (Yusuf)
Bronze'daki JSONL landing dosyalarını okur, parse eder, Silver'a yazar.

- **Girdi:** `bronze/adsblol_realtime/_landing/*.jsonl`
- **Timestamp:** JSONL dosya adından extract edilir (`states-YYYYMMDDTHHMMSSZ.jsonl`)
- Aynı unit dönüşümleri (feet→m, knot→m/s, fpm→m/s)
- `label = null` (canlı adsb'de etiket olmaz)

### `parse_alfa.py` (Anıl)
Bronze'daki `alfa/processed.zip`'i indirip 47 sekan için Silver üretir.

**Parse mantığı:**
1. `processed.zip` içindeki `carbonZ_<datetime>_<label>/` klasörlerini bul
2. Her klasörde `global_position` CSV'yi omurga olarak al (lat/lon/alt)
3. `nav_info-roll`, `nav_info-pitch`, `nav_info-airspeed`, `nav_info-velocity`, `nav_info-yaw` topic'leri `merge_asof` ile birleştir
4. `failure_status-*.csv` varsa gerçek fault başlangıç zamanından etiket üret
5. Yoksa `infer_fault_from_seq_name()` ile klasör adından etiket çıkar

**Etiket sınıfları:**
`normal`, `engine_fault`, `aileron_fault`, `rudder_fault`, `elevator_fault`, `aileron_rudder_fault`, `unknown`

### `parse_uav_attack.py` (Anıl)
Bronze'daki `uav_attack/UAVAttackData.zip`'i indirip Silver üretir.

**Parse mantığı:**
1. Zip içindeki CSV'leri log ID'ye göre grupla (log_id = prefix, topic = suffix)
2. `vehicle_global_position` omurga: lat, lon, alt, UTC timestamp
3. `vehicle_attitude` merge: quaternion → euler (roll, pitch, yaw derece)
4. `vehicle_gps_position` merge: jamming_indicator, noise_per_ms, hdop, satellites_used
5. `battery_status` merge: voltage, current, remaining
6. `infer_label_from_path()` ile klasör adından etiket

**Etiket sınıfları:**
`benign`, `gps_spoofing`, `gps_jamming`, `ping_dos`, `unknown`

**Önemli fix (ADR-003):** Log-ID/topic regex `TOPIC_SUFFIX_PATTERN` gerçek veride kırıktı — log adları kendi içinde alt çizgi barındırıyor. Sabit 4 topic adına (vehicle_global_position, vehicle_attitude, battery_status, vehicle_gps_position) ankrajlı regex ile düzeltildi.

### `parse_generic.py` (Ortak)
Yeni dataset'ler için otomatik format algılayıcı.

**Desteklenen formatlar:** `.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`, `.xlsx`, `.xls`, `.zip` (içini açar), `.tar` (içini açar)

**Ne yapar, ne yapmaz:**
- Yapar: format algıla → DataFrame → provenance ekle → Silver'a yaz
- Yapmaz: unit dönüşümü, etiket çıkarımı (bunlar domain-specific custom parser'ın işi)

**Yeni dataset eklemek için 3 adım:**
```bash
python -m src.ingestion.upload_raw --source yeni --input ~/data.csv
python -m src.silver.parse_generic --bronze-prefix yeni/ --source yeni
# Sonra src/gold/unify.py COLUMN_MAPS'e bir satır ekle
```

---

## 8. `src/gold/` — Gold Katmanı

### `unify.py` (Anıl)
4 kaynağın Silver tablosunu 7+3 ortak şemaya hizalar.

**Gold şeması (7 sayısal + 3 metadata):**

| Kolon | Birim | Açıklama |
|---|---|---|
| `timestamp_utc` | Unix epoch (s) | UTC zaman damgası |
| `lat` | derece (WGS84) | Enlem |
| `lon` | derece (WGS84) | Boylam |
| `altitude_m` | metre | İrtifa |
| `velocity_mps` | m/s | Yer hızı |
| `heading_deg` | derece | Yön |
| `vertical_rate_mps` | m/s | Dikey hız |
| `source_type` | — | Kaynak kimliği |
| `source_id` | — | Araç/uçak ID |
| `label` | — | Anomali etiketi (null veya string) |

**Kolon eşlemesi:**

| Gold | adsblol_hist | adsblol_rt | alfa | uav_attack |
|---|---|---|---|---|
| `timestamp_utc` | timestamp_utc | timestamp_utc | timestamp_utc | timestamp_utc |
| `altitude_m` | alt | alt | alt | alt |
| `velocity_mps` | ground_speed_ms | ground_speed_ms | velocity_measured* | null** |
| `heading_deg` | track_deg | track_deg | yaw_measured | yaw_deg |
| `label` | null | null | label | label |

*ALFA'da `velocity_measured` gerçek veride oluşmuyor (`nav_info-velocity` eşleşmiyor).
**UAV Attack Silver'ında hız kolonu yok.

**Doğrulanmış sonuç** (FakeMinioClient ile, 2026-07-01): ALFA (20.239) + UAV Attack (79.646) = **99.885 satır**, tam 10 kolon.

---

## 9. `src/processing/` — Referans (Aktif Pipeline DEĞİL)

**ADR-004:** ADR-003'ten önce Anıl'ın bireysel ihtiyacı için yazdığı daha geniş Silver denemesi.

| Dosya | Ne |
|---|---|
| `alfa_silver.py` | ALFA: 563 kolon, tüm topic'ler merge, IMU + battery + GPS |
| `uav_attack_silver.py` | UAV Attack: 34 kolon, geniş merge |
| `gold.py` | Eski union (323K satır, 595 kolon) |

**Neden silinmedi?** Silver şeması ileride zenginleştirilmek istenirse (mavctrl/path_dev, IMU, GPS spoofing residual feature'ları), buradaki doğrulanmış merge_asof mantığı hazır referans.

---

## 10. `tests/` — Birim Testleri (91 test, hepsi geçiyor)

| Test dosyası | Ne test ediyor |
|---|---|
| `test_provenance.py` | `add_provenance()` — 4 kolon doğru mu? |
| `test_minio_io.py` | `write_bronze_bytes`, `write_silver`, `read_layer` |
| `test_upload_raw.py` | `upload_raw_file`, `merge_tar_parts` |
| `test_loaders.py` | `upload_raw.py` — byte'lar değişmeden yükleniyor mu? |
| `test_adsblol_realtime.py` | Consumer JSONL landing |
| `test_parse_adsblol_historical.py` | Silver parser: unit dönüşümleri, tar parse, Silver yazma |
| `test_parse_adsblol_realtime.py` | Silver parser: JSONL parse, timestamp extract, Silver yazma |
| `test_parse_alfa.py` | ALFA Silver: merge_asof, etiket çıkarımı |
| `test_parse_uav_attack.py` | UAV Attack Silver: quaternion→euler, etiket |
| `test_alfa_silver.py` | (referans) `processing/alfa_silver.py` |
| `test_uav_attack_silver.py` | (referans) `processing/uav_attack_silver.py` |
| `test_gold_unify.py` | Gold: 7+3 hizalama, boş Silver graceful handling |

**Çalıştırma:** `make test` veya `pytest`

---

## 11. `scripts/` — Lokal Test Scriptleri

Gerçek MinIO/Docker gerekmeden, `FakeMinioClient` kullanarak çalışır.

| Script | Ne yapar |
|---|---|
| `run_alfa_local.py` | ALFA `processed.zip`'i lokal parse eder, sonuçları gösterir |
| `run_uav_attack_local.py` | UAVAttackData.zip'i lokal parse eder, etiket dağılımı gösterir |
| `run_gold_local.py` | Her iki kaynağı Gold'a birleştirir, 10 kolon doğrular |

---

## 12. `docs/` — Dokümantasyon

| Dosya | İçerik |
|---|---|
| `PIPELINE_PLAN (1).md` | Nihai mimari plan, kişi bazlı rehberler |
| `AGENTS.md` | Claude Code için bağlam, bilinen sorunlar, katman kuralları |
| `decisions.md` | ADR-001..005 — tüm mimari kararlar gerekçesiyle |
| `bronze_schema.md` | Bronze şema referansı |
| `silver_schema.md` | (Referans) Eski zengin Silver şeması |
| `PROJECT_OVERVIEW.md` | Bu dosya |

---

## 13. Altyapı

### `docker-compose.yml`
```
Kafka (confluentinc/cp-kafka:7.5.0)     → port 9092
Zookeeper (confluentinc/cp-zookeeper)   → Kafka için
MinIO (minio/minio:latest)              → port 9000 (API) + 9001 (console)
```

### `.env.example`
```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BRONZE_BUCKET=bronze
MINIO_SILVER_BUCKET=silver
MINIO_GOLD_BUCKET=gold
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=uav.raw.states
ADSBLOL_API_KEY=       ← şimdilik boş, auth gerekirse
```

### `Makefile` hedefleri
```makefile
make up                    # Docker (Kafka + MinIO) başlat
make down                  # Docker durdur
make test                  # pytest
make minio-init            # bronze + silver bucket oluştur

# Bronze (ham yükleme)
make bronze-upload-adsb    # tar'ları MinIO bronze'a yükle
make bronze-upload-alfa    # processed.zip'i bronze'a yükle
make bronze-upload-attack  # UAVAttackData.zip'i bronze'a yükle
make bronze-rt-producer    # adsb.lol → Kafka (canlı, loop)
make bronze-rt-consumer    # Kafka → JSONL → MinIO bronze (canlı, loop)

# Silver (parse)
make silver-adsb-hist      # bronze tar → silver parquet
make silver-adsb-rt        # bronze jsonl → silver parquet
make silver-alfa           # bronze alfa zip → silver parquet
make silver-attack         # bronze uav zip → silver parquet

# Gold
make gold                  # tüm silver → unified gold parquet
```

---

## 14. Kim Ne Yaptı?

### Metehan
- Mimari planı oluşturdu (`docs/PIPELINE_PLAN.md`, ADR-001..003)
- MinIO IO altyapısını kurdu (`src/common/minio_io.py`)
- adsb.lol tarihsel Silver parser'ı yazdı (`src/silver/parse_adsblol_historical.py`)
- Kafka consumer'ı ADR-003'e uyguladı (Bronze = JSONL only)
- Generic parser yazdı (`src/silver/parse_generic.py`)
- Pipeline cleanup'ı yaptı (eski loader'ları sildi, klasörleri düzeltti)

### Anıl
- ALFA ve UAV Attack Bronze yükleme (`src/ingestion/upload_raw.py`)
- ALFA Silver parser (`src/silver/parse_alfa.py`) — 47 sekan, merge_asof
- UAV Attack Silver parser (`src/silver/parse_uav_attack.py`) — log/topic regex fix
- Gold birleşim katmanı (`src/gold/unify.py`) — 7+3 şema, gerçek veriyle doğrulandı
- Referans Silver denemesi (`src/processing/` — ADR-004)
- Bilinen sorunları belgeledi (Ping DoS, velocity_mps)

### Yusuf
- adsb.lol realtime producer (`src/ingestion/adsblol_producer.py`)
- Kafka consumer (`src/ingestion/adsblol_consumer.py`)
- Realtime tar'ları Drive'a yükledi (`adsb_datas/` klasörü)

### Birlikte (Claude Code ile)
- FakeMinioClient test altyapısı (`src/common/fakes.py`, `tests/conftest.py`)
- 91 birim test
- Tüm mimari kararlar ve dokümantasyon

---

## 15. Bilinen Açık Sorunlar

| Sorun | Etki | Konum | Durum |
|---|---|---|---|
| `velocity_mps` ALFA'da null | Gold'da ALFA için hız yok | `parse_alfa.py` — nav_info-velocity eşleşmiyor | Belgelendi, düzeltilmedi |
| `velocity_mps` UAV Attack'te null | Gold'da UAV Attack için hız yok | `parse_uav_attack.py` — Silver'da vel_n/e/d yok | Belgelendi, düzeltilmedi |
| MinIO gerçek kurulum yok | Pipeline uçtan uca test edilmedi | docker-compose + `.env` gerekli | Lokal kurulum gerekiyor |

---

## 16. Pipeline'ı Gerçekten Çalıştırmak

### Lokal test (MinIO olmadan)
```bash
pip install -r requirements.txt

# ALFA testi (processed.zip gerekli)
python scripts/run_alfa_local.py --zip data/bronze/alfa/_input/processed.zip

# UAV Attack testi
python scripts/run_uav_attack_local.py --zip data/bronze/uav_attack/_input/UAVAttackData.zip

# Gold testi (her ikisi de çalıştırıldıktan sonra)
python scripts/run_gold_local.py
```

### MinIO ile (Docker gerekli)
```bash
cp .env.example .env          # env'i düzenle
make up                        # Docker başlat
make minio-init                # bucket oluştur

# ALFA + UAV Attack Bronze yükle
make bronze-upload-alfa INPUT=data/bronze/alfa/_input/processed.zip
make bronze-upload-attack INPUT=data/bronze/uav_attack/_input/UAVAttackData.zip

# adsb tar'ları Bronze yükle (büyük dosyalar)
make bronze-upload-adsb INPUT=data/bronze/adsblol_historical/_input/

# Silver parse
make silver-alfa
make silver-attack
make silver-adsb-hist

# Gold birleştir
make gold

# Canlı realtime (isteğe bağlı, 2 terminal)
make bronze-rt-producer        # Terminal 1: API → Kafka
make bronze-rt-consumer        # Terminal 2: Kafka → Bronze JSONL
make silver-adsb-rt            # Terminal 3: Bronze JSONL → Silver Parquet
```

---

## 17. Drive'daki Veriler (Genel Bakış)

Drive: `https://drive.google.com/drive/folders/1AWhBAf98Keg_XH8nwq8kxF6xet2rPyPq`

| Klasör | İçerik | Boyut |
|---|---|---|
| `adsb_datas/` | 7 × adsb.lol tar (2024-09, 2025-02, 2025-06, 2025-10, 2026-02, 2026-06-15, 2026-06-28) | ~21 GB |
| `ALFA/processed/processed/` | 47 sekan klasörü (`carbonZ_...`), per-topic CSV'ler | ~bölünmüş zip |
| `UAVAtackData/UAVAttackData.zip` | 767 CSV, 4 etiket, 2 koleksiyon | 717 MB |

> **Not:** tar ve zip dosyaları binary olduğu için Drive MCP üzerinden doğrudan işlenemez.
> Pipeline testi için local `data/` klasörüne indirilmesi gerekiyor.
