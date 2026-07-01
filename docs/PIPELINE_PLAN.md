# İHA Anomali Tespiti — Nihai Pipeline Planı

> **Bu doküman `docs/bronze_implementasyon_plani.md`'nin yerini alır.** Eski dosyayı sil.
> Claude Code her session'da `docs/AGENTS.md` + `docs/MEMORY.md` + bu dosyayı okumalı.

---

## Mimari özet

```
Bronze (ham, dokunulmaz)          Silver (kaynak-başına parse)       Gold (ortak şema)
┌──────────────────────┐     ┌────────────────────────────┐     ┌──────────────────┐
│ .tar / .zip / .jsonl │ ──▶ │ Parquet + unit + etiket    │ ──▶ │ 7 ortak kolon    │
│ MinIO bronze/        │     │ + provenance               │     │ tüm kaynaklar    │
│ Hiçbir dönüşüm yok  │     │ MinIO silver/              │     │ MinIO gold/      │
└──────────────────────┘     └────────────────────────────┘     └──────────────────┘
```

### Katman kuralları

| Katman | Ne yapar | Ne YAPMAZ |
|---|---|---|
| **Bronze** | Ham dosyayı MinIO'ya olduğu gibi yükler | Parse, unit dönüşümü, filtre, provenance |
| **Silver** | Kaynak-başına parse + unit + etiket + provenance | Kaynakları birleştirme |
| **Gold** | 4 kaynağı 7 ortak kolona hizalar | Kaynak-özel kolon üretme |

### Coğrafi filtre: YOK

Tüm veriler tüm dünya için saklanır. Coğrafi filtreleme (Türkiye veya başka bölge) analiz
aşamasında notebook/query seviyesinde yapılır, pipeline seviyesinde yapılmaz.

---

## Ekip ve sorumluluklar

| Kişi | Veri kaynağı | Bronze | Silver | Gold |
|---|---|---|---|---|
| **Metehan** | adsb.lol historical (tar'lar) | Ham tar → MinIO | Tar parse → Parquet | ortak |
| **Yusuf** | adsb.lol realtime (Kafka) | Producer + consumer → ham JSONL | JSONL → Parquet | ortak |
| **Anıl** | ALFA + UAV Attack | Ham zip → MinIO | Zip parse → Parquet | ortak |

Her kişi kendi bölümünü **bağımsız** takip eder. Bölümlerin birbirine bağımlılığı yok.
Sonunda Gold'da hepsi birleşir.

---

## Gold ortak şema (7 temel kolon + metadata)

4 kaynağın hepsinde ortak olan minimum kolon seti. Her Silver parser'ın çıktısı bu
kolonlara dönüştürülebilir olmalı.

| # | Kolon | Birim | Açıklama |
|---|---|---|---|
| 1 | `timestamp_utc` | Unix epoch (saniye, float) | UTC zaman damgası |
| 2 | `lat` | derece (WGS84) | Enlem |
| 3 | `lon` | derece (WGS84) | Boylam |
| 4 | `altitude_m` | metre | İrtifa (barometrik veya geometrik) |
| 5 | `velocity_mps` | m/s | Yer hızı |
| 6 | `heading_deg` | derece (0-360) | Yön/track |
| 7 | `vertical_rate_mps` | m/s | Dikey hız |

**Metadata kolonları** (ortak ama sayısal değil):

| Kolon | Açıklama |
|---|---|
| `source_type` | `adsblol_hist` / `adsblol_rt` / `alfa` / `uav_attack` |
| `source_id` | Uçak/araç kimliği (ICAO hex / sequence name / log ID) |
| `label` | Varsa: `normal` / `engine_fault` / `gps_spoofing` / `benign` vb. Yoksa `null` |

> **Not:** adsb.lol verisinde `label` her zaman `null` olacak (etiket yok). ALFA ve UAV
> Attack'te etiketler parser tarafından üretiliyor. Gold'da etiketli/etiketsiz veri
> birarada yaşayacak.

Her Silver parser'ın kendi kaynak-özel ek kolonları olabilir (`squawk`, `callsign`,
`roll_deg`, `jamming_indicator` vb.) — bunlar Silver'da kalır, Gold'a sadece yukarıdaki
7+3 kolon geçer.

---

## Ortak altyapı (Claude Code ilk bunu kursun)

### Silinecekler (Faz 0)

```
SİL: docs/bronze_implementasyon_plani.md
SİL: src/ingestion/adsblol_historical_loader.py
SİL: src/ingestion/alfa_loader.py
SİL: src/ingestion/uav_attack_loader.py
SİL: src/streaming/                              (boş klasör)
TAŞI: src/bronze2silverParsers/* → src/silver/    (sonra eski klasörü sil)
```

### Oluşturulacaklar

```
src/
├── common/
│   ├── minio_io.py       # MinIO IO (bronze upload + silver write + download)
│   └── provenance.py     # provenance kolonları (değişmez, Silver'da kullanılır)
├── ingestion/            # === BRONZE ===
│   ├── upload_raw.py     # Lokal dosyayı MinIO bronze'a değişmeden yükle
│   ├── adsblol_producer.py   # (Yusuf) adsb.lol API → Kafka
│   └── adsblol_consumer.py   # (Yusuf) Kafka → ham JSONL → MinIO bronze
├── silver/               # === SILVER ===
│   ├── parse_adsblol_historical.py  # (Metehan)
│   ├── parse_adsblol_realtime.py    # (Yusuf)
│   ├── parse_alfa.py                # (Anıl)
│   └── parse_uav_attack.py          # (Anıl)
└── gold/                 # === GOLD (sonra) ===
    └── unify.py          # 4 Silver → 7 ortak kolon
```

### Docker (docker-compose.yml)

```yaml
services:
  kafka + zookeeper   # mevcut, değişmez
  minio               # mevcut, 9000 (API) + 9001 (console)
```

İki bucket: `bronze`, `silver` (Gold eklenince `gold`). `make minio-init` ikisini de oluşturur.

### `.env.example`

```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BRONZE_BUCKET=bronze
MINIO_SILVER_BUCKET=silver
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=uav.raw.states
ADSBLOL_API_KEY=
```

### Provenance (Silver'da eklenir)

Her Silver Parquet'e şu lineage kolonları eklenir:

| Kolon | Örnek |
|---|---|
| `_source_type` | `adsblol_hist` |
| `_ingest_ts_utc` | `2026-07-01T12:00:00Z` |
| `_source_file` | `adsblol_historical/v2026.06.15.tar` |
| `_schema_version` | `silver_v1` |

---

# ══════════════════════════════════════════════════════════
# METEHAN REHBERİ — adsb.lol Historical
# ══════════════════════════════════════════════════════════

## Veri kaynağı

adsb.lol historical tar arşivleri. Her biri ~3GB, günlük release. İçinde `traces/` altında
per-uçak gzip'li JSON dosyaları var. Format detayı:
`docs/adsblo_data_format_reference.md` (proje dosyalarında mevcut).

Drive'daki ham tar'lar: `staj/proje/Tar/adsb_raw/` altında 7 adet tar (bazıları `.tar.aa` +
`.tar.ab` parçalı). Claude Code bunları Drive linkinden indirebilir.

## Adım 1: Bronze — ham tar'ları MinIO'ya yükle

`src/ingestion/upload_raw.py` ile tar dosyalarını `bronze/adsblol_historical/` altına
değişmeden yükle. Parçalı tar'lar (`.tar.aa` + `.tar.ab`) önce birleştirilir, birleşmiş
`.tar` yüklenir.

```
bronze/adsblol_historical/
├── v2026.06.15-planes-readsb-prod-0.tar
├── v2026.06.28-planes-readsb-prod-0.tar
└── ... (7 tar)
```

**Dokunma, dönüştürme, açma yok.** Sadece yükle.

## Adım 2: Silver — tar'ları parse et

Mevcut `src/bronze2silverParsers/parse_adsb_traces_from_tar_v2.py` **zaten doğru çalışıyor.**
Bu dosyayı `src/silver/parse_adsblol_historical.py`'a taşı ve şu değişiklikleri yap:

1. **Girdi:** MinIO'dan `bronze/adsblol_historical/*.tar` indir (temp'e).
2. **Parse:** Mevcut `parse_trace_bytes()` + batch mantığı aynen kalsın.
   - feet → metre, knot → m/s dönüşümleri **kalsın** (Silver'ın işi).
   - `aircraft_dict` sparse update mantığı (callsign, category, squawk) kalsın.
   - **Coğrafi filtre YOK** — tüm trace noktaları kalır.
3. **Provenance ekle:** `add_provenance(df, source_type="adsblol_hist", source_file=tar_name)`.
4. **Çıktı:** `silver/adsblol_historical/part-NNNNN.parquet` → MinIO'ya yükle.

**Silver çıktı kolonları** (mevcut parser'ın ürettiği — koru):

```
source_type, source_id (icao), timestamp_utc, lat, lon,
alt (metre), alt_geom_m, on_ground, ground_speed_ms, track_deg,
vertical_rate_ms, indicated_airspeed_ms, roll_deg,
flags_stale, flags_new_leg, ads_source_type,
registration, aircraft_type, aircraft_desc, no_reg_data,
flight_callsign, category, squawk, emergency,
nic, rc, nac_p, sil, adsb_version, label (always null)
```

## Adım 3: Test

Sahte küçük tar fixture (2-3 trace'li) ile:
- Parse doğru kolon üretiyor mu?
- Provenance kolonları dolu mu?
- MinIO'ya yazılıyor mu?

## Kabul kriterleri

- [ ] 7 ham tar MinIO `bronze/` altında, orijinal boyutunda.
- [ ] `silver/adsblol_historical/` altında Parquet part'lar, `_source_type = "adsblol_hist"`.
- [ ] Tüm dünya verisi (filtre yok).
- [ ] `label` kolonu hep `null` (adsb'de etiket yok).

---

# ══════════════════════════════════════════════════════════
# YUSUF REHBERİ — adsb.lol Realtime (Kafka)
# ══════════════════════════════════════════════════════════

## Veri kaynağı

adsb.lol canlı API. Endpoint: `https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{nm}`.
Response: `{"ac": [...]}`, her eleman bir uçak state'i. Auth şu an gerekmiyor ama ileride
olabilir — `ADSBLOL_API_KEY` env değişkeni opsiyonel header olarak desteklenmeli.

## Adım 1: Bronze — Producer (API → Kafka)

`src/ingestion/adsblol_producer.py` — mevcut hali büyük ölçüde doğru. Görevler:

- 60 sn'de bir adsb.lol'u poll et (rate limit'e takılmamak için).
- Birden fazla sorgu noktası kullan (tek daire tüm dünyayı kapsamaz). Başlangıç grid'i:
  ```python
  QUERY_POINTS = [
      (41.0, 29.0, 250),   # İstanbul bölgesi
      (39.9, 41.0, 250),   # Doğu Anadolu
      (37.0, 35.5, 250),   # Güney kıyı
      (39.0, 35.0, 250),   # Ankara / merkez
  ]
  ```
  > **UYARI:** Bu noktalar doğrulanmadı. İlk gerçek çalıştırmada log'a bak, gerekirse
  > ekle/ayarla. Sadece Türkiye değil başka bölgeler de istenebilir — kolay genişletilebilir
  > olsun.
- Her `ac[]` entry'sini Kafka topic'e yaz, key = `hex` (ICAO). Ham JSON, dönüşüm yok.
- Graceful shutdown (SIGINT/SIGTERM).

## Adım 2: Bronze — Consumer (Kafka → ham JSONL → MinIO)

`src/ingestion/adsblol_consumer.py` — **sadeleştir:**

- Kafka'dan oku, 500'lük batch'lerde JSONL olarak MinIO'ya yükle.
- Obje adı: `bronze/adsblol_realtime/YYYY/MM/DD/states-<timestamp>.jsonl`.
- **Parquet yok, provenance yok, filtre yok.** Sadece ham JSONL landing.

## Adım 3: Silver — JSONL'leri parse et

`src/silver/parse_adsblol_realtime.py` (YENİ):

1. MinIO'dan `bronze/adsblol_realtime/**/*.jsonl` objelerini listele + indir.
2. Her satır bir `ac` JSON kaydı → DataFrame'e çevir.
3. Unit dönüşümleri:
   - `alt_baro` (feet) → metre (`* 0.3048`). `"ground"` string'i → `on_ground = True`, alt = null.
   - `gs` (knot) → m/s (`* 0.5144`).
   - `baro_rate` / `geom_rate` (fpm) → m/s (`* 0.00508`).
   - `ias` / `tas` (knot) → m/s.
4. Provenance: `add_provenance(df, source_type="adsblol_rt", source_file=jsonl_name)`.
5. `write_silver(df, "adsblol_realtime")` → MinIO silver'a.

**Silver çıktı kolonları** (adsb.lol'un `ac` alanlarından — historical parser'a benzer):

```
source_type ("adsblol_rt"), source_id (hex), timestamp_utc (now),
lat, lon, alt (metre), alt_geom_m, on_ground,
ground_speed_ms, track_deg, vertical_rate_ms,
flight_callsign, category, squawk, emergency,
registration, aircraft_type, label (always null),
_source_type, _ingest_ts_utc, _source_file, _schema_version
```

## Adım 4: Test

- Producer testi: mock requests ile API response simulate.
- Consumer testi: FakeMinioClient ile ham JSONL yükleme.
- Silver testi: sahte JSONL fixture → parse → unit dönüşümleri doğru mu?

## Kabul kriterleri

- [ ] Producer 60 sn'de Kafka'ya yazıyor.
- [ ] Consumer `bronze/adsblol_realtime/` altına ham JSONL yüklüyor (başka hiçbir şey yok).
- [ ] Silver parser JSONL'leri okuyup `silver/adsblol_realtime/` altına Parquet yazıyor.
- [ ] Parquet'te unit dönüşümleri doğru, provenance dolu, `label = null`.

---

# ══════════════════════════════════════════════════════════
# ANIL REHBERİ — ALFA + UAV Attack
# ══════════════════════════════════════════════════════════

## Veri kaynağı 1: ALFA

CMU AirLab'in fault/anomaly dataset'i. 47 otonom uçuş, 8 farklı arıza tipi.
`processed/` altında per-sequence klasörler, her klasörde per-topic CSV'ler.

Dosya adı pattern'i:
```
processed/
├── carbonZ_2018-07-18-12-10-11_no_ground_truth/
│   ├── carbonZ_...-mavros-global_position-global.csv
│   ├── carbonZ_...-mavros-nav_info-roll.csv
│   └── ...
├── carbonZ_2018-07-18-15-53-31_1_engine_failure/
│   ├── carbonZ_...-failure_status-engines.csv    ← ground-truth
│   └── ...
```

**Etiket çıkarımı** (zaten `parse_alfa.py`'da doğru çalışıyor):
- Klasör adından fault tipi (`engine_failure`, `aileron_fault`, `rudder_fault` vb.).
- `failure_status-*.csv` dosyasından fault başlangıç zamanı.

## Veri kaynağı 2: UAV Attack

IEEE DataPort GPS spoofing / jamming / ping DoS dataset'i. PX4 platformları.
Klasör yapısı:
```
UAVAttackData/
├── Live GPS Spoofing and Jamming/
│   ├── Benign Flight/          → label = benign
│   ├── GPS Jamming/            → label = gps_jamming
│   └── GPS Spoofing/           → label = gps_spoofing
└── Simulated - OTU Survey/
    ├── PX4-H480-SITL/
    │   ├── Normal/             → label = benign
    │   ├── GPS Spoofing/       → label = gps_spoofing
    │   └── Ping DoS/           → label = ping_dos
    └── ... (6 platform)
```

**Etiket çıkarımı** (zaten `parse_uav_attack.py`'da doğru çalışıyor):
- En yakın klasör adından label (benign/gps_spoofing/gps_jamming/ping_dos).
- Önemli: üst klasörde "GPS Spoofing" yazsa bile alt klasör "Benign Flight" ise → benign.

## Adım 1: Bronze — ham zip'leri MinIO'ya yükle

```
bronze/alfa/ALFA.zip                        ← (veya processed.zip — ne indirildiyse)
bronze/uav_attack/UAVAttackData.zip
```

`upload_raw.py` ile, dönüşüm yok.

ALFA indirme kaynağı: KiltHub/Figshare
(`https://kilthub.cmu.edu/ndownloader/files/24098639`) veya Drive'daki kopyası.

UAV Attack: IEEE DataPort'tan kullanıcı indirmiş olmalı (login gerekiyor).

## Adım 2: Silver — ALFA parse

Mevcut `src/bronze2silverParsers/parse_alfa.py`'ı `src/silver/parse_alfa.py`'a taşı:

1. MinIO'dan `bronze/alfa/*.zip` indir (temp).
2. Mevcut parse mantığı kalsın:
   - `processed.zip` içindeki sequence klasörlerini bul.
   - Her sequence'ta `global_position` CSV'yi omurga olarak al.
   - `nav_info-roll/pitch/airspeed/velocity/yaw` ile `merge_asof` yap.
   - `failure_status-*.csv`'den etiket üret (fault başlangıç zamanı → `label` kolonu).
   - `infer_fault_from_seq_name()` ile default etiket.
3. **Provenance ekle:** `add_provenance(df, source_type="alfa", source_file="alfa/ALFA.zip")`.
4. MinIO `silver/alfa/` altına yaz.

**Silver çıktı kolonları** (mevcut parser):
```
source_type ("alfa"), source_id (sequence name), timestamp_utc,
lat, lon, alt,
roll_measured, roll_commanded, pitch_measured, pitch_commanded,
airspeed_measured, velocity_measured, yaw_measured, yaw_commanded,
label (normal / engine_fault / aileron_fault / rudder_fault / elevator_fault / unknown),
_source_type, _ingest_ts_utc, _source_file, _schema_version
```

## Adım 3: Silver — UAV Attack parse

Mevcut `src/bronze2silverParsers/parse_uav_attack.py`'ı `src/silver/parse_uav_attack.py`'a taşı:

1. MinIO'dan `bronze/uav_attack/*.zip` indir (temp).
2. Mevcut parse mantığı kalsın:
   - `vehicle_global_position` CSV'yi omurga olarak al (lat, lon, alt).
   - `vehicle_attitude` → quaternion → euler (roll, pitch, yaw).
   - `battery_status`, `vehicle_gps_position` (jamming_indicator vb.) merge.
   - En yakın klasörden etiket çıkar.
   - `time_utc_usec` varsa gerçek UTC, yoksa göreceli timestamp.
3. **Provenance ekle.**
4. MinIO `silver/uav_attack/` altına yaz.

**Silver çıktı kolonları** (mevcut parser):
```
source_type ("uav_attack"), source_id (log ID), timestamp_utc,
lat, lon, alt, roll_deg, pitch_deg, yaw_deg,
eph, epv, voltage_v, remaining, current_a,
jamming_indicator, noise_per_ms, hdop, vdop, satellites_used,
label (benign / gps_spoofing / gps_jamming / ping_dos / unknown),
timestamp_is_real_utc,
_source_type, _ingest_ts_utc, _source_file, _schema_version
```

## Adım 4: Test

- ALFA: 2-3 sahte sequence klasörü (birinde fault, birinde normal) → etiket doğru mu?
- UAV Attack: sahte log (birinde benign, birinde spoofing) → etiket doğru mu?
- İkisinde de provenance dolu mu?

## Kabul kriterleri

- [ ] `bronze/alfa/` ve `bronze/uav_attack/` altında ham zip'ler.
- [ ] `silver/alfa/` Parquet'inde etiketler doğru (engine_fault, normal vb.).
- [ ] `silver/uav_attack/` Parquet'inde etiketler doğru (benign, gps_spoofing vb.).
- [ ] Provenance kolonları her ikisinde de dolu.

---

# ══════════════════════════════════════════════════════════
# ORTAK — Gold (hep birlikte, Silver review'dan sonra)
# ══════════════════════════════════════════════════════════

> Silver review tamamlanmadan Gold'a başlanmaz.

## Gold: 4 kaynağı 7+3 kolona hizala

`src/gold/unify.py`:

1. MinIO'dan 4 Silver Parquet setini oku.
2. Her kaynak için kolon eşlemesi:

| Gold kolonu | adsblol_hist | adsblol_rt | alfa | uav_attack |
|---|---|---|---|---|
| `timestamp_utc` | `timestamp_utc` | `timestamp_utc` | `timestamp_utc` | `timestamp_utc` |
| `lat` | `lat` | `lat` | `lat` | `lat` |
| `lon` | `lon` | `lon` | `lon` | `lon` |
| `altitude_m` | `alt` | `alt` | `alt` | `alt` |
| `velocity_mps` | `ground_speed_ms` | `ground_speed_ms` | `velocity_measured` | hesapla |
| `heading_deg` | `track_deg` | `track_deg` | `yaw_measured` | `yaw_deg` |
| `vertical_rate_mps` | `vertical_rate_ms` | `vertical_rate_ms` | null | null |
| `source_type` | `source_type` | `source_type` | `source_type` | `source_type` |
| `source_id` | `source_id` | `source_id` | `source_id` | `source_id` |
| `label` | null | null | `label` | `label` |

3. `pd.concat()` ile birleştir, `gold/unified/` altına yaz.

> **İleride yeni veri seti eklenirse:** Silver parser'ı yazılır + Gold kolon eşleme tablosuna
> bir satır eklenir. Pipeline genişleyebilir.

---

## Makefile hedefleri

```makefile
# Altyapı
make up                    # docker compose up -d
make down                  # docker compose down
make test                  # pytest
make minio-init            # bronze + silver bucket oluştur

# Bronze (ham yükleme)
make bronze-upload-adsb    # tar'ları MinIO bronze'a yükle
make bronze-upload-alfa    # ALFA zip'i MinIO bronze'a yükle
make bronze-upload-attack  # UAV Attack zip'i MinIO bronze'a yükle
make bronze-rt-producer    # adsb.lol → Kafka (canlı)
make bronze-rt-consumer    # Kafka → ham JSONL → MinIO bronze (canlı)

# Silver (parse)
make silver-adsb-hist      # bronze tar → silver parquet
make silver-adsb-rt        # bronze jsonl → silver parquet
make silver-alfa           # bronze alfa zip → silver parquet
make silver-attack         # bronze uav zip → silver parquet

# Gold
make gold                  # 4 silver → unified gold parquet
```

---

## Silme / güncelleme checklist

| Dosya | Aksiyon |
|---|---|
| `docs/bronze_implementasyon_plani.md` | **SİL** |
| `src/ingestion/adsblol_historical_loader.py` | **SİL** |
| `src/ingestion/alfa_loader.py` | **SİL** |
| `src/ingestion/uav_attack_loader.py` | **SİL** |
| `src/streaming/` | **SİL** (boş) |
| `src/bronze2silverParsers/parse_adsb_traces_from_tar_v2.py` | **TAŞI** → `src/silver/parse_adsblol_historical.py` |
| `src/bronze2silverParsers/parse_alfa.py` | **TAŞI** → `src/silver/parse_alfa.py` |
| `src/bronze2silverParsers/parse_uav_attack.py` | **TAŞI** → `src/silver/parse_uav_attack.py` |
| `src/bronze2silverParsers/` | Taşıma bittikten sonra **SİL** |
| `src/common/io.py` | **YENİDEN YAZ** → `src/common/minio_io.py` |
| `src/common/bbox.py` | **SİL** (coğrafi filtre yok; analiz aşamasında yapılır) |
| `docs/decisions.md` | **ADR-003 EKLE** (Bronze=raw kararı) |
| `docs/AGENTS.md` | **GÜNCELLE** (katman kuralları, sorumluluklar) |
| `docs/bronze_schema.md` | **YENİDEN YAZ** → `docs/schema.md` (3 katman) |
| `README.md` | **GÜNCELLE** (mimari diyagram, MinIO bucket yapısı) |
| `tests/` | Eski loader testleri sil, yeni Silver testleri ekle |

---

## YAPMA listesi

- ❌ Bronze'da parse / unit dönüşümü / provenance / filtre yapma.
- ❌ Silver'da kaynakları birleştirme (o Gold).
- ❌ Gold'a Silver review'sız başlama.
- ❌ `bronze2silverParsers/`'daki transform mantığını yeniden yazma — taşı, IO sar.
- ❌ Hayali dosya adı varsayımı — parser'lar gerçek dosyalardan doğrulanmış.
- ❌ Coğrafi filtre (bbox) pipeline'a koyma — analiz aşamasında yapılır.

---

## `docs/decisions.md`'ye eklenecek ADR

```markdown
## ADR-003: Bronze = raw; parse/provenance Silver'da; coğrafi filtre yok

- Durum: Kabul edildi
- Tarih: 2026-07-01
- ADR-002'yi günceller.

Bronze katmanı ham dosyaları (orijinal .tar/.zip ve realtime ham .jsonl) MinIO'da
değiştirmeden saklar. Parquet dönüşümü, unit dönüşümü, etiket çıkarımı ve provenance
kolonları Silver katmanına taşındı. Coğrafi filtre (Türkiye bbox veya başka) pipeline
seviyesinde uygulanmaz; analiz/notebook aşamasında yapılır. Gold katmanı 4 kaynağı
7 ortak kolona (timestamp, lat, lon, altitude, velocity, heading, vertical_rate) +
3 metadata kolonuna (source_type, source_id, label) hizalar.

Sorumluluklar: adsb historical → Metehan, adsb realtime → Yusuf, ALFA + UAV Attack → Anıl.
```
