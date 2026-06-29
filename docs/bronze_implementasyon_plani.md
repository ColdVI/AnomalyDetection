# Bronze Katmanı — Codex İmplementasyon Planı

> **Amaç:** 4 veri kaynağını (adsb.lol historical, adsb.lol realtime, ALFA, UAV Attack) ham haliyle, provenance metadata ile `data/bronze/` altına indirmek ve GitHub'a pushlamak. **Silver'a geçilmeyecek** — Bronze biter, üç kişi birlikte review eder, sonra Silver.
> **Çalışma şekli:** Her faz bağımsız. Her fazın sonunda kabul kriterleri sağlanınca commit + push.

---

## 0. Bağlam ve Kritik Notlar (önce oku)

### Bu plan resmî ders planından sapıyor — farkında ol
Resmî plan **OpenSky API + MAVLink** diyor. Arkadaşının mimarisi bunları değiştirdi:
- **OpenSky → adsb.lol**: Mantıklı. OpenSky artık OAuth2 + günlük kredi limiti istiyor; adsb.lol ücretsiz, auth'suz (ODbL lisansı). Ders dokümanı zaten "lisans/network/on-prem uyumlu alternatif kullanılabilir" diyor.
- **Generic MAVLink → ALFA + UAV Attack**: Bu *senin bireysel projen* (anomali tespiti) için orijinal plandan **daha iyi**. ALFA = fault/anomaly ground-truth etiketli; UAV Attack = GPS spoofing benign/malicious etiketli. Etiketli anomali verisi olmadan Isolation Forest / Autoencoder değerlendirmesi yapılamaz.

> **Aksiyon:** Bu sapmayı Bronze review'da ekip + mentor ile açıkça konuş. Kabul edilirse `docs/decisions.md`'ye yaz (ADR — Architecture Decision Record).

### Bronze katmanının altın kuralı
Bronze = **ham veriyi olduğu gibi indir, dönüştürme yapma.** Unit dönüşümü, koordinat ölçekleme, harmonizasyon → **hepsi Silver'da**. Bronze'da sadece:
1. Veriyi satır/dosya olarak indir (parse edebilecek kadar minimal işlem).
2. **Provenance kolonları ekle** (aşağıda standart).
3. Türkiye bbox filtresi (sadece adsb kaynakları için — ALFA/UAV Attack zaten küçük ve hepsi lazım).

### Zorunlu provenance standardı (her Bronze kaydında olacak)
Her kaynak için Bronze çıktısına şu kolonlar eklenecek:

| Kolon | Açıklama | Örnek |
|---|---|---|
| `_source_type` | Kaynak kimliği | `adsblol_hist` \| `adsblol_rt` \| `alfa` \| `uav_attack` |
| `_ingest_ts_utc` | İndirme anı (UTC) | `2026-06-29T10:00:00Z` |
| `_source_file` | Orijinal dosya/URI | `2026-06-20.tar / A4B1C2.json` |
| `_schema_version` | Bronze şema versiyonu | `bronze_v1` |
| `_raw` | (opsiyonel) orijinal payload JSON | `{...}` |

Geri kalan tüm orijinal alanlar **ismi değiştirilmeden** korunur.

### Format kararı
- **Bronze tabloları:** Parquet, kaynak başına klasör (`data/bronze/<source>/`).
- `data/bronze/` ve tüm uçuş verisi `.gitignore`'a girer — **repoya veri girmez, sadece kod girer.**
- adsb.lol realtime için ham JSON da JSONL olarak landing edilir (true-Bronze), sonra Parquet'e çevrilir.

### Network notu (Codex için)
Codex/CI ortamından `adsb.lol`, `ieee-dataport.org`, `cmu.edu` erişilemeyebilir. **adsb.lol historical tar'ları, ALFA ve UAV Attack dosyaları kullanıcı tarafından önceden indirilip `data/bronze/<source>/_input/` altına konacak.** Loader'lar lokal dosyadan okur. Sadece adsb.lol *realtime* producer canlı API'ye gider (o da lokal makinede çalışır).

---

## Repo İskeleti (hedef yapı)

```
uav-platform/
├── data/                          # .gitignore'da — repoya girmez
│   ├── bronze/
│   │   ├── adsblol_historical/
│   │   │   └── _input/            # kullanıcının indirdiği .tar'lar
│   │   ├── adsblol_realtime/
│   │   │   └── _landing/          # ham JSONL
│   │   ├── alfa/
│   │   │   └── _input/            # ALFA processed CSV'leri
│   │   └── uav_attack/
│   │       └── _input/            # IEEE DataPort CSV/ULog'ları
│   ├── silver/                    # şimdilik boş
│   └── gold/                      # şimdilik boş
├── src/
│   ├── common/
│   │   ├── provenance.py          # provenance kolonu ekleme yardımcısı
│   │   ├── bbox.py                # Türkiye bbox filtresi
│   │   └── io.py                  # Parquet yazma/okuma standardı
│   ├── ingestion/
│   │   ├── adsblol_historical_loader.py
│   │   ├── adsblol_producer.py
│   │   ├── adsblol_consumer.py
│   │   ├── alfa_loader.py
│   │   └── uav_attack_loader.py
│   └── processing/                # şimdilik boş (Silver'a saklı)
├── notebooks/
│   └── bronze_validation.ipynb
├── docs/
│   ├── decisions.md               # ADR — kaynak değişikliği kararı
│   └── bronze_schema.md           # her kaynağın Bronze kolonları
├── tests/
│   ├── test_provenance.py
│   ├── test_bbox.py
│   └── test_loaders.py            # küçük sample fixture'larla
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

# FAZ 1 — Repo + Docker + Ortak Araçlar

### Hedef
İskeleti kur, Kafka'yı ayağa kaldır, ortak yardımcı modülleri yaz. Henüz veri yok.

### Codex'e verilecek görevler
1. Yukarıdaki repo iskeletini oluştur (boş `__init__.py`'ler dahil).
2. `.gitignore`: `data/`, `.env`, `*.parquet`, `__pycache__/`, `.ipynb_checkpoints/`, `*.tar`, `*.bin`, `*.tlog`, `*.ulg`.
3. `requirements.txt`: `confluent-kafka`, `requests`, `pandas`, `pyarrow`, `pymavlink`, `pyulog`, `python-dotenv`, `pytest`. (ALFA `.bag` için `rosbags` opsiyonel — şimdilik ekleme.)
4. `docker-compose.yml`: zookeeper + kafka (`confluentinc/cp-zookeeper:7.5.0`, `confluentinc/cp-kafka:7.5.0`, port 9092, `KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"`).
5. `src/common/provenance.py`:
   ```python
   def add_provenance(df, source_type, source_file, schema_version="bronze_v1"):
       """Bronze provenance kolonlarını ekler. df'yi mutate etmez, kopya döner."""
   ```
6. `src/common/bbox.py`:
   ```python
   TURKEY_BBOX = {"lat": (36.0, 42.0), "lon": (26.0, 45.0)}
   def in_turkey(lat, lon) -> bool: ...
   ```
   Null-safe olmalı (lat/lon None gelirse `False` dön).
7. `src/common/io.py`: `write_bronze(df, source_type, partition=None)` → `data/bronze/<source>/part-*.parquet` (snappy compression).
8. `Makefile`: `make up` (docker compose up -d), `make down`, `make test` (pytest), `make bronze-hist`, `make bronze-alfa`, `make bronze-attack` (loader'ları çağırır).
9. `README.md`: kurulum + mimari diyagramı (Bronze→Silver→Gold) + "şu an Bronze fazındayız" notu.
10. `docs/decisions.md`: kaynak değişikliği ADR'si (OpenSky→adsb.lol, MAVLink→ALFA+UAV Attack gerekçesi).

### Kabul kriterleri
- [ ] `make up` ile Kafka ayakta, test topic'e mesaj gidip geliyor.
- [ ] `pytest` çalışıyor; `test_provenance.py` ve `test_bbox.py` geçiyor.
- [ ] `add_provenance` ve `in_turkey` unit testlerle doğrulanmış.
- [ ] Repo GitHub'da, README + .gitignore var, `data/` ignore'lu.

### Codex prompt (kopyala-yapıştır)
> Aşağıdaki repo iskeletini oluştur: [yapıyı yapıştır]. `src/common/provenance.py`, `bbox.py`, `io.py` modüllerini verilen imzalarla yaz. Türkiye bbox = lat(36,42) lon(26,45), null-safe. `docker-compose.yml` confluent kafka 7.5.0 + zookeeper, port 9092, auto-create-topics açık. requirements.txt: confluent-kafka, requests, pandas, pyarrow, pymavlink, pyulog, python-dotenv, pytest. `.gitignore` data/ ve tüm log uzantılarını içersin. provenance ve bbox için pytest testleri yaz. Makefile target'ları: up/down/test/bronze-hist/bronze-alfa/bronze-attack.

---

# FAZ 2 — adsb.lol Historical → Bronze

### Hedef
Kullanıcının indirdiği 5 günlük tar'ları parse et, Türkiye bbox filtrele, Bronze'a yaz.

### Kritik teknik notlar
- Tar kaynağı: `github.com/adsblol/globe_history_2026` günlük release'leri. Kullanıcı bunları `data/bronze/adsblol_historical/_input/` altına koyacak.
- Tar içi yapı: gzip'lenmiş per-aircraft trace JSON'ları (`traces/<hex prefix>/<icao>.json` benzeri). Loader `member.name`'de `traces` arayacak (arkadaşın kodu doğru).
- **Trace formatı:** JSON'da top-level `icao`, `timestamp` (base epoch), ve `trace` dizisi var. Her trace elemanı bir dizi: `[saniye_offset, lat, lon, alt, gs, track, flags, vert_rate, ...]`.
  - `t[1]` = lat, `t[2]` = lon, `t[3]` = alt. Arkadaşının indexlemesi **doğru**.
  - **Null-check şart:** `t[1]`/`t[2]` `None` olabilir; `alt` "ground" string'i olabilir → bbox filtresi null-safe çağrılmalı.
  - Gerçek timestamp = `data["timestamp"] + t[0]`.

### Codex'e verilecek görevler
1. `src/ingestion/adsblol_historical_loader.py`:
   - `extract_turkey(tar_path, output_dir)`: tar'ı aç, `traces` member'larını gzip-decompress + json parse et.
   - Her aircraft için Türkiye bbox içindeki trace noktalarını satıra çevir (her satır = bir nokta: icao, gerçek_ts, lat, lon, alt, gs, track, vert_rate).
   - `add_provenance(..., source_type="adsblol_hist", source_file=f"{tar_name}/{icao}")`.
   - `write_bronze(df, "adsblol_historical", partition=tar_date)`.
2. Birden çok tar'ı işleyen CLI: `python -m src.ingestion.adsblol_historical_loader --input data/bronze/adsblol_historical/_input/`.
3. `tests/test_loaders.py`: küçük bir sahte tar fixture'ı ile bbox filtresinin çalıştığını doğrula.

### Kabul kriterleri
- [ ] 5 tar parse ediliyor, hata vermeden bitiyor.
- [ ] `data/bronze/adsblol_historical/` altında Parquet'ler oluşuyor.
- [ ] Çıktıda sadece Türkiye bbox içi noktalar var (spot-check).
- [ ] Her satırda provenance kolonları dolu.
- [ ] Null lat/lon olan trace noktaları drop edilmiş, crash yok.

---

# FAZ 3 — Kafka + adsb.lol Realtime Producer/Consumer → Bronze

### Hedef
Canlı adsb.lol API'sini poll et → Kafka topic → consumer ham JSONL + Parquet Bronze'a yazsın.

### Kritik teknik notlar
- Endpoint: `https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{nm}` (örn. lat=39, lon=35, dist=500 → Türkiye merkezli). **Not:** `v2/lat/lon/dist` deprecated olabilir, `v3` tercih edilir — Codex önce `api.adsb.lol/docs`'tan v3'ü kontrol etsin, varsa v3 kullansın.
- **Auth yok** (şu an), ama "ileride API key" notu var → kodu `ADSBLOL_API_KEY` env değişkenini opsiyonel header olarak destekleyecek şekilde yaz.
- Rate limit ~1 req/sn, dinamik. **Poll aralığı 60 sn** (ders dokümanındaki kredi-bilinçli yaklaşımla uyumlu, hem rate limit'e takılmaz).
- Response: `{"ac": [...]}` — her eleman bir uçak state'i (`hex`, `lat`, `lon`, `alt_baro`, `gs`, `track`, ...).

### Codex'e verilecek görevler
1. `src/ingestion/adsblol_producer.py`:
   - 60 sn'de bir poll, `r["ac"]` içindeki her uçağı `raw.adsblol.states` topic'ine yaz, **key = `hex`** (icao).
   - JSON encode, `producer.flush()` her döngüde.
   - Opsiyonel API key header desteği, `.env`'den.
   - Graceful shutdown (SIGTERM/SIGINT).
2. `src/ingestion/adsblol_consumer.py`:
   - `raw.adsblol.states` dinle, `auto.offset.reset=earliest`.
   - 500 mesaj biriktir → ham JSONL `_landing/`'e + provenance'lı Parquet Bronze'a yaz.
   - Buffer temizle, devam et.
3. `.env.example`: `ADSBLOL_API_KEY=` (boş), `KAFKA_BOOTSTRAP=localhost:9092`.

### Kabul kriterleri
- [ ] Producer 60 sn'de Kafka'ya yazıyor, key olarak hex kullanıyor.
- [ ] Consumer çalışıyor, 500'lük batch'lerde Parquet üretiyor.
- [ ] En az ~30 dk canlı veri toplanmış (Türkiye trafiği).
- [ ] Ham JSONL landing + Bronze Parquet ikisi de var.
- [ ] Producer durup tekrar başlayınca consumer kaldığı yerden devam ediyor.

---

# FAZ 4 — ALFA → Bronze

### Hedef
ALFA fault/anomaly verisini (ground-truth etiketli) Bronze'a indir.

### EN KRİTİK DÜZELTME (arkadaşın planında eksik)
ALFA 4 koleksiyon halinde gelir:
1. **Processed** → `.bag` + `.csv` + `.mat`, **fault tipi ve zamanının ground-truth'u ile.**
2. **Raw Bag** → `.bag`
3. **Telemetry** → `.tlog`, `.tlog.raw`, `mav.parm`
4. **Dataflash** → `.bin`, `.log`, `.param`

> **pymavlink `.bin` ve `.tlog` okur ama ROS `.bag` OKUMAZ.** Ground-truth etiketli "processed" veri `.bag`/`.csv` formatında. Bireysel projen (anomali tespiti) için **etiketler kritik**.
>
> **→ Bronze için birincil yol: "processed" koleksiyonunun CSV'lerini kullan.** Etiketler hazır, parse kolay, pymavlink gerekmez. Arkadaşının `.tlog`/`.bin` + pymavlink yolu ikincil/opsiyonel kalsın. (`.bag` istenirse `rosbags`/`bagpy` gerekir, pymavlink değil.)

### Codex'e verilecek görevler
1. `src/ingestion/alfa_loader.py` — **birincil: CSV yolu**:
   - `data/bronze/alfa/_input/` altındaki processed `*.csv`'leri oku.
   - Dosya adından `failure` tipini ve scenario'yu çıkar (örn. `carbonZ_<datetime>_<failure>.csv` → `failure` alanı). Dosya adı konvansiyonunu gerçek dosyalarla **doğrula**, varsayım yapma.
   - Her CSV'yi topic/mesaj-tipi bilgisiyle birleştir, provenance ekle (`source_type="alfa"`, `source_file=<csv adı>`, ek olarak `_alfa_failure_label`, `_alfa_scenario`).
   - Bronze Parquet'e yaz. **Ham kolon isimleri korunur, dönüşüm yok.**
2. `parse_alfa_mavlink(path)` — **ikincil/opsiyonel**: `.bin`/`.tlog` için pymavlink ile `GLOBAL_POSITION_INT`, `ATTITUDE`, `VFR_HUD` çek (arkadaşın `parse_alfa_tlog` koduna yakın). `recv_match(blocking=False)` `None` dönünce dur.
3. `tests/`: 1-2 satırlık sahte ALFA CSV fixture ile label çıkarımını test et.

### Kabul kriterleri
- [ ] Processed CSV'ler Bronze'a iniyor, `_alfa_failure_label` dolu.
- [ ] Ground-truth fault etiketleri korunmuş (anomali projesi için).
- [ ] Provenance kolonları dolu.
- [ ] (Opsiyonel) `.bin`/`.tlog` pymavlink parse'ı çalışıyor ama bocked değil.

---

# FAZ 5 — UAV Attack → Bronze

### Hedef
IEEE DataPort UAV Attack datasetini (GPS spoofing / DoS, benign+malicious) Bronze'a indir.

### Kritik teknik notlar
- İndirme: **ücretsiz IEEE hesabı login gerekir** (`ieee-dataport.org/open-access/uav-attack-dataset`). Kullanıcı dosyaları `data/bronze/uav_attack/_input/` altına koyacak.
- Orijinal format **ULog**; CSV'ler `pyulog`'un `ulog2csv` script'i ile üretiliyor. Kullanıcıda CSV varsa direkt oku; sadece `.ulg` varsa önce `ulog2csv` ile çevir.
- Platformlar: Holybro S500, 3DR IRIS, Yuneec H480, DeltaQuad VTOL, Standard Tailsitter/Plane.
- **benign vs malicious** ayrımı dosya/klasör adından gelir — `f.split("_")[0]` platform tahmini **gerçek dosya adlarıyla doğrulanmalı**, körlemesine güvenme.

### Codex'e verilecek görevler
1. `src/ingestion/uav_attack_loader.py`:
   - `_input/` altındaki `.csv`'leri tara (yoksa `.ulg`'leri `pyulog ulog2csv` ile çevir).
   - Dosya/klasör adından `platform` ve `label` (`benign`/`malicious`) çıkar — konvansiyonu önce listeleyip doğrula.
   - Provenance ekle (`source_type="uav_attack"`, `_attack_platform`, `_attack_label`).
   - Bronze Parquet'e yaz, ham kolonlar korunur.
2. `tests/`: sahte CSV ile platform+label çıkarımı testi.

### Kabul kriterleri
- [ ] CSV'ler Bronze'a iniyor.
- [ ] `_attack_label` (benign/malicious) ve `_attack_platform` doğru atanmış.
- [ ] (Gerekirse) ULog→CSV çevirimi çalışıyor.
- [ ] Provenance dolu.

---

# FAZ 6 — Bronze Doğrulama + GitHub Teslimi

### Hedef
4 kaynağın Bronze'unu doğrula, dokümante et, pushla. Üçünüzün review edeceği temel bu.

### Codex'e verilecek görevler
1. `notebooks/bronze_validation.ipynb`:
   - Her kaynak için: satır sayısı, null oranı (kolon bazında), zaman aralığı, kolon listesi.
   - adsb kaynakları için Türkiye coverage haritası (Folium, `tiles=None` veya OSM — ders dokümanı offline istiyor).
   - ALFA + UAV Attack için etiket dağılımı (kaç benign / kaç malicious / hangi fault tipleri).
2. `docs/bronze_schema.md`: her kaynağın Bronze kolonları + provenance standardı tablosu.
3. `docs/decisions.md` güncel (kaynak sapması ADR'si).
4. Son commit + push, PR aç.

### Bronze "DONE" tanımı (review checklist)
- [ ] 4 kaynak da `data/bronze/<source>/` altında Parquet üretiyor.
- [ ] Her kayıtta provenance kolonları dolu, `_source_type` doğru.
- [ ] Hiçbir kaynakta unit dönüşümü / harmonizasyon yapılmamış (o Silver'ın işi).
- [ ] adsb kaynakları Türkiye bbox ile filtrelenmiş; ALFA/UAV Attack tam.
- [ ] ALFA ground-truth etiketleri + UAV Attack benign/malicious etiketleri korunmuş.
- [ ] `bronze_validation.ipynb` çalışıyor, null/coverage/etiket özetleri çıkıyor.
- [ ] `docs/bronze_schema.md` + `docs/decisions.md` yazılmış.
- [ ] Tüm loader'ların pytest testleri geçiyor.
- [ ] `data/` repoya **girmemiş**, sadece kod pushlanmış.
- [ ] README'de "Bronze tamamlandı, Silver review bekliyor" notu.

---

## Codex'e Verme Stratejisi (öneri)
1. **Faz 1'i tek başına ver, bitir, pushla.** İskelet + ortak araçlar sağlam olmadan loader yazdırma.
2. Sonra her loader fazını **ayrı ayrı** ver (Faz 2 → 3 → 4 → 5). Her birinden sonra kabul kriterlerini kendin kontrol et, commit at.
3. Faz 6'yı en sona bırak — doğrulama notebook'u 4 kaynak da bittikten sonra anlamlı.
4. Her faz prompt'una şunu ekle: *"Bronze prensibi: ham veriyi koru, dönüşüm yapma, sadece provenance ekle. `src/common/`'daki mevcut yardımcıları kullan, tekrar yazma."*

## Silver'a Geçmeden Önce (sonraki aşama — şimdi DEĞİL)
Üçünüz Bronze'u review edince şunları netleştirin: ortak Gold şeması (`event_id, source_type, platform_id, timestamp_utc, lat, lon, altitude_m, speed_mps, heading_deg, vertical_rate_mps, on_ground, label_available, label_type, quality_score, extra`), her kaynağın Silver dönüşüm kuralları (adsb.lol: ts→UTC, onground filtre; ALFA: lat/lon×1e-7, alt/1000, rad→deg; UAV Attack: ts normalize, benign/malicious flag). Bunlar Bronze review çıktısına göre kesinleşecek.
