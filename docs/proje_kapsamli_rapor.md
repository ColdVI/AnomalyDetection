# İHA/ADS-B Uçuş Verisi Analiz Platformu — Grup Projesi Süreç Raporu (Detaylı)

Bu rapor, GRUP (ortak) projesinin veri kaynağı seçiminden bugünkü son haline kadar geçen
süreci kronolojik olarak, kod düzeyinde detayla anlatır: ne yapıldığını, nasıl yapıldığını,
neden o yolun seçildiğini, karşılaşılan zorlukları ve alınmayan alternatif yolları kapsar.
Sonda, sunumda sorulması muhtemel sorulara hazır cevaplar içeren ayrı bir bölüm var.

Kod blokları çoğunlukla **gerçek repo kodundan alınmış** (bazı yerlerde okunabilirlik için
kısaltılmış) — tamamen uydurma/sembolik değil.

---

## FAZ 0 — Uçtan Uca Pipeline: İki Ayrı Veri Yolu, Tek Ortak Gold Çıkışı

Projede birbirinden **tamamen bağımsız çalışan iki veri yolu** var: biri **hazır/statik veri
setleri** (adsb.lol tarihsel arşiv + ALFA + UAV Attack) için, diğeri **gerçek zamanlı canlı
akış** (adsb.lol live feed) için. İkisi de sonunda aynı yere — Gold katmanına — akar, ama
tetiklenme biçimleri, sıklıkları ve kod yolları tamamen farklı. Bu bölüm ikisini de baştan
sona, hangi kodun hangi anda çalıştığını göstererek anlatıyor.

```
              HAZIR VERI SETI YOLU                        GERCEK ZAMANLI YOL
              (elle / talep uzerine)                      (surekli + gunluk)
              ---------------------                       -------------------
adsb.lol tarihsel .tar arsivleri              adsb.lol "world" REST endpoint (her 60sn)
ALFA / UAV Attack .zip                                     |
        |                                                  v
        v                                    uav_producer.py --> Kafka "uav.flights"
scripts/upload_bronze_all.py                               |
        |                                     +-------------+--------------+
        v                                     |                             |
BRONZE (ham .tar/.zip, MinIO)      dashboard_consumer.py         minio_archiver.py
        |                          (group=dashboard-consumer)    (group=minio-archiver)
        v                          -> Redis (canli durum)        -> Bronze JSONL:
python -m src.silver               -> InfluxDB "uav-history"        adsblol_realtime/_landing/
  .parse_adsblol_historical           (batch)                       states-<ts>.jsonl
  (tar-bazli checkpoint'li)                                         (500 mesaj VEYA 60sn'de flush)
        |                                                           |
        v                                                           v (HER GUN saat 17:00'de)
SILVER "adsblol_historical"                            python -m src.silver.parse_adsblol_realtime
  (Parquet, birim donusumlu,                                        --daily-at 17:00
   is_military etiketli)                                            |
        |                                                            v
        |                                            SILVER "adsblol_realtime" (Parquet)
        |                                            + basarili yazimdan SONRA Bronze
        |                                              JSONL silinir (yaz-sonra-sil)
        |                                            + heartbeat: data/state/
        |                                              silver_realtime_heartbeat.txt
        |                                                            |
        +---------------------------+-------------------------------+
                                     v
                    python -m src.gold.unify   (ELLE calistirilir -- otomatik degil!)
                    stream_unify(): varsayilan modda once eski Gold'u siler
                    (clear_gold_before_unify), sonra TUM Silver source_type'larini
                    (adsblol_historical, adsblol_realtime, alfa, uav_attack) 7+3 ortak
                    semaya donusturup GOLD "unified/<source_type>/" altina yazar
                    (2026-07-18: source_type bazinda partition + --refresh-only ile
                    TEK bir source_type'i, digerlerine dokunmadan, kismi yenileme)
                                     |
                                     v
                    GOLD (2.76 milyar satir, 6.861 parca, unified/<source_type>/ altinda)
                                     |
              +----------------------+----------------------+
              v                      v                       v
  individual/metehan_geo/     team_dashboard/api.py    src/ml/build_features.py
  build_flight_density.py     (Silver+Gold'u DOGRUDAN   (zengin, kaynak-ozgu
  (yogunluk/askeri harita,     okur, export icin --      sema -- ML egitimi
  kendi pickle checkpoint'i)   density'den BAGIMSIZ)      icin ayri Gold cikisi)
```

### 0.1 Hazır (Statik) Veri Seti Pipeline'ı — Adım Adım

Bu yol **tetiklenme bazlı**: bir zamanlayıcı/cron YOK, sadece yeni veri eklendiğinde
(yeni tar indirildiğinde, ya da ALFA/UAV Attack ilk kez işlendiğinde) elle çalıştırılıyor.

1. **İndirme:** Tar dosyaları (`v2025.08.15-planes-readsb-prod-0.tar` gibi) elle indirilip
   `data/bronze/adsblol_historical/_input/` klasörüne konur.
2. **Bronze'a yükleme:** `scripts/upload_bronze_all.py` — idempotent (`_already_uploaded()`
   kontrolü ile daha önce yüklenmiş dosyayı atlar), her tarı MinIO Bronze'a ham `.tar` olarak
   yazar, başarılı yüklemeden sonra yerel kopyayı siler.
3. **Silver'a parse:** `python -m src.silver.parse_adsblol_historical` (gerçek CLI bayrakları:
   `--local-tar`, `--bronze-prefix`, `--batch-size`, `--fresh`). Her tar için: gzip açma, JSON
   parse, `dbFlags` bitfield'inden `is_military` çıkarımı, feet→m/knots→m/s/fpm→m/s birim
   dönüşümü. Tar-bazlı checkpoint (`data/state/silver_historical_checkpoint.json`) sayesinde
   kesinti olursa SADECE yarım kalan tar yeniden işlenir, tamamlanmış tarlar atlanır.
   - **ALFA / UAV Attack:** Bu ikisi artık `archive/2026-07-10_legacy_non_adsb_ml/src/silver/`
     altında, yani ana pipeline'ın PARÇASI değil, arşivlenmiş/legacy — proje ALFA/UAV Attack'i
     bir kez işleyip Gold'a soktuktan sonra (bkz. FAZ 1.1) sürekli yeniden çalıştırılan bir yol
     olarak kullanılmıyor. `parse_alfa.py`/`parse_uav_attack.py`'nin kendi `main()`'i sadece
     `--bronze-object`/`--local-out` alıyor, checkpoint YOK — her çalıştırmada kendi
     source_type'ını (`alfa`/`uav_attack`) tamamen silip sıfırdan yazıyor (küçük, tek seferlik
     veri setleri için checkpoint gereksiz karmaşıklık olurdu).
4. **Gold'a birleştirme:** `python -m src.gold.unify` — **elle çalıştırılır**, otomatik/zamanlı
   DEĞİLDİR. Varsayılan (tam) modda `stream_unify()` önce `clear_gold_before_unify()` ile önceki
   Gold çıktısını TAMAMEN siler, sonra `COLUMN_MAPS`'teki HER source_type için (adsblol_historical,
   adsblol_realtime, alfa, uav_attack) Silver'daki tüm parçaları tek tek okuyup 7+3 ortak şemaya
   çevirip Gold'a yazar (streaming — RAM'e toplu yüklemeden). **2026-07-18 güncellemesi:** her
   parça artık `unified/<source_type>/` altına (source_type'a göre partition'lı) yazılıyor;
   `--refresh-only adsblol_realtime` gibi bir bayrakla SADECE tek bir source_type'ın Gold
   parçaları silinip yeniden yazılabiliyor, diğer ~2.76 milyar satırlık tarihsel veri hiç
   dokunulmadan kalıyor (günlük realtime "catch-up" için eklendi).
5. **Downstream (harita/analiz):** `individual/metehan_geo/build_flight_density.py` (askeri/
   sivil yoğunluk haritası, kendi pickle checkpoint'iyle), `individual/metehan_geo_country`
   (ülke bazlı analiz), `team_dashboard/api.py` (Silver+Gold'u export için doğrudan okur, bu
   ikisi de MANUEL tetiklenir).

**Neden bu yol günlük DEĞİL:** Statik/hazır veri setleri doğası gereği sabit — yeni bir tar
indirilmediği sürece işlenecek yeni veri yok. Bu yüzden zamanlayıcı yerine "yeni veri geldiğinde
elle tetikle" modeli seçildi; gereksiz günlük yeniden-işleme (zaten değişmemiş veriyi tekrar
tekrar okumak) CPU/zaman israfı olurdu.

### 0.2 Gerçek Zamanlı Pipeline'ı — Adım Adım (Günlük Dönüşüm Dahil)

Bu yol **sürekli** (canlı akış) + **günlük** (Silver dönüşümü) iki farklı ritimde çalışıyor.

1. **Canlı toplama (sürekli, 60sn'de bir):** `Dashboard/codes/uav_producer.py` — adsb.lol'ün
   "world" modunda (yarıçap `DEFAULT_WORLD_RADIUS_NM = 12000`, tüm dünyayı kapsayacak şekilde)
   tüm uçakları sorgular, her uçağı Kafka `uav.flights` topic'ine `icao24` key'iyle yazar.
2. **İki bağımsız tüketici (sürekli, aynı topic'i FARKLI `group.id` ile okur):**
   - `dashboard_consumer.py` (`group.id=dashboard-consumer`) — `uav.flights` + `uav.alerts`'i
     dinler, canlı durumu Redis'e, batch halinde InfluxDB `uav-history` bucket'ına yazar.
     Bu, dashboard'daki canlı harita/tablo görünümünü besler.
   - `minio_archiver.py` (`group.id=minio-archiver`) — aynı `uav.flights`'i BAĞIMSIZ olarak
     dinler, ham mesajları JSONL satırları halinde Bronze'a yazar:
     `bronze/adsblol_realtime/_landing/states-<UTC-zaman-damgasi>.jsonl`. Yazma tetikleyicisi:
     **500 mesaj birikince VEYA 60 saniye geçince** (hangisi önce gerçekleşirse) — bu, hem
     çok küçük dosya sayısının patlamasını hem de veri kaybını (uzun süre bekleyip flush
     etmemeyi) önlüyor.
3. **GÜNLÜK dönüşüm — Bronze JSONL → Silver Parquet (asıl "günlük" adım budur):**
   `python -m src.silver.parse_adsblol_realtime --daily-at 17:00` (Windows Görev
   Zamanlayıcı'sında oturum açılışında `scripts/start_silver_realtime_loop.bat` ile
   otomatik başlatılacak şekilde kurulu):
   - **Başlangıçta hemen bir "catch-up" geçişi** yapılır (o ana kadar Bronze'da birikmiş NE
     VARSA işlenir) — sonra döngü, `_seconds_until("17:00")` hesabıyla her gün YEREL saat
     17:00'i hedefler (PC ne zaman açık olursa olsun, sabit bir sayaç yerine gerçek saat
     hedeflenir — PC kapalıyken sayaç donmaz).
   - Bronze'daki `adsblol_realtime/_landing/` altındaki TÜM JSONL dosyaları okunur, sonuç
     Silver'a `adsblol_realtime` source_type'ı olarak Parquet yazılır. **2026-07-18
     düzeltmesi (MetehanSarikaya):** bu modül eskiden artık var olmayan
     `src/ingestion/adsblol_producer.py`'nin ham adsb.lol şemasını (`hex`, `alt_baro`, `gs`,
     `dbFlags`...) bekliyordu ve kendi içinde feet→m/knots→m/s/fpm→m/s birim dönüşümü
     yapıyordu; ama gerçek üretici `Dashboard/codes/uav_producer.py` Kafka'ya yazmadan ÖNCE
     `_normalize_common()` ile alanları zaten yeniden adlandırıp (`icao24`, `alt`,
     `velocity`, `is_military`...) SI birimine çeviriyor. Bu isim uyuşmazlığı yüzünden
     `source_id`/`alt`/`ground_speed_ms`/`vertical_rate_ms`/`flight_callsign` SESSİZCE hep
     `None`, `is_military`/`on_ground` SESSİZCE hep `False` kalıyordu — düzeltme sonrası artık
     hiçbir birim dönüşümü yapılmıyor (değerler zaten SI olarak geliyor), alanlar gerçek
     üreticinin şemasına göre okunuyor. Bu tarihten ÖNCE toplanmış realtime Silver/Gold
     verileri bu hatayı taşıyordu; Bronze'un "yaz-sonra-sil" deseni yüzünden geriye dönük
     düzeltilemedi, sadece bu tarihten SONRA toplanan veri doğru.
   - **Yaz-sonra-sil deseni:** Bir Bronze JSONL dosyası, karşılık gelen Silver Parquet yazımı
     BAŞARILI olduktan SONRA silinir (`_delete_processed`) — yazma yarıda kesilirse Bronze
     dosyası öylece kalır, bir sonraki koşuda tekrar denenir (idempotent, crash-safe).
   - Her başarılı koşuda `data/state/silver_realtime_heartbeat.txt` dosyasına zaman damgası
     yazılır. Bu dosyanın GÜNCEL kalması, döngünün sağlıklı çalıştığının; ESKİMESİ ise
     döngünün sessizce durduğunun (ör. bir istisna fırlatıp döngünün kendisinin çökmesi,
     ya da PC'nin uzun süre kapalı kalması) dış-gözlemlenebilir işareti.
4. **Gold'a yansıma — OTOMATİK DEĞİL (önemli, açık bir mimari nokta):** `COLUMN_MAPS`
   içinde `adsblol_realtime` için bir eşleme HAZIR olsa da, hiçbir zamanlayıcı/script
   `stream_unify()`'i otomatik tetiklemiyor. Yani gerçek zamanlı veri Silver'da HER GÜN
   güncelleniyor, ama bu güncel veri Gold'a (ve dolayısıyla haritalara/export'lara) ancak
   `python -m src.gold.unify` **elle** yeniden çalıştırıldığında yansıyor. Bu, mevcut
   mimarinin bilinçli olarak basit tutulmuş ama tam otomatik olmayan bir noktası.

**Bu oturumda gerçekte yaşanan somut örnek:** Zamanlanmış günlük görev bir noktada (kök neden
tam doğrulanamadı, muhtemelen bir PC/oturum kapanması) 7 gün boyunca sessizce durmuş, bu süre
zarfında Bronze `_landing/` altında ~4.300 civarı işlenmemiş JSONL dosyası birikmişti —
heartbeat dosyasının 7 gün önceki bir tarihte kalmış olması bu durumu tespit etmeyi sağladı.
Host Python'unun (Windows Smart App Control kısıtlaması nedeniyle) doğrudan çalıştırılamadığı
bir anda, aynı işi yapan hafif bir Docker konteyneri (`iha-dashboard:local` imajı + `pyarrow`
kurulumu üzerinden) tek seferlik bir "catch-up" çalıştırması olarak devreye alınıp birikmiş
günler işlendi ve günlük döngü yeniden etkinleştirildi.

### 0.3 İki Yolun Karşılaştırması (Özet Tablo)

| | Hazır Veri Seti (adsb.lol tarihsel + ALFA/UAV Attack) | Gerçek Zamanlı (adsb.lol live) |
|---|---|---|
| Tetiklenme | Elle, yeni veri eklendiğinde | Sürekli toplama + günlük (17:00) Silver dönüşümü |
| Bronze girişi | Tam `.tar`/`.zip` dosyaları | Küçük JSONL parçaları (500 msg/60sn) |
| Checkpoint | Tar-bazlı JSON checkpoint (historical); yok (ALFA/UAV Attack, tek seferlik) | Yok (dosya-bazlı yaz-sonra-sil zaten idempotent) |
| Silver çıktısı | `adsblol_historical`, `alfa`, `uav_attack` | `adsblol_realtime` |
| Gold'a yansıma | Elle (`python -m src.gold.unify`) | Elle (`python -m src.gold.unify`) — realtime Silver güncel olsa da Gold'a OTOMATİK gitmiyor |
| Sağlık göstergesi | Checkpoint dosyasındaki `completed_tars` sayısı | `silver_realtime_heartbeat.txt`'in güncelliği |

---

## FAZ 1 — Ortak Altyapı: Veri Kaynağı, Bronze/Silver/Gold, Kafka, Docker

### 1.1 Veri Kaynağı Kararı: OpenSky yerine adsb.lol

**Ne yaptık:** Proje planı OpenSky Network API + genel MAVLink log örneklerini (ardupilot.org)
öneriyordu. Bunun yerine `adsb.lol` (hem günlük tam-ağ tarihsel arşiv hem gerçek zamanlı REST
feed) ve MAVLink yerine ALFA + UAV Attack açık veri setleri kullanıldı.

**Neden bu yolu seçtik:** OpenSky'ın public API'si kimlik doğrulama gerektiriyor ve günlük
sorgu/kredi limiti var — hem TB'larca tarihsel arşiv çekmeye hem sürekli (60sn'de bir) canlı
sorgu atmaya birlikte yetmiyordu. `adsb.lol` ise:
- Kimlik doğrulama gerektirmiyor, günlük kota yok.
- Hem `.../v2/lat/lon/dist` gibi bir canlı REST endpoint hem `globe_history_*` adlı günlük tam
  ağ trace arşivleri (tar dosyaları) sunuyor — tarihsel VE gerçek zamanlı ihtiyacı TEK
  kaynaktan karşılıyor.

ALFA ve UAV Attack seçimi de benzer bir pragmatizmle: ALFA gerçek arıza ground-truth'u, UAV
Attack ise iyi huylu/kötü niyetli saldırı etiketleri taşıdığından, ileride yapılacak anomali
tespiti çalışması nicel olarak (precision/recall) ölçülebilir hale geliyor — ham MAVLink
log'unda böyle bir etiket yok.

```
# docs/decisions.md ADR-001 (2026-06-29) -- birebir alinti
OpenSky yerine `adsb.lol`; generic MAVLink örnekleri yerine ALFA ve UAV Attack veri
setleri kullanılacaktır. `adsb.lol` kimlik doğrulama ve günlük kredi yükünü azaltırken
Türkiye hava trafiğini hem tarihsel hem gerçek zamanlı sağlar. ALFA fault ground-truth,
UAV Attack ise benign/malicious saldırı etiketleri sunduğundan sonraki anomali tespiti
çalışmasının ölçülebilir olmasını sağlar.

ALFA processed CSV dosyaları ground-truth için birincil Bronze girdisidir. `.bin` ve
`.tlog` dosyalarının `pymavlink` ile ayrıştırılması opsiyonel ek yoldur; ROS `.bag`
dosyaları `pymavlink` ile okunmayacaktır.
```

**Neden MAVLink tamamen atılmadı, "opsiyonel" bırakıldı:** İleride ham otopilot log'u
gerekirse (ör. IMU/battery/GPS-spoofing residual feature'ları) diye `pymavlink` tabanlı bir
generic okuyucu (`archive/.../parse_generic.py`) referans olarak tutuldu, ama aktif hatta
hiç kablolanmadı — YAGNI ("You Aren't Gonna Need It") prensibi: ihtiyaç doğmadan kod
bakım yükü eklemek istenmedi.

### 1.2 Bronze / Silver / Gold Mimarisi (Medallion Architecture)

**Ne yaptık:** Üç katmanlı bir pipeline:

```
adsb.lol (tar arsivleri + canli feed)  ALFA / UAV Attack (zip)
              |                                |
              v                                v
        ============  BRONZE (MinIO, ham, degismemis)  ============
              |
              v  (parse_adsblol_historical.py / parse_alfa.py / parse_uav_attack.py)
        ============  SILVER (MinIO, Parquet, birim donusumlu, provenance'li)  ============
              |
              v  (src/gold/unify.py -- COLUMN_MAPS ile 7+3 semaya hizala)
        ============  GOLD (MinIO, birlesik, "unified/" prefix)  ============
```

- **Bronze**: ham dosyalar (orijinal `.tar`/`.zip`, gerçek zamanlı ham `.jsonl`) hiçbir
  dönüşüm yapılmadan saklanır.
- **Silver**: birim dönüşümü (feet→m, knots→m/s, fpm→m/s), etiket çıkarımı, provenance
  kolonları eklenmiş Parquet.
- **Gold**: tüm kaynak tiplerinin ortak 7+3 şemaya (`COLUMN_MAPS`) hizalandığı birleşik
  veri seti.

**Neden Bronze/Silver/Gold (üç katman), tek bir "temiz tablo" değil:**
1. **Geri dönülebilirlik** — Silver/Gold'un parse mantığı hatalıysa (nitekim iki kez oldu,
   bkz. aşağı), Bronze'daki ham veri bozulmadığı için sadece parser'ı düzeltip YENİDEN
   üretmek yeterli; ham veriyi tekrar indirmeye gerek yok.
2. **Sorumluluk ayrımı** — birim dönüşümü/etiketleme (Silver) ile şema-birleştirme (Gold)
   FARKLI kararlar; ör. Gold'un şeması değişse (yeni bir ortak kolon eklense) Silver'a
   dokunmadan sadece `COLUMN_MAPS` güncellenir.
3. **Test edilebilirlik** — her katman ayrı ayrı, küçük girdilerle test edilebilir
   (`FakeMinioClient` ile, gerçek MinIO sunucusu olmadan).

**Nasıl yaptık — MinIO neden seçildi (gerçek dosya sistemi/Postgres değil):** Bronze başta
yerel diske (`data/bronze/`) yazıyordu (Faz 1); ekip kararıyla MinIO'ya (S3-uyumlu object
store) taşındı çünkü:
- Nesne depolama, TB'larca ham/parquet dosya için doğal bir eşleşme (dosya sistemi
  quota/inode sorunlarına takılmadan).
- S3 API'si sayesinde ileride gerçek AWS S3'e geçiş sadece endpoint/credential değişikliği.
- Docker Compose ile tek komutla ayağa kalkıyor, ekip üyeleri arasında ortak/taşınabilir.

```python
# src/common/minio_io.py -- katmanlar arasi PAYLASILAN tek IO katmani
class ObjectStoreClient(Protocol):
    """Gercek minio.Minio veya test fake'i -- ayni arayuz."""
    def bucket_exists(self, bucket_name: str) -> bool: ...
    def put_object(self, bucket_name, object_name, data, length, content_type="..."): ...
    def list_objects(self, bucket_name, prefix=None, recursive=False): ...
    def get_object(self, bucket_name, object_name): ...
    def remove_object(self, bucket_name, object_name) -> None: ...


def _object_name(source_type: str, partition: str | None) -> str:
    # Zaman damgasi + rastgele uuid -- coklu parca yazan islemler (Silver/Gold
    # streaming) CAKISMADAN, sirayla artan isimler uretir.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"part-{timestamp}-{uuid4().hex[:8]}.parquet"
    return f"{source_type}/{partition}/{filename}" if partition else f"{source_type}/{filename}"


def write_silver(df: pd.DataFrame, source_type: str, *, client=None) -> str:
    """Bir DataFrame'i Silver bucket'ina TEK, degismez (immutable) Parquet
    parcasi olarak yazar; s3:// URI dondurur. Immutable olmasi onemli --
    hicbir yazma islemi baska bir yazmayla YARIS DURUMUNA girmez, her cagri
    kendi benzersiz dosyasini yaratir."""
    ...
```

**Neden immutable (değiştirilemez) parça dosyaları, tek bir büyük dosya/veritabanı satırı
değil:** Çoklu işlem (Silver parser'lar paralel/ardışık, Gold streaming) aynı anda yazarken
kilit/yarış durumu (race condition) olmasın diye her yazma kendi benzersiz dosyasını üretir.
Bunun bedeli: silme/temizlik "bir prefix altındaki TÜM dosyaları sil" şeklinde toplu yapılıyor
(`delete_layer_objects`) — tek satır güncelleme yok, sadece ekle/toplu-sil var.

```python
# src/common/minio_io.py -- delete_layer_objects: bir onceki calismanin
# ciktisini TOPLU temizler (Gold rerun'larinin cift saymamasi icin kritik)
def delete_layer_objects(client, bucket: str, source_type: str) -> int:
    object_names = list_layer_objects(client, bucket, source_type)
    for name in object_names:
        client.remove_object(bucket, name)
    return len(object_names)
```

**Test stratejisi — `FakeMinioClient`:** Gerçek MinIO sunucusu OLMADAN test yazılabilsin diye,
`ObjectStoreClient` bir `Protocol` (yapısal tip) olarak tanımlandı — `src/common/fakes.py`
içindeki `FakeMinioClient`, bellekte (in-memory dict) aynı arayüzü uygular. Testler ve hatta
`scripts/run_alfa_local.py` gibi Docker gerektirmeyen yardımcı script'ler bu sahte
istemciyi enjekte ederek (`client=FakeMinioClient()`) gerçek ağ/disk I/O'suna hiç dokunmadan
çalışabiliyor — CI'da veya hızlı local geliştirmede Docker ayağa kaldırmaya gerek yok.

### 1.3 Silver Katmanı: Parse, Birim Dönüşümü, Provenance

**Ne yaptık:** İki parser tipi:
1. **Kaynak-özel** (`parse_adsblol_historical.py`, `parse_alfa.py`, `parse_uav_attack.py`) —
   domain-spesifik birim dönüşümü ve etiket çıkarımı yapar.
2. **Generic** (`archive/.../parse_generic.py`) — dosya formatını otomatik algılayıp
   (CSV/JSON/JSONL, zip/tar içi dahil) sadece provenance ekleyerek Parquet'e çevirir; yeni
   bir veri seti eklemek için genelde bu yeterli, domain dönüşüm gerekirse üstüne özel
   parser yazılır.

```python
# src/silver/parse_adsblol_historical.py -- parse_trace_bytes, birim donusumu
def parse_trace_bytes(raw: bytes) -> pd.DataFrame:
    data = json.loads(gzip.decompress(raw))
    icao = data.get("icao")
    file_ts = data.get("timestamp")
    trace = data.get("trace", [])

    # dbFlags bitfield'inin 1. biti askeri ucak demek -- dosya-seviyesinde
    # sabit, trace icindeki HER satir icin gecerli.
    is_military = bool(int(data.get("dbFlags", 0) or 0) & 1)

    rows = []
    for row in trace:
        rec = dict(zip(TRACE_COLS, row))
        alt_raw = rec["alt_raw"]
        on_ground = alt_raw == "ground"
        alt_m = None if (on_ground or alt_raw is None) else round(float(alt_raw) * 0.3048, 1)  # feet -> m
        rows.append({
            "source_id": icao,
            "timestamp_utc": file_ts + rec["t_offset"],
            "lat": rec["lat"], "lon": rec["lon"],
            "alt": alt_m,
            "on_ground": on_ground,
            "ground_speed_ms": round(float(rec["gs"]) * 0.5144, 2) if rec["gs"] is not None else None,  # knots -> m/s
            "vertical_rate_ms": round(float(rec["vrate"]) * 0.00508, 3) if rec["vrate"] is not None else None,  # fpm -> m/s
            "is_military": is_military,
        })
    return pd.DataFrame(rows)
```

**Karşılaşılan zorluk — UAV Attack'in regex bug'ı:** `split_log_and_topic()` fonksiyonu,
gerçek `UAVAttackData.zip`'teki (683.9MB, 767 CSV) log_id öneklerinin standart olmadığını
(bazıları kendi içinde alt çizgi taşıyor: `log_12_2020-8-2-14-18-24_...`,
`ace-benign-log_0_...`, `001-2021-01-27-09-08-37-708_...`) hesaba katmıyordu:

```python
# ONCEKI (kirik) regex -- soldan-saga EN ERKEN alt cizgi eslesmesini alir
BROKEN_PATTERN = r"_([a-z0-9_]+?)_(\d+)\.csv$"
# "log_12_2020-8-2-14-18-24_nav_info-altitude.csv" icin bu, topic adini
# YANLIS yerden (ilk alt cizgiden) bolup "12" gibi anlamsiz bir sonuc veriyordu.

# DUZELTME -- bilinen 5 topic adina ANKRAJLI tam eslesme (genel regex yerine
# kapali/sabit liste -- topic sayisi az ve sabit oldugu icin daha guvenilir)
KNOWN_TOPICS = ["nav_info", "attitude", "battery", "gps_info", "actuator_output"]
pattern = re.compile(rf"_({'|'.join(KNOWN_TOPICS)})\.csv$")
```

**Neden genel regex yerine kapalı liste:** Topic sayısı az (5) ve sabit olduğundan, "daha
akıllı" bir regex yazıp yeni bir edge-case'te tekrar kırılma riskini almak yerine, bilinen
isimlere ankrajlı tam eşleşme tercih edildi — basit ve doğrulanabilir.

### 1.4 Gold Katmanı: 7+3 Ortak Şema, Streaming Birleştirme

```python
# src/gold/unify.py -- gercek kod (2026-07-18 guncellemesi, MetehanSarikaya:
# source_type bazinda partition + kismi (partial) yenileme destegi eklendi)
GOLD_COLUMNS = ["timestamp_utc", "lat", "lon", "altitude_m", "velocity_mps",
                "heading_deg", "vertical_rate_mps", "source_type", "source_id", "label"]

def stream_unify(client, *, silver_bucket=None, source_types=tuple(COLUMN_MAPS),
                  gold_bucket=None, refresh_only=None) -> int:
    """Her Silver Parquet dosyasini TEK TEK okuyup 7+3 semaya cevirir ve HER
    dosya icin bir Gold parcasi yazar -- >64M satirlik veri setleri icin
    hepsini RAM'e yuklemeden calisir (streaming). Her parca artik
    unified/<source_type>/ altina yaziliyor (partition'li).

    refresh_only=None (varsayilan): TAM yeniden kurulum -- once TUM
    unified/ prefix'i silinir (clear_gold_before_unify), sonra source_types
    icindeki HER source_type yeniden islenir.

    refresh_only=("adsblol_realtime",) gibi verilirse: SADECE listelenen
    source_type(lar)in Gold parcalari silinip yeniden yazilir
    (clear_gold_source_before_unify); digger source_type'larin Gold verisi
    HIC DOKUNULMADAN kalir -- gunluk realtime "catch-up" gibi tek-kaynakli
    guncellemelerde ~2.76 milyar satirlik degismemis tarihsel veriyi
    yeniden okuyup yazmaktan kacinmak icin eklendi."""
    if refresh_only is not None:
        types_to_process = tuple(refresh_only)
        for source_type in types_to_process:
            clear_gold_source_before_unify(client, source_type, gold_bucket=gold_bucket)
    else:
        types_to_process = source_types
        clear_gold_before_unify(client, gold_bucket=gold_bucket)   # eski rerun'un ciktisini SIL

    total_rows = total_parts = 0
    for source_type in types_to_process:
        mapping = COLUMN_MAPS[source_type]
        object_names = list_layer_objects(client, silver_bucket, source_type)
        if not object_names:
            logger.warning("No Silver data for source_type=%s -- skipping", source_type)
            continue
        for obj_name in object_names:
            df = read_parquet_object(client, silver_bucket, obj_name)
            aligned = _apply_column_map(df, mapping)      # kolon-eslemeyi uygula
            write_gold(aligned, GOLD_NAME, partition=source_type, client=client, bucket=gold_bucket)
            total_rows += len(aligned)
            total_parts += 1
    return total_rows
```

**Neden "streaming" (dosya-dosya), tek seferde RAM'e yükleyip birleştirmek değil:** Gold'un
bu koşusunda toplam veri **2.76 milyar satır, 6.861 Silver parçası** — bunu tek bir pandas
DataFrame'de RAM'e yüklemek pratik olarak imkansız (yüzlerce GB RAM gerektirir). Bunun yerine
her Silver parçası okunup dönüştürülüp HEMEN bir Gold parçası olarak yazılıyor — bellek
kullanımı sabit kalıyor (o an işlenen tek parçanın boyutuyla orantılı, ~1-2GB civarı), toplam
veri boyutundan bağımsız.

**Karşılaşılan zorluk — çift sayım bug'ı (en kritik bulunan hata):**

```python
# BUG (once): her calistirma YENI, benzersiz-isimli parcalar yaziyordu ama
# ONCEKI calistirmanin parcalarini SILMIYORDU
def stream_unify_ESKI(client, ...):
    # clear_gold_before_unify() cagrisi YOK
    for source_type in source_types:
        for obj_name in list_layer_objects(...):
            write_gold(...)   # her seferinde YENI dosya, eskiler duruyor
# SONUC: Gold 2. kez calistirilinca veri 2 katina cikiyor, 3. kez 3 katina...
```

Bu bug, **tek seferlik testte fark edilmedi** — sadece Gold birkaç kez art arda çalıştırılıp
satır sayısının katlanarak arttığı GÖZLEMLENİNCE ortaya çıktı (klasik bir "idempotent olmayan
yeniden çalıştırma" hatası). Düzeltme:

```python
# src/common/minio_io.py -- eklenen fonksiyonlar
def remove_object(self, bucket_name: str, object_name: str) -> None: ...   # Protocol'e eklendi

def delete_layer_objects(client, bucket, source_type) -> int:
    """Bir prefix altindaki TUM objeleri siler."""
    names = list_layer_objects(client, bucket, source_type)
    for name in names:
        client.remove_object(bucket, name)
    return len(names)

# src/gold/unify.py
def clear_gold_before_unify(client, *, gold_bucket=None) -> int:
    removed = delete_layer_objects(client, gold_bucket, GOLD_NAME)
    if removed:
        logger.info("Gold: cleared %d stale part(s) before unify", removed)
    return removed
```

Regresyon testi eklendi ki bu hata bir daha sessizce geri gelmesin:

```python
# tests/test_gold_unify.py -- gercek test
def test_stream_unify_rerun_does_not_double_count_rows(fake_minio_client):
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)
    stream_unify(fake_minio_client)
    first_run_rows = len(read_layer(fake_minio_client, "gold", GOLD_NAME))

    stream_unify(fake_minio_client)   # AYNI veriyle IKINCI kez calistir
    second_run_rows = len(read_layer(fake_minio_client, "gold", GOLD_NAME))

    assert first_run_rows == second_run_rows   # IKI KATINA CIKMAMALI
```

**Neden iki ayrı Gold çıktısı var (kasıtlı tasarım, hata değil):**

| Yol | Konum | Şema | Kullanım |
|---|---|---|---|
| `src/gold/unify.py` | `gold/unified/<source_type>/` (2026-07-18'den itibaren source_type bazında partition'lı) | Dar, 7+3 ortak kolon | Görselleştirme, dashboard, "tüm veri bir arada" sorguları |
| `src/ml/build_features.py` | `data/gold/ml_features/` | Zengin, kaynak-özgü kolonlar korunur (`squawk`, `roll_deg`, `jamming_indicator`...) | ML model eğitimi |

`unify.py`'nin dar şeması ML için yetersiz kalırdı (feature'lar kaybolur); `build_features.py`'nin
zengin şeması ise görselleştirme için gereksiz karmaşık ve yavaş olurdu. Bu yüzden biri
diğerinin yerine geçmiyor, iki ayrı, birbirinden bağımsız üretim yolu var.

### 1.5 Kafka ETL / Gerçek Zamanlı Akış

```python
# Dashboard/codes/uav_producer.py -- gercek kod (ozetlenmis)
TOPIC = "uav.flights"
DEFAULT_WORLD_RADIUS_NM = 12000   # tek sorguda TUM dunyayi kapsayacak yaricap

producer = Producer({"bootstrap.servers": BOOTSTRAP, "linger.ms": 50})

while True:
    aircraft = fetch_from_adsblol(radius_nm=DEFAULT_WORLD_RADIUS_NM)   # 60sn'de bir
    for ac in aircraft:
        producer.produce(TOPIC, key=ac["icao24"], value=json.dumps(ac).encode(),
                          callback=delivery_report)
    producer.flush()
    time.sleep(60)
```

```python
# Dashboard/codes/dashboard_consumer.py -- gercek kod (ozetlenmis)
FLIGHTS_TOPIC = "uav.flights"
ALERTS_TOPIC = "uav.alerts"

consumer = Consumer({
    "bootstrap.servers": BOOTSTRAP,
    "group.id": "dashboard-consumer",       # <-- KENDI grubu
    "auto.offset.reset": "latest",
})
consumer.subscribe([FLIGHTS_TOPIC, ALERTS_TOPIC])
# ... Redis'e canli durum yazar + InfluxDB'ye batch yazar
```

```python
# Dashboard/codes/minio_archiver.py -- gercek kod (ozetlenmis)
TOPIC = "uav.flights"
GROUP_ID = "minio-archiver"                 # <-- FARKLI, BAGIMSIZ grup
BATCH_SIZE = 500

consumer = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": GROUP_ID,
                      "auto.offset.reset": "latest"})
consumer.subscribe([TOPIC])
# ... 500 mesajda/60sn'de bir JSONL olarak Bronze'a yazar
```

**Neden TEK bir consumer'ın 3 sink'e (Redis+InfluxDB+MinIO) yazması yerine İKİ ayrı
consumer:** Mimari diyagram tek consumer öngörüyordu; ekip kararıyla ikiye bölündü —
**Separation of Concerns**: arşivleyici (minio-archiver) çökerse dashboard'un canlı
görselleştirmesi (dashboard-consumer) ETKİLENMEZ, ve tam tersi. Her biri bağımsız
ölçeklenebilir, bağımsız yeniden başlatılabilir, kendi Kafka offset'ini yönetir. Kafka'nın
consumer-group modeli tasarım gereği aynı topic'i birden çok bağımsız grubun (farklı
`group.id`) okumasına izin verir — mesaj kaybı veya çakışma olmadan (her grup kendi offset
takibini tutar).

**Karşılaşılan zorluk — Zookeeper kaldırıldı, KRaft moduna geçildi:**

```yaml
# docker-compose.yml -- kafka servisi, KRaft konfigurasyonu
kafka:
  image: confluentinc/cp-kafka:7.5.0
  environment:
    KAFKA_PROCESS_ROLES: broker,controller   # Zookeeper YOK, kendi kendini yonetiyor
    KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:29093
```

Zookeeper, lokal/native kurulumdan kalma bir kalıntıydı — Docker'da tek-broker için gereksiz
bir servis + ~512MB ekstra RAM demekti. KRaft (Kafka'nın kendi önerdiği/varsayılan yeni yolu,
Zookeeper 4.0'da tamamen kaldırılıyor) tek node'un kendi kendinin controller'ı (quorum
voter'ı) olmasını sağlıyor.

**Karşılaşılan zorluk — çift listener gerekliliği:**

```yaml
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT,CONTROLLER:PLAINTEXT
KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://${KAFKA_LAN_HOST:-localhost}:29092
```

Tek listener (`PLAINTEXT://localhost:9092`) container-içi client'ları kırıyordu — broker
metadata'da "localhost" döndüğünde, container İÇİNDEKİ producer/consumer bunu KENDİ içi
olarak yorumlayıp broker'a değil kendine bağlanmaya çalışıyordu (klasik Kafka+Docker hatası).
Çözüm: biri container-içi (`kafka:9092`), biri host/LAN tarafı (`PLAINTEXT_HOST`) için iki
ayrı listener.

**Ayrıca bulunan gerçek bir operasyonel sorun:** `KAFKA_LAN_HOST` ayarı, aynı Wi-Fi/LAN'daki
bir takım arkadaşının KENDİ makinesindeki ayrı bir producer yerine, PAYLAŞILAN tek Kafka'ya
bağlanabilmesini sağlamak için eklendi — daha önce iki bağımsız producer aynı anda
`adsb.lol`'e istek atıp gereksiz yük/rate-limit riski yaratıyordu.

**InfluxDB sessiz başarısızlık:** `-7d` sorgusu 4.36M+ satıra ulaşınca sürekli
`ReadTimeoutError` ile sessizce başarısız oluyordu:

```python
# ONCEKI -- varsayilan istemci timeout 10sn, gercek sorgu suresi ~170sn
client = InfluxDBClient(url=..., token=..., timeout=10_000)   # ms
query = 'from(bucket: "adsb-history") |> range(start: -7d) |> pivot(...)'   # PAHALI

# SONRAKI -- IKI degisiklik birlikte (biri tek basina yetmezdi)
client = InfluxDBClient(url=..., token=..., timeout=240_000)  # 240sn
query = '''
from(bucket: "adsb-history") |> range(start: -7d)
  |> keep(columns: ["_time", "icao24", "_field", "_value"])
'''
# pivot() SUNUCUDAN kaldirildi, yerel pandas.pivot_table() ile degistirildi
```

**Zamanlama (`--daily-at`) ve heartbeat:**

```python
# parse_adsblol_realtime.py -- gercek mantik
def _seconds_until(hhmm: str) -> float:
    target = datetime.now().replace(hour=int(hhmm[:2]), minute=int(hhmm[3:]), second=0)
    if target <= datetime.now():
        target += timedelta(days=1)
    return (target - datetime.now()).total_seconds()

while True:
    try:
        run()
        _HEARTBEAT_PATH.write_text(f"{datetime.now(timezone.utc).isoformat()} basarili\n")
    except Exception:
        logger.exception("gunluk calisma basarisiz")
        # HEARTBEAT BILEREK YAZILMIYOR -- dosyanin eski kalmasi, dongunun
        # sessizce durdugunun/basarisiz kaldiginin dis-gozlemlenebilir isareti
    time.sleep(_seconds_until("17:00"))
```

**Neden sabit `--interval 86400` değil, `--daily-at`:** PC sadece mesai saatlerinde açık;
sabit bir aralık kullanılsaydı, PC kapalıyken sayaç donar ve PC açılınca rastgele/tutarsız bir
saatte tetiklenirdi. `--daily-at 17:00`, PC ne zaman açılırsa açılsın HER GÜN aynı saati hedefler.

**MinIO'da 7 günlük otomatik silme — kaldırıldı:** Mimari niyet "MinIO'da otomatik silme
kuralı OLMAMALI, sadece InfluxDB'de (168 saat retention) geçici saklama olmalı" idi; ama
incelemede `minio_archiver.py` içinde gerçek bir `ensure_lifecycle()` çağrısının zaten var
olduğu bulundu — yani KOD, mimari kararla ÇELİŞEN bir davranış sergiliyordu. Kural kaldırıldı;
kalıcılık artık sadece günlük Silver işleme ("işle, sonra sil") ile sağlanıyor.

### 1.6 Docker Compose Altyapısı ve Kaynak Sınırları

```yaml
# docker-compose.yml -- kaynak siniri, gercek yorum satiriyla birlikte
kafka:
  deploy:
    resources:
      limits:
        # ONEMLI: 1g -> 2g -- Kafka surekli %94-97 doluluk civarinda
        # seyrediyordu (JVM heap + off-heap page cache/network buffer'lar
        # 1GB'a fazla sikisiyor), OOM-kill riski yakindi. Host'ta bolca bos
        # kapasite var (o an sadece ~1.9GB/33.7GB kullanimdaydi).
        memory: 2g
```

**Profil (`profiles: ["streaming"]`) kullanımı — Faz 2'de sorun kaynağı oldu:**

```yaml
kafka:
  profiles: ["streaming"]
redis:
  profiles: ["streaming"]
influxdb:
  profiles: ["streaming"]
```

`docker-compose up -d` (profilsiz) bu servisleri ATLAR — sadece `docker-compose --profile
streaming up -d` ile başlarlar. Bu, "gerçek zamanlı akışı istemediğim zaman gereksiz servis
çalıştırmayayım" diye bilinçli bir tasarım, ama Faz 2'deki krizde beklenmedik şekilde bu
servislerin PC yeniden başlatılınca ayağa kalkmamasına neden oldu (aşağıda detaylı).

**InfluxDB'nin çözülmemiş sorunu (açık madde):** 1GB bellek limiti 7 günlük birikmiş
realtime veriyi (4M+ satır) sorgulamaya yetmedi — CPU %400+'a çıktı, container zaman zaman
yanıt vermez oldu, bir bucket-delete denemesinde flux `pivot()` dönüşümünde recovered-olmayan
bir nil-pointer panic ile GERÇEKTEN çöktü. `docker inspect` ile `OOMKilled=false` doğrulandı
— yani bellek limiti değil, InfluxDB'nin kendi flux motorundaki bir hataydı. Container
yeniden başlatıldı; kalıcı bir çözüm (bellek artışı veya sorgu mimarisi değişikliği) BU
OTURUMDA yapılmadı, ileride sorgu-yoğun işlemler sırasında eşzamanlı yazma/silme yükünü
azaltmak öneri olarak bırakıldı.

---

## FAZ 2 — Altyapı Krizi: PC Kapanması ve Kurtarma

**Ne oldu:** Bir PC kapanması, devam eden Silver reprocess'i (o zamanki 11 tar) yarıda kesti
ve `profiles: ["streaming"]` gated servislerini (kafka, redis, influxdb) durdurdu.

**Karşılaşılan zorluk 1 — checkpoint yokluğu:** O zamanki `parse_adsblol_historical.run()`,
HER çağrıda `delete_layer_objects()` ile TÜM Silver'ı silip TÜM tar'ları sıfırdan işliyordu
(bkz. eski docstring: *"There is no per-tar resume checkpoint -- every call reprocesses all
tars found under bronze_prefix from scratch"*). Bu yüzden yarıda kalan reprocess devam
ettirilemedi, 11 tar'ın TAMAMI sıfırdan yeniden işlendi (bu sorun Faz 3'te KALICI olarak
çözüldü).

**Karşılaşılan zorluk 2 — Docker profil sürprizi:**

```powershell
# BEKLENEN (ama yetersiz) komut
docker-compose up -d
# SONUC: adsb-producer/dashboard-app/vb. otomatik geri geldi (Docker Desktop'un
# kendi "onceden calisan container'lari yeniden baslat" davranisiyla,
# compose profillerinden BAGIMSIZ) AMA kafka/redis/influxdb GELMEDİ
# (profiles: ["streaming"] gated, bu flag verilmedigi icin atlandi)

# DOGRU komut
docker-compose --profile streaming up -d
```

**Karşılaşılan zorluk 3 — MinIO geçici I/O hatası:** `lstat /data: input/output error` —
kök neden, bind-mount'un işaret ettiği `D:` sürücüsünün Docker'ın kendi başlangıcında henüz
tam hazır olmamasıydı (Windows'ta harici/ikincil sürücülerin boot sırasında geç
"mount" olması bilinen bir durum). Çözüm: tekrar `docker-compose up -d` çalıştırmak (sürücü
tam hazır olduktan sonra) — kalıcı bir kod değişikliği gerekmedi, sadece zamanlama sorunuydu.

**Doğrulama:** `mc ls` ile bronze/silver/gold bucket'larının İÇERİĞİNİN bozulmadığı (veri
kaybı olmadığı) teyit edildi — sadece devam eden reprocess'in ilerlemesi kayboldu, zaten
YAZILMIŞ olan veri sağlamdı.

---

## FAZ 3 — Veri Genişletme: 19 Yeni Tar ve Checkpoint/Resume Sistemi

### 3.1 Yeni Veri Ekleme

**Ne yaptık:** 19 yeni tarihsel tar dosyası (toplam 64.1GB) `data/bronze/adsblol_historical/_input/`
klasörüne indirildi, `scripts/upload_bronze_all.py` ile MinIO Bronze'a yüklendi.

```python
# scripts/upload_bronze_all.py -- ozet
for path in sorted(_input_dir.glob("*.tar")):
    if _already_uploaded(client, bucket, object_name):   # idempotent -- tekrar calistirilabilir
        continue
    with open(path, "rb") as f:
        write_bronze_bytes(f.read(), object_name, client=client)
    path.unlink()   # basarili yuklemeden SONRA yerel kopyayi sil (disk yer acar)
```

Sonuç: **19/19 başarılı, 0 hata** — ~63 MB/s ortalama hızla, ~18 dakikada tamamlandı.

### 3.2 Bulunan Bug: Tar'lar Arası Çakışma Riski (Global Set Dedup)

**Bulunan sorun:** Yoğunluk/agregasyon hesaplayan script, chunk'lar arası sayımı bir
`Counter` ile TOPLUYORDU:

```python
# ONCEKI -- Counter toplama, chunk'lar arasi CIFT SAYIM riski tasiyor
flight_counts = Counter()
for chunk in stream_gold_data():
    dedup = chunk.drop_duplicates(subset=["h3_cell", "source_id", "date"])   # chunk-ICI dedup OK
    flight_counts.update(dedup["h3_cell"].value_counts().to_dict())          # ama chunk'lar ARASI toplama
```

Bu, orijinal 11 tar'ın ~1 aylık aralıklarla seçildiği ve her tar'ın trace rolling-window'unun
(birkaç gün) bir SONRAKİ tar ile ÇAKIŞMADIĞI varsayımına dayanıyordu — bu varsayım koddaki bir
yorumda AÇIKÇA yazılıydı. Yeni 19 tar'ın bazıları (`v2025.08.21` + `v2025.08.26` = 5 gün,
`v2025.10.25` + `v2025.10.28` = 3 gün arayla) bu varsayımı BOZDU: aynı uçağın aynı günkü izi
iki farklı tar'da (iki farklı Silver chunk'ında) görülüp Counter'da İKİ KEZ toplanabilirdi.

**Çözüm — Counter yerine gerçek global set:**

```python
# SONRAKI -- hex basina GERCEK GLOBAL set, chunk sinirlarindan BAGIMSIZ
flight_hex_sets: dict[str, set] = {}
for chunk in stream_gold_data():
    dedup = chunk.drop_duplicates(subset=["h3_cell", "source_id", "date"])
    for h3_cell, source_id, date in zip(dedup["h3_cell"], dedup["source_id"], dedup["date"]):
        flight_hex_sets.setdefault(h3_cell, set()).add((source_id, date))
        # (source_id, date) ayni chunk'ta VEYA FARKLI bir chunk'ta tekrar
        # gorulse bile set zaten iceriyorsa TEKRAR EKLENMEZ -- gercek dedup
flight_count = {h: len(s) for h, s in flight_hex_sets.items()}
```

**Neden bu yolu seçtik (Counter yerine set):** Counter'ın avantajı bellek verimliliğiydi
(sadece sayı tutar, tekil değerleri tutmaz) — ama artık tar'lar arası çakışma mümkün
olduğundan doğruluk, bellekten daha kritikti. Set çözümü biraz daha fazla bellek kullanır
(her hex için tekil (uçak, gün) çiftlerini tutar) ama SONUÇ HER KOŞULDA DOĞRUDUR — tar'ların
ne kadar sık/seyrek seçildiğinden bağımsız.

### 3.3 Bulunan Bug: `src/gold/unify.py`'nin Arşivde Kaybolması

**Bulunan sorun:** `src/gold/unify.py`, önceki bir arşivleme işlemi (`archive/2026-07-10_.../`)
sırasında YANLIŞLIKLA süpürülmüştü — canlı kod ağacında (`src/gold/`) sadece eski, derlenmiş
bir `.pyc` dosyası kalmıştı, kaynak kodu YOKTU. Bu, "Gold nasıl çalışıyor?" diye bakıldığında
fark edildi.

**Tespit yöntemi:**
```powershell
Test-Path "src/gold/unify.py"                     # False
Get-ChildItem "src/gold/__pycache__"              # unify.cpython-314.pyc VAR (eski, 2026-07-06)
Get-ChildItem "archive/2026-07-10_.../src/gold/"   # unify.py BURADA, guncel (is_military destekli)
```

**Çözüm:** Doğru, güncel dosya arşivden `src/gold/unify.py`'ye geri kopyalandı. Aynı
arşivleme hatasıyla `tests/test_gold_unify.py` de kaybolmuştu — o da `tests/`'e geri kondu ve
**7/7 test geçti**, doğrulandı.

### 3.4 En Kritik Talep: Tar-Bazlı Checkpoint/Resume

**Neden gerekliydi:** 30 tar'lık reprocess ~17 saat sürecekti (11 tar ~5 saat sürmüştü, 30
tar orantısal olarak çok daha uzun). PC bu kadar açık tutulamıyordu. Checkpoint'siz mevcut
davranış (her çağrıda sıfırdan başlama) bu ölçekte kabul edilemezdi.

```python
# src/silver/parse_adsblol_historical.py -- eklenen checkpoint mekanizmasi (gercek kod)
CHECKPOINT_PATH = Path("data/state/silver_historical_checkpoint.json")

def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"completed_tars": [], "in_progress": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"completed_tars": [], "in_progress": {}}
    state.setdefault("completed_tars", [])
    state.setdefault("in_progress", {})
    return state

def _save_checkpoint(path: Path, state: dict) -> None:
    """Yaz-sonra-degistir (atomic write): checkpoint HER Silver parcasi
    yazildikca guncellenir (sik cagri) -- islem TAM bu yazma sirasinda
    kesilirse (kill/guc kesintisi) yarim/bozuk JSON kalmasin diye once
    gecici dosyaya yazilip SONRA atomik olarak yerine tasinir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)   # POSIX rename / Windows ReplaceFile -- ATOMIK

def _delete_uris(client, uris: list[str]) -> None:
    """Yarim kalan bir tar'in KISMEN yazilmis Silver parcalarini siler --
    boylece retry'da eski+yeni parcalar YAN YANA durup cift saymaz."""
    for uri in uris:
        _, _, rest = uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        client.remove_object(bucket, key)
```

```python
# run() icindeki entegrasyon (gercek kod, kisaltilmis)
def run(bronze_prefix="adsblol_historical/", *, fresh=False, checkpoint_path=CHECKPOINT_PATH, ...):
    if fresh:
        delete_layer_objects(client, silver_bucket, SOURCE_TYPE)   # --fresh: eski davranis
        state = {"completed_tars": [], "in_progress": {}}
    else:
        state = _load_checkpoint(checkpoint_path)
        for tar_name, partial_uris in list(state["in_progress"].items()):
            # ONCEKI calistirma bu tar'i islerken KESILMISTI -- yarim parcalari
            # sil, tar'i sifirdan islenecekler listesine dahil et
            _delete_uris(client, partial_uris)
            del state["in_progress"][tar_name]

    completed = set(state["completed_tars"])
    for tar_object in tar_objects:
        tar_name = tar_object.split("/")[-1]
        if tar_name in completed:
            logger.info("'%s' zaten tamamlanmis, atlaniyor", tar_name)
            continue   # <-- TAM BURASI: onceden islenmis tar'lar TEKRAR indirilip islenmez

        state["in_progress"][tar_name] = []
        _save_checkpoint(checkpoint_path, state)

        def _on_part_written(uri, _tar=tar_name):
            state["in_progress"][_tar].append(uri)
            _save_checkpoint(checkpoint_path, state)   # HER parcada durabilite

        uris = _parse_tar_fileobj(..., on_part_written=_on_part_written)

        del state["in_progress"][tar_name]
        state["completed_tars"].append(tar_name)
        _save_checkpoint(checkpoint_path, state)
```

**Checkpoint dosyasının gerçek görünümü (çalışma sırasında):**
```json
{
  "completed_tars": ["v2025.08.15-planes-readsb-prod-0.tar", "v2025.08.21-...tar", "..."],
  "in_progress": {
    "v2026.01.03-planes-readsb-prod-0.tar": [
      "s3://silver/adsblol_historical/part-20260713T130034534159Z-c60263e4.parquet",
      "s3://silver/adsblol_historical/part-20260713T130044348924Z-f20d2b06.parquet"
    ]
  }
}
```

**Neden bir JSON dosyası, gerçek bir veritabanı (SQLite/Postgres) değil:** Checkpoint'in
kendisi çok basit bir veri yapısı (bir liste + bir dict) ve tek bir işlem tarafından
yazılıyor (eşzamanlı yazıcı yok, race condition riski yok) — bu ölçekte bir veritabanı
aşırı mühendislik olurdu. Tek gerçek risk, yazma sırasında kesintiydi, o da atomic
write (temp dosya + rename) ile çözüldü.

**Neden "flush-düzeyinde" durabilite, "satır-düzeyinde" değil:** Her Silver parçası
(~300 uçak/parça, ~400 bin satır) tek bir yazma işlemi olarak MinIO'ya gidiyor —
bu, doğal bir "iş biriminin" sınırı. Daha ince taneli (satır bazlı) bir checkpoint hem
gereksiz karmaşıklık hem MinIO'ya çok daha fazla küçük yazma (performans kaybı) demek
olurdu; kesintide en fazla kaybedilecek şey tek bir parçanın (~400 bin satırlık) işlenmemiş
hali, bu kabul edilebilir bir maliyet.

**Karşılaşılan zorluk — canlı geçiş (en riskli adım):** Checkpoint özelliği eklenirken
süreç ZATEN eski (checkpoint'siz) kodla çalışıyordu ve 8-10 tar'ı bitirmişti (~5 saat).
Kodu o an değiştirip yeniden başlatmak, checkpoint dosyası henüz var olmadığından TÜM
ilerlemeyi sıfırlardı. Uygulanan güvenli geçiş prosedürü:

1. Mevcut (eski kod ile çalışan) sürecin, o an işlediği tar'ın TAM bitmesi beklendi
   (log'da `"Downloading v2025.12.01..."` satırının HENÜZ görünmediği, en son tar'ın
   `"Done ..."` satırının göründüğü an).
2. Tam o sınırda (yeni bir tar'ın indirmesi henüz başlamadan) process durduruldu
   (`Stop-Process -Force`).
3. O ana kadar bitmiş tüm tar'ları `completed_tars` listesine elle yazan bir checkpoint
   dosyası oluşturuldu.
4. Yeni (checkpoint destekli) kod başlatıldı — log'da her tar için `"zaten tamamlanmis,
   atlaniyor"` mesajlarını, ardından kaldığı yerden (bir sonraki tar'dan) devam ettiğini
   doğrulandı.

**Neden "tam sınırda durdurma" gerekliydi:** Eğer bir tar İŞLENİRKEN (kısmen parça
yazılmışken) durdurulup checkpoint elle oluşturulsaydı, o tar'ın eski koddan kalma kısmi
parçaları hiçbir checkpoint kaydı olmadan MinIO'da "öksüz" kalırdı — yeni kod o tar'ı
sıfırdan işlerken bu öksüz parçalar silinmeden yanına yeni parçalar eklenir, VERİ MÜKERRER
SAYILIRDI. Bu yüzden geçiş, aktif olarak bir tar'ın TAM bitişini bekleyerek yapıldı.

**Sonuç:** 30 tar'ın tamamı, iki kod versiyonu (eski→yeni) arasında hiçbir veri kaybı veya
mükerrerlik olmadan başarıyla işlendi:

```
Gold unified complete: 2762067817 total rows in 6861 parts
  - adsblol_historical: 6852 parca
  - adsblol_realtime:       7 parca
  - alfa:                   1 parca
  - uav_attack:              1 parca
```

---

## Sunumda Sorulabilecek Sorular ve Cevapları

### Mimari

**S: Neden Bronze/Silver/Gold gibi 3 katman? Neden direkt tek bir temiz tabloya yazmadınız?**
C: Geri dönülebilirlik (parse mantığı hatalıysa ham veri bozulmaz, sadece yeniden üretilir),
sorumluluk ayrımı (birim dönüşümü ile şema-birleştirme farklı kaygılar) ve test edilebilirlik
için. Nitekim Gold'un çift-sayım bug'ı ve UAV Attack'in regex bug'ı, bu katman ayrımı
sayesinde SADECE ilgili katman yeniden çalıştırılarak düzeltildi, tüm pipeline'ı elle
düzeltmeye gerek kalmadı.

**S: Neden MinIO, neden gerçek AWS S3 veya yerel dosya sistemi değil?**
C: MinIO, S3 API'siyle uyumlu ama lokal/ücretsiz çalışıyor — geliştirme ortamında gerçek
bulut maliyeti/gecikmesi olmadan aynı API'yi kullanmayı sağlıyor. İleride gerçek S3'e geçiş
sadece endpoint/credential değişikliği, kod değişmez (`ObjectStoreClient` Protocol'ü zaten
soyutlanmış).

**S: Neden Kafka? Doğrudan veritabanına yazamaz mıydınız?**
C: Kafka, TEK bir üreticiden (producer) GELEN veriyi BİRDEN FAZLA bağımsız tüketiciye
(consumer) dağıtmayı sağlıyor — burada dashboard (Redis/InfluxDB) ve arşivleme (MinIO) iki
ayrı, birbirinden bağımsız ihtiyaç. Doğrudan veritabanına yazsaydık, arşivleme mantığı
dashboard'un kodu içine sıkışırdı veya iki ayrı üretici aynı kaynağa (adsb.lol) iki kez
istek atardı.

**S: Neden tek consumer değil de iki ayrı consumer (aynı topic'i iki kez okuyan)?**
C: Separation of Concerns. Arşivleyici çökerse/yavaşlarsa dashboard etkilenmesin. Kafka'nın
consumer-group modeli bunu doğal olarak destekliyor — her grup kendi offset'ini bağımsız
tutar, mesaj "paylaşılmaz", her bağımsız grup TÜM mesajları görür.

**S: Gold'da iki farklı çıktı üretmek (unify.py + build_features.py) kod tekrarı değil mi?**
C: Hayır, kasıtlı bir ayrım — biri (unify) dar/hızlı/görselleştirme-odaklı, diğeri
(build_features) zengin/ML-odaklı. İkisini TEK bir şemaya sıkıştırsaydık, ya görselleştirme
gereksiz kolonlarla yavaşlardı ya da ML gerekli feature'ları kaybederdi.

### Güvenilirlik / Hata Toleransı

**S: Sistem bir noktada çökerse (Kafka, InfluxDB, MinIO) ne olur, veri kaybolur mu?**
C: Katmana göre değişir: (1) Bronze→Silver işleme "yaz-sonra-sil" deseniyle idempotent —
Silver'a yazma BAŞARILI olmadan Bronze'daki kaynak silinmiyor, yarıda kalan bir hata Bronze'u
olduğu gibi bırakıyor, bir sonraki çalıştırmada tekrar denenir. (2) Gold, her çalıştırmada
TEMİZ yeniden yazıldığı için (clear-then-write) ara bir çökme en kötü ihtimalle eksik/yarım
bir Gold üretir, bir sonraki çalıştırma düzeltir. (3) Silver historical reprocess, artık
tar-bazlı checkpoint'e sahip — kesinti SADECE o an işlenen tek tar'ı etkiler.

**S: Checkpoint dosyasının kendisi bozulursa ne olur?**
C: Checkpoint'in her güncellemesi atomic write (önce `.tmp` dosyasına yaz, sonra
`os.replace()` ile asıl dosyanın yerine geçir) ile yapılıyor — işletim sistemi düzeyinde bu
rename işlemi bölünmez (atomic), yani ya eski içerik ya TAM yeni içerik okunur, asla "yarım
yazılmış" bir JSON okunmaz. `_load_checkpoint()` ayrıca `JSONDecodeError` durumunda (dosya her
nasılsa hâlâ bozuksa) sıfırdan başlamaya güvenli şekilde geri düşer.

**S: Gold'un "her çalıştırmada tamamen sil, yeniden yaz" yaklaşımı ölçeklenebilir mi? Ya
veri çok büyürse?**
C: Şu anki ölçekte (2.76 milyar satır, ~6800 parça) tam yeniden yazım birkaç saat sürüyor —
kabul edilebilir çünkü Gold rebuild sık sık değil, veri kaynağı değiştiğinde (yeni tar
eklendiğinde) tetikleniyor. Çok daha büyük ölçekte (onlarca milyar satır) artımlı
(incremental) bir Gold güncelleme stratejisi gerekebilir, ama bu şu anki gereksinimler için
erken optimizasyon olurdu.

**S: Neden `confluent-kafka` gibi ağır bir kütüphane, neden `kafka-python` gibi saf-Python
bir alternatif değil?**
C: `confluent-kafka`, librdkafka'nın (C kütüphanesi) Python sarmalayıcısı — performans ve
kararlılık açısından üretim ortamlarında daha çok tercih edilir. (Not: bu tercih, Windows'ta
geliştirici makinesinde derleme sorunu yaratabiliyor çünkü C uzantısı derlemek Visual C++
Build Tools istiyor — bu, ayrı bir bulgu olarak test raporunda not edildi.)

### Performans

**S: Silver/Gold rebuild neden bu kadar uzun sürüyor (saatler)?**
C: Toplam veri hacmi büyük (30 tar, ~2.76 milyar satır sonuçta Gold'da) ve işlem tek makinede,
büyük ölçüde tek-thread çalışıyor (parse + parquet yazma CPU-yoğun). Darboğaz disk I/O değil,
CPU (JSON/gzip decode + birim dönüşümü) — bu, `CPU zaman / duvar-saati zamanı` oranının
sürekli yüksek (%80-100) gözlenmesinden anlaşıldı.

**S: Neden paralelleştirmediniz (çoklu işlem/thread)?**
C: Basitlik ve doğruluk önceliği — tek-thread sıralı işleme, checkpoint mantığını (hangi
tar'ın ne zaman bittiğini bilme) çok daha basit tutuyor. Paralel işlemede checkpoint'in
thread-safe olması, parçaların doğru sırayla/çakışmadan yazılması gibi ek karmaşıklık
gerekirdi — bu ölçekte (saatler, günler değil) gerekli görülmedi.

**S: Gerçek zamanlı feed'in 60 saniyelik döngüsü neden seçildi, neden daha sık değil?**
C: `adsb.lol`'ün "world" modundaki tek dev sorgusu zaten önemli bir yük; daha sık sorgu hem
gereksiz yere kaynak kaynağını zorlar hem InfluxDB/Kafka'ya orantısız yük bindirir. 60sn,
uçak pozisyon güncellemesi için yeterince "canlı" hissettiren ama sürdürülebilir bir aralık
olarak seçildi.

### Test / Doğrulama

**S: Testler gerçek MinIO/Kafka sunucusu gerektiriyor mu?**
C: Hayır — `ObjectStoreClient` bir `Protocol` (yapısal arayüz) olarak tanımlandığından,
`FakeMinioClient` (bellek-içi sahte istemci) testlerde gerçek `minio.Minio` istemcisinin
yerine geçebiliyor. Bu sayede testler saniyeler içinde, Docker'a hiç ihtiyaç duymadan
çalışıyor.

**S: Gold'un çift-sayım bug'ı gibi hatalar bir daha nasıl önleniyor?**
C: Regresyon testleriyle — `test_stream_unify_rerun_does_not_double_count_rows` özellikle
"aynı veriyle iki kez çalıştır, satır sayısı katlanmasın" senaryosunu doğruluyor. Bu test,
bug bulunduktan SONRA yazıldı (regresyon testi) ama artık CI'da her çalıştığında bu spesifik
hatanın geri gelmediğini garantiliyor.

### Alternatifler / "Neden Şunu Yapmadınız"

**S: Neden Apache Airflow gibi bir workflow orchestrator kullanmadınız?**
C: Mevcut ölçekte (birkaç script, elle/zamanlanmış tetiklenen) Airflow'un DAG yönetimi,
scheduler'ı, web UI'si aşırı mühendislik olurdu — basit bir `--daily-at` döngüsü + heartbeat
dosyası aynı operasyonel görünürlüğü çok daha az bağımlılıkla sağlıyor. İş akışları
karmaşıklaşıp birbirine bağımlı çok sayıda görev olursa (ör. Bronze→Silver→Gold→density
otomatik zincir), Airflow o zaman değerlendirilebilir.

**S: Neden gerçek bir veritabanı (Postgres/PostGIS) yerine Parquet dosyaları?**
C: Bronze/Silver/Gold, büyük hacimli, çoğunlukla "yaz-bir-kez, çok-oku" (write-once,
read-many) veri için tasarlandı — bu, bir OLTP veritabanından çok bir data lake'in doğal
kullanım şekli. Parquet, sütun bazlı sıkıştırma ve pandas/pyarrow ile doğrudan uyumluluk
sağlıyor; bir veritabanı sunucusu yönetmek (bağlantı havuzu, şema migrasyonu, indeksleme)
bu iş yükü için gereksiz operasyonel yük eklerdi.

**S: PC kapanması/checkpoint sorunu neden en baştan (proje başlarken) düşünülmedi?**
C: Erken optimizasyondan kaçınma prensibi — 11 tar'lık ilk reprocess ~5 saat sürüyordu, bu
ölçekte "sıfırdan başlama" kabul edilebilir bir maliyetti. Checkpoint'in gerçek ihtiyacı,
veri hacmi 30 tar'a (~17 saat tahmini) çıkınca SOMUT olarak ortaya çıktı — yani bu, "önce
basit çöz, ölçek gerektirdiğinde karmaşıklaştır" yaklaşımının bir örneği, eksik planlama
değil.

---

## Genel Sunum Anlatımı — Takım Projesi Baştan Sona (Konuşma Metni)

Bu bölüm, yukarıdaki teknik detaylardan bağımsız olarak, **doğrudan sunumda okunabilecek/
anlatılabilecek** akıcı bir metin. Slayt başlıklarına karşılık gelecek şekilde bölümlere
ayrıldı; her bölüm 30-60 saniyelik bir konuşma parçası düşünülerek yazıldı.

### 1) Problem ve Motivasyon

Bu proje, hava trafiği verisi (uçak konum/hız/irtifa kayıtları) üzerinden **anomali tespiti**
yapabilecek bir veri platformu kurmayı hedefliyor: hem gerçek zamanlı akan uçuşları izleyip
şüpheli davranışları (rota sapması, sinyal kaybı, askeri/sivil karışık trafik gibi) yakalamak,
hem de aylar süren tarihsel veri üzerinde bu davranışların ne kadar yaygın olduğunu ölçmek.
Bunu yapabilmek için önce sağlam, ölçeklenebilir, geri dönülebilir bir **veri altyapısı**
kurmamız gerekiyordu — bu rapor esas olarak o altyapının hikâyesini anlatıyor.

### 2) Veri Kaynağı Kararı

Plan başta OpenSky Network API'sini öngörüyordu, ama OpenSky kimlik doğrulama ve günlük kota
gerektiriyor — hem TB'larca tarihsel arşiv indirmeye hem sürekli canlı sorgu atmaya birlikte
yetmiyordu. Bunun yerine `adsb.lol` seçildi: kota yok, kimlik doğrulama yok, hem **günlük
tam-ağ tarihsel arşiv** (tar dosyaları) hem **canlı REST feed** aynı kaynaktan geliyor. Buna
ek olarak, gerçek arıza/saldırı etiketleri taşıyan iki açık veri seti daha eklendi: **ALFA**
(gerçek uçuş arızası ground-truth'u) ve **UAV Attack** (iyi huylu/kötü niyetli saldırı
etiketleri) — bu ikisi, ileride yapılacak anomali tespitinin precision/recall gibi somut
metriklerle ölçülebilmesini sağlıyor.

### 3) Mimari: Bronze / Silver / Gold

Ham veriyi doğrudan "temiz" bir tabloya yazmak yerine üç katmanlı bir mimari (medallion
architecture) kurduk:

- **Bronze**: hiçbir dönüşüm yapılmadan saklanan ham dosyalar (tar/zip/JSONL).
- **Silver**: birim dönüşümü (feet→metre, knots→m/s), etiket çıkarımı (askeri/sivil),
  provenance bilgisiyle zenginleştirilmiş Parquet.
- **Gold**: tüm farklı kaynakların (adsb.lol tarihsel, adsb.lol gerçek zamanlı, ALFA, UAV
  Attack) ortak 7+3 kolonluk tek bir şemaya hizalandığı, birleşik veri seti.

Bunu üç katmana ayırmamızın nedeni basit: bir parse hatası olduğunda (ki iki kez oldu),
ham veri bozulmadığı için sadece o katmanı düzeltip yeniden üretmek yetiyor — tüm veriyi
yeniden indirmeye gerek kalmıyor. Depolama için MinIO (S3-uyumlu, Docker ile ayağa kalkan,
ücretsiz) kullandık; kod, gerçek S3'e sadece endpoint değiştirerek geçebilecek şekilde
soyutlanmış durumda.

### 4) İki Ayrı Veri Yolu: Hazır Veri Seti ve Gerçek Zamanlı Akış

Platformda iki bağımsız veri yolu işliyor. **Hazır veri seti yolu**, adsb.lol'ün tarihsel tar
arşivlerini ve ALFA/UAV Attack'i elle/talep üzerine Bronze'a yükleyip Silver'da işleyip
Gold'da birleştiriyor — yeni bir tar eklenmediği sürece bu yol tetiklenmiyor. **Gerçek
zamanlı yol** ise sürekli çalışıyor: bir "producer" her 60 saniyede bir tüm dünyadaki uçakları
sorgulayıp Kafka'ya yazıyor, iki bağımsız "consumer" bu veriyi eş zamanlı olarak hem canlı
gösterim için Redis/InfluxDB'ye hem arşivleme için MinIO'ya (ham JSONL olarak) akıtıyor. Bu
ham JSONL'ler her gün saat 17:00'de otomatik olarak Silver'a (Parquet'e) dönüştürülüp
işlenmiş dosya silinerek Bronze temiz tutuluyor — yani gerçek zamanlı veri her gün kendini
"sindirip" kalıcı hale getiriyor. İki yol da sonunda aynı Gold katmanına akıyor, oradan da
haritalar, yoğunluk analizleri ve takım export panosu besleniyor.

### 5) Kafka ile Gerçek Zamanlı Akış — Neden Bu Kadar Katmanlı?

Kafka'yı seçme nedenimiz basit: tek bir üreticiden gelen veriyi birden fazla bağımsız
tüketiciye dağıtmak. Dashboard'un canlı görselleştirmesi ile ham veri arşivleme işi
BİRBİRİNDEN BAĞIMSIZ olsun istedik — biri çökerse diğeri etkilenmesin. Kafka'nın
consumer-group modeli bunu doğal olarak destekliyor: her tüketici kendi ilerleme takibini
(offset) bağımsız tutuyor, aynı mesajı ikisi de görüyor ama biri diğerini bekletmiyor.

### 6) Karşılaşılan Krizler ve Nasıl Çözüldü

Süreç boyunca gerçek, somut sorunlarla karşılaştık — bunlar sadece "planlandığı gibi gitti"
demek yerine, projenin olgunlaştığı yerler:

- **Çift sayım bug'ı:** Gold birleştirme adımı, art arda çalıştırıldığında veriyi ikiye/üçe
  katlıyordu çünkü önceki çalışmanın çıktısını silmiyordu. Bunu, aynı veriyle iki kez
  çalıştırıp satır sayısının katlandığını GÖZLEMLEYEREK bulduk; düzeltip bir daha geri
  gelmesin diye özel bir regresyon testi ekledik.
- **Bellek patlaması:** Yoğunluk/askeri harita hesaplayan script, veri hacmi büyüyünce
  32 GB'a kadar şişip diske takas (swap) yapmaya başladı, saatlerce ilerlemedi. Kök nedeni
  (küme içindeki verinin bellek-verimsiz saklanması) bulup veriyi paketlenmiş tamsayılara
  çevirerek bellek kullanımını ~15 kata kadar düşürdük, üstelik uzun süren işlemler için
  kesintiye dayanıklı bir "checkpoint" (kaldığı yerden devam etme) sistemi ekledik.
  Bu sayede 30 tar'ın işlenmesi (~17 saat) bir PC kapanmasına rağmen veri kaybı olmadan
  tamamlanabildi.
  - **Not (dürüst bir teknik detay):** Checkpoint tek başına "gerçek" bir çözüm değil —
    her koşunun tepe bellek ihtiyacını azaltmıyor, sadece bir kesintide baştan başlamayı
    önlüyor. Veri daha da büyürse (60-100 tar gibi) aynı bellek tavanına yeniden
    çarpılabilir; kalıcı çözüm sıralı/parça-bazlı işleme ya da DuckDB gibi disk-tabanlı bir
    agregasyon motoruna geçiş olurdu — bu, ileride ele alınacak bir iyileştirme olarak not
    edildi.
- **"Rename-drift" (isim değişikliği kalıntısı) sınıfı hatalar:** Üç ayrı gerçek olayda
  (InfluxDB bucket adı, Kafka topic adı, Dashboard Docker imajı), kodda/konfigürasyonda
  yapılan bir isimlendirme değişikliği, UZUN SÜREDİR çalışan eski konteynerlere/imajlara
  yansımamıştı — yani kod güncel ama çalışan altyapı eskiydi. Üçü de kök nedeni bulup ya
  isim güncellemesi ya da imajı yeniden inşa ederek çözüldü; bu, "kod değişince altyapının
  da güncellenmesi gerektiğini" doğrulayan tekrarlayan bir ders oldu.
- **Askeri filtre görünmüyordu (sivil çalışıyordu):** Harita üzerindeki renk skalası
  mantığı, ham veriyi logaritmik dönüşüme soktuktan SONRA artık artan sırada olmayabiliyordu
  — bu, askeri trafiğin %50'sinden fazlasının sıfır olması nedeniyle neredeyse HER ZAMAN
  askeri filtreyi kırıyordu, sivil filtreyi ise neredeyse HİÇ etkilemiyordu (sivil veride
  sıfır neredeyse yok). Kök neden koddan bulundu, düzeltme her iki harita projesine de
  uygulandı.

### 7) Bugünkü Rakamlar (Sunumda Doğrudan Kullanılabilir)

- **Toplam veri:** 30 tarihsel arşiv dosyası (64,1 GB'lık 19 yeni tar dahil), Gold katmanında
  **2.762.067.817 satır**, **6.861 parça**, 4 farklı kaynak tipinde (adsb.lol tarihsel,
  adsb.lol gerçek zamanlı, ALFA, UAV Attack).
  - Yeni veri eklenmeden önceki 11 tar'a kıyasla: benzersiz uçak sayısı 254.909 → **386.779**
    (+%51,7), askeri işaretli uçak 9.308 → **12.753** (+%37,0), toplam satır +%174,2.
    Bu, veri hacmi çok artsa da yeni uçak keşfinin yavaşladığını (aynı ~11 aylık pencere
    yeniden dolduruluyor) gösteriyor — azalan getiri (diminishing returns) örneği.
- **Ülke bazlı proje:** 11.213 benzersiz uçak, 117 farklı ülke, en baskın ülke ABD (%61,2).
- **Okyanus-geçen uçuş tespiti:** Naif yöntem (sadece ülke karşılaştırması) sadece **1**
  uçuş buluyordu; gerçek büyük-daire geometrik yöntem (uçuş rotasını örnekleyip açık okyanus
  kutularıyla kesişim kontrolü) **521 bacak / 186 uçak** buldu — yöntem seçiminin sonucu ne
  kadar değiştirebileceğinin somut bir kanıtı.
- **Uçtan uca işlem süresi (Bronze→Silver→Gold→Yoğunluk):** toplam **~29 saat 40 dakika**
  işlem süresi (Bronze yükleme ~18dk, Silver 30 tar ~17sa12dk, Gold birleştirme ~1sa41dk,
  yoğunluk/askeri analiz ~10sa27dk) — yaklaşık 2,5-3 takvim günü (kesintiler dahil).
- **Yoğunluk analizi (H3 altıgen ızgara, 3 çözünürlük):** en ince çözünürlükte (res5,
  ~8,5km altıgenler) 268.234 hex, bunların 127.883'ünde (%47,7) askeri trafik gözlendi.

### 8) Test ve Doğrulama

Toplam **351/351** test (ML/torch dışı aktif test paketi) geçiyor; bunun 46'sı Bronze/Silver/
MinIO, 7'si Gold birleştirme regresyonu (çift-sayım bug'ını özellikle koruyan test dahil),
40'ı Kafka/Dashboard testleri. Testlerin gerçek bir MinIO/Kafka sunucusu GEREKTİRMEMESİ
kasıtlı bir tasarım: depolama istemcisi bir "Protocol" (yapısal arayüz) olarak tanımlandığı
için testlerde bellek-içi sahte bir istemci gerçek sunucunun yerine geçebiliyor — bu sayede
testler saniyeler içinde, Docker'a hiç ihtiyaç duymadan çalışıyor. Uçtan uca (gerçek tarayıcı,
Playwright) testlerde 13'ten sadece 1'i geçti — bu, henüz tam kök nedeni netleşmemiş, ileride
araştırılacak açık bir bulgu olarak not edildi (dashboard sunucu tarafında ayaktaydı, ama
gerçek tarayıcıda canlı veri render'ı bekleneni vermiyordu).

### 9) Sonuç ve Gelecek Çalışmalar

Bugün elimizde: sağlam, geri dönülebilir, test edilmiş bir Bronze/Silver/Gold veri hattı; hem
hazır veri seti hem gerçek zamanlı akış için çalışan, birbirinden bağımsız iki besleme yolu;
2,76 milyar satırlık birleşik bir Gold veri seti; bunun üzerine kurulu yoğunluk/askeri
haritalar, ülke bazlı analiz ve bir takım export panosu var. Açık kalan noktalar dürüstçe şu:
Gold'un gerçek zamanlı veriyi otomatik almaması (elle tetikleme gerektiriyor), 7 günlük
gerçek zamanlı sorgu modunun büyük veri hacminde hâlâ bellek sorunu yaşaması, ve E2E tarayıcı
testlerinin çoğunun henüz geçmemesi. Bunlar sonraki iterasyon için somut, ölçülebilir bir
yol haritası oluşturuyor — "her şey bitti" değil, "nerede olduğumuzu ve nereye gideceğimizi
tam olarak biliyoruz" diyebileceğimiz bir nokta.
