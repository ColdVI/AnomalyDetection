# Mimari Kararlar

## ADR-001: Bronze veri kaynakları

- Durum: Kabul edildi
- Tarih: 2026-06-29

OpenSky yerine `adsb.lol`; generic MAVLink örnekleri yerine ALFA ve UAV Attack veri
setleri kullanılacaktır. `adsb.lol` kimlik doğrulama ve günlük kredi yükünü azaltırken
Türkiye hava trafiğini hem tarihsel hem gerçek zamanlı sağlar. ALFA fault ground-truth,
UAV Attack ise benign/malicious saldırı etiketleri sunduğundan sonraki anomali tespiti
çalışmasının ölçülebilir olmasını sağlar.

ALFA processed CSV dosyaları ground-truth için birincil Bronze girdisidir. `.bin` ve
`.tlog` dosyalarının `pymavlink` ile ayrıştırılması opsiyonel ek yoldur; ROS `.bag`
dosyaları `pymavlink` ile okunmayacaktır.

Bronze yalnızca ham alanları korur, provenance ekler ve adsb.lol kayıtlarında Türkiye
bbox filtresi uygular. Birim dönüşümü, kolon harmonizasyonu ve koordinat ölçekleme
Silver katmanına aittir.

## ADR-002: Bronze depolama MinIO'da

- Durum: Kabul edildi
- Tarih: 2026-06-30

Mimari diyagram Bronze/Silver/Gold'un tamamının MinIO (S3-uyumlu nesne deposu) üzerinde
tutulmasını öngörüyordu; Faz 1'de yazılan `src/common/io.py` ise yerel diske (`data/bronze/`)
yazıyordu. Ekip kararıyla bu değiştirildi: `write_bronze` ve `write_bronze_bytes` artık
DataFrame'i / ham byte'ları doğrudan MinIO'ya yüklüyor (bucket: `bronze`, env:
`MINIO_BRONZE_BUCKET`), hiç yerel parquet dosyası yazılmıyor. Gerçek zamanlı landing JSONL'i de
aynı şekilde MinIO'ya batch halinde yükleniyor — MinIO native "append" desteklemediği için,
landing artık parquet flush'ıyla aynı cadence'te (varsayılan 500 mesaj) batch olarak yazılıyor;
artık tek bir sürekli büyüyen yerel `.jsonl` dosyası yok.

MinIO client her fonksiyona enjekte edilebilir (`client=` parametresi), bu yüzden testler
gerçek bir MinIO sunucusu gerektirmiyor — `src/common/fakes.py` içindeki `FakeMinioClient`
in-memory bir sahte uyguluyor (`tests/conftest.py` sadece pytest fixture'ı olarak yeniden
sunuyor; `scripts/run_alfa_local.py` gibi test-dışı kod da aynı sahteyi kullanabiliyor).

## ADR-003: Bronze = raw; parse/provenance Silver'da; coğrafi filtre yok; generic parser

- Durum: Kabul edildi
- Tarih: 2026-07-01
- ADR-002'yi günceller.

Bronze katmanı ham dosyaları (orijinal .tar/.zip ve realtime ham .jsonl) MinIO'da
değiştirmeden saklar. Parquet dönüşümü, unit dönüşümü, etiket çıkarımı ve provenance
kolonları Silver katmanına taşındı. Coğrafi filtre (Türkiye bbox veya başka) pipeline
seviyesinde uygulanmaz; analiz/notebook aşamasında yapılır.

Silver'da iki tür parser var: (1) kaynak-özel parser'lar (adsb, alfa, uav_attack) —
domain-specific unit dönüşümü ve etiket çıkarımı yapar; (2) generic parser — dosya
formatını otomatik algılayıp (CSV, JSON, JSONL, zip/tar içindekiler dahil) Parquet'e
çevirir, sadece provenance ekler, domain-specific dönüşüm yapmaz. Yeni bir veri seti
eklemek için generic parser yeterlidir; özel dönüşüm gerekirse üstüne custom parser
yazılır.

Gold katmanı tüm kaynakları 7 ortak kolona (timestamp, lat, lon, altitude, velocity,
heading, vertical_rate) + 3 metadata kolonuna (source_type, source_id, label) hizalar.
Yeni veri seti eklemek için Gold'a sadece bir kolon eşleme satırı eklenir.

Sorumluluklar: adsb historical → Metehan, adsb realtime → Yusuf, ALFA + UAV Attack → Anıl.

Bu ADR'nin uygulanmasıyla değişenler (Anıl'ın bölümü, `docs/PIPELINE_PLAN.md` ANIL
REHBERİ'ne göre):
- `src/ingestion/alfa_loader.py` / `uav_attack_loader.py` (eski, Bronze'da per-topic parse
  eden tasarım) silindi; yerine `src/ingestion/upload_raw.py` (jenerik, ham dosyayı
  değiştirmeden yükler) geldi.
- `src/bronze2silverParsers/parse_alfa.py` / `parse_uav_attack.py`, transform mantığı
  DEĞİŞMEDEN `src/silver/parse_alfa.py` / `parse_uav_attack.py`'a taşındı — IO katmanı
  MinIO'dan zip indirme / Silver'a yazmaya değişti. Tek istisna: `parse_uav_attack.py`'nin
  `split_log_and_topic` regex'i gerçek dosyalarda kanıtlanmış şekilde kırıktı (log_id
  önekleri standart değil, kendi içinde alt çizgi barındırıyor — bkz. aşağıdaki not),
  bilinen 5 topic adına ankrajlı tam-eşleşme ile düzeltildi.
- `src/common/io.py` → `src/common/minio_io.py` (yeniden adlandırma; `write_bronze`/
  `write_bronze_bytes` aynı davranışta kaldı ki Metehan'ın/Yusuf'un mevcut dosyaları
  bozulmasın — sadece import satırları güncellendi).
- `src/common/bbox.py` silindi; coğrafi filtre artık hiçbir yerde yok — Metehan'ın
  (`adsblol_historical_loader.py`) ve Yusuf'un (`adsblol_consumer.py`) Türkiye-bbox
  filtreleri de kaldırıldı (fonksiyonlar `extract_turkey`→`extract_all`,
  `turkey_rows` kaldırıldı), artık tüm dünya verisi tutuluyor.

Gerçek veriyle doğrulanmış sonuçlar (`scripts/run_alfa_local.py`,
`scripts/run_uav_attack_local.py`, FakeMinioClient ile, Docker gerektirmeden):
- **ALFA**: 47 sekans parse edildi.
- **UAV Attack**: gerçek `UAVAttackData.zip`'te (683.9 MB, 767 CSV) `split_log_and_topic`'in
  eski hali (`_([a-z0-9_]+?)_(\d+)\.csv$`, soldan-sağa en erken eşleşme) gerçek log_id
  önekleriyle (`log_12_2020-8-2-14-18-24_...`, `ace-benign-log_0_...`,
  `001-2021-01-27-09-08-37-708_...`) topic'i YANLIŞ (ilk alt çizgiden) bölüyordu; düzeltilmiş
  haliyle doğrulandı.

**ÇÖZÜLDÜ (2026-07-01):** `infer_label_from_path`'e `"ping"`/`"dos"` kontrolü eklendi —
artık tüm Ping DoS satırları `label="ping_dos"` olarak doğru etiketleniyor.

## ADR-005: Gold yazıldı — `src/gold/unify.py`, 7+3 ortak şema

- Durum: Kabul edildi
- Tarih: 2026-07-01
- ADR-003'ü tamamlar (Gold kısmı).

ADR-003/`docs/PIPELINE_PLAN.md` Gold'u "ekibin Silver review'undan sonra" olarak
gated etmişti. Kullanıcı bu review'u beklemeden Gold'un şimdi yazılmasını istedi; bu ADR
bunu ve mevcut kapsamını kaydediyor.

`src/gold/unify.py`, planın verdiği tabloyla birebir aynı `COLUMN_MAPS` dict'ini kullanıyor:
her kaynak (`adsblol_hist`, `adsblol_rt`, `alfa`, `uav_attack`) için 7 ortak kolona
(`timestamp_utc`, `lat`, `lon`, `altitude_m`, `velocity_mps`, `heading_deg`,
`vertical_rate_mps`) + 3 metadata kolonuna (`source_type`, `source_id`, `label`) eşleme.
Yeni bir kaynak eklemek tek satırlık bir `COLUMN_MAPS` girdisi. `adsblol_hist`/`adsblol_rt`
için Silver henüz yazılmadığından (Metehan/Yusuf'un işi), bu iki kaynağın eşlemesi tanımlı
ama `read_layer` boş dönünce sessizce atlanıyor — hata değil, uyarı log'u.

Gerçek veriyle doğrulandı (`scripts/run_gold_local.py`, ALFA `processed.zip` + gerçek
`UAVAttackData.zip`, `FakeMinioClient` ile): **99.885 satır** (ALFA 20.239 + UAV Attack
79.646), tam olarak 10 kolon (7+3).

**BİLİNEN EKSİK (bilerek düzeltilmedi, review'da görülsün diye kaydedildi):** `velocity_mps`
her iki kaynak için de tamamen null çıkıyor — ALFA'da plan'ın önerdiği `velocity_measured`
kolonu gerçek veride hiç oluşmuyor (`nav_info-velocity` topic'i eşleşmiyor), UAV Attack'te
ise plan'ın "hesapla" dediği ham hız alanı (`vel_n`/`vel_e`/`vel_d`) Silver'da hiç yok. Bu,
Gold'un değil ilgili Silver parser'ların kapsamı; detay `docs/AGENTS.md` "Gold — durum".

## ADR-004 (referans/gelecek): Bronze'un üstünde daha zengin bir Silver denemesi

- Durum: Kabul edildi (bireysel referans kapsamda; aktif pipeline DEĞİL — bkz. ADR-003)
- Tarih: 2026-07-01

ADR-003'ten önce, Anıl'ın bireysel anomali tespiti ihtiyacıyla `src/processing/
alfa_silver.py`, `uav_attack_silver.py`, `gold.py` yazılmış ve gerçek veriyle
doğrulanmıştı (ALFA: 47 sekans / 243.455 satır / 563 kolon; UAV Attack: 19 log / 79.646
satır / 34 kolon; Gold union: 323.101 satır / 595 kolon). Bu, o zamanki (Bronze zaten
per-topic parse ediyor) mimariyi varsayıyordu.

Metehan'ın `docs/PIPELINE_PLAN.md`'i (ADR-003) mimariyi tersine çevirince (Bronze=raw,
Silver=parse), bu dosyalar ACTIVE PIPELINE'IN YERİNE GEÇMEDİ — ama silinmedi de,
referans olarak tutuluyor: ALFA/UAV Attack'in gerçek klasör yapısı, gerçek kolon adları,
ve `parse_uav_attack.py`'nin regex hatası (ADR-003'te düzeltildi) buradaki araştırmadan
geliyor. `src/silver/parse_alfa.py`/`parse_uav_attack.py`'nin şu anki dar kolon kümesi
(sadece lat/lon/alt + birkaç nav_info alanı) yeterli gelmezse (ör. `mavctrl/path_dev`,
IMU, battery, GPS-spoofing groundtruth residual feature'ları gerekirse), bu dosyalardaki
zaten-doğrulanmış geniş merge_asof mantığı zenginleştirme için hazır referans olarak
duruyor. Detay: `docs/silver_schema.md` (bu da referans, aktif değil).
