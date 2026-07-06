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

**ÇÖZÜLDÜ (2026-07-06) — `stream_unify()` rerun'larda eski Gold'u siliyordu, artık siliyor:**
Bug: `stream_unify()` (ve `main()`'in `--local-out` yolu) her çalıştırmada
`write_gold()` ile YENİ, zaman damgalı+uuid'li part dosyaları yazıyordu ama
önceki çalıştırmanın `gold/unified/` altındaki part'larını hiç silmiyordu —
tekrarlanan her Gold çalıştırması satırları kümülatif olarak şişiriyordu
(N. çalıştırmadan sonra veri N katı sayılıyordu). `src/common/minio_io.py`'a
`remove_object` (Protocol'e eklendi, `FakeMinioClient`'a da eklendi) ve
`delete_layer_objects()` (bir prefix altındaki tüm objeleri siler) eklendi;
`src/gold/unify.py`'a bunları kullanan `clear_gold_before_unify()` eklendi,
hem `stream_unify()` hem `main()`'in in-memory yolu artık yazmadan önce bunu
çağırıyor. Regresyon testi: `tests/test_gold_unify.py::test_stream_unify_rerun_does_not_double_count_rows`.

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

## ADR-006: ML katmanında iki ayrı Gold çıktısı — kasıtlı tasarım

- Durum: Kabul edildi
- Tarih: 2026-07-03

Pipeline'da Gold iki farklı yoldan üretilir ve bu bir hata değil, bilinçli bir karardır:

1. **`src/gold/unify.py`** → MinIO `gold/unified/` prefix'i. 7 ortak kolon + 3 metadata
   (10 kolon toplam). Her kaynağı aynı daraltılmış şemaya hizalar; görselleştirme,
   dashboard, "bütün veri bir arada" sorgular için optimum.

2. **`src/ml/build_features.py`** → `data/gold/ml_features/`. Silver'ı doğrudan okur,
   ham feature'lar üretir (pencere istatistikleri, ölçekleme, vb.); kaynak-özgü kolonlar
   (ör. `jamming_indicator`, `squawk`, `roll_deg`) burada korunur. Gold'un daraltılmış
   şemasından daha zengindir.

**Neden iki yol?** `unify.py` ML için çok az kolon tutar; `build_features.py` ise ML
için çok fazla kaynak-spesifik bilgi taşır ve görselleştirme için kullanışsızdır. Her yol
kendi sorumluluğunu optimize eder; biri diğerinin yerine geçmez.

## ADR-007: Gerçek zamanlı veri — tek yerine iki ayrı Kafka consumer'ı

- Durum: Kabul edildi
- Tarih: 2026-07-03

Mimari diyagram tek bir consumer'ın aynı anda Redis + InfluxDB + MinIO'ya yazdığını
gösteriyordu (3 sink, 1 consumer). Uygulamada **iki ayrı consumer** tercih edildi:

- `src/ingestion/adsblol_consumer.py` — `group.id = dashboard-consumer`; Redis + InfluxDB
- `Dashboard/minio_archiver.py` — `group.id = minio-archiver`; MinIO Bronze

**Neden?** Separation of Concerns: arşivleyici çöktüğünde dashboard etkilenmiyor;
arşivleyici bağımsız ölçeklenebilir; her consumer kendi offset'ini yönetiyor.
Her iki consumer da aynı Kafka topic'i `adsb.flights`'ı bağımsız okuyor — Kafka'nın
consumer group modeli buna tasarım gereği izin veriyor, veri kaybı yok.

## ADR-008: ML-8A Gate A geçti; Gate B/C reddedildi

- Durum: Development rejected
- Tarih: 2026-07-06

ML-8A descriptor v1 prefix-invariance ve no-future testlerini geçti; uçuş/oturum ayrımı,
train-only scaler, guard-band dışlaması, holdout dışlama assert'i ve şema hash manifesti
doğrulandı. **Gate A geçti.** Mevcut manifest novelty-detection için normal-only train
ürettiğinden, kullanıcı onayıyla orijinal splitleri değiştirmeden `supervised_splits`
alanı eklendi; 5 seed'de oturum kesişimi boş ve final holdout sabittir.

SEAD 5-seed sonucunda LightGBM window AUPRC 0.349, mevcut IF 0.385 oldu ve aynı FA bütçesinde
onset recall üstünlüğü gösteremedi: **Gate B kaldı.** Hiçbir LightGBM+decision bileşimi
kritik/advisory hedefini seed ortalamasında karşılamadı: yeni model için **Gate C kaldı.** Blind
holdout açılmadı ve açılmayacak; ML-8A LightGBM model/policy production adayı değildir.

SEAD'de pre-existing LSTM-AE artifact bulunmadığından orijinal baseline satırı `N/A`dır.
Kullanıcı izniyle ayrı `ml8a_retrained_lstm_ae` bundle'ı üretildi; mevcut artifactler
overwrite edilmedi ve bu model yalnız retrained ek kıyas olarak raporlanır. Sabit reçeteli ALFA
protokolü sonradan tamamlandı: LightGBM AUPRC 0.843 < IF 0.858 < LSTM-AE 0.872. Mevcut
IF+CUSUM advisory 0.625 recall / 7.91 FA-saat ile matris düzeyinde Gate C'yi geçse de LightGBM
geçmedi. Optuna/tuning yapılmaz; sonraki model deneyi ML-8C family-holdout'tur.

Nihai sonuçlar gözleme bağlı endpoint ve `max_gap_s=2` exposure hesabı kullanan
`full_matrix_gapfix/` artifactleridir. İlk `full_matrix/` koşuları ALFA'daki 850 bin saniyelik
telemetry boşluğunu normal exposure saydığı için superseded ilan edilmiştir; model seçimi veya
Gate kararı için kullanılmaz.

### Kontrollü olarak yapılmayan “sonucu kurtarma” adımları

Gate B/C sonucu görüldükten sonra aşağıdaki işlemler **bilinçli olarak yapılmaz**; bunlar eksik
implementasyon değil, test sonucuna bakarak deney tasarımını değiştirmeyi önleyen metodoloji
kontrolleridir:

| Yapılmayan işlem | Neden şimdi yapılmıyor? | Şimdi yapılırsa ne olur? | Ne zaman yeniden açılabilir? |
|---|---|---|---|
| Descriptor v1'e feature ekleme/çıkarma | Şema sonuç görülmeden önce donduruldu. | Test setine uyarlanmış v1 üretir; raporlanan seed sonucu bağımsız kanıt olmaktan çıkar. | Yeni hipotez ve `descriptor_schema_v2` ile, yeni fazda ve önceden yazılmış kabul ölçütleriyle. |
| LightGBM hiperparametre/Optuna taraması | ML-8A sabit reçetenin skor katkısını ölçüyor. | Development test sonuçlarına aşırı uyum ve çoklu-deneme yanlılığı yaratır. | ML-8C sonrasında ayrı bir tuning protokolü, nested/group CV ve ayrı untouched evaluation setiyle. |
| Test sonucuna göre threshold/K/N/CUSUM k/h değiştirme | Policy yalnız validation-normal üzerinde kalibre edilmelidir. | Test FA/recall doğrudan optimize edilir; operasyonel genelleme ölçülemez. | Yeni policy sürümü ve yeni validation kalibrasyon setiyle; mevcut test sonucu tekrar seçim için kullanılmadan. |
| Blind holdout'u açma | Gate B/C geçmedi; model/policy seçimi tamamlanmadı. | Holdout artık blind olmaz ve sonraki model seçimini dolaylı etkiler; tek-seferlik güvence kaybolur. | Yalnız development'ta önceden tanımlı Gate B/C geçen feature/model/policy hash'leri dondurulduktan ve insan kararı kaydedildikten sonra bir kez. |
| Başarısız ML-8A modelini production adayı olarak paketleme | Kritik/advisory fayda hedefleri karşılanmadı. | Yüksek FA veya yetersiz recall operasyonel alarm yorgunluğu/güven kaybı üretir. | Yeni skor ailesi development bütçesini geçtikten sonra ayrı model sürümü olarak. |

Sabit reçeteyle önceden planlanmış ALFA koşusu, aile/fault kırılımları ve raporlama bu kapsama
girmez: bunlar sonucu kurtarmak için model değiştirme değil, aynı deney protokolünü tamamlamadır.
