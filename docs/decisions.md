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

## ADR-009: ML-9 Gate A geçti; kategori residual'ları development'ta reddedildi

- Durum: Development rejected
- Tarih: 2026-07-06

SEAD parser ve feature katmanına geriye uyumlu olarak `ekf_alt_innov`,
`ekf_vertical_vel_innov` ve actuator-output simetri residual'ları eklendi. Aktif motor kanalı eşiği
30 development uçuşundaki gerçek dağılımdan 1 PWM seçildi; expanding aktiflik, rolling ve CUSUM
prefix-invariance testlerinden geçti. Scaler her seed'in yalnız normal train uçuşlarında, feature
CUSUM baseline'ı frozen split_00 normal train'de fit edildi. Karar katmanları ML-8A modülünden
değiştirilmeden kullanıldı. **Gate A geçti.**

Mevcut frozen manifest plan metnindeki eski 412/76 sayısından farklı olarak 611 işlenebilir uçuş ve
131 blind holdout içeriyordu. Yeni split üretilmedi veya manifest yeniden yazılmadı; SHA-256
`0e3047b3abc11d09bc5b2b94e35cca117efce8fa49fdc37393c146550b8d0f0d` kaldı. Holdout telemetry,
feature veya skor akışına alınmadı.

5 seed'de dikey modül `Position.Z` CUSUM/advisory recall'ını pooled EKF'ye göre 0.074'ten 0.096'ya,
motor-simetri modülü `Actuator Outputs+Controls` recall'ını kontrol-cevabına göre 0.180'den 0.205'e
çıkardı; kazançlar önceden dondurulan >=0.05 ve >=3/5 seed kararlılık kuralını birlikte sağlamadı.
**Gate B kaldı.** ML9 fusion CUSUM/advisory 0.222 recall / 25.83 FA-saat, kritik 0.139 / 14.49
verdi; hiçbir policy kritik/advisory fayda+FA hedefini karşılamadı. **Gate C kaldı.** Aday modüller
default'a alınmaz ve blind holdout kapalı kalır.

Sonuç görüldükten sonra motor aktiflik eşiği, feature listesi, IF parametreleri veya karar policy
grid'i değiştirilmez; bunlar mevcut development testine sonucu kurtarma amaçlı uyum olurdu. Nadir
Battery/Vibration/Magnetometer alt-tipleri modellenmez. Yeniden açılma ancak ayrı ML-10 hipotezi,
önceden yazılmış kabul ölçütleri ve yeni development protokolüyle mümkündür. Nihai artifact
`artifacts/ml9/uav_sead/full_matrix/` altındadır; manifest 65 dosyanın checksum'ını taşır.

## ADR-010: ML-10 Gate B mechanical dalında geçti; Gate C reddedildi

- Durum: Development category accepted; operational deployment rejected
- Tarih: 2026-07-06

Zorunlu preflight'ta `amazon/chronos-bolt-tiny` CPU'da gerçek veriyle yüklendi; sıcak tahmin
ortalaması 9.3 ms ölçüldü. Dikey kanal, development doluluk denetiminde `alt` %100 ile en yüksek
olduğu için sonuç görülmeden önce sabitlendi; mechanical kanal mevcut `actuator_output_std` oldu.
Sekiz gerçek uçuşluk fizibilite 480-uçuş tam geçişini 102.4 saniye öngördü ve planın `<3 saat`
kuralıyla tam development/1 s stride kararı kaydedildi. Tam zero-shot precompute 101.2 saniyede
tamamlandı; fine-tuning, gradient veya optimizer adımı yoktur.

Chronos skorları yalnız uçuşun nedensel geçmişinden üretildi, her seed'in normal-validation
dağılımında ampirik olasılığa çevrildi. ML-9 modelleri ve threshold/K-of-N/bootstrap-CUSUM karar
katmanları checksum doğrulamasıyla değişmeden yeniden kullanıldı; skor kalibrasyonu ve max-fusion
tek ortak `src/ml/evaluation/score_fusion.py` yardımcısına taşındı. Future-leak, zero-shot,
karar-katmanı kimliği, füzyon tekilliği ve holdout izolasyonu testleri geçti. **Gate A geçti.**

`Actuator Outputs+Controls` CUSUM/advisory recall'ı `motor_simetrisi` 0.205'ten
`chronos_motor` 0.390'a çıktı (+0.185, 4/5 seed); kritik CUSUM kazancı +0.112 ve yine 4/5 seed
pozitifti. Önceden dondurulan >=0.05 ve >=3/5 şartı sağlandığından **Gate B geçti.** Bu kabul yalnız
mechanical kategori skoruna aittir: `Position.Z` CUSUM/advisory 0.096'dan 0.023'e geriledi.

Sabit `ml10_fusion=max(existing_fusion, chronos_dikey, chronos_motor)` CUSUM/advisory'de
0.213 recall / 23.92 FA-saat, kritik 0.133 / 12.83 verdi. Hiçbir satır kritik veya advisory
fayda+FA hedefini karşılamadığından **Gate C kaldı.** `chronos_motor` default/production fusion'a
alınmaz ve 131-uçuş blind holdout açılmaz. Sonuç görüldükten sonra quantile bandı, context window,
fusion, stride veya policy grid'i değiştirilmez; böyle bir değişiklik yeni, önceden kayıtlı bir faz
ve bağımsız değerlendirme gerektirir. Nihai artifact `artifacts/ml10/uav_sead/full_matrix/` altındadır.

## ADR-011: ML-11 read-only görselleştirme fazı; eğitim izi kalıcı kural oldu

- Durum: Tamamlandı (analiz fazı — gate yok, kabul kriterleri sağlandı)
- Tarih: 2026-07-06

Üç dataset için veri karnesi, PCA/t-SNE projeksiyonları, Spearman + tek-feature×kategori AUC
matrisi ve mevcut artifact modellerle tanılama görselleri üretildi
(`scripts/make_visualizations.py`, `notebooks/09_gorsellestirme_ve_veri_kesfi.ipynb`,
`artifacts/viz/*/viz_manifest.json` checksum'lı). Hiçbir model/eşik/split/scaler/CUSUM
artifact'ı değişmedi; 131-uçuş SEAD blind holdout'u hiçbir figüre/istatistiğe girmedi
(`tests/test_ml11_viz.py` development-küme hash'iyle assert eder).

Kararlar: (1) Projeksiyon figürlerinde RobustScaler çıktısına ±10 IQR **görsel** kırpma
uygulanır — tek uç değer PCA'yı dejenere ediyordu; kırpma hiçbir skor/modele girmez ve figür
başlığında beyan edilir. (2) UMAP/seaborn KURULMADI (numpy düşürme riski; PCA+t-SNE yeterli).
(3) `train_lstm_autoencoder` epoch başına train/val loss geçmişi döndürür ve
`src/ml/training_log.py` bunu `artifacts/training_logs/<source>/<model>/<run_id>/loss.csv`+PNG
olarak yazar; iki LSTM paketleme scripti çağrıyı içerir — bundan sonra eğitilen her model iz
bırakır. RNG tüketimi ve early-stopping kararı bire bir aynı kaldı (davranış değişikliği yok).
(4) Tek-feature AUC'ler keşif çıktısıdır: aday feature'larla aynı veri üzerinde sonuç
raporlanmaz; değerlendirme ancak yeni, ön-kayıtlı bir Gate turunda yapılır. (5) D.1'deki
"~32 oturum" tahmini bu fazda 64 (dev: 49) olarak düzeltildi.

## ADR-012: ML-12 ince-modül Gate B geçti (B1+B2); fusion Gate C yine reddedildi

- Durum: Development category accepted; operational deployment rejected
- Tarih: 2026-07-07

ML-11'in seyrelme hipotezi (H26) ön-kayıtlı iki adayla test edildi (`docs/ML12_INCE_MODUL_PLAN.md`;
listeler, hiperparametreler, füzyon tanımları ve Gate kuralları sonuç görülmeden sabitlendi).
Donmuş ML-9 split scaler'ları ve modelleri checksum doğrulamasıyla yeniden kullanıldı;
`motor_simetrisi`/`existing_fusion` (ML-9) ve `chronos_motor`/`ml10_fusion` (ML-10) baseline
satırları yeniden hesaplanmadan donmuş CSV'lerden alındı ve testle bire bir eşitliği kanıtlandı.
Blind holdout (131 uçuş) hiçbir aşamada okunmadı. **Gate A geçti.**

Tek-feature `itki_komutu` (actuator_thrust_cmd) Actuator Outputs+Controls'te 6 policy/bütçe
kombinasyonunun 5'inde anlamlı kazanç verdi: `motor_simetrisi`ne karşı CUSUM/advisory 0.205→0.459
(+0.254, 4/5 seed), K-of-N/advisory +0.332 (5/5); `chronos_motor`a karşı CUSUM/advisory
0.390→0.459 (+0.068, 3/5), CUSUM/critical +0.132 (5/5). **Gate B hem B1 (gate'i belirleyen) hem
B2'den (bilinen-en-iyi) geçti** — kategori için bilinen en iyi skor artık ince modül.
3-feature `itki_kontrol_ince` her yerde tek-feature'ın altında kaldı: seyrelme 3 feature'da bile
ölçülür; geniş-modül mimarisinin kategori zafiyetindeki payı deneysel olarak doğrulandı.

`ml12_fusion_itki` CUSUM/advisory 0.217 recall / 23.74 FA-saat verdi (hedef ≥0.50 @ ≤12) — mevcut
ve ML-10 füzyonlarıyla pratikte aynı. Kök neden ölçüldü: ince modül normal uçuşlarda 38.1 FA-saat
bırakan bir kategori uzmanı; max-füzyon böyle bir uzmanı hedef bütçede kullanamıyor. **Gate C
kaldı**; holdout açılmaz, `itki_komutu` production füzyona alınmaz. Sonuç görüldükten sonra modül
listesi, füzyon veya policy grid'i değiştirilmez; kategori-bazlı ayrı alarm kanalı gibi bir
mimari ancak yeni, ön-kayıtlı bir fazla değerlendirilebilir. Artifact:
`artifacts/ml12/uav_sead/full_matrix/`.

## ADR-013: ML-13 iki alarm kanalı mimarisi recall kazandı ama FA şartında reddedildi

- Durum: Development rejected; operational deployment rejected
- Tarih: 2026-07-07

ML-12'nin H30 bulgusu ("`itki_komutu` güçlü bir kategori uzmanı ama max-füzyonda FA yükü yüzünden
eriyor") üzerine ML-13 ön-kayıtlı iki-kanal mimarisi uygulandı
(`docs/ML13_KANAL_MIMARISI_PLAN.md`). Sistem kanalı donmuş ML-9 `existing_fusion`, mekanik kanal
ML-12'nin kayıtlı `itki_komutu` split modelleridir. Skorlayıcı model eğitimi yapılmadı; ML-9/ML-12
manifest checksum'ları doğrulandı, 131-uçuş blind holdout okunmadı, decision layer ve
`event_metrics` fonksiyonları değiştirilmeden kullanıldı. Kanal birleşimi, aynı 1 s karar kovasında
çift tetikleri tek operatör bildirimi sayan boolean onset OR'udur. **Gate A geçti.**

Önceden sabitlenen üç bütçe bölüşümü değerlendirildi: advisory 10+2 / 8+4 / 6+6 ve critical
1.67+0.33 / 1.33+0.67 / 1+1. Birleşik kanal en iyi tek-kanal baseline'a karşı recall kazancı verdi:
`dengeli` CUSUM/advisory 0.217→0.291 (+0.074, 5/5 seed), CUSUM/critical 0.137→0.212 (+0.075,
5/5), K-of-N/advisory 0.016→0.122 (+0.105, 5/5). Ancak tüm bu anlamlı recall artışları FA'yı
ön-kayıtlı 1.10x freninin üstüne taşıdı: CUSUM/advisory 23.70→44.70 FA-saat (1.89x),
CUSUM/critical 12.98→27.49 (2.12x), K-of-N/advisory 3.29→12.60 (3.83x). **Gate B kaldı**:
kazanım FA şişirerek satın alındı.

Operasyonel Gate C1 de geçmedi. En iyi birleşik advisory recall 0.291 / 44.70 FA-saat
(hedef ≥0.50 @ ≤12), en iyi critical recall 0.212 / 27.49 FA-saat (hedef ≥0.30 @ ≤2). Bütçe-içi
advisory satırın recall'u yalnız 0.106 / 11.16 oldu. Sınırlı Gate C2 de geçmedi: mekanik kanal
Actuator Outputs+Controls için `dengeli` K-of-N/advisory'de 0.541 recall üretti ama 12.60 FA-saat
ile 12 sınırını aştı; `esit` threshold/advisory 0.498 / 11.16 ile recall eşiğinin az altında kaldı.
Dolayısıyla "mekanik-özel monitör" iddiası da development'ta kanıtlanmış sayılmaz.

Karar: ML-13 mimarisi production'a alınmaz, blind holdout açılmaz, bütçe bölüşümü/policy/OR
semantiği sonuç görülerek değiştirilmez. Artifact `artifacts/ml13/uav_sead/full_matrix/` altındadır;
`tests/test_ml13.py` 8/8 geçti, Gate B/C sayıları ham CSV'lerden bağımsız yeniden türetildi ve MinIO
hariç geniş test paketi 186 passed / 22 deselected olarak tamamlandı.

## ADR-014: RFLY-0 resmi düşük-normal kotayla kabul edildi; pooled SEAD+RFLY operasyonel geçmedi

- Durum: RFLY-only development accepted; pooled-normal operational rejected
- Tarih: 2026-07-09

RflyMAD indirimi tamamlandıktan sonra gerçek havuz 490 uçuş olarak sabitlendi:
Real-Motor 242, Real-Sensors 197, Real-No_Fault 51. Kaggle mirror/listing ve
Bronze/Silver sayımı normal tavanının 51 olduğunu gösterdiği için RFLY-0 §3'e
sonuç görülmeden R1 amendmanı eklendi: resmi ilk kota 30/30 yerine 12/12
normal val/test ve 27 normal train.

Pooled SEAD+RFLY deneyinde yapısal şema farkından doğan "hayalet imputation"
hatası kapatıldı: bir kaynakta hiç bulunmayan kolon artık başka kaynağın train
medyanıyla doldurulmuyor; ilgili modül skoru o kaynak satırında NaN kalıyor ve
fusion mevcut skorlarla devam ediyor. RFLY gold iki geçişli CUSUM ile üretildi;
ALFA/UAV Attack/UAV-SEAD parquet hash'leri ve manifest içindeki mevcut kaynak
içerikleri değişmedi.

Resmi RFLY-only full matrix: Gate R-A geçti, Gate R-B geçti, Gate R-C geçti.
En iyi operasyonel satır `itki_komutu` threshold/critical: 0.573 recall /
1.23 FA-saat; advisory CUSUM 0.749 recall / 10.18 FA-saat ile de bütçe içinde.
Pooled SEAD+RFLY full matrix: Gate R-A ve R-B geçti, Gate R-C kaldı; en iyi
advisory recall 0.388 ama 12.89 FA-saat ile 12 sınırını aştı. Karar: RFLY-only
motor/sensör hattı development'ta kabul edilir; pooled-normal varyantı production
iddiası yapmaz.


### ADR-014 correction note: RFLY-1 interval-truth audit

RFLY-1 found that the RFLY-0 official RFLY-only and pooled metrics above used a
whole-flight anomaly proxy for `Real-*` anomalous flights. That proxy is easier
than the SEAD event-onset task and overstates RFLY performance; the old `0.749`
CUSUM/advisory and `0.573` threshold/critical RFLY-only rows are retained for
audit history only and are now invalid for product-claim evidence.

`parse_rflymad.py` now extracts `(fault_onset_s, fault_end_s)` from the ULog
`rfly_ctrl_lxl` message and the evaluator uses only that interval truth. Five
folder-labeled motor-fault flights had `rfly_ctrl_lxl_no_active_fault`; the
internal control message indicates no active fault trigger, contradicting the
folder label. They were excluded as ambiguous/invalid test cases before any
corrected recall/FA result was produced: 4 from development/test and 1 from
RFLY final holdout (the holdout case remained unopened). They count neither
anomalous nor normal. Artifact: `artifacts/rfly1/interval_truth_report.json`.

Corrected full 5-seed RFLY-only official run:
`artifacts/rfly0/rflymad/official_full_interval_excluding_invalid`.
Gate R-A passed and Gate R-B passed, but Gate R-C failed. Best overall row:
`itki_komutu` CUSUM/advisory `0.526` recall / `22.28` FA-hour. Best critical
recall row: `itki_komutu` CUSUM/critical `0.442` recall / `9.23` FA-hour.
No row met `critical >=0.30 @ <=2` or `advisory >=0.50 @ <=12`.

Corrected full 5-seed pooled SEAD+RFLY official run:
`artifacts/rfly0/pooled_sead_rfly/official_full_interval_excluding_invalid`.
Gate R-A passed and Gate R-B passed, but Gate R-C failed. Best overall row:
`itki_komutu` CUSUM/advisory `0.149` recall / `30.00` FA-hour. Best budget-inside
advisory row among nonzero recalls was `rfly0_fusion` CUSUM/advisory `0.066`
recall / `9.39` FA-hour; best budget-inside critical row was `rfly0_fusion`
CUSUM/critical `0.018` recall / `1.72` FA-hour. No production/operational claim
is accepted from RFLY-0 after interval-truth correction.

## ADR-015: ML-15 drift kalibrasyonu smoke edildi; full koşu maliyeti ayrı planlanmalı

- Durum: Smoke complete; full matrix pending
- Tarih: 2026-07-09

`src/ml/decision/drift_calibration.py` oturum-jackknife drift çarpanı ile mevcut
Threshold/K-of-N/CUSUM policy sınıflarını saran saf bir kalibrasyon katmanı olarak
eklendi; karar katmanı yeniden yazılmadı. Birim testler determinism, floor/cap,
fallback, policy-class identity ve düzeltme yönünü doğruluyor.

`scripts/run_ml15_calibrated_evaluation.py` ML-14 full matrix girdisi üzerinde
`calibration in {none, drift_corrected}` satırlarını aynı CSV şemasında üretir.
Smoke `split_00` tamamlandı ve Gate A geçti; Gate B/C smoke-only olarak raporlandı.
Tek split koşusu yaklaşık 45 dakika sürdüğü için 5-seed full matrix bu oturumda
başlatılmadı; maliyetin nedeni her karar/bütçe/skor için oturum-jackknife ve
CUSUM bootstrap tekrarlarıdır. Parametreler (q=0.75, floor=1.0, cap=5.0,
min_sessions=4) değiştirilmedi.

## ADR-016: ML-16 Kol L — SEAD LSTM-AE güncel splitlerde yeniden eğitildi, resmi hatta
kablolandı; Gate A geçti, Gate B kaldı; skor büyük ölçüde ham genlik-baskın çıktı

- Durum: Development rejected (Gate B); ayrı, önemli bir dürüstlük bulgusu ile birlikte
- Tarih: 2026-07-09

`src/ml/models/lstm_autoencoder.py` (mimari/eğitim döngüsü DEĞİŞTİRİLMEDEN) SEAD için
GÜNCEL `split_manifest.json` (899 normal uçuş, split_00..split_04, development 1044 uçuş,
blind holdout 200 uçuş) üzerinde 5-seed protokolde yeniden eğitildi ve resmi
ml14/ml15 fusion+karar-katmanı hattına kablolandı
(`scripts/run_ml_lstm_sead_evaluation.py`, ön-kayıt: `docs/ML16_KOL_L_LSTM_SEAD_PLAN.md`).
Pencereleme `src/ml/data/windowing.py::build_windows` (window=50, stride=5) DEĞİŞTİRİLMEDEN
kullanıldı; pencere-sonu skorundan 1 sn karar-anına hizalama ML-8A'nin zaten var olan
`_align_score` (`pd.merge_asof(..., direction="backward")`) konvansiyonuyla yapıldı. Üç
skor varyantı ön-kayıtlıydı: (a) `lstm_recon` tek başına, (b) `lstm_recon` ile `ml14_fusion`
max-füzyonu, (c) `lstm_recon` ile `itki_komutu` max-füzyonu.

**Gate A (güvenlik+determinizm): GEÇTİ.** Blind holdout (200 uçuş) hiçbir aşamada
okunmadı. `existing_fusion`/`itki_komutu`/`ml14_fusion` ara sütunları AYNI kodla
(`fit_modular_iforest`, `_score_modules`, `max_score_fusion`) yeniden hesaplandı ve donmuş
`artifacts/ml14/uav_sead/full_matrix` CSV'siyle 90/90 satırda max_abs_diff=7.1e-15 ile
BİREBİR örtüştü (kayan-nokta belirsizliği düzeyinde).

**Gate B (operasyonel hedef): KALDI.** Hiçbir {skor × karar × bütçe} hücresi hedefi
karşılamadı (critical ≥0.30 recall @ ≤2 FA-saat, advisory ≥0.50 recall @ ≤12 FA-saat):

| Skor | Karar | Bütçe | Recall | FA-saat |
|---|---|---|---|---|
| lstm_recon | cusum | advisory | 0.251 | 15.40 |
| lstm_recon | cusum | critical | 0.083 | 3.44 |
| lstm_recon | k_of_n | advisory | 0.204 | 15.50 |
| lstm_recon | k_of_n | critical | 0.190 | 4.30 |
| lstm_recon | threshold | advisory | 0.237 | 16.67 |
| lstm_recon | threshold | critical | 0.219 | 2.92 |
| lstm_ml14_fusion | cusum | advisory | 0.124 | 10.02 |
| lstm_ml14_fusion | cusum | critical | 0.043 | 1.65 |
| lstm_ml14_fusion | k_of_n | advisory | 0.057 | 2.30 |
| lstm_ml14_fusion | k_of_n / threshold | critical / advisory / critical | 0.000 | 0.00 |
| lstm_itki_fusion | cusum | advisory | 0.259 | 25.42 |
| lstm_itki_fusion | cusum | critical | 0.099 | 5.05 |
| lstm_itki_fusion | k_of_n | advisory | 0.179 | 10.93 |
| lstm_itki_fusion | k_of_n | critical | 0.147 | 3.27 |
| lstm_itki_fusion | threshold | advisory | 0.162 | 13.52 |
| lstm_itki_fusion | threshold | critical | 0.098 | 4.43 |

Karşılaştırma (mevcut en iyi, ÖNCEDEN VAR, `artifacts/ml14/uav_sead/full_matrix/metrics.csv`):
`ml14_fusion` CUSUM/advisory recall 0.126 / FA-saat 9.95; CUSUM/critical recall 0.043 /
FA-saat 1.60; K-of-N/advisory recall 0.056 / FA-saat 2.28 (bütçe içi); `itki_komutu` (genel)
CUSUM/advisory recall 0.145 / FA-saat 35.67 (bütçe dışı). `lstm_recon` tek başına ham
recall'da mevcut en iyiyi ~2x geçiyor (ör. CUSUM/advisory 0.251 vs 0.126) ama FA de orantılı
artıyor (15.40 vs 9.95) ve HİÇBİR hücre bütçe içinde kalmıyor. Bütçe İÇİNDE kalan tek yeni
hücreler `lstm_ml14_fusion` K-of-N/advisory (FA 2.30, recall 0.057 — mevcut en iyiyle
pratikte aynı) ve `lstm_itki_fusion` K-of-N/advisory (FA 10.93, recall 0.179 — mevcut bütçe
içi en iyiden [0.056] yüksek ama hedefin [0.50] hâlâ çok altında).

**KRİTİK DÜRÜSTLÜK BULGUSU (koordinatörün çapraz-model tutarlılık kontrolü üzerine, sonuç
raporlanmadan önce araştırıldı):** Bağımsız olarak eğitilmiş Dense-AE ve USAD ajanları
(aynı `_align_score`/pencereleme konvansiyonunu paylaşan), split_00'da `threshold`/`critical`
kararında BİREBİR aynı kategori-bazlı detected/false-alarm sayılarını üretti
(`altitude_anomaly` 1/114, `external_position_anomaly` 70/185, `global_position_anomaly`
6/51, `mechanical_fault` 1/43 — FA sayısı yalnız 7 vs 8 farklı). Üç mimari açısından farklı
model ailesinin aynı sonucu üretmesi, "hepsi eşit derecede iyi öğrendi" ile açıklanamaz.
Araştırma (`scripts/diagnose_ml_lstm_sead_magnitude_domination.py`,
`artifacts/ml_lstm_sead/uav_sead/full_matrix/magnitude_domination_diagnostic.json`) kök
nedeni buldu: eğitilmiş modelin skor SIRALAMASI, tamamen eğitilmemiş (rastgele
başlatılmış) aynı mimari ile Spearman ρ=0.964, model içermeyen saf `‖x‖²` genlik
taban-çizgisiyle ρ=0.965 KORELE — yani eğitim, "girdi ne kadar büyük" ötesine neredeyse
hiçbir şey KATMIYOR. `RobustScaler` (proje çapında, değiştirilmeden kullanılan) aykırı
değerleri kırpmıyor; bu yüzden bir avuç aşırı-genlikli pencere (gerçek GPS-sahtekârlığı
sıçramaları VE en az bir "normal" etiketli ama donmuş-GPS/`eph`≈25000 sentinel içeren uçuş)
herhangi bir sınırlı-çıktılı autoencoder'ın reconstruction hatasına — mimariden bağımsız —
hakim oluyor. `ThresholdPolicy` ayrıca ayrıca doğrudan test edildi ve DEJENERE DEĞİL ("ilk
tanımlı skorda ateşle" davranışı yok — 904 test uçuşunun yalnız 94'ünde alarm var, bunların
sadece %4.3'ü pencere-tamamlanma anıyla çakışıyor); sorun karar katmanında değil, ölçekleme/
özellik seçiminde. **Sonuç:** yukarıdaki `lstm_recon` recall rakamları (özellikle mevcut
en iyiyi geçen CUSUM/advisory 0.251) kısmen ÖĞRENİLMİŞ zamansal örüntüden değil, ham genlik
aykırı-değer tespitinden geliyor olabilir — bu, "LSTM sinyal katıyor" şeklinde
sunulamaz. Metodoloji disiplini gereği (sonuç görüldükten sonra parametre değişikliği yok)
ölçekleme/özellik seçimi bu koşuda DEĞİŞTİRİLMEDİ; bulgu olduğu gibi raporlanıyor.

Karar: Gate B kaldığı için holdout AÇILMAZ, hiçbir varyant production füzyona alınmaz.
`docs/ML_YETERSIZLIKLER_KAYDI.md` C.1 bu sonuçla güncellendi. Genlik-baskınlığı bulgusu
yeni, ayrı bir açık madde olarak kaydedildi (bkz. B.5) — kırpmalı/robust ölçekleme veya
genlik-normalize edilmiş bir reconstruction skoru ancak yeni, ön-kayıtlı bir turda
değerlendirilebilir; bu oturumda post-hoc düzeltme yapılmadı. Artifact:
`artifacts/ml_lstm_sead/uav_sead/full_matrix/`; `tests/test_lstm_sead_integration.py`
13/13 geçti.

## ADR-017: ML-16 Kol D — SEAD Dense-AE (düz otokodlayıcı) güncel splitlerde eğitildi,
resmi hatta kablolandı; Gate A geçti, Gate B kaldı; aynı genlik-baskınlığı bulgusu geçerli

- Durum: Development rejected (Gate B); ADR-016'daki dürüstlük bulgusu bu koşu için de geçerli
- Tarih: 2026-07-09/10

`src/ml/models/dense_autoencoder.py` (yeni: düz/ileri-beslemeli, pencereyi tek vektöre
düzleştiren otokodlayıcı, `LSTMAutoencoder` ile karşılaştırılabilir parametre sayısı) SEAD
için GÜNCEL split_manifest üzerinde 5-seed protokolde eğitildi ve ADR-016 ile AYNI resmi
ml14/ml15 fusion+karar-katmanı hattına, AYNI `_align_score`/pencereleme (window=50,
stride=5) konvansiyonuyla kablolandı (`scripts/run_ml_dense_ae_sead_evaluation.py`,
ön-kayıt: `docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md`). Üç skor varyantı ön-kayıtlıydı:
(a) `dense_ae_recon` tek başına, (b) `ml14_fusion` ile max-füzyonu, (c) `itki_komutu` ile
max-füzyonu.

**Gate A (güvenlik+determinizm): GEÇTİ.** Blind holdout (200 uçuş) hiçbir aşamada
okunmadı. Ara sütunlar donmuş `artifacts/ml14/uav_sead/full_matrix` CSV'siyle 90/90 satırda
max_abs_diff=7.1e-15 ile birebir örtüştü.

**Gate B (operasyonel hedef): KALDI.** Hiçbir hücre hedefi karşılamadı:

| Skor | Karar | Bütçe | Recall | FA-saat |
|---|---|---|---|---|
| dense_ae_recon | cusum | advisory | 0.249 | 13.91 |
| dense_ae_recon | cusum | critical | 0.084 | 3.33 |
| dense_ae_recon | k_of_n | advisory | 0.214 | 14.40 |
| dense_ae_recon | k_of_n | critical | 0.174 | 4.39 |
| dense_ae_recon | threshold | advisory | 0.238 | 15.58 |
| dense_ae_recon | threshold | critical | 0.215 | 2.77 |
| dense_ae_ml14_fusion | cusum | advisory | 0.124 | 10.03 |
| dense_ae_ml14_fusion | cusum | critical | 0.043 | 1.65 |
| dense_ae_ml14_fusion | k_of_n | advisory | 0.057 | 2.30 |
| dense_ae_ml14_fusion | k_of_n/threshold | critical/advisory/critical | 0.000 | 0.00 |
| dense_ae_itki_fusion | cusum | advisory | 0.252 | 25.95 |
| dense_ae_itki_fusion | cusum | critical | 0.102 | 5.00 |
| dense_ae_itki_fusion | k_of_n | advisory | 0.174 | 10.75 |
| dense_ae_itki_fusion | k_of_n | critical | 0.127 | 3.22 |
| dense_ae_itki_fusion | threshold | advisory | 0.164 | 13.45 |
| dense_ae_itki_fusion | threshold | critical | 0.107 | 3.48 |

Karşılaştırma (mevcut en iyi, ÖNCEDEN VAR): `ml14_fusion` CUSUM/advisory recall 0.126 /
FA-saat 9.95; CUSUM/critical recall 0.043 / FA-saat 1.60; K-of-N/advisory recall 0.056 /
FA-saat 2.28 (bütçe içi). `dense_ae_recon` tek başına ham recall'da mevcut en iyiyi
geçiyor (ör. threshold/critical 0.215 vs 0.043) ama HİÇBİR hücre bütçe içinde kalmıyor.

**Dürüstlük bulgusu ADR-016 ile PAYLAŞILIYOR, tekrar üretilmedi.** ADR-016'daki çapraz-model
tutarlılık bulgusu (split_00'da LSTM/Dense-AE/USAD'ın `threshold`/`critical`'da birebir aynı
kategori-bazlı detected-event sayıları üretmesi) ve kök-neden teşhisi
(`artifacts/ml_lstm_sead/uav_sead/full_matrix/magnitude_domination_diagnostic.json`:
eğitilmiş-vs-rastgele-başlatılmış Spearman ρ=0.964, eğitilmiş-vs-ham-genlik ρ=0.965) bu
model için de AYNEN geçerli — mekanizma mimariden bağımsız (proje-çapında kırpmasız
`RobustScaler` + bir avuç aşırı-genlikli pencere/sentinel-değer). Bu yüzden yukarıdaki
`dense_ae_recon` recall rakamları da "Dense-AE sinyal öğrendi" diye SUNULAMAZ.

Karar: Gate B kaldığı için holdout AÇILMAZ. `docs/ML_YETERSIZLIKLER_KAYDI.md` C.1
güncellendi; B.5 (genlik-baskınlığı) bu koşu için de referans veriyor, tekrar açılmadı.
Artifact: `artifacts/ml_dense_ae_sead/uav_sead/full_matrix/`;
`tests/test_dense_ae_sead_integration.py` geçti.

## ADR-018: ML-16 Kol U — SEAD USAD (adversarial çift-otokodlayıcı) güncel splitlerde
eğitildi, resmi hatta kablolandı; Gate A geçti, Gate B kaldı; aynı genlik-baskınlığı
bulgusu geçerli

- Durum: Development rejected (Gate B); ADR-016'daki dürüstlük bulgusu bu koşu için de geçerli
- Tarih: 2026-07-09/10

`src/ml/models/usad.py` (yeni: paylaşımlı kodlayıcı E + iki kod-çözücü D1/D2, yayınlanmış
USAD iki-fazlı adversarial kayıp formülasyonu) SEAD için GÜNCEL split_manifest üzerinde
5-seed protokolde eğitildi ve ADR-016/017 ile AYNI resmi hatta, AYNI `_align_score`/
pencereleme konvansiyonuyla kablolandı (`scripts/run_ml_usad_sead_evaluation.py`,
ön-kayıt: `docs/ML16_KOL_U_USAD_SEAD_PLAN.md`). Üç skor varyantı ön-kayıtlıydı:
(a) `usad_score` tek başına, (b) `ml14_fusion` ile max-füzyonu, (c) `itki_komutu` ile
max-füzyonu.

**Gate A (güvenlik+determinizm): GEÇTİ.** Blind holdout (200 uçuş) hiçbir aşamada
okunmadı. Ara sütunlar donmuş `artifacts/ml14/uav_sead/full_matrix` CSV'siyle 90/90 satırda
max_abs_diff=7.1e-15 ile birebir örtüştü.

**Gate B (operasyonel hedef): KALDI.** Hiçbir hücre hedefi karşılamadı:

| Skor | Karar | Bütçe | Recall | FA-saat |
|---|---|---|---|---|
| usad_score | cusum | advisory | 0.242 | 14.21 |
| usad_score | cusum | critical | 0.080 | 3.18 |
| usad_score | k_of_n | advisory | 0.208 | 15.61 |
| usad_score | k_of_n | critical | 0.194 | 4.24 |
| usad_score | threshold | advisory | 0.233 | 16.86 |
| usad_score | threshold | critical | 0.217 | 2.87 |
| usad_ml14_fusion | cusum | advisory | 0.124 | 10.02 |
| usad_ml14_fusion | cusum | critical | 0.043 | 1.66 |
| usad_ml14_fusion | k_of_n | advisory | 0.057 | 2.31 |
| usad_ml14_fusion | k_of_n/threshold | critical/advisory/critical | 0.000 | 0.00 |
| usad_itki_fusion | cusum | advisory | 0.253 | 24.96 |
| usad_itki_fusion | cusum | critical | 0.097 | 4.78 |
| usad_itki_fusion | k_of_n | advisory | 0.177 | 10.90 |
| usad_itki_fusion | k_of_n | critical | 0.142 | 2.23 |
| usad_itki_fusion | threshold | advisory | 0.163 | 13.47 |
| usad_itki_fusion | threshold | critical | 0.099 | 3.86 |

Karşılaştırma (mevcut en iyi, ÖNCEDEN VAR): `ml14_fusion` CUSUM/advisory recall 0.126 /
FA-saat 9.95; CUSUM/critical recall 0.043 / FA-saat 1.60; K-of-N/advisory recall 0.056 /
FA-saat 2.28 (bütçe içi). `usad_score` tek başına ham recall'da mevcut en iyiyi geçiyor
(ör. threshold/critical 0.217 vs 0.043) ama HİÇBİR hücre bütçe içinde kalmıyor.

**Dürüstlük bulgusu ADR-016/017 ile PAYLAŞILIYOR, tekrar üretilmedi.** Aynı çapraz-model
tutarlılık bulgusu ve kök-neden teşhisi
(`artifacts/ml_lstm_sead/uav_sead/full_matrix/magnitude_domination_diagnostic.json`) bu
model için de geçerli — USAD'ın adversarial eğitimi de kırpmasız `RobustScaler`'ın ürettiği
aşırı-genlikli pencerelerin baskınlığını aşamıyor. `usad_score` recall rakamları "USAD
sinyal öğrendi" diye SUNULAMAZ.

Karar: Gate B kaldığı için holdout AÇILMAZ. `docs/ML_YETERSIZLIKLER_KAYDI.md` C.1
güncellendi; B.5 bu koşu için de referans veriyor. Artifact:
`artifacts/ml_usad_sead/uav_sead/full_matrix/`; `tests/test_usad_sead_integration.py`
geçti.

**Üç kolun (L/D/U) ortak sonucu:** SEAD'de denenen üç derin-öğrenme mimarisi de (LSTM-AE,
Dense-AE, USAD) operasyonel hedefi geçemedi VE üçünün de recall kazancı büyük ölçüde aynı
genlik-baskınlığı artefaktından geliyor — mimari seçimi (tekrarlayan/düz/adversarial) bu
veri setinde şu anki ölçekleme ile ayırt edici değil. Sonraki adım (yeni, ön-kayıtlı bir
turda) kırpmalı/robust ölçekleme veya genlik-normalize skor olmalı; bu üç mimariyi
TEKRAR AYNI ölçeklemeyle koşmak bilgi kazandırmaz.

## ADR-019: ML-16 Kol N — genlik-normalize skor denemesi (yeniden eğitim yok); genlik-
bağımlılığı gerçekten kırıldı ama altından gerçek sinyal çıkmadı

- Durum: Development rejected (Gate B); B.5'i kapatmıyor, tersine doğruluyor
- Tarih: 2026-07-10

ADR-016/017/018'in ortak bulgusuna (B.5 — üç mimarinin de reconstruction skoru kırpılmamış
`RobustScaler` genliğine hâkim) doğrudan cevap: mevcut 3 dondurulmuş model (LSTM-AE,
Dense-AE, USAD split_00..04 checkpoint'leri) YENİDEN EĞİTİLMEDEN, ham reconstruction
hatasının skora çevrilme biçimi değiştirilerek 2 yeni skor türetildi (ön-kayıt:
`docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md`, kod: `src/ml/evaluation/
magnitude_normalized_scoring.py`, `scripts/run_ml16_kol_n_magnitude_normalized_scoring.py`):
(a) **bağıl hata** `|x-x̂|/(|x|+ε)`, kanal başına, (b) **kanal-başına yüzdelik-sıra** —
her kanalın hatası KENDİ train-normal dağılımına göre percentile'a çevrilip ortalanıyor.
`masked_mse` DEĞİŞTİRİLMEDİ; yanına saf-ekleme `masked_mse_per_channel` eklendi (toplamı
`masked_mse` ile birebir eşleştiği testle kanıtlı).

**Gate A (güvenlik+determinizm): GEÇTİ** — 3 mimari × 5 split, kör holdout okunmadı,
hiçbir `.fit(`/eğitim çağrısı yok (statik testle kanıtlı).

**Gate B (operasyonel hedef): KALDI** — 3 mimari × 2 varyant × 3 karar × 2 bütçe, HİÇBİR
hücre geçmedi. En iyi hücreler (5-seed ortalama): `dense_ae_rankpct` cusum/advisory 0.287
recall/16.9 FA-saat; `usad_rankpct` cusum/advisory 0.287/19.3; `lstm_rankpct` threshold/
advisory 0.270/40.6 — hepsi bütçe dışı. `relerr` varyantı genelde daha da düşük recall
verdi (ör. `lstm_relerr` threshold/critical 0.024/5.65).

**ASIL SONUÇ, sayılardan daha önemli — genlik-bağımlılığı ölçümü (5 split ortalaması,
tam tablo `artifacts/ml16_kol_n/summary.json`):**

| Mimari | Varyant | Eğitilmiş-vs-rastgele ρ | Eğitilmiş-vs-genlik ρ |
|---|---|---|---|
| LSTM | rankpct | ~0.81 (0.71-0.91) | ~0.79 (0.67-0.92) |
| LSTM | relerr | ~0.15 (-0.18/+0.43) | ~-0.09 (-0.42/+0.35) |
| Dense-AE | rankpct | ~0.83 (0.68-0.90) | ~0.83 (0.71-0.94) |
| Dense-AE | relerr | ~0.55 (0.33-0.68) | ~0.23 (-0.04/+0.53) |
| USAD | rankpct | ~0.72 (0.38-0.91) | ~0.80 (0.50-0.97) |
| USAD | relerr | ~0.37 (0.18-0.56) | ~0.13 (-0.33/+0.38) |

(Baseline, ADR-016: ham skor için ρ≈0.96-0.97, üç mimaride de.)

**Yorum:** `rankpct` genlik-bağımlılığını neredeyse hiç azaltmadı (hâlâ 0.7-0.9 civarı) —
recall'ı korudu ama sorunu çözmedi. `relerr` genlik-bağımlılığını GERÇEKTEN kırdı (3
mimaride de 0.96'dan ~0.15-0.55'e, bazı split'lerde negatife düştü) — ama bunun karşılığında
recall neredeyse yok oldu (çoğu hücrede critical recall <0.06). **Bu, ADR-016/017/018'deki
"ham recall kazancının çoğu genlik artefaktı" hipotezini DOĞRULUYOR**: genliği gerçekten
temizleyince altından operasyonel olarak kullanılabilir bir sinyal çıkmadı. Üç mimarinin de
bu özellik setiyle SEAD'de gerçekten öğrendiği şey sınırlı; sorun mimari seçimi değil.

Karar: Gate B kaldığı için holdout AÇILMAZ. `docs/ML_YETERSIZLIKLER_KAYDI.md` B.5
güncellendi — madde KAPANMADI ama artık "kırpma/normalize skor bunu tek başına çözmez,
altta yatan sinyal zaten zayıf" şeklinde daha kesin bir teşhisle kayıtlı. Bir sonraki
mantıklı adım (bu oturumda başlatılmadı) veri-kalitesi kanalının davranışsal skordan
ayrılması (GPT'nin önerisi) veya tamamen farklı bir feature ailesi olabilir — genlik-
normalize skor denemesinin kendisi tükenmiş bir yön olarak kapatıldı.

## ADR-020: ML-15 session-jackknife paralelleştirildi; tam 5-seed drift kalibrasyonu Gate B/C'yi geçemedi

- Durum: Altyapı tamamlandı; development rejected (Gate B ve Gate C)
- Tarih: 2026-07-10
- ADR-015'i tamamlar.

ML-15'in kayıtlı algoritması, eşikleri ve bütçeleri değiştirilmeden yalnız hesap yürütümü
hızlandırıldı. Temsilî, gerçek üretim parametreli `cProfile` ölçümünde tek CUSUM fit'i
16.433 saniye sürdü; bunun 16.310 saniyesi (%99.2) 16 eşik değerlendirmesindeki
`cusum_alarm_onsets`, yalnız 0.105 saniyesi `_moving_block_bootstrap` idi. Bu nedenle
bootstrap vektörleştirilmedi. Reset/refractory taşıyan durumlu CUSUM döngüsü de sayısal
semantiği riske atmamak için değiştirilmedi. Bağımsız leave-one-session-out jackknife
fitleri `joblib` süreçleriyle paralelleştirildi; sonuç sırası sıralı session listesine göre
korundu. Aynı anda çalışan ML-16 işi ve 16 GB RAM sınırı nedeniyle split-düzeyi iç içe
paralellik açılmadı; gerçek koşu 4 jackknife işçisiyle, splitler ardışık yürütüldü.

**Doğruluk kanıtı:** threshold, K-of-N ve CUSUM için ardışık/paralel birim regresyonları
birebir eşleşti (`tests/test_ml15.py`, 10/10). Gerçek `split_00` yeniden koşusu eski
`artifacts/ml15/uav_sead/smoke_split_00` ile karşılaştırıldı: `metrics.csv` 36/36,
`flight_label_metrics.csv` 180/180 ve `category_metrics.csv` 324/324 satırda `rtol=atol=1e-12`
ile eşleşti; `policies.json` birebir, `drift_reports.json` yalnız yeni işçi-sayısı provenance
alanı çıkarıldığında birebir eşleşti. `gates.json` kararları aynıydı; yalnız yaklaşık
1e-16 düzeyinde JSON kayan-nokta yazım farkları vardı. Yeni smoke 1026.528 saniye
(17.11 dakika) sürdü; ADR-015'teki yaklaşık 45 dakikaya göre 2.63x hızlanmadır.

**Tam koşu:** `artifacts/ml15/uav_sead/full_matrix/`, split_00..04, 5743.139 saniye
(95.72 dakika). Manifestteki 59 dosyanın SHA-256 değeri yeniden doğrulandı. Blind holdout
200 uçuş olarak manifestte kaldı ve hiçbir telemetri/feature/skor okumasına girmedi;
Gate A geçti.

**Gate B KALDI:** ön-kayıtlı kural 4 CUSUM hücresinin en az 3'ünün median FA<=bütçe ve
en az 4/5 seed'de <=1.25x bütçe olmasını istiyordu; yalnız 2/4 hücre geçti.
`existing_fusion` advisory median 8.749953 FA/saat (5/5 seed) ve critical median 0.903227
(4/5) geçti. `itki_komutu` advisory median 13.213611 (yalnız 3/5) ve critical median
7.353321 (0/5) kaldı. Drift multiplier 90 fit boyunca min/median/max = 1.0/1.0/5.0;
CUSUM advisory medianı 1.331623 idi.

**Gate C KALDI:** hiçbir drift-corrected hücre kritik >=0.30 recall @ <=2 FA/saat veya
advisory >=0.50 @ <=12 FA/saat hedefini karşılamadı. En iyi bütçe-içi advisory satırı
`ml14_fusion` CUSUM: 0.114504 recall / 8.414598 FA-saat. Aynı skorun kritik satırı
0.043257 / 1.595978 idi. `existing_fusion` CUSUM advisory 0.108397 / 8.338657;
critical 0.041730 / 1.596345 verdi. Drift düzeltmesi FA'yı bazı hücrelerde bütçe içine
çekti, fakat operasyonel recall açığını kapatmadı.

Karar: holdout açılmaz; policy/bütçe/quantile/floor/cap sonuçtan sonra değiştirilmez.
ML-15 full-matrix hesap açık işi kapanmıştır, fakat SEAD operasyonel alarm bütçesi
sınırlaması kapanmamıştır.

## ADR-021: ADSB-0 başlatıldı — segmentasyon + fiziksel-tutarlılık residual altyapısı yazıldı, gerçek veri erişimi bekliyor

- Durum: Altyapı tamamlandı (kod+test); gerçek veri üstünde doğrulama BLOKLU
- Tarih: 2026-07-10

9 farklı yöntemin (IF, LightGBM, Chronos, LSTM-AE, Dense-AE, USAD, genlik-normalize skor,
drift-kalibreli füzyon; ADR-008..020) SEAD/RFLY'de Gate C'yi geçememesi ve kök nedeninin
(küçük/heterojen normal havuzu + genlik-baskınlığı artefaktı) netleşmesi üzerine, kullanıcı yeni
ve paralel bir keşif hattı başlatma kararı aldı: adsb.lol/readsb ADS-B telemetrisinde, öğrenilmiş
bir model yerine **aritmetik fiziksel-tutarlılık residual'ları** (bildirilen kanal vs ham
lat/lon/alt'tan türetilen karşılığı) ile anomali tespiti. Bu tasarım, genlik-baskınlığı
artefaktının yapısal olarak oluşamayacağı bir zemin sunuyor. Track adı "ML-N"/"RFLY-N" isim
alanlarıyla çakışmasın diye **ADSB-0/ADSB-1** olarak ayrıldı; kod tamamen yeni `src/adsb/`
paketinde, `src/ml/` ve RFLY'ye hiç dokunulmadan yürütülüyor (plan onayı, aynı gün).

Repoda bu iş için önemli altyapı zaten mevcuttu: takım arkadaşı Metehan'ın staj projesinin ortak
fazında kurduğu adsb.lol tar→Bronze→Silver pipeline'ı (`src/silver/parse_adsblol_historical.py`,
ADR-003) ve `src/gold/unify.py`'deki Gold şema eşlemesi. En az bir günlük tar zaten parse edilmiş
(97.8M satır, 67.577 uçak — `logs/parallel_parse/*.log`, `Dashboard/FULL_PROJECT_HANDOFF.md` §3.3)
ama bu veri Metehan'ın kendi makinesindeki Docker `minio_data` volume'ünde; bu makinede ne ham
tar ne de parse edilmiş Silver verisi var, MinIO da şu an erişilemez durumda (2026-07-10 denendi,
Docker başlatılamadı). **Bu açık bir engel** — kullanıcının Drive'daki ham tar'lardan birini bu
makineye getirmesi ya da Metehan'dan veri aktarımı istemesi gerekiyor; kod tarafı bu olmadan da
tamamen sentetik veriyle geliştirilip test edildi.

**Yazılan ve test edilen (veri gerektirmez, sentetik):**
- `src/adsb/segmentation.py` — `assign_flight_ids`/`segment_flights` (boşluk-tabanlı uçuş bölme,
  `windowing.py`'nin `max_gap_s` deseninden esinli, repoda hazır eşdeğeri yoktu) +
  `new_leg_agreement` (readsb'nin kendi `flags_new_leg` bayrağıyla çapraz-doğrulama).
  `tests/test_adsb_segmentation.py`: 9/9 geçti.
- `src/adsb/physics_features.py` — 4 residual: `vertical_rate_residual`, `speed_residual`,
  `heading_residual` (tam kapsama), `turn_bank_residual` (roll_deg'e bağımlı, ~%8.5 kapsama,
  eksikse NaN — birincil karar buna bağımlı olmayacak). `tests/test_adsb_physics_features.py`:
  9/9 geçti; kritik doğrulama: `EARTH_RADIUS_M` sabiti hem modülde hem testte AYNI olmalı
  (ilk denemede WGS84 111320 m/derece ile modülün haversine 6371km yarıçapı arasındaki ~%0.1
  fark, testte sahte bir sistematik residual üretmişti — düzeltildi).
- `src/adsb/injection.py` — `src/ml/injection.py`'deki freeze/bias/noise/dropout'un ADS-B
  kolonlarında doğrudan yeniden kullanımı (kolon-adı-agnostik, doğrulandı) + yeni
  `inject_position_ramp` (mevcut `inject_gps_ramp`'in keyfi-kerteriz + saniye-zaman-damgası
  genellemesi) + `PHYSICS_BREAK_RECIPES` (5 adlandırılmış senaryo, ADSB-1'in doğrulama scripti
  için). `tests/test_adsb_injection.py`: 4/4 geçti.
- Tam paket: `pytest -q` → 329 geçti, 6 atlandı, yalnız önceden bilinen 4 MinIO SDK sürüm
  uyumsuzluğu hatası (bu değişiklikle ilgisiz).

**Yazılmadı (veri engeli / sonraki adım):** `scripts/make_adsb_visualizations.py` + galeri
(ADSB-0 §5), gerçek veri üstünde `new_leg_agreement` oranının gözlemlenmesi ve ADSB-0 faz
kapısının (residual≈0 temiz uçuşlarda) fiilen geçilmesi. Detaylar: `docs/ADSB0_INGEST_SEGMENT_
PLAN.md`, `docs/ADSB1_PHYSICS_DETECTOR_PLAN.md` (ikincisi ADSB-1'in ön-kayıtlı doğrulama
protokolünü de içeriyor: recall≥0.70 @ FA≤0.05, sonuç görüldükten sonra değişmez).

Karar: holdout kavramı henüz açılmadı (ADSB-1'de tanımlanacak, kör-holdout birkaç gün trafiği
olarak ayrılacak). Bu ADR, kod+test tamamlanmış bir ara durumu kaydeder; gerçek veri erişimi
çözülmeden ADSB-0 faz kapısı kapanmış sayılmaz.

## ADR-022: ADS-B sıfırlaması + gerçek 3-günlük veriyle ilk model turu (Dense-AE/LSTM-AE/USAD/LSTM-forecaster)

- Durum: Altyapı + ilk eğitim tamamlandı; USAD sayısal kararsız, sonuçlar dürüstçe karışık
- Tarih: 2026-07-10

ADR-021'deki `src/adsb/` (Claude) ile paralel, koordinesiz geliştirilen `src/adsb_behavioral/`
(Codex) denemesi karşılaştırıldığında ikisinin de aynı probleme iki ayrı isim alanı ve planla
başladığı görüldü; Codex'in denemesi düzeltme sonrası %97.6 sentetik recall ama doğal veride
25.54 yeni-alarm/saat üretti (kullanılamaz). Kullanıcı kararı: her ikisi de
`archive/2026-07-10_{legacy_non_adsb_ml,rejected_adsb_attempts}/` altına kaldırıldı (hiçbir
şey silinmedi), eski ML-0..16/RFLY hattı da aynı arşive taşındı, ve tamamen yeni, tek bir
`adsb/` (kök dizin, `src/adsb/` DEĞİL) hattı başlatıldı. Not: Dashboard/individual (Yusuf/
Metehan'ın kendi işleri) yanlışlıkla aynı arşive girmişti, aynı gün fark edilip geri çıkarıldı.

Kullanıcı ayrıca gerçek ADS-B verisi indirdi: 3 gün (`v2026.02.28`, `v2026.03.01`,
`v2026.03.16`, her biri ~3GB), `STORAGE_BACKEND=local` ile Docker'sız parse edildi
(256.150.550 satır toplam, 638 Silver parça).

**Faz 0 (madde 1,2,4,6) gerçek veriyle tamamlandı:** `adsb/inventory.py` envanteri (format
3 günde de stabil, hiç UAV/drone kategorisi yok — 2026-03-01 örnekleminde 25 kez görülen `B6`
istisnası henüz doğrulanmadı), `adsb/segmentation.py` gerçek veride çalıştırıldı (1500 uçak →
4230 uçuş, `flags_new_leg` uyuşma oranı **%60.4**), `adsb/reports/measurability_table.md`
(gerçek satır-düzeyi kapsama: `alt`/`vertical_rate_ms` %89.4, `ground_speed_ms` %98.1,
`track_deg` %95.2, `roll_deg` %28.4 — forward-fill YOK, format referansının eski %8.5
rakamından yüksek ama yine azınlık), `adsb/synthetic.py` (5 `PHYSICS_BREAK_RECIPES`
senaryosu, test-only, `save_synthetic_batch` path guard'lı). Madde 3 (galeri) hâlâ yazılmadı.

**Kullanıcı onay-kapısını bilinçli olarak esnetti:** "şimdi paralel başlat" talimatıyla,
Faz 0 tamamlanmadan Dense-AE/LSTM-AE/USAD/LSTM-forecaster mimarilerine (`adsb/models/`)
paralel başlandı — `adsb/README.md`'de bu istisna açıkça not edildi.

**İlk eğitim turu (ölçeksiz) — SEAD dersi bilerek tekrar test edildi:** `adsb/diagnostics.py`
(`magnitude_domination_check`, SEAD'in ADR-016 bulgusunu ZORUNLU standart hale getiren modül)
ilk turda LSTM-AE (ρ=0.919), USAD (ρ=0.986) ve LSTM-forecaster'ı (ρ=0.890) işaretledi; yalnız
Dense-AE (ρ=0.086) temiz çıktı. USAD'ın loss'u ayrıca sayısal olarak patladı (loss1 15.
epoch'ta 134 milyar). Kök neden: feature'lar (`alt` binler, `vertical_rate_ms` tek hane) hiç
ölçeklenmemişti — SEAD'in "kırpılmamış ölçekleme" hatasının ölçekleme YAPMAMA versiyonu.

**Düzeltme:** `adsb/scaling.py` (`ClippedRobustScaler`, train-only fit, clip=5.0 — SEAD'in
kırpma dersi doğrudan uygulanıyor) + 4 mimariye gradient clipping (`max_norm=1.0`) eklendi.
İkinci turda: Dense-AE ŞİMDİ işaretlendi (ρ=0.81/0.83), LSTM-AE işaretlenmedi ama sınırda
(ρ=0.77/0.80), USAD işaretlenmedi (ρ≈-0.05 — ama loss HÂLÂ patlıyor, 23 milyar; diagnostic
bunu yakalamıyor çünkü patlamış skor ne genlikle ne rastgele-init'le korele — **diagnostic'in
kendisi yeterli değil, eğitim-kararlılığı ayrıca kontrol edilmeli**, açık madde), LSTM-forecaster
işaretli kaldı (ρ=0.90/0.89).

**Sentetik-bozulma doğrulaması (5 senaryo × 4 mimari, 20 val uçuşu):** Dense-AE/LSTM-AE/
LSTM-forecaster üçü de 5/5 senaryoda corrupt>clean gösterdi, ama ayrım büyüklüğü çok değişken:
`ground_speed_biased` güçlü (2.3x-10.75x), `vertical_rate_frozen` orta (1.03x-2.28x),
`position_ramp_stealthy`/`track_frozen`/`altitude_dropout` zayıf (1.01x-1.17x — pratikte
ayırt edilemeyebilir). USAD 0/5 (patlamış eğitim yüzünden skor tamamen gürültü).

Karar: hiçbiri production/headline recall adayı DEĞİL — bu, ADSB'nin ilk kez gerçek veriyle
"pipeline çalışıyor mu" sorusuna cevap. Açık maddeler: USAD'ın sayısal kararsızlığı çözülmedi,
galeri (Faz 0 madde 3) yazılmadı, eğitim-kararlılığı diagnostic'e eklenmedi, tam-hacim (3 gün)
eğitim yapılmadı (yalnız 10/638 Silver parça, 3000 uçak kullanıldı). Kör-holdout henüz
tanımlanmadı.
