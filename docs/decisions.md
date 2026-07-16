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
- `Dashboard/codes/minio_archiver.py` — `group.id = minio-archiver`; MinIO Bronze

**Neden?** Separation of Concerns: arşivleyici çöktüğünde dashboard etkilenmiyor;
arşivleyici bağımsız ölçeklenebilir; her consumer kendi offset'ini yönetiyor.
Her iki consumer da aynı Kafka topic'i `uav.flights`'ı bağımsız okuyor — Kafka'nın
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
(97.8M satır, 67.577 uçak — `logs/parallel_parse/*.log`, `Dashboard/docs/FULL_PROJECT_HANDOFF.md` §3.3)
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

## ADR-023: Kalıcı sentetik test/validation korpusu + parser envanteri + shuffle bug düzeltmesi

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Parser envanteri (kullanıcı sorusu üzerine):** Tarihsel adsb.lol tar arşivlerini parse eden
TEK implementasyon var — `src/silver/parse_adsblol_historical.py::parse_trace_bytes()` (ADR-003
ile `src/bronze2silverParsers/parse_adsb_traces_from_tar_v2.py`'den taşınmış, mevcut proje
altyapısı, bu oturumda yeniden yazılmadı). Arşivlenmiş Codex denemesi bile
(`archive/2026-07-10_rejected_adsb_attempts/src/adsb_behavioral/reader.py`) kendi parser'ını
yazmak yerine AYNI `parse_trace_bytes()`'i import ediyor — iki bağımsız denemede de parse mantığı
hiç ayrışmamış. Farklı veri kaynağı için ayrı bir parser var (`parse_adsblol_realtime.py`, canlı
adsb.lol API'si) ama birim dönüşümleri (feet→m ×0.3048, knots→m/s ×0.5144) ikisinde de birebir
aynı — tutarsızlık yok. `individual/metehan_geo/data.py` parse etmiyor, zaten üretilmiş Gold
katmanını okuyor.

**Kalıcı sentetik korpus:** `adsb/synthetic.py::save_synthetic_batch()` daha önce yazılmış ama
hiç çağrılmıyordu (`run_synthetic_check()` bozulmayı yalnız bellekte, geçici olarak, 20 val
uçuşuyla üretip atıyordu). Yeni `scripts/adsb_generate_synthetic_dataset.py`, 60 Silver parçadan
(22.520.807 satır, 18.000 uçak) segment_flights + aynı 80/20 train/val bölmesiyle (SEED=0) 8910
val-uçuşu seçip 5 PHYSICS_BREAK_RECIPES'i uyguladı. Çıktı: `data/objectstore/synthetic/adsb/`
altında 6 parquet dosyası (clean + 5 recipe, her biri 4.467.115 satır/8910 uçuş, toplam 765MB) +
`manifest.json`. Gerçek Silver'a hiç dokunulmadı (`save_synthetic_batch`'in path-guard'ı
zorluyor). İlk tasarım uçuş-başına-recipe-başına AYRI dosya yazıyordu (~59.000 küçük dosya, 60
parçada) — Windows'ta çok yavaş olduğu için recipe-başına TEK konsolide dosyaya çevrildi.

**Bug bulundu ve düzeltildi:** `np.random.Generator.shuffle()`, pandas'ın `.unique()`'ından gelen
`ArrowStringArray`'i (pyarrow-destekli string dtype) "Sequence değil" diye şikayet ediyordu.
Küçük ölçekte (3 parça, 2440 uçuş) zararsız çıktı (0.09s, tekrar yok) ama asıl risk teyit
edilmedi büyük ölçekte — güvenli olması için hem `scripts/adsb_generate_synthetic_dataset.py`
hem de `scripts/adsb_train_baseline_models.py::prepare_windows()` (gerçek model eğitiminin
train/val bölmesi) `np.array(..., dtype=object)` ile düz numpy array'e çevrilecek şekilde
düzeltildi.

**Açık madde:** `seg[seg["flight_id"] == fid]` her uçuş için TÜM segment tablosunda (22.5M satır)
tam tarama yapıyor — 8910 uçuş × tam-tablo-taraması ~25 dakika sürdü. Sonraki ölçek büyütmede
(tam 638 parça) `seg.groupby("flight_id")` ile tek seferlik gruplama kullanılmalı, yoksa süre
doğrusal değil süper-doğrusal büyür.

## ADR-024: İlk tam eğitim turu (60 parça) + z-score güven skoru + literatür taraması

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Yeni feature (literatür bulgusu):** Web taraması, ADS-B sahtekarlık tespitinde "self-consistency
check" (barometrik vs jeometrik irtifa, NIC/NACp/SIL bütünlük alanları) sinyalinin öncelikli
literatür bulgusu olduğunu gösterdi (PIR-PSO-XGBoost, F1=0.9528; NIC/NACp/SIL tutarlılık kontrolleri
sahte mesaj tespitinde merkezi). Silver'da hem `alt` (barometrik, %89.4 kapsam) hem `alt_geom_m`
(jeometrik, %89.2) zaten var — `adsb/features.py::altitude_source_residual()` eklendi: iki kaynak
arasındaki FARKIN zaman-türevi (fark sabit değil ama normalde ~sabit, ani sıçrama kaynaklardan
birinin bozulduğunu gösterir). `PRIMARY_FEATURES`'a eklendi (artık 8 kanal). `nic`/`nac_p`/`sil`
alanları da Silver'da mevcut — S2 kural-kanalı (henüz yazılmadı) için veri hazır.

**z-score güven skoru:** `adsb/diagnostics.py::fit_score_baseline()` (train-skorunun medyan/MAD'i,
SEAD dersi gereği ortalama/std değil) + `z_score_confidence()` (standart normal CDF, istatistiksel
p-değeri İDDİASI TAŞIMAZ, yorumlanabilir tek-yönlü "ne kadar uç" ölçeği).

**Eğitim turu (`scripts/adsb_train_baseline_models.py`, N_PARTS=60 — sentetik korpusla AYNI split,
sızıntı çalışma-zamanında doğrulandı: train 39.555 / sentetik-korpus 8.910 uçuş, kesişim sıfır):**
Dense-AE, LSTM-AE, LSTM-forecaster eğitildi (USAD bu turda HARİÇ, ADR-022/023'teki sayısal
kararsızlık hâlâ çözülmedi). Yol boyunca gerçek bir bug bulundu: LSTM skorlaması 2.85M pencereyi
TEK forward'ta işlemeye çalışıp ~20.8GB istedi ve çöktü — `_score_batched()` ile düzeltildi
(20.000'lik gruplar), `BATCH_SIZE` eğitimde de 64→512 (36dk tahmini Dense-AE süresini ~2.4dk'ya
indirdi, CPU-only ortamda batch-başı sabit-maliyet baskınlığı yüzünden).

**Sonuçlar — dürüstçe, üçü de aynı sorunu gösteriyor:**

| Model | magnitude_domination | pooled AUC (5 senaryo havuzu) |
|---|---|---|
| Dense-AE | FLAGGED (ρ=0.86/0.90) | 0.572 |
| LSTM-AE | FLAGGED (ρ=0.84/0.89) | 0.568 |
| LSTM-forecaster | FLAGGED (ρ=0.94/0.92, EN KÖTÜ) | 0.552 |

Senaryo bazında (3 modelde de aynı örüntü): `ground_speed_biased` orta (AUC 0.65-0.74) — diğer
dördü rastgele-yakını veya ALTINDA (`track_frozen`/`position_ramp_stealthy` ~0.51-0.55,
`altitude_dropout` **0.48-0.50, iki modelde rastgeleden KÖTÜ**). LSTM-forecaster'ın loss eğrisi
15 epoch boyunca neredeyse DÜZ (~1.6'dan ~1.55'e) — pratikte öğrenmiyor, en yüksek ρ ve en düşük
AUC ile tutarlı.

**Kök neden teşhis edildi (bu oturumun daha önceki bir önerisinin doğrudan deneysel kanıtı):**
`masked_mse`, 8 kanalı EŞİT ağırlıkla ortalıyor. `alt`/`ground_speed_ms`/`track_deg` gibi ham
kanallar büyük, yapılı, kolay-öğrenilir varyansa sahip (iniş/kalkış eğrisi tahmin edilebilir) —
residual kanalları ise normal uçuşta ~gürültü (öğrenilecek yapı yok, "doğru cevap" zaten sıfıra
yakın rastgelelik). Eşit-ağırlıklı ortak loss altında optimizasyon doğal olarak kolay/büyük
kanallara odaklanıp residual kanalları pratikte görmezden geliyor — bu hem magnitude-domination'ı
(skor ham-karmaşıklığı yansıtıyor, fizik-ihlalini değil) HEM düşük AUC'yi (çoğu senaryo ham
kanalların büyüklüğünü değiştirmiyor) tek bir mekanizmayla açıklıyor. Tek istisna
(`ground_speed_biased`, literal +4σ ekliyor) ham-kanal büyüklüğünü DE değiştirdiği için "yakalanmış"
görünüyor — ama muhtemelen doğru fizik-ihlali anlayışından değil, büyüklük artışından.

**altitude_dropout'un rastgeleden kötü AUC'si (0.48-0.50) muhtemelen bir ölçüm artefaktı**: dropout
NaN üretiyor, maskeleniyor, bu da `masked_mse`'nin payda'sını (aktif eleman sayısı) küçültüyor —
gerçek ters-sinyal değil, muhtemelen skorun payda-küçülmesinden kaynaklanan bir yapı; doğrulanmadı,
açık madde.

**Grafikler:** `artifacts/adsb/plots/` (loss_curves, roc_curves, auc_heatmap, confusion_matrices,
score_distributions) — `scripts/adsb_make_training_plots.py`, salt-okunur, rapordan üretiyor.

**Açık maddeler:** (1) Ağırlıklı-loss deneyi (residual kanallarına 3-5x ağırlık) henüz koşulmadı —
kök-neden teşhisinin doğrudan testi. (2) USAD hâlâ hariç. (3) `altitude_dropout` ters-sinyali
doğrulanmadı. (4) Kategori-koşullu normallik (literatür: CNN uçak-tipine-göre trafik-şekli
tutarsızlığı tespiti) heterojen-normal sorununa yeni bir yaklaşım olarak NOT edildi, denenmedi.
(5) S2 kural-kanalı (squawk/nic/nac_p/sil) hâlâ yazılmadı, veri hazır.

Kaynaklar: [PIR-PSO-XGBoost GPS spoofing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12656533/),
[ADS-B ML anomali tespiti](https://www.sciencedirect.com/science/article/abs/pii/S2542660524003573),
[NIC/NACp/SIL bütünlük alanları](https://navi.ion.org/content/72/3/navi.716).

## ADR-025: Kural-bazlı penalty skorlayıcı — NN'leri geçti + AUC-tavanı bulgusu

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Motivasyon (kullanıcı talebi + ADR-024 kök-neden teşhisi):** "Kural bazlı matematiksel formüllü
bir destek — hatta destekten öte ana odak; penalty/reward mantığımız yok mu?" Residual'lar zaten
aritmetik özdeşlik olduğu için öğrenmeye gerek yok: `adsb/rules.py::ResidualRuleScorer` — kanal
başına robust z (train-normal medyan/MAD), `pen_c = min(max(0, z_c − 3), 10)`, satır penalty'si
= ağırlıklı toplam (uniform w=1, ÖN-KAYITLI, sonuç görülüp ayarlanmadı), pencere skoru = pencere-içi
ortalama (NN'lerle aynı birim). ML-12 dersiyle aynı ruh (tek domain-seçilmiş sinyal 16-feature
modeli geçmişti).

**İki tur (ikisi de raporlu, `artifacts/adsb/models/rule_scorer_report*.json`):**
- Tur 1 (MAD-floor): `altitude_source_residual` MAD=0 çıktı (irtifa 25ft-kuantize → fark-türevi
  çoğunlukla TAM 0), floor=1e-6 kanalı "kıl tetik" yaptı — normal pencerelerin %93.8'i sıfır-üstü
  penalty aldı, pooled AUC 0.582.
- Tur 2 (genel kural: **train MAD'i tam 0 çıkan kanal kalibre-edilemez sayılır, skordan hariç**
  — arşivlenen denemenin zero-MAD dersinin doğru genellemesi; floor çökmeyi önler ama kıl-tetiği
  önlemez): normal sıfır-penalty oranı %6.2→%31.3, **pooled AUC 0.600 — üç NN'i de geçti**
  (0.552-0.572).

| Senaryo | Kural (tur 2) | En iyi NN | Fark |
|---|---|---|---|
| ground_speed_biased | 0.727 | 0.743 (LSTM-AE) | ≈ |
| track_frozen | **0.679** | 0.523 | +0.16 KURAL |
| vertical_rate_frozen | 0.579 | 0.600 (Dense-AE) | ≈ |
| position_ramp_stealthy | 0.519 | 0.519 | ≈ (ikisi de kör — CUSUM işi) |
| altitude_dropout | 0.494 | 0.498 | ≈ (kural için KAPSAM DIŞI beyanlı: NaN→katkı 0; ayrı S2 veri-kalite kanalının işi) |

**KRİTİK DEĞERLENDİRME BULGUSU (zaman-çizgisi grafikleri açığa çıkardı,
`artifacts/adsb/plots/injection_timelines/`):** bozuk dosyaların onset-ÖNCESİ pencereleri (uçuşun
ilk yarısı, onset_frac=0.5) birebir temiz olduğu halde AUC hesabında "anomali" (y=1) sayılıyor —
MÜKEMMEL bir dedektörün bile ulaşabileceği AUC ~0.75 ile sınırlı (pencerelerin ~yarısı yapısal
olarak ayırt-edilemez). RFLY-1'in "whole-flight proxy → interval truth" düzeltmesinin birebir
tekrarı: dosya-kimliği proxy etiketi yerine `label` kolonundan pencere-etiketi türetilmeli
(yalnız onset-sonrası satır içeren pencereler y=1). Bu düzeltmeyle ground_speed_biased 0.727 ve
track_frozen 0.679 fiilen tavana yakın — sonraki turda etiketleme düzeltilip yeniden ölçülecek
(ÖN-KAYIT: düzeltme yalnız etiket yönünde, skorlara/parametrelere dokunulmayacak).

Zaman-çizgisi gözlemleri: `ground_speed_biased`/`track_frozen` onset'te sıçrayıp KALICI ~10
penalty bandına oturuyor (temiz ~0'da) — tespit tam olması gerektiği gibi görünüyor;
`vertical_rate_frozen` yalnız onset anında kısa patlama (donmuş değer, irtifa değişmedikçe
tutarlı kalıyor); `position_ramp_stealthy` onset spike'ı var ama sonrası eşik altı (2 m/s <
3×MAD≈5.8 m/s — birikimli/CUSUM olmadan yapısal olarak görünmez); `altitude_dropout` düz
(beyan edilen kapsam dışılık doğrulandı).

**Açık maddeler:** (1) pencere-etiket düzeltmesi + yeniden ölçüm; (2) stealthy ramp için
uçuş-içi CUSUM birikimi (S3 madde 9); (3) tam-hacim (638 parça + yeni indirilen v2025.06.15
tar'ı); (4) NN'ler için ağırlıklı-loss deneyi hâlâ koşulmadı (kural skorlayıcı ana-odak olunca
önceliği düştü ama kök-neden testinin kanıt değeri duruyor).

## ADR-026: Değişmez ADS-B run manifesti ve provenance sözleşmesi

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Karar:** Her yeni ADS-B koşusu yeni ve mevcutsa hata veren bir `run_dir` ile başlar;
`adsb/run_manifest.py` girdilerin byte/SHA-256/Parquet footer+schema hash'ini, açık uçuş
splitlerini ve hashlerini, config/Git dirty-state'ini ve sentetik path/uçuş-ID sızıntı
korumasını kaydeder. CLI girdi keşfi yapmaz ve holdout/Downloads varsayılanı taşımaz.

**Kanıt:** `tests/test_adsb_run_manifest.py` ile fail-if-exists, deterministik split, hash/footer,
sentetik guard ve CLI dahil **9 test**; `tests/test_adsb_evaluation.py` ile birim ayrımına ait
**5 test**, toplam **14/14 geçti**. Manifest provenance'i footer ölçümü
**256.155.009**, eski belgeli toplam **256.150.550** ve açıklanmamış fark
**+4.459** satırı birlikte, `unresolved_do_not_silently_correct` olarak taşır.

**Açık madde:** Somut skor koşuları kendi donmuş config/splitleri belli olduğunda ayrı
manifestlerle yaratılacak; holdout havuzu bu adımda erişilmedi veya freeze edilmedi.

## ADR-027: Sentetik interval truth v2 ve ayrı, değişmez korpus

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Karar:** `adsb/truth.py` satır düzeyinde `injection_active`, `observable_changed`,
`evaluable_truth` ile event aralığını ayrı tutar. Rule/AE desteği tüm pencere, forecaster desteği
yalnız açık hedef satırlarıdır; birincil etiket `q_w>0`, ikincil steady-state yalnız
`q_w∈{0,1}`'dir. Dropout yalnız eski RNG'nin gerçek bloğunu, ramp onset satırı ise `dt=0`
nedeniyle aktif fakat değişmemiş gözlemi kaydeder. V1 dosyaları değiştirilmedi.

**Kanıt:** Tam korpus `data/objectstore/synthetic/adsb_v2_20260713_01/manifest.json` altında
**8.910 uçuş × (clean + 5 senaryo)**, dosya başına **4.467.115**, toplam **26.802.690 satır** ve
**646.160.578 byte** olarak fail-if-exists yazıldı; her çıktı SHA-256/footer taşır. Dropout'ta
**666.739 aktif** satırın **570.666**'sı gözlenebilir değişimdir; onset→uçuş-sonu proxy'si
kullanılmadı. Tüm ADS-B regresyonu skor koşusundan önce **161/161 geçti**. İlk yavaş smoke
namespace'i manifestsiz olduğu için `data/objectstore/synthetic/adsb_v2/INCOMPLETE_DO_NOT_USE.md`
ile açıkça geçersiz işaretlendi, üstüne yazılmadı.

**Açık madde:** Donmuş kuralın corrected-truth skorlaması Adım 3'tür; sentetik v2 hiçbir
fit/train/calibration rolüne alınmayacaktır.

## ADR-028: Donmuş kuralın corrected truth-v2 üzerinde yeniden skorlanması

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Karar:** ADR-025'teki kural, kalibrasyon veya eşik değiştirilmeden truth-v2 üzerinde yeniden
skorlandı. Negatif havuz her reçete için aynı değiştirilmemiş temiz referanstır; bozuk dosyanın
q_w=0 pencereleri AUC negatiflerine katılmadı ve yalnız zaman çizgisi sanity kontrolünde tutuldu.
NN checkpoint'i bulunmadığından eski NN JSON'u yalnız
historical_label_bugged_no_checkpoint_rescore olarak kaydedildi; corrected iddiası taşımıyor.

**Kanıt:** Değişmez koşu
artifacts/adsb/runs/20260713_step3_corrected_rule_v1/ altındadır. Pooled birincil pencere
AUROC/AUPRC **0.764883/0.883313** (705.787 temiz negatif + 1.511.700 q_w>0 pozitif);
reçete AUROC'ları dropout **0.566965**, ground-speed **0.974023**, ramp **0.558927**,
track **0.889552**, vertical-rate **0.696337**'dir. Aynı donmuş eşikte değiştirilmemiş temiz
referansta **40.308 episode / 8.382599 scoreable uçuş-saat = 4.808533 episode/saat** ve
alarm gören uçuş oranı **0.892356** oldu. Örneğin ground-speed event recall'ı **0.963659**
(medyan gecikme **19.31 s**), track **0.951804** (**56.75 s**); ramp event recall'ı
**0.801347** olsa da aktif-aralık micro coverage yalnız **0.183902**'dir. Sentetik ölçüler bu
doğal yükten ayrı yorumlanmadı. Event tablosu **44.550** satırdır.

summary.json SHA-256
4b81c37af0f1e3c0a4ef6a1f135bb726b6fd01be64d3ee260f707b06378348f0,
event tablosu SHA-256
c8dda720da9e8e0333dfcb910c07cb5e60c4a41fee9bf7f12205b7d4725b4051 ve manifest
SHA-256 fc188839a114052dde7ed7d870876780c73f327d261d9fb73245b0a358840fb8 olarak
doğrulandı; corrected-truth/evaluation regresyonu **18/18** geçti. Dış komut zarfı,
çıktı yolu basıldıktan sonra 1.287 saniyelik sınırda 124 döndürdü; Python süreci durmuştu,
üç çıktı parse/hash/footer ve testlerle eksiksiz doğrulandı.

**Açık madde:** Bu tarihsel 0.95 skor eşiği operasyonel olarak onaylanmış bir eşik değildir.
Doğal yük ve günler-arası kararlılık Adım 5'te, CUSUM ile birlikte ön-kayıtlı normal
kalibrasyon/development/rehearsal akışında ölçülecektir.

## ADR-029: İki eksenli hız residual'ı ve causal Page CUSUM çekirdeği

- Durum: Kabul edildi
- Tarih: 2026-07-13

**Karar:** adsb/features.py raporlanan doğu/kuzey hız bileşeni ile ardışık konumdan türetilen
bileşen arasındaki işaretli residual'ları üretir. adsb/cusum.py her eksen için pozitif/negatif
Page state'i taşır; 2 m/s vektör hedefini yön-bağımsız 2/sqrt(2) eksen alt sınırına ve
train-normal medyan/MAD kalibrasyonuna çevirir. Uçuş/ground/gap/zaman resetleri, dt=0 skip,
eksiklikte süreli carry-reset ve prefix nedenselliği sözleşmedir. h çekirdek tarafından
seçilmez; dışarıdan donmuş doğal-burden kalibrasyonu zorunludur.

**Kanıt:** tests/test_adsb_cusum.py ve tests/test_adsb_features.py güncel halde **30/30**
geçti. MAD=0 kanal floor'lanmadan hariç tutulur. İki eksenden biri hariç kalırsa serileştirme
aktif state sayısını doğru verir ve axis_coverage_status=degraded_axis_coverage yazar;
Adım 5/7 bunu geçiş değil gate engeli sayacaktır. CUSUM kanalları NN PRIMARY_FEATURES listesine
eklenmedi; yeni bir NN eğitimi yapılmadı.

**Açık madde:** Sayısal h henüz seçilmedi/dondurulmadı. Yalnız 2026-02-28 normal kalibrasyon
bölümündeki ön-kayıtlı adaylar ve doğal episode/saat bütçesiyle Adım 5'te seçilecek; sentetik
recall bu seçime girmeyecektir.

## ADR-030: Tam-hacim streaming baseline kanıtı; ana konfigürasyon freeze'i reddedildi

- Durum: Kanıt olarak kabul edildi; konfigürasyon freeze'i reddedildi
- Tarih: 2026-07-13

**Karar:** 638 açık Silver parçası üzerinde 2026-02-28 fit/calibration, 2026-03-01 development
ve 2026-03-16 donmuş rehearsal akışı tamamlandı. V1 artefaktı değişmez tarihçe olarak korunur,
ancak seçilen CUSUM h=1 doğal alarm doygunluğu nedeniyle ana rule+CUSUM konfigürasyonu olarak
dondurulmaz. 12 episode/saat sınırı ölçülmüş operasyonel gereksinim değil, bu turda
ön-kayıtlanmış engineering-advisory varsayımıdır; bunu sağlamak tek başına geçiş sayılmaz.

**Kanıt:** artifacts/adsb/runs/20260713_step5_full_streaming_v1 koşusu **10.588 saniyede**
exit 0 ile tamamlandı. Toplam **256.155.009 satır / 638 parça**; roller fit **149.462**,
calibration **37.208**, validation **8.910**, development **181.828**, rehearsal **165.053**
uçuş segmentidir. Sentetik eğitim satırı **0**; 8.910 sentetik-kaynak ID'nin tamamı fit ve
calibration dışında validation rolündedir. Fit'te **67.360.196 satır** ve **59.682.640**
vektör-uygun geçiş görüldü. Doğu/kuzey MAD'leri **1.736830/1.542729** ile iki eksen aktif;
altitude_source_residual MAD=0 olduğu için floor'suz dışlandı.

Rule diagnostic episode/saat calibration/development/rehearsal için
**1.171104/1.061339/1.158345**, alarm gören scoreable-flight oranı
**0.523444/0.506338/0.523398** oldu. CUSUM h=1 episode/saat
**6.071336/5.738076/5.326149** görünse de alarm gören scoreable-flight oranı
**0.991196/0.991636/0.991540**, evaluable-row alarm oranı yaklaşık
**0.78282/0.77709/0.80189** oldu. Cadence tabakalarında hızlı ve yavaş gruplar arasında
yaklaşık 2.1--2.5 kat burden farkı vardır; 60 saniyelik episode merge doygunluğu gizleyebilir.

V1 raw moving-block p95'i **5.745749/saat**, tam-uçuş observed değer **6.071336/saat** idi;
upper niceliğinin nokta tahmininden düşük olması correctness invariant'ını ihlal etti. Eski
dosyalar değiştirilmeden artifacts/adsb/runs/20260713_step5_cusum_upper_audit_v1 altında
conservative upper=max(observed, raw p95) denetimi yapıldı: 18 adayın 9'u yukarı düzeldi,
fakat 18/18 yine 12/saat varsayımını geçti ve seçim **h=1 -> h=1** değişmedi.

V1 final report SHA-256
e906a348e03d866dd9f9cfabe470a1e9a8735b04e12986bcd0b0d3e92f299a6a,
natural report SHA-256
37a8bf99ac0221b51bc5c9864f3c870b081394f3c1d32124b505728ee3eba766,
checksum index SHA-256
8706a8f79df559571077d7301e224556fc9ed7a731d337f196481817692112f2'dir.
İndeks, kendisi hariç **9/9** artefaktın byte/hash'ini doğruladı; runner sonu code/config
değişmezliği true ve rehearsal geri-beslemesi false'tur.

**Açık madde:** Truth-v2 üzerinde donmuş CUSUM event recall/gecikme/aktif coverage henüz bu
ADR anında ölçülmedi. S2 doğal burden ve corrected CUSUM değerlendirmesi tamamlanacak; Adım 7
incelemesinde mevcut doygunluk nedeniyle freeze önerilmeyecek. Yeni h/bütçe seçimi bu sonuçlara
bakılarak bu run içinde yapılamaz; kullanıcı onaylı yeni ön-kayıt ve yeni namespace gerektirir.

## ADR-031: Tam-hacim S2 doğal reason burden; residual penalty'den ayrı

- Durum: Kanıt olarak kabul edildi; Adım 7 gate kararı bekleniyor
- Tarih: 2026-07-14

**Karar:** S2 declared-status, position-quality, altitude-availability ve message-gap kanalları
residual penalty/CUSUM'dan ayrı tutuldu; saldırı ground-truth'u veya false-positive oranı diye
adlandırılmadı. Legacy Silver'da update-age alanları bulunmadığı için tüm 256.155.009 satır
freshness_unknown kaldı; eşleşen squawk/emergency değerleri fresh varsayılıp corroborated
yapılmadı. State reason'ları yalnız gerçek rising edge'de, MESSAGE_GAP ise her post-gap satırında
point event olarak sayıldı.

**Kanıt:** artifacts/adsb/runs/20260714_step6_s2_natural_v4 koşusu 4 deterministik worker ile
638/638 parçayı ve 256.155.009 satırı 272,2 saniyede exit 0 ile tamamladı. Gün bazında
satır/uçuş/scoreable-saat değerleri 2026-02-28 için
88.762.032 / 195.580 / 172.927,359925; 2026-03-01 için
85.991.023 / 181.828 / 168.752,015006; 2026-03-16 için
81.401.954 / 165.053 / 155.891,525003'tür. Pooled toplam 542.461 uçuş segmenti ve
497.570,899933 scoreable uçuş-saattir.

Pooled doğal reason burden: MESSAGE_GAP **2,890511 episode/saat**; NIC
reported-unknown/unavailable **0,528053/saat**; NACp missing **0,195202/saat**; SIL missing
**0,195202/saat**; all-altitude-unavailable **0,005083/saat**; declared-status
not-corroborated **0,000330/saat**. MESSAGE_GAP günler arasında
2,899321 / 2,811996 / 2,965729 episode/saat; NIC unknown
0,608492 / 0,499870 / 0,469333 episode/saat oldu. Bu değerler nominal operasyonel
reason burden'dır, anomali etiketi değildir.

run_manifest.json SHA-256
ac60638bbc708064f25f535141a35e0f724674722620648c253724ecc604b15b,
s2_natural_burden_report.json SHA-256
4ddb42293c4f0d6d89b7dec19631b9ff8a3a9a87c888b5878d105474bb6edb53 ve
artifact_checksums.json SHA-256
12ea24922a4ee96e16869bde8c461578b39a689915c5c23652175afb0b1f9214'tür. Checksum indexindeki
2/2 dosyanın byte/SHA değerleri yeniden doğrulandı; holdout_accessed=false'tur. İlgili
segmentation/S2/parser regresyonu **54/54 geçti**. Önceki v1/v2/v3 namespace'leri incomplete
marker ile geçersizdir ve yeniden kullanılmayacaktır.

**Açık madde:** Adım 7 corrected CUSUM truth-v2 değerlendirmesi henüz tamamlanmadı. Geniş
regresyonda 221/222 test geçti; tek hata Step-5 manifestindeki donmuş adsb/features.py byte
hash'inin güncel checkout ile eşleşmemesidir. Bu fail-closed engel çözülmeden hash kontrolü
gevşetilmeyecek, corrected CUSUM iddiası kurulmayacak ve ana konfigürasyon dondurulmayacaktır.

## ADR-032: Adım 7 gate FAIL; ana konfigürasyon dondurulmadı

- Durum: Gate tamamlandı — FAIL / kullanıcı sert durma noktası
- Tarih: 2026-07-14

**Karar:** Rule+CUSUM/S2 ana konfigürasyonu dondurulmadı. Adım 7'nin zorunlu kanıtlarından
provenance ve S2 tamdır; fakat CUSUM doğal alarm doygunluğu operasyonel gate'i geçmez, üç NN
magnitude şartını geçmez ve corrected CUSUM truth-v2 ölçümü frozen scoring-source snapshot'ı
geri getirilemediği için fail-closed blokludur. Bu üç koşuldan herhangi biri freeze'i engellemeye
yeterlidir. Sonuçlara bakılarak yeni h/eşik seçilmedi.

**Gate kanıtı:**

- Truth-v2 kuralı corrected etiketlerle tamamlandı: pooled AUROC/AUPRC
  0,764883/0,883313. Ground-speed event recall 0,963659 (medyan 19,31 s), track recall
  0,951804 (56,75 s); stealthy ramp event recall 0,801347 görünse de aktif-aralık micro
  coverage yalnız 0,183902'dir. Aynı blok temiz doğal burden 4,808533 episode/saat ile
  eşlenmiştir.
- Tam-hacim CUSUM h=1 calibration/development/rehearsal burden
  6,071336/5,738076/5,326149 episode/saat görünürken scoreable-flight alarm oranı
  0,991196/0,991636/0,991540 ve evaluable-row alarm oranı yaklaşık
  0,78282/0,77709/0,80189'dur. Episode merge doygunluğu gizlemektedir. Cadence tabakaları
  arasında yaklaşık 2,1–2,5 kat fark vardır.
- S2 v4 provenance ve doğal burden ADR-031'de PASS'tir. MESSAGE_GAP gün burden'ı
  2,899321/2,811996/2,965729 episode/saat; NIC unknown
  0,608492/0,499870/0,469333 episode/saat oldu. Bu S2 reason'ları saldırı etiketi değildir.
- Dense-AE, LSTM-AE ve LSTM-forecaster magnitude-domination kontrolünde sırasıyla yaklaşık
  rho 0,86/0,90; 0,84/0,89; 0,94/0,92 ile FLAGGED'dir. Pooled tarihsel label-bugged AUC'ler
  0,572/0,568/0,552'dir; corrected checkpoint rescore yoktur.
- Step-5 artefakt/hash zinciri ve Step-6 v4 checksum zinciri PASS'tir. Step-7 evaluator,
  frozen scoring dependency byte kontrolünde doğru biçimde durur. adsb/cusum.py canonical-LF
  SHA-256 değeri manifestteki 44b87b2a983ce5775ca3d609a79570a83ce0a3df0d28840cc365e558927ffe61
  ile eşleşir; fark yalnız checkout EOL'üdür. adsb/features.py ise canonical-LF
  cacb7febcab35c57849fe0d1cd3853e7346fc10b81d13ca7057618780b4e1470 olup frozen
  37eae2b84927b3d07b7a2b281dcd073f18e874338cb740ab522a7dfb83ed2fc4 ile eşleşmez.
  3–15 KB aralığındaki 735 yerel Git blob adayında frozen SHA-256 bulunamadı. Hash kontrolü
  gevşetilmedi ve current-code sonucu corrected frozen-CUSUM diye sunulmadı.

**Sonuç:** Genel Adım 7 gate **FAIL**. Ana config freeze yoktur. Adım 8'in “yalnız Adım 7
stabilse” önkoşulu sağlanmadığı için Dense-AE 4x treatment ve USAD smoke testi başlatılmadı.
Adım 9 holdout freeze/unseal işine geçilmedi; üç raw tar açılmadı veya hashlenmedi.

**Açık madde / sonraki kullanıcı kararı:** Yeni CUSUM adayı ancak kullanıcı-onaylı yeni
ön-kayıt, operasyonel burden bütçesi ve yeni namespace ile seçilebilir. Corrected CUSUM
karşılaştırması için ya gerçek frozen features.py byte snapshot'ı dış kaynaktan geri
getirilmeli ya da mevcut code version açıkça yeni bir aday olarak baştan natural calibration
ve truth-v2 akışından geçirilmelidir; mevcut Step-5 adayı adına post-hoc ikame yapılamaz.

## ADR-033: Contextual-physics v1 ayrı aday altyapısı; sayısal alarm bütçesi bekleniyor

- Durum: Yapısal sözleşme ve test altyapısı kabul edildi; bilimsel config/threshold donmadı
- Tarih: 2026-07-14

**Karar:** Kullanıcının anomaly türü özelinde threshold ve heterojen normal-uçuş dağılımı
araştırmasını uygulama onayıyla `contextual_physics_v1` ayrı aday namespace'i açıldı. Eski
Step-5/Step-7 artefaktı değiştirilmedi. Uçuş fazı yalnız geçmiş vertical-rate satırlarından
nedensel çıkarılır; track sin/cos, gerçek delta-t ve cadence açık girdidir. NN ham telemetriyi
reconstruct etmek yerine residual kanal başına next-step location/scale üretir. Strict scaler
MAD=0 kanalı floorsuz dışlar. Natural-only hierarchical conformal calibration
`channel+phase+cadence -> channel+phase -> channel` fallback uygular. Spike, persistence ve
saniye-normalize accumulation profilleri ile channel alpha payları ayrı tutulur; sessiz fusion
yasaktır.

**Kanıt:** Yeni uygulama `adsb/context.py`, `adsb/contextual_scaling.py`,
`adsb/contextual_windowing.py`, `adsb/conditional_calibration.py`,
`adsb/contextual_decision.py` ve `adsb/models/contextual_residual_forecaster.py` içindedir.
Sentetik fit/calibration, yanlış veri rolü, implicit alpha, toplam bütçeyi aşan channel payı,
MAD=0 floor, karışık-channel fusion, geleceğe bakan phase ve uçuşlar-arası pencere geçişi
fail-closed test edilir. Yeni hedefli test paketi **21/21 geçti**. Geniş ADS-B/parser regresyonu
**242 geçti / 1 deselect** ile tamamlandı; deselect yalnız ADR-032'de kayıtlı kayıp frozen
`features.py` hash testidir. Bu sayılar yalnız yazılım/smoke kanıtıdır; gerçek-veri detection
metriği değildir.

**Açık madde:** Toplam operasyonel alert-alpha/burden bütçesi ve sayısal config kullanıcı
tarafından henüz dondurulmadı. Bu nedenle gerçek normal fit/calibration, truth-v2 evaluation,
aday terfisi ve holdout erişimi yapılmadı. Açık sayılar
`docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md` içinde listelenmiştir; sonuç görülerek
aynı namespace'te doldurulamaz.

## ADR-034: Contextual-physics v1 normal-only eğitim config'i sonuç öncesi donduruldu

- Durum: Eğitim için onaylandı; alarm/threshold seçimi kapsam dışı
- Tarih: 2026-07-14

**Karar:** Kullanıcının “bir eğitim yapalım” onayıyla tek `contextual_physics_v1` eğitim
config'i `configs/adsb_contextual_physics_v1_train.json` içinde sonuç görülmeden donduruldu.
Step-5 manifestindeki yalnız fit rolünden deterministik yüzde 2 uçuş (seed 20260714) model/scaler
için; ayrık calibration rolünden yüzde 2 uçuş (seed 20260715) yalnız doğal magnitude diagnostic
için seçilir. Config 12-history/1-horizon, lagged 3-row phase, 2/5/15 s cadence, 60 s max gap,
robust clip 5, hidden 32/tek LSTM, bounded scale 0.1–5.0, beş epoch, batch 512, learning-rate
0.001, seed 0 ve scaled active kanallara explicit 1.0 loss ağırlığı kullanır. Sweep yoktur.

**Kanıt:** `scripts/adsb_train_contextual_physics_v1.py` 237 fit-day Parquet girdisinin
Step-5 byte/SHA kayıtlarını doğrular; seçili source'ları part içinde filtreleyip feature/pencereyi
bellek sınırlı üretir. Sentetik/truth-v2/development/rehearsal/holdout okumaz. Run manifestini
optimizer'dan önce, scaler ve derived model config'ini ilk optimizer adımından önce yazar; tracked
Git temizliği ile başlangıç/son code hash eşitliğini zorunlu tutar. Runner unit/integration smoke
testleri **3/3**, önceki contextual hedefli paket **21/21** geçti.

**Açık madde:** Bu ADR bir detection sonucu değildir. Model eğitimi henüz çalıştırılmadı;
sayısal alarm bütçesi, conformal alpha ve anomaly-profile temporal eşikleri hâlâ kullanıcı kararı
bekler ve bu eğitim koşusunda seçilemez.

## ADR-035: Contextual-physics v1 normal-only eğitim tamamlandı; magnitude gate PASS

- Durum: Model eğitildi; threshold/anomaly fayda değerlendirmesi henüz yapılmadı
- Tarih: 2026-07-14

**Karar:** Dondurulmuş `0a0ed73` commit'i ve
`configs/adsb_contextual_physics_v1_train.json` ile
`artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1` değişmez koşusu tamamlandı.
Model `trained_not_thresholded` durumundadır; bu sonuç ana anomaly detector terfisi veya Step-7
FAIL kararının geri alınması değildir.

**Kanıt:** Step-5 fit rolündeki 149.462 uçuştan deterministik 2.929 uçuş ve 1.267.625 satır
scaler/model fit için seçildi. Beş epoch'un her birinde 1.180.160 pencere ve 2.417 batch görüldü;
weighted Gaussian NLL sırasıyla **0,795375 / 0,738245 / 0,723818 / 0,714755 / 0,708696** oldu.
Toplam süre **1.702,158 s** idi. `vertical_rate_residual`, `speed_residual`, `heading_residual`,
`east_velocity_residual` ve `north_velocity_residual` aktif; `altitude_source_residual` MAD=0
nedeniyle floorsuz dışlandı.

Ayrı 770 natural-calibration diagnostic uçuşunda 332.510 pencere skorlandı; optimizer veya
threshold seçimine geri beslenmedi. Trained-vs-untrained Spearman rho **0,649633**,
trained-vs-target-magnitude rho **0,654240** ve `magnitude_domination_flagged_at_0_8=false` oldu.
Bu yalnız magnitude gate PASS'tir; synthetic AUC/recall ölçülmedi. Checkpoint 9.546 sonlu
parametreyle `strict=True` yeniden yüklendi.

Model checkpoint SHA-256
`3a04b2ceb64b2f11df8f8c108a2721a5dbdff96b49080028c56902e2ce3d354d`, training report
SHA-256 `16f87e44a056e070fef90d26217c0481cf0ea6f91ab6470321a1a78f25e1a5f4`, run manifest
SHA-256 `833c6705953a21c09d7450efa059d981ecaf6b84e65c8541bfa7c2fbea70b555` ve checksum index
SHA-256 `de7a2b04837089671f60c80c38098c3260f11a2180ef82949c35e4ee67b2a44b`'dir. İndeksteki
5/5 dosyanın byte/SHA değeri yeniden doğrulandı. Sentetik eğitim satırı 0; threshold sweep,
truth-v2, development, rehearsal ve holdout erişimi yok; kod/Git koşu boyunca değişmedi.

**Açık madde:** Sonraki bilimsel adım için kullanıcı toplam operasyonel alert-alpha/burden
bütçesini ve channel paylarını sonuçtan bağımsız olarak tanımlamalıdır. Bundan sonra ayrı natural
calibration conformal tail, development/rehearsal burden ve en son truth-v2 event recall/delay/
active-coverage ölçümü yapılabilir. Eğitim loss'u veya magnitude PASS tek başına detection
başarısı değildir.

## ADR-036: Contextual eğitim sonucu matematiksel olarak yorumlandı; detection sonucu ayrıştırıldı

- Durum: Raporlandı; calibration/evaluation bütçe kararı bekliyor
- Tarih: 2026-07-14

**Karar:** `docs/adsb_overall_model_report_2026-07-14.md`, aktif contextual modelin uçuş-içi
time-series sözleşmesini, residual denklemlerini, Gaussian NLL eğitimini, kanal-bazlı
standardized-surprise skorunu ve sonraki conformal/temporal karar katmanını tek raporda kaydeder.
Mevcut `rho < 0,8` magnitude PASS bir detection başarısı veya Step-7 kararının geri alınması
olarak sunulmaz.

**Kanıt:** Beş epoch loss'u, 332.510 doğal diagnostic pencere, kanal p95'leri, iki Spearman rho,
9.546 parametreli strict checkpoint ve sentetik fit/calibration=0 değerleri ADR-035'in hashli
artefaktından alınmıştır. Eski NN, corrected rule ve doygun CUSUM kıyasları mevcut ADR-024/025/
028/030/032 kayıtlarından taşınmış; yeni contextual aday için AUROC/AUPRC/recall uydurulmamıştır.

**Açık madde:** Kullanıcı toplam doğal alarm episode bütçesini ve channel paylarını sayısal olarak
dondurmadan conformal calibration, temporal threshold, development/rehearsal veya truth-v2
değerlendirmesi çalıştırılmaz. Üçlü holdout havuzu açılmaz.

## ADR-037: Alarm bütçesi Pareto ızgarası + kanal payı + temporal profil ön-kaydı

- Durum: Kabul edildi — sonuç görülmeden dondurulmuş
- Tarih: 2026-07-14

**Karar:** ADR-036'nın açık maddesi kapatıldı. Kullanıcı üç kararı AskUserQuestion ile onayladı:
(1) toplam bütçe tek sayı yerine 5 noktalık Pareto ızgarası
(`[0.1, 0.5, 1.0, 2.0, 5.0]` episode/100 scoreable uçuş-saat); (2) kanal/S2 payı kanıta ağırlıklı;
(3) instant/persistence/accumulation temporal eşikleri Claude tarafından önerilip onaylandı.
Tam türetim ve gerekçe `docs/adsb_contextual_physics_v1_alarm_budget_prereg_2026-07-14.md` ve
makine-okunur `configs/adsb_contextual_physics_v1_alarm_budget.json`'dadır.

**Kanıt:** Kanal payları ADR-028'in corrected truth-v2 pooled reçete AUROC'larından
(ground-speed 0.974023, track 0.889552, vertical-rate 0.696337) `skill=AUROC-0.5` normalize
edilerek türetildi. `east_velocity_residual`/`north_velocity_residual` için eski skaler `ramp`
AUROC'u (0.558927) BİLEREK kullanılmadı — ADR-029 bu temsili tam olarak bu zayıflığı gidermek için
terk etmişti; onun yerine kanıtlanmış en zayıf kanalın (vertical_rate, skill 0.196337) payı taban
alındı. S2 veri-kalitesi katmanı AUROC taşımadığı için (deterministik bayrak, öğrenilmiş residual
değil) ayrı, sabit %15 pay aldı; kalan %85 beş fizik kanalına orantılı bölündü. Nihai paylar
toplamı `budget_shares_of_total` alanında test edilebilir şekilde tam 1.0'dır. Persistence penceresi
30s olarak, track kanalının zaten gözlenen ~57s doğal gecikmesini kötüleştirmeyecek şekilde
seçildi. CUSUM `h` eşiği bu belgeyle SEÇİLMEDİ — ADR-029 sözleşmesi gereği her Pareto noktasında
doğal kalibrasyondan ayrı türetilecek.

**Açık madde:** Gerçek conformal calibration/development/rehearsal koşusu henüz başlatılmadı; bu
ADR yalnız bütçe/pay/profil yapısını dondurur. Kullanıcı ayrıca Isolation Forest'ın yeni
residual/bağlam çerçevesinde paralel, ayrı ön-kayıtlı bir keşif olarak şimdi başlatılmasını
onayladı (bkz. ADR-038).

## ADR-038: Isolation Forest — paralel keşif, ayrı namespace

- Durum: Kabul edildi — kod yazıldı, gerçek veriyle ilk magnitude self-check bekliyor
- Tarih: 2026-07-14

**Karar:** `isolation_forest_contextual_v1` adıyla `contextual_physics_v1`'i bloklamayan, onu
ikame etmeyen paralel bir keşif açıldı. Gerekçe: IF çok-değişkenli izolasyona bakar,
`contextual_physics_v1` zamansal-sürprize — kör noktaları farklı olabilir, füzyon AYRI bir
ön-kayıt gerektirir, bu karar şimdi verilmedi. Tam sözleşme
`docs/adsb_isolation_forest_contextual_v1_prereg_2026-07-14.md`'dedir.

**Kanıt:** `adsb/models/isolation_forest_residual.py` yazıldı — `contextual_physics_v1` ile
BİREBİR aynı 5 residual kanalını ve aynı `StrictNaturalRobustScaler`'ı (medyan/MAD, clip=5.0,
MAD=0 floor'suz dışlama) paylaşır; `fit()` yalnız `natural_clean_fit` rolünü kabul eder ve
`contains_synthetic=True` bayrağında ValueError fırlatır (çalışma-zamanı zorlaması, konvansiyon
değil). Skorlama yalnız `score_samples()` üzerinden sürekli değer döner; `predict()`'in ikili
çıktısı hiçbir yerde kullanılmaz. 6 test yazıldı (synthetic-reddi, MAD=0 dışlama, aykırı-değer
sıralaması, NaN satırın düşürülmeden NaN skorlanması, determinizm) — 6/6 geçti.

**Bilinen sınırlama (dürüstçe beyan):** LSTM tarafının availability mask'i burada yok — aktif 5
kanaldan biri NaN olan satır TAMAMEN atılır (complete-case). Eksik veri deseni zaten ayrı S2
katmanında yakalandığı için örtüşme yok, ama IF'in kapsamı LSTM'den yapısal olarak daha dar.

**İlk magnitude self-check sonucu (aynı gün, gerçek veriyle):** `scripts/adsb_isolation_forest_
magnitude_check.py`, Step-5 manifestinin fit-rolünden 40/237 parçalık bir alt-örneklemle
(12.003.593 fit satırı/26.241 uçuş, ayrı 2.573.412 skorlanabilir diagnostic satırı/6.561 uçuş)
koşuldu — tam-hacim/hash-zincirli üretim koşusu DEĞİL, bilinçli bir keşif alt-örneklemi.

**Sonuç, dürüstçe:** `rho_trained_vs_magnitude = 0.995958`, `rho_trained_vs_shuffled_channel_fit
= 0.995841`. İkisi de contextual_physics_v1'in kendi rho'sundan (0.65) ve eski üç NN'in
FLAGGED sınırından (0.84–0.94) ÇOK daha yüksek. Kanal-bazında karıştırılmış (yapısı bozulmuş)
veriyle fit edilen IF, gerçek veriyle fit edilenle NEREDEYSE AYNI sıralamayı üretiyor — bu, mevcut
5 kanallı ham IF'in gerçek çok-değişkenli yapı öğrenmediğini, fiilen ölçekli-residual'ların
öklid/Mahalanobis-benzeri büyüklüğünü ölçtüğünü gösteriyor. Beklenen bir matematiksel sonuç:
bağlamdan (phase/cadence) bağımsız, düz bir IF, farklı uçuş rejimlerinin (tırmanış/cruise/iniş)
doğal olarak farklı residual büyüklük aralıklarını "izolasyon" sanıyor olabilir.

**Yorum:** Bu, IF'i şu haliyle DEĞERLİ bir tamamlayıcı sinyal olarak DESTEKLEMİYOR — mevcut ham
hâliyle basit bir büyüklük kuralından anlamlı şekilde ayrışmıyor. Füzyona veya kıyasa
sokulmayacak. Tek makul sonraki adım (denenmedi, ayrı ön-kayıt gerektirir): contextual model
gibi phase/cadence-koşullu, AYRI IF fit'leri — düz/havuzlanmış IF'in bu sonucu tekrar
üretmeyeceği garanti değil, bu yüzden iddialı bir beklenti kurulmuyor.

## ADR-039: contextual_physics_v1 — ilk gerçek natural-calibration turu

- Durum: Tamamlandı (keşif alt-örneklemi) — alarm/burden ölçümü henüz yok
- Tarih: 2026-07-14

**Karar:** Veri rolü sırasının 2. adımı (`docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md`)
ilk kez gerçek veriyle çalıştırıldı. `scripts/adsb_contextual_physics_v1_calibrate.py`,
eğitilmiş checkpoint'i (`artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1/`) yükler,
Step-5 split_contract'inin `calibration` rolündeki (37.208 uçuş, fit ile aynı günden —
2026-02-28 — ama fit'ten tamamen ayrık) satırları skorlar ve
`adsb/conditional_calibration.py::HierarchicalConformalCalibrator`'i fit eder.

**Kanıt:** 237 fit-günü parçasının 60'lık alt-örneklemi (tam-hacim değil, bilinçli keşif ölçeği)
tarandı: 8.182 calibration-rolü uçuş bulundu (split_contract'teki 37.208'in ~%22'si — 60/237≈%25
ile tutarlı), 3.321.776 skorlanabilir pencere, 16.001.483 satırlık uzun-format (channel, phase,
cadence, score) kalibrasyon tablosu. `min_group_size=1000` (ADR-037/Q2'deki türetim: hedeflenen
en küçük p'nin ~10 katı) ile hiyerarşik gruplama sağlıklı çıktı: `speed_residual`/
`heading_residual`/`east_velocity_residual`/`north_velocity_residual` için 20 phase×cadence
kombinasyonundan 19'u DOĞRUDAN destekli (fallback'e nadiren düşülecek); `vertical_rate_residual`
biraz daha zayıf (16 kombinasyondan 12'si doğrudan destekli) ama channel-seviyesi fallback her
zaman mevcut.

**Yol boyunca bulunan bug:** (1) `split_contract`'teki flight_id'ler zaten `"2026-02-28:"`
önekini taşıyor — kodum bunu bir kez daha ekleyip sıfır eşleşme üretti, düzeltildi. (2) Aynı
oturumun daha önce iki kez düzelttiği desen tekrarladı: `contextual_channel_scores`, 3.3M
pencereyi TEK forward'ta işlemeye çalışıp ~29.8GB istedi ve çöktü — `_score_batched()` (20.000'lik
gruplar) ile düzeltildi.

**Açık madde:** Bu yalnız kalibrasyon-sağlığı raporudur — hiçbir alarm üretilmedi, hiçbir p-değeri
eşiklenmedi. Sıradaki adım: ADR-037'nin Pareto ızgarası + kanal bütçe paylarını kullanarak
natural-development verisinde (rol #3) gerçek alfa-arama/doğal-yük ölçümü. Tam-hacim (237/237)
koşusu ve hash-zincirli manifest de henüz yapılmadı — bu keşif turu, üretim kalibrasyonu değil.

## ADR-040: contextual_physics_v1 — ilk gerçek doğal-yük (development) ölçümü

- Durum: Tamamlandı (çok küçük ölçekli ilk bakış) — operasyonel karar için YETERSİZ, sıradaki
  büyütmenin temelini kanıtlıyor
- Tarih: 2026-07-14

**Karar:** Veri rolü sırasının 3. adımı (natural development) ilk kez gerçek veriyle, uçtan uca
çalıştırıldı: `scripts/adsb_contextual_physics_v1_development_burden.py` — ADR-039'daki
kalibrasyon çıktısını yeniden üretip (2026-02-28), donmuş modeli 2026-03-01 (development günü,
hiç fit/calibration görmemiş) verisinde skorlar, `adsb/contextual_decision.py`'nin
instant/persistence karar profilleriyle önceden-belirlenmiş bir alfa ızgarasında doğal alarm
yükünü (episode/scoreable uçuş-saat) ölçer. Sonuç bir ARAMA değil, sabit ızgarada bir ÖLÇÜMdür.

**Yol boyunca iki gerçek performans sorunu bulundu:** (1) development gününün parça-başına
yoğunluğu fit/calibration gününden çok daha fazla çıktı — 15 parçada bile pencereleme tek
seferde 4.56GB'lık tek array istedi, bellek patladı; parça-parça (chunk) işleyip skorları
biriktirmeye çevrildi. (2) `apply_detector_profile`'in satır-başına `.loc` atamalı Python
döngüsü, 8-alfa × 5-profil taramasında 2.1M satırda 15+ dakikada tek kanalı bile bitiremedi —
bu ÇÖZÜLMEDİ, yalnız ölçek (1 parça, 4-nokta alfa ızgarası) küçültülerek etrafından geçildi.
Büyük ölçekli bir sonraki turda bu döngü gerçek bir darboğaz olacak, ayrı bir performans işi
gerekecek.

**Kanıt (704 uçuş, TEK development parçası — istatistiksel olarak KÜÇÜK, yalnız mekanizma
kanıtı):** 5 profilin 4'ünde (freeze, bias, inconsistency, ve spike'ların küçük-alfa ucunda)
en gevşek test edilen alfa'da bile doğal yük ÇOK düşük çıktı (ör. `speed_bias` alfa=3.68e-4'te
0.0053 episode/saat, `heading_inconsistency` aynı alfa'da 0.0068 episode/saat) — Pareto
ızgarasının EN SIKI noktası (0.1 episode/100 saat = 0.001/saat) bile bu aralıkta rahatça
karşılanabilir görünüyor. `vertical_rate_spike` farklı davrandı: alfa=0.0136'da bile 4.89
episode/saat'e sıçradı — bu kanalın doğal conformal p-değeri dağılımı diğerlerinden çok daha
"yayvan," aynı hedefe çok daha küçük bir alfa gerektiriyor. En gevşek alfa (0.5) her profilde
beklendiği gibi doygunluğa yakın (7.6–11.4 episode/saat).

**Bilinen, dürüstçe beyan edilen kapsam daralması:** (1) yalnız 3/5 kanal ölçüldü —
`east_velocity_residual`/`north_velocity_residual` (stealthy ramp accumulation) bu turda HİÇ
çalıştırılmadı, kod yalnız 3 kanal için profil tanımlıyor; ADR-037'nin öngördüğü ortak 2-eksenli
CUSUM entegrasyonu da yapılmadı. (2) tek development parçası (216'nın 1'i), 704 uçuş — hiçbir
operasyonel "bu alfa'yı kullanalım" kararı bu sayıdan verilemez. (3) donmuş rehearsal (rol #4)
ve truth-v2 karşılaştırması (rol #6) hâlâ başlamadı.

**Açık madde:** Sıradaki adım öncelik sırasıyla: (a) `apply_detector_profile` sweep'i için
vektörleştirilmiş/hızlı bir yol (mevcut fonksiyon değiştirilmeden, ayrı bir performans katmanı);
(b) east/north kanallarının accumulation profillerini eklemek; (c) ölçeği kademeli büyütmek
(1→10→50 parça); (d) yalnız bundan sonra donmuş rehearsal'a geçmek.

## ADR-041: contextual_physics_v1 — performans darboğazı çözüldü, ölçek 20x büyütüldü,
east/north kanalları ortak Page-CUSUM ile ilk kez ölçüldü

- Durum: Tamamlandı (hâlâ tam-hacim değil, ama ADR-040'ın iki açık maddesi kapandı)
- Tarih: 2026-07-14

**Karar A — performans darboğazı:** ADR-040'ın çözülmemiş bıraktığı `apply_detector_profile`
satır-başına `.loc` atamalı Python döngüsü, dondurulmuş fonksiyon DEĞİŞTİRİLMEDEN, yanına ayrı
bir performans katmanı olarak ele alındı: `adsb/contextual_decision_fast.py::
apply_detector_profile_fast`. Aynı tam recurrence'ı (instant/persistence/accumulation, gap-reset
dahil) tekrar eder ama pandas `.loc` yazma yerine düz numpy dizileri kullanır ve DataFrame'e
sonunda TEK seferde yazar; `instant` modu için döngü tamamen gereksiz (satırlar arası durum
taşımıyor), tam vektörize edildi. `tests/test_adsb_contextual_decision_fast.py` (8 test:
3 mod × 2 senaryo + tek-satır uç durumu + karışık-kanal reddi) eski/yeni fonksiyonun BİT-BİT
aynı `alarm`/`temporal_evidence`/`reset_reason` çıktısını verdiğini kanıtlıyor. Ölçüm: 300
uçuş / 144.856 satırlık sentetik veride eski fonksiyon 17.05s, yenisi 0.11s (~150x).

**Karar B — east/north kanalları:** ADR-040'ta hiç ölçülmemiş `east_velocity_residual`/
`north_velocity_residual`, ADR-037/ADR-029'un öngördüğü ORTAK 2-eksenli Page-CUSUM ile
ölçüldü — `adsb/cusum.py::VectorPageCUSUM`, mevcut kural-skorlayıcı hattından SIFIRDAN
YAZILMADAN yeniden kullanıldı (`scripts/adsb_contextual_physics_v1_cusum_burden.py`, yeni).
`threshold_h`'nin p-değeri gibi doğal bir aralığı olmadığından (config bunu açıkça "derived
per budget_grid point from natural calibration" olarak bırakmıştı), sıra: (1) fit-rolünde
(20 parça, 4.698.464 satır/10.671 uçuş) median/MAD kalibrasyonu — hiçbir kanal dışlanmadı;
(2) calibration-rolünde (60 parça, 3.568.945 satır/3.158.246 değerlendirilebilir) TEK bir
`score_rows()` geçişiyle (h'den bağımsız) skor dağılımından 16 adet h adayı türetildi ve her
Pareto bütçe noktasına en yakın h donduruldu; (3) o donmuş h'ler development-rolünde (20 parça,
7.763.478 satır/6.787.342 değerlendirilebilir, 15.467 uçuş, 13.928 uçuş-saat), DEĞİŞTİRİLMEDEN
uygulandı.

**Kanıt B:** Bu ortak dedektör diğer üç kanaldan (hız/yön/dikey-hız) belirgin şekilde daha
sessiz: Pareto=2.0 hedefinde (h=155.88) development'ta 0.00474 episode/saat, Pareto=5.0'da
(h=93.49) 0.02039 episode/saat. Kalibrasyon→development genelleme kalitesi bütçe noktasına göre
DEĞİŞTİ: gevşek uçta (V=2.0, V=5.0) iki gün arası oran ~1.15–1.26x (iyi genelleme); en sıkı uçta
(V=0.1, h=312.54) kalibrasyon tahmini SADECE TEK bir gözlenen episode'a dayanıyordu (0.000157/saat)
ve development'ta ~7.8x daha yüksek çıktı (0.00122/saat) — mutlak sayılar hâlâ küçük ama bu,
en sıkı bütçe noktasının şu ölçekte GÜVENİLİR olmadığının açık kanıtı.

**Kanıt A (büyütülmüş ölçek, 3 kanal):** Aynı 3 kanal (hız/yön/dikey-hız), development parçası
1→20'ye, alfa ızgarası 4→12 noktaya çıkarıldı (7.252.704 skorlanabilir pencere, ~14.800–16.300
uçuş-saat/kanal — ADR-040'ın 704 uçuşuna göre ~20x). ADR-040'ın "vertical_rate_spike diğerlerinden
çok daha alfa-hassas" bulgusu ÇOK daha sağlam istatistikle doğrulandı: alfa=1.91e-4'te
`vertical_rate_spike` 0.0927/saat iken aynı alfa'da `heading_inconsistency` 0.0040/saat,
`speed_bias` 0.0010/saat — ~10-90x fark. Pareto=1.0 hedefine en yakın noktalar: vertical_rate_spike
alfa=1e-5→0.0035/saat (izgaranın en sıkı ucu bile hedefin üstünde kalıyor), vertical_rate_freeze
alfa=1.91e-4→0.0008/saat, speed_spike alfa=1e-5→0.0043/saat, speed_bias alfa=5.11e-4→0.0035/saat,
heading_inconsistency alfa=7.15e-5→0.0015/saat.

**Yeni, beklenmedik gözlem:** Çok gevşek alfa'larda (0.187→0.5) episode/saat sayısı bazı
profillerde DÜŞÜYOR (ör. vertical_rate_spike 14.92→7.46/saat) — bu alarmın azaldığı anlamına
gelmiyor; `alerted_flight_fraction` aynı aralıkta hâlâ artıyor (%98→%99). Neredeyse her satırın
alarm verdiği rejimde, 60 saniyelik episode-birleştirme kuralı çok sayıda ardışık satırı TEK
uzun episode'a indiriyor — episode/saat metriği tek başına gevşek alfa ucunda yanıltıcı,
`alerted_flight_fraction` ile birlikte okunmalı. Kayıt altına alınan bir metodoloji notu, bir
model/decision değişikliği değil.

**Bilinen, dürüstçe beyan edilen kalan boşluklar:** (1) hâlâ tam-hacim değil — LSTM tarafı
20/216 development parçası, CUSUM tarafı aynı 20/216 + 60/237 calibration; (2) CUSUM'un h
adayları yalnız 16 noktalık bir quantile ızgarası — en sıkı Pareto noktasında (V=0.1) tek-episode
istatistiğine dayandığı için bu nokta GÜVENİLMEZ ilan edildi, kullanılmamalı; (3) donmuş rehearsal
(rol #4) ve truth-v2 karşılaştırması (rol #6) hâlâ başlamadı; (4) `apply_detector_profile_fast`
sadece decision-katmanını hızlandırıyor — LSTM forward-pass ve pencereleme adımları ayrı, onlar
zaten `_score_batched`/parça-parça yükleme ile bellek-güvenli ama hız-optimize değil.

**Açık madde:** Sıradaki adım: (a) ölçeği tekrar büyütmek (20→50→tam 216/237 parça); (b) CUSUM
h-adayları ızgarasını sıkı uçta daha ince hale getirmek (daha fazla calibration parçası, en sıkı
Pareto noktası için); (c) yalnız bundan sonra donmuş rehearsal'a (rol #4) geçmek.

## ADR-042: contextual_physics_v1 — rol #4 (donmuş rehearsal) ve rol #5 (truth-v2 gerçek
recall) tamamlandı; dondurulmuş alarm bütçeleri gerçek olay yakalama için çok sıkı çıktı

- Durum: Development rejected (mevcut Pareto bütçe noktalarında); yeni, geniş bir bütçe
  ızgarası için ayrı ön-kayıt gerektiren açık, önemli bir bulgu
- Tarih: 2026-07-16

**Karar — rol #4 (rehearsal):** `scripts/adsb_contextual_physics_v1_rehearsal.py` (yeni),
ADR-041'de dondurulmuş alfa (LSTM, 5 profil) ve h (CUSUM) değerlerini, hiçbir yeni seçim
yapmadan, üçüncü, bağımsız bir günde (2026-03-16, rehearsal rolü — ne fit ne calibration ne
development görmüş) aynen uyguladı. 60 calibration + 20 fit + 20 rehearsal parçası tarandı
(8.102.088 LSTM penceresi, 8.643.797 CUSUM satırı). **Sonuç: çoğunlukla kararlı** — orta/gevşek
Pareto noktalarında (V≥0.5) gerçekleşen oran hedefin ~0.3–2 katı arasında kaldı; heading_
inconsistency en istikrarlısı (hedefin %93–130'u, her noktada). En sıkı nokta (V=0.1) yine en
kırılgan: vertical_rate_spike hedefin 15 katı, speed_spike 11 katı çıktı — ADR-041'de zaten
görülen "tek-episode istatistiği kırılgan" örüntüsünün üçüncü bağımsız günde de tekrarı.

**Karar — rol #5 (truth-v2):** `scripts/adsb_contextual_physics_v1_truth_v2_eval.py` (yeni),
gerçek/enjekte edilmiş 8.910 uçuşluk sentetik truth-v2 corpus'unu (`data/objectstore/synthetic/
adsb_v2_20260713_01`, yalnız değerlendirmede kullanıldı — fit/calibration/scaling'e hiç girmedi)
ADR-041'de dondurulmuş alfa/h değerleriyle skorladı. 4 recipe 5 fizik kanalıyla eşleşti
(`vertical_rate_frozen`→vertical_rate_residual, `ground_speed_biased`→speed_residual,
`track_frozen`→heading_residual, `position_ramp_stealthy`→east/north CUSUM); `altitude_dropout`
kapsam dışı bırakıldı (S2 veri-kalitesi katmanının işi, NN/CUSUM fizik kanallarıyla eşleşmiyor).

**Ön-kontrol (bug mü gerçek mi):** Küçük bir örneklemde (200 uçuş) recall neredeyse sıfır
çıkınca, önce alfa'yı kasıtlı olarak çok gevşek (0.5) bir değere zorlayıp aynı testi tekrar
çalıştırdık. Recall %74–92'ye çıktı (ama karşılığında temiz veride saatte ~9 alarm — kullanılamaz).
Bu, olay-hizalama/alarm mekanizmasının DOĞRU çalıştığının kanıtı; sorun kod değil, bütçe.

**Kanıt (tam corpus, 8.910 olay):**

| Profil | Recall (V=0.1, en sıkı) | Recall (V=5.0, en gevşek) | Gecikme medyanı (V=5.0) |
|---|---|---|---|
| vertical_rate_spike | %0.06 | %0.06 (alfa zaten izgaranın en sıkı ucuna sabit) | 316 s |
| vertical_rate_freeze | %0 | %0.31 | 682 s |
| speed_spike | %2.53 | %5.60 | 0 s (anlık) |
| speed_bias | %0.01 | %1.28 | 20 s |
| heading_inconsistency | %0 | %0.09 | 189 s |
| east_north_cusum (ortak) | %7.13 | **%49.72** | 1101 s |

**En önemli bulgu:** 5 profilin 4'ünde (spike/freeze/bias/inconsistency) recall Pareto=5.0'da
bile **%6'nın altında** — dondurulmuş bütçelerle gerçek anomalilerin neredeyse hiçbiri
yakalanmıyor. Kök neden ölçüldü, tahmin değil: bütçe "100 skorlanabilir uçuş-SAATİNDE kaç
alarm" birimindeydi (ADR-037), ama truth-v2'de her olay TEK bir uçuşun ~0.5–1 saatlik penceresinde
kanıtlanmak zorunda. Hedef oranı (saatte 0.0001–0.06) o pencereye çarpınca beklenen alarm sayısı
zaten yüzde birkaçın altında çıkıyor — model zayıf olduğu için değil, bütçe birimi ile
değerlendirme birimi uyuşmadığı için. Yukarıdaki gevşek-alfa kontrolü bunu doğruluyor.

**Ayrı ve önemli bir gözlem — CUSUM belirgin şekilde daha iyi genelledi:** `east_north_cusum`
(ADR-029'un ortak, sürekli-biriken Page-CUSUM istatistiği) aynı doğal-yük mantığıyla kalibre
edildiği hâlde V=0.1'de bile %7.1, V=5.0'da %49.7 recall verdi — LSTM tabanlı beş profilin
hepsinden kat kat yüksek. Muhtemel neden: CUSUM kanıtı ham (robust-z) alanda, uçuş boyunca
sıfırlanmadan sürekli biriktiriyor; LSTM'in persistence/accumulation modları ise conformal
p-değeri üzerinden çalışıyor ve yalnız TEK uçuş süresiyle sınırlı — kanıt birikecek zamanı
yapısal olarak daha az. Bu, sonraki tasarım turunda ciddiye alınması gereken somut bir ipucu.

**Karar (metodoloji disiplini gereği):** ADR-037'nin dondurulmuş Pareto ızgarası/bütçe payları
BU ADR'de DEĞİŞTİRİLMEDİ — sonucu görüp aynı run içinde parametre düzeltmek, projenin ML-0..16
hattını çökerten tam hatadır. Bunun yerine: (1) bu bulgu olduğu gibi kaydedildi; (2) yeni, çok
daha geniş bir Pareto ızgarası (ör. 100 saatte 5 değil, 50–500 arası) ve/veya persistence/
accumulation pencere tasarımının (tek-uçuş sınırı yerine daha uzun kanıt birikimi) yeniden
tasarımı için AYRI, sonuç görülmeden yazılacak bir ön-kayıt gerekiyor — bu oturumda başlatılmadı.

**Bilinen sınırlamalar:** Ölçek hâlâ tam değil (calibration 60/237, rehearsal 20/185,
truth-v2 corpus'un tamamı ama yalnız 4/5 recipe NN kapsamında). `active_interval_coverage`
(alarmın olayı ne kadar kapsadığı) çoğu profilde neredeyse sıfır; CUSUM'da V=5.0'da %30.9 —
bu da ayrıca izlenmesi gereken bir metrik. Rol #6 (ADR-025 kuralıyla eşit-bütçeli kıyas) ve
rol #7 (üçlü kör holdout) hâlâ başlamadı; holdout ayrı bir unseal kararı olmadan açılmayacak.

Artifact'lar: `artifacts/adsb/runs/20260715_contextual_physics_v1_rehearsal_v1/rehearsal_report.json`,
`artifacts/adsb/runs/20260715_contextual_physics_v1_truth_v2_eval_v1/truth_v2_eval_report.json`.
