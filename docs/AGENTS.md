# AGENTS.md — Proje Bağlamı (her Codex/Claude Code session'ı bunu okumalı)

## Bu proje ne?
8 haftalık İHA (UAV) veri mühendisliği internship projesi. 3 stajyer, Hafta 1-4 ortak
"Bronze→Silver→Gold" veri pipeline'ı, Hafta 5-8 bireysel projeler. Bu repo'yu yazan kişinin
bireysel projesi: **UAV anomali tespiti** (Isolation Forest, Autoencoder — etiketli/etiketsiz
anomali tespiti literatürü temelinde).

## Şu an neredeyiz?
**Mimari ADR-003 ile değişti (2026-07-01, `docs/PIPELINE_PLAN.md`).** Artık "sadece Bronze"
değiliz — Bronze=raw upload, Silver=parse/etiket/provenance her kaynak için bağımsız
aktif. **Gold da yazıldı** (`src/gold/unify.py`, ADR-005, 2026-07-01): şu an sadece `alfa` ve
`uav_attack` Silver'ları mevcut olduğu için Gold onlarla çalışıyor; `adsblol_hist`/
`adsblol_rt` için kolon eşlemesi tabloda hazır (`COLUMN_MAPS`), Metehan'ın/Yusuf'un Silver
parser'ları `src/silver/`'a taşınınca hiçbir kod değişikliği gerekmeden otomatik dahil
olacaklar (eksik kaynak sessizce atlanıyor, hata vermiyor).

> **Not:** Plan "Silver review tamamlanmadan Gold'a başlanmaz" diyordu — bu adım kullanıcının
> açık isteğiyle (review beklenmeden) şimdi yazıldı. Ekip review'unda bunu göz önünde bulundurun.

Detaylı plan ve her kişinin adım adım rehberi: `docs/PIPELINE_PLAN.md` (her session'dan
önce oku). Mimari kararlar: `docs/decisions.md` (ADR-001..005).

## Ekip ve sorumluluklar (docs/PIPELINE_PLAN.md)
| Kişi | Veri kaynağı | Bronze | Silver |
|---|---|---|---|
| Metehan | adsb.lol historical (tar) | `src/ingestion/upload_raw.py --source adsblol_historical` | `src/silver/parse_adsblol_historical.py` |
| Yusuf | adsb.lol realtime (Kafka) | `src/ingestion/adsblol_producer.py` + `adsblol_consumer.py` | `src/silver/parse_adsblol_realtime.py` |
| **Anıl** | **ALFA + UAV Attack** | `src/ingestion/upload_raw.py` | `src/silver/parse_alfa.py` + `parse_uav_attack.py` |

Herkes kendi bölümünü bağımsız yürütür. Gold'da hepsi birleşecek (henüz yok).

## Katman kuralları
| Katman | Ne yapar | Ne YAPMAZ |
|---|---|---|
| **Bronze** | Ham dosyayı MinIO'ya olduğu gibi yükler (`write_bronze_bytes`) | Parse, unit dönüşümü, filtre, provenance |
| **Silver** | Kaynak-özel parse + unit dönüşümü + etiket + provenance (`add_provenance`, varsayılan `schema_version="silver_v1"`) | Kaynakları birleştirme |
| **Gold** | Tüm kaynakları 7+3 ortak kolona hizalar (`src/gold/unify.py`) | Kaynak-özel kolon üretme |

**Coğrafi filtre YOK** — Türkiye bbox dahil hiçbir geo-filtre pipeline seviyesinde
uygulanmaz (`src/common/bbox.py` silindi); tüm dünya verisi saklanır, filtreleme
analiz/notebook aşamasında yapılır.

## Anıl'ın bölümü (ALFA + UAV Attack) — durum
- `src/ingestion/upload_raw.py`: ham `.zip`'i (ALFA `processed.zip`, UAV Attack
  `UAVAttackData.zip`) değiştirmeden `bronze/<source>/<dosya adı>` altına yükler.
- `src/silver/parse_alfa.py`, `parse_uav_attack.py`: Bronze'daki zip'i indirir, parse eder
  (transform mantığı eski `src/bronze2silverParsers/parse_alfa.py`/`parse_uav_attack.py`'den
  DEĞİŞMEDEN taşındı — tek istisna: UAV Attack'in log_id/topic regex'i gerçek dosyalarda
  kanıtlanmış şekilde kırıktı, düzeltildi, bkz. ADR-003), `write_silver` ile Silver'a yazar.
  Gerçek veriyle doğrulandı: `scripts/run_alfa_local.py` / `scripts/run_uav_attack_local.py`
  (Docker/MinIO gerektirmeden, `FakeMinioClient` ile).
- `src/processing/alfa_silver.py`/`uav_attack_silver.py`/`gold.py`: ADR-003'ten ÖNCEKİ,
  daha geniş bir Silver denemesi — artık aktif pipeline değil, referans olarak duruyor
  (bkz. ADR-004, `docs/silver_schema.md`). Silinmedi çünkü Silver şeması ileride
  zenginleştirilmek istenirse (ör. `mavctrl/path_dev`, IMU, GPS-spoofing groundtruth
  residual feature'ları) buradaki merge_asof mantığı hazır referans.

## Gold (`src/gold/unify.py`) — durum
7 ortak kolon (`timestamp_utc, lat, lon, altitude_m, velocity_mps, heading_deg,
vertical_rate_mps`) + 3 metadata kolonu (`source_type, source_id, label`) — `COLUMN_MAPS`
dict'inde her kaynak için bir satır (plan'ın Gold tablosuyla birebir). Eksik/henüz
yazılmamış bir kaynağın Silver'ı `read_layer` boş dönerse o kaynak sessizce atlanır
(hata değil, uyarı log'u).

Gerçek veriyle doğrulandı (`scripts/run_gold_local.py`, 2026-07-01, ALFA `processed.zip` +
gerçek `UAVAttackData.zip`, `FakeMinioClient` ile, Docker'sız): **99.885 satır** (ALFA
20.239 + UAV Attack 79.646), 10 kolon.

**BİLİNEN EKSİK — `velocity_mps` her iki kaynakta da tamamen null:**
- ALFA: plan `velocity_measured` kolonunu öneriyor ama gerçek `processed.zip`'te
  `nav_info-velocity` topic'i hiç eşleşmiyor — `parse_alfa.py`'nin çıktısında bu kolon hiç
  oluşmuyor (20.239 satırın tamamı için yok).
- UAV Attack: plan "hesapla" diyor ama `parse_uav_attack.py` şu an hiç ham hız alanı
  (`vel_n`/`vel_e`/`vel_d`) taşımıyor, hesaplanacak bir kaynak yok.

Uydurmak yerine `None` bırakıldı (`src/gold/unify.py`'deki `COLUMN_MAPS["alfa"]["velocity_mps"]`
yorumuna bkz.). Düzeltmek için Silver parser'lara yeni ham kolon eklemek gerekir — bu Gold'un
değil Silver'ın (Anıl'ın) kapsamı; bilerek şimdi yapılmadı, review'da görülsün diye kaydedildi.

## ÇÖZÜLDÜ — UAV Attack "Ping DoS" etiketi (2026-07-01)
`src/silver/parse_uav_attack.py`'deki `infer_label_from_path`'e `"ping"`/`"dos"` kontrolü
eklendi. Daha önce 79.646 satırın 29.200'ü (%37) yanlışlıkla `label="unknown"` çıkıyordu;
artık `label="ping_dos"` olarak doğru etiketleniyor. Bkz. `docs/decisions.md` ADR-003.

## Mimari sapma — bilerek yapıldı
Resmî ders planı OpenSky API + generic MAVLink diyordu. Bu proje yerine şunları kullanıyor:
- **adsb.lol** (OpenSky yerine — auth'suz, kredi limitsiz, ODbL lisans)
- **ALFA dataset** (ground-truth fault/anomaly etiketli — bireysel anomali projesi için gerekli)
- **UAV Attack dataset** (GPS spoofing benign/malicious etiketli — aynı sebep)

Gerekçe `docs/decisions.md`'de. Bunu sorgulama, kabul edilmiş bir karar.

## KRİTİK — gerçek veri yoksa dur, üretme
ALFA ve UAV Attack dosya adlandırma konvansiyonları standardize değil. Gerçek dosya
yoksa sahte/hayali bir format varsayıp kod yazma; mevcut dosyaları incele ve gerçek
adlandırmayı kodda belgele. Bu proje boyunca birden fazla kez, "muhtemelen doğru" bir
regex/pattern gerçek veriyle test edilince kırık çıktı (bkz. ADR-003 — UAV Attack'in
log_id/topic regex'i) — her zaman gerçek veriyle doğrula.

## Ortak araçlar (tekrar yazma, kullan)
`src/common/provenance.py`, `src/common/minio_io.py` (eski adı `io.py`), `src/common/fakes.py`
(`FakeMinioClient`, testler ve `scripts/`'teki yerel doğrulama script'leri için). Her
parser/loader bunları import etmeli. Provenance kolonları: `_source_type`, `_ingest_ts_utc`,
`_source_file`, `_schema_version`.

## Ekip / review akışı
Kod tek kişi tarafından Claude Code/Codex ile yazılıyor ama üç stajyer birlikte review
edecek. Bu yüzden:
- Commit mesajları açık olsun (`feat(silver): ALFA parser MinIO'ya taşındı`).
- Her faz kendi commit'i/PR'ı olsun, tek mega-commit yapma.
- README ve şema dokümantasyonu (`docs/bronze_schema.md`/`docs/silver_schema.md`, ileride
  `docs/schema.md`'ye birleşecek) her parser eklendiğinde güncellensin.
