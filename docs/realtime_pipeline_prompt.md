# Bireysel Proje — Realtime Veri Pipeline Mimarisi

Bireysel projemde (H3 yoğunluk haritası + DBSCAN rota kümeleme) mevcut
historical/batch analiz akışına ek olarak, realtime veri için 3 yeni
analiz modu eklemek istiyorum. Mimariyi birlikte tasarlayalım.

## Mevcut durum

- Yusuf'un realtime Kafka pipeline'ı ADS-B verisini akıtıyor,
  MinIO Bronze katmanına yazılıyor.
- Bu realtime veri, aynı zamanda **günlük periyotla** (batch job ile)
  mevcut Bronze→Silver→Gold pipeline'ından geçirilip Gold/Parquet'e
  de işleniyor — yani realtime veri sadece "sıcak" depoda kalmıyor,
  her gün bir kez kalıcı/tarihsel katmana da ekleniyor.
- Grup projesinde Bronze→Silver→Gold pipeline'ı yeni tamamlandı
  (11 tar, 2.500 Silver parçası, Gold streaming unify işlemi bitti).
- Benim projemde şu an sadece historical (Gold/Parquet) veriden
  H3-bazlı yoğunluk haritası ve DBSCAN rota kümeleme yapılıyor
  (batch/toplu analiz).

## Uygulama sırası (önce bunu onaylayalım)

1. **Önce mevcut historical akışı Gold'daki güncel veriyle uçtan uca
   test edelim** — `load_adsb_gold_data()` → H3 density → DBSCAN
   akışının tamamı çalışıyor mu, sağlam bir baseline var mı doğrulayalım.
   Bu adım bitmeden realtime katmanına geçmeyelim; aksi halde bir hata
   çıkarsa historical mantıkta mı yoksa yeni realtime katmanında mı
   olduğunu ayırt etmek zorlaşır.
2. **Baseline doğrulandıktan sonra** aşağıdaki realtime mimarisine geçelim.

## İhtiyaç: 3 yeni yoğunluk analizi modu

1. **Son 24 saat** — rolling window
2. **Son 1 hafta** — rolling window (retention: 7 günden eski veri
   otomatik düşsün)
3. **Anlık (near-live) yoğunluk haritası** — tam gerçek-zamanlı
   olması şart değil; kullanıcı talep ettiğinde en son ~1-2 dakika
   içinde gelen veriyi göstermesi yeterli (birkaç dakikalık gecikme
   kabul edilebilir, ama ne kadar taze olursa o kadar iyi).

## Historical taraftan taşınması gereken kritik öğrenimler

Historical (Gold/Parquet) tarafında çalışırken birkaç önemli bug/tasarım
kararıyla karşılaştık, bunların realtime tarafında da baştan doğru
kurulması lazım — aksi halde aynı hatalar sessizce tekrarlanabilir:

1. **Metrik tanımı netleşmeli: ham nokta sayısı ≠ benzersiz uçuş sayısı.**
   Historical'da `point_count` (ham GPS nokta sayısı) yerine `flight_count`
   (benzersiz `icao24`/`source_id` sayısı) kullanmamız gerektiğini fark
   ettik — çünkü ADS-B saniyede birkaç nokta üretiyor, yavaş/bekleyen
   uçaklar (havaalanı yakını gibi) ham nokta sayısını suni olarak
   şişiriyor. Realtime tarafında da `--live`/`--24h`/`--7d` modları için
   AYNI metrik tanımını (flight_count, unique aircraft bazlı) kullanmamız
   lazım — yoksa historical ve realtime görünümler birbirinden farklı
   ve tutarsız bir "yoğunluk" tanımı taşır.
   - Ek not: unique count'lar toplanabilir değildir (`nunique(A)+nunique(B)
     ≠ nunique(A∪B)`). Eğer realtime tarafında da çoklu-çözünürlük (H3
     res 3/4/5) göstermek istersek, her çözünürlüğü kendi verisinden
     ayrı ayrı hesaplamamız gerekir — düşük çözünürlüğü yüksek
     çözünürlükten `cell_to_parent()` ile toplayarak türetemeyiz.

2. **`ads_source_type` (MLAT vb.) alanının realtime'a taşınıp
   taşınmayacağı netleşmeli.** Historical Silver şemasında
   `ads_source_type` (adsb_icao, mlat, tisb_icao vb.) alanı var ve bunu
   ayrı bir katman olarak gösterebiliyoruz. Kafka→Influx akışında bu
   alanın taşınıp taşınmayacağına karar vermemiz lazım — eğer
   `--live`/`--24h`/`--7d` modlarında da MLAT/kaynak ayrımı istiyorsak,
   Influx şemasına bu alanı tag olarak eklemeliyiz.

## Önerilen mimari (değerlendirip onaylamamı/düzeltmemi isterim)

**İki paralel veri akışı olacak:**

- **Akış 1 — "sıcak" veri (benim yeni eklediğim kısım):**
  Realtime Kafka consumer, gelen her noktayı MinIO/Bronze'a yazmaya
  ek olarak **InfluxDB**'ye de yazsın (7 günlük retention policy ile).
  InfluxDB burada sadece zaman-aralığı bazlı ham nokta sorgusu
  (lat/lon/timestamp) için kullanılacak — H3 binning ve density
  hesaplaması (mevcut `compute_hex_density()` fonksiyonum) yine
  Python tarafında, Influx'tan çekilen veri üzerinde yapılacak, ve
  yukarıdaki `flight_count` tanımıyla tutarlı şekilde hesaplanacak.
  Bu akış `--live` / `--24h` / `--7d` modlarını besleyecek.

- **Akış 2 — "soğuk" veri (grup pipeline'ının zaten yaptığı kısım):**
  Aynı realtime veri, **günlük periyotla** (örn. her gece bir kez)
  batch job ile Bronze→Silver→Gold'dan geçip kalıcı Parquet arşivine
  ekleniyor. Bu akış `--historical` modunu besliyor ve InfluxDB'nin
  7 günlük retention'ından sonra düşen veri için de kalıcı referans
  oluyor.

Yani Influx ve Gold birbirini **tekrar etmiyor**, tamamlıyor:
Influx = son 7 günün hızlı/sıcak sorgu katmanı (dakika/saat
seviyesinde taze), Gold = kalıcı/tam tarihsel arşiv (günlük
periyotla güncellenen, retention limiti olmayan).

**Not:** Günlük periyot nedeniyle `--historical` mod en fazla ~1 gün
gecikmeli olacak (bugünün verisi henüz Gold'a işlenmemiş olabilir) —
bu gecikme `--live`/`--24h`/`--7d` modlarıyla (Influx üzerinden, çok daha
taze) tamamlanıyor, yani kullanıcı en güncel veriyi zaten başka
moddan alabiliyor.

## InfluxDB şema tasarımı (netleştirilmeli)

InfluxDB'de tag (indexlenmiş, filtrelenebilir ama kardinalitesi düşük
tutulmalı) ve field (ham değer, indexlenmez) ayrımı performans açısından
kritik. Önerim (tartışmaya açık):

- **Tags:** `source_type` (adsb_icao/mlat/tisb vb.), belki `h3_cell`
  (eğer binning'i yazarken önceden yapıyorsak) — ama `icao24`'ü tag
  yapmak muhtemelen YANLIŞ olur çünkü kardinalitesi çok yüksek olur
  (binlerce farklı uçak = binlerce ayrı tag değeri, Influx performansını
  düşürür). `icao24`'ü field olarak tutmak daha güvenli.
- **Fields:** `lat`, `lon`, `altitude_m`, `velocity_mps`, `heading_deg`,
  `vertical_rate_mps`, `icao24`/`source_id`, `nic`/`nac_p` (ileride
  jamming haritası için).

## Deduplication ve hata yönetimi

- Kafka consumer aynı noktayı hem MinIO'ya hem Influx'a yazarken biri
  başarısız olursa ne olacak? (retry mı, drop mu, dead-letter queue mu)
- Aynı noktanın (network hatası/consumer restart sonucu) tekrar
  işlenmesi ihtimaline karşı bir dedup stratejisi lazım — muhtemelen
  `(icao24, timestamp)` bazlı bir benzersizlik kontrolü (Influx bunu
  aynı tag+timestamp için otomatik "upsert" olarak ele alabilir, bunu
  doğrulamamız lazım).

## Day-count / tutarlılık kavramının realtime'a taşınması (opsiyonel, düşük öncelik)

Historical'da "kaç farklı günde görüldü" filtresi (day_count) kalıcı
koridor ile tek seferlik yoğunluğu ayırt etmemizi sağladı. `--7d`
modunda benzer bir "son 7 günün kaçında bu hex aktifti" ayrımı işe
yarayabilir mi? Zorunlu değil ama düşünmeye değer — ilk versiyonda
atlanabilir, sonradan eklenebilir.

### main.py'ye eklenecek modlar

- `--live` → son ~1-2 dk (Influx'tan, "near-live")
- `--24h` → son 24 saat (Influx'tan)
- `--7d` → son 7 gün (Influx'tan, retention sınırı burada)
- `--historical` → mevcut Gold/Parquet akışı (değişmiyor, günlük
  periyotla realtime veriyle de beslenmeye devam ediyor)

## Sorular / birlikte karar vermek istediklerim

1. InfluxDB entegrasyonu için ayrı bir modül mü açalım
   (örn. `influx_client.py`, mevcut `minio_client.py`'ye benzer şekilde),
   yoksa mevcut Kafka consumer'a mı ekleyelim?
2. Retention policy'yi InfluxDB tarafında native mi tanımlayalım,
   yoksa uygulama seviyesinde mi yönetelim?
3. "Near-live" mod için Influx sorgusunu her istek geldiğinde mi
   çalıştıralım, yoksa kısa süreli (örn. birkaç saniye-dakika TTL'li)
   bir cache mi ekleyelim? (Redis opsiyonel, zorunlu değil.)
4. Günlük Bronze→Silver→Gold batch job'ı nasıl tetiklenecek — cron
   mu, manuel script çalıştırma mı, yoksa başka bir zamanlayıcı mı?
5. Bu üç yeni modun mevcut FAZ planımdaki hangi adıma denk geldiğini
   ve zaman tahminini nasıl etkileyeceğini birlikte değerlendirelim.
6. `flight_count` metriğini realtime tarafında (Influx sorgusu üzerinden)
   hesaplarken, historical'daki gibi tek seferlik toplu tarama yerine
   sürekli akan veriyle nasıl doğru şekilde (mükerrer saymadan)
   hesaplarız? Bu, streaming context'te distinct-count'un nasıl
   yönetileceği sorusu — belki her sorguda Influx'tan ham noktaları
   çekip Python'da `nunique()` yapmak yeterlidir (7 günlük veri hacmi
   küçük olduğu için), ama teyit edelim.
7. InfluxDB tag/field ayrımını (yukarıdaki taslak) onaylıyor musun,
   yoksa `icao24` gibi alanları farklı ele almamız mı gerekiyor?

---

Lütfen önce 1. adımı (historical baseline doğrulama) birlikte yapalım,
sonra bu tasarımı gözden geçir, varsa eksik/riskli noktaları söyle,
sonra adım adım realtime implementasyona geçelim.
