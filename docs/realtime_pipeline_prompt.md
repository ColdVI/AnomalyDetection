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

## Önerilen mimari (değerlendirip onaylamamı/düzeltmemi isterim)

**İki paralel veri akışı olacak:**

- **Akış 1 — "sıcak" veri (benim yeni eklediğim kısım):**
  Realtime Kafka consumer, gelen her noktayı MinIO/Bronze'a yazmaya
  ek olarak **InfluxDB**'ye de yazsın (7 günlük retention policy ile).
  InfluxDB burada sadece zaman-aralığı bazlı ham nokta sorgusu
  (lat/lon/timestamp) için kullanılacak — H3 binning ve density
  hesaplaması (mevcut `compute_hex_density()` fonksiyonum) yine
  Python tarafında, Influx'tan çekilen veri üzerinde yapılacak.
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

---

Lütfen önce 1. adımı (historical baseline doğrulama) birlikte yapalım,
sonra bu tasarımı gözden geçir, varsa eksik/riskli noktaları söyle,
sonra adım adım realtime implementasyona geçelim.
