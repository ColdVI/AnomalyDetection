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
gerçek bir MinIO sunucusu gerektirmiyor — `tests/conftest.py` içindeki `FakeMinioClient`
in-memory bir sahte uyguluyor.
