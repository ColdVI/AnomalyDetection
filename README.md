# ADS-B Anomaly Detection — Clean Restart

Bu repo 2026-07-10 tarihinde sadeleştirildi. Önceki ALFA, UAV Attack, UAV-SEAD,
RFLY ve ML-0…ML-16 deneyleri aktif çalışma alanından çıkarıldı. Aynı gün yazılan
iki ADS-B model denemesi de kullanıcı tarafından baseline olarak kabul edilmedi ve
ayrı bir arşive kaldırıldı.

Aktif hedef: gerçek ADS-B arşivlerinden, tanımı baştan açık kurulmuş bir
`anomaly = yes/no` sistemi geliştirmek.

Başlangıç noktaları:

- `adsb/README.md`: sıfırdan başlangıç sözleşmesi;
- `docs/adsblo_data_format_reference (1) 2026-07-10 amt 11.03.27.md`: veri formatı;
- `src/silver/parse_adsblol_historical.py`: korunmuş ham veri okuma altyapısı;
- `archive/README.md`: önceki çalışmaların indeksi.

`archive/` altındaki kod ve sonuçlar aktif baseline değildir; yeni modele import
edilmez veya başarı kanıtı olarak kullanılmaz.

## Kurulum ve çalıştırma

Gerekli: Docker Desktop (Compose ile) ve Python 3.11+ (yerelde script/test
çalıştırmak isteyenler için).

1. Repoyu klonla, `.env.example` dosyasını `.env` olarak kopyala. Varsayılan
   değerler her makinede çalışır; `OPENSKY_CLIENT_ID`/`SECRET` gibi alanlar
   opsiyoneldir, boş bırakılırsa ilgili servis düşük öncelikli bir moda
   (anonim erişim vb.) düşer. `.env` git'e gitmez, kimse birbirinin yerel
   ayarını ezmez.
2. Tüm sistemi (Kafka, Redis, InfluxDB, MinIO ve dört Dashboard servisi)
   ayağa kaldır:

   ```
   docker compose --profile streaming up -d --build
   ```

   (Eşdeğeri: `make up-streaming`, ama o `--build` yapmıyor; ilk çalıştırmada
   veya `requirements.txt`/Dockerfile değiştiğinde yukarıdaki komutu kullan.)
3. Kontrol et:
   - Dashboard: http://localhost:8050
   - API sağlık kontrolü: http://localhost:8000/api/health
   - MinIO konsolu: http://localhost:9001 (varsayılan kullanıcı/şifre:
     `minioadmin`/`minioadmin`)
4. Kapatmak için: `docker compose --profile streaming down` (eşdeğeri:
   `make down`).

Sadece MinIO'yu (Docker/Kafka'sız, tek başına pipeline geliştirmek için)
ayağa kaldırmak yeterliyse: `make up-storage`. Docker hiç kurulmadan da
pipeline'ı denemek istersen `.env`'de `STORAGE_BACKEND=local` yap — Bronze/
Silver/Gold, MinIO yerine yerel diske (`data/objectstore/`) yazılır.

Yerel Python ortamı (test çalıştırmak, pipeline scriptlerini Docker'sız
denemek için): `pip install -r requirements.txt`, sonra `pytest -q`
(eşdeğeri: `make test`).
