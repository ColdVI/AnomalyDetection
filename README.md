# ADS-B Anomaly Detection

Bu repo, ortak gerçek-zamanlı ADS-B altyapısını (bu README) ve üzerine kurulu üç
ayrı çalışmayı bir arada barındırıyor. Klasörlerin kime/neye ait olduğu için
**[REPO_YAPISI.md](REPO_YAPISI.md)**'ye bakın — bu dosya güncel haritadır.

Repo 2026-07-10'da bir kere sadeleştirildi: o tarihten önceki ALFA, UAV Attack,
UAV-SEAD, RFLY ve ML-0…ML-16 denemeleri (iki reddedilen ADS-B model denemesi
dahil) `main`'den çıkarılıp arşivlendi. Bu arşiv artık `main`'de değil, ayrı bir
**`arsiv`** branch'inde tutuluyor (`git checkout arsiv` ile erişilebilir) — aktif
baseline değildir, yeni modele import edilmez veya başarı kanıtı olarak
kullanılmaz.

Ortak altyapının aktif hedefi: gerçek ADS-B verisinden gerçek-zamanlı bir
dashboard beslemek (bkz. `src/`, `Dashboard/`). Bireysel projelerin kendi
hedefleri için `REPO_YAPISI.md`'deki ilgili bölümlere bakın.

Başlangıç noktaları:

- `REPO_YAPISI.md`: hangi klasör kime ait, hangi rapor nerede;
- `docs/adsblo_data_format_reference (1) 2026-07-10 amt 11.03.27.md`: veri formatı;
- `src/silver/parse_adsblol_historical.py`: korunmuş ham veri okuma altyapısı.

## Kurulum ve çalıştırma

Gerekli: Docker Desktop (Compose ile) ve Python 3.13+ (yerelde script/test
çalıştırmak isteyenler için — kök `requirements.txt`'teki `numpy==2.5.0` pin'i
3.12 altında kurulamıyor, `Dashboard/Dockerfile` de bu yüzden `python:3.13-slim`
kullanıyor).

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
