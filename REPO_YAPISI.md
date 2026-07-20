# Repo Yapısı — Yol Haritası

Bu repo üç ayrı çalışmayı ve onların paylaştığı ortak veri altyapısını barındırır.
Aşağıdaki harita, hangi klasörün kime/neye ait olduğunu özetler.

## Ortak altyapı (takım) — gerçek-zamanlı ADS-B pipeline

| Klasör | İçerik |
|---|---|
| [src/](src/) | Ingestion → Silver → Gold veri hattı (adsb.lol tarihsel + gerçek-zamanlı), MinIO/local depolama |
| [Dashboard/](Dashboard/) | Gerçek-zamanlı Dash tabanlı uçuş dashboard'u (Docker servisi) |
| [team_dashboard/](team_dashboard/) | Takım paneli (statik harita + ülke katmanları) |
| [configs/](configs/) | Çalışma zamanı konfigürasyonları |
| `docker-compose.yml`, `Makefile`, `.env.example` | Kafka/Redis/InfluxDB/MinIO + Dashboard servislerini ayağa kaldırır |

Çalıştırma için ana [README.md](README.md)'ye bakın.

## Bireysel proje — ML Anomali-Tespiti Fizibilitesi (Anıl)

Gerçek ADS-B ve İHA telemetrisinde operasyonel bir anomali dedektörünün
kurulabilirliğini araştıran çalışma. Sonuç: disiplinli **NO-GO** — sinyal
gösterilebiliyor ama dondurulmuş yanlış-alarm bütçesi altında operasyonel eşik
kurulamıyor.

| Klasör / dosya | İçerik |
|---|---|
| [adsb/](adsb/) | ADS-B contextual-physics residual modeli + kural/CUSUM karar katmanı |
| [residual_v1/](residual_v1/) | Komut→tepki residual tabanlı İHA FDI hattı (ALFA + RflyMAD) |
| [uav_gnss/](uav_gnss/) | UAV GNSS bütünlük pilotu (PX4/RflyMAD) |
| [anomaly_core/](anomaly_core/) | Paylaşılan CUSUM/forecaster/kalibrasyon çekirdeği |
| [scripts/](scripts/) | `adsb_*`, `residual_v1_*`, `*_uav_gnss_*` deney/rapor sürücüleri |
| [tests/](tests/) | `test_adsb_*`, `test_residual_v1_*`, `test_uav_gnss_*` |
| [artifacts/](artifacts/) | Kayıtlı run çıktıları/manifestler (büyük veri gitignore'da) |
| **Raporlar (`docs/`):** | |
| [docs/RESIDUAL_V1.md](docs/RESIDUAL_V1.md) | RESIDUAL-V1 tasarım + sonuç + NO-GO (birleşik) |
| [docs/ADSB_CONTEXTUAL_DENEY.md](docs/ADSB_CONTEXTUAL_DENEY.md) | ADS-B contextual deney kayıtları (birleşik) |
| [docs/PROJE_SUREC_VE_SONUC.md](docs/PROJE_SUREC_VE_SONUC.md) | Durum/teşhis, skor defteri, ön-kayıt, sunum notları (birleşik) |
| [docs/final_rapor_ml_fizibilite_2026-07-16.md](docs/final_rapor_ml_fizibilite_2026-07-16.md) | Fizibilite final raporu |
| [docs/decisions.md](docs/decisions.md) | ADR karar günlüğü |

## Bireysel proje — Coğrafi Rota Analizi (Metehan)

| Klasör | İçerik |
|---|---|
| [individual/metehan_geo/](individual/metehan_geo/) | Uçuş yoğunluğu/kümeleme, gerçek-zamanlı katmanlar |
| [individual/metehan_geo_country/](individual/metehan_geo_country/) | Ülke/rota katmanları, günlük snapshot'lar |

İlgili tasarım/araştırma belgeleri `docs/` altında (`BIREYSEL_PROJE_MASTER`,
`PROMPT_COGRAFI_*`, `*_prompt.md`, `sayisal_veriler_*`, `proje_kapsamli_rapor` vb.).

## Arşiv (git geçmişinde)

Eski non-ADS-B ML hattı (ALFA/UAV-SEAD/RFLY, ML-0…ML-16) ve süreç/AI-inceleme
notları `main`'de değil, ayrı **`arsiv`** branch'inde tutulur
(`git checkout arsiv`). Bu, `main`'i teslime hazır ve hafif tutmak içindir; içerik
kaybolmadı, iki makineden `git fetch` ile erişilebilir.
