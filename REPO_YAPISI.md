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
| [adsb/](adsb/) | ADS-B contextual-physics residual modeli + kural/CUSUM karar katmanı (dokunulmadı, kök dizinde) |
| [gecmis_calismalar/residual_v1/](gecmis_calismalar/residual_v1/) | Komut→tepki residual tabanlı İHA FDI hattı (ALFA + RflyMAD) — 2026-07-22'de buraya taşındı |
| [gecmis_calismalar/uav_gnss/](gecmis_calismalar/uav_gnss/) | UAV GNSS bütünlük pilotu (PX4/RflyMAD) — 2026-07-22'de buraya taşındı |
| [gecmis_calismalar/anomaly_core/](gecmis_calismalar/anomaly_core/) | Paylaşılan CUSUM/forecaster/kalibrasyon çekirdeği — 2026-07-22'de buraya taşındı |
| [gecmis_calismalar/rfly_full/](gecmis_calismalar/rfly_full/), [gecmis_calismalar/rfly_dl/](gecmis_calismalar/rfly_dl/) | RflyMAD-Full v2 hattı (bugünkü iş dahil) + erken direct-DL pilotu — 2026-07-22'de buraya taşındı |
| [scripts/](scripts/) | `adsb_*` kökte; `residual_v1_*` → `scripts/_ortak_residual_v1_ALFA_RFLYMAD/`, `rfly_full/rfly_dl` → `scripts/RFLYMAD_rfly_full_v2/`, `uav_gnss` → `scripts/RFLYMAD_uav_gnss/` (2026-07-22 alt-klasörleme) |
| [tests/](tests/) | `test_adsb_*` kökte; `test_residual_v1_*` → `tests/_ortak_residual_v1_ALFA_RFLYMAD/`, `test_rfly_full_*`/`test_rfly_dl` → `tests/RFLYMAD_rfly_full_v2/`, `test_uav_gnss*` → `tests/RFLYMAD_uav_gnss/`, `test_anomaly_core_*` → `tests/_ortak_anomaly_core/` (`pytest.ini`'deki `testpaths = tests` değişmedi, hepsi otomatik taranır) |
| [artifacts/](artifacts/) | Kayıtlı run çıktıları/manifestler (büyük veri gitignore'da, fiziksel taşınmadı) |
| [gecmis_calismalar/](gecmis_calismalar/) | **2026-07-22 dosyalama:** ALFA/UAV-Attack/UAV-SEAD/RFLYMAD'ın (canlı paketler dahil) dataset × dosya-türü bazında yeniden klasörlenmiş hâli — bkz. `gecmis_calismalar/README.md` |
| **Raporlar (`docs/`):** | |
| [gecmis_calismalar/_ortak/raporlar/RESIDUAL_V1.md](gecmis_calismalar/_ortak/raporlar/RESIDUAL_V1.md) | RESIDUAL-V1 tasarım + sonuç + NO-GO (birleşik, ALFA+RFLYMAD) |
| [docs/ADSB_CONTEXTUAL_DENEY.md](docs/ADSB_CONTEXTUAL_DENEY.md) | ADS-B contextual deney kayıtları (birleşik) |
| [docs/PROJE_SUREC_VE_SONUC.md](docs/PROJE_SUREC_VE_SONUC.md) | Durum/teşhis, skor defteri, ön-kayıt, sunum notları (birleşik) |
| [docs/final_rapor_ml_fizibilite_2026-07-16.md](docs/final_rapor_ml_fizibilite_2026-07-16.md) | Fizibilite final raporu |
| [docs/decisions.md](docs/decisions.md) | ADR karar günlüğü |
| [docs/ADSB_BASIT_ANOMALI_PLAN_20260722.md](docs/ADSB_BASIT_ANOMALI_PLAN_20260722.md) | Yeni, indirgenmiş ADS-B işi: irtifa + GPS/rota sapması planı |
| [docs/raporlar_html_tex/](docs/raporlar_html_tex/) | Biçimsel .tex/.pdf raporlar + atlas/yönetici html'leri (2026-07-22 toparlandı) |
| [docs/panolar/](docs/panolar/) | İnteraktif deney/yük dashboard html'leri |
| [docs/haftalik_takip/](docs/haftalik_takip/) | Stajyer haftalık takip raporları (html + .docx, kronolojik seri) |
| [docs/sunumlar/](docs/sunumlar/) | Takım sunumu (.pptx) |

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
