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
kurulabilirliğini araştıran çalışma. Beş veri kümesi, 12'den fazla yöntem
ailesi, 17'den fazla deney turu. Sonuç: disiplinli **NO-GO** — sinyal
gösterilebiliyor ama dondurulmuş yanlış-alarm bütçesi altında operasyonel eşik
kurulamıyor.

Tüm kaynak kod, eğitilmiş modeller, artefaktlar ve raporlar tek klasörde:
[individual_anil/](individual_anil/).

| Belge | İçerik |
|---|---|
| [individual_anil/README.md](individual_anil/README.md) | Giriş kapısı — çalışma nedir, sonuç nedir, nereden başlanır |
| [individual_anil/KOD_HARITASI.md](individual_anil/KOD_HARITASI.md) | Modül modül kod haritası |
| [individual_anil/BULGU_KOD_ESLEMESI.md](individual_anil/BULGU_KOD_ESLEMESI.md) | Üç yapısal bulgu → hangi dosya/fonksiyonda ölçüldü |
| [individual_anil/CALISTIRMA.md](individual_anil/CALISTIRMA.md) | Demo, testler, gerçek veriyle çalıştırma, sonuçlara bakma |
| [individual_anil/demo/](individual_anil/demo/) | Veri gerektirmeyen uçtan uca demo (`demo_calistir.py`) |
| [individual_anil/raporlar/](individual_anil/raporlar/) | Tüm ADR/rapor/sunum malzemesi |

## Bireysel proje — Coğrafi Rota Analizi (Metehan)

| Klasör | İçerik |
|---|---|
| [individual/metehan_geo/](individual/metehan_geo/) | Uçuş yoğunluğu/kümeleme, gerçek-zamanlı katmanlar |
| [individual/metehan_geo_country/](individual/metehan_geo_country/) | Ülke/rota katmanları, günlük snapshot'lar |

İlgili tasarım/araştırma belgeleri `docs/` altında (`BIREYSEL_PROJE_MASTER`,
`PROMPT_COGRAFI_*`, `*_prompt.md`, `sayisal_veriler_*`, `proje_kapsamli_rapor` vb.).

## Arşiv (git geçmişinde)

Anomali-tespiti çalışmasının güncel hâli artık `main`'de, `individual_anil/`
altında. Çalışmanın tam commit geçmişi (adım adım deney kararları, ara
sonuçlar) ayrı **`arsiv`** branch'inde tutulur (`git checkout arsiv`) —
arkeoloji için, `main`'in içeriği için değil.
