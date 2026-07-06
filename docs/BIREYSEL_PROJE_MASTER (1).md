# Bireysel Proje — Coğrafi Rota Kümeleme ve Rota Sapması Tespiti

> **Bu dosya, Claude Code'un bu bireysel projeyi baştan sona uygulaması için
> tek ve eksiksiz kaynaktır.** Proje bağlamını, teknik kararları ve
> implementasyon adımlarını (fonksiyon fonksiyon) içerir. `individual/metehan_geo/`
> altında çalış, ortak pipeline'a (`src/`) dokunma — sadece oku/import et.

---

## 1. Bağlam ve amaç

Bu proje, staj programının orijinal planındaki "Stajyer 2 — Coğrafi Uçuş
Verisi Analizi ve Rota Kümeleme" (Hafta 5-8) tanımına dayanıyor. Orijinal
planda veri kaynağı OpenSky Network API olarak belirtilmişti; bu proje
**onun yerine takımın kendi ürettiği Gold katmanını** (adsb.lol historical +
realtime, MinIO'da) kullanıyor — amaç ve teslimler aynı, veri kaynağı
değişti.

**Amaç:** adsb.lol Gold verisinden (Türkiye ve/veya global — coğrafi filtre
analiz aşamasında senin kararın) yaygın uçuş koridorlarını çıkarmak, hava
sahası yoğunluk haritaları üretmek, benzer rotaları kümeleyerek tipik uçuş
profillerini tanımlamak, ve **ek olarak** belirli bir rota çiftinin (örn.
iki havalimanı arası) normal rotasından ne kadar saptığını tespit eden bir
mekanizma kurmak.

**Orijinal plandaki teorik araştırma alanları** (rapor/literatür özeti için,
Claude Code'un değil, kullanıcının okuyacağı kısım — kod yazarken bu
başlıklara referans ver ama okumayı kullanıcı yapacak):
- Coğrafi Veri Madenciliği (Spatial Data Mining)
- Yörünge Madenciliği (Trajectory Mining)
- Heksagonal Grid Sistemleri (H3 vs S2)
- Hava Sahası Yoğunluk Modelleme (KDE)
- Coğrafi Kümeleme Algoritmaları (DBSCAN, ST-DBSCAN, OPTICS)

---

## 2. Veri kaynağı ve bağlantı

- **Konum:** MinIO `gold` bucket, `unified/*.parquet` objeleri.
- **MinIO şu an native Windows process olarak çalışıyor** (Docker değil —
  altyapı sorunları nedeniyle geçici olarak `minio.exe` ile ayakta).
  Bağlantı bilgisi değişmedi: `.env`'de `MINIO_ENDPOINT=localhost:9000`,
  `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY` aynı. Kod tarafında hiçbir fark
  yok — `src/common/minio_io.py`'deki `get_minio_client()` aynı şekilde
  çalışır.
- **Filtre:** `source_type` kolonu `adsblol_historical` / `adsblol_hist` /
  `adsblol_realtime` / `adsblol_rt` olan satırlar (isimlendirme geçiş
  sürecinde iki varyant da olabilir, ikisini de kapsa).
- **Kolonlar (Gold 7+3 şema):** `timestamp_utc`, `lat`, `lon`,
  `altitude_m`, `velocity_mps`, `heading_deg`, `vertical_rate_mps`,
  `source_type`, `source_id` (ICAO hex), `label` (adsb'de her zaman null).
- **Coğrafi filtre pipeline'da YOK** (bilinçli mimari karar, ADR-003) —
  Türkiye ya da başka bir bölge filtresi **bu projenin kendi kodunda**
  uygulanır, `src/`'e asla eklenmez.
- **Erişim:** Salt okunur. `from src.common.minio_io import get_minio_client, read_layer`
  ile import et, pipeline kodunu değiştirme.

---

## 3. Teknoloji yığını

Python, pandas, GeoPandas, H3 (`h3-py`), Folium, scikit-learn (DBSCAN,
KMeans), requests (adsbdb.com API için).

---

## 4. Klasör yapısı

```
individual/metehan_geo/
├── __init__.py
├── data.py             # Bolum 5, Adim 1-2
├── geo.py              # Bolum 5, Adim 3-6
├── viz.py              # Bolum 5, Adim 7-8
├── clustering.py        # Bolum 5, Adim 9-13
├── routes.py             # Bolum 6 (rota sapmasi tespiti)
├── build_baseline.py     # Bolum 6, tek seferlik baseline kurulumu
├── update_baseline.py    # Bolum 6, haftalik guncelleme
├── report.py              # ozet istatistikler
├── main.py                # Bolum 5'i uctan uca calistiran CLI
├── NOTLAR.md               # her fazdan sonra kisa ozet buraya birikir
└── tests/
    ├── test_data.py
    ├── test_geo.py
    ├── test_clustering.py
    └── test_routes.py
```

---

## 5. FAZ A — Yoğunluk haritası + rota kümeleme (ana teslim)

Aşağıdaki 14 fonksiyonu **bu sırayla** yaz — her biri bir öncekinin çıktısını
kullanıyor, atlama. Her fonksiyon bitince kullanıcıya kısa bir özet ver (ne
yazıldı, hangi parametre/karar neden öyle seçildi) — kullanıcı paralelde
konuyu okuyup takip ediyor, bu özetler onun `NOTLAR.md`'sine gidecek.

### 1. `load_adsb_gold_data(client=None) -> pd.DataFrame`
MinIO Gold'dan `adsblol_hist`/`adsblol_historical`/`adsblol_rt`/`adsblol_realtime`
satırlarını çeker, birleştirir. `src/common/minio_io.py`'deki
`get_minio_client()`/`read_layer()`'ı çağır, yeniden yazma.

### 2. `clean_coordinates(df) -> pd.DataFrame`
Null `lat`/`lon` at, geçersiz aralık dışını (`lat` -90/90, `lon` -180/180)
filtrele. Kaç satır silindiğini logla.

### 3. `filter_bbox(df, min_lat, max_lat, min_lon, max_lon) -> pd.DataFrame`
İsteğe bağlı bölgesel filtre (senin analiz kararın, pipeline'da yok).

### 4. `assign_h3_cell(df, resolution) -> pd.DataFrame`
`h3.latlng_to_cell(lat, lon, resolution)` ile her satıra `h3_cell` kolonu
ekle. `resolution`'ı CLI argümanı yap, sabit yazma.

### 5. `h3_cell_to_polygon(hex_id) -> list[tuple[float, float]]`
`h3.cell_to_boundary(hex_id)` ile hex sınır koordinatlarını Folium'un
beklediği formata çevir.

### 6. `compute_hex_density(df) -> pd.DataFrame`
`.groupby("h3_cell").size().reset_index(name="point_count")`.

### 7. `build_density_map(density_df, center_lat, center_lon) -> folium.Map`
Her hex'i yoğunluğa göre renklendirilmiş poligon olarak çiz. Alternatif/daha
kolay başlangıç: `folium.plugins.HeatMap`.

### 8. `save_map(map_obj, path) -> None`
`folium.Map.save(path)`.

### 9. `prepare_clustering_input(df) -> np.ndarray`
`lat`/`lon`'u array'e çevir. Haversine mesafe kullanacaksan
`np.radians(df[["lat","lon"]])` ile radyana çevir.

### 10. `run_dbscan_clustering(X, eps, min_samples) -> np.ndarray`
`sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric="haversine")`.
**Dikkat:** haversine metric kullanılıyorsa `eps` km değil radyan olmalı
(`eps_km / 6371`).

### 11. `attach_cluster_labels(df, labels) -> pd.DataFrame`
`df["cluster_id"] = labels` — index sırası bozulmamış olmalı.

### 12. `build_cluster_map(df, center_lat, center_lon) -> folium.Map`
Her `cluster_id`'yi farklı renkte göster, `-1` (gürültü) ayrı renk.

### 13. `summarize_clusters(df) -> pd.DataFrame`
`.groupby("cluster_id").agg(...)` — nokta sayısı, ortalama irtifa, benzersiz
uçak sayısı.

### 14. `main()`
1-13'ü sırayla çağır, ara adımlarda satır/hex/küme sayısını logla. CLI
argümanları: `--h3-resolution`, `--eps-km`, `--min-samples`, bbox sınırları.

---

## 6. FAZ B — Rota sapması tespiti (ek özellik)

`routes.py`:

```python
def resolve_route(callsign: str) -> dict:
    """adsbdb.com GET /v0/callsign/{callsign} -- local cache'li (kucuk
    sqlite ya da dict), tekrar tekrar ayni callsign'i sorgulama."""

def segment_flight(df_one_aircraft: pd.DataFrame) -> list[pd.DataFrame]:
    """on_ground False->True / True->False gecisleriyle ucusu segmentlere
    ayirir. on_ground yoksa (adsb.lol raw'da alt_baro=='ground' -> on_ground)
    turetilmis olmasi lazim, Silver semasinda var mi kontrol et."""

def sample_every_n_minutes(flight_segment: pd.DataFrame, minutes: int = 15) -> pd.DataFrame: ...

def build_baseline(flights: list[pd.DataFrame], h3_resolution: int) -> pd.DataFrame:
    """Cruise-fazi (terminal fazi haric) noktalarindan hex frekans tablosu."""

def score_flight_against_baseline(flight_hexes: set[str], baseline_hexes: set[str]) -> float:
    """Cruise noktalarinin yuzde kaci baseline'da degil -- sapma orani."""

def calibrate_threshold(baseline_scores: list[float], n_std: float = 2.0) -> float:
    """Esik SABIT SAYI DEGIL -- baseline'in kendi skor dagilimindan
    (ortalama + n_std*std) turetilir."""

def is_anomalous(score: float, threshold: float) -> bool: ...
```

**`build_baseline.py`** (tek seferlik): 4 mevsim penceresinden, 7 gün
arayla (ardışık değil — pseudo-replication'dan kaçınmak için) alınmış
historical tar'ları parse edip `build_baseline()`'a besler.

**`update_baseline.py`** (haftalık): Realtime Gold'dan o haftanın
uçuşlarını çeker, önce mevcut baseline'a göre skorlar, **sadece normal
çıkanları** baseline'a ekler (self-contamination'ı önleme — anormal
uçuşlar baseline'a asla karışmasın).

---

## 7. Test

`tests/conftest.py`'deki `FakeMinioClient`'ı (ortak pipeline'da zaten var)
import et, gerçek MinIO gerektirmeyen testler yaz. `resolve_route`
(adsbdb'ye gerçek HTTP isteği atan) `unittest.mock` ile mock'lanmalı,
testte gerçek ağ isteği atılmamalı.

---

## 8. Haftalık teslim eşleşmesi (orijinal plana göre)

| Hafta | Bölüm |
|---|---|
| 5 | Bölüm 5, Adım 1-5 (veri çekme, H3 grid) |
| 6 | Bölüm 5, Adım 6-8 (yoğunluk haritası) |
| 7 | Bölüm 5, Adım 9-13 (rota kümeleme) + Bölüm 6 (rota sapması, süre kalırsa) |
| 8 | Bölüm 5 Adım 14 (main) + rapor + sunum |

---

## 9. Genel kurallar

- `src/` klasörüne dokunma, sadece import et.
- Coğrafi filtre `src/`'e asla eklenmez — bu projenin kendi kodunda kalır.
- Gerçek veri (Gold'dan gelen) olmadan kolon/format varsayımı yapma — önce
  `load_adsb_gold_data()`'yı çalıştırıp gerçek çıktıyı göster.
- H3 resolution, DBSCAN eps/min_samples, rota sapması eşiği — hiçbiri sabit
  yazılmaz, ya CLI argümanı ya da veriden empirik çıkarılır.
- Her fonksiyon bitince kısa özet ver, `individual/metehan_geo/NOTLAR.md`'ye
  ekle (kullanıcı raporunda kullanacak).
- Bir fonksiyon çalışır hale gelmeden bir sonrakine geçme.
