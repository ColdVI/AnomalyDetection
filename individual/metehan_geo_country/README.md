# ICAO Hex → Ülke Çıkarımı + Ülke Bazlı Harita

Ana projeye (`individual/metehan_geo/`) **ek**, bağımsız bir modül —
country_heatmap_prompt.md'deki hocanın isteği üzerine. Ana proje dosyalarına
hiçbir değişiklik yapılmadı; sadece `individual/metehan_geo/geo.py`'den
(`assign_h3_cell`, `h3_cell_to_polygon`) salt-okunur import var. Beğenilmezse
bu klasörü silmek yeterli, ana projeyi etkilemez.

## Veri

- `data/aircraft_dump_20260707_181141.csv`: hocanın verdiği CSV (Drive'dan
  indirildi, kolonlar: `id, hex, seen_at, latitude, longitude, altitude,
  velocity, heading, callsign, source`). 1.304.711 satır, 11.222 benzersiz hex.
  **Git'e dahil DEĞİL** (~124MB, GitHub'ın 100MB limitini aşıyor + zaten "veri",
  kod değil) — Drive'dan ayrıca indirilip bu klasöre (`data/`) konmalı.
- `data/ICAOHexRange.csv`: `rikgale/ICAOList` reposundan indirilen ICAO
  Annex 10 hex-blok → ülke tahsis tablosu (200 aralık, iç içe/nested bloklar
  dahil). Küçük/sabit referans tablo olduğu için git'te kalıyor.
- `data/ne_50m_admin_0_countries.geojson`: Natural Earth 50m ülke sınırları
  (110m değil — küçük ülkeler, ör. Bahreyn/Malta/Singapur, 110m'de yok).
  Git'e dahil değil (`.gitignore`'daki `**/data/*.geojson` kuralı) — gerekirse
  tekrar indirilebilir, bkz. "Çalıştırma".
- `data/hex_country_lookup.csv`, `data/country_counts.csv`: TÜRETİLMİŞ
  (bizim script'lerimizin ürettiği) küçük dosyalar, git'te kalıyor —
  `step1_coverage.py`/`step2_build_country_layers.py` çalıştırılarak yeniden
  üretilebilir.
- `viz/data/h3_*.geojson`, `viz/data/country_basemap.geojson`: TÜRETİLMİŞ
  büyük görselleştirme çıktıları (6-62MB), git'e dahil değil
  (`**/data/*.geojson`) — `step2_build_country_layers.py` ile yeniden üretilir.

## Adım 1 sonucu: coverage

- Benzersiz hex bazında **%99.9** (11.213/11.222) bir ülkeye eşleşiyor.
- 9 hex ICAO'nun kendi rezerve/tahsis-etmediği bir bloğa düşüyor (veri hatası
  değil, tablonun kendi boşluğu).
- **Kritik sınırlama**: bu, uçağın **tescilli olduğu** ülkeyi verir, o an
  **hangi ülke üzerinde uçtuğunu değil**. `latitude`/`longitude` zaten ayrı
  bir bilgi, tescil ülkesiyle karıştırılmamalı.

## Adım 2: görselleştirme (2026-07-08, birkaç revizyondan sonraki hali)

Sadece **H3 + baskın ülke** modu var (choropleth/ülke-sınırı renklendirmesi
hocanın isteğiyle kaldırıldı) — H3 res5 çözünürlüğünde, her hex'in rengi/
opaklığı toplam benzersiz uçak sayısına göre (log-scale, sarı→kırmızı ramp).

- **Zaman aralığı**: 1 Gün / 1 Hafta / 30 Gün / Tümü — verinin KENDİ son
  kaydına (2026-07-07) göre hesaplanır (gerçek "şimdi"ye göre değil, çünkü
  veri 62 gün önce durdu).
- **Ülke seçimi**: bir ülke seçilince SADECE baskın olduğu değil, o ülkeye
  tescilli uçakların **geçtiği tüm hex'ler** gösterilir (dropdown, geçerli
  zaman penceresindeki tüm ülkelerden otomatik dolduruluyor).
- **Hover popup**: tek "baskın ülke" değil, o hex'teki **en baskın 3 ülke**
  (sayılarıyla) gösterilir.
- **3D Küre** ve **Fiziki (hillshade)** toggle'ları ana projedekiyle aynı.

Bilinen boşluk: **Yugoslavya** (1991'de dağıldı) ICAOList'te hâlâ bir blok
olarak duruyor ama bugün karşılık gelen tek bir ülke polygonu yok — sadece
1 hex etkileniyor, `country_counts.csv`'de görünür ama harita katmanlarına
dahil edilmiyor.

## Çalıştırma

Önce (git'te olmayan iki dosya):
1. `aircraft_dump_*.csv` (veya `.csv.gz`) dosyasını Drive'dan indirip
   `data/`'ya koy (gz ise çıkart, script'ler düz `.csv` bekliyor).
2. `ne_50m_admin_0_countries.geojson`'ı indir (Natural Earth, public domain):
   `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson`
   → `data/`'ya kaydet.

Sonra:
```
python -m individual.metehan_geo_country.step1_coverage
python -m individual.metehan_geo_country.step2_build_country_layers
cd individual/metehan_geo_country/viz && python -m http.server 8091
```
