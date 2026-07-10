# Araştırma Bulguları — Dış Kaynaklar (Rapor için)

> Bu dosya, `koala73/worldmonitor` reposu ve marksblogg tech blog'u üzerinde
> yapılan incelemenin doğrulanmış halidir. Rapor/literatür bölümüne
> doğrudan aktarılabilir. Metehan'ın bireysel projesi (`docs/BIREYSEL_PROJE_MASTER (1).md`)
> için mimari/yöntem referansı olarak kullanılır.

---

## 1. Dual Map Engine Mimarisi — `koala73/worldmonitor`

**Kaynak:** https://github.com/koala73/worldmonitor (AGPL-3.0 — bkz. Bölüm 4, lisans notu)

`src/components/MapContainer.ts` içinde, üç render motorunu (`GlobeMap.ts`
[3D, globe.gl], `DeckGLMap.ts` [2D, deck.gl], `Map.ts` [eski SVG/D3]) tek
bir facade sınıf yönetiyor.

**Kademeli fallback zinciri:**
```
3D Globe (globe.gl, WebGL1 yeterli)
    → başarısız olursa → 2D DeckGL (WebGL2 gerekli)
        → başarısız olursa → SVG (her zaman çalışır, son çare)
```
Yazılım tabanlı rasterize'lar (`swiftshader`, `llvmpipe`) WebGL2'yi
reddedebiliyor — bu durumda otomatik bir alt seviyeye düşülüyor
(`handleGlobeInitFailure`, `createDeckGLMap`'in catch bloğu).

**Lazy + demand-gated yükleme:** Ağır render motorları (deck.gl/MapLibre)
`await import(...)` ile sadece gerektiğinde çekiliyor. Yükleme, kullanıcı
etkileşimine (tıklama/wheel/scroll) veya harita görünür + tarayıcı idle
olana kadar erteleniyor (`IntersectionObserver` + `requestIdleCallback`,
3.5sn gecikme, 12sn üst sınır) — ilk sayfa boyaması (LCP) hafif bir
"shell" ile hızlı kalıyor.

**Token tabanlı iptal:** Kullanıcı hızlıca render modu değiştirirse
(`globeInitToken`/`rendererInitToken`), yarım kalan eski init işlemleri
sessizce iptal ediliyor — yarış koşulu (race condition) oluşmuyor.

**Rehydration:** `MapContainer`, her veri türünü kendi üzerinde
cache'liyor (`cachedAircraftPositions` bizim projemiz için doğrudan
örnek). Render motoru değişince (`switchToGlobe`/`switchToFlat`),
`rehydrateActiveMap()` tüm veriyi yeni motora tekrar besliyor — veri
kaynağına (API/DB) tekrar gitmeden sorunsuz geçiş sağlıyor.

**Uçak pozisyon katmanı (`DeckGLMap.ts`, `createAircraftPositionsLayer`)
— doğrudan uygulanabilir desen:**
- `IconLayer` ile uçak ikonu
- `getAngle: -trackDeg` — heading'e göre ikon döndürme
- İrtifaya göre renk gradyanı
- Yerdeyken (`on_ground`) küçük/gri ikon, havadayken irtifa-renkli ikon

**Not (2026-07):** Metehan'ın kendi projesinde bu üç-motorlu (globe.gl/deck.gl/SVG)
dual-engine mimarisinin tamamı yerine, tek kütüphanede hem 2D hem 3D
(globe projeksiyonu) veren **MapLibre GL JS** kullanılmasına karar
verildi — gerekçe ve detay `docs/BIREYSEL_PROJE_MASTER (1).md` Bölüm 5.1'de.
Yukarıdaki fallback/lazy-load/rehydration desenleri yine de referans
değerinde (özellikle stil değişiminde custom layer'ları yeniden ekleme
ihtiyacı, worldmonitor'un rehydration deseniyle aynı problem sınıfı).

---

## 2. Redis 3-Katmanlı Cache Mimarisi — `koala73/worldmonitor`

Tek bir Redis değil, **3 katman:**

1. **Redis (Upstash REST)** — paylaşılan, TTL'li ana cache.
2. **Isolate-local bellek fallback** — Redis'e ulaşılamazsa devreye giren,
   boyutu sınırlı (maks. 5000 kayıt), kısa ömürlü (≤30sn) bir `Map`. Hem
   pozitif hem **negatif** sonuç tutuyor — başarısız bir fetch'i art arda
   tekrar denememek için "negative sentinel" + cooldown mekanizması var.
3. **İstemci tarafı kalıcı cache** — tarayıcıda IndexedDB → localStorage
   sırayla fallback; masaüstünde (Tauri) native dosya cache'i.

**İki ek desen:**
- **In-flight request coalescing:** Aynı anahtar için eşzamanlı gelen
  istekler tek bir upstream çağrısını paylaşıyor — "cache stampede"
  (aynı anda binlerce isteğin aynı veriyi tekrar tekrar çekmesi) önleniyor.
- **Stale-while-revalidate + negatif TTL sentinel:** Bir fetch `null`
  dönerse, kısa bir süre boyunca tekrar denenmiyor (gereksiz başarısız
  istek tekrarını önlüyor).

**Bizim projemize uygulanabilirlik:** Bizim mimaride Redis zaten Yusuf'un
canlı dashboard'unda var (TTL ~120sn) ama tek katmanlı. 3. katman (istemci
tarafı kalıcı cache) ve negative-sentinel deseni, dashboard tarafında
ileride "MinIO/Redis'e erişilemezse ne olur" senaryosu için referans
alınabilir — şu an zorunlu değil, gelecek iyileştirme notu.

---

## 3. H3 + MovingPandas ile Rota Genelleştirme — marksblogg tech blog

**Kaynak (doğrulandı):**
- https://tech.marksblogg.com/global-flight-tracking-adsb.html
- https://tech.marksblogg.com/aircraft-route-analysis-adsb.html

Tek bir "150km tolerans" parametresi değil — **üç ayrı MovingPandas adımı**,
üç farklı parametre grubuyla çalışıyor:

| Adım | MovingPandas sınıfı | Parametre | Ne yapıyor |
|---|---|---|---|
| 1 | `ObservationGapSplitter` | `gap=30 dakika` | Sürekli pozisyon akışını, aralarda 30 dk'dan büyük boşluk varsa ayrı "trip"lere böler |
| 2 | `MinDistanceGeneralizer` | `tolerance=100 metre` | Rotanın şeklini bozmadan gereksiz ara noktaları sadeleştirir (Douglas-Peucker benzeri) |
| 3 | `TrajectoryCollectionAggregator` | `max_distance=150km`, `min_distance=5km`, `min_stop_duration=30 dk`, `min_angle=45°` | **Asıl kümeleme adımı** — trajectory'lerden "önemli noktaları" çıkarır, kümeler, kümeler arası "akış" (flow) çizgilerini üretir. Çıktı: küme merkezleri (centroid + nokta sayısı) + akış çizgileri (kaç trajectory'nin o akışı kullandığı) |

**Veri altyapısı (marksblogg):** DuckDB + Parquet (günlük ~45 dosya, dosya
başına 1M kayıt, 8GB RAM'li makinede test edilmiş). `h3_5` (H3 resolution 5)
kolonu kaynak veride bazen zaten hazır geliyor, QGIS + Tile+ eklentisiyle
görselleştiriliyor.

**Bizim `routes.py`/`build_baseline.py` için uygulanabilirlik:** Mevcut
plan (`docs/BIREYSEL_PROJE_MASTER (1).md` Bölüm 5, Adım 10) DBSCAN
kullanıyor. `TrajectoryCollectionAggregator`, DBSCAN'e **alternatif değil,
tamamlayıcı**:
- DBSCAN → ham noktaları kümeler (mevcut plan, yoğunluk haritası için kalır).
- `TrajectoryCollectionAggregator` → trajectory'lerin önemli noktalarını
  çıkarıp kümeler + aralarındaki akışları birlikte üretir — rota
  kümeleme + koridor tespitinde DBSCAN'den daha zengin bir çıktı (flow/akış
  bilgisi DBSCAN'de yok), ayrıca FAZ B'nin (rota sapması tespiti) baseline
  kalitesini artırabilir: "bilinen hex seti" yerine gerçek bir akış
  çizgisine göre sapma ölçülebilir. Detay: `BIREYSEL_PROJE_MASTER (1).md`
  Bölüm 6.

**Literatür özeti için taslak cümle:**
> "Yörünge kümeleme için hem yoğunluk-tabanlı (DBSCAN) hem de
> trajectory-özel (MovingPandas'ın Andrienko & Andrienko (2011)
> algoritmasına dayanan `TrajectoryCollectionAggregator`) yaklaşımlar
> değerlendirildi; ikincisi kümeleme ile birlikte kümeler arası akış
> (flow) bilgisini de ürettiği için rota koridoru tespitinde ek değer
> sağlayabilir."

---

## 4. Lisans notu

`koala73/worldmonitor` **AGPL-3.0** lisanslı. Bölüm 1-2'deki bulgular
**mimari/yaklaşım referansı** olarak kullanılmalı — kod satırları
kopyalanmamalı, kendi implementasyonumuz yazılır. marksblogg'un kod
örnekleri (Bölüm 3) blog yazısı içinde paylaşılmış; kullanılacaksa kendi
lisans notları ayrıca teyit edilmeli.

MapLibre GL JS (BSD-3-Clause / MIT ekosistem, telif sorunu yok) ve
globe.gl (MIT) kütüphanelerinin kendisi worldmonitor'un koduyla alakasız
— bunları doğrudan kullanmak lisans açısından sorunsuz.
