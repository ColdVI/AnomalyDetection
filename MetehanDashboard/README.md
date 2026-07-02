# MetehanDashboard — MapLibre GL tabanlı bireysel ADS-B dashboard

Yusuf'un `Dashboard/` klasöründeki Dash+Leaflet arayüzünden **tamamen
bağımsız**, ayrı bir Next.js/React/TypeScript uygulaması. Onun kodunu
değiştirmiyor, sadece onun **FastAPI backend'ini** (aynı veri, farklı
görselleştirme) kullanıyor.

## Neden ayrı proje, fork değil

`src/` (Python pipeline) ve `Dashboard/` (Yusuf'un Dash arayüzü) klasörlerine
hiç dokunulmadı. Bu proje kendi bağımsız klasöründe (`MetehanDashboard/`),
kendi `package.json`'ıyla yaşıyor — git'te ayrı bir dal/klasör olarak durur,
Yusuf'un çalışmasıyla sıfır çakışma riski var.

## Mimari — tek gerçek bağımlılık

```
Yusuf'un altyapısı (DEĞİŞMEDİ)              Bu proje (YENİ)
┌──────────────────────────────┐            ┌─────────────────────┐
│ adsb_producer.py              │            │                     │
│   → Kafka (adsb.flights)      │            │  Next.js (port 3000)│
│ dashboard_consumer.py         │            │  MapLibre GL harita │
│   → Redis + InfluxDB          │            │  Cluster + tekil    │
│ app.py                        │───HTTP────▶│  görünüm            │
│   → FastAPI (port 8000, embed)│  :8000/api │                     │
│   → Dash UI (port 8050)       │            │                     │
└──────────────────────────────┘            └─────────────────────┘
```

**Önemli:** Bu dashboard'un veri görebilmesi için Yusuf'un `python app.py`si
arka planda çalışıyor olmalı — onun Dash arayüzünü (8050) hiç kullanmasak
bile, FastAPI (8000) o process'in **içinde** bir thread olarak başlıyor,
ayrı bir servis değil. Yani çalıştırman gereken (Yusuf'un kendi
`Dashboard/README`'sindeki sıra):

```
# Terminal 1 (Yusuf'un klasöründe)
python adsb_producer.py

# Terminal 2 (Yusuf'un klasöründe)
python dashboard_consumer.py

# Terminal 3 (Yusuf'un klasöründe) -- SADECE port 8000 icin lazim,
# tarayicida onun 8050 arayuzunu ACMANA gerek yok
python app.py

# Terminal 4 (BU proje)
npm install
npm run dev
```

Sonra tarayıcıda: **http://localhost:3000** (Yusuf'unki 8050'de, çakışma yok,
ikisi aynı anda açık kalabilir).

## Kurulum

```bash
cd MetehanDashboard
npm install
cp .env.local.example .env.local   # gerekirse API adresini degistir
npm run dev
```

## Neler var

- **`components/ui/mapcn-map-arc.tsx`** — sana attığın shadcn/mapcn component'i,
  hiç değiştirilmeden kopyalandı (`prompt.txt`'deki talimata birebir uyuldu).
- **`components/dashboard.tsx`** — asıl uygulama. İki görünüm modu:
  - **Tekil görünüm** (varsayılan): her uçak kendi dönen ikonuyla,
    alarm varsa kırmızı — Yusuf'un Dash'indeki `_airplane_icon` mantığının
    birebir React karşılığı.
  - **Küme görünümü** (sağ üstteki buton): `MapClusterLayer` ile GPU
    hızlandırmalı clustering. Aşağıdaki performans bölümüne bak.
- **`lib/api.ts`** — Yusuf'un `/api/flights`, `/api/alerts`, `/api/route/{callsign}`,
  `/api/aircraft_info/{icao24}`, `/api/history/{icao24}` endpoint'lerine
  fetch sarmalayıcıları. Backend şeması değişmedikçe buraya dokunmana gerek yok.
- **`OPSIYONEL_backend_route_lat_lon_yamasi.md`** — rota okunu (arc)
  haritada çizebilmek için Yusuf'un backend'ine önerilen, **isteğe bağlı**,
  geriye dönük uyumlu 4 satırlık ek. Uygulamazsan hiçbir şey bozulmaz,
  sadece ok çizilmez, rota bilgisi panelde metin olarak kalır.

## "Çok uçak verisiyle kasar mı?" — gerçek cevap

Bunu gerçekten inceledim, iki modu bu yüzden ekledim:

- **Tekil görünüm** (DOM tabanlı marker, Yusuf'un Dash+Leaflet'iyle aynı
  yaklaşım): yüzlerce uçakta (şu anki Türkiye-merkezli 500nm kapsama) sorunsuz.
  Binlere çıkarsan tarayıcı yavaşlamaya başlar.
- **Küme görünümü** (`MapClusterLayer`, MapLibre GL'in native GPU clustering'i):
  bu tam olarak bunun için var — on binlerce noktayı bile WebGL üzerinde
  agregatlar, DOM'a hiç dokunmaz. "Gerçek şirkette global ölçek" hedefliyorsan
  cevap bu — kod zaten yazılı, sadece butona basman yeterli.

CPU-only makinede ikisi de sorunsuz çalışır — burada hiçbir model eğitimi
yok, sadece görselleştirme + HTTP polling (15sn'de bir, Yusuf'un
producer'ıyla aynı ritim).

## Bilinen sınırlamalar (v1, dürüstçe)

- Redis okuma deseni backend tarafında hâlâ N ayrı `GET` (`_get_flights()`),
  bu proje onu değiştirmiyor — gerçekten global ölçeğe çıkarsan Yusuf'un
  tarafında `MGET`'e geçmek gerekir (bu repo'nun kapsamı dışında, onun kodu).
- Rota arc'ı sadece opsiyonel backend yaması uygulanırsa çizilir (yukarı bak).
- Next.js 14.2.35 kullanılıyor (14.x hattının en güncel yamalı sürümü,
  `npm audit`'in gösterdiği kalan uyarılar sadece internete açık production
  deploy'ları ilgilendiriyor — bu `localhost`'ta kalacak bir araç).

## Doğrulama notu

Bu proje burada gerçekten `npm install` + `npx tsc --noEmit` + `npm run build`
ile test edildi (uydurma kod değil) — production build temiz derleniyor,
sıfır TypeScript hatası. `maplibre-gl` ilk denemede v4.7.1 ile
`setProjection` tip hatası verdi (bu API MapLibre'de v5'te eklenmiş),
v5.24.0'a yükseltilip düzeltildi.
