# adsb.lol / readsb globe_history Veri Formatı Referansı

Bu doküman, ALFA ve UAV Attack Dataset paper'larının muadili olarak hazırlanmıştır.
adsb.lol'ün kendi yayımlanmış bir paper'ı yok — format, alttaki açık kaynak yazılım olan
**readsb**'nin (wiedehopf fork) resmi dokümantasyonundan derlenmiştir.

Kaynak: https://github.com/wiedehopf/readsb/blob/dev/README-json.md
Repo: https://github.com/adsblol/globe_history_2026

---

## 1. Genel Yapı

Her günlük release bir `.tar` arşividir (parçalı indirilirse `.tar.aa` + `.tar.ab` birleştirilir).
Açıldığında üç ana klasör çıkar:

```
traces/    → asıl veri, bizim kullanacağımız kısım
heatmap/   → replay/heatmap görselleştirmesi için, bize gerekmiyor
acas/      → Airborne Collision Avoidance System verisi, bize gerekmiyor
```

`traces/` altında 2 haneli alt klasörler var (`00/`, `01/`, ... `ff/` — ICAO24'ün son byte'ına göre
gruplama), her birinin içinde **bir uçağın bir günlük tüm trace'i, gzip'li tek bir JSON dosyası**:

```
traces/c0/trace_full_c03001.json.gz
```

Dosya adındaki hex (`c03001`) = ICAO24 transponder ID.

---

## 2. Trace Dosyasının Üst Seviye Alanları

Dosyayı açtığında (`gunzip` + `json.loads`) şu yapı çıkar:

| Alan | Açıklama |
|---|---|
| `icao` | 24-bit ICAO transponder ID (hex string, 6 karakter) |
| `r` | Tail number / kayıt numarası (örn. "C-FSEQ") |
| `t` | ICAO aircraft type code (örn. "B38M" = 737 MAX 8) |
| `desc` | Uçak tipi açık isim (örn. "BOEING 737 MAX 8") |
| `dbFlags` | Bitfield: military, interesting, PIA, LADD bayrakları |
| `ownOp` | Operatör/havayolu adı (örn. "Air Canada") |
| `year` | Uçağın üretim yılı (varsa) |
| `timestamp` | Bu dosyanın referans unix timestamp'i (saniye) — trace'teki her satır buna **göreceli** |
| `trace` | Asıl pozisyon verisi — array of arrays (bkz. Bölüm 3) |

**Önemli:** `dbFlags`, `r`, `t`, `desc`, `ownOp`, `year` alanları sadece readsb'nin aircraft
database'inde (tar1090-db) o ICAO24 kayıtlıysa gelir. Kayıtlı değilse bu alanlar dosyada
**hiç olmaz** (null değil, key'in kendisi yok).

---

## 3. Trace Array — Asıl Pozisyon Verisi

`trace` alanı, her elemanı bir pozisyon kaydı olan bir array'dir. Her kayıt **sabit pozisyonlu bir
array** (dict değil!), sütun isimleri yok — index'e göre okunmalı:

```python
trace[i] = [
    0,   # [0] seconds_after_timestamp  -> gerçek zaman: dosya timestamp + bu değer
    1,   # [1] lat
    2,   # [2] lon
    3,   # [3] altitude  -> feet cinsinden sayı, veya "ground" (string), veya null
    4,   # [4] ground_speed  -> knots, veya null
    5,   # [5] track  -> derece (0-359); EĞER altitude=="ground" ise bu heading olur, track değil
    6,   # [6] flags  -> bitfield, bkz. Bölüm 4
    7,   # [7] vertical_rate  -> fpm (feet/minute), veya null
    8,   # [8] aircraft_dict  -> ek detaylar, dict veya null, bkz. Bölüm 5
    9,   # [9] source_type  -> "adsb_icao", "mlat", "tisb_icao" vb. (sadece 2022+ dosyalarda)
    10,  # [10] geometric_altitude  -> feet, GNSS/INS bazlı (sadece 2022+ dosyalarda)
    11,  # [11] geometric_vertical_rate  -> fpm (sadece 2022+ dosyalarda)
    12,  # [12] indicated_airspeed  -> knots (sadece 2022+ dosyalarda)
    13,  # [13] roll_angle  -> derece, negatif=sol roll (sadece 2022+ dosyalarda)
]
```

### Pratik notlar

- **Timestamp hesaplama:** `gerçek_zaman = dosya["timestamp"] + trace[i][0]`
- **Altitude iki türlü:** `trace[i][3]` barometrik (basınca dayalı), `trace[i][10]` geometrik
  (GNSS'e dayalı). İkisi arasında ~50-200m fark normal — biri MSL diğeri farklı referans.
- **`altitude == "ground"` durumu:** Uçak yerdeyken bu alan sayı değil, `"ground"` string'i
  olur. Filtrelerken tip kontrolü şart (`isinstance(x, (int,float))`).
- **2020-2021 dosyalarında** index 9-13 (source_type, geometric verileri, IAS, roll) **yok** —
  array sadece 9 elemanlı olabilir. Bizim indirdiğimiz 2026 dosyalarında bu sorun olmamalı
  ama kod yazarken `len(trace_row)` kontrolü eklemek güvenli.

---

## 4. Flags Bitfield (index 6)

Bitwise AND ile okunur:

| Bit | Maske | Anlamı |
|---|---|---|
| 0 | `flags & 1 > 0` | Pozisyon "stale" (bu kayıttan önce 20+ saniye pozisyon gelmemiş) |
| 1 | `flags & 2 > 0` | Yeni "leg" başlangıcı — readsb'nin iniş/kalkış ayrımı tespiti (uçuş segmentasyonu için kullanılabilir) |
| 2 | `flags & 4 > 0` | Vertical rate barometrik değil, geometrik |
| 3 | `flags & 8 > 0` | Altitude barometrik değil, geometrik |

`flags & 2` özellikle bizim için değerli — **readsb'nin kendi uçuş segmentasyon mantığı**, bizim
manuel "30 dakika gap → yeni flight" kuralımıza alternatif/referans olarak kullanılabilir.

---

## 5. Aircraft Dict (index 8) — Ek Detaylar

Her trace satırında olmaz, sadece veri değiştiğinde gelir (önceki satırdaki değer hâlâ geçerli
sayılır — yani sparse update). `null` olduğunda bir önceki dolu satırdaki değerler kullanılmalı.

En sık kullanılacak alt alanlar:

| Alan | Açıklama |
|---|---|
| `flight` | Callsign, 8 karaktere tamamlanmış (örn. `"ACA971  "` — sondaki boşluklara dikkat, strip gerekir) |
| `category` | Emitter kategorisi (A0-D7) — örn. A1=küçük uçak, A3=büyük uçak, B6=UAV/drone |
| `squawk` | 4 haneli oktal transponder kodu |
| `emergency` | "none" veya acil durum tipi |
| `nic`, `rc`, `nac_p`, `nac_v`, `sil` | Pozisyon doğruluk/güvenilirlik metrikleri (ADS-B versiyon 2 standardı) |
| `alert`, `spi` | Flight status bitleri |
| `version` | ADS-B versiyon numarası (0, 1, 2) |

`category` alanı anomali/sınıflandırma için ilginç olabilir — B6/B7 gibi kodlar UAV/drone
emitter'larını işaretler, A serisi ise konvansiyonel uçakları.

### 5.1 Ampirik Doğrulama (300 dosyalık örneklem, v2026.06.15)

Yukarıdaki alan listesi, gerçek veriden 300 uçak / ~459K trace noktası taranarak doğrulandı.

**Dosya seviyesi alanların görülme sıklığı:**

| Alan | Görülme | Not |
|---|---|---|
| `icao`, `version`, `timestamp`, `trace` | 300/300 (%100) | Zorunlu |
| `r` | 270/300 (%90) | Opsiyonel |
| `dbFlags` | 270/300 (%90) | Opsiyonel |
| `t` | 268/300 (%89) | Opsiyonel |
| `desc` | 254/300 (%85) | Opsiyonel |
| `ownOp` | 162/300 (%54) | Opsiyonel |
| `year` | 153/300 (%51) | Opsiyonel |
| `noRegData` | 30/300 (%10) | Sadece kayıtsız uçaklarda |

**Trace array uzunluğu:** Örneklemdeki 459.287 kaydın **tamamı 14 elemanlı** — sapma yok.
2020-2021 formatındaki 9 elemanlı kayıtlardan bu tarih aralığında (2026) hiç görülmedi.

**Aircraft dict alanlarının görülme sıklığı (459K kayıt içinde, çoktan aza):**

| Alan | Görülme | Alan | Görülme |
|---|---|---|---|
| `type`, `nic`, `rc` | 114.876 | `nic_baro` | 101.573 |
| `sil_type` | 114.138 | `emergency` | 100.368 |
| `category` | 113.287 | `gva` | 99.638 |
| `version` | 113.268 | `baro_rate` | 90.656 |
| `nac_p`, `sil` | 112.458 | `nav_altitude_mcp` | 88.380 |
| `track` | 110.977 | `nav_qnh` | 87.580 |
| `nac_v` | 110.831 | `geom_rate` | 54.850 |
| `alert` | 110.505 | `nav_heading` | 51.312 |
| `spi` | 110.500 | `true_heading` | 49.274 |
| `flight` | 109.601 | `mag_heading` | 40.892 |
| `squawk` | 108.685 | `ias`, `mach` | ~40.350 |
| `sda` | 107.471 | `roll`, `tas` | ~39.000 |
| `alt_geom` | 104.253 | `wd`, `ws` | 35.782 |

**Daha önce dokümana eklenmemiş, ama sık görülen ek alanlar** (meteorolojik/performans verileri,
çoğunlukla büyük/yüksek-versiyonlu uçaklarda mevcut):

| Alan | Açıklama |
|---|---|
| `mach` | Mach sayısı |
| `tas` | True airspeed (gerçek hava hızı, knot) |
| `ias` | Indicated airspeed (knot) |
| `wd`, `ws` | Hesaplanan rüzgar yönü/hızı |
| `oat`, `tat` | Dış/toplam hava sıcaklığı (°C) |
| `nav_modes` | Aktif otopilot modları (autopilot, vnav, lnav, tcas vb.) |
| `nav_altitude_fms` | FMS hedef irtifa |
| `mag_heading`, `true_heading` | Manyetik/gerçek pusula yönü |
| `track_rate` | Track açısının değişim hızı (derece/sn) |

---

## 6. Pandas/DuckDB'ye Çevirme — Hedef Şema

Silver katmanı için önerilen düz (flat) tablo:

```python
COLUMNS = [
    "icao24",            # dosya seviyesi
    "registration",      # r
    "aircraft_type",     # t
    "desc",
    "owner_op",          # ownOp
    "timestamp_utc",     # file.timestamp + trace[0]
    "lat",                # trace[1]
    "lon",                # trace[2]
    "altitude_baro_ft",   # trace[3], "ground" ise None + on_ground=True
    "on_ground",          # trace[3] == "ground"
    "ground_speed_kt",    # trace[4]
    "track_deg",          # trace[5]
    "vertical_rate_fpm",  # trace[7]
    "flight_callsign",    # trace[8].flight, strip edilmiş
    "category",           # trace[8].category
    "squawk",              # trace[8].squawk
    "source_type",         # trace[9]
    "altitude_geom_ft",    # trace[10]
    "is_new_leg",           # flags & 2
]
```

---

## 7. Colab İçin Hızlı İnceleme Kodu

Drive'da tutulan tar dosyalarından **tek bir trace dosyasını** çıkarıp ilk 10 kaydı göstermek için:

```python
from google.colab import drive
drive.mount('/content/drive')

import tarfile, gzip, json, glob, os

TAR_PATH = "/content/drive/MyDrive/staj/proje/Tar/adsb_raw/v2026.06.15-planes-readsb-prod-0.tar.aa"
# Not: .tar.aa + .tar.ab parçalıysa önce birleştirmen gerekir, aşağıda script var.

EXTRACT_DIR = "/content/adsb_sample"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# --- Parçaları birleştir (sadece bir kere çalıştır) ---
def merge_tar_parts(base_path_no_ext, parts=["aa", "ab"]):
    merged_path = base_path_no_ext + ".tar"
    if os.path.exists(merged_path):
        print("Zaten birleşmiş:", merged_path)
        return merged_path
    with open(merged_path, "wb") as out:
        for p in parts:
            part_file = f"{base_path_no_ext}.tar.{p}"
            if os.path.exists(part_file):
                print("Ekleniyor:", part_file)
                with open(part_file, "rb") as f:
                    out.write(f.read())
    return merged_path

# --- Tar'dan SADECE birkaç trace dosyasını çıkar (RAM dostu) ---
def peek_traces(tar_path, n_files=5):
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if "traces" in m.name and m.name.endswith(".json")]
        print(f"Toplam trace dosyası: {len(members)}")
        sample = members[:n_files]
        for m in sample:
            f = tar.extractfile(m)
            raw = f.read()
            try:
                data = json.loads(gzip.decompress(raw))
            except OSError:
                data = json.loads(raw)  # bazı sürümlerde .json uzantılı ama gzip olmayabilir
            print("\n--- ", m.name, " ---")
            print({k: v for k, v in data.items() if k != "trace"})
            print(f"Trace uzunluğu: {len(data.get('trace', []))}")
            if data.get("trace"):
                print("İlk kayıt:", data["trace"][0])
                if len(data["trace"]) > 1:
                    print("İkinci kayıt:", data["trace"][1])

# --- Çalıştır ---
merged = merge_tar_parts(TAR_PATH.replace(".tar.aa", ""))
peek_traces(merged, n_files=5)
```

Bu kod tüm 47K dosyayı açmadan, sadece **ilk 5 uçağın** trace yapısını gösterir — Colab'ı
çökertmeden format doğrulaması yapmak için yeterli.

---

## 8. Bizim İçin Önemli Olmayan Kısımlar

- `heatmap/` klasörü — bizim kendi H3 density haritamızı kendimiz üreteceğiz, hazır heatmap
  binary'sine ihtiyacımız yok.
- `acas/` klasörü — collision avoidance verisi, coğrafi analiz veya UAV anomali projemizle
  ilgisi yok.
- `mlat`, `tisb` ile gelen pozisyonlar düşük güvenilirlikli — `source_type` alanına göre
  istersek bunları filtreleyebiliriz (`source_type == "adsb_icao"` en güvenilir olanı).

---

## 9. Lisans

ODbL 1.0 (Open Database License). Paylaşım/atıf şartlarına dikkat — raporda kaynak olarak
`adsb.lol / readsb (wiedehopf)` belirtilmeli.
