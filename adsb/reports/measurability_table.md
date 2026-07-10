# Faz 0.5 — Fiziksel İlişki Ölçülebilirlik Tablosu

Kaynak: `data/objectstore/silver/adsblol_historical` (tar-1, `v2026.02.28`, 550 Silver
parça, 11.142.219 satır — ilk 30 parça üstünde satır-düzeyi kapsama ölçüldü;
0.1'deki `ac_dict_field_presence` uçak-düzeyi "en az bir kez görüldü" istatistiğiydi,
bu tablo daha sıkı olan **satır-düzeyi** kapsamayı kullanıyor).

## Kolon kapsaması (satır-düzeyi, gerçek veri)

| Kolon | Kapsama | Not |
|---|---|---|
| `lat`, `lon` | %100 | her zaman var |
| `on_ground` | %100 | boolean, her zaman var |
| `ground_speed_ms` | %98.1 | neredeyse tam |
| `track_deg` | %95.2 | neredeyse tam |
| `alt`, `alt_geom_m` | %89.4 / %89.2 | `on_ground=True` iken null (beklenen) |
| `vertical_rate_ms` | %89.4 | `alt` ile birlikte hareket ediyor |
| `indicated_airspeed_ms` | %29.7 | seyrek |
| `roll_deg` | %28.4 | **forward-fill YOK** (`parse_adsblol_historical.py`da `rec["roll"]`
  doğrudan kullanılıyor, `last_ac.get(...)` gibi sparse-update taşıma yok) — bu
  yüzden format referansındaki eski %8.5 rakamından daha yüksek ama yine de azınlık |

## Fiziksel ilişki kararları

| İlişki | Karar | Gerekçe |
|---|---|---|
| `vertical_rate_ms` ↔ `Δalt/Δt` | **Ölçülebilir** | İkisi de ~%89 kapsamda, aynı satırlarda birlikte var/yok |
| `ground_speed_ms` ↔ haversine(`Δlat,Δlon`)/`Δt` | **Ölçülebilir** | lat/lon %100, speed %98.1 |
| `track_deg` ↔ bearing(`Δlat,Δlon`) | **Ölçülebilir** | lat/lon %100, track %95.2 |
| dönüş hızı (`Δtrack/Δt`) ↔ `g·tan(roll)/v` | **Yalnız-geçişte-ölçülebilir** | `roll_deg` %28.4, forward-fill yok — yalnız bu alt-kümede birincil sinyal, genelde ikincil/tamamlayıcı |
| gerçek hava hızı (rüzgarsız/rüzgarlı ayrımı) | **ADS-B'den ölçülemez** | `indicated_airspeed_ms` %29.7 ve rüzgar (`wd`/`ws`) verisi de seyrek/opsiyonel; TAS-GS farkından rüzgar çıkarımı yapılabilir ama bu fazda kapsam dışı |

## Karar: ilk-geçiş feature seti (ADSB-1 eğitimi için)

**Birincil (tam kapsamaya yakın, model girdisinin çekirdeği):**
`alt`, `ground_speed_ms`, `track_deg`, `vertical_rate_ms` + üç fiziksel-tutarlılık
residual'ı (`vertical_rate_residual`, `speed_residual`, `heading_residual`).

**İkincil (seyrek, ayrı maske kanalıyla, birincil karar buna bağımlı değil):**
`roll_deg`, `turn_bank_residual`.

Kapsam dışı bırakıldı: `indicated_airspeed_ms` tek başına (seyrek + `roll_deg` gibi
forward-fill'siz), meteorolojik alanlar (`wd`/`ws`/`oat`/`tat` vb.).

## Karar: değerlendirme birimi

**Sabit pencere** (fixed window) seçildi — dört model mimarisi de (Dense-AE, LSTM-AE,
USAD, LSTM-forecaster) zaten pencere üstünde çalışıyor (`adsb/windowing.py`), bu
yüzden pencere-düzeyi skor doğal birincil birim. Uçuş-düzeyi ve event-düzeyi metrikler
pencere skorlarının üstüne AGGREGATION olarak (max/mean, event-onset recall) daha
sonra eklenir — modelleme katmanında ayrı bir birim seçimi gerekmiyor. Bu, kullanıcı
onayı beklemeden ilerlemeyi mümkün kılan pragmatik bir varsayılan seçim; itiraz
edilirse değiştirilebilir.
