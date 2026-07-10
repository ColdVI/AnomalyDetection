# ADSB-0 Planı — Altyapı Doğrulama, Uçuş Segmentasyonu, Fiziksel-Tutarlılık Residual'ları

> Bu doküman bir coding-agent (Codex / Claude Code) talimatıdır. Repo: `ColdVI/AnomalyDetection`.
> `docs/decisions.md` kuralları geçerlidir. `src/ml/` (ML-0..16) ve RFLY-0/1'e **hiç dokunulmaz** —
> bu tamamen ayrı, paralel bir keşif hattıdır (bkz. plan onayı, 2026-07-10).

## 0. Bağlam ve deney sorusu

9 farklı yöntem (IF, LightGBM, Chronos, LSTM-AE, Dense-AE, USAD, genlik-normalize skor,
drift-kalibreli füzyon) SEAD/RFLY üzerinde Gate C'yi (işe yararlık) geçemedi. Kök neden: (1) küçük/
heterojen normal havuzu, (2) LSTM/Dense-AE/USAD'ın kırpılmamış ölçekleme yüzünden "genlik-
baskınlığı" artefaktına yakalanması (ADR-016..019) — model gerçekten fizik öğrenmiyor, sadece
büyük sayılara bakıyor.

**Deney sorusu:** adsb.lol/readsb ADS-B telemetrisinde, bildirilen kanallarla (vertical_rate,
ground_speed, track) bunların ham lat/lon/alt'tan **aritmetik olarak türetilen** karşılıkları
arasındaki tutarsızlık (residual), öğrenilmiş bir modele hiç ihtiyaç duymadan anlamlı bir anomali
sinyali üretir mi? Bu, "genlik-baskınlığı" artefaktının yapısal olarak oluşamayacağı bir tasarım
(residual bir özdeşlik farkı, öğrenilmiş bir skor değil).

ADSB-0'ın kapsamı **sadece altyapı**: veri erişimi, uçuş segmentasyonu, residual hesaplama,
görselleştirme. Model/eşik/anomali kararı YOK — o ADSB-1'in işi.

## 1. Kapsam / kapsam dışı

**Kapsamda:** adsb.lol historical Silver şeması (`src/silver/parse_adsblol_historical.py`
çıktısı) üzerinde uçuş segmentasyonu + 4 fiziksel-tutarlılık residual'ı + görselleştirme galerisi.

**Kapsam DIŞI:** herhangi bir anomali modeli/eşiği (ADSB-1), gerçek/nadir ADS-B anomalisi arama,
`src/ml/` `src/silver/parse_adsblol_historical.py` dahil mevcut ingestion koduna müdahale (yalnız
OKUNUR, değiştirilmez), RFLY/SEAD öncelik sırasına dokunma.

## 2. Veri kaynağı ve mevcut altyapı (repo'da zaten var, doğrulandı)

- `src/silver/parse_adsblol_historical.py` — tar → Silver Parquet parser (Metehan, ADR-003).
  Silver şeması: `source_id`(=icao24), `timestamp_utc`, `lat`, `lon`, `alt`, `alt_geom_m`,
  `on_ground`, `ground_speed_ms`, `track_deg`, `vertical_rate_ms`, `indicated_airspeed_ms`,
  `roll_deg`, `flags_stale`, `flags_new_leg`, `label`(=her zaman None), + metadata.
- En az bir günlük tar zaten parse edilmiş: 97.8M satır, 67.577 uçak (`logs/parallel_parse/
  v2026.06.28-planes-readsb-prod-0.log`, `Dashboard/FULL_PROJECT_HANDOFF.md` §3.3) — ama bu
  MetehanSarikaya'nın kendi makinesinde commit'lenmiş (`git log`, 2026-07-07); Silver Parquet'in
  kendisi Docker-yönetimli `minio_data` volume'ünde, makineye özel, BU makinede değil.
- Bu makinede: ham `.tar` yok (`data/bronze/adsblol_historical/` boş), MinIO şu an çalışmıyor
  (Docker erişilemedi, 2026-07-10 denendi). **AÇIK ENGEL:** kullanıcının Drive'daki ham tar'lardan
  birini bu makineye indirip `python -m src.silver.parse_adsblol_historical --local-tar <path>`
  ile yerel parse çalıştırması, ya da Metehan'dan Silver parquet aktarımı istemesi gerekiyor.
  Aşağıdaki §3-4 kodu bu veri olmadan da sentetik veriyle geliştirilip test edilebilir (ve
  edildi) — gerçek veri geldiğinde sadece uçtan uca çalıştırma kalıyor.

## 3. `src/adsb/segmentation.py` — DURUM: YAZILDI + test edildi

```
assign_flight_ids(df, *, id_col="source_id", time_col="timestamp_utc", gap_s=1800.0) -> pd.Series
segment_flights(df, ...) -> pd.DataFrame   # + flight_id kolonu, id+zaman sirali
new_leg_agreement(df, ...) -> float        # gap-kurali vs flags_new_leg uyusma orani
```

Boşluk-tabanlı kesim `src/ml/data/windowing.py`'deki `max_gap_s` diff-kesme desenini örnek alır
(repoda hazır bir uçuş-segmentasyon fonksiyonu YOKTU, sıfırdan yazıldı — doğrulandı, bkz. Explore
araştırması 2026-07-10). Varsayılan eşik 30dk, `gap_s` parametresiyle değiştirilebilir.
`flags_new_leg` (readsb'nin kendi segmentasyonu) kör biçimde ezilmez — `new_leg_agreement()` iki
yöntemin uyuşma oranını raporlar.

**Test kapsamı** (`tests/test_adsb_segmentation.py`, 9 test, hepsi geçti): boşluklu/boşluksuz
bölme, sırasız girdiyle tutarlılık, boş df, çıktı sıralaması, agreement=1.0/0.0/NaN (sınır yokken)
durumları.

## 4. `src/adsb/physics_features.py` — DURUM: YAZILDI + test edildi

Dört residual, hepsi `flight_id` bazında gruplanmış diff/shift (uçuş sınırı dışına taşmaz):

| Fonksiyon | Residual | Kapsama |
|---|---|---|
| `vertical_rate_residual` | bildirilen `vertical_rate_ms` − ölçülen `Δalt/Δt` | tam |
| `speed_residual` | bildirilen `ground_speed_ms` − haversine(`Δlat,Δlon`)/`Δt` | tam |
| `heading_residual` | bildirilen `track_deg` − bearing(`Δlat,Δlon`) (dairesel) | tam |
| `turn_bank_residual` | dönüş hızı (`Δtrack/Δt`) − `g·tan(roll)/v` | ~%8.5 (roll_deg seyrek) |

`compute_physics_residuals()` dördünü birden ekler; `roll_deg` kolonu yoksa `turn_bank_residual`
tüm satırlarda NaN döner (birincil karar bu kanala bağımlı OLMAYACAK — ADSB-1'in tasarım kuralı).

**Test kapsamı** (`tests/test_adsb_physics_features.py`, 9 test, hepsi geçti): elle inşa edilmiş
fiziksel-tutarlı sentetik uçuşlarda residual≈0 (haversine ile AYNI Dünya yarıçapı kullanılarak,
`EARTH_RADIUS_M`), enjeksiyon sonrası residual'ın büyümesi, `roll_deg` eksikliğinde NaN, uçuş
sınırı dışına diff sızmaması.

**Önemli metodolojik not:** residual'lar öğrenilmiş DEĞİL, aritmetik özdeşlik — bu yüzden
"genlik-baskınlığı" artefaktı (ADR-016) burada yapısal olarak oluşamaz: residual sıfıra
yakınsa kanal fiziksel olarak tutarlıdır, büyüklüğün (magnitude) kendisi hiçbir şey ifade etmez.

## 5. `scripts/make_adsb_visualizations.py` + galeri — DURUM: BEKLİYOR (veri engeli)

Plan: her uçuş için çok-panelli PNG (lat/lon rota + irtifa + hız + dikey-hız + track + 4
residual), `scripts/make_visualizations.py`'nin matplotlib deseni taklit edilir (per-flight PNG,
`ax.axvspan`/`ax.plot` düzeni). `scripts/build_viz_gallery.py`'nin statik HTML index üretim
mantığı (`GRAPH_TYPES` sınıflandırması, gömülü CSS/JS) yeniden kullanılarak
`artifacts/adsb/viz/index.html` üretilir. **Gerçek veri gelmeden anlamlı şekilde yazılamaz/
test edilemez** — §2'deki engel çözülünce bu adım tamamlanır.

## 6. Faz kapısı (model YOK, insan-gözü doğrulama)

ADSB-1'e geçmeden önce: en az bir günlük gerçek trafik üstünde, birkaç temiz uçuşta residual'lar
görsel/istatistiksel olarak ~0 civarında olmalı (birim/işaret hatası yok testi). Segmentasyonun
`new_leg_agreement()` oranı da raporlanır (belirli bir eşik önceden burada sabitlenmiyor — ilk
gerçek veri koşusunda gözlemlenen oran, sonraki fazın makul kabul edip etmeyeceğine karar
vermesi için ADR'ye yazılır; sonuç ne olursa olsun dürüstçe raporlanır).

## 7. Test listesi (pytest, tamamlandı)

- `tests/test_adsb_segmentation.py` — 9/9 geçti
- `tests/test_adsb_physics_features.py` — 9/9 geçti
- `tests/test_adsb_injection.py` — 4/4 geçti (ADSB-1 için önceden hazırlandı, bkz. o plan)
- Tam paket: `pytest -q` → 329 geçti, 6 atlandı, yalnız önceden bilinen 4 MinIO SDK hatası
  (bu değişiklikle ilgisiz, Codex tarafından da 2026-07-10'da doğrulandı).

## 8. Bu turda YAPILMAYACAKLAR

- `scripts/make_adsb_visualizations.py`'nin gerçek veri üstünde çalıştırılması (veri engeli).
- Herhangi bir model/eşik (ADSB-1'in işi).
- `src/silver/parse_adsblol_historical.py`'a değişiklik (zaten doğru çalışıyor, dokunulmaz).

## 9. Sonraki faz

`docs/ADSB1_PHYSICS_DETECTOR_PLAN.md` — fiziksel-tutarlılık detektörü + sentetik-enjeksiyon
doğrulaması.
