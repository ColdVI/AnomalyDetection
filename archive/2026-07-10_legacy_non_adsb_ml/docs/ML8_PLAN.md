# ML-8A Planı — Temporal Boosting + Karar Katmanı Ayrıştırması

> Bu doküman bir coding-agent (Codex / Claude Code) talimatıdır. Repo: `ColdVI/AnomalyDetection`.
> `docs/AGENTS.md` ve `docs/decisions.md` kuralları geçerlidir. **Yeni modül yolları mevcut
> yapıyla çelişirse MEVCUT YAPIYI esas al** (özellikle §2'deki uyarılar). Bu plan, ML8A ham
> taslağının repo gerçeğiyle doğrulanmış ve üç noktada düzeltilmiş halidir (bkz. §16 değişiklik
> kaydı) — ham taslakla çelişki olursa bu doküman kazanır.

## 0. Bağlam ve deney sorusu

ML-1..7 özeti (`docs/ML1_BULGULAR_VE_HATALAR.md`): uçuş-düzeyi ROC yeterli (ALFA LSTM-AE 0.918,
SEAD 0.678±0.013) fakat operasyonel onset tespiti FA bütçesi altında başarısız (kritik 0.030,
advisory 0.164 → `development_rejected`). Elimizdeki alarm taramasında (378 kombinasyon) FA
sınırı tamamen kaldırılsa bile onset recall **%37.6**'da tavan yapıyor — bu bir sinyal tavanı,
sadece eşik ayarı sorunu değil.

**Deney sorusu:** UAV-SEAD ve ALFA'da, dondurulmuş causal temporal descriptor'larla eğitilmiş
class-balanced LightGBM, aynı karar katmanları altında mevcut IF-füzyon/LSTM-AE'ye göre
FA-bütçeli event-onset recall ve detection delay'i iyileştiriyor mu — ve iyileşme **skor
katmanından mı karar katmanından mı** geliyor?

Üç katman ayrı ölçülür, birbirine karıştırılmaz:
`Evidence (feature) → Score (model) → Decision (alarm politikası)`

## 1. Kapsam / kapsam dışı

**Kapsamda:** UAV-SEAD (349 uçuşluk mevcut subset) + ALFA (54 uçuş, rosbag'ler dahil).
Supervised binary pencere sınıflandırma. Üç karar katmanı. Mevcut modellerle **aynı karar
katmanları altında** adil kıyas.

**Kapsam DIŞI (bu fazda yapma):** UAV Attack MIL (ML-8B), DevNet/Deep SAD (ML-8C), Chronos/TSFM
(ML-9), GDN/Transformer, Optuna/hyperparameter search, yeni descriptor ekleme, MLflow,
blind holdout'u açmak.

## 2. Açılacak / genişletilecek dosyalar

```
YENİ:
  src/ml/features/window_descriptors.py
  src/ml/models/temporal_boosting.py
  src/ml/decision/decision_layers.py
  scripts/run_ml8a_temporal_boosting.py
  notebooks/08_temporal_boosting.ipynb
  tests/test_ml8a_descriptors.py
  tests/test_ml8a_decision_layers.py

GENİŞLET (yeniden yazma):
  src/ml/evaluation/events.py        ← event/onset metrik kodu ZATEN BURADA
  docs/ML1_BULGULAR_VE_HATALAR.md    ← "ML-8A sonuçları" bölümü
  docs/decisions.md                  ← Gate A/B/C kararları
```

**DÜZELTME 1 — `src/ml/eval/` AÇMA.** Event/onset metrikleri zaten
[src/ml/evaluation/events.py](../src/ml/evaluation/events.py)'te (`k_of_n_alarm`,
`persistent_alarm`, `event_metrics` — hepsi nedensel, prefix-invariant). Yeni bir `eval/`
dizini `evaluation/` ile karışır. LightGBM yolu bu **aynı** `event_metrics`'i çağırmalı;
kopya çıkarma. `test_event_metrics_matches_ml7` bunu doğrulasın.

**DÜZELTME 2 — MLflow YOK.** Repoda MLflow yok ve yerleşik desen
[src/ml/artifacts.py](../src/ml/artifacts.py)'teki checksum'lu JSON manifest
(`save_modular_iforest_bundle` / `manifest.json` + SHA-256). ML-8A da bu deseni kullanır;
MLflow eklenmez (§9).

## 3. `window_descriptors.py` spesifikasyonu

### Pencereleme
- Causal (sağa hizalı) pencere: **10 s**, stride **1 s**. Pencere yalnız `[t-10s, t]` görür.
- Pencere hiçbir zaman uçuş sınırını aşamaz; her uçuş (`source_id`) bağımsız işlenir.
- Girdi feature tabloları `data/gold/ml_features/<source>/<source>_ml_features.parquet`;
  zaman ekseni ID kolonu `t_rel_s` (saniye, uçuş-içi göreli). Pencere `t_rel_s` üzerinden
  kesilir; gerçek örnekleme hızına göre satır sayısı değişebilir (`min_periods` kullan).
- ALFA yüksek-Hz backbone'da descriptor öncesi kanal bazında yalnız **geçmişe-bakan**
  downsample yapılabilir (bfill/interpolate ile ileriye taşıma YASAK).

### Girdi kanalları (kaynak başına — REPO İLE DOĞRULANDI)
Kanal listesi kaynak başına `artifacts/ml8a/descriptor_schema_v1.json` içine açıkça yazılır.
Feature tablosunda **var olduğu doğrulanmış** çekirdek kanallar:

| Kaynak | Çekirdek kanallar (mevcut) |
|---|---|
| **ALFA** | `xtrack_error`, `alt_error`, `gps_speed_residual` (+ `roll_error`, `pitch_error`, `airspeed_error`, `path_dev_mag` — feature tablosunda mevcut olanları ekle) |
| **UAV-SEAD** | `gps_speed_residual`, `alt_baro_residual`, `alt_local_residual`, `attitude_error_mag`, `control_strain` (+ EKF reject bit-count'ları `*_bit_count`, missingness maskeleri) |

> ALFA'da `alt_baro_residual`/`attitude_error_mag` YOK, SEAD'de `xtrack_error`/`alt_error` YOK —
> şemayı kaynağa göre ayır, olmayan kanalı sessizce atla (KeyError verme). Şemaya yalnız
> `df.columns`'ta gerçekten bulunan kanallar girer; hangi kanalın hangi kaynakta kullanıldığı
> şema JSON'una yazılır.

### Descriptor seti (kanal başına, v1 — DONDURULMUŞ, 20 adet)
```
mean, std, min, max, median, q10, q25, q75, q90, range,
first, last, last_minus_first, linear_slope,
diff_mean, diff_std, diff_abs_max,
lag1_autocorrelation,
missing_fraction, stale_fraction
```

### Kurallar
- Geleceğe bakmak YOK. `bfill` YOK. Yalnız sınırlı `ffill` (maks. 2 s) + `age_since_last_observation`.
- Scaler/imputation istatistikleri **yalnız train'de** fit edilir (mevcut
  [src/ml/data/scaling.py](../src/ml/data/scaling.py) desenini kullan).
- `descriptor_schema_v1.json` üretilir (kaynak→kanal listesi + descriptor listesi + pencere/stride
  + ffill limiti). SHA-256 hash'i her koşu manifest'ine yazılır. **Test sonucu görüldükten sonra
  şemaya ekleme/çıkarma YASAK**; değişiklik ancak `v2` olarak yeni faz açar.
- **Prefix-invariance testi:** Bir uçuşun ilk T saniyesiyle hesaplanan descriptor'lar, uçuşun
  tamamı verildiğinde ilk T saniye için birebir aynı çıkmalı (ML-6 CUSUM testiyle aynı desen:
  bkz. `tests/test_ml_features.py::test_cusum_is_prefix_invariant...`). pytest'e ekle.

## 4. Label politikası

Pencere etiketi, pencere ile anomali aralığının **kesişim ORANI** üzerinden verilir:

| Kesişim oranı | Train etiketi |
|---|---|
| ≥ %50 | positive |
| = %0 | negative |
| %0 < oran < %50 (guard-band) | **train'den DIŞLANIR** |

Guard-band dışlaması yalnız eğitim içindir; **değerlendirme her zaman tam skor akışı üzerinde**
event metrikleriyle yapılır (guard-band pencereleri değerlendirmede skorlanır, sadece train
etiketi almazlar).

### UAV-SEAD — mevcut range→satır kodunu YENİDEN KULLAN
- Ranges kaynağı: `data/objectstore/bronze/uav_sead/labels.json`. Yapı:
  `meta["ranges"] = [[[sinyal_adı, [[start_us, end_us], ...]], ...]]` (mutlak PX4 μs).
  Ayrıştırma için [scripts/run_ml6_events.py](../scripts/run_ml6_events.py) `load_ranges()`
  fonksiyonunu **aynen kullan** (yeni parser yazma).
- `t_rel_s` → mutlak μs eşlemesi: `absolute_us = t0[sid] + t_rel_s*1e6`, burada `t0` =
  `data/silver/uav_sead_silver.parquet`'in `source_id` başına min `timestamp`'ı. Bu tam olarak
  `run_ml6_events.py`'nin yaptığı şey — kopyalamak yerine ortak yardımcıya taşıyıp ikisi de
  kullanabilir.
- Positive kaynak: yalnız **TRAIN oturumlarındaki** range aralıkları.
- Negative kaynak: normal uçuş pencereleri. Anomalili oturumların **range-DIŞI** pencereleri
  bu fazda train-negative YAPILMAZ (sınır belirsizliği kontaminasyonu).
- Blind holdout (`final_holdout`: 40 anomalili + 36 normal) HİÇ OKUNMAZ (§5, §10-Gate A).

### ALFA
- Positive: fault onset SONRASI pencereler (aynı %50 kuralı). ALFA Silver'da label satır-bazlı;
  onset = ilk anomali-etiketli satır.
- Negative: normal uçuşlar + fault öncesi bölüm.
- Binary **tek** model; rudder/aileron_rudder gibi az-örnekli tipler için ayrı classifier YOK.
- ALFA `development-only` etiketi korunur (kör holdout ALFA'da yok).

### Sınıf dengesi
`class_weight="balanced"` esas. Eğitim süresi sorun olursa negatifler 10:1'e kadar seed'li
rastgele altörneklenebilir (manifest'e loglanır); pozitife dokunulmaz.

## 5. Split protokolü — MEVCUT MANİFESTİ YENİDEN KULLAN

- Manifest: `data/gold/ml_features/split_manifest.json`. SEAD için oturum-bazlı split
  ([src/ml/data/splits.py](../src/ml/data/splits.py) `session_of`, anomalili oturum
  normallerinin karantinası) ve `split_00`'ın `final_holdout`/`development_anomalous`
  ayrımı ZATEN var. **Yeni split üretme.**
- Aynı 5 development seed (`split_00..split_04`) kullanılır → ML-6/7 ile seed-bazında
  kıyaslanabilirlik şart.
- Pencereleme split'ten SONRA, uçuş bazında yapılır (satır bazlı split yok → overlap sızıntısı
  yapısal olarak imkânsız).

## 6. `temporal_boosting.py`

**Bağımlılık:** `lightgbm` kurulu DEĞİL. Adım olarak `pip install lightgbm`, sonra çözülen
sürümü `requirements.txt`'e **tam pinle** (torch/scipy'yi pinlediğimiz disiplin). Kurulmadan
kod yazma.

```python
LGBMClassifier(
    objective="binary",
    class_weight="balanced",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=seed,
)
```
- Hyperparameter search YOK. Model kötü çıkarsa Optuna ile kurtarma YOK (Gate B'ye bak).
- Early stopping validation **AUPRC** ile (val = train-split içinden, test'e bakmadan).
- Feature importance (gain) kaydedilir → hangi kanal/descriptor gruplarının taşıdığını raporla.
- Çıktı olasılığı **kalibre değildir** — yalnız sıralama/skor akışı olarak kullanılır (§7 CUSUM
  bunu logit + val-normal normalizasyonuyla ele alır).

## 7. `decision_layers.py` — üç karar katmanı

Hepsi aynı arayüz: `fit_calibrate(val_normal_score_streams, fa_budget_per_hour) -> policy`,
`apply(score_stream) -> alarm_onsets`. **Kalibrasyon YALNIZ validation normal uçuşları
üzerinde**; test'te politika donmuş.

### 7.1 Threshold-only
`alarm: p_t > τ`. τ, val-normal akışlarında FA/saat ≤ bütçe olacak en düşük eşik.

### 7.2 K-of-N
Mevcut mekanizma: son N pencerenin K'sı τ' üstünde. `(K, N, τ')` val-normal üzerinde bütçeye
kalibre edilir. [src/ml/evaluation/events.py](../src/ml/evaluation/events.py) `k_of_n_alarm` /
`persistent_alarm` implementasyonu KULLANILIR — yeniden yazma.

### 7.3 Causal CUSUM + bootstrap-ARL (yeni)
Somut reçete:
1. `z_t = logit(clip(p_t, 1e-6, 1-1e-6))`
2. Val-normal akışlarından `μ_N, σ_N`; `ẑ_t = (z_t - μ_N) / σ_N`
3. Artım: `inc_t = ẑ_t - k`, allowance `k = 0.5` (v1 sabiti; TARANMAZ)
4. `S_t = max(0, S_{t-1} + inc_t)`; alarm: `S_t > h`
5. **h kalibrasyonu (moving-block bootstrap):** val-normal `ẑ` akışlarından 60 s'lik bloklarla
   ≥ 200 sentetik normal saat üret; her aday h için FA/saat ölç; bütçeyi sağlayan en düşük h seç.
6. Alarm sonrası reset: `S=0` + 30 s refractory (aynı event'in çift sayımını önler).
7. CUSUM parametreleri (`μ_N, σ_N, k, h`) `artifacts/ml8a/cusum/` altına checksum'lu yazılır;
   prefix-invariance testi bu katman için de geçmeli.

## 8. Değerlendirme

### Adil kıyas matrisi (fazın kalbi) — 3×3 TAMAMI koşulur
Skor ve karar etkisi ancak böyle ayrışır:

| Skor kaynağı | threshold | K-of-N | CUSUM |
|---|---|---|---|
| LightGBM (yeni) | ✓ | ✓ | ✓ |
| IF-füzyon (mevcut artifact) | ✓ | ✓ | ✓ |
| LSTM-AE (mevcut artifact) | ✓ | ✓ | ✓ |

Mevcut model skorları `artifacts/models/<source>/` altındaki ML-6/7 bundle'larından **YÜKLENİR**
([src/ml/artifacts.py](../src/ml/artifacts.py) `load_modular_iforest_bundle`), yeniden
eğitilmez. IF/LSTM-AE skor akışları da aynı üç karar katmanından geçirilir — yoksa "skor mu
karar mı kazandı" sorusu cevapsız kalır.

### Metrik seti (seed başına, sonra mean±std)
```
window AUROC, window AUPRC (+ prevalence taban çizgisiyle birlikte — H7 kuralı),
event onset recall (ML-7 tanımı: yalnız event içinde BAŞLAYAN alarm; events.py event_metrics),
event precision, preexisting_alarm_events,
median / p90 detection delay,
false alarms / hour,
anomali ailesi bazında onset recall (SEAD: external_position, global_position, altitude,
  mechanical; ALFA: fault tipi bazında),
ALFA için resmi sequence-level Max/Avg Detection Time.
```

### Operasyonel bütçeler (ML-7 ile aynı)
```
critical: ≤ 2 FA/saat, hedef recall ≥ 0.30
advisory: ≤ 12 FA/saat, hedef recall ≥ 0.50
```
Point-adjust veya "event'in herhangi bir yerine dokundum" tipi metrik **KULLANILMAZ**.

## 9. Artifact kuralları (MLflow YOK)

Her koşu `artifacts/ml8a/<source>/<run_name>/` altına şunları checksum'lu `manifest.json` ile
paketler ([src/ml/artifacts.py](../src/ml/artifacts.py) deseni):
```
model.txt (LightGBM booster) / scaler.json / descriptor_schema_v1.json
cusum/  (μ_N, σ_N, k, h)   policy.json (kalibre τ/K/N/h + bütçe)
metrics.json (tüm metrikler, seed başına + mean±std)
manifest.json (dataset sürümü, split/seed id, şema hash'i, dosya SHA-256'ları)
```
Ayrı bir tracking server (MLflow vb.) KURULMAZ; tüm kayıt lokal JSON + checksum.

## 10. Faz kapıları

**Gate A — Veri ve leakage (HEPSİ geçmeli):**
```
test∩train oturum/sequence kesişimi = ∅
descriptor prefix-invariance testi geçiyor
CUSUM prefix-invariance testi geçiyor
scaler yalnız train'de fit
guard-band dışlaması uygulanmış (train pozitif/negatif sayıları loglanmış)
blind holdout hiç okunmamış (final_holdout uçuşlarının hiçbir feature/skor akışına girmediği
  assert'le doğrulanmış)
descriptor şema hash'i manifest'e loglanmış
```

**Gate B — Skor kalitesi (en az biri):**
```
SEAD window AUPRC, mevcut en iyi baseline'dan seed-std dikkate alınarak anlamlı yüksek
VEYA aynı FA/saat'te onset recall mevcut sistemden yüksek
```
Geçemezse: LightGBM'i kurtarmaya çalışma → ML-8C'ye (DevNet/Deep SAD) geç, bulguyu yaz.

**Gate C — Operasyonel fayda (en az biri):**
```
critical: recall ≥ 0.30 @ ≤ 2 FA/saat
VEYA advisory: recall ≥ 0.50 @ ≤ 12 FA/saat
```
Geçerse: feature/model/policy hash'leri dondurulur, **blind holdout açılması AYRI bir insan
kararıdır (otomatik açma YOK).** Geçmezse holdout kapalı kalır.

> **Gerçekçilik notu:** ML-7 dersi, ≤2 FA/saat'te onset recall'un yapısal olarak zor olduğuydu.
> Critical bütçenin yine reddedilmesi olası; advisory geçerse bu başarısızlık değil, "kritik
> alarm için skor kalitesi hâlâ sınırlayıcı" bulgusudur ve ML-8C'nin gerekçesidir.

## 11. Test listesi (pytest)

```
test_descriptor_prefix_invariance
test_descriptor_no_future_leak          # pencere sonrası veriyi değiştir → descriptor değişmemeli
test_guard_band_labeling                # sentetik range ile %0/%30/%60 kesişim senaryoları
test_split_no_session_overlap           # (mevcut splits testleriyle tutarlı)
test_cusum_prefix_invariance
test_cusum_fa_calibration               # sentetik normal akışta kalibre h'nin FA/saat ≤ bütçe verdiği
test_event_metrics_matches_ml7          # events.py event_metrics ile birebir aynı sonuç (kopya değil)
test_refractory_no_double_onset
```

## 12. Çalışma sırası

1. Bu dokümanı oku; `pip install lightgbm` + `requirements.txt`'e tam sürüm pinle.
2. `window_descriptors.py` + şema json + prefix/leak testleri. **Gate A'nın descriptor kısmı
   geçmeden devam etme.**
3. Label üretimi (guard-band + SEAD `load_ranges`/absolute_us yeniden kullanımı) +
   `test_guard_band_labeling`.
4. SEAD split_00 tek-seed smoke run: descriptor → LightGBM → window AUROC/AUPRC.
   Sanity: AUPRC prevalence taban çizgisinin belirgin üstünde mi? Değilse DUR, bulguyu yaz.
5. `decision_layers.py` (üçü de) + kalibrasyon + CUSUM testleri.
6. Tam matris: SEAD 5 seed × 3 skor × 3 karar katmanı. Lokal JSON manifest'e logla.
7. Aynı altyapıyla ALFA 5 seed (binary; guard-band; resmi Detection Time metrikleri dahil).
8. `notebooks/08_temporal_boosting.ipynb`: sonuç tabloları + feature-importance + aile-bazlı
   kırılım + Gate B/C değerlendirmesi. Notebook yalnız görselleştirme/rapor; hesap
   `scripts/run_ml8a_temporal_boosting.py`'de yaşar.
9. `docs/ML1_BULGULAR_VE_HATALAR.md`'ye "ML-8A sonuçları" bölümü — mevcut format: her madde
   "neyi denedik, ne çıktı, neden, ne yapılacak"; yeni hipotezlere **H15+** numarası ver.
10. Gate kararlarını (A/B/C, geçti/kaldı, gerekçe) `docs/decisions.md`'ye işle.

## 13. Yapılmayacaklar (tekrar)

Chronos/TSFM, GDN, Transformer, Optuna, şemaya yeni descriptor, MLflow, blind holdout'u açmak,
test sonucuna bakıp geriye dönük feature/eşik ayarı, point-adjust metrikleri, `bfill`.

## 14. Referans / citation kuralı

- **Taşıyıcı teori:** CUSUM'un yanlış-alarm kısıtı altında minimax gecikme optimalliği —
  Lorden (1971), Moustakides (1986). "Öğrenilmiş skor + bootstrap-ARL kalibrasyonu" bunun
  standart modern uygulanışıdır. Rapor bu klasik referanslara dayanır.
- **DeepLLR-CUSUM** (NeurIPS 2025 workshop, OpenReview `XXpt1NHd4B`): gerçek ve mekanizması
  planınkiyle örtüşür, ancak alanı Site Reliability Engineering / ağ verisi (CESNET), havacılık
  DEĞİL. Yalnız "yakın modern uygulama örneği" olarak dipnotta kullanılabilir; "havacılıkta
  kanıtlandı" iddiası kurulamaz.
- **AeroTSBoost:** tek kaynaklı ve doğrulanmadı → **rapora kanıt olarak KOYMA.** LightGBM'in
  tabular descriptor'lara uygulanması egzotik citation gerektirmeyen standart pratiktir;
  gerekçe budur, hayali bir makale değil.

## 15. Sonraki fazlar (bu fazda BAŞLAMA)

- **ML-8B:** UAV Attack top-k MIL (bag=uçuş, instance=pencere, `s_flight = mean(TopK(s_i))`).
- **ML-8C:** Family-holdout (örn. train: external+global position → test: altitude) + LightGBM
  görülmemiş ailede çökerse DevNet/Deep SAD kafası.
- **ML-9:** Chronos-2 / MOMENT zero-shot residual kanalı pilotu; dual-stream TCN / physics-graph
  ablation.

## 16. Bu planın ham taslaktan farkı (değişiklik kaydı)

1. **`src/ml/eval/` → mevcut `src/ml/evaluation/events.py`** genişletilir (yol çakışması giderildi).
2. **MLflow kaldırıldı** → mevcut checksum'lu JSON manifest deseni (`src/ml/artifacts.py`).
3. **Kanal listesi kaynağa göre ayrıştırıldı** ve repo feature tablolarıyla doğrulandı
   (ALFA'da baro/attitude residual yok; SEAD'de xtrack/alt_error yok).
4. **SEAD range→satır etiketleme** için mevcut `run_ml6_events.py::load_ranges` + `absolute_us`
   reconstruction'ı yeniden kullanma talimatı eklendi (yeni parser yazılmaz).
5. **lightgbm** yeni bağımlılık olarak işaretlendi (kurulu değil; kur + pinle).
6. **Citation kuralı** eklendi (§14): DeepLLR-CUSUM dipnot, AeroTSBoost yasak, taşıyıcı = klasik.
