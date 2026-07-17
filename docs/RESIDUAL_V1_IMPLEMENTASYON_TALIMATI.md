# RESIDUAL-V1 İmplementasyon Talimatı (Kodlama Ajanı İçin)

Referans tasarım: `RESIDUAL_V1_DENEY_TASARIMI.md` (bu talimatla birlikte repoya `docs/` altına konacak).
Hedef repo: `github.com/ColdVI/AnomalyDetection` (main).
Rol: Sen bir implementasyon ajanısın. Bu belgedeki görevleri SIRAYLA yaparsın. Her görevin kabul testi vardır; test geçmeden sonraki göreve geçemezsin. Tasarım kararlarını değiştiremezsin — belirsizlik görürsen `docs/residual_v1_questions.md` dosyasına soru yazıp o görevi atlayıp atlayamayacağını kontrol edersin (bağımlılık yoksa devam, varsa dur).

---

## 0. Genel kurallar (her görevde geçerli)

1. Yeni kod TEK pakette yaşar: repo kökünde `residual_v1/`. Mevcut `adsb/`, `src/`, `anomaly_core/` paketlerine dokunulmaz — İSTİSNA: `anomaly_core/sequential.py` içindeki `MultiChannelPageCUSUM` yeniden KULLANILIR (import edilir, kopyalanmaz).
2. Arşivdeki (`archive/2026-07-10_legacy_non_adsb_ml/`) kod READ-ONLY referanstır. ALFA/RflyMAD parser mantığı oradan OKUNARAK `residual_v1/ingest/` altına temiz yeniden yazılır; arşivden import YAPILMAZ.
3. Her çalıştırılabilir script `scripts/residual_v1_*.py` adlandırmasıyla; her config `configs/residual_v1_*.json`; her test `tests/test_residual_v1_*.py`. Mevcut pytest düzenine uyulur.
4. Tüm rastgelelik `seed` parametresiyle; varsayılan seed listesi `[11, 23, 37, 41, 53]`.
5. Çıktılar `artifacts/residual_v1/runs/<YYYYMMDD_gorevadi>/` altına; her run klasöründe `manifest.json` (git SHA, config hash SHA-256, girdi dosya hash'leri, seed). Var olan run klasörünün üstüne yazma girişimi → hata (fail-if-exists).
6. MLflow: experiment adı `residual_v1`; her koşuda config, manifest, metrikler ve grafikler artifact olarak loglanır.
7. YASAKLAR (tasarım AP-1..AP-7'nin kod karşılığı):
   - Tepki değişkeninin kendi geçmişini (lag'li y) feature olarak eklemek — feature şemasında `target_history` diye bir alan AÇILMAYACAK.
   - Silver katmanında herhangi bir interpolasyon/resample (`.interpolate`, `.resample().mean()` vb.) — lint kuralı Görev 1.4'te.
   - ADS-B, UAV-SEAD, sentetik enjeksiyon kodu çağırmak.
   - Test/holdout rolündeki uçuşları eşik seçimi, model seçimi veya hata analizinde kullanmak.
   - `point_adjust` benzeri değerlendirme gevşetmeleri.
8. Durdurma kuralları (STOP): Görev 3.5 görsel doğrulama ve Görev 5.4 S-3 kapısı "insan onayı" bekler — bu noktalarda özet rapor üretip DURURSUN, devam komutu gelmeden sonraki faza geçmezsin.

---

## 1. Faz A — Veri katmanı (Silver-v2)  [tasarım §3–4]

### Görev 1.1 — Paket iskeleti
`residual_v1/{__init__.py, ingest/, features/, models/, decision/, eval/, viz/}` oluştur. `residual_v1/schema.py` içinde:
```python
@dataclass(frozen=True)
class ChannelSpec:
    name: str            # ör. "imu_gyro_x"
    topic: str           # kaynak topic/dosya
    unit: str            # "rad_s", "m_s", "pwm", ...
    valid_min: float
    valid_max: float
    nominal_hz: float
    is_angle: bool = False   # wrap-aware fark için
```
ALFA ve RflyMAD kanal envanterleri (tasarım §3.1–3.2 tabloları) `residual_v1/ingest/alfa_channels.py` ve `rfly_channels.py` içinde `CHANNELS: tuple[ChannelSpec, ...]` sabitleri olarak kodlanır.
Kabul: `pytest tests/test_residual_v1_schema.py` — ChannelSpec doğrulamaları (min<max, hz>0), envanterlerde isim tekilliği.

### Görev 1.2 — ALFA ingest (doğal hızda, topic başına parquet)
`residual_v1/ingest/alfa.py`: processed-CSV kökünden uçuş başına, topic başına parquet yazar: `artifacts/residual_v1/silver/alfa/<flight_id>/<topic>.parquet`. Kurallar: timestamp'ler float saniyeye normalize (uçuş t0'ına göre); dt≤0 satırlar atılır ve sayısı `ingest_report.json`'a yazılır; İNTERPOLASYON YOK; açı kanalları radyana çevrilir ve (−π, π] aralığına sarılır; quaternion işaret devamlılığı düzeltilir (ardışık dot<0 → −q). `failure_status` topic'inden `events.json` üretilir: `{fault_class, onset_s, end_s}` (onset = ilk emisyon).
Kabul: `tests/test_residual_v1_alfa_ingest.py` — sentetik mini-CSV fixture ile: dt filtresi, wrap, quaternion düzeltmesi, onset çıkarımı birim test edilir.

### Görev 1.3 — RflyMAD-Real ingest
`residual_v1/ingest/rfly.py`: ULog → topic parquet (pyulog). `rfly_ctrl_lxl`'den interval truth; arşiv raporundaki 5 çelişkili uçuş ID'si `configs/residual_v1_rfly_exclusions.json`'a kodlanır ve dışlanır. `battery_status.voltage_v` bağlam kanalı olarak alınır (dedektör kanalı DEĞİL — şemada `role: "context"` alanıyla işaretle; `ChannelSpec`'e `role: Literal["response","command","context"]` alanı ekle, Görev 1.1'i güncelle).
Kabul: `tests/test_residual_v1_rfly_ingest.py` (sentetik ULog fixture veya mock'lanmış pyulog).

### Görev 1.4 — Hijyen profili + interpolasyon lint'i
`scripts/residual_v1_profile.py --dataset {alfa,rfly}`: uçuş başına JSON+HTML profil — kanal başına null oranı, dt histogramı, aralık ihlali sayısı, donmuş-sensör (`stale`) segment listesi (2 s değişimsizlik kuralı), toplam süre. Aralık ihlali >%1 olan uçuş `quarantine.json`'a düşer ve sonraki fazlarda otomatik dışlanır.
Ek: `tests/test_residual_v1_no_interpolation_lint.py` — `residual_v1/` kaynak ağacında `interpolate(`, `.resample(` ve `fillna(method=` desenlerini grep'leyen ve bulursa FAIL eden test (kaba ama etkili kilit).
Kabul: iki veri setinde profil koşusu tamamlanır; MLflow'a `phaseA_profile` run'ı olarak özet metrikler (uçuş sayısı, karantina sayısı) loglanır.

### Görev 1.5 — Split manifesti (oturum-bazlı)
`residual_v1/ingest/splits.py`: uçuşları oturum anahtarına (ALFA: kayıt günü; RflyMAD: metadata'daki test-session alanı, yoksa gün) grupla; oturum düzeyinde 70/15/15 development/test/holdout böl, 5 seed. Arıza sınıfı stratifikasyonu oturum düzeyinde (her sınıftan her bölmeye en az 1 oturum; n<8 sınıflar yalnız development'a — vaka analizi orada yapılacak). Çıktı: `artifacts/residual_v1/splits/<dataset>_seed<k>.json`, SHA-256'ları manifest'e.
Kabul: `tests/test_residual_v1_splits.py` — aynı oturunun iki bölmeye düşmediği, seed determinizmi, stratifikasyon kuralı.

---

## 2. Faz B — Faz segmentasyonu ve hizalama  [tasarım §4.1, §4.4]

### Görev 2.1 — Uçuş fazı segmentasyonu
`residual_v1/features/phases.py`: kural-bazlı `label_phases(flight) -> DataFrame[t, phase]`; fazlar `{ground, takeoff, cruise, maneuver, landing}`; eşikler `configs/residual_v1_phases.json`'da (ALFA: yer hızı<3 m/s & |climb|<0.3 → ground; |roll|>25° veya |roll_rate|>15°/s → maneuver; vb. — config'e yaz, koda gömme). Faz geçişlerinde ±1 s `phase_boundary=True` maskesi.
Kabul: sentetik trajektori fixture'ında beklenen faz dizisi; boundary maskesi genişliği.

### Görev 2.2 — Referans-saat hizalama
`residual_v1/features/align.py`: `align_to_clock(flight, clock_topic, tolerances) -> DataFrame` — hedef saat ALFA'da `nav_info` (≈20 Hz), RflyMAD'de `vehicle_attitude`; diğer kanallar `merge_asof(direction="backward", tolerance=kanal_hz'e_göre)`; her düşük hızlı kanala `<kanal>_staleness_ms` kolonu. Tolerans aşımı → NaN + staleness=inf (asla ileriye taşıma yok). `stale` (Görev 1.4) segmentlerinde ilgili kanal NaN'lanır.
Kabul: `tests/test_residual_v1_align.py` — backward yönü, tolerans, staleness hesabı, stale maskeleme.

---

## 3. Faz C — Residual kanalları + G0  [tasarım §4.2–4.3, §5.1]

### Görev 3.1 — Feature şeması (frozen)
`residual_v1/features/spec.py`: her residual kanalı için
```python
@dataclass(frozen=True)
class ResidualChannelSpec:
    name: str                     # "R1_aileron_roll_rate"
    command_inputs: tuple[str, ...]   # yalnız role∈{command,context}
    response: str                     # role=response, ASLA girdide değil
    horizon_s: float = 0.5            # hedef: gelecek 0.5 s ortalaması
    lag_summary: str = "tri4"         # son değer + 3 üçgen-ağırlıklı pencere (0–0.25, 0.25–0.5, 0.5–1.0 s)
```
ALFA R1–R6 ve RflyMAD Q1–Q4, tasarım §4.2–4.3'teki tanımlarla kodlanır. R3'te `g*tan(roll)/V` ve R4/Q3'te türev, `residual_v1/features/physics.py` yardımcılarında (türev = 0.5 s merkezi fark, uçlarda tek yönlü). Şemanın JSON dump'ının SHA-256'ı = `descriptor_schema_residual_v1` olarak her run manifest'ine.
Konstrüktör kuralı: `response` adı `command_inputs` içinde veya lag'li türevlerinde geçerse `ValueError` (AR-sızıntı kilidi, tasarım §2.3a).
Kabul: `tests/test_residual_v1_feature_spec.py` — AR-sızıntı kilidi testi dahil.

### Görev 3.2 — Feature matrisi üretimi
`residual_v1/features/build.py`: `build_xy(flight_aligned, spec, phases) -> (X, y, meta)`; X = komut lag-özetleri + [V, V², faz one-hot, V×son_komut]; y = response'un ileri 0.5 s ortalaması; `ground` fazı ve `phase_boundary` satırları atılır; NaN içeren satır atılır ve oranı meta'ya yazılır (>%20 ise uyarı logu). Eğitim maskeleri: normal uçuşların tamamı + arızalı uçuşlarda `t < onset − 10 s` (guard band).
Kabul: birim test — guard band, faz dışlama, horizon hesabı; ve "X kolonları arasında response türevi yok" şema denetimi.

### Görev 3.3 — G0 fizik kuralları
`residual_v1/models/g0_rules.py`: üç kural (tasarım §5.1) → kanal başına skor serisi; parametreler `configs/residual_v1_g0.json`. Çıktı formatı G1 ile aynı (aşağıdaki `ScoreFrame` sözleşmesi) ki karar katmanı ortak olsun:
`ScoreFrame = DataFrame[flight_id, t, channel, z]` (z = robust-z, scaler Görev 5.1'de).
Kabul: sentetik "komut var tepki yok" fixture'ında kural (i)'nin ateşlemesi.

### Görev 3.4 — Residual hesap koşusu
`scripts/residual_v1_build_features.py --dataset alfa --seed 11`: tüm development uçuşları için X/y üret, parquet'e yaz, MLflow'a satır sayıları + NaN oranları.

### Görev 3.5 — GÖRSEL DOĞRULAMA (STOP noktası, tasarım G3 checkpoint)
`scripts/residual_v1_sanity_plots.py`: (a) bir ALFA engine-fault uçuşunda R4 ham residual'ının (henüz modelsiz: y − komut-koşullu-medyan gibi kaba tahmin DEĞİL — burada yalnız y ve komutun zaman serisi + onset çizgisi) görselleştirilmesi; (b) en agresif manevralı 3 normal uçuşta R1 girdi/çıktı serileri. Çıktı: `artifacts/residual_v1/runs/<ts>_sanity/plots/*.png` + `SANITY_REPORT.md` (her grafiğe 2 cümle otomatik özet: onset sonrası airspeed türev işareti, manevra sırasında komut-tepki eşzamanlılığı).
**Burada DUR ve raporu sun.** (İnsan gözü tasarım Senaryo A/B'nin gerçek veride tuttuğunu onaylayacak.)

---

## 4. Faz D — G1/G2 modelleri  [tasarım §5.2–5.3]

### Görev 4.1 — G1 ridge
`residual_v1/models/g1_ridge.py`: kanal başına `Ridge(alpha)` — alpha, development-içi 5-fold (OTURUM-bazlı fold!) ile `{0.1,1,10,100}` ızgarasından seçilir (bu, eşik değil model hiperparametresi; development içinde serbest). Fit yalnız train maskesinde. Çıktı: residual `r = y − ŷ` → ScoreFrame (z'leme Görev 5.1'de). Model + katsayılar + kanal başına R² MLflow'a; katsayıların fiziksel işaret kontrolü (`aileron→roll_rate` kazancı pozitif mi vb.) `coeff_sanity.json`'a.
Kabul: `tests/test_residual_v1_g1.py` — sentetik doğrusal dinamikte katsayı geri-kazanımı (bilinen kazançla üretilen veride öğrenilen kazanç ±%10).

### Görev 4.2 — G2 LightGBM (aynı sözleşme)
`residual_v1/models/g2_lgbm.py`: aynı X/y, `LGBMRegressor` (max_depth≤6, n_estimators≤400, early stopping oturum-bazlı val ile). Karşılaştırma metriği: kanal başına test-DEĞİL, development-val residual varyans oranı `var(r_G2)/var(r_G1)`. Bu oran her headline kanalda ≥0.8 ise (yani <%20 iyileşme) G2 elenir ve Faz E'ye G1 ile gidilir — karar `g2_decision.json`'a otomatik yazılır (tasarım İ-4'ün testi).
Kabul: birim test — early stopping ve karar kuralı mantığı.

### Görev 4.3 — S-4 AR/sızıntı ablasyonu
`scripts/residual_v1_s4_ablation.py`: seçilen modeli komut girdileri ÇIKARILMIŞ (yalnız bağlam) versiyonuyla yeniden eğit; `var(r_sakat)/var(r_tam)` < 1.15 çıkan kanal FLAGGED (model komutu kullanmıyor → sızıntı şüphesi) → `flags.json`. FLAGGED kanal karar katmanına giremez.

---

## 5. Faz E — Skorlama, sanity kapıları, CUSUM  [tasarım §6–7]

### Görev 5.1 — Robust z + scaler sözleşmesi
`residual_v1/decision/scaling.py`: kanal başına median/MAD train-normal'den; MAD=0 → kanal dışlanır ve manifest'e `excluded_channels` yazılır; z clip=8 (CUSUM z_clip ile uyumlu). Scaler parametreleri run artifact'ı.

### Görev 5.2 — Sanity kapıları S-1/S-2/S-3
`residual_v1/eval/sanity_gates.py`:
- S-1: kanal z'lerinin uçuş-içi |z| ortalaması vs aynı pencerenin ham-girdi normu ‖x‖ Spearman ρ; ρ≥0.5 → FLAG.
- S-2: yalnız G2/G3 için (öğrenilmiş ağaç/ağ); rastgele-init eşdeğeriyle sıralama ρ; ≥0.7 → FLAG. (G1 ridge için S-2 atlanır, katsayı-sanity zaten var.)
- S-3: development etiketli olaylarda kanal başına KS testi — |z| dağılımı `[onset, onset+15 s]` vs `[onset−60, onset−10 s]`; headline sınıf başına en az bir kanal p<0.01 değilse **STOP**: `S3_FAILURE_REPORT.md` üret (kanal başına KS istatistiği + en kötü 3 olayın grafiği) ve dur. Eşik/kalibrasyon koduna geçiş S-3 PASS koşuluna programatik olarak bağlanır (`raise GateError`).

### Görev 5.3 — CUSUM (mevcut çekirdeğin yeniden kullanımı)
`residual_v1/decision/cusum.py`: `anomaly_core.sequential.MultiChannelPageCUSUM`'ı sar; k=1.0, İKİ YÖNLÜ (mevcut sınıf tek yönlüyse −z ile ikinci geçiş; koda bak, gerekiyorsa sarmalayıcıda çöz, çekirdeği değiştirme). Refractory 60 s. Alarm kaydı: `{flight_id, t_alarm, channel_contributions}`.

### Görev 5.4 — Blok-bootstrap eşik kalibrasyonu — **STOP / NO-GO**
STOP — bkz. docs/RESIDUAL_V1_KALIBRASYON_NOGO_RAPORU.md.
`residual_v1/decision/calibrate.py`: development-NORMAL uçuşlarından 60 s bloklu bootstrap (B=500) ile kanal başına h çöz: hedef, kanal FA katkısı = 0.5/(aktif kanal sayısı) alarm/uçuş-saati. Çözüm monoton arama (h ↑ → FA ↓). Çıktı: `thresholds_frozen.json` + bootstrap FA dağılım grafiği. Bu dosya yazıldıktan sonra değiştirilemez (fail-if-exists) — test değerlendirmesi yalnız bunu okur.
Kabul: `tests/test_residual_v1_calibration.py` — monotonluk, blok örnekleme, determinizm (seed'li).

---

## 6. Faz F — Değerlendirme ve hata analizi  [tasarım §9]

### Görev 6.1 — Olay-bazlı değerlendirme
`residual_v1/eval/events.py`: olay başına `detected (t_alarm ∈ [onset, min(end, onset+60 s)])`, `delay = t_alarm − onset`; uçuş-saati başına FA (normal uçuşlar + arızalı uçuşların onset-öncesi guard-dışı kısmı). Sınıf başına: olay-recall, medyan/p90 gecikme, bootstrap (olay-düzeyi, B=2000) %95 CI. n<8 sınıflar otomatik olarak `case_studies/` klasörüne yönlenir (sayı tablosuna girmez).
`scripts/residual_v1_evaluate.py --split test --seed all`: 5 seed × frozen eşik; MLflow'a tam tablo.

### Görev 6.2 — Zorunlu hata analizi üretimi
`residual_v1/eval/postmortem.py`: kaçırılan HER olay için 4-panelli PNG (komut / tepki / kanal z'leri / CUSUM + eşik + onset) ve `miss_taxonomy.csv`'ye otomatik ön-sınıflama: onset±15 s'de max|z|<2 → "sinyal-yok"; max|z|≥3 ama CUSUM<h → "birikim-yavaş"; z yüksek ama FLAGGED kanalda → "maskeleme"; onset±5 s'de veri boşluğu/stale → "etiket-veri-şüpheli". (Nihai etiketleme insan işi; otomatik atama öneri kolonudur.)

### Görev 6.3 — RflyMAD zayıf-süpervizyon kolu (yalnız Faz F'e kadar her şey PASS ise)
`residual_v1/models/g_sad.py`: residual-z 10 s pencereleri (stride 1 s) üzerinde Deep SAD-lite — 2 katmanlı MLP (64,32), merkez c train-normal ortalamasından; kayıp: normal pencerede ‖φ(x)−c‖², etiketli arıza penceresinde 1/‖φ(x)−c‖² (η=1). Skor → aynı z'leme → aynı CUSUM → AYNI frozen-bütçe prosedürüyle (development'ta yeniden kalibre edilen KENDİ eşiği; test'e bir kez). Önceden kayıtlı karar kuralı: medyan gecikme iyileşmesi <%20 VEYA FA bütçe aşımı → unsupervised kazanır, `g_sad_decision.json`.

### Görev 6.4 — Kör holdout (yalnız insan onayıyla)
`scripts/residual_v1_holdout.py`: `--confirm-open` bayrağı olmadan çalışmaz; çalışınca holdout split'ini bir kez skorlar, sonucu ayrı MLflow run'ına yazar ve `HOLDOUT_OPENED.flag` bırakır (ikinci çalıştırma hata verir).

---

## 7. Rapor ve teslim

`scripts/residual_v1_final_report.py`: tüm run manifest'lerinden `docs/residual_v1_sonuc_raporu.md` derler — başarı sözleşmesi tablosu (hedef vs ölçülen, CI'larla), sanity kapı durumları, İ-1..İ-4 iddialarının tek tek doğrulanma/yanlışlanma durumu, miss taksonomisi dağılımı, kanal-katkı örnek alarm demoları. Sunum için ayrıca 3 "hero" grafiği: Senaryo-A gerçekleşmesi (engine olayında R4+CUSUM), Senaryo-B gerçekleşmesi (agresif manevrada sakin R1), miss-taksonomi pastası.

## 8. Görev sırası özeti ve STOP noktaları

1.1 → … → **3.5 STOP** → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 (**S-3 FAIL ise STOP**) → 5.3 → **5.4 STOP / NO-GO**. Mevcut durumda Faz F'ye geçilmez; yeniden açma koşulları NO-GO raporundadır.

Definition of done: tüm kabul testleri yeşil; yeterli maruziyet kapısı geçilirse `thresholds_frozen.json` tek kez yazılmış; İ-1..İ-4 durumu raporda; hiçbir yasak desende (Görev 1.4 lint + bu belgenin §0.7'si) ihlal yok. Mevcut tur bu tanıma ulaşmadı; NO-GO raporunda kayıtlıdır.
