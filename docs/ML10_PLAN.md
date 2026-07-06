# ML-10 Planı — Zero-Shot Foundation-Model Forecast-Residual Pilotu (SEAD altitude + mechanical)

> Bu doküman bir coding-agent (Codex / Claude Code) talimatıdır. Repo: `ColdVI/AnomalyDetection`.
> `docs/AGENTS.md`, `docs/decisions.md` ve `docs/ML8_PLAN.md`/`docs/ML9_PLAN.md` kuralları geçerlidir
> (Gate disiplini, checksum'lu artifact deseni, kör holdout'a hiç dokunmama). ML-9'un planlanmış
> devamıdır (`docs/ML9_PLAN.md` §11, ADR-009: "Gate B kalırsa forecast-residual/foundation-model
> pilotu"). ML-9 Gate B/C kaldı (H19-H21) — bu faz o taahhüdün karşılığıdır. Yeni model mimarisi
> mevcut modüler IF + karar katmanı iskeletine EK bir skor kaynağı olarak eklenir, iskelet değişmez.

## 0. Bağlam ve deney sorusu

ML-9, SEAD'in zayıf `Position.Z` (altitude_anomaly) ve `Actuator Outputs+Controls` (mechanical_fault)
kategorilerine kategori-eşleşmeli feature ekledi. Yön doğruydu ama dondurulmuş büyüklük+kararlılık
şartını geçemedi: Position.Z'de dikey modül +0.021 recall (4/5 seed), Actuator Outputs+Controls'te
motor-simetri modülü +0.024 (2/5 seed). Fusion düzeyinde de FA bütçesi karşılanmadı (en iyi: 0.222
recall/25.83 FA-saat).

Ortak neden: mevcut TÜM dedektörler (IF-tabanlı modüller) **popülasyon-düzeyi sabit bir normal
profiline** göre skorluyor — train-normal üzerinde fit edilmiş bir Isolation Forest'in "bu değer
popülasyona göre ne kadar tuhaf" sorusu. SEAD'in normal sınıfı heterojen (398 normal uçuş bir avuç
oturuma dağılıyor — literatür notu, `docs/ML1_BULGULAR_VE_HATALAR.md`); popülasyon eşiği "biraz
farklı ama makul" ile "gerçekten anormal" arasını heterojen normalde ayırt etmekte zorlanıyor.

**Deney sorusu:** Zero-shot, önceden eğitilmiş bir zaman-serisi foundation modeli (Chronos), HER
UÇUŞUN KENDİ NEDENSEL GEÇMİŞİNE bakarak ("bu uçuşun buraya kadarki seyrine göre şimdiki değer ne
kadar şaşırtıcı") bir sonraki değeri olasılıksal tahmin edip forecast-residual skoru üretirse,
popülasyon eşiğinin kaçırdığı onset'leri yakalar mı?

**Bu, reddedilen "session-koşullu normallik modeli" fikri DEĞİLDİR** (bkz. proje ilkesi: normal
sınıfı homojenleştirme önerilmez, `feedback-anomaly-detection-principles` ilkesi). Session/context
kimliğine göre AYRI modeller kurulmuyor; HER uçuş için AYNI, TEK, genel, önceden-eğitilmiş model
çalışıyor. Koşullandırma yalnız o anki uçuşun kendi (zaten nedensel olarak mevcut) geçmiş penceresine
— bu, yeni/görülmemiş bir uçuşta da anında kullanılabilir (hiçbir "önce bu oturumu/uçuşu tanı" ön
koşulu yok, sadece havada birkaç saniye/dakika geçmiş olması yeterli).

## 1. Doğrulanmış ön-koşullar (bugün kontrol edildi, varsayım değil)

**Bu ortamda (Windows, Python 3.14.6, torch 2.12.1+cpu, GPU YOK) iki foundation-model adayı
gerçek `pip install --dry-run` ile test edildi:**

- **MOMENT (`momentfm`) KULLANILAMAZ.** `pip install --dry-run momentfm` derleme aşamasında
  gerçek bir hatayla başarısız oldu: `AttributeError: module 'pkgutil' has no attribute
  'ImpImporter'`. Sebep: momentfm'in eski bir `numpy` pinni, Python 3.12+'ta kaldırılan
  `pkgutil.ImpImporter`'a bağımlı `setuptools`/`pkg_resources` üzerinden kaynak derlemesi
  gerektiriyor ve bu ortamda derlenemiyor. Bu spekülatif bir risk değil, tekrar edilebilir,
  gerçek bir kurulum hatasıdır. **MOMENT bu fazda DENENMEYECEK** (ayrı bir Python 3.10/3.11
  sanal ortamı kurmak teknik olarak mümkün olsa da, bu proje tek `.venv` kullanıyor; ek ortam
  yönetimi kapsam dışı ve orantısız).
- **Chronos (`chronos-forecasting`) KULLANILABİLİR.** `pip install --dry-run chronos-forecasting`
  tam bir bağımlılık çözümü üretti: `chronos-forecasting-2.3.1`, `transformers-5.13.0`,
  `torch<3,>=2.2` (zaten kurulu `torch==2.12.1+cpu` ile uyumlu) — kurulum gerçekten çalışıyor.
- **Gerçek API (resmi GitHub README'sinden doğrulandı, uydurulmadı):**
  ```python
  from chronos import BaseChronosPipeline
  import torch
  pipeline = BaseChronosPipeline.from_pretrained(
      "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
  quantiles, mean = pipeline.predict_quantiles(
      context=torch.tensor(series), prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
  # quantiles shape: [batch, prediction_length, len(quantile_levels)]
  ```

**Codex'in İMPLEMENTASYONDAN ÖNCE ayrıca doğrulaması gerekenler (aşağıdakiler ben tarafımdan
doğrulanmadı — varsayım olarak kullanılmayacak, gerçek kontrolle kilitlenecek):**

1. En küçük/en hızlı Chronos varyantının (`amazon/chronos-t5-tiny` veya bir bolt varyantı — HF'de
   gerçek adı teyit edilecek) bu paket sürümüyle gerçekten yüklenip CPU'da makul sürede (tek
   pencere tahmini ~saniyenin çok altında) çalıştığı küçük bir gerçek yükleme+tahmin denemesiyle
   kanıtlanmalı. Büyük/large varyantlar bu ortamda (CPU-only, GPU yok) denenmeyecek.
2. Altitude/dikey kanal seçimi: `local_alt_m`, `baro_alt_m` ve GPS-türetilmiş irtifa arasından,
   development uçuşlarında EN YÜKSEK doluluk oranına sahip olan seçilmeli. `docs/ML_ORNEK_INPUT_
   OUTPUT.md` Örnek 3'te `baro_alt_m`'in yalnızca ~%7-14 uçuşta dolu olduğu zaten bilinen bir bulgu
   — muhtemelen `local_alt_m` daha iyi kapsar ama bu iddia edilmeden gerçek veride ÖLÇÜLMELİ.
3. `actuator_output_std` (ML-9'da zaten üretilmiş, `uav_attack_features.py`) mechanical kanalı
   için doğrudan kullanılabilir — yeni bir kanal keşfi gerekmiyor.

## 2. Kapsam / kapsam dışı

**Kapsamda:** UAV-SEAD (mevcut 611 uçuş, frozen `split_manifest.json`). Yalnız İKİ hedef kanal:
dikey irtifa (Position.Z için) ve `actuator_output_std` (Actuator Outputs+Controls için). Zero-shot
kullanım (fine-tuning YOK). Forecast-residual skorunun mevcut modüler mimariye (fusion + threshold/
K-of-N/CUSUM karar katmanları) yeni bir "score_source" olarak eklenmesi.

**Kapsam DIŞI:** MOMENT (yukarıda gerekçeli dışlandı), fine-tuning/eğitim (herhangi bir gradyan
güncellemesi), ALFA/UAV Attack (bu fazın hedefi SEAD), blind holdout'u açmak, Battery/Vibration/
Magnetometer/Actuator Thrust/Velocity alt-tipleri (n=1-3, `docs/ML_YETERSIZLIKLER_KAYDI.md`'de
kayıtlı), multivariate/covariate-informed Chronos-2 kullanımı (univariate `ChronosPipeline` yeterli,
gereksiz karmaşıklık katmaz), yeni bir Isolation-Forest-dışı karar mimarisi (`decision_layers.py`
aynen kullanılır).

## 3. Fizibilite kontrol noktası — ZORUNLU İLK ADIM (Gate A'dan ÖNCE)

Foundation-model CPU çıkarımı mevcut sklearn IsolationForest'ten çok daha yavaştır. Tam development
setine geçmeden önce KÜÇÜK bir gerçek zaman denemesi yapılmalı:

1. 5-10 gerçek development uçuşunda, hedef kanal(lar) için causal rolling forecast (bkz. §4)
   çalıştırılıp uçuş başına gerçek duvar-saati ölçülür.
2. Karar kuralı (ÖNCEDEN sabit, sonuca göre değişmez):
   - Tüm development setinin (~480 uçuş, TEK geçiş — skor split'ten bağımsız, bkz. §4 not) tahmini
     toplam süresi **3 saatin altındaysa**: tam sette devam.
   - 3-8 saat arası: karar adımını (decision stride) mevcut 1 sn yerine 5 sn'ye seyreltip yeniden
     ölç (`decision_layers.py` zaten `stride_seconds` parametresi alıyor — bu ücretsiz bir
     geriye-uyumlu seçenek).
   - 8 saati aşıyorsa: development setinin sabit, önceden belirlenmiş bir alt kümesine (Position.Z
     ve Actuator Outputs+Controls etiketli TÜM uçuşlar + eşit sayıda rastgele normal, n belirtilerek
     raporlanır) daraltılır; bu daraltma açıkça "hesaplama bütçesi kısıtı" olarak kayıt altına
     alınır, sonucu iyileştirmek için yapılan bir seçim DEĞİLDİR.
3. Sonuç (ölçülen süre, alınan karar) `artifacts/ml10/uav_sead/feasibility_check.json`'a yazılır
   ve H22+ olarak dokümante edilir.

## 4. Skor spesifikasyonu — causal forecast-residual

Yeni bir precompute adımı (`scripts/build_ml10_forecast_residual.py` veya `build_features.py`
içinde yeni bir fonksiyon): her development uçuşu için, HER NEDENSEL karar noktasında (mevcut 1 sn
karar stride'ı, §3 sonucuna göre değişebilir):

```
context = kanal[t - context_window : t]   # yalnız GEÇMİŞ, o ana kadar gözlenmiş
quantiles, mean = pipeline.predict_quantiles(context, prediction_length=1,
                                              quantile_levels=[0.1, 0.5, 0.9])
actual = kanal[t]
# quantile bandının dışına ne kadar tastigi (0 = bant icinde, normalize edilmis)
residual = max(0, actual - q90, q10 - actual) / (q90 - q10 + eps)
```

Bu skor **split'ten (train/val/test) tamamen bağımsızdır** — Chronos hiçbir SEAD verisiyle fit
edilmiyor (zero-shot). Bu yüzden **BİR KEZ** hesaplanıp mevcut gold feature tablosuna yeni kolon(lar)
olarak yazılabilir (`chronos_alt_residual`, `chronos_actuator_std_residual`) — 5 seed için 5 kez
yeniden hesaplamaya GEREK YOK; yalnız karar katmanı kalibrasyonu (threshold/K-of-N/CUSUM eşikleri)
her seed'in kendi normal-val'inde ayrı fit edilir (mevcut desenle aynı). Bu, §3'teki bütçe hesabını
da fiilen 5'e böler.

**Nedensellik (Gate A kritik şart):** `context_window` yalnız `t`'den ÖNCEKİ gözlenmiş satırları
içerir; gelecekteki hiçbir satır skor hesaplamasına giremez. `test_ekf_alt_innov_no_future_leak`
ile AYNI desende bir test yazılmalı: gelecek satırlara aşırı değer eklenip geçmiş satırların
skorunun DEĞİŞMEDİĞİ kanıtlanmalı.

## 5. Modüler entegrasyon

Yeni skor(lar), `run_ml9_category_evaluation.py`'deki `_score_modules`/`_empirical_probability`
deseniyle AYNI şekilde normal-val'e göre empirik olasılığa çevrilip mevcut `existing_fusion`/
`ml9_fusion` ile aynı seviyede yeni bir `score_source` (`chronos_dikey`, `chronos_motor`,
`ml10_fusion = max(existing_fusion, chronos_dikey, chronos_motor)`) olarak eklenir. Bu mantık
`run_ml9_category_evaluation.py`'den KOPYALANMAZ — ya oradan import edilir ya da ortak bir
`src/ml/evaluation/score_fusion.py` yardımcı modülüne çıkarılıp HER İKİ script de (ML-9 ve ML-10)
oradan import eder (mekanizma Codex'e ait, ama kopya-yapıştır YASAK — ML-9'un
`test_ml9_decision_layers_reused_not_reimplemented` testine benzer bir identity-check testiyle
kanıtlanmalı, bkz. §8).

`modular_iforest.py`'ye yeni bir IsolationForest modülü EKLENMEZ — Chronos skoru zaten tek-boyutlu
ve kalibre edilmiş bir "şaşırtıcılık" ölçüsüdür, ayrıca bir IsolationForest'e sarmak gereksiz
karmaşıklık katar.

## 6. Split / karar katmanı / artifact — HİÇBİRİ YENİDEN YAZILMAZ

ML-9 ile birebir aynı disiplin: mevcut `split_manifest.json` (611 uçuş, 5 seed, 131 kör holdout)
aynen kullanılır, yeni split üretilmez; `decision_layers.py` değiştirilmeden import edilir;
artifact `artifacts/ml10/uav_sead/<run_name>/` altına, checksum'lı manifest ile. MLflow YOK,
`src/ml/eval/` YOK (ML-8A/9'daki aynı iki düzeltme burada da geçerli).

## 7. Gate'ler

**Gate A (hepsi geçmeli):** forecast-residual skoru nedensel (§4'teki future-leak testi geçmeli);
kör 131-uçuş holdout hiç okunmadı; Chronos hiçbir SEAD verisiyle fine-tune edilmedi (kod
denetiminde eğitim/gradyan adımı YOK — `test_chronos_zero_shot_no_training_step`); §3 fizibilite
kontrolü tamamlandı ve kararı kayıtlı.

**Gate B (en az biri — ML-9'un KENDİ en iyi adayına göre, eski/daha zayıf pooled/kontrol-cevabı
baseline'ına göre DEĞİL, çünkü asıl soru "şu ana kadar bulduğumuz en iyi şeyi geçiyor mu"):**
(a) Position.Z'de `chronos_dikey`, mevcut EN İYİ aday `dikey_tutarlilik`'e (CUSUM/advisory recall
0.096, `artifacts/ml9/uav_sead/full_matrix/category_metrics.csv`) göre aynı policy/bütçede
ortalama recall kazancı >=0.05 VE >=3/5 seed pozitif; VEYA (b) Actuator Outputs+Controls'te
`chronos_motor`, mevcut EN İYİ aday `motor_simetrisi`'ne (0.205) göre aynı kural. Geçemezse:
rescue yok, bulguyu H22+ olarak yaz; bu iki kategori için mevcut mimarinin mevcut veriyle
ulaşılabilir tavana yaklaştığı dürüst bir sınır olarak kabul edilir.

**Gate C:** aynı kritik (≤2 FA/saat, ≥0.30 recall) / advisory (≤12 FA/saat, ≥0.50 recall) bütçeleri,
BASE_MODULES+chronos füzyonu (`ml10_fusion`) üzerinde — ML-8A/9 ile birebir aynı sayısal hedefler
(karşılaştırılabilirlik için hedef değiştirilmez, kolaylaştırılmaz). Geçmezse holdout kapalı kalır.

## 8. Test listesi

```
test_chronos_forecast_residual_no_future_leak       # causal, gelecek sizintisi yok
test_chronos_zero_shot_no_training_step             # egitim/gradyan adimi olmadigini dogrular
test_ml10_decision_layers_reused_not_reimplemented   # decision_layers.py'den import, kopya yok
test_ml10_score_fusion_not_duplicated                # ML-9 ile ortak fusion yardimcisinin tekilligi
```

## 9. Çalışma sırası

1. §1'deki üç Codex-doğrulaması (model boyutu/hız, kanal doluluk oranı, aktarım) — sonuç
   kaydedilmeden ilerlenmez.
2. §3 fizibilite kontrol noktası — karar kaydedilmeden tam sete geçilmez.
3. `chronos-forecasting` `requirements.txt`'e pinlenir (`==2.3.1`, bu oturumda dry-run ile
   doğrulandı).
4. Forecast-residual precompute (§4) — gold feature tablosuna yeni kolon(lar).
5. Skor-füzyon entegrasyonu (§5).
6. Gate A/B/C değerlendirmesi; `docs/ML1_BULGULAR_VE_HATALAR.md`'ye "ML-10 sonuçları" (H22+).
7. `docs/decisions.md`'ye ADR-010.
8. `docs/ML_YETERSIZLIKLER_KAYDI.md`'yi güncelle (sonuç ne olursa olsun — geçerse "kapandı" olarak
   işaretlenir, kalırsa yeni bulgu eklenir).

## 10. Yapılmayacaklar

MOMENT (gerekçeli dışlandı), fine-tuning, ALFA/UAV Attack, Chronos-2/multivariate/covariate-
informed kullanım, yeni IsolationForest modülü, blind holdout açma, nadir alt-tipleri modelleme,
sonucu görüp fizibilite/kanal seçimini veya Gate eşiklerini geriye dönük değiştirme.

## 11. Sonraki fazlar (bu fazda BAŞLAMA)

- Bu pilot da geçemezse: mevcut modüler IF + karar katmanı mimarisinin bu iki kategori için mevcut
  veriyle ulaşılabilir tavana yaklaştığı sonucu dürüst bir sınır olarak kabul edilip kapatılabilir
  (istisna: SEAD çok-sınıflı 8 uçuşluk gerçek/mütevazı veri boşluğu, `docs/ML_YETERSIZLIKLER_
  KAYDI.md` §A.3 — bu ayrı, veri-tabanlı bir iş kalemi, model işi değil).
- Kafka/Docker altyapı fazı (kullanıcının ayrı planladığı iş, ML-N sayaçlarından bağımsız).
