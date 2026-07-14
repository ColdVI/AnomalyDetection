# Claude handoff — ADS-B contextual anomaly detection

Tarih: 2026-07-14  
Repo: `ColdVI/AnomalyDetection`  
Aktif dal: `main`  
Aktif aday: `contextual_physics_v1`

## Bu belgenin amacı

Bu belge, kullanıcı ile Codex arasındaki son konuşmayı, tamamlanan ADS-B model eğitimini,
Codex'in teknik yorumlarını ve açık bilimsel kararları Claude'a aktarır. Claude'dan beklenen,
mevcut kanıtları bağımsız biçimde eleştirmesi ve özellikle anomaly validation'ın normal
confidence/calibration ile nasıl ilişkilendirilmesi gerektiğini değerlendirmesidir.

## Kullanıcının soruları ve yönlendirmeleri

Kullanıcı sırasıyla şunları istedi veya sorguladı:

1. Yeni contextual model için gerçek bir eğitim yapılması.
2. Eğitimin sonuçlarının çıkarılması ve overall bir rapor hazırlanması.
3. Anomaly detection sırasında sistemin neye baktığının, uçuşun kendi içindeki zaman serisini
   kullanıp kullanmadığının ve model çıktısının matematiksel anlamının açıklanması.
4. Kısaltmaların önce uzun halleriyle verilmesi ve yorumların sayılarla desteklenmesi.
5. Son olarak şu araştırma sorusu:

> Anomaly örneklerini validation'a sokmak, modelin normal hakkındaki confidence-score tahmininde
> normal kümelerini daha fazla daraltmamızı sağlar mı?

## Değişmez proje sınırları

- `archive/` salt-okunur; oradan kod kopyalanmaz veya import edilmez.
- Sentetik anomaly hiçbir train/fit/normal calibration aşamasına giremez.
- Sonuç görüldükten sonra aynı run/config içinde threshold veya hyperparameter ayarı yapılamaz.
- MAD değeri sıfır olan kanal yapay floor ile kurtarılmaz; dışlanır.
- Satır, pencere, event, uçuş ve scoreable uçuş-saati metrikleri birbirine karıştırılmaz.
- Sentetik detection sonucu doğal alarm burden ile birlikte yorumlanır.
- Rehearsal sonucu seçime geri beslenmez.
- Üç dosyalık kör holdout havuzu ayrı unseal onayı olmadan açılmaz.
- Eski Step-7 FAIL sonucu yeni aday tarafından geriye dönük değiştirilmez.

## Önce kısaltmalar

- **Automatic Dependent Surveillance–Broadcast (ADS-B):** Uçağın durum/konum yayın sistemi.
- **Long Short-Term Memory (LSTM):** Zamansal bağımlılıkları taşıyan sinir ağı.
- **Neural Network (NN):** Sinir ağı.
- **Median Absolute Deviation (MAD):** Uç değerlere dayanıklı medyan tabanlı ölçek.
- **Negative Log-Likelihood (NLL):** Olasılıksal tahmin uyumsuzluğu kaybı.
- **Area Under the Receiver Operating Characteristic Curve (AUROC):** Threshold'dan bağımsız
  pozitif-negatif sıralama metriği.
- **Area Under the Precision–Recall Curve (AUPRC):** Precision-recall eğrisinin alanı.
- **Cumulative Sum (CUSUM):** Küçük ve sürekli sapmaları zamanla biriktiren yöntem.
- **Navigation Integrity Category (NIC), Navigation Accuracy Category for Position (NACp),
  Source Integrity Level (SIL):** ADS-B bütünlük/doğruluk beyan alanları.

## Tamamlanan eğitim

Eğitim koşusu:

```text
artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1
```

Dondurulmuş config:

```text
configs/adsb_contextual_physics_v1_train.json
```

Sonuçlar:

- Durum: `trained_not_thresholded`
- Fit rolündeki 149.462 uçuştan deterministik %2 seçim: **2.929 uçuş**
- Seçili satır: **1.267.625**
- Epoch başına pencere: **1.180.160**
- Epoch başına batch: **2.417**
- Epoch: **5**
- Toplam süre: **1.702,158 saniye**
- Ayrı natural-calibration diagnostic: **770 uçuş / 332.510 pencere**
- Sentetik train satırı: **0**
- Sentetik calibration satırı: **0**
- Model parametresi: **9.546**, tamamı sonlu; strict checkpoint reload PASS
- Artefakt checksum doğrulaması: **5/5 PASS**

Weighted Gaussian NLL:

| Epoch | NLL |
|---:|---:|
| 1 | 0,795375 |
| 2 | 0,738245 |
| 3 | 0,723818 |
| 4 | 0,714755 |
| 5 | 0,708696 |

İlk-son epoch göreli düşüşü yaklaşık **%10,90**. Bu, optimizer'ın fit verisinde tahmini
iyileştirdiğini gösterir; anomaly recall kanıtı değildir.

## Time-series mekanizması

Bu model gerçek bir uçuş-içi next-step time-series modelidir:

- Aynı `flight_id` içindeki önceki **12 satır**, bir sonraki satırı tahmin etmek için kullanılır.
- Pencere başka bir uçuşa geçmez.
- Zaman ters/tekrarlıysa veya mesaj gap'i 60 saniyeyi aşarsa pencere üretilmez.
- Skorlanan satırın residual'ı giriş geçmişinde bulunmaz.
- Flight phase, mevcut satırdaki vertical-rate ile değil yalnız önceki üç satırın lagged median'ı
  ile çıkarılır; böylece anomaly kendi calibration bağlamını değiştiremez.
- Model uçuş sırasında online yeniden eğitilmez. Global normal davranış ile o uçuşun yakın geçmişini
  birlikte kullanır.

Kritik yorum: uzun süreli anomaly modelin 12 satırlık geçmişini zamanla kirletebilir. Anlık
surprise başlangıçta yüksekken model girdisi anomaly rejimine girdikçe düşebilir. Bu yüzden spike,
persistence ve accumulation/CUSUM kararları ayrı tutulmalıdır.

## Fiziksel residual kanalları

Aktif beş kanal:

```text
vertical_rate_residual
speed_residual
heading_residual
east_velocity_residual
north_velocity_residual
```

Temel anlamları:

```text
vertical-rate residual
  = bildirilen dikey hız - delta altitude / delta time

speed residual
  = bildirilen yer hızı - ardışık konumdan türetilen hız

heading residual
  = bildirilen track - ardışık konumun dairesel bearing'i

east/north velocity residual
  = bildirilen hız vektörü - konum geçişinden türetilen hız vektörü
```

`altitude_source_residual`, barometrik/geometrik irtifa farkının zaman türevidir. Fit-normal MAD
değeri tam sıfır çıktığı için floor verilmeden dışlandı.

## Model girdisi

Toplam 19 input feature:

- beş scaled residual;
- `log(1 + delta-time)`;
- `sin(track)` ve `cos(track)`;
- `ground/climb/level/descent/unknown` phase one-hot alanları;
- cadence bucket alanları;
- availability mask.

Track'in 359 derece ile 1 derece arasındaki dairesel yakınlığı sin/cos temsiliyle korunur.

## Matematik

Her residual fit-normal medyan ve MAD ile ölçeklenir:

```text
z[c,t] = clip((r[c,t] - median[c]) / MAD[c], -5, 5)
```

LSTM her kanal için bir sonraki residual'ın merkezini ve ölçeğini üretir:

```text
mu[c,t]    = beklenen merkez
sigma[c,t] = beklenen normal oynaklık
0.1 <= sigma[c,t] <= 5.0
```

Kanal-bazlı Gaussian NLL:

```text
NLL[c,t] = log(sigma[c,t])
           + 0.5 * ((z[c,t] - mu[c,t]) / sigma[c,t])^2
```

Maskeli ve açık ağırlıklı toplam loss:

```text
loss = sum(w[c] * mask[c,t] * NLL[c,t])
       / sum(w[c] * mask[c,t])
```

Detection'a taşınan kanal skoru:

```text
S[c,t] = abs(z[c,t] - mu[c,t]) / sigma[c,t]
```

Yorum:

- `S ~= 0`: model beklentisine yakın;
- `S ~= 1`: yaklaşık bir predicted-scale uzakta;
- `S ~= 2`: yaklaşık iki predicted-scale uzakta;
- yüksek S: ilgili kanal ve bağlam için daha sıra dışı.

S değeri doğrudan anomaly olasılığı veya confidence değildir. Normal calibration ile conformal
p-değerine çevrilmesi gerekir:

```text
p[c,t] = (1 + count(S_calibration >= S[c,t])) / (n + 1)
```

## Normal bağlam ve hiyerarşik calibration

Planlanan normal calibration hiyerarşisi:

```text
channel + phase + cadence
          ↓ destek yetersizse
channel + phase
          ↓ destek yetersizse
channel
          ↓ destek yoksa
unscoreable
```

Amaç, tırmanış, level flight ve farklı mesaj cadence'lerinin normal dağılımlarını tek threshold'a
zorlamamaktır.

Dar kümenin avantajı: daha homojen normal dağılım ve küçük anomaly'ye daha yüksek hassasiyet.

Dar kümenin riski: calibration örneği azalır ve p-değeri çözünürlüğü kötüleşir. En küçük mümkün
conformal p-değeri `1/(n+1)`'dir:

| Küme örneği n | En küçük p |
|---:|---:|
| 100 | yaklaşık 0,0099 |
| 1.000 | yaklaşık 0,0010 |
| 10.000 | yaklaşık 0,0001 |

Bu nedenle yalnız “daha fazla cluster” daha iyi değildir; homojenlik ile örnek desteği arasında
bias-variance/support dengesi vardır.

## Magnitude-domination sonucu

Natural diagnostic sonuçları:

- trained-vs-untrained Spearman rho: **0,649633**
- trained-vs-target-magnitude Spearman rho: **0,654240**
- önceden dondurulmuş fail sınırı: `rho >= 0,8`
- sonuç: `magnitude_domination_flagged=false`, gate **PASS**

Eski üç NN'de rho yaklaşık `0,84–0,94` idi ve magnitude-domination FLAGGED olmuştu. Yeni skor salt
genlik kopyası değildir; ancak `rho ~= 0,65` nedeniyle genlik ilişkisi tamamen ortadan kalkmış da
değildir.

Doğal diagnostic kanal p95 surprise değerleri yaklaşık `1,840–2,052` aralığındadır. Normal
pencerelerin yaklaşık %95'i kanal bazında kabaca iki predicted-scale içinde kalmıştır. Bu bir
Gaussian goodness-of-fit kanıtı değil, yalnız skor ölçeği sanity kontrolüdür.

## Eski adaylarla karşılaştırma

| Sistem | Sonuç | Yorum |
|---|---|---|
| Eski Dense-AE | tarihsel pooled AUROC 0,572; rho 0,86/0,90 | magnitude FLAGGED |
| Eski LSTM-AE | tarihsel pooled AUROC 0,568; rho 0,84/0,89 | magnitude FLAGGED |
| Eski LSTM forecaster | tarihsel pooled AUROC 0,552; rho 0,94/0,92 | magnitude FLAGGED |
| Corrected residual rule | AUROC 0,764883; AUPRC 0,883313 | ayrıştırma güçlü, doğal alarm yükü yüksek |
| Eski vector CUSUM h=1 | doğal uçuşların yaklaşık %99,1'i alarm görüyor | doygun, Step-7 FAIL |
| Yeni contextual LSTM | rho 0,649633/0,654240, magnitude PASS | detection sonucu henüz yok |

Corrected residual rule için ayrıca:

- doğal burden: **4,808533 episode / scoreable uçuş-saat**;
- alarm gören uçuş oranı: **0,892356**;
- ground-speed event recall: **0,963659**, medyan gecikme **19,31 s**;
- track event recall: **0,951804**, medyan gecikme **56,75 s**;
- stealthy ramp event recall: **0,801347**;
- stealthy ramp active-interval micro coverage: yalnız **0,183902**.

Bu, tek bir event hit ile olay boyunca faydalı coverage'ın aynı olmadığını gösterir.

## Anomaly validation hakkındaki Codex bulgusu

Anomaly validation normal confidence hesabını doğrudan kalibre etmek için kullanılmamalıdır.
Roller ayrılmalıdır:

```text
normal fit
  -> model normal next-step davranışını öğrenir

normal calibration
  -> conformal p-değeri ve doğal tail hesaplanır

normal development/rehearsal
  -> doğal alarm burden ve kararlılık ölçülür

anomaly development
  -> channel/context/temporal tasarımın anomaly sensitivity'si sınanır

dokunulmamış anomaly test
  -> nihai recall, delay, coverage ve AUROC/AUPRC ölçülür
```

Anomaly örnekleri şu amaçlarla faydalı olabilir:

- hangi normal context ayrımının detection'a gerçekten katkı verdiğini sınamak;
- hangi residual kanalının hangi anomaly ailesine tepki verdiğini görmek;
- instant/persistence/accumulation profilini değerlendirmek;
- aşırı dar kümelerin recall'ı öldürüp öldürmediğini görmek;
- persistent anomaly'nin model geçmişine karışıp skorun sönmesini ölçmek.

Anomaly örnekleri şu hesaplara karıştırılmamalıdır:

- normal MAD/scaler;
- normal conformal tail;
- normal confidence p-değeri;
- doğal alarm burden threshold'u.

Anomaly skorları normal tail'e katılırsa dağılım genişler, threshold yükselir ve küçük anomaliler
normalleşebilir.

## Önemli metodolojik sonuç

Anomaly validation'a bakarak clustering, model config veya temporal threshold değiştirilirse bu
set artık test değildir; development setidir. Sistem fiilen semi-supervised model-selection
haline gelir. Bu bilimsel olarak yapılabilir, fakat:

1. yeni config yeni namespace altında önceden kaydedilmelidir;
2. mevcut truth-v2 development rolüne düşer;
3. nihai iddia için görülmemiş truth-v3 veya eşdeğer bağımsız anomaly test gerekir;
4. üçlü kör raw holdout ayrı unseal onayı olmadan kullanılamaz.

Mevcut `contextual_physics_v1` sözleşmesinde truth-v2 feedback yasaktır. Bu nedenle anomaly
validation ile cluster/threshold tuning mevcut v1 içinde post-hoc yapılamaz.

## Henüz olmayan sonuçlar

Yeni contextual model için henüz aşağıdakiler yoktur:

- AUROC;
- AUPRC;
- event recall;
- first-alarm delay;
- active-interval coverage;
- doğal operator-facing alarm episode burden;
- alarm gören uçuş oranı;
- anomaly profile bazlı threshold.

Bu değerler hesaplanmadığı için “yeni model anomalilerin %X'ini yakalıyor” denemez. Mevcut sonuç
yalnız model eğitimi ve magnitude gate sonucudur.

## Açık operasyonel karar

Calibration/evaluation öncesinde kullanıcı tarafından şu sayı tanımlanmalıdır:

```text
100 scoreable uçuş-saatinde kabul edilen maksimum operator-facing alarm episode sayısı
```

Ardından toplam bütçenin beş residual channel ve ayrı S2 reason-code katmanları arasında nasıl
paylaşılacağı önceden dondurulmalıdır. Bu karar anomaly sonuçları görüldükten sonra verilmemelidir.

## Claude'dan istenen bağımsız değerlendirme

Lütfen aşağıdaki soruları açıkça yanıtla:

1. Anomaly validation kullanmadan normal context kümelerini daraltmak için mevcut
   `channel + phase + cadence` hiyerarşisine hangi nedensel, anomaly'den etkilenmeyen bağlamlar
   eklenebilir?
2. Minimum calibration support ve fallback seviyeleri nasıl seçilmeli? Conformal p-resolution ile
   cluster homojenliği arasındaki dengeyi öner.
3. Anomaly development kullanılması bilimsel olarak değerli mi? Değerliyse `contextual_physics_v2`
   için veri rolleri ve dokunulmamış final test nasıl kurulmalı?
4. Persistent anomaly'nin 12 satırlık history'ye karışarak surprise skorunu söndürmesi nasıl
   ölçülmeli ve önlenmeli?
5. Kanal-bazlı conformal p-değerleri instant/persistence/time-normalized accumulation profillerine
   nasıl bağlanmalı?
6. Kullanıcı operasyonel burden sayısını henüz bilmiyorsa, sonucu görmeden seçilebilecek savunulabilir
   bir burden duyarlılık protokolü nedir? Tek sayı yerine önceden dondurulmuş Pareto eğrisi kabul
   edilebilir mi?
7. Mevcut magnitude PASS kanıtı yeterli mi; hangi ek falsification testleri yapılmalı?
8. Yeni adayın eski residual rule ile aynı doğal burden seviyesinde adil karşılaştırması nasıl
   yapılmalı?

## İlgili repo belgeleri

- `docs/adsb_overall_model_report_2026-07-14.md`
- `docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md`
- `docs/decisions.md` — ADR-024, ADR-025, ADR-028, ADR-030, ADR-032–036
- `configs/adsb_contextual_physics_v1_train.json`
- `adsb/models/contextual_residual_forecaster.py`
- `adsb/contextual_windowing.py`
- `adsb/context.py`
- `adsb/conditional_calibration.py`
- `adsb/contextual_decision.py`

Son yayımlanmış ilgili commitler:

```text
3de9c9f  record contextual ADS-B training result
fd8c7ea  document contextual ADS-B model interpretation
```
