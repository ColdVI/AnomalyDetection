# ADS-B Contextual-Physics Anomali Tespiti — Deney Kayıtları

> **Bu dosya, ADS-B `contextual_physics_v1` aday ailesinin tüm sonuç ve ön-kayıt
> belgelerini tek yerde toplar.** Daha önce ayrı duran dört belge buraya mantıksal
> sırayla birleştirildi; içerik ve sayılar değiştirilmeden korundu. Orijinal ayrık
> dosyalar ile ilgili AI-inceleme kayıtları `arsiv` branch'inde durmaya devam ediyor.

## Yönetici özeti

`contextual_physics_v1`, ham irtifa/hız/track'i yeniden kurmak yerine, aynı uçuşun
önceki 12 satırına bakıp bir sonraki satırdaki **beş fiziksel residual kanalının**
beklenen merkezini ve belirsizliğini tahmin eden bir **uçuş-içi zamansal-sürpriz**
modelidir.

**Bugünkü durum: eğitildi, magnitude gate geçti, alarm/recall kararı verilmedi.**

- Eğitim: 2.929 fit uçuşu, 1.267.625 satır. Weighted Gaussian NLL
  `0,795375 → 0,708696` (≈ %10,90 göreli düşüş).
- Magnitude-domination testi: trained-vs-untrained ρ = 0,649633, trained-vs-magnitude
  ρ = 0,654240 — her ikisi de dondurulmuş `ρ ≥ 0,8` sınırının altında → **gate PASS**
  (önceki autoencoder'ların ρ 0,84–0,94 artefaktı bu modelde tekrarlanmadı).
- Henüz **yok**: alarm threshold'u, doğal alarm yükü, truth-v2 recall. Bu yüzden bu
  model için bugün "%X anomali yakalıyor" demek sayı uydurmak olur.

Bağlam olarak eski ADS-B hattı: kural-bazlı skorlayıcı (AUC 0,60) üç NN'i de geçti;
düzeltilmiş truth-v2 kuralı güçlü ayrıştı (AUROC 0,764883 / AUPRC 0,883313) ama
CUSUM doğal alarm doygunluğu nedeniyle Adım-7 operasyonel gate'i FAIL oldu (ADR-032).

## İçindekiler

1. Genel Teknik Sonuç ve Yorum Raporu
2. Aday Ön-kayıt Sözleşmesi
3. Alarm Bütçesi / Kanal Payı Ön-kaydı
4. Isolation Forest Paralel Keşif Ön-kaydı


---

## ADS-B anomaly detection — genel teknik sonuç ve yorum raporu

Tarih: 2026-07-14  
Aktif aday: `contextual_physics_v1`  
Durum: **eğitildi, magnitude gate geçti, henüz threshold/alarm kararı dondurulmadı**

### 1. Önce kısaltmalar

- **Automatic Dependent Surveillance–Broadcast (ADS-B):** Uçağın kimlik, konum, hız ve benzeri durum bilgilerini yayınladığı gözetim sistemi.
- **Long Short-Term Memory (LSTM):** Sıralı/zamansal veride geçmiş bilgiyi taşıyan sinir ağı.
- **Neural Network (NN):** Sinir ağı.
- **Median Absolute Deviation (MAD):** Medyandan mutlak sapmaların medyanı; uç değerlere dayanıklı ölçek ölçüsü.
- **Negative Log-Likelihood (NLL):** Modelin gözlenen değere verdiği olasılıksal uyumsuzluk kaybı.
- **Area Under the Receiver Operating Characteristic Curve (AUROC):** Rastgele seçilen bir anomalinin normalden daha yüksek skor alma olasılığı olarak yorumlanabilen, threshold'dan bağımsız sıralama ölçüsü.
- **Area Under the Precision–Recall Curve (AUPRC):** Precision-recall eğrisinin alanı; sınıf dengesizliğinde pozitif sınıf performansını özetler.
- **Cumulative Sum (CUSUM):** Küçük fakat sürekli sapmaları zaman boyunca biriktiren yöntem.
- **Navigation Integrity Category (NIC), Navigation Accuracy Category for Position (NACp), Source Integrity Level (SIL):** ADS-B mesajının bütünlük/doğruluk beyan alanları.

### 2. Sonuç özeti

Yeni model, önceki autoencoder'ların yaptığı gibi ham irtifa/hız/track'i yeniden kurmuyor. Aynı uçuşun önceki **12 satırına** bakıp bir sonraki satırdaki **beş ayrı fiziksel residual kanalının** beklenen merkezi ve normal belirsizliğini tahmin ediyor. Bu nedenle gerçek bir **uçuş-içi time series** modelidir.

Eğitim, fit rolündeki uçuşların önceden dondurulmuş deterministik %2 örneği olan 2.929 uçuş, 1.267.625 satır ve epoch başına 1.180.160 pencereyle tamamlandı. Weighted Gaussian NLL `0,795375 -> 0,708696` düştü; göreli düşüş yaklaşık **%10,90**. Ayrı 770 doğal calibration-diagnostic uçuşundaki 332.510 pencere için trained-vs-untrained ve trained-vs-magnitude Spearman korelasyonları sırasıyla **0,649633 / 0,654240** oldu. Önceden dondurulmuş `rho >= 0,8` magnitude-domination sınırı aşılmadı; gate **PASS**.

Bu, modelin anomaly yakaladığı anlamına gelmez. Bugünkü doğru sonuç şudur:

> Model normal uçuş zaman serisinden bağlama bağlı next-step beklenti ve kanal-bazlı sürpriz üretmeyi öğrendi; skorun yalnız ham büyüklüğün kopyası olduğuna dair test geçildi. Alarm threshold'u, doğal alarm yükü ve truth-v2 anomaly recall henüz ölçülmedi.

### 3. Uçuşun kendi içindeki anomaliye bakıyor muyuz?

**Skorlarken evet; eğitirken anomaly etiketiyle hayır.** Ayrım önemlidir.

Her hedef satır `t` için yalnız aynı `flight_id` içindeki önceki 12 satır `[t-12, ..., t-1]` modele girer. Uçuş sınırından pencere geçmez. Zaman ters giderse veya ardışık iki mesaj arası 60 saniyeyi aşarsa pencere üretilmez. Model mevcut satırdaki fiziksel residual'ı önceden görmeden tahmin eder.

Modelin ağırlıkları uçuş sırasında yeniden eğitilmez. Dolayısıyla yaptığı karşılaştırma iki parçalıdır:

1. bütün doğal-fit uçuşlarından öğrendiği genel, bağlama-koşullu normallik;
2. o uçuşun son 12 satırındaki kendi yakın geçmişi.

Eğitime anomaly etiketi verilmedi ve sentetik eğitim satırı **0**'dır. Eğitim verisi `natural_clean_fit` kabul edilen uçuşlardan gelir. Eğer bu havuzda bilinmeyen gerçek anomaliler varsa contamination riski vardır: model bunların sık olanlarını normal sanabilir. Normal-only novelty detection'ın temel riski budur.

Uzun süre devam eden bir bozulma, 12 satırlık geçmişi zamanla kendisiyle kirletebilir ve anlık skoru düşürebilir. Bu yüzden tek bir NN skoru yeterli değildir: spike için anlık, freeze/bias için persistence, stealthy ramp için gerçek saniyeyle CUSUM-benzeri accumulation gerekir.

### 4. Model neye bakıyor?

Model doğrudan “uçak anormal mi?” sorusunu öğrenmiyor. Önce fiziksel tutarsızlıklar hesaplanıyor:

- vertical-rate residual = bildirilen dikey hız − `delta altitude / delta time`;
- speed residual = bildirilen yer hızı − ardışık konumdan türetilen hız;
- heading residual = bildirilen track − ardışık konumun dairesel bearing'i;
- east/north velocity residual = bildirilen hız vektörü − konum geçişinden türetilen hız vektörü.

`altitude_source_residual` barometrik/geometrik irtifa farkının zaman türevidir; fit MAD değeri tam sıfır çıktığı için yapay floor verilmeden dışlandı. Aktif kanal sayısı bu nedenle altı değil **beş** oldu.

Model girdisi 19 boyutludur: beş residual, gerçek `log(1 + delta-t)`, track'in `sin/cos` gösterimi, yalnız geçmişten çıkarılmış flight phase, cadence bucket'ları ve her girdinin availability maskesi. Track'in 359 derece ile 1 derece arasındaki dairesel yakınlığı `sin/cos` ile korunur.

### 5. Matematik: eğitim ve anomaly skoru

Her aktif residual önce fit-normal medyan ve MAD ile ölçeklenip `[-5, 5]` aralığında kırpılır:

```text
z[c,t] = clip((r[c,t] - median[c]) / MAD[c], -5, 5)
```

LSTM geçmiş 12 satırdan her kanal için iki çıktı verir:

```text
mu[c,t]     = beklenen bir sonraki residual merkezi
sigma[c,t]  = beklenen normal oynaklık, 0.1 <= sigma <= 5.0
```

Eğitim kaybı, eksik hedefleri maskeler ve beş kanalı açıkça eşit ağırlıklandırır:

```text
NLL[c,t] = log(sigma[c,t])
           + 0.5 * ((z[c,t] - mu[c,t]) / sigma[c,t])^2

loss = sum(w[c] * mask[c,t] * NLL[c,t]) / sum(w[c] * mask[c,t])
```

Detection katmanına taşınan esas kanal skoru standartlaştırılmış mutlak sürprizdir:

```text
S[c,t] = abs(z[c,t] - mu[c,t]) / sigma[c,t]
```

Pratik yorum:

- `S ~= 0`: gerçekleşen residual model beklentisine çok yakın;
- `S ~= 1`: yaklaşık bir model ölçeği uzaklık;
- `S ~= 2`: yaklaşık iki model ölçeği uzaklık;
- daha büyük değer: o kanal ve bağlam için daha sıra dışı gözlem.

Bu değer doğrudan olasılık veya alarm değildir. Ayrı doğal calibration dağılımında conformal p-değerine çevrilecektir:

```text
p[c,t] = (1 + calibration'da S >= S[c,t] olan örnek sayısı) / (n + 1)
```

Calibration hiyerarşisi `channel + phase + cadence -> channel + phase -> channel` biçimindedir. Yeterli örnek yoksa deterministik olarak daha kaba gruba düşer; hiçbir destek yoksa skor verilmez. Böylece tırmanıştaki normal dikey hareket ile level-flight normalliği aynı threshold'a zorlanmaz.

### 6. Sayılar bize ne söylüyor?

| Kanıt | Sonuç | Doğru yorum |
|---|---:|---|
| Epoch NLL | 0,795375 -> 0,708696 | Fit verisinde olasılıksal next-step tahmini iyileşti |
| NLL göreli düşüş | yaklaşık %10,90 | Optimizasyon çalıştı; detection recall kanıtı değildir |
| Calibration diagnostic pencere | 332.510 | Magnitude testi için optimizer dışı doğal örnek |
| Trained vs untrained rho | 0,649633 | Eğitim skor sıralamasını değiştirdi; hâlâ orta düzey ilişki var |
| Trained vs magnitude rho | 0,654240 | Skor salt genlik değil; genlik etkisi tamamen yok da değildir |
| Magnitude gate | PASS (`rho < 0,8`) | Önceki NN'lerdeki bariz artefakt bu testte görülmedi |
| Checkpoint | 9.546 sonlu parametre, strict reload PASS | Artefakt teknik olarak tekrar yüklenebilir |
| Sentetik train/calibration | 0 / 0 | Sentetik anomaly modele öğretilmedi |

Doğal diagnostic p95 kanal skorları `1,840–2,052` aralığındadır. Yani normal calibration pencerelerinin yaklaşık %95'i ilgili kanalda kabaca iki predicted-scale içinde kalmıştır. Bu Gaussian varsayımının kanıtı değildir; yalnız skor ölçeğinin pratik kontrolüdür.

### 7. Eski adaylarla overall durum

| Sistem | Bilinen sonuç | Karar |
|---|---|---|
| Eski Dense-AE / LSTM-AE / LSTM-forecaster | pooled tarihsel AUROC 0,572 / 0,568 / 0,552; magnitude rho 0,84–0,94 | Detection adayı olarak güvenilmez |
| Residual rule, corrected truth-v2 | AUROC 0,764883; AUPRC 0,883313 | Ayrıştırma güçlü, fakat doğal burden 4,808533 episode/saat ve alarm gören uçuş oranı 0,892356 |
| Eski vector CUSUM h=1 | doğal günlerde alarm gören uçuş oranı yaklaşık 0,991 | Doygun; Step-7 FAIL |
| Yeni contextual LSTM | magnitude rho 0,649633 / 0,654240; gate PASS | Umut verici eğitim, henüz AUROC/AUPRC veya recall yok |

Eski residual rule corrected truth-v2'de ground-speed event recall `0,963659`, track event recall `0,951804` verdi; stealthy ramp recall `0,801347` görünse de aktif-aralık micro coverage yalnız `0,183902` idi. Yani bir olaya bir kez dokunmak ile olay boyunca faydalı biçimde alarmda kalmak aynı şey değildir. Overall karar bu yüzden yalnız AUROC veya event recall ile verilmez.

### 8. Anomaly tespit ederken hangi kanala ve zamansal davranışa dikkat edeceğiz?

| Anomaly ailesi | Birincil kanıt | Karar davranışı |
|---|---|---|
| Ground-speed spike/bias | speed residual surprise | spike için instant; bias için persistence |
| Vertical-rate spike/freeze | vertical-rate residual surprise | spike instant; freeze persistence |
| Track frozen/direction inconsistency | heading residual surprise | persistence |
| Stealthy position ramp | east/north velocity residual surprise | saniye-normalize accumulation/CUSUM |
| Altitude dropout | availability/S2 reason code | NN residual'ına karıştırılmaz |
| Mesaj gap'i | cadence/message-gap S2 reason code | ayrı veri-kalitesi episode'u |
| NIC/NACp/SIL bozulması | declared-quality S2 reason code | fizik residual'ından ayrı yorum |

Her anomaly türü için ayrı threshold fikri doğrudur; fakat threshold doğrudan ham skora değil, önce channel/phase/cadence koşullu conformal p-değerine uygulanmalıdır. Böylece normal uçuşların heterojenliği hesaba katılır. Channel'lar sessizce toplanmaz; her biri toplam alarm bütçesinden önceden pay alır.

### 9. Modelin şu an verdiği sonuç tam olarak nedir?

Modelin ürün çıktısı henüz `anomaly=yes/no` değildir. Her scoreable satır ve kanal için şu paket olmalıdır:

```text
flight_id, target_time, channel, phase, cadence,
predicted_mu, predicted_sigma, observed_scaled_residual,
standardized_surprise, conformal_p_value,
temporal_evidence, alarm
```

Bugünkü checkpoint conformal öncesi model/skor alanlarını üretebilir; gerçek conformal p-değeri, temporal evidence ve alarm için operasyonel bütçe/config dondurulmamıştır. Dolayısıyla bu model için bugün AUROC, AUPRC, event recall veya “%X anomaly yakalıyor” sayısı söylemek sayı uydurmak olur.

### 10. Sonraki zorunlu karar ve değerlendirme sırası

Sonuç çıkarma sırası değiştirilmemelidir:

1. Kullanıcı kabul edilebilir **toplam doğal alarm burden** değerini tanımlar; örneğin `100 scoreable uçuş-saatinde en fazla X operator-facing episode`.
2. Bu toplam bütçe beş residual channel ve S2 reason'ları arasında önceden paylaştırılır.
3. Minimum conditional-calibration desteği ve anomaly başına instant/persistence/accumulation eşikleri truth-v2 açılmadan dondurulur.
4. Ayrı doğal calibration ile conformal tail kurulur.
5. Natural development ve donmuş rehearsal üzerinde episode/saat, alarm gören uçuş oranı ve cadence/phase kararlılığı ölçülür.
6. Ancak doğal burden geçerse truth-v2 üzerinde event recall, first-alarm delay, active coverage, AUROC/AUPRC ve channel attribution çıkarılır.
7. Üçlü kör holdout havuzu ayrı unseal onayı olmadan açılmaz.

Bu sırada eksik olan tek bir “istatistiksel metrik” değil, detector'ın operasyonel maliyet sözleşmesidir. Bir threshold'u anomaly sonuçlarına bakıp seçmek kolayca iyi görünen ama sahada alarm yağdıran bir sistem üretir; önceki rule ve CUSUM sonuçları bunun sayısal örneğidir.


---

## ADS-B contextual-physics candidate v1 — ön-kayıt sözleşmesi

### Durum ve kapsam

Bu belge, kullanıcının 2026-07-14 tarihli açık yönlendirmesiyle araştırılan ve uygulanması
onaylanan yeni aday ailesinin yapısal sözleşmesidir. Adayın namespace'i
`contextual_physics_v1`'dir. Step-5 CUSUM, ADR-025 kuralı, tarihsel üç NN veya Step-7 FAIL
sonucu değiştirilmez; bu aday bunlardan birinin yeniden adlandırılmış devamı değildir.

Bu revizyonda gerçek-veri bilimsel koşusu, threshold seçimi, config freeze, Step-8 terfisi veya
holdout erişimi **yoktur**. Operasyonel toplam alarm bütçesi kullanıcı tarafından henüz sayısal
olarak tanımlanmadığı için kod alert alpha olmadan alarm üretmez.

### Problem sözleşmesi

Çıktı tek ve açıklamasız bir reconstruction skoru değildir. Her skor aşağıdakileri taşır:

- `anomaly_type` ve fiziksel `channel`;
- skorun ait olduğu nedensel normal bağlam (`context_phase`, `context_cadence`);
- conformal p-değeri, kullanılan fallback seviyesi ve calibration örnek sayısı;
- anlık/süreklilik/birikimli temporal evidence;
- `alarm=yes|no` veya veri yetersizse score dışı durum.

Satır, event, uçuş ve scoreable uçuş-saat metrikleri ayrı raporlanır. Sentetik truth-v2 yalnız
değerlendirmede kullanılabilir; fit, robust scaling, normal-tail calibration veya threshold
seçiminde kullanılamaz.

### Yapısal olarak dondurulan kararlar

1. Uçuş fazı, skorlanan mevcut satırın vertical-rate değerini kullanmaz. Yalnız aynı uçuşun
   önceki satırlarından lagged rolling median ile `ground/climb/level/descent/unknown` üretilir.
2. Track ham derece olarak NN'ye verilmez; `sin(track)` ve `cos(track)` kullanılır.
3. Gerçek `delta-t` ve cadence bağlamı model girdisinde açıkça bulunur. Büyük gap pencereyi böler.
4. Model yalnız fiziksel residual kanallarının bir sonraki değerine kanal-bazlı location/scale
   tahmini yapar. Ham altitude/speed/track reconstruction hedefi değildir.
5. Availability mask modele açık girdidir. Missingness/altitude-dropout residual MSE içinde
   kaybedilmez; S2/state kanalı olarak ayrı kalır.
6. Fit-normal median/MAD scaling uygulanır. MAD=0 veya verisiz kanal floor'lanmadan tamamen
   dışlanır ve manifestte yazılır.
7. NN loss ve çıktı skoru kanal bazında korunur. Kanal ağırlıkları zorunlu ve config'te açık
   olmalıdır; sessiz equal-weight aggregation veya erken fusion yoktur.
8. Normal calibration skorları bağlam hiyerarşisinde conformal p-değerine çevrilir:
   `channel+phase+cadence -> channel+phase -> channel`. Minimum destek yoksa daha kaba bağlama
   deterministik fallback yapılır; channel fallback de yoksa skor verilmez.
9. Spike, persistence ve accumulation temporal modları ayrıdır. Accumulation mesaj sayısıyla
   değil gerçek saniye ile güncellenir. Böylece yüksek cadence'in sırf daha çok örnek nedeniyle
   daha hızlı alarm üretmesi önlenir.
10. Her fizik channel'ı toplam alert-alpha bütçesinden önceden pay alır. Payların toplamı bütçeyi
    aşamaz. Bir profile başka channel sessizce katılamaz; fusion ayrı ön-kayıt ister.

### Aday fizik kanalları

- `speed_residual`: ground-speed spike/bias;
- `vertical_rate_residual`: vertical-rate spike/freeze;
- `heading_residual`: track-frozen/direction inconsistency;
- `east_velocity_residual`, `north_velocity_residual`: stealthy position ramp için işaretli,
  zaman-birikimli kanıt;
- `altitude_source_residual`: yalnız fit-normal MAD pozitifse;
- altitude availability, message gap ve declared quality: NN residual'ına karıştırılmadan mevcut
  S2 reason-code katmanında.

### Sayısal freeze öncesi açık kullanıcı kararları

Aşağıdaki değerler sonuç görülmeden tek config olarak dondurulmalıdır:

- toplam operasyonel alert-alpha/burden bütçesi ve channel payları;
- lagged phase history satırı ve level-rate sınırı;
- cadence sınırları, max gap ve history satır sayısı;
- robust clip;
- model hidden size/layer, location-scale sınırları, seed, epoch, batch ve learning rate;
- explicit channel loss ağırlıkları;
- anomaly profile başına instant/persistence/accumulation modu ve zaman/evidence eşikleri;
- minimum conditional-calibration grup desteği.

Bu değerlerin hiçbiri truth-v2 recall/AUC sonuçlarına bakılarak seçilemez. Kullanıcı sayısal
operasyonel bütçeyi tanımlamadan gerçek calibration veya evaluation koşusu başlatılmaz.

### Veri rolleri ve değerlendirme sırası

1. Natural fit: scaler ve NN; sentetik satır sayısı zorunlu olarak sıfır.
2. Ayrı natural calibration: conformal tail ve önceden verilmiş bütçeye göre karar config'i.
3. Natural development: episode/uçuş/uçuş-saat burden ve bağlam/cadence kararlılığı.
4. Donmuş natural rehearsal: geri besleme yok.
5. Truth-v2: yalnız corrected event recall, delay, active coverage ve channel attribution.
6. Ana aday kıyası: aynı natural burden'da ADR-025 rule ve yeni aday; pooled AUC tek gate değil.
7. Üçlü kör holdout havuzu bu adayın geliştirme verisi değildir ve ayrı unseal kararı olmadan
   açılmaz.

### Normal-only eğitim freeze'i — 2026-07-14

Kullanıcı gerçek eğitimin başlatılmasını açıkça onayladı. Alarm/threshold bütçesi hâlâ açık
olduğu için bu freeze yalnız normal-only model fit ve doğal calibration diagnostic kapsamındadır.
Tek çalıştırılabilir config `configs/adsb_contextual_physics_v1_train.json` dosyasıdır; sweep yoktur.

- Kaynak split: hash'i config'te bağlı Step-5 açık-Silver manifestindeki yalnız `fit` uçuşları.
- Eğitim örneği: fit uçuşlarının SHA-256 ile deterministik yüzde 2'si; seed 20260714.
- Optimizer dışı diagnostic: calibration uçuşlarının ayrı deterministik yüzde 2'si; seed 20260715.
- Kanallar: vertical-rate, speed, heading, altitude-source, east-velocity ve north-velocity
  residual; MAD=0 kanal fit sonrasında floorsuz dışlanır.
- Bağlam: 3 geçmiş satırlı lagged phase, 1.0 m/s level sınırı, 2/5/15 s cadence sınırları,
  60 s max gap.
- Pencere/model: 12 geçmiş satır, bir sonraki satır hedefi, hidden 32, tek LSTM katmanı,
  scale aralığı 0.1–5.0.
- Scaling/training: robust clip 5, active scaled kanallara açık 1.0 ağırlık, 5 epoch,
  batch 512, Adam learning-rate 0.001, seed 0, gradient clip 1.0.

Bu sayılar model sonucu görülmeden kaydedildi. Eğitim truth-v2, development, rehearsal ve holdout
okumaz; alarm alpha, persistence veya accumulation threshold'u seçmez.

### İlk gate

Aday ancak provenance/checksum tam, sentetik eğitim sıfır, magnitude-domination false, conditional
calibration desteği yeterli ve natural development/rehearsal burden önceden verilen bütçeyi sağlıyor
ise truth-v2 fayda kıyasına geçebilir. Bir koşul fail olursa config sonuçlara göre aynı run içinde
değiştirilmez; yeni config yeni namespace ve yeni ön-kayıt gerektirir.


---

## contextual_physics_v1 — alarm bütçesi / kanal payı / temporal profil ön-kaydı

Tarih: 2026-07-14
Durum: **sonuç görülmeden dondurulmuş** — kalibrasyon/development/rehearsal koşusu başlamadan önce yazıldı
Kapsam: `docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md`'nin "Sayısal freeze öncesi açık
kullanıcı kararları" bölümünün ilk üç maddesini kapatır. Kullanıcı üç soruyu da AskUserQuestion ile
onayladı: (1) Pareto ızgarası, (2) kanıta ağırlıklı kanal payı, (3) Claude'un önerdiği temporal
eşikler.

### 1. Toplam operasyonel alarm bütçesi — Pareto ızgarası

Tek sayı yerine, 100 scoreable uçuş-saatte kabul edilen maksimum operator-facing episode sayısı için
5 noktalık dondurulmuş bir ızgara:

```text
budget_grid_per_100h = [0.1, 0.5, 1.0, 2.0, 5.0]
```

Her nokta için development/rehearsal turunda AYRI burden/recall/coverage raporlanacak. Nihai
operasyonel seçim kullanıcı tarafından bu ızgaradan yapılacak — hangi noktanın "en iyi" göründüğüne
bakılarak ızgaranın kendisi genişletilmeyecek/daraltılmayacak.

### 2. Kanal/S2 bütçe payı — kanıta ağırlıklı türetim

Girdi kanıtı: ADR-028'in corrected truth-v2 pooled reçete AUROC'ları (donmuş kural, değiştirilmemiş
kalibrasyon, `artifacts/adsb/runs/20260713_step3_corrected_rule_v1/`):

| Reçete → kanal | AUROC | Skill = AUROC − 0.5 |
|---|---:|---:|
| ground-speed → `speed_residual` | 0.974023 | 0.474023 |
| track → `heading_residual` | 0.889552 | 0.389552 |
| vertical-rate → `vertical_rate_residual` | 0.696337 | 0.196337 |

**`east_velocity_residual` / `north_velocity_residual` için özel karar:** eski `ramp` reçetesinin
skaler-residual AUROC'u (0.558927) bu iki kanala DOĞRUDAN taşınmadı — çünkü bu sayı, ADR-029'da
tam olarak bu zayıflığı gidermek için terk edilen ESKİ (skaler) temsili ölçüyor, yeni iki-eksenli
temsili değil. Onun yerine, kanıtı henüz ölçülmemiş bu iki kanala en zayıf KANITLANMIŞ kanalın
(`vertical_rate_residual`, skill 0.196337) payı taban olarak verildi — ne eski-zayıf sayıyla
cezalandırılıyor ne de ölçülmemiş bir iyimserlikle şişiriliyor.

Beş fizik kanalı arasında normalize edilmiş pay (toplam skill = 1.452586):

| Kanal | Fizik-havuzu payı |
|---|---:|
| `speed_residual` | %32.6 |
| `heading_residual` | %26.8 |
| `vertical_rate_residual` | %13.5 |
| `east_velocity_residual` | %13.5 |
| `north_velocity_residual` | %13.5 |

**S2 veri-kalitesi katmanı için ayrı muamele:** S2 (squawk/emergency/NIC/NACp/SIL/message-gap
reason-code'ları) bir öğrenilmiş fizik-residual DEĞİL, deterministik bir bütünlük bayrağıdır — AUROC
kavramı ona uygulanmaz. Bu yüzden "kanıta ağırlıklı" ilkesi burada AUROC yerine S2'nin zaten ölçülmüş
doğal-yük kanıtına (ADR-031: MESSAGE_GAP ~2.9 episode/saat, NIC-unknown ~0.5 episode/saat) dayanarak
yorumlandı: S2 kendi başına büyük, öngörülemez bir yük taşıyabildiği için toplam bütçeden SABİT ve
mütevazı bir pay (**%15**) ayrılır, kalan **%85** yukarıdaki fizik-kanal oranlarıyla bölünür.

**Nihai toplam bütçe payları:**

| Katman | Toplam bütçe payı |
|---|---:|
| S2 (veri-kalitesi, ayrı) | %15.0 |
| `speed_residual` | %27.7 |
| `heading_residual` | %22.8 |
| `vertical_rate_residual` | %11.5 |
| `east_velocity_residual` | %11.5 |
| `north_velocity_residual` | %11.5 |

Bu paylar her Pareto ızgara noktasına aynı oranda uygulanır (örn. 1.0 episode/100 saatte
`speed_residual`'a düşen pay 0.277 episode/100 saattir).

### 3. Temporal karar profilleri (instant / persistence / accumulation)

Mevcut CUSUM sözleşmesinden (ADR-029, `adsb/cusum.py`) ve corrected truth-v2'nin gözlenen doğal
gecikmelerinden (ground-speed medyan 19.31s, track medyan 56.75s) türetildi. Sayılar sonuç
görülmeden dondurulmuştur; anomaly-development rolü yalnız MOD ATAMASINI (hangi kanal hangi profili
kullanır) sınayabilir, bu sayıları geriye dönük değiştiremez.

| Kanal | Alt-mod | Profil | Tanım |
|---|---|---|---|
| `speed_residual` | spike | instant | Tek skorlanan satırda `p < alpha` → alarm |
| `speed_residual` | bias | persistence | 30s gerçek-zaman pencerede medyan `p < alpha` |
| `vertical_rate_residual` | spike | instant | Tek satır |
| `vertical_rate_residual` | freeze | persistence | 30s pencere |
| `heading_residual` | — | persistence | 30s pencere (track-frozen doğası gereği anlık değil) |
| `east_velocity_residual` + `north_velocity_residual` | — | accumulation | Mevcut donmuş 2-eksenli causal Page CUSUM (ADR-029): `target_vector_shift_mps=2.0`, `k` train-MAD'den türetilir, `h` BUGÜN seçilmez — her Pareto bütçe noktasında doğal kalibrasyondan türetilir |
| S2 reason-code'ları | — | ayrı deterministik episode mantığı | Fizik residual skoruna karışmaz (mevcut S2 modülü) |

`speed_residual` ve `vertical_rate_residual`'ın instant/persistence alt-modları arasındaki bütçe
payı, sonuç görülmeden **%50/%50** olarak varsayılan alınmıştır — bu varsayılan yalnız
anomaly-development rolünde, yeni bir ön-kayıtlı versiyon açılarak değiştirilebilir.

Persistence penceresi (**30 saniye**) seçimi: track kanalının zaten gözlenen ~57s doğal gecikmesini
kötüleştirmeyecek, ama tek-satır gürültüsünü reddedecek kadar uzun bir orta nokta.

### Açık madde

`h` (CUSUM eşiği) bu belgeyle seçilmedi — her Pareto bütçe noktası için doğal
calibration/development turunda ayrı türetilecek (ADR-029'un kendi kuralı). Bu belge yalnız
BÜTÇE PAYLARINI ve zaman-profili YAPISINI dondurur, gerçek eşik sayılarını değil.


---

## isolation_forest_contextual_v1 — paralel keşif ön-kaydı

Tarih: 2026-07-14
Durum: **keşif** — `contextual_physics_v1`'i bloklamaz, onu ikame etmez, ana adaya dahil edilmez
Kullanıcı onayı: AskUserQuestion, "Şimdi paralel başlat"

### Neden ayrı bir aday, ayrı bir mekanizma

`contextual_physics_v1` bir **zamansal-sürpriz** dedektörüdür: "bu değer, bu uçuşun kendi yakın
geçmişine göre beklenmedik mi?" sorusuna cevap verir. Isolation Forest yapısal olarak **farklı bir
soru** sorar: "bu residual vektörü, TÜM normal uçuşların residual-uzayında ne kadar izole?" — hiçbir
uçuş-içi geçmişe bakmaz, saf çok-değişkenli yoğunluk/izolasyon mantığıdır.

Bu, iki dedektörün KÖR NOKTALARININ farklı olabileceği anlamına gelir:

- IF, bağlamdan bağımsız ama kanal-KOMBİNASYONU nadir olan bir noktayı yakalayabilir (LSTM bunu
  "geçmişe göre normal" sayıp kaçırabilir).
- LSTM, tek başına nadir olmayan ama BU UÇUŞUN akışına göre ani bir sıçramayı yakalayabilir
  (IF bunu havuzda sık görülen bir kombinasyon sayıp kaçırabilir).

Bu yüzden IF, `contextual_physics_v1`'in YERİNE değil, ONUNLA birlikte değerlendirilecek ayrı bir
sinyal olarak keşfediliyor. Füzyon (varsa) ayrı, sonraki bir ön-kayıt gerektirir — bu belge füzyon
kararı VERMİYOR.

### Sözleşme (contextual_physics_v1 ile aynı disiplin)

- Eğitim SADECE `natural_clean_fit` rolündeki satırlardan; sentetik satır sayısı zorunlu sıfır
  (`StrictNaturalRobustScaler` bunu çalışma zamanında zorlar, adsb/contextual_scaling.py).
- Kanallar: `speed_residual`, `vertical_rate_residual`, `heading_residual`,
  `east_velocity_residual`, `north_velocity_residual` — `contextual_physics_v1` ile BİREBİR aynı 5
  kanal (`altitude_source_residual` aynı sebeple, MAD=0, dışlanır).
- Ölçekleme: aynı `StrictNaturalRobustScaler` (medyan/MAD, clip=5.0, MAD=0 floor'suz dışlama) —
  iki aday arasında ölçekleme kaynaklı bir fark OLMASIN diye kasıtlı paylaşım.
- Sonuç görülmeden dondurulmuş hiperparametreler: `n_estimators=200`, `max_samples="auto"`,
  `contamination="auto"` (yalnız sklearn'in iç `predict()` ofsetini etkiler, KULLANILMAZ),
  `random_state=0`. Skorlama `score_samples()` üzerinden sürekli değer olarak yapılır, ikili
  `predict()` çıktısı hiçbir yerde kullanılmaz.
- **Bilinen basitleştirme (dürüstçe beyan):** LSTM tarafının `availability mask`'ı burada YOK —
  IF, aktif 5 kanaldan herhangi biri NaN olan satırları TAMAMEN atar (complete-case). Bu, eksik
  veri deseni gerçek bir anomali sinyaliyse (örn. altitude_dropout) IF'in o satırları hiç
  göremeyeceği anlamına gelir — S2 katmanı zaten bu durumu ayrı yakalıyor, çakışma yok ama IF'in
  kapsamı LSTM'den DAHA DAR'dır, bu bir sınırlama olarak kayıtlıdır.
- Sentetik veri yalnız DEĞERLENDİRMEDE (anomaly-development rolünde, kalıcı korpus
  `data/objectstore/synthetic/adsb/`) kullanılır — fit'e asla girmez.
- Aynı alarm bütçesi/Pareto ızgarası ve kanal payı çerçevesi (ADR-037) IF'e de uygulanacak;
  IF kendi ayrı eşiğini alacak, `contextual_physics_v1`'in eşiğini paylaşmayacak.

### İlk gate

Provenance/checksum tam, sentetik eğitim sıfır, ve doğal calibration diagnostic üzerinde IF
skorunun ham-genlik taban çizgisiyle (`adsb/diagnostics.py::magnitude_only_score`) Spearman
korelasyonu ölçülüp raporlanmadan (FLAG/PASS iddiası olmadan, yalnız ölçüm) hiçbir "IF daha iyi/
kötü" karşılaştırması yapılmaz.
