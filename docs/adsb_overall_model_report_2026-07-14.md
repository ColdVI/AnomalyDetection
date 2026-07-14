# ADS-B anomaly detection — genel teknik sonuç ve yorum raporu

Tarih: 2026-07-14  
Aktif aday: `contextual_physics_v1`  
Durum: **eğitildi, magnitude gate geçti, henüz threshold/alarm kararı dondurulmadı**

## 1. Önce kısaltmalar

- **Automatic Dependent Surveillance–Broadcast (ADS-B):** Uçağın kimlik, konum, hız ve benzeri durum bilgilerini yayınladığı gözetim sistemi.
- **Long Short-Term Memory (LSTM):** Sıralı/zamansal veride geçmiş bilgiyi taşıyan sinir ağı.
- **Neural Network (NN):** Sinir ağı.
- **Median Absolute Deviation (MAD):** Medyandan mutlak sapmaların medyanı; uç değerlere dayanıklı ölçek ölçüsü.
- **Negative Log-Likelihood (NLL):** Modelin gözlenen değere verdiği olasılıksal uyumsuzluk kaybı.
- **Area Under the Receiver Operating Characteristic Curve (AUROC):** Rastgele seçilen bir anomalinin normalden daha yüksek skor alma olasılığı olarak yorumlanabilen, threshold'dan bağımsız sıralama ölçüsü.
- **Area Under the Precision–Recall Curve (AUPRC):** Precision-recall eğrisinin alanı; sınıf dengesizliğinde pozitif sınıf performansını özetler.
- **Cumulative Sum (CUSUM):** Küçük fakat sürekli sapmaları zaman boyunca biriktiren yöntem.
- **Navigation Integrity Category (NIC), Navigation Accuracy Category for Position (NACp), Source Integrity Level (SIL):** ADS-B mesajının bütünlük/doğruluk beyan alanları.

## 2. Sonuç özeti

Yeni model, önceki autoencoder'ların yaptığı gibi ham irtifa/hız/track'i yeniden kurmuyor. Aynı uçuşun önceki **12 satırına** bakıp bir sonraki satırdaki **beş ayrı fiziksel residual kanalının** beklenen merkezi ve normal belirsizliğini tahmin ediyor. Bu nedenle gerçek bir **uçuş-içi time series** modelidir.

Eğitim, fit rolündeki uçuşların önceden dondurulmuş deterministik %2 örneği olan 2.929 uçuş, 1.267.625 satır ve epoch başına 1.180.160 pencereyle tamamlandı. Weighted Gaussian NLL `0,795375 -> 0,708696` düştü; göreli düşüş yaklaşık **%10,90**. Ayrı 770 doğal calibration-diagnostic uçuşundaki 332.510 pencere için trained-vs-untrained ve trained-vs-magnitude Spearman korelasyonları sırasıyla **0,649633 / 0,654240** oldu. Önceden dondurulmuş `rho >= 0,8` magnitude-domination sınırı aşılmadı; gate **PASS**.

Bu, modelin anomaly yakaladığı anlamına gelmez. Bugünkü doğru sonuç şudur:

> Model normal uçuş zaman serisinden bağlama bağlı next-step beklenti ve kanal-bazlı sürpriz üretmeyi öğrendi; skorun yalnız ham büyüklüğün kopyası olduğuna dair test geçildi. Alarm threshold'u, doğal alarm yükü ve truth-v2 anomaly recall henüz ölçülmedi.

## 3. Uçuşun kendi içindeki anomaliye bakıyor muyuz?

**Skorlarken evet; eğitirken anomaly etiketiyle hayır.** Ayrım önemlidir.

Her hedef satır `t` için yalnız aynı `flight_id` içindeki önceki 12 satır `[t-12, ..., t-1]` modele girer. Uçuş sınırından pencere geçmez. Zaman ters giderse veya ardışık iki mesaj arası 60 saniyeyi aşarsa pencere üretilmez. Model mevcut satırdaki fiziksel residual'ı önceden görmeden tahmin eder.

Modelin ağırlıkları uçuş sırasında yeniden eğitilmez. Dolayısıyla yaptığı karşılaştırma iki parçalıdır:

1. bütün doğal-fit uçuşlarından öğrendiği genel, bağlama-koşullu normallik;
2. o uçuşun son 12 satırındaki kendi yakın geçmişi.

Eğitime anomaly etiketi verilmedi ve sentetik eğitim satırı **0**'dır. Eğitim verisi `natural_clean_fit` kabul edilen uçuşlardan gelir. Eğer bu havuzda bilinmeyen gerçek anomaliler varsa contamination riski vardır: model bunların sık olanlarını normal sanabilir. Normal-only novelty detection'ın temel riski budur.

Uzun süre devam eden bir bozulma, 12 satırlık geçmişi zamanla kendisiyle kirletebilir ve anlık skoru düşürebilir. Bu yüzden tek bir NN skoru yeterli değildir: spike için anlık, freeze/bias için persistence, stealthy ramp için gerçek saniyeyle CUSUM-benzeri accumulation gerekir.

## 4. Model neye bakıyor?

Model doğrudan “uçak anormal mi?” sorusunu öğrenmiyor. Önce fiziksel tutarsızlıklar hesaplanıyor:

- vertical-rate residual = bildirilen dikey hız − `delta altitude / delta time`;
- speed residual = bildirilen yer hızı − ardışık konumdan türetilen hız;
- heading residual = bildirilen track − ardışık konumun dairesel bearing'i;
- east/north velocity residual = bildirilen hız vektörü − konum geçişinden türetilen hız vektörü.

`altitude_source_residual` barometrik/geometrik irtifa farkının zaman türevidir; fit MAD değeri tam sıfır çıktığı için yapay floor verilmeden dışlandı. Aktif kanal sayısı bu nedenle altı değil **beş** oldu.

Model girdisi 19 boyutludur: beş residual, gerçek `log(1 + delta-t)`, track'in `sin/cos` gösterimi, yalnız geçmişten çıkarılmış flight phase, cadence bucket'ları ve her girdinin availability maskesi. Track'in 359 derece ile 1 derece arasındaki dairesel yakınlığı `sin/cos` ile korunur.

## 5. Matematik: eğitim ve anomaly skoru

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

## 6. Sayılar bize ne söylüyor?

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

## 7. Eski adaylarla overall durum

| Sistem | Bilinen sonuç | Karar |
|---|---|---|
| Eski Dense-AE / LSTM-AE / LSTM-forecaster | pooled tarihsel AUROC 0,572 / 0,568 / 0,552; magnitude rho 0,84–0,94 | Detection adayı olarak güvenilmez |
| Residual rule, corrected truth-v2 | AUROC 0,764883; AUPRC 0,883313 | Ayrıştırma güçlü, fakat doğal burden 4,808533 episode/saat ve alarm gören uçuş oranı 0,892356 |
| Eski vector CUSUM h=1 | doğal günlerde alarm gören uçuş oranı yaklaşık 0,991 | Doygun; Step-7 FAIL |
| Yeni contextual LSTM | magnitude rho 0,649633 / 0,654240; gate PASS | Umut verici eğitim, henüz AUROC/AUPRC veya recall yok |

Eski residual rule corrected truth-v2'de ground-speed event recall `0,963659`, track event recall `0,951804` verdi; stealthy ramp recall `0,801347` görünse de aktif-aralık micro coverage yalnız `0,183902` idi. Yani bir olaya bir kez dokunmak ile olay boyunca faydalı biçimde alarmda kalmak aynı şey değildir. Overall karar bu yüzden yalnız AUROC veya event recall ile verilmez.

## 8. Anomaly tespit ederken hangi kanala ve zamansal davranışa dikkat edeceğiz?

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

## 9. Modelin şu an verdiği sonuç tam olarak nedir?

Modelin ürün çıktısı henüz `anomaly=yes/no` değildir. Her scoreable satır ve kanal için şu paket olmalıdır:

```text
flight_id, target_time, channel, phase, cadence,
predicted_mu, predicted_sigma, observed_scaled_residual,
standardized_surprise, conformal_p_value,
temporal_evidence, alarm
```

Bugünkü checkpoint conformal öncesi model/skor alanlarını üretebilir; gerçek conformal p-değeri, temporal evidence ve alarm için operasyonel bütçe/config dondurulmamıştır. Dolayısıyla bu model için bugün AUROC, AUPRC, event recall veya “%X anomaly yakalıyor” sayısı söylemek sayı uydurmak olur.

## 10. Sonraki zorunlu karar ve değerlendirme sırası

Sonuç çıkarma sırası değiştirilmemelidir:

1. Kullanıcı kabul edilebilir **toplam doğal alarm burden** değerini tanımlar; örneğin `100 scoreable uçuş-saatinde en fazla X operator-facing episode`.
2. Bu toplam bütçe beş residual channel ve S2 reason'ları arasında önceden paylaştırılır.
3. Minimum conditional-calibration desteği ve anomaly başına instant/persistence/accumulation eşikleri truth-v2 açılmadan dondurulur.
4. Ayrı doğal calibration ile conformal tail kurulur.
5. Natural development ve donmuş rehearsal üzerinde episode/saat, alarm gören uçuş oranı ve cadence/phase kararlılığı ölçülür.
6. Ancak doğal burden geçerse truth-v2 üzerinde event recall, first-alarm delay, active coverage, AUROC/AUPRC ve channel attribution çıkarılır.
7. Üçlü kör holdout havuzu ayrı unseal onayı olmadan açılmaz.

Bu sırada eksik olan tek bir “istatistiksel metrik” değil, detector'ın operasyonel maliyet sözleşmesidir. Bir threshold'u anomaly sonuçlarına bakıp seçmek kolayca iyi görünen ama sahada alarm yağdıran bir sistem üretir; önceki rule ve CUSUM sonuçları bunun sayısal örneğidir.
