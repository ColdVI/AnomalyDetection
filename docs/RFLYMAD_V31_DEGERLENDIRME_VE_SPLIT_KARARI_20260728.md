# RflyMAD v3.1 — değerlendirme, yanlış-alarm teşhisi ve split kararı

Tarih: 2026-07-28

Statü: Magnitude gate PASS; dondurulmuş çalışma noktası operasyonel NO-GO;
553 uçuşluk final-fault-test kapalıdır.

## 1. Koşunun kapsamı

RflyMAD probabilistic v3.1 modeli yalnız normal eğitim uçuşlarında, NVIDIA L4
üzerinde 30 epoch eğitildi. Checkpoint yalnız normal-validation Gaussian NLL ile
seçildi; en düşük değer epoch 30'da `-0,150592` oldu. Dondurulmuş normal-validation
0,995 quantile alarm eşiği `10,094001` olarak uygulandı.

Development değerlendirmesi:

- 91 group-safe normal-test uçuşu;
- 557 anomaly-development uçuşu;
- 553 mühürlü final-fault-test uçuşu; erişilmedi.

Kaynak koşu:
`artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728`.

## 2. Magnitude diagnostic

| Karşılaştırma | Spearman rho | Dondurulmuş kapı |
|---|---:|---:|
| Trained skor / random-init skor | 0,3288 | < 0,8 PASS |
| Trained skor / standardized hedef büyüklüğü | 0,3495 | < 0,8 PASS |

`magnitude_domination_flagged_at_0_8=false` olumlu fakat sınırlı bir kontroldür.
Model skorunun bu iki vekille yüksek korelasyonlu olmadığını gösterir; doğru
eşik, kabul edilebilir yanlış alarm veya güçlü anomaly ayrımı kanıtlamaz.

## 3. Uçuş düzeyi sonuç

| | Alarm yok | Alarm var |
|---|---:|---:|
| Normal | TN=14 | FP=77 |
| Anomaly-development | FN=95 | TP=462 |

Türetilen ölçümler:

| Ölçüm | Değer |
|---|---:|
| Sensitivity / anomaly recall | %82,94 |
| Specificity | %15,38 |
| Normal uçuş alarm oranı | %84,62 |
| Balanced accuracy | %49,16 |
| Accuracy | %73,46 |
| Precision | %85,71 |
| Flight ROC-AUC | 0,6411 |
| Flight average precision | 0,9211 |

Değerlendirme havuzunun 557/648'i, yani %85,96'sı anomalilidir. Rastgele bir
sıralayıcının AP tabanı yaklaşık bu prevalanstır. AP'nin 0,9211 olması tabanın
yalnız yaklaşık 0,061 üzerindedir; tek başına güçlü başarı olarak sunulamaz.
Her uçuşu anomaly sayan bir sistem yaklaşık %85,96 accuracy elde edeceği için
mevcut %73,46 accuracy da yanıltıcıdır. Ana bulgu %82,94 recall değil, normal
uçuşların %84,62'sinde alarm üretilmesidir.

## 4. Neden bu kadar çok normal uçuş alarm aldı?

Mevcut evaluator, uçuşta tek bir pencere eşik aşarsa uçuşu alarm verilmiş sayar.
Normal-validation eşiği pencere bazında 0,995 quantile iken uzun bir uçuşta çok
sayıda pencere denenir. Bağımsızlık varsayımı gerçekçi olmasa da
`1 - 0,995^N` ifadesi çoklu-deneme etkisinin yönünü gösterir: pencere sayısı
arttıkça en az bir aşım olasılığı büyür.

Olası ve ölçülmesi gereken katkılar:

1. Normal train/validation gruplarının normal-test senaryolarını kapsamaması.
2. Ground/takeoff/cruise/landing rejimlerinin tek dağılım gibi ölçeklenmesi.
3. Uçuş uzunluğu ve pencere sayısının `any alarm` kararını domine etmesi.
4. Tek pencerenin uçuş alarmı için yeterli sayılması.
5. Öğrenilen Gaussian sigma değerlerinin bazı rejimlerde aşırı dar olması.
6. Görülmemiş normal scenario/domain aileleri.
7. Global pencere quantile'ının uzun validation uçuşları tarafından domine
   edilmesi.

Bu olasılıklar sonuç değildir; yeni development tanısallarında ayrı ayrı
ölçülmelidir.

## 5. Fault ailesi tanısalı

| Etiket | Uçuş | Bayraklanan | Oran | Median max score |
|---|---:|---:|---:|---:|
| Environment | 69 | 68 | %98,55 | 123,61 |
| Motor | 243 | 204 | %83,95 | 114,30 |
| Propeller | 129 | 122 | %94,57 | 96,07 |
| Sensor | 110 | 65 | %59,09 | 23,76 |
| Voltage | 6 | 3 | %50,00 | 8,91 |
| NoFault | 91 | 77 | %84,62 | 42,89 |

Voltage satırı yalnız altı uçuş içerdiğinden genellenebilir bir fault-family
iddiası değildir.

## 6. V2 source-level ile v3.1 group-safe split karşılaştırması

### V2 balanced source-level

- 309 normal train, 66 normal validation;
- primary test 66 normal + 66 anomaly;
- exact `source_id` örtüşmesi yok;
- SIL/HIL/Real normal dağılımı stratified;
- interval ROC-AUC 0,907, flight-mean AUC 0,915;
- dondurulmuş event referansında 37/66 event recall (%56,1) ve
  0,768 false event/saat.

V2 satır-random veya aynı-uçuş sızıntılı değildir. Ancak canonical
scenario/domain ailelerini v3.1 kadar katı biçimde ayırmaz; daha iyimser in-domain
development baseline'ıdır.

### V3.1 group-safe

- 260 normal train / 7 grup;
- 90 normal validation / 3 grup;
- 91 normal test / 3 ayrı grup;
- 557 anomaly-development / 12 grup;
- 553 final-fault-test / 13 grup ve SHA-256 mührü;
- grup ve kaynak dengeli epoch sampler.

V3.1'in daha kötü görünmesi splitin daha kötü olduğu anlamına gelmez. Aksine,
görülmemiş normal gruplara genelleme zayıflığını görünür kılmıştır. Resmî split
v3.1 group-safe kalmalıdır; v2 yalnız iyimser in-domain referans olarak korunur.

V2 ve v3.1 sonuçları doğrudan aynı metrik değildir. V2'nin güçlü sonucu gerçek
interval/event değerlendirmesinden, v3.1'in mevcut alarm oranı ise
`any-window -> flight` özetinden gelir. Performans düşüşünü yalnız splite
bağlamak için aynı model, seed, sampler, checkpoint, threshold ve event
aggregation ile yalnız splitin değiştiği kontrollü ablation gerekir.

## 7. Test verisine temas kuralı

77 normal false-positive uçuşu ayrıntılı incelemek yararlıdır; fakat bu inceleme
sonucuna göre model, eşik, event kuralı veya feature değiştirilirse 91 uçuşluk
normal-test artık test değildir. Böyle bir çalışma yapılacaksa bu 91 uçuş yeni
namespace'te development olarak yeniden sınıflandırılmalı ve yeni görülmemiş
normal holdout ayrılmalıdır. Final-fault-test bu sırada yine kapalı kalır.

## 8. Sonraki doğru deney

Yeni model karmaşıklığına geçmeden önce yeni ve sonuç-öncesi dondurulmuş bir
development protokolü şu tanısalları üretmelidir:

1. Train/validation/test normal skor dağılımları grup, domain ve scenario bazında.
2. Uçuş süresi/pencere sayısı ile alarm olasılığı ilişkisi.
3. Tahmin sigma dağılımı, sigma-floor temas oranı ve standardized residual.
4. %50/%80/%90/%95 prediction-interval coverage.
5. Pencere alarmından event üretmek için persistence, CUSUM ve debounce.
6. Önceden dondurulmuş %1/%5/%10 normal-flight alarm ve false-event/saat
   bütçelerinde anomaly-development recall eğrisi.
7. Nedensel ground/takeoff/cruise/landing context ablation'ı; ilk baseline tek
   modele flight-phase context eklemeli, doğrudan dört ayrı modelle başlamamalı.

Checkpoint seçimi normal-validation NLL'de kalmalıdır. Anomaly-development
epoch seçmek için kullanılamaz; yalnız tanısal değerlendirme ve önceden
tanımlanmış aday karşılaştırması için kullanılabilir.

## 9. Nihai karar

> RflyMAD probabilistic v3.1 group-safe baseline: magnitude PASS, zayıf fakat
> sıfır olmayan sıralama sinyali, başarısız operating point ve normal-group
> genellemesi; operasyonel NO-GO. Normal-only paradigma korunabilir. Final test
> kapalı kalmalıdır.

Ana baseline raporu:
`docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`.
