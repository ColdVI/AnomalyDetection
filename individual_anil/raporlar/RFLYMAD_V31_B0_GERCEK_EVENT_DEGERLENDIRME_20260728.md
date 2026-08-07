# RflyMAD v3.1 B0 gerçek-event değerlendirmesi

Tarih: 2026-07-28
Namespace: `rflymad_probabilistic_v31_b0_event_v1`
Durum: **Tamamlandı; B0 operasyonel terfi için NO-GO**
Final holdout: **553 final-fault-test uçuşuna erişilmedi**

## Yönetici özeti

Tamamlanmış epoch-30 RflyMAD v3.1 B0 modeli yeniden eğitilmeden gerçek arıza
zamanlarına bağlandı. 90 normal-validation uçuşunda dondurulmuş alarm ızgarası
kalibre edildi; 91 bağımsız normal-test uçuşunda yanlış alarm yükü ve 557
anomaly-development uçuşunda gerçek-event ölçümleri üretildi.

Modelde yararlı bir sinyal vardır, fakat sinyal group/domain değişiminde kararlı bir
operasyonel alarm bütçesine dönüşmemektedir. Primary 1 event/normal-saat noktasında
en dengeli görünen K-of-N politikası anomaly-dev eventlerinin `%43,27`'sini
yakalarken normal-testte `0,654 event/saat` üretmiştir. Buna karşılık persistence ve
CUSUM aynı validation bütçesi altında normal-testte sırasıyla `4,577` ve `5,884
event/saat` üretmiştir. En yüksek gözlenen recall `%58,71` olsa da bunun bedeli
`9,153 false event/saat`tir. Test sonuçlarından kazanan seçilmemiştir.

Bu sonuç önceki `any-window` flight diagnostiklerini geçersiz kılmaz; onları doğru
seviyeye yerleştirir. Önceki `%84,62 normal flight alarm fraction`, debounced event
yükü değil, uçuşta en az bir ham eşik aşımıydı.

## 1. Değiştirilmeyen model ve veri sözleşmesi

- Temel koşu: `rflymad_colab_l4_20260728`.
- Seçili checkpoint: epoch 30; model, scaler ve feature listesi değiştirilmedi.
- Magnitude-domination: `false`; kapı PASS.
- Skor standardizasyonu yalnız 70.486 validation penceresinde fit edildi:
  median `-0,692386`, `1,4826 × MAD = 0,679642`.
- Görünür skor defteri: 681.690 pencere / 738 uçuş / 16,7 MB.
- Roller: 90 validation, 91 normal-test, 557 anomaly-dev.
- Gerçek truth: parse edilmiş `fault_active | condition_active`; bütün uçuşu
  anomalili sayma ve point-adjustment uygulanmadı.
- Final-fault-test telemetrisi hiçbir kod yolunda açılmadı.

Ön-kayıt ve grid sonuç çalıştırılmadan önce
`FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_B0_EVENT_PREREG_20260728.md` ile
`rflymad_probabilistic_v31_event_grid.json` dosyalarında donduruldu.

## 2. Dondurulmuş değerlendirme kapsamı

Validation üzerinde 144 aday hesaplandı. Üç alarm ailesi, üç false-event bütçesi
ve iki refractory süresi toplam 54 rapor noktası oluşturdu. CUSUM'un `0,5
event/saat` bütçesindeki altı noktası, dondurulmuş aday h değerlerinden hiçbiri
bütçeyi sağlayamadığı için ön-kayıt gereği `unavailable` bırakıldı; ekstrapolasyon
yapılmadı. Kalan 48 noktanın tamamı raporlandı.

Bağımsız normal-test toplam exposure yalnız `1,5295 saat`tir. Bu nedenle
false-event/saat kestirimleri birkaç olaya duyarlıdır ve uçuş-bootstrap aralıkları
geniştir. Bu veri gerçeği gizlenmemeli; oranların yanında event sayısı ve uçuş yükü
de verilmelidir.

## 3. Primary 1 event/saat, 30 saniye refractory görünümü

| Alarm ailesi | Politika | Val event/saat | Normal-test event/saat | Normal uçuşta ≥1 false event | Event recall | Medyan gecikme |
|---|---|---:|---:|---:|---:|---:|
| K-of-N | 2-of-3 | 0,511 | **0,654** | **%1,10** | **%43,27** | 3,2 s |
| K-of-N | 3-of-5 | 0,000 | 0,654 | %1,10 | %35,01 | 5,9 s |
| K-of-N | 5-of-10 | 0,000 | 0,654 | %1,10 | %27,29 | 5,1 s |
| CUSUM | allowance 0,25 | 0,511 | 5,884 | %9,89 | %48,29 | 36,8 s |
| CUSUM | allowance 0,50 | 0,511 | 5,884 | %9,89 | %48,29 | 36,8 s |
| CUSUM | allowance 1,00 | 0,511 | 5,884 | %9,89 | %47,76 | 36,3 s |
| Persistence | 0,5 s | 0,511 | 4,577 | %7,69 | %42,19 | 6,3 s |
| Persistence | 1,0 s | 0,511 | 3,923 | %6,59 | %24,96 | 17,7 s |
| Persistence | 2,0 s | 0,511 | 6,538 | %10,99 | %25,67 | 18,1 s |

K-of-N 2-of-3 referans noktasının kaynak-bootstrap `%95` aralıkları:

- false event/saat: `[0,000; 1,965]`;
- event recall: `[%40,04; %46,86]`;
- medyan gecikme: `[1,3; 4,0] s`;
- time-weighted range precision: `[%69,47; %86,28]`;
- time-weighted range recall: `[%0,65; %1,14]`.

Yüksek range precision, yüksek coverage anlamına gelmemektedir. Alarm parçaları
fault aralığının doğru bölgelerine düşse de toplam fault süresinin çok küçük bir
bölümünü kaplamaktadır.

## 4. Recall–false-alarm değiş tokuşu

Dondurulmuş gridde en yüksek recall, K-of-N 5-of-10 / 2 event-saat / 10 saniye
refractory noktasında `%58,71`dir. Bunun bağımsız normal-test yükü `9,153
event/saat`, en az bir false event gören normal uçuş oranı `%12,09`dur. `%95`
bootstrap aralıkları recall için `[%55,12; %62,30]`, false-event/saat için
`[4,569; 15,035]`tir. Bu nokta bir terfi adayı değil, trade-off eğrisinin yüksek
alarm-yüklü ucudur.

Primary K-of-N 2-of-3 noktasında fault-family ve domain ayrımı:

| Kırılım | Uçuş | Event recall | Medyan gecikme |
|---|---:|---:|---:|
| Motor | 243 | %60,08 | 0,1 s |
| Environment | 69 | %53,62 | 15,1 s |
| Propeller | 129 | %39,53 | 3,2 s |
| Voltage | 6 | %16,67 | 3,5 s |
| Sensor | 110 | **%5,45** | **41,0 s** |
| HIL | 248 | %56,85 | 0,1 s |
| SIL | 245 | %38,78 | 17,4 s |
| Real | 64 | **%7,81** | **41,2 s** |

Toplam skor tek başına yeterli değildir: temel model özellikle Real domain ve
Sensor fault ailesine genellenememektedir. Bu, yalnız eşik ayarıyla çözülecek bir
sorun görüntüsü vermemektedir.

## 5. Karar

**B0 için operasyonel terfi NO-GO'dur.** Gerekçeler:

1. Validation bütçesi persistence/CUSUM ailelerinde bağımsız normal-testte ciddi
   biçimde taşmaktadır; group/domain shift vardır.
2. Bütçeyi en iyi koruyan K-of-N noktasında recall yalnız `%43,27`dir.
3. Real ve Sensor recall'ı sırasıyla `%7,81` ve `%5,45`tir.
4. Fault interval coverage düşüktür; iyi görünen range precision tek başına
   yanıltıcıdır.
5. Normal-test exposure kısa olduğu için false-alarm belirsizliği büyüktür.

Bu karar grid veya parametre değiştirilmeden kaydedilmiştir. Final holdout hâlâ
mühürlüdür ve bu NO-GO kararını düzeltmek için açılamaz.

## 6. Meşru sonraki çalışma

Yeni threshold/epoch araması yapılmamalıdır. Sonraki araştırma yeni namespace ve
ön-kayıtla B1 olarak kurulmalıdır:

1. false eventlerin takeoff/landing/manevra geçişleri ve SIL/HIL/Real gruplarındaki
   dağılımını incelemek;
2. aynı tek modele causal flight-phase/domain context eklemek;
3. dört ayrı phase modeli yerine önce ortak model + phase context kullanmak;
4. ardından control/setpoint residual, actuator, estimator innovation, IMU
   transient, energy ve environment bloklarını tek tek ablate etmek;
5. Gaussian scale-floor çökmesi ve kanal-bazlı NLL katkılarını ayrıca raporlamak.

## 7. Üretilen kanıtlar

`artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/` altında:

- `score_streams.parquet` ve `score_export_manifest.json`;
- `calibration_candidates.csv` ve `operating_grid.csv`;
- `event_metrics.csv`, `domain_family_breakdown.csv`;
- `source_event_metrics.parquet`, `bootstrap_intervals.json`;
- `evaluation_report.json`;
- `plots/01_recall_vs_false_event_rate.png`;
- `plots/02_reference_budget_event_recall.png`;
- `plots/03_normal_test_alarm_burden.png`;
- `plots/04_normal_false_event_timeline.png`;
- `plots/05_detected_motor_event_timeline.png`;
- `plots/06_missed_real_sensor_timeline.png`.

## 8. Provenance olayı

Colab run manifestinin gömdüğü preregistration byte-hash'i, daha sonra commitlenen
checkout'taki aynı adlı v3.1 karar belgesinin byte-hash'iyle eşleşmemektedir. Bu
kusur saklanmamıştır. Kabul edilen koşunun gömülü sözleşmesi kendi içinde
tutarlıdır; config hash'i, roller hash'i, veri fingerprintleri, model/scaler ve run
artifact hash'leri eksiksiz eşleşmektedir. Event evaluator mevcut checkout
sözleşmesini sessizce yeniden imzalamamış; temel run'ın `6c10569e...` gömülü
contract'ını ve bu provenance farkını birlikte kaydetmiştir.

| Kanıt | SHA-256 |
|---|---|
| Run içine gömülü prereg hash | `9674a1a91da2427ad45a81d8492bbc653008f09dd0312e88fac100a78a889c5c` |
| Sonraki checkout prereg hash | `21b4c99cf03e0f316418340fd814e5195b61579bea16d439ea9d70f29d58a086` |
| Değişmeyen config hash | `36fd8d78bb9ca098cfab6a5f7c3330702dabf3f6aad001188ee0634fb89932a4` |
| Run'ın gömülü contract hash'i | `6c10569e2b99c885b662bffcd7f554799582c28ebcb0536a998b51a8624d4ba2` |
| Checkout'tan ham yeniden hesaplanan contract | `08bdbff46aab9fef27a5d98d036e901132ab1b091c1598daf140a1950b1f7eda` |

Bu olay gelecekte contract tasarımında yalnız dosya byte-hash'i yerine normalize
edilmiş içerik hash'i, artifact içine prereg snapshot kopyası ve açık
`contract_version` saklanması gerektiğini gösterir.
