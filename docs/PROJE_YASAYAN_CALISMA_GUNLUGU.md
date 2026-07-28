# Proje yaşayan çalışma günlüğü — uçtan uca kronoloji, yöntemler ve bulgular

İlk yayın: 2026-07-28

Son güncelleme: 2026-07-28

Durum: Yaşayan ana kayıt. Yeni deneyler tamamlandıkça eski sonuçlar silinmeden,
yeni tarihli bölümlerle güncellenecektir.

## 1. Belgenin amacı

Bu belge, proje boyunca yapılan çalışmaların sunum ve nihai rapor için tek bir
kronolojik omurgada tutulması içindir. ALFA, UAV Attack, UAV-SEAD, RflyMAD ve
ADS-B hatlarında yapılan veri hazırlama, provenance, pruning, feature
engineering, modelleme, kalibrasyon, evaluator düzeltmeleri, split değişimleri,
GPU koşuları ve NO-GO kararlarını birbirine bağlar.

Bu belge bir “en iyi skorlar listesi” değildir. Düşük veya başarısız sonuçlar
silinmez; çünkü projenin ana bilimsel çıktılarından biri, yüksek görünen bazı
sonuçların etiket, ölçek, split veya metrik düzeyi düzeltildiğinde neden
düştüğünün gösterilmesidir.

## 2. Okuma ve raporlama kuralları

Her sonuç şu beş eksenle birlikte okunmalıdır:

1. **Veri rolü:** train, validation/calibration, development, rehearsal, test,
   stress veya sealed final holdout.
2. **Birim:** satır, pencere, gerçek zaman aralığı, event veya uçuş.
3. **Ground truth:** uçuş etiketi, gerçek onset/interval, proxy veya sentetik
   enjeksiyon.
4. **Split bağımsızlığı:** source-safe, session/campaign-safe, group-safe veya
   domain-transfer.
5. **Seçim disiplini:** feature, checkpoint, threshold ve event kuralının hangi
   veri görülmeden önce dondurulduğu.

Satır ROC-AUC, interval ROC-AUC, event recall ve uçuş ROC-AUC birbirinin yerine
kullanılmaz. `Herhangi bir pencere alarmı -> uçuş alarmı` sonucu debounce
edilmiş event confusion matrix değildir.

## 3. Bir bakışta ana zaman çizgisi

```text
Veri/platform altyapısı ve provenance
  -> ALFA ilk normal/anomaly modelleri
  -> UAV Attack saldırı görünürlüğü denemeleri
  -> UAV-SEAD veri büyütme + feature/model turları
  -> RflyMAD gerçek/simülasyon fault turları
  -> ortak pruning, causal truth ve magnitude denetimleri
  -> 2026-07-10 ADS-B için temiz namespace/reset
  -> ADS-B basit fizik kuralları
  -> ADS-B contextual_physics v1
  -> ADS-B contextual_physics v2, 8 epoch L4
  -> dört UAV veri setinde ortak probabilistic GPU v1/v2
  -> interval/event evaluator düzeltmesi
  -> group-safe v3 ve v3.1
  -> ALFA group-CV NO-GO + RflyMAD group-safe operating-point NO-GO
```

“Dört veri setine dönme/indirgenme”, ADS-B verisinin dört parçaya indirilmesi
değildir. ADS-B hattından öğrenilen normal-only Gaussian forecasting, robust
ölçekleme, magnitude diagnostic ve event-evaluation disiplininin dört etiketli
UAV veri setinde ortak bir baseline olarak tekrar sınanmasıdır.

## 4. Veri ve altyapı temeli

### 4.1 Veri kaynakları

- **ALFA:** sabit-kanat İHA; motor ve kumanda yüzeyi arızaları.
- **UAV Attack:** PX4 çok-rotorlu; benign, GPS spoofing, jamming ve Ping DoS.
- **UAV-SEAD:** ULog tabanlı; irtifa, external/global position ve mekanik
  anomaliler.
- **RflyMAD:** SIL, HIL ve Real uçuşlar; motor, propeller, sensor, voltage ve
  environment fault aileleri.
- **ADS-B/adsb.lol:** büyük hacimli gerçek hava trafiği; doğal anomaly ground
  truth'u yoktur.

Ham ULog, CSV, ROS bag, MAT ve telemetri arşivlerinin tamamı doğrudan optimizer'a
verilmedi. Ham dosyalar veri kaynağıdır; model girdisi parser, birim dönüşümü,
zaman hizası, kalite kontrolü ve feature üretiminden sonra oluşan Silver/Gold
veya parsed tablolardır.

### 4.2 Bronze/Silver/Gold ve provenance

Proje üç katmanlı veri mimarisi kullandı:

- Bronze: ham ve değişmemiş kaynak;
- Silver: parse edilmiş, birimleri dönüştürülmüş ve provenance kolonlu veri;
- Gold/parsed: model için ortak veya dataset-specific feature tablosu.

Kaynak kimliği, role üyeliği, config/model/scaler hashleri, run manifestleri ve
SHA-256 zincirleri saklandı. ADS-B sentetik corpusunda fail-if-exists ve
`contains_synthetic`/`data_role` kontrolleriyle sentetik verinin train/scaler/
calibration'a girmesi kodla engellendi.

2026-07-10'da eski non-ADS-B ML hattı ve iki ilk ADS-B yaklaşımı `archive/`
altına alındı. Yeni ADS-B hattı temiz namespace ile sıfırdan kuruldu; arşiv kodu
aktif modele geri bağlanmadı.

## 5. İlk veri seti turları

### 5.1 ALFA ile eğitim ve değerlendirme

İlk ALFA turlarında monolitik satır-bazlı Isolation Forest rastgele seviyeye
yakın sonuç verdi. Modüler feature grupları ve validation-normalize füzyonla
uçuş ROC-AUC 0,833 görüldü. İlk LSTM-AE, küçük veri koşulunda 0,731 ile IF'in
gerisinde kaldı.

ROS bag genişletmesiyle veri büyüdüğünde LSTM-AE 0,918 uçuş ROC-AUC'ye çıktı;
bu proje içindeki az sayıdaki net derin öğrenme iyileşmesinden biridir. Ancak
sonraki causal truth düzeltmeleri daha önce yüksek görünen sonuçların bir kısmını
düşürdü:

- `alt_error_cusum` uçuş ROC-AUC: 0,878 -> 0,611;
- overlap/point-adjust benzeri event recall: 0,594 -> 0,194–0,224.

RESIDUAL-V1 aşamasında bazı fizik residual'ları sanity kapılarını geçti; fakat
normal kalibrasyon maruziyeti yalnız 0,168846 saat iken 2 saat gerekiyordu.
Threshold dondurulamadı ve sonuç NO-GO oldu. ALFA'nın temel sınırı az bağımsız
normal session ve küçük fault aileleridir.

### 5.2 UAV Attack ile eğitim ve değerlendirme

UAV Attack havuzu 19 logdan oluşur: 6 normal, 6 Ping DoS, 6 GPS spoofing ve
1 GPS jamming. Monolitik IF satır ROC-AUC 0,21 verdi. Ping DoS'un bazı örneklerde
seçili fizik/telemetri kanallarına yeterince yansımadığı görüldü; bir saldırı
etiketinin varlığı, gözlenebilir fiziksel anomaly garantisi değildir.

Normal kaynak azlığı nedeniyle split ve calibration yapısal olarak zayıftır.
Son four-dataset probabilistic v2 değerlendirmesinde flight agregasyon AUC'leri
0,000, normal testte false event yükü 32,417/saat ve alarm time fraction %75,3
oldu. Group registry denetiminde yalnız bir saf-normal campaign bulunması,
ayrık train/validation/normal-test kurmayı imkânsızlaştırdı. Bu nedenle v3.1'de
ana training benchmark olmaktan çıkarıldı; campaign-safe yeni normal veri
gelene kadar external stress/exploratory statüsündedir.

### 5.3 UAV-SEAD ile eğitim ve değerlendirme

UAV-SEAD hattı 60 -> 179 -> 349 -> 611 -> 1.044 development uçuşuna kadar
büyütüldü; 899 normal development uçuşu kullanıldı, 200 uçuşluk blind holdout
açılmadı. Gün/oturum bazlı split seed varyansını yaklaşık ±0,212'den ±0,012'ye
indirdi; adil satır ROC-AUC 0,474 -> 0,799 yükseldi. Bu iyileşme modelden çok
veri temizliği ve split disiplininden geldi.

Denenen ana yöntemler:

- modüler Isolation Forest ve CUSUM/K-of-N karar katmanları;
- Dense-AE, LSTM-AE ve USAD;
- LightGBM supervised window modeli;
- EKF innovation/test-ratio ve motor-simetri feature'ları;
- Chronos zero-shot forecast residual;
- tek-feature `itki_komutu` ince modülü;
- mekanik/sistem iki-kanal füzyonu;
- daha fazla normal veri ve jackknife drift calibration.

Önemli sonuçlar:

- LightGBM AUPRC 0,349 ile IF 0,385'in gerisinde kaldı.
- EKF/motor feature mühendisliği kazanımları yaklaşık +0,02 düzeyindeydi.
- Chronos mekanik dalda recall 0,205 -> 0,390 ile yöntemsel Gate B'yi geçti;
  toplam füzyonda operasyonel kapıyı açmadı.
- Tek-feature `itki_komutu` 0,205 -> 0,459 ile en güçlü kategori kazancını
  verdi; normalde 38,1 false alarm/saat üretti.
- İki-kanal mimarisi recall'ı artırırken false alarmı 1,89–3,83 kat büyüttü.
- Normal veri 899'a büyütülünce false alarm 23,6 -> 9,95/saat düştü; recall
  0,21 -> 0,126 geriledi.

Dense/LSTM-AE/USAD skorları trained-vs-random ve trained-vs-raw-magnitude için
yaklaşık 0,964/0,965 Spearman korelasyonu verdi. Genlik-normalize `relerr`
skoru korelasyonu 0,15–0,23'e düşürdü fakat recall <%6 oldu. Bu, önceki
kazancın önemli bölümünün öğrenilmiş zamansal yapı değil ham büyüklük farkı
olduğunu gösterdi.

Probabilistic v2'nin gerçek interval değerlendirmesinde UAV-SEAD interval AUC
0,747 olsa da yalnız 7/206 event (%3,40) yakalandı ve time-weighted range recall
%0,264 kaldı. Sıralama sinyali vardır; operasyonel event detector yoktur.

V3 strict-parent denetiminde 1.244 ULog, 259 parent gruba indi; 510 normal kayıt
anomalili kayıtlarla mixed parent içinde kaldı. UUID çok kaba, tam parameter
hash neredeyse source-id kadar ince çıktı. Savunulabilir mission/session anahtarı
bulunamadığı için UAV-SEAD ana v3.1 eğitimi beklemektedir.

### 5.4 RflyMAD ile eğitim ve değerlendirme

RflyMAD Full-v2 6.605 canonical uçuş ve 1.225 locked-test içeren en büyük
etiketli fault hattıdır. Dar GNSS Gaussian forecaster pilotu ve sonrasında
normal temporal AE, supervised TCN, direct Dense/LSTM-AE/USAD ve robustness
turları çalıştırıldı.

İlk RFLY proxy değerlendirmesinde 0,749 recall görüldü. Gerçek
`fault_active/condition_active` interval truth kullanıldığında sonuç
0,526 recall / 22,28 false alarm-saat oldu. SEAD+RFLY havuzlaması 0,149 /
30,00 ile daha da kötüleşti; heterojen normal domainleri körlemesine birleştirmenin
zararı ölçüldü. Full-v2 robustness ve Real-transfer kapıları operasyonel GO
vermedi.

Daha sonra four-dataset probabilistic v2 source-level splitte RflyMAD en güçlü
araştırma sinyalini üretti:

- interval ROC-AUC 0,907;
- interval AP 0,805;
- flight-mean AUC 0,915;
- 37/66 event recall (%56,1);
- 0,768 false event/saat;
- range recall %16,54.

Bu sonuç koşullu araştırma GO, operasyonel olmayan baseline olarak kaydedildi.
Group-safe v3.1 sonucu §10'da ayrıca yer alır.

## 6. Proje boyunca pruning, feature engineering ve provenance

### 6.1 Veri pruning ve karantina

Pruning, sonucu güzelleştirmek için test örneği silmek anlamında kullanılmadı.
Uygulanan işlemler şunlardır:

- parse edilemeyen veya schema/quality kontrolü geçmeyen kaynakların karantinası;
- gerçek fault mesajıyla klasör etiketi çelişen örneklerin sonuç görülmeden
  dışlanması;
- sentetik anomaly'nin optimizer/scaler/calibration'dan yasaklanması;
- aynı source/session/campaign/group üyelerinin farklı rollere bölünmemesi;
- final holdoutların development tamamlanana kadar kapalı tutulması;
- ground/on-ground contamination'ın ayrı quarantine raporlarına yazılması.

### 6.2 Feature pruning

- Train MAD değeri sıfır veya `<=1e-6` olan kanallar dışlandı; yapay epsilon
  floor ile sahte hassasiyet oluşturulmadı.
- Tamamen boş veya train içinde gözlenmeyen kanallar aktif feature diye
  gösterilmedi. RflyMAD v3.1'de `battery_voltage`, `battery_current`, `gps_eph`
  ve `gps_epv` bu nedenle dışlandı.
- Missing değerler maskeyle birlikte modele verildi; future bilgiyle doldurma
  yapılmadı.
- Fault ID, dosya adındaki fault kodu, injection komutu ve onset metadata'sı
  model feature'ı olarak yasaklandı.

### 6.3 Feature engineering envanteri

Başlıca feature aileleri:

- konum/hız/ivme ve attitude;
- actuator output/control ve motor simetrisi;
- EKF innovation/test ratio;
- altitude/local-position ve vertical consistency;
- ADS-B vertical-rate, speed, heading, east/north velocity ve altitude-source
  fizik residual'ları;
- cadence, gap, phase ve veri-kalitesi bağlamı;
- fault'tan bağımsız flight-phase bağlamı için ground, takeoff/climb,
  cruise/maneuver ve descent/landing taslağı.

Feature engineering her zaman iyileştirme üretmedi. ML-9'da yeni fizik feature
kazanımları küçük kaldı; tek-feature modül güçlü sinyal buldu fakat false alarmı
yüksekti. Bu nedenle feature sayısının artması başarı sayılmadı.

### 6.4 Model/aday pruning

- USAD, ALFA'da LSTM-AE'nin gerisinde kaldığı ve ADS-B'de sayısal loss patlaması
  ürettiği için ana aday olmadı.
- Isolation Forest ADS-B contextual keşfinde magnitude rho yaklaşık 0,996
  verdiği için elendi.
- ADS-B reconstruction NN'leri magnitude gate'i geçmedi.
- Sonuç sonrası epoch/LR/threshold avı yasaklandı; başarısız adaylar yeni
  namespace ve yeni ön-kayıt olmadan yeniden açılmadı.

## 7. ADS-B pivotu ve basit anomaly detection

İlk ADS-B ML denemeleri %97,6 sentetik recall fakat 25,54 doğal alarm/saat
ürettiği için reddedildi. 2026-07-10 resetinden sonra fiziksel açıklanabilirlik,
doğal alarm yükü ve provenance önceliklendirildi.

Basit anomaly turu aynı dondurulmuş 100 doğal uçuş / 53.714 satır üzerinde iki
kuralı karşılaştırdı:

| Kural | Değerlendirilebilir uçuş | Triggerlı uçuş | Event | Bulgusu |
|---|---:|---:|---:|---|
| İrtifa sapması | 57 | 2 (%3,51) | 2 | Faz sınırı ve çok seviyeli cruise false positive |
| GPS/rota sapması | 95 | 13 (%13,68) | 24 | Tamamı düşük-hız bearing kararsızlığıyla uyumlu |

Doğrulanmış doğal anomaly bulunmadı. Rota kuralı ilerletilmedi; irtifa hattı
yalnız açıklanabilir keşif adayı olarak kaldı. Bu çalışma gerçek ground truth
içermediğinden daha sonraki truth-v2 sentetik recall sonuçlarıyla doğrudan
kazanan karşılaştırması değildir.

## 8. ADS-B contextual physics v1

ADS-B hattında üç büyük gün yaklaşık 256,15 milyon satır, 638 Silver parçası ve
yüz binlerce uçuş segmenti sağladı. Truth-v2 corpusu 8.910 uçuşta temiz ve beş
fizik-bozma reçetesini içerdi. İlk “onsetten dosya sonuna kadar anomaly” proxy'si
yerine `injection_active`, `observable_changed` ve `evaluable_truth` ayrıldı.

Öğrenmesiz robust-z kuralı pooled AUC 0,600 ile ilk NN adaylarını geçti; fakat
doğal alarm yükü yüksekti. Contextual-physics v1, uçuş phase/cadence bağlamlı
normal-only next-step Gaussian LSTM olarak kuruldu. Beş epoch eğitimde magnitude
rho yaklaşık 0,65 ile PASS verdi; projenin magnitude kapısını geçen ilk öğrenilmiş
ADS-B modeliydi.

Hiyerarşik conformal kalibrasyon ve Page-CUSUM geliştirildi. ADR-042'de
dondurulmuş dar alarm bütçesi altında 5 profilin 4'ü truth-v2 recall'da %6'nın
altında kaldı. East/north Page-CUSUM yaklaşık %49,7 recall ile belirgin istisna
oldu. Kök neden yalnız model değildi: “100 uçuş-saatinde çok az alarm” bütçesi,
tek 0,5–1 saatlik injection penceresinde alarm üretme olasılığını matematiksel
olarak aşırı kısıtladı.

## 9. ADS-B contextual physics v2 — veri genişletme ve 8 epoch

V2, mimariyi sonuç sonrası değiştirmek yerine ADR-042'nin istediği daha geniş
alarm bütçesi ve daha fazla doğal veriyle yeni ön-kayıt altında açıldı.

### 9.1 Veri genişletme

Yeni günler:

- 2024-09-01;
- 2025-02-15;
- 2025-06-15 (`-003` kaynak shard adı provenance içinde korundu).

Üç yeni tar Silver'a 0 hata ile işlendi; toplam Silver parça sayısı 1.185 oldu.
Fit-expansion manifest günleri contract ile doğrulandı. Epoch, feature, split,
alarm grid'i ve truth-v2 corpusu sonuç görülerek değiştirilmedi.

### 9.2 CPU incident ve recovery dersi

Yerel v4 CPU koşusu yaklaşık 24 saat 14 dakikada epoch 1'i tamamladı, epoch 2
sırasında exception veya nihai rapor bırakmadan sessizce sonlandı. Epoch 1:
236.129.563 pencere ve 461.574 batch. Checkpoint yazılmış fakat trainer'da
resume yolu olmadığı için operasyonel recovery eksikliği ayrıca kaydedildi ve
lineage doğrulayan resume yolu eklendi. Bu incident bilimsel model sonucu olarak
kullanılmadı.

### 9.3 Tamamlanan L4 eğitimi

Google Colab NVIDIA L4 koşusu dondurulmuş sekiz epochu tamamladı:

- epoch başına 236.129.563 pencere;
- toplam 1.889.036.504 window-epoch;
- epoch 1 NLL 0,81609 -> epoch 8 NLL 0,80753;
- sentetik training satırı 0;
- threshold selection eğitim sırasında yapılmadı;
- trained-vs-random rho 0,64989;
- trained-vs-target-magnitude rho 0,65731;
- `magnitude_domination_flagged_at_0_8=false`.

Model Colab'da eğitildi; çıktı ZIP'i yerel projeye alındıktan sonra contract ve
hashleri doğrulandı. Kalibrasyon ve evaluation yerel olarak devam etti.

### 9.4 Faz D–G

- Hierarchical conformal + değiştirilmeyen CUSUM + cumulative persistence_v2
  kalibre edildi.
- Doğal calibration formülü `reference_shift_multiplier=1,11` üretti.
- Mevcut 8.910-uçuş truth-v2 corpusu yeniden üretilmeden kullanıldı.
- Geniş alarm bütçesi `[0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500]`
  sonuç görülmeden dondurulmuştu.
- Model/CUSUM/persistence eventleri ortak simple-anomaly event şemasına çevrildi.
- Timeline, evaluation ve alarm PNG'leri üretildi.

ADR-046 sonucu:

- tek evrensel dedektör olarak NO-GO;
- recipe-bazlı araştırma devamı GO;
- speed-bias persistence, V=0,1'de %57,18 recall ve yaklaşık 0,0009 event/saat;
- position-ramp CUSUM, V=5'te %51,27 recall ve yaklaşık 0,0172 event/saat;
- daha gevşek bütçelerde recall yükseldi fakat tek operasyonel evrensel nokta
  kurulmadı.

## 10. Dört veri setine ortak probabilistic dönüş

### 10.1 Neden tekrar dört UAV veri seti?

ADS-B çalışmasının taşınan metodolojik bileşenleri şunlardı:

- normal-only causal next-step forecasting;
- kanal başına `mu` ve sınırlı `sigma` üreten Gaussian NLL;
- train-only robust median/MAD scaling;
- magnitude-domination diagnostic;
- pencere, interval, event ve flight metriklerini ayırma;
- threshold ve checkpointi testten ayırma;
- immutable contract ve hash doğrulaması.

Bu dört dataset turu, önceki modelleri geçersiz kılmak için değil, aynı
probabilistic baseline'ın datasetler arası davranışını ölçmek için açıldı.

### 10.2 “WindowScaling” tam olarak neydi?

Projede resmi adı `WindowScaling` olan tek bir algoritma yoktur. Kullanılan
tasarım üç parçadır:

1. 32 geçmiş satırlık nedensel pencere; uçuş ve 5 saniyelik gap sınırını geçmez.
2. Yalnız normal train'de median/MAD (`1,4826`) robust scaling; `[-5,5]` clip ve
   dejenere kanal dışlama.
3. Bir sonraki standardize feature vektörü için kanal başına Gaussian `mu/sigma`
   tahmini ve standardized surprise/NLL skoru.

Buna ek olarak v3.1'de uzun kaynak ve büyük grupların pencere sayısıyla optimizerı
domine etmemesi için önce grup, sonra kaynak dengeli epoch kotası kullanıldı.

### 10.3 Probabilistic v1 ve balanced v2

V1 mevcut splitleri kullanan ortak GPU smoke/baseline idi. V2 model ailesini
değiştirmeden normal kaynakları yaklaşık 70/15/15 train/validation/clean-test
olarak yeniden dengeledi ve anomaly primary testini label-stratified kurdu:

| Dataset | Normal train | Normal validation | Primary test | Normal/anomaly |
|---|---:|---:|---:|---:|
| ALFA | 10 | 2 | 9 | 3 / 6 |
| UAV Attack | 4 | 1 | 4 | 1 / 3 |
| UAV-SEAD | 629 | 135 | 268 | 134 / 134 |
| RflyMAD | 309 | 66 | 132 | 66 / 66 |

L4 üzerinde 30 epoch eğitim döngüsü süreleri ALFA 6,31 s, Attack 7,78 s,
RflyMAD 189,37 s ve UAV-SEAD 457,72 s oldu. Ham veri yaklaşık 30,43 GiB iken
modele taşınan parsed/Gold bundle toplamı yaklaşık 980,87 MiB idi; ham arşivin
tamamı optimizer girdisi değildi.

### 10.4 İlk evaluator sorunu ve düzeltme

İlk evaluator anomalili etiketli uçuşun bütün pencerelerini pozitif sayıyor,
uçuş skorunu max pencere olarak alıyor ve tek alarm penceresiyle uçuşu detected
ilan ediyordu. Bu nedenle çok yüksek normal `any-alarm` oranları gerçek
false-event/saat değildi.

Ayrı event evaluator gerçek interval truth, persistence, false-event/saat,
detection delay ve range coverage üretti. Düzeltilmiş v2 sonucu:

| Dataset | Interval AUC | Ana event sonucu | Karar |
|---|---:|---|---|
| ALFA | 0,483 | 1/5 event; yalnız 2 normal validation | NO-GO / smoke |
| UAV Attack | N/A | interval truth yok; normal burden çok yüksek | Açık NO-GO |
| UAV-SEAD | 0,747 | 7/206 event; %0,264 range recall | Ranking var, detector NO-GO |
| RflyMAD | 0,907 | 37/66 event; 0,768 false event/saat | Koşullu araştırma GO |

Bu aşama önceki proje bulgularını tekrar gösterdi: daha fazla veri veya GPU,
az bağımsız normal grup, proxy truth, magnitude ve event aggregation sorunlarını
tek başına çözmedi.

### 10.5 Splitlerin group-safe v3/v3.1'e değişmesi

V2 exact-source ayrılığı sağladı fakat ALFA session, Attack campaign ve UAV-SEAD
parent/mission ailelerinde rol çakışması riski bulundu. Önce grup ayrıklığı,
sonra sınıf dengesi ilkesiyle v3 üretildi.

- ALFA: session-family group-CV;
- UAV Attack: campaign + platform + live/simulation;
- UAV-SEAD: strict parent/session;
- RflyMAD: canonical scenario/domain group.

RflyMAD v3.1 rolleri:

| Rol | Grup | Uçuş | Kullanım |
|---|---:|---:|---|
| Train | 7 | 260 normal | Optimizer ve preprocessing |
| Validation | 3 | 90 normal | NLL checkpoint ve calibration |
| Normal test | 3 | 91 normal | False-alarm evaluation |
| Anomaly development | 12 | 557 fault | Development karşılaştırması |
| Final fault test | 13 | 553 fault | Mühürlü final |

### 10.6 Group-safe sonuç değişiklikleri

ALFA 4-fold teknik pipeline smoke PASS oldu; bilimsel baseline NO-GO:

- yalnız fold 0 magnitude gate'i geçti, fault detection 0;
- fold 1–3 magnitude gate'i geçmedi;
- flight AUC'ler 0,409–0,545 bandında kaldı.

RflyMAD v3.1 L4 sonucu:

- 30 epoch, seçilen checkpoint epoch 30;
- magnitude rho 0,3288 / 0,3495, PASS;
- flight ROC-AUC 0,6411;
- TP=462, FN=95, FP=77, TN=14;
- anomaly recall %82,94;
- normal-flight alarm %84,62;
- balanced accuracy %49,16;
- dondurulmuş operating point operasyonel NO-GO.

V2'nin daha güçlü görünmesi v2 splitini resmî olarak “daha iyi” yapmaz. V2
daha iyimser source-level/in-domain baseline, v3.1 daha güvenilir group
genelleme testidir. İki sonuç aynı metrik düzeyinde de değildir; split etkisini
izole etmek için kontrollü ablation gerekir. Final-fault-test açılmamıştır.

Ayrıntı:
`docs/RFLYMAD_V31_DEGERLENDIRME_VE_SPLIT_KARARI_20260728.md`.

## 11. Gözlenen gerçek iyileşmeler ve sınırlar

| Değişiklik | Gözlenen iyileşme | Sınır |
|---|---|---|
| ALFA veri büyütme | LSTM-AE 0,731 -> 0,918 | Küçük/session bağımlı corpus |
| SEAD session split/temizlik | 0,474 -> 0,799 satır AUC | Model kazancı değil; event GO değil |
| Chronos mekanik | 0,205 -> 0,390 recall | Füzyonda kayboldu |
| Tek-feature itki | 0,205 -> 0,459 | 38,1 FA/saat |
| SEAD daha çok normal | FA 23,6 -> 9,95/saat | Recall 0,21 -> 0,126 |
| Magnitude normalize skor | rho belirgin düştü | Recall <%6; sahte kazanç ortaya çıktı |
| Rfly interval truth | Gerçek zaman lokalizasyonu | Proxy 0,749 -> gerçek 0,526 |
| ADS-B contextual forecaster | Magnitude PASS | Dar bütçede çoğu recall <%6 |
| ADS-B geniş v2 grid | Recipe recall %51–57 düşük bütçede | Evrensel detector NO-GO |
| Four-dataset event evaluator | Gerçek interval/event metrikleri | İlk yüksek flight sonuçları zayıfladı |
| Group-safe split | Daha dürüst genelleme ölçümü | Görünen metrikler düştü |

En önemli ortak desen: recall artışı çoğu kez false alarm artışıyla satın alındı;
false alarm azaltıldığında recall düştü. Daha yüksek epoch veya GPU bu yapısal
trade-off'u çözmedi.

## 12. Uçuş modu/phase yönü

Ground, takeoff/climb, cruise/maneuver ve descent/landing ayrımı güçlü ve
test-edilebilir bir sonraki hipotezdir. Mevcut dört-UAV probabilistic v2/v3.1
dört ayrı phase modeli eğitmedi. İlk doğru ablation:

1. Phase yalnız geçmiş ve mevcut telemetriden nedensel hesaplanır.
2. Fault/onset, dosya adı veya future örnek kullanılmaz.
3. Aynı group-safe splitte önce global model, sonra tek modele phase context
   eklenmiş aday karşılaştırılır.
4. Faza özel scaler/threshold yalnız yeterli normal-validation exposure varsa
   açılır; aksi halde global calibration korunur.
5. Ana sonuç toplam ve phase-bazlı event recall, false event/saat ve gecikmedir.

RflyMAD v3.1 normal-test false-positive'ları incelenip sisteme göre değişiklik
yapılırsa bu 91 uçuş artık test sayılamaz; development'a dönüştürülmeli ve yeni
normal holdout ayrılmalıdır.

## 13. Güncel proje karar matrisi

| Hat | Güncel statü | Korunan olumlu bulgu | Ana engel |
|---|---|---|---|
| ALFA | Group-safe bilimsel NO-GO | Hızlı pipeline smoke; bir veri-büyütme DL kazanımı | Az bağımsız normal session |
| UAV Attack | NO-GO / external stress | Bazı attack örneklerinde görünür sinyal | 6 normal log, campaign ayrımı, interval truth |
| UAV-SEAD | Ana v3.1 eğitim beklemede | Interval ranking ve bazı mekanik feature sinyalleri | Savunulabilir session key ve event recall |
| RflyMAD | Araştırma adayı; v3.1 operating point NO-GO | V2 interval AUC 0,907; v3.1 magnitude PASS | Group-safe normal false alarm %84,62 |
| ADS-B basic | Keşif | Açıklanabilir false-positive mekanizmaları | Doğal anomaly ground truth yok |
| ADS-B contextual v2 | Recipe-bazlı araştırma GO; evrensel NO-GO | Speed persistence ve position CUSUM | Tek evrensel bütçe/recipe yok |

## 14. Sunumda kullanılacak ana anlatı

1. Proje önce etiketli İHA veri setlerinde geniş bir model ve feature uzayını
   denedi.
2. Yüksek görünen bazı skorların proxy truth, session yakınlığı veya ham
   magnitude ile açıklandığı bulundu.
3. Causal onset/interval truth, train-only scaling, magnitude diagnostic ve
   provenance kapıları getirildi.
4. ADS-B pivotu açıklanabilir fizik residual'ları, conformal calibration,
   persistence ve CUSUM'u olgunlaştırdı.
5. Sekiz epochluk büyük ADS-B v2 koşusu magnitude gate'i geçti; geniş grid
   recipe-bazlı güçlü sonuçlar verdi, evrensel dedektör yine oluşmadı.
6. Bu metodoloji dört UAV veri setine geri taşındı. Balanced source-level
   değerlendirmede RflyMAD güçlü görünürken gerçek event evaluator ve group-safe
   split genelleme sınırlarını ortaya çıkardı.
7. Sonuç “hiçbir şey çalışmadı” değildir: hangi sinyalin hangi bağlamda çalıştığı,
   hangi yüksek skorun artefakt olduğu ve güvenilir değerlendirmenin nasıl
   kurulacağı ölçüldü.

## 15. Güncelleme protokolü

Bu dosya ilerleyen raporlama boyunca şu biçimde güncellenecektir:

- tamamlanan her yeni koşu için tarih, namespace, contract hash ve veri rolleri;
- satır/window/interval/event/flight seviyeleri ayrı;
- başarı kadar NO-GO ve incident kayıtları;
- sonuç görüldükten sonra değiştirilmeyen ve değiştirilen değerler;
- final holdout erişim durumu;
- yeni grafik ve karar belgesi bağlantıları;
- eski sonuç silmeden “yerine geçen değerlendirme” notu.

## 16. Ana kaynak dizini

- İlk geniş süreç ve skor defteri: `docs/PROJE_SUREC_VE_SONUC.md`
- ML fizibilite sentezi: `docs/final_rapor_ml_fizibilite_2026-07-16.md`
- ADS-B basit kurallar: `docs/ADSB_BASIT_ANOMALI_KARSILASTIRMA_20260722.md`
- ADS-B contextual v2 planı:
  `docs/ADSB_CONTEXTUAL_PHYSICS_V2_CODEX_ILERLEME_PLANI_20260723.md`
- ADS-B contextual v2 ön-kayıt:
  `docs/adsb_contextual_physics_v2_prereg_20260723.md`
- ADS-B Faz G karşılaştırması:
  `docs/ADSB_CONTEXTUAL_PHYSICS_V2_KURAL_KARSILASTIRMA_20260727.md`
- ADS-B nihai karar: `docs/decisions.md`, ADR-046
- Four-dataset tarihsel audit:
  `docs/FOUR_DATASET_PROBABILISTIC_GPU_V1_HISTORICAL_AUDIT_20260727.md`
- Four-dataset v2 final:
  `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`
- Group split audit:
  `docs/FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_AUDIT_20260728.md`
- v3.1 protokol:
  `docs/FOUR_DATASET_NORMAL_ONLY_V31_PROTOCOL_DECISIONS_20260728.md`
- ALFA v3.1 smoke:
  `docs/FOUR_DATASET_PROBABILISTIC_V31_ALFA_CV_SMOKE_20260728.md`
- RflyMAD v3.1 baseline:
  `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`
- RflyMAD yorum ve split kararı:
  `docs/RFLYMAD_V31_DEGERLENDIRME_VE_SPLIT_KARARI_20260728.md`

## 17. 2026-07-28 — RflyMAD v3.1 B0 gerçek-event katmanı

Epoch-30 group-safe B0 modeli değiştirilmeden 90 normal-validation, 91 bağımsız
normal-test ve 557 anomaly-development uçuşu için 681.690 zaman hizalı pencere
skoru dışarı aktarıldı. Gerçek event truth'u `fault_active | condition_active`
alanlarından bağlandı; bütün uçuşu anomalili sayan eski flight proxy'si kullanılmadı.
553 final-fault-test uçuşu mühürlü kaldı.

Sonuç görülmeden threshold+persistence, zamana dayalı K-of-N ve validation-normal
standardized-NLL CUSUM aileleri donduruldu. 144 calibration adayı 54 rapor noktasına
indirildi; 48 nokta ölçülebildi, CUSUM 0,5 event/saat bütçesindeki altı nokta
ekstrapolasyon yapılmadan unavailable bırakıldı.

Primary 1 event/saat ve 30 saniye refractory görünümünde K-of-N 2-of-3,
validation'da 0,511 ve normal-testte 0,654 false event/saat ile %43,27 gerçek-event
recall verdi. Persistence 0,5 saniye aynı validation oranına rağmen normal-testte
4,577 event/saat ve %42,19 recall; CUSUM allowance 0,25 ise 5,884 event/saat ve
%48,29 recall verdi. En yüksek grid recall'ı %58,71 oldu fakat 9,153 false
event/saat gerektirdi. Real-domain recall %7,81, Sensor recall %5,45 kaldı.

Karar: B0 operasyonel terfi **NO-GO**. Sonuç threshold/epoch avcılığına değil,
yeni namespace altında causal flight-phase/domain context ve kanal-bazlı NLL/scale
teşhisine yönlendirildi. Ayrıntılı rapor:
`docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.

Provenance notu: Colab run manifestinin gömülü prereg byte-hash'i sonraki checkout
belgesiyle eşleşmedi; fakat gömülü sözleşme kendi içinde tutarlı ve config, roles,
data fingerprint, model, scaler ve run artifact hash'lerinin tamamı eşleşti. Kusur
gizlenmeden event manifestine ve ayrıntılı rapora kaydedildi.
