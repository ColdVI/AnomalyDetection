# RflyMAD-Full v2 — Wind/Real Robustness Deney Sözleşmesi

> Durum: **ONAYLANDI — development-only deneyler açıldı**  
> Yazım tarihi: 2026-07-22 (Europe/Istanbul)  
> `approved=true`  
> Kapsam: yalnız development split; locked-test feature okuma yasaktır.

## 1. Amaç ve değişmez sınırlar

Bu belge, sonuçlara bakıldıktan sonra başarı ölçütü değiştirilmesini önlemek için
Wind ve Real-domain geliştirme deneylerinin ölçütlerini önceden dondurur. Belge
onaylanmadan aşağıdaki robustness deneyleri çalıştırılmaz.

- Wind sistem arızası/pozitif sınıf değildir; `environment_robustness` olarak
  yalnız false-alarm yüküyle değerlendirilir.
- Real, HIL ve SIL metrikleri ayrı raporlanır. SIL/HIL başarısı Real başarısı
  yerine kullanılamaz.
- Scaler, model, eşik ve aday seçimi yalnız development verisiyle yapılır.
- Satır/window, uçuş-event ve uçuş düzeyi metrikler karıştırılmaz.
- Tüm çıktılar `status=development_only_robustness` ve
  `operational_claim_allowed=false` taşır.
- Frozen karşılaştırma tabanı:
  `normal_temporal_ae/sweep_20260722_093049`.

## 2. Dondurulmuş baseline

| Ölçüt | Critical baseline |
|---|---:|
| Tüm sistem-arızası uçuş recall ort. | %60,43 |
| En düşük rotasyon recall | %51,84 |
| Tüm nonfault FA/saat ort. | 1,28 |
| En yüksek rotasyon nonfault FA/saat | 1,65 |
| Wind FA/saat ort. | 28,46 |
| Real-Motor recall | %20,20 |
| Real-Sensor recall | %8,35 |

Bu değerler hedef seçmek için yalnız başlangıç zorluğunu gösterir; aynı development
sonuçları aday seçimi ve nihai aday değerlendirmesinde tekrar tekrar kullanılmaz.
Her aday, dış beş validation rotasyonunun içinde ayrı inner-development seçim
katmanı kullanır.

## 3. Real-domain başarı kapıları

Real örneklemi küçük olduğu için pooled recall kullanılmaz. Birincil ölçüt,
Real-Motor ve Real-Sensor uçuş-recall değerlerinin eşit ağırlıklı makro
ortalamasıdır. İki aile ayrı ayrı da raporlanır.

### 3.1 Araştırma-promosyon kapısı

Bir Real adayı sonraki geliştirme aşamasına ancak aşağıdakilerin **tamamını**
sağlarsa taşınır:

1. Beş dış rotasyonda ortalama Real macro recall `>= %40`.
2. Real-Motor ve Real-Sensor ortalama recall değerlerinin her biri `>= %30`.
3. Hiçbir dış rotasyonda Real macro recall `< %25` değil.
4. Held-out development Real-NoFault FA ortalaması `<= 4/saat`, rotasyon
   maksimumu `<= 8/saat`.
5. Genel critical sistem-arızası recall baseline'a göre `>5 yüzde puan`
   düşmüyor ve hiçbir rotasyonda `%50` altına inmiyor.
6. Genel critical tüm-nonfault FA ortalaması `<= 2/saat`.

Bu kapı yalnız “araştırmaya değer Real sinyali” anlamına gelir; fizibilite veya
operasyonel başarı değildir.

### 3.2 Daha sıkı fizibilite kapısı

İleride Real-domain fizibilite adayı denebilmesi için, yeni sonuç görmeden
dondurulan daha sıkı kapı şudur:

- Real macro recall `>= %50`;
- her Real fault ailesi recall `>= %40`;
- Real-NoFault FA ortalaması `<= 2/saat`, rotasyon maksimumu `<= 4/saat`;
- cluster-bootstrap %95 güven aralığının alt sınırı macro recall için `>= %35`.

Bu kapı development üzerinde geçse bile locked-test protokolü ayrıca onaylanmadan
operasyonel iddia kurulmaz.

## 4. Wind başarı kapıları

Wind için ana ölçüt critical `environment_fa_per_hour` değeridir. Recall olarak
raporlanmaz ve sistem-arızası confusion matrix'ine katılmaz.

### 4.1 Dondurulmuş ara hedef

Bir Wind robustness adayı ana geliştirme hattına ancak:

1. Beş rotasyon ortalama Wind FA `<= 15/saat`;
2. hiçbir rotasyonda Wind FA `>20/saat` değil;
3. frozen 28,46/saat baseline'a göre ortalama azalma `>= %40`;
4. genel critical recall kaybı `<=5 yüzde puan`;
5. Real macro recall kaybı `<=5 yüzde puan`;
6. genel tüm-nonfault FA ortalaması `<=2/saat`

koşullarının tamamını sağlarsa taşınır. `15/saat`, mevcut yükün yaklaşık yarısına
inen anlamlı ama nihai olmayan bir ara hedeftir; “kabul edilebilir operasyonel
alarm yükü” diye sunulamaz.

### 4.2 Dondurulmuş nihai araştırma hedefi

Wind için daha sonraki nihai araştırma hedefi `<=5 FA/saat` ve hiçbir rotasyonda
`>8 FA/saat` olmamasıdır. Bu hedef bu sözleşmedeki sınırlı aday sayısını artırmak
için kullanılmaz.

## 5. Development-only aday sırası

Her aşamada eşik/model seçimi outer evaluation rotasyonundan ayrılmış inner
development fold'larında yapılır. Bir sonraki aşamaya yalnız önceki aday kendi
promosyon kapısını geçmezse veya farklı bir kapıyı hedefliyorsa geçilir.

1. **R1 — Real threshold-only baseline.** Model ve scaler dondurulur. Yalnız
   development Real-NoFault inner fold'larıyla Real critical/advisory quantile
   seçimi yapılır. Fault uçuşları threshold seçiminde kullanılmaz.
2. **W1 — Environment-aware calibration.** Model dondurulur. Development Wind,
   pozitif/anomaly değil ayrı nonfault stres dağılımı olarak threshold
   kalibrasyonuna eklenir. Ağırlık önceden `Real/HIL/SIL NoFault : Wind = 1:1`
   olarak sabittir; ağırlık taraması yapılmaz.
3. **W2 — Environment-aware normal training.** Wind pencereleri ayrı sampler
   stratum'u olarak normal-only AE eğitimine eklenir. NoFault ve Wind toplam loss
   katkıları `1:1` tutulur; fault pencereleri eğitime girmez.
4. **R2 — Real normal-only fine-tune, kısa.** Frozen base AE yalnız development
   Real-NoFault ile `3 epoch`, learning rate `1e-4` fine-tune edilir.
5. **R3 — Real normal-only fine-tune, uzun.** R2 aynı koşullarda `8 epoch`
   çalışır; 3 ve 8 dışında epoch taraması yapılmaz.
6. **RW1 — Birleşik aday.** Yalnız ayrı bir Real adayı ve ayrı bir Wind adayı
   kendi promosyon kapısını geçmişse bu iki önceden seçilmiş mekanizma tek koşuda
   birleştirilir. Yeni hiperparametre aranmaz.

En fazla **6 aday konfigürasyonu** değerlendirilir. Aynı adayın seed tekrarı
hiperparametre denemesi sayılmaz; yalnız önceden sabit seed seti
`[20260721, 20260722, 20260723]` ile stabilite ölçümü yapılabilir ve seçim ortalama
sonuca göre değil, yukarıdaki beş dış rotasyon kapısına göre verilir.

## 6. Raporlanacak metrikler

Her aday için aşağıdakiler aynı tabloda baseline ile yan yana yazılır:

- uçuş-event düzeyi genel critical/advisory recall;
- Real-Motor, Real-Sensor ve eşit ağırlıklı Real macro recall;
- held-out Real-NoFault FA/saat ve alarm alan uçuş oranı;
- tüm nonfault FA/saat;
- Wind FA/saat (sistem arızalarından ayrı);
- beş rotasyon mean/std/min/max;
- uçuş veya session cluster-bootstrap %95 güven aralıkları;
- `locked_test_features_read`, `status`, `operational_claim_allowed`.

Bir metric eksikse aday kapıyı geçmiş sayılmaz. NaN/insufficient-data başarıya
çevrilmez.

## 7. Durdurma ve karar kuralı

- Altı adaydan hiçbiri Real araştırma-promosyon kapısını geçmezse Real-domain
  başarı iddiası kapsamdan çıkarılır. Sonuç “mevcut veri/temsil ile Real transfer
  gösterilemedi” diye raporlanır; yeni Real veri veya yeni temsil sözleşmesi
  olmadan threshold araması sürdürülmez.
- Hiçbir aday Wind ara kapısını geçmezse Wind robustness “çözülmedi” olarak
  kalır. Wind anomaly pozitifine çevrilmez ve alarm yükünü saklamak için rapordan
  çıkarılmaz.
- Ayrı Real ve Wind adayları geçip birleşik RW1 geçmezse tek bir ana model ilan
  edilmez; trade-off açıkça raporlanır.
- Herhangi bir aday genel recall/FA koruma kapısını bozarsa kendi hedef metriği
  iyi görünse bile elenir.
- Bu kararlar sonrasında kullanıcı ayrıca onay vermeden locked test açılmaz.

## 8. Kullanıcı onayı

Bu belge onaylanmıştır; Bölüm 5 deneyleri yalnız yukarıdaki sıra ve durdurma
kurallarıyla çalıştırılır.

- [x] Real araştırma ve fizibilite kapıları onaylandı.
- [x] Wind ara ve nihai hedefleri onaylandı.
- [x] En fazla 6 aday ve durdurma kuralı onaylandı.
- [x] Development-only/locked-test kapısı onaylandı.

Onay kaydı: **KULLANICI “tamamdır, devam” diyerek onayladı**  
Onaylayan: kullanıcı (sohbet kaydı)  
Onay zamanı: 2026-07-22T11:31:22+03:00
