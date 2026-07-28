# Four-dataset v2 — yorum, v3 yönü ve raporlama kaydı

Tarih: 2026-07-28
Kaynak: Kullanıcı tarafından sağlanan teknik değerlendirme; kabul edilmiş v2
artifactleri ve `four_dataset_probabilistic_event_eval_v1` çıktılarıyla
uzlaştırılmıştır.
Statü: İleride yazılacak tez, mentör raporu ve sunum için kalıcı yorum kaydıdır.

## Bu belgenin kullanım kuralı

Bu metin üç kanıt düzeyini açıkça ayırır:

- **Ölçüm:** Yerel artifact veya evaluator raporundan doğrudan doğrulanan sayı.
- **Yorum:** Ölçümün desteklediği, fakat tek başına nedensellik kanıtlamayan sonuç.
- **Hipotez:** V3 ablation ile sınanması gereken mekanizma açıklaması.

Hipotezler raporda “olası açıklama” olarak yazılmalı; doğrulanmış mekanizma gibi
sunulmamalıdır. Özellikle RflyMAD skorunun arızalı rejime “alışması” tek-adım
tahmin modelinin davranışıyla uyumlu bir hipotezdir, henüz ablation ile
kanıtlanmış değildir.

## Metodolojik düzeltmenin bilimsel anlamı

Yapılan çalışma önceki v2 modellerini değiştirmeden evaluator'ı düzeltmiştir:

- Model yeniden eğitilmedi.
- 30. epoch final checkpointleri sonuçtan önce sabitti.
- Ham pencere skorları değiştirilmedi; truth yenilemesinde immutable skor
  payload hash'leri aynı kaldı.
- Gerçek arıza zamanları yalnız değerlendirme katmanına bağlandı.
- `1 s persistence + validation'da 1 false event/saat` testten seçilmiş bir
  kazanan değil, sunum için sabit referans hücresidir.
- Timeline uçuşları skora göre değil, sınıf içindeki süre medyanına yakınlıkla
  seçildi.
- Bütün dondurulmuş quantile × persistence × bütçe hücreleri raporda tutuldu.

Dolayısıyla yapılan işlem kötü sonucu iyileştirmek için model veya metrik
ayarlamak değildir. Eski evaluator uçuşun tamamını tek etiket ve tek maksimum
skorla özetliyordu; yeni evaluator aynı skorları gerçek arıza zamanı, nedensel
süreklilik, event recall, gecikme ve normal false-event yüküyle ölçüyor.

## Veri seti bazında doğrulanmış yorum

### RflyMAD — gerçek ve geliştirilebilir sinyal

**Ölçümler**

- Interval ROC-AUC: 0.907.
- Interval AP: 0.805.
- Referans event recall: 37/66 = %56.1.
- Test false-event yükü: 0.768 event/normal saat.
- Flight mean-score AUC: 0.915; keşifsel agregasyondur.
- Source-bootstrap %95 event-recall aralığı: %42.4–68.2.
- Referans confusion matrix interval-window recall: %16.69.
- Time-weighted range recall: %16.54.

**Yorum**

Model arızalı zamanları normal zamanlardan güçlü biçimde sıralıyor ve 66 olayın
çoğunda en azından kısa bir uyarı işareti üretiyor. Buna rağmen arızanın tüm
süresini takip edemiyor. Event recall ile window/range recall arasındaki fark
çelişki değildir: yüzlerce pencerelik bir olayın kısa bir parçasındaki alarm,
olayı “yakalanmış” yaparken anomalili pencerelerin çoğu kaçabilir.

Raporlanabilir tanım:

> RflyMAD v2, birçok arızada kısa bir erken uyarı işareti üreten; ancak arızalı
> rejimin tamamını güvenilir biçimde izleyemeyen bir araştırma baseline'ıdır.

**V3 hipotezi**

Tek-adım model ilk rejim değişimine yüksek sürpriz veriyor olabilir. Arızalı
örnekler giriş penceresine girdikçe yeni rejim tahmin edilebilir hâle geliyor ve
skor düşüyor olabilir. Bu hipotez multi-horizon tahmin ve zaman içinde kanıt
biriktirme ablation'larıyla sınanacaktır.

### UAV-SEAD — ranking var, operasyonel event detector yok

**Ölçümler**

- Interval ROC-AUC: 0.747.
- Interval AP: 0.263.
- Referans event recall: 7/206 = %3.40.
- Referans test false-event sayısı: 0; normal exposure 5.414 saattir.
- Flight mean-score AUC: 0.772; keşifsel agregasyondur.
- Source-bootstrap %95 event-recall aralığı: %0.53–7.04.
- Interval-window recall: %0.136; time-weighted range recall: %0.264.

**Yorum**

Anomali pencereleri genel olarak daha yüksek skor alma eğilimindedir; fakat
mevcut skor ve dondurulmuş validation kalibrasyonu kabul edilen false-event
bütçesinde gerçek olay duyarlılığına dönüşmemektedir. Sıfır test false eventi,
%3.4 event recall ile birlikte okunmalıdır; tek başına başarı değildir.

Raporlanabilir tanım:

> UAV-SEAD v2'de istatistiksel bir sıralama sinyali vardır, ancak bu sinyal
> mevcut karar katmanında kullanılabilir event-detection duyarlılığı üretmez.

### ALFA — küçük ve session-kısıtlı fizibilite sonucu

**Ölçümler**

- Interval ROC-AUC: 0.483.
- Interval AP: 0.108.
- Referans event recall: 1/5 = %20.
- Yalnız 2 normal validation ve 3 normal test uçuşu vardır.
- Bir `no_ground_truth` anomalili uçuş event metriklerinden dışlanmıştır.
- Source-bootstrap %95 event-recall aralığı: %0–60.
- Interval-window recall: %5.78.

**Yorum**

İki normal validation uçuşundan türetilen sıfır false-event sonucu güvenilir
operasyonel güvence değildir. Interval AUC'nin 0.5 altında olması, anomalili
zamanların genel olarak daha yüksek skorlanmadığını gösterir. Kısa spike'lar bir
olayı yakalamıştır, fakat sonuç yüksek varyanslıdır.

Raporlanabilir tanım:

> ALFA sonucu bir başarı göstergesi değil, küçük ve session-kısıtlı veri üzerinde
> evaluator ve yöntem fizibilitesi kaydıdır.

### UAV Attack — fizik modeli yerine domain/campaign sorunu baskın

**Ölçümler**

- Bütün raporlanan flight agregasyonlarında ROC-AUC: 0.000.
- Tek normal test kaynağında 4 alarm eventi ve 32.42 false event/saat vardır.
- Referans alarm-time fraction: %75.3.
- Üç attack test uçuşunda interval truth yoktur; event recall hesaplanmamıştır.

**Yorum**

Normal test kaynağının attack uçuşlarından daha yüksek skorlanması persistence
veya başka bir flight agregasyonuyla çözülecek bir karar-katmanı sorunu değildir.
Campaign/domain farkı ve fiziksel kanallarda düşük saldırı gözlenebilirliği ana
sorundur.

Raporlanabilir tanım:

> UAV Attack v2, mevcut normal kalibrasyon kaynakları ve fiziksel feature setiyle
> kullanılamaz; önce campaign-safe ayrım ve network/timing gözlenebilirliği gerekir.

## V3 dataset karar matrisi

| Veri seti | Karar | Amaç |
|---|---|---|
| RflyMAD | Ana bilimsel v3 adayı | Güçlü interval sinyalini daha yüksek event coverage ve daha düşük gecikmeye çevirmek |
| UAV-SEAD | İkinci ana v3 adayı | Phase/context ve yeni kanalların ranking sinyalini event recall'a çevirip çevirmediğini sınamak |
| ALFA | Sınırlı smoke-test ve grouped-CV | Pipeline, session-safe split ve phase-aware residual fizibilitesi |
| UAV Attack | Ayrı track | Önce campaign-safe normal calibration, sonra network/timing feature hattı |

Mühendislik sırası ALFA → RflyMAD → UAV-SEAD → UAV Attack olarak kalır. Bu sıra
bilimsel önem sırası değildir: ALFA hızlı smoke-test, RflyMAD ana deneydir.

## V3'te uygulanacak değişiklikler

### 1. Group-safe split

- ALFA: aynı temel tarih/session ailesinin bütün normal/fault parçaları tek rol.
- UAV Attack: campaign + platform + live/simulation tek rol.
- UAV-SEAD: doğrulanmış mission/session; metadata yetersizse muhafazakâr parent.
- RflyMAD: canonical case bütünlüğü; in-domain ve SIL/HIL/Real transfer ayrı
  protokoller.

Önce grup ayrıklığı, sonra sınıf dengesi uygulanır. Satır/pencere random split
yasaktır.

### 2. Uçuş bağlamı

İlk phase deneyi dört bağımsız model değil:

> Tek model + nedensel context input + yeterli normal validation exposure varsa
> faza özel scaler/threshold.

Bağlam sınıfları `ground`, `takeoff_climb`, `cruise_maneuver`,
`descent_landing`; RflyMAD için SIL/HIL/Real domain context'i ayrıca korunur.

### 3. Raw feature ablation

Bloklar birlikte değil, sırayla denenir:

1. Base Gold features;
2. `+ phase/domain context`;
3. `+ setpoint-response residual`;
4. `+ actuator/motor detail`;
5. `+ estimator innovations`;
6. `+ IMU transient summaries`;
7. `+ energy/battery`.

Her aday aynı group-safe split ve evaluator ile karşılaştırılır. Fault/onset,
dosya etiketi ve injection metadata modele giremez.

### 4. Zaman içinde kanıt biriktirme

RflyMAD hipotezi nedeniyle aşağıdaki karar katmanları önceden dondurulmuş aday
ailesi olarak karşılaştırılacaktır:

- ham causal persistence baseline;
- EWMA;
- CUSUM;
- hysteresis;
- threshold üstündeki alan;
- son sabit sürede yüksek-skor oranı;
- multi-horizon forecasting skoru.

Testten kazanan seçilemez. Kalibrasyon ve aday seçimi yalnız grouped normal
validation ve, event parametresi gerekiyorsa, development truth ile yapılır.
Testte bütün dondurulmuş adaylar raporlanır.

### 5. Checkpoint seçimi

V3 checkpoint politikası test açılmadan dondurulacaktır. Tercih edilen ilk
politika: en fazla 30 epoch eğitim, grouped normal-validation NLL üzerinde en
iyi checkpoint, eşitlikte en erken epoch; test NLL veya event metriği checkpoint
seçiminde kullanılmaz. Bu karar v2 eğitim eğrilerinden öğrenilen metodoloji
dersidir ve v3 test sonucu görülmeden kayda alınmalıdır.

## ADS-B ile ilişki

Ortak olan feature seti değil değerlendirme metodolojisidir:

```text
window score
→ causal persistence / CUSUM / evidence accumulation
→ event
→ false events/hour
→ event recall
→ detection delay ve interval coverage
```

ADS-B'de IMU, motor, actuator ve estimator innovation yoktur. Uygun context
adayları uçuş fazı, irtifa bandı, tırmanış/alçalma, dönüş/manevra, havaalanı
yakınlığı, mesaj cadence/gap, uçak/rota/gün domaini ve kinematik residual'lardır.

Mevcut ADS-B frozen-checkpoint evaluator çalışması zaten event, persistence ve
CUSUM katmanını uygulamıştır. Buradaki yeni ders ADS-B'yi hemen yeniden eğitmek
değil; mevcut truth-v2 sonuçları üzerinden phase/cadence ablation gereksinimini
belirlemektir.

Hesap ölçeği:

- ADS-B: 1,889,036,504 window-epoch.
- Dört UAV veri seti toplamı: 35,314,530 window-epoch.
- Oran: 53.49×.
- Dört kabul edilmiş L4 GPU eğitim döngüsü: toplam 11.02 dakika.
- ADS-B raporlanmış L4 eğitim oturumu: yaklaşık 42.3 dakika.
- ADS-B truth-v2 CPU hazırlama/denetimi: yaklaşık 129.4 dakika.

Bu süreler notebook uçtan uca benchmarkı değildir; transfer, yükleme veya skor
export kapsamı artifacte göre değişir.

## İleride rapora aktarılacak bölüm yapısı

### 1. Önceki yaklaşımın sınırlaması

> İlk değerlendirmede anomalili olarak işaretlenen bir uçuşun tüm zaman
> pencereleri anomalili kabul edilmiş ve uçuş skoru en yüksek pencere skoruyla
> belirlenmişti. Bu yaklaşım arıza başlamadan önceki normal bölümleri yanlış
> etiketliyor ve uzun uçuşlarda tek bir yüksek skor nedeniyle tüm uçuşun alarm
> üretmesine yol açıyordu.

### 2. Yapılan düzeltme

> Eğitilmiş modeller ve ham skorlar değiştirilmeden, gerçek arıza başlangıç ve
> bitiş zamanları değerlendirme katmanına bağlandı. Ardışık yüksek skorların tek
> alarm olayı olarak birleştirildiği nedensel event değerlendirmesi oluşturuldu.
> Eşik ve süreklilik adayları validation verisiyle kalibre edildi; test üzerinde
> yeniden ayar yapılmadı.

### 3. Ana sonuç

> Düzeltilmiş değerlendirme veri setleri arasında belirgin farklar gösterdi.
> RflyMAD modelinde arızalı zamanları sıralayan ve olayların %56.1'inde uyarı
> üreten gerçek bir sinyal bulundu; fakat arızalı sürenin yalnız sınırlı kısmı
> alarm altında kaldı. UAV-SEAD'de skor ayrımı görülmesine rağmen event recall
> %3.4'te kaldı. ALFA'da bağımsız normal uçuş azlığı güvenilir kalibrasyonu
> engelledi. UAV Attack'ta normal test uçuşu attack uçuşlarından daha yüksek
> skorlandığı için mevcut feature ve veri ayrımı kullanılamaz bulundu.

### 4. Bulguların anlamı

> Eğitim loss'unun düşmesi tek başına anomali tespit başarısı değildir. En güçlü
> gelişim alanları model büyütmeden önce session-safe veri ayrımı, uçuş fazı
> bağlamı, gerçek event truth ve zaman içinde kanıt biriktiren alarm mantığıdır.

### 5. V3 planı

> Yeni sürüm önce session/campaign/domain bazlı group-safe split kuracak; ardından
> flight phase, control residual, actuator, estimator, IMU transient ve energy
> bloklarını tek tek ekleyerek her katkıyı mevcut baseline'a karşı ölçecektir.

## Sunum görsel hiyerarşisi

Ana sunum:

1. Detection summary — dört veri setinin tek sayfalık sonucu.
2. RflyMAD timeline — ilk sinyal ile devam eden event coverage farkı.
3. Training diagnostics — düşük loss'un başarı olmadığını, özellikle UAV Attack
   üzerinden gösterir.

Teknik ek:

- operating-grid;
- interval-truth confusion matrix;
- bütün dataset timeline'ları;
- bootstrap aralıkları ve group-overlap audit'i.

Compute grafiği yalnız ölçek/maliyet bölümünde kullanılmalıdır. Ana mesaj:

> Dört UAV veri setinin model-ready transferi yaklaşık 981 MiB, kabul edilmiş L4
> eğitim döngüsü yaklaşık 11 dakikadır; asıl mühendislik maliyeti ham veri,
> group metadata ve event-ground-truth eşlemesindedir.

## Tek cümlelik nihai bulgu

> Mevcut model ailesi dört veri setinde genel bir anomali dedektörü oluşturmadı;
> ancak evaluator düzeltmesi RflyMAD üzerinde gerçek ve geliştirilebilir bir
> event-detection sinyali bulunduğunu, diğer veri setlerindeki ana sınırların
> sırasıyla olay duyarlılığı, bağımsız normal veri miktarı ve domain uyuşmazlığı
> olduğunu gösterdi.

## İlgili kayıtlar

- Nihai ölçüm raporu: `FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`
- Group/split denetimi: `FOUR_DATASET_PROBABILISTIC_GPU_V2_GROUP_SPLIT_AUDIT_20260728.md`
- V3 group registry v1 denetimi: `FOUR_DATASET_PROBABILISTIC_V3_GROUP_REGISTRY_V1_AUDIT_20260728.md`
- V3 group-safe split v1 ön-kayıt: `FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_PREREG_20260728.md`
- V3 group-safe split v1 sonuç denetimi: `FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_AUDIT_20260728.md`
- Event ön-kayıt: `FOUR_DATASET_PROBABILISTIC_EVENT_EVAL_V1_PREREG_20260728.md`
- V3 phase/raw planı: `FOUR_DATASET_PROBABILISTIC_V3_PHASE_RAW_ABLATION_PLAN_20260728.md`
