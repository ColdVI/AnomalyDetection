# Four-dataset v3 — group-safe, flight-phase ve raw kanal ablation planı

Tarih: 2026-07-28

V3.1 uygulama eki: Bu taslağın split sonrası uygulama kararı
`FOUR_DATASET_NORMAL_ONLY_V31_PROTOCOL_DECISIONS_20260728.md` dosyasında
dondurulmuştur. Mevcut v3 manifesti korunmuş; RflyMAD fault kaynakları iteratif
geliştirme için `anomaly_dev` ve tek-seferlik `final_fault_test` rollerine grup
düzeyinde ayrılmıştır. UAV-SEAD strict-parent benchmark, ULog header metadata
denetimi daha savunulabilir bir session anahtarı üretmediği için korunmuştur.
Statü: Yeni eğitim başlatmadan önce uygulanacak çalışma sözleşmesi taslağıdır;
v2 artifactlerini veya v2 sonuçlarını değiştirmez.

İlerleme notu (2026-07-28): Adım 1 registry denetimi ve Adım 2 group-safe split
üretimi tamamlandı. Registry için kanonik kayıt
`FOUR_DATASET_PROBABILISTIC_V3_GROUP_REGISTRY_V1_AUDIT_20260728.md` dosyasındadır.
Split ön-kayıt ve sonuç denetimi sırasıyla
`FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_PREREG_20260728.md` ve
`FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_AUDIT_20260728.md` dosyalarındadır.
UAV Attack group-safe split için veri-sözleşmesi NO-GO'dur; diğer üç veri setinin
rolleri dondurulmuştur.

## Amaç

V2'nin cevapladığı soru “normal uçuş dinamiğini öğrenen küçük probabilistic
forecaster bir veri setinde sinyal üretiyor mu?” idi. V3'ün sorusu daha sıkıdır:
aynı sinyal bağımsız session/campaign/domain gruplarına genelleniyor mu ve uçuş
rejimi değişimlerini gerçek arıza/saldırı eventlerinden ayırabiliyor mu?

Başarı ölçütü yalnız ROC-AUC değildir. Ana çıktılar interval/event recall,
normal false event/saat, detection delay ve time-weighted range precision/recall
olacaktır. Flight agregasyonları yine offline triage ve ikincil analizdir.

## Değiştirilemez sıra

1. Dataset-specific group registry oluştur ve doğrula.
2. Group-safe train/validation/test rollerini dondur.
3. V2 feature/model ailesiyle yeniden baseline kur.
4. Flight-phase context ablation'ı ekle.
5. Raw telemetri bloklarını tek tek ekle.
6. Her aday için yalnız normal validation ile alarm kalibrasyonu yap.
7. Testte bütün dondurulmuş operating grid'i göster; testten kazanan seçme.

Group-safe baseline geçmeden phase veya raw-feature kazancı iddia edilemez.

## Dataset-specific group registry

| Veri seti | Grup anahtarı | Zorunlu kontrol |
|---|---|---|
| ALFA | `carbonZ_YYYY-MM-DD-HH-MM-SS` temel session ailesi | Aynı session'ın normal ve fault parçaları tek rolde |
| UAV Attack | campaign tarihi + platform + live/simulation | Aynı benign/attack campaign farklı rollere bölünmez |
| UAV-SEAD | doğrulanmış mission/session; yoksa muhafazakâr parent path | Parent/session train-val-test kesişimi sıfır |
| RflyMAD | canonical case bütünlüğü + domain protokolü | In-domain grouped test ve SIL/HIL/Real transfer ayrı |

Önce grup ayrıklığı, sonra olabildiği ölçüde sınıf dengesi uygulanır. Satır veya
pencere random split yasaktır.

## Flight-phase v1

İlk deney dört ayrı modeli değil, tek modelde nedensel context kanalını kullanır.
Bu sayede hem veri azlığında model parçalanmaz hem phase kanalının katkısı temiz
bir ablation olarak ölçülür.

Fazlar:

- `ground`: armed/in-air değil veya kalıcı düşük hız ve düşük bağıl irtifa;
- `takeoff_climb`: in-air ve kalıcı pozitif vertical rate;
- `cruise_maneuver`: in-air, climb/descend koşulu yok;
- `descent_landing`: kalıcı negatif vertical rate; touchdown sonrası ground.

Kurallar yalnız geçmiş/mevcut telemetriyi kullanır. Centered rolling window,
gelecek touchdown bilgisi, fault başlangıcı veya dosya etiketi kullanılamaz.

Karşılaştırma:

- B0: group-safe v2 feature baseline;
- B1: B0 + categorical/one-hot phase context;
- B2: yalnız yeterli normal-validation exposure olan fazlarda phase-specific
  normalizasyon/eşik; diğer fazlarda global kalibrasyon.

Her faz için normal exposure saati, gerçek event sayısı, event recall, false
event/saat ve gecikme ayrı verilir. Faz sınırındaki ±5 saniyelik bölge ayrıca
raporlanır; otomatik olarak ground truth dışına atılmaz.

## Zaman içinde kanıt biriktirme

RflyMAD v2'de event recall %56.1 iken time-weighted range recall %16.54'tür.
Bu ölçüm, birçok eventte kısa bir skor işareti bulunduğunu fakat kanıtın event
boyunca korunmadığını gösterir. Tek-adım modelin arızalı rejime uyum sağlaması
olası açıklamadır; mekanizma henüz kanıtlanmış değildir.

V3 karar-katmanı aday ailesi test açılmadan dondurulacaktır:

1. ham causal persistence baseline;
2. EWMA;
3. CUSUM;
4. hysteresis;
5. threshold üzerindeki alan;
6. son sabit zaman aralığında yüksek-skor oranı;
7. multi-horizon forecast skor birleşimi.

Kalibrasyon normal grouped validation ile yapılır. Event-specific parametre
gerekiyorsa yalnız development truth kullanılabilir. Test sonucundan yöntem,
decay, horizon, pencere veya threshold seçilemez; testte bütün dondurulmuş
adaylar raporlanır.

## Raw kanal ablation blokları

Raw dosyalar topluca modele dökülmez. Her blok B1 üzerine tek başına eklenir;
katkı ancak aynı group-safe split ve aynı evaluator altında ölçülür.

1. `control`: setpoint, measured response, tracking residual;
2. `actuator`: control output, motor/servo dispersion, saturation;
3. `estimator`: EKF innovations/test ratios/reset counters;
4. `imu_transient`: yüksek frekanslı IMU transient/rate/vibration özetleri;
5. `energy`: battery voltage/current/power ve thrust ilişkisi;
6. `environment_context`: GPS quality, wind, vibration, platform/domain ve
   flight phase.

Yasak leakage kanalları: fault/injection kimliği, `fault_id`, `fault_mode`,
fault başlangıç/bitiş zamanı, attack dosya adı ve future-derived phase. PX4
native health/failure flagleri ayrı bir baseline olarak tutulur.

## Checkpoint politikası

V2'de final epochu sabitlemek metodolojik olarak temizdi; eğitim eğrileri ise
özellikle UAV Attack'ta train-validation kopması gösterdi. V3 test sonuçları
görülmeden aşağıdaki politika dondurulur:

- maksimum 30 epoch;
- seçim metriği yalnız grouped normal-validation Gaussian NLL;
- en düşük validation NLL checkpointi;
- eşitlikte en erken epoch;
- test loss'u, event recall veya test false-event yükü checkpoint seçemez.

Bu politika v2 gözleminden öğrenilmiş yeni-deney kararıdır; v2 checkpointini
geçmişe dönük değiştirmez.

## Dataset-specific araştırma track'leri

- RflyMAD ana fiziksel-fault v3 track'idir: phase/domain, raw block ve evidence
  accumulation ablation'larının tamamı uygulanır.
- UAV-SEAD phase/context ve interval-localization track'idir; hedef yalnız AUC
  artırmak değil, %3.4 event recall'ın gelişip gelişmediğini sınamaktır.
- ALFA küçük grouped-CV ve pipeline smoke-testidir; ana başarı tablosuna tek
  split sonucu olarak taşınmaz.
- UAV Attack aynı fizik-v3 hattına doğrudan sokulmaz. Önce campaign-safe normal
  calibration ve message timing/topic rate/jitter/gap feature sözleşmesi gerekir.

## Uygulama sırası, veri ve süre planı

| Sıra | Veri seti | Yerel raw | Mevcut optimizer girdisi | Kabul edilmiş L4 30-epoch döngüsü | V3 amacı |
|---:|---|---:|---:|---:|---|
| 1 | ALFA | 0.253 GiB | 0.0175 GiB | 6.31 s | Pipeline/group/phase smoke-test; başarı iddiası değil |
| 2 | RflyMAD | 18.910 GiB | 0.3157 GiB Colab subset | 189.37 s | Ana fiziksel-fault baseline ve domain analizi |
| 3 | UAV-SEAD | 10.597 GiB | 0.6109 GiB | 457.72 s | Ana phase/context ve interval-localization testi |
| 4 | UAV Attack | 0.668 GiB | 0.0131 GiB | 7.78 s | Campaign-safe normal kalibrasyon sağlanırsa yeniden dene |

GPU eğitimi toplam işin küçük kısmıdır. Tahmini hazırlama süreleri, yerel SSD ve
mevcut parserların yeniden kullanıldığı tek makine için planlama aralığıdır:

| Veri seti | Group/metadata denetimi | Raw feature extraction | Her L4 aday eğitimi + skor exportu |
|---|---:|---:|---:|
| ALFA | 10–30 dk | 5–20 dk | 2–8 dk notebook E2E |
| RflyMAD | 30–90 dk | mevcut parsed katmanda 20–60 dk; raw rebuild 2–6 saat | 8–25 dk |
| UAV-SEAD | 1–3 saat | 1–4 saat | 15–40 dk |
| UAV Attack | 30–90 dk | 15–60 dk | 3–10 dk |

Bu aralıklar benchmark değil, I/O/parser ağırlıklı planlama tahminidir. Her
stage başlangıç/bitiş zamanı, okunan byte, kaynak sayısı ve peak RAM'i JSON
progress kaydına yazılarak ilk gerçek turdan sonra tahminlerin yerini ölçüm alır.

## Donanım

- NVIDIA L4 mevcut küçük LSTM için yeterlidir ve v2'de doğrulanmıştır.
- T4 ALFA/Attack için yeterli olabilir; SEAD/RflyMAD'de L4 daha tutarlı seçimdir.
- A100 bu mimaride ana darboğazı çözmez; raw parse, Parquet decode ve feature
  join CPU/RAM/I/O ağırlıklıdır.
- Raw extraction için 16 GiB RAM alt sınır, 32 GiB rahat çalışma hedefidir.
  Dosya-bazlı streaming yapılmalı; 10–20 GiB raw veri tek DataFrame'e alınmamalıdır.

## Go/no-go kuralları

- Split group overlap varsa eğitim başlamaz.
- Validation normal exposure ön-kayıt minimumunu karşılamıyorsa operasyonel
  eşik iddiası kurulmaz.
- B1 phase context toplam metriği iyileştirirken bir fazda false-event yükünü
  ağırlaştırırsa sonuç birlikte raporlanır; faz gizlenmez.
- Raw blok katkısı testten seçilmez. Ablation adayları önceden dondurulur;
  başarısız bloklar raporda kalır.
- UAV Attack interval truth olmadan window/event-localization iddiası üretmez.

## Beklenen bilimsel sonuç

En makul iyileşme beklentisi UAV-SEAD ve RflyMAD'dedir. RflyMAD v2 interval AUC
0.907 ile gerçek zaman bölgelerini sıralayabildiğini göstermiştir; phase context
normal rejim geçişi false alarmını azaltabilir. UAV-SEAD interval AUC 0.747 iken
referans event recall yalnız %3.4'tür; burada phase/context'in amacı skoru daha
yüksek yapmak değil, kısa ve kopuk sinyali kalıcı event kararına dönüştürmektir.

ALFA'nın 10 normal uçuşluk temel envanteri ve group-family çakışmaları nedeniyle
yüksek varyans beklenir. UAV Attack'ta v2 AUC 0.0 olduğu için yeni feature'dan
önce campaign-safe calibration verisi sorunu çözülmelidir.
