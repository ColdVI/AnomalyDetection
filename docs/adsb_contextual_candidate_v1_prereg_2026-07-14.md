# ADS-B contextual-physics candidate v1 — ön-kayıt sözleşmesi

## Durum ve kapsam

Bu belge, kullanıcının 2026-07-14 tarihli açık yönlendirmesiyle araştırılan ve uygulanması
onaylanan yeni aday ailesinin yapısal sözleşmesidir. Adayın namespace'i
`contextual_physics_v1`'dir. Step-5 CUSUM, ADR-025 kuralı, tarihsel üç NN veya Step-7 FAIL
sonucu değiştirilmez; bu aday bunlardan birinin yeniden adlandırılmış devamı değildir.

Bu revizyonda gerçek-veri bilimsel koşusu, threshold seçimi, config freeze, Step-8 terfisi veya
holdout erişimi **yoktur**. Operasyonel toplam alarm bütçesi kullanıcı tarafından henüz sayısal
olarak tanımlanmadığı için kod alert alpha olmadan alarm üretmez.

## Problem sözleşmesi

Çıktı tek ve açıklamasız bir reconstruction skoru değildir. Her skor aşağıdakileri taşır:

- `anomaly_type` ve fiziksel `channel`;
- skorun ait olduğu nedensel normal bağlam (`context_phase`, `context_cadence`);
- conformal p-değeri, kullanılan fallback seviyesi ve calibration örnek sayısı;
- anlık/süreklilik/birikimli temporal evidence;
- `alarm=yes|no` veya veri yetersizse score dışı durum.

Satır, event, uçuş ve scoreable uçuş-saat metrikleri ayrı raporlanır. Sentetik truth-v2 yalnız
değerlendirmede kullanılabilir; fit, robust scaling, normal-tail calibration veya threshold
seçiminde kullanılamaz.

## Yapısal olarak dondurulan kararlar

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

## Aday fizik kanalları

- `speed_residual`: ground-speed spike/bias;
- `vertical_rate_residual`: vertical-rate spike/freeze;
- `heading_residual`: track-frozen/direction inconsistency;
- `east_velocity_residual`, `north_velocity_residual`: stealthy position ramp için işaretli,
  zaman-birikimli kanıt;
- `altitude_source_residual`: yalnız fit-normal MAD pozitifse;
- altitude availability, message gap ve declared quality: NN residual'ına karıştırılmadan mevcut
  S2 reason-code katmanında.

## Sayısal freeze öncesi açık kullanıcı kararları

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

## Veri rolleri ve değerlendirme sırası

1. Natural fit: scaler ve NN; sentetik satır sayısı zorunlu olarak sıfır.
2. Ayrı natural calibration: conformal tail ve önceden verilmiş bütçeye göre karar config'i.
3. Natural development: episode/uçuş/uçuş-saat burden ve bağlam/cadence kararlılığı.
4. Donmuş natural rehearsal: geri besleme yok.
5. Truth-v2: yalnız corrected event recall, delay, active coverage ve channel attribution.
6. Ana aday kıyası: aynı natural burden'da ADR-025 rule ve yeni aday; pooled AUC tek gate değil.
7. Üçlü kör holdout havuzu bu adayın geliştirme verisi değildir ve ayrı unseal kararı olmadan
   açılmaz.

## Normal-only eğitim freeze'i — 2026-07-14

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

## İlk gate

Aday ancak provenance/checksum tam, sentetik eğitim sıfır, magnitude-domination false, conditional
calibration desteği yeterli ve natural development/rehearsal burden önceden verilen bütçeyi sağlıyor
ise truth-v2 fayda kıyasına geçebilir. Bir koşul fail olursa config sonuçlara göre aynı run içinde
değiştirilmez; yeni config yeni namespace ve yeni ön-kayıt gerektirir.
