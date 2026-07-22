# RflyMAD-Full v2 — Wind/Real Robustness Sonuçları

> Tarih: 2026-07-22 (Europe/Istanbul)  
> Durum: `development_only_robustness_complete`  
> Locked test: okunmadı  
> Operasyonel iddia: yasak

Bu rapor, kullanıcı tarafından sonuçlar görülmeden önce onaylanan
`RFLYMAD_V2_ROBUSTNESS_SOZLESMESI_20260722.md` sözleşmesinin sonuç kaydıdır.

## Deney tasarımı

- Dondurulmuş manifest: 5.380 development uçuşu; locked-test satırı yok.
- Beş dış rotasyonun her birinde normal ve Wind için train, inner kalibrasyon ve
  outer değerlendirme grupları ayrıdır.
- Scaler yalnız train NoFault uçuşlarıyla fit edildi.
- Fault uçuşları model, scaler veya threshold seçiminde kullanılmadı.
- W2'de Wind fault/pozitif yapılmadı; NoFault ve Wind reconstruction loss katkısı
  sampler ile 1:1 tutuldu.
- Seed, epoch, learning rate, aday sırası ve kapılar sonuçlardan önce donduruldu.
- Cluster-bootstrap canonical uçuş kimliği düzeyindedir; aynı uçuşun rotasyonları
  aynı cluster içinde tutuldu (`1000` örnek, seed `20260722`).

Frozen baseline eski tek-holdout sweep'tir. Yeni adaylar nested inner/outer
protokolü kullandığından baseline doğrudan eşdeğer bir yeniden-koşu değil, sözleşmede
dondurulmuş bağlam ve koruma hedefidir.

## Critical sonuçlar — beş rotasyon ortalaması

| Aday | Recall | Nonfault FA/s | Wind FA/s | Real Motor | Real Sensor | Real macro | Real normal FA/s | Real alarm uçuşu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | %60,43 | 1,28 | 28,46 | %20,20 | %8,35 | %14,28 | 0,80 | %2,00 |
| R1 | %59,55 | 3,28 | 28,13 | %23,88 | %22,03 | %22,95 | 12,08 | %19,82 |
| W1 | %10,78 | 2,54 | 7,85 | %23,88 | %22,03 | %22,95 | 12,08 | %19,82 |
| W2 | %55,90 | 2,65 | 17,78 | %10,61 | %17,47 | %14,04 | 3,79 | %6,64 |
| R2 | %59,46 | 3,24 | 27,83 | %23,37 | %21,77 | %22,57 | 11,68 | %19,82 |
| R3 | %58,90 | 3,20 | 27,83 | %23,27 | %21,65 | %22,46 | 11,68 | %19,82 |
| R4 convergence | %54,61 | 5,24 | 28,75 | %29,39 | %26,84 | %28,11 | 12,98 | %21,64 |

Critical/advisory değerlerin tamamı ile mean/std/min/max istatistikleri
`candidate_comparison_by_policy.csv` içindedir.

## Kapı kararları

| Aday | Hedef | Sonuç | Ana neden |
|---|---|---|---|
| R1 | Real | Geçmedi | Real macro %22,95 < %40; Real FA 12,08/s > 4/s |
| W1 | Wind | Geçmedi | Recall %10,78; Wind max 23,67/s; nonfault FA 2,54/s |
| W2 | Wind | Geçmedi | Wind ort. 17,78/s ve max 24,67/s; nonfault FA 2,65/s |
| R2 | Real | Geçmedi | Real macro %22,57; Real FA 11,68/s |
| R3 | Real | Geçmedi | Real macro %22,46; Real FA 11,68/s |
| R4 | Real convergence | Geçmedi | Real macro %28,11; Real FA 12,98/s; genel FA 5,24/s |
| RW1 | Birleşik | Çalıştırılmadı | Ayrı Real ve Wind adayı geçmedi |

R1 Real macro cluster-bootstrap %95 güven aralığı `%20,36–%25,62`, R2'nin
`%20,14–%25,34`, R3'ün `%19,95–%25,17` oldu. Hiçbiri araştırma-promosyon
eşiğine yaklaşmadı.

W1 Wind yükünü baseline'a göre yaklaşık %72 azalttı, fakat bunu HIL/SIL
threshold'larını yükseltip genel recall'ı yaklaşık 50 yüzde puan düşürerek yaptı.
W2'nin Wind azalması yaklaşık %37,5 ile dondurulmuş %40 koşulunun altında kaldı;
diğer FA/max koşulları da başarısızdı.

## Kullanıcı-onaylı convergence follow-up — R4

Kullanıcı sekiz epoch'un az olabileceğini belirttikten sonra ayrı ek sözleşme
sonuç koşusundan önce donduruldu:
`docs/RFLYMAD_V2_CONVERGENCE_EK_SOZLESME_20260722.md`.

R4 sabit epoch kullanmadı. Epoch 0 base checkpoint'i de aday olacak şekilde inner
Real-NoFault validation loss, `patience=12`, `min_delta=1e-4` ile izlendi. İlk 100
epoch tavanına dayanan rotasyonlar 500'e, yine sınıra dayananlar 2000 güvenlik
tavanına uzatıldı. Uzatma outer metriğe göre yapılmadı.

| Rotasyon | İlk val loss | En iyi val loss | En iyi epoch | Durma epoch | Stop |
|---:|---:|---:|---:|---:|---|
| 0 | 7,6656 | 7,3868 | 1 | 13 | patience |
| 1 | 8,4032 | 3,9386 | 780 | 792 | patience |
| 2 | 8,7191 | 8,1356 | 217 | 229 | patience |
| 3 | 5,8375 | 5,7902 | 1 | 13 | patience |
| 4 | 7,3151 | 3,9436 | 628 | 640 | patience |

Bu tablo sekiz epoch'un rotasyon 1, 2 ve 4 için gerçekten az olduğunu doğrular;
rotasyon 0 ve 3 içinse en iyi checkpoint epoch 1'dir. Tek bir sabit epoch bütün
session splitlerine uygun değildir.

Cap uzatmalarındaki deterministik validation replay maksimum mutlak farkı
`0–8,9e-16` aralığındadır. Böylece uzatma aynı eğitimin devamıdır.

R4, R3'e göre Real macro recall'ı `%22,46 → %28,11` yükseltti. Ancak:

- gerekli Real macro `%40` eşiğine ulaşmadı;
- Real macro bootstrap %95 GA yalnız `%24,92–%31,44` oldu;
- genel recall `%58,90 → %54,61` düştü;
- tüm nonfault FA `3,20 → 5,24/saat` yükseldi;
- Real-NoFault FA `11,68 → 12,98/saat` yükseldi;
- en düşük rotasyon Real macro yalnız `%12,24` oldu.

Dolayısıyla uzun fine-tune bir kısım Real transfer sinyalini artırdı, fakat domain
genellemesini ve alarm yükünü bozdu. Sorun yalnız epoch yetersizliği değildir.
R4 altıncı/son konfigürasyondur; yeni LR/patience/threshold taraması yapılmayacaktır.

Epoch grafikleri:

- `candidates/R4/00_validation_loss_R2_R3_R4_first25.png`: ilk 25 epoch yakın görünüm.
- `candidates/R4/01_validation_loss_R2_R3_R4.png`: tam R2/R3/R4 validation eğrileri.
- `candidates/R4/02_R4_train_validation_by_epoch.png`: R4 train/validation eğrileri.
- `candidates/R4/03_R4_best_and_stop_epochs.png`: best/stop epoch özeti.
- Ham epoch verisi: her `candidates/R4/rotation_*/training_history.csv`.
- Görselli ayrı deney raporu:
  `docs/RFLYMAD_V2_CONVERGENCE_DENEY_RAPORU_20260722.md`.

## Dondurulmuş durdurma kararının uygulanması

- Mevcut veri/temsil ile Real transfer gösterilemedi.
- Wind robustness çözülmedi.
- Ayrı adaylar geçmediğinden RW1 koşulu oluşmadı.
- Yeni Real veri veya yeni temsil sözleşmesi olmadan threshold/fine-tune avı
  sürdürülmeyecek.
- Kullanıcı ayrıca yeni protokol onaylamadan locked test açılmayacak.

Bu sonuçlar “model operasyonel”, “fizibil” veya “başarılı” anlamına gelmez.

## Son doğrulama

- TCN development testleri dahil güncel ilgili RflyMAD suite: `50 passed`.
- Dondurulmuş manifest: `5380/5380 development`.
- Altı adayın her birinde `5/5` outer rotasyon tamamlandı.
- Denetlenen 46 summary dosyasının tümünde `locked_test_features_read=false`.
- Çalışan `python/pythonw` süreci: `0`.
- `archive/` değişikliği: `0`.

## Artefaktlar

Kök:
`artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/`

- `final_summary.json`: nihai karar ve `rw1_required=false`.
- `candidate_comparison.csv`: kısa critical tablo.
- `candidate_comparison_by_policy.csv`: tam iki-politika karşılaştırması.
- `rw1_decision.json`: koşullu adayın neden çalıştırılmadığı.
- `development_manifest.parquet`: dondurulmuş development-only manifest.
- `base/rotation_*/`: nested base checkpoint/scaler/skorları.
- `candidates/<aday>/gate_summary.json`: koşul bazında kapı sonuçları.
- `candidates/<aday>/bootstrap_ci.json`: uçuş-cluster güven aralıkları.
- `candidates/<aday>/rotation_*/`: rotation/per-flight/family metrikleri ve ilgili
  checkpoint/training history.
