# RflyMAD-Full v2 — Convergence Fine-Tune Ek Sözleşmesi

> Tarih: 2026-07-22 (Europe/Istanbul)  
> Durum: kullanıcı tarafından açıkça onaylanmış takip deneyi  
> Kapsam: development-only; locked test okunmaz

## Neden ek deney var?

Önceki sözleşmedeki R2 ve R3 yalnız `3` ve `8` epoch'luk sınırlı adaylardı.
R3 eğitim geçmişinde iki rotasyonun en iyi inner validation değeri epoch 7–8'de
olduğu için sekiz epoch bu rotasyonlarda convergence kanıtı değildir. Kullanıcı
“converge edene kadar runlasak” diyerek uzun koşuyu açıkça istedi.

Bu belge ilk beş adayın preregistered sonucunu geriye dönük değiştirmez. Yeni koşu
ayrı bir kullanıcı-onaylı follow-up/diagnostic adaydır.

## Dondurulmuş R4 protokolü

- Başlangıç: her outer rotasyonun mevcut nested base AE checkpoint'i.
- Fine-tune verisi: yalnız o rotasyonun train Real-NoFault uçuşları.
- Checkpoint ölçütü: inner Real-NoFault masked reconstruction validation loss.
- Epoch 0, yani değiştirilmemiş base model, geçerli en iyi checkpoint'tir. Fine-tune
  validation'ı bozarsa zorla fine-tuned model seçilmez.
- Learning rate: `1e-4`; AdamW weight decay: `1e-4`; batch size: `128`.
- Maksimum epoch: `100`.
- Early stopping patience: iyileşmesiz `12` epoch.
- Minimum anlamlı iyileşme: `1e-4` validation loss.
- Her rotasyonda minimum inner-validation checkpoint saklanır; outer sonuç hiçbir
  epoch veya checkpoint seçiminde kullanılmaz.
- Threshold: mevcut normal-only inner kalibrasyon ve standart critical/advisory
  bütçeleri; R1 threshold gevşetmesi eklenmez.
- Fault uçuşları scaler, fine-tune, early stopping veya threshold seçiminde yoktur.
- Beş outer rotasyonun tümü raporlanır; mevcut Real/Wind kapıları değiştirilmez.

R4 altıncı ve son değerlendirilen konfigürasyondur. İlk sözleşmedeki koşullu RW1'in
ön şartı oluşmadığı ve RW1 çalıştırılmadığı için toplam değerlendirilen konfigürasyon
sayısı yine altıdır. R4 sonucu başarısız olursa yeni epoch/LR/patience taraması
yapılmaz.

## Zorunlu epoch raporları

Her rotasyon için:

- epoch 0 dahil validation loss;
- epoch başına train ve validation loss;
- en iyi epoch ve durma epoch'u;
- stop reason ve en iyi checkpoint'in base olup olmadığı;
- R2/R3/R4 validation-loss karşılaştırma grafiği;
- R4 train/validation eğrileri ve best/stop epoch grafiği.

Tüm çıktılar `locked_test_features_read=false` ve
`operational_claim_allowed=false` taşır.

## Kullanıcı onayı

Kullanıcı 2026-07-22 tarihinde “8 epoch az değil midir… converge edene kadar
runlasak… otonom devam et” diyerek bu takip deneyini açıkça onayladı.
