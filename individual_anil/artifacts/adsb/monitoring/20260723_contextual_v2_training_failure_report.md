# contextual_physics_v2 Faz C eğitim başarısızlığı

Tarih: 2026-07-23 (Europe/Istanbul)

## Durum

Koşu tamamlanmadı ve hiçbir model checkpoint'i veya training_report.json
üretmedi. Run dizinindeki INCOMPLETE_DO_NOT_USE.md marker'ı otoritatif
sonuçtur; bu koşunun artefaktları model sonucu olarak kullanılamaz.

## Zaman çizelgesi

- Eğitim başlangıcı: 2026-07-23 12:43:38 +03:00, PID 19464.
- run_manifest.json: 12:44:07.
- fit_scaler.json: 13:02:48.
- derived_training_config.json: 13:03:21.
- Son canlı monitor örneği: 19:17:08.
- Hata marker'ı: 19:19:49.
- Sürecin kaybolduğu ilk monitor örneği: 19:22:08.

## Otoritatif hata

ValueError: unexpected contextual forecaster input shape

## Kök neden

Eğitim döngüsü NumPy pencere dizisini doğrudan bir torch.Tensor ile
indeksliyordu. Bir parçanın son minibatch'i tam bir satır içerdiğinde tek
elemanlı Torch indeksi NumPy tarafından skaler indeks gibi yorumlanıyor ve
baş batch eksenini sessizce düşürüyordu:

- beklenen: (1, history_rows, input_features)
- elde edilen: (history_rows, input_features)

Forecaster'ın 3-D giriş sözleşmesi bu nedenle doğru biçimde fail-loudly hata
verdi. Bu threshold, veri rolü, magnitude sonucu veya model mimarisi sorunu
değildir; batching uygulama hatasıdır.

## Hazırlanan düzeltme

scripts/adsb_train_contextual_physics_v2.py içindeki minibatch indeksi açıkça
1-D NumPy dizisine çevriliyor. Permütasyon sırası ve Torch RNG tüketimi
değişmiyor; yalnız tek satırlı remainder batch'in baş ekseni korunuyor.

Regresyon testi:

- tests/test_adsb_train_contextual_physics_v2.py
- Contextual ilgili test turu: 39 passed.
- Yeni shape testi + persistence_v2 turu: 12 passed.

## Bilimsel kapı durumu

- model_state.pt: YOK
- training_report.json: YOK
- magnitude_domination_flagged_at_0_8: DEĞERLENDİRİLEMEDİ
- Faz D-G: BAŞLATILMADI

## Yeniden başlatma önkoşulları

1. Minimal tracked düzeltme ve regresyon testi kullanıcı tarafından gözden
   geçirilip commit edilmeli; eğitim scripti temiz tracked worktree ister.
2. Başarısız v1 run dizini audit kanıtı olarak korunmalı.
3. Yeni, boş bir run dizininde (öneri:
   20260724_contextual_physics_v2_train_v2) aynı dondurulmuş config/veriyle
   eğitim yeniden başlatılmalı.
4. Uzun koşudan önce model+optimizer+RNG içeren atomik epoch checkpoint/resume ve
   kalıcı stdout log desteği ayrıca sonuçtan bağımsız yürütme kontrolü olarak
   ele alınmalı; mevcut patch bunu henüz eklemiyor.
