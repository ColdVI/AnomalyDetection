# ADS-B contextual_physics_v2 v4 sessiz sonlanma incident'i — 2026-07-25

## Yönetici özeti

`20260724_contextual_physics_v2_train_v4` koşusu yaklaşık 24 saat 14 dakika
duvar saati ve 33,5 CPU-saat tüketip epoch 1'i tamamladıktan sonra epoch 2
sırasında **hiçbir Python exception, stderr kaydı veya nihai rapor bırakmadan**
sonlandı. Bu küçük bir konsol aksaklığı değildir: 275.468.643 fiziksel satırdan
üretilen 236.129.563 pencere üzerinde 461.574 minibatch işleyen bir epoch'un
ardından süreç dışarıdan veya çalışma ortamı tarafından sessizce kaybolmuştur.

Run nihai model, `training_report.json`, magnitude-domination sonucu veya
checksum zinciri üretmedi. Bu nedenle v4 bilimsel sonuç değildir ve Faz D–G
girdisi olarak kullanılamaz. Eşik, alarm bütçesi, epoch sayısı, veri günleri,
model mimarisi ve ön-kayıt değerleri değiştirilmemiştir.

## Kanıtlanmış zaman çizelgesi (Europe/Istanbul)

- 2026-07-24 11:08:05 — PID 16908 başladı.
- 2026-07-24 11:25:29 — scaler sonrası epoch hesaplaması başladı.
- 2026-07-25 11:10:27 — epoch 1 atomik checkpoint'i yazıldı.
- 2026-07-25 11:21:47 — harici monitor PID'yi son kez canlı gördü;
  CPU=120.701,4 s, RSS=2.555.498.496 byte.
- 2026-07-25 11:22:24 — PID bulunamadı; rapor/error/INCOMPLETE artefaktı yoktu.

Epoch-1 kaydı:

- windows: 236.129.563
- batches: 461.574
- mean weighted Gaussian NLL: 0,8160950942642353

## Kök neden durumu

Kök neden **bilinmiyor**. Stderr boş olduğu ve trainer'ın exception handler'ı
çalışmadığı için normal bir Python exception lehine kanıt yoktur. Olası işletim
sistemi/host/process-lifecycle sonlandırmalarından biri olabilir; mevcut kanıt
bunların arasında dürüstçe ayrım yapmaya yetmez. Bu olay magnitude-domination,
model kalitesi, truth-v2 sonucu veya veri günü ayrışması olarak yorumlanamaz.

## İkinci altyapı kusuru: yazılan fakat okunmayan checkpoint

Orijinal trainer checkpoint'i “incident recovery” amacıyla model, optimizer,
PyTorch RNG ve history ile atomik yazmasına rağmen onu yükleyen bir resume yolu
içermiyordu. Aynı komutu yeniden çalıştırmak mevcut run-dir nedeniyle hemen
başarısız olacak; yeni run-dir ise epoch 1'i baştan hesaplayacaktı. Dolayısıyla
“recovery checkpoint” sözü operasyonel olarak eksikti.

Bu eksikliği gidermek için `scripts/adsb_resume_contextual_physics_v2.py`
eklendi. Recovery yeni run kimliği kullanır; parent manifest/checkpoint hash'ini,
tamamlanan epoch sayısını ve kod lineage'ını kaydeder; model+optimizer+RNG
state'ini aynen yükler; dondurulmuş toplam 8 epoch'un yalnız kalan kısmını
çalıştırır. Orijinal v4 dizini değiştirilemez incident delili olarak kalır.

## Contract doğrulaması

Sonlanma sonrasında v4 manifest'inde kayıtlı yedi eğitim kodu hash'i, dondurulmuş
training config hash'i ve Step-5 manifest hash'i güncel dosyalarla bire bir
eşleşti (`all_match=true`). Fit-expansion günleri değişmedi:
`2024-09-01`, `2025-02-15`, `2025-06-15`.

Recovery yükleme denetimi epoch 1 history'sini, model config'ini, altı aktif
scaler kanalını, sekiz optimizer state girdisini ve 5.056-byte PyTorch RNG
state'ini başarıyla doğruladı. İlgili contextual regresyon ailesi:
`31 passed`.

## Kullanım yasağı

v4'ün checkpoint'i yalnız recovery parent'ıdır. Nihai model değildir;
kalibrasyon, eşik seçimi, truth-v2 değerlendirmesi veya Faz D–G için doğrudan
kullanılamaz. Yalnız lineage'ı doğrulanmış recovery run'ının eksiksiz
`training_report.json` ve checksum zinciri üretmesi bu yasağı kaldırabilir.
