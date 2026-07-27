# Dört Dataset Probabilistic GPU v1 — Sonuç Öncesi Kayıt

**Dondurma tarihi:** 2026-07-27  
**Namespace:** `four_dataset_probabilistic_gpu_v1`

## Amaç

ALFA, UAV-Attack, UAV-SEAD ve RflyMAD üzerinde daha önce aynı biçimde
çalıştırılmamış ortak bir temporal baseline kurulur: yalnız normal/benign train
uçuşlarından, geçmiş 32 satıra bakarak bir sonraki satırın kanal bazlı ortalaması
ve ölçeği tahmin edilir. Optimizasyon maskeli Gaussian NLL ile yapılır.

Bu çalışma eski sonuçları geçersiz kılmaz. ALFA ve UAV-Attack veri-tavanlı küçük
development corpuslarıdır; UAV-SEAD holdout'u kapalı kalır; RflyMAD'in daha önce
başarısız olan Wind/Real transfer kapıları yeniden yorumlanmaz.

## Neden Gaussian NLL?

Önceki AE/TCN turlarının çoğu MSE reconstruction ya da BCE/CrossEntropy kullandı.
Gaussian NLL burada "daha havalı" olduğu için değil, uçuş rejimine göre doğal
tahmin belirsizliği değişebildiği için kullanılır. Model her kanal için hem
`mu` hem `sigma` üretir. Büyük fakat öngörülebilir oynaklık ile küçük fakat
beklenmedik sapma böylece aynı şey sayılmaz.

## Dondurulmuş sözleşme

- Kaynak: ALFA, UAV-Attack ve UAV-SEAD için mevcut Gold feature Parquet'leri;
  RflyMAD için mevcut Full-v2 `parsed_10hz` Parquet'leri. Ham veriler yeniden
  parse edilmez.
- Split: ilk üç corpus için mevcut `split_manifest.json` içindeki yalnız
  `split_00`; RflyMAD için Full-v2 manifestindeki development fold'larından
  normal train=`2,3,4`, normal validation=`0`, tüm test=`1`. Locked-test açılmaz.
  Tüm ayrımlar uçuş düzeyindedir.
- Optimizer girdisi: yalnız train rolündeki normal/benign uçuşlar.
- Robust ölçek: train median/MAD, `1.4826`, `MAD <= 1e-6` kanal dışlama,
  ardından `[-5, 5]` clip.
- Pencere: 32 geçmiş satır, tek satır ileri hedef, uçuş ve `5 s` gap sınırını
  geçmez.
- Model: tek katmanlı LSTM, hidden 64, kanal başına `mu` ve sınırlı `sigma`.
- Eğitim: 30 epoch, batch 1024, Adam `1e-3`, gradient clip `1.0`, seed `20260727`.
- Mixed precision kapalıdır; CUDA kullanılır.
- Alarm eşiği yalnız normal validation skorunun sabit `0.995` quantile'ıdır.
- Test label'ları model, scaler veya eşik seçiminde kullanılmaz.
- Magnitude diagnostic: eğitilmiş skorun hem random-init skorla hem hedef
  büyüklüğüyle Spearman korelasyonu; herhangi biri `>= 0.8` ise sonuç
  magnitude-dominated olarak işaretlenir.

Tüm sayısal değerler ve feature listeleri
`configs/four_dataset_probabilistic_gpu_v1.json` içinde sonuç görülmeden
dondurulmuştur. V1 sonuçları görüldükten sonra feature, epoch, clip veya eşik
değiştirilemez; yeni deneme yeni namespace ve yeni ön-kayıt gerektirir.
