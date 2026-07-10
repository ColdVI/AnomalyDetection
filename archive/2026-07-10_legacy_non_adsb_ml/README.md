# Legacy Non-ADS-B ML Archive

Arşiv tarihi: 2026-07-10

## Neden arşivlendi?

ALFA, UAV Attack, UAV-SEAD ve RFLY üzerinde ML-0…ML-16 boyunca çok sayıda model,
eşik, füzyon, kalibrasyon ve değerlendirme denemesi birikti. Çalışmaların önemli
bölümü operasyonel recall/yanlış-alarm kapılarını birlikte geçemedi; aktif repo
görünümü hangi hattın güncel olduğunu anlatamaz hale geldi. Kullanıcı yeni odağı
yalnız ADS-B olarak belirlediği için bu hat aktif geliştirmeden çıkarıldı.

## İçerik

- `src/ml/`: eski model ve değerlendirme kütüphanesi;
- `notebooks/`: eski analiz/eğitim notebookları;
- `scripts/`, `tests/`: ML ve RFLY çalıştırıcıları ile testleri;
- `docs/`: ML planları, bulgular, eski karar hafızası ve pipeline dokümanları;
- `artifacts/`: model, scaler, policy, metrik, grafik ve eğitim izleri;
- `logs/`: kökte birikmiş çalışma logları.

## Son durum

Bu arşivdeki modeller production adayı veya yeni ADS-B baseline'ı değildir. Ham
veriler `data/` altında yerinde bırakıldı; bu taşıma veri silme işlemi değildir.
Arşiv, geçmişte ne denendiğini gerektiğinde denetlemek için süresiz korunacaktır.
