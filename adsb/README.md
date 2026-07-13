# ADS-B — Sıfırdan Başlangıç

Durum (2026-07-10 güncellemesi): Aşama 0 (madde 1-4, 6) gerçek veriyle tamamlandı.
Madde 5 pragmatik varsayılanla karara bağlandı (aşağıda). Kullanıcı, Aşama 0 arka
planda sürerken model ailesine (Dense-AE/LSTM-AE/USAD/LSTM-forecaster) paralel
başlanmasını EXPLICIT olarak istedi ("şimdi paralel başlat") — bu, aşağıdaki "ilk
onay kapısı" kuralının bilinçli bir istisnasıdır, kural iptal edilmedi.

## Ana hedef

Bir ADS-B uçuş gözleminde davranışsal anomaly olup olmadığına açık bir binary cevap
üretmek:

`anomaly = yes | no`

Modelin ayrıca karar nedeni, kullanılan gözlem aralığı ve veri-kalitesi durumunu
raporlaması gerekir. `unknown / insufficient_data`, anomaly ile aynı sınıf değildir.

## Aşama 0 — modelden önce tamamlanacaklar

1. ✅ Üç tar arşivinin (2026-02-28, 03-01, 03-16 — toplam 256.150.550 satır, 638 Silver
   parça, `STORAGE_BACKEND=local`) envanteri çıkarıldı: `adsb/reports/inventory_profile.json`
   (`adsb/inventory.py`, `scripts/adsb_inventory_report.py`). Format 3 günde de stabil
   (her satır 14 elemanlı), hiçbir tar'da UAV/drone kategorisi (B6/B7) — tek istisna
   2026-03-01 örnekleminde 25 kez görülen `B6`, henüz görsel doğrulaması yapılmadı.
2. ✅ Uçuş başlangıç/bitiş: `adsb/segmentation.py` (boşluk-tabanlı, 1800s varsayılan +
   `flags_new_leg` çapraz-doğrulama). Gerçek veride test edildi: 1500 uçak → 4230 uçuş,
   `flags_new_leg` uyuşma oranı **%60.4** (tam değil, dürüstçe raporlanıyor).
3. ⏳ Ham zaman serisi/harita galerisi henüz üretilmedi (bir sonraki adım).
4. ✅ Fiziksel ilişki ölçülebilirlik kararı: `adsb/reports/measurability_table.md`
   (gerçek satır-düzeyi kapsama oranlarıyla — `alt`/`vertical_rate_ms` %89.4,
   `ground_speed_ms` %98.1, `track_deg` %95.2, `roll_deg` %28.4 — forward-fill YOK).
5. Binary değerlendirme birimi: **sabit pencere** seçildi (pragmatik varsayılan,
   gerekçe measurability_table.md'de) — dört model mimarisi de zaten pencere üstünde
   çalışıyor.
6. ✅ Sentetik bozulmalar: `adsb/synthetic.py` (`PHYSICS_BREAK_RECIPES`, 5 senaryo),
   test-only, gerçek veriye asla yazılmıyor (`save_synthetic_batch` path guard'ı).
   Kalıcı korpus artık diskte: `data/objectstore/synthetic/adsb/` (8910 val-uçuşu ×
   5 senaryo + clean, 765MB, `scripts/adsb_generate_synthetic_dataset.py` — bkz.
   ADR-023). Şu an 60/638 Silver parça; tam-hacim sonraki ölçek büyütme adımı.

## İlk onay kapısı

Aşama 0 veri raporu kullanıcı tarafından görülüp onaylanmadan model ailesi, eşik,
normal öğrenme yöntemi veya headline recall seçilmeyecek. **İstisna (2026-07-10):**
kullanıcı Aşama 0 arka planda sürerken Dense-AE/LSTM-AE/USAD/LSTM-forecaster
mimarilerine paralel başlanmasını açıkça istedi — bu kod yazıldı ve gerçek veriyle
ilk eğitim/tanı turu çalıştırıldı (`scripts/adsb_train_baseline_models.py`), ama
bu bir headline recall/production karar ANLAMINA GELMİYOR — yalnız pipeline
doğrulaması. Galeri (madde 3) ve tam-hacim eğitim hâlâ bekliyor.

## Aktif olmayanlar

Önceki ADS-B kodu ve sonuçları
`archive/2026-07-10_rejected_adsb_attempts/` altındadır. Yeni çalışma onların devamı
değildir.
