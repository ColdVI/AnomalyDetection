# Aktif Çalışma Kuralları

## Proje durumu

Proje 2026-07-10 tarihinde ADS-B için sıfırlandı. Eski non-ADS-B ML hattı ve aynı
gün üretilen iki ADS-B yaklaşımı `archive/` altındadır.

## Zorunlu sınırlar

- `archive/` salt-okunur tarihçedir; açık kullanıcı talebi olmadan değiştirme,
  oradan kod import etme veya eski modeli yeniden aktifleştirme.
- `src/silver/parse_adsblol_historical.py` veri okuma altyapısıdır, kabul edilmiş
  bir anomaly modeli değildir.
- Yeni ADS-B model koduna başlamadan önce `adsb/README.md` içindeki Aşama 0
  çıktıları hazırlanmalı ve problem sözleşmesi kullanıcıyla netleştirilmelidir.
- Satır, event ve uçuş düzeyi metrikleri birbirine karıştırılmamalıdır.
- Sentetik anomaly yalnız enjeksiyonun fiziksel anlamı ve gözlenebilirliği
  doğrulandıktan sonra değerlendirme ground-truth'u olabilir.
- Yeni çalışma kendi temiz namespace'ini kullanmalı; arşivdeki `src/adsb` veya
  `src/adsb_behavioral` paketlerini kopyalayarak başlamamalıdır.
