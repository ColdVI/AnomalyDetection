# Aktif Çalışma Kuralları

## Proje durumu

Proje 2026-07-10 tarihinde ADS-B için sıfırlandı. Eski non-ADS-B ML hattı
`individual_anil/gecmis_calismalar/` altında (arşivsel, çalıştırılabilir
değil), aynı gün üretilen iki reddedilen ADS-B yaklaşımı
`individual_anil/reddedilen_denemeler/` altındadır.

## Zorunlu sınırlar

- `individual_anil/reddedilen_denemeler/` salt-okunur tarihçedir; açık kullanıcı
  talebi olmadan değiştirme, oradan kod import etme veya eski modeli yeniden
  aktifleştirme.
- `src/silver/parse_adsblol_historical.py` veri okuma altyapısıdır, kabul edilmiş
  bir anomaly modeli değildir.
- Yeni ADS-B model koduna başlamadan önce `individual_anil/adsb/README.md`
  içindeki Aşama 0 çıktıları hazırlanmalı ve problem sözleşmesi kullanıcıyla
  netleştirilmelidir.
- Satır, event ve uçuş düzeyi metrikleri birbirine karıştırılmamalıdır.
- Sentetik anomaly yalnız enjeksiyonun fiziksel anlamı ve gözlenebilirliği
  doğrulandıktan sonra değerlendirme ground-truth'u olabilir.
- Yeni çalışma kendi temiz namespace'ini kullanmalı; arşivdeki `src/adsb` veya
  `src/adsb_behavioral` paketlerini kopyalayarak başlamamalıdır.
