# ADS-B Basit Anomali — Faz Eşiği Kalibrasyon Profili

> Bu rapor yalnız `vertical_rate_ms`/`alt` doğal dağılımını profiller.
> Anomali tetik sayısı hesaplanmamıştır.

- Silver parçası: `data/objectstore/silver/adsblol_historical/part-20260710T125240580717Z-fcdf1dca.parquet`
- SHA-256: `2769ba79cb2611ed2b7d39ecae8c7fe1db6975b771e4195d530dbf2a071aa5e1`
- Kaynak: `v2026.02.28-planes-readsb-prod-0.tar`
- Örnek: 100 uçuş / 53,714 satır
- Eligible havuz: 640 uçuş

## Dağılımlar

- Örnekleme aralığı (s): `{'p50': 4.2799999713897705, 'p75': 12.0, 'p90': 19.710000038146973, 'p95': 19.880000114440918, 'p99': 26.790800189971943}`
- Dikey hız (m/s): `{'p01': -13.98, 'p05': -9.428, 'p10': -6.502, 'p25': -3.576, 'p50': 0.0, 'p75': 1.3, 'p90': 10.404, 'p95': 13.98, 'p99': 18.857}`
- Mutlak dikey hız (m/s): `{'p25': 0.325, 'p50': 2.926, 'p75': 6.828, 'p90': 11.867, 'p95': 14.63, 'p99': 19.507}`
- İrtifa (m): `{'p01': 38.1, 'p05': 182.9, 'p25': 640.1, 'p50': 1935.5, 'p75': 8793.5, 'p95': 11582.4, 'p99': 12192.0}`
- Uçuş robust irtifa aralığı (m): `{'p05': 4.332000000000381, 'p25': 481.92500000000007, 'p50': 4842.880000000001, 'p75': 10312.94, 'p95': 11452.900000000001}`

## Taslak eşiklerin doğal kapsama oranı

- `|vertical_rate_ms| < 1.0`: 34.611%
- `vertical_rate_ms > 2.5`: 21.149%
- `vertical_rate_ms < -2.5`: 28.637%

Bu oranlar anomali sonucu veya başarı metriği değildir; yalnız faz sınırı ön-kayıt girdisidir.
