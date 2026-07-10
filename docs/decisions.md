# Aktif Kararlar

## ADR-001 — ADS-B model çalışması sıfırlandı

- Tarih: 2026-07-10
- Durum: accepted

Önceki non-ADS-B ML hattı
`archive/2026-07-10_legacy_non_adsb_ml/`, mevcut iki ADS-B yaklaşımı ise
`archive/2026-07-10_rejected_adsb_attempts/` altına kaldırıldı.

Gerekçe: biriken deneyler aktif kapsamı anlaşılmaz hale getirdi; mevcut ADS-B
sonuçları da gerçek binary anomaly detector başarısını göstermedi. Yeni ADS-B
çalışması model koduyla değil veri envanteri, zaman serisi/harita incelemesi,
gözlenebilirlik tablosu ve açık değerlendirme sözleşmesiyle başlayacaktır.

Korunan aktif altyapı: gerçek historical ADS-B tar parserı ve onun ortak
provenance/IO bağımlılıkları. Bu altyapı model veya baseline olarak kabul edilmez.
