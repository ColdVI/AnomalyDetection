# Mimari Kararlar

## ADR-001: Bronze veri kaynakları

- Durum: Kabul edildi
- Tarih: 2026-06-29

OpenSky yerine `adsb.lol`; generic MAVLink örnekleri yerine ALFA ve UAV Attack veri
setleri kullanılacaktır. `adsb.lol` kimlik doğrulama ve günlük kredi yükünü azaltırken
Türkiye hava trafiğini hem tarihsel hem gerçek zamanlı sağlar. ALFA fault ground-truth,
UAV Attack ise benign/malicious saldırı etiketleri sunduğundan sonraki anomali tespiti
çalışmasının ölçülebilir olmasını sağlar.

ALFA processed CSV dosyaları ground-truth için birincil Bronze girdisidir. `.bin` ve
`.tlog` dosyalarının `pymavlink` ile ayrıştırılması opsiyonel ek yoldur; ROS `.bag`
dosyaları `pymavlink` ile okunmayacaktır.

Bronze yalnızca ham alanları korur, provenance ekler ve adsb.lol kayıtlarında Türkiye
bbox filtresi uygular. Birim dönüşümü, kolon harmonizasyonu ve koordinat ölçekleme
Silver katmanına aittir.
