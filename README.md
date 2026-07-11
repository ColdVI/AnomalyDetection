# ADS-B Anomaly Detection — Clean Restart

Bu repo 2026-07-10 tarihinde sadeleştirildi. Önceki ALFA, UAV Attack, UAV-SEAD,
RFLY ve ML-0…ML-16 deneyleri aktif çalışma alanından çıkarıldı. Aynı gün yazılan
iki ADS-B model denemesi de kullanıcı tarafından baseline olarak kabul edilmedi ve
ayrı bir arşive kaldırıldı.

Aktif hedef: gerçek ADS-B arşivlerinden, tanımı baştan açık kurulmuş bir
`anomaly = yes/no` sistemi geliştirmek.

Başlangıç noktaları:

- `adsb/README.md`: sıfırdan başlangıç sözleşmesi;
- `docs/adsblo_data_format_reference (1) 2026-07-10 amt 11.03.27.md`: veri formatı;
- `src/silver/parse_adsblol_historical.py`: korunmuş ham veri okuma altyapısı;
- `archive/README.md`: önceki çalışmaların indeksi.

`archive/` altındaki kod ve sonuçlar aktif baseline değildir; yeni modele import
edilmez veya başarı kanıtı olarak kullanılmaz.
