# ADS-B contextual_physics_v2 calibration veri-kalitesi olayi

Tarih: 2026-07-27
Faz: D - dogal calibration
Durum: raporlandi, karantinaya alindi, kosu tamamlandi

## Olay

Ilk Faz D kosusu ayni flight_id + timestamp_utc anahtarinda celisen
on_ground degerleri gordugu icin fail-loudly durdu. Kullanici, bozuk kayitlarin
ayri raporlanip ayiklanarak calismanin devam etmesini acikca istedi.

Belirsiz bir cogunluk/son-satir secimi yapmak yerine, herhangi bir celiski
tasiyan ucusun tamami calibration rolunden karantinaya alindi. Bu politika model
agirliklarini, dondurulmus alarm butcesini, epoch sayisini, threshold adaylarini
veya truth-v2 corpus'unu degistirmedi.

## Karantina envanteri

| flight_id | timestamp_utc | degerler | kaynak satir |
|---|---:|---|---:|
| 2026-02-28:a6e4e8_000 | 1772252095.720000 | False, True | 2 |
| 2026-02-28:ac1df7_001 | 1772259002.190000 | False, True | 3 |
| 2026-02-28:86837c_003 | 1772270168.530000 | False, True | 2 |

Kaynaklar sirasiyla:

- part-20260710T125856787951Z-311b43f9.parquet
- part-20260710T130028669534Z-4b485007.parquet
- part-20260710T130913260131Z-50d25ee6.parquet

## Etki

- Secilen calibration ucusu: 763
- Karantinaya alinan ucus: 3
- Tutulan calibration ucusu: 760
- Celiskili anahtar: 3
- Celiskili kaynak satiri: 7
- Karantinaya giren tum feature satirlari: 4.298
- Nihai skorlanan model penceresi: 318.053

## Artefaktlar

- Karantina tablosu:
  artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1/calibration_on_ground_quarantine.parquet
- Faz D raporu:
  artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1/calibration_report.json
- Faz D rapor SHA-256:
  8e6fa2a355fa4f49ba96769b4f3a56700a5ede39e08d87f07672d08079e86b73
- Tum Faz D artifact checksum kontrolleri: 0 hata

Bu olay model performansi veya threshold sonucu degildir; kaynak kayitlarin
ayni kimlik-zaman anahtarindaki zemin-durumu tutarsizligidir.
