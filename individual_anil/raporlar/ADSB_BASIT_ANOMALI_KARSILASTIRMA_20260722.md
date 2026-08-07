# ADS-B Basit Anomali — İki Kuralın Karşılaştırması ve İlerleme Kararı

> Tarih: 2026-07-22
> Kapsam: aynı dondurulmuş 100-uçuş Silver örneği
> Karar türü: keşif önceliği; operasyonel/model başarısı değildir

## Ortak kapsam

- 100 uçuş / 53.714 satır.
- Üç-faz kuralı 57 uçuşu çözdü; 43 eksik/uyumsuz trace `uncertain` kaldı.
- Eşikler sonuçlardan önce
  `ADSB_BASIT_ANOMALI_ONKAYIT_20260722.md` içinde donduruldu.
- Ground truth yok; triggerlar doğrulanmış anomaly değildir.

## Yan yana sonuç

| Boyut | İrtifa sapması | GPS/rota sapması |
|---|---:|---:|
| Değerlendirilebilir uçuş | 57 | 95 |
| Triggerlı uçuş | 2 (%3,51) | 13 (%13,68) |
| Olay | 2 | 24 |
| Ana bağlam sorunu | Faz sınırı / meşru seviye değişimi | Düşük-hız bearing kararsızlığı |
| Cruise olayı | 2 | 0 |
| Nitel incelemede doğrulanmış anomaly | 0 | 0 |

## İrtifa kuralının okuması

İki olay da uzun ve sayısal olarak belirgindir: `277–478 s`, peak `244–335 m`.
Ancak ham çevre incelemesi:

- bir olayın landing başlamadan hemen önceki normal descent bölümünü cruise içinde
  yakaladığını;
- diğerinin 1402 m ve 1158 m'deki iki stabil uçuş seviyesinden ilkini, tüm-cruise
  medyanına göre sapma saydığını gösterdi.

İki olayda da barometrik/geometrik kaynak uyuşmazlığı bayrağı yoktur. Mevcut iki
trigger doğrulanmış anomaly değil; fakat false-positive mekanizması açık ve dar:
faz sınırı ile çok-seviyeli cruise referansı.

## GPS/rota kuralının okuması

24 olayın tamamı düşük-hız bağlamındadır. Olayların 23'ünde bütün satırlar
`<30 m/s`; kalan olayda düşük-hız oranı `%69,23`tür. Faz dağılımı 17 takeoff,
5 landing, 2 uncertain ve 0 cruise'dur. Peak residual'ların çoğunun 180 dereceye
yaklaşması, uçuş-rotası sapmasından çok düşük hızda iki noktalı bearing'in
kararsız/ters yönlü hesaplanmasıyla uyumludur.

Bu nedenle mevcut rota kuralı GPS/spoofing keşfi olarak ilerletilmez. İleride
yeniden açılacaksa minimum hız, timestamp hizası ve position-derived bearing
güvenilirliği yeni bir ön-kayıtta ele alınmalıdır; mevcut sonuç görülüp eşik
sessizce değiştirilmeyecektir.

## Karar

**İlerletilecek bölüm: irtifa sapması — yalnız keşif/teşhis hattı olarak.**

Gerekçe:

1. Trigger yükü küçük ve bütünü elle incelenebilir.
2. Kural cruise'a özgü, fiziksel olarak doğrudan açıklanabilir bir sinyal kullanıyor.
3. Mevcut hataların mekanizması dar ve gözlenebilir; rota kuralındaki bütün
   triggerları etkileyen düşük-hız artefaktından daha kolay ayrıştırılabilir.

Bu karar mevcut iki olayı anomaly ilan etmez. Sonraki irtifa turundan önce yeni
bir ön-kayıt gerekir. O sözleşmede yalnız iki tasarım konusu ele alınmalıdır:

- landing'e geçiş çevresini cruise referansından ayırmak;
- tek global cruise medyanı yerine birden fazla stabil flight-level segmentini
  normal kabul eden, yine yorumlanabilir bir referans tanımlamak.

Yeni sözleşme yazılmadan başka Silver parçalarında threshold/phase ayarı veya
sonuç-odaklı tarama yapılmaz.

## Raporlar

- `docs/ADSB_BASIT_IRTIFA_KESIF_RAPORU_20260722.md`
- `docs/ADSB_BASIT_ROTA_KESIF_RAPORU_20260722.md`
- Ham yerel sonuçlar: `artifacts/adsb/simple_anomaly_20260722/`

## Doğrulama

- Yeni basit-anomali + ADS-B feature suite: `37 passed in 0.55s`.
- Geniş ADS-B regresyon turu: `258 passed, 1 failed`.
- Tek failure, bu çalışmada değiştirilmeyen `adsb/cusum.py` ile immutable
  Step-5 CUSUM artefaktının tarihsel hash uyuşmazlığıdır. Frozen artefakt veya
  `archive/` değiştirilmedi.
- Ön-kayıt SHA-256 ve Silver SHA-256, sonuç summary'siyle yeniden doğrulandı.
- `operational_claim_allowed=false`, `ground_truth_available=false`.
