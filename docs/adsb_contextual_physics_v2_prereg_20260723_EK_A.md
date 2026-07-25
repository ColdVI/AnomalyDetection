# ADS-B contextual_physics_v2 ?n-Kay?t ? Ek A

**Dondurma zaman?:** 2026-07-24, Faz C e?itimi ve herhangi bir v2 calibration
??kt?s? tamamlanmadan ?nce.

Bu ek, `docs/adsb_contextual_physics_v2_prereg_20260723.md` dosyas?n? de?i?tirmez.
Yaln?z o dosyada do?al calibration verisinden t?retilece?i belirtilip say?sal
prosed?r? a??k b?rak?lan `contextual_persistence_v2` parametrelerini dondurur.
Truth-v2, development, rehearsal ve holdout verileri bu se?imlere giremez.

## A.1 Veri ve conformal p-de?erleri

- Kaynak, Faz C ile ayn? deterministik `%2` `calibration_selected` u?u? k?mesidir.
- Fit u?u?lar?yla overlap s?f?r olmak zorundad?r.
- Model skoru `HierarchicalConformalCalibrator` ile
  `channel+phase+cadence -> channel+phase -> channel` fallback s?ras?yla p-de?erine
  ?evrilir; `min_group_size=1000` kullan?l?r.
- Yaln?z do?al calibration sat?rlar? kullan?l?r; sentetik sat?r say?s? s?f?rd?r.

## A.2 Sabit persistence parametreleri

- `max_gap_s = 30.0` ? v2 train config'de ?nceden yaz?l? de?er.
- `missing_reset_s = 60.0` ? de?i?tirilmemi? CUSUM hatt?ndaki do?al reset s?resi.
- `surprise_clip = 6.0` ? tek sat?r?n katk?s?n? `p=1e-6` s?rprizinde s?n?rlar.
- Alarm kar??la?t?rmas? `state > threshold_h` olarak kal?r.

Bu ?? de?er calibration veya truth-v2 sonucu g?r?ld?kten sonra de?i?tirilemez.

## A.3 `reference_shift_multiplier` t?retme kural?

Her persistence kanal?nda, ge?erli do?al conformal p-de?erleri i?in

`s = min(-log10(p), 6.0)`

hesaplan?r. Kanal ortalamalar? `mu_c`, teorik null ortalamas? ise
`mu_0 = 1 / ln(10)` olur. Tek ve ortak multiplier a?a??daki kapal? kuralla
hesaplan?r:

`m = ceil(100 * max(1.01, 1.10 * max_c(mu_c / mu_0))) / 100`

Yani do?al calibration'da en y?ksek ortalama s?rprize sahip kanal?n ?zerinde
%10 negatif-drift marj? b?rak?l?r ve de?er yukar? do?ru iki ondal??a yuvarlan?r.
Bo?, NaN veya sonsuz bir kanal ortalamas? se?im hatas?d?r ve Faz D'yi durdurur.
Candidate grid veya truth-recall aramas? yap?lmaz.

## A.4 B?t?e ba??na `threshold_h` t?retme kural?

Multiplier dondurulduktan sonra do?al calibration ak??? bir kez skorlan?r.
Her `(kanal, temporal profil, b?t?e V)` i?in hedef y?k:

`V / 100 * channel_budget_share * profile_budget_fraction`

alarm episode/saat olarak hesaplan?r. Persistence ve CUSUM `h` adaylar?, do?al
calibration skorlar?n?n a?a??daki ?nceden dondurulmu? quantile noktalar?d?r:

`[0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999,
0.99995, 0.99999, 0.999995, 0.999999, 0.9999995, 0.9999999]`

`threshold_h`, bu adaylar aras?ndan hedef y?ke mutlak olarak en yak?n y?k? veren
de?erdir. E?it uzakl?kta daha y?ksek `h` se?ilir. Instant conformal profiller i?in
aday alpha ?zgaras? v1 do?al-y?k hatt?yla ayn? sabit
`geomspace(1e-5, 0.5, 12)` ?zgaras?d?r; ayn? en-yak?n-y?k ve konservatif tie-break
kural? uygulan?r. Bunlar yaln?z do?al alarm y?k? e?lemeleridir; truth-v2 recall
bu se?imlere giremez.

CUSUM `h` de?erleri de ayn? b?t?e ?zgaras? i?in mevcut v1 do?al-calibration
prosed?r?yle t?retilir; `adsb/cusum.py` ve fiziksel CUSUM sabitleri de?i?tirilmez.

## A.5 Mekanik sonu? kayd?

Faz D tamamland???nda hesaplanan multiplier, kanal ortalamalar?, persistence ve
CUSUM b?t?e e?lemeleri calibration artifact'?na yaz?l?r. Bu dosyaya daha sonra
yaln?z A.3 form?l?n?n ?retti?i say?sal multiplier ve artifact SHA-256's? eklenebilir;
prosed?r veya sabitler de?i?tirilemez.
