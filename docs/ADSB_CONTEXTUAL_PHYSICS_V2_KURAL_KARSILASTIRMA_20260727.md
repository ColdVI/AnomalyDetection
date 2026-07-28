# ADS-B contextual_physics_v2 — model/CUSUM/persistence_v2 ile basit kural turu karşılaştırması

> Tarih: 2026-07-27
> Faz G; eşik veya bütçe seçimi değildir.

## Sonuçların doğru okuma birimi

Bu iki çalışma aynı veri ve ground-truth rejiminde değildir. Basit kural turu 100 doğal uçuşluk keşif örneğidir ve ground truth içermez; contextual-v2 ise mevcut 8.910-uçuş truth-v2 corpusunda enjekte olay recall'ü ile, eşlenik temiz uçuşlarda doğal alarm yükünü ölçer. Bu nedenle aşağıdaki tablolar yan yana bağlam sağlar; doğrudan bir kazanan sıralaması değildir.

## Ortak event şeması

Model, persistence_v2 ve CUSUM alarm emisyonları 60 saniyelik episode birleştirmesinden sonra `adsb.contextual_events.contextual_alarm_events` ile aşağıdaki ortak prefix'e dönüştürülür:

`event_id, flight_id, start_time, end_time, duration_s, n_samples`

Bu prefix `adsb.simple_anomaly` içindeki hem irtifa hem rota event tablolarıyla birebir aynıdır. Detector/channel/profile/budget/threshold/recipe alanları ortak prefix'in arkasına eklenir; temel alanların anlamı değiştirilmez.

## Basit kural turu — doğal keşif örneği

| Kural | Değerlendirilebilir uçuş | Triggerlı uçuş | Event | Doğrulanmış anomaly | Ana bağlam |
|---|---:|---:|---:|---:|---|
| İrtifa sapması | 57 | 2 (3.51%) | 2 | 0 | Faz sınırı / meşru seviye değişimi |
| Rota sapması | 95 | 13 (13.68%) | 24 | 0 | Düşük-hız bearing kararsızlığı |

## contextual_physics_v2 — dondurulmuş bütçe noktaları

| Recipe | Detector / kanal-profili | V=0.1 recall / temiz yük | V=5 recall / temiz yük | V=50 recall / temiz yük | V=500 recall / temiz yük |
|---|---|---:|---:|---:|---:|
| vertical_rate_frozen | model / vertical_rate_residual/spike | 0.00% / 0.000 ep/saat | 0.00% / 0.000 ep/saat | 3.89% / 0.018 ep/saat | 12.02% / 0.219 ep/saat |
| vertical_rate_frozen | persistence_v2 / vertical_rate_residual/freeze | 26.26% / 0.004 ep/saat | 26.34% / 0.004 ep/saat | 40.81% / 0.022 ep/saat | 58.48% / 0.181 ep/saat |
| ground_speed_biased | model / speed_residual/spike | 0.00% / 0.001 ep/saat | 12.25% / 0.026 ep/saat | 17.41% / 0.069 ep/saat | 49.07% / 0.656 ep/saat |
| ground_speed_biased | persistence_v2 / speed_residual/bias | 57.18% / 0.001 ep/saat | 57.18% / 0.001 ep/saat | 79.66% / 0.054 ep/saat | 89.98% / 0.526 ep/saat |
| track_frozen | persistence_v2 / heading_residual/default | 11.58% / 0.000 ep/saat | 14.82% / 0.000 ep/saat | 62.46% / 0.098 ep/saat | 77.61% / 0.682 ep/saat |
| position_ramp_stealthy | CUSUM / east_north_velocity_residual/accumulation | 9.35% / 0.000 ep/saat | 51.27% / 0.017 ep/saat | 65.73% / 0.082 ep/saat | 85.61% / 0.636 ep/saat |

## Karşılaştırmalı yorum

- Kural turundaki event sayıları anomaly recall değildir; ground truth yoktur ve elle doğrulanmış anomaly sayısı sıfırdır.
- contextual-v2 satırlarında recall yalnız observable-eligible enjekte olaylar üzerindedir; temiz yük aynı uçuşların enjeksiyonsuz eşlerinde episode/saat birimindedir.
- V=0.1, 5, 50 ve 500 noktaları sonuç görüldükten sonra seçilmedi; önceden dondurulmuş 11-noktalı ızgaranın sabit temsilcileridir. Ara noktalar makine okunur evaluation artifact'ında aynen korunur.
- Bu rapor threshold, epoch, grid, channel share veya persistence parametresi değiştirmez ve operasyonel başarı iddiası üretmez.

## Kaynaklar

- `docs/ADSB_BASIT_ANOMALI_KARSILASTIRMA_20260722.md`
- `artifacts/adsb/simple_anomaly_20260722/summary.json`
- Faz E `truth_v2_eval_report.json`
- `adsb/contextual_events.py`
