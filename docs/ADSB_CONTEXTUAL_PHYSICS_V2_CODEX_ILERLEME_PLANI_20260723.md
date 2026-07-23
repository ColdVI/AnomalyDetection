# ADS-B `contextual_physics_v2` — Codex İlerleme Planı

> Tarih: 2026-07-23 (Europe/Istanbul)
> Onaylı plan: `.claude/plans/fluttering-waddling-frost.md` (Faz A-G) — bu doküman
> onun somut, adım adım yürütme talimatıdır.
> Ön-kayıt (değiştirilemez): `docs/adsb_contextual_physics_v2_prereg_20260723.md`

## 0. Neden bu iş var — 2 cümlede

`contextual_physics_v1` (16 Tem, ADR-042) mimari olarak doğru şeyi öğrendi ama
yanlış birimde tanımlanmış bir alarm bütçesi yüzünden gerçek olayların
>%94'ünü kaçırdı; bu iş AYNI mimariyi, ADR-042'nin bizzat istediği çok daha
geniş bir bütçe ızgarası ve çok daha büyük veriyle, sonuç görülmeden dondurulmuş
yeni bir ön-kayıtla yeniden açıyor. **Isolation Forest veya ham-reconstruction
AE/USAD'a DÖNÜLMÜYOR** — ikisi ayrı ayrı NO-GO oldu (bkz. ön-kayıt §0, ADR-032).

## 1. Şu ana kadar YAPILANLAR (tekrar üretme, aynen kullan)

| Ne | Dosya | Durum |
|---|---|---|
| Ön-kayıt (dondu) | `docs/adsb_contextual_physics_v2_prereg_20260723.md` | Tamam |
| Eğitim config'i | `configs/adsb_contextual_physics_v2_train.json` | Tamam (epoch=8, prob=1.0) |
| Bütçe ızgarası config'i | `configs/adsb_contextual_physics_v2_alarm_budget.json` | Tamam (0.1-500) |
| Yeni persistence modülü | `adsb/models/contextual_persistence_v2.py` | Tamam, **10/10 test geçti** |
| Persistence testleri | `tests/test_adsb_contextual_persistence_v2.py` | Tamam |
| Fit-expansion manifest üretici | `scripts/adsb_build_fit_expansion_manifest_v2.py` | Tamam, henüz ÇALIŞTIRILMADI |
| Eğitim script'i (v1'den uyarlanmış) | `scripts/adsb_train_contextual_physics_v2.py` | Tamam, syntax doğrulandı, henüz ÇALIŞTIRILMADI |
| 3 yeni tar → Silver | `data/objectstore/silver/adsblol_historical/` | **Arka planda çalışıyor**, tamamlanmamış olabilir — önce kontrol et (§2) |

3 yeni tar (`v2024.09.01`, `v2025.02.15`, `v2025.06.15-003`) taşındığı geçici
konum: `C:\Users\PC_5812_YD26\Downloads\_yeni_adsb_tarlar_staging\`. Komut:

```
python scripts/parallel_parse_all.py --tar-dir "C:\Users\PC_5812_YD26\Downloads\_yeni_adsb_tarlar_staging" --workers 3 --skip-clear --log-dir "logs/parallel_parse_20260723"
```

**`--skip-clear` OLMADAN asla çalıştırma** — mevcut 638 parçayı (3 gün, zaten
kullanılan calibration/development/rehearsal rolleri) siler.
`scripts/process_tars_sequential.py` ve `scripts/upload_bronze_all.py`
KULLANMA — ikisi de kaynak tar'ı koşulsuz siliyor.

## 2. İlk adım — veri hazır mı kontrol et

```
python -c "
import pandas as pd
from pathlib import Path
n = len(list(Path('data/objectstore/silver/adsblol_historical').glob('*.parquet')))
print('toplam parça:', n, '(638 eskiyse + yeni 3 tar'in parçaları beklenir, ~1140 civarı)')
"
```

Log'lardaki `Progress: N/TOTAL members` satırlarının üçü de `TOTAL/TOTAL`'a
ulaşmadıysa (`logs/parallel_parse_20260723/*.log`), önce bitmesini bekle. Süreç
kendi kendine biter, MinIO/Docker gerekmez (`.env`: `STORAGE_BACKEND=local`).

## 3. Faz A doğrulama + fit-expansion manifest

```
python scripts/adsb_build_fit_expansion_manifest_v2.py
```

Beklenen çıktı: 3 günün (`2024-09-01`, `2025-02-15`, `2025-06-15`) her biri için
parça sayısı. **Farklı bir gün seti çıkarsa DURDUR, ilerleme** — bu, script'in
kasıtlı fail-loudly davranışı (`FitExpansionManifestError`), sessizce farklı
veriyle eğitime geçmeyi engelliyor.

## 4. Faz C — Model eğitimi (ÇALIŞTIR, henüz çalıştırılmadı)

```
python scripts/adsb_train_contextual_physics_v2.py --run-dir "artifacts/adsb/runs/20260723_contextual_physics_v2_train_v1"
```

Script kendi sözleşme kontrollerini yapar (git temiz ağaç, kod-hash kilidi,
Step-5 manifest SHA-256, forbidden path). **Beklenen süre:** v1 (2.989 uçuş,
tek gün) birkaç dakika sürmüştü; bu turda veri ~50 kat büyük (149.462+3-gün
uçuş) — çok daha uzun sürebilir, arka planda çalıştır.

**Kritik kontrol — eğitim bitince İLK bakılacak şey:**
`artifacts/adsb/runs/20260723_contextual_physics_v2_train_v1/training_report.json`
içindeki `natural_calibration_diagnostic.magnitude_domination_flagged_at_0_8`
alanı **`false`** olmalı (v1'de `false`, rho=0.65 idi). `true` çıkarsa DUR,
kod/mimari değişmedi ama veri ölçeği bir şeyi bozmuş demektir — rapor et,
Faz D'ye geçme.

## 5. Faz D — Kalibrasyon (henüz yazılmadı, Codex yazacak)

`scripts/adsb_contextual_physics_v1_calibrate.py`'nin deseni yeni namespace'e
uyarlanacak (yeni dosya: `scripts/adsb_contextual_physics_v2_calibrate.py`):

- `adsb/conditional_calibration.py::HierarchicalConformalCalibrator` — DEĞİŞTİRME,
  aynen kullan. `channel+phase+cadence → channel+phase → channel` fallback.
- `adsb/cusum.py::VectorPageCUSUM` kalibrasyonu — DEĞİŞTİRME, aynen kullan
  (`east_north_cusum`, ADR-042'nin en iyi sonucu, referans/karşılaştırma için).
- **YENİ:** `adsb/models/contextual_persistence_v2.py::CumulativeConformalPersistence`
  kalibrasyonu — `PersistenceV2Config`'i `configs/adsb_contextual_physics_v2_alarm_budget.json`'daki
  `temporal_profiles.*.mode == "persistence_v2_cumulative"` kanallarına uygula.
  `reference_shift_multiplier` ön-kayıtta YOK, kalibrasyon-doğal veriden
  türetilecek şekilde tasarlandı ama **DEĞERİ sonuç görülmeden, sadece doğal
  kalibrasyon dağılımına bakılarak seçilmeli** (örn. doğal veride yanlış-alarm
  oranı hedeflenen bir üst sınırın altında kalacak en küçük multiplier) — bu
  seçim ayrı bir küçük ön-kayıt notu olarak kaydedilmeli
  (`docs/adsb_contextual_physics_v2_prereg_20260723.md`'ye EK olarak, o dosyanın
  kendisi değiştirilmeden — yeni bir "Ek A" bölümü).

Girdi kaynağı: `calibration_selected` flight'ları (mevcut Step-5 manifest'in
calibration rolü, %2 tanı-örneklemi — Faz C'nin `_natural_diagnostics`'inde
zaten kullanıldı, aynı flight seti).

## 6. Faz E — Değerlendirme (henüz yazılmadı, Codex yazacak)

`scripts/adsb_contextual_physics_v1_truth_v2_eval.py`'nin deseni yeni namespace,
yeni bütçe ızgarası (`configs/adsb_contextual_physics_v2_alarm_budget.json`,
0.1-500) ve persistence_v2 kanalıyla tekrarlanır (yeni dosya:
`scripts/adsb_contextual_physics_v2_truth_v2_eval.py`). Girdi: mevcut truth-v2
corpus'u (`data/objectstore/synthetic/adsb_v2_20260713_01`, 8.910 uçuş — YENİDEN
ÜRETME, zaten var). Çıktı: `docs/decisions.md`'ye ADR-042'nin devamı olarak yeni
bir madde (ADR numarası, dosyanın en sonundaki son ADR'den devam ettir).

**Ön-kayıt madde 6 (değişmezlik taahhüdü) burada da geçerli:** ızgara/persistence
parametreleri sonucu görüp DEĞİŞTİRİLMEZ. Sonuç ne çıkarsa çıksın olduğu gibi
raporlanır — 4 önceki ADS-B model denemesinin hepsi NO-GO oldu, bu turun da
NO-GO çıkması dürüst ve kabul edilebilir bir sonuçtur, "iyileştirmeye" çalışıp
disiplini bozma.

## 7. Faz F — Örnek anomali-etiketleme grafiği

`scripts/adsb_plot_injection_timelines.py`'nin deseni genişletilir (yeni dosya:
`scripts/adsb_plot_contextual_v2_timelines.py`): aynı uçuşun temiz/enjekte
versiyonlarında **model conformal-p-değeri + persistence_v2 kümülatif skoru +
CUSUM skoru** üst üste çizilir (3 çizgi, ortak zaman ekseni), enjeksiyon-onset
dikey kesikli çizgiyle işaretlenir. matplotlib/Agg, PNG — mevcut rapor
konvansiyonu (bkz. mevcut script, `adsb/synthetic.py::PHYSICS_BREAK_RECIPES`
üzerinden truth-v2 recipe'leri okunur). Çıktı:
`artifacts/adsb/plots/contextual_v2_timelines/`.

## 8. Faz G — Kural + model hibrit karşılaştırma

Model+CUSUM+persistence_v2 birleşik alarmları, `adsb/simple_anomaly.py`'nin
event şemasıyla (`event_id/flight_id/start_time/end_time/duration_s/...`)
aynı formatta bir tabloya dönüştürülür. Kural-turu sonuçlarıyla
(`docs/ADSB_BASIT_ANOMALI_KARSILASTIRMA_20260722.md`) yan yana yeni bir
karşılaştırma dokümanında raporlanır.

## Genel disiplin kuralları (her fazda geçerli)

1. `docs/adsb_contextual_physics_v2_prereg_20260723.md` ve iki config dosyası
   SONUÇ GÖRÜLMEDEN DONDULAR — hiçbir sayı/eşik/epoch geriye dönüp değiştirilmez.
   Değişiklik gerekiyorsa yeni tarihli, sonuç görülmeden yazılmış bir ön-kayıt.
2. `adsb/cusum.py`, `adsb/models/contextual_residual_forecaster.py`,
   `adsb/context.py`, `adsb/contextual_scaling.py`, `adsb/contextual_windowing.py`,
   `adsb/conditional_calibration.py` — DEĞİŞTİRME, aynen kullan (dondurulmuş,
   çalışan, v1'den kalma kod).
3. Sentetik veri (`adsb/synthetic.py`, truth-v2 corpus) SADECE değerlendirme
   için; eğitim/kalibrasyon verisine asla girmez (`data_role` tip kontrolleri
   kod seviyesinde zaten bunu zorluyor).
4. Her faz sonunda özet + doğrulama çıktısı raporlanır; commit/push kararı
   kullanıcıya bırakılır (otomatik commit YAPMA).
5. Beklenmeyen bir sonuç (örn. `magnitude_domination_flagged_at_0_8=true`,
   ya da fit-expansion manifest'in farklı bir gün seti bulması) DURDURMA
   sebebidir, "düzeltip devam et" değil — rapor et, kullanıcıya sor.
