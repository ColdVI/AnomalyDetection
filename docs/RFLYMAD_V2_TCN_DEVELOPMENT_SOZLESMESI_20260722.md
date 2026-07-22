# RflyMAD-Full v2 — TCN development-only uzun koşu sözleşmesi

> Tarih: 2026-07-22 (Europe/Istanbul)  
> Durum: **SONUÇLARDAN ÖNCE DONDURULDU**  
> Dayanak: `RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md` Bölüm 12/9 ve kullanıcının
> otonom devam onayı.  
> Locked test: **okunmayacak**  
> Operasyonel/fizibilite iddiası: **yasak**

## 1. Amaç

Truth schema v2 ile tamamlanan 3-epoch TCN sanity koşusundan sonra supervised TCN'in
development verisinde çok-epoch davranışını, fold kararlılığını, Real transferini
ve Wind yanlış-alarm yükünü ölçmek. Bu deney AE robustness sözleşmesindeki altı aday
hakkını yeniden açmaz ve yeni bir AE threshold/LR/epoch araması değildir.

## 2. Veri ve split sözleşmesi

- Yalnız `split=development` feature dosyaları okunur.
- Beş outer fold'un her biri tam bir kez değerlendirme fold'u olur.
- Kalibrasyon/validation fold'u outer fold'un bir önceki mod-5 fold'udur:

| Outer değerlendirme | Validation/kalibrasyon | Eğitim fold'ları |
|---:|---:|---|
| 0 | 4 | 1, 2, 3 |
| 1 | 0 | 2, 3, 4 |
| 2 | 1 | 0, 3, 4 |
| 3 | 2 | 0, 1, 4 |
| 4 | 3 | 0, 1, 2 |

- Scaler yalnız eğitimdeki `normal_reference` uçuşlarından öğrenilir.
- Temperature ve critical/advisory eşikleri yalnız validation fold'unda seçilir.
- Outer fold metrikleri model, epoch, temperature veya threshold seçiminde kullanılmaz.
- Her koşuda `development_smoke_fold=<outer>` zorunludur. Alan adı tarihsel olsa da
  12+ epoch koşuların statüsü `development_only` olacaktır.
- Her summary için `locked_test_features_read=false` ve
  `operational_claim_allowed=false` zorunludur.

## 3. Eğitim ve epoch sözleşmesi

- Model: mevcut `supervised_tcn_multitask`; mimari değiştirilmez.
- Train cap: 50.000 pencere; validation-loss cap: 20.000 pencere.
- Başlangıç epoch tavanı: 12.
- Checkpoint: aynı koşudaki en düşük validation-loss epoch'u.
- Tavan uzatma kararı yalnız validation loss ile ve otomatik verilir:
  - en iyi epoch son iki epoch içindeyse ve önceki en iyi loss'a göre iyileşme
    en az `1e-4` ise `12 → 25 → 50` uzatılır;
  - aksi halde o fold durur;
  - 50 kesin tavandır.
- Outer recall/FA sonuçları uzatma kararına giremez.
- Bir uzatma gerektiğinde koşu aynı seed ile baştan deterministik tekrar edilir.

## 4. Dondurulmuş değerlendirme kapıları

### 4.1 Critical development kapısı

Hepsi gerekli:

- ortalama event recall `>= %50`;
- minimum fold recall `>= %40`;
- ortalama tüm-nonfault FA `<= 2/saat`;
- maksimum fold FA `<= 4/saat`.

### 4.2 Advisory development kapısı

Hepsi gerekli:

- ortalama event recall `>= %60`;
- minimum fold recall `>= %50`;
- ortalama tüm-nonfault FA `<= 12/saat`;
- maksimum fold FA `<= 15/saat`.

### 4.3 Real research kapısı — critical

AE robustness sözleşmesiyle uyumlu olarak hepsi gerekli:

- Real macro recall ortalaması `>= %40`;
- Real Motor ve Real Sensor ortalamalarının her biri `>= %30`;
- minimum-fold Real macro recall `>= %25`;
- Real-NoFault FA ortalaması `<= 4/saat`, maksimumu `<= 8/saat`.

### 4.4 Wind ara kapısı — critical

Hepsi gerekli:

- Wind FA ortalaması `<= 15/saat`, maksimumu `<= 20/saat`;
- genel recall ortalaması `>= %50`;
- tüm-nonfault FA ortalaması `<= 2/saat`.

## 5. Yorum ve durdurma kuralları

- Beş fold tamamlanmadan başarı/başarısızlık kararı verilmez.
- TCN yalnız critical veya advisory development kapısından en az birini geçerse
  “ikinci ana deneysel aday” olarak anılabilir.
- Real ve Wind kapıları ayrı raporlanır; genel kapının geçmesi onları çözülmüş yapmaz.
- Hiçbir development sonucu locked test'i açma, fizibilite veya operasyonel başarı
  ilanı değildir.
- Tavan 50'de validation loss hâlâ sınırda iyileşiyorsa sonuç “epoch-limited” diye
  işaretlenir; sonuç görüldükten sonra tavan artırılmaz, yeni sözleşme gerekir.

## 6. Çalıştırma ve artefakt

```powershell
.venv\Scripts\python.exe `
  scripts\run_rfly_full_v2_supervised_development_sweep.py
```

Deney kökü:
`artifacts/rfly_full/v2/supervised_tcn/development_5fold_20260722_v1/`

Zorunlu nihai dosyalar: `sweep_state.json`, `summary.json`,
`outer_fold_metrics.csv`, `aggregate_metrics.csv`, `training_history.csv` ve
`gate_summary.json`.
