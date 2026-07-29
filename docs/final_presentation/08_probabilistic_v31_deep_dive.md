# Probabilistic v3.1 derinlemesine inceleme

## Bilimsel soru

Normal uçuşlardan öğrenilen causal next-step olasılık modeli, görmediği scenario/domain gruplarında gerçek fault aralıklarını düşük normal alarm yüküyle yerelleştirebilir mi? v3.1 sözleşmesi RflyMAD’de bu soruyu train/validation/normal-test/anomaly-development/final-fault rolleriyle ayırır. Kaynak: `configs/four_dataset_probabilistic_v31_evaluation_contract.json`; `artifacts/four_dataset_probabilistic_v31/split_report.json`.

## Veri ve split

| Rol | Uçuş / grup | Modele görünürlük | Kaynak |
|---|---:|---|---|
| normal train | 260 / 7 | Ölçekleyici ve ağırlık öğrenimi | `artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json` |
| normal validation | 90 / 3 | Epoch/checkpoint ve policy bütçesi | `artifacts/four_dataset_probabilistic_v31/split_report.json`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json` |
| normal test | 91 / 3 | Yalnız final normal burden | aynı |
| anomaly development | 557 / 12 | v3.1 teşhis ve B0 gerçek olay değerlendirmesi | aynı |
| final fault test | 553 / 13 | Mühürlü, açılmadı | `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |

Train havuzu HIL=120, SIL=120, Real=20; 213.567 satır, 5,9252 saat ve 205.247 aday pencere içerir. Hard-check’ler rol/grup bütünlüğünü ve feature sözleşmesini geçti. Kaynak: `artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json`.

## Ön işleme ve pencereleme

- Yalnız train normal verisinden medyan ve `1,4826×MAD` hesaplanır; değerler `[-5,5]` aralığına clip edilir. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `configs/four_dataset_probabilistic_v31_rflymad.json`.
- Train MAD≤`1e-6` kanallar degenerate sayılıp çıkarılır; eksiklik maskeyle taşınır, future fill yapılmaz. Kaynak: aynı.
- Her örnek aynı uçuşta ve aynı kesintisiz segmentte 32 geçmiş adım kullanır; >5 s gap pencereyi böler. Girdi scaled değerler, observed mask ve cadence bilgisidir. Hedef bir sonraki çok-kanallı vektördür. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py` (`_make_windows`).
- 28 istenen özellikten 24’ü aktiftir. `battery_voltage`, `battery_current`, `gps_eph`, `gps_epv` train-degenerate/kontrat dışı bırakılmıştır. Kaynak: `configs/four_dataset_probabilistic_v31_rflymad.json`; `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`.

## Model

Bir katmanlı LSTM (`hidden_size=64`) son gizli durumu lineer head’e verir. Head her aktif kanal için iki sayı üretir: ortalama `μ` ve ham scale. Scale sigmoid ile `[0,05, 5,0]` aralığına sınırlandırılır. Eğitim normal-only masked Gaussian NLL ile Adam (`lr=0,001`, batch=1024, grad clip=1) kullanır. Maksimum 30 epoch, seed `20260727` ve group/source-balanced epoch quotas vardır. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py` (`GaussianForecaster`, training loop); `scripts/four_dataset_probabilistic_v31_runner.py`; `configs/four_dataset_probabilistic_v31_rflymad.json`.

## Gaussian NLL ve negatif değer

Kanal başına kullanılan kayıp:

`NLL = 0.5 * ((y - μ) / σ)^2 + log(σ)`

yalnız gözlenen hedef kanallarında ortalanır. Kod `0.5 log(2π)` sabitini eklemez. `σ<1` ise `log(σ)<0`; standardized hata küçükse toplam değer negatiftir. Dolayısıyla epoch-30 train NLL `-0,960454` ve validation NLL `-0,150592` matematiksel olarak mümkündür. Negatiflik, detection başarısı değil normal validation fit’idir. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py` (`_masked_gaussian_nll`); `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`; `validation_summary.json`.

## Checkpoint ve eşik seçimi

Her epoch sonunda normal validation ortalama NLL hesaplanır; en düşük validation NLL seçilir, eşitlikte erken epoch tercih edilir. v3.1 Rfly’da epoch 30 seçildi. Diagnostic pencere eşiği normal validation skorlarının `q=0,995` quantile’ıdır ve `10,0940005779` bulundu. Bu eşik B0 event policy seçiminin yerine geçmez; v3.1’in ilk any-window teşhisi içindir. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `configs/four_dataset_probabilistic_v31_rflymad.json`; `training_report.json`.

## Magnitude kontrolü

Validation’da en fazla 50.000 örnekte üç dizi karşılaştırılır: trained model NLL, aynı mimarinin random-init NLL’si ve standardized hedef RMS magnitude. Spearman korelasyonlarından herhangi biri `≥0,8` ise magnitude flag true olur. Rfly v3.1 sonuçları:

- trained–random `ρ=0,328813`
- trained–magnitude `ρ=0,349469`
- flag `false`

Bu sonuç, SEAD’deki `ρ≈0,964/0,965` kadar açık bir magnitude kestirmesi olmadığını gösterir. “Model iyi detector” sonucu çıkarmaz. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py` (`_magnitude_diagnostic`); `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.

## v3.1 any-window teşhisi

648 uçuş ve 611.204 pencere değerlendirildi. Window ROC-AUC `0,627143`, AP `0,940306`; flight ROC-AUC `0,641052`, AP `0,921051`. 91 normal uçuşun 77’sinde, 557 anomaly-development uçuşun 462’sinde en az bir pencere eşiği aştı: TP=462, FN=95, FP=77, TN=14; anomaly flight detection `%82,94`, normal flight alarm `%84,62`, balanced accuracy `%49,16`. Test havuzu anomaly-ağırlıklı olduğu için AP başlık metriği yapılmamalıdır. Bu matris event confusion matrix değildir. Kaynak: `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`; `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md`; `scripts/plot_four_dataset_probabilistic_v31_rflymad.py`.

## B0 gerçek event katmanı

B0, 681.690 pencere skoru / 738 uçuş üzerinde, validation’da policy seçip bağımsız normal test ve anomaly development’ta raporladı; 553 final fault uçuş kapalı kaldı. Persistence, approximate-time K-of-N ve standardized-NLL CUSUM ailelerinde 144 adaydan 54 ön-kayıtlı nokta, 48 mevcut nokta değerlendirildi. Kaynak: `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`; `scripts/evaluate_rflymad_probabilistic_v31_events.py`.

| Policy / nokta | Normal test false event/h | Event recall | Yorum | Kaynak |
|---|---:|---:|---|---|
| 2-of-3 approximate 10 Hz; budget 1/h; refractory 30 s | 0,654 | %43,27 | Birincil referans; NO-GO | `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |
| 0,5 s persistence | 4,577 | %42,19 | Benzer recall, çok daha yüksek burden | aynı |
| CUSUM allowance 0,25 | 5,884 | %48,29 | Küçük recall kazancı, büyük burden | aynı |
| Tarama içindeki max recall | 9,153 | %58,71 | Operasyonel nokta değil | aynı |

Birincil noktada family recall: Motor `%60,08`, Environment `%53,62`, Propeller `%39,53`, Voltage `%16,67`, Sensor `%5,45`; domain recall: HIL `%56,85`, SIL `%38,78`, Real `%7,81`. Range precision yüksek görünse de range recall yalnız yaklaşık `%0,65–1,14`; alarm truth süresinin çok küçük kısmını kapsıyor. Kaynak: `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.

## v2 ile v3.1 neden doğrudan kıyaslanamaz?

v2 source/in-domain split’te Rfly interval AUC `0,907`, flight mean AUC `0,915`, event recall `%56,1` ve `0,768` false event/h verdi. v3.1 scenario/domain group-safe ayrım kullanır; B0’nun değerlendirme havuzu ve policy sözleşmesi farklıdır. v2 daha iyimser fakat farklı soruyu, v3.1 daha zorlu grup genellemesini ölçer. Sayıları “model geriledi” diye doğrudan çıkarmak geçersizdir. Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json`; `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`.

## Nihai hüküm ve sıradaki meşru soru

Magnitude kapısının geçilmesi öğrenme hijyenini destekledi; ancak normal yük, Real/Sensor kör noktaları ve düşük range recall nedeniyle operating point **NO-GO**’dur. Mevcut belgede sıradaki ön-kayıtlı yön B1: causal phase/domain context ile kanal-bazlı NLL/scale teşhisidir. Bu görevde yeni deney yapılmamıştır. Kaynak: `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`; `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.
