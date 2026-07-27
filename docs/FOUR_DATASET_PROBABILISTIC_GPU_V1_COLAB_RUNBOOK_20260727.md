# Four-Dataset Probabilistic GPU v1 — Colab çalıştırma notu

Tarih: 2026-07-27  
Namespace: `four_dataset_probabilistic_gpu_v1`

## Drive'a yüklenecekler

Yerel klasör:

`artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer/`

Bu klasördeki şu yedi dosyanın tamamını Drive'da aşağıdaki klasöre yükleyin:

`MyDrive/bykr/four_dataset_probabilistic_v1_transfer/`

- `transfer_index.json`
- `bundle_manifest.json`
- `four_dataset_probabilistic_gpu_v1_code_and_contract.zip`
- `four_dataset_probabilistic_gpu_v1_alfa_gold.zip`
- `four_dataset_probabilistic_gpu_v1_uav_attack_gold.zip`
- `four_dataset_probabilistic_gpu_v1_uav_sead_gold.zip`
- `four_dataset_probabilistic_gpu_v1_rflymad_development_folds_0_4.zip`

ZIP'leri açmadan yükleyin. Notebook, gereken code ZIP'i ile kendi dataset ZIP'ini
`transfer_index.json` içinden seçer; byte ve SHA-256 doğrulaması geçmeden veriyi
açmaz.

## Notebook eşlemesi

| Dataset | Açılacak notebook | Drive run dizini |
|---|---|---|
| ALFA | `notebooks/alfa_probabilistic_gpu_v1_colab.ipynb` | `.../four_dataset_probabilistic_v1_runs/alfa_gpu_v1` |
| UAV-Attack | `notebooks/uav_attack_probabilistic_gpu_v1_colab.ipynb` | `.../four_dataset_probabilistic_v1_runs/uav_attack_gpu_v1` |
| UAV-SEAD | `notebooks/uav_sead_probabilistic_gpu_v1_colab.ipynb` | `.../four_dataset_probabilistic_v1_runs/uav_sead_gpu_v1` |
| RflyMAD | `notebooks/rflymad_probabilistic_gpu_v1_colab.ipynb` | `.../four_dataset_probabilistic_v1_runs/rflymad_gpu_v1` |

Colab'de GPU runtime seçip notebook'u yukarıdan aşağı çalıştırın. CUDA yoksa
notebook durur. L4 tercih edilir; başka CUDA GPU da sözleşmeyi bozmaz.

## Resume ve sonuç

- Her tamamlanan epoch `training_epoch_checkpoint.pt` dosyasına atomik yazılır.
- Aynı train hücresini ve aynı `RUN_DIR`'ı tekrar çalıştırmak tamamlanmış son
  epochtan devam eder.
- `training_progress.json` o anki kaynak/epoch ilerlemesini gösterir; yarım epoch
  checkpoint değildir. Yarım kalan epoch baştan alınır.
- Sabit 30 epoch tamamlanınca `training_report.json`, `validation_summary.json`,
  `flight_metrics.csv` ve `model_state.pt` yazılır.
- `magnitude_domination_flagged_at_0_8=true` çıkarsa sonuç başarısız/uyarı olarak
  olduğu gibi kaydedilir; aynı v1 içinde feature, epoch, clip veya threshold
  değiştirilmez.

Pencere alarmları debounced event değildir. Rapordaki alarm-window/saat değeri,
eski event-level FA/saat sayılarıyla doğrudan karşılaştırılmaz. ALFA ve
UAV-Attack development-only'dir; UAV-SEAD blind holdout'u ve RflyMAD locked-test'i
bu notebooklar tarafından açılmaz.

## Paketleri yeniden üretme

Kaynaklardan aynı transfer klasörünü yeniden kurmak için:

```powershell
.venv\Scripts\python.exe scripts\prepare_four_dataset_probabilistic_gpu_v1_colab_bundles.py --overwrite
```

Kod, config, ön-kayıt veya veri değişirse arşiv hashleri de değişir. Böyle bir
değişiklik yeni paket ve notebook'taki transfer-index hash kilidinin birlikte
yenilenmesini gerektirir.
