# UAV-SEAD — veri ve büyük artefakt konumu

UAV-SEAD'ın da **hiç canlı kodu yok** — `kaynak_kod/` git geçmişinden restore
edilmiş ölü koddur. Büyük veri taşınmadı:

- Ham/silver/gold veri: `data/objectstore/bronze/uav_sead/` (11G, en büyük
  ham veri ayağı — `.ulg` PX4 logları),
  `data/silver/uav_sead_silver.{csv,parquet}` (parquet 480M),
  `data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet` (656M),
  `uav_sead_ml10_forecast_residual.parquet` (4.6M, ML-10 Chronos pilotu).
- **Dikkat:** `data/gold/ml6/sead_*.csv` (`sead_2x2_ablation.csv`,
  `sead_alarm_policy_by_class.csv`, `sead_alarm_policy_sweep.csv`,
  `sead_base_thresholds.csv`, `sead_event_metrics.csv`) + ilişkili
  `ml7_alarm_policy_*.csv`, `causal_cusum_single_features.csv`,
  `causal_modular_remeasure.csv` — bunlar SEAD-döneminden kalma ve **hiç
  arşivlenmemiş** legacy ML-6/7 çıktılarıdır, hâlâ `data/gold/ml6/` içinde
  canlı duruyor. Fiziksel taşınmadı (yine gitignore/`data/` kuralı gereği),
  ama gelecekte bu dataset üzerinde çalışan biri için not: bu dosyalar SEAD'e
  ait, `ml7_alarm_policy_*` adları SEAD-prefiksli değil ama aynı döneme aittir.
- `egitilmis_modeller/` altında: `ml8a/` (development eval, Gate A geçti/B-C
  kaldı), `ml9/`..`ml16/`, `ml16_kol_n/`, `ml_dense_ae_sead/`, `ml_lstm_sead/`,
  `ml_usad_sead/` (üçü de Gate B'de kaldı — bkz. proje hafızası "ML-16 Kol
  L/D/U"), `cusum_baseline.json`. `ml12/itki_komutu.joblib` en iyi kategori
  sonucu (0.459) olarak not edilmişti.
