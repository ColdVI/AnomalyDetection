# ALFA — veri ve büyük artefakt konumu

Bu klasördeki kod/rapor/görsel fiziksel olarak taşındı; aşağıdaki büyük
veri/artefakt ağaçları `.gitignore` çakışmasını önlemek için **taşınmadı**,
orijinal konumlarında kalıyor:

- Ham/silver/gold veri: `data/objectstore/bronze/alfa/` (260M),
  `data/objectstore/silver/alfa/` (5M), `data/silver/alfa_silver.{csv,parquet}`,
  `data/silver/alfa_rosbag_silver.parquet`,
  `data/gold/ml_features/alfa/alfa_ml_features.parquet` (18.8M).
- Canlı ingest kodu: `gecmis_calismalar/residual_v1/ingest/alfa.py`,
  `alfa_channels.py` — `residual_v1` ALFA+RFLYMAD birleşik pipeline paketidir,
  bölünmeden tek parça olarak `gecmis_calismalar/residual_v1/` altına taşındı.
- Run/split/handout çıktıları: `artifacts/residual_v1/` (4.1G) içindeki ALFA'ya
  özel alt yollar (`splits/alfa_seed*.json`, `silver/alfa/`,
  `features/alfa/`, `phase_e_handout_20260717/handout/flights/alfa/` — burada
  onlarca `carbonZ_*` uçuşu için ayrı `REPORT.md` var).
- Konfig: `configs/residual_v1_*.json` (ALFA+RFLYMAD ortak).
- Birleşik tasarım/sonuç raporu: `_ortak/raporlar/RESIDUAL_V1.md`.
