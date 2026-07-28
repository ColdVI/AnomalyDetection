# Four-dataset probabilistic GPU v2 — balanced Colab runbook

## What changed

Completed v1 artifacts remain unchanged. V2 uses a new deterministic source-level
contract:

- normal sources: 70% train, 15% validation, 15% clean test;
- optimizer and threshold validation remain normal-only;
- the primary test contains held-out normal sources and a label-stratified
  anomaly sample;
- remaining anomalies are isolated as stress_test and do not affect the primary
  result;
- RflyMAD normal allocation is also stratified across SIL, HIL, and Real.

## Files already on Drive that are reused

Do not upload these large dataset ZIPs again:

- four_dataset_probabilistic_gpu_v1_alfa_gold.zip
- four_dataset_probabilistic_gpu_v1_uav_attack_gold.zip
- four_dataset_probabilistic_gpu_v1_uav_sead_gold.zip
- four_dataset_probabilistic_gpu_v1_rflymad_development_folds_0_4.zip

## New small files to add to the same transfer directory

From artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer:

- four_dataset_probabilistic_gpu_v2_code_and_contract.zip
- transfer_index_v2.json
- bundle_manifest_v2.json

The new code ZIP is about 0.7 MiB. Dataset bytes are unchanged.

## Notebooks

- notebooks/alfa_probabilistic_gpu_v2_colab.ipynb
- notebooks/uav_attack_probabilistic_gpu_v2_colab.ipynb
- notebooks/uav_sead_probabilistic_gpu_v2_colab.ipynb
- notebooks/rflymad_probabilistic_gpu_v2_colab.ipynb

Run every notebook in a fresh v2 run directory. Do not point a v2 notebook at
an existing v1 checkpoint: the contract hash must differ and the runner will
reject it.

## Frozen primary role counts

| Dataset | Normal train | Normal validation | Primary test | Normal/anomaly in primary test |
|---|---:|---:|---:|---:|
| ALFA | 10 | 2 | 9 | 3 / 6 |
| UAV-Attack | 4 | 1 | 4 | 1 / 3 |
| UAV-SEAD | 629 | 135 | 268 | 134 / 134 |
| RflyMAD | 309 | 66 | 132 | 66 / 66 |

ALFA and UAV-Attack cannot achieve an exact 50/50 test while also retaining at
least one example of every anomaly family. This limitation must remain visible
in their reports.

RflyMAD reuses the v1 transfer ZIP boundary: all 441 development NoFault
sources and the 1,110 fold-1 sources are represented by the split contract.
The 1,044 fold-1 anomalies outside the balanced primary test remain stress-test
only. Other anomaly folds are not part of the transferred v2 contract.

## Result discipline

Report the magnitude gate, ROC-AUC, average precision, normal-flight alarm
fraction, anomalous-flight detection rate, and label-level results exactly as
produced. A failed v2 result does not authorize changes to the split, epochs,
features, scale bounds, or threshold quantile.
