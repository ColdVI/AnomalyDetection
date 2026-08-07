# Four-dataset probabilistic GPU v2 — balanced split preregistration

Date: 2026-07-27

This is a new contract. It does not rewrite or reinterpret any completed v1 run.

## Objective

Retain the normal-only causal Gaussian next-step forecaster while correcting the
source-level development imbalance. Splits are source/flight based; row-random
splitting is prohibited.

## Frozen split policy

- Seed: 20260727.
- Normal sources: 70% optimizer train, 15% threshold validation, 15% clean test.
- Train and validation contain only the dataset's declared normal label.
- The primary test contains every held-out clean source plus a deterministic,
  label-stratified anomaly sample.
- The primary anomaly sample targets the clean-test source count, but includes
  at least one source from every anomaly label when the corpus permits it.
- Remaining anomaly sources form stress_test. They are not used for scaling,
  optimization, threshold selection, primary model selection, or the primary
  v2 result.
- RflyMAD is additionally stratified by SIL/HIL/Real domain for the normal
  70/15/15 allocation. To preserve the verified transfer boundary, its anomaly
  pool is the already packaged development fold 1; all development NoFault
  sources remain eligible for the normal allocation. Other anomaly folds and
  the locked partition remain unopened by this contract.

## Frozen model and threshold

The v1 model family and optimization values remain unchanged: 32 history rows,
one-step prediction, one-layer LSTM with hidden size 64, batch 1024, 30 epochs,
learning rate 0.001, robust train-only scaling, and validation-only 0.995 alarm
quantile. The magnitude-domination gate remains Spearman rho 0.8.

## Interpretation

Primary ROC-AUC and average precision are reported on the balanced primary test.
Stress-test results, if later produced, must be labeled separately and cannot
replace the primary result. No post-result split, feature, epoch, threshold, or
class-ratio adjustment is permitted within v2.
