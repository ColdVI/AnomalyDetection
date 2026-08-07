# RFLYMAD — veri ve büyük artefakt konumu

RFLYMAD'ın **3 paralel canlı pipeline'ı** var, aynı ham veriyi okuyor ama
birbirinin yerine geçmezler:

1. **`gecmis_calismalar/residual_v1/`** — `ingest/rfly.py`, `rfly_channels.py`
   (ALFA ile paylaşılan komut→tepki residual FDI pipeline'ının RFLYMAD yarısı;
   paket bölünmeden tek parça taşındı, bkz. `ALFA/VERI_ARTEFAKT_KONUMU.md`).
   Artefaktlar: `artifacts/residual_v1/splits/rfly_seed*.json` + `rfly_hashes.json`,
   `artifacts/residual_v1/silver/rfly/`, `features_k1k4/rfly/`.
2. **`gecmis_calismalar/uav_gnss/`** — GNSS bütünlük pilotu (PX4/RflyMAD).
   Artefaktlar: `artifacts/uav_gnss_integrity_v1/` (1.1M — `lstm_state.pt`
   checkpoint, calibration/development/rehearsal/wind-stress JSON'ları, 2
   `.tex` raporu). Sonuç: **NO-GO** (dev+rehearsal kapılarını geçemedi).
3. **`gecmis_calismalar/rfly_full/` + `gecmis_calismalar/rfly_dl/`** —
   bugünkü (2026-07-22) büyük çalışma turu dahil, en güncel ve en kapsamlı
   hat. Artefaktlar: `artifacts/rfly_full/` (1.7G —
   `v2/dataset_manifest.parquet`, `v2/normal_temporal_ae/`,
   `v2/supervised_tcn/`, `v2/truth_audit/`, `v2/normal_temporal_ae/robustness/`)
   ve `artifacts/rfly_dl/` (7.3M, erken direct-DL pilotu).

Legacy RFLY-0/1 (`legacy_rfly0_1/`) ölü koddur (git geçmişinden restore
edildi); tek fiziksel artefaktı `artifacts/rfly0/rflymad/smoke/` idi, o da
bu klasöre taşındı (`legacy_rfly0_1/artifacts/rfly0/`).

Büyük veri (`.gitignore` çakışmasını önlemek için taşınmadı):

- `data/objectstore/bronze/rflymad/` = **19G** (en büyük tek ayak — alt
  klasörler `HIL-Wind` 4.9G, `Real-Motor` 5.7G, `Real-No_Fault` 1.0G,
  `Real-Sensors` 4.2G, `SIL-Wind` 3.4G).
- `data/silver/rflymad_silver.parquet` (91M),
  `data/gold/ml_features/rflymad/rflymad_ml_features.parquet` (133M).

`raporlar/` altında: 12 `RFLYMAD_V2_*.md` doküman + `Codex_RFLYMAD_conversation.md`
(bugünkü Codex sohbet kaydı) + `RFLY_DL_DOGRUDAN_DEGERLENDIRME_PLANI.md` +
TCN/convergence görselleri (`assets/`) + çalıştırılmış özet notebook
(`notebooks/RFLYMAD_V2_TUM_DENEYLER_CALISTIRILMIS_20260722.ipynb` + besleyen
`notebooks/data/rflymad_v2/*.csv,json`).

**Güncel durum özeti (2026-07-22 itibarıyla):** parser truth-domain hatası
düzeltildi, 6-adaylı Wind/Real robustness sweep (R1-R4, W1, W2) preregistered
kapıların hiçbirini geçmedi, TCN development sweep de aynı kapıları geçmedi —
Real transfer gösterilemedi, Wind robustness çözülmedi. Ayrıntı:
`raporlar/RFLYMAD_V2_SONRAKI_ADIMLAR_20260722.md`.
