# UAV-Attack — veri ve büyük artefakt konumu

UAV-Attack'ın **hiç canlı kodu yok** — `kaynak_kod/` altındaki dosyalar
git geçmişinden (`53b9f1f~1`) restore edilmiş, artık import edilmeyen ölü
koddur. Büyük veri `.gitignore` çakışmasını önlemek için taşınmadı:

- Ham/silver/gold veri: `data/objectstore/bronze/uav_attack/` (684M),
  `data/objectstore/silver/uav_attack/` (6.2M),
  `data/silver/uav_attack_silver.{csv,parquet}`,
  `data/gold/ml_features/uav_attack/uav_attack_ml_features.parquet` (14.1M).
- Model artefaktları: eski ML-8A..16 bucket'ları içinde UAV-Attack'a özel bir
  ayrım bulunamadı (bkz. `gecmis_calismalar/UAV_SEAD/egitilmis_modeller/` —
  çoğu ML-9..16 artefaktı isimlendirmeye göre SEAD-ağırlıklı); UAV-Attack
  kendi başına bir model/checkpoint bucket'ına sahip değildi.
