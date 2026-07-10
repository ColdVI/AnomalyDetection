# ML-12: İnce-Modül (Thin-Module) Hipotezi Planı

Durum: UYGULANDI (2026-07-07; plan sonuç görülmeden yazılıp sabitlendi, sonra koşuldu).
Sonuç: Gate A GEÇTİ, Gate B GEÇTİ (B1+B2 — `itki_komutu` 0.459, hem `motor_simetrisi` 0.205'i
hem `chronos_motor` 0.390'ı geçti), Gate C KALDI (füzyon 0.217/23.74 FA-saat). Ayrıntı:
`docs/ML1_BULGULAR_VE_HATALAR.md` "ML-12 sonuçları" (H29-H30), ADR-012, C.7.
Artifact: `artifacts/ml12/uav_sead/full_matrix/`.
Kaynak hipotez: ML-11 görselleştirme fazının feature×kategori AUC matrisi
(`artifacts/viz/uav_sead/s3_features/feature_auc_matrix.csv`, H26).

## §0 Hipotez ve gerekçe

ML-11 bulgusu: `actuator_thrust_cmd` tek başına Actuator Outputs+Controls
satırlarını normalden **AUC 0.983** ile ayırıyor; ama bu feature 16 feature'lık
`kontrol_cevabi` modülünün içinde ve 6-modüllü max-füzyonda **seyreliyor**
(füzyonun kategori onset-recall'u yalnız 0.205-0.390 bandında). Hipotez:

> Güçlü tekil sinyali kendi **ince modülüne** (1-3 feature'lık IF) ayırıp aynı
> kalibrasyon/karar hattından geçirmek, kategori recall'unu geniş modüldeki
> seyrelmiş halinden anlamlı biçimde yukarı taşır.

Bu, ML-11'deki "füzyon/model sorunu vs veri/kapsam sorunu" ayrımının
**model-sorunu kolunu** test eden kontrollü deneydir. Position.Z bilerek kapsam
DIŞI: ML-11 oradaki güçlü ayrıştırıcıların baro-tabanlı (n=33 satır, %7
doluluk) olduğunu, yani sorunun veri/kapsam olduğunu gösterdi — ince modül onu
çözmez.

## §1 Aday modüller (SABİT — sonuç görüldükten sonra değiştirilemez)

`src/ml/models/modular_iforest.py::PX4_ML12_THIN_MODULES`:

| Modül | Feature listesi | Gerekçe (ML-11 AUC) |
|---|---|---|
| `itki_komutu` | `actuator_thrust_cmd` | 0.983 — saf tekil-sinyal testi |
| `itki_kontrol_ince` | `actuator_thrust_cmd`, `attitude_error_mag`, `control_strain` | 0.983/0.783/0.748 — üç farklı fizik, hâlâ ince |

İki aday da ÖNCEDEN kayıtlıdır ve ikisi de raporlanır; sonuç görüp aralarından
seçme ("garden of forking paths") yok. Hiperparametreler ML-9 ile birebir aynı
ve sabit: `IsolationForest(n_estimators=300, max_samples=256, random_state=seed)`;
eğitim yalnız split train-normallerinde; kalibrasyon `empirical_probability` ile
yalnız split val-normallerinde (ortak `src/ml/evaluation/score_fusion.py`).
Scaler YENİDEN FİT EDİLMEZ: her split'in donmuş ML-9 scaler'ı
(`artifacts/ml9/uav_sead/full_matrix/split_XX/scaler.json`, checksum'lu) kullanılır.

## §2 Değerlendirme protokolü

`scripts/run_ml12_thin_module_evaluation.py`, ML-10 runner'ının kalıbıyla:

1. Donmuş ML-9 split modelleri + scaler checksum doğrulamasıyla yüklenir;
   `existing_fusion` satır akışları aynen yeniden üretilir (ML-10'daki gibi).
2. İnce modüller her split'in train-normalinde fit edilir, tüm development
   satırları skorlanır, val-normal CDF'ine kalibre edilir.
3. Füzyon adayları (her ikisi de önceden kayıtlı):
   - `ml12_fusion_itki   = max(existing_fusion, itki_komutu)`
   - `ml12_fusion_ince   = max(existing_fusion, itki_kontrol_ince)`
4. Karar katmanları DEĞİŞMEDEN `src/ml/decision/decision_layers.py`'den:
   threshold / K-of-N / bootstrap-CUSUM × {critical: 2 FA-saat, advisory: 12
   FA-saat} bütçeleri, 1 s stride, val-normal akışlarında fit.
5. Baseline satırları YENİDEN HESAPLANMAZ; checksum'u doğrulanmış ML-9
   (`motor_simetrisi`, `existing_fusion`, `ml9_fusion`) ve ML-10
   (`chronos_motor`, `ml10_fusion`) CSV satırları aynen alınır (ML-10'un
   ML-9'u aldığı kalıp).

## §3 Gate tanımları (SABİT)

- **Gate A (zorunlu):** 131-uçuş blind holdout hiçbir tabloda okunmadı
  (assert); split/feature/silver ve ML-9/ML-10 manifest checksum'ları eşleşti;
  karar katmanları ve score-fusion yardımcıları değişmeden import edildi
  (identity test); ince modül feature listeleri bu plandakiyle birebir aynı
  (test assert eder). Geçmezse dur.
- **Gate B (kategori, `Actuator Outputs+Controls`):** ML-9/ML-10 ile AYNI kural —
  eşleşen policy+bütçede ortalama onset-recall kazancı **≥0.05** VE **≥3/5
  seed** pozitif.
  - **B1 (gate'i belirleyen):** aday (`itki_komutu` veya `itki_kontrol_ince`)
    vs `motor_simetrisi` (aynı model ailesi, production-uyumlu baseline).
    Herhangi bir aday B1'i geçerse Gate B GEÇTİ sayılır; iki aday da tüm
    karşılaştırma satırlarıyla raporlanır.
  - **B2 (bilgilendirici, önceden kayıtlı):** aynı kural aday vs
    `chronos_motor` (bilinen en iyi kategori skoru, 0.390). "Bilinen en iyiyi
    geçti" iddiası ancak B2 ile kurulabilir; B2 gate durumunu belirlemez
    (chronos production füzyonunda değil ve ayrı bir bağımlılık hattı).
- **Gate C (operasyonel, değişmedi):** herhangi bir `ml12_fusion_*` satırı
  critical'da ≥0.30 recall @ ≤2 FA-saat VEYA advisory'de ≥0.50 @ ≤12 FA-saat
  sağlarsa geçer. Geçmezse holdout AÇILMAZ, aday production füzyona ALINMAZ.

Sonuç görüldükten sonra modül listesi, hiperparametre, füzyon tanımı, policy
grid'i veya bütçe DEĞİŞTİRİLMEZ; değişiklik yeni ön-kayıtlı faz gerektirir.

## §4 Dosyalar ve doğrulama

| Dosya | İş |
|---|---|
| `src/ml/models/modular_iforest.py` | `PX4_ML12_THIN_MODULES` sözlüğü (yalnız ekleme; mevcut modüller değişmez) |
| `scripts/run_ml12_thin_module_evaluation.py` (yeni) | §2 protokolü; `--splits split_00` smoke desteği |
| `tests/test_ml12.py` (yeni) | modül listeleri plandakiyle özdeş; karar-katmanı/score-fusion identity; artifact manifest'inde holdout izolasyonu + development-id hash |
| `artifacts/ml12/uav_sead/<run>/` | metrics/flight_label/category CSV + gates.json + checksum'lu manifest |

Kabul: smoke (split_00) sonra tam 5-seed koşu; tam `pytest` yeşil (bilinen 4
MinIO hariç); bulgular `docs/ML1_BULGULAR_VE_HATALAR.md`'ye, karar ADR-012'ye.
