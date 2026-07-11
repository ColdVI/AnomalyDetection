# ML-9 Planı — Kategori-Eşleşmeli Residual'lar (SEAD altitude + mechanical)

> Bu doküman bir coding-agent (Codex / Claude Code) talimatıdır. Repo: `ColdVI/AnomalyDetection`.
> `docs/AGENTS.md`, `docs/decisions.md` ve `docs/ML8_PLAN.md` kuralları geçerlidir (özellikle
> Gate disiplini, checksum'lu artifact deseni, blind holdout'a hiç dokunmama). Yeni modül yolları
> mevcut yapıyla çelişirse mevcut yapıyı esas al.

## 0. Bağlam ve deney sorusu

ML-8A'nın Gate B sonucu (`docs/ML1_BULGULAR_VE_HATALAR.md` H15): class-balanced LightGBM,
mevcut yarı-denetimli IF-füzyon/LSTM-AE'yi **geçemedi** (SEAD AUPRC 0.349 < 0.385). Literatür
bunu doğruluyor — az etiketli veride yarı-denetimli, tam-denetimliyi geçer. **Bu yüzden ML-9
YENİ BİR MODEL AİLESİ denemiyor; mevcut kazanan yaklaşımı (modüler Isolation Forest) daha
isabetli feature'larla besliyor.**

SEAD'in `altitude_anomaly` ve `mechanical_fault` sınıfları zayıf kaldı (onset recall sırasıyla
~0.03-0.04 ve ~0.04-0.08). SEAD havuzu artık **fiilen tükendi** (altitude 73/73, mechanical
41/41 — hepsi indirildi, `docs/ML1_BULGULAR_VE_HATALAR.md` "SEAD tamamlama" notu). Yani bundan
sonraki kaldıraç veri değil, **feature isabeti**.

Bugün `data/objectstore/bronze/uav_sead/labels.json`'daki `ranges` alanının annotasyon
KATEGORİ adını (`Position.Z`, `Actuator Outputs` vb.) doğrudan kontrol ettim — bu şimdiye
kadar hiç kullanılmamış bir bilgi:

```
altitude_anomaly (73 uçuş)   : Position.Z 72, Actuator Outputs 1, Actuator Thrust 1
                                -> NEREDEYSE TAMAMEN tek bir kanal (dikey pozisyon)
mechanical_fault (41 uçuş)    : Actuator Outputs 27, Actuator Controls 9, Magnetometer 3,
                                Battery 2, Vibration 1, Raw Accel 1, Actuator Thrust 1
                                -> HETEROJEN -- 4 farklı fiziksel imzanın ortak etiketi;
                                   Actuator Outputs+Controls tek başına %88'i kapsıyor
```

**Deney sorusu:** `altitude_anomaly` için mevcut havuzlanmış (yatay+dikey karışık) EKF
innovation yerine **yalnız dikey bileşeni** ayırmak, `mechanical_fault` için **motor-çıkışı
simetri residual'i** eklemek — bu iki kategori-eşleşmeli feature, aynı IF-füzyon mimarisinde
mevcut modüllerden daha iyi ayrışma sağlıyor mu? Ayrışma hem eski sınıf-düzeyi metriklerle
hem de YENİ annotasyon-kategorisi-düzeyi metriklerle ölçülür (ikincisi daha keskin bir test).

## 1. Kapsam / kapsam dışı

**Kapsamda:** UAV-SEAD (412 uçuş, mevcut Silver/feature). İki yeni feature ailesi (altitude
EKF dikey ayrıştırma, mechanical motor-simetri residual'i). Bunları mevcut modüler IF
mimarisine yeni modül olarak ekleme. Mevcut ML-8A karar katmanlarını (threshold/K-of-N/CUSUM)
DEĞİŞTİRMEDEN yeniden kullanma. Annotasyon-kategorisi-düzeyi yeni bir değerlendirme katmanı.

**Kapsam DIŞI (bu fazda yapma):** Yeni model mimarisi (LightGBM/forecast-residual/Chronos —
bunlar ML-10 adayı), ALFA (bu fazın hedefi SEAD), UAV Attack, blind holdout'u açmak,
Battery/Vibration/Magnetometer alt-tipleri için özel feature (her biri mechanical_fault'un
sadece 1-3 uçuşu — n çok küçük, kapsam beyanı maddesi olarak kalır, MODELLENMEZ).

## 2. Doğrulanmış ön-koşullar (bugün kontrol edildi, varsayım değil)

- `vel_pos_innov[2]` (dikey hız innovation) ve `vel_pos_innov[5]` (dikey pozisyon/irtifa
  innovation) zaten `ekf2_innovations`'tan parse ediliyor ama
  [parse_uav_sead.py:146](../src/silver/parse_uav_sead.py#L146) satırında yalnız pooled
  `ekf_vel_innov_mag`/`ekf_pos_innov_mag` hesaplandıktan sonra **drop ediliyor**. Yeni
  feature YENİ BİR TOPIC GEREKTİRMİYOR, sadece bu iki sütunu drop etmeden önce ayrıca
  saklamak yeterli.
- `actuator_outputs` topic'i gerçek ULog dosyalarında kontrol edildi: 15 mechanical_fault +
  15 normal uçuşluk rastgele örneklemde **15/15 ve 15/15** mevcut (sensor_baro/
  vehicle_local_position gibi seyrek DEĞİL). Alanlar: `timestamp`, `noutputs`,
  `output[0..15]` (PWM, tipik aralık ~900-2000). `noutputs` uçuşa göre değişebilir (bir
  örnekte 8) — Codex aktif/anlamlı çıkışları **ampirik olarak** belirlemeli (ör. flight
  boyunca varyansı sıfıra yakın olmayan kanallar), `noutputs` değerine körü körüne güvenme.

## 3. Feature spesifikasyonu

### 3.1 `ekf_alt_innov` / `ekf_vertical_vel_innov` — SEAD Silver parser

[src/silver/parse_uav_sead.py](../src/silver/parse_uav_sead.py) `parse_ulg_bytes` içinde,
mevcut pooled hesaplamanın YANINA (yerine değil — geriye uyumluluk, mevcut artifact'ler
`ekf_vel_innov_mag`/`ekf_pos_innov_mag`'a bağımlı):

```python
base["ekf_vertical_vel_innov"] = base["vel_pos_innov[2]"].abs()   # dikey hiz (D bileseni)
base["ekf_alt_innov"] = base["vel_pos_innov[5]"].abs()             # dikey pozisyon/irtifa
```

Bu iki sütun drop edilmeden ÖNCE hesaplanmalı. `src/ml/features/uav_attack_features.py`'nin
`_EKF_COLS` listesine eklenir (yalnız `if c in g.columns` koşuluyla — ALFA/UAV Attack'ta bu
sütunlar yok, otomatik elenir, mevcut desen zaten bunu yapıyor).

### 3.2 `actuator_output_imbalance` — motor simetri residual'i

Yeni `_topic_df(ulog, "actuator_outputs", [f"output[{i}]" for i in range(16)] + ["noutputs"])`
parse'ı eklenir. Feature builder'da (yeni fonksiyon veya `uav_attack_features.py` içinde
SEAD-özel blok — ALFA/UAV Attack'ta bu topic yok):

```python
active = [c for c in output_cols if g[c].std() > <esik>]   # ampirik aktif-kanal tespiti
actuator_output_std = g[active].std(axis=1)      # anlik motor-arasi dagilim
actuator_output_range = g[active].max(axis=1) - g[active].min(axis=1)
```

Beklenti: sağlıklı uçuşta simetrik yük dağılımı (küçük std/range); motor/ESC arızasında
kalıcı asimetri. Bu ANALİTİK REDUNDANCY prensibiyle aynı aile (gps_speed_residual,
turn_residual gibi) — "sağlıklı sistemde olması gereken bir simetri/ilişki, arızada bozulur".

### 3.3 Rolling/CUSUM

Mevcut desenle aynı: `rolling_stats(..., WIN_5S, stats=("max","mean"))` VE ayrıca **max-tabanlı**
istatistik önceliklendirilmeli — bugünkü ALFA rudder analizi (`docs/ML_ORNEK_INPUT_OUTPUT.md`
Örnek 2) sinyalin seyrek/sivri olabileceğini gösterdi, yalnız ortalama-tabanlı CUSUM yetersiz
kalabilir. `cusum_kwargs`/`fit_cusum_baselines` mevcut nedensel mekanizması aynen kullanılır
(yeni kolonlar `CUSUM_SOURCE_COLUMNS`'a eklenir).

## 4. Modüler IF entegrasyonu

[src/ml/models/modular_iforest.py](../src/ml/models/modular_iforest.py)'e yeni aday modül
(varsayılana ANCAK Gate B/C geçerse alınır — ML-7/ML-8A deseniyle aynı disiplin):

```python
PX4_ML9_CANDIDATE_MODULES = {
    **PX4_ML7_CANDIDATE_MODULES,
    "dikey_tutarlilik": [
        "ekf_alt_innov", "ekf_vertical_vel_innov",
        "ekf_alt_innov_5s_max", "ekf_vertical_vel_innov_5s_max",
        "ekf_alt_innov_cusum_pos",
    ],
    "motor_simetrisi": [
        "actuator_output_std", "actuator_output_range",
        "actuator_output_std_5s_max", "actuator_output_range_5s_max",
        "actuator_output_std_cusum_pos",
    ],
}
```

## 5. YENİ: annotasyon-kategorisi-düzeyi değerlendirme

Mevcut `flight_label` (altitude_anomaly/mechanical_fault) düzeyi ölçüm çok kaba —
mechanical_fault'un %66'sı Actuator Outputs, geri kalanı 4 farklı imza. Bu, "iyileşme
gerçekten hedeflenen kategoride mi oldu" sorusunu cevapsız bırakıyor.

`src/ml/evaluation/events.py`'ye ekle (mevcut `load_uav_sead_ranges`'ın YANINA, onu bozmadan):

```python
def load_uav_sead_ranges_by_category(labels_path) -> dict[str, dict[str, list[tuple]]]:
    """flight -> {annotation_category: [(start_us, end_us), ...]}"""
```

`run_ml9_*` betiği hem eski (flight_label bazlı) hem yeni (kategori bazlı) onset recall'ı
YAN YANA raporlar. Kategori bazlı recall < 10 event içeren kategoriler için **n belirtilerek**
raporlanır, genelleme iddiası yapılmaz (Battery n=2, Vibration n=1, Magnetometer n=3 gibi).

## 6. Split / karar katmanı / artifact — HİÇBİRİ YENİDEN YAZILMAZ

- Split: mevcut `split_manifest.json` (oturum-bazlı, 5 seed, `final_holdout` 76 uçuş dahil)
  aynen kullanılır. Yeni split üretilmez.
- Karar katmanları: `src/ml/decision/decision_layers.py`'deki threshold/K-of-N/CUSUM
  fonksiyonları DEĞİŞTİRİLMEDEN import edilir (ML-8A'da genel-amaçlı yazıldılar, skor
  kaynağından bağımsız çalışırlar).
- Artifact: `src/ml/artifacts.py`'deki checksum'lu manifest deseni; çıktı
  `artifacts/ml9/uav_sead/<run_name>/` altına.
- MLflow YOK, `src/ml/eval/` YOK (ML-8A'daki aynı iki düzeltme burada da geçerli).

## 7. Gate'ler

**Gate A (hepsi geçmeli):** yeni feature'lar nedensel (gelecek satıra bakmıyor — actuator
symmetry residual anlık satır-bazlı olduğu için otomatik nedensel, ama rolling/CUSUM
varyantları prefix-invariance testinden geçmeli); final_holdout hiç okunmadı; scaler/CUSUM
baseline yalnız train'de fit edildi.

**Gate B (en az biri):** (a) annotasyon-kategorisi-düzeyinde Position.Z onset recall, mevcut
pooled `ekf_pos_innov_mag`/`ekf_vel_innov_mag`'a göre anlamlı yüksek; VEYA (b) Actuator
Outputs+Controls kategorisinde onset recall, mevcut modüllere göre anlamlı yüksek.
Geçemezse: rescue yok, bulguyu H19+ olarak yaz, ML-10'a (forecast-residual/Chronos/MOMEN pilotu)
geç.

**Gate C:** aynı kritik (≤2 FA/saat, ≥0.30 recall) / advisory (≤12 FA/saat, ≥0.50 recall)
bütçeleri, BASE_MODULES+yeni-modül füzyonu üzerinde. Geçmezse holdout kapalı kalır
(otomatik açma yok, ML-8A'daki gibi).

## 8. Test listesi

```
test_ekf_alt_innov_no_future_leak         # rolling/cusum varyantlari icin prefix-invariance
test_actuator_output_active_channel_detection   # sentetik veri: sabit kanallar disari elensin
test_actuator_output_imbalance_flags_stuck_motor  # sentetik: bir kanal sabit -> residual yukselir
test_category_ranges_preserve_category_label      # load_uav_sead_ranges_by_category
test_ml9_decision_layers_reused_not_reimplemented  # decision_layers.py'den import, kopya yok
```

## 9. Çalışma sırası

1. `parse_uav_sead.py`: `ekf_alt_innov`/`ekf_vertical_vel_innov` (drop'tan önce sakla) +
   `actuator_outputs` parse'ı. Silver yeniden üret (`data/objectstore/silver/uav_sead`
   siline GEREKMEZ — `--local-bronze-dir` yolu objectstore'a yazmıyor, bkz. ML-8A notu).
2. `uav_attack_features.py`'ye SEAD-özel yeni feature bloğu (mevcut H13 EKF blok deseniyle
   aynı `if c in g.columns` korumasıyla).
3. `build_features.py` yeniden çalıştırılır (cusum baseline iki-geçişli fit zaten otomatik).
4. `modular_iforest.py`'ye `PX4_ML9_CANDIDATE_MODULES`.
5. `src/ml/evaluation/events.py`'ye `load_uav_sead_ranges_by_category`.
6. `scripts/run_ml9_category_evaluation.py`: split_00 (+ 5 seed) development'ta BASE_MODULES
   vs BASE+ML9 füzyonunu hem flight_label hem annotasyon-kategorisi düzeyinde ölçer.
7. Gate A/B/C değerlendirmesi; `docs/ML1_BULGULAR_VE_HATALAR.md`'ye "ML-9 sonuçları" (H19+).
8. `docs/decisions.md`'ye ADR-009 (Gate kararları).

## 10. Yapılmayacaklar

Yeni model mimarisi, ALFA/UAV Attack, Battery/Vibration/Magnetometer için özel feature
(n çok küçük), blind holdout açma, mevcut `ekf_vel_innov_mag`/`ekf_pos_innov_mag`'ı silme
(geriye uyumluluk — sadece yanına ekle), MLflow, `src/ml/eval/` (mevcut `evaluation/`
kullanılır).

## 11. Sonraki fazlar (bu fazda BAŞLAMA)

- **ML-10:** Gate B kalırsa forecast-residual pilotu (MOMENT — anomali için pretrain edilmiş
  tek foundation model; Chronos alternatif), önce altitude/mechanical zayıf kategorilerinde
  zero-shot prob olarak.
- **ML-11:** Kafka/Docker altyapı fazı (kullanıcının ayrı planladığı iş).
