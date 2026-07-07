# Claude Oturum Belleği — Geçici Dışa Aktarım

> **Bu dosya nedir:** Claude Code'un bu proje için tuttuğu kalıcı bellek dosyalarının
> (`~/.claude/projects/.../memory/`) birebir kopyası. Codex ve claude.ai gibi bu belleğe
> erişemeyen araçların proje durumunu okuyabilmesi için **geçici olarak** docs'a konuldu
> (2026-07-06). Kaynak bellek güncellendikçe bu kopya ESKİYEBİLİR — çelişki durumunda
> `docs/ML1_BULGULAR_VE_HATALAR.md`, `docs/decisions.md` ve `docs/ML_YETERSIZLIKLER_KAYDI.md`
> esas alınmalı.
>
> **Not:** `docs/MEMORY.md` bu dosyadan FARKLIDIR — o, proje başında (Bronze planlama dönemi)
> Codex için yazılmış eski bilgi tabanıdır; içindeki Faz 0-3 / MLflow planları kısmen aşılmıştır.

---

## İndeks

- [ML fazları durumu (ML-0..ML-11)](#1-ml-fazları-durumu-ml-0ml-11) — splitler, causal CUSUM, gitignore bug, ML-8A/ML-9 Gate B/C kaldı, ML-10 Chronos mechanical Gate B geçti/fusion Gate C kaldı, ML-11 görselleştirme tamam (feature×kategori AUC + top-10 aday + eğitim izi kuralı)
- [Anomali tespiti ilkeleri (feedback)](#2-anomali-tespiti-ilkeleri-feedback) — normal-sınıfı yapay homojenleştirme önerme, bu unsupervised'ın doğasına ters
- [Git commit no-coauthor (feedback)](#3-git-commit-no-coauthor-feedback) — bu repoda commit'lere Co-Authored-By ekleme
- [Dış paydaşa jargonsuz anlat (feedback)](#4-dış-paydaşa-jargonsuz-anlat-feedback) — mentöre yönelik içerikte ML-N faz numarası kullanma

---

## 1. ML fazları durumu (ML-0..ML-11)

*Tür: project — "ML-0..ML-11 faz durumu — feature tabloları, splitler, UAV-SEAD 611 uçuş, ML-9 Gate B/C kaldı, ML-10 mechanical Gate B geçti/fusion kaldı, ML-11 görselleştirme tamam"*

2026-07-02 itibarıyla ML-0 fazı tamam (FableChat.md/LastChat.md planına göre; bu dosyalar repo kökünde değil, kullanıcının paylaştığı sohbet dökümleri):

- `src/ml/` yeni modül: `features/temporal.py` (wrap-aware açı, CUSUM, geçmişe-bakan rolling, haversine, spektral, freeze), `features/alfa_features.py` (73 feature), `features/uav_attack_features.py` (`build_px4_features` — UAV Attack VE UAV-SEAD aynı feature uzayı; indoor uçuşlar için pos_x_m/pos_y_m öklid fallback), `data/splits.py` (uçuş bazlı, normal-only train, 5 seed + LOFO, `split_manifest.json`), `data/scaling.py` (train-only RobustScaler, JSON parametre).
- Çıktılar: `data/gold/ml_features/{alfa,uav_attack,uav_sead}/*_ml_features.parquet` + `split_manifest.json`; scaler'lar `artifacts/scalers/`.
- UAV-SEAD: HF `aykutkabaoglu/uav-flight-anomaly-dataset`ten 60 uçuş alt kümesi (20 normal + 4 sınıf × 10) `src/ingestion/uav_sead_downloader.py` ile Bronze'a indirildi; `src/silver/parse_uav_sead.py` pyulog ile parse ediyor. İç mekân uçuşları GPS taşımaz → `vehicle_local_position` fallback şart.
- velocity_mps kök nedeni ÇÖZÜLDÜ: ALFA `nav_info-velocity` kolonları `meas_x/des_x` adında (find_col("measured") eşleşmiyordu); UAV Attack `vehicle_gps_position` zaten `vel_m_s` taşıyordu. Gold null oranı %100→%9.3.
- Tuzak: `write_silver` her çalıştırmada yeni immutable part ekler, `read_layer` HEPSİNİ okur → Silver'ı iki kez çalıştırıp Gold'u üretince satırlar katlanır. Yeniden üretmeden önce `data/objectstore/silver/<source>` silinmeli.
- MinIO yok; `STORAGE_BACKEND=local` (`.env`) ile `LocalObjectStoreClient` (`src/common/local_store.py`) kullanılıyor.

ML-1 de tamam (2026-07-02): `notebooks/01_veri_ve_feature_incelemesi.ipynb` + `02_isolation_forest_cusum_egitim.ipynb` (çalıştırılmış, çıktılar gömülü; nbformat/nbclient ile üretildi). Kritik bulgular:
- Monolitik satır-bazlı IF başarısız (ALFA satır-ROC 0.50, UAV 0.21): etiket semantiği (saldırı logunun tüm satırları etiketli ama imza birkaç satırda), feature sulanması, NaN-impute'un anomalileri merkeze çekmesi. Değerlendirme UÇUŞ düzeyinde yapılmalı.
- Modüler IF + val-normalize füzyon (FableChat mimarisi): ALFA uçuş-ROC 0.833±0.17 (rehberlik modülü 0.864), AvgDT 3.7s/MaxDT 68s; UAV jamming 1.0 / spoofing 0.7 / ping_dos 0.37 tespit.
- ALFA'nın en güçlü tek dedektörü: alt_error (otopilot residual'ı) üzerinde CUSUM, uçuş-ROC 0.878.
- `gps_speed_residual` (konumdan hesaplanan hız vs receiver vel_m_s) GİZLİ live/hackrf spoofing'i yakalıyor (5.9 vs normal ~2) — SITL spoofing zaten 78 km sıçramayla trivial.
- Ping DoS 4/6 logda bu topic'lerden tespit edilemez (imza fiziğe yansımıyor) — kapsam beyanı maddesi.
- SEAD kalibrasyonsuz transfer tamamen başarısız (normal FA 1.0) — beklenen; yeni platformda tau kalibrasyonu şart.
- Feature'lar v2: ALFA 85, UAV/SEAD 58; split kotaları ALFA 6/2/2, UAV 4/1/1, SEAD 14/3/3. anomaly_events tablosu `data/gold/anomaly_events/`.

ML-2 de tamam (2026-07-02): `notebooks/03_autoencoder_lstm_ae_egitim.ipynb` (torch CPU, Dense AE + LSTM-AE, maske-ağırlıklı MSE, POT/GPD eşiği, `src/ml/data/windowing.py`). Sonuç: ALFA'da IF-füzyon (0.833) > LSTM-AE (0.731); UAV'de AE'ler (0.677) > IF (0.600) ve tip bazında LSTM-AE spoofing 1.00 / ping_dos 0.53'e çıkardı. Hatalar `docs/ML1_BULGULAR_VE_HATALAR.md`'de H1-H11 olarak numaralı (H10: max-pencere skoru yanlış alarm üretiyor; H11: pencere etiket semantiği — SEAD `ranges` ile çözülecek). Metot araştırması: `docs/ANOMALI_METOT_ARASTIRMASI.md` (POT/SPOT/DSPOT, USAD/TranAD, matrix profile, missing-aware modeller; hata modu→metot eşleme tablosu).

ML-3 de tamam (2026-07-02): `notebooks/04_ablation_enjeksiyon_usad.ipynb` + `src/ml/injection.py` (freeze/bias/drift/noise/gps_ramp/dropout, 6 test; 129 toplam test). Ana sonuçlar: (1) ALFA'da sadece-rehberlik modülü (0.864) tam füzyondan iyi — kontrol_tepki gürültü katıyor; (2) SEAD kalibre transfer 0.375→0.783, FA 1.00→0.33 — "model taşınır, eşik platforma kalibre edilir" tezi deneysel kanıtlandı (projenin ana iddiası); (3) H10 hipotezi reddedildi — uçuş skoru max kalır, oran-skoru sinyali seyreltiyor; (4) UAV enjeksiyonları mevcut eşiklerle kaçtı (H12: 2 m/s ramp val sınırında — şiddet taraması gerek); ALFA'da drift 0.75/0.28s (CUSUM doğrulandı), bias/freeze zayıf (B2: frozen-count feature'ları modüllere eklenmeli); (5) USAD (0.45/0.53) < LSTM-AE — elendi. Tüm bulgular H1-H12/B1-B2 olarak docs/ML1_BULGULAR_VE_HATALAR.md'de.

ML-4 (veri büyütme) de tamam (2026-07-03, notebooks/05): ALFA 54 uçuş/15 normal (raw rosbag'lerden `src/silver/parse_alfa_rosbag.py` + `scripts/inventory_alfa_raw.py`; 8 eksik bag'den 7 parse, 5 normal), UAV-SEAD 179 uçuş/59 normal (downloader skip-existing'li). ANA SONUÇ: **ALFA LSTM-AE 0.731→0.918** (H9 çözüldü — veri azlığı hipotezi doğrulandı; ALFA varsayılan modeli artık LSTM-AE). Dersler: IF-füzyon heterojen normallere kırılgan (B3, nav_info'suz rosbag uçuşları eşiği bozuyor); SEAD ranges adil satır-ROC 0.474 → UAV-Attack feature'ları SEAD state-estimation anomalilerini yakalamıyor (H13: SEAD'e özgü feature seti gerekir — EKF innovation vb.; SEAD şimdilik eşik kalibrasyonu + normal havuz işlevi görüyor). Çapraz havuz SEAD-test'i 0.617→0.671 iyileştirdi. 133 test geçiyor.

ML-5 de tamam (2026-07-03, notebooks/06): SEAD 349 uçuş/199 normal (%57 doğal dağılım, ExtPos 60; 1 uçuş HF'te kalıcı 404); oturum-bazlı split (`session_of` tarih klasörü, anomalili oturum normalleri karantinada — SEAD seed-std ±0.212→±0.012!); parse_uav_sead'e estimator_status test-ratio'ları + ekf2_innovations eklendi. SONUÇLAR: adil satır-ROC 0.474→**0.799** (iyileşme EKF'ten değil veri+oturum-temizliğinden!); **H14**: EKF test-ratio'ları TERS sinyal (0.354 — anomalide ölçüm reddi innovation'ı bastırıyor; reject-counter'larla birleşmeden kullanılmaz, varsayılan füzyona alınmadı, kolonlar Silver'da); tip tespiti ExtPos 0.82/GlobalPos 0.70, altitude 0.27 zayıf; POT ilk kez val-max'ı geçti (uçuş eşiğinde varsayılan POT). 135 test. Downloader'da --ext-pos flag'i ve class_overrides var.

ML-6/7 de tamam (2026-07-06, Codex çalıştı, bağımsız doğrulandı): **causal CUSUM düzeltmesi** — `temporal.py::cusum()` artık yalnız split_00 train-normal'den sabit `center/k` alıyor (fit_cusum_baselines, artifacts/cusum/), prefix-invariance testiyle kanıtlı. Bedeli: "ALFA'nın en güçlü sinyali" sanılan alt_error_cusum (0.878) causal ölçümde **0.611**'e düştü, gerçek en güçlü sinyal xtrack_error (0.751) çıktı. **Event-onset recall düzeltmesi (H16)**: eski event_recall event başlamadan önce açık kalmış alarmı da sayıyordu (overlap 0.594), gerçek "yeni alarm" onset recall'ı **0.194-0.224**. SEAD blind final holdout eklendi (`final_holdout_fraction` — 5 seed'de sabit, hiç açılmadı). Model/threshold artifact paketleme (`src/ml/artifacts.py`, `artifacts/models/<source>/`) ve K-of-N/CUSUM alarm politikası (`src/ml/evaluation/events.py`) eklendi. **KRİTİK ALTYAPI HATASI BULUNDU VE DÜZELTİLDİ**: `.gitignore`'daki ankorsuz `data/` deseni `src/ml/data/`yı da eşleştiriyordu — splits.py/scaling.py/windowing.py ML pipeline'ın en başından (70932ca) beri **hiçbir commit'te yoktu**, temiz clone'da ImportError verirdi. `/data/` olarak ankorlanıp düzeltildi.

ML-8A de tamam (2026-07-06): dondurulmuş causal 10s/1s pencere descriptor'ları (`src/ml/features/window_descriptors.py`, 20 descriptor/kanal) + class-balanced LightGBM (`src/ml/models/temporal_boosting.py`) + 3 karar katmanı (threshold/K-of-N/causal-CUSUM+bootstrap-ARL, `src/ml/decision/decision_layers.py`) + supervised split katmanı (`splits.py::add_supervised_splits` — normal-only novelty split'ten AYRI, session-izole, holdout-hariç). **SONUÇ: Gate B KALDI** — LightGBM window AUPRC SEAD'de 0.349 < mevcut IF-füzyon 0.385 (ALFA'da 0.843 < IF 0.858 < LSTM 0.872) — yeni model mevcut yöntemleri geçemedi. Gate C de LightGBM için hiç geçmedi; tek geçen hücre ALFA'da mevcut IF skoru + CUSUM karar katmanıydı (karar katmanının kazanımı, modelin değil). Literatür bu sonucu doğruluyor: az etiketli veride yarı-denetimli > tam-denetimli. Gerçek bir pencereleme bug'ı da bulunup düzeltildi (yoğun 1s grid telemetri boşluklarında hayali pencere üretiyordu → `full_matrix_gapfix/`, eski çıktı `SUPERSEDED.md` ile işaretli).

**Literatür notu (2026-07-06)**: (1) Rudder zayıflığının (ALFA, 0.333) kök nedeni ARAŞTIRILDI — fizik-prior zaten var (`turn_residual = yaw_rate - g·tan(roll)/V`, alfa_features.py:105), ama rudder_fault sadece **4 farklı uçuşta**, aileron_rudder_fault **1 uçuşta** ("1.00 tespit" sayıları n=1'den — istatistiksel olarak anlamsız). turn_residual'ın rudder_fault'ta ORTALAMASI normalle neredeyse aynı (-1.32 vs -2.10) — sinyal seyrek bir uç-değer (max 86.68), CUSUM/mean-tabanlı yöntemler bunu kaçırıyor. (2) "Normal veri çok fazla olursa anomali gizlenir mi": **"Heterogeneous Normal Classes Pose a Challenge for Anomaly Detection"** (OpenReview) — normal sınıf heterojense performans veri ARTSA BİLE kötüleşiyor; swamping/masking. SEAD'in normal uçuşları sadece **32 farklı oturuma** dağılıyor — gerçek bağımsız örneklem 199/398 değil 32. **ÖNEMLİ**: "oturum-koşullu model" mitigasyonu önerildi ama kullanıcı haklı olarak reddetti — unsupervised anomali tespitinin "normal karmaşık olabilir, buna rağmen sapmayı yakala" ilkesine ters. Artık "çözülecek iş" değil, dürüst bir SINIR. (3) Az etiketli veride yarı-denetimli > tam-denetimli; ML-8A Gate B sonucu bunun deneysel teyidi.

`docs/ML_ORNEK_INPUT_OUTPUT.md` yazıldı (2026-07-06): 5 gerçek input→output örneği — spoofing (başarı), rudder (kısmi), SEAD altitude (dürüst başarısızlık), EKF ters-sinyal (H14 öğretici), true-negative.

**SEAD tamamlama (2026-07-06)**: mapping.json'daki TÜM tek-sınıflı havuz indirildi — 413 uçuş; 412'si parse edildi. **Dürüst sonuç**: `alt_local_residual` altitude_anomaly'de hâlâ **%0 uçuşta dolu** (topic eksikliği — veri büyütmeyle düzelmiyor). AMA `hgt_test_ratio` ortalaması altitude_anomaly'de 0.047→**0.132** (normal sabit ~0.041) — gerçek bir ayrışma iyileşmesi. IF-fusion onset recall: altitude 0.043, global_position 0.137, mechanical 0.044, external_position 0.70-0.74 (en güçlü). SEAD tek-sınıf havuzu FİİLEN TÜKENDİ (mech/global/altitude %100 indirildi).

**SEAD 611 uçuşa büyütüldü (2026-07-06)**: normal havuz 199→398 (2 uçuş kalıcı olarak kurtarılamadı). Silver/features/split_manifest/scaler/CUSUM baseline yeniden üretildi (611 işlenebilir uçuş, 480 development + 131 blind holdout).

**Küçük-n araştırması (2026-07-06)**: (1) **SEAD'de gerçek ama mütevazı boşluk**: `select_flights()` çok-sınıflı 8 uçuşu sessizce atlıyor (altitude 73→78, mechanical 41→47 büyütürdü) — ML-9'un kategori-bazlı range altyapısıyla kullanılabilir, henüz YAPILMADI. (2) **ALFA'da boşluk YOK**: resmi makale (arXiv 1907.06268) tam 47 işlenmiş uçuş bildiriyor; `Desktop/ALFA/processed/processed/` da tam 47 klasör — rudder=4/elevator=2/aileron_rudder=1 sayıları akademik veri setinin kendi yapısal boyutu. `dataflash/` klasörü redundant (aynı iki günün paralel logu).

**ML-9 de tamam (2026-07-06, Codex çalıştı, bağımsız doğrulandı — checksum'lar yeniden hesaplandı, sayılar ham CSV'den türetildi)**: `parse_uav_sead.py`'de `ekf_alt_innov`/`ekf_vertical_vel_innov` drop'tan önce saklanıyor; `actuator_output_imbalance()` nedensel/expanding motor-simetri residual'i (gelecek sızıntısı yok, testle kanıtlı); `PX4_ML9_CANDIDATE_MODULES` (dikey_tutarlilik + motor_simetrisi). **Gate A GEÇTİ** (480+131=611, sıfır kesişim). **Gate B KALDI** — Position.Z'de dikey_tutarlilik 0.0956 vs pooled_ekf 0.0743 (+0.021, büyüklük barajı ≥0.05 geçilemedi); Actuator'da motor_simetrisi 0.2049 vs kontrol_cevabi 0.1805 (+0.024, 2/5 seed — kararlılık barajı ≥3/5 geçilemedi). **Gate C KALDI** — ml9_fusion 0.222 recall/25.83 FA-saat (bütçe ~2 kat aşılıyor). Aday modüller production'a alınmadı, holdout kapalı. Örüntü ML-8A ile aynı: küçük feature adımları recall'ı marjinal kımıldatıyor ama operasyonel kazanca dönüşmüyor.

**ML-10 de tamam (2026-07-06, Codex çalıştı, bağımsız doğrulandı)**: `amazon/chronos-bolt-tiny` zero-shot CPU'da `alt` + `actuator_output_std` kanallarında causal forecast-residual; tam koşu 101.2 s. **Gate A GEÇTİ** (future-leak/zero-shot/holdout-izolasyon testli). **Gate B mechanical dalında GEÇTİ**: `chronos_motor` CUSUM/advisory recall 0.205→0.390 (+0.185, 4/5 seed); Position.Z reddedildi (0.096→0.023). **Gate C KALDI**: ml10_fusion 0.213 recall/23.92 FA-saat (hedef ≥0.50/≤12) — kategori kazancı fusion düzeyinde kayboldu. `chronos_motor` production'a alınmadı, holdout kapalı. H22-H24, ADR-010, C.6. (`momentfm` Python 3.14'te kurulamıyor — F.4 yapısal sınır; `chronos-forecasting==2.3.1` temiz.)

**ML-12 de tamam (2026-07-07, Claude uyguladı, ön-kayıtlı — `docs/ML12_INCE_MODUL_PLAN.md`)**: ML-11'in seyrelme hipotezi kontrollü test edildi. Tek-feature `itki_komutu` (actuator_thrust_cmd, 1-feature IF) Actuator Outputs+Controls CUSUM/advisory recall'ını 0.205→**0.459**'a taşıdı (**Gate B B1 geçti**, 4/5 seed) VE ML-10'un `chronos_motor`'unu (0.390) da geçti (**B2 geçti**) — kategori için bilinen en iyi skor artık tek-feature modül; 3-feature varyant bile her yerde altında kaldı (seyrelme 3 feature'da ölçülür). **Gate C yine kaldı** (füzyon 0.217/23.74 FA-saat): kök neden ölçüldü — ince modül normal uçuşlarda 38.1 FA-saat bırakan kategori uzmanı, max-füzyon bunu hedef bütçede kullanamıyor (H30). Holdout kapalı, production füzyon değişmedi. H29-H30, ADR-012, C.7. Sonraki aday (ML-13): kategori-bazlı ayrı alarm kanalı mimarisi (ön-kayıt şart).

**ML-11 de tamam (2026-07-06, Claude uyguladı)**: `scripts/make_visualizations.py` (read-only) 3 dataset için veri karnesi / PCA+t-SNE (4 boyama, ±10 IQR görsel kırpma) / Spearman + **feature×kategori AUC matrisi** / model tanılama üretti → `artifacts/viz/*/viz_manifest.json` (checksum + development-id-hash). **ANA BULGU (H26)**: zayıf kategoriler ikiye ayrıştı — *füzyon/model sorunu*: `actuator_thrust_cmd` tek başına AUC 0.983 (Actuator O+C) ama 16-feature modülde seyreliyor; *veri/kapsam sorunu*: Position.Z'nin en iyi ayrıştırıcıları baro-tabanlı (AUC 0.996, n=33 satır — %7 doluluk), Actuator Thrust'ta hiçbir feature ayrışmıyor. Füzyon skoru doygun (normal satırlar 0.92-1.0; H27); ALFA LSTM-AE eşiği aşırı muhafazakâr (ROC 0.750 ama 3/38 tespit @ 0 FA; H28). **Sayı düzeltmesi**: "398 normal ~32 oturum" yanlıştı — gerçek **64 oturum** (dev: 324/49; D.1 güncellendi). Top-10 manuel feature-engineering adayı `docs/ML1_BULGULAR_VE_HATALAR.md` "Görselleştirme sonuçları"nda (keşif amaçlı; değerlendirme ancak yeni ön-kayıtlı Gate turunda). Eğitim izi kalıcı kural: `train_lstm_autoencoder` → `info["history"]`, `src/ml/training_log.py` → `artifacts/training_logs/...` loss.csv+PNG. Notebook 09 çalıştırılmış/gömülü; tam pytest 185 + bilinen 4 MinIO; ADR-011. Yetersizlikler kaydı `docs/ML_YETERSIZLIKLER_KAYDI.md`: 29 madde (A-G tema, 🔴🟡⚪✅).

---

## 2. Anomali tespiti ilkeleri (feedback)

*Tür: feedback — "Kullanıcı, normal-sınıfı yapay olarak homojenleştirmeye çalışan (session-koşullu model gibi) önerileri reddetti"*

Anomali tespiti önerirken "normal sınıfı homojenleştir/koşullandır" tarzı kısayollar önerme.

**Neden:** SEAD'de normal uçuşların sadece 32 oturuma dağıldığını ve bunun tespiti zorlaştırabileceğini söyleyince, mitigasyon olarak "oturum-koşullu normallik modeli" önerildi. Kullanıcı haklı olarak reddetti: "normalleri homojen yapmaya çalışmak zaten unsupervised learning'in işine ters değil mi?" Doğru — semi-supervised/unsupervised anomali tespitinin bütün amacı "normal karmaşık ve çeşitli olabilir, buna rağmen sapmayı yakala"dır. Bir oturuma/gruba göre model koşullandırmak, yeni/görülmemiş bir bağlamda modelin hiç referansı kalmaması riskini taşır — sorunu çözmüyor, gizliyor.

**Nasıl uygulanır:** Heterojen-normal + zayıf-sınıf durumunu "mühendislikle düzeltilecek iş kalemi" olarak sunma — dürüst bir SINIR/bulgu olarak raporla. Meşru alternatifler: (a) split/eşiği oturum-farkındalıklı tutmak (yapılıyor), (b) yeni bağımsız feature/veri kaynağı aramak, (c) sonucu olduğu gibi (zayıf) raporlamak.

---

## 3. Git commit no-coauthor (feedback)

*Tür: feedback — "Bu repoda commit'lere Co-Authored-By trailer'ı EKLENMEMESİ isteniyor"*

Bu repoda (AnomalyDetection) `git commit` atarken `Co-Authored-By: Claude...` trailer'ı EKLEME. Kullanıcı iki ayrı seferde açıkça istedi; commit sahipliği tamamen kendi adına görünmeli.

---

## 4. Dış paydaşa jargonsuz anlat (feedback)

*Tür: feedback — "Mentöre/dış paydaşa yönelik içerik üretirken repo-içi 'ML-N' faz jargonu kullanma"*

Mentöre veya proje mimarisini bilmeyen bir dış paydaşa yönelik tablo/rapor/döküman üretirken repo-içi "ML-1", "ML-8A" gibi faz numaralarına atıfta bulunma; yöntemi sade dille, NE YAPTIĞI üzerinden anlat (ör. "normal uçuş verisiyle eğitilen, sapmayı skorlayan bir model"). İç/teknik takip dokümanlarında ML-N numaralandırması olduğu gibi kalır — bu kural yalnızca DIŞA-DÖNÜK içerik için geçerli.
