# Kod Haritası

Bu klasör iki katmandan oluşuyor: **`adsb/`** (güncel, aktif ADS-B anomali
paketi) ve **`gecmis_calismalar/`** (veri kümesi bazında geçmiş çalışmalar +
paylaşılan çekirdek modüller). `scripts/`, `tests/`, `notebooks/`,
`reddedilen_denemeler/` ayrı bölümlerde.

Çalışma dizini `individual_anil/` olduğunda tüm import'lar ve `configs/`,
`artifacts/` yolları olduğu gibi çalışır (bkz. [CALISTIRMA.md](CALISTIRMA.md)).

## `adsb/` — güncel ADS-B anomali paketi

En yeni nesil: contextual-physics v1/v2 + kural/CUSUM karar katmanı.

| Dosya | Ne yapıyor |
|---|---|
| `features.py` | Fiziksel-tutarlılık residual'ları + birleşik feature tablosu. Residual'lar öğrenilmiş değil, aritmetik özdeşlik — genlik-baskınlığı burada yapısal olarak oluşamaz (bkz. bulgu 1). |
| `rules.py` | Kural-tabanlı, formül-tabanlı fizik-residual penalty skorlayıcısı (`ResidualRuleScorer`) — öğrenme yok, yalnız kalibre edilmiş eşik + penalty toplamı. |
| `cusum.py` | İşaretli ADS-B hız residual'ları için nedensel, iki-taraflı Page CUSUM. Eşik yalnız donmuş girdi (`CusumConfig`), **arama metodu yok**. |
| `cusum_truth_v2_eval.py` | Dondurulmuş Step-5 vektör-CUSUM için fail-closed truth-v2 değerlendirmesi (yalnız değerlendirme, hiçbir dedektörü fit etmez). |
| `contextual_decision.py` | Kanal başına açık alarm bütçeleri + zaman-duyarlı anomali karar profilleri. |
| `contextual_decision_fast.py` | `contextual_decision.apply_detector_profile`'ın vektörize, bit-bit eşdeğer performans ikizi. |
| `contextual_events.py` | Contextual dedektör alarm emisyonlarından simple-anomaly olaylarına adaptör. |
| `contextual_scaling.py` | Contextual model kanalları için katı, yalnız-normal robust ölçekleme. |
| `contextual_windowing.py` | Contextual residual forecaster için sızıntısız (leakage-safe) sıradaki-satır pencereleri. |
| `conditional_calibration.py` | Kanal skorları için hiyerarşik koşullu conformal kalibrasyon (yalnız doğal-temiz kalibrasyon verisi kabul eder). |
| `context.py` | ADS-B anomali kalibrasyonu için nedensel normal-bağlam feature'ları (bir anomali etiketi DEĞİL). |
| `diagnostics.py` | **Zorunlu eğitim-sonrası tanı** — `magnitude_domination_check()`, `magnitude_only_score()` (bulgu 1'in ölçüldüğü yer). |
| `evaluation.py` | Açık pencere/olay/uçuş birimli değerlendirme ilkelleri — `truth_event_table()`, `active_interval_coverage()`, `event_observability_denominators()`, `scoreable_exposure()`, `natural_alert_burden()`, `alarm_episodes()`, `event_detection_metrics()`. Eşik seçimi yapmaz, dondurulmuş bir kararı alır. |
| `truth.py` | Sentetik ADS-B olayları için satır/pencere truth-v2 sözleşmesi — `attach_event_truth_v2()`, `attach_clean_truth_v2()`. |
| `synthetic.py` | Test-ONLY sentetik bozulma enjeksiyonu (`inject_freeze`, `inject_bias`, `inject_noise`, `inject_position_ramp`). Enjekte edilmiş veri asla eğitime girmez; `save_synthetic_batch` çıktı yolunun `synthetic` içerdiğini zorunlu kılar. |
| `inventory.py` | Bir adsb.lol/readsb tar arşivini tam parse etmeden hızlı profil çıkarır. |
| `segmentation.py` | Sürekli ICAO24 trace akışını ayrık uçuşlara böler. |
| `scaling.py` | Kırpılı robust ölçekleme (SEAD dersinin doğrudan uygulanması — kırpılmamış ölçekleme birkaç aşırı-genlik kanalın skoru sürüklemesine yol açmıştı). |
| `windowing.py` | Uçuş-düzeyi feature tablosunu sabit uzunluklu model-girdisi pencerelerine çevirir. |
| `streaming.py` | Tam-gün ADS-B baseline değerlendirmesi için sınırlı-bellek yardımcıları. |
| `s2.py` / `s2_streaming.py` | S2 durum/veri-kalitesi neden kodları (penalty skorlayıcısından bağımsız) + vektörize doğal-yük özetleri. |
| `simple_anomaly.py` | Küçük, ön-kayıtlı ADS-B faz/anomali kuralları — öğrenilmiş model veya eşik araması içermez. |
| `run_manifest.py` | Değişmez, denetlenebilir run manifestleri — bir run dizini asla yeniden kullanılmaz. |
| `models/contextual_persistence_v2.py` | Conformal p-değerleri üzerinde kümülatif, tüm-uçuş kalıcılık (contextual_physics_v2 karar katmanı). `score()` metodu. |
| `models/contextual_residual_forecaster.py` | Fizik residual kanalları için küçük, bağlam-duyarlı sıradaki-adım tahmincisi. |
| `models/isolation_forest_residual.py` | Isolation Forest tabanlı contextual v1 keşif kolu — contextual_physics_v1 ile aynı 5 residual kanalı. |
| `models/dense_autoencoder.py` | Düz feed-forward otokodlayıcı — pencereyi vektöre açıp yeniden kurar. |
| `models/lstm_autoencoder.py` | LSTM seq2seq otokodlayıcı — zamansal örüntü öğrenmesi umulan mimari. |
| `models/lstm_forecaster.py` | LSTM-AE'nin türevi: yeniden kurmak yerine `horizon` adımını tahmin eder. |
| `models/usad.py` | USAD (UnSupervised Anomaly Detection, Audibert ve ekibi 2020) implementasyonu. |
| `reports/` | `measurability_table.md` (hangi ilişkinin ölçülebilir/yalnız-geçişte-ölçülebilir/ölçülemez olduğu kararı) + envanter/kalibrasyon çıktıları. |

## `gecmis_calismalar/` — veri kümesi bazında geçmiş çalışmalar + paylaşılan çekirdek

### Paylaşılan çekirdek modüller

| Modül | Dosya | Ne yapıyor |
|---|---|---|
| `anomaly_core/` | `sequential.py` | Nedensel çok-kanallı Page CUSUM (`MultiChannelPageCUSUM`, `PageCUSUMConfig`) — demo'nun 4. bölümünde çağrılan sınıf. |
| | `calibration.py` | Yalnız-doğal hiyerarşik konformal kalibrasyon. |
| | `forecaster.py` | Küçük, yalnız-doğal konum/ölçek LSTM tahmincisi. |
| `residual_v1/` | `run.py`, `schema.py`, `tracking.py` | Değişmez run/provenance yardımcıları, dondurulmuş ham-kanal şeması, opsiyonel MLflow adaptörü. |
| | `decision/` | `cusum.py` (paylaşılan CUSUM çekirdeğinin RESIDUAL-V1 sarmalayıcısı), `calibrate.py` (60s block-bootstrap eşik kalibrasyonu), `scaling.py` (dondurulmuş train-normal median/MAD ölçekleme). |
| | `eval/` | `sanity_gates.py` (**eşikten bağımsız akıl-sağlığı kapıları** — `s1_magnitude_gate()`, `require_s3_pass()`), `s4_ablation.py` (S-4 komut-çıkarma ablasyonu). |
| | `features/` | `align.py` (nedensel doğal-oranlı hizalama), `build.py` (residual feature-matrisi), `phases.py` (kural-tabanlı uçuş fazı bölümleme), `physics.py`, `spec.py` (dondurulmuş kanal tanımları + AR-sızıntı bekçisi), `waypoints.py`. |
| | `ingest/` | `alfa.py`, `rfly.py` (doğal-oranlı ALFA/RflyMAD-Real alımı), `alfa_channels.py`/`rfly_channels.py` (ön-kayıtlı kanal envanteri), `splits.py` (**oturum-düzeyinde grup-güvenli bölme**), `profile.py`, `common.py`. |
| | `models/` | `g0_rules.py` (öğrenilmemiş komut/tepki fizik taban çizgisi), `g1_ridge.py` (geliştirme-only, oturum-gruplu ridge). |
| | `viz/` | `handout.py` (sızıntı-farkında, uçuş-düzeyi ham telemetri raporu). |
| `rfly_dl/` | `config.py`, `data.py`, `models.py`, `experiment.py`, `decision.py`, `evaluation.py`, `reporting.py` | RflyMAD "doğrudan" (direct) derin öğrenme deneyi — dondurulmuş config, sızıntı-kontrollü veri, kapasite-kontrollü modeller, yalnız-normal-validasyon alarm politikaları, beş-bölmeli uçtan uca deney. |
| `rfly_full/` | `contract.py` | RflyMAD-Full v2 veri kümesi sözleşmesi + kanonik uçuş manifestosu. |
| | `pipeline.py`, `expansion.py`, `v2_parser.py` | Kontrol noktalı indirme/parse hattı + resmi SIL/HIL aynalarının ayrı alımı + nedensel 10 Hz çok-oranlı parser. |
| | `normal_ae.py`, `normal_ae_reporting.py` | Alan-kalibreli, yalnız-normal zamansal otokodlayıcı + görsel raporlama. |
| | `supervised.py`, `supervised_sweep.py` | Denetimli zamansal TCN taban çizgisi + geliştirme-only beş-katlı tarama. |
| | `robustness.py` | Ön-kayıtlı, geliştirme-only Wind/Real gürbüzlük deneyleri. |
| | `dl_worker.py` | Parse edilen batch'leri izler, dondurulmuş bir Dense-AE eğitir, değerlendirir. |
| | `ae_diagnostics.py` | Dondurulmuş 1Hz Dense-AE taban çizgisi için post-hoc tanılar (genlik-baskınlığı denetiminin RflyMAD tarafı). |
| | `truth_audit.py` | RflyMAD-Full v2 10Hz parse için truth denetimi (bulgu 2'nin RflyMAD tarafı). |
| | `summary.py`, `visualize.py` | Metrik seviyelerini karıştırmadan kontrol-noktalı kanıt özeti + geliştirme-only görsel tanılar. |
| `uav_gnss/` | `pipeline.py` | Karar-dondurulmuş PX4 GNSS-bütünlük fizibilite pilotu çalıştırıcısı. |
| | `frozen_runner.py` | **Yapılandırma kilitli** governance-sertleştirilmiş çalıştırıcı. |
| | `catalog.py`, `features.py`, `evaluation.py` | RflyMAD gerçek-uçuş keşfi + truth doğrulama, nedensel PX4 feature çıkarımı, açık episode/yük/olay-son-tarih/güven metrikleri. |
| | `final_report.py` | Dondurulmuş pilot artefaktlarından Overleaf-hazır detaylı NO-GO raporu üretir. |

### Veri kümesi bazında arşiv (çalıştırılabilir değil, dondurulmuş)

`ALFA/`, `UAV_ATTACK/`, `UAV_SEAD/`, `RFLYMAD/` — her biri aynı iç yapı:
`VERI_ARTEFAKT_KONUMU.md` (ham veri nerede, repoda değil), `kaynak_kod`/
`kaynak_kod_legacy`/`legacy_rfly0_1` (dataset'e özel parser/model kodu),
`egitim_yontemleri_ve_modeller`/`egitilmis_modeller` (eğitilmiş model
artefaktları), `raporlar/` (dataset'e özel bulgu raporları + `gorseller/`).

**Bu dört klasördeki kod arşivseldir, ortak veri hattının eski hâline
bağımlıdır ve çalıştırılabilir değildir** — `from src.silver.parse_uav_sead
import ...` gibi satırlar içerirler ama o modül artık `src/silver/` altında
yok (`arsiv`'de de kırık; taşımada bozulmadı, zaten dondurulmuş arşiv kodu).

`_ortak/` — üç alt klasör: `legacy_ml_kutuphanesi/` (ML-0…ML-16 boyunca
biriken eski model/değerlendirme kütüphanesi — `src_ml/`, kendi
`notebooks/`/`scripts/`/`tests/`/`artifacts/`'ı ile; ayrı `README.md`'sinde
neden arşivlendiği yazıyor), `raporlar/` (RESIDUAL-V1 ve RFLYMAD-doğrudan
birleşik raporları — `RESIDUAL_V1.md`, `RFLY_DL_DOGRUDAN_DEGERLENDIRME_PLANI.md`),
`gorseller_sunum_hafta3/` (hafta-3 sunum görselleri: AUC ısı haritası, ROC
eğrileri, karışıklık matrisleri, skor dağılımları).

## `scripts/` — önek → hangi hat

| Önek | Hat |
|---|---|
| `adsb_*` | Güncel ADS-B contextual-physics v1/v2 deney/rapor sürücüleri |
| `build_four_dataset_*`, `four_dataset_probabilistic_*`, `prepare_four_dataset_*`, `plot_four_dataset_*`, `audit_four_dataset_*`, `evaluate_rflymad_probabilistic_*`, `export_rflymad_probabilistic_*`, `plot_rflymad_probabilistic_*`, `build_rflymad_probabilistic_*` | Dört-veri-kümesi (ALFA/UAV Attack/UAV-SEAD/RflyMAD) grup-güvenli olasılıksal değerlendirme v3/v3.1 ailesi |
| `build_experiment_dashboard.py`, `build_adsb_burden_dashboard.py`, `audit_uav_sead_session_keys_v1.py` | Rapor/denetim üretimi |
| `RFLYMAD_rfly_full_v2/` | RflyMAD-Full v2 hattının tüm sürücüleri (indirme, parse, eğitim, sweep, rapor) |
| `RFLYMAD_uav_gnss/` | UAV GNSS-bütünlük pilotu çalıştırıcıları |
| `_ortak_residual_v1_ALFA_RFLYMAD/` | RESIDUAL-V1 hattının tüm sürücüleri (ingest, feature, kalibrasyon, handout) |

## `tests/`

Hepsi veri/ağ/Docker gerektirmiyor. Çoğu davranışsal (belirli bir girdi için
beklenen çıktı), bir kısmı davranış değil **matematiksel özellik** doğruluyor
— örnek: tutarlı bir uçuşta residual sıfıra yakın çıkması, `MultiChannelPageCUSUM`'un
uçuş sınırında sıfırlanması, `scaling.py`'deki ölçekleyicinin uçuş dışına
sızmaması. Kök klasördeki `test_adsb_*`/`test_four_dataset_*`/`test_rflymad_*`
dosyaları + üç alt klasör (`RFLYMAD_rfly_full_v2/`, `RFLYMAD_uav_gnss/`,
`_ortak_residual_v1_ALFA_RFLYMAD/`, `_ortak_anomaly_core/`) yukarıdaki
`scripts/` önekleriyle bire bir eşleşir.

## `notebooks/`

10 Colab GPU eğitim defteri — dört veri kümesi (ALFA/RflyMAD/UAV Attack/
UAV-SEAD) × olasılıksal v1/v2 (+ RflyMAD için ayrıca v3.1) ailesi, artı
`adsb_contextual_physics_v2_colab.ipynb`. Yerel GPU olmadan eğitim/sweep
turlarını çalıştırmak için — girdi/çıktı sözleşmesi ilgili `scripts/*_colab_*`
betikleriyle eşleşir.

## `reddedilen_denemeler/`

2026-07-10'da reddedilen iki ilk ADS-B yaklaşımı: `src/adsb/` (ADSB-0/ADSB-1
segmentasyon + fizik residual + enjeksiyon taslağı) ve `src/adsb_behavioral/`
(ayrı geliştirilen robust-rule/Isolation Forest/hard-physics yaklaşımı).
Reddedilme nedeni kendi `README.md`'sinde: düzeltme sonrası sentetik kolay
anomalilerde yüksek event recall görüldü, ama doğal veride kullanılamaz
seviyede yeni-alarm/saat oluştu — yüksek sentetik recall gerçek başarı olarak
yorumlanamadı. Salt-okunur; yeni çalışma buradan kod kopyalamadan başlar.

## `dokunti_arsivi/`

Kök dizinden arşivlenmiş eski çalışma-logları (`loglar/`) ve gevşek
script'ler (`gevsek_scriptler/`) — 2026-07-22 dosyalama. Kod değil, çalışma
zamanı artefaktı.
