# Görsel varlık envanteri

## Kullanım sınıfları

- **A — Doğrudan kullanılabilir:** Güncel sözleşmeyle üretilmiş ve sunum mesajı nettir.
- **B — Bağlam eklenerek kullanılabilir:** Gerçek artefakttır; development/triage/legacy sınırı slaytta görünmelidir.
- **C — Yalnız tarihçe:** Eski veya düzeltilmiş bir hattı anlatır; güncel sonuç diye gösterilemez.
- **D — Kullanma:** Reddedilmiş yaklaşım, yanlış truth/evaluator veya sunum kapsamı dışı hesap altyapısı.

## Güncel ana görseller — tek tek

| Dosya | Eksenler / kodlama | Ne anlatır? | Statü ve önerilen slayt | Kaynak |
|---|---|---|---|---|
| `artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/01_training_and_selected_checkpoint.png` | x=epoch; y=train/validation masked Gaussian NLL; seçilen epoch çizgisi | 30-epoch normal-only fit ve checkpoint seçimi | **A**, yöntem/fit slaytı; detection sonucu diye sunma | `scripts/plot_four_dataset_probabilistic_v31_rflymad.py`; aynı klasördeki PNG |
| `.../02_flight_max_score_distribution.png` | x=flight max score; y=yoğunluk/sayı; normal/anomaly sınıfları ve threshold | Uçuş max skor örtüşmesini | **B**, flight triage; event sonucu değil | `scripts/plot_four_dataset_probabilistic_v31_rflymad.py` |
| `.../03_flight_flag_matrix.png` | y=true normal/anomaly flight; x=no-alarm/alarm | TP=462, FN=95, FP=77, TN=14 any-window matrisi | **A**, ancak başlık zorunlu: “event confusion matrix değildir”; final sonuç slaytında B0 ile yan yana | `scripts/plot_four_dataset_probabilistic_v31_rflymad.py`; `docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_BASELINE_20260728.md` |
| `.../04_label_level_diagnostics.png` | fault label/domain bazında alarm veya skor özeti | Aile/domain heterojenliğini | **B**, ayrıntı/appendix; B0 event family grafikleri daha güncel | `scripts/plot_four_dataset_probabilistic_v31_rflymad.py` |
| `artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/01_recall_vs_false_event_rate.png` | x=false events/normal hour; y=event recall; renk=event policy family | Alarm yükü–recall Pareto yüzeyini | **A**, en güçlü final sonuç grafiği | `scripts/evaluate_rflymad_probabilistic_v31_events.py`; `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md` |
| `.../02_reference_budget_event_recall.png` | x=policy; y=event recall; referans validation bütçesi=1 false event/h | Referans noktada persistence, K-of-N, CUSUM kıyasını | **A**, karar katmanı slaytı | `scripts/evaluate_rflymad_probabilistic_v31_events.py` |
| `.../03_normal_test_alarm_burden.png` | x=policy; y=normal flights with any false event fraction | Yanlış olayların normal uçuşlara yayılımını | **A**, final NO-GO kanıtı | `scripts/evaluate_rflymad_probabilistic_v31_events.py` |
| `.../04_normal_false_event_timeline.png` | x=uçuş içi zaman; y üst=standardized NLL, y alt=Gaussian NLL; threshold/event işaretleri | Bağımsız normal uçuşta yanlış olay örneğini | **A**, vaka slaytı | `scripts/plot_rflymad_probabilistic_v31_event_timelines.py` |
| `.../05_detected_motor_event_timeline.png` | x=zaman; y=standardized ve raw Gaussian NLL; truth ve alarm aralıkları | Yakalanan Motor olay örneğini | **A**, vaka slaytı | `scripts/plot_rflymad_probabilistic_v31_event_timelines.py` |
| `.../06_missed_real_sensor_timeline.png` | x=zaman; y=standardized ve raw Gaussian NLL; truth aralığı | Kaçan Real/Sensor olay örneğini | **A**, kör nokta slaytı | `scripts/plot_rflymad_probabilistic_v31_event_timelines.py` |

`...` üstte bir önceki açık klasör yolunu kısaltır; gerçek dosya yolları `artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/plots/` ve `artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/` altındadır.

## Dört veri seti probabilistic v2 olay paketi

| Dosya | Eksenler / amaç | Statü / sunum |
|---|---|---|
| `artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/01_training_diagnostics.png` | x=epoch; y=masked Gaussian NLL; dört veri seti fit eğrileri | **B**; ortak yöntem transferi slaytı. Kaynak: `scripts/plot_four_dataset_probabilistic_v2_results.py`. |
| `.../02_detection_summary.png` | Panel A interval ROC-AUC; B flight max/mean ROC-AUC; C event recall; D false events/normal hour | **A/B**; metrik seviyelerini ayırmak için ideal. Rfly v2’yi v3.1 ile doğrudan kıyaslama. Kaynak: `scripts/plot_four_dataset_probabilistic_v2_results.py`. |
| `.../03_event_operating_grid.png` | x=validation false-event budget; y=persistence seconds; hücre=event recall | **A**, event policy seçimi öğretimi. Kaynak: `scripts/plot_four_dataset_probabilistic_v2_results.py`. |
| `.../04_reference_confusion_matrices.png` | x=predicted normal/alarm; y=interval truth | **B**; interval confusion, uçuş/event confusion değildir. Kaynak: `scripts/plot_four_dataset_probabilistic_v2_results.py`. |
| `.../05_alfa_reference_timelines.png` | x=Gold uçuş başlangıcından saniye; y=score (symlog) | **B**; az normal exposure uyarısıyla. Kaynak: `scripts/plot_four_dataset_probabilistic_v2_results.py`. |
| `.../05_rflymad_reference_timelines.png` | aynı | **B**; v2 source-split tarihsel örnek, v3.1 final değil. |
| `.../05_uav_attack_reference_timelines.png` | aynı | **B/C**; interval truth olmadığı ve alarm yükü yüksek olduğu yazılmalı. |
| `.../05_uav_sead_reference_timelines.png` | aynı | **B**; düşük event recall örneği. |

## ADS-B görselleri

| Klasör / dosyalar | İçerik ve eksen | Statü / öneri | Kaynak |
|---|---|---|---|
| `docs/assets/adsb_simple_anomaly/01_phase_distribution.png` | Uçuş phase dağılımı; kategori x sayım/oran | **B**, ilk fizik-bağlam teşhisi | `artifacts/adsb/simple_anomaly_20260722/summary.json` |
| `.../02_altitude_summary.png` | İrtifa kuralı değerlendirme ve tetik dağılımı | **B**, 57 evaluable / 2 trigger / 0 verified anomaly açık yazılmalı | aynı |
| `.../03_altitude_examples.png` | x=zaman; y=irtifa/residual; örnek uçuşlar | **B**, örnek; kabul kararı değil | aynı |
| `.../04_route_summary.png` | Rota kuralı değerlendirilen/tetiklenen/event sayısı | **A/B**, 95/13/24 ve düşük-hız artefaktı mesajı | aynı |
| `.../05_route_examples.png` | x=zaman/rota; y=bearing/konum türevi | **B**, artefakt vaka analizi | aynı |
| `artifacts/adsb/plots/contextual_v2_evaluation/event_detection_outcome_matrices.png` | recipe × detector outcome matrisi | **A**, ADS-B v2 senaryo sonucu | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` |
| `.../event_recall_heatmap.png` | recipe × detector; hücre=event recall | **A**, recipe-level research GO / universal NO-GO | aynı |
| `.../paired_clean_natural_burden_heatmap.png` | recipe/detector × temiz/doğal alarm yükü | **A**, recall ile burden’ı birlikte ver | aynı |
| `artifacts/adsb/plots/contextual_v2_timelines/*.png` (5) | x=zaman; y=ilgili kanal/skor; her sentetik tarif için timeline | **B**, tariflerin fiziksel anlatımı | `artifacts/adsb/plots/contextual_v2_timelines/` |
| `artifacts/adsb/plots/contextual_v2_detected_examples/*.png` (2) | x=zaman; y=skor/sinyal; yakalanan örnek | **B**, cherry-pick uyarısıyla vaka | ilgili klasör |
| `artifacts/adsb/plots/contextual_v2_flight_alarms/*.png` (2) | uçuş dashboard’u; skor/eşik/alarm | **B**, alarm katmanı örneği | ilgili klasör |
| `artifacts/adsb/plots/contextual_v2_preview*.png` (2) | enjeksiyon önizleme | **B**, yöntem şeması | ilgili klasör |
| `artifacts/adsb/plots/injection_timelines/*.png` (3) | clean vs injected kanal timeline | **A/B**, label-before/after benzeri fiziksel görünürlük | ilgili klasör |
| `artifacts/adsb/plots/reporting_summary/adsb_research_progression.png` | kronolojik ilerleme özeti | **B**, bu yeni sunum kronolojisiyle sayı/dil kontrolü sonrası | `artifacts/adsb/plots/reporting_summary/` |
| `.../anomaly_detector_map_V5.png` | tarif–detector haritası | **B**, yöntem haritası | aynı |
| `artifacts/adsb/plots/{auc_heatmap,confusion_matrices,loss_curves,roc_curves,score_distributions}.png` | eski ADS-B model kıyas grafikleri | **C**; yalnız tarihçe, güncel v2/B0 sonucu değil | `artifacts/adsb/plots/` |
| `archive/2026-07-10_rejected_adsb_attempts/**/plots/*.png` (25) | hard-rule ve injected-example figürleri | **D**, reddedilmiş yaklaşım; yalnız “neden sıfırlandı?” slaydında kırmızı “rejected” etiketiyle küçük örnek olabilir | `archive/2026-07-10_rejected_adsb_attempts/`; `AGENTS.md` |

Hesap yükü görselleri `artifacts/adsb/plots/compute_load_comparison/` ve `artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_compute_measured*/` altındadır; sunum kapsamı platform değil anomaly araştırması olduğundan **D — kullanma**.

## RflyMAD Full v2 ve sunum-seçilmiş görseller

| Klasör / dosyalar | İçerik | Statü / öneri | Kaynak |
|---|---|---|---|
| `docs/sunum_hafta5_gorseller/rflymad_frozen_ae_tcn_hedefli_karsilastirma.png` | x=false alarm/saat, y=recall; hedef bölgesi | **B**, eski v2 robustness; magnitude kontrolü bu turda yapılmadı notuyla | `docs/sunum_hafta5_gorseller/captions.md` |
| `.../rflymad_tcn_5kat_kararliligi.png` | x=fold, y=recall/FA | **B**, split oynaklığı | aynı |
| `.../rflymad_ae_tcn_karsilastirma.png` | AE ve TCN politika/domain kıyası | **B**, veri kapsamları farklı uyarısı | aynı |
| `.../rflymad_uzun_egitim_real_tradeoff.png` | eğitim adayı × Real recall / genel recall / FA | **B**, “daha uzun eğitim” trade-off’u | aynı |
| `.../rflymad_alarm_zaman_serisi_ornekleri.png` | x=zaman, y=model skoru/eşik; üç vaka | **B**, örnek; gate kararı değil | aynı |
| `artifacts/rfly_full/v2/visuals/01_data_composition.png` | domain/family/rol sayımları | **B**, Rfly veri kartı | `artifacts/rfly_full/v2/visuals/README.md` |
| `.../02_feature_completeness_heatmap.png` | x=özellik, y=domain/family; hücre=completeness | **B**, veri kalitesi | aynı |
| `.../03_correlation_normal.png`, `04_correlation_fault_active.png`, `05_correlation_delta.png` | özellik×özellik korelasyonları | **B**, appendix; performance değildir | aynı |
| `.../06_pca_tsne_family_domain.png` | PCA/t-SNE embedding; renk=family/domain | **B**, domain kümelenmesi; genelleme başarısı değildir | aynı |
| `.../07_knn_neighbor_agreement.png`, `08_knn_fold0_confusion_matrix.png` | komşuluk uyumu / fold-0 matrisi | **B**, temsil teşhisi; development-only | aynı |
| `.../09_feature_shift_by_domain.png` | x=özellik, y=domain shift | **B**, domain shift | aynı |
| `.../10_model_confusion_matrices.png` | Dense AE/TCN matrisleri | **C/B**, kapsamları farklı; doğrudan kıyaslama yapma | aynı |
| `.../11_ae_score_by_phase_domain.png` | phase/domain bazında AE skor dağılımı | **B**, phase/domain teşhisi | aynı |
| `artifacts/rfly_full/v2/dense_ae_diagnostics/{domain_family_recall,threshold_recall_fa_curve}.png` | recall ayrışımı; x=FA, y=recall | **B**, legacy v2 teşhis | ilgili klasör |
| `artifacts/rfly_full/v2/normal_temporal_ae/sweep_*/01..04*.png` (12) | rotation trade-off, kararlılık, family heatmap, confusion | **C/B**, yalnız robustness kronolojisi | ilgili klasörler |
| `artifacts/rfly_full/v2/normal_temporal_ae/robustness/.../R4/00..08*.png` (9) | loss, epoch seçimi, metric trade-off, vaka | **C/B**, R4 başarısızlık analizi | `docs/sunum_hafta5_gorseller/captions.md` |
| `gecmis_calismalar/RFLYMAD/raporlar/assets/rflymad_v2_convergence/00..08*.png` (9) | üstteki R4 paketinin rapor kopyası | **C**, duplicate; artefact kopyasını tercih et | ilgili klasör |
| `gecmis_calismalar/RFLYMAD/raporlar/assets/rflymad_v2_tcn_development/01..05*.png` (5) | TCN loss/fold/AE kıyas/Real trade-off | **C/B**, tarihsel TCN slaytı | ilgili klasör |
| `artifacts/rfly_dl/direct_v1_5split_20260720/*.png` (10) | training, split stability, ROC/FA/confusion/delay | **C**, direct DL development history; v3.1 final değil | ilgili klasör |

## ALFA, Attack, SEAD ve residual tarihsel portföyleri

| Görsel ailesi | Dosyalar / eksen kodu | Statü ve kullanım | Kaynak |
|---|---|---|---|
| Hafta-3 GNSS paketi | `gecmis_calismalar/_ortak/gorseller_sunum_hafta3/{auc_heatmap,confusion_matrices,roc_curves,score_distributions}.png`; senaryo/detector AUC, ROC, skor dağılımı | **B/C**; Dense-AE/LSTM-AE/LSTM-forecaster alternatiflerini anlatır, “basit istatistik” deme | `docs/PROJE_SUREC_VE_SONUC.md` |
| ALFA portföy | `gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/artifacts/viz/alfa/s1_portfolio/*.png` (3), `s2_embeddings/*.png` (7), `s3_features/*.png` (3), `s4_model/*.png` (13) | **C/B**; veri kompozisyonu/örnek timeline için kullanılabilir, eski model ROC/confusion final değil | ilgili klasörler; `docs/final_rapor_ml_fizibilite_2026-07-16.md` |
| UAV Attack portföy | `.../artifacts/viz/uav_attack/s1_portfolio/*.png` (3), `s2_embeddings/*.png` (7), `s3_features/*.png` (3) | **C/B**; veri kalitesi ve gözlenebilirlik bağlamı, final performance değil | ilgili klasörler |
| UAV-SEAD portföy | `.../artifacts/viz/uav_sead/s1_portfolio/*.png` (5), `s2_embeddings/*.png` (9), `s3_features/*.png` (3), `s4_model/*.png` (19) | **C/B**; session/domain ve magnitude hikâyesine seçici kullan; eski confusion final değil | ilgili klasörler |
| SEAD eğitim loss’ları | `.../artifacts/training_logs/uav_sead/ml16_*/**/loss.png` (çoklu split/run) | **C**, model öğrenmesini kanıtlamaz; appendix dışında kullanma | ilgili klasörler |
| SEAD mentor pack | `gecmis_calismalar/UAV_SEAD/egitilmis_modeller/ml14/uav_sead/mentor_pack/figures/{ml14_fa_drift_old_vs_new,ml14_recall_vs_false_alarm,rflymad_parsed_pool}.png` | **C/B**, eski deney kıyasları | ilgili klasör |
| Residual handout | `artifacts/residual_v1/phase_e_handout_20260717/01..07*.png` | **B/C**, leakage/magnitude/calibration coverage dersleri | `gecmis_calismalar/_ortak/raporlar/RESIDUAL_V1.md` |
| Residual vaka görselleri | `artifacts/residual_v1/runs/**/plots/*.png` ve `.../handout/flights/**/plot.png` | **C/B**, yüzlerce uçuş-vaka plot’u; toplu sonuç değil, yalnız önceden belirlenmiş örnek | ilgili artefact kökü |
| Basit sınıf/completeness kopyaları | `gecmis_calismalar/{ALFA,UAV_ATTACK}/raporlar/gorseller/*.png`; `gecmis_calismalar/RFLYMAD/raporlar/gorseller/*.png` | **C/B**, veri kartı; duplicate ise ana artefactı seç | ilgili klasörler |

## Sunum için önerilen mevcut görsel seti

1. Veri/split: `artifacts/rfly_full/v2/visuals/01_data_composition.png` ve `06_pca_tsne_family_domain.png` (**B**).
2. Metrik düzeyi: `artifacts/four_dataset_probabilistic_event_eval_v1/plots/20260728_v2/02_detection_summary.png` (**A/B**).
3. v3.1 model fit: `artifacts/four_dataset_probabilistic_v31/plots/rflymad_colab_l4_20260728/01_training_and_selected_checkpoint.png` (**A**).
4. Any-window yanılsaması: aynı klasörde `03_flight_flag_matrix.png` (**A**, açık uyarıyla).
5. Nihai sonuç: B0 `01_recall_vs_false_event_rate.png`, `03_normal_test_alarm_burden.png` ve iki timeline (**A**).
6. ADS-B fiziksel ders: `docs/assets/adsb_simple_anomaly/04_route_summary.png` + `05_route_examples.png` (**B**).

Bu envanter geçici pytest/PDF extraction görsellerini sunum varlığı saymaz; `archive/` yalnız tarihçe, duplicate rapor kopyaları ana artefactın yerine geçmez. Kaynak sınıflandırması `AGENTS.md` ve yukarıdaki artefact/rapor provenance’larına dayanır.
