# Eksik görseller planı

Bu dosya yeni deney veya model eğitimi önermez. Yalnız mevcut rapor/JSON/CSV artefaktlarından sunum görseli üretme planıdır. Ham skor gerektiren kalemler ayrıca işaretlenmiştir.

| Öncelik | Eksik görsel | Tam tasarım | Veri kaynağı | Üretim yolu | Kullanılacak slayt |
|---:|---|---|---|---|---|
| P0 | Tek sayfalık proje kronolojisi | x=zaman/aşama; y=veri, model, ölçüm-disiplini üç yüzme şeridi; düğüm rengi=GO/NO-GO/düzeltme | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`; `01_project_timeline.md` | Yeni çizim script’i gerekir; elle SVG/PPT de yapılabilir. Yeni deney yok. | 2–3 |
| P0 | Veri seti seçim/fizibilite matrisi | x=ALFA, Attack, SEAD, Rfly, ADS-B; y=truth, normal çeşitlilik, bağımsız grup, fiziksel gözlenebilirlik, event interval; hücre=iyi/sınırlı/yok | `02_dataset_inventory.md`; `docs/final_rapor_ml_fizibilite_2026-07-16.md` | Markdown tablodan doğrudan sunum-native matrise; script zorunlu değil | 4–5 |
| P0 | Metrik merdiveni | Soldan sağa window score → flight aggregate → event policy → event recall/false events-h; her basamakta “neyi kanıtlamaz?” notu | `04_metric_evolution.md`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json` | Yeni şema; sunum aracıyla vektör | 6 |
| P0 | Truth/parser önce-sonra | x=uçuş zamanı; y=fault active; üst=sahte t=0, alt=düzeltilmiş gerçek onset; yan kart=2.712/6.605 | `gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md` | Gerçek tek-uçuş satırı gerekirse canonical Gold’dan seçilmeli; yalnız kavramsal şema yapılırsa “schematic” etiketi | 8 |
| P0 | Gaussian forecaster şeması | 32-adım causal pencere + mask/cadence → LSTM → μ,σ → NLL → event policy | `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `configs/four_dataset_probabilistic_v31_rflymad.json` | Yeni vektör şema; deney yok | 14 |
| P0 | Negatif NLL açıklama grafiği | x=standardized residual z; y=`0.5z²+logσ`; 3 eğri σ=0,2/1/2; negatif bölge taralı | `_masked_gaussian_nll` in `scripts/four_dataset_probabilistic_gpu_v1_runner.py` | Formülden deterministik çizim; yeni model/deney yok | 15 |
| P0 | v2→v3.1 “adil kıyas değil” kartı | İki kolon: v2 source/in-domain vs v3.1 scenario/domain group-safe; satırlar rol, truth, metric, final erişim | `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`; `configs/four_dataset_probabilistic_v31_evaluation_contract.json` | Sunum-native tablo | 16 |
| P0 | Nihai sonuç kartı | Üç sayı: %43,27 event recall; 0,654 false event/h; Real %7,81 & Sensor %5,45; altında NO-GO | `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`; `artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval/` | Mevcut `01` ve `03` plotlarından sayılar; sunum-native kart | 18 |
| P1 | Trained–random–magnitude üçlü scatter | Panel A x=random score y=trained score; Panel B x=target RMS magnitude y=trained; ρ anotasyonları | `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json` yalnız özet ρ içeriyor; nokta verisi persist edilmemiş olabilir | Ham validation skorları varsa yeniden plot; yoksa checkpoint scorer çalıştırmak gerekir. Bu **yeni eğitim değildir**, fakat kullanıcı onayı olmadan bu görevde çalıştırılmadı. | 13/15 |
| P1 | Group-safe split Sankey/şerit | x=Rfly scenario/domain groups; y=role; bant genişliği=uçuş sayısı; hiçbir grup rol kesmiyor | `artifacts/four_dataset_probabilistic_v31/split_report.json`; `artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json` | JSON’dan yeni plot script’i | 12 |
| P1 | Beş veri seti outcome small multiples | x=false event/h; y=event recall; ayrı paneller ALFA/Attack/SEAD/Rfly; ADS-B recipe panel ayrı | Dört-set v2 event eval CSV/JSON; ADS-B contextual v2 summary | Mevcut `02_detection_summary.png` ve ADS-B heatmap’lerinden sunum-native yeniden düzenleme | 10–11 |
| P1 | Gözlenebilirlik örneği | Sol: etiketli Ping DoS ama residual düz; sağ: ADS-B low-speed bearing artefaktı; “etiket ≠ ölçülebilir sinyal” | UAV Attack score/source artefaktları; `docs/assets/adsb_simple_anomaly/05_route_examples.png` | Attack uygun timeline bulunursa yeni plot; yoksa yalnız ADS-B + metin | 7 |
| P1 | Alarm bütçesi düzlemi | x=false alarms/events per hour; y=recall; critical/advisory hedef bölgeleri; PX4/CUSUM/LSTM noktaları | `docs/PROJE_SUREC_VE_SONUC.md`; `artifacts/uav_gnss_integrity_v1/` | Rapor tablosundan scatter; yeni deney yok | 9 |
| P2 | Normal veri büyümesi trade-off | x=normal uçuş sayısı/aşama; sol y=false alarms/h, sağ y=recall; SEAD 23,6→9,95 ve 0,21→0,126 | `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md` | Yalnız iki doğrulanmış nokta varsa slope chart; ara noktalar uydurulmaz | 9/appendix |
| P2 | B0 family/domain heatmap | x=Motor, Environment, Prop, Voltage, Sensor; y=HIL/SIL/Real veya iki ayrı bar panel; renk=event recall | `docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`; B0 sonuç JSON/CSV | Mevcut CSV’den yeni plot | 19 |

## En çok ihtiyaç duyulan yeniden üretimler

1. **Trained vs random scatter:** Sunum promptunun istediği en önemli eksik kanıt görselidir. Özet korelasyon mevcut, fakat nokta bulutu mevcut PNG envanterinde yoktur. Ham skorlar persist edilmişse salt plot; değilse mevcut checkpoint ile validation scoring gerekir. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py`; `artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728/training_report.json`.
2. **Label/truth before-after:** 2.712 uçuşluk parser düzeltmesini tek bakışta anlatacak görsel yoktur. Gerçek bir etkilenen uçuş seçilmeden kavramsal çizim “schematic” diye etiketlenmelidir. Kaynak: `gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md`.
3. **Group split görseli:** Rol sayıları JSON’da var, fakat group crossing olmadığını görselleştiren grafik yoktur. Kaynak: `artifacts/four_dataset_probabilistic_v31/split_report.json`.
4. **NLL matematik görseli:** Negatif validation NLL’nin hata olmadığını açıklayan grafik yoktur. Kod formülünden veri gerektirmeden üretilebilir. Kaynak: `scripts/four_dataset_probabilistic_gpu_v1_runner.py`.

## Yeniden üretim kuralları

- Mevcut script varsa aynı script ve aynı artefact kullanılır; yeni threshold/model seçimi yapılmaz. Kaynak: `scripts/plot_four_dataset_probabilistic_v2_results.py`; `scripts/plot_four_dataset_probabilistic_v31_rflymad.py`; `scripts/plot_rflymad_probabilistic_v31_event_timelines.py`.
- Rfly final fault test açılmaz. Kaynak: `configs/four_dataset_probabilistic_v31_evaluation_contract.json`.
- `archive/` altından kod import edilmez veya güncel görsel üretilmez. Kaynak: `AGENTS.md`.
- Her grafikte veri rolü, metrik birimi ve truth türü başlık/altbaşlıkta yazılır. Kaynak: `docs/PROJE_YASAYAN_CALISMA_GUNLUGU.md`.
- Sıfır yanlış alarm, exposure ile birlikte gösterilir; ALFA v2 örneğinde yalnız 0,131 normal saat vardır. Kaynak: `docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`.
