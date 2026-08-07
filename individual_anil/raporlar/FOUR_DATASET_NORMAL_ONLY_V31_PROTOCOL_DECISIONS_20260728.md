# Four-dataset normal-only v3.1 protokol kararları

Tarih: 2026-07-28
Statü: Uygulandı; yeni eğitim başlatılmadı, final fault test açılmadı.
Kapsam: RflyMAD, UAV-SEAD, ALFA ve UAV Attack için grup-bağımsız normal-only
eğitim ve değerlendirme disiplini.

## Neden v3.1 gerekti?

Normal-only optimizer bu problem için korunmuştur: model ağırlıkları yalnız normal
uçuş davranışından öğrenilecektir. Ancak normal-only eğitim tek başına yeterli
değildir; aynı session/campaign ailesinin farklı rollere bölünmemesi ve fault
testinin iteratif model seçimine dönüşmemesi de gerekir.

Mevcut v3 manifesti değiştirilmemiştir. RflyMAD v3 testindeki 91 normal ve 1.110
fault kaynak, üyelik değiştirilmeden v3.1'de aşağıdaki rollere ayrılmıştır:

| Rol | Grup | Kaynak | Kullanım |
|---|---:|---:|---|
| train | 7 | 260 normal | optimizer ve train-only preprocessing |
| val | 3 | 90 normal | normal NLL checkpoint seçimi ve false-alarm kalibrasyonu |
| normal_test | 3 | 91 normal | false-alarm değerlendirmesi; eşik seçmez |
| anomaly_dev | 12 | 557 fault | model/event kuralı karşılaştırması; optimizer görmez |
| final_fault_test | 13 | 553 fault | tek seferlik nihai değerlendirmeye kadar kapalı |

Fault grupları domain ve fault-family tabakaları içinde, kaynak sayısını yaklaşık
dengeleyen deterministik hash kuralıyla ayrılmıştır. Bölme satır veya pencere
düzeyinde değildir. `anomaly_dev` ve `final_fault_test` arasında source/group
örtüşmesi sıfırdır. Final rol ayrıca içerik-listesi SHA-256 mührüyle sabitlenmiştir.
Bu protokol **normal-only optimizer + semi-supervised model selection** olarak
raporlanacaktır; yalnız “unsupervised” denmeyecektir.

Kanonik dosyalar:

- `configs/four_dataset_probabilistic_v31_split_manifest.json`
- `configs/four_dataset_probabilistic_v31_evaluation_contract.json`
- `artifacts/four_dataset_probabilistic_v31/split_report.json`
- `scripts/build_four_dataset_probabilistic_v31_protocol.py`

## RflyMAD normal-train kalite ve kapsama denetimi

Denetim yalnız 260 train kaynağını okumuş, validation/test/fault telemetrisi
okumamıştır. Sonuç:

- 260 kaynak, 7 bağımsız grup, SIL/HIL/Real dağılımı 120/120/20;
- 213.567 satır ve 205.247 kullanılabilir causal pencere;
- kaynak başına pencere min/medyan/p95/maksimum: 492/772,5/1.040/1.354;
- en uzun kaynak bütün train pencerelerinin %0,66'sını, en büyük grup %20,19'unu
  üretmektedir;
- source kimliği, `NoFault` etiketi, `quality_status=ok`, gerçek fault aralığı ve
  aktif fault/condition kontrolleri geçmiştir;
- `battery_voltage` ve `battery_current` train içinde tamamen boştur. Bunlar veri
  varmış gibi yorumlanmayacak ve train-only feature-availability kontrolünde
  dışlanacaktır;
- 240 simülasyon kaydında boş `fault_id=[]` ile birlikte 0--uçuş süresi biçiminde
  `planned_fault_*` şablon alanı vardır. Gerçek `fault_start_s/fault_end_s` yoktur;
  satır düzeyinde `fault_active` ve `condition_active` daima false'dur. Bu durum
  kontaminasyon diye gizlenmemiş, şema tutarsızlığı olarak kayda alınmıştır.

Uzun uçuş ve büyük scenario gruplarının optimizerı domine etmesini önlemek için
v3.1 runner her epoch toplam pencere bütçesini önce gruplara, sonra grup içindeki
kaynaklara eşit kota olarak dağıtır; kısa kaynaklarda deterministik replacement
sampling uygulanır ve miktarı training history içinde raporlanır.
Scaler, imputer, feature availability/degeneracy seçimi, residual katsayıları ve
phase-normalizasyon yalnız train rolünde fit edilebilir.

Kanonik denetim:

- `scripts/audit_four_dataset_probabilistic_v31_normal_train.py`
- `artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json`

## UAV-SEAD session-key denetimi

Mevcut strict-parent v3 benchmark korunmuştur ve ana eğitim hâlâ beklemededir.
1244 seçili ULog'un 11.378.112.716 byte'lık envanterinde yalnız header/parameter
metadata okunmuştur; sinyal gövdesi decode edilmemiştir. Bütün kaynaklarda UUID,
donanım, firmware, OS ve parameter metadata bulunmuştur.

Gözlenen anahtar davranışı:

| Aday anahtar | Anahtar sayısı | Tekil anahtar | Yorum |
|---|---:|---:|---|
| system UUID | 5 | 0 | Çok kaba; en büyük araç 993 log ve birden çok label/parent içeriyor |
| UUID + firmware | 10 | 1 | Hâlâ çok kaba; label ve parent sınırlarını geçiyor |
| UUID + firmware + tüm parameter hash | 1.239 | 1.234 | Aşırı ince; pratikte source kimliğine dönüşüyor |
| UUID + firmware + kimlik parametreleri | 13 | 1 | Yine çok kaba |

ULog başlangıç timestamp'i boot-relative olduğundan duvar-saati session anahtarı
değildir. Metadata denetimi, strict parent'ı güvenle inceltecek mission/waypoint
kanıtı üretmemiştir. Bu nedenle 510 mixed-parent normal kaydı eğitime geri
alınmamış, sonuç **`strict_parent_retained_no_defensible_refined_key_yet`** olarak
dondurulmuştur. Daha ince bir SEAD registry ancak mission/waypoint veya bağımsız
temporal-session kanıtı incelenip yeni immutable registry/split üretilirse geçerli
olacaktır.

Kanonik denetim:

- `scripts/audit_uav_sead_session_keys_v1.py`
- `artifacts/four_dataset_probabilistic_v31/uav_sead_session_metadata_v1.parquet`
- `artifacts/four_dataset_probabilistic_v31/uav_sead_session_key_audit_v1.json`

## Dört veri seti için uygulanan karar

| Veri seti | Uygulanan karar |
|---|---|
| RflyMAD | v3.1 baseline eğitimine yetkili; anomaly-dev ve final fault test ayrıldı |
| UAV-SEAD | strict-parent benchmark korundu; session anahtarı kanıtı yetersiz olduğu için ana eğitim beklemede |
| ALFA | fold içi preprocessing ve threshold ile 4-fold group-safe CV smoke |
| UAV Attack | ana training benchmark değil; yalnız external stress/exploratory |

RflyMAD'de önce v2 ile aynı feature/model/hiperparametre ailesiyle yalnız split
etkisi ölçülecektir. Ardından phase context ve raw/residual ablation'ları yalnız
`anomaly_dev` üzerinden karşılaştırılabilir. `final_fault_test` bu aşamalarda
erişilemez.

## Raporlama sözleşmesi

Normal tarafta ana metrikler false event/saat, en az bir yanlış alarm üreten uçuş
oranı, false event/uçuş, alarm time fraction ve grup bazında eşik kararlılığıdır.
Fault tarafında event/flight recall, ilk tespit gecikmesi, fault-family ve varsa
severity/magnitude kırılımı, interval/flight AUC ve PR-AUC raporlanır. Accuracy,
precision ve F1 fault-enriched corpus nedeniyle ana sonuç değildir.

Güven aralıkları source/group bootstrap ile üretilecektir; aynı uçuşun satır veya
pencereleri bağımsız örnek sayılmayacaktır. Final sonuç görüldükten sonra split,
eşik, event kuralı, grid, epoch veya feature kararı değiştirilemez.

## Bir sonraki yürütme kapısı

V3.1 contract-aware runner ve ALFA 4-fold smoke tamamlanmıştır. ALFA'nın teknik
pipeline smoke sonucu PASS, bilimsel baseline sonucu magnitude gate nedeniyle
NO-GO'dur; ayrıntı `FOUR_DATASET_PROBABILISTIC_V31_ALFA_CV_SMOKE_20260728.md`
dosyasındadır. RflyMAD için checksum-pinned Colab notebook ve transfer sözleşmesi
hazırdır. Sonraki kapı RflyMAD L4 eğitimidir; kendi magnitude diagnostic sonucu
false olmadan phase/raw/event ablation aşamasına geçilemez.
