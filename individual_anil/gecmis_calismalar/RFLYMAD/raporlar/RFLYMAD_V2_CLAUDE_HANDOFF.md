# RflyMAD-Full v2 — Claude çalışma devri ve tamamlama planı

> Son güncelleme: 2026-07-21, Europe/Istanbul  
> Amaç: Claude Code'un mevcut işi yeniden keşfetmeden, veri sızıntısı yaratmadan
> ve kısmi sonuçları nihai sonuç sanmadan devam edebilmesi.

## 1. Önce bunları oku

1. Kök dizindeki `AGENTS.md` kuralları geçerlidir.
2. `archive/` salt-okunur tarihçedir; buradan kod import edilmemeli ve eski hat
   yeniden aktifleştirilmemelidir.
3. Bu çalışma `rfly_full/` temiz namespace'inde yürütülmektedir.
4. Kullanıcının açık talebi olmadan commit/push yapılmamalıdır.
5. Çalışma ağacı hâlihazırda kirli ve çok sayıda yeni RflyMAD dosyası içeriyor;
   kullanıcı değişiklikleri silinmemeli veya resetlenmemelidir.

## 2. Yönetici özeti

RflyMAD için iki ayrı fakat birbirini tamamlayan model hattı kabul edilmiştir:

1. **Normal-only novelty detection:** Yalnız normal uçuşları öğrenir. Bilinmeyen
   arızalara genelleme ihtimalini ölçmek için kullanılır.
2. **Supervised temporal fault detection:** Normal ve etiketli arıza pencerelerini
   birlikte kullanır. Bilinen Motor, Propeller, Sensor, Load ve Voltage
   arızalarında ana adaydır.

Yalnız anomalilerle eğitip yalnız normallerle test etmek ana sözleşme değildir.
Bu ters one-class yaklaşım yalnız bilinen arıza imzası tanıma yan deneyi olabilir;
anomali test örneği olmadan recall, FN ve detection delay ölçülemez.

Mevcut bulgu: normal-only temporal convolutional AE, eski Dense AE'ye göre çok
daha güçlü development sonucu üretmiştir. Ancak önceki sweep yalnız 69 arızalı
uçuş içeriyordu ve Wind alarm yükü çok yüksekti. Tam veri değerlendirmesi ve
Real transfer kanıtı olmadan başarı ilan edilemez.

## 3. En güncel veri durumu

### İndirme ve manifest

- Resmî Kaggle ana/SIL/HIL kaynaklarının indirme kuyruğu tamamlandı.
- `artifacts/rfly_full/expansion_state.json`: `queue_complete`, başarısız batch yok.
- Güncel manifest: **6.605 canonical uçuş**.
- Exact ULog SHA-256 duplicate: **0**.
- Kilitli test: **1.225 uçuş**, toplamın yaklaşık `%18,55`i.
- Güncel özet: `artifacts/rfly_full/v2/dataset_manifest_summary.json`.
- Exact hash duplicate bulunmaması semantik veya near-duplicate olmadığı anlamına
  gelmez; scenario/oturum gruplaması korunmalıdır.

Manifest aile toplamları:

| Domain | NoFault | Motor | Sensor | Propeller | Load | Voltage | Wind/Environment |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real | 51 | 245 | 197 | 0 | 0 | 0 | 0 |
| HIL | 240 | 921 | 690 | 435 | 291 | 36 | 443 |
| SIL | 240 | 921 | 690 | 435 | 291 | 36 | 443 |

### 10 Hz v2 parse

- **6.605 / 6.605 uçuş parse edildi.**
- `artifacts/rfly_full/v2/parse_10hz_state.json`: `complete`, `remaining=0`,
  `failed={}` olmalıdır.
- Parser causal/backward merge kullanır; 1 Hz tarihsel parse üzerine yazmaz.
- Yüksek frekans IMU/motor özetleri mean, std, RMS, peak-to-peak ve
  first-difference RMS olarak üretilir.
- Windows atomik state replace sırasında kısa Windows kilidi görüldü;
  `rfly_full.pipeline._atomic_json` içine `PermissionError` retry eklendi.

### Truth kalitesi

Manifestte eski parse'a göre:

- 4.746 uçuş `provisional_testinfo_truth`,
- 1.839 uçuş `ok`,
- 20 uçuş `missing_fault_interval`.

V2 parser aktif aralıkta önce `rfly_ctrl_lxl`, bulunamazsa `TestInfo` fallback
kullanır. Tam parse sonrası gerçek v2 truth-source dağılımı ve
`truth_crosscheck_disagreement` sayısı ayrıca çıkarılmalıdır; eski manifest
özetindeki provisional sayıları nihai v2 truth sayısı gibi kullanma.

## 4. Doğru deney sözleşmesi

### 4.1 Normal-only temporal AE

Kod: `rfly_full/normal_ae.py`  
CLI: `scripts/run_rfly_full_v2_normal_ae.py`

Zorunlu kurallar:

- Scaler fit: yalnız `split=development` ve `evaluation_role=normal_reference`.
- Model train: yalnız development NoFault.
- Validation: her Real/HIL/SIL domain'inden bir tam `split_group_id`.
- Aynı uçuş/scenario/Real session train ve validation'a giremez.
- Domain dengesizliği `WeightedRandomSampler` ile azaltılır.
- Tek ortak model kullanılır; alarm eşikleri Real/HIL/SIL için ayrı kalibre edilir.
- Eşik seçimi yalnız held-out development normal uçuşlarıyla yapılır.
- Development arızaları eşik/model dondurulduktan sonra diagnostic evaluation'da
  kullanılır.
- Wind sistem arızası değildir; `environment_robustness` olarak ayrı raporlanır.
- Kilitli test özellikleri bu hatta okunmaz.
- Alarm kararı: 4-of-6 saniye, 30 saniye refractory.
- FA/saat; NoFault validation, arıza pre/post normal maruziyeti ve Wind için ayrı
  raporlanır.

Model: zaman eksenini 4 kat sıkıştıran temporal convolutional autoencoder;
eksiklik maskesi giriş kanalındadır, loss yalnız gözlenen değerlerde hesaplanır.

### 4.2 Supervised TCN

Kod: `rfly_full/supervised.py`  
CLI: `scripts/run_rfly_full_v2_supervised.py`

Zorunlu kurallar:

- Negatif train penceresi: yalnız NoFault ve hiç fault-active olmayan pencere.
- Pozitif train penceresi: tamamı fault-active olan sistem arızası penceresi.
- Transition, pre/post karışık pencereler ve Wind train loss'una girmez.
- Missingness ayrı giriş kanalıdır.
- İki head vardır: binary anomaly + conditional fault-family.
- Kalibrasyon development validation üzerinde yapılır.
- Satır/window, event ve flight metrikleri birbirine karıştırılmaz.
- SIL/HIL başarısı Real başarısı sayılmaz.

**Önemli bellek uyarısı:** Mevcut `_build_windows` bütün pencereleri bellekte
oluşturup daha sonra `_cap_balanced` uygular. 6.605 uçuşta doğrudan full TCN
çalıştırmak RAM taşmasına yol açabilir. Full TCN'yi başlatmadan önce pencere
üretimini streaming/lazy dataset, memmap veya üretim sırasında sınırlama ile
refactor et. Test evaluation da uçuş-bazlı streaming score üretmelidir.

### 4.3 Neden anomaly-only train + normal-only test değil?

- Arıza sınıfı kapalı ve tek biçimli değildir; bilinen fault pattern ezberlenir.
- Unseen fault kolayca normal kabul edilebilir.
- Testte pozitif yoksa TP/FN, recall ve detection delay hesaplanamaz.
- Yalnız false alarm/specificity ölçülür; ana fizibilite sorusu cevaplanmaz.
- İstenirse yalnız `known-fault signature retrieval` veya contrastive pretraining
  ablation'ı olarak, normal ve unseen-fault testleriyle ayrıca denenebilir.

## 5. Şimdiye kadar elde edilen sonuçlar

### 5.1 Tarihsel Dense AE baseline — başarısız

Kaynak: `artifacts/rfly_full/v2/dense_ae_diagnostics/summary.json`

- Sistem arızası event recall: `%2,79`.
- Pooled AUROC: `0,556`; AUPRC: `0,498`.
- Pre/post dahil tüm nonfault maruziyeti: `5,55 FA/saat`.
- Kritik 2 FA/saat bütçesinde feasible threshold bulunamadı.
- Ana model veya hibrit kapısı olarak kabul edilmemelidir.

### 5.2 Supervised TCN — yalnız kısmi smoke

Kaynak: `artifacts/rfly_full/v2/supervised_tcn/run_20260721_125530/`

- 128 uçuşluk kısmi 10 Hz havuz, 12 epoch.
- Kritik: `3/18`, recall `%16,67`, `6,37 FA/saat`.
- Advisory: `11/18`, recall `%61,11`, `15,93 FA/saat`.
- Her iki politika da ilgili FA kapısını geçmedi.
- Bu çıktı `smoke_only`; full-data performans iddiası değildir.

### 5.3 Normal-only Temporal AE — kısmi havuzda 5 rotasyon

Sweep: `artifacts/rfly_full/v2/normal_temporal_ae/sweep_20260721_130641/`

Bu sweep sırasında 334 normal train, 92 normal validation ve yalnız 69 development
arıza uçuşu mevcuttu.

| Politika | Ortalama recall | Recall std | Pre/post dahil FA/saat | FA std | Wind FA/saat |
|---|---:|---:|---:|---:|---:|
| Critical | %58,84 | %8,36 | 1,20 | 0,36 | 26,35 |
| Advisory | %71,59 | %6,92 | 10,24 | 1,09 | 27,79 |

Yorum:

- Development kapılarında umut vericidir.
- Wind alarm yükü kabul edilemez derecede yüksektir.
- Real arıza uçuşu yalnız 8 olduğu için Real transfer kanıtı değildir.
- Tam fault havuzu sonuçları bu değerlerden farklı olabilir.
- Görseller sweep klasöründe recall–FA, stability, family heatmap ve confusion
  matrix olarak üretilmiştir.

### 5.4 Tam parse sonrası otomatik normal-AE

Worker: `scripts/run_rfly_full_v2_postparse_normal_ae.py`  
State: `artifacts/rfly_full/v2/normal_temporal_ae/postparse_training_state.json`

Bu handoff yazılırken parse tamamlanmış ve worker tam development havuzunda
rotation-0 normal AE eğitimi/değerlendirmesini başlatmıştı. Önce state dosyasını
kontrol et:

Not: worker `run()` çağrısından hemen önce ayrıca `status=training` yazmıyor.
State `waiting_for_parser` + `parser_stop_reason=complete` gösterirken worker
Python süreci CPU kullanıyorsa eğitim/değerlendirme fiilen devam ediyor olabilir.

```powershell
Get-Content artifacts\rfly_full\v2\normal_temporal_ae\postparse_training_state.json
Get-Content rfly_postparse_normal_ae.out.log -Tail 20
Get-Content rfly_postparse_normal_ae.err.log -Tail 40
```

`status=complete` ise `model_output` yolundaki `summary.json`,
`operational_metrics.csv` ve `domain_family_metrics.csv` ilk incelenecek
çıktılardır.

## 6. Test körlüğü ve contamination notu

- Normal-only Temporal AE hattı kilitli test feature'larını okumadı.
- Ancak eski supervised TCN smoke koşusu, o an mevcut kilitli test altkümesinden
  **29 uçuşu** değerlendirdi.
- Dolayısıyla bütün 1.225 uçuş için koşulsuz “hiç görülmemiş pristine test” iddiası
  kurulamaz.
- `run_20260721_125530/per_flight_metrics.csv` içindeki 29 canonical ID exposed
  olarak kaydedilmeli.
- Final audit için bu 29 uçuşu dışlayan, daha önce model skoru üretilmemiş kilitli
  gruplardan immutable bir `final_audit_registry.json` oluşturulması önerilir.
- Bu registry model seçimi yapılmadan önce dondurulmalı; sonradan iyi sonuç için
  uçuş taşınmamalıdır.

## 7. Tamamlama planı

### Aşama A — Tam parse ve truth audit

Durum: parse tamamlandı; truth audit bekliyor.

Yapılacaklar:

1. `parse_10hz_state.json` için `complete`, 6.605 completed, 0 failed doğrula.
2. Tüm parquet şemalarının `V2_FEATURES` içerdiğini doğrula.
3. `truth_source`, `truth_crosscheck_disagreement`, eksik aktif aralık ve domain/aile
   dağılımlarını CSV + Markdown olarak çıkar.
4. Fault-active sürelerin negatif/taşmış olmadığını test et.
5. ULog exact hash yanında scenario/session near-duplicate audit yap.

Done kriteri: tüm 6.605 uçuş için parse/kalite tablosu ve açıklanmış istisnalar.

### Aşama B — Tam normal-only AE doğrulaması

1. Postparse rotation-0 koşusunun tamamlandığını doğrula.
2. Tam development havuzunda beş validation rotasyonunu yeniden çalıştır:

```powershell
.venv\Scripts\python.exe scripts\run_rfly_full_v2_normal_ae_sweep.py `
  --rotations 0 1 2 3 4 --epochs 25 --torch-threads 4
```

3. Yeni sweep'i görselleştir:

```powershell
.venv\Scripts\python.exe scripts\render_rfly_full_v2_normal_ae_sweep.py `
  artifacts\rfly_full\v2\normal_temporal_ae\sweep_YYYYMMDD_HHMMSS
```

4. Family/domain recall, FA/saat, confusion matrix, score distribution, training
   curve ve validation-rotation variance raporla.
5. Wind için ayrı robust threshold veya explicit rejection/abstention deneyi yap;
   Wind'i normal ya da sistem arızası diye sessizce yeniden etiketleme.

Done kriteri: beş rotasyonda gate kararlılığı, Real sonuçları ve Wind yükü açık.

### Aşama C — Supervised TCN bellek refactor

1. Window üretiminde cap'i üretim sırasında uygula.
2. Train için domain/family dengeli reservoir veya lazy sampler kullan.
3. Validation/test score'u uçuş bazında streaming üret.
4. Deterministik seed ve tekrar koşusunda aynı çıktı testi ekle.
5. Peak RAM ve runtime ölç; dry-run/smoke/full modları ayır.

Done kriteri: full manifestte OOM olmadan çalışan, aynı seed ile deterministik TCN.

### Aşama D — Full supervised deney matrisi

Zorunlu deneyler:

1. Beş grouped development fold'unda full TCN.
2. `Motor`, `Sensor`, `Propeller`, `Load`, `Voltage` leave-one-family-out.
3. Simulation-only train → Real test.
4. Real-only baseline.
5. Simulation pretrain → Real fine-tune; fine-tune yalnız Real development train
   gruplarında yapılmalı.
6. Domain-specific threshold ile ortak threshold karşılaştırması.

Mevcut `cv_fold=0` normal validation yalnız HIL içeriyordu. Full TCN kalibrasyonunda
Real/HIL/SIL normal exposure bulunan task-specific grouped validation registry
oluştur veya beş fold sonuçlarını domain bazında kalibre et. Kilitli test
registry'sini yeniden yazma.

Done kriteri: her protokol için event/flight/window metrikleri, domain/aile
breakdown ve grouped bootstrap güven aralığı.

### Aşama E — Model seçimi ve final audit

1. Yalnız development sonuçlarıyla model, eşik ve alarm policy seç.
2. Aşağıdaki kapıları önceden dondur:
   - Critical: recall `>= %30`, FA/saat `<= 2`.
   - Advisory: recall `>= %50`, FA/saat `<= 12`.
3. Wind için kabul kapısını final audit öncesi yazılı olarak belirle.
4. Exposed 29 TCN-smoke uçuşunu hariç tutan fresh final audit registry'yi dondur.
5. Final audit'i **tek sefer** çalıştır.
6. Sonuçları Real, HIL, SIL ve fault family bazında ayrı raporla.

Done kriteri: final audit tekrar tekrar eşik ayarlamak için kullanılmamış olmalı.

### Aşama F — Fizibilite kararı

Proje ancak şu koşullar birlikte sağlanırsa feasible denebilir:

- Gate sonuçları birden fazla grouped splitte kararlı.
- Real domain sonuçları hedefi karşılıyor.
- Wind/çevresel dayanıklılık kabul edilebilir.
- Unseen-family performansı yalnız bilinen aile recall'ından ibaret değil.
- FA/saat gerçek normal maruziyette ölçülmüş.
- Detection delay operasyonel olarak anlamlı.

Bunlardan biri sağlanmıyorsa sonuç “eşik ayarı eksik” diye yumuşatılmamalı;
hangi fiziksel/domain sınırlamasının fizibiliteyi bozduğu açık yazılmalıdır.

## 8. Öncelikli teknik borçlar ve riskler

1. Full TCN window materialization OOM riski.
2. Real normal sayısı yalnız 51; threshold güven aralığı geniş olabilir.
3. Domain clustering güçlü: 5-NN aynı-domain oranı `%94,84` bulunmuştu.
4. Wind yükü yüksek; model uçuş çevresini arıza sanıyor olabilir.
5. `gps_eph-gps_epv`, `output_std-output_range`, `act_thrust-output_mean` yüksek
   korelasyonlu; redundancy ablation yapılmalı.
6. 4.746 eski interval TestInfo provisional; v2 truth audit zorunlu.
7. TCN smoke 29 locked uçuşu expose etti; test körlüğü notu saklanmalı.
8. Full parse sonrasında üretilen yeni sonuçlar eski 69-fault sweep ile aynı tabloda
   “doğrudan kıyas” etiketi olmadan birleştirilmemeli.

## 9. Hızlı durum ve test komutları

Parse durumu:

```powershell
.venv\Scripts\python.exe -c "import json; s=json.load(open(r'artifacts/rfly_full/v2/parse_10hz_state.json')); print(len(s['completed']), s['remaining'], s['stop_reason'], len(s['failed']))"
```

Normal AE worker:

```powershell
Get-Content artifacts\rfly_full\v2\normal_temporal_ae\postparse_training_state.json
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,Path
```

İlgili testler:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests\test_rfly_full_contract.py `
  tests\test_rfly_full_dl_worker.py `
  tests\test_rfly_full_pipeline.py `
  tests\test_rfly_full_split_contract.py `
  tests\test_rfly_full_supervised.py `
  tests\test_rfly_full_v2_parser.py `
  tests\test_rfly_full_normal_ae.py -q
```

Son doğrulamada 15 test geçti; pytest cache yazma uyarısı test başarısızlığı
değildir.

## 10. İlgili dosya haritası

| Amaç | Yol |
|---|---|
| Dataset/split contract | `rfly_full/contract.py` |
| 10 Hz causal parser | `rfly_full/v2_parser.py` |
| Normal-only Temporal AE | `rfly_full/normal_ae.py` |
| Normal AE sweep | `scripts/run_rfly_full_v2_normal_ae_sweep.py` |
| Normal AE görselleri | `rfly_full/normal_ae_reporting.py` |
| Supervised TCN | `rfly_full/supervised.py` |
| Dense AE posthoc audit | `rfly_full/ae_diagnostics.py` |
| Genel EDA/görseller | `rfly_full/visualize.py` |
| Dataset manifest | `artifacts/rfly_full/v2/dataset_manifest.csv` |
| Split registry | `artifacts/rfly_full/v2/split_registry.json` |
| Parse state | `artifacts/rfly_full/v2/parse_10hz_state.json` |
| Kısmi AE sweep | `artifacts/rfly_full/v2/normal_temporal_ae/sweep_20260721_130641/` |
| TCN smoke | `artifacts/rfly_full/v2/supervised_tcn/run_20260721_125530/` |
| EDA görselleri | `artifacts/rfly_full/v2/visuals/` |

## 11. Claude için ilk çalışma sırası

1. Bu dosyayı ve `AGENTS.md`yi oku.
2. Postparse normal-AE worker sonucunu kontrol et; hata varsa yalnız kök nedeni
   düzelt ve aynı rotation'ı yeniden çalıştır.
3. Aşama A truth audit'ini üret.
4. Tam veride beş normal-AE rotasyonunu çalıştır ve raporla.
5. TCN full run başlatmadan önce bellek refactorunu yap ve peak RAM smoke testi
   göster.
6. Full supervised deney matrisine geç.
7. Final audit registry'yi development model seçimi bitmeden dondur; final testi
   erken çalıştırma.
