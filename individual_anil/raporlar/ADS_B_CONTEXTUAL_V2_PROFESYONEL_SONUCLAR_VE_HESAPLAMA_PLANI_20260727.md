# ADS-B contextual_physics_v2 — Profesyonel sonuç, görsel okuma ve hesaplama planı

Tarih: 2026-07-27
Durum: Faz C–G tamamlandı; raporlama paketi genişletildi

## 1. Yönetici özeti

`contextual_physics_v2`, yalnız doğal temiz ADS-B verisiyle eğitilen nedensel bir
residual forecaster, doğal kalibrasyondan türetilen conformal skorlar, yeni
`persistence_v2` birikimi ve değiştirilmeyen iki-eksenli Page-CUSUM bileşimidir.
Eğitim NVIDIA L4 üzerinde 8 epoch tamamlandı. Eğitim girdisi 275.468.643 fiziksel
satır ve epoch başına 236.129.563 pencereydi. Magnitude-domination kapısı
`false` ile geçti.

Truth-v2 değerlendirmesinde 26.802.690 satır denetlendi. Çelişkili `on_ground`
kaydı olan 58 uçuş bütün paired dosyalardan karantinaya alındı ve 8.852 paired
uçuş korundu. Sonuç tek bir evrensel dedektör için operasyonel GO üretmedi;
ancak speed-bias persistence ve position-ramp CUSUM kollarında önceki turlardan
belirgin derecede güçlü, recipe-bazlı araştırma kanıtı üretti.

## 2. “İki anomaly” meselesi: anomaly ile detector aynı şey değildir

Bu çalışmada yalnız iki anomaly yoktur. Truth-v2 corpusunda dört fizik anomaly
mekanizması değerlendirme kapsamındadır:

| Truth-v2 recipe | Fizik anlamı | Birincil kanal / mekanizma |
|---|---|---|
| `vertical_rate_frozen` | Dikey hızın donması | vertical-rate model + persistence |
| `ground_speed_biased` | Yer hızına kalıcı bias eklenmesi | speed model + persistence |
| `track_frozen` | Uçuş yönünün donması | heading persistence |
| `position_ramp_stealthy` | Konumun yavaş, birikimli kaydırılması | east/north Page-CUSUM |

`altitude_dropout` beşinci sentetik recipe’dir; fakat ön-kayıtta S2 veri-kalitesi
katmanına ait olduğu için contextual model/CUSUM fizik değerlendirmesinin dışında
tutulmuştur.

Bunlardan ayrı olarak üç detector davranışı vardır:

- **Instant model alarmı:** conformal `p` değeri dondurulmuş `alpha` değerinin altına iner.
- **Persistence_v2 alarmı:** küçük conformal p-değerlerinin uçuş içinde biriken durumu dondurulmuş `h` değerini aşar.
- **CUSUM alarmı:** east/north velocity residual kanıtı Page-CUSUM içinde birikerek dondurulmuş `h` değerini aşar.

Yeni tek-uçuş alarm dashboardlarında iki farklı anomaly mekanizması özellikle yan
yana gösterilmiştir: `ground_speed_biased` ve `position_ramp_stealthy`. Bu seçim
“toplam iki anomaly var” anlamına gelmez; persistence ve CUSUM alarm anatomisini
aynı uçuş kimliği üzerinde okunabilir kılmak içindir.

## 3. V=5 tam olarak ne demektir?

`V`, modelin confidence skoru veya yüzde değildir. Sistem seviyesinde doğal alarm
bütçesidir:

`V episode / 100 scoreable flight-hours`

Dolayısıyla `V=5`, bütün sistem için hedeflenen toplam yükün:

`5 / 100 = 0,05 episode/saat`, yani ortalama **20 skorlanabilir uçuş-saatinde bir episode**

olması demektir. Bu toplam bütçe sonuçlardan önce dondurulan kanal ve profil
paylarıyla bölünür:

| V=5 kolu | Toplam bütçe payı | Kalibrasyon hedefi | Yaklaşık karşılığı |
|---|---:|---:|---:|
| Speed bias persistence | `0,277 × 0,5` | 0,006925 ep/saat | 144 uçuş-saatinde 1 episode |
| Vertical freeze persistence | `0,115 × 0,5` | 0,002875 ep/saat | 348 uçuş-saatinde 1 episode |
| Heading persistence | `0,228` | 0,0114 ep/saat | 88 uçuş-saatinde 1 episode |
| East+north CUSUM | `0,115 + 0,115` | 0,0115 ep/saat | 87 uçuş-saatinde 1 episode |

Bu değerler doğal **kalibrasyon hedefleridir**. Truth-v2 paired-clean uçuşlarda
gerçekleşen yük birebir aynı olmak zorunda değildir. Örneğin V=5’te speed-bias
persistence için gözlenen paired-clean yük 0,00087 ep/saat, position CUSUM için
0,01723 ep/saat çıkmıştır. Bu fark eşik değiştirme gerekçesi yapılmamış, dağılım
transferinin dürüst sonucu olarak korunmuştur.

## 4. Tam-corpus sonuçları

| Recipe / profil | V=0,1 recall / yük | V=5 recall / yük | V=50 recall / yük | V=500 recall / yük |
|---|---:|---:|---:|---:|
| Vertical spike | %0,00 / 0,0000 | %0,00 / 0,0000 | %3,89 / 0,0179 | %12,02 / 0,2193 |
| Vertical persistence | %26,26 / 0,0041 | %26,34 / 0,0038 | %40,81 / 0,0217 | %58,48 / 0,1810 |
| Speed spike | %0,00 / 0,0005 | %12,25 / 0,0265 | %17,41 / 0,0695 | %49,07 / 0,6558 |
| Speed-bias persistence | **%57,18 / 0,0009** | **%57,18 / 0,0009** | %79,66 / 0,0544 | %89,98 / 0,5260 |
| Track persistence | %11,58 / 0,0000 | %14,82 / 0,0001 | **%62,46 / 0,0985** | %77,61 / 0,6820 |
| Position Page-CUSUM | %9,35 / 0,0003 | **%51,27 / 0,0172** | %65,73 / 0,0817 | %85,61 / 0,6361 |

Recall yalnız observable-eligible sentetik eventler üzerindedir. “Yük” paired-clean
uçuşlardaki doğal alarm episode/saat değeridir. Satır, event ve uçuş metrikleri
birbirinin yerine kullanılmamıştır.

## 5. Tek uçuşta alarm noktalarının okunması

Yeni dashboardlar dört panel içerir:

1. Clean ve injected fizik residual’ı,
2. conformal p-değeri, dondurulmuş alpha ve instant alarm noktaları,
3. persistence_v2 durumu, dondurulmuş h ve persistence alarm noktaları,
4. CUSUM skoru, dondurulmuş h ve CUSUM alarm noktaları.

Turuncu bant observable anomaly aralığıdır. Noktalı çizgi attack onset, kesikli
çizgi observable onset, kırmızı semboller alarm üreten gerçek timestamp/window’lardır.

İki ayrı örnekleme politikası özellikle korunmuştur:

- **Deterministik ilk uçuş (`000001_003`):** detector performansına bakılmadan
  seçilmiştir. V=5’te ana speed persistence ve position CUSUM alarm üretmemiştir.
  Bu, full-corpus recall’ünün %100 olmamasının dürüst tek-uçuş örneğidir.
- **Detected-example (`0081ef_002`):** V=5’te observable onset sonrasında alarm
  veren ilk leksikografik uçuş, yalnız alarm anatomisini göstermek için seçilmiştir.
  Performans tahmini olarak kullanılmaz; performans için yukarıdaki tam-corpus
  tablo kullanılır.

Detected-example üzerinde:

| Anomaly / detector | Alarm window | Observable onset sonrası ilk alarm |
|---|---:|---:|
| Speed bias / persistence_v2 | 121 | 2.322,7 s (38,7 dk) |
| Position ramp / CUSUM | 55 | 2.559,7 s (42,7 dk) |

Bu grafikler yalnız “yakalandı” demekle yetinmeyip önemli sınırlamayı da görünür
kılar: seçilen örnekte alarm vardır, fakat gecikme uzundur.

## 6. Önceki ADS-B çalışmalarından bu tura teknik ilerleme

| Dönem | Ana bulgu | Sonraki tasarıma etkisi |
|---|---|---|
| ADR-025–032 | Kural baseline’ı bazı NN denemelerini geçti; AE/USAD ve ana freeze kapıları NO-GO oldu | Reconstruction yaklaşımına geri dönülmedi |
| ADR-029 | İki-eksenli causal Page-CUSUM çekirdeği kuruldu | Position ramp için uzun kanıt birikimi mümkün oldu |
| ADR-035–041 | Contextual v1 magnitude kapısını geçti; doğal burden ve performans darboğazları ölçüldü | Modelin çalışması ile alarm kararının başarısı ayrıştırıldı |
| ADR-042 | Dar V=0,1–5 ızgarasında çoğu model profili düşük recall verdi; CUSUM V=5’te %49,72’ye ulaştı | Sonuç görülmeden V=0,1–500 geniş ızgarası ve persistence_v2 ön-kaydedildi |
| ADR-046 / v2 | 275,5M doğal satır, 8 epoch L4 eğitim, magnitude PASS; full truth-v2 ve üç detector mekanizması | Recipe-bazlı güçlü kollar bulundu; evrensel operasyonel GO yine verilmedi |

Profesyonel sonuç “model başarılı/başarısız” şeklinde tek cümle değildir:

- Speed-bias persistence düşük bütçede güçlüdür.
- Position ramp için CUSUM ana değer üretmeye devam etmektedir.
- Track yüksek bütçelerde anlamlı recall üretmektedir.
- Vertical instant spike kolu zayıftır.
- Bazı yakalamalarda alarm gecikmesi operasyonel kullanım için uzundur.

## 7. ADS-B ile dört yeni probabilistic dataset koşusunun yükü

Dört notebook ortak, dondurulmuş bir temporal baseline çalıştırır: history=32,
LSTM hidden=64, 30 epoch, batch=1024, yalnız normal train ve validation-only
0,995 quantile alarm eşiği.

| Dataset | Train kaynak | Train satırı | Epoch başına pencere | Epoch | Window-epoch | Test penceresi |
|---|---:|---:|---:|---:|---:|---:|
| ADS-B contextual v2 | 149.462 fit uçuş | 275.468.643 fiziksel | 236.129.563 | 8 | **1.889.036.504** | 26.802.690 truth-v2 satırı |
| ALFA | 11 uçuş | 11.681 | 11.296 | 30 | 338.880 | 24.342 |
| UAV-Attack | 4 uçuş | 17.882 | 17.754 | 30 | 532.620 | 56.282 |
| UAV-SEAD | 23 uçuş | 34.290 | 33.554 | 30 | 1.006.620 | 1.191.954 |
| RflyMAD | 221 uçuş | 171.743 | 164.671 | 30 | 4.940.130 | 1.300.807 |

ADS-B optimizer exposure’ı dört dataset toplamının yaklaşık **277 katıdır**.
Dört-dataset transfer paketinin tamamı 986,04 MiB iken ADS-B eğitim Parquet
girdisi yaklaşık 17,67 GiB’dir. Bu nedenle ADS-B’de L4 training gerçek bir GPU
işidir; küçük dört-dataset modellerinde Parquet/ZIP/Drive I/O ve kaynak başına
pencere hazırlama toplam sürenin daha büyük bölümünü oluşturacaktır.

## 8. Süre ve GPU planı

| İş | Donanım | Süre | Kanıt türü |
|---|---|---:|---|
| ADS-B final raporlanan eğitim oturumu | NVIDIA L4 | 42,3 dk | `training_report.json::elapsed_seconds_this_session` |
| ADS-B full truth-v2 | CPU | 129,4 dk | Bu oturumda ölçülen wall-clock |
| ALFA 30 epoch + değerlendirme | L4 | 4–8 dk | Planlama tahmini |
| UAV-Attack 30 epoch + değerlendirme | L4 | 4–8 dk | Planlama tahmini |
| UAV-SEAD 30 epoch + değerlendirme | L4 | 8–20 dk | Planlama tahmini |
| RflyMAD 30 epoch + değerlendirme | L4 | 12–30 dk | Planlama tahmini |

Tahminler tam GPU benchmark değildir. Tam satır/pencere envanteri, model boyutu,
30 epoch, 67,3 saniyelik yerel inventory/window geçişi ve Drive hash/extract/
checkpoint payı kullanılarak konservatif verilmiştir. İlk tamamlanan epoch’un
`elapsed_seconds` değeri geldikten sonra otoritatif ETA şu şekilde güncellenir:

`ETA ≈ ilk epoch süresi × kalan epoch + final validation/random-init/test payı`

ADS-B’deki 42,3 dakika alanı oturum kapsamlıdır; önceki resume oturumları varsa
kümülatif toplamı temsil etmez. Bu nedenle dört-dataset tahminleri için tek başına
doğrusal throughput benchmarkı olarak kullanılmamıştır.

GPU önerisi:

- **ADS-B:** L4 doğrulanmış ve yeterlidir. A100 yalnız wall-clock çok kritikse düşünülür.
- **ALFA / UAV-Attack:** T4 yeterlidir; dört koşuda aynı ortamı korumak için L4 tercih edilir.
- **UAV-SEAD / RflyMAD:** L4 önerilir.
- **A100:** Bu küçük hidden=64 tek-katman LSTM’lerde genellikle ekonomik değildir;
  RflyMAD’in 1.551 küçük dosyalı I/O darboğazını çözmez.

Dört koşu tek L4 üzerinde sırayla yaklaşık 28–66 dakika planlanabilir. Dört ayrı
GPU runtime gerçekten erişilebiliyorsa teorik alt sınır en uzun koşu olan 12–30
dakikadır; hesap kotası, eşzamanlı runtime izni ve upload süresi bu tahmine dahil değildir.

## 9. Dört notebook ile şimdi ne yapılacak?

1. Drive’a `artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer/`
   içindeki yedi dosya bir kez yüklenir; ZIP’ler açılmaz.
2. Her notebook ayrı açılır:
   - `notebooks/alfa_probabilistic_gpu_v1_colab.ipynb`
   - `notebooks/uav_attack_probabilistic_gpu_v1_colab.ipynb`
   - `notebooks/uav_sead_probabilistic_gpu_v1_colab.ipynb`
   - `notebooks/rflymad_probabilistic_gpu_v1_colab.ipynb`
3. L4 runtime seçilir. Notebook yukarıdan aşağı çalıştırılır; SHA/byte doğrulaması
   geçmeden ZIP açılmaz ve CUDA yoksa CPU fallback yapılmaz.
4. Aynı `RUN_DIR` yeniden kullanılır. Her tamamlanan epoch Drive checkpoint’ine
   atomik yazıldığı için kesinti sonrası son tamamlanan epoch’tan devam edilir.
5. Epoch 1 tamamlanınca `training_history.json` içindeki `elapsed_seconds` ile ETA
   yenilenir. Frozen 30 epoch sonucu görerek değiştirilmez.
6. `training_report.json` oluşunca önce
   `magnitude_domination_flagged_at_0_8` kontrol edilir. True sonuç saklanır ve
   aynı v1 içinde feature/epoch/clip/eşik değiştirilmez.
7. Her run dizininden en az şu dosyalar geri alınır:
   `training_report.json`, `validation_summary.json`, `flight_metrics.csv`,
   `training_history.json`, `model_state.pt`, `training_epoch_checkpoint.pt`.
8. Dört sonuç geldikten sonra ortak raporlama grafikleri üretilir: train/validation
   NLL, label bazlı ROC-AUC/AP, normal alarm-window/saat, magnitude korelasyonları
   ve dataset kapsam sınırlamaları.

ALFA ve UAV-Attack development-only ve veri-tavanlıdır. UAV-SEAD blind holdout’u,
RflyMAD locked-test’i bu notebooklar tarafından açılmaz. Bu notebookların
alarm-window/saat metriği ADS-B’nin debounced event episode/saat metriğiyle doğrudan
yarıştırılmaz.

## 10. Görsel paket

- `artifacts/adsb/plots/contextual_v2_detected_examples/`: Alarm veren iki uçuş-anatomisi örneği.
- `artifacts/adsb/plots/contextual_v2_flight_alarms/`: Detector-seçimsiz deterministik iki örnek.
- `artifacts/adsb/plots/contextual_v2_timelines/`: Beş recipe için clean/injected skor timeline’ları.
- `artifacts/adsb/plots/contextual_v2_evaluation/`: Recall, natural burden ve event TP/FN matrisleri.
- `artifacts/adsb/plots/compute_load_comparison/adsb_vs_four_dataset_compute_load.png`:
  ADS-B ve dört dataset için optimizer exposure, veri boyutu, runtime ve GPU planı.
- `artifacts/adsb/plots/reporting_summary/adsb_research_progression.png`:
  Önceki ADS-B kapılarından contextual-v2 sonucuna araştırma evrimi.
- `artifacts/adsb/plots/reporting_summary/anomaly_detector_map_V5.png`:
  V=5 için anomaly mekanizması, atanmış detector ve event recall haritası.

Makine-okunur ana kanıtlar değiştirilmeden korunur: Faz E
`truth_v2_eval_report.json`, calibration report, training report ve dört-dataset
frozen config/transfer manifestleri.
