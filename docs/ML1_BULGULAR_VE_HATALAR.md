# ML-1 Bulgular ve Hatalar Raporu (2026-07-02)

Kaynak: `notebooks/02_isolation_forest_cusum_egitim.ipynb` (çalıştırılmış çıktılar) +
`notebooks/01_veri_ve_feature_incelemesi.ipynb`. Bu doküman ML-2/ML-3 tasarımının ve
metot araştırmasının girdisidir — her madde "neyi denedik, ne çıktı, neden, ne yapılacak" formatındadır.

## H1 — Monolitik satır-bazlı Isolation Forest başarısız

**Gözlem:** Tüm feature setiyle satır bazlı IF: ALFA satır-ROC **0.497±0.026** (yazı-tura),
UAV Attack **0.209±0.138** (terse dönmüş — anomaliler normalden DÜŞÜK skorluyor).

**Üç kök neden:**
1. **Etiket semantiği:** Saldırı/arıza logunun *tüm* satırları anomali etiketli, ama gerçek imza birkaç
   satırda yaşıyor (78 km GPS sıçraması ~2-3 satır). Satır-bazlı ROC, imza taşımayan binlerce satırın
   yakalanmasını bekliyor — yapısal olarak adaletsiz hedef.
2. **Feature sulanması:** IF bölme feature'ını rastgele seçer; 58-85 boyutun ~3-4'ü bilgilendiriciyken
   ağaçların çoğu gürültü boyutlarında bölünüyor.
3. **NaN-impute etkisi:** Eksik değerler train medyanıyla dolduruluyor → attitude'u tamamen eksik
   ping_dos satırları "tam ortalama" görünüp normalden bile derin izole oluyor. UAV'de ROC'un 0.5
   *altına* düşmesinin nedeni bu.

**Çözüm yönü:** Uçuş-düzeyi değerlendirme + modüler dar-feature dedektörler (uygulandı, H2);
imputasyon yerine maske-farkındalıklı modeller (ML-2 araştırma konusu).

## H2 — Modüler füzyon çalışıyor ama kararsız

**Gözlem:** ALFA uçuş-ROC **0.833±0.172** (rehberlik modülü tek başına 0.864±0.081);
UAV füzyon **0.600±0.212**. Ama `normal_yanlis_alarm@1` seed'ler arası 0.0↔1.0 salınıyor.

**Neden:** τ_m = val uçuşlarının maksimum skoru; val'de yalnızca 1-2 (ALFA 2, UAV 1) normal uçuş var.
Tek uçuşluk val ile eşik tamamen o uçuşun karakterine bağlı — varyansın kaynağı bu.

**Çözüm yönü:** LOFO (10/6/20 fold, manifest'te hazır) ile eşik dağılımı çıkarmak;
val skorlarına parametrik kuyruk (EVT/POT — Peaks Over Threshold) oturtmak (araştırma maddesi R4).

## H3 — Ping DoS 4/6 logda tespit edilemiyor

**Gözlem:** ping_dos tespiti 0.37. 6 DoS logundan yalnızca 2'sinde attitude tamamen eksik
(veri-kalitesi imzası); kalan 4'ünde konum/attitude/GPS-sağlık kolonlarında ayırt edici HİÇBİR imza yok.

**Neden:** DoS ağ-katmanı saldırısı; uçuş fiziğine yansımamışsa mevcut 4 uORB topic'inden görünmez.
Ham .ulg loglarındaki zamanlama/paket metadata'sı (mesaj arrival aralıkları, drop sayaçları) Silver'a taşınmıyor.

**Çözüm yönü:** (a) Kapsam beyanına "network-layer DoS bu telemetri kesitiyle kısmen tespit edilir" maddesi;
(b) Silver'a mesaj-aralığı (inter-arrival) istatistikleri eklemek — `timestamp` diff'lerinin kendisi feature olabilir.

## H4 — UAV sinyal-kalitesi modülü ROC < 0.5

**Gözlem:** sinyal_kalitesi modülü uçuş-ROC 0.292±0.034 — sistematik ters.

**Neden (hipotez, doğrulanacak):** SITL loglarında GPS sağlık alanları sabit-sıfır (simülatör doldurmuyor);
canlı (live) loglarda gerçek gürültü değerleri var. Test setindeki canlı-normal uçuş, SITL-saldırı
loglarından daha "anormal" skorluyor → live-vs-SITL domain karışması. Jamming'i yine de %100 yakalıyor
çünkü jamming değerleri her şeyin dışında.

**Çözüm yönü:** SITL/live'ı ayrı kalibre etmek (platform bazlı normal profil — FableChat'in araç-başına
kalibrasyon tezi) veya sabit-sıfır kolonları uçuş bazında maskelemek.

## H5 — SEAD'e kalibrasyonsuz transfer tamamen başarısız

**Gözlem:** UAV Attack'ta eğitilen modüller + eşikler SEAD'de: normal yanlış alarm **1.0**
(her normal uçuş alarm), uçuş-ROC 0.375.

**Neden:** Farklı platform, farklı sensör gürültü tabanı, farklı uçuş rejimi → tüm uçuşlar val eşiğini aşıyor.
Bu, FableChat'in öngördüğü sonuç: "tek global eşik gerçekçi değil; yeni araçta normal uçuşla kalibrasyon şart".

**Çözüm yönü:** SEAD split'leri manifest'te hazır — SEAD normal'leriyle τ yeniden kalibre edilip
transfer yeniden ölçülecek (feature semantiği transferi vs eşik transferi ayrımı raporlanacak).

## H6 — ALFA rudder/aileron_rudder zayıf

**Gözlem:** Tespit oranları: engine 0.86, elevator 0.80, aileron 0.77, **rudder 0.33, aileron_rudder 0.20**.

**Neden (hipotez):** Rudder arızası sabit kanatta önce yaw/sideslip'e yansır; yaw_error feature'ı var ama
sideslip verisi yok; ayrıca rudder senaryoları kısa (~290 satır) — rolling pencereler ısınamadan bitiyor.
aileron_rudder tek uçuş — istatistik anlamsız (n=1).

**Çözüm yönü:** ML-2'de sequence modeli (LSTM-AE) yaw dinamiğini zamansal bağlamda öğrenebilir;
kısa pencere (2 sn) varyantı denenmeli.

## H7 — PR-AUC yanıltıcı raporlanıyordu

**Gözlem:** Satır PR-AUC 0.94-0.97 "iyi" görünüyor ama test satırlarının ~%90'ı anomali etiketli —
şans çizgisi (prevalence) zaten ~0.9.

**Kural:** PR-AUC her zaman prevalence taban çizgisiyle birlikte raporlanacak; asıl eksen uçuş-ROC.

## H8 — Veri/altyapı tuzakları (tekrarlanmasın)

1. **Part çoğalması:** `write_silver` her koşuda yeni immutable part ekler; `read_layer` HEPSİNİ okur →
   Silver'ı iki kez çalıştırıp Gold üretince satırlar katlanır. Yeniden üretim öncesi
   `data/objectstore/silver/<kaynak>` temizlenmeli. (Kullanıcının gördüğü "20k→200k" bunun + eski
   referans scriptin bileşimiydi.)
2. **`battery_power_w` %96 NaN:** `current_a` bu datasette çoğunlukla -1 sentinel. Feature listede
   kalabilir (impute güvenli) ama batarya modülü kurulacaksa önce veri kalitesi analizi şart.
3. **`velocity_mps` kök nedeni** (çözüldü ama ders): kolon adı varsayımı (`measured`) gerçek veriyle
   (`meas_x`) doğrulanmadan yazılmıştı. Kural: her yeni parser alanı gerçek dosyaya karşı doğrulanır.
4. **UAV-SEAD iç mekân uçuşları GPS taşımaz** — `vehicle_local_position` fallback'i olmadan alt kümenin
   %80'i sessizce atlanıyordu. Kural: parse coverage'ı (kaç uçuş atlandı) her zaman loglanır ve raporlanır.

## Doğrulanan tezler (pozitif bulgular)

- **Otopilot residual'ı + CUSUM = en güçlü tek dedektör:** `alt_error_cusum_pos` uçuş-ROC **0.878**.
- **Analytical redundancy gizli saldırıyı yakalıyor:** `gps_speed_residual` live/hackrf spoofing'i
  (GPS sıçraması OLMADAN) ayırıyor: 5.9 vs normal ~2.
- **Detection time rekabetçi:** rehberlik modülü AvgDT **3.7 s**, MaxDT **68.1 s** (34/36 uçuşta alarm).
- Modüler mimari `predicted_category` (baskın modül) ile ücretsiz açıklanabilirlik veriyor.

## ML-2 sonuçları (2026-07-02, `notebooks/03_autoencoder_lstm_ae_egitim.ipynb`)

Uçuş-ROC (5 seed ort.): **ALFA** IF-füzyon 0.833 > LSTM-AE 0.731 > Dense AE 0.622;
**UAV Attack** LSTM-AE/Dense AE 0.677 > IF-füzyon 0.600.

- **H9 — LSTM-AE ALFA'da IF-füzyonu geçemedi:** ~10 dk normal veriyle sequence modeli beklendiği gibi
  veri-aç; literatür deseniyle tutarlı (MSAD: iyi feature + basit yöntem çoğu zaman kazanır). Zamansal
  bağlam yine de Dense AE'ye karşı +0.11 ROC katkı verdi.
- **Pozitif — UAV tip bazında sıçrama:** LSTM-AE spoofing tespiti **1.00** (IF 0.70), jamming **1.00**,
  ping_dos **0.53** (IF 0.37 — zamansal bağlam DoS'un kısmi zamanlama imzasını yakalıyor, H3 ile tutarlı).
- **H10 — max-pencere uçuş skoru yanlış alarm üretiyor:** normal uçuş yanlış alarmı 0.5-0.8 (1-2 normal
  test uçuşunda 0/1 salınımı). Yüzlerce pencerenin maksimumu alınınca P(en az biri > Q99) doğal olarak
  yüksek. ML-3: uçuş skoru = eşik-üstü pencere ORANI veya pencere-P99; POT'un q'su pencere sayısına göre
  Bonferroni benzeri ayarlanmalı.
- **H11 — pencere-ROC < 0.5 aynı etiket semantiği sorunu:** saldırı uçuşunun TÜM pencereleri anomali
  sayılıyor (H1'in pencere versiyonu); pencere-düzeyi metrik ancak zaman-aralıklı etiketle (UAV-SEAD
  `ranges` alanı!) adil olur — SEAD'in aralık etiketleri bunun için kullanılacak.
- POT ≈ Q99 çıktı (val penceresi bol olduğu için ikisi de kararlı); POT'un asıl değeri az-örnekli
  uçuş-düzeyi eşiklerde (H2) — ML-3'te uçuş skorlarına uygulanacak.

## ML-3 sonuçları (2026-07-02, `notebooks/04_ablation_enjeksiyon_usad.ipynb`)

**Ablation (IF-füzyon, 5 seed uçuş-ROC):**
- ALFA: sadece-rehberlik **0.864** > tam füzyon 0.833 > airspeed'siz 0.808 ≫ sadece-kontrol 0.511.
  → **B1:** kontrol_tepki modülü ALFA'da gürültü katıyor; varsayılan dedektör "rehberlik-yalnız" olmalı.
  → Airspeed katkısı küçük ama pozitif; artifact bağımlılığı düşük (ablation kaygısı büyük ölçüde temizlendi).
- UAV: tam 0.600 > sadece-nav 0.554 > missingness'siz 0.423. ping_dos tespiti missingness'ten
  bağımsız çıktı (0.37 = 0.37) → DoS tespitindeki pay "availability imzası" değil; ama genel ROC'a
  missingness katkısı büyük → raporda ikisi ayrı cümlelerle verilmeli.

**Sentetik enjeksiyon (temiz split_00 modeliyle, val+test-normal uçuşlara):**
- ALFA: drift(alt) **0.75 tespit, 0.28 s gecikme** (CUSUM'un var oluş amacı — doğrulandı),
  noise(roll) 0.75/7.2 s; bias(pitch) 0.25, freeze(roll) 0.25 → **B2:** adım-tipi/donma arızaları
  mevcut feature setiyle zayıf; frozen-count/varyans-çöküşü feature'ları modüllere alınmalı.
- UAV: **hiçbir enjeksiyon uçuş-düzeyi alarm veremedi (0.0)** → **H12:** 2 m/s stealthy ramp
  `gps_speed_residual`i ancak normal-val maksimumu kadar oynatıyor (2 m/s ≈ val sınırı);
  şiddet taraması (2→20 m/s) yapılmadan "stealthy spoofing yakalanamıyor" da "yakalanıyor" da denemez.
  Dropout enjeksiyonu SITL normallerindeki doğal %13 attitude eksikliğinin gölgesinde kaldı.

**H10 kararı — hipotez reddedildi:** eşik-üstü-oran skoru max'tan KÖTÜ (ALFA ROC 0.742→0.417).
İmza az sayıda pencerede yaşadığı için oran-tabanlı skor sinyali seyreltiyor. Uçuş skoru **max kalır**;
yanlış alarm sorunu eşik tarafında (POT uçuş-düzeyi, LOFO) çözülecek.

**SEAD kalibre transfer — tezin ana doğrulaması:** UAV modeli + SEAD-normalleriyle kalibre eşik:
uçuş-ROC **0.375 → 0.783**, normal yanlış alarm **1.00 → 0.33**, global_position tespiti **1.00**.
→ Proje iddiasının deneysel kanıtı: **feature semantiği + model platformlar arası taşınır, eşik platforma aittir.**

**USAD:** ALFA 0.450, UAV 0.531 — LSTM-AE'nin altında. Az-veri rejiminde adversarial eğitim
kararsız (beklenen). Karar: sequence modeli olarak **LSTM-AE kalır**; USAD ML-4'e taşınmaz.

## ML-4 sonuçları — veri büyütme (2026-07-03, `notebooks/05_veri_buyutme_yeniden_olcum.ipynb`)

Veri: ALFA 47→**54 uçuş / 10→15 normal** (raw rosbag'lerden `parse_alfa_rosbag.py` ile +5 normal,
+2 engine_fault; 8 eksik bag'in envanteri `scripts/inventory_alfa_raw.py`); UAV-SEAD 60→**179 uçuş /
59 normal** (skip-existing indirme; 1 uçuş HF deposunda 404).

- **H9 ÇÖZÜLDÜ — turun ana kazanımı:** ALFA LSTM-AE uçuş-ROC **0.731 → 0.918 ± 0.104**.
  Normal veri +%50 artınca sequence modeli IF-füzyonu (0.832) net geçti. "LSTM-AE'nin geri kalması
  veri azlığındandı" hipotezi deneysel olarak doğrulandı; ALFA'nın varsayılan modeli artık LSTM-AE.
- **B3 (yeni):** IF-füzyon yeni veriyle iyileşmedi, seed-std'si arttı (0.172→0.283). Neden: rosbag
  normallerinin 2'si nav_info'suz (rehberlik feature'ları imputed) — val'e düştükleri seed'lerde eşik
  bozuluyor. Ders: heterojen normal havuzda IF eşiği kırılgan, LSTM-AE dayanıklı.
- **H11 ölçüldü — SEAD ranges ile ilk adil satır-ROC:** naive 0.563 → adil **0.474**. Yani mevcut
  (UAV-Attack-tasarımlı) modül feature'ları SEAD'in state-estimation anomalilerini satır düzeyinde
  YAKALAMIYOR (yazı-tura civarı). Uçuş-düzeyi SEAD kendi-tespiti de ~0 (bölüm 5).
  → **H13 (yeni):** SEAD anomali aileleri için kaynak-uygun feature seti gerekiyor (EKF innovation,
  local-position tutarlılığı, baro/GPS irtifa farkı) — SEAD şimdilik yalnızca *eşik kalibrasyonu ve
  normal-havuz zenginleştirme* işlevi görüyor, kendi anomalilerinin dedektörü değil.
- **Çapraz-platform havuz:** UAV+SEAD normalleriyle eğitim: SEAD-test ROC 0.617→0.671,
  tespit@1 0.23→0.37, FA 0.00. (ML-3'ün 0.783'üyle doğrudan kıyas yanlış: test seti 40→120 anomalili
  uçuşa büyüdü — daha zor ve daha temsili bir sınav.)
- **POT uçuş-düzeyi:** SEAD kendi-tespiti zaten ~0 olduğundan POT-vs-valmax kıyası bilgi vermedi
  (her iki eşikte de tespit≈0) — POT değerlendirmesi anlamlı sinyali olan ALFA/UAV'de tekrarlanmalı.

## ML-5 sonuçları — SEAD büyütme + oturum-split + EKF feature'ları (2026-07-03, `notebooks/06_sead_ekf_oturum_split.ipynb`)

Veri: SEAD 179 → **349 uçuş / 199 normal** (doğala yakın dağılım, %57 normal; ExtPos 60).
Split artık **oturum-bazlı** (`session_of`: tarih klasörü; aynı günün uçuşları tek tarafta,
anomalili oturumların normalleri train/val'den karantinaya) — train-test oturum kesişimi 0.

- **H13 kısmen ÇÖZÜLDÜ ama beklenmedik yoldan:** adil (ranges) satır-ROC **0.474 → 0.799**
  — iyileşmeyi sağlayan EKF feature'ları DEĞİL, veri büyütme + oturum-temiz train seti oldu
  (eski 3 modül yeni veriyle 0.799; +EKF füzyonu 0.781'e düşürüyor).
- **H14 (yeni) — EKF test-ratio'ları ters sinyal veriyor:** yalnız-EKF modülü satır-ROC **0.354**
  (0.5 altı). Muhtemel mekanizma: anomali sırasında EKF ölçümü REDDEDİYOR → reddedilen ölçüm
  innovation üretmiyor → test ratio düşük kalıyor → anomali "temiz" görünüyor. PX4 hazır-residual'ı
  ancak ölçüm-reddi sayaçlarıyla (innovation_check_flags, reject counters) birlikte anlamlı olur.
  **Karar:** EKF modülü varsayılan füzyona ALINMADI; kolonlar Silver'da duruyor (ileriki iş).
- **Oturum-split kararlılığı:** SEAD uçuş-ROC 0.696 ± **0.012** (ML-4'te ±0.212 idi) —
  seed-varyansının ana kaynağı oturum-içi benzerlik sızıntısıymış; kapandı.
- Tip bazında tespit@1 (+EKF, split_00): external_position **0.82**, global_position **0.70**,
  mechanical 0.43, altitude 0.27 → altitude/mechanical için feature işi devam ediyor.
- **POT ilk kez değer gösterdi:** 199-normalli rejimde POT tespit 0.43 / FA 0.225 vs
  val-max 0.40 / 0.238, iki metrikte de daha düşük std — uçuş-düzeyi eşikte varsayılan POT olur.

## ML-6 metodoloji düzeltmesi (2026-07-03, `notebooks/07_ml6_causal_degerlendirme.ipynb`)

ML-1..5 denetiminde uçuş-içi CUSUM slack'inin bütün uçuşun MAD'inden hesaplandığı ve gelecekteki
anomali eklenince geçmiş CUSUM değerlerinin değiştiği doğrulandı. CUSUM artık yalnız `split_00`
normal-train popülasyonundan öğrenilen sabit center/k ile hesaplanır; parametreler
`artifacts/cusum/` altında saklanır ve prefix-invariance testiyle korunur.

- **Eski CUSUM iddiası geri çekildi:** `alt_error_cusum_pos` tek-feature uçuş ROC'u causal yeniden
  ölçümde **0.878 → 0.611** oldu. En güçlü tek ALFA sinyali artık `xtrack_error` (**0.751**).
  Causal modular IF: ALFA **0.824 ± 0.229**, UAV Attack **0.608 ± 0.220**. LSTM-AE girişlerinde
  CUSUM bulunmadığından önceki LSTM-AE sonucu bu düzeltmeden doğrudan etkilenmez.
- **Blind holdout:** SEAD anomaly oturumlarının sabit %30'u ayrıldı; güncel manifestte 40 anomalili
  + aynı oturumlardan 36 normal uçuş final holdout'tur. Beş development seed'i bu 76 uçuşu görmez.
  ALFA/UAV Attack kıt anomaly nedeniyle açıkça `development-only` işaretlenir.
- **Session-aware LOSO:** SEAD LOFO yerine leave-one-session-out kullanır; aynı oturumun kardeş
  uçuşları validation ve train'e bölünmez.
- **2x2 ablation (blind holdout hariç):** 179/flight ROC 0.574±0.130; 179/session 0.602±0.044;
  349/flight 0.674±0.065; 349/session **0.678±0.013**. Veri büyütme seviye performansını,
  session split özellikle seed kararlılığını artırdı.
- **POT düzeltmesi:** EKF'siz varsayılan modüllerde POT tespit 0.293 / FA 0.114; val-max tespit
  0.373 / FA 0.171. POT FA'yı azaltırken tespiti de düşürdüğü için otomatik varsayılan değildir;
  operasyonel alarm bütçesine göre seçilecektir.
- **Event/K-of-N ilk ölçümünün nüansı:** 165 development eventi üzerinde raporlanan 1-of-1 overlap
  recall 0.594 / 279.2 FA-saat; 2-of-3 0.576 / 138.9; 3-of-5 0.570 / 108.7 idi. Sonraki alarm
  denetiminde bu recall'ın event başlamadan önce açık kalmış alarmı da başarı saydığı görüldü; aşağıdaki
  ML-7 başlangıcı bu metriği yeni-alarm başlangıcı (onset) olarak düzeltti.
- **Artifact:** modular IF joblib modelleri ve LSTM-AE checkpoint'leri; feature listeleri, scaler,
  CUSUM baseline, threshold ve checksum'lu manifestlerle `artifacts/models/<source>/` altında paketlenir.

## ML-7 başlangıcı — operasyonel alarm denetimi ve fiziksel residual v2 (2026-07-03)

Alarm metriği artık yalnız event aralığında üretilen **yeni bildirim başlangıcını** tespit sayar. Eventten
önce başlamış ve event içine taşmış alarm ayrıca `preexisting_alarm_events` olarak raporlanır. Bu daha
katı ama canlı kullanım açısından doğru tanımla base modelin 1-of-1 onset recall'ı **0.224** (overlap
0.594), 2-of-3 **0.200**, 3-of-5 **0.194** oldu; 3-of-5'te 62 event önceden açık alarm nedeniyle eski
metrikte başarı görünüyordu.

Development üzerinde iki operasyonel bütçe tarandı: kritik bildirim için ≤2 FA/saat ve ≥0.30 recall;
advisory için ≤12 FA/saat ve ≥0.50 recall. Bunlar şimdilik ürün varsayımıdır. Base modelde en iyi bütçeli
adaylar sırasıyla **0.030 recall / 1.55 FA-saat** ve **0.164 / 7.16** verdi; ikisi de asgari faydayı
karşılamadığı için policy artifact açıkça `development_rejected` yazıldı ve final holdout açılmadı.

Ardından ham ULog'lardan global-local-baro irtifa residual'ları, attitude/rate setpoint hataları,
actuator effort/control strain ve EKF innovation/gps/fault rejection bitmask feature'ları üretildi.
V2 aday modüller bütçeli kritik recall'ı **0.091**'e, advisory recall'ı **0.182**'ye yükseltti; fakat
Altitude **0.024**, Global Position **0.070**, Mechanical **0.029** seviyesinde kaldı. Sonuç: eşik ve
persistence optimizasyonu tek başına yeterli değildir; sıradaki model işi anomaly-onset odaklı
forecasting/sequence residual (TCN/GRU veya hafif GRU-AE) ve oturum-başı nedensel adaptasyondur.

Kör SEAD holdout ancak bu model development bütçesini karşılayıp feature/model/policy hash'leri
dondurulduktan sonra **bir kez** açılacaktır.

## ML-8A sonuçları — temporal boosting + karar katmanı ayrıştırması (2026-07-06)

**Ne denedik:** Dondurulmuş 10 saniyelik causal descriptor v1 (kanal başına 20 descriptor),
class-balanced LightGBM ve threshold/K-of-N/bootstrap-CUSUM karar katmanları. SEAD development
setinde 5 oturum-izole supervised seed koşuldu; 76 uçuşluk blind holdout hiçbir feature/skor
akışına alınmadı. Descriptor şema SHA-256: `364cb56ab174540fee53fb991bee85c6811a9a68eeec859bfa84a8cf7bcd2670`.

- **H15 — Temporal descriptor + LightGBM skor katmanını iyileştirmedi (Gate B KALDI).**
  SEAD seed-ortalama window AUPRC LightGBM'de **0.349**, mevcut IF-füzyonda **0.385** oldu
  (retrained LSTM-AE: 0.395; bu model eski baseline değildir). Split_00 smoke sonucu 0.666 idi,
  fakat beş seed genellemesi bu ilk iyimser sonucu doğrulamadı. Neden: descriptor özetleri
  oturumlar arasında kararlı onset ayrımı üretmiyor; tek seed'e bakmak belirgin seçim yanlılığı.
  **Ne yapılacak:** LightGBM/Optuna ile kurtarma yok; ML-8C family-holdout DevNet/Deep SAD fazına geçilecek.
- **H16 — Validation-normal FA kalibrasyonu test oturumlarına taşınmadı (Gate C KALDI).**
  LightGBM'in hiçbir karar katmanı seed-ortalamasında kritik (recall>=0.30 @ <=2 FA/saat)
  veya advisory (recall>=0.50 @ <=12 FA/saat) hedefini karşılamadı. En yüksek LightGBM
  SEAD onset recall threshold/advisory'de 0.302'ydi, fakat FA **38.87/saat** oldu. CUSUM karar katmanı
  skor tavanını aşamadı. **Ne yapılacak:** Karar katmanı taraması genişletilmeyecek; yeni skor
  ailesi olmadan policy tuning yapılmayacak.
- **H17 — SEAD için mevcut LSTM baseline artifact'i yoktu.** Orijinal 3x3 tablonun LSTM satırı
  `N/A` kaydedildi. Kullanıcı izniyle eski artifactleri ezmeden `ml8a_retrained_lstm_ae` üretildi
  ve ayrı/etiketli kıyas satırı olarak koşuldu; bunu ML-6/7 baseline'ı gibi sunmak yasaktır.
- **H18 — “Sonucu kurtarma” ile “protokolü tamamlama” ayrıldı.** Gate sonucu görüldükten sonra
  descriptor v1, LightGBM hiperparametreleri veya test-eşiği değiştirilmedi ve holdout açılmadı;
  bunlardan herhangi biri mevcut development sonucunu seçim verisine dönüştürüp iyimser yanlılık
  üretirdi. Bunlar ancak yeni sürüm/yeni faz, önceden yazılmış hipotez ve ayrı değerlendirme setiyle
  yeniden açılabilir. Buna karşılık sabit reçeteli ALFA ve aile kırılımları tuning değildir ve
  deney protokolünü kapatmak için çalıştırılır.

**ALFA sabit-reçete tamamlaması:** 5 seed'de LightGBM AUPRC **0.843**, IF **0.858**, mevcut
LSTM-AE **0.872** oldu; LightGBM yine skor üstünlüğü göstermedi. LightGBM'in hiçbir policy'si
operasyonel hedefi karşılamadı. Buna karşılık mevcut IF + CUSUM advisory satırı **0.625 onset
recall / 7.91 FA-saat** ile advisory hedefini geçti (AvgDT 34.65 s, seed-ortalama MaxDT 81.08 s).
Bu, karar katmanının ALFA'daki mevcut skora faydasını gösterir; LightGBM Gate B/C başarısı değildir.
Fault kırılımında LightGBM+CUSUM advisory: engine 0.741, aileron 1.00, aileron-rudder 1.00,
elevator 1.00, rudder 0.333 (seedler boyunca event-ağırlıklı).

**SEAD aile kırılımı:** LightGBM threshold/critical onset recall external-position 0.561 iken
global-position 0.064, mechanical 0.080 ve altitude 0.000 kaldı. İyileşme tek aileye yoğunlaşmış,
genel operasyonel faydaya dönüşmemiştir.

**Gate kararı:** Gate A GEÇTİ; Gate B KALDI. Gate C matris düzeyinde ALFA mevcut IF+CUSUM ile
GEÇTİ, fakat yeni LightGBM için ve SEAD'de KALDI. Blind holdout açılmadı. Gap-aware nihai
90'ar matris satırı `artifacts/ml8a/<source>/full_matrix_gapfix/metrics.json` altındadır;
önceki `full_matrix/` çıktıları telemetry-gap payda hatası nedeniyle superseded'dır.

## ML-9 sonuçları — kategori-eşleşmeli residual'lar (2026-07-06)

**Ne denedik:** SEAD Silver'a pooled EKF innovation'ları silmeden dikey hız/irtifa innovation'ları
ve 16 `actuator_outputs` kanalı eklendi. Gerçek development verisindeki 15 mechanical + 15 normal
uçuş denetiminde kullanılmayan kanalların uçuş-içi std'si 0, aktif kanalların std'si >10 PWM idi;
aktiflik eşiği 1 PWM seçildi. Gelecek bilgisiyle kanal seçmemek için aktiflik tüm uçuş std'siyle
değil, geçmişe-bakan expanding std ile açıldı. Rolling/CUSUM prefix-invariance hem sentetik hem gerçek
uçuş prefix'inde geçti. Mevcut threshold/K-of-N/bootstrap-CUSUM kodu değiştirilmeden import edildi.

- **H19 — Kategori-eşleşmesi yönsel sinyal verdi ama Gate B büyüklük/kararlılık şartını geçmedi.**
  Gate B kuralı sonuç görülmeden önce aynı policy+bütçede ortalama recall kazancı >=0.05 ve en az
  3/5 seed'de pozitif kazanç olarak donduruldu. `Position.Z` için dikey modül CUSUM/advisory'de
  **0.096**, pooled EKF **0.074** recall verdi (+0.021; 4/5 seed): yön tutarlı, etki yetersizdi.
  Threshold/advisory kazancı da yalnız +0.034'tü. `Actuator Outputs+Controls` için motor-simetri
  CUSUM/advisory **0.205**, mevcut kontrol-cevabı **0.180** verdi (+0.024; 2/5 seed). Kritik
  CUSUM'da +0.054 görünse de yalnız 1/5 seed pozitif olduğundan genellenebilir kabul edilmedi.
  **Gate B KALDI.**
- **H20 — Yeni modüller fusion skorunu operasyonel bütçeye taşımadı (Gate C KALDI).** ML9 fusion
  CUSUM/advisory seed ortalaması **0.222 onset recall / 25.83 FA-saat**, kritik **0.139 / 14.49**
  oldu; her ikisi de FA bütçesini aştı. Bütçe içinde kalan K-of-N/advisory **3.40 FA-saat** verdi
  ama recall yalnız **0.017** idi. Mevcut fusion CUSUM/advisory **0.212 / 23.63** olduğundan yeni
  feature'lar toplam recall'ı yalnız ~0.011 artırıp FA'yı da yükseltti. Kritik/advisory hedefini
  karşılayan satır yoktur; blind holdout kapalı kaldı.
- **H21 — Plan sayıları ile frozen gerçek artifact arasında veri-sürümü farkı var.** ML9 planı
  412 uçuş/76 holdout yazsa da çalıştırma anındaki mevcut manifest ve yeniden üretilen gerçek veri
  **611 işlenebilir uçuş/131 holdout** içeriyordu (labels.json'da 612 kayıt; telemetry tablosunda 611).
  §6 gereği yeni split üretilmedi; mevcut manifest byte-byte korundu (SHA-256
  `0e3047b3abc11d09bc5b2b94e35cca117efce8fa49fdc37393c146550b8d0f0d`). Development matrisinde
  `Position.Z` n=113 event, `Actuator Outputs+Controls` n=41 eventtir. `Actuator Thrust`, `Battery`
  ve `Velocity` yalnız n=2 olduğundan raporlanır ama genelleme iddiası yapılmaz; bunlara özel feature
  eklenmedi.

**Gate kararı:** Gate A GEÇTİ; Gate B ve Gate C KALDI. `PX4_ML9_CANDIDATE_MODULES` aday olarak
kalır, default/production modüllerine alınmaz. Threshold/K/N/CUSUM sonucu görerek genişletilmez,
motor eşiği yeniden ayarlanmaz, nadir mechanical alt-tipleri modellenmez ve holdout açılmaz. Planın
talimatıyla sıradaki araştırma ML-10 forecast-residual/foundation-model pilotudur; bu fazda başlatılmadı.
Nihai checksum'lu 5-seed artifact: `artifacts/ml9/uav_sead/full_matrix/` (65 kayıtlı dosya).

## ML-10 sonuçları — zero-shot Chronos forecast-residual pilotu (2026-07-06)

**Ne denedik:** Python 3.14/CPU ortamında `chronos-forecasting==2.3.1` ile resmi
`amazon/chronos-bolt-tiny` (8.65M parametre, revision
`a0e552de83495b5c28c14c71c374f3e33280b340`) modeli zero-shot kullanıldı. Development
doluluk denetiminde dikey adaylar `alt` **%100**, `local_alt_m` **%99.998** ve `baro_alt_m`
**%6.87** çıktı; sonuç görülmeden önce `alt` sabitlendi. Mechanical kanal olarak ML-9'un
`actuator_output_std` kolonu kullanıldı (%99.21 satır, 480 uçuşun 479'unda en az bir değer).
Her 1 saniyelik karar noktasında yalnız önceki en fazla 512 gözlemle q10/q50/q90 tahmini
yapıldı; bant dışı normalize residual bir kez hesaplandı ve seedler arasında tekrar üretilmedi.

- **H22 — Zorunlu CPU fizibilitesi açık farkla geçti.** 512 örnekli gerçek development
  penceresinde sıcak tek tahmin ortalaması **9.3 ms** idi. Uçuş boyutu dağılımının sabit sekiz
  kantilinde iki kanal birlikte **2.57 s** sürdü; 480 development uçuşuna projeksiyon
  **102.4 s / 0.028 saat** oldu. Sabit `<3 saat` kuralıyla 1 s stride ve tam development kapsamı
  seçildi. Gerçek tam precompute **101.2 s** sürdü; 57.323 irtifa ve 56.812 actuator residual'i
  üretildi. Fizibilite sonucu görülerek stride/subset değiştirilmedi.
- **H23 — Gate B mechanical forecast-residual ile GEÇTİ; irtifa hipotezi reddedildi.** Önceden
  dondurulan aynı policy+bütçede ortalama recall kazancı >=0.05 ve en az 3/5 seed pozitif şartı,
  `Actuator Outputs+Controls` için birden çok satırda sağlandı. En güçlü karşılaştırma
  CUSUM/advisory'de `chronos_motor` **0.390** vs ML-9'un en iyi `motor_simetrisi` **0.205**
  (+**0.185**, 4/5 seed); CUSUM/critical **0.205 vs 0.093** (+0.112, 4/5) oldu. Buna karşılık
  `Position.Z` CUSUM/advisory'de `chronos_dikey` **0.023**, `dikey_tutarlilik` **0.096**
  (-0.073, yalnız 1/5 pozitif) verdi. Sonuç: uçuşun kendi geçmişine koşullu residual actuator
  dinamiğinde gerçek ve kararlı ek sinyal sağlıyor, fakat irtifa onset'ini çözmüyor.
- **H24 — Kategori kazancı fusion düzeyinde operasyonel hedefe dönüşmedi (Gate C KALDI).** Planın
  sabit `ml10_fusion=max(existing_fusion, chronos_dikey, chronos_motor)` skoru CUSUM/advisory'de
  **0.213 onset recall / 23.92 FA-saat**, kritik **0.133 / 12.83** verdi; hedefler sırasıyla
  `>=0.50 @ <=12` ve `>=0.30 @ <=2` idi. Sonuç mevcut fusion'a (0.212 / 23.63) neredeyse eşit,
  ML-9 fusion'dan (0.222 / 25.83) daha düşük FA ama daha düşük recall'dır. Mechanical kanalın tekil
  kazancı max-fusion ve bütün-event validation kalibrasyonunda korunmadı. Sonuç görülerek fusion,
  Chronos bağlamı, quantile bandı veya policy grid'i değiştirilmedi.

**Gate kararı:** Gate A GEÇTİ (future-leak, zero-shot/no-training, ortak füzyon ve donmuş karar
katmanı testleri 9/9; 131-uçuş holdout okunmadı). Gate B mechanical dalından GEÇTİ. Gate C KALDI;
bu yüzden blind holdout kapalı kaldı ve `chronos_motor` development'ta kanıtlanmış bir aday skor
olsa da default/production fusion'a alınmadı. Nihai checksum'lu artifact:
`artifacts/ml10/uav_sead/full_matrix/`; fizibilite ve precompute manifestleri aynı ağacın altındadır.

## Görselleştirme sonuçları — veri keşfi ve tanılama (ML-11, 2026-07-06, `notebooks/09_gorsellestirme_ve_veri_kesfi.ipynb`)

**Ne yaptık:** `scripts/make_visualizations.py` ile üç dataset için read-only analiz turu:
veri karnesi (sınıf/oturum/doluluk/süre), PCA+t-SNE projeksiyonları (4 boyama), Spearman
korelasyon + **tek-feature × kategori AUC matrisi** (`artifacts/viz/<dataset>/s3_features/feature_auc_matrix.csv`)
ve mevcut artifact modellerle tanılama görselleri (violin, ROC/PR, karar matrisi, örnek zaman
serileri). Hiçbir model/eşik/split/scaler/CUSUM artifact'ı değişmedi; 131-uçuş SEAD holdout'u
hiçbir figüre girmedi (`tests/test_ml11_viz.py` hash denetimiyle assert eder; deterministik
subsample seed=42, uçuş başına ≤200 satır). Projeksiyon figürlerinde ±10 IQR görsel kırpma
kullanıldı (tek uç değer PCA'yı tek noktaya sıkıştırıyordu); kırpma hiçbir skor hesabına girmez.

- **H25 — SEAD normal sınıfının heterojenliği artık görsel kanıtlı.** Development'taki 324
  normal uçuş **49 oturuma** dağılıyor (en büyük oturum 21 uçuş); t-SNE'de normal noktalar tek
  küme değil çok sayıda ada ve oturum boyaması adaların büyük ölçüde oturumu izlediğini
  gösteriyor. "Normali öğren" hedefi tek yoğunluk değil, oturum ailelerinin birleşimi (D.1'in
  görsel teyidi).
- **H26 — Zayıf kategorilerin kök nedeni ikiye ayrıştı (ANA BULGU).** Tek-feature AUC matrisi
  iki farklı hikâye anlatıyor:
  - *Füzyon/model sorunu:* `Actuator Outputs+Controls` için `actuator_thrust_cmd` tek başına
    **AUC 0.983** ayrıştırıyor ve feature zaten `kontrol_cevabi` modülünün İÇİNDE — ama 16
    feature'lık geniş modül + 6-modüllü max-füzyon içinde seyreliyor. Aynı ailede
    `attitude_error_mag` (0.783), `control_strain` (0.748), `pitch_rate_error` (0.747) de
    tekil olarak füzyonun kategori recall'undan (ML-9: 0.205) çok daha güçlü.
  - *Veri/kapsam sorunu:* `Position.Z`'nin en iyi üç ayrıştırıcısı baro tabanlı
    (`alt_baro_residual_5s_max` **AUC 0.996**) fakat yalnız **n=33 satırda** mevcut (B.1/B.2:
    baro kanalı ~%7 dolulukta). Baro'suz en iyi aday `ekf_alt_innov_cusum_pos` (0.741) zaten
    ML-9'da denendi ve Gate B'yi geçemedi. `Actuator Thrust` (n=113 satır) hiçbir feature'da
    anlamlı ayrışmıyor → bu kategori için sinyal mevcut kolonlarda yok.
- **H27 — Füzyon skoru doygun: kalibre max-füzyon normal satırlarda bile 0.92-1.0 bandında.**
  6 modülün empirik-CDF max'ı tabanı yukarı itiyor; violin grafiklerinde normal ile kategori
  dağılımları büyük ölçüde örtüşüyor. Kategori-recall kaybının bir kısmı skor değil karar
  marjı sorunu. Uçuş düzeyi ROC (5 seed, IF-füzyon) **0.557** — SEAD'de ayrım uçuş düzeyinde
  değil event düzeyinde yaşanıyor; advisory CUSUM noktasında 5-seed toplam karar matrisi
  TP 416 / FN 364 / FP 380 / TN 543.
- **H28 — ALFA'da paketli LSTM-AE eşiği aşırı muhafazakâr.** Uçuş ROC AUC **0.750**
  (IF-füzyon 0.622) ama val-q99 pencere eşiği çalışma noktasında 38 anomalili uçuştan yalnız
  **3'ü** alarm üretiyor (0 yanlış alarm). Eşik 2 normal val uçuşundan kalibre edilmek zorunda —
  H2'deki küçük-n eşik kararsızlığının çalışma-noktası görünümü. ALFA tek-feature tarafında
  rudder için `path_dev_mag` (sep. 0.931) ve `roll/pitch_spec_energy_5s` (0.91) güçlü —
  bunlar da modüllerin içinde zaten var; sorun yine seyrelme + eşik.
- **Yorum uyarısı (D.2 küçük-n disiplini):** tek-feature AUC'lerin bir kısmı arıza fiziği değil
  uçuş-profili karışıklığı yansıtabilir (ör. ALFA `engine_fault` için `wp_dist` 0.913 — arızalı
  uçuşlar eve dönüyor). CUSUM-tipi feature'larda q99 oranı normal q99≈0 olduğu için astronomik
  çıkabilir; oran tek başına değil AUC ile birlikte okunmalı.

**Top-10 manuel feature-engineering adayı** (keşif listesi; seçim ancak yeni bir Gate turunda,
farklı validation ile değerlendirilebilir — aynı veriyle sonuç raporlanmaz):

| # | Aday | Kategori | AUC | Not |
|---|---|---|---|---|
| 1 | `actuator_thrust_cmd` (ince modül/tek başına) | Actuator O+C | 0.983 | Modülde var, füzyonda seyreliyor |
| 2 | `attitude_error_mag` + `_5s_rms` | Actuator O+C | 0.783/0.775 | Aynı seyrelme sorunu |
| 3 | `control_strain` | Actuator O+C | 0.748 | " |
| 4 | `pitch_rate_error` | Actuator O+C | 0.747 | " |
| 5 | `pos_test_ratio_5s_mean` | Velocity | 0.997 | q99 oranı ~66× — güçlü kuyruk imzası |
| 6 | `gps_frozen_count` | Velocity | 0.978 | Seyrek uç-değer imzası (q99 patlıyor) |
| 7 | `gps_speed_residual` | Position.X | 0.890 | q99 ~5.3× |
| 8 | `hgt_test_ratio_5s_max` | Battery | 0.972 | n=51 satır — küçük-n uyarısıyla |
| 9 | `ekf_alt_innov_5s_mean/max` | Position.Z | 0.673/0.675 | q99 5-11× (kuyruk var, mean zayıf); baro'suz en gerçekçi dikey aday |
| 10 | ALFA: `path_dev_mag`, `roll/pitch_spec_energy_5s` (rudder'a özel ince modül) | rudder_fault | 0.93/0.91 | Rudder H6 zafiyetine hedefli |

Baro tabanlı `alt_baro_residual*` (AUC 0.996) aday DEĞİL: %7 doluluk kapsam engeli — bu bir
feature-engineering değil veri temini işi (baro kanalı tamamlanırsa yeniden değerlendirilir).

**Eğitim izi kuralı (Bölüm 5):** `train_lstm_autoencoder` artık epoch başına train/val loss
geçmişi döndürüyor; `src/ml/training_log.py::write_training_log` bunu
`artifacts/training_logs/<source>/<model>/<run_id>/loss.csv` + `loss.png` olarak yazıyor ve iki
LSTM paketleme scripti çağrıyı içeriyor. Bundan sonra eğitilen her model iz bırakır (IF'in epoch
kavramı yok; ML-10 Chronos zero-shot olduğundan kapsam dışı). Bu fazda model eğitilmedi.

**Doğrulama:** tam `pytest` 185 geçti + bilinen 4 MinIO SDK hatası (ML dışı, F.2); 7'si yeni
`tests/test_ml11_viz.py` (deterministik subsample, holdout-hash izolasyonu, viz manifest
checksum'ları, eğitim izi). Checksum'lu çıktılar: `artifacts/viz/{alfa,uav_attack,uav_sead}/viz_manifest.json`.

## ML-2/ML-3'e devredilen iş listesi

| # | İş | Adres |
|---|---|---|
| 1 | LSTM-AE (10 sn pencere, normal-only) | H1, H6 |
| 2 | EVT/POT tabanlı eşik + LOFO eşik dağılımı | H2 |
| 3 | Inter-arrival zamanlama feature'ları | H3 |
| 4 | SITL/live ayrı kalibrasyon | H4 |
| 5 | SEAD τ-kalibrasyonlu transfer deneyi | H5 |
| 6 | Ablation: missingness±, airspeed±, residual-only | H1/H7 |
| 7 | Sentetik enjeksiyon test seti (freeze/bias/drift/stealthy-ramp) | kapsam |
