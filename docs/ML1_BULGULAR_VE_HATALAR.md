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
