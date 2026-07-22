# ML Sistemi — Bilinen Yetersizlikler ve Sınırlamalar Kaydı

Bu doküman ML-0'dan ML-10'a kadar (2026-06-29 → 2026-07-06) bulunan **tüm** bilinen eksiklik,
sınırlama ve açık işi TEK bir yerde toplar. Amaç: mentör/ekip incelemesinde hiçbir bulgunun
`docs/ML1_BULGULAR_VE_HATALAR.md`'nin 24 H-maddesi ve `docs/decisions.md`'nin 10 ADR'si arasında
kaybolmaması. Her madde kaynağını (H-no/ADR-no/dosya) gösterir — burada anlatılan hiçbir şey
yeni bir iddia değildir, hepsi başka yerde zaten kanıtlanmış bulguların tek-yerde konsolidasyonudur.

**Nasıl okunur:** her madde bir **Durum** etiketi taşır:
- 🔴 **Yapısal sınır** — mevcut veri kaynağının/yöntemin doğal boyutu; daha fazla mühendislikle
  düzelmez (ör. akademik veri setinin kendi küçüklüğü). Kapatmanın tek yolu yeni bir veri kaynağı.
- 🟡 **Gerçek ama küçük açık iş** — bilinen, ölçülmüş, düzeltilebilir ama henüz yapılmamış.
- ⚪ **Bilinçli kapsam dışı** — düzeltilebilir olabilir ama proje bunu şu an bilerek yapmıyor
  (istatistiksel olarak anlamsız n, veya metodoloji disiplini gereği).
- ✅ **Kapandı/telafi edildi** — bir zamanlar sorundu, artık giderildi veya kabul edilebilir
  bir telafisi var; yalnızca denetim izi için burada.

---

## A. Veri kaynağı sınırlamaları

### A.1 — ALFA: rudder=4, elevator=2, aileron_rudder=1 uçuş (🔴 Yapısal sınır)

`docs/ML1_BULGULAR_VE_HATALAR.md` H6. Resmi ALFA makalesi (Keipour/Mousaei/Scherer, arXiv
1907.06268) toplam **47 işlenmiş uçuş** bildiriyor (23 motor arızası + 24 diğer arıza tipi).
2026-07-06'da `Desktop/ALFA/processed/processed/` klasörü kontrol edildi: tam **47 klasör**,
isimleri makaleyle birebir eşleşiyor. Mevcut 54 uçuşumuz (47 resmi + 7 raw-rosbag kurtarma,
`docs/ML1_BULGULAR_VE_HATALAR.md` ML-4 bölümü) bu resmi külliyatın **tamamını zaten kapsıyor**.
`dataflash/` klasörü (60 dosya, yalnız 2 tarih) ayrıca kontrol edildi — aynı iki güne ait
zaten-ingest-edilmiş uçuşların paralel ArduPilot log kaydı, yeni oturum değil.
**Kapatmak için:** yeni bir kaynak gerekir (ör. gerçek/simüle ek rudder arıza uçuşu — proje
kapsamında yok). Bu sayılarla istatistiksel genelleme iddiası yapılamaz; rapor bunu her seferinde
n belirterek yapmalı.

### A.2 — SEAD: 8 çok-sınıflı uçuş sessizce dışlanıyor (🟡 Gerçek ama küçük açık iş)

2026-07-06 araştırması (bu oturum). `src/ingestion/uav_sead_downloader.py::select_flights()`
yalnız `len(real_classes)==1` uçuşları alıyor; mapping.json'daki 1396 uçuşun **8 tanesi**
çok-sınıflı (ör. `['Altitude','Mechanical']`) ve tamamen atlanıyor. Tek-sınıf havuzları
büyütürdü: altitude 73→78, mechanical 41→47, global_position 40→41, external_position 193→197
(mechanical için +%15). **Kapatmak için:** ML-9'un zaten kurduğu `load_uav_sead_ranges_by_category`
(annotasyon-kategorisi bazlı range) altyapısı bu uçuşları doğal olarak destekler — flight_label
tekil bir string olduğu için değil, kategori bazlı etiketleme kullanıldığı için. Henüz
UYGULANMADI; `select_flights()`'ın filtresi gevşetilip downstream etiketleme kategori-bazlı
yapılmalı.

### A.3 — SEAD: 141 "Uncategorized"-only uçuş (⚪ Bilinçli kapsam dışı)

Aynı 2026-07-06 araştırması. Bu uçuşların TÜM annotasyonları "Uncategorized" — gerçek ground-truth
belirsiz. Doğru şekilde dışlanıyor, bu bir eksiklik değil, gerekçeli bir hariç tutma.

### A.4 — SEAD: Battery/Vibration/Magnetometer/Actuator Thrust/Velocity alt-tipleri (⚪ Bilinçli kapsam dışı)

`docs/ML9_PLAN.md` §0, H21 (`docs/ML1_BULGULAR_VE_HATALAR.md`). `mechanical_fault`'un annotasyon-
kategori kırılımında Actuator Outputs 27, Actuator Controls 9 iken Magnetometer 3, Battery 2,
Vibration 1, Raw Accel 1, Actuator Thrust 1. ML-9'un development matrisinde Actuator Thrust/
Battery/Velocity yalnız **n=2** event. Bilerek modellenmiyor — istatistiksel olarak anlamsız,
özel feature eklense bile genelleme iddiası yapılamaz.

### A.5 — UAV Attack: ping_dos 4/6 logda tespit edilemiyor (🔴 Yapısal sınır)

`docs/ML1_BULGULAR_VE_HATALAR.md` H3. 6 DoS logundan 2'sinde attitude eksikliği bir imza
bırakıyor (dolaylı tespit); kalan 4'ünde network-katmanı saldırısı mevcut 4 uORB topic'inin
hiçbirine yansımıyor. **Kapatmak için:** ham `.ulg`'daki mesaj-varış zamanlaması/paket metadata'sı
Silver'a taşınmalı (inter-arrival feature'ları) — bu veri büyütmeyle değil, yeni bir parse
kapsamıyla çözülür; henüz yapılmadı (`docs/ML1_BULGULAR_VE_HATALAR.md` devir listesi #3).

### A.6 — UAV Attack: gps_jamming n=1, aileron_rudder_fault n=1 (🔴 Yapısal sınır)

Tek örnekli sınıflar; bu veri setlerinde büyütme yok (UAV Attack sabit, ALFA A.1'de açıklandığı
gibi tükendi). "1.00 tespit" gibi sayılar n=1'den geliyor — istatistiksel olarak anlamsız,
her raporda böyle işaretlenmeli.

### A.7 — UAV Attack normal havuzu yalnızca 6 log (🔴 Yapısal sınır)

Dataset bu kadar. SEAD normalleriyle çapraz-platform havuzu (ML-4, `docs/ML1_BULGULAR_VE_HATALAR.md`)
kısmi telafi sağlıyor (SEAD-test ROC 0.617→0.671) ama UAV Attack'ın kendi normal çeşitliliğini
artırmıyor.

### A.8 — `velocity_mps` iki kaynakta da tamamen null (✅ Kapandı/kabul edilebilir telafi)

ADR-005 "BİLİNEN EKSİK". ALFA'da `nav_info-velocity` topic'i eşleşmiyor, UAV Attack'ta ham hız
alanı (`vel_n/vel_e/vel_d`) Silver'da hiç yok. Gold'un değil Silver parser'ların kapsamı; ML
feature katmanında (`src/ml/features/`) ayrıca hesaplanan `gps_speed_calc_mps`/`log_gps_speed`
gibi türetilmiş hız kolonlarıyla fiilen telafi edilmiş durumda (H1 pozitif bulgu:
`gps_speed_residual` bu türetilmiş hızla çalışıyor). Gold'daki ham `velocity_mps` kolonu hâlâ
null — düzeltilmedi, bilerek not edildi.

### A.9 — `battery_power_w` %96 NaN (⚪ Bilinçli kapsam dışı)

`docs/ML1_BULGULAR_VE_HATALAR.md` H8.2. `current_a` bu datasette çoğunlukla -1 sentinel değer.
Feature listede impute-güvenli olarak kalıyor ama bir "batarya modülü" kurulacaksa önce ayrı bir
veri kalitesi analizi şart — henüz talep edilmedi.

---

## B. Feature / sensör sınırlamaları

### B.1 — `alt_local_residual` altitude_anomaly için yapısal olarak ölü (🔴 Yapısal sınır)

`docs/ML_ORNEK_INPUT_OUTPUT.md` Örnek 3 (2026-07-06 doğrulaması). Bu feature altitude_anomaly
etiketli uçuşların **%0**'ında dolu — kaynak topic (yerel konum referansı) bu uçuşlarda hiç
loglanmamış. 413→611 uçuşa büyümeyle bile değişmedi (aynı doğrulama iki farklı veri hacminde
tekrarlandı). **Veri büyütme bunu çözmez** — eksik olan topic'in kendisi, miktar değil.

### B.2 — `alt_baro_residual` altitude_anomaly'de yalnızca %7 uçuşta dolu (🔴 Yapısal sınır)

Aynı kaynak. Barometre topic'i çoğu uçuşta yok. `hgt_test_ratio` (EKF irtifa tutarlılığı) tek
kalan sinyal — ortalamada normalle neredeyse ayrışmıyor (0.041 vs 0.047) ama uçuş-başı MAKSİMUM
istatistiğinde zayıf bir umut var (p90: 0.258→0.899, SEAD 611'e büyüdükten sonra 0.047→0.132'ye
çıktı — bu KISMEN gerçek bir iyileşme, ML-1 Bölümü'nde kayıtlı).

### B.3 — EKF test-ratio'ları TERS sinyal veriyor (✅ Kapandı — bilerek füzyon dışı bırakıldı)

H14 (`docs/ML1_BULGULAR_VE_HATALAR.md`). Anomali sırasında EKF ölçümü REDDEDİYOR → reddedilen
ölçüm innovation üretmiyor → test ratio düşük kalıp anomali "temiz" görünüyor (yalnız-EKF modülü
satır-ROC 0.354, rastgeleden kötü). Reject-bayraklarıyla (`innovation_check_flags_*`) birleşmeden
kullanılamaz. **Karar:** EKF test-ratio modülü varsayılan füzyona hiç alınmadı; kolonlar Silver'da
duruyor, ileride reject-counter'larla birlikte yeniden değerlendirilebilir.

### B.4 — Ping DoS network-katmanı imzası telemetriye yansımıyor (bkz. A.5)

### B.5 — Reconstruction skorları kırpılmamış `RobustScaler` genliğine hâkim (🟡 Gerçek açık iş, ADR-016, bu oturum)

ML-16 Kol L (LSTM-AE SEAD yeniden eğitimi) sırasında, bağımsız olarak eğitilmiş Dense-AE ve
USAD ajanları (aynı `_align_score`/pencereleme konvansiyonunu paylaşan) split_00'da
`threshold`/`critical` kararında LSTM ile BİREBİR aynı kategori-bazlı detected/false-alarm
sayılarını üretti — üç farklı mimarinin "eşit derecede iyi öğrenmesi" ile açıklanamayacak bir
örtüşme. Araştırma (`scripts/diagnose_ml_lstm_sead_magnitude_domination.py`,
`artifacts/ml_lstm_sead/uav_sead/full_matrix/magnitude_domination_diagnostic.json`) kök nedeni
buldu: eğitilmiş LSTM'in skor SIRALAMASI, tamamen eğitilmemiş (rastgele başlatılmış) aynı
mimariyle Spearman ρ=0.964, modelsiz saf `‖x‖²` genlik taban-çizgisiyle ρ=0.965 KORELE —
eğitim "girdi ne kadar büyük" ötesine neredeyse hiçbir şey katmıyor. `RobustScaler` (proje
çapında kullanılan, aykırı değer kırpmayan ölçekleyici) sayesinde bir avuç aşırı-genlikli
pencere — gerçek GPS-sahtekârlığı sıçramaları (`external_position_anomaly`, beklenen/kısmen
meşru) VE en az bir "normal" etiketli ama donmuş-GPS + `eph`≈25000 sentinel içeren uçuş
(muhtemelen yanlış etiketli veya loglama artefaktı, İSTENMEYEN) — herhangi bir sınırlı-çıktılı
autoencoder'ın reconstruction hatasına mimariden bağımsız hâkim oluyor.
`ThresholdPolicy`/`_align_score`/`build_windows` doğrudan test edildi ve DEJENERE DEĞİL ("ilk
tanımlı skorda ateşle" davranışı yok); sorun karar katmanında değil, ölçekleme/özellik
seçiminde. **Sonuç:** ADR-016'daki LSTM recall rakamları (özellikle mevcut en iyiyi ham
sayıda geçen CUSUM/advisory 0.251) "öğrenilmiş sinyal" olarak sunulamaz — büyük kısmı genlik
aykırı-değer tespiti olabilir. **Kapatmak için:** kırpmalı/robust ölçekleme (ör. görselleştirme
fazının ±10 IQR kırpması gibi ama SKORA giren) veya genlik-normalize edilmiş bir reconstruction
skoru; ayrıca "eph≈25000" sentinel'in bir SEAD veri kalitesi sorunu olup olmadığı (yanlış
etiketleme mi, gerçek sensör arızası mı) ayrı araştırılmalı. Bu oturumda post-hoc
DÜZELTİLMEDİ (sonuç görülmeden ön-kayıtlı disiplin gereği); yeni, ayrı bir ön-kayıtlı turda
değerlendirilmeli.

**2026-07-10 güncellemesi:** Dense-AE ve USAD'ın kendi tam 5-seed koşuları (ADR-017/018)
split_00 smoke bulgusunu doğruladı — üç mimarinin de ham recall'ı benzer aralıkta
(threshold/critical: LSTM 0.219, Dense-AE 0.215, USAD 0.217) ve üçü de aynı bütçe-dışı
FA'ya sahip. Bu, tek-split bir tesadüf olmadığını, sistematik bir ölçekleme sorunu
olduğunu doğruluyor.

**2026-07-10 güncellemesi #2 (ML-16 Kol N, ADR-019) — düzeltme denendi, kısmen işe yaradı
ama madde KAPANMADI:** Aynı 3 dondurulmuş modelden (yeniden eğitim yok) iki yeni skor
türetildi: bağıl hata ve kanal-başına yüzdelik-sıra. Kanal-başına yüzdelik-sıra genlik-
bağımlılığını neredeyse hiç azaltmadı (ρ hâlâ ~0.7-0.9). Bağıl hata genlik-bağımlılığını
GERÇEKTEN kırdı (3 mimaride de ρ 0.96'dan ~0.15-0.55'e düştü, bazı split'lerde negatif) —
ama bunun karşılığında recall neredeyse sıfırlandı (çoğu hücrede critical recall <0.06).
**Sonuç: kırpma/normalize sorunu tek başına çözmüyor çünkü genliği temizleyince altta
operasyonel olarak kullanılabilir bir sinyal yok.** Kapatma yolu artık yalnızca "daha iyi
ölçekleme" değil — ya davranışsal skoru veri-kalitesi sinyalinden tamamen ayırmak ya da
farklı bir feature ailesi/veri kaynağı gerekiyor olabilir; bu oturumda başlatılmadı.

---

## C. Model / yöntem sınırlamaları (Gate başarısızlıkları)

### C.1 — ML-8A: LightGBM temporal-descriptor skorlayıcı Gate B/C'yi geçemedi (✅ Kapandı, ADR-008; LSTM takibi ADR-016 ile tamamlandı)

SEAD window AUPRC LightGBM 0.349 < mevcut IF-füzyon 0.385 (ALFA'da 0.843 < IF 0.858 < LSTM-AE
0.872). Hiçbir karar katmanı kombinasyonu kritik/advisory FA bütçesini karşılamadı (en iyi
threshold/advisory: 0.302 recall ama 38.87 FA/saat). Literatürle tutarlı: az etiketli veride
yarı-denetimli > tam-denetimli. **Kurtarma yapılmadı** (Optuna/hiperparametre taraması bilerek
atlandı) — bu metodolojik disiplin, eksiklik değil.

**2026-07-09 güncellemesi (ML-16 Kol L, ADR-016):** "ML-8A/başka model ailesi (planlanmadı)"
maddesi artık kapalı — SEAD LSTM-AE (`src/ml/models/lstm_autoencoder.py`, H17'de bir kez
yan-karşılaştırma olarak eğitilmişti) GÜNCEL split_manifest.json'da (899 normal uçuş, ML-14
zenginleştirmesi sonrası) 5-seed yeniden eğitildi ve resmi ml14/ml15 fusion+karar-katmanı
hattına kablolandı (`scripts/run_ml_lstm_sead_evaluation.py`). Gate A (holdout izolasyonu +
determinizm) GEÇTİ. **Gate B (operasyonel hedef) KALDI** — üç ön-kayıtlı varyanttan
(`lstm_recon` tek başına, `+ml14_fusion`, `+itki_komutu`) hiçbiri critical ≥0.30 recall @ ≤2
FA-saat veya advisory ≥0.50 recall @ ≤12 FA-saat şartını karşılamadı; en iyi ham recall
`lstm_recon` CUSUM/advisory 0.251 @ 15.40 FA-saat (bütçe dışı). **Ayrıca ÖNEMLİ bir
dürüstlük bulgusu ortaya çıktı ve B.5'e kaydedildi:** bu recall'ların büyük kısmı öğrenilmiş
zamansal örüntüden değil, kırpılmamış `RobustScaler` çıktısındaki ham genlik aykırı-
değerlerinden geliyor olabilir (bkz. B.5). Tam sayılar ve karşılaştırma: ADR-016.

**2026-07-10 güncellemesi (ML-16 Kol D + Kol U, ADR-017/ADR-018):** Aynı GÜNCEL
split_manifest ve AYNI resmi hatta iki model daha kablolandı — düz Dense-AE
(`src/ml/models/dense_autoencoder.py`) ve USAD (`src/ml/models/usad.py`). İkisi de Gate A
geçti, Gate B kaldı (en iyi ham recall: Dense-AE threshold/critical 0.215 @ 2.77 FA-saat;
USAD threshold/critical 0.217 @ 2.87 FA-saat — ikisi de bütçe dışı). B.5'teki genlik-
baskınlığı bulgusu mimariden bağımsız olduğu için ikisine de AYNEN uygulanıyor (ayrıca
teşhis edilmedi, ADR-016'nın teşhisi referans verildi). **Üç kolun (L/D/U) ortak sonucu:**
SEAD'de üç farklı derin-öğrenme mimarisi de denendi, üçü de operasyonel hedefi geçemedi ve
üçünün de ham recall kazancı aynı ölçekleme artefaktından geliyor — bu veri setinde mimari
seçimi şu anki ölçeklemeyle ayırt edici değil. "ML-8A/başka model ailesi (planlanmadı)"
maddesi artık tam anlamıyla kapalı: üç aile de denendi, sonuç dürüstçe negatif.

### C.2 — ML-9: kategori-eşleşmeli residual'lar Gate B/C'yi geçemedi (✅ Kapandı, ADR-009)

Position.Z'de dikey modül +0.021 recall (4/5 seed, gereken ≥0.05), Actuator Outputs+Controls'te
motor-simetri +0.024 (2/5 seed, gereken ≥3/5). Fusion en iyi hâliyle 0.222 recall/25.83 FA-saat —
kritik/advisory hiçbir bütçeyi karşılamadı. Ayrıntı: yukarıdaki ML-9 doğrulama raporu / H19-H21.
**ML-10 uygulandı:** actuator forecast-residual'i bu kategori baseline'ını anlamlı geçti;
irtifa dalı ve operasyonel fusion hedefi geçmedi (bkz. C.6 / ADR-010).

### C.3 — USAD, LSTM-AE'nin altında kaldı (✅ Kapandı, ML-3; SEAD'de güncel veriyle tekrar denendi, ADR-018)

ALFA 0.450, UAV 0.531 — az-veri rejiminde adversarial eğitim kararsız (beklenen sonuç). USAD
elendi, LSTM-AE sequence modeli olarak kaldı.

**2026-07-10 güncellemesi (ML-16 Kol U, ADR-018):** O zamanki karar "az veri" varsayımına
dayanıyordu; SEAD'in normal-uçuş havuzu o zamandan beri 324→899'a çıktığı için USAD SEAD'de
güncel veriyle YENİDEN denendi (`src/ml/models/usad.py`, 5-seed, resmi hatta kablolu). Gate
B yine kaldı (en iyi ham recall threshold/critical 0.217 @ 2.87 FA-saat, bütçe dışı) — ama
bu kez "az veri" değil B.5'teki genlik-baskınlığı artefaktı asıl sebep. Yani veri miktarı
artışı USAD'ı gerçek anlamda kurtarmadı; ham genlik-aykırı-değer duyarlılığı diğer iki
mimariyle (LSTM-AE, Dense-AE) aynı seviyede.

### C.4 — IF-füzyon heterojen normal havuzuna kırılgan (🟡 Gerçek ama küçük açık iş, B3)

ML-4. Rosbag'ten kurtarılan 2 normal uçuş nav_info'suz (rehberlik feature'ları imputed) — bu
uçuşlar val'e düştüğü seed'lerde eşik bozuluyor, seed-std 0.172→0.283'e çıktı. LSTM-AE bu
heterojenliğe karşı çok daha dayanıklı. **Kapatılmadı** — nav_info'suz uçuşları val'den hariç
tutmak veya ayrı bir eşik stratejisi denenebilir, henüz yapılmadı.

### C.5 — Eşik-üstü-oran skoru denendi, max'tan kötü çıktı (✅ Kapandı — hipotez reddedildi, H10)

ALFA ROC 0.742→0.417 (oran skoruyla). İmza az sayıda pencerede yaşadığından oran-tabanlı skor
sinyali seyreltiyor. Uçuş skoru max kalıyor — bu artık kapanmış bir tasarım kararı, tekrar
denenmeyecek.

### C.6 — ML-10: mechanical Gate B geçti, fusion Gate C kaldı (✅ Pilot tamamlandı, ADR-010)

Zero-shot `chronos_motor`, Actuator Outputs+Controls CUSUM/advisory recall'ını ML-9'un en iyi
adayına göre 0.205→0.390 yükseltti (+0.185, 4/5 seed); kategori düzeyinde gerçek ek sinyal
kanıtlandı. `chronos_dikey` Position.Z'de 0.096→0.023 ile geriledi. Sabit BASE+Chronos max-fusion
0.213 recall/23.92 FA-saat verdi ve operasyonel Gate C'yi geçmedi. **Kurtarma yapılmadı:** context,
quantile, fusion veya policy sonucu görerek ayarlanmadı; holdout kapalı kaldı. Pilot tamamlanmıştır,
ancak skor production/default adayı değildir.

### C.7 — ML-12: ince-modül Gate B geçti (B1+B2), fusion Gate C yine kaldı (✅ Tur tamamlandı, ADR-012)

ML-11'in seyrelme hipotezi ön-kayıtlı testte doğrulandı: tek-feature `itki_komutu`
(actuator_thrust_cmd) Actuator Outputs+Controls CUSUM/advisory recall'ını 0.205→**0.459**'a
taşıdı ve `chronos_motor`u da (0.390) geçti — kategori için bilinen en iyi skor artık bu.
3-feature ince varyant bile tek-feature'ın altında kaldı (seyrelme 3 feature'da ölçülür). AMA
füzyon yine operasyonel hedefe dönüşmedi (0.217/23.74 FA-saat): kök neden ölçüldü — ince modül
normal uçuşlarda 38.1 FA-saat bırakan bir **kategori uzmanı**; max-füzyon böyle bir uzmanı hedef
bütçede kullanamıyor. Kategori-bazlı ayrı alarm kanalı gibi bir mimari değişiklik ancak yeni
ön-kayıtlı fazla değerlendirilir. Holdout kapalı.

### C.8 — ML-13: iki ayrı alarm kanalı recall kazandırdı ama FA şartında kaldı (✅ Tur tamamlandı, ADR-013)

ML-12'nin H30 hipotezi ön-kayıtlı iki-kanal mimariyle test edildi: `sistem=existing_fusion`,
`mekanik=itki_komutu`, model eğitimi yok, kanal onset'leri 1 s kovada boolean OR. Recall gerçekten
arttı (`dengeli` CUSUM/advisory 0.217→0.291, +0.074, 5/5 seed; K-of-N/advisory 0.016→0.122,
+0.105, 5/5), ama FA şişirme freni tüm anlamlı kazanımları reddetti: CUSUM/advisory 23.70→44.70
FA-saat (1.89x), K-of-N/advisory 3.29→12.60 (3.83x). Gate C1 yine kaldı. Gate C2 de geçmedi:
mekanik kanal `dengeli` K-of-N/advisory'de Actuator O+C recall 0.541 üretti ama 12.60 FA-saat ile
12 sınırını aştı; `esit` threshold/advisory 0.498 / 11.16 ile recall eşiğinin az altında kaldı.
Sonuç: ayrı kanal mimarisi bu development protokolünde production ya da mekanik-monitör iddiası
üretmez; holdout kapalı kalır.

---

## D. Metodolojik / istatistiksel sınırlamalar

### D.1 — SEAD normal sınıfı heterojen (🔴 Yapısal sınır, BİLEREK "düzeltilmiyor")

398 normal uçuş yalnızca 64 farklı oturuma dağılıyor (development: 324 uçuş / 49 oturum; en
kalabalık oturum 21 uçuş) — gerçek bağımsız örneklem 398 değil ~64'e yakın. *(Sayı düzeltmesi,
görselleştirme fazı 2026-07-06: önceki kayıtlardaki "~32 oturum" tahmini yanlıştı; sayı
`session_of` ile split manifest'ten yeniden türetildi ve
`artifacts/viz/uav_sead/s1_portfolio/session_histogram.png` figürüyle belgelendi. Bulgunun özü
değişmez: bağımsız örneklem uçuş sayısının çok altında.)* Literatür ("Heterogeneous Normal Classes Pose a
Challenge for Anomaly Detection", OpenReview) bunu doğruluyor: normal sınıf heterojense
performans veri artsa bile kötüleşebilir (swamping/masking). **KRİTİK NOT:** bu maddenin
"çözümü" olarak session-koşullu/context-koşullu bir normallik modeli daha önce önerilmiş ve
kullanıcı tarafından haklı olarak REDDEDİLMİŞTİR (`feedback-anomaly-detection-principles`
belleği) — böyle bir koşullandırma yeni/görülmemiş oturumlarda modelin referans kaybetmesine
yol açar, sorunu çözmez gizler. Bu madde artık bir "yapılacak iş" değil, **kabul edilmiş, dürüst
bir sınır** olarak kapalı kalacak; ML-10'un forecast-residual yaklaşımı bunun BİR alternatifi
(session kimliğine değil, uçuşun kendi anlık geçmişine koşullanıyor) ama garantili çözüm değil.

### D.2 — Küçük-n istatistiksel kırılganlık (yaygın, bkz. A.1/A.4/A.6)

Birçok alt-tipte (rudder=4, aileron_rudder=1, gps_jamming=1, Battery=2, Velocity=2) tespit oranı
tek haneli/çift haneli uçuş sayısından geliyor. Kural: bu sayılar HER raporda n belirtilerek
sunulmalı, genelleme iddiası yapılmamalı. Bu bir "düzeltilecek hata" değil, kalıcı bir raporlama
disiplinidir.

### D.3 — Enjeksiyon şiddet taraması hiç yapılmadı (🟡 Gerçek açık iş, H12)

ML-3. UAV Attack'ta hiçbir sentetik enjeksiyon (freeze/bias/drift/gps_ramp/dropout) uçuş-düzeyi
alarm üretemedi (0.0). 2 m/s stealthy GPS ramp'i `gps_speed_residual`'i ancak normal-val
maksimumu kadar oynatıyor — yani 2→20 m/s şiddet taraması yapılmadan "stealthy spoofing
yakalanamıyor" da "yakalanıyor" da iddia edilemez. **Kapatılmadı**, devir listesinde (#7).

### D.4 — SITL/live domain karışması (🟡 Gerçek açık iş, H4)

`sinyal_kalitesi` modülü UAV Attack'ta sistematik ters (ROC 0.292). Hipotez: SITL loglarında GPS
sağlık alanları sabit-sıfır, canlı loglarda gerçek gürültü var; test setindeki canlı-normal uçuş
SITL-saldırı loglarından daha "anormal" skorlanıyor. **Doğrulanmadı ve düzeltilmedi** — devir
listesinde (#4), platform-bazlı ayrı kalibrasyon önerisi var.

---

## E. Operasyonel / karar katmanı sınırlamaları

### E.1 — SEAD zayıf kategoriler için hiçbir policy kritik/advisory bütçeyi karşılamıyor (🔴 Güncel durum)

ML-7'den ML-13'e kadar hiçbir fusion/policy konfigürasyonu Position.Z veya Actuator Outputs+Controls için
≤2 FA/saat @ ≥0.30 recall (kritik) ya da ≤12 FA/saat @ ≥0.50 recall (advisory) hedefini
karşılamadı. Kategori skoru iki turda üst üste belirgin iyileşti (ML-10 `chronos_motor` 0.390,
ML-12 `itki_komutu` **0.459** CUSUM/advisory recall) ve ML-13'te iki ayrı kanal recall'u daha da
yukarı taşıyabildi (`dengeli` CUSUM/advisory 0.291 birleşik recall), ama FA şartı bozuldu
(44.70 FA-saat). ML-13'ün sınırlı mekanik-monitör C2 iddiası da kıl payı kaldı: 0.541 recall /
12.60 FA-saat veya 0.498 / 11.16. Dolayısıyla kategori sinyali var; operasyonel bütün-sistem açığı
FA kayması ve alarm bütçesi problemine takılıyor. Mevcut development sonucuna tuning yapılmayacak;
yeniden açılma ancak yeni veri, farklı ön-kayıtlı FA-kalibrasyon hipotezi veya başka bağımsız
protokolle mümkün.

ML-15'in ön-kayıtlı session-jackknife drift kalibrasyonu tam 5-seed olarak tamamlandı (ADR-020).
Gate B yalnız 2/4 CUSUM hücresinde geçtiği için kaldı; Gate C de kaldı. En iyi bütçe-içi
advisory satırı `ml14_fusion` CUSUM 0.114504 recall / 8.414598 FA-saat, kritik satırı
0.043257 / 1.595978 verdi. Dolayısıyla FA-kalibrasyonu hesaplanmış ve sınanmış olsa da E.1'i
kapatmadı; aynı development sonucu üzerinde post-hoc policy ayarı yapılmaz.

### E.2 — Blind holdout (131 SEAD uçuşu) hiç açılmadı — gerçek "nihai" sayı yok (⚪ Bilinçli, doğru davranış)

Bu bir eksiklik değil kasıtlı bir metodoloji: development Gate B/C geçmeden holdout açılmaz.
Ama pratik sonucu şu: şu ana kadar raporlanan TÜM sayılar development tahminleridir, gerçek
"canlı" performans hâlâ ölçülmedi. Mentöre sunumda bu ayrım netleştirilmeli.

### E.3 — ALFA'da tek geçen operasyonel satır karar katmanına ait, modele değil (✅ Not edildi, ADR-008)

ALFA'da IF+CUSUM advisory 0.625 recall/7.91 FA-saat ile Gate C'yi geçiyor — ama bu LightGBM'in
başarısı değil, mevcut skorun ÜZERİNE oturan karar katmanının (CUSUM+bootstrap-ARL) katkısı.
Raporlarda bu ayrım karıştırılmamalı.

---

## F. Altyapı / kod sınırlamaları (denetim izi için)

### F.1 — `.gitignore`'daki ankorsuz `data/` deseni (✅ Kapandı, bu oturumda düzeltildi)

`src/ml/data/` da eşleşiyordu — `splits.py`/`scaling.py`/`windowing.py` ML pipeline'ın en
başından beri (70932ca) hiçbir commit'te YOKTU; temiz clone ImportError verirdi. `/data/`
olarak ankorlandı. **Ders:** yeni bir `.gitignore` deseni eklerken `git check-ignore -v` ile
istenmeyen eşleşme kontrolü standart olmalı.

### F.2 — Silver part-çoğalması tuzağı (⚪ Bilinçli — prosedürle yönetiliyor, H8.1)

`write_silver` her koşuda yeni immutable part ekliyor, `read_layer` HEPSİNİ okuyor. Silver'ı
iki kez çalıştırıp Gold üretmek satırları katlıyor. Düzeltme yok (tasarım gereği) — prosedür:
yeniden üretmeden önce `data/objectstore/silver/<kaynak>` silinmeli, VEYA `--local-bronze-dir`
modu (objectstore'a hiç yazmaz) kullanılmalı. Her yeni ajan/oturumda hatırlatılmalı.

### F.3 — 4 önceden var olan MinIO SDK test hatası (⚪ Bilinçli kapsam dışı, her oturumda görülüyor)

`tests/test_minio_retention.py` — `minio.lifecycleconfig.Filter` kurulu minio paket sürümüyle
uyuşmuyor (`ImportError`). ML çalışmasıyla ilgisi yok, hiçbir ML fazı bunu bozmadı/düzeltmedi.
Kurulu `minio` paketi güncellenirse veya kodun kendi sürüm-uyum katmanı yazılırsa kapanır —
bilerek ML kapsamının dışında tutuluyor.

### F.4 — MOMENT foundation modeli bu ortamda kurulamıyor (🔴 Yapısal sınır, bu oturumda bulundu)

`pip install --dry-run momentfm`: `AttributeError: module 'pkgutil' has no attribute
'ImpImporter'` — eski `numpy` pinni Python 3.14'te (bu makinenin sürümü) derlenemiyor. Chronos
(`chronos-forecasting==2.3.1`) aynı ortamda temiz kuruluyor — bkz. `docs/ML10_PLAN.md` §1.
**Kapatmak için:** ayrı bir Python 3.10/3.11 sanal ortamı gerekir; şu an orantısız/kapsam dışı.

---


### C.10 - RFLY-0 whole-flight proxy RFLY basarisini ustten tahmin etti (acik is, RFLY-1)

RFLY-0 official run anomalous Real-* ucuslarda olay araligi yerine tum ucusu anomalous saydi. RFLY-1 parser artik `rfly_ctrl_lxl` interval truth cikariyor ve evaluator whole-flight proxy'ye dusmeyi reddediyor. `rfly_ctrl_lxl_no_active_fault` olan 5 motor ucusu belirsiz/gecersiz test olarak haric tutuldu (4 development/test, 1 final holdout unopened). Duzeltilmis 5-seed official run tamamlandi: RFLY-only ve pooled R-A/R-B gecti ama R-C kaldi; eski 0.749 proxy sonucu gecersiz kabul ediliyor.

## G. Özet tablo — hızlı tarama için

| # | Madde | Kaynak | Durum | Kapatma yolu |
|---|---|---|---|---|
| A.1 | ALFA rudder/elevator/aileron_rudder n=1-4 | H6, bu oturum | 🔴 Yapısal | Yeni veri kaynağı (proje kapsamında yok) |
| A.2 | SEAD 8 çok-sınıflı uçuş dışlanıyor | bu oturum | 🟡 Açık iş | `select_flights()` gevşet + kategori-etiketleme |
| A.3 | SEAD 141 Uncategorized-only | bu oturum | ⚪ Bilinçli | — (doğru davranış) |
| A.4 | SEAD Battery/Vibration/Magnetometer n≤3 | H21 | ⚪ Bilinçli | Yeni fiziksel arıza verisi gerekir |
| A.5 | ping_dos 4/6 log tespit edilemiyor | H3 | 🔴 Yapısal | Inter-arrival feature (yeni parse kapsamı) |
| A.6 | gps_jamming/aileron_rudder n=1 | — | 🔴 Yapısal | Yeni veri kaynağı |
| A.7 | UAV Attack normal havuzu 6 log | — | 🔴 Yapısal | Yeni veri kaynağı |
| A.8 | `velocity_mps` null | ADR-005 | ✅ Telafi edildi | Türetilmiş hız kolonları kullanılıyor |
| A.9 | `battery_power_w` %96 NaN | H8.2 | ⚪ Bilinçli | Ayrı veri kalitesi analizi |
| B.1 | `alt_local_residual` altitude'da %0 dolu | bu oturum | 🔴 Yapısal | Kaynak topic eksik, veri büyütme çözmez |
| B.2 | `alt_baro_residual` %7 dolu | bu oturum | 🔴 Yapısal | Kaynak topic eksik |
| B.3 | EKF test-ratio ters sinyal | H14 | ✅ Kapandı | Füzyon dışı bırakıldı (reject-counter ile birleşmeden kullanılmaz) |
| B.5 | Reconstruction skoru kırpılmamış RobustScaler genliğine hâkim (3 mimaride doğrulandı) | ADR-016/017/018 | 🟡 Açık iş | Kırpmalı/robust ölçekleme veya genlik-normalize skor |
| C.1 | ML-8A LightGBM Gate B/C kaldı, LSTM/Dense-AE/USAD takibi üçü de Gate B kaldı | ADR-008, ADR-016/017/018 | ✅ Kapandı | Üç derin-öğrenme ailesi de koşuldu; sonuç negatif, dürüstçe raporlandı |
| C.2 | ML-9 kategori residual Gate B/C kaldı | ADR-009 | ✅ Kapandı | **ML-10 planlandı** (`docs/ML10_PLAN.md`) |
| C.6 | ML-10 mechanical Gate B geçti, fusion Gate C kaldı | H22-H24, ADR-010 | ✅ Pilot tamamlandı | Yeni bağımsız protokol olmadan tuning/holdout yok |
| C.7 | ML-12 ince-modül Gate B geçti (B1+B2), fusion Gate C kaldı | H29-H30, ADR-012 | ✅ Tur tamamlandı | ML-13 denendi; production'a dönüşmedi |
| C.8 | ML-13 iki kanal recall kazandı ama FA şartında kaldı | H31, ADR-013 | ✅ Tur tamamlandı | Yeni bağımsız FA-kalibrasyon/veri hipotezi olmadan tuning yok |
| C.9 | RFLY pooled-normal kategori kazandı ama operasyonel Gate R-C kaldı | ADR-014 | ✅ Tur tamamlandı | RFLY-only ayrı kabul; pooled iddia yok |
| C.3 | USAD < LSTM-AE (ALFA, az veri); SEAD'de güncel veriyle tekrar denendi, yine Gate B kaldı | ML-3, ADR-018 | ✅ Kapandı | — (karar verildi, iki kez doğrulandı) |
| C.4 | IF-füzyon heterojen normale kırılgan | B3 | 🟡 Açık iş | nav_info'suz uçuşları val'den ayır |
| C.5 | Oran-skoru max'tan kötü | H10 | ✅ Kapandı | — (hipotez reddedildi) |
| D.1 | SEAD normal sınıfı heterojen | literatür notu | 🔴 Yapısal, bilerek | **Session-koşullama ÖNERİLMEZ** (reddedildi) |
| D.2 | Küçük-n kırılganlık (yaygın) | A.1/A.4/A.6 | ⚪ Bilinçli | Her raporda n belirtilir |
| D.3 | Enjeksiyon şiddet taraması yok | H12 | 🟡 Açık iş | 2→20 m/s taraması (devir #7) |
| D.4 | SITL/live domain karışması | H4 | 🟡 Açık iş | Platform-bazlı kalibrasyon (devir #4) |
| E.1 | Kritik/advisory bütçe karşılanmıyor (SEAD zayıf) | H20, H24, H31 | 🔴 Güncel | Yeni veri veya ayrı ön-kayıtlı FA-kalibrasyon hipotezi |
| E.2 | Blind holdout hiç açılmadı | metodoloji | ⚪ Bilinçli | Gate B/C geçmeden açılmaz (doğru) |
| E.3 | ALFA'daki tek geçiş karar katmanına ait | ADR-008 | ✅ Not edildi | Raporda ayrım netleştirilir |
| F.1 | `.gitignore` `data/` bug'ı | bu oturum | ✅ Kapandı | Düzeltildi |
| F.2 | Silver part-çoğalması | H8.1 | ⚪ Bilinçli | Prosedürle yönetiliyor |
| F.3 | 4 MinIO SDK test hatası | — | ⚪ Bilinçli | Kapsam dışı, minio paketi güncellenirse kapanır |
| F.4 | MOMENT bu ortamda kurulamıyor | bu oturum | 🔴 Yapısal | Ayrı Python 3.10/3.11 ortamı (orantısız) |
| F.5 | ML-15 full matrix hesap maliyeti | ADR-015/020 | ✅ Kapandı | Jackknife süreç paralelliği; 5-seed 95.72 dk tamamlandı |

**Sayaç:** 34 madde — 8 🔴 yapısal sınır, 6 🟡 gerçek açık iş, 8 ⚪ bilinçli kapsam dışı,
12 ✅ kapandı/telafi edildi.
