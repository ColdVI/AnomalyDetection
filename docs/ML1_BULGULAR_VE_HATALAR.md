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
