# ADS-B Anomali Tespiti — Durum, Teşhis ve Yol Haritası

> **Tarih:** 2026-07-12
> **Amaç:** Şu ana kadarki bütün bulguları, koyduğumuz çıtaları, neyi yanlış
> yaptığımızın dürüst teşhisini ve bundan sonra **temelden başlayan** somut bir
> yol haritasını tek yerde toplamak. Gerçek ADS-B verisi diğer makinede olduğu
> için, o makinede çalışırken bu belge birincil referanstır.
>
> Bu belge yeni bir iddia içermez — hepsi `docs/decisions.md` (ADR-008…022),
> `archive/2026-07-10_legacy_non_adsb_ml/docs/ML_YETERSIZLIKLER_KAYDI.md` (34
> madde) ve `docs/anomali_türleri_adsb.md` içindeki kanıtlanmış bulguların
> konsolidasyonudur.

---

## 0. Bir cümlelik özet

Dört etiketli veri setinde de aynı duvara toslamamızın sebebi beceriksizlik
değil; **araştırma-kalitesi küçük veriyle, operasyonel-dağıtım seviyesinde bir
çıta kanıtlamaya çalıştık.** Doğru yol: çıtayı gerçekçi bir *temel* seviyeye
indirip oradan yukarı tırmanmak. **Temel olmadan çatı (çıta) olmaz.**

---

## 1. Nerede duruyoruz

- **9 farklı ML yöntemi** (IF, LightGBM, Chronos zero-shot, LSTM-AE, Dense-AE,
  USAD, genlik-normalize skor, drift-kalibreli füzyon) **4 etiketli veri setinde**
  denendi (ML-8A…ML-16, RFLY-0/1).
- **Hepsi operasyonel çıtayı (Gate C) geçemedi.** A ve B kapıları geçilebildi,
  kırmızı çizgi hep aynı yerde: operasyonel yanlış-alarm bütçesi.
- Eski hattın tamamı **2026-07-10'da arşivlendi**; yerine tek, temiz bir `adsb/`
  hattı başladı — öğrenen model yerine **aritmetik fizik-tutarlılık residual'ları**
  + kural-tabanlı sinyaller.
- **Gerçek 3 günlük ADS-B verisi** (`v2026.02.28`, `03.01`, `03.16`; toplam
  **256.150.550 satır**, 638 Silver parça) **diğer makinede**. Bu makinede veri
  yok — bu açık bir engel.
- İlk model turu gerçek veriyle koşuldu ama **yalnız pipeline doğrulaması** —
  headline recall / production kararı DEĞİL.

---

## 2. Dört veri seti — bulgular

Dört etiketli veri seti: **ALFA, UAV Attack, UAV-SEAD, RflyMAD (RFLY)**.
Beşincisi ADS-B — etiketsiz, şu anki iş.

### 2.1 ALFA — sabit-kanat İHA, motor/kumanda yüzeyi arızaları

| Boyut | **47 uçuş** (resmi külliyatın tamamı; rudder=4, elevator=2, aileron_rudder=1) |
|---|---|
| Araştırma metriği | LSTM-AE AUPRC **0.872** > IF 0.858 > LightGBM 0.843 → **literatürle uyumlu, iyi** |
| Operasyonel | IF+CUSUM advisory 0.625 recall / 7.91 FA-saat Gate C'yi *geçti* — ama bu modelin değil, **karar katmanının (CUSUM)** başarısı (E.3) |
| Ana sınır | 🔴 Yapısal: bazı arıza tipleri tek örnekli (n=1); genelleme iddiası yapılamaz |

### 2.2 UAV Attack — PX4 çok-rotorlu, GPS spoofing / jamming / DoS

| Boyut | Normal havuzu **yalnız 6 log**; gps_jamming n=1, aileron_rudder n=1 |
|---|---|
| Bulgu | ping_dos **6 logdan 4'ünde tespit edilemez** — ağ-katmanı saldırısı mevcut telemetriye hiç yansımıyor (A.5) |
| Bulgu | 2 m/s sinsi GPS-ramp'i `gps_speed_residual`'i ancak normal-val maksimumu kadar oynatıyor; şiddet taraması (2→20 m/s) hiç yapılmadı → "stealthy yakalanır/yakalanmaz" ikisi de iddia edilemez (D.3) |
| Ana sınır | 🔴 Yapısal: veri çok küçük + SITL/canlı log karışması sinyali sistematik ters çeviriyor (D.4) |

### 2.3 UAV-SEAD — dronlarda GPS-spoofing + irtifa/mekanik/konum anomalileri

| Boyut | ~1396 uçuş; **398 normal ama yalnız ~64 bağımsız oturum** → gerçek örneklem göründüğünden çok küçük (D.1) |
|---|---|
| En iyi kategori sinyali | `itki_komutu` (actuator_thrust_cmd) CUSUM/advisory recall 0.205 → **0.459** (iki turda gerçekten iyileşti: ML-10, ML-12) |
| Operasyonel | **Hiçbir** füzyon/policy Gate C'yi geçemedi — her recall kazancı FA'yı bütçenin üstüne taşıdı |
| Ana sınır | 🔴 Heterojen normal + 🟡 **genlik-baskınlığı artefaktı**: 3 farklı mimari (LSTM-AE/Dense-AE/USAD) birebir aynı sonucu verdi; eğitilmiş model ρ=0.964 rastgele-init ile, ρ=0.965 ham genlik ile korele — "örüntü" değil "büyüklük" öğrenilmiş (B.5) |

### 2.4 RflyMAD (RFLY) — gerçek+simüle dron, motor/sensör arızası

| Boyut | 490 gerçek uçuş (Motor 242, Sensör 197, **No_Fault yalnız 51**) |
|---|---|
| Kritik düzeltme | İlk sonuç 0.749 recall'du ama **tüm-uçuş "proxy" etiketiyle** (kolay soru). Gerçek olay-aralığı (`rfly_ctrl_lxl`) etiketiyle düzeltilince **0.526 recall / 22.28 FA-saat**'e düştü (C.10) |
| Operasyonel | Düzeltilmiş halde Gate R-C **kaldı** (ne RFLY-only ne pooled geçti) |
| Ana sınır | 🔴 Yapısal: temiz "normal" tavanı 51 uçuş |

### 2.5 ADS-B (adsb.lol) — etiketsiz, şu anki iş

| Boyut | 3 gün · **256.150.550 satır** · 638 Silver parça (ilk turda yalnız 10 parça / 3000 uçak kullanıldı) |
|---|---|
| İlk tur sonucu | Sentetik-bozulma ayrımı: `ground_speed_biased` **güçlü (2.3–10.75×)**; `vertical_rate_frozen` orta (1.03–2.28×); sinsi spoofing / track / altitude **çok zayıf (1.01–1.17×)**; USAD sayısal patladı (0/5) |
| Durum | Pipeline doğrulaması; hiçbiri headline/production adayı değil |

---

## 3. Çektiğimiz çıtalar (gates)

Her deney üç kapıdan geçmek zorundaydı. Bunlar sonuç görülmeden donduruldu.

| Kapı | Ne soruyor | Sonuç |
|---|---|---|
| **Gate A** — güvenlik/determinizm | Kör test setine sızıntı yok mu? Tekrar üretilebilir mi? | ✅ Her zaman geçti |
| **Gate B** — anlamlı kazanç | Yeni yöntem, öncekini önceden-belirlenmiş bir farkla ve yeterli seed'de geçti mi? | ◐ Bazen geçti (ML-10, ML-12, RFLY) |
| **Gate C** — operasyonel hedef | **critical:** recall ≥ 0.30 @ ≤ 2 yanlış-alarm/saat · **advisory:** recall ≥ 0.50 @ ≤ 12 YA/saat | ❌ **Hiçbir veri setinde, hiçbir yöntemle geçilmedi** |

Yeni ADS-B hattının kendi çıtası daha da sıkı: **recall ≥ 0.70 @ FA ≤ 0.05**.

**Dikkat:** kırmızı çizgi hep Gate C. Dört veri setinde de aynı yerde takıldık —
operasyonel yanlış-alarm bütçesi.

---

## 4. Teşhis — neyi yanlış yaptık

Sistemik bir şey var, ama "beceriksizlik" değil. İki gerçek hata + bir büyük
yanılsama. En önemliden başlayarak:

### 4.1 (En kritik) — Yanlış metriğe göre kendimizi yargıladık

Literatürdeki "%95 başarı" rakamları **AUC/F1** metriğidir: düzenli, dengeli test
setinde. **Biz o metrikte camiayı zaten yakalıyoruz** — ALFA'da LSTM-AE 0.872
AUPRC, yayınlanmış sonuçlarla aynı ligde. Bizim takıldığımız Gate C ise
**"saatte kaç yanlış alarm"** metriği — ki makaleler bunu **genelde hiç
raporlamaz.**

> Yani "camia yapıyor biz yapamıyoruz" doğru değil: camianın ölçtüğü şeyi biz de
> yapıyoruz. Biz kendimize camianın ölçmediği, çok daha zor bir **operasyonel**
> çıta koyduk ve ona takıldık.

### 4.2 Küçük/heterojen etiketli veri o çıtayı kaldıramaz

ALFA 47 uçuş, UAV Attack 6 normal log, SEAD ~64 bağımsız oturum, RFLY 51 temiz
uçuş. "Saatte 2 yanlış alarm" gibi operasyonel bir iddia için bu veri **yapısal
olarak çok küçük.** Etiket olması kurtarmıyor: etiket "hangi uçuş bozuk" der; biz
"bozukluk hangi saniyede başladı ve normal uçuşlarda saatte kaç kez boş yere
öttük" diye soruyoruz. Etiketler bir soruya, bizim skorumuz başka bir soruya ait.
(RFLY bunu kanıtladı: kolay soruya 0.749, gerçek soruya 0.526.)

### 4.3 Novelty-detection, heterojen normalde çöküyor

"Sadece normali öğren, sapanı yakala" yaklaşımı, **normalin kendisi dağınıksa**
çöküyor. Literatür doğruluyor ("Heterogeneous Normal Classes Pose a Challenge for
Anomaly Detection"): normal çeşitliyse, veri **artsa bile** performans
kötüleşebilir (swamping/masking). Dört veri setinin normal havuzu da küçük ve
heterojen. Model net bir "normal" referansı kuramayınca her şey biraz anormal
görünüyor → yanlış alarm patlıyor.

### 4.4 Genlik-baskınlığı — gerçek teknik hata (ama yakalandı)

Kırpılmamış `RobustScaler` yüzünden derin modeller "örüntü" değil "büyüklük"
öğrendi. Bir avuç aşırı-genlikli pencere (gerçek GPS-sıçraması + "normal" etiketli
ama donmuş-GPS sentinel'i) her autoencoder'ın skoruna hâkim oldu. Bu gerçek bir
hataydı — ama fark edip dürüstçe raporladık, üstünü örtmedik. ML-16-N: genliği
temizleyince altından operasyonel sinyal çıkmadı.

### 4.5 Disiplin faturası (dürüstlük)

"Geçemedi" satırlarının çoğu aslında **disiplin.** Kör test setini açmadık, sonucu
görüp parametre oynamadık. Hile yapan bir hat bu rakamları daha parlak gösterirdi.
Tablodaki başarısızlığın bir kısmı beceriksizlik değil, **dürüstlüğün faturası.**

---

## 5. Asıl ders — çıtayı çok yükseğe koyduk

En iyisini herkes ister. Ama şu aşamada **basic/temel seviyeyi yapmadan en üste
inemeyiz.** Doğrudan operasyonel Gate C'yi (saatte 2 yanlış alarm) hedeflemek,
daha ortada çalışan bir temel varken çatıyı aramaktı.

**Çözüm: kademeli çıta.** Tek bir "her şeyi kesen" çıta yerine, temelden çatıya
doğru dört seviye. Her seviye geçilmeden bir üstü hedeflenmez:

| Seviye | Çıta | Ne kanıtlar |
|---|---|---|
| **S0 — Temel** | Temiz uçuşta residual ≈ 0; sentetik bozulmada `corrupt > clean` (5/5 senaryo) | Boru hattı ve fizik doğru çalışıyor |
| **S1 — Tek kanal** | Tek net kanal (hız residual) tek bir anomali türünü basit eşikle ayırt ediyor (ROC/AUC anlamlı) | Gerçek sinyal var, ölçülebilir |
| **S2 — Kesin sinyaller** | Kural-tabanlı sinyaller (acil squawk 7500/7600/7700) çalışıyor + dashboard'da | Sıfır-ML, kesin, operasyonel değer |
| **S3 — Çatı** | Operasyonel Gate C (critical ≥0.30 @ ≤2 FA/saat, advisory ≥0.50 @ ≤12) | Dağıtıma hazır |

Şu an S0'dayız. Önce S0'ı gerçek veride kapatıp S1/S2'ye geçeceğiz; S3
(operasyonel çıta) **ancak S1/S2 sağlamken** gündeme gelir.

---

## 6. Yol haritası — temelden çatıya

### S0 — Temel (gerçek veride kapatılacak, ML gerektirmez)
1. **Galeri (Faz 0 madde 3, hâlâ yazılmadı):** ham zaman-serisi + harita görselleri
   (`scripts/make_adsb_visualizations.py`). Gözle "normal uçuş nasıl görünür"ü görmeden
   model çıtası koyulmaz.
2. **Residual doğrulaması:** temiz gerçek uçuşlarda `speed_residual`/`vertical_rate_residual`
   ≈ 0 mı? (Fiziğin doğruluğu.)
3. **Sentetik bozulma tekrarı** tam-hacimde, S0 kapı ölçütüyle (`corrupt > clean` 5/5).

### S1 — Tek kanal (en güçlü sinyalden başla)
4. **Hız residual eşiği:** `ground_speed_biased` ve `speed_residual` zaten en güçlü
   kanal (2.3–10.75×). Bunu tek bir anomali türü için basit eşikle ROC/AUC olarak
   ölç. Öğrenen model şart değil — aritmetik residual + eşik.
5. **Kör-holdout tanımı:** birkaç günlük trafiği ayır, hiç dokunma (ADSB-1'de tanımlı).

### S2 — Kural-tabanlı kesin sinyaller (bedava kazanım)
6. **Acil squawk kanalı (7500/7600/7700):** doğrudan `squawk` alanından, ML yok.
   En ucuz, en kesin sinyal ailesi; **henüz dashboard'a bağlanmadı.** Bu, veri
   gerektirmeden bile yazılabilir (kod hazır, gerçek veride doğrulanır).
7. ICAO24 çakışması, bütünlük metriği (`nic/nac_p/sil`) düşüşü — yine kural tabanlı.

### S3 — Çatı (yalnız S1/S2 sağlamken)
8. Fizik-residual kanallarının füzyonu + karar katmanı, operasyonel FA bütçesiyle.
9. En zor hedef — sinsi/kademeli spoofing (B.8, şu anki en zayıf nokta 1.01×) —
   residual'ları CUSUM ile birikimli izleyerek.

---

## 7. Diğer bilgisayardan çalışma planı

Gerçek veri o makinede; kod ve bu yol haritası git'te. Sıra:

1. **`git pull`** — bu belge ve güncel `adsb/` hattı gelir.
2. **Veriyi doğrula:** `data/objectstore/silver/adsblol_historical` yerinde mi
   (638 parça)? Yoksa 3 ham tar'dan `STORAGE_BACKEND=local` ile yeniden parse et.
3. **S0'ı kapat:** galeri + residual doğrulaması + tam-hacim sentetik tekrar
   (§6 madde 1-3). **Onay kapısı:** S0 raporu görülmeden model ailesi/çıta seçilmez.
4. **S2 kural kanalını yaz** (squawk) — veri gerektirmeyen parça bu makinede de
   yazılabilir; gerçek veride yalnız doğrulanır.
5. **S1'e geç:** hız residual eşiği + kör-holdout tanımı.
6. Her seviye kapandığında bu belgeyi güncelle (durum + geçilen çıta).

> **Kural:** çıtayı seviye atlayarak değil, sırayla yükseltiyoruz. S3 (operasyonel
> Gate C) bir hedef, bir *başlangıç* şartı değil. Temel sağlamsa çatı gelir.

---

## Kaynaklar
- `docs/decisions.md` — ADR-008…022 (tüm model turları, gate kararları)
- `archive/2026-07-10_legacy_non_adsb_ml/docs/ML_YETERSIZLIKLER_KAYDI.md` — 34 madde konsolide yetersizlik kaydı
- `docs/anomali_türleri_adsb.md` — 37 maddelik anomali taksonomisi
- `adsb/reports/measurability_table.md` — gerçek satır-düzeyi kolon kapsaması
- `adsb/README.md` — sıfırdan başlangıç sözleşmesi + Aşama 0 durumu
