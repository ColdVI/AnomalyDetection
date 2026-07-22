# Proje Süreci ve Sonuçları — Durum, Skor Defteri, Ön-kayıt ve Sunum Notları

> **Bu dosya, bireysel ML anomali-tespiti fizibilite çalışmasının süreç, teşhis
> ve toplu sonuç belgelerini tek yerde toplar.** Daha önce ayrı duran belgeler
> buraya birleştirildi; içerik ve sayılar değiştirilmeden korundu. Orijinal ayrık
> dosyalar `arsiv` branch'inde durmaya devam ediyor.

## Yönetici özeti

Çalışma boyunca beş veri seti (ALFA, UAV Attack, UAV-SEAD, RflyMAD ve etiketsiz
ADS-B) üzerinde ~40 farklı model/konfigürasyon denendi. **Ortak sonuç: hiçbir
konfigürasyon operasyonel hedefi (Gate C — recall + saatlik yanlış-alarm birlikte)
tam anlamıyla karşılamadı.**

Kök teşhis tek bir cümlede: *araştırma-kalitesi küçük/heterojen veriyle,
operasyonel-dağıtım seviyesinde bir çıta kanıtlanmaya çalışıldı.* Literatürün ölçtüğü
AUC/AUPRC metriğinde camia seviyesi yakalandı (ALFA'da LSTM-AE AUPRC 0,872); asıl
duvar, makalelerin çoğunun hiç raporlamadığı "saatte kaç yanlış alarm" bütçesiydi.

İki en güncel disiplinli çalışma açık NO-GO ile kapandı; ikisinde de "sinyal var
ama operasyonel eşik kurulamadı" ayrımı belgelendi:
- **UAV GNSS Integrity v1** — tam NO-GO (hiçbir yöntem kritik+advisory'yi birlikte
  tutturamadı).
- **RESIDUAL-V1** — NO-GO; sanity kapıları (S-1/S-3) PASS ama kalibrasyon için
  yeterli bağımsız normal uçuş-saati yok. (Ayrıntı: `RESIDUAL_V1.md`.)

**2026-07-22 güncellemesi:** RflyMAD-Full v2 hattında ayrı, çok daha büyük bir
turda (6605 uçuş, düzeltilmiş truth) 6-adaylı preregistered Wind/Real
robustness sweep (R1-R4, W1, W2) ve development-only TCN sweep çalıştırıldı —
**hiçbiri kendi kapısını geçmedi**; aynı desen burada da tekrarlandı: sinyal
kısmen var (Real macro recall %14→%28'e çıktı) ama genel recall/FA maliyeti
ve Wind alarm yükü operasyonel/araştırma eşiğinin altında kaldı. Ayrıntı:
bölüm 6 ve `gecmis_calismalar/RFLYMAD/raporlar/RFLYMAD_V2_SONRAKI_ADIMLAR_20260722.md`.

## İçindekiler

1. Durum, Teşhis ve Yol Haritası
2. Proje Skor ve Başarısızlık Defteri (tüm hatlar, tüm sayılar)
3. UAV GNSS Integrity v1 — Ön-kayıt
4. Sunum 1–2. Hafta Revizyon Talimatları
5. Sunum 3. Hafta Revizyon Talimatları
6. RflyMAD-Full v2 — 2026-07-22 Güncellemesi


---

## ADS-B Anomali Tespiti — Durum, Teşhis ve Yol Haritası

> **Tarih:** 2026-07-12
> **Amaç:** Şu ana kadarki bütün bulguları, koyduğumuz çıtaları, neyi yanlış
> yaptığımızın dürüst teşhisini ve bundan sonra **temelden başlayan** somut bir
> yol haritasını tek yerde toplamak. Gerçek ADS-B verisi diğer makinede olduğu
> için, o makinede çalışırken bu belge birincil referanstır.
>
> Bu belge yeni bir iddia içermez — hepsi `docs/decisions.md` (ADR-008…022),
> `gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML_YETERSIZLIKLER_KAYDI.md` (34
> madde) ve `docs/anomali_türleri_adsb.md` içindeki kanıtlanmış bulguların
> konsolidasyonudur.

---

### 0. Bir cümlelik özet

Dört etiketli veri setinde de aynı duvara toslamamızın sebebi beceriksizlik
değil; **araştırma-kalitesi küçük veriyle, operasyonel-dağıtım seviyesinde bir
çıta kanıtlamaya çalıştık.** Doğru yol: çıtayı gerçekçi bir *temel* seviyeye
indirip oradan yukarı tırmanmak. **Temel olmadan çatı (çıta) olmaz.**

---

### 1. Nerede duruyoruz

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

### 2. Dört veri seti — bulgular

Dört etiketli veri seti: **ALFA, UAV Attack, UAV-SEAD, RflyMAD (RFLY)**.
Beşincisi ADS-B — etiketsiz, şu anki iş.

#### 2.1 ALFA — sabit-kanat İHA, motor/kumanda yüzeyi arızaları

| Boyut | **47 uçuş** (resmi külliyatın tamamı; rudder=4, elevator=2, aileron_rudder=1) |
|---|---|
| Araştırma metriği | LSTM-AE AUPRC **0.872** > IF 0.858 > LightGBM 0.843 → **literatürle uyumlu, iyi** |
| Operasyonel | IF+CUSUM advisory 0.625 recall / 7.91 FA-saat Gate C'yi *geçti* — ama bu modelin değil, **karar katmanının (CUSUM)** başarısı (E.3) |
| Ana sınır | 🔴 Yapısal: bazı arıza tipleri tek örnekli (n=1); genelleme iddiası yapılamaz |

#### 2.2 UAV Attack — PX4 çok-rotorlu, GPS spoofing / jamming / DoS

| Boyut | Normal havuzu **yalnız 6 log**; gps_jamming n=1, aileron_rudder n=1 |
|---|---|
| Bulgu | ping_dos **6 logdan 4'ünde tespit edilemez** — ağ-katmanı saldırısı mevcut telemetriye hiç yansımıyor (A.5) |
| Bulgu | 2 m/s sinsi GPS-ramp'i `gps_speed_residual`'i ancak normal-val maksimumu kadar oynatıyor; şiddet taraması (2→20 m/s) hiç yapılmadı → "stealthy yakalanır/yakalanmaz" ikisi de iddia edilemez (D.3) |
| Ana sınır | 🔴 Yapısal: veri çok küçük + SITL/canlı log karışması sinyali sistematik ters çeviriyor (D.4) |

#### 2.3 UAV-SEAD — dronlarda GPS-spoofing + irtifa/mekanik/konum anomalileri

| Boyut | ~1396 uçuş; **398 normal ama yalnız ~64 bağımsız oturum** → gerçek örneklem göründüğünden çok küçük (D.1) |
|---|---|
| En iyi kategori sinyali | `itki_komutu` (actuator_thrust_cmd) CUSUM/advisory recall 0.205 → **0.459** (iki turda gerçekten iyileşti: ML-10, ML-12) |
| Operasyonel | **Hiçbir** füzyon/policy Gate C'yi geçemedi — her recall kazancı FA'yı bütçenin üstüne taşıdı |
| Ana sınır | 🔴 Heterojen normal + 🟡 **genlik-baskınlığı artefaktı**: 3 farklı mimari (LSTM-AE/Dense-AE/USAD) birebir aynı sonucu verdi; eğitilmiş model ρ=0.964 rastgele-init ile, ρ=0.965 ham genlik ile korele — "örüntü" değil "büyüklük" öğrenilmiş (B.5) |

#### 2.4 RflyMAD (RFLY) — gerçek+simüle dron, motor/sensör arızası

| Boyut | 490 gerçek uçuş (Motor 242, Sensör 197, **No_Fault yalnız 51**) |
|---|---|
| Kritik düzeltme | İlk sonuç 0.749 recall'du ama **tüm-uçuş "proxy" etiketiyle** (kolay soru). Gerçek olay-aralığı (`rfly_ctrl_lxl`) etiketiyle düzeltilince **0.526 recall / 22.28 FA-saat**'e düştü (C.10) |
| Operasyonel | Düzeltilmiş halde Gate R-C **kaldı** (ne RFLY-only ne pooled geçti) |
| Ana sınır | 🔴 Yapısal: temiz "normal" tavanı 51 uçuş |

#### 2.5 ADS-B (adsb.lol) — etiketsiz, şu anki iş

| Boyut | 3 gün · **256.150.550 satır** · 638 Silver parça (ilk turda yalnız 10 parça / 3000 uçak kullanıldı) |
|---|---|
| İlk tur sonucu | Sentetik-bozulma ayrımı: `ground_speed_biased` **güçlü (2.3–10.75×)**; `vertical_rate_frozen` orta (1.03–2.28×); sinsi spoofing / track / altitude **çok zayıf (1.01–1.17×)**; USAD sayısal patladı (0/5) |
| Durum | Pipeline doğrulaması; hiçbiri headline/production adayı değil |

---

### 3. Çektiğimiz çıtalar (gates)

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

### 4. Teşhis — neyi yanlış yaptık

Sistemik bir şey var, ama "beceriksizlik" değil. İki gerçek hata + bir büyük
yanılsama. En önemliden başlayarak:

#### 4.1 (En kritik) — Yanlış metriğe göre kendimizi yargıladık

Literatürdeki "%95 başarı" rakamları **AUC/F1** metriğidir: düzenli, dengeli test
setinde. **Biz o metrikte camiayı zaten yakalıyoruz** — ALFA'da LSTM-AE 0.872
AUPRC, yayınlanmış sonuçlarla aynı ligde. Bizim takıldığımız Gate C ise
**"saatte kaç yanlış alarm"** metriği — ki makaleler bunu **genelde hiç
raporlamaz.**

> Yani "camia yapıyor biz yapamıyoruz" doğru değil: camianın ölçtüğü şeyi biz de
> yapıyoruz. Biz kendimize camianın ölçmediği, çok daha zor bir **operasyonel**
> çıta koyduk ve ona takıldık.

#### 4.2 Küçük/heterojen etiketli veri o çıtayı kaldıramaz

ALFA 47 uçuş, UAV Attack 6 normal log, SEAD ~64 bağımsız oturum, RFLY 51 temiz
uçuş. "Saatte 2 yanlış alarm" gibi operasyonel bir iddia için bu veri **yapısal
olarak çok küçük.** Etiket olması kurtarmıyor: etiket "hangi uçuş bozuk" der; biz
"bozukluk hangi saniyede başladı ve normal uçuşlarda saatte kaç kez boş yere
öttük" diye soruyoruz. Etiketler bir soruya, bizim skorumuz başka bir soruya ait.
(RFLY bunu kanıtladı: kolay soruya 0.749, gerçek soruya 0.526.)

#### 4.3 Novelty-detection, heterojen normalde çöküyor

"Sadece normali öğren, sapanı yakala" yaklaşımı, **normalin kendisi dağınıksa**
çöküyor. Literatür doğruluyor ("Heterogeneous Normal Classes Pose a Challenge for
Anomaly Detection"): normal çeşitliyse, veri **artsa bile** performans
kötüleşebilir (swamping/masking). Dört veri setinin normal havuzu da küçük ve
heterojen. Model net bir "normal" referansı kuramayınca her şey biraz anormal
görünüyor → yanlış alarm patlıyor.

#### 4.4 Genlik-baskınlığı — gerçek teknik hata (ama yakalandı)

Kırpılmamış `RobustScaler` yüzünden derin modeller "örüntü" değil "büyüklük"
öğrendi. Bir avuç aşırı-genlikli pencere (gerçek GPS-sıçraması + "normal" etiketli
ama donmuş-GPS sentinel'i) her autoencoder'ın skoruna hâkim oldu. Bu gerçek bir
hataydı — ama fark edip dürüstçe raporladık, üstünü örtmedik. ML-16-N: genliği
temizleyince altından operasyonel sinyal çıkmadı.

#### 4.5 Disiplin faturası (dürüstlük)

"Geçemedi" satırlarının çoğu aslında **disiplin.** Kör test setini açmadık, sonucu
görüp parametre oynamadık. Hile yapan bir hat bu rakamları daha parlak gösterirdi.
Tablodaki başarısızlığın bir kısmı beceriksizlik değil, **dürüstlüğün faturası.**

---

### 5. Asıl ders — çıtayı çok yükseğe koyduk

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

### 6. Yol haritası — temelden çatıya

#### S0 — Temel (gerçek veride kapatılacak, ML gerektirmez)
1. **Galeri (Faz 0 madde 3, hâlâ yazılmadı):** ham zaman-serisi + harita görselleri
   (`scripts/make_adsb_visualizations.py`). Gözle "normal uçuş nasıl görünür"ü görmeden
   model çıtası koyulmaz.
2. **Residual doğrulaması:** temiz gerçek uçuşlarda `speed_residual`/`vertical_rate_residual`
   ≈ 0 mı? (Fiziğin doğruluğu.)
3. **Sentetik bozulma tekrarı** tam-hacimde, S0 kapı ölçütüyle (`corrupt > clean` 5/5).

#### S1 — Tek kanal (en güçlü sinyalden başla)
4. **Hız residual eşiği:** `ground_speed_biased` ve `speed_residual` zaten en güçlü
   kanal (2.3–10.75×). Bunu tek bir anomali türü için basit eşikle ROC/AUC olarak
   ölç. Öğrenen model şart değil — aritmetik residual + eşik.
5. **Kör-holdout tanımı:** birkaç günlük trafiği ayır, hiç dokunma (ADSB-1'de tanımlı).

#### S2 — Kural-tabanlı kesin sinyaller (bedava kazanım)
6. **Acil squawk kanalı (7500/7600/7700):** doğrudan `squawk` alanından, ML yok.
   En ucuz, en kesin sinyal ailesi; **henüz dashboard'a bağlanmadı.** Bu, veri
   gerektirmeden bile yazılabilir (kod hazır, gerçek veride doğrulanır).
7. ICAO24 çakışması, bütünlük metriği (`nic/nac_p/sil`) düşüşü — yine kural tabanlı.

#### S3 — Çatı (yalnız S1/S2 sağlamken)
8. Fizik-residual kanallarının füzyonu + karar katmanı, operasyonel FA bütçesiyle.
9. En zor hedef — sinsi/kademeli spoofing (B.8, şu anki en zayıf nokta 1.01×) —
   residual'ları CUSUM ile birikimli izleyerek.

---

### 7. Diğer bilgisayardan çalışma planı

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

### Kaynaklar
- `docs/decisions.md` — ADR-008…022 (tüm model turları, gate kararları)
- `gecmis_calismalar/_ortak/legacy_ml_kutuphanesi/docs/ML_YETERSIZLIKLER_KAYDI.md` — 34 madde konsolide yetersizlik kaydı
- `docs/anomali_türleri_adsb.md` — 37 maddelik anomali taksonomisi
- `adsb/reports/measurability_table.md` — gerçek satır-düzeyi kolon kapsaması
- `adsb/README.md` — sıfırdan başlangıç sözleşmesi + Aşama 0 durumu


---

## Proje Skor ve Başarısızlık Defteri — tüm hatlar, tüm sayılar

Tarih: 2026-07-17 · Kapsam: projenin başından bugüne kadar **her** Gate/kapı
FAIL'i, NO-GO'su, hedefi tutturamayan sonucu — dataset, kanal/kategori, algoritma
ve gerçek metrik değerleriyle. Kaynak: `docs/decisions.md` (ADR'ler), bellek
kayıtları ve bu oturumda bağımsız doğrulanan RESIDUAL-V1/GNSS çalışması.
Başarılı/kabul edilen sonuçlar da bağlam için işaretlendi (✅), ama odak FAIL/NO-GO.

**Okuma notu:** "Gate B" = yeni yöntem eskisini belirli bir marjla geçmeli;
"Gate C" = operasyonel bütçe (recall + saatlik yanlış alarm birlikte); "S-1/S-2/S-3/S-4"
= RESIDUAL-V1'in threshold-bağımsız sanity testleri. Her satır bir ADR'ye veya bu
oturumdaki doğrulamaya bağlanabilir.

---

### 1. Legacy ML hattı — ALFA / UAV Attack / UAV-SEAD (arşivlendi 2026-07-10)

`archive/2026-07-10_legacy_non_adsb_ml/` altında. 17 fazlık ilk yaklaşım, sonunda
tamamen arşivlenip ADS-B hattına geçildi (bkz. §3).

#### 1.1 Temel modeller (ML-1..ML-3)

| Dataset | Kanal/Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| ALFA | tüm satırlar | Monolitik satır-bazlı Isolation Forest | ROC | 0.50 | FAIL (rastgele) |
| UAV Attack | tüm satırlar | Monolitik satır-bazlı IF | ROC | 0.21 | FAIL (rastgeleden kötü) |
| UAV Attack | ping_dos | Modüler IF-füzyon | uçuş-recall | 4/6 log tespit edilemedi | FAIL — imza fiziğe yansımıyor |
| ALFA | genel | LSTM-AE (10dk azveri) | uçuş-ROC | 0.731 vs IF-füzyon 0.833 | FAIL — IF'i geçemedi |
| UAV-SEAD | kalibrasyonsuz transfer | IF-füzyon | normal FA | 1.00 | FAIL — tam yanlış alarm |
| ALFA | rudder_fault (n=4) | CUSUM | "1.00 tespit" | n=1'den — istatistiksel anlamsız | FAIL (istatistiksel tiyatro, sonradan reddedildi) |
| ALFA/UAV | H10 hipotezi (max-pencere skoru) | — | — | reddedildi | FAIL — oran-skoru sinyali seyreltiyor |
| ALFA/UAV | drift/bias/freeze enjeksiyonu | USAD | recall | 0.45 / 0.53 (< LSTM-AE) | FAIL — elendi |

#### 1.2 Veri büyütme sonrası (ML-4..ML-5)

| Dataset | Kanal | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD | ranges (adil) | IF-füzyon | satır-ROC | 0.474 | FAIL — UAV-Attack feature'ları SEAD'i yakalamıyor (H13) |
| UAV-SEAD | EKF test-ratio | IF-füzyon | korelasyon yönü | TERS sinyal (0.354) | FAIL — arızada ölçüm reddi innovation'ı bastırıyor (H14) |
| UAV-SEAD | altitude | tip tespiti | recall | 0.27 | FAIL — zayıf |

#### 1.3 Causal düzeltme sonrası gerçek sayılar (ML-6/7)

| Dataset | Kanal | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| ALFA | alt_error_cusum | CUSUM (causal) | uçuş-ROC | 0.878 → **0.611** | FAIL — "en güçlü sinyal" sanılan kanal causal ölçümde çöktü |
| ALFA | tüm kanallar | event-onset recall | recall | 0.594 → **0.194–0.224** | FAIL — eski point-adjust benzeri şişirme düzeltildi |

#### 1.4 LightGBM / kategori residual / iki-kanal (ML-8A, ML-9, ML-13)

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD | genel | LightGBM window | AUPRC | 0.349 vs IF 0.385 | **Gate B kaldı** |
| UAV-SEAD | tüm policy | LightGBM+decision | kritik/advisory recall+FA | hiçbiri hedefi karşılamadı | **Gate C kaldı** |
| ALFA | sabit reçete | LightGBM | AUPRC | 0.843 < IF 0.858 < LSTM-AE 0.872 | FAIL — en zayıf model |
| UAV-SEAD | Position.Z | dikey_tutarlilik (ML-9) | recall farkı | +0.021 (4/5 seed, <0.05 baraj) | **Gate B kaldı** — büyüklük yetersiz |
| UAV-SEAD | Actuator O+C | motor_simetrisi (ML-9) | recall farkı | +0.024 (2/5 seed, <3/5 kararlılık) | **Gate B kaldı** |
| UAV-SEAD | tüm kategoriler | ML9-fusion | CUSUM/advisory recall/FA | 0.222 / 25.83 FA-saat | **Gate C kaldı** |
| UAV-SEAD | mekanik+sistem | iki-kanal (ML-13) `dengeli` | CUSUM/advisory recall/FA | 0.217→0.291 (+0.074) ama FA 23.70→**44.70** (1.89×) | **Gate B kaldı** — kazanım FA şişirerek satın alındı |
| UAV-SEAD | | | CUSUM/critical FA | 12.98→27.49 (2.12×) | aynı |
| UAV-SEAD | | | K-of-N/advisory FA | 3.29→**12.60** (3.83×) | aynı |

#### 1.5 Chronos zero-shot ve ince modül (ML-10, ML-12) — Gate B geçen NADİR örnekler

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD | Actuator O+C | Chronos zero-shot (`chronos_motor`) | CUSUM/advisory recall | 0.205→**0.390** (+0.185, 4/5 seed) | ✅ Gate B geçti |
| UAV-SEAD | Position.Z | Chronos (`chronos_dikey`) | recall | 0.096→**0.023** | ❌ REDDEDİLDİ — 6/6 kombinasyon negatif |
| UAV-SEAD | tüm kategoriler | `ml10_fusion` | CUSUM/advisory recall/FA | 0.213 / 23.92 FA-saat (hedef ≥0.50/≤12) | **Gate C kaldı** |
| UAV-SEAD | Actuator O+C | ince-modül `itki_komutu` (tek feature) | CUSUM/advisory recall | 0.205→**0.459** (+0.254, en iyi kategori sonucu) | ✅ Gate B geçti (hem B1 hem B2) |
| UAV-SEAD | | `itki_komutu` | normal uçuşlarda FA | **38.1 FA-saat** | FAIL nedeni — yüksek-FA uzman kanal |
| UAV-SEAD | tüm kategoriler | `ml12_fusion_itki` | CUSUM/advisory recall/FA | 0.217 / 23.74 FA-saat | **Gate C kaldı** (yine) |

#### 1.6 Veri büyütme + derin öğrenme yeniden deneme (ML-14, ML-16)

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| UAV-SEAD (899 normal) | tüm | `ml14_fusion` CUSUM/advisory | recall/FA | 0.126 / 9.95 FA-saat (eski 0.21/23.6) | FAIL — recall/precision değiş-tokuşu, hedef karşılanmadı |
| UAV-SEAD | | CUSUM/critical | recall/FA | 0.043 / 1.60 FA-saat | FAIL |
| UAV-SEAD | | K-of-N/advisory | recall/FA | 0.056 / 2.28 FA-saat | bütçe içi ama düşük recall |
| UAV-SEAD | tüm (5-seed) | LSTM-AE (Kol L) | threshold/critical recall, FA | ~0.22 ham, FA~2.8-2.9 | **Gate B kaldı** |
| UAV-SEAD | | Dense-AE (Kol D) | aynı | benzer | **Gate B kaldı** |
| UAV-SEAD | | USAD (Kol U) | aynı | benzer | **Gate B kaldı** |
| UAV-SEAD | (üçü de) | trained vs untrained-random | Spearman ρ | **0.964** | Genlik-baskınlığı artefaktı — eğitim ~hiçbir şey katmıyor |
| UAV-SEAD | (üçü de) | trained vs raw-‖x‖² | Spearman ρ | **0.965** | Aynı artefakt |
| UAV-SEAD | relerr-düzeltmeli | LSTM-AE/Dense-AE/USAD | düzeltme sonrası ρ | 0.15 / 0.23 / 0.13 (FLAGGED eşiği 0.80 altı) | ✅ genlik-bağımlılığı kırıldı |
| UAV-SEAD | relerr-düzeltmeli | aynı üçü | recall | **<0.06** | FAIL — kazancın büyük kısmı ham genlik farkıymış |

---

### 2. RflyMAD (RFLY-0 / RFLY-1)

| Dataset | Kategori | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|---|
| RFLY-only | motor/sensör | `itki_komutu` (whole-flight proxy, **geçersiz**) | CUSUM/advisory recall/FA | 0.749 / — | ❌ GEÇERSİZ — whole-flight proxy hatası (RFLY-1 ile reddedildi) |
| RFLY-only | | threshold/critical (proxy) | recall/FA | 0.573 / 1.23 FA-saat | ❌ GEÇERSİZ (aynı neden) |
| Pooled SEAD+RFLY | | full matrix (proxy) | Gate R-C | kaldı | FAIL bile proxy ile |
| **RFLY-only (düzeltilmiş, interval-truth)** | motor/sensör | `itki_komutu` | CUSUM/advisory recall/FA | **0.526 / 22.28 FA-saat** | Gate R-A/R-B geçti, **Gate R-C kaldı** (FA hedefin ~2 katı) |
| RFLY-only (düzeltilmiş) | | `itki_komutu` critical | recall/FA | 0.442 / 9.23 FA-saat | FAIL (critical hedefi ≥0.30@≤2 karşılamadı) |
| Pooled SEAD+RFLY (düzeltilmiş) | | `itki_komutu` | CUSUM/advisory recall/FA | **0.149 / 30.00 FA-saat** | FAIL — havuzlama ciddi kötüleştiriyor |
| Pooled (düzeltilmiş) | | `rfly0_fusion` advisory | recall/FA | 0.066 / 9.39 FA-saat | bütçe içi ama çok düşük recall |
| Pooled (düzeltilmiş) | | `rfly0_fusion` critical | recall/FA | 0.018 / 1.72 FA-saat | aynı |

5 uçuşta ULog arıza-onay mesajı klasör etiketiyle çelişti → ambiguous/invalid
sayılıp sonuç görülmeden dışlandı (istatistiksel temizlik, hata değil).

---

### 3. ADS-B hattı (aktif — "3. hafta" pivotu)

#### 3.1 Kural-bazlı skorlayıcı vs nöral alternatifler

| Senaryo | Algoritma | Metrik | Değer | Sonuç |
|---|---|---|---|---|
| pooled (tüm senaryolar) | Kural-bazlı penalty scorer | AUC | **0.600** | ✅ üç NN'i de geçti |
| pooled | Dense-AE | AUC | 0.572 | FAIL — kural-bazlıyı geçemedi |
| pooled | LSTM-AE | AUC | 0.568 | FAIL |
| pooled | LSTM-forecaster | AUC | 0.552 | FAIL |
| ground_speed_biased | Dense-AE / LSTM-AE / LSTM-forecaster | AUC | 0.737 / 0.743 / 0.648 | kısmi (büyüklüğü yakaladı) |
| vertical_rate_frozen | aynı üçü | AUC | 0.600 / 0.579 / 0.552 | zayıf |
| track_frozen | aynı üçü | AUC | 0.523 / 0.521 / 0.551 | FAIL — rastgeleden farksız |
| position_ramp_stealthy | aynı üçü | AUC | 0.513 / 0.519 / 0.513 | FAIL — rastgele |
| altitude_dropout | aynı üçü | AUC | 0.489 / 0.480 / 0.498 | FAIL — rastgeleden kötü |
| genel | tüm modeller (magnitude-domination) | trained-random / trained-raw ρ | Dense 0.86/0.90, LSTM-AE 0.84/0.89, forecaster 0.94/0.92 | FLAGGED (genlik-baskınlığı) |
| pooled (proxy pencere etiketi) | — | AUC tavanı | **~0.75** | Yapısal sınır — mükemmel dedektör bile burada tavan yapar |

#### 3.2 Truth-v2 düzeltilmiş kural (daha güçlü ama Adım 7'de FAIL)

| Senaryo | Metrik | Değer | Sonuç |
|---|---|---|---|
| pooled | AUROC / AUPRC | **0.764883 / 0.883313** | ✅ güçlü ayrışma |
| ground_speed | event recall (medyan gecikme) | 0.963659 (19.31 s) | ✅ |
| track | event recall (medyan gecikme) | 0.951804 (56.75 s) | kısmi (gecikme yüksek) |
| stealthy_ramp | event recall | 0.801347 | görünüşte iyi AMA aktif-aralık micro coverage yalnız **0.183902** | FAIL — kapsam çok dar |
| — | doğal temiz burden | 4.808533 episode/saat | referans |
| tam-hacim | CUSUM h=1 burden (calib/dev/rehearsal) | 6.07 / 5.74 / 5.33 episode/saat | scoreable-flight alarm oranı ~%99 — episode-merge doygunluğu gizliyor |

**ADR-032 — Adım 7 gate FAIL (2026-07-14):** Rule+CUSUM ana konfigürasyonu
dondurulamadı — üç ayrı, herhangi biri tek başına yeterli üç engel: (1) CUSUM doğal
alarm doygunluğu operasyonel gate'i geçmiyor, (2) üç NN magnitude şartını geçmiyor
(yukarıdaki ρ 0.84-0.94 FLAGGED), (3) corrected CUSUM truth-v2 ölçümü frozen
scoring-source snapshot'ı geri getirilemediği için fail-closed bloklandı (hash
uyuşmazlığı: `adsb/features.py` canonical-LF hash frozen manifestle eşleşmiyor,
735 yerel blob adayında bulunamadı). **Sonuç: Genel gate FAIL, sonraki adımlar
(Adım 8/9) başlatılmadı.**

---

### 4. UAV GNSS Bütünlük Fizibilitesi v1 — tam NO-GO (16 Temmuz 2026)

Bu oturumda tam metni okunup bağımsız doğrulandı
(`artifacts/uav_gnss_integrity_v1/uav_gnss_integrity_v1_final_no_go_report.tex`).

| Rol | Sözleşme | Yöntem | Recall | Alarm/uçuş-saati | Sonuç |
|---|---|---|---:|---:|---|
| development | critical | uçuş kontrol göstergeleri (PX4-native) | %58.8 | 19.58 | FAIL |
| development | critical | CUSUM | %0.0 | 0.00 | FAIL |
| development | critical | contextual LSTM | %47.1 | 20.32 | FAIL |
| development | advisory | PX4-native | %58.8 | 19.58 | FAIL |
| development | advisory | CUSUM | %52.9 | 0.00 | FAIL |
| development | advisory | LSTM | %82.4 | **101.60** | FAIL — alarm yükü aşırı |
| rehearsal | critical | PX4-native | %90.0 | 28.86 | FAIL — alarm yükü |
| rehearsal | critical | CUSUM | %0.0 | 0.00 | FAIL |
| rehearsal | critical | LSTM | %90.0 | 0.00 | FAIL — development'a genellemiyor |
| rehearsal | advisory | PX4-native | %90.0 | 28.86 | FAIL |
| rehearsal | advisory | CUSUM | %70.0 | 7.21 | FAIL — %90 hedefi tutturamadı |
| rehearsal | advisory | LSTM | %100.0 | 44.86 | FAIL — alarm yükü |
| — | — | LSTM trained-random Spearman | ρ | 0.678 | 0.80 altı (magnitude-domination tekrarlanmadı ama…) |
| — | — | LSTM trained-raw-magnitude | ρ | 0.690 | aynı |
| SIL-Wind | advisory | CUSUM / LSTM | alarm/saat | 8.27 / 6.88 | bütçe (12) içinde |
| HIL-Wind | advisory | CUSUM / LSTM | alarm/saat | **20.53 / 22.61** | FAIL — domain kaymasına aşırı hassas |

**Sonuç: NO-GO / not achievable with current data and instrumentation.** Hiçbir
yöntem hiçbir rolde kritik+advisory'yi birlikte tutturamadı; kör holdout hiç
açılmadı.

---

### 5. RESIDUAL-V1 — Görev 4.1 (G1 ridge) ve Faz E (bu oturum)

#### 5.1 Görev 4.1 — G1 ridge, development-only

| Dataset | Kanal | CV R² | Train R² | Sonuç |
|---|---|---:|---:|---|
| ALFA | R1_aileron_roll_rate | — | — | FAIL — eğitilemedi (1 oturum, ≥2 gerekli) |
| ALFA | R2_elevator_pitch_rate | — | — | aynı |
| ALFA | R3_rudder_coordinated_yaw_rate | — | — | aynı |
| ALFA | R4_throttle_airspeed_derivative | — | — | aynı |
| ALFA | R5_pitch_throttle_climb_rate | — | — | aynı |
| RFLY | Q1_attitude_setpoint_rate_response | **0.0114** | 0.3638 | FAIL — pratikte zayıf, train-CV farkı büyük |
| RFLY | Q2_motor_pwm_distribution | **0.4564** | 0.7109 | ✅ eğitildi, en güçlü G1 sonucu |
| RFLY | Q3_total_pwm_vertical_acceleration | **0.0003** | 0.0085 | FAIL — pratikte sinyalsiz |
| RFLY | Q4_position_setpoint_velocity_response | — | — | FAIL — train-eligible satır yok |

#### 5.2 Faz E — sanity kapıları (hepsi PASS) ama kalibrasyon NO-GO

| Test | Kanal | Metrik | Değer | Sonuç |
|---|---|---|---:|---|
| S-4 (komut ablasyonu) | Q1 | var(sakat)/var(tam) | 1.1992 (eşik 1.15) | ✅ PASS |
| S-4 | Q2 | aynı | 2.4972 | ✅ PASS |
| S-4 | Q3 | aynı | **1.0081** (eşik 1.15) | ❌ FLAGGED — karar hattından çıkarıldı |
| S-4 | Q4 | — | — | not_evaluable/model_unavailable |
| S-1 (büyüklük korelasyonu) | R6 | Spearman ρ | 0.4718 (eşik 0.5) | ✅ PASS |
| S-1 | Q1 | ρ | 0.1398 | ✅ PASS |
| S-1 | Q2 | ρ | 0.0179 | ✅ PASS |
| S-3 (KS ayrışması) | ALFA/engine (R6) | KS / p | 0.1646 / 5.5e-18 | ✅ PASS |
| S-3 | RFLY/motor (Q1) | KS / p | 0.1772 / ≈0 | ✅ PASS |
| S-3 | RFLY/motor (Q2) | KS / p | 0.5702 / ≈0 | ✅ PASS |
| S-3 | RFLY/sensor (Q1) | KS / p | 0.2740 / ≈0 | ✅ PASS |
| S-3 | RFLY/sensor (Q2) | KS / p | 0.0340 / 2.3e-91 | ✅ PASS (küçük etki) |
| **Kalibrasyon** | ALFA/R6 | mevcut/gereken normal saat | 0.168846 / 2.0 (**11.845× açık**) | ❌ **NO-GO** — thresholds_frozen.json yazılmadı |
| **Kalibrasyon** | RFLY/Q1,Q2 | mevcut/gereken normal saat | 0.786237 / 4.0 (**5.088× açık**) | ❌ **NO-GO** |
| Veri tavanı testi | ALFA | toplam normal uçuş (evren) | 11 / 47 sabit corpus | Ek veri yolu KAPALI |
| Veri tavanı testi | RFLY | resmî kaynak / projede / eksik üst sınır | 84 / 51 / 33 | En iyimser ingest bile 1.419h → **2.581h/~168 uçuş açık kalır** |

**Sonuç: NO-GO / mevcut development-normal maruziyetle elde edilemez** (ADR-043,
2026-07-17). Kritik ayrım: bu bir sinyal-yokluğu değil — S-3 üç sınıfta da PASS —
salt kalibrasyon için yeterli bağımsız normal uçuş-saati yok.

---

### Özet — kaç FAIL/NO-GO, kaç PASS/Gate-geçen

| Hat | Toplam denenen konfigürasyon (bu deftere göre) | Gate C / operasyonel hedefi geçen | Gate B (yöntemsel iyileşme) geçen | Tam NO-GO |
|---|---:|---:|---:|---|
| Legacy ML (ALFA/UAV/SEAD) | ~20 | **0** | 2 (Chronos ML-10, ince-modül ML-12) | — (arşivlendi) |
| RflyMAD (RFLY-0/1) | 6 | **0** | Gate R-A/B geçti, R-C hep kaldı | — |
| ADS-B | ~10 | 0 (Adım 7 FAIL) | kural-bazlı NN'leri geçti (AUC 0.60) | ADR-032 FAIL |
| UAV GNSS Integrity v1 | 12 satır (3 yöntem × 2 rol × 2 sözleşme) | **0** | — | ✅ NO-GO (ADR — GNSS raporu) |
| RESIDUAL-V1 | 9 kanal + 4 sanity kapısı + kalibrasyon | S-1/S-3 hepsi PASS, **kalibrasyon NO-GO** | Q2 tek güçlü ridge sonucu | ✅ NO-GO (ADR-043) |

**Tek satırlık gerçek:** Projenin başından beri **hiçbir konfigürasyon operasyonel
Gate C/final bütçe hedefini tam anlamıyla karşılamadı** — en yakın yaklaşımlar
RFLY-only `itki_komutu` (0.526 recall ama FA hedefin 2 katı) ve ADS-B kural-bazlı
skorlayıcı (AUC 0.60-0.88 aralığında, ama tavan ~0.75-0.88 senaryoya göre) idi.
İki en güncel çalışma (GNSS Integrity v1, RESIDUAL-V1) disiplinli NO-GO ile kapandı;
ikisinde de "sinyal var ama operasyonel eşik kurulamadı" ayrımı belgelendi.


---

## UAV GNSS Integrity v1 — ön-kayıt

Bu çalışma genel anomaly detector veya motor-sağlığı ürünü değildir. Tek iddia,
PX4'ün mevcut EKF/GNSS telemetrisiyle RflyMAD gerçek uçuşlarındaki `ID=123456`
GNSS noise ve scale-factor arızalarının dondurulmuş gecikme ve alarm-yükü
sözleşmesi altında tespit edilebilirliğidir.

- Fit ve threshold calibration yalnız `Real-No_Fault` uçuşlarından yapılır.
- GPS klasöründe bulunmasına rağmen `ID=123455` taşıyan altı magnetometre vakası
  karantinaya alınır.
- Satır, alarm episode, event, uçuş ve scoreable-flight-hour birimleri ayrıdır.
- `not_evaluable` normal veya anomaly sınıfına çevrilmez.
- Kritik sözleşme: 5 saniye, en fazla 2 episode/uçuş-saat.
- Advisory sözleşme: 15 saniye, en fazla 12 episode/uçuş-saat.
- Yöntemler yalnız PX4-native, çok-kanallı Page CUSUM ve contextual
  location/scale LSTM'dir. Sonuç sonrası fusion veya model kataloğu genişletme yoktur.
- LSTM, trained-vs-random veya trained-vs-magnitude Spearman korelasyonu 0.80 ve
  üzerindeyse geçersiz sayılır.
- Holdout ayrı `HOLDOUT_UNSEAL.json` onayı olmadan okunamaz.
- Holdout sonucu görüldükten sonra değişiklik yeni namespace ve yeni prereg ister.

SIL/HIL `*-Wind` havuzları beş rüzgâr etkisini temsil eder; GNSS arıza
ground-truth'u veya ürün recall kanıtı değildir.



---

## 1-2. Hafta Sunumu — Revizyon Talimatları (Claude web için)

Bu dosyayı olduğu gibi Claude web'e (claude.ai) yapıştır. Sayılar zaten gerçek ve
doğrulanmış (kullanıcı decisions.md/archive ADR'lerinden teyit etti) — buradaki
eleştiri placeholder değil, **görsel şablon + jargon + bağlam eksikliği** üzerine.

**Genel teşhis:**
1. Slayt 3 (Ortak Proje) ve Slayt 5'in sol grafiği (Şekil 3, RflyMAD Proxy Düzeltmesi)
   yine kutu+ok şablonunda — 3. hafta sunumunda eleştirilen aynı kalıp. Slayt 5'in sağ
   grafiği (Şekil 4) ve Slayt 4'ün grafiği (Şekil 2) zaten gerçek bar chart, bunlara
   dokunma, iyi durumdalar.
2. **Slayt 4'te dış paydaşa gösterilemeyecek repo-içi jargon var: "LightGBM (ML-8A),
   kategori residual'ları (ML-9), iki-kanal mimarisi (ML-13)" ve "Chronos zero-shot:
   ML-10; ince-modül itki_komutu: ML-12".** Bu ML-N faz numaraları yalnız bu reponun
   iç iş-takibi, mentörün proje mimarisinden haberi yok — bunları temizle, yöntemi ne
   olduğu üzerinden anlat (ör. "LightGBM tabanlı pencere modeli", "kategori-özel
   residual feature'ları", "iki ayrı alarm kanalı mimarisi", "sıfır-atış zaman-serisi
   modeli (Chronos)", "ince, tek-feature'lı modül").
3. ADR kodları ("ADR-001", "ADR-002/ADR-005"...) çıplak bırakılmış — mentör bu kodların
   ne karar olduğunu bilemez. Ya kaldır ya da her birine üç-beş kelimelik bir açıklama
   ekle.
4. Slayt 2 (İÇİNDEKİLER) bu şablonun 3. hafta sürümünde başlık/daire çakışması vardı —
   aynı şablon ailesi olduğu için bu slaytta da aynı sorun olabilir, kontrol et.

---

### Gerçek görseller (repoda bulundu, kopyalandı)

`docs/sunum_1_2_hafta_gorseller/` klasörüne kopyaladım — Claude web'e doğrudan
yükleyebilirsin:

- `alfa_completeness_heatmap.png`, `alfa_class_counts.png` — ALFA hattının (senin
  geliştirdiğin) gerçek veri-kalite/kapsam görselleri. `completeness_heatmap`
  `velocity_mps` null sorununu görsel olarak zaten kanıtlıyor.
- `uav_attack_completeness_heatmap.png`, `uav_attack_class_counts.png` — aynısı UAV
  Attack hattı için (senin geliştirdiğin ikinci hat).
- `rflymad_parsed_pool.png` — RflyMAD'ın gerçek 490 uçuşluk kompozisyonu (Real-Motor
  ~242, Real-Sensors ~197, Real-No_Fault ~51 — toplamı 490'a denk geliyor).
- `rflymad_recall_vs_false_alarm.png` — gerçek recall-vs-saatlik-yanlış-alarm scatter
  grafiği, kritik/advisory hedef çizgileriyle. **Dikkat:** bu görselin başlığında
  "ML-14" yazıyor — olduğu gibi yüklersen jargon kuralını ihlal eder. Ya Claude web'den
  aynı veriyle başlıksız/jargonsuz bir versiyon çizmesini iste, ya da görseli başlığı
  kırparak kullan.

Kaynak: `magnitude_domination_diagnostic.json` (archive) — Slayt 5'teki ρ iddialarının
ham verisi: `trained_vs_untrained_random_init_spearman = 0.9637`,
`trained_vs_magnitude_only_spearman = 0.9645` (n=235.625 test penceresi). Bu, "eğitim
neredeyse hiçbir şey katmıyor" iddiasının kaynağı — slaytta bu n sayısını eklemek
iddiayı somutlaştırır.

---

### Slayt 3 — Ortak Proje (pipeline)

**Sorun:** Bronze→Silver→Gold üç kutu+ok — jenerik. Asıl mesaj ("ALFA ve UAV Attack
hattının TAMAMI benim tarafımdan geliştirildi") kutu diyagramında hiç görünmüyor, en
altta küçük bir caption'da ("Sorumluluklar: Metehan → ..., Yusuf → ..., Anıl → ...")
gömülü kalmış — bu slaydın en önemli cümlesi bu, görsel olarak en gizli yerde.

**Önerilen görsel:** Aynı Bronze→Silver→Gold akışını koru ama her kutunun altına/içine
kimin geliştirdiğini renk/etiketle göster (ör. Anıl'ın geliştirdiği ALFA+UAV Attack
akışı vurgulu renkte, adsb.lol hattı nötr gri) — "katkılarım" başlığının görsel
karşılığı bu olmalı. İkinci olarak yukarıdaki `alfa_completeness_heatmap.png` /
`uav_attack_completeness_heatmap.png` görsellerinden birini slayda ekle — "gerçek
veriyle doğrulandı" cümlesinin kanıtı olarak.

**İçerik derinleştirme:**
- Her ADR'nin ne olduğu bir-cümle: örn. "ADR-005: velocity_mps null sorununun
  belgelenip kabul edilmesi" gibi (gerçek içeriği decisions.md'den al).
- 99.885 satır/10 kolon sayısının hangi katmana (Gold) ait olduğunu açıkça belirt;
  Bronze/Silver katman satır sayıları da varsa ekle (pipeline'ın hacmini gösterir).
- 91 test sayısını "bilinen açık" cümlesinin yanına değil, pipeline güvenilirliğinin
  kanıtı olarak ayrı vurgula.

---

### Slayt 4 — İlk Yöntem Denemeleri (Gate B/C)

**Sorun:** Grafik zaten iyi (gerçek bar chart, 0.205→0.390→0.459, hedef çizgisi 0.50).
Tek sorun metindeki ML-N jargonu (yukarıya bkz.) ve "hepsi Gate B veya C'de kaldı"
cümlesinin hangi yöntemin tam olarak nerede kaldığını söylememesi.

**İçerik derinleştirme:**
- Jargon temizliği (zorunlu, feedback kuralı): "LightGBM (ML-8A)" → "LightGBM tabanlı
  pencere modeli"; "kategori residual'ları (ML-9)" → "kategoriye özel residual
  feature'ları"; "iki-kanal mimarisi (ML-13)" → "sistem+mekanik ayrı alarm kanalı
  mimarisi"; "Chronos zero-shot: ML-10" → "sıfır-atış zaman-serisi modeli (Chronos)";
  "ince-modül itki_komutu: ML-12" → "tek-feature'lı ince modül (itki komutu sinyali)".
- Grafikteki üç çubuğun her biri için tek satır "neden" ekle: neden `motor_simetrisi`
  (başlangıç) 0.205'te kaldı, Chronos'un zero-shot olması neden iyileştirdi, ince
  modülün TEK feature'a indirgenmesi neden daha da iyi sonuç verdi (seyrelme/dilution
  hipotezi — çok feature'lı versiyonların sinyali sulandırdığı gerçek bulgu, mentöre
  aktarılmaya değer).
- LightGBM/kategori-residual gibi Gate B'de bile kalan denemeler için en az bir sayı
  ekle (kullanıcının teklif ettiği AUPRC 0.349 gibi) — "hepsi kaldı" çok genel.

---

### Slayt 5 — Gerçek Veri ve Genlik-Baskınlığı

**Sorun:** Sol taraf (Şekil 3) iki kutu+ok — "0.749 recall" → "0.526/22.28 FA-sa" iyi
bir sayı ama kutu diyagramı METODOLOJİK FARKI göstermiyor (neden 0.749 yanlıştı).

**Önerilen görsel (sol, Şekil 3 yerine):** Kutu yerine tek bir uçuşun zaman çizelgesi
üzerinde iki etiketleme stratejisini üst üste göster: bir satırda "whole-flight proxy"
(arıza başlangıcından uçuş sonuna kadar TAMAMI kırmızı), altındaki satırda
"interval-truth" (yalnız gerçek arıza aralığı kırmızı, geri kalan yeşil). Bu, "neden
0.749 yanlıştı" sorusunu tek bakışta cevaplar — sayı değişikliğinden çok daha güçlü bir
demo. `rflymad_parsed_pool.png`'i de bu slayda küçük bir yan-görsel olarak ekleyebilirsin
(490 rakamının nereden geldiğini gösterir).

**Sağ taraf (Şekil 4):** İyi durumda, dokunma. İçeriğe eklenecek: yukarıdaki gerçek
n=235.625 test penceresi sayısı ve "relerr düzeltmesi" ifadesinin ne olduğu bir cümleyle
(göreli hata ile ölçekleme — ham genlik yerine oransal sapmaya bakmak) — şu an "relerr"
tanımsız bir kısaltma olarak duruyor.

**İçerik derinleştirme:**
- LSTM-AE/Dense-AE/USAD'ın üçünün de "aynı örtük soruna düştüğü" iddiası güçlü ama
  NEDEN aynı sorun olduğu eksik: kök neden RobustScaler'ın aykırı değerleri
  kırpmaması + bir uçuşun donmuş-GPS/eph≈25000 sentinel artefaktı taşıması — bu kök
  neden cümlesi mentöre "üç farklı mimari tesadüfen mi aynı hataya düştü" sorusunu
  önceden cevaplar (hayır, ortak bir veri/ölçekleme kusuru).
- Kullanıcının kendi notu: hafta4 sunumundaki "AUC≈0.54" ile bu slaydaki genlik-
  baskınlığı bulgusu aynı örüntünün tekrarı — bunu değerlendirme slaydına (Slayt 6)
  bir cümle olarak ekle: "3. haftadaki [FLAGGED] sonuç, burada [2. haftada] zaten bir
  kez keşfedilip belgelenmiş genlik-baskınlığı örüntüsünün tekrarıydı" gibi — bu iki
  sunum arasında bilimsel süreklilik kurar, mentöre "aynı hatayı fark edip
  belgeleyebiliyoruz" mesajı verir.

---

### Slayt 6 — Değerlendirme

Zaman çizelgesi (Şekil 5) zaten iyi bir format, template kutu değil — dokunma. Tek
ekleme: yukarıdaki hafta2↔hafta4 bağlantı cümlesi ve "9 yöntem" listesinin en az
isim olarak (jargonsuz) tek satırda sayılması ("basit istatistiksel eşik, LightGBM
penceresi, kategori-özel residual, iki-kanal füzyon, sıfır-atış zaman-serisi modeli,
ince tek-feature modül, ..." gibi) — şu an "9 yöntem" sadece bir sayı, hangi 9 olduğu
hiçbir yerde toplu görünmüyor.

---

### Slayt 1-2 (Kapak / İçindekiler)

Slayt 2'yi 3. hafta sunumunun aynı şablonundaki başlık/daire çakışma hatasına karşı
kontrol et (bu depoda önceki turda tespit edilmişti). Aynı hata burada da olabilir.

---

### Placeholder kuralı

Bu sunumda zaten placeholder yok (sayılar gerçek) — ama Claude web'den yeni bir görsel
istediğinde (ör. Slayt 3'ün kimlik-vurgulu pipeline diyagramı) sayı bilmediği bir yer
çıkarsa köşeli parantez (`<...>`) yazıp tasarımın içine literal metin olarak basmasın;
bilmediği yeri boş bıraksın ya da ayrı bir not olarak dışarıda belirtsin.

---

### Claude web'e verilecek özet talimat

> Slayt 3 ve Slayt 5'in sol grafiğini kutu+ok şablonundan çıkar (yukarıdaki önerilere
> göre: Slayt 3'te kimlik-vurgulu pipeline + gerçek completeness görseli, Slayt 5'te
> whole-flight-proxy vs interval-truth zaman çizelgesi karşılaştırması). Slayt 4'teki
> tüm "ML-N" faz numaralarını ve mümkünse ADR kodlarını jargonsuz açıklamalarla
> değiştir. Slayt 5 ve 6'ya yukarıdaki içerik-derinleştirme cümlelerini ekle
> (relerr tanımı, kök neden cümlesi, hafta2↔hafta4 bağlantısı, 9 yöntemin isim listesi).
> Slayt 2'yi başlık/daire çakışmasına karşı kontrol et. `docs/sunum_1_2_hafta_gorseller/`
> içindeki gerçek PNG'leri kullan, `ml14_recall_vs_false_alarm.png`'i yüklüyorsan
> başlığındaki "ML-14" ifadesini kırp veya aynı veriyle jargonsuz yeniden çizdir.


---

## 3. Hafta Sunumu — Revizyon Talimatları (Claude web için)

Bu dosyayı olduğu gibi Claude web'e (claude.ai) yapıştır. Aşağıda artık **placeholder
değil, repodaki gerçek çalışmalardan çıkarılmış gerçek sayı ve gerçek PNG dosyaları**
var — v1'de "sen dolduracaksın" dediğim yerlerin çoğu artık dolu. Kalan birkaç yer için
de nasıl doldurulacağı (hangi dosyadan) yazılı.

**v2'de görülen iki yeni sorun** (bu revizyonda özellikle düzeltildi):
1. Önceki turda Claude web `<N>`, `<model türü>` gibi köşeli-parantez placeholder'ları
   SİLİP gerçek değer koymak yerine **olduğu gibi, literal metin olarak slayda bastı**.
   Bu yüzden bu dosyada artık `<...>` yazım biçimi hiç kullanılmıyor — bkz. "Placeholder
   kuralı" ve dosya sonundaki özet talimat.
2. v2'de kutular/metin ekran dışına taşıyor ve üst üste biniyor (slayt 2 ve 6'da başlık
   ile içerik çakışması, slayt 3'te mini akış kutuları grafiğin/paragrafın üzerine
   biniyor). Bkz. "Taşma/Bindirme Sorunları" bölümü — bunlar v1'den beri hiç
   düzelmemiş, aynı hata tekrar ediyor.

---

### Gerçek veri/görsel kaynakları (repoda bulundu)

Sunumdaki üç teknik slayt aslında **iki ayrı gerçek çalışmaya** karşılık geliyor:

- **Slayt 3 + 4** (kalibrasyon + alternatif yöntemler) → uçak takip (ADS-B) verisiyle
  çalışan kural-bazlı anormallik skorlayıcısı ve onunla karşılaştırılan nöral
  alternatifler (`artifacts/adsb/...`).
- **Slayt 5** (GPS-bütünlük) → PX4/İHA GNSS bütünlük pilotu, 16 Temmuz 2026 tarihli
  nihai NO-GO raporu (`artifacts/uav_gnss_integrity_v1/...`).

#### Slayt 3 — gerçek eşik/hassasiyet verisi

`artifacts/adsb/runs/20260714_contextual_physics_v1_development_burden_v2/development_burden_curves.json`
dosyasında tam olarak bu slaydın anlattığı deney var: `calibration_day: 2026-02-28`
(normal-bilinen gün, eğitim) → `development_day: 2026-03-01` (görülmemiş yeni gün, test),
12 seviyelik bir `alpha_grid` (hassasiyet eşiği) taraması, her seviye için
`alert_episodes_per_scoreable_flight_hour` (saatlik yanlış-alarm oranı). Bir kanal
(`vertical_rate_residual` → `vertical_rate_spike`) için gerçek 12 nokta:

| Hassasiyet (alpha) | Saatlik alarm/uçuş-saati |
|---|---|
| 0.00001 | 0.0035 |
| 0.0000267 | 0.0119 |
| 0.0000715 | 0.0378 |
| 0.000191 | 0.0927 |
| 0.000511 | 0.2476 |
| 0.001367 | 0.6249 |
| 0.003657 | 1.6289 |
| 0.009778 | 3.9678 |
| 0.026148 | 8.7537 |
| 0.069922 | 14.3586 |
| 0.186978 | 14.9156 |
| 0.5 | 7.4616 |

Bu tabloyu Claude web'e direkt ver, "bu gerçek veriyle çizgi grafiği çiz" de — placeholder
istemesine gerek yok. Not: eğri en yüksek iki alpha'da beklenmedik biçimde düşüyor
(14.92 → 7.46) — bu muhtemelen episode-merge etkisinden kaynaklanıyor, veriyi düzeltme,
olduğu gibi göster (gerçek sonucu "temizlemek" yanıltıcı olur). Dosyada başka
kanallar/senaryolar da var (`channels` altında); istersen farklı bir kanalı örnek al.

#### Slayt 4 — gerçek PNG'ler VE bir içerik düzeltmesi

Şu dosyalar gerçek, üretilmiş grafikler — Claude web'e **doğrudan yükleyebilirsin**,
yeniden çizdirmene gerek yok. Kolay bulman için hepsini
`docs/sunum_3_hafta_gorseller/` klasörüne kopyaladım:
- `docs/sunum_3_hafta_gorseller/roc_curves.png` — 3 alternatif model × 5 senaryo ROC eğrisi
- `docs/sunum_3_hafta_gorseller/auc_heatmap.png` — aynı verinin ısı haritası (0.48–0.74 arası)
- `docs/sunum_3_hafta_gorseller/score_distributions.png` — temiz vs bozuk skor dağılımları
- `docs/sunum_3_hafta_gorseller/confusion_matrices.png` — 0.95 güven eşiğinde karışıklık matrisleri (opsiyonel)

**İçerik düzeltmesi (önemli):** Sunum metni "daha basit istatistiksel yöntemler" diyor
ama bu PNG'lerdeki gerçek alternatifler **Dense-AE, LSTM-AE ve LSTM-forecaster** —
yani basit istatistiksel kurallar değil, ana modelden farklı mimaride üç ayrı nöral ağ.
Gerçek AUC sonuçları: `ground_speed_biased` senaryosunda 0.648–0.743 (büyüklüğü
yakaladılar), ama `track_frozen`/`position_ramp_stealthy`/`altitude_dropout`'ta
0.48–0.55 (rastgeleden farksız — inceliği kaçırdılar). Yani gerçek hikaye "basit
yöntem yetersiz kaldı" değil, **"daha karmaşık nöral alternatifler bile ana modelin
(kural-bazlı skorlayıcı) yakaladığı inceliği yakalayamadı"** — bu aslında daha güçlü
bir sonuç, metni buna göre düzelt. Slayt metnindeki "basit istatistiksel yöntemler"
ifadesini "farklı mimaride nöral alternatifler (Dense-AE / LSTM-AE / LSTM-forecaster)"
ile değiştir.

#### Slayt 5 — gerçek sonuç tablosu (16 Temmuz 2026 NO-GO raporu)

`docs/sunum_3_hafta_gorseller/uav_gnss_integrity_v1_final_no_go_report.tex` (kaynağı:
`artifacts/uav_gnss_integrity_v1/`) bu slaydın birebir konusu — dar kapsamlı
GPS-bütünlük pilotu, sonuç görülmeden dondurulmuş kriterlerle test edilmiş, **NO-GO**
kararıyla kapanmış. Bu bir LaTeX kaynak dosyası (PDF değil) — Claude web'e metin
olarak yapıştırabilir veya doğrudan yükleyebilirsin, o da içeriği okuyabilir.
Placeholder'a hiç gerek yok, gerçek development/rehearsal sonuçları:

| Rol | Sözleşme | Yöntem | Recall | Alarm/uçuş-saati | Kapı |
|---|---|---|---|---|---|
| development | critical | uçuş kontrol göstergeleri (PX4-native) | %58.8 | 19.58 | FAIL |
| development | critical | istatistiksel değişim tespiti (CUSUM) | %0.0 | 0.00 | FAIL |
| development | critical | derin öğrenme (contextual LSTM) | %47.1 | 20.32 | FAIL |
| development | advisory | uçuş kontrol göstergeleri | %58.8 | 19.58 | FAIL |
| development | advisory | CUSUM | %52.9 | 0.00 | FAIL |
| development | advisory | derin öğrenme | %82.4 | 101.60 | FAIL |
| rehearsal | critical | uçuş kontrol göstergeleri | %90.0 | 28.86 | FAIL |
| rehearsal | critical | CUSUM | %0.0 | 0.00 | FAIL |
| rehearsal | critical | derin öğrenme | %90.0 | 0.00 | FAIL |
| rehearsal | advisory | uçuş kontrol göstergeleri | %90.0 | 28.86 | FAIL |
| rehearsal | advisory | CUSUM | %70.0 | 7.21 | FAIL |
| rehearsal | advisory | derin öğrenme | %100.0 | 44.86 | FAIL |

Hedef: kritik alarm için 5 saniyede tespit + en fazla 2 alarm/uçuş-saati; advisory için
15 saniyede tespit + en fazla 12 alarm/uçuş-saati. **Hiçbir yöntem, hiçbir rolde ikisini
birlikte karşılayamadı** — bu tablo scatter grafiğin (aşağıdaki "Önerilen görsel") ham
verisi. Ayrıca her yöntemin *neden* başarısız olduğuna dair gerçek gerekçe raporda var:
uçuş kontrol göstergeleri sinyali yakalıyor ama saatte ~20-29 alarm üretiyor (operatör
yükü kabul edilemez); CUSUM kritik bütçede hiç alarm üretmeyecek kadar muhafazakâr
kalmaya zorlanmış (kritik recall sıfır); derin öğrenme modeli rehearsal'da parlak
görünüyor ama development'ta aynı donmuş karar çok daha kötü performans veriyor —
yani rol-arası genellemiyor. Bu üç cümleyi doğrudan slayt metnine ekle, "hepsi
yetersiz kaldı" yerine.

Veri ölçeği (aynı raporda, "Veri Denetimi" tablosu): Fit 20 uçuş, Calibration 10 uçuş,
Development 23 uçuş, Rehearsal 15 uçuş, mühürlü holdout 20 uçuş (hiç açılmadı, çünkü
hiçbir yöntem geçmedi).

---

**Placeholder kuralı (değişti):** Yukarıdaki üç slayt için artık gerçek sayı var, hiç
`<...>` yazmana gerek yok. Eğer ileride gerçekten bilinmeyen bir değer kalırsa,
**tasarımın içine köşeli parantez yazma** — Claude web bunu literal metin olarak
slayda basıyor (v2'de olan buydu). Onun yerine ya o cümleyi/veri noktasını o an
atla, ya da görselin DIŞINA, açıkça "[DOLDURULACAK: kaynak X]" diye ayrı bir not
kutusu olarak koy — tasarımın gövdesine karışmasın.

---

### Taşma / Bindirme Sorunları (v2'de hâlâ var)

- **Slayt 2**: "İÇİNDEKİLER" başlığı ile altındaki dört daire yine üst üste biniyor —
  v1'de de aynı sorun vardı, v2'de düzelmemiş. Başlık kutusunun altına yeterli boşluk
  bırakılmamış.
- **Slayt 3**: Üstte küçük tuttuğun mini akış kutuları ("Eğitim", "Skor üretimi",
  "Eşik taraması") hem gövde paragrafının son cümlesiyle hem de grafiğin y-ekseni
  etiketiyle çakışıyor; üçüncü kutu ("Eşik taraması") sağ kenardan taşıyor gibi
  duruyor. **Öneri: bu mini akış kutularını tamamen kaldır** — artık gerçek eşik
  verisi var, grafik tek başına yeterli, süreç adımlarını metin zaten anlatıyor.
- **Slayt 6**: Aynı başlık/gövde çakışması burada da var ("...ADIMLAR" başlığı ile
  "Bu hafta hem umut verici..." paragrafı iç içe giriyor) — slayt 2'yle birebir aynı
  şablon hatası, muhtemelen aynı kök neden (başlık placeholder'ı sabit/yetersiz
  yükseklikte, içerik onun üzerine taşıyor).

Bu üçü aynı kök nedene işaret ediyor: başlık alanı ile gövde alanı arasında sabit,
yeterli bir boşluk/margin tanımlı değil. Claude web'e şunu açıkça söyle: "her slaydı
16:9 (1280×720 veya eşdeğeri) sabit bir tuval olarak tasarla, hiçbir öğe bu tuvalin
dışına taşmasın veya başka bir öğeyle kesişmesin; başlık ile gövde arasında en az
sabit bir boşluk bırak; tasarımı bitirdikten sonra her slaydı tek tek kontrol edip
üst üste binen/taşan öğe olup olmadığını doğrula."

---

### Slayt 6 — küçük bir ek not

Zaman çizelgesi düğümlerinden biri ("Hafta 3 — bu hafta") tek daire içine iki ayrı
durumu sıkıştırıyor ("Kalibrasyon: İlerliyor" + "GPS-bütünlük: Durduruldu"). Bunu tek
daire yerine iki küçük alt-etiket olarak yan yana/alt alta, daire dışında göster —
okunabilirlik için.

---

### Claude web'e verilecek özet talimat

> Slayt 3, 4 ve 5'i, bu dosyada verdiğim gerçek tablo ve gerçek PNG dosyalarını
> kullanarak yeniden tasarla — sayı uydurma veya köşeli parantez placeholder basma,
> hepsi burada mevcut. Slayt 4'ün metnini "basit istatistiksel yöntemler" yerine
> gerçek yöntem adlarıyla (Dense-AE / LSTM-AE / LSTM-forecaster) düzelt. Slayt 3'teki
> mini akış kutularını kaldır, sadece gerçek eşik/alarm eğrisini göster. Slayt 2 ve
> 6'daki başlık/gövde çakışmasını düzelt — her slaydı sabit 16:9 tuval olarak tasarla,
> hiçbir öğe taşmasın veya kesişmesin, bitince tek tek kontrol et. Slayt 6'daki "bu
> hafta" düğümünde iki durumu ayrı etiket olarak göster. ML faz numarası veya iç repo
> jargonu kullanma, yöntemleri ne yaptıkları üzerinden anlat.

---

## 6. RflyMAD-Full v2 — 2026-07-22 Güncellemesi

> Tam ayrıntı: `gecmis_calismalar/RFLYMAD/raporlar/` altındaki 12
> `RFLYMAD_V2_*.md` dokümanı, özellikle `RFLYMAD_V2_SONRAKI_ADIMLAR_20260722.md`
> (durum) ve `RFLYMAD_V2_CONVERGENCE_DENEY_RAPORU_20260722.md` (görselli detay).

Bu bölümdeki `rfly_full/` hattı, yukarıdaki RESIDUAL-V1 ve UAV-GNSS
çalışmalarından **ayrı, çok daha büyük ölçekli** bir RFLYMAD turudur (6605
uçuş, 10 Hz parse, truth-schema v2). Aynı gün içinde şu adımlar tamamlandı:

1. **Truth/parser hatası düzeltmesi:** `v2_parser.py`'de yanlış domain
   algılama nedeniyle 2712 uçuşta arıza başlangıcı yanlış hesaplanıyordu
   (sahte "t=0'dan itibaren aktif arıza"). Düzeltildi, ilgili uçuşlar yeniden
   parse edildi, truth audit temizlendi.
2. **Preregistered Wind/Real robustness sözleşmesi** kullanıcı onayıyla
   dondurulup 6 aday (R1 threshold-only, W1 eşik-kaydırma, W2 Wind-eğitime-dahil,
   R2/R3 kısa/uzun fine-tune, R4 kullanıcı-onaylı convergence-follow-up) bu
   sözleşme altında çalıştırıldı.
3. **Sonuç: altı adayın da hiçbiri kendi promosyon kapısını geçemedi.** En iyi
   Real macro recall (R4) %14,3 baseline'dan %28,1'e çıktı ama hedef %40'ın
   altında kaldı; ayrıca genel recall düştü (%60,4→%54,6) ve FA yükü arttı. En
   iyi Wind azaltımı (W2) %37,5 oldu, hedef %40'ın az altında.
4. **Development-only supervised TCN sweep** de aynı kapılarla test edildi,
   aynı şekilde geçemedi — sorunun yalnız AE'ye özgü olmadığını, muhtemelen
   veri/temsil kısıtından geldiğini güçlendirdi.

**Genel projeye katkısı:** bu tur, dosyanın giriş bölümündeki "araştırma-
kalitesi küçük/heterojen veriyle operasyonel çıta kanıtlanmaya çalışıldı" teşhisini
bağımsız bir dördüncü hat üzerinde bir kez daha doğruladı — Real-domain
transferi ve Wind-robustness'ı için preregistered disiplin altında bile aynı
duvara toslandı. Sıradaki meşru yollar (yeni Real veri/temsil değişikliği,
veya TCN'in uzun-development koşusu) `RFLYMAD_V2_SONRAKI_ADIMLAR_20260722.md`
içinde ayrıntılandırıldı; mevcut sözleşme kapsamında yeni threshold/epoch
avcılığı yapılmayacak.
