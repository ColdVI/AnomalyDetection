# Final Sunum v3 — Doğruluk Denetimi, Bilgi Kaybı Analizi ve Yeniden Yazım

**Denetlenen dosyalar**
- `Downloads/Final_Sunum_IHA_Anomali_Tespiti_Basliklari_Teknik_v3.pptx` — 22 slayt (şu an üzerinde çalışılan)
- `Downloads/Final_Sunum_IHA_Anomali_Tespiti_v2_slide_thematic_design.pptx` — 41 slayt (görsel kirlilik nedeniyle terk edilen)

**Doğrulama kaynakları:** `docs/final_presentation/01..09_*.md`, `docs/decisions.md`,
`docs/RFLYMAD_V31_B0_GERCEK_EVENT_DEGERLENDIRME_20260728.md`,
`docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_FINAL_REPORT_20260728.md`,
`docs/final_rapor_ml_fizibilite_2026-07-16.md`,
`artifacts/four_dataset_probabilistic_v31/**`. Aşağıdaki hiçbir sayı yorumla üretilmedi;
her biri bu dosyalardan okundu.

---

# BÖLÜM A — Doğruluk denetimi

## A.1 Ciddi hatalar (sunum öncesi mutlaka düzeltilmeli)

### H1 — Slayt 13 grafiği "monolitik / modüler" dersini tersine çeviriyor
**Slaytta ne var:** `ALFA ranking signals` grafiği: `IF monolithic = 0,833`, `IF best module = 0,864`.

**Gerçek:** 0,833 **modüler** Isolation Forest'ın uçuş ROC-AUC'sidir. Monolitik kurulum
ALFA'da rastgele sıralamaya yakındı ve UAV Attack'ta 0,21'de kaldı.
Kaynak: `03_method_inventory.md` satır 7-8 — *"ALFA modüler IF flight AUC 0,833"*,
*"Monolitik IF'ye göre ALFA flight AUC 0,833'e çıktı"*.

**Neden ciddi:** Projenin en anlaşılır mühendislik dersi bu — "tüm feature'ları tek modele
verince güçlü ama dar bir sinyal seyreliyor". Grafik şu hâliyle bu dersin tam tersini
söylüyor: monolitik zaten iyiymiş, modüler marjinal katkı yapmış gibi görünüyor.

**Düzeltme:** Grafik dört bar olmalı: `Monolitik IF (ALFA) — rastgeleye yakın` /
`Monolitik IF (UAV Attack) — 0,21` / `Modüler IF (ALFA) — 0,833` / `En güçlü tek modül — 0,864`.
İlk iki barın farklı veri setlerine ait olduğu grafik altında yazmalı.

---

### H2 — %97,6 ve 25,54/saat, reddedilmiş bir arşiv denemesine ait; Gaussian NLL hattına ait değil
**Slaytta ne var:** Slayt 9 başlığı *"First Probabilistic Forecasting Trial on ADS-B"*,
hemen ardından Slayt 10 aynı anlatı içinde `97.6% synthetic anomaly recall` ve
`25.54/h natural false alarm load` kartlarını veriyor. Slayt 9'un takeaway'i:
*"ADS-B + Gaussian NLL, final yöntemin ilk versiyonu gibi sunulmalı."*

**Gerçek:** Bu iki sayı `archive/2026-07-10_rejected_adsb_attempts/` altındaki, **reddedilip
arşivlenmiş** ADS-B denemesine aittir. Gaussian NLL hattının ilk temiz üyesi 14 Temmuz'daki
contextual v1'dir ve o koşunun karnesi farklıdır (genlik kapısı GEÇTİ, ρ=0,65).
Kaynak: `docs/decisions.md:835`, `09_number_consistency_audit.md` yasak listesi madde 1 —
*"%97,6 sentetik recall'ı güncel ADS-B başarısı gibi kullanma"*.

**Neden ciddi:** Repo'nun kendi kuralını ihlal ediyor ve arşivlenmiş bir sonucu final yöntemin
soyağacına bağlıyor. Mentör "bu neden reddedildi de sunumda duruyor" diye sorarsa savunulamaz.

**Düzeltme:** Bu iki sayı kalmalı — ama **kendi slaytında, reddedilme hikâyesiyle**:
"Bir yaklaşım sentetik anomalilerin %97,6'sını yakaladı, aynı sistem doğal trafikte 25,54
alarm/saat üretti; yaklaşım reddedildi ve arşivlendi. O günden sonra yakalama oranı ve alarm
yükünü ayrı raporlamak yasaklandı." Gaussian NLL soyağacı ayrı slaytta anlatılmalı.

---

### H3 — "ADS-B / OpenSky" beş kaynaktan biri olarak gösteriliyor
**Slaytta ne var:** Slayt 5'te kaynak adı `ADS-B / OpenSky`; Slayt 12 *"OpenSky / ADS-B fault
start time vermediği için…"*; Slayt 22 referanslarında `Project datasets: ADS-B/OpenSky`.

**Gerçek:** OpenSky **hiç deney veri kaynağı olmadı**. 29 Haziran 2026'da (ADR-001) kimlik
doğrulama ve kota kısıtları nedeniyle `adsb.lol`'e geçildi. Repo'nun kanıtlanabilir çalışma
havuzu: ALFA, UAV Attack, UAV-SEAD, RflyMAD ve etiketsiz **adsb.lol** trafiği.
Kaynak: `01_project_timeline.md` satır 38; `02_dataset_inventory.md` satır 11.

**Neden ciddi:** Kaynak adı yanlış verilmiş bir veri seti, tüm veri anlatısının güvenilirliğini
zedeler. Ayrıca `01_project_timeline.md` açıkça uyarıyor: *"Sunumda bunun ötesinde aday veri
seti isimleri uydurulmamalıdır."*

**Düzeltme:** Tüm slaytlarda kaynak adı `ADS-B (adsb.lol)`. OpenSky yalnız tek cümlelik bir
başlangıç notu olarak, açıkça "terk edildi" diye geçmeli.

---

### H4 — Slayt 20'nin iki grafiği tamamen uydurma sayılardan üretilmiş
**Slaytta ne var:** "Detected motor fault" ve "Missed real sensor fault" grafiklerinin veri
serileri: `score = [0.25, 0.35, 0.72, 0.78, 0.70]`, `threshold = [0.6, 0.6, 0.6, 0.6, 0.6]`.

**Gerçek:** Bunlar gerçek koşudan gelmiyor; elle yazılmış temsilî sayılar. Aynı iki olayın
**gerçek** grafikleri repoda mevcut: `05_detected_motor_event_timeline.png` ve
`06_missed_real_sensor_timeline.png`. Gerçek eşik değeri 10,0940'tır, 0,6 değil.

**Neden ciddi:** Sonuç sunan bir slaytta uydurma eğri kullanmak, tüm decksin dürüstlük
iddiasını çökertir — ki bu projenin ana satış argümanı zaten dürüst ölçüm.

**Düzeltme:** Gerçek artefaktları kullan. İngilizce başlıkları varsa slaydın kendi Türkçe
başlığı ve altyazısı bunu taşır; grafiğin kendisi ham çıktı olarak kalabilir.

---

### H5 — Nihai karar (NO-GO) ve mühürlü testin hiç açılmadığı deckte hiç geçmiyor
**Slaytta ne var:** Slayt 21 sonucu *"hangi koşullarda güvenilir alarm üretilebildiğini ve
hangi koşullarda bunun mümkün olmadığını gösterdi"* diye kapatıyor.

**Gerçek:** Projenin resmî bilimsel kararı açık bir **NO-GO**'dur ve bu karar nedeniyle
RflyMAD'in 553 uçuşluk mühürlü final test seti **hiç açılmadı**. ALFA v3.1 group-CV de
NO-GO. GNSS bütünlük pilotu da NO-GO. Kaynak: `01_project_timeline.md` satır 26-28;
`02_dataset_inventory.md` satır 21.

**Neden ciddi:** Bu proje bir "negatif ama disiplinli sonuç" projesi. Kararın kendisini
söylemeyen bir sunum, projenin en güçlü kısmını atlıyor — üstelik "mühürlü testi hiç
açmadık" cümlesi metodolojik olarak decksin en etkileyici cümlesi.

**Düzeltme:** Kapanışta net karar cümlesi + mühürlü test kartı.

---

## A.2 Düzeltilmesi gereken ama daha küçük sorunlar

| # | Slayt | Sorun | Düzeltme |
|---|---|---|---|
| K1 | 10 | `Physics rules 0,600 vs Neural 0,552/0,572` karşılaştırması "temiz kazanç" gibi sunulmuş | Üç sinir ağının hepsi genlik-baskınlığı kontrolünde FLAGGED'dı ve bu AUC'ler etiket hatalı tarihsel koşudan; grafik altına "karşılaştırılan sinir ağları ayrı bir kontrolde elendi" notu şart (`docs/decisions.md:1260-1262`) |
| K2 | 10 | Başlık "Large-Scale Normal Traffic" ama grafik sentetik senaryo havuzu AUC'si | Grafiği kendi slaytına al veya altına "sentetik senaryo havuzu" yaz |
| K3 | 13 | `0,918` hiçbir uyarı olmadan duruyor | Bu sayı nedensellik düzeltmesinden ÖNCEKİ değer; iki slayt sonra 0,611'e düşüyor. Grafiğin içinde görünür uyarı olmalı |
| K4 | 11/13 | Sıralama ters: nedensellik düzeltmesi (slayt 11), düzeltilen sayıdan (slayt 13) önce geliyor | Kronolojiyi düzelt: önce yüksek sonuç, sonra düzeltme |
| K5 | 15 | `0,749 → 0,526` verilmiş ama yanındaki 22,28 yanlış alarm/saat yok | Repo kuralı: yakalama ve alarm yükü **asla** ayrı sunulmaz. 22,28 eklenmeli |
| K6 | 12 | `73–85 feature family` bir aralık gibi yazılmış | Bu bir aralık değil, bir **öncesi/sonrası**: kolon eşleşme hatası düzeltilince 73 → 85 |
| K7 | 17 | Grafiğin veri serisi `Captures = [1, 1, 1, 1]` — bilgi taşımayan sahte grafik | Grafiği çıkar; yerine tek örnekli "aynı hata, iki farklı σ" şeması koy |
| K8 | 6 | `True-onset` ilkesi anlatılmış ama sayısı yok | 0,594 → 0,194–0,224 düşüşü ilkenin kanıtı; ilke sayısız kalınca havada duruyor |
| K9 | 5 | "Bütün yöntemler her dataset üzerinde aynı kapsamda çalıştırılmadı" notu doğru ve dürüst | ✅ Değişiklik gerekmez — bu iyi yazılmış |
| K10 | 18, 19 | 43,3 / 0,654 / 58,7 / 9,15 ve Motor 0,601 / Sensor 0,055 / HIL 0,569 / SIL 0,388 / Real 0,078 | ✅ **Hepsi doğru** (`RFLYMAD_V31_B0_...md` satır 96-116) |

## A.3 Doğru çıkan sayılar (dokunma)

256.150.550 satır · ~497.571 uçuş-saati · 0,878→0,611 · 0,751 cross-track ·
10→15 normal uçuş · 8 aday / 7 kurtarılan · 2.712 parser hatası · 0,747 interval AUC ve
%3,4 SEAD event recall · ±0,212→±0,012 · %43,27 @ 0,654 · %58,71 @ 9,153 ·
Motor %60,08 · Sensor %5,45 · HIL %56,85 · SIL %38,78 · Real %7,81 ·
pooled AUC 0,600 vs 0,552–0,572.

---

# BÖLÜM B — 41 → 22 geçişinde bilgi kaybı

41 sayfalıktan cayma sebebin görüntü kirliliğiydi ve bu doğru bir karardı. Ama kesim
görsel değil **içerik** tarafından da yapılmış. Kaybolanlar:

## B.1 Tamamen kaybolan bulgular (geri gelmeli)

| Kaybolan | 41'deki yeri | Neden geri gelmeli |
|---|---|---|
| **Genlik baskınlığı** — trained-random ρ≈0,964, score-magnitude ρ≈0,965; göreli hatayla yeniden skorlanınca recall %6'nın altına düştü | Slayt 20, başlığı bile "EN KRİTİK BULGU" | Projenin metodolojik imzası. Üç ayrı derin mimarinin neden aynı sonucu verdiğini açıklayan tek şey bu; ve o günden sonra rastgele-model kontrolü zorunlu kapı oldu. v3'te yalnız "magnitude shortcut riski" diye bir tablo hücresinde geçiyor |
| **Dört veri setinin final tablosu** — RflyMAD 0,907/%56,1/0,768 · SEAD 0,747/%3,40/0 · ALFA ~%20/0 ama yalnız 0,131 saat maruziyet · Attack 0,000/32,42 | Slayt 33 | v3 yalnız RflyMAD sonucunu gösteriyor. Beş veri setinin dördü sunumda tanıtılıp sonuçsuz bırakılmış |
| **Any-window yanılsaması** — anomali uçuşlarında %82,94 alarm, normal uçuşlarda %84,62, balanced accuracy %49,16 | Slayt 35 | "Uçuşta alarm var" ile "olayı yakaladık" arasındaki farkın en çarpıcı sayısal kanıtı |
| **UAV-SEAD yöntem turu** — EKF innovation 0,354 (ters yön), LightGBM 0,349 < IF 0,385, Chronos 0,205→0,390, tek özellik 0,459 @ 38,1 alarm/saat, fusion ×1,89–3,83 | Slayt 19 | Sekiz yöntem ailesinin **neden** denendiğini ve her birinin neyi öğrettiğini gösteren tek slayt. v3'te yöntem tablosu var ama sonuç yok |
| **Gerçek başlangıç düzeltmesinin sayısı** — %59,4 → %19,4–22,4 | Slayt 16 | v3 ilkeyi anlatıyor, kanıtı vermiyor |
| **Grup-güvenli bölme rol kartı** — 260 / 90 / 91 / 557 / 553 uçuş, 7/3/3/12/13 grup, sıfır kesişim | Slayt 31 | "Group-safe split" v3'te bir cümle; asıl ikna edici olan rakamlarla gösterilen rol ayrımı |
| **Genlik kapısı PASS kanıtı** — ρ 0,3288/0,3495 (kapı 0,80), epoch 30, eşik 10,0940 | Slayt 32 | SEAD'deki 0,96'lardan buraya gelmek projenin tek gerçek "iyileşme" hikâyesi |
| **ALFA'nın son sözü** — dört grup CV, yalnız fold-0 kapıyı geçti ve orada sıfır ayrım | Slayt 34 | ALFA sunumun en çok anlatılan veri seti; sonucu olmadan bitiyor |

## B.2 Kaybolan ama appendix'e gidebilecekler

Sentetik bozulma taksonomisi (6 tip, CUSUM drift'te ~%75) · RflyMAD dayanıklılık turu
(%14,3→%28,1 ama genel %60,4→%54,6) ve supervised TCN'in erken ezberlemesi ·
GNSS bütünlük pilotu (üç yöntem, hepsi NO-GO) · komut→tepki residual kalibrasyon açığı
(ALFA ~1/11,8, RflyMAD ~1/5,1) · ham telemetriden alarma dokuz adımlık dönüşüm zinciri ·
ADS-B 100 uçuşluk kural taraması (rota olaylarının 24/24'ü düşük-hız artefaktı).

## B.3 v3'ün 41'e göre **kazandırdıkları** (koru)

Bunlar iyi eklemeler, yeniden yazımda korunuyor: ayrı problem tanımı slaydı ·
değerlendirme protokolü slaydı (group-safe / causal / frozen / true-onset dörtlüsü) ·
Gaussian NLL formül slaydı · feature design slaydı · related work + references ·
her slaytta tek cümlelik "takeaway" disiplini · İngilizce teknik terimlerin korunması.

---

# BÖLÜM C — Yeniden yazılmış sunum

## C.0 Yazım kuralları (bu deck'te uygulanan)

1. **Her slayt bir motiv → bir işlem → bir sonuç → bir sonraki soru.** "Şunu gözlemledik"
   diye biten slayt yok; "bu yüzden şunu yaptık" ile biter.
2. **Repo bilgisi varsayılmaz.** ADR numarası, faz kodu (`ML-12`, `v3.1`, `B0`), dosya yolu
   ve "birinci kırılma / ikinci kırılma" gibi iç kısaltmalar slaytta geçmez. Bir karar
   anlatılacaksa kararın kendisi anlatılır, kaydının adı değil.
3. **Teknik terimler İngilizce kalır:** event recall, false alarm, ROC-AUC, autoencoder,
   CUSUM, residual, ground truth, group-safe split, causal scoring, Gaussian NLL, threshold.
   Anlatım Türkçedir; terim çevrilmez.
4. **Yakalama oranı ve alarm yükü asla ayrı verilmez.** Bir slaytta recall varsa, aynı
   slaytta false alarm/saat vardır.
5. **Her sayının yanında hangi turdan geldiği belli olur.** Farklı split'lerden gelen iki
   sayı asla aynı eksende karşılaştırılmaz.
6. **Ses tonu proje lideri.** "Denedik, olmadı" değil; "şu riski gördük, şu kararı aldık,
   şunu maliyetlendirdik, sıradaki iş şu."

## C.1 Görsel sistem (beyaz tema, sade)

**Beyaz olan zemindir, grafikler değil.** Renk anlam taşır: her ton bir role bağlıdır ve deck
boyunca aynı rolü gösterir — okuyucu ikinci slayttan sonra rengi okumaya başlar.

| Rol | Renk | Nerede |
|---|---|---|
| Zemin | `#FFFFFF` | Her slayt. Gradient, doku, tam sayfa fotoğraf yok |
| **Güncel / geçerli sonuç** | mavi `#2A78D6` | 0,833 · %43,27 · referans çalışma noktası |
| **Düzeltilmiş / iyileşme / geçen kapı** | yeşil `#1BAF7A` | 0,864 · cross-track 0,751 · ρ 0,329 · Motor %60,08 |
| **Bedel / öncesi / uyarı** | turuncu `#EB6834` | monolitik sonuçlar · alarm yükü · anomali development |
| **Kritik / geçersiz / reddedilen** | kırmızı `#D03B3B` | 25,54 alarm/saat · 38,1 alarm/saat · ρ 0,964 · Real %7,81 · mühürlü set |
| **İkincil kategori** | mor `#4A3AA7` | SIL · CUSUM referansı · TCN · parser hatası |
| Üstü çizilmiş / bağlam | gri `#898781` | elenmiş sinir ağları · any-window barları |
| Metin | `#0B0B0B` / `#52514E` | başlık mavisi `#1C5CAB` |

**Palet doğrulandı** (dataviz altı-kontrol validator'ının Python portu, beyaz zemin):
3 kategorik slot all-pairs → CVD ΔE **9,2**, normal-görüş ΔE **24,0**;
6 slot adjacent → CVD ΔE **9,1**, normal-görüş ΔE **22,9**. Hepsi PASS.
Yeşil (2,82:1) ve sarı (2,17:1) beyaz üstünde 3:1'in altında → **relief kuralı**: bu tonlardaki
her bara değeri doğrudan yazılır, eksene bırakılmaz. Yazılıyor.

| Diğer kurallar | |
|---|---|
| Grafik | Izgara çizgisi yok veya çok açık. Y ekseni etiketi yerine bar üstünde doğrudan sayı |
| Yasak | 3B, gölge, gradient dolgu, pasta grafik, çift y-ekseni, ikon kalabalığı |
| Slayt yoğunluğu | Başlık + 1 görsel + en fazla 3 kısa madde. Ayrıntı konuşma notuna gider |
| Sayı biçimi | Türkçe ondalık virgül (0,833). Yüzde işareti sayıdan önce: %43,27 |

---

## C.2 Slayt planı

> **Not:** Aşağıdaki taslak 26 slayt üzerine kurulmuştu. Nihai deck 36 slayt: appendix dağıtıldı
> (bölüm C.3), her yöntem için "bu metot şunu yapar → o yüzden burada kullandık" gerekçesi eklendi.
> Nihai slayt sırası ve konuşma notlarının tamamı `build_final_deck.py` içindeki `SLIDES` tablosunda —
> orası tek doğruluk kaynağıdır, aşağısı gerekçeyi anlatan taslaktır.

### PERDE I — Ne yapıyoruz, neye göre başarılı diyoruz (Slayt 1-5)

---

**SLAYT 1 — Kapak**

> İHA Telemetrisinde Anomaly Detection
> **Sinyalden operasyonel alarma**
>
> Beş veri kaynağı · sekiz yöntem ailesi · beş ölçüm düzeltmesi · bir dürüst karar

**[GÖRSEL]** Yok. Beyaz sayfa, tek satır lacivert ince çizgi, altta kurum adı.

---

**SLAYT 2 — Bu çalışma neyi iddia ediyor, neyi iddia etmiyor**

**[SLAYT METNİ]**
- **İddia:** İHA telemetrisinde hangi arıza ailesinde ve hangi veri koşulunda güvenilir
  alarm üretilebildiğini, hangilerinde üretilemediğini ölçülebilir biçimde gösterdik.
- **İddia etmiyoruz:** Her arıza tipini kapsayan genel bir UAV anomaly detector veya
  sahaya konabilecek dondurulmuş bir threshold.
- **Bu deck'in tek kuralı:** Yakalama oranı ve yanlış alarm yükü hiçbir slaytta ayrı verilmez.

**[KONUŞMA]** "Bu projede asıl risk düşük skor değildi; iyi görünen bir skorun gerçek
zamanda taşınabilir alarm yüküne dönüşmemesiydi. Bu yüzden başarı tanımını en başta
sabitledik ve bir kere bile gevşetmedik. Bugün göreceğiniz her sayı bu tanıma göre üretildi."

**[GÖRSEL]** Yok veya çok sade: iki sütunlu "İddia / İddia değil" kartı, çerçevesiz.

---

**SLAYT 3 — Problem: aynı veriden üç farklı soru sorulabilir**

**[SLAYT METNİ]** Üç düzey birbirinin yerine geçmez:
- **Window** — modelin skor ürettiği kısa telemetri dilimi
- **Event** — ardışık alarm pencerelerinin birleşip gerçek arıza aralığıyla eşleşmesi
- **Flight** — "bu uçuşta en az bir alarm var mı"

**[KONUŞMA]** "Proje boyunca en pahalı hatamız bu üçünü karıştırmaktı. Bir uçuşta alarm
görülmesi, arızanın yakalandığı anlamına gelmiyor — birazdan bunun sayısal kanıtını
göstereceğim. Operatör açısından anlamlı olan tek düzey ortadaki: doğru zaman aralığında
açılan bir alarm."

**[GÖRSEL]** Tek bir zaman ekseni. Üstte pencere kutucukları, ortada gerçek arıza bandı
(açık lacivert dolgu), altta uçuş çubuğu. Üç düzey aynı eksende hizalı. Renk: lacivert + gri.

---

**SLAYT 4 — Başarı ölçütü ve neden bu kadar zor**

**[SLAYT METNİ]**
- Geçerli alarm = arıza **başladıktan sonra** açılan ve arıza aralığıyla kesişen alarm
- Arıza başlamadan açılmış alarm, sonradan aralığa girse bile yakalama sayılmaz
- Ölçüt çifti: **event recall** ve **false event / normal flight-hour**

**[KONUŞMA]** "Zorluk üç yerden geliyor. Arıza örneği az ve pahalı — gerçek uçakta arıza
üretmek riskli, ALFA'nın rudder sınıfında sadece 4 uçuş var. Normal davranış ise çok
çeşitli; agresif bir manevra ile bir arıza telemetride benzer görünebiliyor. Ve aynı skor
farklı uçuş fazında farklı anlama geliyor. Bu yüzden AUC'yi hiçbir zaman tek başına başarı
saymadık; AUC yalnız sıralama gücü gösterir, alarm garantisi vermez."

**[GÖRSEL]** Tek zaman ekseni: gerçek arıza bandı + üç alarm oku — biri erken (geçersiz,
gri), biri aralık içinde (geçerli, lacivert), biri geç (geçersiz, gri).

---

**SLAYT 5 — Değerlendirme sözleşmesi: dört kural, en baştan dondurulmuş**

**[SLAYT METNİ]**
- **Group-safe split** — aynı kaynak/oturum/senaryo ailesi iki role birden giremez
- **Causal scoring** — t anındaki skor yalnız `x ≤ t` ile üretilir
- **Frozen policy** — ölçekleme, checkpoint ve threshold yalnız validation'da seçilir, test
  sonucuna göre oynatılmaz
- **True-onset detection** — alarm arızadan sonra başlamalıdır

**[KONUŞMA]** "Bu dört kural sonradan eklenmedi; her biri bir hatanın bedelini ödedikten
sonra kurala dönüştü — birazdan dördünün de nasıl doğduğunu tek tek göreceksiniz. Şunu
şimdiden söyleyeyim: bu kuralların her biri elimizdeki en iyi sayıyı düşürdü. Sunum boyunca
göreceğiniz sayı düşüşlerinin çoğu model kötüleşmesi değil, ölçümün dürüstleşmesidir."

**[GÖRSEL]** 2×2 kart ızgarası, çerçevesiz, her kartta bir kural + tek satır tanım.

---

### PERDE II — Veriyi nasıl kullanılabilir hâle getirdik (Slayt 6-9)

---

**SLAYT 6 — Neden dışarıdan gözlem yetmedi**

**[SLAYT METNİ]**
- İlk plan hava trafiği yayınıyla başlıyordu — rota sapması, ani irtifa değişimi ve sinyal
  kaybı **dışarıdan** görülebilir
- Motor gücünün düşmesi, bir sensörün bozulması, GPS spoofing ile gerçek manevra ayrımı
  **görülemez**
- Sonuç: dedektörü sayısal olarak doğrulamak için araç içi etiketli arıza verisi gerekti

**[KONUŞMA]** "Projeye hava trafiği verisiyle başladık. Kısa sürede şunu gördük: dışarıdan
gözlem aracın içinde ne olduğunu söylemiyor. Bu veriyi bırakmadık — ama rolünü değiştirdik.
Etiket üretemediği için 'arıza yakalama' verisi olmaktan çıktı, devasa gerçek normal trafik
sunduğu için 'yanlış alarm stres testi' verisi oldu. Bu arada kimlik doğrulama ve kota
kısıtları nedeniyle sağlayıcıyı da değiştirdik; çalışmanın tamamı `adsb.lol` verisi üzerinde."

**[GÖRSEL]** İki panelli karşılaştırma: solda "dışarıdan gözlem" (konum/irtifa/hız izleri —
3 kanal), sağda "araç içi telemetri" (komut, tepki, sensör, EKF, motor — 8+ kanal).
Altında tek satır: *görünen kanal sayısı değil, gözlenebilir arıza sayısı belirleyicidir.*

---

**SLAYT 7 — Beş veri kaynağı, beş ayrı teknik soru**

**[SLAYT METNİ — tablo]**

| Kaynak | Ne sağladı | Hangi soruyu cevapladı |
|---|---|---|
| ADS-B (adsb.lol) | ~497.571 skorlanabilir normal flight-hour, etiket yok | Alarm bütçesi gerçek trafikte ne kadar hızlı bozulur? |
| ALFA | Sabit kanatlı gerçek uçuş, motor ve control-surface arıza aralıkları | Etiketli ama küçük veride ne öğrenilebilir? |
| UAV Attack | Ping DoS, GPS spoofing, jamming senaryoları | Etiketin varlığı, ölçülebilir sinyalin varlığı mıdır? |
| UAV-SEAD | 1.044 development uçuşu | Uçuş sayısı bağımsız deney sayısı mıdır? |
| RflyMAD | 6.605 uçuş; Real / HIL / SIL; 5 arıza ailesi | Domain farkı ve gerçek interval truth altında ne kalır? |

**[KONUŞMA]** "Bu beş kaynak beş ayrı sonuç üretmek için seçilmedi. Her biri problemin
başka bir bileşenini izole ediyor: etiket kalitesi, fiziksel gözlenebilirlik, veri
bağımsızlığı, domain kayması ve yanlış alarm maruziyeti. Bir kaynağın zayıf olduğu yer,
diğerinin test alanı oldu. Dürüst olmak gerekirse: her yöntemi her veri setinde aynı
kapsamda çalıştırmadık, yöntemleri hedefli kullandık."

**[GÖRSEL]** Tablo, çizgisiz. Sağ sütun lacivert vurgulu.

---

**SLAYT 8 — Ham kolonlardan feature'a: veri ilk hâliyle kullanılamıyordu**

**[SLAYT METNİ]** ALFA örneği — üç somut müdahale:
- `velocity_mps` kolonu %100 boş görünüyordu. Sebep veri eksikliği değil, **topic/kolon
  eşleşme hatası**ydı. Düzeltildi.
- İşlenmiş kayıtların ham veriyi tam kapsamadığı görüldü: 8 eksik aday denetlendi, 7'si
  yeniden parse edildi → normal uçuş havuzu **10 → 15**
- Bu iki düzeltmenin ardından feature sayısı **73 → 85**

**[KONUŞMA]** "Buradaki ders şu: model kurmadan önce en az bir kez veriyi kolon kolon
denetlemek gerekiyor. Boş görünen bir kolon çoğu zaman eksik veri değil, yanlış eşleştirme
oluyor. Ve normal uçuş sayısını 10'dan 15'e çıkarmak küçük bir sayı gibi görünüyor ama
normal-only bir modelde normal havuzunu %50 büyütmek demek — birazdan bunun sonuca etkisini
göreceksiniz."

**[GÖRSEL]** Veri doluluk haritası (`completeness_heatmap` artefaktı) — düzeltme öncesi
boş sütun kırmızı değil koyu gri; sonrası dolu. Sade, tek renkli heatmap.

---

**SLAYT 9 — Feature tasarımı: arıza "motor_failure" adlı bir kolonda gelmiyor**

**[SLAYT METNİ]**
- Aranan şey ham büyüklük değil, arızanın uçuş dinamiğinde bıraktığı **ikincil fiziksel iz**
- Üretilen aileler: command–response residual, attitude/airspeed error, route deviation,
  freeze counters, CUSUM birikimleri
- UAV Attack'ta en değerli tek feature: **GPS speed residual** — GPS'in raporladığı hız ile
  ardışık konumlardan hesaplanan hızın farkı

**[KONUŞMA]** "Somut örnek: motor arızasında irtifa hemen düşmez. Önce motor komutu yüksek
kalırken airspeed azalır, sonra autopilot telafi ederken attitude davranışı değişir. Yani ham
irtifa kanalına bakan bir model arızayı geç görür; komut ile tepki arasındaki farka bakan bir
model erken görür. Feature engineering'in amacı modele ezber verdirmek değil, arızanın izini
görünür hale getirmek."

**[GÖRSEL]** Yatay akış şeması: `komut (throttle/attitude)` → `ölçülen tepki` →
`residual = tepki − beklenen tepki` → `context (phase/cadence)` → `event score`.
Beş kutu, ince lacivert oklar, gölgesiz.

---

### PERDE III — Yöntemler: her adım bir öncekinin bıraktığı boşluğu test etti (Slayt 10-14)

---

**SLAYT 10 — Neden normal-only öğrenme**

**[SLAYT METNİ]**
- Model **yalnız normal uçuşlarla** eğitilir; etiketler sadece threshold kalibrasyonu ve
  değerlendirmede kullanılır, eğitimde asla
- Gerekçe: arıza örneği az ve heterojen; supervised model gördüğü arıza tipine aşırı uyum
  sağlar ve görmediğine kör kalır
- İstisna açıkça işaretlendi: LightGBM ve supervised TCN ayrı karşılaştırma kollarıydı

**[KONUŞMA]** "Bu bir tercih değil, veri yapısının dayattığı bir zorunluluktu. Normal veri
büyütülebilir bir kaynak — daha çok uçuş, daha çok saat. Etiketli arıza verisi büyütülemez;
gerçek uçakta arıza üretmek riskli ve pahalı. Bu yüzden ana hattı normal-only kurduk. Ve
denetimli alternatifi de körü körüne reddetmedik — denedik, birazdan sonucunu göstereceğim."

**[GÖRSEL]** İki kutu şeması: `Normal uçuşlar → MODEL` (kalın lacivert ok) /
`Arızalı uçuşlar → yalnız değerlendirme` (kesikli gri ok, modele girmiyor).

---

**SLAYT 11 — İlk yöntem ve ilk ders: tek modele her şeyi vermek sinyali seyreltiyor**

**[SLAYT METNİ]**
- **Motiv:** Fiziksel feature'lar basit bir normal-only skorla ayrışıyor mu? Isolation Forest
  hızlı ve yorumlanabilir bir baseline verdi.
- **Gözlem:** Monolitik kurulumda güçlü ama dar bir sinyal, alakasız feature'lar içinde
  seyreldi — ALFA'da rastgele sıralamaya yakın, UAV Attack'ta 0,21.
- **İşlem:** Feature'lar fiziksel modüllere ayrıldı (control-response, guidance, navigation,
  signal quality), her modül ayrı skorlandı, skorlar normal validation verisinde ölçeklenip
  birleştirildi.
- **Sonuç:** ALFA flight ROC-AUC **0,833**; en güçlü tek modül (guidance) **0,864**.

**[KONUŞMA]** "Kazanım modelden gelmedi — model aynı Isolation Forest. Kazanım sinyalin
seyrelmesini önlemekten geldi. Bir uyarı: 0,21 ve 0,833 farklı veri setlerine ait, aynı
modelin önce/sonra karşılaştırması değil; monolitik yaklaşımın iki ayrı veri setinde de
zayıf kaldığının kanıtı."

**[GÖRSEL]** Dört bar, tek eksende ama iki gruba ayrılmış:
`Monolitik — ALFA (rastgeleye yakın, gri)` · `Monolitik — UAV Attack 0,21 (gri)` ‖
`Modüler — ALFA 0,833 (lacivert)` · `En güçlü tek modül 0,864 (lacivert)`.
Grafik altı notu: *ilk iki bar farklı veri setlerine aittir.*
⚠️ **Bu grafik v3'te ters etiketliydi — H1'e bakınız.**

---

**SLAYT 12 — Derin modele geçiş: aynı model, daha çeşitli veri**

**[SLAYT METNİ]**
- **Motiv:** Isolation Forest zaman sırasını kullanmıyordu. Dense ve LSTM autoencoder ile
  sequence reconstruction denendi.
- **Teknik detay:** Eksik sensör hücreleri sıfır hata sayılmadı; yalnız gözlenen değerler
  loss'a katkı verdi.
- **İlk sonuç:** Küçük havuzda flight ROC-AUC **0,731** — modüler Isolation Forest'ın gerisinde.
- **Slayt 8'deki veri onarımından sonra, model hiç değişmeden:** **0,918**.

**[KONUŞMA]** "Kolay yorum 'derin model bu problemde çalışmıyor' olurdu. Ama 5 normal uçuş
eklediğimizde aynı mimari çok farklı bir sıralama sinyali üretti. Küçük veride bir derin
modelin zayıf çıkması her zaman modelin suçu değildir. **Ama şunu şimdi söylemem lazım:**
bu 0,918 nihai bir sonuç değil, iki slayt sonra 0,611'e düşecek — çünkü ölçümde bir hata
bulacağız. Bu sayıyı başarı olarak sunmuyorum, sadece veri çeşitliliğinin etkisini gösteriyorum."

**[GÖRSEL]** İki bar: `Küçük normal havuz — 0,731 (gri)` → `+5 normal uçuş — 0,918 (lacivert)`.
Sağda **kırmızı kenarlı uyarı kartı**: *"Bu değer causal düzeltmeden öncedir. Düzeltilmiş
karşılığı Slayt 14'te."*

---

**SLAYT 13 — Sekiz yöntem ailesi denendi; hiçbiri rastgele seçilmedi**

**[SLAYT METNİ — tablo]**

| Yöntem | Neden denendi | Ne çıktı |
|---|---|---|
| Isolation Forest | Normal-only hızlı baseline | ALFA modüler 0,833 — dirençli rakip kaldı |
| LSTM / LSTM-AE / Dense-AE | Zaman yapısının katkısı | Genlik artefaktı bulundu (Slayt 16) |
| EKF innovation | Uçuş kontrolcüsünün kendi tutarlılık göstergesi | **0,354 — ters yön** |
| Supervised LightGBM | Denetimli öğrenme daha mı iyi? | AP 0,349 < Isolation Forest 0,385 |
| Chronos (zero-shot forecasting) | Hazır bir zaman serisi modeli yeter mi? | Mechanical recall 0,205 → **0,390** |
| Tek feature: thrust command | En güçlü tek kanal ne kadar götürür? | **0,459** ama **38,1 yanlış alarm/saat** |
| İki kanallı fusion | Kapsamı genişletmek | Yanlış alarm **×1,89–3,83** |
| Gaussian NLL forecaster | Hata ve belirsizliği birlikte skorlamak | Final protokol (Slayt 18) |

**[KONUŞMA]** "İki tanesini vurgulayayım. EKF innovation ters yönde sinyal verdi — sebebi
şu: anomalili ölçüm estimator tarafından reddedildiğinde innovation düzenli üretilmiyor,
yani en sorunlu örnekler daha 'temiz' görünüyor. Bu bir bug değil, göstergenin doğası. İkincisi:
tek bir feature, thrust command, en yüksek kategori yakalamasını verdi — ama saatte 38 yanlış
alarmla. Soldan sağa yakalama gerçekten yükseliyor; her adımda alarm yükü daha hızlı yükseldi.
Bu tablo 'daha iyi model bulduk' değil, 'aynı duvara farklı yollardan çarptık' diyor."

**[GÖRSEL]** Tablo. Son sütunda 0,354 ve 38,1 kırmızı; 0,390 ve 0,459 lacivert.
⚠️ **Bu slayt v3'te tamamen eksikti — B.1'e bakınız.**

---

**SLAYT 14 — Etiketsiz devasa trafikte iki ders**

**[SLAYT METNİ]**
- **Ölçek:** 256.150.550 satır işlendi, ~497.571 skorlanabilir flight-hour
- **Ders 1 — sentetik başarı tuzağı:** Bir yaklaşım sentetik anomalilerin **%97,6**'sını
  yakaladı; **aynı sistem doğal trafikte 25,54 alarm/saat** üretti. Yaklaşım reddedildi ve
  arşivlendi. Bu olaydan sonra yakalama oranını alarm yükü olmadan raporlamak yasaklandı.
- **Ders 2 — basit fizik kuralları sinir ağlarını geçti:** 5 tutarlılık kuralı (dikey
  hız ↔ irtifa türevi, yer hızı ↔ konum türevi …) pooled AUC **0,600**; karşılaştırılan üç
  sinir ağı 0,552–0,572.

**[KONUŞMA]** "İlk ders bu projenin dönüm noktasıydı. Sentetikte %97,6 gören bir sistemin
gerçek trafikte saatte 25 alarm üretmesi, o sistemin kullanılamaz olduğu anlamına gelir.
Yaklaşımı arşivledik ve raporlama kuralını değiştirdik. İkinci ders daha ilginç: basit fizik
kuralları sinir ağlarını geçti. Ama dürüst olayım — karşılaştırdığımız o üç sinir ağı zaten
ayrı bir kontrolde elenmişti, birazdan anlatacağım genlik problemi yüzünden. Yani bu kuralın
zaferi kadar, sinir ağlarının o turdaki geçersizliğinin de sonucu."

**[GÖRSEL]** Sol: iki büyük sayı yan yana — `%97,6 sentetik recall` (lacivert) ve
`25,54 alarm/saat` (kırmızı), aralarında kalın eşitsizlik işareti.
Sağ: dört bar — `Fizik kuralları 0,600` lacivert, üç NN gri, altlarında küçük gri etiket
*"bu üç model genlik kontrolünde elendi"*.

---

### PERDE IV — Beş ölçüm düzeltmesi: sayılar neden düştü (Slayt 15-19)

---

**SLAYT 15 — Düzeltme 1: model geleceğe bakıyordu**

**[SLAYT METNİ]**
- **Motiv:** Tek pencere eşiği küçük ama kalıcı driftleri kaçırıyordu. CUSUM, sapma
  kanıtını zaman içinde biriktirmek için eklendi.
- **Sorun:** CUSUM'un baseline'ı (median/MAD) **uçuşun tamamından** hesaplanıyordu. Yani
  arıza sonrası değerler, arıza öncesi skoru dolaylı olarak etkiliyordu.
- **Düzeltme:** Baseline yalnız normal training uçuşlarından hesaplandı; t anındaki skor
  yalnız `x ≤ t` ile üretildi.
- **Sonuç:** En güçlü sanılan sinyal **0,878 → 0,611**. Düzeltmeden sonraki gerçek en güçlü
  tek sinyal: cross-track error, **0,751**.

**[KONUŞMA]** "CUSUM'u eklememizin sebebi netti: tek pencerede görünmeyen ama süren bir
sapmayı biriktirerek görünür kılmak. Yöntem doğruydu, kurulumunda hata vardı. Baseline'ı
uçuşun tamamından hesaplayınca model, geçmişte karar verirken gelecekteki veriden
etkileniyordu. Bu sinsi bir hata — kod çalışıyor, sonuç iyi görünüyor, ama gerçek zamanda
üretilemeyecek bir skor üretiyorsunuz. Düzelttik, sayı 0,878'den 0,611'e düştü. Bu bir
performans kaybı değil; o 0,878 hiçbir zaman gerçek değildi."

**[GÖRSEL]** Üç bar: `Full-flight baseline 0,878 (gri, üstünde küçük "geçersiz" etiketi)` →
`Causal düzeltme sonrası 0,611 (lacivert)` · `Düzeltme sonrası en güçlü sinyal:
cross-track 0,751 (lacivert)`. Aralarında ince ok.

---

**SLAYT 16 — Düzeltme 2: alarm arızadan önce açılmışsa yakalama değildir**

**[SLAYT METNİ]**
- **Sorun:** Eski değerlendirici yalnız **çakışmaya** bakıyordu. Arıza başlamadan açılıp
  aralığa kadar süren bir alarm da "yakalandı" sayılıyordu.
- **Düzeltme:** Değerlendirici, arıza başladıktan **sonra** yeni bir alarm başlangıcı arıyor.
- **Sonuç:** Çakışma tabanlı yakalama **%59,4** iken gerçek başlangıç yakalaması
  **%19,4–22,4**'e düştü.
- Bu ilke sonradan RflyMAD ve ADS-B değerlendirmelerinde de aynen kullanıldı.

**[KONUŞMA]** "Operasyonel olarak düşünün: alarm zaten açıktı, arıza sonra başladı. Bu
alarm arızayı haber vermedi, tesadüfen üstüne denk geldi. Ölçüm bunu yakalama sayarsa,
sistemin gerçek uyarı kabiliyetini üç kat fazla gösterir — nitekim öyle olmuş. Bu düzeltmeden
sonra elimizdeki bütün geçmiş event sonuçlarını geçersiz saydık ve yeniden hesapladık."

**[GÖRSEL]** İki satırlı zaman ekseni. Üstte: alarm arızadan önce başlıyor, aralıkla
kesişiyor → gri, "sayılmaz". Altta: alarm aralık içinde başlıyor → lacivert, "sayılır".
Sağda iki sayı: `%59,4` (gri) → `%19,4–22,4` (lacivert).
⚠️ **v3'te ilke vardı, sayı yoktu — K8'e bakınız.**

---

**SLAYT 17 — Düzeltme 3: uçuş sayısı bağımsız deney sayısı değil**

**[SLAYT METNİ]**
- **Motiv:** UAV-SEAD havuzu 60'tan 1.044 development uçuşuna büyütüldü. Daha çok veri,
  daha güvenilir sonuç beklendi.
- **Sorun:** Aynı gün ve aynı oturumdan gelen akraba uçuşlar train ve test rollerine
  dağılıyordu. Model, test uçuşunun neredeyse aynısını eğitimde görmüştü.
- **Düzeltme:** Session-level split — akraba uçuşlar aynı role kilitlendi.
- **Sonuç:** Seed oynaklığı **±0,212 → ±0,012**; adil satır AUC **0,474 → 0,799**.

**[KONUŞMA]** "Buradaki iyileşme yeni bir modelden gelmedi, doğru veri bölünmesinden geldi —
ve dikkat edin, bu sefer sayı **yükseldi**. Oynaklığın on yediye bir düşmesi ise asıl kritik
olan: öncesinde aynı deneyi farklı rastgele tohumla çalıştırdığınızda çok farklı sonuç
alıyordunuz, yani hiçbir sonuca güvenilemezdi. Bu dersi sonrasında bütün veri setlerine
taşıdık: RflyMAD'de senaryo ve domain grupları, UAV Attack'ta kampanya ve platform grupları."

**[GÖRSEL]** İki bar grubu yan yana: solda `Seed oynaklığı ±0,212 → ±0,012`, sağda
`Adil satır AUC 0,474 → 0,799`. Gri → lacivert.

---

**SLAYT 18 — Düzeltme 4: model örüntü değil, sinyal büyüklüğü öğreniyordu**

**[SLAYT METNİ]**
- **Şüphe:** Üç farklı derin mimari (LSTM-AE, Dense-AE, USAD) neredeyse aynı sonucu verdi.
  Bu, mimarilerin iyi olmasından çok, üçünün de aynı kestirme yolu bulmuş olmasına benziyordu.
- **Teşhis:** Eğitilmiş model skoru ile **hiç eğitilmemiş rastgele** model skoru arasında
  Spearman **ρ ≈ 0,964**; ham sinyal büyüklüğü ile **ρ ≈ 0,965**.
- **Doğrulama:** Skorlar göreli hata ile yeniden hesaplandığında genlik korelasyonu
  0,15–0,55'e düştü — ama **yakalama %6'nın altına indi**. Önceki kazancın büyük kısmı sahteydi.
- **Karar:** Rastgele-model kontrolü o günden sonra **zorunlu güvenlik kapısı** oldu (eşik ρ < 0,80).

**[KONUŞMA]** "Bu projenin en önemli bulgusu. Eğitilmiş modelin skoru ile hiç eğitilmemiş bir
modelin skoru neredeyse mükemmel korele — yani ağırlıkların hiçbir katkısı yok. Model
'anomali' öğrenmemiş, sadece 'büyük değer' öğrenmiş. Agresif bir manevrada roll büyüktür,
model onu anomali sanar. Bunu doğrulamak için skoru genlikten arındırdık; genlik bağımlılığı
düştü ama yakalama da çöktü — demek ki elimizdeki sinyal zaten büyüklükten ibaretti. O günden
sonra hiçbir modeli bu kontrolden geçmeden rapor etmedik. Bu kapı, sunumun geri kalanındaki
bütün sonuçların ön şartıdır."

**[GÖRSEL]** İki panel. Sol: dağılım grafiği — x ekseni rastgele model skoru, y ekseni
eğitilmiş model skoru, noktalar neredeyse diyagonal üstünde, köşede `ρ = 0,964`.
Sağ: iki bar — `Genlik düzeltmesi öncesi recall` vs `sonrası < %6`, ikincisi çarpıcı biçimde kısa.
⚠️ **v3'te tamamen eksikti — B.1'e bakınız.**

---

**SLAYT 19 — Düzeltme 5: soruyu değil modeli düzeltmeye çalışıyorduk**

**[SLAYT METNİ]**
- **Sorun 1 — etiketin kapsamı:** RflyMAD'de arızalı uçuşun **tamamı** anomali sayılıyordu.
  Gerçek arıza aralığı truth'una geçilince yakalama **0,749 → 0,526**, yanında **22,28
  yanlış alarm/saat**.
- **Sorun 2 — parser hatası:** 2.712 SIL/HIL uçuşunda ayrıştırıcı **t=0'da sahte arıza**
  üretiyordu. Etkilenen uçuşlar yeniden işlendi; ilk örnekten itibaren aktif görünen arıza
  sayısı 1.354'ten 0'a indi.
- **Modelde hiçbir değişiklik yapılmadı.** Yalnız sorulan soru düzeltildi: *"bu uçuş arızalı
  mı"* yerine *"gerçek arıza aralığını buldun mu"*.

**[KONUŞMA]** "İki hata da modelin değil, ground truth'un hatasıydı. Birincisi ölçüm
tanımından geliyordu: arızalı bir uçuşun tamamını anomali saymak, modelin uçuşun herhangi bir
yerinde alarm vermesini yeterli kılıyor. İkincisi bir yazılım hatasıydı — 2.712 uçuşta arıza
sanki ilk saniyeden başlıyor gibi görünüyordu. Bu hatadan etkilenen bütün geçmiş sonuçları
geçersiz ilan ettik. Projedeki en büyük kazanımlardan biri yeni bir model değil, doğru bir
ground truth oldu."

**[GÖRSEL]** Sol: iki bar `Tüm-uçuş etiketi 0,749 (gri)` → `Gerçek aralık truth 0,526
(lacivert)`, altında kırmızı küçük kart `+22,28 yanlış alarm/saat`.
Sağ: sayı kartı `2.712 uçuş yeniden işlendi · sahte t=0 arıza: 1.354 → 0`.

---

### PERDE V — Final protokol ve sonuç (Slayt 20-25)

---

**SLAYT 20 — Final yaklaşım: reconstruction yerine next-step forecasting**

**[SLAYT METNİ]**
- Model son **32 causal adımı** okur ve bir sonraki telemetri vektörünü tahmin eder — tek bir
  değer değil, **kanal başına bir dağılım (μ, σ)**
- Gövde: tek katmanlı LSTM, 64 gizli birim
- Ön işleme: yalnız training verisinden robust ölçekleme (median, 1,4826×MAD), [-5,+5] clip;
  5 saniyeden uzun boşluklarda ve uçuş sınırında pencere kesilir
- Aktif kanal: 28 kanaldan 24 (training'de MAD'i sıfır çıkan kanallar kalibre edilemez sayılıp çıkarıldı)

**[KONUŞMA]** "Reconstruction'dan forecasting'e geçmemizin sebebi doğrudan bir önceki
slayttaki genlik problemi. Bir autoencoder girdiyi yeniden kurmaya çalışır ve büyük değerleri
yeniden kurmak doğal olarak zordur — bu yüzden skor büyüklüğü takip eder. Forecasting farklı
bir soru sorar: 'bir sonraki adım ne olmalı'. Agresif bir manevrada değerler büyük olabilir
ama komuta uygun tepki varsa tahmin hatası küçük kalır. Bu, genlik baskınlığını yapısal
olarak azaltır. Ve şunu da net söyleyeyim: bu bir algoritma değil, bir protokoldür — asıl
değeri, sonuçların dört veri seti arasında karşılaştırılabilir olmasını sağlaması."

**[GÖRSEL]** Yatay altı kutulu akış: `son 32 causal adım` → `LSTM 64` →
`kanal başına μ, σ` → `masked Gaussian NLL` → `threshold + persistence` → `event`.
İlk üç kutu "tahmin", son üç kutu "karar" diye üstten iki gri parantezle ayrılmış.

---

**SLAYT 21 — Skor: masked Gaussian NLL**

**[SLAYT METNİ]**

> `NLL ≈ ½ · ((x − μ)² / σ²) + log(σ)`, yalnız gözlenen kanallar üzerinden

- **Küçük hata + düşük belirsizlik** → model eminse küçük bir sapma bile anlamlı anomali
- **Büyük hata + yüksek belirsizlik** → doğal değişkenliğin yüksek olduğu bölgede aynı hata
  daha az anomalik
- **Masked:** eksik sensör hücreleri loss ve skor dışında tutulur; eksik veri skoru şişirmemeli

**[KONUŞMA]** "Bu skorun klasik reconstruction error'dan ayrıldığı yer burası: 'tahmin ne
kadar yanlıştı' ile 'model bu bölgede ne kadar emindi' sorularını aynı anda cevaplıyor.
Bir teknik not: validation NLL'imiz negatif çıkıyor. Bu bir hata değil — kod sabit terimi
eklemiyor ve σ birden küçük olduğunda log(σ) negatif oluyor. Daha negatif bir validation
NLL yalnızca normal veriyi daha iyi açıklayan bir checkpoint demektir; anomali yakalama
kanıtı değildir."

**[GÖRSEL]** Tek şema: aynı büyüklükte iki hata oku, biri dar bir σ bandının içinde
(yüksek skor, lacivert), diğeri geniş bir σ bandının içinde (düşük skor, gri).
Formül üstte, ince tipografi. Grafik yok.
⚠️ **v3'teki `[1,1,1,1]` sahte grafiği kaldırılmalı — K7.**

---

**SLAYT 22 — Bağımsızlık sözleşmesi: beş rol, sıfır kesişim**

**[SLAYT METNİ — RflyMAD rol kartı]**

| Rol | Uçuş | Bağımsız grup | Ne için kullanıldı |
|---|---:|---:|---|
| Normal training | 260 | 7 | Ölçekleyici ve model fit |
| Normal validation | 90 | 3 | Checkpoint, threshold ve event policy seçimi |
| Bağımsız normal test | 91 | 3 | Yanlış alarm yükü ölçümü |
| Anomali development | 557 | 12 | Teşhis ve event değerlendirmesi |
| **Mühürlü final test** | **553** | **13** | **Hiç açılmadı** |

**[KONUŞMA]** "Genelleme iddiası, akraba uçuşlar rollere dağılmadan yapılamaz. Her veri
setine kendi bağımsızlık birimini tanımladık; RflyMAD'de bu senaryo ve domain gruplarıydı.
Son satır önemli: 553 uçuşluk final test setini mühürledik ve **hiç açmadık**. Çünkü
development sonucu geçme eşiğimizin altında kaldı — mühürlü seti açmak, sonuca bakıp karar
vermek olurdu. Bu setin kapalı kalması bu projenin en savunulabilir kararıdır."

**[GÖRSEL]** Tablo + sağda beş dikey bant (uçuş sayısına orantılı yükseklikte), aralarında
kalın beyaz boşluk. Son bant kesikli çerçeveli ve gri — "mühürlü".
⚠️ **v3'te tamamen eksikti — B.1.**

---

**SLAYT 23 — Eğitim ve öğrenme kontrolü: ilk kez temiz geçen model**

**[SLAYT METNİ]**
- 30 epoch eğitildi; checkpoint **yalnız normal validation NLL** ile seçildi (seçilen: 30)
- Dondurulmuş threshold: **10,0940**
- **Genlik kontrolü:** eğitilmiş–rastgele **ρ = 0,3288**, eğitilmiş–genlik **ρ = 0,3495** —
  kapı 0,80'in belirgin altında ✅
- Karşılaştırma: aynı kontrol daha önce **0,964/0,965** vermişti

**[KONUŞMA]** "Bu, projedeki tek gerçek 'iyileşme' hikâyesi. Slayt 18'de gösterdiğim
0,964'ten buraya, 0,33'e geldik — model artık gerçekten örüntü öğreniyor, büyüklük
kopyalamıyor. Ama şunu net söylemem lazım: bu kapıyı geçmek 'model başarılı' demek değil.
Yalnızca 'bu spesifik kestirme yolunu kullanmıyor' demek. Tespit performansı ayrı bir soru
ve cevabını bir sonraki slaytta vereceğim."

**[GÖRSEL]** İki bar yan yana: `Önceki tur ρ = 0,964 (gri)` ve `Bu tur ρ = 0,329 (lacivert)`,
aralarında yatay kesikli **kapı çizgisi ρ = 0,80** — ilk bar çizgiyi aşıyor, ikincisi altında.

---

**SLAYT 24 — Dört veri setinde ne çıktı**

**[SLAYT METNİ — tablo]**

| Veri seti | Interval ROC-AUC | Event recall | Yanlış olay / normal saat |
|---|---:|---:|---|
| RflyMAD | **0,907** | **%56,1** | **0,768** |
| UAV-SEAD | 0,747 | %3,40 (206 olayın 7'si) | 0 |
| ALFA | — | ~%20 | 0 — *ama yalnız 0,131 saat bağımsız normal maruziyet* |
| UAV Attack | 0,000 (uçuş) | interval truth yok | **32,42** |

**[KONUŞMA]** "Dört sonuç, dört farklı ders. RflyMAD tek koşullu 'devam' kararı — sıralama
sinyali gerçekten var. UAV-SEAD çarpıcı: interval AUC 0,747, yani skor sıralama yapabiliyor;
ama 206 gerçek olayın sadece 7'sini bulmuş. Sıralama sinyalinin olay yakalamaya
dönüşmediğinin en net kanıtı bu. ALFA'da sıfır yanlış alarm görüyorsunuz — bunu başarı
diye sunmuyorum, çünkü bağımsız normal test maruziyetimiz sadece 0,131 saat; sıfır alarm
bu maruziyette güçlü kanıt değil. UAV Attack ise zaten interval truth'u olmayan bir veri
seti; oradaki tek anlamlı sayı 32 yanlış olay/saat."

**[GÖRSEL]** Tablo. RflyMAD satırı hafif lacivert dolgu. ALFA'nın "0" hücresinde küçük
gri dipnot işareti. Grafik yok — tablo yeter.
⚠️ **v3'te bu tablo tamamen yoktu; sadece RflyMAD gösteriliyordu — B.1.**

---

**SLAYT 25 — Çalışma noktası: yakalama ve alarm yükü aynı yerde buluşmadı**

**[SLAYT METNİ]**
- Referans çalışma noktası (2-of-3 persistence): **%43,27 event recall @ 0,654 yanlış olay/saat**
- Daha yüksek yakalama seçeneği: **%58,71 @ 9,153 yanlış olay/saat**
- Yani **+15 puan recall için alarm yükü 14 kat** arttı
- 144 aday çalışma noktasından 54'ü ön-kayıtlı olarak değerlendirildi; hiçbiri hedefi tutturmadı

**[KONUŞMA]** "Bu slayt sunumun en önemli sayısal sonucu. Threshold'u gevşetince daha fazla
arıza yakalıyorsunuz — ama normal uçuşlardaki yanlış alarm çok daha hızlı artıyor. Operatör
açısından saatte 9 yanlış olay demek, sistemin ilk gün kapatılması demek. Sıralama sinyali
var, işletilebilir bir çalışma noktası yok. Ve bunu 54 farklı noktayı önceden kaydedip
değerlendirerek gösterdik — sonuca bakıp en iyi noktayı seçmedik."

**[GÖRSEL]** Tek eğri: x ekseni yanlış olay/saat (log ölçek), y ekseni event recall.
İki nokta işaretli ve doğrudan etiketli. Eğrinin dikliği kendi başına mesajı veriyor.
Renk: tek lacivert çizgi, gri ızgara yok.

---

**SLAYT 26 — Kör noktalar: ortalama yanıltıyor**

**[SLAYT METNİ]**

| Arıza ailesi | Recall | | Domain | Recall | Gecikme |
|---|---:|---|---|---:|---:|
| Motor | %60,08 | | HIL (donanım döngüde) | %56,85 | — |
| **Sensor** | **%5,45** | | SIL (simülasyon) | %38,78 | 17,4 s |
| | | | **Real (gerçek uçuş)** | **%7,81** | **41,2 s** |

**[KONUŞMA]** "Ortalama %43'lük yakalama tek bir yerde toplanıyor: simülasyondaki motor
arızaları. Sistemin asıl hedefi olan gerçek uçuş verisinde yakalama %7,8 — ve yakaladığı
olaylarda bile 41 saniye gecikiyor. Motor arızalarında gecikme 0,1 saniye. Yani model
komut–tepki ilişkisinin bozulduğu ani arızaları görüyor, yavaş gelişen sensör bozulmalarını
görmüyor. Bu bir model ayarı sorunu değil, feature ve gözlenebilirlik sorunu — sıradaki
işin adresi burası."

**[GÖRSEL]** İki küçük yatay bar grubu yan yana. Yüksek değerler lacivert, %5,45 ve %7,81
gri + ince kırmızı kenar. Gecikme sütunu sayı olarak, grafik değil.

---

**SLAYT 27 — Aynı model, iki farklı soru** *(kritik dürüstlük slaydı)*

**[SLAYT METNİ]**
- Soru A — *"Bu uçuşta en az bir alarm var mı?"* → anomali uçuşlarının **%82,94**'ünde alarm
- Aynı eşikle, **normal** uçuşların **%84,62**'sinde de alarm
- Dengeli doğruluk: **%49,16** — yani rastgele tahmin düzeyi
- Soru B — *"Alarm gerçek arıza aralığında mı başladı?"* → **%43,27 @ 0,654 yanlış olay/saat**

**[KONUŞMA]** "Bu slaydı özellikle koydum. Eğer sunumu 'model anomali uçuşlarının %83'ünü
yakaladı' diye bitirseydim, teknik olarak yanlış bir cümle kurmuş olmazdım. Ama aynı eşik
normal uçuşların %85'inde de alarm veriyor — yani model hiçbir ayrım yapmıyor. Dengeli
doğruluk %49, tam olarak yazı-tura. Bu, tek bir metriğin bir sonucu nasıl tamamen ters
gösterebileceğinin kanıtı ve bizim raporlama kuralımızın neden bu kadar katı olduğunun cevabı."

**[GÖRSEL]** İki bar, neredeyse aynı yükseklikte: `Anomali uçuşları %82,94` ve
`Normal uçuşlar %84,62` — ikisi de gri, yan yana, bilinçli olarak ayırt edilemez.
Altında tek satır lacivert: `dengeli doğruluk %49,16`.
⚠️ **v3'te tamamen eksikti — B.1.**

---

**SLAYT 28 — Tek uçuşta bir yakalanan, bir kaçan olay**

**[SLAYT METNİ]**
- **Yakalanan (motor arızası):** skor arıza aralığı içinde eşiği aşıyor, alarm aralık
  içinde başlıyor
- **Kaçan (gerçek uçuş sensör arızası):** skor arıza boyunca eşiğin altında kalıyor; tek
  alarm arıza başlamadan önce üretildiği için yakalama sayılmıyor

**[KONUŞMA]** "Bu iki grafik, sunumdaki bütün sayıların tek uçuş ölçeğindeki karşılığı.
Soldaki, sistemin işe yaradığı senaryo. Sağdaki, en çok önemsediğimiz senaryo — gerçek
uçuşta sensör arızası — ve orada skor hiç yükselmiyor. Sağdaki grafiği sunumdan çıkarabilirdim;
çıkarmadım, çünkü asıl teknik sorunun nerede olduğunu en açık gösteren şey bu."

**[GÖRSEL]** Repodaki **gerçek** artefaktlar: `05_detected_motor_event_timeline.png` ve
`06_missed_real_sensor_timeline.png`, yan yana. Her birinin üstünde Türkçe altbaşlık.
⚠️ **v3'teki uydurma eğriler kaldırılmalı — H4.**

---

### PERDE VI — Karar (Slayt 29-30)

---

**SLAYT 29 — Karar ve sıradaki iş**

**[SLAYT METNİ]**

**Karar: hedef karşılanmadı.** Mühürlü final test seti açılmadı.

| Güçlü olduğu yer | Darboğaz | Sıradaki iş |
|---|---|---|
| Motor / komut–tepki arızaları | Yanlış alarm yükü | Bağımsız normal uçuş süresini artırmak |
| Simülasyon ve HIL domainleri | Gerçek uçuş domain farkı | Zaman etiketli kontrollü arıza kampanyası |
| Fiziksel tutarlılık kuralları | Sensör arızası gözlenebilirliği | Arıza ailesine ve domain'e özel policy |

**[KONUŞMA]** "Kararımız net: bu protokol hedeflenen güvenilirliği karşılamadı, bu yüzden
mühürlü final test setini açmadık. Ama bu sonucun kendisi bir çıktı. Bugün şunları biliyoruz:
alarm bütçesinin gerçek trafikte ne kadar hızlı bozulduğunu; bir modelin örüntü mü büyüklük mü
öğrendiğini nasıl test edeceğimizi; uçuş sayısının bağımsız deney sayısı olmadığını; ve etiket
zaman doğruluğunun model seçiminden daha belirleyici olduğunu. Sıradaki veri kampanyasının
neye ihtiyaç duyduğunu artık tahmin etmiyoruz, ölçtük. Yanlış bir sistemi sahaya vermektense,
iyi belgelenmiş bir olumsuz sonuç vermeyi tercih ettik."

**[GÖRSEL]** Üç sütunlu tablo. "Karar" satırı üstte, lacivert ince çerçeveli tek kutu içinde.

---

**SLAYT 30 — Teşekkür / referanslar**

**[SLAYT METNİ]**
Chandola et al. (2009) — Anomaly Detection: A Survey · Pang et al. (2021) — Deep Learning
for Anomaly Detection · Liu et al. (2008) — Isolation Forest · sequential change detection
(CUSUM) literatürü · UAV telemetry anomaly detection ve GNSS integrity literatürü

Veri kaynakları: adsb.lol · ALFA · UAV Attack · UAV-SEAD · RflyMAD

---

## C.3 Appendix yok — hepsi ana akışta

Kullanıcı kararı (4 Ağustos): ayrı bir appendix bölümü olmayacak. Sonuç kendi slaytında tartışılıp
devam edilecek; yalnız referanslar en son sayfada. İlk taslakta appendix'e ayrılan altı konu ana
anlatıya şu şekilde yerleştirildi:

| İlk taslakta appendix'te | Nihai yeri |
|---|---|
| Sentetik bozulma taksonomisi | **Slayt 13** — "Modelin neye duyarlı olduğunu nasıl ölçtük" |
| Supervised TCN | **Slayt 15** — LightGBM ile birlikte "Denetimli öğrenme neden ana hat olmadı" |
| GNSS bütünlük pilotu | **Slayt 22** — "Dar çerçeve denemesi: yalnız GPS bütünlüğü" |
| Komut→tepki residual | **Slayt 23** — "Genlik problemine yapısal cevap" |
| ALFA group-CV son sözü | **Slayt 29** — "ALFA'nın son sözü" |
| RflyMAD dayanıklılık turu | **Slayt 31** içine ikinci kart olarak ("Bunu düzeltmeyi denedik") |

Ayrıca ilk taslaktaki ayrı "Related work" slaydı (v3'ün 3. slaydı) kaldırıldı — açılışta literatür
anlatmak tempoyu düşürüyordu. Referanslar son sayfada duruyor.

---

## C.4 Üretilen dosyalar

| Dosya | Ne |
|---|---|
| `docs/final_presentation/Final_Sunum_IHA_Anomali_Tespiti_v4.pptx` | **36 slayt**, 16:9, beyaz tema, konuşma notları not panelinde |
| `docs/final_presentation/presentation_assets/final_deck/*.png` + `*.svg` | 29 görsel |
| `.../source_data/render_final_deck.py` | Görselleri üreten script — model eğitmez, eşik değiştirmez, mühürlü teste dokunmaz |
| `.../source_data/build_final_deck.py` | Slayt metni + notlar + yerleşim; pptx'i kurar |

Yeniden üretmek için:

```
python docs/final_presentation/presentation_assets/source_data/render_final_deck.py
python docs/final_presentation/presentation_assets/source_data/build_final_deck.py
```

**Yerleşim kuralı:** Grafik slaytlarında pptx ayrı bir başlık eklemez — görselin kendisi 16:9'dur ve
kendi başlığını/altbaşlığını taşır, tam sayfa yerleşir. Başlık yalnız metin, tablo ve ham artefakt
slaytlarında pptx tarafından çizilir.

**İki gerçek artefakt** (`S20_detected_motor_timeline.png`, `S20_missed_real_sensor_timeline.png`)
olduğu gibi kullanıldı; eksen etiketleri İngilizce olduğu için altlarına "orijinal araştırma çıktısı"
notu düşüldü. Bunlar v3'teki uydurma eğrilerin yerini alıyor (H4).

**Üretim sırasında yakalanan iki kendi hatam:** ilk turda monolitik ALFA sonucu için `0,520` diye bir
sayı basmıştım — kaynak yalnız "rastgeleye yakın" diyor, sayı yok; bar etiketine niteliksel ifade
kondu. İkincisi, çalışma noktası eğrisine ara noktalar uydurmuştum — yalnız raporda geçen üç
ön-kayıtlı nokta (%43,27@0,654 · %48,29@5,884 · %58,71@9,153) çizildi.

---

## C.4 v3 → yeni deck eşleme (neyi at, neyi taşı, neyi ekle)

| v3 slaytı | Karar |
|---|---|
| 1 Kapak | Taşı → yeni 1 (alt başlık sadeleşti) |
| 2 Operational framing | Taşı → yeni 2 |
| 3 Related work | **Appendix'e** — açılışta literatür anlatmak tempoyu düşürüyor; referanslar zaten kapanışta |
| 4 Problem definition | Taşı → yeni 3 + 4 (window/event/flight ayrımı eklendi) |
| 5 Dataset rationale | Taşı → yeni 7, **OpenSky adı düzeltilerek** |
| 6 Evaluation protocol | Taşı → yeni 5 |
| 7 Method progression | Taşı → yeni 13, **sonuç sütunu eklenerek** |
| 8 Feature design | Taşı → yeni 9, **+ yeni 8 (kolon onarımı) eklendi** |
| 9 + 10 ADS-B | **Birleştir → yeni 14**, %97,6'nın reddedilme hikâyesi düzeltilerek |
| 11 CUSUM correction | Taşı → yeni 15 |
| 12 ALFA data audit | Taşı → yeni 8 |
| 13 ALFA baselines | **Böl → yeni 11 + 12**, grafik etiketi düzeltilerek |
| 14 Dataset validation | **Böl → yeni 17** (SEAD split) + UAV Attack gözlenebilirlik yeni 7'nin satırına |
| 15 RflyMAD ground truth | Taşı → yeni 19 |
| 16 Final model | Taşı → yeni 20 |
| 17 Scoring function | Taşı → yeni 21, **sahte grafik kaldırılarak** |
| 18 Operating point | Taşı → yeni 25 |
| 19 Fault/domain | Taşı → yeni 26 |
| 20 Event interpretation | Taşı → yeni 28, **gerçek artefaktlarla** |
| 21 Conclusion | Taşı → yeni 29, **net karar cümlesi eklenerek** |
| 22 References | Taşı → yeni 30 |
| — | **YENİ 16** — gerçek başlangıç düzeltmesi (sayılarla) |
| — | **YENİ 18** — genlik baskınlığı |
| — | **YENİ 22** — rol kartı ve mühürlü test |
| — | **YENİ 23** — genlik kapısı PASS |
| — | **YENİ 24** — dört veri seti tablosu |
| — | **YENİ 27** — any-window yanılsaması |

**Toplam:** 30 ana slayt (v3'ün 22'sine karşılık), 6 appendix. 41'e göre 11 slayt daha az,
ama v3'te düşen 6 kritik bulgunun tamamı geri geldi.
