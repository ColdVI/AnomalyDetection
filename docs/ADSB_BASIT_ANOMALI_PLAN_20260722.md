# ADS-B — İndirgenmiş, Basit Anomali Tespiti Planı

> **Uygulama durumu (2026-07-22):** Madde 7'deki beş adım tamamlandı. Aynı
> dondurulmuş 100-uçuş örneğinde irtifa ve rota raporları üretildi. Mevcut rota
> kuralı düşük-hız bearing artefaktına dönüştüğü için durduruldu; yalnız irtifa
> faz/referans teşhisi yeni bir ön-kayıtla ilerletilecek. Karar belgesi:
> `docs/ADSB_BASIT_ANOMALI_KARSILASTIRMA_20260722.md`.

> Yazıldı: 2026-07-22 (Europe/Istanbul)
> Önkoşul: bu dosyadan önce `gecmis_calismalar/README.md` (dört-dataset dosyalama
> haritası) okunmalı — bu plan, o dört dataset'teki (ALFA/UAV-Attack/UAV-SEAD/
> RFLYMAD) yoğun ama sonuçsuz çabadan sonra **bilinçli olarak küçük tutulan**
> yeni bir başlangıçtır.

## 0. Neden bu kadar küçük tutuyoruz

`docs/final_rapor_ml_fizibilite_2026-07-16.md` ve bugünkü RFLYMAD-Full v2 turu
(bkz. ADR-044) aynı dersi iki bağımsız hatta verdi: **"özelden genele"** yerine
"genelden (operasyonel anomali tespit sistemi) özele" giden her girişim, küçük/
heterojen etiketli veri + saatlik yanlış-alarm bütçesi duvarına çarptı. Bu planın
amacı tam tersi yönde ilerlemek:

- Tek, **biz tanımlayan**, tartışmasız basit bir anomali (irtifa sapması VEYA
  GPS/rota sapması — ikisi de aşağıda ayrı ayrı planlı).
- Model YOK — öğrenmesiz, yorumlanabilir bir eşik/kural.
- Operasyonel "saatte N alarm" iddiası YOK — yalnız "bu kuralla gerçek veride
  kaç uçuşta, ne sıklıkta bir sapma gözlemleniyor" sorusuna dürüst bir cevap.
- Kapsam bilinçli olarak dar: 3 uçuş fazı (kalkış/seyir/iniş), 2 anomali türü,
  tek adımda ikisi birden değil.

## 1. Yeniden kullanılacak mevcut altyapı (sıfırdan yazma)

`adsb/` paketi (ADS-B contextual-physics/CUSUM çalışmasından, NO-GO ile
kapandı ama altyapısı sağlam ve dokunulmadan duruyor):

- **Uçuş segmentasyonu:** `adsb/segmentation.py` — `assign_flight_ids()` zaten
  sürekli ICAO24 trace'ini ayrık uçuşlara bölüyor (1800s boşluk kuralı +
  `flags_new_leg` çapraz-doğrulama, gerçek veride test edildi: 4230 uçuş,
  %60.4 uyuşma). **Doğrudan kullanılır, değiştirilmez.**
- **Fiziksel residual feature'lar:** `adsb/features.py` şunları zaten üretiyor
  (`PRIMARY_FEATURES`/`VECTOR_RESIDUAL_FEATURES`): `alt`, `vertical_rate_ms`,
  `vertical_rate_residual` (bildirilen dikey hız vs irtifa türevi),
  `altitude_source_residual` (barometrik vs geometrik irtifa), `ground_speed_ms`,
  `track_deg`, `speed_residual`, `heading_residual`, `east/north_velocity_residual`.
  **İrtifa sapması VE GPS/rota sapması için gereken ham sinyallerin hepsi zaten
  hesaplanıyor** — yeni feature mühendisliği gerekmiyor.
- **Ölçüm kapsamı gerçekçiliği:** `adsb/reports/measurability_table.md` —
  `alt`/`vertical_rate_ms` satır-kapsamı %89.4, `ground_speed_ms` %98.1,
  `track_deg` %95.2 (gerçek, forward-fill'siz ölçülmüş). Bu planın hedefleri bu
  kapsama oranlarının üstüne çıkamaz; ön-kayıt bunu kabul eder.
- **Eksik olan tek şey — faz bölme:** mevcut kodda kalkış/seyir/iniş ayrımı YOK
  (yalnız uçuş-bütünü segmentasyon var). Bu plan kapsamında **yeni, küçük** bir
  `flight_phase` fonksiyonu yazılacak (madde 2).

## 2. Yeni parça: 3 fazlı basit uçuş bölme

Karmaşık bir faz sınıflandırıcı değil, **irtifa ve dikey hız eşiğine dayalı**
basit, yorumlanabilir bir kural önerilir (ön-kayıt gerektirir — sonuçlara
bakılmadan dondurulmalı):

- **Kalkış:** uçuşun başından, `vertical_rate_ms` sürekli pozitif bir eşiği
  (örn. `> 2.5 m/s`, ölçülmüş gürültü tabanına göre kalibre edilecek) aşmayı
  bırakana kadar (ya da irtifa göreli tavan yüksekliğinin bir yüzdesine
  ulaşana kadar).
- **Seyir:** `vertical_rate_ms` bir bant içinde kaldığı (`|vr| < eşik`) ve
  irtifa görece stabil olduğu orta bölüm.
- **İniş:** uçuşun sonuna doğru sürekli negatif `vertical_rate_ms` ile
  başlayıp uçuş bitimine kadar.

Kenar durumları (hiç seyir fazı olmayan kısa uçuşlar, sürekli tırmanan/alçalan
anormal uçuşlar) ayrı bir `phase = "belirsiz"` etiketiyle işaretlenir, sessizce
"seyir" sayılmaz — bu, ALFA/RFLY'deki proxy-etiket hatalarının (bkz. ADR
kayıtları) aynısını burada tekrarlamamak içindir.

## 3. Bölüm A — İrtifa sapması

**Tanım (ön-kayıt, sonuçlardan önce dondurulacak):** bir uçuşun **seyir**
fazında, `alt` değeri o fazın kendi medyan irtifasından **±X metre** (örn.
150m — gerçek ADS-B irtifa gürültüsüne göre kalibre edilecek, uydurma değil)
sapıp bu sapmanın **Y dakikadan** (örn. 2 dk) uzun sürmesi durumu **irtifa
sapması anomalisi** sayılır. `altitude_source_residual` aynı anda büyükse
(barometrik/geometrik uyumsuzluk) bu ayrıca "veri kalitesi şüphesi" olarak
etiketlenir, anomaliyle karıştırılmaz.

**Kalkış/iniş fazlarında** aynı kural uygulanmaz (o fazlarda irtifa değişimi
zaten beklenen davranıştır) — onun yerine `vertical_rate_residual`'ın (beklenen
tırmanma/alçalma oranından sapma) kendi fazına özgü, ayrı ve daha gevşek bir
eşiği olur (bu ikinci adımda detaylandırılır, madde 5).

## 4. Bölüm B — GPS/rota sapması

**Tanım (ön-kayıt):** bir uçuşun herhangi bir fazında, ardışık konumlardan
hesaplanan rota (`heading_residual`) ile bildirilen `track_deg` arasındaki fark
**±Z derece** (örn. 20°) eşiğini **W ardışık örnekte** (örn. 4 örnek ~ ADS-B
örnekleme hızına göre birkaç saniye) aşarsa **GPS/rota sapması anomalisi**
sayılır. Alternatif/ek sinyal: `east_velocity_residual`/`north_velocity_residual`
vektör toplamının büyüklüğü — konumdan türetilen hız ile bildirilen hız
vektörü arasındaki tutarsızlık, sinsi GPS-spoofing'in klasik imzasıdır (bkz.
final_rapor §2.5, UAV-Attack `gps_speed_residual` deneyiminden ders).

**Faz-bazlı ayrım:** seyirde rota değişimi zaten az beklenir (eşik sıkı
tutulabilir); kalkış/inişte pist hizalaması nedeniyle rota değişimi normaldir
(eşik gevşetilir veya bu fazlarda bu anomali türü hiç uygulanmaz — karar
sonuçlardan önce yazılacak).

## 5. Ön-kayıt disiplini (dört dataset'teki dersin doğrudan uygulanması)

Sonuçlara bakılmadan önce yazılı olarak dondurulacaklar:

1. Eşik değerleri (X metre, Y dakika, Z derece, W örnek) — küçük bir gerçek-veri
   örnekleminde (örn. 50-100 uçuş) irtifa/rota gürültüsünün doğal dağılımına
   bakılarak kalibre edilir, ama **anomali sayısına bakılarak değil**.
2. Başarı ölçütü **operasyonel bir "saatte N alarm" bütçesi DEĞİL** — bu, dört
   dataset'i batıran tuzaktı. Bunun yerine: "kaç uçuşta bu basit kural en az bir
   kez tetikleniyor, tetiklenme süresi/şiddeti ne, kaç tanesi elle bakıldığında
   gerçekten fiziksel olarak anlamlı görünüyor (rastgele bir alt-örneklem elle
   incelenir)." Bu bir keşif/karakterizasyon adımıdır, kapı/gate değildir.
3. Kalkış/seyir/iniş faz sınırları da dahil, hiçbir eşik "daha çok anomali
   bulmak için" sonradan değiştirilmez; değiştirilirse yeni bir ön-kayıt yazılır.
4. Sentetik enjeksiyon **kullanılmaz** (dört dataset'teki proxy-etiket
   sorunlarının kök nedenlerinden biri buydu) — yalnız gerçek veride gözlemsel
   karakterizasyon yapılır. İleride etiketli/doğrulanmış olay verisi bulunursa
   bu ayrıca değerlendirilir.

## 6. Kapsam dışı (bilinçli olarak)

- ALFA/UAV-Attack/UAV-SEAD/RFLYMAD'daki gibi ağır model/robustness/cross-domain
  sözleşmeleri, çok-adaylı sweep'ler, preregistered "operasyonel kapı" dili.
- Makine öğrenmesi modeli (AE, LSTM, TCN vb.) — bu ilk turda YOK.
- Saatlik/günlük alarm bütçesi taahhüdü.
- 3 fazdan fazla uçuş segmentasyonu (taxi, holding pattern vb. — ileride).

## 7. Somut ilk adımlar

1. `adsb/features.py` çıktısından, küçük bir gerçek-veri örnekleminde (örn. tek
   bir Silver parçası) `vertical_rate_ms`/`alt` dağılımına bakıp madde 2'deki
   faz-bölme eşiklerini taslak olarak kalibre et (yeni bir keşif scripti,
   `scripts/` altına — ADS-B kapsamında olduğu için kök `scripts/`'te kalır).
2. Yeni `flight_phase()` fonksiyonunu yaz + birim testle (sentetik, elle
   inşa edilmiş 3-fazlı bir uçuş örneğiyle — gerçek veri sızıntısı riski yok
   çünkü bu saf bir fonksiyon testi).
3. Bölüm A (irtifa) kuralını uygula, küçük gerçek-veri örnekleminde çalıştır,
   madde 5.2'deki keşif raporunu üret.
4. Aynısını Bölüm B (GPS/rota) için tekrarla.
5. İki rapor birlikte kullanıcıya sunulur; hangisinin (varsa) daha ileri
   götürülmeye değer olduğuna kullanıcıyla birlikte karar verilir — bu plan
   ikisini de eşit öncelikte açık bırakıyor.
