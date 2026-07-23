# ADS-B Basit Anomali — Ön-Kayıt Sözleşmesi

> Tarih: 2026-07-22 (Europe/Istanbul)
> Durum: **SONUÇLAR GÖRÜLMEDEN DONDURULDU**
> Kapsam: tek Silver parçası, öğrenmesiz iki keşif kuralı
> Operasyonel başarı/gate iddiası: yok

## 1. Kalibrasyon verisi ve ayrımı

Yalnız eşik kalibrasyonu için lexicographic ilk açık Silver parçası kullanıldı:

- `data/objectstore/silver/adsblol_historical/part-20260710T125240580717Z-fcdf1dca.parquet`
- SHA-256: `2769ba79cb2611ed2b7d39ecae8c7fe1db6975b771e4195d530dbf2a071aa5e1`
- Kaynak tar: `v2026.02.28-planes-readsb-prod-0.tar`
- Footer: 365.847 satır
- Stable-hash seed: `20260722`
- Örnek: 100 uçuş / 53.714 satır; eligible havuz 640 uçuş
- Seçili flight-ID listesi SHA-256:
  `7b7fd5ae2f2a2aebcfce24c41d2ec5a31c050dad5ced02a31bd8fab20321e280`

Seçim önkoşulları: en az 20 satır, en az 300 s süre, `alt` ve
`vertical_rate_ms` için ayrı ayrı en az %60 kapsama. Kalibrasyon aşamasında
yalnız `vertical_rate_ms`, `alt`, uçuş süresi ve örnekleme aralığı dağılımları
okundu; anomaly trigger sayısı hesaplanmadı.

Doğal örnekleme aralığı p50 `4,28 s`, p95 `19,88 s`, p99 `26,79 s` bulundu.
Taslak faz eşiklerinin satır kapsaması: `|vr|<1,0` için `%34,61`, `vr>2,5`
için `%21,15`, `vr<-2,5` için `%28,64`. Bu değerler yalnız eşik
kalibrasyonudur, başarı/anomaly sonucu değildir.

## 2. Dondurulmuş faz kuralı

Etiketler: `takeoff`, `cruise`, `landing`, `uncertain`.

- Uçuşlar `adsb.segmentation.segment_flights(..., gap_s=1800)` ile ayrılır.
- Tüm faz kanıtları en fazla `30 s` iç boşlukla ardışık sayılır.
- Süreklilik: en az `4` ardışık örnek.
- Takeoff kanıtı: `vertical_rate_ms > 2,5 m/s`.
- Cruise çekirdeği: `|vertical_rate_ms| < 1,0 m/s` ve çekirdek medyan
  irtifası uçuşun robust `[q05, q95]` irtifa aralığının en az `%60` seviyesinde.
- Landing kanıtı: `vertical_rate_ms < -2,5 m/s`.
- Robust irtifa aralığı `<300 m` ise üç-faz ayrımı yapılmaz.
- Sıralı bir climb run, ardından yüksek/level cruise çekirdeği ve daha sonra
  descent run bulunması zorunludur. Bu üç kanıttan biri yoksa uçuşun tamamı
  `uncertain` olur; eksik uçuş sessizce cruise sayılmaz.
- Geçerli uçuşta cruise başlangıcı level çekirdeğin ilk satırıdır; landing
  başlangıcı cruise çekirdeğinden sonraki ilk sürdürülebilir descent run'dır.
  Öncesi `takeoff`, arası `cruise`, sonrası `landing` etiketlenir.

Bu kural offline keşif segmentasyonudur; causal/operasyonel faz iddiası değildir.

## 3. Dondurulmuş irtifa sapması kuralı

- Yalnız `phase == cruise` satırları değerlendirilir.
- Referans, aynı uçuşun cruise fazındaki medyan `alt` değeridir.
- Sapma: `abs(alt - cruise_median_alt) >= 150 m`.
- Süre: aynı yöndeki ardışık sapma `>120 s` sürmelidir.
- Ardışıklık, iç satır boşluğu `<=30 s` iken korunur.
- Aynı run içinde sapma yönü değişirse yeni run başlar.
- `altitude_source_residual` aynı olay içinde `abs >=5,0 m/s` ise olay
  `data_quality_suspect=true` olarak ayrıca işaretlenir. Bu bayrak anomaly
  kararını değiştirmez.

## 4. Dondurulmuş GPS/rota sapması kuralı

- Bütün fazlar değerlendirilir; faz yalnız bağlam olarak raporlanır.
- Sinyal: `adsb.features.heading_residual`.
- Sapma: `abs(heading_residual) >=20 derece`.
- Süreklilik: en az `4` ardışık örnek.
- Ardışıklık, iç satır boşluğu `<=30 s` iken korunur.
- `ground_speed_ms <30 m/s` olan triggerlar silinmez; düşük-hız/bearing
  kararsızlığı bağlamı olarak ayrıca raporlanır.
- East/north velocity residual büyüklüğü olay şiddeti için yardımcı kolon olarak
  raporlanır; trigger kararına girmez.

## 5. Keşif raporu sözleşmesi

Her kural ayrı raporlanır:

- değerlendirilebilir uçuş/satır kapsamı;
- en az bir olayı olan uçuş sayısı ve oranı;
- olay sayısı, süre ve şiddet dağılımı;
- faz dağılımı;
- veri-kalitesi/düşük-hız bağlamı;
- stable-hash ile seçilen en fazla 10 olaylık nitel inceleme tablosu/görseli.

Bu bir gate değildir. “Saatte N alarm”, doğruluk, recall veya operasyonel başarı
iddiası kurulmaz; ground truth yoktur. Sıfır, çok az veya çok fazla trigger
çıkması halinde eşikler bu turda değiştirilmez. Değişiklik ancak yeni tarihli
bir ön-kayıtla yapılabilir.

Sentetik veri gerçek keşif sonucuna girmez. Yalnız `flight_phase()` ve kural
mantığının saf birim testlerinde elle inşa edilmiş sentetik tablolar kullanılabilir.
