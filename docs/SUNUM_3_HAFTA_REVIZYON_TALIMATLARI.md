# 3. Hafta Sunumu — Revizyon Talimatları (Claude web için)

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

## Gerçek veri/görsel kaynakları (repoda bulundu)

Sunumdaki üç teknik slayt aslında **iki ayrı gerçek çalışmaya** karşılık geliyor:

- **Slayt 3 + 4** (kalibrasyon + alternatif yöntemler) → uçak takip (ADS-B) verisiyle
  çalışan kural-bazlı anormallik skorlayıcısı ve onunla karşılaştırılan nöral
  alternatifler (`artifacts/adsb/...`).
- **Slayt 5** (GPS-bütünlük) → PX4/İHA GNSS bütünlük pilotu, 16 Temmuz 2026 tarihli
  nihai NO-GO raporu (`artifacts/uav_gnss_integrity_v1/...`).

### Slayt 3 — gerçek eşik/hassasiyet verisi

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

### Slayt 4 — gerçek PNG'ler VE bir içerik düzeltmesi

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

### Slayt 5 — gerçek sonuç tablosu (16 Temmuz 2026 NO-GO raporu)

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

## Taşma / Bindirme Sorunları (v2'de hâlâ var)

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

## Slayt 6 — küçük bir ek not

Zaman çizelgesi düğümlerinden biri ("Hafta 3 — bu hafta") tek daire içine iki ayrı
durumu sıkıştırıyor ("Kalibrasyon: İlerliyor" + "GPS-bütünlük: Durduruldu"). Bunu tek
daire yerine iki küçük alt-etiket olarak yan yana/alt alta, daire dışında göster —
okunabilirlik için.

---

## Claude web'e verilecek özet talimat

> Slayt 3, 4 ve 5'i, bu dosyada verdiğim gerçek tablo ve gerçek PNG dosyalarını
> kullanarak yeniden tasarla — sayı uydurma veya köşeli parantez placeholder basma,
> hepsi burada mevcut. Slayt 4'ün metnini "basit istatistiksel yöntemler" yerine
> gerçek yöntem adlarıyla (Dense-AE / LSTM-AE / LSTM-forecaster) düzelt. Slayt 3'teki
> mini akış kutularını kaldır, sadece gerçek eşik/alarm eğrisini göster. Slayt 2 ve
> 6'daki başlık/gövde çakışmasını düzelt — her slaydı sabit 16:9 tuval olarak tasarla,
> hiçbir öğe taşmasın veya kesişmesin, bitince tek tek kontrol et. Slayt 6'daki "bu
> hafta" düğümünde iki durumu ayrı etiket olarak göster. ML faz numarası veya iç repo
> jargonu kullanma, yöntemleri ne yaptıkları üzerinden anlat.
