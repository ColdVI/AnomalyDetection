# Codex ADS-B Behavioral Anomaly — V1 Pilot Sonuçları
> **GEÇERSİZ IMPLEMENTASYON BASELINE'I:** Aşağıdaki V1 recall değerleri veri veya
> problemin zorluğu hakkında geçerli bir sonuç değildir. Zero-MAD skor patlaması ve
> event değerlendirmesinde alarm `onset`/aktif durum karışıklığı nedeniyle bariz
> anomaliler false-negative yazılmıştır. Düzeltilmiş hard-physics sanity koşusu
> `hard_physics_sanity_v3` kabul kapısını geçmiştir.

Tarih: 2026-07-10
Plan: `docs/CODEX_ADSB_BEHAVIORAL_ANOMALY_ROADMAP.md`
Artifact: `artifacts/adsb_behavioral_stage1/pilot_3archives_v1/`

## 1. Veri ve split

Kullanıcının `Downloads` klasöründeki üç gerçek readsb arşivi kullanıldı:

| Rol | Arşiv tarihi | Uçuş | Satır |
|---|---:|---:|---:|
| train-normal | 2026-02-28 | 250 | 136,843 |
| validation-normal | 2026-03-01 | 250 | 150,522 |
| test-normal | 2026-03-16 | 250 | 149,663 |

Her tar içindeki 150 aircraft trace üyesi dosya adı hash'iyle deterministik seçildi.
Toplam 750 gerçek uçuş segmenti kullanıldı. Testteki 250 uçuşun her biri `easy` ve
`medium` olmak üzere iki ayrı kopyada bozuldu: 500 injected uçuş / 299,326 satır.
Orijinal tarlar salt okunur kaldı ve SHA-256 değerleri manifestte kaydedildi.

Fizik residual smoke'unda kalite-uygun satır oranı `%97.17` idi. Ana feature dolulukları
`%88–100`; roll tabanlı dönüş residual'ı `%42.2` doluydu.

## 2. Modeller

1. **Physics rule:** uçuş fazına göre train-normal median/MAD ile normalize edilmiş
   fizik residual'larının en büyük iki skorunun ortalaması.
2. **Isolation Forest:** aynı dokuz residual üzerinde normal-only eğitim.
3. Her modelin eşiği 2026-03-01 validation-normal verisinde `2-of-3` kararıyla
   `<=0.10 false event/saat` koşulunu karşılayan en düşük eşik olarak seçildi.

Enjeksiyon train veya validation-normal fit'ine girmedi.

## 3. Geçersiz V1 koşusunun tarihsel sonucu

### Gate A — GEÇTİ

- tarih bazlı train/validation/test ayrımı;
- train enjeksiyon satırı sıfır;
- residual'lar causal/prefix-invariant;
- veri-kalitesi ve behavioral skor sözleşmesi ayrı;
- kaynak tar hash'leri manifestte.

### Gate B — KALDI

Ön-kayıtlı hedef: easy recall `>=0.90`, medium recall `>=0.70`, aynı anda
`<=0.10 false event/saat`.

| Model | Easy recall | Medium recall | Test FA/saat |
|---|---:|---:|---:|
| Physics rule | 0.000 | 0.004 | 0.0193 |
| Isolation Forest | 0.092 | 0.028 | 0.0483 |

Isolation Forest yalnız `position_drift` easy (`0.351`) ve `coherent_route_drift` easy
(`0.182`) tiplerinde kısmi sinyal buldu. Altitude/speed/track bias tipleri pratikte sıfırdı.

### Gate C — KALDI

Isolation Forest macro event recall'ı (`0.0550`) physics rule'dan (`0.00169`) `+0.05`
fazlaydı; fakat test FA/saat de daha yüksekti (`0.0483 > 0.0193`). Ön-kayıtlı “daha yüksek
recall ve daha yüksek olmayan FA” şartı birlikte sağlanmadı.

### Gate D — BEKLİYOR

Etiketsiz doğal ADS-B'deki top-100 skorlar `natural_top100_audit.csv` dosyasına yazıldı.
Manuel time-series/harita audit'i tamamlanmadan doğal anomaly precision iddiası yoktur.

## 4. Kök-neden teşhisi

### 4.1 Physics rule MAD çökmesi

Physics rule validation eşiği `46,207,340.46` çıktı. Bazı zero-inflated feature'larda
train median ve MAD sıfıra yakın olduğu için scale `1e-6` tabanına düştü:

- `duplicate_position_moving`: median `0`, scale `1e-6`;
- `abs_baro_geom_delta_rate_mps`: global scale `1e-6`;
- bazı fazlarda vrate/track-rate scale `1e-6`.

Normal verideki tek bir sıfırdan-farklı değer milyonluk skor üretti. Validation FA bütçesi
eşiği aşırı yukarı itti ve kontrollü enjeksiyonların çoğu bastırıldı. Bu bir veri sonucu
değil, V1 robust-scaling tasarım yetersizliğidir. V1 artifact'i audit için değiştirilmez.

### 4.2 Isolation Forest çok-değişkenli seyrelme

Tek kolonu etkileyen speed/track/vertical-rate bias, dokuz feature'lı IF skorunda yeterince
öne çıkmadı. Daha önce UAV-SEAD'de görülen feature-seyrelme örüntüsünün ADS-B karşılığıdır:
tek fizik ilişkisinin güçlü bozulması, diğer normal ilişkilerle ortalanıyor.

### 4.3 Persistence kararı spike'ları kaçırıyor

`2-of-3`, tek-adımlık position/altitude jump'ı gürültü olarak bastırıyor. Donmuş V1 IF skoru
üzerinde karar tanısı:

| Karar | Easy | Medium | Validation FA/saat | Test FA/saat |
|---|---:|---:|---:|---:|
| 1-of-1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2-of-3 | 0.092 | 0.028 | 0.0811 | 0.0483 |
| 3-of-5 | 0.200 | 0.120 | 0.0919 | 0.1158 |

`3-of-5` daha düşük eşik kullanabildiği için recall'ı artırdı, ancak test FA bütçesini
aştı ve yine hedeflerin çok altında kaldı. Sorun yalnız karar katmanı değildir.

### 4.4 Bazı enjeksiyonlar ADS-B'den tek başına sürekli gözlenemez

Sabit altitude bias yalnız olay başlangıcı/bitişinde dikey tutarlılığı bozar; sonrasında
trajectory yeniden pürüzsüz görünür. Haricî referans veya ikinci sensör yoksa sürekli
“gerçek irtifa yanlış” iddiası yapılamaz. Benzer biçimde fiziksel olarak tutarlı biçimde
birlikte değiştirilmiş konum+hız+track, yalnız kısa-adım residual'larıyla zor tespit edilir.

## 5. Düzeltilmiş hard-physics sanity sonucu

Artifact: `artifacts/adsb_behavioral_stage1/hard_physics_sanity_v3/`

Bu koşu aynı donmuş örneklem üzerinde implementasyon sanity kontrolüdür; bağımsız nihai
test değildir. Binary `anomaly=yes/no` durumu event boyunca ölçülmüş, alarm başlangıcı
yalnız false-event/saat hesabında kullanılmıştır.

| Easy anomaly tipi | Event recall |
|---|---:|
| altitude bias | 1.0000 |
| position jump | 1.0000 |
| speed bias | 1.0000 |
| track bias | 1.0000 |
| vertical-rate bias | 0.9688 |
| freeze | 0.9667 |

Tüm easy recall `0.976`, tüm medium recall `0.944`; hard sanity kapısı geçti. Ayrıca
reported vertical-rate sıfırken tek adımda `+10,000 ft` irtifa sıçramasının aynı satırda
`altitude_vertical_rate_mismatch` üretmesi regresyon testiyle sabitlenmiştir.

Doğal veride `25.54 yeni alarm/saat` görülmesi nedeniyle bu katman tek başına production
model değildir. Hard kurallar yüksek-recall integrity katmanıdır; doğal veri-kalitesi
olayları ayrıştırılmalı ve behavioral katman validation-normal üzerinde ayrıca kalibre
edilmelidir.

## 6. Karar

Bu V1 model production adayı değildir. Ancak çalışma boşuna değildir:

- üç gerçek tarla uçtan uca binary pipeline çalıştı;
- gerçek şema ve sampling doğrulandı;
- 750 uçuşluk zaman-genellemeli protokol oluştu;
- hangi anomaly ailelerinin kısa-adım fizik residual'ıyla gözlenebilir olduğu ayrıştı;
- zero-MAD, feature seyrelmesi ve persistence sorunları ölçüldü.

## 7. Sonraki model için ön-kayıt

V2 aynı 2026-03-16 testinde “başarı” aramayacaktır. Yeni, görülmemiş bir tarih olmadan
V2 nihai Gate sonucu raporlanmaz.

Önerilen değişiklikler:

1. Zero-MAD güvenli ampirik upper-tail/quantile skor; binary feature için scale=1.
2. Tek büyük fizik ihlalini koruyan feature-bazlı uzman skorlar; dokuz feature'ı tek IF
   skorunda seyreltmeme.
3. İki kanal:
   - ani/hard constraint için instant data-integrity alarmı;
   - yavaş behavioral sapma için persistent 3-of-5 alarmı.
4. Validation enjeksiyonları yalnız model/policy seçimi için; yeni tarih final test.
5. Altitude bias gibi haricî referans gerektiren senaryoları “gözlenebilirlik sınırlı” olarak
   ayrı raporlama.

V2 için Drive'dan yerelde hiç kullanılmamış bir sonraki arşiv (örneğin 2026-03-31 veya
2026-04-16) final test olarak ayrılmalıdır.
