# ML-13: Kategori-Bazlı Ayrı Alarm Kanalı Mimarisi Planı

Durum: ÖN-KAYIT (2026-07-07, sonuçlar görülmeden yazıldı ve sabitlendi).
Kaynak hipotez: ML-12'nin H30 bulgusu (`docs/ML1_BULGULAR_VE_HATALAR.md`).

## §0 Hipotez ve gerekçe

ML-12 kanıtladı: `itki_komutu` (tek-feature IF) Actuator Outputs+Controls için
bilinen en iyi kategori skoru (CUSUM/advisory onset recall 0.459), AMA normal
uçuşlarda 38.1 FA-saat bırakan bir **kategori uzmanı** — max-füzyona eklenince
füzyon eşiği yukarı itiliyor ve kazanç bütün-event kalibrasyonunda eriyor
(ml12_fusion 0.217/23.74; mevcut füzyonla pratikte aynı).

Hipotez:

> Uzman skoru füzyona KARIŞTIRMAK yerine **kendi bütçesi ve policy'siyle ayrı
> bir alarm kanalı** olarak çalıştırmak (operatöre "mekanik uyarı" kanalı),
> aynı TOPLAM FA bütçesinde birleşik recall'u tek-kanallı max-füzyondan yukarı
> taşır. Çünkü iki kanal iki ayrı çalışma noktasına izin verir; tek füzyon
> eşiği iki farklı skor ölçeğini aynı anda optimize edemez.

Dürüst risk beyanı (sonuç öngörüsü değil, bilinen ölçüm): önceki tüm fazlarda
val-kalibre FA bütçesi test'te aşıldı (val→test FA kayması; heterojen normal
sınıfı D.1). İki kanal bu kaymayı ÇÖZMEZ; plan bunu ölçecek ve birleşik FA
test değeriyle raporlanacak. Gate C yine kalabilir — o zaman kazanım, sınırlı
C2 iddiası (aşağıda) ve kaymanın kanal-bazlı ölçümü olur.

## §1 Mimari (SABİT)

İki kanal, ikisi de mevcut donmuş artifact'lardan:

| Kanal | Skor | Kaynak |
|---|---|---|
| `sistem` | `existing_fusion` | Donmuş ML-9 split modelleri (checksum'lu), ML-12 runner'daki üretimle birebir |
| `mekanik` | `itki_komutu` | ML-12'nin split başına kaydettiği ince modeller (`artifacts/ml12/uav_sead/full_matrix/split_XX/models/itki_komutu.joblib`, checksum'lu) |

- Hiçbir model YENİDEN EĞİTİLMEZ; skorlar ML-12 runner'ının deterministik
  yoluyla yeniden üretilir (donmuş scaler + donmuş modeller + val-normal
  `empirical_probability` kalibrasyonu, ortak `score_fusion.py`).
- Her kanal kendi bütçe payıyla, val-normal akışlarında, DEĞİŞMEYEN
  `decision_layers.py` fonksiyonlarıyla kalibre edilir. Bir satırda iki kanal
  aynı karar tipini kullanır (cusum-cusum, k_of_n-k_of_n, threshold-threshold);
  çapraz kombinasyon yok (çokluk kontrolü).
- **Birleşik alarm semantiği**: karar noktası başına kanal onset'lerinin
  mantıksal VEYA'sı (aynı 1 s kovasında iki kanal birden tetiklerse operatöre
  TEK bildirim sayılır — boolean OR bunu doğal sağlar). Birleşik akış,
  değişmeyen `event_metrics` yoluyla değerlendirilir (onset'ler skor olarak
  verilir, eşik 0.5 — ML-9'dan beri kullanılan kalıp).
- Birleşik recall: event, HERHANGİ bir kanal event aralığında yeni onset
  üretirse yakalanmış sayılır. Birleşik FA: normal anlardaki birleşik
  onset'ler (çift bildirim yok, OR'lanmış).

## §2 Ön-kayıtlı bütçe payları (SABİT)

Toplam bütçeler değişmez (critical 2, advisory 12 FA-saat). Kanal payları
kalibrasyonda kullanılır; üç sabit bölüşüm değerlendirilir:

| Bölüşüm | advisory (sistem+mekanik) | critical (sistem+mekanik) |
|---|---|---|
| `agirlikli_sistem` | 10 + 2 | 1.67 + 0.33 |
| `dengeli` | 8 + 4 | 1.33 + 0.67 |
| `esit` | 6 + 6 | 1.0 + 1.0 |

3 bölüşüm × 3 karar tipi × 2 bütçe sınıfı = 18 birleşik satır/seed. Bölüşüm
listesi sonuç görülüp genişletilmez/daraltılmaz.

## §3 Gate tanımları (SABİT)

- **Gate A (zorunlu):** 131-uçuş blind holdout hiçbir tabloda okunmadı;
  ML-9/ML-10/ML-12 manifest checksum'ları doğrulandı; model yeniden eğitimi
  yok (statik kontrol: runner'da `fit_modular_iforest`/`IsolationForest` fit
  çağrısı yok — test assert eder); karar katmanları ve `event_metrics`
  değişmeden import edildi (identity test); kanal-OR yardımcısı birim testli
  (aynı kovada çift tetik = tek bildirim).
- **Gate B (mimari karşılaştırması):** ML-9/12 ile aynı kural — eşleşen karar
  tipi + bütçe sınıfında **birleşik (tüm-event) onset recall** kazancı
  **≥0.05** VE **≥3/5 seed** pozitif; ek şart: aynı satırın birleşik FA'sı
  baseline'ın FA'sının **1.10 katını aşmayacak** (recall'u FA şişirerek satın
  almayı engeller). Baseline'lar (donmuş CSV'lerden, yeniden hesaplanmaz):
  - B1 (gate'i belirleyen): en iyi tek-kanal füzyon satırı — `existing_fusion`
    (ML-9) ve `ml12_fusion_itki` (ML-12) hangisi o satırda yüksekse.
  - Herhangi bir ön-kayıtlı bölüşüm B1'i geçerse Gate B GEÇTİ; tüm bölüşümler
    raporlanır.
- **Gate C1 (operasyonel, değişmedi):** herhangi bir birleşik satır critical'da
  ≥0.30 recall @ ≤2 FA-saat VEYA advisory'de ≥0.50 @ ≤12 FA-saat (test
  değerleriyle) sağlarsa geçer. Geçmezse holdout AÇILMAZ, mimari production'a
  ALINMAZ.
- **Gate C2 (sınırlı "mekanik monitör" iddiası, bilgilendirici ama ön-kayıtlı):**
  yalnız `mekanik` kanal, kendi payı ne olursa olsun test'te ölçülen KENDİ
  FA'sıyla: Actuator Outputs+Controls onset recall **≥0.30 @ kanal FA ≤2**
  (critical-eşdeğer) veya **≥0.50 @ ≤12** (advisory-eşdeğer) sağlarsa,
  "mekanik-özel uyarı kanalı" sınırlı iddiası development'ta kanıtlanmış
  sayılır. C2, C1'i geçmez/ikame etmez; holdout açtırmaz; yalnız kapsamı
  daraltılmış, dürüst bir ürün iddiasının ön koşuludur.

Sonuç görüldükten sonra bölüşüm listesi, karar grid'i, OR semantiği veya
bütçeler DEĞİŞTİRİLMEZ; değişiklik yeni ön-kayıtlı faz gerektirir.

## §4 Dosyalar, testler, kabul

| Dosya | İş |
|---|---|
| `scripts/run_ml13_channel_evaluation.py` (yeni) | §1-§2 protokolü; `--splits split_00` smoke; skor üretimi ML-12 runner'la ortak yardımcılardan |
| `src/ml/decision/channel_union.py` (yeni, ~20 satır) | onset-OR yardımcısı (saf fonksiyon, birim testli) |
| `tests/test_ml13.py` (yeni) | (a) OR-dedupe birim testi, (b) eğitim-yok statik kontrolü, (c) karar-katmanı/event_metrics identity, (d) manifest holdout-hash + checksum, (e) baseline satırlarının donmuş CSV'lerle eşitliği |
| `artifacts/ml13/uav_sead/<run>/` | metrics/category CSV + gates.json + checksum'lu manifest |

Kabul: smoke (split_00) → tam 5-seed; Gate B/C sayıları ham CSV'den bağımsız
yeniden türetilir; tam `pytest` yeşil (bilinen 4 MinIO hariç); bulgular
`docs/ML1_BULGULAR_VE_HATALAR.md`'ye (H31+), karar ADR-013'e, kayıt C.8/E.1'e.

## §5 Bilinen sınırlar

1. **Val→test FA kayması çözülmüyor, ölçülüyor**: kanal payları val'de
   kalibre edilir; test FA'sı payın üstüne çıkabilir (ML-12'de 12 → 23.7).
   Bu planın Gate C1 değerlendirmesi test değerleriyle yapılır — kayma
   satır satır raporlanır.
2. **Mekanik kanal yalnız bir kategoriyi güçlendirir**: altitude/Position.Z
   zafiyeti (veri/kapsam sorunu, H26) bu fazın kapsamı dışında kalır.
3. Çokluk: 18 satır/bütçe-sınıfı "herhangi biri geçer" kuralıyla
   değerlendirilir (önceki fazlarla tutarlı); satır sayısı önceden sabit
   olduğu için sonradan-seçme yok, ama okuyucu satır sayısını bilerek
   yorumlamalı.
