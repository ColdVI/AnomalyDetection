# ML-16: Öz-Koşullu İnce Dedektörler Planı (Chronos genişletme + TabFM pilotu)

Durum: ÖN-KAYIT (2026-07-07, sonuçlar görülmeden sabitlendi).
Üst plan: `docs/ML14_MASTER_IYILESTIRME_PLANI.md` (Kaldıraç 3).
Bağımlılıklar: yeni-dönem veri (ML-14) + kayma-düzeltmeli kalibrasyon (ML-15).
TabFM kurulum/hız preflight'ı erken koşulabilir; tam değerlendirme ML-15 sonrası.

## §0 Gerekçe

Projede baseline'ı Gate B'de kesin geçen iki fikir de aynı ailedendir:
uçuşu/satırı KÜRESEL normal manifolduna değil kendi bağlamına koşullayan ince
skorlar (ML-10 chronos_motor: uçuşun kendi geçmişi; ML-12 itki_komutu: tek
kanalın kendi dağılımı). Bu faz aileyi iki kolda genişletir; küresel-normal
heterojenliğine (D.1) en az bağımlı skor ailesini tamamlar.

## §1 Kol F — Chronos forecast-residual genişletmesi

ML-10'un DONMUŞ boru hattı (`scripts/build_ml10_forecast_residual.py` kalıbı:
causal yalnız-geçmiş bağlam, q10/q90 bant-dışı normalize residual, aynı model
`amazon/chronos-bolt-tiny` + revizyon pini) yeni kanallara uygulanır.

**Kanal seçim kuralı (SABİT, sonuç görülmeden):** ML-11 AUC matrisinde hedef
kategorisi için separation ≥0.65 olan, yeni-dönem development'ta doluluğu ≥%99
olan HAM zaman-serisi kanalları; aday listesi:

| Kanal | Hedef kategori | ML-11 kanıtı |
|---|---|---|
| `gps_speed_calc_mps` (veya doluluğu daha yüksekse `vel_m_s`) | Velocity / Position.X | gps_speed_residual ailesi 0.89 |
| `hgt_test_ratio` | Position.Z (baro'suz tek yaşayan sinyal) | 5s_max 0.66, q99 ~5× |
| (referans) `actuator_output_std` | Actuator O+C | ML-10'da kanıtlı |

Doluluk kontrolü preflight'ta yapılır ve seçim ORADA dondurulur (ML-10'un `alt`
kanalı kalıbı). Zorunlu fizibilite: 8-uçuş zaman projeksiyonu, <3 saat kuralı.

## §2 Kol T — TabFM cross-feature residual pilotu

Google TabFM 1.0.0 (`google/tabfm-1.0.0-pytorch`, Apache-2.0; bu ortamda
`pip --dry-run` temiz — 2026-07-07 doğrulandı). **Yalnız regresyon-residual
olarak** kullanılır; sınıflandırıcı olarak KULLANILMAZ (ML-8A dersi).

1. **Kurulum pini:** `tabfm==1.0.0` requirements'a eklenmeden önce ve sonra tam
   `pytest` (ML-11 UMAP kuralıyla aynı disiplin).
2. **Kurgu:** `TabFMRegressor`, in-context = split train-normal satırlarından
   deterministik alt-örneklem (**N_ctx = 4096**, seed=split seed); hedef kanal
   y, öngörücüler X = hedefin ailesi DIŞINDAKİ feature'lar. **Aile dışlama
   kuralı (SABİT):** hedefle aynı kök adı taşıyan tüm kolonlar (prefix eşleşme,
   ör. hedef `actuator_output_std` → tüm `actuator_output_*`) X'ten çıkarılır —
   sızıntı/trivyal kopya engeli.
3. **Hedef kanallar (SABİT):** `actuator_output_std` (Actuator O+C — mevcut en
   iyilerle doğrudan kıyas) ve `hgt_test_ratio` (Position.Z).
4. **Skor:** residual = |y − ŷ|; val-normal CDF'ine `empirical_probability` ile
   kalibrasyon (ortak score_fusion yolu).
5. **Nedensellik:** kesitsel tahmin (aynı zaman adımının diğer kanalları +
   donmuş train-normal bağlamı) — gelecek satır değişince geçmiş skor değişmez;
   ML-10 tarzı future-leak testi assert eder. Zero-shot: kaynakta
   `.fit(` yalnız in-context anlamında; gradient/optimizer çağrısı olmadığı
   statik testle kanıtlanır.
6. **Maliyet kontrolü (SABİT):** skorlar yalnız 1 sn karar kovalarının son
   satırları için üretilir (karar kadansı; tüm 5 Hz satırlar DEĞİL). Zorunlu
   preflight: model yükleme + 1.000 gerçek satır × N_ctx=4096 CPU zamanlaması →
   tam development projeksiyonu; **<3 saat kuralı**. Aşarsa Kol T "bu ortamda
   fizibil değil" olarak (MOMENT/F.4 kalıbında) belgelenir, Kol F devam eder.

## §3 Değerlendirme ve Gate'ler (SABİT)

Tüm adaylar ML-9/12 kalıbında bağımsız skor kaynağı olarak, **ML-15'in
kayma-düzeltmeli kalibrasyonuyla** değerlendirilir (3 karar × 2 bütçe × 5 seed;
smoke split_00 önce). Artifact: `artifacts/ml16/uav_sead/<run>/`.

- **Gate A (zorunlu):** holdout kapalı; donmuş dizinler değişmedi; future-leak +
  zero-shot statik testleri (iki kol); karar katmanı/score_fusion identity;
  kanal/aile-dışlama seçimlerinin preflight'ta dondurulduğu kanıtı.
- **Gate B (kategori):** aday vs **yeni-dönem en iyi** aynı-kategori skoru
  (ML-14'te yeniden ölçülen `itki_komutu` Actuator O+C için; Position.Z /
  Velocity için ML-14'teki en iyi mevcut kaynak). Kural değişmez: eşleşen
  policy+bütçede ortalama onset-recall kazancı ≥0.05 VE ≥3/5 seed pozitif.
  Position.Z satırları ayrıca "veri/kapsam sorunu" hipotezinin testidir:
  hgt_test_ratio-tabanlı adaylar da ayrışmazsa H26'nın kapsam teşhisi kesinleşir.
- **Gate C (operasyonel, değişmez):** düzeltilmiş-kalibrasyonlu satırlarda
  critical ≥0.30 @ ≤2 veya advisory ≥0.50 @ ≤12.

## §4 Dosyalar, testler, kabul

| Dosya | İş |
|---|---|
| `scripts/build_ml16_residual_channels.py` (yeni) | Kol F precompute (ML-10 kalıbı: --preflight/--feasibility/--full) |
| `scripts/build_ml16_tabfm_residual.py` (yeni) | Kol T precompute (--preflight zorunlu ilk adım) |
| `scripts/run_ml16_evaluation.py` (yeni) | §3 değerlendirme (ML-15 kalibrasyon sarmalayıcısını import eder) |
| `tests/test_ml16.py` (yeni) | future-leak (iki kol), zero-shot statik, aile-dışlama kuralı birim testi, N_ctx/seed determinizmi, identity, manifest holdout-hash |
| `requirements.txt` | `tabfm==1.0.0` (yalnız preflight geçerse) |

Kabul: preflight raporları (kanal doluluk + zaman projeksiyonu) sonuç
görülmeden dondurulmuş seçimleri içerir; smoke + tam koşu; gate sayıları ham
CSV'den bağımsız türetilebilir; tam pytest yeşil (4 MinIO hariç); H34+ /
ADR-016 / yetersizlikler kaydı; holdout NE OLURSA OLSUN kapalı; commit'ler
co-author'suz.

## §5 Sıralama notu

ML-16 tam koşusu ML-15'e bağımlıdır (kalibrasyon sarmalayıcısı). Erken
başlanabilecekler: TabFM preflight (kurulum + hız mikro-ölçümü) ve iki
precompute script'inin iskeleti + testleri.
