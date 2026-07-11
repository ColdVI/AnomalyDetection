# ADSB-1 Planı — Fiziksel-Tutarlılık Detektörü + Sentetik-Enjeksiyon Doğrulaması

> Bu doküman bir coding-agent (Codex / Claude Code) talimatıdır. Repo: `ColdVI/AnomalyDetection`.
> `docs/decisions.md` kuralları geçerlidir. ADSB-0'ın (`docs/ADSB0_INGEST_SEGMENT_PLAN.md`)
> tamamlanmasına (özellikle §6 faz kapısı) bağımlıdır — segmentasyon/residual kodu yazılıp test
> edildi, ama gerçek veri üstünde henüz doğrulanmadı (veri erişim engeli, bkz. o doküman §2).

## 0. Bağlam ve deney sorusu

Gerçek adsb.lol trafiğinde (ticari havacılık) neredeyse hiç gerçek etiketli anomali yok — 300
uçaklık örneklemde hiç UAV/drone kategorisi görülmedi (format referansı §5.1). Bu yüzden
ADSB-1'in ana değerlendirmesi SEAD/RFLY'deki gibi "gerçek etiketli arıza recall'ü" DEĞİL,
**sentetik-enjeksiyon doğrulaması**: temiz bir gerçek uçuşa bilinen bir fiziksel-tutarsızlık
enjekte edilir, dedektörün bozulanı yakalayıp orijinali yakalamaması beklenir. Zemin gerçek-ama-
gürültülü etiket değil, bilinen sentetik gerçek olduğu için bu **tamamen kapatılabilir** bir test
— SEAD/RFLY'nin açık kalan "gerçekten işe yarıyor mu" belirsizliğinden farklı olarak.

**Deney sorusu:** ADSB-0'ın 4 fiziksel-tutarlılık residual'ı üstünde en basit mümkün karar katmanı
(robust eşik), gerçek kanalı fiziksel karşılığından koparan sentetik bozulmaları, sırf-genliğe-
bakan bir dedektörün kaçıracağı durumlarda dahi yakalıyor mu?

## 1. Kapsam / kapsam dışı

**Kapsamda:** ADSB-0 residual'ları üstünde basit istatistiksel eşik/füzyon. Sentetik enjeksiyon
enjektörleri. Enjeksiyon-öncesi/sonrası recall@FA karşılaştırması.

**Kapsam DIŞI:** derin öğrenme (ML-16 Kol L/D/U'nun tam tersi ders — önce en basiti, karmaşıklık
hiçbir SEAD/RFLY denemesinde darboğaz olmadı), gerçek/nadir ADS-B anomalisi arama (bu fazın
sentetik doğrulaması temiz geçtikten SONRAKİ bir madde), `src/ml/` `src/adsb/segmentation.py`
`src/adsb/physics_features.py`'a değişiklik (ADSB-0'da kilitlendi, yalnız import edilir).

## 2. `src/adsb/injection.py` — DURUM: YAZILDI + test edildi

`src/ml/injection.py`'deki `inject_freeze/inject_bias/inject_noise/inject_dropout` kolon-adı-
agnostik (doğrulandı, Explore araştırması 2026-07-10) — doğrudan re-export edilip ADS-B
kolonlarında (`alt`, `ground_speed_ms`, `track_deg`, `vertical_rate_ms`, ...) değişiklik
gerekmeden çalışır. `label` kolonu Silver şemasında zaten var (her zaman `None`), `_mark()`'ın
gerektirdiği kolon mevcut.

**Kritik gözlem:** bu genel fonksiyonların çoğu, "bildirilen kanalı fiziksel karşılığından
koparan" senaryoları zaten karşılıyor — örn. `inject_freeze(df, "vertical_rate_ms", ...)` =
"irtifa gerçekte değişirken bildirilen dikey hız sabit kalıyor". Donmuş değer (örn. 0 m/s) sırf-
büyüklüğe-bakan bir dedektörün NORMAL sayacağı bir değer; fiziksel-tutarlılık residual'i ise
bunu yakalar (`tests/test_adsb_physics_features.py::test_vertical_rate_residual_nonzero_after_
freeze_injection` bunu tam olarak gösteriyor).

YENİ eklenen: `inject_position_ramp()` — `src.ml.injection.inject_gps_ramp`'in genellemesi
(yalnız kuzey-lat yerine keyfi kerteriz, PX4 mikrosaniye yerine adsb'nin saniye-cinsinden
`timestamp_utc`'u). Konum yavaşça kayar, bildirilen hız/track DEĞİŞMEZ — `speed_residual`/
`heading_residual`'in tam yakalaması gereken durum.

`PHYSICS_BREAK_RECIPES` sözlüğü 5 adlandırılmış senaryo tanımlar (`vertical_rate_frozen`,
`ground_speed_biased`, `track_frozen`, `position_ramp_stealthy`, `altitude_dropout`) — §5'teki
doğrulama scripti bunları doğrudan kullanır.

**Test kapsamı** (`tests/test_adsb_injection.py`, 4 test, hepsi geçti): genel enjektörlerin adsb
kolonlarında çalıştığı, `inject_position_ramp`'in doğru yönde/büyüklükte kayma ürettiği ve
hız/track'i değiştirmediği, tüm reçetelerin çağrılabilir olduğu.

## 3. `src/adsb/detector.py` — DURUM: YAZILMADI (bu fazın asıl işi)

Planlanan tasarım (en basitten başla ilkesi):

```
robust_residual_score(residuals: pd.DataFrame, *, cols: list[str]) -> pd.Series
    # her residual kolonu icin MAD-tabanli robust z-skor (egitim/kalibrasyon
    # kumesinden ogrenilen medyan+MAD; ONLY normal-donem/temiz ucuslardan)
    # sonra kolonlar arasi max/OR fuzyon (basit; SEAD'deki max_score_fusion
    # deseniyle ayni ruh, ama src/ml/decision/'dan import DEGIL -- ayri kod alani)
```

Kalibrasyon SADECE temiz (enjeksiyonsuz) gerçek uçuşlardan yapılır — etiket zaten yok, leakage
riski yapısal olarak sıfır. `roll_deg`'e bağımlı `turn_bank_residual`, birincil karar katmanında
AĞIRLIKSIZ/ikincil sinyal olarak kullanılır (kapsama ~%8.5, ADSB-0 §4).

## 4. Kör-holdout ve geliştirme kümesi ayrımı

RFLY/SEAD disiplini burada da geçerli: gerçek trafiğin bir kısmı (birkaç gün) **kör-holdout**
olarak ayrılır ve hiçbir kalibrasyon/eşik/enjeksiyon denemesinde kullanılmaz. Enjeksiyon
doğrulaması yalnız geliştirme kümesinde çalışır. Sonuç görüldükten sonra eşik/parametre
değiştirilmez (post-hoc değişiklik yasağı, proje geneli kural).

## 5. Ön-kayıtlı doğrulama protokolü (bu fazın Gate B/C'si) — DURUM: YAZILMADI

`scripts/run_adsb1_synthetic_validation.py` (planlanan):

1. Geliştirme kümesindeki her temiz uçuşa `PHYSICS_BREAK_RECIPES`'teki 5 senaryodan her biri
   ayrı ayrı enjekte edilir (orijinal ASLA değiştirilmez, kopya üstünde çalışılır).
   `onset_frac` taranır (örn. 0.3/0.5/0.7) — literatürdeki ablation şiddet taraması geleneği
   (`docs/ML_YETERSIZLIKLER_KAYDI.md`'de daha önce planlanmış ama SEAD'de hiç uygulanmamış madde
   burada ilk kez gerçek karşılığını buluyor).
2. Her (senaryo, onset_frac) çifti için: dedektör bozulmuş uçuşta alarm veriyor mu (recall), aynı
   dedektör TEMİZ orijinalde alarm vermiyor mu (false alarm)?
3. **Ön-kayıtlı hedef (bu doküman yazılırken sabitlenir, sonuç görüldükten sonra DEĞİŞMEZ):**
   her senaryo için recall ≥ 0.70 (5 senaryonun en az 4'ünde) VE temiz uçuşlarda false-alarm
   oranı ≤ 0.05 — SEAD/RFLY'nin gevşek "advisory" bütçesinden kasıtlı olarak daha sıkı, çünkü
   zemin gürültüsüz sentetik gerçek (SEAD/RFLY'deki etiket belirsizliği burada yok).
4. Sonuç ne olursa olsun (geçse de kalsa da) `docs/decisions.md`'ye ADR olarak, dürüstçe
   yazılır — tıpkı ADR-016..020'de olduğu gibi.

## 6. Bu turda YAPILMAYACAKLAR

- Gerçek-dünya (etiketsiz) ADS-B trafiğinde nadir gerçek anomali arama — §5 temiz geçtikten
  sonraki bir sonraki-faz maddesi, bu planın kapsamı dışında.
- Derin öğrenme / karmaşık model — yalnız §5 sonucu basit eşiğin yetersiz olduğunu gösterirse
  gündeme gelir, ve o zaman bile önce ADR'ye "neden basit yetmedi" dürüstlük kaydı girilir.
- RflyMAD simülasyon indirmesi, SEAD B.5 düzeltmesi — öncelik sırası değişmiyor.

## 7. Bağımlılıklar / engeller

- ADSB-0 §2'deki veri erişim engeli çözülmeden §5'in gerçek veri üstünde çalıştırılması mümkün
  değil. §3 (detector.py) ve enjeksiyon reçeteleri sentetik veriyle geliştirilip test edilebilir.

## 8. Test listesi (pytest)

- `tests/test_adsb_injection.py` — 4/4 geçti (tamamlandı, §2).
- `tests/test_adsb_detector.py` — YAZILMADI (§3 ile birlikte).
- `tests/test_adsb1_synthetic_validation.py` — YAZILMADI (§5 ile birlikte, muhtemelen sentetik
  uçuş + bilinen enjeksiyon üstünde uçtan uca recall/FA doğrulaması).
