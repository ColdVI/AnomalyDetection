# ADS-B `contextual_physics_v2` — Ön-Kayıt Sözleşmesi

> Tarih: 2026-07-23 (Europe/Istanbul)
> Durum: **SONUÇLAR GÖRÜLMEDEN DONDURULDU**
> Önceki: `contextual_physics_v1`, ADR-042 ile "development rejected" — bkz. §0.
> Bu doküman ADR-042'nin doğrudan devamıdır, yeni bir keşif değil.

## 0. Neden yeniden açılıyor (ADR-042 özeti)

`contextual_physics_v1` (LSTM residual-forecaster + hierarchical conformal
kalibrasyon) genlik-baskınlığı sağlamlık testini GEÇTİ
(`rho_trained_vs_untrained=0.65`, 0.8 flag eşiğinin altında — mimari gerçekten
fiziksel yapıyı öğreniyor). Ama dondurulmuş alarm bütçesiyle gerçek enjekte
olayların 5 kanaldan 4'ünde **>%94'ünü kaçırdı**. Kök neden ölçüldü: bütçe
"100 skorlanabilir uçuş-SAATİNDE V" biriminde tanımlıydı, ama her olay TEK bir
uçuşun ~0.5-1 saatlik penceresinde aranıyordu — saatlik oranı bu kısa pencereye
çarpınca beklenen alarm sayısı yapısal olarak %birkaçın altında kalıyor,
modelin kalitesinden bağımsız olarak. Kasıtlı gevşetilmiş alfa testi bunu
doğruladı (recall %74-92'ye çıktı, ama saatte ~9 sahte alarmla). ADR-042'nin
resmi kararı, yeniden açılışın **sonuç görülmeden yazılan, çok daha geniş bir
bütçe ızgarasıyla** ayrı bir ön-kayıt gerektirdiğiydi — bu doküman odur.

Ayrıca ADR-042, `adsb/cusum.py`'nin `east_north_cusum` kanalının (öğrenmesiz,
ham robust-z uzayında uçuş boyunca sıfırlanmadan biriken) aynı ızgarada
**%49.7 recall** verdiğini kaydetti — LSTM'in "persistence" modunun (30 saniyelik
sabit pencere) aksine, CUSUM kanıtı çok daha uzun süre biriktirebiliyor. Bu, §3'teki
persistence yeniden tasarımının doğrudan gerekçesidir.

## 1. Veri kapsamı — "tüm data" genişletmesi

Mevcut Step-5 manifest'i (`artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json`,
SHA-256 `e2fccd40b2b3059eb4d9585980d8296f31462210fe88284295ed33291b236f5f`)
**değiştirilmiyor** — rol/gün ataması doğrulanmış ve temiz:

| Rol | Kaynak gün | Parça | Uçuş |
|---|---|---:|---:|
| fit | 2026-02-28 | 237 | 149.462 |
| calibration | 2026-02-28 (fit'ten ayrık flight_id, overlap=0) | — | 37.208 |
| development | 2026-03-01 | 216 | 181.828 |
| rehearsal | 2026-03-16 | 185 | 165.053 |
| validation | truth-v2 sentetik corpus (`adsb_v2_20260713_01`) | — | 8.910 |

**Karar:** development/rehearsal/validation rolleri DOKUNULMAZ (zaten belirli bir
günün tamamına atanmış, şimdi karıştırmak sızıntı yaratır). Genişleme sadece
`fit` rolünde, iki koldan:

1. Mevcut Feb-28 fit havuzu: `fit_flight_sample_probability` **0.02 → 1.0**
   (149.462 uçuşun tamamı, önceki ~2.989 yerine — calibration'ın 37.208 uçuşu
   ayrık kaldığı için sızıntı yok, `overlap=0` doğrulandı).
2. 3 yeni, daha önce HİÇ kullanılmamış gün, tamamı fit rolüne, **probability=1.0**
   (sub-sampling yok — tamamen temiz veri, tutumlu olmaya gerek yok):
   - 2024-09-01 (~57M satır tahmini)
   - 2025-02-15 (~59M satır tahmini)
   - 2025-06-15 (~85M satır tahmini)

   Bu 3 gün için ayrı, yeni bir "fit-expansion manifest"i Faz A'nın (tar→Silver
   işleme) gerçek çıktısı üzerinden Faz C'de üretilecek (path/bytes/SHA-256,
   v1'in `_verify_fit_inputs` desenindeki aynı bütünlük kontrolüyle) — dosyalar
   henüz yok, hash şimdiden yazılamaz; **ama kural şimdiden dondu: tüm uçuşlar,
   seçici örnekleme yok, sonuca bakılmadan.**

## 2. Epoch sayısı: 8

v1'in 5'inden ölçülü bir artış. Gerekçe: v1'in `rho_trained_vs_untrained=0.65`
zaten 0.8 flag eşiğinin rahat altındaydı — kapasite/overfitting sorunu yoktu.
Fit verisi artık ~50 kat büyüdüğü (149.462+3-gün vs ~2.989 uçuş) için optimizer
zaten çok daha fazla gradyan adımı görecek; 8 epoch yakınsama payı için yeterli.
**Sabit, sonuç görülmeden dondurulmuş.** Epoch sayısı üzerinde arama/erken-durdurma
YAPILMAYACAK (`selection_prohibitions.hyperparameter_sweep=true`, v1'den aynen).

## 3. Genişletilmiş Pareto bütçe ızgarası

v1'in ızgarası (`configs/adsb_contextual_physics_v1_alarm_budget.json`):
`[0.1, 0.5, 1.0, 2.0, 5.0]` "episodes per 100 scoreable flight-hours" biriminde.
**Birim AYNI kalıyor** (yeniden icat etmiyoruz — v1'in kendi gevşek-alfa
sanity testi zaten bu birimde recall'ün V büyüdükçe gerçekten yükseldiğini
kanıtladı, yani birim çalışıyor, sadece ızgara çok dardı). **Aralık genişliyor:**

```
[0.1, 0.5, 1.0, 2.0, 5.0, 10, 25, 50, 100, 250, 500]
```

Matematiksel gerekçe: V=5'te (~0.75 saatlik uçuş için) beklenen alarm =
5/100 × 0.75 ≈ 0.0375 — yapısal olarak <%4 recall tavanı. V=500'de aynı hesap
500/100 × 0.75 ≈ 3.75 — gerçek bir anomalinin uçuş penceresinde en az bir kez
yakalanma ihtimali artık makul. Bu, önceki NN 74-92%'ye çıkan gevşek-alfa
bulgusuyla tutarlı. Log-aralıklı 11 nokta, düşükten yükseğe **tüm eğriyi**
dürüstçe karakterize eder — hiçbir nokta "iyi çıksın diye" seçilmedi.

## 4. Persistence/accumulation yeniden tasarımı

LSTM'in mevcut "persistence" modu (`temporal_profiles`, `window_s=30.0`) sabit,
kısa bir kayan pencerede K-of-N kontrolü. CUSUM ise ham robust-z alanında UÇUŞ
BOYUNCA (segment/gap/ground-transition sınırları hariç) sıfırlanmadan biriken bir
Page-CUSUM istatistiği — bu yapısal fark, ADR-042'nin gözlemlediği %7→%50 recall
farkının olası nedeni.

**Yeni skor:** LSTM'in `conformal_p_value` çıktısı üzerinde, `adsb/cusum.py`'nin
reset mantığıyla BİREBİR aynı sınırlarda (flight-start, ground-transition,
geçersiz/büyük zaman boşluğu → sıfırla) çalışan bir Page-CUSUM benzeri kümülatif
istatistik: her adımda `state = max(0, state + (-log10(p_value) - k))`, `k`
sabit bir referans kayması. Bu, mevcut dondurulmuş `adsb/cusum.py`'yi
DEĞİŞTİRMEZ — yeni, ayrı bir fonksiyon (`adsb/models/contextual_persistence_v2.py`,
Faz C'de yazılacak), CUSUM'un kod deseninden esinlenerek ama girdi olarak ham
z yerine model p-değerini alır. Referans kayması `k` ve reset kuralları da bu
ön-kayıtla dondurulur (v1'in `reference_shift_z`/`max_gap_s` değerleriyle aynı
büyüklük mertebesinde başlanır, sonuçtan önce sabitlenir).

## 5. Değerlendirme ve sentetik veri disiplini

- Eğitim SADECE `data_role="natural_clean_fit"` ile işaretli doğal veriyle
  (v1'deki tip-seviyesi zorunluluk aynen: `train_forecaster` sentetik satır
  görürse hata fırlatır).
- Recall ölçümü SADECE mevcut truth-v2 corpus'uyla (`adsb_v2_20260713_01`,
  8.910 uçuş, zaten üretilmiş) — yeni sentetik üretim YOK.
- `docs/ADSB_BASIT_ANOMALI_ONKAYIT_20260722.md`'nin (kural turu) disiplini
  burada da geçerli: sentetik veri hiçbir zaman eğitime girmez.

## 6. Değişmezlik taahhüdü

Bu doküman + `configs/adsb_contextual_physics_v2_train.json` +
`configs/adsb_contextual_physics_v2_alarm_budget.json` yazılıp kaydedildikten
SONRA hiçbir eşik/ızgara/epoch/persistence-parametresi "sonucu iyileştirmek
için" değiştirilmeyecek. Değişiklik ancak yeni tarihli, sonuç görülmeden
yazılmış bir ön-kayıtla mümkündür (`docs/decisions.md`'ye yeni bir ADR olarak
kaydedilir).
