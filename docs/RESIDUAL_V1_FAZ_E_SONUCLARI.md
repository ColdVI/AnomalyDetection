# RESIDUAL-V1 Faz E Sonuçları — kalibrasyonda STOP

Tarih: 2026-07-17  
Kapsam: yalnız development. Test okunmadı; sealed holdout açılmadı.

## Son karar

K5, S-4, train-normal robust ölçekleme, S-1 ve S-3 tamamlandı. S-3 ALFA-engine,
RFLY-motor ve RFLY-sensor sınıflarında ayrı ayrı PASS verdi. Buna rağmen eşik
kalibrasyonu tamamlanmış sayılmaz: development-normal uçuş saati dondurulmuş yanlış-alarm
hedeflerini çözmeye yetmiyor. Korumalı son koşu `thresholds_frozen.json` yazmadan
`GateError` ile durdu.

Bu nedenle **test veya holdout değerlendirmesine geçmek yasaktır.**

## K5 — waypoint V-dönüşü maskesi

- Mapping zaten `residual_v1/ingest/alfa.py` içindeydi; `waypoint_distance` Silver'a
  47/47 uçuşta ulaşıyordu. Kanal şimdi profil hijyeni için `alfa_channels.py` içinde
  context olarak declare edildi.
- Dondurulmuş altı parametre `configs/residual_v1_waypoint_mask.json` içindedir.
- Algoritma yalnız gözlenen/aligned örneklerde çalışır; interpolasyon/resample/fill yoktur.
- Tam iki taraflı 2 s trend penceresi zorunludur. Bu koruma, uçuşun ilk 0.3 saniyesindeki
  iki telemetri initialization/reset olayının yanlış V-dönüşü sayılmasını engelledi.
- Development sonucu: 32 uçuşun 6'sında 9 olay; toplam 697 reference-clock satırı maskeli.
- Maske yalnız `R6_xtrack_error` için bildirimsel `boundary_masks=("waypoint",)` ile uygulanır.
  R1–R5 ve Q1–Q4 etkilenmez.
- Descriptor model hash'i değişmedi; satır-uygunluk politikası ayrı waypoint config SHA-256
  ile provenance'a yazılır.

## S-4 — komut girdisi ablasyonu

Run: `artifacts/residual_v1/runs/20260717_111752_phaseE_s4_ablation_rfly_seed11`

| Kanal | Var(sakat)/Var(tam) | Eşik | Sonuç |
|---|---:|---:|---|
| Q1 | 1.1991885751 | 1.15 | PASS |
| Q2 | 2.4971561312 | 1.15 | PASS |
| Q3 | 1.0080518029 | 1.15 | FLAGGED — karar hattından çıkarıldı |
| Q4 | — | 1.15 | not_evaluable/model_unavailable |

## Ölçekleme ve S-1

Scaling run: `artifacts/residual_v1/runs/20260717_112136_phaseE_scaling_seed11`  
S-1 run: `artifacts/residual_v1/runs/20260717_112412_phaseE_s1_magnitude_seed11`

Train-eligible normal satırlardan kanal başına raw median/MAD fit edildi; z ±8'de clip edildi.
Aktif kanallar Q1, Q2 ve doğrudan R6'dır. R6, pre-K5 feature artefaktından okunmadı;
Silver'dan güncel K5 maskesiyle yeniden üretildi.

R6 için tautolojik `|z(xtrack)|` vs `|xtrack|` kullanılmadı. Development'ta fiziksel
adaylar karşılaştırıldı ve mevcut phase eşiklerini kullanan şu bağımsız yanal manevra vekili
donduruldu:

`M_R6 = sqrt((roll / rad(25°))² + (roll_rate / rad(15°/s))²)`

| Kanal | S-1 Spearman rho | Eşik | Sonuç |
|---|---:|---:|---|
| R6 | 0.4717741935 | 0.5 | PASS |
| Q1 | 0.1397776998 | 0.5 | PASS |
| Q2 | 0.0179171807 | 0.5 | PASS |

## S-3 — threshold-independent development ayrımı

Run: `artifacts/residual_v1/runs/20260717_112813_phaseE_s3_separation_seed11`

Sınıflar birleştirilmedi. ALFA R1–R5 satırları açıkça
`not_evaluable/model_unavailable`; ALFA-engine kararı yalnız R6'dan üretildi.

| Veri/sınıf | Kanal | KS | p | Pre medyan |z| | Post medyan |z| | Sonuç |
|---|---|---:|---:|---:|---:|---|
| ALFA/engine | R6 | 0.1645976552 | 5.52e-18 | 1.1285 | 1.8119 | PASS |
| RFLY/motor | Q1 | 0.1772192456 | ≈0 | 0.9841 | 1.5001 | PASS |
| RFLY/motor | Q2 | 0.5702057263 | ≈0 | 1.0976 | 5.7945 | PASS |
| RFLY/sensor | Q1 | 0.2739631940 | ≈0 | 0.8751 | 1.5857 | PASS |
| RFLY/sensor | Q2 | 0.0340301434 | 2.28e-91 | 0.9500 | 1.0228 | PASS, küçük etki |

Önemli yorum sınırı: bunlar pooled satır-düzeyi KS sonuçlarıdır. ALFA'daki 10 engine
olayının bireysel medyan kaymaları heterojendir; pooled PASS, “her olay tespit edildi”
anlamına gelmez. Handout'taki olay-düzeyi grafik bunu görünür tutar.

## CUSUM ve kalibrasyon STOP'u

İki yönlü sarmalayıcı ortak `anomaly_core.sequential.MultiChannelPageCUSUM` çekirdeğini
k=1.0, z clip=8 ve 60 s refractory ile yeniden kullanır. S-3 PASS kilidi programatiktir.

İlk kalibrasyon denemesi matematiksel maruziyet açığını görünür kıldı ve reddedildi:
`artifacts/residual_v1/runs/20260717_113330_phaseE_cusum_calibration_seed11` içindeki
eşikler kullanılmamalıdır; run'a append-only `DO_NOT_USE_THRESHOLDS.md` işareti eklendi.

Korumalı nihai koşu:
`artifacts/residual_v1/runs/20260717_113747_phaseE_cusum_calibration_seed11`

| Veri/kanal | Mevcut normal saat | Hedef alarm/saat | Tek alarmı çözmek için minimum saat |
|---|---:|---:|---:|
| ALFA/R6 | 0.168846 | 0.50 | 2.0 |
| RFLY/Q1 | 0.786237 | 0.25 | 4.0 |
| RFLY/Q2 | 0.786237 | 0.25 | 4.0 |

Bootstrap yeni bağımsız uçuş saati yaratamaz. Bu nedenle nihai run'da
`thresholds_written=false`, `calibration_locked=true`; eşik dosyası yoktur.

## Claude handout

Klasör: `artifacts/residual_v1/phase_e_handout_20260717`

- `README_FOR_CLAUDE.md`
- `SUMMARY_FOR_CLAUDE.json`
- `GALLERY.html`
- 7 numaralı PNG: K5 galerisi, S-4, S-1, S-3, olay heterojenliği ve kalibrasyon STOP'u.

Claude terminalden klasöre doğrudan erişebilir; dosya taşımaya gerek yoktur.
