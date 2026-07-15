# contextual_physics_v1 — alarm bütçesi / kanal payı / temporal profil ön-kaydı

Tarih: 2026-07-14
Durum: **sonuç görülmeden dondurulmuş** — kalibrasyon/development/rehearsal koşusu başlamadan önce yazıldı
Kapsam: `docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md`'nin "Sayısal freeze öncesi açık
kullanıcı kararları" bölümünün ilk üç maddesini kapatır. Kullanıcı üç soruyu da AskUserQuestion ile
onayladı: (1) Pareto ızgarası, (2) kanıta ağırlıklı kanal payı, (3) Claude'un önerdiği temporal
eşikler.

## 1. Toplam operasyonel alarm bütçesi — Pareto ızgarası

Tek sayı yerine, 100 scoreable uçuş-saatte kabul edilen maksimum operator-facing episode sayısı için
5 noktalık dondurulmuş bir ızgara:

```text
budget_grid_per_100h = [0.1, 0.5, 1.0, 2.0, 5.0]
```

Her nokta için development/rehearsal turunda AYRI burden/recall/coverage raporlanacak. Nihai
operasyonel seçim kullanıcı tarafından bu ızgaradan yapılacak — hangi noktanın "en iyi" göründüğüne
bakılarak ızgaranın kendisi genişletilmeyecek/daraltılmayacak.

## 2. Kanal/S2 bütçe payı — kanıta ağırlıklı türetim

Girdi kanıtı: ADR-028'in corrected truth-v2 pooled reçete AUROC'ları (donmuş kural, değiştirilmemiş
kalibrasyon, `artifacts/adsb/runs/20260713_step3_corrected_rule_v1/`):

| Reçete → kanal | AUROC | Skill = AUROC − 0.5 |
|---|---:|---:|
| ground-speed → `speed_residual` | 0.974023 | 0.474023 |
| track → `heading_residual` | 0.889552 | 0.389552 |
| vertical-rate → `vertical_rate_residual` | 0.696337 | 0.196337 |

**`east_velocity_residual` / `north_velocity_residual` için özel karar:** eski `ramp` reçetesinin
skaler-residual AUROC'u (0.558927) bu iki kanala DOĞRUDAN taşınmadı — çünkü bu sayı, ADR-029'da
tam olarak bu zayıflığı gidermek için terk edilen ESKİ (skaler) temsili ölçüyor, yeni iki-eksenli
temsili değil. Onun yerine, kanıtı henüz ölçülmemiş bu iki kanala en zayıf KANITLANMIŞ kanalın
(`vertical_rate_residual`, skill 0.196337) payı taban olarak verildi — ne eski-zayıf sayıyla
cezalandırılıyor ne de ölçülmemiş bir iyimserlikle şişiriliyor.

Beş fizik kanalı arasında normalize edilmiş pay (toplam skill = 1.452586):

| Kanal | Fizik-havuzu payı |
|---|---:|
| `speed_residual` | %32.6 |
| `heading_residual` | %26.8 |
| `vertical_rate_residual` | %13.5 |
| `east_velocity_residual` | %13.5 |
| `north_velocity_residual` | %13.5 |

**S2 veri-kalitesi katmanı için ayrı muamele:** S2 (squawk/emergency/NIC/NACp/SIL/message-gap
reason-code'ları) bir öğrenilmiş fizik-residual DEĞİL, deterministik bir bütünlük bayrağıdır — AUROC
kavramı ona uygulanmaz. Bu yüzden "kanıta ağırlıklı" ilkesi burada AUROC yerine S2'nin zaten ölçülmüş
doğal-yük kanıtına (ADR-031: MESSAGE_GAP ~2.9 episode/saat, NIC-unknown ~0.5 episode/saat) dayanarak
yorumlandı: S2 kendi başına büyük, öngörülemez bir yük taşıyabildiği için toplam bütçeden SABİT ve
mütevazı bir pay (**%15**) ayrılır, kalan **%85** yukarıdaki fizik-kanal oranlarıyla bölünür.

**Nihai toplam bütçe payları:**

| Katman | Toplam bütçe payı |
|---|---:|
| S2 (veri-kalitesi, ayrı) | %15.0 |
| `speed_residual` | %27.7 |
| `heading_residual` | %22.8 |
| `vertical_rate_residual` | %11.5 |
| `east_velocity_residual` | %11.5 |
| `north_velocity_residual` | %11.5 |

Bu paylar her Pareto ızgara noktasına aynı oranda uygulanır (örn. 1.0 episode/100 saatte
`speed_residual`'a düşen pay 0.277 episode/100 saattir).

## 3. Temporal karar profilleri (instant / persistence / accumulation)

Mevcut CUSUM sözleşmesinden (ADR-029, `adsb/cusum.py`) ve corrected truth-v2'nin gözlenen doğal
gecikmelerinden (ground-speed medyan 19.31s, track medyan 56.75s) türetildi. Sayılar sonuç
görülmeden dondurulmuştur; anomaly-development rolü yalnız MOD ATAMASINI (hangi kanal hangi profili
kullanır) sınayabilir, bu sayıları geriye dönük değiştiremez.

| Kanal | Alt-mod | Profil | Tanım |
|---|---|---|---|
| `speed_residual` | spike | instant | Tek skorlanan satırda `p < alpha` → alarm |
| `speed_residual` | bias | persistence | 30s gerçek-zaman pencerede medyan `p < alpha` |
| `vertical_rate_residual` | spike | instant | Tek satır |
| `vertical_rate_residual` | freeze | persistence | 30s pencere |
| `heading_residual` | — | persistence | 30s pencere (track-frozen doğası gereği anlık değil) |
| `east_velocity_residual` + `north_velocity_residual` | — | accumulation | Mevcut donmuş 2-eksenli causal Page CUSUM (ADR-029): `target_vector_shift_mps=2.0`, `k` train-MAD'den türetilir, `h` BUGÜN seçilmez — her Pareto bütçe noktasında doğal kalibrasyondan türetilir |
| S2 reason-code'ları | — | ayrı deterministik episode mantığı | Fizik residual skoruna karışmaz (mevcut S2 modülü) |

`speed_residual` ve `vertical_rate_residual`'ın instant/persistence alt-modları arasındaki bütçe
payı, sonuç görülmeden **%50/%50** olarak varsayılan alınmıştır — bu varsayılan yalnız
anomaly-development rolünde, yeni bir ön-kayıtlı versiyon açılarak değiştirilebilir.

Persistence penceresi (**30 saniye**) seçimi: track kanalının zaten gözlenen ~57s doğal gecikmesini
kötüleştirmeyecek, ama tek-satır gürültüsünü reddedecek kadar uzun bir orta nokta.

## Açık madde

`h` (CUSUM eşiği) bu belgeyle seçilmedi — her Pareto bütçe noktası için doğal
calibration/development turunda ayrı türetilecek (ADR-029'un kendi kuralı). Bu belge yalnız
BÜTÇE PAYLARINI ve zaman-profili YAPISINI dondurur, gerçek eşik sayılarını değil.
