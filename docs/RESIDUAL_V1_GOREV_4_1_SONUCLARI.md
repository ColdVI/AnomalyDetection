# RESIDUAL-V1 Görev 4.1 Sonuçları

Tarih: 2026-07-17  
Descriptor hash:
86cd49485995b4779934c6b02cf85a26bf4cf303d59c987600ed00a1080c80ca

## Sonuç

Görev 4.1 development-only ve oturum-bazlı CV sözleşmesiyle tamamlandı.
Test telemetrisi model seçimine girmedi; holdout telemetrisi açılmadı.

ALFA'da R1–R5 için beş feature taşıyan development uçuşunun tamamı aynı
2018-07-18 oturumunda. En az iki oturum gerektiren CV kapısı nedeniyle ALFA G1
modeli eğitilmedi. Uçuşları fold'lara bölerek aynı oturumun iki tarafa
sızdırılması reddedildi. R6, K6 gereği öğrenmesiz/doğrudan kanal olarak G1
dışında tutuldu.

RFLY'de Q1–Q3, 12 development oturumu üzerinde gerçek 5-fold session CV ile
eğitildi. Q4'ün train-eligible satırı olmadığı için model kurulmadı.

| Veri | Kanal | Train satır | Uçuş | Oturum | Alpha | CV R² | Train R² | Karar |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ALFA | R1 | 1.361 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R2 | 1.361 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R3 | 1.361 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R4 | 1.299 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R5 | 1.302 | 5 | 1 | — | — | — | yetersiz oturum |
| ALFA | R6 | 16.793 | 32 | 3 | — | — | — | K6: doğrudan kanal |
| RFLY | Q1 | 821.977 | 238 | 12 | 0,1 | 0,0114 | 0,3638 | eğitildi; zayıf genelleme |
| RFLY | Q2 | 821.980 | 238 | 12 | 100 | 0,4564 | 0,7109 | eğitildi; en güçlü G1 |
| RFLY | Q3 | 822.005 | 238 | 12 | 100 | 0,0003 | 0,0085 | eğitildi; pratikte sinyalsiz |
| RFLY | Q4 | 0 | 0 | 0 | — | — | — | train coverage yok |

Q1 ve özellikle Q3'ün train–CV farkı/çok düşük CV R² değeri, bu kanalların
Faz E'ye otomatik kabulü anlamına gelmez. Sonraki model adımı S-1/S-3/S-4
kapıları ve development hata analizi olmalıdır. S-3 geçmeden eşik
kalibrasyonuna gidilemez.

## Kapsam beyanı

**ALFA headline iddiası tek test oturumuna dayanır ve holdout'ta R1–R5 kapsamı beklenmez.**

ALFA için model kurulamaması başarısızlığı gizlemek üzere test oturumunun
training'e alınmasına izin vermez. Tasarım değişikliği istenirse bu ayrı insan
kararı ve yeni şema/ön-kayıt gerektirir.

## Artefaktlar

- ALFA G1 run:
  artifacts/residual_v1/runs/20260717_081305_phaseD_g1_ridge_alfa_seed11
- RFLY G1 run:
  artifacts/residual_v1/runs/20260717_090138_phaseD_g1_ridge_rfly_seed11
- Yeni ALFA Silver:
  artifacts/residual_v1/silver/alfa_preonset_trim_v2
- K1/K4 feature kökü:
  artifacts/residual_v1/features_k1k4
- Tüm uçuş görsel/rapor handout'u:
  artifacts/residual_v1/runs/20260717_065725_full_flight_handout/handout

Her eğitilmiş kanalın joblib modeli, residual parquet'i, fold oturumları,
alpha adayları, katsayıları ve coverage raporu kendi run klasöründedir.
