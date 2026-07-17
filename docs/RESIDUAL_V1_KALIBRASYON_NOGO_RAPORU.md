# RESIDUAL-V1 — Nihai Kalibrasyon NO-GO Raporu

Tarih: 2026-07-17  
Karar: **NO-GO — mevcut development-normal maruziyetle elde edilemez**  
Kapsam: ALFA/engine (`R6_xtrack_error`) ve RFLY/motor-sensor (`Q1`, `Q2`)

## Yönetici özeti

RESIDUAL-V1'in eşik kalibrasyonu mevcut veri ve enstrümantasyonla tamamlanamaz. Bu karar
genel bir “dedektör çalışmadı” sonucu değildir. Tam tersine, eşikten önce sınanan yöntem
zinciri çalışmıştır: K5 waypoint maskesi uygulanmış, S-4 komut ablasyonu karar hattındaki
Q1/Q2'yi doğrulamış, robust ölçekleme ve tautoloji-düzeltilmiş S-1 üç aktif kanalda geçmiş,
S-3 ise **ALFA/engine, RFLY/motor ve RFLY/sensor sınıflarının üçünde de threshold-bağımsız
ayrışma göstermiştir**.

**Bu NO-GO'nun nedeni sinyal yokluğu değil, dondurulmuş yanlış-alarm bütçesini güvenilir
biçimde çözmek için yeterli bağımsız normal uçuş-saati bulunmamasıdır.** Bu ayrım sonucu
“başarısız dedektör” diye özetlemeyi bilimsel olarak yanlış kılar: yöntem sinyali S-3 ile
kanıtlanmış, operasyonel eşik ise maruziyet çözünürlüğü kapısında fail-closed durmuştur.

GNSS-bütünlük pilotuyla karşılaştırmada RESIDUAL-V1'in güçlü farkı budur: burada üç başlık
sınıfı için önceden tanımlı, threshold-bağımsız bir sinyal kapısı açıkça PASS vermiştir.
GNSS raporu bazı telemetri ayrışmaları veya gevşek eşikte tepki bulunduğunu not etmiş olsa
da, kendi kayıtlı uçtan uca kabul kapılarından geçen operasyonel bir yöntem kuramamıştı.
Dolayısıyla bu rapor GNSS'te “literatürde hiçbir sinyal yoktu” gibi daha geniş bir iddia
kurmaz; RESIDUAL-V1 için daha güçlü ve dar kanıtı öne çıkarır: **sinyal var, kalibrasyon
maruziyeti yok.**

Sonuç olarak `thresholds_frozen.json` üretilmemiştir; Faz F test/holdout değerlendirmesine
geçilemez.

## Bu turda çalışan ve doğrulanan parçalar

### K5 — waypoint V-dönüşü maskesi

`waypoint_distance` sinyalinin Silver'a ulaştığı doğrulandı ve altı parametreli V-dönüşü
sözleşmesi development verisinde donduruldu. K5 yalnız R6'ya uygulanır; R1–R5 ve Q1–Q4'ü
etkilemez. Development'ta 32 uçuşun 6'sında 9 olay bulundu ve 697 reference-clock satırı
maskelendi. Başlangıç resetlerinin olay sayılmaması için iki taraflı trend penceresi zorunlu
tutuldu. K5 için ayrı bir run dizini açılmadı; güncel maskeyle R6 yeniden üretiminin provenance'ı
scaling run'ı ve Faz E handout'undadır.

### S-4 — komut ablasyonu

| Kanal | `var(r_sakat)/var(r_tam)` | Karar |
|---|---:|---|
| Q1 | 1.1991885751 | PASS |
| Q2 | 2.4971561312 | PASS |
| Q3 | 1.0080518029 | FLAGGED — karar hattından çıkarıldı |
| Q4 | — | not_evaluable/model_unavailable |

Q1 ve Q2'nin komut bilgisini gerçekten kullandığı doğrulandı. Q3'ün elenmesi gizlenmiş bir
başarı değil, S-4 kapısının amaçlandığı gibi çalıştığının kanıtıdır.

### Robust ölçekleme ve S-1

Aktif kanallar Q1, Q2 ve R6'dır. Train-eligible normal satırlardan median/MAD fit edilmiş,
skorlar ±8'de kırpılmıştır. R6 için `xtrack_error`ı kendisiyle karşılaştıran tautolojik bir
test kullanılmamış; bağımsız yanal manevra vekili dondurulmuştur:

`M_R6 = sqrt((roll / rad(25°))² + (roll_rate / rad(15°/s))²)`

| Kanal | S-1 Spearman ρ | FLAG eşiği | Karar |
|---|---:|---:|---|
| R6 | 0.4717741935 | 0.5 | PASS |
| Q1 | 0.1397776998 | 0.5 | PASS |
| Q2 | 0.0179171807 | 0.5 | PASS |

### S-3 — threshold-bağımsız sinyal kanıtı

Sınıflar birbirine karıştırılmamış ve veri setleri arasında sonuç taşınmamıştır. ALFA R1–R5
`not_evaluable/model_unavailable` olarak kalmış; ALFA/engine kararı yalnız R6'dan verilmiştir.

| Veri/sınıf | Kanal | KS | p | Pre medyan |z| | Post medyan |z| | Karar |
|---|---|---:|---:|---:|---:|---|
| ALFA/engine | R6 | 0.1645976552 | 5.52e-18 | 1.1285 | 1.8119 | PASS |
| RFLY/motor | Q1 | 0.1772192456 | ≈0 | 0.9841 | 1.5001 | PASS |
| RFLY/motor | Q2 | 0.5702057263 | ≈0 | 1.0976 | 5.7945 | PASS |
| RFLY/sensor | Q1 | 0.2739631940 | ≈0 | 0.8751 | 1.5857 | PASS |
| RFLY/sensor | Q2 | 0.0340301434 | 2.28e-91 | 0.9500 | 1.0228 | PASS, küçük etki |

Bu sonuçlar pooled satır-düzeyi dağılım ayrışmasıdır; “her olay yakalandı” veya operasyonel
recall kanıtı değildir. Bununla birlikte üç headline sınıfta eşikten önce ölçülebilir sinyal
bulunduğunu gösterir ve kalibrasyon kapısına geçiş için tanımlanmış S-3 koşulunu karşılar.

## Kalibrasyon maruziyet açığı

Korumalı kalibrasyon koşusu, hedef alarm oranında tek bir alarmı dahi çözebilmek için gereken
asgari süreyi development-normal maruziyetle karşılaştırmış ve eşik yazmadan durmuştur.

| Veri/kanal | Mevcut normal maruziyet (saat) | Asgari hedef (saat) | Gereken çarpan | Sonuç |
|---|---:|---:|---:|---|
| ALFA/R6 | 0.168846 | 2.0 | 11.845× | Yetersiz |
| RFLY/Q1,Q2 | 0.786237 | 4.0 | 5.088× | Yetersiz |

Bootstrap mevcut blokları yeniden örnekleyebilir; yeni bağımsız uçuş-saati yaratamaz. İlk
keşif koşusunda çıkan sıfır-alarm eşikleri bu nedenle güvenilir kalibrasyon sayılmamış ve
`DO_NOT_USE_THRESHOLDS.md` ile açıkça reddedilmiştir. Son korumalı koşunun durumu
`stopped_insufficient_calibration_exposure`, `thresholds_written=false`'dur.

## Veri tavanı testi

| Veri | Doğrulanan veri tavanı | Mevcut kullanım/split | En iyimser ekleme | En iyimser sonuç | Hedefe kalan açık |
|---|---|---|---:|---:|---:|
| ALFA | Sabit 47 uçuşluk corpus; toplam 11 normal | development 9, holdout 1, test 1 | 2 normal uçuş | 9→11 uçuş; yalnız +%22 | 11.845× gereksinimi kapanmaz |
| RFLY | Resmî kaynakta 84 `Real-No_Fault` | projede 51; development 41, holdout 10 | en çok 33 aday | ≈1.419 saat | ≈2.581 saat / ≈135 uçuş |

ALFA'da test ve holdout'taki iki normal uçuşun development'a taşınması hem rol izolasyonunu
bozardı hem de sayısal açığı kapatmazdı. Bu nedenle redistribution denenmemiştir.
ALFA'nın 47 uçuşluk corpus tavanı Keipour, Mousaei ve Scherer'in yayımlanmış
[ALFA çalışması](https://arxiv.org/abs/1907.06268) ve yerel corpus sayımıyla doğrulanmıştır.

RFLY'nin resmî veri sayfası 84 Real-No_Fault uçuş bildirmektedir. Projedeki 51 uçuşa göre
kalan en çok 33 adayın tümünün aynı ölçüde kullanılabilir olduğunu varsayan iyimser üst sınır:

`0.786237 + 33 × (0.786237 / 41) = 1.419062 saat`

Bu dahi 4.0 saat hedefinin 2.580938 saat altındadır. Development ortalaması
0.0191765 saat/uçuşla açık yaklaşık 135 ilave uçuş daha gerektirir. Resmî kaynaktaki 33 aday,
kalibrasyon için gereken toplam ek miktarın çok altındadır; bu nedenle indirme ve ingest de
başlatılmamıştır. Böylece hem redistribution hem mevcut resmî kaynaktan ek ingest yolu,
test/holdout açılmadan ve veri indirilmeden matematiksel olarak elenmiştir.

RFLY kaynak sayımı: [RflyMAD resmî dataset sayfası](https://rfly-openha.github.io/documents/4_resources/dataset.html)
(`Real Flight / No Fault = 84`). Alt-durum dağılımı resmî sayfada verilmediği için 33 sayısı
“kesin kullanılabilir uçuş” değil, **en iyimser aday tavanıdır**.

## Nihai karar ve yeniden açma koşulları

Karar **NO-GO / not achievable with current development-normal exposure** olarak dondurulmuştur.
Mevcut yanlış-alarm hedefi, sonucu gördükten sonra gevşetilmeyecek; mevcut veriyle tersine
mühendislik yapılmayacaktır.

Çalışma ancak aşağıdakilerden biri için ayrı kapsam, bütçe ve insan onayı verilirse yeniden
açılabilir:

- ALFA için mevcut 47 uçuşluk akademik corpus'un ötesinde yeni, kontrollü bir normal-uçuş
  kampanyası;
- RFLY için resmî kaynağın mevcut tavanının da ötesinde, benzer kullanılabilir maruziyet
  sağlayan yaklaşık 135 veya daha fazla yeni normal uçuş;
- ya da yanlış-alarm hedefinin ayrı bir ön-kayıtla yeniden müzakere edilmesi. Bu son seçenek
  mevcut deneyin devamı değil, yeni bir deney sözleşmesidir.

Bunlar uygulama ayrıntısı değil, yeni veri toplama veya bilimsel hedef değiştirme kararıdır;
projenin bu turdaki yetki ve kapsamının dışındadır.

## Kesinlik ve izolasyon sınırları

- Kör holdout açılmadı; açma koşulu oluşmadı.
- Test rolü kalibrasyon, hata analizi veya hedef seçimi için kullanılmadı.
- İlk keşif koşusundaki eşikler hiçbir üretim/karar hattına girmedi ve
  `DO_NOT_USE_THRESHOLDS.md` ile işaretli kaldı.
- Korumalı koşu `thresholds_frozen.json` yazmadı.
- `configs/residual_v1_cusum.json` değiştirilmedi. Rapor yazımı öncesi ve sonrası doğrulanan
  SHA-256: `627948fbfd060aa39f881f72c25cf359694642d546b2e02ee6a0a2e4d0777584`.
- Bu rapor event/uçuş recall'ı veya saha güvenilirliği iddia etmez. Kanıtlanan şey
  threshold-bağımsız dağılım ayrışmasıdır; operasyonel sınır kalibre edilmemiştir.

## Provenance ve denetim izi

Tüm yollar repo köküne göredir:

- K5 doğrulaması ve güncel R6 üretimi:
  `artifacts/residual_v1/runs/20260717_112136_phaseE_scaling_seed11`
- S-4: `artifacts/residual_v1/runs/20260717_111752_phaseE_s4_ablation_rfly_seed11`
- Scaling: `artifacts/residual_v1/runs/20260717_112136_phaseE_scaling_seed11`
- S-1: `artifacts/residual_v1/runs/20260717_112412_phaseE_s1_magnitude_seed11`
- S-3: `artifacts/residual_v1/runs/20260717_112813_phaseE_s3_separation_seed11`
- Reddedilen ilk kalibrasyon:
  `artifacts/residual_v1/runs/20260717_113330_phaseE_cusum_calibration_seed11`
- Korumalı kalibrasyon STOP'u:
  `artifacts/residual_v1/runs/20260717_113747_phaseE_cusum_calibration_seed11`
- Claude handout özeti:
  `artifacts/residual_v1/phase_e_handout_20260717/SUMMARY_FOR_CLAUDE.json`
- Görsel handout kökü: `artifacts/residual_v1/phase_e_handout_20260717`
- Maruziyet hata raporu:
  `artifacts/residual_v1/runs/20260717_113747_phaseE_cusum_calibration_seed11/CALIBRATION_COVERAGE_FAILURE.md`
- Reddedilen eşik işareti:
  `artifacts/residual_v1/runs/20260717_113330_phaseE_cusum_calibration_seed11/DO_NOT_USE_THRESHOLDS.md`
- RFLY kaynak sayımı: resmî 84, projede 51, en çok 33 ingest edilmemiş aday;
  indirme/ingest yapılmadı.

Bu rapor Görev 5.4'ün başarıyla tamamlandığını değil, kalibrasyonun fail-closed biçimde ve
sayısal gerekçeyle durduğunu kaydeder.
