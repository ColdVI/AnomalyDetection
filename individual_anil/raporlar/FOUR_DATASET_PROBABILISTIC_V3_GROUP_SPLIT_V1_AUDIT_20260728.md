# Four-dataset probabilistic v3 — group-safe split v1 sonuç denetimi

Tarih: 2026-07-28
Ön-kayıt:
`FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_PREREG_20260728.md`
Statü: ALFA, RflyMAD ve UAV-SEAD splitleri üretildi; bütün contract kontrolleri
geçti. UAV Attack veri-sözleşmesi NO-GO olarak kaydedildi ve eğitimi başlatılmadı.

## Üretilen kayıtlar

- Split manifesti: `configs/four_dataset_probabilistic_v3_split_manifest.json`
- Makine denetimi: `artifacts/four_dataset_probabilistic_v3/group_split_v1_report.json`
- Üretici: `scripts/build_four_dataset_probabilistic_v3_group_splits.py`
- Test: `tests/test_four_dataset_probabilistic_v3_group_splits.py`

Manifest, kullandığı registry Parquet ve JSON reportunun SHA-256 değerlerini
taşır. V2 splitleri ve sonuç artefaktları değiştirilmemiştir.

## Contract sonucu

- Her protokolde group overlap: 0.
- Her protokolde source overlap: 0.
- Train anomaly source: 0.
- Validation anomaly source: 0.
- Uygun üç veri setinde test hem normal hem anomalili kaynak içeriyor.
- Attack kaynaklarının 19'u da korunuyor fakat yeni role atanmıyor.
- Herhangi bir test metriği, loss, AUC veya event sonucu split üretiminde
  kullanılmadı.

## Sabit split sayıları

### RflyMAD — group-safe in-domain v1

| Rol | Grup | Kaynak | Normal | Anomali |
|---|---:|---:|---:|---:|
| Train | 7 | 260 | 260 | 0 |
| Validation | 3 | 90 | 90 | 0 |
| Test | 28 | 1.201 | 91 | 1.110 |

Her SIL/HIL/Real domaininde bir normal grup validation, bir normal grup test,
kalan normal gruplar train rolündedir. Test bütün 25 fault-scenario grubunu
içerir. Normal havuz kendi içinde `%59,0 train / %20,4 validation / %20,6 test`
olmuştur; 70/15/15'ten sapma, Real domaininde yalnız üç ve SIL/HIL'de beşer büyük
scenario grubu bulunmasının zorunlu sonucudur.

Testin toplam kaynakların %77,4'ünü içermesi supervised sınıf dengesizliği gibi
yorumlanmamalıdır: model normal-only forecaster'dır ve 1.110 fault kaynağının
hiçbiri optimizer'a giremez. Fault grubunu testten atıp toplam oranı güzelleştirmek
değerlendirme kapsamını yapay olarak daraltırdı.

### UAV-SEAD — group-safe parent v1

| Rol | Grup | Kaynak | Normal | Anomali |
|---|---:|---:|---:|---:|
| Train | 105 | 271 | 271 | 0 |
| Validation | 22 | 59 | 59 | 0 |
| Test | 137 | 914 | 568 | 346 |

Normal-only 388 kaynak kendi içinde `%69,8 train / %15,2 validation / %14,9
normal test` hedefini karşılar. Testteki 568 normal kaynağın 510'u anomalili
kaynaklarla aynı parent gruptadır. Bunları train/validation'a taşımak etikete göre
aynı parent grubunu bölmek olacağı için yapılmadı. Sonuçta test büyük görünür;
bu bir bug değil, group-safe sözleşmenin ölçülen maliyetidir.

Test bütün anomaly label envanterini korur: 72 altitude, 193 external-position,
40 global-position ve 41 mechanical-fault kaynak.

### ALFA — dört fold group-safe smoke/CV

Her foldda train dört, validation iki saf-normal session kaynağı içerir. Test
foldları aşağıdaki gibidir:

| Fold | Test grup | Test kaynak | Test normal | Test anomali | Deferred/stress kaynak |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 12 | 3 | 9 | 36 |
| 1 | 10 | 13 | 2 | 11 | 35 |
| 2 | 9 | 15 | 4 | 11 | 33 |
| 3 | 9 | 14 | 6 | 8 | 34 |

Mixed sessionlar nedeniyle test normal sayısı outer folda atanan iki saf-normal
kaynaktan daha yüksek olabilir. Her session tek roldedir. Foldlar arasında en iyi
sonuç seçilmeyecek; tamamı ve foldlar-arası dağılım raporlanacaktır. Yalnız iki
normal validation session bulunduğundan operasyonel false-event/saat güvencesi
kurulamaz; ALFA smoke/CV statüsünü korur.

### UAV Attack — veri-sözleşmesi NO-GO

Archive-backed gruplamada yalnız bir normal-only grup vardır:
`simulated:PX4-QUAD-SITL:2020-08-01`. Diğer beş normal kaynak, aynı mode/platform/
campaign grubundaki saldırı kaynağıyla birlikte mixed gruptadır.

Bir saf-normal grupla ayrık train, validation ve normal-test kurmak mümkün
değildir. Bu nedenle `physics_v3_training_authorized=false` donduruldu. Sorun GPU,
epoch veya model mimarisi değildir; yeni bağımsız normal campaign gerektirir.

## Bilimsel yorum kuralı

RflyMAD ve UAV-SEAD tablolarında testin büyük olması “80/10/10 random split”e
dönüş değildir. V3 önce grup bağımsızlığını korur, sonra yalnız normal-only havuzu
train/validation/normal-test arasında böler; bütün anomalili veya mixed grupları
değerlendirmede tutar. Raporlarda bu nedenle üç oran ayrı verilecektir:

1. saf-normal havuzun train/validation/test oranı;
2. tüm kaynakların rol oranı;
3. testin normal/anomali kompozisyonu.

Bu üç düzey birbirine karıştırılmayacaktır.

## Sonraki adım

V3 baseline eğitim kodu bu manifesti doğrudan okuyacak ve başlangıçta registry/
manifest hashlerini yeniden doğrulayacaktır. Uygulama sırası değişmez:

1. ALFA dört-fold pipeline smoke;
2. RflyMAD group-safe fiziksel-fault baseline;
3. UAV-SEAD group-safe phase/context baseline;
4. UAV Attack için eğitim yok; yalnız ayrı timing/network veri sözleşmesi.

Phase veya raw kanal kazancı, aynı splitteki v3 feature-baseline geçmeden iddia
edilmeyecektir.
