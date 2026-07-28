# Four-dataset probabilistic GPU v2 — group/sızıntı denetimi

Tarih: 2026-07-28

Güncelleme: Bu ilk hızlı denetimden sonra tekrar üretilebilir v3 registry v1
oluşturuldu. UAV-SEAD root-source kuralını açıklaştıran ve RflyMAD'in mevcut
`split_group_id` metadata'sını kullanan kanonik sayılar
`FOUR_DATASET_PROBABILISTIC_V3_GROUP_REGISTRY_V1_AUDIT_20260728.md` içindedir.
Bu belgedeki UAV-SEAD 259/137/1.088 sayıları ve yalnız exact-source düzeyindeki
RflyMAD yorumu tarihsel ilk-audit kaydı olarak korunmuştur; v3 kararı için yeni
registry raporu kullanılmalıdır.

## Kapsam ve statü

Bu denetim tamamlanmış v2 splitini değiştirmez. Amaç, `source_id` ayrıklığının
aynı session/campaign ailesini de ayrık tutup tutmadığını kontrol etmek ve yeni
eğitim gerekirse group-safe v3 sözleşmesinin gerekçesini dondurmaktır.

ADS-B contextual-physics hattı ile bu dört UAV veri-seti hattı tek bir transfer
modeli değildir. Ortak olan normal-only probabilistic forecasting fikridir;
veri şeması, scaler, model ve threshold her UAV veri setinde ayrıdır.

## Sonuç

V2 satır-random split yapmamış ve aynı `source_id`yi iki role koymamıştır. Buna
rağmen ALFA'da kesin session-aile çakışmaları, UAV Attack'ta campaign-aile
çakışmaları vardır. UAV-SEAD kaynak yolları da güçlü bir mission/campaign
çakışması riski göstermektedir. Bu nedenle v2, dengeli source-level development
deneyi olarak korunur; yeni genellenebilirlik iddiasının split temeli olamaz.

## ALFA — doğrulanmış session-aile çakışması

Grup anahtarı kaynak kimliğinin ilk
`carbonZ_YYYY-MM-DD-HH-MM-SS` bölümüdür. Bu, aynı temel oturumun `_1`, `_2`,
`_3` ve fault/no-fault parçalarını birlikte tutar.

- 8 temel session ailesi birden fazla role dağılmıştır.
- `2018-09-11-15-05-11`: normal parça validation, elevator-fault parça primary
  testtedir.
- `2018-10-05-14-34-20`: normal parça train, right-aileron-fault parça primary
  testtedir.
- `2018-10-05-15-52-12`: iki normal parça train/validation'a, engine-fault
  parça stress-test'e dağılmıştır.
- Başka normal/fault eşleşmeleri train ile stress-test arasında da vardır.

Bu yalnız teorik benzerlik değildir: aynı timestamp/session ailesinin parçaları
farklı rollerdedir. ALFA v3 bütün aileyi tek role/folda atamalıdır.

## UAV Attack — campaign/domain çakışması

Kaynak adı prefixi ve tarihinden türetilen muhafazakâr campaign anahtarıyla üç
campaign ailesi birden fazla role yayılmıştır:

- `ace_2033-8-19`: benign train, jamming primary test, spoofing stress-test;
- `log_2020-8-2`: benign train/validation/test ve spoofing test/stress-test;
- `numeric_2021-01-27`: Ping-DoS primary test ve stress-test.

Bu veri setinde yalnız bir normal validation uçuşu vardır. Event-evaluation v1
ön-kaydındaki en az beş normal validation uçuşu şartı nedeniyle mevcut split
operasyonel kalibrasyon iddiasına zaten uygun değildir. V3; platform,
live/simulation ve campaign'i açık grup anahtarı yapmalı ve mümkünse
leave-one-benign-campaign/domain-out raporlamalıdır.

## UAV-SEAD — yüksek risk, grup semantiği doğrulama bekliyor

Kaynak kimliğinin parent yolu geçici mission/campaign vekili olarak
kullanıldığında:

- 1,244 kaynak 259 parent grubuna inmektedir;
- 137 parent grubu birden fazla role yayılmaktadır;
- bu gruplarda toplam 1,088 kaynak bulunmaktadır.

Bu rakam doğrudan “1,088 sızıntılı uçuş” iddiası değildir. Bazı parent klasörler
birden fazla bağımsız görevi kapsıyor olabilir; kök dizinde kalan altı kimlik de
ayrı ele alınmalıdır. Ancak çok sayıda aynı-gün ve aynı-session klasörünün
train/validation/test/stress rollerine bölündüğü kesindir. V3 öncesinde kaynak
paketin mission/campaign metadata'sı doğrulanmalı; bulunamazsa parent/session
yolu muhafazakâr grup anahtarı olarak kullanılmalıdır.

## RflyMAD — exact-source ayrılığı var, transfer ekseni ayrı kalmalı

Seçili 1,551 kaynağın exact `canonical_case_id`/object kimlikleri benzersizdir.
V2 normal train/validation ayrımını SIL/HIL/Real domainine göre stratify eder:

| Domain | Train | Validation | Primary test | Stress-test |
|---|---:|---:|---:|---:|
| HIL | 140 | 30 | 49 | 468 |
| Real | 29 | 6 | 14 | 109 |
| SIL | 140 | 30 | 69 | 467 |

Bu, exact-source sızıntısı göstermemektedir; fakat in-domain dengeli test ile
SIL→HIL→Real transfer iddiası aynı şey değildir. Mevcut transfer kapıları ayrı
raporlanmalı ve başarısız statüleri v2 dengeli ROC sonucu ile
değiştirilmemelidir. Yeni RflyMAD v3, in-domain grouped evaluation ile domain
transferini iki ayrı protokol olarak tutmalıdır.

## V3 split kararı

Yeni model eğitimi ancak dataset-specific grup registry'si üretildikten sonra
başlayabilir:

- ALFA: temel timestamp/session ailesi;
- UAV Attack: campaign + platform + live/simulation;
- UAV-SEAD: doğrulanmış mission/campaign/session parent;
- RflyMAD: canonical case bütünlüğü; in-domain grup değerlendirmesi ve ayrı
  SIL/HIL/Real transfer protokolü.

Group anahtarı train/validation/test arasında kesişemez. Sınıf oranı ancak bu
kısıttan sonra dengelenebilir. GroupKFold/StratifiedGroupKFold araçtır; bilimsel
kararı dataset-specific grup anahtarının doğruluğu belirler.

## Feature ve rejim sırası

Evaluator ve group-safe split tamamlanmadan yeni ULog bloklarıyla başarı iddiası
aranmayacaktır. V3 sırası:

1. group-safe split;
2. mevcut feature seti ve aynı model ailesiyle baseline;
3. control, actuator, estimator, energy ve context bloklarını tek tek ablation;
4. bundan sonra flight-regime/domain conditioning.

`rfly_ctrl_lxl.fault_id`, `fault_mode`, injection command, filename fault kodu
ve onset metadata yalnız truth içindir; modele verilemez. PX4'in kendi health/
fault flagleri kullanılırsa fizik öğrenen model kanalı gibi gizlenmez, ayrı
PX4-native baseline ve açık ablation olarak raporlanır.
