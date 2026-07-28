# Four-dataset probabilistic v3 — group registry v1 denetimi

Tarih: 2026-07-28
Statü: Registry üretildi ve sözleşme kontrolleri geçti. Bu artefakt yeni v3
train/validation/test rolü atamaz; dondurulmuş v2 rollerinin daha üst düzey
session/campaign/scenario gruplarını kesip kesmediğini ölçer.

## Neden bu adım gerekliydi?

V2 satır veya pencere bazında random split kullanmadı ve aynı `source_id`yi iki
role koymadı. Buna rağmen iki farklı kaynak kaydı aynı uçuş oturumunun parçaları,
aynı saldırı campaign'i veya aynı simülasyon scenario ailesi olabilir. Bu durumda
source-level ayrıklık tek başına bağımsız genelleme kanıtı değildir.

V3'ün ilk kapısı bu nedenle şöyledir:

> Önce her kaynak dataset-specific bir gruba bağlanır; yeni train, validation ve
> test rolleri ancak bu grup anahtarları arasında sıfır kesişim üretecek biçimde
> dondurulur.

## Üretilen artefaktlar

- Registry: `artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet`
- Makine-okunur denetim: `artifacts/four_dataset_probabilistic_v3/group_registry_v1_report.json`
- Üretici: `scripts/build_four_dataset_probabilistic_v3_group_registry.py`
- Test: `tests/test_four_dataset_probabilistic_v3_group_registry.py`

Registry her v2 kaynağı için `dataset`, `source_id`, `v2_role`, `flight_label`,
`is_normal`, `group_id`, grup anahtarının kanıt temeli, metadata statüsü ve
grubun kaç v2 role yayıldığını kaydeder. RflyMAD satırlarında domain, platform,
flight status ve fault-family metadata'sı da korunur.

## Contract doğrulamaları

- 2.868 v2 kaynağının tamamı tam bir kez registry'ye girdi.
- Dört veri setinin train/validation/test/stress rol kapsamı eksiksiz ve ayrıktı.
- ALFA, UAV-SEAD ve UAV Attack Gold Parquet kaynak kümeleri v2 manifestiyle tam
  eşleşti.
- RflyMAD'in seçili 1.551 `canonical_case_id` kaydı dataset manifestinde tam bir
  kez bulundu; `split_group_id` alanı boş değildi.
- RflyMAD v2 `flight_label` değerleri manifestteki `fault_family` değerleriyle
  bire bir eşleşti.
- Registry üretimi mevcut v2 splitini, checkpointleri, skorları veya sonuçları
  değiştirmedi.

## Ölçülen v2 grup çakışmaları

Buradaki “çakışan kaynak”, mutlaka yinelenmiş aynı uçuş demek değildir. Yalnızca
seçilen muhafazakâr bağımlılık grubunun birden fazla v2 role dağıldığını söyler.

| Veri seti | V2 kaynak | Registry grubu | Birden fazla v2 role yayılan grup | Bu gruplardaki kaynak | Kaynak oranı |
|---|---:|---:|---:|---:|---:|
| ALFA | 54 | 38 | 8 | 20 | %37,0 |
| RflyMAD | 1.551 | 38 | 31 | 1.458 | %94,0 |
| UAV-SEAD | 1.244 | 264 | 136 | 1.082 | %87,0 |
| UAV Attack | 19 | 13 | 5 | 11 | %57,9 |
| **Toplam** | **2.868** | **353** | — | — | — |

Veri setleri arasında grup tanımı farklı olduğu için “çakışan grup” toplamı tek
bir kalite skoru olarak toplanmaz.

## Dataset-specific bulgular

### ALFA

Grup anahtarı kaynak adındaki `carbonZ_YYYY-MM-DD-HH-MM-SS` temel session
ailesidir ve deterministik kaynak gramerine dayanır. Sekiz session ailesi v2
rollerini keser. Önemli örnekler:

- `2018-09-11-15-05-11`: normal parça validation, elevator-fault parça test;
- `2018-10-05-14-34-20`: normal parça train, aileron-fault parça test;
- `2018-10-05-15-52-12`: normal parçalar train/validation, engine-fault parça
  stress-test.

Bu anahtar v3 grouped-CV/split için kullanılabilir durumdadır.

### RflyMAD

Grup anahtarı kaynak paketin kendi `dataset_manifest.split_group_id` alanıdır;
bu nedenle dört veri seti içindeki en güçlü metadata dayanağına sahiptir. Seçili
1.551 kaynak yalnız 38 gerçek-session veya simülasyon-scenario grubuna iner;
31 grup v2 rollerini keser.

Normal HIL ve SIL kaynaklarında aynı `circling`, `dece`, `hover`, `velocity` ve
`waypoint` scenario ailelerinin train/validation/teste bölünmesi; Real normal
session günlerinin de aynı üç role yayılması ölçülmüştür. Fault scenario
ailelerinin bir bölümü primary test ile stress-test arasında bölünmüştür.

Bu bulgu v2 ROC-AUC ölçümünü silmez ve her scenario üyesinin kopya olduğunu iddia
etmez. Ancak v2'nin exact-source ayrıklığını scenario-aile genellenebilirliği
olarak yorumlamayı engeller. RflyMAD v3 hem `split_group_id`-safe in-domain
protokolü hem de ayrı SIL/HIL/Real transfer protokolü kullanmalıdır.

### UAV-SEAD

Doğrulanmış mission kimliği henüz aktif Gold katmanda bulunmadığı için kaynak
parent path'i muhafazakâr proxy olarak kullanıldı. Root seviyesindeki altı kaynak,
birlikte olduklarına dair kanıt bulunmadığından ayrı gruplar olarak tutuldu.

Bronze `labels.json` ayrıca denetlendi: 1.246 kaydın tamamında yalnız `class`,
`label`, `object_name`, `ranges` ve `size_bytes` alanları vardır; mission/session
alanı yoktur. V2'de seçili 1.244 kaynağın tamamı bu kayıtta bulunur, iki ek kaynak
(`2019-01-22/07_55_19`, `2019-03-04/08_04_13`) v2 kapsamının dışındadır.

Bu açık kural 1.244 kaynağı 264 gruba indirdi; 136 grup birden fazla v2 role ve
toplam 1.082 kaynağa yayılıyor. Önceki hızlı denetimdeki 259/137/1.088 sayıları
root kayıtlarını tek parent altında birleştiren örtük kurala dayanıyordu. Registry
v1'deki 264/136/1.082 ölçümü açık ve tekrar üretilebilir kuralın kanonik sonucudur.

Parent path gerçek mission metadata'sı değildir. Aktif Gold ve Bronze label
metadata'sında daha güçlü bir mission/session alanı bulunmadığı doğrulandığı için
bu muhafazakâr fallback ilk v3 protokolü için açık sınırlamasıyla donduruldu.
Gelecekte daha güçlü harici metadata bulunursa mevcut kayıt sessizce değiştirilmez;
yeni versioned registry üretilir.

### UAV Attack

Raw `UAVAttackData.zip` dizin envanteri 786 dosya içerir: 601'i simulated OTU
Survey, 185'i live GPS spoofing/jamming ağacındadır. Seçili 19 kaynağın her biri
tek bir archive dizinine eşlendi. Grup anahtarı bu doğrulanmış live/simulated
modu, platform dizini ve source-id içindeki normalize campaign tarihidir. On üç
grup oluştu; beşi rolleri kesiyor:

- `live:live_platform_unspecified:2033-08-19`: train/test/stress-test;
- `simulated:PX4-H480-SITL:2020-08-02`: train/test;
- `simulated:PX4-PLANE-SITL:2020-08-02`: validation/stress-test;
- `simulated:PX4-TAIL-SITL:2020-08-02`: test/stress-test;
- `simulated:PX4-VTOL-SITL:2020-08-02`: train/stress-test.

Live kayıtlarda platform adı archive tarafından verilmediği için açıkça
`live_platform_unspecified` olarak tutuldu; bir platform tahmin edilmedi. Attack
v3 fizik modeli hattına katılmadan önce campaign-safe normal calibration
yeterliliği ayrıca ölçülmeli; message timing/topic-rate/jitter/gap özellikleri
ayrı sözleşmeyle ele alınmalıdır.

## Karar ve sonraki kapı

- ALFA session anahtarı ve RflyMAD `split_group_id` anahtarı v3 split tasarımı
  için kullanılabilir.
- UAV Attack mode/platform bilgisi raw ZIP yolu ile doğrulandı; live platform adı
  pakette bulunmadığı için eksiklik açık bir değerle korunuyor.
- UAV-SEAD parent path anahtarı
  `frozen_conservative_proxy_active_metadata_exhausted` statüsündedir; bu isim
  hem kararı hem sınırlamasını taşır.
- Adım 2 split manifesti daha sonra bu registry anahtarları değiştirilmeden
  üretildi; sonuç `FOUR_DATASET_PROBABILISTIC_V3_GROUP_SPLIT_V1_AUDIT_20260728.md`
  dosyasındadır.
- Yeni v3 split üretildiğinde train/validation/test group kesişimi sıfır değilse
  eğitim başlamayacaktır.
- Sınıf dengesi grup ayrıklığından sonra optimize edilecek; satır/pencere random
  split kullanılmayacaktır.

## Group-safe split fizibilitesi

Bir grup normal ve anomalili kaynakları birlikte içeriyorsa, anomalili üyeyi
train/validation dışarıda tutup normal üyeyi içeri almak grup ayrıklığını bozar.
Bu nedenle optimizer ve normal calibration yalnız `normal_only` gruplardan
beslenebilir.

| Veri seti | Normal-only grup/kaynak | Mixed grup/kaynak | Anomaly-only grup/kaynak | Üç normal rol için temel fizibilite |
|---|---:|---:|---:|---|
| ALFA | 8 / 8 | 6 / 15 | 24 / 31 | Var; fakat küçük grouped-CV/smoke-test |
| RflyMAD | 13 / 441 | 0 / 0 | 25 / 1.110 | Var; domain-stratified group split |
| UAV-SEAD | 149 / 388 | 85 / 811 | 30 / 45 | Var; 510 mixed normal kaynak optimizer dışı kalır |
| UAV Attack | 1 / 1 | 5 / 11 | 7 / 7 | **Yok** |

Attack için yalnız bir saf-normal grup vardır; aynı anda group-safe normal train,
validation ve test kurmak matematiksel olarak mümkün değildir. Bu, model sonucu
değil veri sözleşmesi NO-GO'sudur. Attack kaynakları silinmez ve v2 raporu
korunur; yeni normal campaign toplanmadan fizik-v3 eğitimi başlatılmaz. Ayrı
network/timing tanımlayıcı analizi bu karardan bağımsız bir araştırma track'idir.

## Tekrar üretim

```powershell
.venv\Scripts\python.exe scripts/build_four_dataset_probabilistic_v3_group_registry.py
.venv\Scripts\python.exe -m pytest tests/test_four_dataset_probabilistic_v3_group_registry.py -q
```

İlk üretimde 4/4 unit test geçti. Mevcut artefaktın bilerek yeniden üretilmesi
gerekirse önce rapor incelenmeli ve komut açıkça `--overwrite` ile çalıştırılmalıdır.

## İleride rapora aktarılacak kısa metin

> Kaynak düzeyinde ayrık olan v2 splitleri, dataset-specific session/campaign/
> scenario anahtarlarıyla yeniden denetlendi. Denetim aynı kaynak kimliğinin
> roller arasında yinelenmediğini doğrularken, daha üst bağımlılık gruplarının
> dört veri setinde de rolleri kestiğini gösterdi. Bu nedenle v2 sonuçları
> development baseline olarak korundu; v3 genellenebilirlik deneyi için grup
> ayrıklığı eğitim öncesi zorunlu sözleşme kapısı hâline getirildi.
