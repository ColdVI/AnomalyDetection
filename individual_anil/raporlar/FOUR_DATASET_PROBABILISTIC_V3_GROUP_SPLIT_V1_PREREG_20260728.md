# Four-dataset probabilistic v3 — group-safe split v1 ön-kayıt

Tarih: 2026-07-28
Statü: Model eğitimi ve v3 test skoru görülmeden önce dondurulmuş split üretim
sözleşmesidir. V2 splitini, modellerini, skorlarını veya raporlarını değiştirmez.

## Girdi sözleşmesi

Tek grup kaynağı
`artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet` dosyasıdır.
Manifest bu dosya ile makine-okunur registry raporunun SHA-256 değerlerini taşır.

Global kurallar:

- split birimi dataset-specific `group_id`dir;
- aynı grup iki role giremez;
- train ve validation yalnız normal-only gruplardan oluşur;
- normal ile anomalili kaynak taşıyan mixed grup etikete göre parçalanamaz;
- test metriği split, seed veya protokol seçemez;
- satır, pencere ve bağımsız `source_id` random split yasaktır;
- seed `20260728`dir.

## ALFA protokolü

ALFA tek splitli operasyonel sonuç değil, dört fold group-safe smoke/CV
protokolüdür.

1. Sekiz normal-only session grubu SHA-256 tabanlı deterministik sıraya konur ve
   dört outer folda ikişer atanır.
2. Mixed/anomaly-only session grupları label-signature içinde deterministik
   round-robin ile outer foldlara dağıtılır.
3. Her turda ilgili outer fold `test`tir.
4. Bir sonraki outer foldun iki normal-only grubu `validation`dır.
5. Kalan normal-only gruplar `train`; test foldunda olmayan anomalili/mixed
   gruplar `stress_test`tir ve optimizer/threshold göremez.

Her validation ve normal-test yalnız iki bağımsız normal session içerdiği için
ALFA sonucu operasyonel false-event/saat iddiasına uygun değildir. Dört foldun
tamamı ayrı raporlanır; en iyi fold seçilmez.

## RflyMAD protokolü

Ana protokol `group_safe_in_domain_v1`dir.

- Normal-only gruplar SIL, HIL ve Real içinde ayrı sıralanır.
- Her domainde bir grup validation, bir grup normal test, kalan gruplar train
  rolüne atanır.
- Bütün mixed/anomaly-only gruplar testtedir; tek bir scenario grubu primary ve
  stress arasında parçalanmaz.
- Test bütün fault gruplarını içerir; test sonucu görülerek fault grubu seçilmez.
- SIL→HIL→Real transfer protokolleri bu in-domain splitten ayrı tutulur ve sonraki
  versioned sözleşmede ayrıca dondurulur.

## UAV-SEAD protokolü

Ana protokol `group_safe_in_domain_v1`dir.

- Aktif metadata'da mission/session alanı bulunmadığı için dondurulmuş
  muhafazakâr parent-path grupları kullanılır.
- Normal-only gruplar, grup büyüklüğü korunarak deterministik ağırlıklı
  `%70 train / %15 validation / %15 normal test` kaynak hedeflerine atanır.
- Bütün 85 mixed ve 30 anomaly-only grup testtedir.
- Mixed gruplardaki normal kaynaklar train/validation'a taşınmaz; aksi davranış
  parent grubunu etikete göre bölerek sözleşmeyi bozar.

Bu karar kullanılabilir normal optimizer havuzunu 898'den 388 kaynağa indirir.
Kayıp saklanmaz; v3 raporu eğitim ve değerlendirme exposure'ını grup ve kaynak
düzeyinde birlikte gösterecektir.

## UAV Attack NO-GO

Archive-backed live/simulated + platform + campaign gruplaması altında yalnız
bir normal-only grup vardır. Disjoint normal train, validation ve test rollerini
aynı anda doldurmak mümkün değildir.

Bu nedenle:

- physics-v3 eğitimi başlatılmaz;
- mevcut 19 kaynak silinmez veya başka campaign gibi yeniden etiketlenmez;
- v2 negatif sonuç korunur;
- yeni bağımsız normal mode/platform/campaign grupları gelene kadar durum
  `no_go_group_safe_split_not_feasible` kalır;
- network/timing tanımlayıcı analizi ayrı bir track olabilir, fakat physics-v3
  başarısı gibi sunulamaz.

## Üretilecek kayıtlar

- `configs/four_dataset_probabilistic_v3_split_manifest.json`
- `artifacts/four_dataset_probabilistic_v3/group_split_v1_report.json`
- üretici: `scripts/build_four_dataset_probabilistic_v3_group_splits.py`

Manifest bütün grup ve source rol listelerini taşır. Report her rol için grup,
normal/anomali kaynak ve label sayılarını; sıfır overlap sonucunu ve Attack
NO-GO'sunu saklar.

## Eğitim kapısı

Eğitim yalnız aşağıdaki kontrollerin tamamı geçerse başlayabilir:

1. registry ve report hashleri manifest girdileriyle eşleşir;
2. her protokolde group overlap sıfırdır;
3. source overlap sıfırdır;
4. train/validation anomaly sayısı sıfırdır;
5. testte normal ve anomalili kaynak vardır;
6. Attack physics-v3 işi başlatılmamıştır.

Bu kapılardan biri bozulursa eşik veya seed değiştirilmez; contract hatası olarak
raporlanır.
