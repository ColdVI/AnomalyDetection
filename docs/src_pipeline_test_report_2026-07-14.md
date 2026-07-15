# `src/` Veri Pipeline'ı — Test Raporu (2026-07-14)

## Kapsam

Bu rapor, `src/` paketinin (ingestion → silver → gold veri pipeline'ı,
`Dashboard/` hariç) test kapsamını konu alır. Görev: mevcut testleri
incelemek, gereksiz olanları kaldırmak, her özellik için test yazmak ve
bulguları bu raporda özetlemek.

## Özet sonuç

| | Önce | Sonra |
|---|---|---|
| Test dosyası sayısı | 9 | 10 |
| Toplam test | ~68 | **129** |
| Geçen | — | **129** |
| Kalan (fail) | 4 (`test_minio_retention.py`) | **0** |

```
129 passed in 1.62s
```

## 1. Kaldırılan testler

### `tests/test_loaders.py` → `tests/test_upload_raw.py` içine birleştirildi

`test_loaders.py` ve `test_upload_raw.py`, `upload_raw_file()`'ı neredeyse
birebir aynı senaryolarla (byte-for-byte kopyalama, dosya adı korunumu) iki
ayrı dosyada test ediyordu — gerçek bir kapsam farkı yoktu. Birleştirirken:

- İki dosyanın üzerinde çakıştığı testlerden biri (yalnızca dosya adını
  kontrol eden, daha zayıf olan) elendi.
- Sadece `test_loaders.py`'de bulunan `merge_tar_parts()` testleri
  (`concatenates_in_order`, `noop_if_merged_exists`, `raises_when_no_parts_exist`)
  korunarak `test_upload_raw.py`'ye taşındı.
- Sonuç dosya, projedeki `test_<source_module>.py` adlandırma kuralına uyuyor.
- `test_loaders.py` silindi.

### `tests/test_minio_retention.py` — tamamen kaldırıldı

**Bulgu:** `src/common/minio_io.py::apply_realtime_retention()` şu anda hem
(a) production kodunda hiçbir yerden çağrılmıyor, hem de (b) kurulu
`minio==7.2.20` paketiyle **çalıştırılamaz durumda** — fonksiyonun içindeki
`from minio.lifecycleconfig import Expiration, Filter, LifecycleConfig, Rule`
satırı `ImportError: cannot import name 'Filter' from 'minio.lifecycleconfig'`
fırlatıyor (kurulu sürümde `Filter` sınıfı yok).

Bu üç kanıt bir araya geldiğinde fonksiyonun tamamen terk edilmiş olduğu
netleşiyor:

1. `grep` ile doğrulandı: `apply_realtime_retention` sadece kendi tanımında,
   kendi test dosyasında ve arşivlenmiş eski bir Makefile'da
   (`archive/2026-07-10_legacy_non_adsb_ml/project_support/Makefile_OLD`)
   geçiyor — aktif hiçbir production kodu tarafından çağrılmıyor.
2. `Dashboard/codes/minio_archiver.py`'nin docstring'i, 2026-07-09 tarihli bir
   ürün kararıyla bu "7 günlük otomatik silme" özelliğinin **kaldırıldığını**
   zaten açıkça belgeliyor ("*2026-07-09 KARARI: MinIO'da 7 GÜNLÜK OTOMATIK
   SİLME KURALI YOK/KALDIRILDI*").
3. Fonksiyon kurulu `minio` paketi sürümüyle artık **çağrıldığı anda
   crash ediyor** — yani kullanılmaya kalkışılsa bile çalışmaz.

**Karar (kullanıcı onayıyla):** `test_minio_retention.py` silindi, kaynak
koddaki `apply_realtime_retention()` fonksiyonuna dokunulmadı (bu, "testleri
incele" kapsamının dışına çıkıp production kodu değiştirmek anlamına
gelirdi). Fonksiyon `src/common/minio_io.py` içinde hâlâ duruyor ama
**ölü ve bozuk** olarak işaretlendi — ileride tamamen silinmesi veya
güncel `minio` API'sine uyarlanması ayrı bir karar konusu.

## 2. Yeni eklenen testler (dosya bazında)

### `tests/test_local_store.py` — YENİ DOSYA (13 test)

`LocalObjectStoreClient`'ın (Docker/MinIO kurulmadan pipeline'ı çalıştırmayı
sağlayan disk-tabanlı `ObjectStoreClient` implementasyonu) daha önce **hiç**
testi yoktu. Kapsanan davranışlar: bucket oluşturma/idempotency,
`put_object`/`get_object` round-trip, dosya benzeri veri kabul etme, iç içe
dizin oluşturma, `remove_object` (var olan + olmayan dosya), `list_objects`
(boş bucket, olmayan bucket, sıralı+recursive listeleme, prefix filtreleme).

### `tests/test_adsblol_producer.py` — YENİ DOSYA (11 test)

`src/ingestion/adsblol_producer.py`'nin de daha önce **hiç** testi yoktu.
Sahte `requests.Session` (`_FakeSession`/`_FakeResponse`) ile hermetik test
edildi. Kapsanan davranışlar:
- `_build_headers()`: API key yok/var/sadece boşluk.
- `fetch_point()`: doğru URL üretimi, `ac` alanı eksikse boş liste, auth
  header iletimi, HTTP hatasında exception fırlatma.
- `poll_once()`: birden fazla sorgu noktasından gelen aynı `hex`'in
  deduplike edilmesi (son gelen kazanır), `hex`'i olmayan kayıtların
  atlanması, bir noktanın ağ hatası vermesinin diğer noktaları
  etkilememesi, boş nokta listesi.

`run()` (sinyal işleme + sonsuz döngü) kasıtlı olarak test edilmedi —
projedeki diğer tüm producer/consumer main loop'larıyla aynı kural: iş
mantığı saf fonksiyonlara çıkarılır, sadece onlar test edilir.

### `tests/test_provenance.py` — 5 yeni test eklendi (toplam 17)

`add_provenance()`'ın girdi doğrulama mantığı (TypeError/ValueError
yolları) daha önce test edilmiyordu. Eklenenler: özel `schema_version`
parametresi, DataFrame olmayan girdinin reddi (`None`, dict, list, string —
parametrize), boş/None/sayısal `source_type`/`source_file`/`schema_version`
değerlerinin reddi (her biri parametrize).

### `tests/test_minio_io.py` — 8 yeni test eklendi (toplam 23)

- `write_bronze`/`write_silver`/`write_gold`'un üçünün de DataFrame olmayan
  girdiyi reddettiği (tek bir ortak `_write_layer()` kontrolünden miras
  aldıkları) parametrize test ile doğrulandı.
- `ensure_bucket()`: eksik bucket'ı oluşturma, var olan bucket'a dokunmama.
- `get_minio_client()`: `STORAGE_BACKEND` ayarlanmadığında gerçek `Minio`
  dönmesi, `STORAGE_BACKEND=local` (ve büyük/küçük harf duyarsız `LOCAL`)
  ayarlandığında `LocalObjectStoreClient` dönmesi ve doğru `LOCAL_STORAGE_DIR`
  ile başlatılması. (`.env`'de `STORAGE_BACKEND` tanımlı olmadığı doğrulanarak
  varsayılan-davranış testinin yanlış pozitif vermeyeceğinden emin olundu.)

### `tests/test_gold_unify.py` — 6 yeni test eklendi (toplam 13)

- `is_military` alanının SADECE adsb.lol kaynaklı tablolarda doldurulduğu,
  diğer kaynaklarda (alfa, uav_attack) her zaman null kaldığı (2026-07-10
  kararı — `COLUMN_MAPS`'te `None`).
- `unify()`/`stream_unify()`'ın kayıtlı olmayan bir `source_type` için
  `ValueError` fırlatması.
- `clear_gold_before_unify()`: önceki unified parçaları temizleme ve
  temizlenecek bir şey yokken 0 dönme.

### `tests/test_parse_adsblol_realtime.py` — 5 yeni test eklendi (toplam 16)

- `_parse_ac_record`'ın `is_military` bit-flag mantığı (`dbFlags & 1`),
  6 senaryo parametrize edildi: bit set, bit+başka bit set, hiç bit yok,
  sadece farklı bit set, alan hiç gelmemiş, sayıya çevrilemeyen değer
  (crash yerine güvenli `False` varsayılanı).
- `parse_jsonl_bytes()`'in bozuk JSON satırlarını atlayıp devam etmesi
  (tamamı bozuksa boş DataFrame dönmesi dahil).
- `run()`'ın Bronze'da hiç JSONL yokken boş liste dönmesi ve Silver'a
  başarıyla yazılan Bronze dosyalarını silmesi (2026-07-09 kararı —
  aksi halde `run()` tekrar çağrıldığında aynı dosyalar ikinci kez işlenip
  Silver'da kopya satır üretir).

### `tests/test_parse_adsblol_historical.py` — en büyük ekleme (toplam 26)

- `is_military` bit-flag testi (realtime ile aynı parametrize desen).
- `_parse_tar_fileobj`'ın her flush sonrası `on_part_written` callback'ini
  çağırdığı, bozuk bir üyeyi atlayıp devam ettiği.
- **Checkpoint/resume sistemi** (2026-07-13, kullanıcı isteği üzerine
  eklenmişti — "dur deyince dursam ertesi gün kaldığı yerden devam
  edemez miyiz"): bu bölüm eklenene kadar **hiç testi yoktu**. Eklenenler:
  - `_load_checkpoint`: eksik dosyada boş state, round-trip, bozuk JSON'da
    boş state'e düşme, eksik anahtarları varsayılanla doldurma.
  - `_delete_uris`: her nesneyi silme, bir silme hatasından sonra devam etme.
  - `run()`: tar objesi yokken boş liste dönme, işlenen tar'ı
    `completed_tars`'a işaretleyip tekrar çalıştırmada atlama, `fresh=True`
    bayrağının yeniden işleyip önceki Silver'ı temizlemesi.
  - **En kritik senaryo** — kesinti sonrası devam etme: bir tar
    `in_progress` olarak işaretliyken (yarım kalmış bir Silver parçası
    referanslıyken) `run()` tekrar çağrıldığında, (1) o yarım parçanın
    silindiği, (2) tar'ın sıfırdan yeniden işlendiği, (3) checkpoint'in
    doğru güncellendiği doğrulandı.

## 3. Kapatılan kapsam boşlukları (özet)

| Alan | Önceki durum | Şimdi |
|---|---|---|
| `LocalObjectStoreClient` | 0 test | 13 test |
| `adsblol_producer.py` | 0 test | 11 test |
| `add_provenance` doğrulama hataları | test edilmiyordu | 8 senaryo |
| `is_military` bit-flag mantığı (3 dosyada) | test edilmiyordu | her üç dosyada parametrize test |
| Checkpoint/resume sistemi | 0 test | 9 test (kesinti senaryosu dahil) |
| `get_minio_client()` backend switching | test edilmiyordu | 3 test |
| Non-DataFrame girdi reddi (bronze/silver/gold) | test edilmiyordu | parametrize test |

## 4. Bilinen/kalan sorunlar

- **`apply_realtime_retention()`** (`src/common/minio_io.py`) hâlâ kaynak
  kodda duruyor, kullanılmıyor ve kurulu `minio` sürümüyle çalıştırılamıyor.
  Bu görevin kapsamı testlerle sınırlı olduğundan fonksiyona dokunulmadı;
  ileride ya tamamen silinmesi ya da güncel `minio.lifecycleconfig` API'sine
  (artık `Filter` yerine farklı bir yapı kullanıyor olabilir) uyarlanması
  önerilir.

## 5. Çalıştırma komutu

```bash
python -m pytest tests/test_adsblol_realtime.py tests/test_gold_unify.py \
  tests/test_local_store.py tests/test_minio_io.py tests/test_minio_io_delete.py \
  tests/test_parse_adsblol_historical.py tests/test_parse_adsblol_realtime.py \
  tests/test_provenance.py tests/test_upload_raw.py tests/test_adsblol_producer.py -v
```

Sonuç: **129 passed in 1.62s**, 0 fail.
