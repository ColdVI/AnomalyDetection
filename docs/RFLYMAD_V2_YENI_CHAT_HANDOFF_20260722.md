# RflyMAD-Full v2 — Yeni Sohbet Handoff

> Son güncelleme: 2026-07-22 (Europe/Istanbul)  
> Kapsam: RflyMAD-Full v2 veri, truth, Temporal AE ve supervised TCN çalışmaları.  
> Amaç: Yeni sohbetin yeniden keşif yapmadan güvenli biçimde devam edebilmesi.

## 0. Kritik güncel durum — truth schema v2 sonrası

Bu dosyanın aşağıdaki tarihsel bölümlerinde parser düzeltmesi öncesi sonuçlar da
korunmaktadır. Güncel ve yetkili durum şudur:

- `rfly_full/v2_parser.py` canonical `domain_of()` kullanacak biçimde düzeltildi;
  `TRUTH_SCHEMA_VERSION=2` eklendi.
- State migration tam **2.712** underscore SIL/HIL uçuşunu seçici olarak invalidated
  etti; diğer 3.893 uçuş yeniden parse edilmedi.
- Reparse tamamlandı: **6.605 completed / 0 failed / 0 remaining**.
- Yeni truth audit: `active_from_first_sample=0`, missing truth=5 (yalnız Real/Motor),
  interval violation=0, eksik v2 feature şeması=0, güçlü trajectory leakage kümesi=0.
- `truth_crosscheck_disagreement=5.577` hâlâ yüksektir. Ham ULog denetimi bunun
  özellikle SIL'de control topic'in yerel uçuş eksenine ~13–14 saniye geç başlaması
  ve TestInfo/control bitiş tanımlarının farklı olması nedeniyle mevcut `%1 örnek
  farkı` metriğinin aşırı hassas olduğunu gösterdi. Bu sayı parser düzeltmesinin
  başarısızlığı olarak yorumlanmamalı; cross-check metriği ayrıca iyileştirilmelidir.
- Düzeltilmiş beş-rotasyon AE sweep'i tamamlandı:
  `artifacts/rfly_full/v2/normal_temporal_ae/sweep_20260722_093049/`.

| Politika | Recall ort. | Recall std | Tüm nonfault FA/saat | FA std | Wind FA/saat |
|---|---:|---:|---:|---:|---:|
| Critical | %60,43 | %5,54 | 1,28 | 0,35 | 28,46 |
| Advisory | %69,84 | %5,85 | 3,64 | 1,32 | 31,54 |

Critical artık beş rotasyonun tamamında `recall >= %50` ve `FA <= 2/saat`
kapılarını geçmektedir (min recall %51,84; max FA 1,65/saat). Model/eşik ve NoFault
validation FA değerleri önceki koşuyla deterministik olarak aynıdır. Recall neredeyse
değişmemiştir; asıl değişim, düzeltilmiş fault pre/post-normal maruziyetinin
`all_nonfault_fa_per_hour` paydasını doğru büyütmesidir. Wind ve Real-domain sorunları
çözülmemiştir; operasyonel iddia hâlâ yasaktır.

## 1. Yeni sohbet için kısa başlangıç mesajı

Bu dosyayı oku, ardından `AGENTS.md` kurallarına uyarak RflyMAD-Full v2
çalışmasına devam et. Parser truth schema v2 düzeltmesi, 2.712 uçuşluk seçici
reparse, truth audit ve beş-rotasyon AE tekrar koşusu tamamlandı. Sıradaki iş
locked test'i kapalı tutarak Wind/Real-domain robustness deney sözleşmesini
netleştirmek; ardından development-only 3-epoch supervised TCN sanity koşusudur.
`archive/` salt-okunurdur. Kullanıcı istemeden commit/push yapma.

## 2. Zorunlu proje sınırları

- Kök `AGENTS.md` geçerlidir.
- `archive/` salt-okunur tarihçedir; eski kod buradan import edilmemelidir.
- Aktif çalışma temiz `rfly_full/` namespace'indedir.
- Satır/window, event ve uçuş metrikleri birbirine karıştırılmamalıdır.
- SIL/HIL sonucu Real-domain başarısı olarak sunulmamalıdır.
- Kilitli test, model/threshold seçimi için okunmamalıdır.
- Wind sistem arızası değil, ayrı bir çevresel robustness problemidir.
- Çalışma ağacı kirlidir ve RflyMAD dosyalarının çoğu henüz untracked'dir.
  Kullanıcı değişikliklerini silme/resetleme.
- Bu handoff yazılırken commit veya push yapılmadı.

## 3. Veri durumu

- Resmî RflyMAD kaynaklarının yerel indirme/parçalama kuyruğu tamamlandı.
- Canonical uçuş sayısı: **6.605**.
- Domain/aile dağılımı:

| Domain | NoFault | Motor | Sensor | Propeller | Load | Voltage | Wind |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real | 51 | 245 | 197 | 0 | 0 | 0 | 0 |
| HIL | 240 | 921 | 690 | 435 | 291 | 36 | 443 |
| SIL | 240 | 921 | 690 | 435 | 291 | 36 | 443 |

- Manifestte exact ULog SHA-256 duplicate bulunmadı.
- 10 Hz v2 parse ilk turu: **6.605/6.605**, `failed=0`, `remaining=0`.
- İlk “parse complete” durumu truth hatası keşfedilmeden önceydi. Bölüm 8'deki
  2.712 uçuş truth schema v2 ile seçici olarak yeniden parse edildi; güncel state
  truth açısından da `complete` durumundadır.

Temel dosyalar:

- `artifacts/rfly_full/v2/dataset_manifest.parquet`
- `artifacts/rfly_full/v2/dataset_manifest_summary.json`
- `artifacts/rfly_full/v2/parse_10hz_state.json`
- `artifacts/rfly_full/v2/parsed_10hz/`
- Parser: `rfly_full/v2_parser.py`
- Manifest/split: `rfly_full/contract.py`

## 4. Üretilen v2 özellikler ve preprocessing

Parser causal/backward merge ve 10 Hz ortak zaman ekseni kullanır. Tarihsel 1 Hz
çıktıların üzerine yazmaz. Temel telemetriye ek olarak yüksek frekanslı IMU ve
aktüatör özetleri üretilir:

- ivme ve gyro magnitude: mean, standard deviation, RMS, peak-to-peak;
- ardışık örnek farkı RMS;
- actuator output: mean, standard deviation, peak-to-peak ve difference RMS;
- eksikler model girişinde ayrıca maskelenir;
- scaler yalnız development verisinin ilgili eğitim bölümüne fit edilir.

Normal Temporal AE girişinde pencere başına normalize edilmiş telemetri ve
missingness maskesi bulunur. Loss yalnız gözlenen değerlerde hesaplanır. TCN'de
de telemetri ile missingness ayrı kanallar halinde kullanılır.

## 5. Deney sözleşmesi

### Normal-only Temporal AE

- Eğitim: yalnız `split=development`, `evaluation_role=normal_reference`.
- Validation: her Real/HIL/SIL domain'inden tamamen ayrılmış normal grup.
- Tek ortak temporal convolutional AE; domain-bazlı eşik kalibrasyonu.
- Eşik: yalnız held-out development normal verisi.
- Arızalı development uçuşları model/eşik dondurulduktan sonra değerlendirilir.
- Alarm politikası: 4-of-6 saniye, 30 saniye refractory.
- NoFault, fault pre/post-normal ve Wind FA/saat ayrı raporlanır.
- Kilitli test özellikleri okunmaz.

Kod ve komutlar:

- `rfly_full/normal_ae.py`
- `scripts/run_rfly_full_v2_normal_ae.py`
- `scripts/run_rfly_full_v2_normal_ae_sweep.py`
- `scripts/render_rfly_full_v2_normal_ae_sweep.py`

### Supervised multitask TCN

- Negatif eğitim penceresi: yalnız NoFault ve tamamen fault-inactive.
- Pozitif pencere: tamamı fault-active olan sistem arızası penceresi.
- Transition, pre/post karışık pencere ve Wind training loss'a girmez.
- İki head: binary anomaly ve conditional fault-family.
- Temperature calibration development validation üzerinde yapılır.
- Normal + etiketli arızalarla eğitim yapılır; yalnız anomalilerle eğitip yalnız
  normalleri test etmek ana sözleşme değildir.

Kod ve komutlar:

- `rfly_full/supervised.py`
- `scripts/run_rfly_full_v2_supervised.py`
- `scripts/benchmark_rfly_full_v2_tcn_memory.py`

## 6. Truth audit ve sızıntı denetimi

Yazılan bileşenler:

- `rfly_full/truth_audit.py`
- `scripts/run_rfly_full_v2_truth_audit.py`
- `tests/test_rfly_full_truth_audit.py`
- Çıktı: `artifacts/rfly_full/v2/truth_audit/`

İlk audit sonucu, düzeltme öncesi:

| Bulgular | Sayı |
|---|---:|
| Audit edilen uçuş | 6.605 |
| `rfly_ctrl_lxl` truth | 6.069 |
| `normal_no_fault` | 531 |
| Missing truth | 5 |
| Sistem arızalı uçuş | 5.188 |
| Cross-check disagreement | 5.583 |
| İlk örnekten aktif görünen | 1.354 |
| Aralık ihlali | 0 |
| Eksik v2 feature şeması | 0 |

Manifestte bulunan eski **4.746 provisional TestInfo truth** sayısı v2 truth
dağılımı değildir. V2'de `test_info_fallback=0`; dağılım 6.069 control truth,
531 normal ve 5 missing şeklindedir. Ancak control truth içindeki alt çizgili
paket hatası düzeltilmeden bu da nihai değildir.

Near-duplicate denetimi:

- Zayıf duration-signature tier: 1.285 küme; tek başına sızıntı kanıtı değildir.
- Güçlü trajectory-fingerprint tier: **0 küme**.
- Development ve locked-test arasında güçlü trajectory riski: **0**.
- Önceden görülen risk kümeleri zayıf heuristic yanlış pozitifleridir.

## 7. Düzeltme öncesi tam veri Temporal AE sweep — tarihsel/provisional

Klasör:

`artifacts/rfly_full/v2/normal_temporal_ae/sweep_20260721_170900/`

Beş validation rotasyonu tamamlandı ve recall–FA scatter, stabilite grafiği,
domain/aile heatmap ve confusion matrix üretildi.

| Politika | Recall ort. | Recall std | Tüm nonfault FA/saat | FA std | Wind FA/saat |
|---|---:|---:|---:|---:|---:|
| Critical | %60,45 | %5,55 | 2,31 | 0,63 | 28,46 |
| Advisory | %69,89 | %5,86 | 6,53 | 2,29 | 31,54 |

- Advisory development kapısı (`recall>=%50`, `FA<=12/saat`) geçiyor.
- Critical recall kapısı geçiyor fakat ortalama FA 2,31 ile 2/saat bütçesini
  aşıyor; rotasyon aralığı yaklaşık 1,30–2,97 ve kararlı geçiş yok.
- Wind tüm rotasyonlarda çözülmemiş durumda.

Domain/aile ortalama recall:

| Domain / aile | Advisory | Critical |
|---|---:|---:|
| HIL / Load | %65,57 | %56,88 |
| HIL / Motor | %76,29 | %62,88 |
| HIL / Propeller | %92,76 | %86,62 |
| HIL / Sensor | %50,58 | %40,58 |
| HIL / Voltage | %36,67 | %26,67 |
| SIL / Load | %74,43 | %65,65 |
| SIL / Motor | %92,18 | %83,76 |
| SIL / Propeller | %96,27 | %92,37 |
| SIL / Sensor | %46,30 | %38,33 |
| SIL / Voltage | %50,67 | %38,00 |
| Real / Motor | %31,12 | %20,20 |
| Real / Sensor | %27,34 | %8,35 |

Önemli yorum:

- Real transfer kanıtı yok; özellikle Real-Sensor critical çok zayıf.
- HIL/SIL Motor ve Propeller recall değerleri Bölüm 8'deki truth hatasından
  doğrudan etkilenir ve şu anda başarı kanıtı olarak kullanılamaz.
- Normal eğitim uçuşları ve NoFault FA kalibrasyonu doğrudan bozulmamıştır;
  bozuk olan arıza başlangıcı/aktif aralığı, recall ve detection-delay hesabıdır.
- Hata bazı uçuşlarda fault onset'i erkene taşıdığı için pre-fault alarmları TP
  sayabilir; pooled recall yapay biçimde yüksek olabilir.

## 8. Kritik açık hata: SIL/HIL underscore paketlerinin truth'u

### Kök neden

`rfly_full/v2_parser.py` içinde control sentinel seçimi şu mantıkla yapılmıştır:

```python
str(package).split("-", 1)[0].upper()
```

Bu, `HIL-Sensors` için `HIL` üretirken `HIL_Motor_1` için `HIL_MOTOR_1`
üretir. `_active_control()` yalnız domain tam olarak `SIL` veya `HIL` ise idle
sentinel'i `0` seçmektedir; yanlış domain yüzünden `1500` seçilmiş ve geçerli
`ctrl_id=0, ctrl_mode=0` idle durumu arıza kabul edilmiştir.

### Ham ULog kanıtı

- `HIL_Motor_1`: kontrol `(0,0)` ile başlıyor; yaklaşık 18,68 s'de
  `(123450, motor_mode)` değerine geçiyor. TestInfo onset yaklaşık 19 s.
- `HIL_Motor_2`: `(0,0)` ile başlıyor; yaklaşık 54,55 s'de `(123450,1)`.
  TestInfo onset yaklaşık 54 s.
- `HIL_Prop`: `(0,0)` ile başlıyor; yaklaşık 54,44 s'de `(123451,1)`.
  TestInfo onset yaklaşık 54 s.
- Dolayısıyla 1.354 adet “t=0 aktif” örüntüsü fiziksel arıza değil, parser/domain
  ayrıştırma hatasıdır.

Etkilenen paketler ve toplam uçuş:

| Paket | Uçuş |
|---|---:|
| HIL_Motor_1 | 486 |
| HIL_Motor_2 | 435 |
| HIL_Prop | 435 |
| SIL_Motor_1 | 486 |
| SIL_Motor_2 | 435 |
| SIL_Prop | 435 |
| **Toplam** | **2.712** |

SIL paketlerinde ilk control kaydı bazen uçuş başlangıcından yaklaşık 13 saniye
sonra geldiğinden audit bunları `active_from_first_sample` saymamıştır; fakat ilk
control kaydından gerçek fault komutuna kadar olan `(0,0)` bölüm yine yanlış aktif
etiketlenmiştir. Bu nedenle yalnız 1.354 HIL değil, toplam 2.712 uçuş reparsedilmelidir.

### Düzeltme uygulandı ve doğrulandı

2026-07-22'de:

- `domain_of()` parser'a import edildi ve `_control_domain()` eklendi;
- control sentinel ile parquet `domain` alanı aynı canonical domain kaynağını
  kullanacak biçimde düzeltildi;
- `TRUTH_SCHEMA_VERSION = 2` eklendi;
- state migration yalnız `^(SIL|HIL)_` kapsamındaki 2.712 canonical ID'yi
  invalidated etti;
- hyphen/underscore domain ve migration idempotency regresyon testleri eklendi;
- 2.712 uçuş yeniden parse edildi ve state `6605 complete / 0 failed` oldu.

### Uygulanan güvenli düzeltme

1. `rfly_full.contract.domain_of` fonksiyonunu parser'a import et.
2. Control sentinel ve parquet `domain` alanı için aynı canonical domain'i kullan:

```python
domain = domain_of(str(package)).upper()
control_active = _active_control(base, ulog, domain)
base["domain"] = domain
```

3. `TRUTH_SCHEMA_VERSION = 2` ekle.
4. State migration sırasında yalnız package regex'i `^(SIL|HIL)_` olan 2.712
   canonical ID'yi `completed` kümesinden çıkar. Böylece diğer 3.893 ULog boşuna
   yeniden parse edilmez.
5. `tests/test_rfly_full_v2_parser.py` içine hem `-` hem `_` ayırıcısını kapsayan
   regresyon testi ekle.
6. Parser'ı normal şekilde çalıştır; atomik parquet replace ile yalnız etkilenen
   uçuşlar yenilensin.

Tüm state dosyasını elle silmek veya bütün parsed klasörünü silmek önerilmez.

## 9. Supervised TCN refactor ve bellek smoke sonucu

Yapılan refactor:

- Eğitim pencereleri sınıf-bazlı reservoir sampling ile üretim sırasında cap edilir.
- Validation-loss pencereleri ayrıca sınırlandırılır.
- Dense validation/test skorları bütün split'i RAM'e almadan uçuş uçuş streaming
  hesaplanır.
- Development-only smoke modu eklendi: validation fold ile smoke fold ayrıdır ve
  kilitli test feature'ları okunmaz.
- İlgili alanlar: `development_smoke_fold`, `locked_test_features_read`.

Bellek benchmark:

- Klasör: `artifacts/rfly_full/v2/supervised_tcn/memory_smoke_20260721_181256/`
- Durum: complete.
- Süre: 242,42 saniye.
- Peak RSS: **1.167,27 MB**.
- Güvenlik kapısı: 4.096 MB; geçti.
- Train cap: 5.000 pencere.
- Validation-loss cap: 2.000 pencere.
- Epoch: 1.
- Validation fold: 0; development smoke fold: 1.
- `locked_test_features_read=false`.

TCN 1-epoch smoke çıktısı:

`artifacts/rfly_full/v2/supervised_tcn/run_20260721_181609/`

| Politika | Event recall | FA/saat | Median delay |
|---|---:|---:|---:|
| Critical | %11,50 | 1,10 | 52 s |
| Advisory | %53,06 | 13,88 | 22 s |

Bu sonuç yalnız bellek/akış doğrulamasıdır; `status=smoke_only` ve
`operational_claim_allowed=false`. Tek epoch performans karşılaştırması değildir.
Üstelik truth hatalı veriye dayanır. Truth düzeltilmeden uzun TCN koşusu başlatma.

## 10. Test durumu

2026-07-22 tarihinde parser düzeltmesi ve yeni regresyon testleri dahil ilgili
RflyMAD suite yeniden çalıştırıldı:

```text
33 passed in 12.57s
```

Parser/audit odaklı alt suite ayrıca `14 passed` sonucu verdi. Test failure veya
çalışan Python süreci kalmadı.

## 11. Şu anda en umut veren yaklaşım hangisi?

**Deneysel lead olarak normal-only Temporal AE en umut veren çalışmadır.** Bunun
nedeni tam development havuzunda truth schema v2 ile beş rotasyon çalışmış olması,
advisory ve critical kapılarını geçmesi ve bilinmeyen arıza fikrine supervised
modele göre daha uygun olmasıdır.

Ancak şu ayrım korunmalıdır:

- En güçlü deneysel aday: **evet**.
- Doğrulanmış operasyonel/fizibil aday: **hayır**.

Başarı ilanını engelleyen üç ana sorun:

1. Real-domain recall çok düşük ve dengesiz.
2. Wind yaklaşık 24–34 alarm/saat üretiyor.
3. Robustness sözleşmesi ve locked-test final audit protokolü henüz dondurulmadı.

TCN ise bellek açısından çalışabilir hale gelmiştir fakat performansı henüz yalnız
1-epoch, eski-truth smoke seviyesindedir. Yeni truth ile development-only 3-epoch
sanity çalıştırıldıktan sonra bilinen fault-family tespiti ve SIL/HIL → Real
transfer deneyi için ikinci ana aday olarak değerlendirilebilir.

## 12. Kesin devam sırası

1. ✅ Parser domain/sentinel hatası ve regresyon testleri düzeltildi.
2. ✅ Yalnız etkilenen 2.712 uçuş yeniden parse edildi.
3. ✅ Parse state `6605 complete / 0 failed / 0 remaining` olarak doğrulandı.
4. ✅ Truth audit yeniden çalıştırıldı.
5. ✅ Beklenen truth kontrolleri geçti:
   - `active_from_first_sample`: 1.354 → 0;
   - Motor/Prop control onset ham ULog geçişleriyle eşleşiyor;
   - missing truth yalnız 5 Real/Motor istisnası;
   - interval violation ve feature-schema ihlali sıfır.
6. ✅ Beş rotasyonlu normal Temporal AE sweep aynı seed/sözleşmeyle tekrar çalıştı.
7. ✅ Eski/yeni domain-aile karşılaştırması yapıldı. Flight-level recall neredeyse
   aynı; HIL/SIL Motor/Propeller değişimleri 0 ile -0,31 yüzde puan arasında.
8. **Sıradaki:** locked test kapalıyken Wind ve Real-domain için preregistered
   threshold/robustness deneyi yap. Paralelde TestInfo/control farklı zaman
   referanslarını bilen daha anlamlı cross-check metriği tasarla.
9. Ardından development-only, çok epoch'lu supervised TCN çalıştır; önce kısa
   3-epoch sanity, sonra gerekirse 12+ epoch. Kilitli test kapalı kalsın.
10. Model ve eşikler yalnız development verisiyle dondurulduktan sonra final audit
    protokolü ayrıca kararlaştırılsın.

## 13. Hızlı kontrol komutları

```powershell
# Çalışan süreç var mı?
Get-Process python,pythonw -ErrorAction SilentlyContinue

# Parse durumu
.venv\Scripts\python.exe -c "import json; from pathlib import Path; s=json.loads(Path('artifacts/rfly_full/v2/parse_10hz_state.json').read_text()); print(s['stop_reason'], len(s['completed']), len(s['failed']), s['remaining'])"

# İlgili testler
.venv\Scripts\python.exe -m pytest `
  tests\test_rfly_full_contract.py `
  tests\test_rfly_full_pipeline.py `
  tests\test_rfly_full_split_contract.py `
  tests\test_rfly_full_v2_parser.py `
  tests\test_rfly_full_truth_audit.py `
  tests\test_rfly_full_normal_ae.py `
  tests\test_rfly_full_supervised.py `
  tests\test_rfly_full_dl_worker.py -q

# Truth audit (reparse tamamlandıktan sonra)
.venv\Scripts\python.exe scripts\run_rfly_full_v2_truth_audit.py

# AE sweep (truth audit temizlendikten sonra)
.venv\Scripts\python.exe scripts\run_rfly_full_v2_normal_ae_sweep.py `
  --rotations 0 1 2 3 4 --epochs 25 --torch-threads 4
```

## 14. Temel artefakt dizini

- Manifest: `artifacts/rfly_full/v2/dataset_manifest.parquet`
- Parse state: `artifacts/rfly_full/v2/parse_10hz_state.json`
- Truth audit: `artifacts/rfly_full/v2/truth_audit/`
- Provisional tam AE sweep:
  `artifacts/rfly_full/v2/normal_temporal_ae/sweep_20260721_170900/`
- Truth schema v2 sonrası düzeltilmiş AE sweep:
  `artifacts/rfly_full/v2/normal_temporal_ae/sweep_20260722_093049/`
- TCN memory smoke:
  `artifacts/rfly_full/v2/supervised_tcn/memory_smoke_20260721_181256/`
- TCN smoke model çıktısı:
  `artifacts/rfly_full/v2/supervised_tcn/run_20260721_181609/`

## 15. Son durum tek cümle

RflyMAD-Full v2 truth schema v2 düzeltmesi, 2.712 uçuşluk seçici reparse, temiz
truth audit ve düzeltilmiş beş-rotasyon Temporal AE sweep'i tamamlandı; critical
development kapısı artık kararlı geçse de Real recall ve Wind alarm yükü çözülmeden,
locked test protokolü ayrıca dondurulmadan fizibilite/operasyonel başarı iddiası
kurulamaz.

## 16. Codex ilerleme planı sonuçları — 2026-07-22

### Görev A — Cross-check v2

- Legacy `truth_crosscheck_disagreement` geriye dönük kıyas için korundu.
- Yeni alanlar eklendi: eligibility, signed onset/offset delta, interval overlap,
  `truth_crosscheck_disagreement_v2` ve cross-check schema version.
- V2 kararı: `|onset delta| <= 16 saniye` ve aktif interval overlap zorunluluğu.
  Offset delta aileye göre farklı bitiş semantiği taşıdığı için ayrı raporlanıyor.
- Yalnız development parquetleri post-process edildi: `5380/5380`, failed=0,
  `locked_test_features_read=false`.
- Development audit: legacy disagreement `4544/4934`; v2 disagreement `0/4585`.
  Onset |delta| maksimumu 15,9 s. 30 s kaymalı ve örtüşmeyen sentetik örnek v2
  tarafından hâlâ disagreement olarak yakalanıyor.
- Artefakt: `artifacts/rfly_full/v2/truth_audit_development/`.

Kapsam caveat'i: Görev A başlangıcındaki başarısız bir keşif sorgusu daha önce
üretilmiş all-split audit CSV'sini açtı, ancak `split` kolon çakışmasında hiçbir
istatistik döndürmeden durdu. Tolerans seçimi, post-processing ve rapor daha sonra
yalnız development parquetleriyle yeniden üretildi; locked sonuç seçimde
kullanılmadı. Yine de literal “hiç açma” kuralı açısından bu teknik olay saklanmaz.

### Görev B — Robustness sözleşmesi

- Sözleşme: `docs/RFLYMAD_V2_ROBUSTNESS_SOZLESMESI_20260722.md`.
- Durum: `approved=true`; kullanıcı 2026-07-22'de “tamamdır, devam” diyerek onayladı.
- Real research/fizibilite kapıları, Wind ara/nihai kapıları, en fazla altı aday
  ve durdurma kuralları sonuç çalıştırılmadan önce yazıldı.
- Onay sonrasında development-only nested robustness dizisi çalıştırıldı; sonuçlar
  aşağıdaki Bölüm 17'de ve ayrı sonuç belgesinde kayıtlıdır.

### Görev C — Yeni truth ile 3-epoch TCN smoke

- Run: `artifacts/rfly_full/v2/supervised_tcn/run_20260722_111938/`.
- Memory benchmark: `memory_smoke_20260722_111203/`; peak RSS 3563,21 MB,
  4096 MB kapısı geçti.
- Zorunlu alanlar: `status=smoke_only`, `development_smoke_fold=1`,
  `locked_test_features_read=false`, `operational_claim_allowed=false`.

| Model/politika | Recall | FA/saat | Median delay |
|---|---:|---:|---:|
| TCN critical, 3 epoch | %13,68 | 2,43 | 32,5 s |
| TCN advisory, 3 epoch | %71,40 | 13,59 | 18,0 s |
| AE critical, 5-rotasyon ort. | %60,43 | 1,28 | raporlanmıyor |
| AE advisory, 5-rotasyon ort. | %69,84 | 3,64 | raporlanmıyor |

TCN advisory head anlamlı sinyal öğreniyor, fakat FA bütçesini aşıyor. Critical
head hem recall hem FA kapısını geçmiyor. Epoch 3 validation loss yükseldiği için
bu smoke, doğrudan 12+ epoch veya “en iyi model” kararı vermez.

Doğrulama: supervised suite `7 passed`; ilgili tam RflyMAD suite `37 passed`.
Uyarı yalnız `.pytest_cache` yazımıyla ilgilidir.

## 17. Onaylı Wind/Real robustness deneyleri — 2026-07-22

Sözleşme onaylandıktan sonra her rotasyonda birbirinden ayrık train, inner
kalibrasyon ve outer değerlendirme gruplarıyla beş aday çalıştırıldı. Dondurulmuş
development manifesti 5.380 uçuş içerir; tüm split değerleri `development` ve tüm
aday summary'leri `locked_test_features_read=false` taşır.

Deney kökü:
`artifacts/rfly_full/v2/normal_temporal_ae/robustness/approved_20260722_nested_v1/`

Critical beş-rotasyon ortalamaları:

| Aday | Genel recall | Tüm nonfault FA/s | Wind FA/s | Real Motor | Real Sensor | Real macro | Real-NoFault FA/s | Real alarm uçuş oranı |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | %60,43 | 1,28 | 28,46 | %20,20 | %8,35 | %14,28 | 0,80 | %2,00 |
| R1 Real threshold | %59,55 | 3,28 | 28,13 | %23,88 | %22,03 | %22,95 | 12,08 | %19,82 |
| W1 env-aware threshold | %10,78 | 2,54 | 7,85 | %23,88 | %22,03 | %22,95 | 12,08 | %19,82 |
| W2 Wind-normal training | %55,90 | 2,65 | 17,78 | %10,61 | %17,47 | %14,04 | 3,79 | %6,64 |
| R2 Real fine-tune 3 epoch | %59,46 | 3,24 | 27,83 | %23,37 | %21,77 | %22,57 | 11,68 | %19,82 |
| R3 Real fine-tune 8 epoch | %58,90 | 3,20 | 27,83 | %23,27 | %21,65 | %22,46 | 11,68 | %19,82 |
| R4 convergence fine-tune | %54,61 | 5,24 | 28,75 | %29,39 | %26,84 | %28,11 | 12,98 | %21,64 |

Karar:

- R1, R2, R3 ve kullanıcı-onaylı convergence follow-up R4 Real
  research-promotion kapısını geçmedi. En iyi Real macro ortalaması R4'te %28,11;
  gerekli eşik %40. R4 bootstrap %95 GA `%24,92–%31,44`; dış Real-NoFault FA
  12,98/saat.
- W1 Wind ortalamasını 7,85/saat'e indirdi fakat genel recall %10,78'e düştü,
  maksimum Wind 23,67/saat ve tüm nonfault FA 2,54/saat oldu; kapı geçilmedi.
- W2 daha dengeli fakat yetersiz kaldı: Wind 17,78/saat, maksimum 24,67/saat,
  genel recall %55,90 ve tüm nonfault FA 2,65/saat; kapı geçilmedi.
- Ayrı bir Real ve ayrı bir Wind adayı geçmediği için koşullu RW1 çalıştırılmadı.
- Sözleşmenin durdurma kuralı uygulandı: mevcut veri/temsil ile Real transfer
  gösterilemedi; Wind robustness çözülmedi. Yeni veri veya yeni temsil sözleşmesi
  olmadan threshold/fine-tune araması sürdürülmez.
- Convergence testi eklenmiş ilgili tam suite son çalıştırmada `44 passed` verdi;
  çalışan Python süreci ve `archive/` değişikliği yoktur.

Frozen baseline eski tek-holdout protokolündedir; yeni adaylar daha sıkı nested
inner/outer protokolündedir. Bu yüzden baseline kıyas satırı bağlam içindir ve
adaylar arasındaki kapı kararı yalnız önceden dondurulmuş kurallarla verilmiştir.
Hiçbir sonuç fizibilite veya operasyonel başarı iddiası değildir.

Ana raporlar:

- `final_summary.json`: nihai durdurma ve RW1 kararı.
- `candidate_comparison_by_policy.csv`: critical/advisory, mean/std/min/max tam tablo.
- Her aday altında `gate_summary.json`, `bootstrap_ci.json`, beş rotasyon metrikleri,
  per-flight sonuçlar ve model üreten adaylarda checkpoint/training history.
- Ayrı okunabilir rapor: `docs/RFLYMAD_V2_ROBUSTNESS_SONUCLARI_20260722.md`.

### R4 convergence ayrıntısı

Sekiz epoch'un convergence için yeterli olmadığı kullanıcı tarafından sorgulandı ve
`RFLYMAD_V2_CONVERGENCE_EK_SOZLESME_20260722.md` ile ayrı takip protokolü açıldı.
Inner Real validation early stopping sonucu best/stop epoch'ları sırasıyla
`1/13`, `780/792`, `217/229`, `1/13`, `628/640` oldu. Böylece 8 epoch'un üç
rotasyon için az, iki rotasyon için ise fazla olduğu doğrulandı.

Uzun fine-tune Real macro recall'ı R3'teki %22,46'dan %28,11'e çıkardı; fakat genel
recall %54,61'e düştü, tüm nonfault FA 5,24/saat ve Real FA 12,98/saat oldu.
Sonuç: daha uzun eğitim Real sinyalini artırıyor ama transfer/FA sorununu çözmüyor;
R4 de kapıyı geçmedi. Epoch grafikleri R4 aday klasöründedir.
Görselli tam rapor: `docs/RFLYMAD_V2_CONVERGENCE_DENEY_RAPORU_20260722.md`.
