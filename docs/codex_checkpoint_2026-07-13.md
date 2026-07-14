# Codex çalışma checkpoint'i — 2026-07-13

## Durum

2026-07-14 devam turunda Adım 6 ve Adım 7 tamamlandı. Karar kayıtları docs/decisions.md
içinde ADR-026–ADR-032 olarak bulunuyor. Adım 6 tam-hacim S2 v4 koşusu PASS, Adım 7 genel
gate kararı FAIL'dir. Çalışma kullanıcı sert durma noktasındadır.

Ana konfigürasyon dondurulmadı. Adım 8 Dense-AE/USAD başlatılmadı. Adım 9 holdout freeze
başlatılmadı; Downloads/raw/archive ve üç kör holdout tarının içeriği açılmadı.

## 2026-07-14 devam sonucu

- Deterministik 4-worker ve vektörize S2 runner gerçek 365.847 satırlık parçada yaklaşık
  1,5 saniye ölçüldü; hedefli segmentation/S2/parser testleri 54/54 geçti.
- artifacts/adsb/runs/20260714_step6_s2_natural_v4 koşusu 638/638 parça ve
  256.155.009 satırı 272,2 saniyede exit 0 ile tamamladı. Checksum indexi 2/2 doğrulandı.
- ADR-031 S2 doğal reason burden'ı kaydeder; residual/CUSUM ile birleştirme ve saldırı
  ground-truth iddiası yoktur.
- ADR-032 Adım 7 gate kararını FAIL olarak kapatır. CUSUM h=1 doğal alarm doygunluğu,
  üç NN'in magnitude-domination flag'i ve kayıp frozen features.py byte snapshot'ı ana
  konfigürasyon freeze'ini engeller.
- Corrected CUSUM evaluator hash kontrolünü gevşetmedi. Mevcut features.py yeni bir aday
  olarak baştan ön-kayıtlanmadan Step-5 frozen adayının yerine geçirilemez.

## Tamamlanan kanıtlar

- Adım 1 / ADR-026: değişmez run manifesti, fail-if-exists ve giriş/split provenance.
  Açık Silver toplamı 256.155.009 satır; eski belgeli 256.150.550 toplamıyla +4.459 fark
  çözülmemiş provenance notu olarak korunuyor.
- Adım 2 / ADR-027: truth-v2 korpusu 8.910 uçuş x (clean + 5 senaryo), toplam
  26.802.690 satır ve 646.160.578 byte. Sentetik veri eğitim/fit/calibration'a girmedi.
- Adım 3 / ADR-028: donmuş kural corrected truth-v2 üzerinde yeniden ölçüldü. Pooled
  AUROC/AUPRC 0.764883/0.883313; doğal temiz burden 4.808533 episode/saat.
- Adım 4 / ADR-029: iki eksenli causal Page CUSUM ve reset/missingness/prefix sözleşmesi.
  MAD=0 kanal floor uygulanmadan hariç tutuluyor.
- Adım 5 / ADR-030: 638 parça ve 256.155.009 satırlık tam-hacim streaming kanıtı
  tamamlandı. CUSUM h=1 skorlanabilir uçuşların yaklaşık yüzde 99'unu ve evaluable
  satırların yaklaşık yüzde 78–80'ini alarma soktuğu için ana freeze reddedildi.
  Engineering-advisory 12 episode/saat sınırı kullanıcı-onaylı operasyonel gereksinim
  değildir. Bootstrap upper doğruluk denetimi seçimi değiştirmedi.
- CUSUM truth-v2 değerlendirme kodu hazır ve bağımsız incelemeden PASS aldı. Fit,
  calibration, threshold sweep veya fusion yapmıyor; tek clean negatif havuzu, corrupt q0
  dışlama ve doğal-burden eşlemesi fail-closed.
- S2 kodunda bulunan iki blocker kapatıldı: state episode'ları explicit inactive satırda
  bölünüyor fakat sparse cadence tek başına bölmüyor; MESSAGE_GAP her satırda point event.
  Step 6 kod hash zinciri artık adsb/run_manifest.py dosyasını da kapsıyor. Kök doğrulamada
  S2/parser için 39 test, CUSUM truth-v2 kapsamı için 34 test geçti.

## Tamamlanmamış Adım 6 koşusu

İlk tam-hacim deneme dizini:

artifacts/adsb/runs/20260713_step6_s2_natural_v1

Koşu kullanıcı zaman önceliği nedeniyle yaklaşık 14 dakika sonra kontrollü durduruldu.
run_manifest.json yazılmıştı fakat final S2 raporu ve checksum indexi üretilmedi. Bu namespace
yeniden kullanılmamalı ve bilimsel sonuç sayılmamalıdır; yanında INCOMPLETE_DO_NOT_USE.md
işareti bulunur.

Ölçülen çalışma davranışı: 20 mantıksal işlemciden tek çekirdek kullanıldı; süreç canlıydı,
yaklaşık 814 CPU-s tüketmiş ve yaklaşık 906 MB working set kullanmıştı. İlk 50/638 ilerleme
satırı henüz gelmemişti. Hata veya veri sözleşmesi ihlali gözlenmedi; durdurma nedeni yalnız
duvar-saatiydi.

## Devam sırası

1. Kullanıcı Adım 7 FAIL kararını ve yeni çalışma yönünü onaylamalıdır.
2. Eski frozen features.py byte snapshot'ı bulunabiliyorsa corrected CUSUM değerlendirmesi
   aynı Step-5 adayı için tamamlanabilir.
3. Snapshot bulunamıyorsa mevcut code version yalnız yeni bir aday olarak; yeni ön-kayıtlı
   operasyonel burden bütçesi, natural calibration ve yeni namespace ile baştan çalıştırılabilir.
4. Kullanıcı onayı olmadan ana konfigürasyonu dondurma, Adım 8'e veya Adım 9'a geçme.

## Değişmez kısıt hatırlatması

- Sonuç görüldükten sonra aynı run içinde parametre/eşik ayarı yok.
- Sentetik veri train/fit/calibration'a girmez.
- archive içinden kod kopyalanmaz veya import edilmez.
- MAD=0 kanal floor'lanmaz; hariç tutulur.
- Sentetik recall doğal burden yanında raporlanır.
- Satır, event, uçuş ve uçuş-saati birimleri karıştırılmaz.
- Rehearsal geri-beslemesi ve holdout seçimi yapılmaz.
- Üç holdout tarı tek havuzdur; freeze ve unseal ayrı kullanıcı kararlarıdır.
- Commit mesajına Co-Authored-By eklenmez.

## Git yayın checkpoint'i

Kaynak/test/dokümantasyon kapsamı agent/adsb-rule-cusum-checkpoint dalında
26b0225 adsb rule cusum evidence checkpoint commit'i olarak push edildi:

origin/agent/adsb-rule-cusum-checkpoint

528 MB'lık generated baseline raporu ve generated parse logu Git indexinden çıkarıldı;
.gitignore ile gelecekte yeniden stage edilmeleri engellendi, yerel dosyalar silinmedi.
Generated ADS-B run/model/plot çıktıları da ignore kapsamındadır. Kullanıcı yalnız commit+push
istediği için draft PR açılmadı. Branch, origin/main'in önceki durumundan türetilmiştir;
main'deki 10 yeni commit ile rebase/PR senkronizasyonu sonraki yayın adımıdır.
