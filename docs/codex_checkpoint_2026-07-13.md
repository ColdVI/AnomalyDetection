# Codex çalışma checkpoint'i — 2026-07-13

## Durum

Çalışma kullanıcı isteğiyle güvenli biçimde duraklatıldı. Kaldığı yerden devam edilebilir.
Adım 1–5 tamamlandı ve karar kayıtları docs/decisions.md içinde ADR-026–ADR-030 olarak
bulunuyor. Adım 6 kodu ve testleri hazır; tam-hacim ölçümü tamamlanmadı. Adım 7 gate
incelemesi bu nedenle henüz kapanmadı.

Ana konfigürasyon dondurulmadı. Adım 8 Dense-AE/USAD başlatılmadı. Adım 9 holdout freeze
başlatılmadı; Downloads/raw/archive ve üç kör holdout tarının içeriği açılmadı.

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

1. S2 runner'a yalnız yürütme düzeyinde deterministik process paralelliği ekle. Bilimsel
   config, reason-code, eventizer, split, threshold veya denominator değişmemeli. Önerilen
   güvenlik sözleşmesi: sabit input sırası, worker başına tek Parquet parçası, parent'ta
   deterministik merge ve source ownership kontrolü, bir ve çok worker sonuç-eşitliği testi,
   bounded worker sayısı ve başlangıç/bitiş code hash guard'ı.
2. Yeni ve mevcut olmayan bir namespace kullan; v1'i silme veya üzerine yazma. Önerilen ad:
   artifacts/adsb/runs/20260713_step6_s2_natural_v2
3. Önce S2/parser regresyonlarını çalıştır; sonra Step 5 manifestindeki aynı 638 açık Silver
   girdisi üzerinde tam-hacim Step 6 koşusunu tamamla. Sonuçtan sonra ADR-031 yaz.
4. Hazır donmuş CUSUM truth-v2 değerlendirmesini şu yeni namespace ile çalıştır:
   .venv/Scripts/python.exe scripts/adsb_evaluate_cusum_truth_v2.py --run-dir
   artifacts/adsb/runs/20260713_step7_cusum_truth_v2_v1
5. Artefakt hash/footer/checksum ve doğal-burden eşlemesini doğrula.
6. Adım 7 gate incelemesini tamamla ve ADR-032 yaz. Mevcut doğal doygunluk nedeniyle başlangıç
   önerisi FAIL / ana konfigürasyonu dondurma yönündedir; eksik corrected CUSUM ve S2 kanıtı
   görülmeden nihai metin yazılmamalı.
7. Adım 7 sonunda sert durma noktasında kullanıcıya dön. Onay olmadan konfigürasyonu dondurma,
   Adım 8'e veya Adım 9'a geçme.

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

Çalışma ağacı main dalında ve origin/main'in 10 commit gerisindedir. GitHub CLI bu makinede
kurulu değildir. Ayrıca izlenen artifacts/adsb/models/baseline_training_report.json dosyası
yaklaşık 528 MB'dır ve normal GitHub blob sınırını aşar. Publish öncesinde branch/senkronizasyon
ve büyük-artefakt stratejisi bilinçli biçimde çözülmelidir; bu dosya sessizce silinmemeli veya
geri alınmamalıdır.
