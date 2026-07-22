# RflyMAD-Full v2 — Codex İlerleme Planı

> **2026-07-22 durum notu:** A, B ve C tamamlandı; ardından kullanıcı onayıyla
> AE robustness/convergence ve TCN development-only 5-fold takipleri de
> tamamlandı. Güncel sonuç için `RFLYMAD_V2_SONRAKI_ADIMLAR_20260722.md` okunmalı.

> Yazıldı: 2026-07-22 (Europe/Istanbul)
> Hedef okuyucu: bu depoyu ilk kez açan bir kodlama ajanı (Codex).
> Kapsam: sadece aşağıdaki 3 görev. Başka bir şeye dokunma.

## 0. Önce oku

Bu dosyadan önce `docs/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md` dosyasının
tamamını, özellikle Bölüm 0 ve Bölüm 2'yi oku. Orada yazan mevcut-durum ve
sınır bilgisini burada tekrar etmiyorum, sadece özetliyorum:

- 6.605 uçuşluk `rfly_full/` v2 hattı parse+truth-audit açısından temiz.
- Düzeltilmiş 5-rotasyon normal-AE sweep tamamlandı; critical politika kapıyı
  geçiyor ama bu **ölçüm düzeltmesinden** geliyor, model iyileşmesinden değil.
- Real-domain recall çok zayıf, Wind FA çok yüksek — **hiçbiri çözülmedi**.
- TCN, yeni (düzeltilmiş) truth ile hiç çalıştırılmadı.
- Commit/push yapılmadı; çalışma ağacı kirli/untracked.

## 1. Sabit kurallar (ihlal etme)

- `archive/` salt-okunur. Oradan kod import etme.
- Kilitli test (`split == "locked_test"`) bu üç görevin hiçbirinde okunmaz,
  filtrelenmez, özet istatistiği bile alınmaz. Sadece development split
  kullanılır.
- Hiçbir görev "operasyonel/fizibil model" iddiası üretmez. Çıktılara
  `status` ve `operational_claim_allowed` alanlarını dürüstçe yaz
  (bkz. `rfly_full/supervised.py:625-636` deki örnek desen).
- Wind bir sistem arızası değildir, ayrı bir `environment_robustness`
  problemidir; recall/FA tablolarında sistem arızalarıyla karıştırma.
- SIL/HIL sonucu Real-domain başarısı olarak sunulmaz.
- Kullanıcı açıkça istemeden commit/push yapma. Görev sonunda diff'i ve
  yeni dosyaları listele, commit'i kullanıcıya bırak.
- Yeni script/modül eklerken mevcut `rfly_full/` namespace'ini kullan,
  paralel bir yapı kurma.
- Her görev bittiğinde ilgili pytest suite'ini çalıştır ve sonucu raporla.

Sanal ortam: `.venv\Scripts\python.exe` (Windows/PowerShell).

## 2. Görev A — Cross-check metriğini düzelt

**Neden:** `truth_crosscheck_disagreement` şu an 5.577/6.069 uçuşta `True`
çıkıyor ve bu bir "kırmızı bayrak" gibi görünüyor. Ama ham ULog incelemesi şunu
gösterdi: SIL'de kontrol topic'i uçuşun yerel eksenine göre ~13-14 saniye geç
başlıyor, ayrıca TestInfo ile kontrol sinyalinin "bitiş" tanımları farklı.
Mevcut metrik bu zaman kaymasını hesaba katmadığı için gerçek bir parser
hatası yokken bile "uyuşmazlık" işaretliyor. Bu metrik gerçek hataları
maskeleyebilir çünkü şu an neredeyse her uçuş zaten "uyuşmazlık" diyor.

**Mevcut kod:** `rfly_full/v2_parser.py:225-228`

```python
base["truth_crosscheck_disagreement"] = bool(
    control_active is not None and control_active.any() and published is not None
    and np.mean(control_active != planned) > 0.01
)
```

**Yapılacak:**
1. `control_active` (kontrol sinyali kaynaklı aktif aralık) ile `planned`
   (TestInfo kaynaklı aktif aralık) arasında ham örnek-örnek eşitlik yerine,
   zaman kayması toleranslı bir karşılaştırma tasarla — örneğin onset/offset
   zaman farkı bir tolerans penceresinin (örn. ±15 s, ham ULog kanıtına göre
   kalibre et) içindeyse "uyuşuyor" say. Sabit toleransı büyütme; onset ve
   offset farkını ayrı ayrı hesaplayıp rapora yaz, sadece nihai boolean'ı
   gevşetme — böylece hâlâ gerçek büyük sapmaları yakalayabilirsin.
2. Yeni alanı eski `truth_crosscheck_disagreement` alanının **yanına** ekle
   (örn. `truth_crosscheck_onset_delta_s`, `truth_crosscheck_disagreement_v2`);
   eski alanı silme, geriye dönük kıyas için sakla.
3. `rfly_full/truth_audit.py` içindeki raporlama kısmını yeni alanı da
   gösterecek şekilde güncelle (`artifacts/rfly_full/v2/truth_audit/`).
4. `tests/test_rfly_full_v2_parser.py` ve `tests/test_rfly_full_truth_audit.py`
   içine: (a) küçük zaman kaymalı ama aynı mantıksal aralığı temsil eden bir
   sentetik örnekte yeni metriğin `disagreement=False` demesi gerektiğini,
   (b) gerçekten uyuşmayan (örn. tamamen farklı fault_family/onset) bir
   sentetik örnekte hâlâ `True` demesi gerektiğini doğrulayan regresyon
   testleri ekle.
5. Parser'ı sadece etkilenen alanı yeniden hesaplayacak şekilde çalıştır
   (tam reparse gerekmeyebilir — önce `v2_parser.py` içinde bu alanın
   ayrı bir post-processing adımı olarak çıkarılıp çıkarılamayacağını
   değerlendir; olmuyorsa mevcut reparse akışını kullan).
6. `scripts\run_rfly_full_v2_truth_audit.py` çalıştır, yeni disagreement
   sayısını `truth_audit_summary.md`'ye yaz.

**Bitti sayılma koşulu:** yeni metrik ile disagreement oranı önemli ölçüde
düşüyor (beklenti: çoğunluk artık `False`) VE hâlâ gerçek sapmaları
(sentetik testte) yakalıyor VE ilgili testler geçiyor.

## 3. Görev B — Wind/Real robustness deney sözleşmesini preregister et

**Neden:** Sonuca bakıp sonra "başarı kriteri buymuş" demek (post-hoc
rasyonalizasyon) yöntemsel olarak geçersizdir. Kriterler sonuçları görmeden
önce yazılı olarak dondurulmalı.

**Yapılacak — yeni dosya:** `docs/RFLYMAD_V2_ROBUSTNESS_SOZLESMESI_<tarih>.md`

İçermesi gerekenler (kilitli teste **hiç bakmadan**, sadece development
split ve mevcut Bölüm 7/kapı tanımlarına dayanarak yaz):

1. **Real-domain için ayrı başarı kapısı.** Şu an genel kapı (`recall>=%50
   critical`, `FA<=2/saat`) tüm domain'leri havuzda karıştırıyor ve Real
   zaten çok az örnekle (51 normal, 245 Motor, 197 Sensor) eziliyor. Real
   için ayrı, daha küçük örneklem büyüklüğüne uygun bir recall/FA hedefi
   ve bunun altında kalırsa ne yapılacağı (örn. "Real-only fine-tune
   denenecek" / "simulation-to-real transfer protokolü B'ye geçilecek")
   önceden yazılsın.
2. **Wind için ayrı kapı.** Wind bir arıza değil, bir stres testi;
   "Wind FA <= X/saat" hedefi ve X'in nasıl seçildiği (mevcut 28-31/saat'e
   göre gerçekçi bir ara hedef mi, yoksa nihai hedef mi) yazılsın.
3. **Hangi development-only deneylerin çalıştırılacağı, hangi sırayla.**
   Örnek adaylar: (a) domain-bazlı ayrı eşik kalibrasyonu (Real için ayrı
   quantile), (b) Wind örneklerini training/threshold kalibrasyonuna dahil
   etme (an environment-aware baseline), (c) SIL/HIL→Real fine-tune.
   Her biri için "hangi metrik iyileşirse bu adayı ana hatta taşırız"
   önceden yazılsın.
4. **Durdurma/karar kuralı.** Kaç development-only deneme yapılacağı ve
   hiçbiri hedefi tutturmazsa ne olacağı (örn. "Real-domain iddiası proje
   kapsamından şimdilik çıkarılır ve bu açıkça raporlanır") yazılsın —
   hedefe ulaşana kadar sınırsız deneme/threshold avcılığı yapılmayacağının
   güvencesi budur.

**Bitti sayılma koşulu:** sözleşme dosyası yazıldı, kullanıcı onayına
sunuldu (bu görev kod değişikliği değildir, sadece dokümandır — kullanıcı
onaylamadan sözleşmedeki deneyleri Görev B kapsamında **çalıştırma**).

## 4. Görev C — TCN development-only 3-epoch sanity koşusu

**Neden:** Mevcut tek TCN sonucu (`run_20260721_181609`, critical recall
%11,5) eski/hatalı truth'a dayanıyor ve tek epoch'luk bellek/akış
doğrulamasıydı. Yeni düzeltilmiş truth ile TCN'in en azından makul bir
sinyal öğrenip öğrenmediğine dair development-only, kısa bir sağlık
kontrolü gerekiyor — 12+ epoch'luk uzun/pahalı koşuya geçmeden önce.

**Komut** (`development-smoke-fold` ile kilitli test hiç okunmaz):

```powershell
.venv\Scripts\python.exe scripts\run_rfly_full_v2_supervised.py `
  --validation-fold 0 --development-smoke-fold 1 --epochs 3 `
  --max-train-windows 50000 --max-val-windows 20000
```

**Yapılacak:**
1. Yukarıdaki komutu çalıştır, çıktı klasörünü not et
   (`artifacts/rfly_full/v2/supervised_tcn/run_<timestamp>/`).
2. `summary.json` içindeki `status` alanının `smoke_only` olduğunu ve
   `locked_test_features_read=false`, `development_smoke_fold=1` olduğunu
   doğrula — biri bile yanlışsa dur ve raporla, sonucu kullanma.
3. Critical/Advisory recall ve FA/saat değerlerini normal-AE sweep
   sonuçlarıyla (Bölüm 0'daki tablo) yan yana koy; TCN'in AE'ye kıyasla
   makul bir sinyal taşıyıp taşımadığını (rastgeleden anlamlı şekilde
   farklı mı) değerlendir — "en iyi model" ilanı yapma, sadece "3-epoch'ta
   sinyal var/yok" tespiti yap.
4. Peak RSS'in daha önceki bellek smoke koşusundaki (~1.167 MB) gibi
   kapı-içi kaldığını doğrula.
5. Sonucu `docs/RFLYMAD_V2_YENI_CHAT_HANDOFF_20260722.md` tarzında kısa bir
   ek not olarak (yeni bölüm veya yeni dosya, mevcut handoff'u ELLE
   SİLME/ÜZERİNE YAZMA) raporla.

**Bitti sayılma koşulu:** koşu tamamlandı, `status=smoke_only` doğrulandı,
sonuç sayıları ve yorum yazıldı, ilgili testler
(`tests\test_rfly_full_supervised.py`) hâlâ geçiyor.

## 5. Görev sırası ve bağımlılıklar

1. Görev A (cross-check düzeltmesi) önce — bağımsız, veri kalitesini
   etkiler.
2. Görev B (sözleşme) A ile paralel yazılabilir — kod değişikliği gerektirmez.
3. Görev C (TCN sanity), A tamamlandıktan SONRA başlatılmalı çünkü A truth
   audit çıktısını değiştirebilir (reparse gerekiyorsa). A reparse
   gerektirmiyorsa (sadece rapor alanı ekliyorsa) C, A'yı beklemeden de
   başlayabilir — ama A'nın truth verisini bozmadığından emin ol.
4. Görev B'deki deneyler (fine-tune, ayrı eşik vb.) bu planın kapsamı
   DIŞINDA — kullanıcı sözleşmeyi onayladıktan sonra ayrı bir görev olarak
   başlatılmalı.

## 6. Her görev sonunda çalıştırılacak doğrulama

```powershell
.venv\Scripts\python.exe -m pytest `
  tests\test_rfly_full_contract.py `
  tests\test_rfly_full_pipeline.py `
  tests\test_rfly_full_split_contract.py `
  tests\test_rfly_full_v2_parser.py `
  tests\test_rfly_full_truth_audit.py `
  tests\test_rfly_full_normal_ae.py `
  tests\test_rfly_full_supervised.py `
  tests\test_rfly_full_dl_worker.py -q

Get-Process python,pythonw -ErrorAction SilentlyContinue
```

Süreç bitmeden hiçbir arka plan python süreci bırakılmamalı; bırakıldıysa
raporda açıkça belirtilmeli.

## 7. Raporlama formatı

Her görev bitiminde kullanıcıya (Türkçe, jargon minimum):
- ne değişti (dosya + kısa neden),
- yeni sayılar varsa eski sayılarla yan yana,
- hangi kapının geçtiği/geçmediği,
- "operasyonel iddia" içerip içermediği (genelde hayır),
- sıradaki önerilen adım.

Bu üç görevin hiçbiri tek başına "model çalışıyor" veya "proje bitti"
sonucuna varmaz. Amaç, Bölüm 11'deki (`YENI_CHAT_HANDOFF`) üç açık sorunu
(Real recall, Wind FA, robustness sözleşmesi eksikliği) sıraya koyup
metodik biçimde küçültmektir.
