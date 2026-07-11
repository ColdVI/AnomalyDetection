# ML-15: Kayma-Farkındalıklı FA Kalibrasyonu Planı

Durum: ÖN-KAYIT (2026-07-07, sonuçlar görülmeden sabitlendi).
Üst plan: `docs/ML14_MASTER_IYILESTIRME_PLANI.md` (Kaldıraç 2).
Bağımlılık: **tam koşu ML-14 yeniden inşası bitmeden yapılamaz** (yeni split
manifest + yeni feature tablosu gerekir); modül + birim testler ÖNCE yazılabilir.

## §0 Problem (ölçülmüş)

Val-normal akışlarında B bütçesine kalibre edilen her policy, test'te sistematik
olarak ~2× FA üretiyor (ML-10: 12→23.9; ML-12: 12→23.7; ML-13 kanal bazında da
aynı). Kök neden: val'in temsil ettiği normal çeşitliliği testtekinden dar
(heterojen-normal, D.1). ML-14 çeşitliliği büyütüyor; ML-15 kalan kaymayı
ÖLÇÜP telafi ediyor. Bu, normali homojenleştirme DEĞİL; çalışma noktasının
dürüst kalibrasyonudur — eşiği "val'de B" yerine "test'te B" hedefine kurar.

## §1 Yöntem (SABİT): oturum-jackknife kayma tahmini + bütçe düzeltmesi

Yeni modül `src/ml/decision/drift_calibration.py` (karar katmanlarını
DEĞİŞTİRMEZ, sarar):

```
fit_drift_corrected_policy(val_streams_by_session, budget, decision_fit_fn, seed)
```

1. Val normal uçuşları oturuma göre gruplanır (`session_of`). ML-14 kotası
   (≈%15) sayesinde val artık çok-oturumlu.
2. Her val oturumu s için: policy val−s akışlarında B bütçesiyle fit edilir,
   s akışlarında gerçekleşen FA ölçülür → r_s = FA_s / B.
3. Kayma çarpanı **D̂ = quantile_0.75({r_s})**, taban 1.0, tavan 5.0 (guard).
4. Nihai policy TÜM val'de **B_eff = B / D̂** ile fit edilir; D̂, {r_s} listesi
   ve B_eff rapora yazılır.
5. **Fallback (ön-kayıtlı):** val oturum sayısı < 4 ise jackknife yapılmaz;
   D̂ = ML-14 raporundaki havuzlanmış (kaynak×bütçe CUSUM) medyan kayma oranı
   kullanılır ve fallback kullanımı raporda işaretlenir.

Parametreler SABİT: q=0.75, floor=1.0, cap=5.0, min_sessions=4. Sonuç görülüp
q/floor/cap OYNANMAZ.

## §2 Değerlendirme protokolü

`scripts/run_ml15_calibrated_evaluation.py` — ML-14 runner'ının aynı yeni-dönem
splitleri üzerinde, TEK farkla: policy fit'i `fit_drift_corrected_policy`
sarmalayıcısından geçer.

- Skor kaynakları (SABİT, ML-14 ile aynı): `existing_fusion`, `itki_komutu`,
  `ml14_fusion`.
- 3 karar tipi × {critical: 2, advisory: 12}; 5 seed; smoke `--splits split_00`.
- ML-14'ün düzeltmesiz satırları yan yana raporlanır (aynı CSV şemasında
  `calibration ∈ {none, drift_corrected}` kolonu) — recall maliyeti görünür olur.
- Artifact: `artifacts/ml15/uav_sead/<run>/` (CSV'ler + gates.json + split başına
  policies + D̂ raporları + checksum'lı manifest + development-id-hash +
  `blind_holdout_read: false`).

## §3 Gate tanımları (SABİT)

- **Gate A (zorunlu):** holdout okunmadı; donmuş dizinler değişmedi; karar
  katmanı fonksiyonları değişmeden import (identity test); sarmalayıcının
  döndürdüğü policy nesneleri MEVCUT sınıflardan (Threshold/KOfN/Cusum —
  yeni karar mekaniği yok, test assert eder); jackknife deterministik.
- **Gate B (kalibrasyon başarısı — fazın asıl sorusu):** her
  (kaynak, CUSUM, bütçe) hücresinde 5-seed **medyan test FA ≤ 1.00 × bütçe**
  VE **≥4/5 seed ≤ 1.25 × bütçe**. En az {existing_fusion, itki_komutu} × 
  {critical, advisory} = 4 hücrenin 3'ünde sağlanırsa GEÇTİ. (Diğer karar
  tipleri bilgi amaçlı raporlanır.)
- **Gate C (operasyonel, değişmez):** düzeltilmiş satırlar arasında herhangi
  biri critical ≥0.30 recall @ ≤2 FA-saat VEYA advisory ≥0.50 @ ≤12 sağlarsa
  GEÇTİ. Ek dürüstlük raporu: aynı hücrede düzeltmesiz vs düzeltilmiş recall
  (kalibrasyonun recall maliyeti gizlenmez).

Gate C geçerse: ML-17 endgame önerisi hazırlanır (holdout HÂLÂ açılmaz —
o kullanıcı onaylı ayrı adım). Geçmezse: "bu veri + bu skor ailesiyle uyumlu
FA'da ulaşılabilir recall haritası" dürüst çıktı olarak yazılır ve ML-16'ya
geçilir.

## §4 Dosyalar, testler, kabul

| Dosya | İş |
|---|---|
| `src/ml/decision/drift_calibration.py` (yeni) | §1 sarmalayıcı; saf, birim-testli |
| `scripts/run_ml15_calibrated_evaluation.py` (yeni) | §2 protokol |
| `tests/test_ml15.py` (yeni) | jackknife determinizmi; floor/cap/fallback yolları; policy-sınıf identity; sentetik akışta "D̂ doğru yönde düzeltiyor" testi; manifest holdout-hash + checksum |

Kabul: birim testler + smoke + tam 5-seed; Gate sayıları ham CSV'den bağımsız
türetilebilir; tam pytest yeşil (bilinen 4 MinIO hariç); bulgular H33+ /
ADR-015 / yetersizlikler kaydı güncellemesi; commit'ler co-author'suz.
