# ML-14 Uygulama Planı — SEAD Veri Yenileme + Yeniden İnşa + D-Gate Ölçümü

Durum: ÖN-KAYIT (2026-07-07). Üst plan: `docs/ML14_MASTER_IYILESTIRME_PLANI.md` §3
(teşhis ve D1/D2/D3 gate tanımları orada; bu doküman UYGULAMA ayrıntısıdır).
Veri çekimi ayrıca başlatıldı (aşağıda §1 yalnız doğrulama yapar, indirme yapmaz).

Bu turun tek sorusu: **normal-sınıf oturum çeşitliliğini ~2× büyütmek (64→~137
oturum) val→test FA kaymasını düşürüyor mu ve Gate C nihayet geçiyor mu?**

---

## §0 Değişmezler (SÖZLEŞME — ihlal = faz geçersiz)

1. **Blind holdout asla değerlendirilmez/incelenmez.** Eski 131-uçuş holdout'un
   HİÇBİR üyesi yeni development'a giremez (aşağıda §2 donmuş-holdout kuralı).
2. **Donmuş tarih dizinlerine dokunulmaz:** `artifacts/ml8a|ml9|ml10|ml12|ml13/**`
   ve `artifacts/models/**`, `artifacts/viz/**` aynen kalır (belgelenmiş geçmiş).
   ML-14 çıktıları yalnız `artifacts/ml14/**` + yeniden üretilen veri katmanları
   (`data/objectstore/silver/uav_sead`, `data/silver/uav_sead_*`, Gold/feature
   tabloları, `split_manifest.json`, `artifacts/scalers/uav_sead_*`,
   `artifacts/cusum/uav_sead_*`).
3. **ALFA / UAV Attack veri katmanlarına dokunulmaz.** Silver/Gold/feature yeniden
   üretimi YALNIZ uav_sead kapsamındadır (`build_features` çağrısında diğer
   kaynakların tabloları değişmeden kalmalı — parquet baytları bile; gerekirse
   uav_sead-dışı yazımları atlayan bir yol ekle). `split_manifest.json` tek global
   dosya olduğu için yeniden yazılır; ALFA/UAT split İÇERİKLERİ birebir aynı
   kalmalı (girdi tabloları değişmediği için deterministik — testle assert et).
4. Karar katmanları / event_metrics / score_fusion DEĞİŞMEZ, import edilir
   (identity testleri). Model hiperparametreleri ML-9 ile aynı
   (IsolationForest n_estimators=300, max_samples=256, random_state=seed).
5. Sonuç görüldükten sonra hiçbir eşik/kota/gate tanımı değiştirilmez.
6. Commit'lerde Co-Authored-By YOK; mantıksal commit'lere böl.

---

## §1 Adım 0 — Bronze tamlık doğrulaması (indirme BİTTİKTEN sonra)

İndirme ayrı süreçte koşuyor (`--normal 900 --per-class 200 --ext-pos 193`,
skip-existing). Bittiğinde:

1. `labels.json`'u oku; sınıf başına sayıları raporla. Beklenen (404'ler hariç):
   normal ≈ 897-900, external_position ≈ 191-193, altitude 73, mechanical 41,
   global_position 40. (Geçmişte 2 normal + 1 uçuş kalıcı 404; bu koşuda en az
   1 yeni 404 görüldü: `2018-07-04/19_38_27`.)
2. Kalıcı indirilemeyen uçuşların TAM listesini
   `artifacts/ml14/uav_sead/bronze_refresh_report.json`'a yaz
   (sınıf sayıları, oturum sayıları, 404 listesi, toplam bayt).
3. Bronze'daki .ulg sayısı == labels.json uçuş sayısı assert'i.
4. Eksik kritik sınıf yoksa devam; normal < 850 ise DUR ve raporla (indirme
   yarım kalmış olabilir — downloader'ı yeniden koşmak güvenli, skip-existing).

## §2 Donmuş-holdout genişletmesi (EN KRİTİK PARÇA)

**Problem:** `make_group_split`, holdout'u anomali-oturumlarının %30'unu
`holdout_seed` ile karıp seçiyor. ExtPos havuzu 60→193 büyüyünce oturum listesi
değişir → aynı seed FARKLI bir %30 seçer → eski holdout uçuşları development'a,
eski development uçuşları (ML-6..13 boyunca defalarca değerlendirildi!) holdout'a
düşer → **yeni holdout kontamine olur, körlük yanar.**

**Çözüm — artımlı donmuş holdout:** `src/ml/data/splits.py`'ye opsiyonel parametre:

```python
build_split_manifest(..., frozen_holdout={"uav_sead": <eski manifestteki
    splits.split_00.final_holdout listesi>})
```

Semantik (uav_sead için):
1. `frozen_sessions = {session_of(f) for f in eski_final_holdout}` — bu oturumlar
   KOŞULSUZ holdout'tur (yeni indirilen uçuşları dahil: aynı oturuma yeni normal/
   anomali geldiyse o da holdout'a gider — hiç görülmemiş oldukları için körlüğü
   bozmaz, tersine korur).
2. Yalnız YENİ anomali oturumları (eski manifestte hiç uçuşu olmayan oturumlar)
   üzerinde `_anomaly_dev_holdout` mantığı aynı `holdout_seed=20260703` ile koşar
   → yeni-oturumların ~%30'u holdout'a eklenir.
3. `final_holdout = frozen_sessions uçuşları ∪ yeni-seçilen oturun uçuşları`;
   kalan her şey development havuzu.
4. **Zorunlu assert'ler** (fonksiyon içinde, testte de):
   - eski_final_holdout ⊆ yeni_final_holdout (körlük korunur),
   - eski_development ∩ yeni_final_holdout == ∅ (kontaminasyon yok),
   - yeni_final_holdout ∩ yeni_development == ∅.

Eski manifest, yeniden üretimden ÖNCE
`artifacts/ml14/uav_sead/previous_split_manifest.json` olarak kopyalanır (hem
girdi hem denetim izi; sha256'sı rapora yazılır).

Unit testler (`tests/test_ml14.py`): sentetik mini-manifest ile (a) frozen
oturunlar korunuyor, (b) eski-dev → holdout sızıntısı imkânsız, (c) yeni
oturunlar %30 kuralıyla bölünüyor, (d) frozen_holdout verilmeyince davranış
birebir eski (regresyon yok — mevcut split testleri geçmeye devam etmeli).

## §3 Yeniden inşa sırası (uav_sead kapsamlı)

1. **Silver temizliği (H8.1/F.2 tuzağı):** `data/objectstore/silver/uav_sead`
   TAMAMEN silinir, sonra `python -m src.silver.parse_uav_sead` (parse coverage
   loglanır; 1247'ye yakın uçuşun kaçı parse edildi → rapora).
2. **Feature/split/scaler/CUSUM:** `python -m src.ml.build_features` — ama §0.3
   gereği uav_sead-dışı tabloları YAZMADAN. Mevcut CLI'da yalnız
   `--skip-uav-sead` var; tersi (yalnız-uav_sead) için küçük bir bayrak ekle
   (`--only-uav-sead` gibi) — ALFA/UAT parquet'lerini yeniden yazmasın.
3. **Split kotası (ÖN-KAYIT):** FA kalibrasyon varyansı bu projenin ana duvarı
   olduğu için val artık normal çeşitliliğini taşımalı:
   `n_val = n_test_normal = max(30, round(0.15 × development_normal_uçuş_sayısı))`
   — sayı Adım 0/D1 sayımından hesaplanır, `SPLIT_QUOTAS["uav_sead"]`'e yorumla
   yazılır ve rapora kaydedilir. (Oturum-bazlı doldurma zaten mevcut mantık.)
4. Yeni `split_manifest.json` + `uav_sead_robust_scaler.json` +
   `uav_sead_cusum_baseline.json` üretimi (hepsi split_00 train-normal'den —
   mevcut kurulu davranış).

## §4 Gate D1 — Veri kalitesi raporu

`artifacts/ml14/uav_sead/rebuild_report.json` (checksum'lı):
- parse coverage (kaç uçuş girdi/çıktı, atlananlar listesi),
- part-çoğalması kontrolü: feature tablosunda source_id başına satırlar TEK
  üretimden mi (ör. silver part dosya sayısı == 1 koşu; ve/veya uçuş başına
  satır sayısının eski tabloya göre ~aynı ölçekte olduğu örneklem kontrolü),
- development normal oturum sayısı (beklenti: 64 → ≥110) ve uçuş sayıları,
- yeni holdout boyutu ve §2 assert'lerinin çıktıları,
- ALFA/UAT split içerik-eşitliği kanıtı (eski manifest vs yeni: alfa/uav_attack
  alt-ağaçlarının JSON eşitliği),
- yeni feature tablosu satır/uçuş sayısı + sha256.

**D1 geçmezse dur.**

## §5 Gate D2/D3 — Ölçüm runner'ı (`scripts/run_ml14_enrichment_evaluation.py`)

ML-9 protokolünün yeni manifest üzerinde yeniden koşumu + ince modül:

1. Split başına: scaler'ı train-normal'de fit et (`fit_scaler_params` — ML-9
   runner kalıbı), `PX4_ML7_CANDIDATE_MODULES` + `PX4_ML12_THIN_MODULES`
   (yalnız `itki_komutu`) modüllerini train'de fit et, val'e kalibre et
   (`_score_modules` yeniden kullan), 1 sn akışlara indir.
2. Skor kaynakları (SABİT): `existing_fusion`, `itki_komutu`,
   `ml14_fusion = max(existing_fusion, itki_komutu)`.
3. Politikalar: threshold / k_of_n / cusum × {critical:2, advisory:12} —
   `_fit_policies` import (identity test).
4. Değerlendirme `_evaluate` import; metrics/flight_label/category CSV'leri +
   split başına policies.json + modeller (`models/*.joblib`) yazılır.
5. **Gate D2 (kayma):** her (score_source ∈ {existing_fusion, itki_komutu},
   budget) için CUSUM satırlarında kayma oranı r = test_FA_saat / bütçe.
   ESKİ referans: donmuş `artifacts/ml9/.../metrics.csv` (existing_fusion) ve
   `artifacts/ml12/.../metrics.csv` (itki_komutu) CUSUM satırları (checksum
   doğrulamalı okunur, yeniden hesaplanmaz). KURAL (sabit): 4 hücrenin
   {2 kaynak × 2 bütçe} havuzlanmış medyan r'si, eskiye göre **≥%15 göreli
   düşerse D2 GEÇTİ**. Hücre bazında da raporlanır; diğer karar tipleri bilgi
   amaçlı.
6. **Gate D3 (operasyonel):** yeni matriste HERHANGİ bir satır
   (3 kaynak × 3 karar × 2 bütçe) critical ≥0.30 recall @ ≤2 FA-saat VEYA
   advisory ≥0.50 @ ≤12 sağlarsa GEÇTİ. Ek rapor (gate değil): Actuator
   Outputs+Controls recall'u (ML-12 0.459 ile kıyas), Position.X/Y recall
   eski-vs-yeni (ExtPos zenginleştirmesinin beklenen faydası), 5-seed
   normal-FA dağılımı.
7. `gates.json` = {gate_d1 (rapor özeti + assert'ler), gate_d2, gate_d3};
   manifest: girdi hash'leri (yeni split manifest, yeni feature tablosu, eski
   ml9/ml12 manifestleri), development-id-hash, holdout sayısı,
   `blind_holdout_read: false`, tüm dosya checksum'ları.
8. Önce `--splits split_00` smoke, sonra tam 5-seed. (Yeni tablo ~1.5-2× satır;
   tam koşu süresi artar — smoke ile projeksiyon yap, raporla.)

## §6 Test güncellemeleri

1. **`tests/test_ml14.py` (yeni):** §2 donmuş-holdout unit testleri; D-gate
   json tutarlılığı (status, satırlardan türetilebilir olmalı); runner identity
   (`_evaluate`/`_fit_policies`/score_fusion aynı nesne); manifest checksum +
   holdout-hash; ALFA/UAT split-içerik eşitliği testi.
2. **Dönem-sabitleme (era-pinning) — ZORUNLU:** eski faz testleri "güncel
   split_manifest"e karşı çapraz kontrol yapıyor ve yeniden inşadan sonra
   YANLIŞ nedenle kırılacaklar. Kural: her eski-faz testi, artifact
   manifestindeki `split_manifest_sha256` güncel dosyanın sha256'sına EŞİTSE
   çapraz kontrolü yapar; değilse o alt-assert'i
   `pytest.skip("eski veri dönemi artifact'ı")` ile atlar — dosya-içi checksum
   kontrolleri HER ZAMAN çalışır. Uygulanacak yerler: `tests/test_ml10.py`
   (precompute≡development kontrolü), `tests/test_ml12.py`,
   `tests/test_ml13.py`, `tests/test_ml11_viz.py`. `grep -rn SPLIT_PATH tests/`
   ile tara, hepsini yakala. HİÇBİR eski testi silme.
3. Tam `pytest` yeşil (bilinen 4 MinIO hariç); yeni testler dahil sayıyı raporla.

## §7 Dokümantasyon + kabul

- Bulgular `docs/ML1_BULGULAR_VE_HATALAR.md`'ye "ML-14 sonuçları" (H32+):
  oturum sayısı değişimi, D2 kayma tablosu (eski vs yeni), D3 sonucu, dürüst
  yorum (veri duvarı yıktı mı, yıkmadı mı — iki sonuç da değerli).
- `docs/decisions.md` → ADR-014 (kota değişikliği + donmuş-holdout kuralı +
  gate sonuçları). `docs/ML_YETERSIZLIKLER_KAYDI.md` → D.1/E.1 güncellemesi +
  yeni durum; sayaç güncelle. `docs/CLAUDE_MEMORY.md` senkron.
- **Kabul kriterleri:**
  1. §2 assert'leri + unit testleri geçiyor (körlük matematiksel olarak korunmuş).
  2. D1 raporu tam; dev normal oturum sayısı ≥110.
  3. Smoke + tam 5-seed koşu tamam; gates.json + checksum'lı manifest.
  4. D2/D3 sayıları ham CSV'den bağımsız türetilebilir (gates ile eşleşme).
  5. Tam pytest yeşil (4 MinIO hariç); eski testler era-pin'li, silinmemiş.
  6. Donmuş dizinler bayt-bayt değişmemiş (`git status` ile kanıt).
  7. Sonuç NE ÇIKARSA ÇIKSIN holdout açılmaz; production model/threshold
     paketlerine (artifacts/models/**) dokunulmaz — onlar ML-15+ konusu.

## §8 Çalışma sırası

1. §2 splits.py donmuş-holdout + unit testler (indirme sürerken yazılabilir)
2. §1 bronze doğrulama (indirme bitince)
3. §3 yeniden inşa + §4 D1 raporu
4. §5 runner: smoke → tam matris
5. §6 test güncellemeleri + tam pytest
6. §7 docs + commit'ler (mantıksal bölünmüş, co-author'suz)
