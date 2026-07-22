# RflyMAD-Full v2 — Sonraki Adımlar (robustness + TCN sweep sonrası)

> Yazıldı: 2026-07-22 (Europe/Istanbul)
> Önkoşul: bu dosyadan önce `docs/RFLYMAD_V2_ROBUSTNESS_SONUCLARI_20260722.md` ve
> `docs/RFLYMAD_V2_CONVERGENCE_DENEY_RAPORU_20260722.md` ve
> `docs/RFLYMAD_V2_TCN_DEVELOPMENT_DENEY_RAPORU_20260722.md` okunmalı.

## 0. Neredeyiz

`RFLYMAD_V2_ROBUSTNESS_SOZLESMESI_20260722.md` sözleşmesi kapsamında altı aday
(R1, W1, W2, R2, R3, R4) development-only olarak koşuldu. **Altısı da kendi
promosyon kapısını geçemedi:**

- Real araştırma-promosyon kapısı: en iyi aday (R4) Real macro recall'ı
  %14,28 baseline'dan %28,11'e çıkardı ama hedef %40'ın altında; ayrıca genel
  recall düştü (%60,4→%54,6) ve FA yükü arttı.
- Wind ara kapısı: en iyi aday (W2) Wind FA'yı 28,46'dan 17,78/saate indirdi
  ama hedef (≤15 ort., ≤20 max, ≥%40 azalma) tutmadı; W1 Wind'i çok indirdi
  (%72) ama genel recall'ı ~50 puan düşürerek.
- Birleşik RW1 koşulmadı (koşulu oluşmadı).

Sözleşmenin kendi durdurma kuralı bu noktada devreye girer: **"yeni Real veri
veya yeni temsil sözleşmesi olmadan threshold/fine-tune avı sürdürülmez"** ve
**"Wind robustness çözülmedi olarak kalır."** Bu, başarısızlık değil — sözleşme
tam olarak böyle bir sonucu öngörüp p-hacking'i önlemek için yazılmıştı ve
işe yaradı: sonuç net, dürüst ve tekrar üretilebilir şekilde raporlandı.

Paralelde: truth cross-check metriği düzeltildi (`v2_parser.py` içine
`truth_crosscheck_disagreement_v2` + onset-tolerance eklendi), ve düzeltilmiş
truth ile development-only 3-epoch TCN sanity koşusu yapıldı
(`artifacts/rfly_full/v2/supervised_tcn/run_20260722_111938/`):
critical recall %13,68/FA 2,43 saat, advisory recall %71,40/FA 13,59 saat,
`status=smoke_only`. Bu sadece "sinyal var mı" testidir, kapı kararı değildir.

Ardından sonuçlardan önce dondurulan sözleşmeyle TCN development-only 5-fold
sweep tamamlandı. En iyi epoch'lar `5, 5, 3, 2, 2` olduğundan hiçbir fold
12-epoch sınırında uzatılmadı. Critical recall/FA `%28,87 / 2,87-saat`, advisory
`%67,86 / 12,54-saat` oldu. Critical, advisory, Real ve Wind kapılarının hiçbiri
geçilmedi. Bu sonuç TCN'in de mevcut veri/temsil üzerinde genel çözüm olmadığını
ve problemin yalnız AE'ye özgü olmadığını güçlendirir.

## 1. Zorunlu sınır — bunu yapma

Mevcut robustness sözleşmesi kapsamında **yeni bir Real veya Wind AE
konfigürasyonu (yeni epoch/LR/threshold/ağırlık kombinasyonu) çalıştırma.**
Altı aday hakkı kullanıldı, sonuç dondu. Yeni bir deneme, sonuç beğenilmeyince
kuralı esnetmek anlamına gelir ve tam olarak bu sözleşmenin önlemeye çalıştığı
şeydir. Bu satırı ihlal eden herhangi bir istek (kullanıcıdan gelse bile)
önce yeni bir yazılı sözleşme gerektirir.

## 2. Üç meşru ileri yol

### Yol A — TCN development-only uzun koşu (**tamamlandı**)

Orijinal handoff sırası tamamlandı: 3-epoch sanity sonrasında beş disjoint
development outer fold, validation-only checkpoint/uzatma seçimiyle çalıştırıldı.

```powershell
.venv\Scripts\python.exe `
  scripts\run_rfly_full_v2_supervised_development_sweep.py
```

- **Düzeltme:** `development_smoke_fold=None` mevcut kodda locked-test manifestini
  seçer ve `locked_test_features_read=true` üretir; development-only koşuda
  kullanılamaz. Sweep çalıştırıcısı beş outer fold için bu alanı zorunlu ve
  birbirinden ayrık biçimde ayarlar.
- Beş dış rotasyonun hepsi sonuçlardan önce dondurulan
  `RFLYMAD_V2_TCN_DEVELOPMENT_SOZLESMESI_20260722.md` ile çalıştırılır.
- Sonuç AE'nin 5-rotasyon sweep'iyle Real/Wind dahil yan yana raporlandı.
- TCN hiçbir ana development kapısını geçmedi ve ikinci ana deneysel aday
  sayılmadı.

### Yol B — Real-domain için temsil/veri değişikliği (yüksek çaba, kök nedene iner)

Convergence raporunun kendi teknik yorumu (Bölüm 6) bunu zaten işaret ediyor:
uzun fine-tune Real sinyalini biraz güçlendiriyor ama genelleme ve alarm
yükünü bozuyor — bu, **threshold veya epoch sorunu değil, temsil/aşırı-uyum
sorunu**. Olası yönler (hiçbiri şu an onaylı değil, her biri yeni bir yazılı
sözleşme ister):

1. Flight-phase normalization (kalkış/seyir/iniş fazlarını ayrı normalize et).
2. Domain-invariant feature analizi (Real'i ayırt eden özelliklerin
   SIL/HIL'de neden farklı dağıldığını incele).
3. Encoder katmanlarını kısmen dondurup yalnız üst katmanı fine-tune et
   (tam-model fine-tune yerine).
4. Daha fazla bağımsız Real-NoFault session — şu an yalnız 3 session grubu
   var (R4'ün 780/628 epoch'a rağmen session'a aşırı uyum yapması bunun
   göstergesi); yeni veri toplanmadan bu kısıtlama muhtemelen aşılamaz.

**Bu yol veri kısıtlı olabilir** — (4) mümkün değilse (1)-(3) muhtemelen tavan
yapar. Başlamadan önce kullanıcıyla hangi alt-yönün denenebilir olduğu
netleştirilmeli.

### Yol C — Wind için kapsamı daralt veya farklı bir mekanizma dene

W1/W2 ikisi de "threshold'u/eğitimi Wind'e göre kaydır" yaklaşımıydı ve
ikisi de recall veya FA'den ödün verdi. Alternatif, aynı ailede bir varyant
değil, farklı bir mekanizma olurdu: örneğin ayrı bir "ortam durumu"
sınıflandırıcısı (Wind algıla → o pencerede farklı/gevşetilmiş eşik uygula)
gibi **iki-aşamalı gating**. Bu da yeni sözleşme gerektirir ve bu turun
kapsamı dışındadır — burada yalnız seçenek olarak not ediliyor.

**Alternatif (daha ucuz):** Wind'i şimdilik "çözülmedi" olarak dondurup
operasyonel kapsamdan açıkça çıkarmak (sözleşmenin zaten izin verdiği sonuç)
ve enerjiyi Yol A/B'ye vermek.

## 3. Önerilen sıradaki tek adım

Yeni model koşusu başlatmadan **development-only temsil/domain teşhisi** yap:
Real Sensor, Real Motor, Wind ve diğer nonfault domainlerde mevcut AE/TCN skor ve
feature dağılımlarının hangi flight-phase/özelliklerde ayrıştığını ölç. Bu analiz
model/threshold seçmez ve yeni aday tüketmez; amacı Yol B için sonuçlardan önce
dondurulabilecek tek bir müdahale hipotezi üretmektir.

Teşhis sonunda yeni veri ihtiyacı baskınsa ek bağımsız Real-NoFault session
toplanmadan yeni training sweep açma. Temsil kusuru baskınsa flight-phase
normalization veya domain-invariant representation seçeneklerinden yalnız biri
yeni yazılı sözleşmeyle seçilmeli. Wind iki-aşamalı gating ayrı araştırma hattı
olarak kalmalı.

## 4. Ev işleri

- Yeni koşu öncesi Python süreç durumu yeniden doğrulanmalı.
- Model checkpoint/parquet ve kök logları `.gitignore` kapsamındadır; küçük
  CSV/JSON özetleri, rapor görselleri ve çalıştırılmış notebook yayınlanır.
- İlgili test suite'i her yayın öncesi çalıştırılmalı:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -k rfly_full -q
```

## 5. Raporlama ilkesi (değişmedi)

Her yeni sonuç: hangi kapı/hedef, geçti mi/geçmedi mi, `operational_claim_allowed`
değeri, ve bir önceki sonuçla yan yana sayı. "Recall arttı" gibi tek taraflı
ifadeler FA/genelleme maliyeti belirtilmeden yazılmaz — bugünkü R4 örneği
(Real macro arttı ama genel recall ve FA kötüleşti) bunun neden zorunlu
olduğunu gösteriyor.
