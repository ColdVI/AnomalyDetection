# RflyMAD-Full v2 — Sonraki Adımlar (robustness sweep sonrası)

> Yazıldı: 2026-07-22 (Europe/Istanbul)
> Önkoşul: bu dosyadan önce `docs/RFLYMAD_V2_ROBUSTNESS_SONUCLARI_20260722.md` ve
> `docs/RFLYMAD_V2_CONVERGENCE_DENEY_RAPORU_20260722.md` okunmalı.

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

## 1. Zorunlu sınır — bunu yapma

Mevcut robustness sözleşmesi kapsamında **yeni bir Real veya Wind AE
konfigürasyonu (yeni epoch/LR/threshold/ağırlık kombinasyonu) çalıştırma.**
Altı aday hakkı kullanıldı, sonuç dondu. Yeni bir deneme, sonuç beğenilmeyince
kuralı esnetmek anlamına gelir ve tam olarak bu sözleşmenin önlemeye çalıştığı
şeydir. Bu satırı ihlal eden herhangi bir istek (kullanıcıdan gelse bile)
önce yeni bir yazılı sözleşme gerektirir.

## 2. Üç meşru ileri yol

### Yol A — TCN'i development-only uzun koşuya çıkar (düşük risk, altyapı hazır)

Bu, orijinal handoff'un zaten onaylı sırasıdır (`YENI_CHAT_HANDOFF` Bölüm 12,
madde 9): "önce kısa 3-epoch sanity, sonra gerekirse 12+ epoch." 3-epoch
sanity bugün tamamlandı ve makul bir sinyal gösterdi (advisory recall %71,4
— AE'nin advisory recall'ından farklı bir profil). Yeni bir sözleşme
gerekmez, çünkü bu zaten var olan supervised TCN deney sözleşmesinin
(`YENI_CHAT_HANDOFF` Bölüm 5.2) bir sonraki adımıdır.

```powershell
.venv\Scripts\python.exe scripts\run_rfly_full_v2_supervised.py `
  --validation-fold 0 --epochs 12 `
  --max-train-windows 50000 --max-val-windows 20000
```

- Kilitli test bu koşuda da kapalı kalmalı mı yoksa 12-epoch tam-protokol
  koşusu mu (locked test hâlâ yalnız final audit'te açılır, bu koşu yalnız
  development+outer-rotation kullanır, `development-smoke-fold` OLMADAN)
  — bu ayrım netleştirilmeli: 12-epoch koşu "tam development" sayılır,
  `development_smoke_fold=None` ile çalıştırılabilir çünkü locked test split'i
  zaten ayrı ve okunmuyor.
- Beş dış rotasyonun hepsinde koş (supervised TCN zaten `--validation-fold`
  parametresiyle 5-fold destekliyor).
- Sonucu AE'nin 5-rotasyon sweep'iyle (Real/Wind dahil) yan yana raporla.
- TCN'in Real transfer'de AE'den daha iyi/kötü olup olmadığını gözlemsel
  olarak not et — ama TCN için de aynı disiplin: sonuçlara bakmadan önce
  "TCN'i ikinci ana aday sayarız" eşiğini kabaca handoff'taki mevcut
  fizibilite kapılarından (Bölüm 11) türet, sonradan bükme.

**Bu yolun avantajı:** yeni veri/temsil gerektirmiyor, altyapı zaten yazıldı
ve bellek açısından doğrulandı (1.167 MB, kapı 4.096 MB). En hızlı somut
sonraki bilgi kazancı budur.

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

**Yol A'yı (TCN development-only 12-epoch) önce çalıştır.** Gerekçe: sıfır
yeni sözleşme maliyeti (zaten onaylı), altyapı hazır ve doğrulanmış, veri
toplama gerektirmiyor, ve sonucu Yol B'ye girip girmemeye karar vermek için
kullanılabilir (TCN da Real'de başarısızsa, sorunun AE'ye özgü olmadığı,
muhtemelen veri/temsil kaynaklı olduğu — yani doğrudan Yol B — daha güçlü
kanıtlanır).

Yol B ve C, kullanıcıyla hangi alt-seçeneğin (yeni veri var mı, hangi
temsil değişikliği önceliklendirilsin) netleştirilmeden başlatılmamalı.

## 4. Ev işleri (bu tur bitmeden)

- Şu an arka planda çalışan python süreci yok (kontrol edildi); bu böyle
  kalmalı — yeni bir koşu başlatılırsa süreç durumu tekrar doğrulanmalı.
- Çalışma ağacı hâlâ tamamen untracked/kirli: `artifacts/rfly_full/` çok
  büyük olabilir (model checkpoint'leri, parquet skorlar) ve repo kök
  dizininde çok sayıda başıboş `.log` dosyası var
  (`rfly_*.err.log`/`.out.log`, `rfly_full/`, vb.). Commit yapılmadan önce
  kullanıcıyla şunlar netleştirilmeli: (a) `artifacts/` gerçekten repoya mı
  girecek yoksa `.gitignore`'a mı alınacak, (b) kök dizindeki log
  dosyalarının hepsi gerekli mi yoksa temizlenebilir mi. Bu, benim veya
  Codex'in tek taraflı karar vereceği bir şey değil.
- `tests/` altında bugünkü değişikliklerle ilgili suite'in tamamı hâlâ
  yeşil olmalı; yeni bir koşu öncesi tekrar doğrulanmalı:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -k rfly_full -q
```

## 5. Raporlama ilkesi (değişmedi)

Her yeni sonuç: hangi kapı/hedef, geçti mi/geçmedi mi, `operational_claim_allowed`
değeri, ve bir önceki sonuçla yan yana sayı. "Recall arttı" gibi tek taraflı
ifadeler FA/genelleme maliyeti belirtilmeden yazılmaz — bugünkü R4 örneği
(Real macro arttı ama genel recall ve FA kötüleşti) bunun neden zorunlu
olduğunu gösteriyor.
