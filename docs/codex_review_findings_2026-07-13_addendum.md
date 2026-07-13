# Üç Dosyalı Kör-Holdout Havuzu — Ek İnceleme Bulguları

**Tarih:** 2026-07-13  
**Dayanak:** `docs/codex_review_prompt_2026-07-13.md` ve
`docs/codex_review_prompt_2026-07-13_addendum.md`  
**İlişki:** Onaylanmış `docs/codex_review_findings_2026-07-13.md` değiştirilmedi. Bu rapor
yalnız Adım 9'daki tek-dosyalı holdout tanımını üç-dosyalı havuza genişletir; Adım 1–8 ve
dokuz dokunulmaz kısıt aynen kalır.

## Yönetici kararı

Üç tarın hiçbiri eğitim, kalibrasyon, development veya parser geliştirme verisi yapılmamalıdır.
Önerilen kullanım, dosyaları sırayla açıp her sonuçtan sonra karar vermek değil, **tek mantıksal
açılışta, ara sonuçları ambargolu ortak bir holdout batch'i** olarak tüketmektir.

Üç dosya aynı `pool_id` altında, yalnız takvim uzaklığına dayalı üç zorunlu rapor stratum'udur:

- `2024-09-01`: `far_backcast`;
- `2025-02-15`: `mid_backcast`;
- `2025-06-15`: `near_backcast`, fakat `scope_status=unknown` (`-003`).

Bu adlar model sonucu veya içerik bilgisi kullanmaz; rol seçimi değildir. Üçünün gerçek rolü
aynıdır: `blind_backcast_test`. Mühendislik dosyaları seri işlemeyi gerektirirse deterministik
kronolojik sıra kullanılabilir, ancak üç job tamamlanana kadar hiçbir metrik veya mekanik sonuç
insana açılmaz; arada kod/config/eşik değişmez ve erken durdurma yapılmaz.

Birincil endpoint, üç mechanically eligible artefaktın toplam alert episode sayısının toplam
scoreable flight-hour'a oranıdır. Üç stratumun burden, scoreability ve attrition sonuçları da
zorunlu ve eksiksiz yayımlanır. En iyi gün seçilmez. Equal-date macro ortalama yalnız secondary
sensitivity'dir; üç tarih rastgele/temsili bir örnek değildir.

## 1. Metadata ve temporal kapsam

**Bulgu.** Addendum'daki üç dosya ve metadata değeri dosya sisteminde doğrulandı. İçerikleri
açılmadı, üyeleri listelenmedi, hashlenmedi veya parse edilmedi. Üç tarih de fit gününden
eskidir; bu havuz forward deployment holdout'u değil, geriye-dönük temporal/source-transfer
stres panelidir.

**Kanıt.** Hedefli `Get-Item` ile yapılan inceleme ölçümü:

| Artefakt | Byte | Yerel mtime | Fit gününden önce |
|---|---:|---|---:|
| `v2024.09.01-planes-readsb-prod-0.tar` | 2,084,157,440 | 2026-07-13 15:13:26 | 545 gün |
| `v2025.02.15-planes-readsb-prod-0.tar` | 2,146,856,960 | 2026-07-13 15:13:03 | 378 gün |
| `v2025.06.15-planes-readsb-prod-0-003.tar` | 3,093,094,400 | 2026-07-13 12:02:04 | 258 gün |

Byte ve mtime değerleri addendum ile aynıdır
(`docs/codex_review_prompt_2026-07-13_addendum.md:12-19`). Gün farkları, filename tarihi ile
2026-02-28 arasındaki bu inceleme-türevi takvim hesabıdır; yaklaşık ay ifadesi değildir. Üç
byte değerinin türetilmiş toplamı **7,324,108,800 byte**'tır. Dosya büyüklüğü kapsam veya
“tam gün” kanıtı değildir.

**Öneri.** Şimdilik yalnız bu metadata kaydı korunsun. SHA-256, ancak Adım 9 kullanıcı/gate
kararıyla başladığında, üç dosya için aynı manifest turunda hesaplansın. Hash raw byte'ları
okur ama tar üyesi listelemez/decompress etmez; erişim günlüğünde bu ayrım açıkça yazılsın.
Manifest mtime'ı hem yerel timezone bilgisiyle hem UTC olarak kaydetsin.

**Tahmini efor.** Metadata/freeze şablonu 0.5 iş günü; ölçülmüş süre değildir. Hash çalışma
süresi ölçülmedi ve burada tahmin edilmedi.

**Risk.** Filename tarihi gerçek iç timestamp kapsamını kanıtlamaz. Üç eski tarihte başarı,
2026-03-16 sonrası geleceğe genelleme kanıtı değildir; prospektif holdout ihtiyacı sürer.

## 2. Neden tek batch ve sonuç ambargosu

**Bulgu.** “Önce birini aç, sonuca göre diğerine rol ver” tasarımı ilk dosyayı fiilen
development verisine dönüştürür ve optional stopping/post-hoc dosya seçimine kapı açar. Üç
dosyanın tek mantıksal açılışı bu riski en doğrudan kapatır.

**Kanıt.** Addendum, üçlü havuzun kullanım şeklinin sonuç görülmeden önce seçilmesini istiyor
(`docs/codex_review_prompt_2026-07-13_addendum.md:23-31`). Ana rapor da parser, truth,
scoreability, feature/rule/CUSUM/eşik/event/gate sözleşmesinin açılıştan önce dondurulmasını ve
holdout'un yalnız development gate'inden sonra bir kez açılmasını şart koşuyor
(`docs/codex_review_findings_2026-07-13.md:559-591`).

**Öneri.** Açılış protokolü:

1. Tek champion sistem, parser commit/şema, scaler/rule/CUSUM/S2 artefaktları, threshold,
   eventizer, scoreability, metric ve gate hashleri önce dondurulur. Holdout üzerinde model
   seçimi veya challenger seçimi yapılmaz.
2. Runner; output isolation, restart, attrition ve rapor akışı dahil, yalnız non-blind 2026
   development raw'ı veya sentetik fixture üzerinde uçtan uca dry-run edilir.
3. Tek kullanıcı/gate kararı üç raw artefaktı birlikte `unseal` eder. Fiziksel okuma sırası
   gerekiyorsa `2024-09-01 → 2025-02-15 → 2025-06-15` olur; bu yalnız deterministik execution
   sırasıdır.
4. Job çıktıları erişim-kontrollü staging alanında tutulur. Üç job ve birleşik attrition raporu
   tamamlanmadan insan ara sonucu göremez; early-stop veya “üçüncüyü açmama” yoktur.
5. Her artefakt ayrı write-once output namespace'ine parse edilir; doğrulamadan sonra namespace
   sealed/read-only yapılır. Normal Silver target'ına yazılmaz.
6. Altyapı kesintisi yalnız aynı raw hash, container/commit, config ve checkpoint ile restart
   edilebilir. Bilimsel schema/parser uyumsuzluğu config değiştirip “blind retry” gerekçesi
   değildir.

**Tahmini efor.** Freeze/dry-run denetimi 0.5–1 iş günü; kullanıcı gate'i sonrasında üç izole
parse/evaluation ve birleşik rapor 1–2 iş günü. Bunlar planlama tahminidir.

**Risk.** Tek batch üç blind varlığı aynı anda tüketir. Bunun karşılığı, dosyaları sırayla
görerek ayarlama özgürlüğünün bilinçli olarak kapatılmasıdır. Dry-run yapılmazsa basit altyapı
hatası üç varlığı birden tüketebilir.

## 3. `-003` mekanik kapsam belirsizliği

**Bulgu.** Yalnız 2025-06-15 adındaki `-003`, shard olabileceğini düşündürür fakat kanıtlamaz.
Dosyanın daha büyük olması da tam-gün kanıtı değildir. Freeze öncesi “tam gün” varsayılmamalı;
freeze sonrası ilk erişimde kapsam mekanik olarak sınıflandırılmalıdır.

**Kanıt.** Belirsizlik addendum'da açıkça kaydedilmiştir
(`docs/codex_review_prompt_2026-07-13_addendum.md:14-16,32-35`). Ana rapor, ilk mekanik
kontrolü holdout erişimi sayar; schema/parser başarısızlığını dış-geçerlilik sonucu olarak
raporlar ve aynı holdout'a özel tamirle ikinci bir blind iddiayı yasaklar
(`docs/codex_review_findings_2026-07-13.md:569-589`).

**Öneri.** Freeze manifestinde 2025-06-15 için baştan:

- `declared_date_from_filename=2025-06-15`;
- `scope_status=unknown`;
- `full_day_claim=not_asserted`;
- `filename_suffix=-003`

kaydedilsin. İlk izinli açılışta, skordan önce ama freeze sonrasında, aynı otomatik batch içinde
önceden sınırlanmış şu mekanik kontroller çalışsın: tar integrity, member naming/index,
timestamp min/max ve saat/gap coverage, required schema/type/unit, duplicate kuralı, parse hata
ve attrition sayıları. Alarm/metrik sonucu eligibility belirleyemez.

`-003` partial capture çıkarsa post-hoc dışlanmasın veya başka dosyayla değiştirilmesin;
scoreable exposure'u saat-normalize pooled endpoint'e girsin, kapsamı zorunlu stratified
tabloda `partial_capture` diye gösterilsin ve tam-gün drift iddiası kurulmasın. Parser/schema
uyumsuzsa stratum `external_validity_failure/inconclusive` olarak tüketilmiş sayılır; diğer iki
job yine tamamlanır.

**Tahmini efor.** Mekanik scope/attrition doğrulaması üçlü evaluation süresine dahildir;
ayrı ölçülmüş süre yoktur.

**Risk.** Kapsam kontrolünü skordan sonra yorumlamak, iyi/kötü sonuca göre dosyayı dahil etme
riskini doğurur. Partial capture'ın exposure-normalizasyonu kapsam eksikliğini düzeltmez;
yalnız alarm yükü paydasını dürüstleştirir.

## 4. Endpoint, strata ve çoklu seçim kontrolü

**Bulgu.** Üç tarih daha zengin stres kapsamı sağlar, fakat üç ayrı model seçme yarışına veya
“en iyi gün” raporuna dönüştürülürse körlük avantajı kaybolur. Tek pooled primary endpoint ve
zorunlu tam stratified panel, en az seçim serbestliğiyle en çok bilgiyi korur.

**Kanıt.** Ana rapor doğal trafik için alert episode/scoreable flight-hour, clean flight alarm
oranı ve scoreability'yi ayrı ister; sentetik recall'ın doğal burden olmadan raporlanmasını
yasaklar (`docs/codex_review_findings_2026-07-13.md:499-543`). Üç tarın hiçbirinde gerçek
anomaly etiketi olduğu ölçülmedi; dolayısıyla primary outcome doğal alert burden ve
scoreability'dir, recall değildir.

**Öneri.** Sonuç sözleşmesi:

- **Tek primary endpoint:** mechanically eligible içerikte
  `Σ alert episodes / Σ scoreable flight-hours`. Episode merge/refractory ve flight-hour
  tanımı development'ta önceden dondurulur.
- **Zorunlu strata:** her artefakt için aynı burden, scoreability, toplam exposure, alarm
  episode sayısı ve eksiksiz attrition/reason-code tablosu. Dosya/saat/aircraft seçilmez.
- **Safety guardrail:** pooled ortalamanın kötü bir stratum'u gizlememesi için her eligible
  stratum aynı ön-kayıtlı burden sınırına karşı gösterilir. Per-stratum pass/fail kullanılacaksa
  aile-düzeyi tek-taraflı interval/multiplicity yöntemi development'ta sayısal olarak
  dondurulur; bu rapor ölçülmemiş bir alpha veya burden limiti uydurmaz.
- **Secondary:** equal-date macro burden ve tarih-stratified sensitivity. Üç tarih rastgele
  örnek olmadığı için “drift eğrisi” veya population trend iddiası yoktur.
- **Sentetik secondary:** yalnız donmuş recipe/truth ile tam `3 artefakt × tüm senaryolar`
  matrisi; her hücre aynı artefaktın doğal burden'ı yanında sunulur. Hücre veya senaryo seçimi
  yapılmaz.
- **Tek champion:** holdout öncesi seçilir. Rule/NN/USAD sıralaması bu havuzda yapılmaz;
  diagnostic component skorları yayınlansa bile sonraki model seçimi için bu üçlü tekrar blind
  sayılmaz.

Havuz sonucu sınıfları da önceden dondurulsun:

- `pass`: üç manifest entry'si de mechanically eligible olur; pooled endpoint ve ön-kayıtlı
  zorunlu guardrail'ler geçer;
- `performance_fail`: mechanically eligible stratum/pooled gate performans nedeniyle kalır;
- `external_validity_failure/inconclusive`: kapsam/schema/parser/scoreability sözleşmesi
  değerlendirilemez; sessizce pass veya exclusion yapılmaz.

Batch tamamlandığında üç dosya da tüketilmiş test setidir. Model/parser/eşik değişikliği yeni
versiyondur; bu havuzdaki tekrar skor “blind” diye sunulamaz. Yeni iddia yeni, tercihen
2026-03-16 sonrası prospektif holdout gerektirir.

**Tahmini efor.** Metric/guardrail/attrition sözleşmesini freeze etmek 0.5 iş günü; ölçülmüş
süre değildir.

**Risk.** Pooled oran tek başına kötü günü maskeleyebilir; post-hoc per-day gate ise
multiplicity yaratır. Tam panel ve ön-kayıtlı guardrail ikisini birlikte sınırlar. Üç gün aynı
provider ailesinden olabilir ve bağımsız/temsili örnek oldukları ölçülmedi.

## Güncellenmiş Adım 9 — Adım 1–8 değişmez

**Bulgu.** Önceki planın sırası değişmemelidir; yalnız Adım 9'un nesnesi tek tar yerine üçlü
havuzdur (`docs/codex_review_prompt_2026-07-13_addendum.md:3-4,23-26`).

**Kanıt.** Önceki Adım 9 hâlâ tek 2025-06-15 tarını adlandırır
(`docs/codex_review_findings_2026-07-13.md:624-638`); addendum bunu açıkça üçlü havuza
genişletmiştir.

**Öneri — Adım 9'un yerine geçecek metin.**

> **9 / Gün 10:** Kullanıcı kararıyla 2024-09-01, 2025-02-15 ve 2025-06-15 raw tarları,
> içerikleri açılmadan aynı `pool_id` altında raw path/byte/mtime/SHA-256 ve
> `scope_status` alanlarıyla freeze edilir. Holdout runner üçünde de dry-run edilmez; dry-run
> yalnız non-blind development/fixture üzerinde tamamlanır. Bu iki haftada varsayılan olarak
> tar üyesi listeleme, parse veya evaluation yoktur. Daha sonraki tek gate kararı üç artefaktı
> aynı mantıksal batch'te, sonuç ambargosuyla açar; üçü de test-only kalır.

**Tahmini efor.** Adım 9 freeze işi 0.5–1 iş günü; açılış/evaluation daha sonraki ayrı gate'in
1–2 iş günlük planlama tahminidir.

**Risk.** Eski tek-dosyalı satırı yeni havuzla birlikte iki alternatif plan gibi bırakmak
yorum serbestliği doğurur. Bu ek rapor, yalnız Adım 9 için normatif replacement'tır; onaylanmış
Adım 1–8'e dokunmaz.
