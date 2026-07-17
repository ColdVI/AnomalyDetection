# 1-2. Hafta Sunumu — Revizyon Talimatları (Claude web için)

Bu dosyayı olduğu gibi Claude web'e (claude.ai) yapıştır. Sayılar zaten gerçek ve
doğrulanmış (kullanıcı decisions.md/archive ADR'lerinden teyit etti) — buradaki
eleştiri placeholder değil, **görsel şablon + jargon + bağlam eksikliği** üzerine.

**Genel teşhis:**
1. Slayt 3 (Ortak Proje) ve Slayt 5'in sol grafiği (Şekil 3, RflyMAD Proxy Düzeltmesi)
   yine kutu+ok şablonunda — 3. hafta sunumunda eleştirilen aynı kalıp. Slayt 5'in sağ
   grafiği (Şekil 4) ve Slayt 4'ün grafiği (Şekil 2) zaten gerçek bar chart, bunlara
   dokunma, iyi durumdalar.
2. **Slayt 4'te dış paydaşa gösterilemeyecek repo-içi jargon var: "LightGBM (ML-8A),
   kategori residual'ları (ML-9), iki-kanal mimarisi (ML-13)" ve "Chronos zero-shot:
   ML-10; ince-modül itki_komutu: ML-12".** Bu ML-N faz numaraları yalnız bu reponun
   iç iş-takibi, mentörün proje mimarisinden haberi yok — bunları temizle, yöntemi ne
   olduğu üzerinden anlat (ör. "LightGBM tabanlı pencere modeli", "kategori-özel
   residual feature'ları", "iki ayrı alarm kanalı mimarisi", "sıfır-atış zaman-serisi
   modeli (Chronos)", "ince, tek-feature'lı modül").
3. ADR kodları ("ADR-001", "ADR-002/ADR-005"...) çıplak bırakılmış — mentör bu kodların
   ne karar olduğunu bilemez. Ya kaldır ya da her birine üç-beş kelimelik bir açıklama
   ekle.
4. Slayt 2 (İÇİNDEKİLER) bu şablonun 3. hafta sürümünde başlık/daire çakışması vardı —
   aynı şablon ailesi olduğu için bu slaytta da aynı sorun olabilir, kontrol et.

---

## Gerçek görseller (repoda bulundu, kopyalandı)

`docs/sunum_1_2_hafta_gorseller/` klasörüne kopyaladım — Claude web'e doğrudan
yükleyebilirsin:

- `alfa_completeness_heatmap.png`, `alfa_class_counts.png` — ALFA hattının (senin
  geliştirdiğin) gerçek veri-kalite/kapsam görselleri. `completeness_heatmap`
  `velocity_mps` null sorununu görsel olarak zaten kanıtlıyor.
- `uav_attack_completeness_heatmap.png`, `uav_attack_class_counts.png` — aynısı UAV
  Attack hattı için (senin geliştirdiğin ikinci hat).
- `rflymad_parsed_pool.png` — RflyMAD'ın gerçek 490 uçuşluk kompozisyonu (Real-Motor
  ~242, Real-Sensors ~197, Real-No_Fault ~51 — toplamı 490'a denk geliyor).
- `rflymad_recall_vs_false_alarm.png` — gerçek recall-vs-saatlik-yanlış-alarm scatter
  grafiği, kritik/advisory hedef çizgileriyle. **Dikkat:** bu görselin başlığında
  "ML-14" yazıyor — olduğu gibi yüklersen jargon kuralını ihlal eder. Ya Claude web'den
  aynı veriyle başlıksız/jargonsuz bir versiyon çizmesini iste, ya da görseli başlığı
  kırparak kullan.

Kaynak: `magnitude_domination_diagnostic.json` (archive) — Slayt 5'teki ρ iddialarının
ham verisi: `trained_vs_untrained_random_init_spearman = 0.9637`,
`trained_vs_magnitude_only_spearman = 0.9645` (n=235.625 test penceresi). Bu, "eğitim
neredeyse hiçbir şey katmıyor" iddiasının kaynağı — slaytta bu n sayısını eklemek
iddiayı somutlaştırır.

---

## Slayt 3 — Ortak Proje (pipeline)

**Sorun:** Bronze→Silver→Gold üç kutu+ok — jenerik. Asıl mesaj ("ALFA ve UAV Attack
hattının TAMAMI benim tarafımdan geliştirildi") kutu diyagramında hiç görünmüyor, en
altta küçük bir caption'da ("Sorumluluklar: Metehan → ..., Yusuf → ..., Anıl → ...")
gömülü kalmış — bu slaydın en önemli cümlesi bu, görsel olarak en gizli yerde.

**Önerilen görsel:** Aynı Bronze→Silver→Gold akışını koru ama her kutunun altına/içine
kimin geliştirdiğini renk/etiketle göster (ör. Anıl'ın geliştirdiği ALFA+UAV Attack
akışı vurgulu renkte, adsb.lol hattı nötr gri) — "katkılarım" başlığının görsel
karşılığı bu olmalı. İkinci olarak yukarıdaki `alfa_completeness_heatmap.png` /
`uav_attack_completeness_heatmap.png` görsellerinden birini slayda ekle — "gerçek
veriyle doğrulandı" cümlesinin kanıtı olarak.

**İçerik derinleştirme:**
- Her ADR'nin ne olduğu bir-cümle: örn. "ADR-005: velocity_mps null sorununun
  belgelenip kabul edilmesi" gibi (gerçek içeriği decisions.md'den al).
- 99.885 satır/10 kolon sayısının hangi katmana (Gold) ait olduğunu açıkça belirt;
  Bronze/Silver katman satır sayıları da varsa ekle (pipeline'ın hacmini gösterir).
- 91 test sayısını "bilinen açık" cümlesinin yanına değil, pipeline güvenilirliğinin
  kanıtı olarak ayrı vurgula.

---

## Slayt 4 — İlk Yöntem Denemeleri (Gate B/C)

**Sorun:** Grafik zaten iyi (gerçek bar chart, 0.205→0.390→0.459, hedef çizgisi 0.50).
Tek sorun metindeki ML-N jargonu (yukarıya bkz.) ve "hepsi Gate B veya C'de kaldı"
cümlesinin hangi yöntemin tam olarak nerede kaldığını söylememesi.

**İçerik derinleştirme:**
- Jargon temizliği (zorunlu, feedback kuralı): "LightGBM (ML-8A)" → "LightGBM tabanlı
  pencere modeli"; "kategori residual'ları (ML-9)" → "kategoriye özel residual
  feature'ları"; "iki-kanal mimarisi (ML-13)" → "sistem+mekanik ayrı alarm kanalı
  mimarisi"; "Chronos zero-shot: ML-10" → "sıfır-atış zaman-serisi modeli (Chronos)";
  "ince-modül itki_komutu: ML-12" → "tek-feature'lı ince modül (itki komutu sinyali)".
- Grafikteki üç çubuğun her biri için tek satır "neden" ekle: neden `motor_simetrisi`
  (başlangıç) 0.205'te kaldı, Chronos'un zero-shot olması neden iyileştirdi, ince
  modülün TEK feature'a indirgenmesi neden daha da iyi sonuç verdi (seyrelme/dilution
  hipotezi — çok feature'lı versiyonların sinyali sulandırdığı gerçek bulgu, mentöre
  aktarılmaya değer).
- LightGBM/kategori-residual gibi Gate B'de bile kalan denemeler için en az bir sayı
  ekle (kullanıcının teklif ettiği AUPRC 0.349 gibi) — "hepsi kaldı" çok genel.

---

## Slayt 5 — Gerçek Veri ve Genlik-Baskınlığı

**Sorun:** Sol taraf (Şekil 3) iki kutu+ok — "0.749 recall" → "0.526/22.28 FA-sa" iyi
bir sayı ama kutu diyagramı METODOLOJİK FARKI göstermiyor (neden 0.749 yanlıştı).

**Önerilen görsel (sol, Şekil 3 yerine):** Kutu yerine tek bir uçuşun zaman çizelgesi
üzerinde iki etiketleme stratejisini üst üste göster: bir satırda "whole-flight proxy"
(arıza başlangıcından uçuş sonuna kadar TAMAMI kırmızı), altındaki satırda
"interval-truth" (yalnız gerçek arıza aralığı kırmızı, geri kalan yeşil). Bu, "neden
0.749 yanlıştı" sorusunu tek bakışta cevaplar — sayı değişikliğinden çok daha güçlü bir
demo. `rflymad_parsed_pool.png`'i de bu slayda küçük bir yan-görsel olarak ekleyebilirsin
(490 rakamının nereden geldiğini gösterir).

**Sağ taraf (Şekil 4):** İyi durumda, dokunma. İçeriğe eklenecek: yukarıdaki gerçek
n=235.625 test penceresi sayısı ve "relerr düzeltmesi" ifadesinin ne olduğu bir cümleyle
(göreli hata ile ölçekleme — ham genlik yerine oransal sapmaya bakmak) — şu an "relerr"
tanımsız bir kısaltma olarak duruyor.

**İçerik derinleştirme:**
- LSTM-AE/Dense-AE/USAD'ın üçünün de "aynı örtük soruna düştüğü" iddiası güçlü ama
  NEDEN aynı sorun olduğu eksik: kök neden RobustScaler'ın aykırı değerleri
  kırpmaması + bir uçuşun donmuş-GPS/eph≈25000 sentinel artefaktı taşıması — bu kök
  neden cümlesi mentöre "üç farklı mimari tesadüfen mi aynı hataya düştü" sorusunu
  önceden cevaplar (hayır, ortak bir veri/ölçekleme kusuru).
- Kullanıcının kendi notu: hafta4 sunumundaki "AUC≈0.54" ile bu slaydaki genlik-
  baskınlığı bulgusu aynı örüntünün tekrarı — bunu değerlendirme slaydına (Slayt 6)
  bir cümle olarak ekle: "3. haftadaki [FLAGGED] sonuç, burada [2. haftada] zaten bir
  kez keşfedilip belgelenmiş genlik-baskınlığı örüntüsünün tekrarıydı" gibi — bu iki
  sunum arasında bilimsel süreklilik kurar, mentöre "aynı hatayı fark edip
  belgeleyebiliyoruz" mesajı verir.

---

## Slayt 6 — Değerlendirme

Zaman çizelgesi (Şekil 5) zaten iyi bir format, template kutu değil — dokunma. Tek
ekleme: yukarıdaki hafta2↔hafta4 bağlantı cümlesi ve "9 yöntem" listesinin en az
isim olarak (jargonsuz) tek satırda sayılması ("basit istatistiksel eşik, LightGBM
penceresi, kategori-özel residual, iki-kanal füzyon, sıfır-atış zaman-serisi modeli,
ince tek-feature modül, ..." gibi) — şu an "9 yöntem" sadece bir sayı, hangi 9 olduğu
hiçbir yerde toplu görünmüyor.

---

## Slayt 1-2 (Kapak / İçindekiler)

Slayt 2'yi 3. hafta sunumunun aynı şablonundaki başlık/daire çakışma hatasına karşı
kontrol et (bu depoda önceki turda tespit edilmişti). Aynı hata burada da olabilir.

---

## Placeholder kuralı

Bu sunumda zaten placeholder yok (sayılar gerçek) — ama Claude web'den yeni bir görsel
istediğinde (ör. Slayt 3'ün kimlik-vurgulu pipeline diyagramı) sayı bilmediği bir yer
çıkarsa köşeli parantez (`<...>`) yazıp tasarımın içine literal metin olarak basmasın;
bilmediği yeri boş bıraksın ya da ayrı bir not olarak dışarıda belirtsin.

---

## Claude web'e verilecek özet talimat

> Slayt 3 ve Slayt 5'in sol grafiğini kutu+ok şablonundan çıkar (yukarıdaki önerilere
> göre: Slayt 3'te kimlik-vurgulu pipeline + gerçek completeness görseli, Slayt 5'te
> whole-flight-proxy vs interval-truth zaman çizelgesi karşılaştırması). Slayt 4'teki
> tüm "ML-N" faz numaralarını ve mümkünse ADR kodlarını jargonsuz açıklamalarla
> değiştir. Slayt 5 ve 6'ya yukarıdaki içerik-derinleştirme cümlelerini ekle
> (relerr tanımı, kök neden cümlesi, hafta2↔hafta4 bağlantısı, 9 yöntemin isim listesi).
> Slayt 2'yi başlık/daire çakışmasına karşı kontrol et. `docs/sunum_1_2_hafta_gorseller/`
> içindeki gerçek PNG'leri kullan, `ml14_recall_vs_false_alarm.png`'i yüklüyorsan
> başlığındaki "ML-14" ifadesini kırp veya aynı veriyle jargonsuz yeniden çizdir.
