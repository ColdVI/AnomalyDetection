# A→B Rota Tespiti + Sapma (Deviation) Uyarı Sistemi

Bireysel projeme yeni bir analiz katmanı eklemek istiyorum: iki nokta
(havaalanı) arasında tipik olarak izlenen rotayı çıkarıp, bu rotadan
belirgin şekilde sapan uçuşları arayüzde işaretlemek.

## Kapsam notu

Bu, `individual/metehan_geo/`'nun BAŞTAN PLANLANMIŞ bir parçası
(DBSCAN rota kümeleme) — şu ana kadar sadece H3 density tarafını
tamamladık, DBSCAN kısmına hiç girmedik. Yani bu, ülke projesi gibi
ayrı bir "eklenti" değil, ana projenin eksik kalan bir modülü —
`individual/metehan_geo/` içinde yeni dosyalar olarak (örn.
`route_clustering.py`, `deviation.py`) ekleyelim, mevcut `data.py`/
`geo.py` fonksiyonlarını (streaming Gold okuma, H3 binning) tekrar
kullanarak.

## Adım 1: A→B çiftlerini belirleme (havaalanı eşleme)

Callsign metninden rota çıkarmak güvenilir değil (callsign rotayı
kodlamıyor). Bunun yerine:

1. OurAirports açık veri setini indir (ücretsiz, ICAO/IATA kodlu
   havaalanı koordinatları): `https://davidmegginson.github.io/ourairports-data/airports.csv`
2. **UÇUŞ TANIMI — DİKKAT:** `source_id` + takvim-günü kombinasyonunu
   (build_flight_density.py'den ödünç alınan konvansiyon) BURADA
   KULLANMA. Bunu test ettik: gece yarısını (UTC) geçen gerçek uçuşlar
   bu tanımla ikiye bölünüyor, her parçanın ilk/son noktası havada bir
   yerde kalıyor ve hiçbir zaman gerçek havaalanına denk gelmiyor —
   eşleşmeyen trace'lerin %22-23'ü tam bu sebepten kayboluyordu
   (uniform beklenti %4.17'nin 5.5 katı, doğrulanmış bulgu).
   Bunun yerine: aynı `source_id`'nin trace'ini zaman sırasına göre
   tara, ardışık noktalar arası boşluk **>45-60 dakika** ise yeni bir
   uçuş başlat (zaman-boşluğu bazlı segmentasyon). Bu, gece yarısı
   sınırını sorunsuz geçer.
3. Her uçuşun İLK ve SON noktasını, **~15km yarıçap** içindeki en
   yakın havaalanına eşle (snap) — ilk denemede 8km kullanmıştık,
   eşleşmeyenlerin %30'unun 15-30km bandında kaldığını gördük, bu
   yüzden 15km'ye gevşetiyoruz. Yine de eşleşmeyen (bu yarıçapa da
   girmeyen) trace'leri hariç tut — muhtemelen okyanus/son-mil kapsama
   boşluğu kaynaklı gerçek kesik trace'lerdir (bunu ayrıca doğruladık).
4. Callsign'i (varsa) İKİNCİL bir doğrulama sinyali olarak kullan
   (örn. aynı havayolu+numara kombinasyonunun günler boyunca hep aynı
   A-B çiftine denk gelip gelmediğini kontrol etmek için), ama birincil
   yöntem olmasın.

## Adım 2: Yeterli örneklem olan A-B çiftlerini filtrele

Sadece belirli bir minimum uçuş sayısına (örn. ≥15-20 uçuş) sahip A-B
çiftleri için "tipik rota" hesapla — az sayıda uçuşla anlamlı bir
dağılım/kümeleme çıkarılamaz. Bu eşiği veri gerçek dağılımına göre
birlikte ayarlayalım (önce kaç A-B çiftinin bu eşiği geçtiğini raporla).

## Adım 3: Tipik rota / "çekirdek koridor" çıkarma

Önerilen yöntem (mevcut H3 altyapımızla uyumlu, DBSCAN'i tam
trajectory-clustering yerine daha basit bir başlangıç noktası olarak
kullanabiliriz):

1. Seçilen A-B çiftinin tüm uçuşlarının H3 hex setini (res-5 civarı)
   çıkar.
2. Her hex için, o A-B çiftinin uçuşlarının kaçının o hex'ten geçtiğini
   say (uçuş bazlı distinct count, ham nokta değil — flight_count
   mantığıyla aynı, önceki bug'ı tekrarlamayalım).
3. "Çekirdek koridor" = uçuşların en az %X'inin (örn. %70) geçtiği
   hex'ler kümesi.
4. Bunu DBSCAN ile de karşılaştıralım: trace noktalarını A-B arası
   normalize edilmiş ilerlemeye (0-100%) göre yeniden örnekleyip
   DBSCAN uygulamak, "ana küme" ile "aykırı" tekil rotaları ayırabilir
   — hangi yöntemin daha iyi sonuç verdiğini birlikte değerlendirelim,
   ilk aşamada basit hex-yüzdesi yöntemiyle başlayalım.

## Adım 4: Sapma (deviation) skoru + uyarı

Her bireysel uçuş için: trace'inin hex'lerinin ne kadarı çekirdek
koridor DIŞINDA kalıyor, bunu bir yüzde/skor olarak hesapla. Bir eşik
belirleyip (örn. >%30 dışarıda ise) bu uçuşu "sapma" olarak işaretle.

Arayüze:
- A-B çifti seçici (dropdown, yeterli örneklemi olan çiftlerden).
- Seçilen çift için çekirdek koridoru haritada vurgula.
- Sapma olarak işaretlenen uçuşları ayrı renk/stille (örn. kırmızı
  çizgi) göster, tıklanınca sapma yüzdesini ve muhtemel sebep
  ipucu (varsa irtifa/hız anormalliği) göster.

## Sıra

Önce Adım 1-2'yi (havaalanı eşleme + yeterli örneklemli A-B çiftlerini
bulma) yapıp bana kaç çift/uçuş bulduğunu raporla, ben onayladıktan
sonra Adım 3-4'e geçelim.
