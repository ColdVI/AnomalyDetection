# Coğrafi Trafik Hub Tespiti — DBSCAN (A-B Eşleşmesinden Bağımsız)

Resmi proje planındaki "Rota kümeleme modeli" görevini, A-B havaalanı
eşleşmesine ihtiyaç duymayan ayrı bir yaklaşımla da uygulamak istiyorum:
tüm ham konum verisini doğrudan kümeleyerek, hava trafiğinin doğal
coğrafi merkezlerini (hub) veri-güdümlü şekilde keşfetmek — önceden
tanımlanmış ülke/idari sınır kullanmadan.

## Yöntem

1. **Girdi:** Gold verisindeki (veya örneklenmiş bir alt kümesi —
   1 milyar satırın tamamı yerine, örn. flight_count hesaplamasında
   kullandığımız benzersiz uçuş bazlı nokta seti, ya da H3 hex
   centroid'leri + o hex'in flight_count'u ağırlık olarak) lat/lon
   noktaları. Tüm ham noktaları kullanmak hesaplama açısından çok
   pahalı olabilir — önce hangi granülaritenin (ham nokta mı, hex
   centroid mi, hangi resolution) pratik olduğunu değerlendirelim.

2. **DBSCAN uygula** (`sklearn.cluster.DBSCAN`, `metric='haversine'`
   ile küresel mesafe kullanarak, koordinatları radyana çevirip):
   - `eps`: kilometre cinsinden bir mesafe eşiği (örn. başlangıç için
     50-100km dene, k-distance/elbow yöntemiyle veriye göre ayarla).
   - `min_samples`: bir bölgenin "hub" sayılması için gereken minimum
     nokta/ağırlıklı-uçuş sayısı (örn. 50-100 gibi, veri yoğunluğuna
     göre ayarlanacak).
   - Eğer hex centroid + flight_count ağırlığı kullanıyorsak, DBSCAN
     ağırlıklı örnekleri desteklemiyor — bunun yerine her hex'i
     flight_count'una ORANTILI olarak tekrar örnekleyip (weighted
     resampling) ham DBSCAN'e verebiliriz, ya da HDBSCAN gibi ağırlık
     destekli bir alternatifi değerlendirebiliriz. Hangisi daha
     pratik, birlikte karar verelim.

3. **Çıktı:**
   - Her küme = bir "trafik hub'ı" (örn. Batı Avrupa, Körfez, Güney
     Asya gibi çıkması bekleniyor, önceden varsaymadan).
   - `-1` (noise) = hiçbir yoğun kümeye dahil olmayan, seyrek/izole
     noktalar.
   - Her küme için: merkez koordinatı, kapsadığı toplam uçuş/nokta
     sayısı, yaklaşık coğrafi sınır (convex hull veya alpha shape).

4. **Görselleştirme:** Mevcut haritaya (index.html) yeni bir katman/mod
   olarak ekle — her küme farklı bir renk/sınır ile gösterilsin, tıklanınca
   o kümenin istatistiklerini (uçuş sayısı, merkez, yaklaşık kapladığı
   alan) göstersin. Bunu var olan hexagon/heatmap katmanlarının YANINA
   opsiyonel bir "Trafik Kümeleri" modu olarak ekleyelim, mevcut
   görünümleri bozmadan.

## Önemli çerçeveleme notu (rapor için)

Bu, "anomali/sapma tespiti" DEĞİL — amaç, veri-güdümlü şekilde hava
trafiğinin doğal coğrafi merkezlerini bulmak (spatial data mining,
mekansal kümeleme). Sonuçları sunarken "algoritma X kümesini önceden
tanımlanmış hiçbir ülke/bölge sınırı kullanmadan, sadece nokta
yoğunluğuna bakarak keşfetti" şeklinde çerçevele.

## Hyperparameter Tuning — eps/min_samples'ı veriden hesapla

`eps` ve `min_samples`'ı elle deneme-yanılma ile sabitlemek yerine,
k-distance/elbow yöntemiyle veriden türet — bu, ML sürecinin standart
bir parçası (hyperparameter tuning) ve raporda ayrı, değerli bir
metodoloji notu olur:

1. `min_samples`'ı önceden seç (örn. 50 ya da 2×boyut kuralı gibi
   makul bir varsayılan).
2. Her noktanın `min_samples`'ıncı en yakın komşusuna olan mesafesini
   hesapla (`sklearn.neighbors.NearestNeighbors`).
3. Bu mesafeleri küçükten büyüğe sırala ve çiz (k-distance grafiği).
4. Grafikteki "dirsek" (elbow) noktasını — eğrinin eğiminin keskin
   şekilde arttığı yeri — otomatik ya da görsel olarak tespit et, bunu
   `eps` olarak kullan.
5. Bu süreci (grafik + seçilen eps değeri) bir çıktı/görsel olarak
   sakla — raporda "eps'i nasıl seçtik" sorusuna görsel kanıt olarak
   kullanılabilir.

Bunu sabit bir global `eps` olarak mı uygulayalım, yoksa bölgesel
alt kümeler (örn. Avrupa vs. Güney Asya, yoğunlukları çok farklı
olabilir) için ayrı ayrı mı hesaplayalım — veri yoğunluğu bölgeden
bölgeye çok değişkense (ki muhtemelen öyle), ikincisi daha doğru
sonuç verir, birlikte karar verelim.

## Sıra

1. Önce granülarite/performans kararını netleştir (ham nokta örneklemi
   mi, hex centroid+weighted resampling mi) ve tahmini çalışma süresini
   raporla.
2. Küçük bir bölgesel alt küme (örn. sadece Avrupa+Ortadoğu) üzerinde
   prototip çalıştır, kümeleri görselleştirip bana göster.
3. Sonuç mantıklı görünürse (bilinen hub'larla örtüşüyorsa) tüm veriye
   ölçekle.
