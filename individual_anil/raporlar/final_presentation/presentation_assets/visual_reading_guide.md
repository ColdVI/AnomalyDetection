# Görsel okuma rehberi

## S01 — Pencere, olay ve uçuş kavramı

Görsele soldaki “Pencere” kartından başlanmalı ve oklar izlenerek “Olay” ile “Uçuş” düzeylerine geçilmelidir. Yatay yön bir zaman ekseni değil, değerlendirme zinciridir; dikey eksen yoktur. Doğru yön, tek pencere skorundan açık bir olay politikasına ve ardından yalnız gerektiğinde uçuş özetine ilerlemektir. Ana bulgu, bu üç düzeyin birbirinin yerine kullanılamamasıdır. Uçuşta alarm görülmesi event recall olarak yorumlanmamalıdır.

## S03 — Üç şeritli proje kronolojisi

Önce üstteki “Veri ve gerçeklik” şeridine, ardından model ve değerlendirme şeritlerindeki aynı sütuna bakılmalıdır. Soldan sağa proje olgunlaşırken, dikey konum üç paralel çalışma alanını ayırır. İyi yön sağa doğru truth, bağımsızlık ve gerçek olay değerlendirmesinin güçlenmesidir. Ana bulgu, sonuçların yalnız daha karmaşık modelle değil, veri ve metrik düzeltmeleriyle güvenilir hâle gelmesidir. Düğüm sayıları deney adedi veya başarı oranı değildir.

## S04 — Veri seti fizibilite matrisi

Önce bir sütun seçilip o veri setinin beş ölçütteki renk dizisi yukarıdan aşağı okunmalıdır. X ekseni veri setlerini, Y ekseni olay truth’u, normal çeşitlilik, bağımsız grup, gözlenebilirlik ve doğal etiket ölçütlerini gösterir. Yeşil daha güçlü bilimsel dayanak, sarı dikkat gerektiren sınır, kırmızı kritik eksik anlamındadır. RflyMAD’in bu çalışma için en kapsamlı etiketli kaynak, ADS-B’nin ise gerçek fakat doğal etiketsiz trafik olduğu görülür. Hücreler model performansı veya kesin kalite puanı değildir; audit belgelerinden türetilmiş nitel kodlamadır.

## S05 — RflyMAD veri kompozisyonu

Görsele önce toplam veri bileşimini veren ana panelden, sonra domain ve fault ailesi kırılımlarından başlanmalıdır. Kategorik X eksenleri veri bölümlerini, Y ekseni uçuş sayılarını gösterir. Tek bir domainin baskın olmaması çeşitlilik açısından iyi, ancak domainler arasındaki büyük dağılım farkları genelleme açısından zordur. Ana bulgu, RflyMAD’in büyük fakat homojen olmayan bir havuz olduğudur. Bu Full-v2 kompozisyonu v3.1 rol sayıları veya model başarısı gibi okunmamalıdır.

## S06 — Metrik merdiveni

Görsel soldan sağa ve aşağıdan yukarı yükselen dört basamak izlenerek okunmalıdır. Basamakların konumu sayısal eksen değil, değerlendirme olgunluğudur. İyi yön Gaussian fit’ten pencere sıralamasına, ardından olay politikasına ve nihayet recall–alarm yükü çiftine ulaşmaktır. Ana bulgu, ROC-AUC veya NLL’nin tek başına kullanılabilir alarm sistemi kanıtı olmadığıdır. Basamak yüksekliği yöntemlerin göreli performansını göstermez.

## S07 — ADS-B rota artefaktı

Önce değerlendirilen uçuş sayısına, sonra tetiklenen uçuş ve olay sayılarına bakılmalıdır. Kategoriler X ekseninde, sayı veya oranlar Y ekseninde gösterilir. Daha az tetik tek başına iyi değildir; tetiklerin fiziksel olarak doğrulanması gerekir. Ana bulgu, 24 rota olayının 24’ünün de düşük hızlı bearing kararsızlığıyla açıklanmasıdır. Bu görsel doğrulanmış 24 doğal anomaly bulunduğu şeklinde yorumlanmamalıdır.

## S08 — Truth/parser önce ve sonra

Önce üstteki kırmızı “düzeltme öncesi” çizgi, sonra alttaki geç başlayan kırmızı bölüm okunmalıdır. Yatay eksen şematik uçuş zamanını; renk ise arızanın aktif sayılıp sayılmadığını gösterir. İyi durum, arızanın t=0’dan değil gerçek başlangıç noktasından itibaren aktif olmasıdır. Ana bulgu, 2.712 uçuşun yeniden parse edilmesini gerektiren truth hatasının model metriklerinden önce çözülmüş olmasıdır. Bu gerçek bir uçuş sinyal grafiği değildir; üzerinde belirtildiği gibi şematik gösterimdir.

## S09 — Yöntem trade-off’u

Noktaların hedef bölgesine göre konumundan başlanmalıdır. X ekseninde yanlış alarm/saat, Y ekseninde recall vardır; sola ve yukarı hareket tercih edilir. İyi bir yöntem hem hedef recall’ın üstünde hem alarm bütçesinin altında kalmalıdır. Ana bulgu, AE ve TCN’nin tek tek sinyal bulmasına karşın iki koşulu birlikte karşılayamamasıdır. Bu development Full-v2 sonucu, son probabilistic v3.1 sonucu veya eğitilmiş–rastgele kontrolünün geçtiği bir deney gibi sunulmamalıdır.

## S10 — Benzer grupları ayıran veri bölünmesi

Kartlar soldan sağa eğitim, doğrulama, bağımsız normal test, anomali geliştirme ve kapalı nihai test olarak okunmalıdır. Kart içindeki ilk sayı uçuş, ikinci sayı bağımsız grup adedidir; geometrik konum performans ekseni değildir. İyi durum, kesik rol sınırlarının hiçbir kaynak veya grup tarafından aşılmamasıdır. Ana bulgu kaynak ve grup kesişiminin sıfır olmasıdır. Son kartın varlığı nihai testin değerlendirildiği anlamına gelmez; veri kapalı tutulmuştur.

## S11 — Eğitilmiş–rastgele model kontrolü

Önce soldaki UAV-SEAD kartı, sonra sağdaki RflyMAD kartı ve en alttaki ρ=0,80 kontrol eşiği okunmalıdır. Bu kartta X/Y ekseni yoktur; iki farklı korelasyon çifti sayısal olarak karşılaştırılır. Düşük korelasyon belirgin genlik kestirmesi açısından daha iyidir; SEAD’in yaklaşık 0,964/0,965 değerleri kritik artefaktı gösterir. RflyMAD’in 0,3288/0,3495 değerleri bu özel kontrolün altında kaldığını gösterir. Nokta skorları bulunmadığı için bu bir scatter değildir ve Rfly modelinin başarılı detector olduğunu kanıtlamaz.

## S12 — ADS-B araştırma ilerlemesi

Görselde araştırma aşamaları soldan sağa izlenmelidir. Yatay akış tarihsel ilerlemeyi, katmanlar yöntem ve değerlendirme değişimlerini gösterir. İyi yön sentetik recall’a odaklanmaktan gerçek trafik alarm yükü ve fiziksel bağlama geçiştir. Ana bulgu, ilk yüksek sentetik sonucun doğal alarm yükü nedeniyle kabul edilmemesidir. Görsel güncel, kabul edilmiş bir ADS-B anomaly modeli bulunduğu şeklinde yorumlanmamalıdır.

## S13 — Olasılıksal değerlendirme dönüşümü

Her satırda önce sarı soldaki eski sözleşme, sonra ok izlenerek yeşil sağdaki güncel sözleşme okunmalıdır. X ekseni yerine önce–sonra karşılaştırması, Y yönünde dört sözleşme boyutu vardır. İyi yön kaynak içi bölünmeden bağımsız gruplara, tüm uçuş truth’undan gerçek aralıklara ve tek alarmdan olay politikasına geçiştir. Ana bulgu, daha sıkı değerlendirmenin daha düşük ama daha güvenilir sonuç verebilmesidir. v2 ve v3.1 sayıları aynı testte model gerilemesi gibi karşılaştırılmamalıdır.

## S14 — Dört veri seti detection özeti

Önce panel başlıkları okunmalı; A ve B sıralama, C olay yakalama, D normal alarm yüküdür. X ekseninde veri setleri, Y ekseninde paneline göre AUC, event recall veya yanlış olay/normal saat bulunur. İyi yön AUC ve recall için yukarı, yanlış olay/saat için aşağıdır. Ana bulgu, RflyMAD dışında yüksek uçuş/pencere sinyalinin gerçek olay yerelleştirmesine çoğunlukla dönüşmemesidir. ALFA’daki sıfır yanlış olay yalnız 0,131 normal saat exposure içerdiğinden güçlü sıfır-alarm kanıtı sayılmamalıdır.

## S16 — Gaussian forecaster mimarisi

Soldaki 32-adımlık geçmiş pencereden başlanıp oklarla olay katmanına kadar ilerlenmelidir. Yatay yön model ve karar zincirini gösterir; sayısal eksen yoktur. İyi tasarım, yalnız normal geçmişten kanal başına μ ve σ tahmin edip eksik hedefleri maskeyle dışlamaktır. Ana bulgu, Gaussian NLL’nin henüz olay olmadığı; persistence, K-of-N veya CUSUM ile olaya dönüştürülmesi gerektiğidir. Şema belirli bir threshold veya politikanın üstünlüğünü iddia etmez.

## S17 — Eğitim ve checkpoint grafiği

Önce doğrulama eğrisinin en düşük noktasına ve seçilen epoch işaretine bakılmalıdır. X ekseni epoch, Y ekseni masked Gaussian NLL’dir; daha düşük doğrulama NLL normal veriyi daha iyi açıklayan checkpoint anlamına gelir. Ana bulgu 30. epoch’un yalnız normal validation sözleşmesiyle seçilmesidir. Eğitim ve doğrulama eğrilerinin düşmesi fit açısından olumlu olsa da anomaly yakalama sonucu değildir. Bu grafik event recall veya yanlış olay yükünü göstermez.

## S17 — Negatif Gaussian NLL

Önce sıfır yatay çizgisi ve mavi σ=0,2 eğrisinin taralı negatif bölgeye girdiği alan incelenmelidir. X ekseni standartlaştırılmış hata `z`, Y ekseni kodda kullanılan `0.5·z²+log(σ)` değeridir. Küçük hata ve σ<1 durumunda negatif NLL matematiksel olarak beklenir; hata büyüdükçe tüm eğriler yükselir. Ana bulgu, negatif validation NLL’nin yazılım hatası olmadığıdır. Grafik deterministik formül çizimidir ve model deney sonucu olarak yorumlanmamalıdır.

## S18 — Checkpoint ve öğrenme kontrolü

Önce seçilen epoch ve NLL kartları, sonra iki yeşil korelasyon kartı okunmalıdır. Kartların konumu eksen değildir; her biri farklı bir doğrulama değerini gösterir. İyi yön korelasyonların 0,80 kontrol eşiğinin altında olmasıdır. Ana bulgu, checkpoint’in normal doğrulama NLL ile seçilmesi ve belirgin trained–random/genlik kestirmesinin işaretlenmemesidir. Bu kontrolün geçmesi event detection başarısı anlamına gelmez.

## S19 — Uçuşta en az bir pencere alarmı matrisi

Önce satırların gerçek normal/anomali uçuşu, sütunların alarm yok/alarm olduğunu doğrulayarak matrise bakılmalıdır. Hücreler uçuş sayılarını gösterir; klasik event confusion matrix eksenleri değildir. Anomali uçuşlarda %82,94 alarm ilk bakışta yüksek görünür, fakat normal uçuşlarda %84,62 alarm olması kötüdür. Ana bulgu, uçuşta tek pencere alarmının ayırt edici bir işletim noktası üretmemesidir. %82,94 event recall olarak söylenmemelidir.

## S20 — Recall ve yanlış olay oranı

Grafiğe sol üst yönün hedef olduğu bilgisiyle başlanmalıdır. X ekseni yanlış olay/normal saat, Y ekseni event recall’dır; yukarı iyi, sağa gitmek kötüdür. Noktalar farklı sabit olay politikalarını ve bütçe seçeneklerini gösterir. Ana bulgu, maksimum %58,71 recall noktasının 9,153 yanlış olay/saat üretmesi ve dengeli bir nokta bulunmamasıdır. Grafik final fault test sonucu değildir; geliştirme ve bağımsız normal test rollerine dayanır.

## S20 — Normal alarm yükü

Önce her politikanın turuncu bar yüksekliği karşılaştırılmalıdır. X ekseni olay politikalarını, Y ekseni en az bir yanlış olay taşıyan normal uçuş oranını gösterir; daha düşük daha iyidir. Ana bulgu, yanlış olayların bağımsız normal uçuşlara yayılmasıdır. Bu sonuç operatörün kaç uçuşta gereksiz alarm göreceğini tamamlayıcı olarak anlatır. Bar oranı false events/normal hour ile aynı metrik değildir.

## S20 — Yakalanan Motor olayı

Önce truth aralığı, sonra aynı zaman bölümündeki standardized NLL ve üretilen olay işaretleri izlenmelidir. X ekseni uçuş içi zaman, iki Y paneli standardized ve raw Gaussian NLL’yi gösterir. İyi durum predicted event’in truth aralığıyla kesişmesidir. Ana bulgu, bazı Motor olaylarında skor artışının karar katmanına dönüşebildiğidir. Bu tek örnek toplam Motor recall’ını veya genel başarıyı kanıtlamaz.

## S20 — Kaçan Real/Sensor olayı

Önce truth aralığı bulunmalı, ardından aynı aralıkta skorun ve olay işaretinin neden yeterli olmadığı incelenmelidir. X ekseni uçuş içi zaman, Y eksenleri standardized ve raw Gaussian NLL’dir. İyi durum truth ile eşleşen olay olurdu; bu örnekte yoktur. Ana bulgu, Real domain ve Sensor ailesindeki sırasıyla %7,81 ve %5,45 recall kör noktasının somutlaşmasıdır. Tek vaka tüm Real/Sensor dağılımının şekli gibi genellenmemelidir.

## S20 — Final sonuç kartı

Önce üst soldaki %43,27 event recall, sonra 0,654 yanlış olay/normal saat ve sağdaki Real/Sensor kör noktaları okunmalıdır. Kartta sayısal eksen yoktur; renkler recall, alarm yükü, risk ve karar türlerini ayırır. Ana referans noktası ancak recall ile normal alarm yükü birlikte okunduğunda anlamlıdır. Mor kart en yüksek tarama recall’ının kabul edilemez alarm maliyetini gösterir ve kırmızı karar kartı “Hedef karşılanmadı” sonucunu verir. Kart, kapalı nihai test verisi üzerinde başarısızlık iddia etmez; final test hiç açılmamıştır.
