Bu yaklaşımla (ADS-B telemetri + fiziksel-tutarlılık residual'ları) tespit edebileceğimiz olası anomali türleri
A) Fiziksel-tutarlılık ihlalleri — asıl yaklaşımımızın çekirdeği
Dikey hız ↔ irtifa değişimi uyuşmazlığı — bildirilen dikey hız ile gerçek irtifa değişimi birbirini tutmuyor (sensör arızası, yanlış kalibrasyon, telemetri bozulması). Test ettik, orta güçte ayrım (1.03x-2.28x).
Yer hızı ↔ GPS-konum-türevi hız uyuşmazlığı — bildirilen hız ile lat/lon değişiminden hesaplanan gerçek hız tutmuyor. Test ettik, en güçlü sinyal (5x-10.75x) — bu kanal şu ana kadarki en güvenilir bulgumuz.
Track/yön ↔ GPS-konum-türevi kerteriz uyuşmazlığı — bildirilen yön ile gerçek hareket yönü tutmuyor (pusula/heading sensör hatası, yanlış bildirim). Test ettik, zayıf ayrım (1.07x-1.17x) — henüz güçlendirilmedi.
Dönüş hızı ↔ bank açısı tutarsızlığı — koordineli dönüş fiziğine (g·tan(roll)/v) uymayan dönüş oranı. Anormal manevra veya sensör hatası işareti. Kodladık ama roll_deg verisi seyrek (%28) olduğu için henüz gerçek veriyle test edilmedi.
Barometrik irtifa ↔ geometrik irtifa arasında normalin dışında sapma — basınç sensörü arızası, yanlış QNH ayarı, ya da GNSS bozulması. Henüz kodlanmadı, ölçülebilirlik tablosunda "ölçülebilir" olarak işaretli, sıradaki aday.
Fiziksel olarak imkânsız ivmelenme — saniyeler içinde 100 knot'tan 0'a düşme gibi, hiçbir uçağın gerçek performans zarfına sığmayan değişim. Kavramsal olarak mevcut residual'lardan türetilebilir, ayrı bir eşik/kural olarak henüz yazılmadı.
B) GPS/konum kaynaklı anomaliler
Ani konum sıçraması (GPS "jump") — alıcı hatası, multipath, jamming sonrası ani toparlanma.
Yavaş/sinsi konum kayması (kademeli spoofing) — tam olarak position_ramp_stealthy senaryomuzun simüle ettiği durum. Test ettik, şu ana kadarki EN ZAYIF sonuç (1.01x-1.03x) — bu, projenin en önemli açık maddesi: sinsi/kademeli sahtecilik tam da tasarım amacımızdı ama şu anki modeller bunu neredeyse hiç yakalamıyor.
Konum donması — uçak hareket ederken lat/lon sabit kalıyor (GPS kaybı/sensör donması).
Fiziksel olarak imkânsız rota — ani ters yön, "teleport" tarzı sıçramalar.
Deniz/boş bölge üstünde tutarsız iniş-kalkış paterni — konum-irtifa-hız üçlüsünün coğrafi bağlamla çelişmesi.
Çoklu-yol (multipath) kaynaklı kısa-süreli konum titremesi — gerçek anomali değil ama benzer görünen gürültü; residual'larla gerçek anomaliden ayrıştırılabilir mi, henüz test edilmedi.
C) İrtifa kaynaklı anomaliler
İrtifa donması — uçak hareket ederken irtifa hep aynı kalıyor. inject_freeze fonksiyonuyla üretilebilir, henüz ayrı senaryo olarak koşulmadı (şu an vertical_rate_frozen var, alt_frozen yok).
Ani/fiziksel olarak imkânsız irtifa sıçraması — saniyeler içinde binlerce feet.
Anlamsız irtifa değerleri — deniz seviyesinin çok altı, gerçekçi olmayan yükseklik (veri bozulması ya da parse hatası işareti).
Barometrik-geometrik irtifa arasında kalıcı/büyüyen sapma — kalibrasyon kayması (madde 5'in zaman-içinde-trend versiyonu).
on_ground bayrağı ile gerçek irtifa/hız tutarsızlığı — yerde derken hızlı hareket ediyor, ya da havadayken yerde işaretli. Doğrudan Silver şemasında mevcut, henüz kural yazılmadı.
D) Hız kaynaklı anomaliler
Uçak tipi için fiziksel olarak imkânsız hız — aircraft_type alanıyla çapraz kontrol edilip performans zarfı aşımı tespit edilebilir (henüz yapılmadı, tip-bazlı referans tablosu gerekir).
Havadayken ani hız-sıfırlanması — donma değil, gerçekten anlamsız "0 hız" bildirimi.
Ani hız-bias'ı (ofset hatası) — sensör kalibrasyon kayması. Test ettik, en güçlü sonucumuz bu kategoride (madde 2 ile aynı residual).
E) Transponder/protokol seviyesi anomaliler (ADS-B'ye özgü, fizik gerektirmez)
Acil durum squawk kodları — 7500 (kaçırma), 7600 (haberleşme kaybı), 7700 (genel acil durum). Doğrudan squawk/emergency alanından okunur, bizim residual yaklaşımımıza hiç ihtiyaç yok — en "ucuz" ve güvenilir sinyal ailesi, henüz dashboard'a bağlanmadı.
ICAO24 (transponder ID) çakışması/klonlanması — iki farklı uçağın aynı anda aynı transponder kimliğini yayınlaması (spoofing işareti).
Aynı ICAO24'ün coğrafi olarak imkânsız mesafede "aynı anda" görülmesi — madde 22'nin özel bir tespit yöntemi.
category alanı ile gerçek uçuş davranışı tutarsızlığı — kategori "büyük yolcu uçağı" ama davranış küçük/çevik bir platform gibi (ya da tam tersi, B6/B7 = drone kodlarının davranışsal doğrulaması — 2026-03-01'de gördüğümüz 25 kayıtlık B6 bulgusu tam bu kategoriye örnek, henüz incelenmedi).
Bütünlük/güvenilirlik metriklerinin (nic, nac_p, sil) ani düşüşü — uçağın kendi bildirdiği konum-güven skorunun bozulması, erken uyarı sinyali.
source_type değişimi (adsb_icao → mlat/tisb geçişi) — doğrudan sinyal kalitesi/görünürlük düşüşü işareti.
flags_stale bayrağının sık/uzun aktif olması — kesintili sinyal alım paterni.
F) Uçuş davranışı / manevra anomalileri
Fiziksel sınırların dışında ani dönüş — madde 4'ün eşik-tabanlı, roll verisi olmadan da uygulanabilir basit versiyonu (yalnız track'in kendi değişim hızına bakarak).
Aşırı dikey hız — tırmanma/alçalma performans limitlerinin dışında.
Havada beklenmeyen "dur-kalk" paterni.
Rota/koridor sapması — bir uçağın, aynı A-B hattını daha önce uçan uçaklardan belirgin şekilde sapması. Bu, benim residual-tabanlı yaklaşımımdan FARKLI bir yöntem (tipik-rota kümeleme) gerektiriyor — takımda Metehan'ın individual/metehan_geo/ altında ayrıca planladığı DBSCAN rota-kümeleme işi bu kategoriye giriyor, benim ADSB-1 hattımın parçası değil ama aynı ham veriyi kullanıyor, tamamlayıcı.
Zikzak/salınımlı hareket paterni — sensör gürültüsü mü gerçek kontrol sorunu mu ayrımı, henüz ele alınmadı.
G) Segmentasyon kaynaklı, "anomali gibi görünen ama veri artefaktı olan" durumlar (dürüstlük için ayrı kategori)
Uzun sinyal kaybı sonrası "ışınlanma" gibi görünen sıçrama — aslında iki ayrı temas penceresi, gerçek anomali değil. flags_new_leg uyuşmasının %60.4 çıkması tam bu belirsizliğin göstergesi.
Segmentasyon eşiği çok gevşekse iki gerçek uçuşun birleşmesi.
Eşik çok sıkıysa tek bir gerçek uçuşun yanlışlıkla ikiye bölünmesi.
H) "Anomali değil, veri kalitesi" — karıştırılmaması gereken durumlar
roll_deg gibi seyrek alanların yokluğu — kapsama %28, veri eksikliği anomali değil.
Havaalanı yakınında on_ground geçişlerinin ürettiği normal ama "ani" görünen irtifa/hız değişimleri.
Kısaca: 37 maddenin büyük kısmı kavramsal olarak ulaşılabilir (veri zaten elimizde), ama şu ana kadar GERÇEKTEN test ettiğimiz yalnızca 5 tanesi (A.1, A.2, A.3, B.8, C-benzeri altitude_dropout) — ve onlardan bile sadece hız-bozulması (A.2/D.20) net bir sinyal veriyor. En değerli/en zor kategori olan "sinsi spoofing" (B.8) şu anki en zayıf noktamız — asıl kanıtlamak istediğimiz şey tam olarak bu, ve henüz başaramadık.