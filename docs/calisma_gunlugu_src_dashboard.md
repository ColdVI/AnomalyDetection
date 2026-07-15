# Çalışma Günlüğü: src (Veri Pipeline'ı) ve Dashboard

Bu günlük src klasöründeki veri pipeline'ını (ingestion, silver, gold, common,
processing) ve Dashboard klasöründeki canlı harita uygulamasını kapsıyor. ML
özellik mühendisliği ve model çalışması (src/ml, adsb) kapsam dışı tutuldu,
çünkü o tamamen ayrı bir hat.

Günlük, git geçmişindeki gerçek commit tarihleri/mesajları ve decisions.md
dosyasındaki tarihli mimari karar kayıtları (ADR) üzerinden hazırlandı. İş
aslında 2026-06-29 ile 2026-07-14 arasında 16 takvim gününe yayılmıştı;
aktivitenin en az olduğu iki gün komşu güne katıştırılarak 14 güne indirildi
(1. gün ve 5. gün).

---

## 1. Gün: 2026-06-29 / 2026-06-30. Literatür taraması ve mimari planlama

Literatür taraması ve planlama: OpenSky ile adsb.lol karşılaştırıldı (kimlik
doğrulama yükü, günlük kredi limiti, Türkiye hava trafiği kapsamı) ve adsb.lol
seçildi. Etiketli UAV veri setleri de araştırıldı: generic MAVLink örnekleri
yerine ALFA (fault ground truth) ve UAV Attack (benign/malicious saldırı
etiketleri) seçildi, çünkü ikisi de ölçülebilir bir anomali tespiti çalışması
yapmaya izin veriyordu (ADR-001, 29 Haziran).

Depolama katmanı için de bir araştırma yapıldı: mimari diyagramın öngördüğü
"Bronze, Silver, Gold tamamı MinIO'da" tasarımı ile ilk yazılan yerel disk
(data/bronze) implementasyonu karşılaştırıldı ve MinIO'ya geçilmesine karar
verildi. Test edilebilirlik için client injection (fonksiyonlara "client"
parametresi geçme) ve bellek içi sahte istemci (FakeMinioClient) fikri bu
aşamada planlandı (ADR-002, 30 Haziran).

Uygulama tarafında ise ingestion katmanının ilk iskeleti kuruldu (bronze
katmanı için ingestion temeli açıldı).

---

## 2. Gün: 2026-07-01. Bronze, Silver, Gold mimarisinin kurulması

Bugün önce bir planlama kararı verildi: Bronze katmanı artık sadece ham veri
tutacak, ayrıştırma ve provenance işi Silver'a taşınacak; coğrafi filtre
pipeline'dan tamamen kaldırılacak; kaynak özel parser'lara ek olarak yeni bir
veri seti eklemeyi tek dosyaya indiren "generic parser" tasarlanacak. Bu,
önceki tasarımın (Bronze'da konu bazlı parse etme) bilinçli olarak tersine
çevrilmesiydi (ADR-003). Aynı gün Gold katmanının şeması da netleşti: her
kaynağı 7 ortak kolon artı 3 metadata kolonuna hizalayan tek bir tablo
(ADR-005). Ekip görev dağılımı da netleşti: adsb tarihsel veri Metehan'da,
adsb gerçek zamanlı Yusuf'ta, ALFA ve UAV Attack Anıl'da.

Bu, projenin en yoğun günlerinden biriydi, 13 ayrı commit atıldı. Bronze
loader'ları (Faz 2-5) yazıldı ve MinIO'ya taşındı, 33 test geçti. alt_baro
alanındaki karışık tip Parquet hatası düzeltildi. Bronze kuralını ihlal eden
bir alan (on_ground) Bronze'dan kaldırıldı. io.py, minio_io.py olarak yeniden
adlandırıldı ve coğrafi filtre tamamen kaldırıldı. ALFA ve UAV Attack için
gerçek veriyle doğrulanmış Silver ve Gold referans implementasyonu yazıldı.
Metehan ve Yusuf'un Silver parser'ları, generic parser ve genel bir temizlik
tamamlandı. Silver'dan Gold'a 7 artı 3 ortak şema hizalaması yazıldı
(src/gold/unify.py). Dashboard tarafında da ilk demo ve güncelleme
çalışmaları başladı.

---

## 3. Gün: 2026-07-02

10 commit atıldı. Docker/MinIO kurulmadan pipeline'ı çalıştırabilmek için
yerel depolamaya düşen bir yol eklendi. Gold streaming akışındaki bir hata
düzeltildi, gitignore güncellendi. Silver telemetrisi zenginleştirildi, Gold
velocity eşlemeleri iyileştirildi. Kaynak tipi ile MinIO önekinin ve Gold
kolon eşleme anahtarının tutarsız olduğu bir hata düzeltildi (adsblol
tarihsel ve gerçek zamanlı kaynaklar için). Dashboard tarafında harita
güncellendi, birkaç ekleme yapıldı ve yanlışlıkla silinen Dashboard dosyaları
geri yüklendi.

---

## 4. Gün: 2026-07-03. Mimari genişleme kararları ve gerçek zamanlı altyapı

Yine planlamayla başlayan bir gün. Gold'un ML için ve görselleştirme için iki
ayrı yoldan, kasıtlı olarak üretilmesine karar verildi: biri dar ve 10
kolonluk ortak şemayı üreten unify.py, diğeri zengin ve kaynağa özgü
kolonları koruyan build_features.py (ADR-006). Ayrıca gerçek zamanlı veri
akışında tek bir tüketici yerine iki ayrı Kafka tüketicisi kullanılmasına
karar verildi: biri Redis ve InfluxDB'ye yazan adsblol_consumer, diğeri MinIO
Bronze'a yazan Dashboard'daki minio_archiver. Bu ayrım sorumlulukları
birbirinden bağımsız kılıyor, biri çökerse diğeri etkilenmiyor (ADR-007).

Uygulama tarafında gerçek zamanlı veri için saklama süresi mantığı, sahte
Minio istemcisinin yaşam döngüsü ve Docker kaynak limitleri eklendi. Dashboard
tarafında ise Türkiye kutusundan (bbox) dünya ölçeğine geçildi.

---

## 5. Gün: 2026-07-04 / 2026-07-05

4 Temmuz'da OpenSky, adsb.lol'a alternatif ikinci bir veri kaynağı olarak
Dashboard'a entegre edildi. 5 Temmuz'da bu kapsamda (src veya Dashboard'la
ilgili) kayıtlı bir commit yok.

---

## 6. Gün: 2026-07-06

Gold tarafında optimizasyonlar yapıldı. Dashboard tarafında sinyal tazelik
(staleness) gösterimi iyileştirildi, sinyalin ne kadar eski olduğu artık
daha doğru yansıtılıyor.

---

## 7. Gün: 2026-07-07

Dashboard Docker'a taşındı ve Kafka, Zookeeper bağımlılığını kaldıran KRaft
moduna geçirildi; bu geçişin hazırlığı da aynı gün yapıldı. Ayrıca yerdeki
uçakları askeri veya sivil uçaklar gibi açılıp kapanabilir hale getiren bir
filtre eklendi.

---

## 8. Gün: 2026-07-08

Dashboard'a OpenSky için OAuth2 desteği eklendi, Kafka'nın yerel ağda
paylaşılması sağlandı, uçak çizimi/tıklama/renklendirme iyileştirildi ve
önceki uçuşlar listesi eklendi. Ayrıca kaynak ekleme işlemi kolaylaştırıldı.

---

## 9. Gün: 2026-07-09. En yoğun gün: pipeline sağlamlaştırma ve Dashboard'da özellik patlaması

15 commit atıldı, projenin en yoğun günüydü. Bronze'a dokunmadan Silver ve
Gold'un tekrar tekrar çoğalmasını önleyen bir düzeltme yapıldı ve bu iş
otomasyona bağlandı. Native InfluxDB kurulum kalıntıları kaldırılarak
Docker'a tam geçişin ilk adımı atıldı. MinIO'nun gerçek zamanlı veriyi
arşivleyip düzenli aralıklarla Silver'a çevirmesi sağlandı. Dashboard'a aynı
günde üç büyük özellik birden eklendi: acil durum uyarısı, CSV olarak dışa
aktarma ve geçmişi tekrar oynatma (replay). Son olarak adsb.lol'daki gibi bir
irtifa renklendirmesi eklendi ve Kafka'nın bellek limiti artırıldı.

---

## 10. Gün: 2026-07-10. Kapsam daraltma ve proje temizliği

Eski ML ve ADS-B denemeleri arşivlendi, aktif çalışma alanı ADS-B altyapısına
indirildi; bu esas olarak ML tarafını etkiledi ama repo genelinde bir
temizlikti. Arşivleme sırasında yanlışlıkla taşınan Dashboard'a özel dosyalar
geri çıkarıldı. Dashboard'da uçak tipini ayrı ve seçilebilir hale getiren bir
özellik eklendi. Karar kayıtları dosyası (decisions.md) güncellendi.

---

## 11. Gün: 2026-07-11. Test günü

İrtifa ve replay'de yaşanan yarış durumları düzeltildi, havayolu firma
filtresi eklendi, koyu tema ve durum çubuğu iyileştirildi. En önemlisi,
Dashboard için sıfırdan 182 testlik hermetik bir birim test paketi yazıldı.
Bunun dışında birkaç küçük düzeltme daha yapıldı.

---

## 12. Gün: 2026-07-12

Bu kapsamda kayıtlı bir commit yok. O gün üretilen tek doküman ADS-B ve ML
tarafına ait bir durum teşhis ve yol haritası belgesiydi, bu günlüğün
kapsamının dışında kaldığı için burada yer almıyor.

---

## 13. Gün: 2026-07-13

Dashboard klasörü temizlendi ve ADS-B'ye özgü isimlendirmeler daha genel
hale getirildi. Tarayıcı sekme başlığı sadece "Dashboard" olarak
sadeleştirildi. Durum çubuğundaki saat artık saniye saniye ilerliyor. 15
saniyelik "tick" aralığının nereden geldiği ve neyi kapsadığı yorum olarak
belgelendi, bu sırada OpenSky aralığıyla ilgili bir yanlışlık da (300 değil
90 saniye olduğu) düzeltildi.

---

## 14. Gün: 2026-07-14 (bugün). Modüler refactor, kapsamlı test ve dokümantasyon

Tek parça halindeki 4292 satırlık app.py dosyası kademeli olarak, birkaç
adımda, texts, styles, constants, layout, server ve api modüllerine
bölündü. Ayrıca artık kullanılmayan native Windows kurulum dosyaları
kaldırıldı, sadece Docker üzerinden kurulum bırakıldı; son bir kontrolde de
ölü hale gelmiş re-export'lar temizlendi.

Test tarafında src pipeline'ının tüm test dosyaları gözden geçirildi ve test
sayısı 129'a çıkarıldı. Daha önce hiç test edilmeyen alanlar kapatıldı: yerel
depolama istemcisi, adsb.lol üretici scripti, kesinti sonrası kaldığı yerden
devam edebilme (checkpoint/resume) sistemi ve askeri uçak bayrağının bit
mantığı gibi. Kullanılmayan ve kurulu minio paketiyle artık çalışmayan bir
saklama süresi testi de kaldırıldı.

Son olarak Dashboard klasöründeki tüm fonksiyon ve callback'lerin (90'ın
üzerinde) bir envanteri çıkarıldı ve gezilebilir bir referans dokümanı
hazırlandı. Projede kullanılan yazılım tasarım desenleri de (Singleton,
Adapter, Strategy, yayıncı/abone modeli, önbellekleme, Facade, Observer,
Repository, Factory Method, Medallion mimarisi gibi) tespit edilip
belgelendi.
