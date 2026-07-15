# İHA Dashboard — Güncelleme Notu (Claude.ai sohbetinden Claude Code'a geçiş)

**Kapsam:** Bu dosya, `FULL_PROJECT_HANDOFF.md`'den SONRA, uzun bir Claude.ai sohbetinde yapılan
tüm çalışmayı özetler. Claude Code'a bu iki dosyayı (eski handoff + bu dosya) birlikte ver.

---

## 1. Bu sohbette neler değişti (özet)

Eski handoff dosyasından beri **TTL mantığı, rota sorgulama, sinyal tazeliği** gibi konularda
ciddi mimari değişiklikler yapıldı. Kod dosyalarının kendisi zaten çok detaylı yorumlarla
belgelendi (neden, ne zaman, hangi hata bulundu) — Claude Code repo'yu okuyunca bu bağlamın
çoğuna zaten sahip olacak. Burada sadece **büyük resmi ve son durumu** özetliyorum.

## 2. Redis'in geçerlilik modeli — 3 iterasyon geçirdi

1. **TTL (ilk hâl):** Her kayıt Redis'e bir `ex=` (yaşam süresi) ile yazılıyordu. Sorun:
   `iha:active_flights` kümesine sadece `SADD` yapılıyordu, hiç `SREM` yoktu — TTL dolan
   tekil kayıtlar kümede "hayalet" olarak kalmaya devam ediyordu.
2. **Zaman penceresi:** TTL yerine sabit saniyelik bir pencere (`WINDOW_SEC`) denendi — hayaletleri
   çözdü ama pencere birden fazla üretim cycle'ını kapsadığında ekranda **%13'e kadar fazlalık**
   yarattı (birden fazla cycle'ın birleşimi gösteriliyordu).
3. **(GÜNCEL) Cycle-ID:** `uav_producer.py` artık her kayda bir `cycle_id` (artan sayaç) ekliyor.
   `dashboard_consumer.py`, gelen `cycle_id` değiştiğinde "önceki cycle tamamlandı" der ve o
   cycle'da hiç görünmeyen eski kayıtları **o an** siler — saniye tahmini YOK, ortalama fazlalık
   ~%0'a indi. **ÖNEMLİ:** "Tüm cycle'ı bellekte toplayıp tek seferde atomik yaz" da denendi
   (tutarsızlığı komple bitirmek için) ama bu bir CYCLE'lık (~56sn) gecikme yarattı — **bu bir
   regresyondu, geri alındı**. Mesajlar hâlâ geldikçe anında yazılıyor, sadece cycle geçişinde
   kısa (birkaç saniyelik) bir örtüşme kabul edilen bir bedel.

Redis şeması artık:
- `iha:state:{icao24}` — TTL YOK, geçerlilik cycle-id mantığıyla belirleniyor
- `iha:active_flights` — aktif icao24 kümesi
- `iha:route:{callsign}` — rota cache'i (12sa bulundu / 1sa bulunamadı)

## 3. Rota (kalkış/varış) sorgulama — iki kaynaklı, doğrulamalı

`adsbdb.com`'un statik callsign→rota veritabanı **bazen yanlış** (havayolları uçuş numaralarını
farklı günlerde farklı rotalar için yeniden kullanıyor — örn. WZZ43 gerçekte Kraków→Stavanger
iken adsbdb Londra→Budapeşte diyordu).

**Çözüm — iki katmanlı:**
1. **Önce adsb.lol'ün kendi routeset API'sini dener** (`https://api.adsb.lol/api/0/routeset`,
   POST). Bunu çalıştırmak için **iki şey zorunluydu** (çok uğraşarak bulundu):
   - İstek gövdesinde `callsign` YETMEZ, uçağın **gerçek anlık `lat`/`lng`'i de zorunlu**
   - `Origin: https://adsb.lol` + `Referer: https://adsb.lol/` başlıkları **zorunlu**
     (yoksa endpoint sessizce boş bir "201" dönüyor, format tahminiyle asla çözülmez)
2. **Başarısız olursa veya lat/lon yoksa**, sessizce `adsbdb.com`'a düşer.

**Ek doğrulama:** Bulunan rota, uçağın **gerçek anlık yönüyle** (`track`) karşılaştırılıyor
(`_route_is_plausible`, büyük daire başlangıç açısı formülü) — 90°'den fazla sapma varsa
(WZZ43'te 180° bulunmuştu) rota "şüpheli" işaretleniyor: metin altında uyarı, harita çizgisi
soluk/kesikli. **Rota hiçbir zaman gizlenmiyor** (heuristik kesin değil), sadece işaretleniyor.

Harita çizgisi düz değil — **great-circle (büyük daire) eğrisi** (`_great_circle_points`,
Ed Williams formülü), 180. meridyen geçişlerinde doğru "unwrap" ediliyor.

## 4. Sinyal tazeliği — "ölü sinyal" ayrımı

`readsb` (adsb.lol'ün altyapısı), sinyali kesilen bir uçağı **60 saniyeye kadar** listede tutar.
Bu yüzden bir uçak listede "var" görünse bile, verisi bayat olabilir. Çözüm: `seen_pos` alanı
(adsb.lol) / `last_contact`'tan hesaplanan eşdeğer (OpenSky) artık `signal_age_sec` olarak
şemaya ekleniyor, haritada bu değere göre uçak ikonu **soluklaştırılıyor** (10sn altı net,
40sn+ belirgin soluk). Bu, cycle-id temizliğinin **yakalayamadığı** bir durumu (zayıf sinyal
bölgesinde konum donarken diğer mesajların gelmeye devam etmesi) ortaya çıkarıyor.

## 5. Diğer önemli notlar

- **adsb.lol'ün "total aircraft" sayısı bizden yüksek çıkabilir** — biz bilerek yerdeki
  (`alt == "ground"`) uçakları filtreliyoruz, adsb.lol'ün kendi sayacı muhtemelen dahil ediyor.
- **Kafka bağlantı kopmaları** (`ConnectionResetError`, `Read timed out`) adsb.lol'ün dünya
  ölçekli sorgularında ara sıra oluyor — retry (3 deneme, 3sn bekleme) + medyan-tabanlı ölçüm
  ile ele alınıyor, kök sebep (adsb.lol sunucu tarafı mı, yerel ağ mı) kesin ayrılamadı ama
  büyük ihtimalle adsb.lol tarafı (küçük sorgular hep sorunsuzdu, büyük sorgu bazen yavaşladı).
- **OpenSky'nin kendi rota endpoint'i VAR ama İŞE YARAMAZ** — sadece geceki toplu işlemden
  önceki günlerin verisini veriyor, canlı uçuş için boş dönüyor. Bu yüzden route lookup'ımız
  kaynaktan bağımsız (hem adsb.lol hem OpenSky ile takip edilen uçaklar için aynı mekanizma).

## 6. DOCKER GEÇİŞİ İÇİN — bu oturumda düzeltildi

`app.py` ve `dashboard_consumer.py`'de Redis/InfluxDB/Kafka bağlantıları hardcoded
`"localhost"` idi — bu artık ortam değişkenleriyle ayarlanabilir:

| Ortam değişkeni | Varsayılan (Windows/tek makine) | Docker'da ne olmalı |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis container'ının servis adı (örn. `redis`) |
| `INFLUX_HOST` | `http://localhost:8086` | örn. `http://influxdb:8086` |
| `KAFKA_BOOTSTRAP` | `localhost:9092` | örn. `kafka:9092` |

`uav_producer.py` zaten `KAFKA_BOOTSTRAP` ortam değişkenini kullanıyordu (Docker'a hazırdı).

`app.py` içindeki `http://localhost:8000/api/...` çağrıları (Dash→FastAPI arası) **DEĞİŞMEDİ**
ve değişmemeli — FastAPI ve Dash aynı process içinde (`app.py`), aynı container'da kalmalı.

**Henüz yapılmadı, Docker geçişinde ele alınmalı:**
- MinIO bronze/silver/gold hiç bu dashboard tarafından kullanılmıyor (Yusuf'un/Metehan'ın işi,
  ayrı bir consumer pipeline'ı) — bu geçişte muhtemelen ayrı bir docker-compose servisi olacak.
- `KAFKA_SCHEMA.md`, ekip için güncel şema referansı — Docker'a geçerken bu dosyayı da taşı,
  `cycle_id` ve `signal_age_sec` alanları en son eklenenler, ekibin haberi olmayabilir.
