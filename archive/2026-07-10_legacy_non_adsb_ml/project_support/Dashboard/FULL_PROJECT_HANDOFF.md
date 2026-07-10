# İHA (UAV) Staj Projesi — Tam Aktarım Dosyası
**Kapsam:** 22 Haziran 2026 (proje başlangıcı) — 2 Temmuz 2026 (bu dosyanın yazıldığı an)
**Amaç:** Bu dosya, yeni bir Claude sohbetine yapıştırılıp/eklenip, önceki tüm çalışmanın bağlamının anında aktarılması için hazırlandı.

---

## 1. Proje Özeti

Veri madenciliği & big data alanında 3 stajyerlik bir İHA (İnsansız Hava Aracı) projesi.
- **Süre:** 8 hafta (Hafta 1-4 ortak proje, Hafta 5-8 bireysel projeler)
- **Konu:** İHA uçuş verisi analiz platformu — anomali tespiti (arıza, GPS spoofing/jamming) + canlı dashboard
- **Orijinal plan:** Kafka tabanlı ETL + Spark Streaming (Lambda mimarisi) + OpenSky/MAVLink verisi — **bu plan zamanla önemli ölçüde değişti** (aşağıda açıklanıyor).

## 2. Ekip ve Roller

| Kişi | Sorumluluk |
|---|---|
| **Sen (bu sohbetin sahibi)** | Canlı ADS-B dashboard (Kafka → Redis/InfluxDB → Dash) |
| **Metehan** | adsb.lol tarihsel veri (bronze/silver/gold lakehouse, MinIO) |
| **Yusuf** | adsb.lol canlı veri → Kafka → MinIO bronze (arşivleme, model eğitimi için) |
| **Anıl** | ALFA + UAV Attack Dataset → model eğitimi (arıza/GPS saldırı tespiti) |

Ekip, ayrı ayrı Claude Code oturumlarıyla çalışıyor; `PIPELINE_PLAN.md` adında paylaşılan bir mimari doküman var (bronze/silver/gold, MinIO, generic parser deseni). Detaylı analiz için bkz. Bölüm 11.

## 3. Veri Kaynakları

| Kaynak | Ne için | Format |
|---|---|---|
| **ALFA** (CMU AIRLab) | UAV arıza/anomali tespiti (model eğitimi) | processed.zip → CSV (per-topic), 47 sequence |
| **UAV Attack Dataset** (Whelan ve ekibi, IEEE) | GPS spoofing/jamming tespiti (model eğitimi) | ulog2csv çıktısı, 767 CSV, ~19 log |
| **adsb.lol / readsb** | Canlı dashboard + tarihsel arşiv | Canlı: point API JSON. Tarihsel: gzip JSON trace dosyaları (.tar) |
| ~~OpenSky, MAVLink~~ | Orijinal plandaydı, **kullanılmadı** | — |

### 3.1 ALFA Detayları
- Carbon-Z T-28 sabit kanat platform, Pittsburgh'da toplanmış
- 8 arıza tipi: engine, rudder (sol/sağ/orta), elevator, aileron (sol/sağ/ikisi)
- `processed.zip` içinde: 47 bag + 47 mat + 1590 CSV (her sequence için ~34 topic, sadece 6'sı bizim kullandığımız: global_position, nav_info/{roll,pitch,airspeed,yaw}, failure_status)
- Etiketleme: klasör adından varsayılan (engine_failure, no_failure, vb.) + `failure_status-*.csv`'den satır-bazlı gerçek arıza başlangıcı

### 3.2 UAV Attack Dataset Detayları
- İki senaryo kümesi: "Simulated - OTU Survey" (601 dosya, 6 platform: VTOL/TAIL/QUAD-SITL/H480/PLANE/QUAD-HITL) + "Live GPS Spoofing and Jamming" (185 dosya, gerçek uçuş)
- Etiketler: benign / gps_spoofing / gps_jamming / ping_dos(unknown olarak bırakıldı) — **klasör bazlı**, dosya yolunun EN YAKIN klasörüne bakarak (tüm path'e değil — kritik bug fix, bkz. Bölüm 6)
- Zaman damgası: `timestamp` kolonu PX4'ün açılıştan beri sayan mikrosaniye sayacı, GERÇEK UTC DEĞİL — gerçek UTC için `vehicle_gps_position.time_utc_usec` kullanılıyor

### 3.3 adsb.lol Detayları
- Canlı: `https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}` — Türkiye merkezli (39, 35), 500nm yarıçap
- Tarihsel: `adsb_raw/` altında 7 tar dosyası (v2024.09 - v2026.06), her biri `traces/XX/trace_full_*.json.gz` — 14 elemanlı sabit-pozisyonlu array, ICAO24 bazlı günlük trace
- 97.8 milyon satır, 67.577 uçak, 24 saatlik veri doğrulandı (tüm konvansiyonel uçak, hiç UAV/drone kategorisi yok — beklenen)

## 4. Faz 1 — Colab Denemeleri (Proje başlangıcı, önceki oturum)

Colab'da tam bir Lambda mimarisi denendi: Kafka + Spark(*) + Redis + InfluxDB + MinIO + FastAPI(SSE) + Dash + PyDeck.

**(*) ÖNEMLİ: Spark Streaming HİÇBİR ZAMAN gerçekten kullanılmadı.** Orijinal plandaydı, Colab'da da denenmedi — Kafka'dan sonrasını hep düz Python consumer işledi. Neden gerekmediği ayrıca tartışıldı: bizim ölçeğimizde (~200-300 uçak/15sn, dünya geneli olsa bile ~saniyede 1000 event) Spark'ın çözdüğü sorunlar (milyon/saniye ölçek, dağıtık işleme, kompleks windowed agregasyon) hiç yaşanmıyor. Asıl darboğaz olsaydı bile çözüm InfluxDB batch-write moduna geçmek + Kafka partition'larıyla paralel consumer olurdu, Spark değil.

### Mimari evrimi (Colab'da):
- HDFS yerine **MinIO** (S3-uyumlu, kurulumu kolay)
- WebSocket yerine **SSE** (tek yönlü, dashboard için yeterli)
- Streamlit yerine **Dash** (karmaşık callback mimarisi için)
- Folium yerine **PyDeck → sonra Plotly Scattermapbox** (PyDeck TextLayer render sorunları)

### Çözülen teknik sorunlar (Colab, Linux):
- Kafka indirme: dlcdn.apache.org'dan 3.9.2 (3.7.0 CDN'de yoktu)
- InfluxDB: `setup_api()` yok → REST `/api/v2/setup` direkt kullanıldı, token dosyaya kaydedildi
- MinIO: Colab'ın kendi PHP-FPM'i port 9000'i tutuyordu → 9100 kullanıldı
- Redis: `zadd`+`sadd` aynı key'de çakıştı (`WRONGTYPE`) → sadece `sadd`
- `zrange(rev=True)` desteklenmiyordu (Redis 6.0.16) → `zrevrange`

### Sonuç: Colab pipeline'ı çalışır hâle geldi ama **kalıcı değildi** (Colab oturumu kapanınca sıfırlanıyor) — bu yüzden projeyi **lokal Windows'a taşıma kararı** alındı (Bölüm 5).

---

## 5. Faz 2 — Veri Hazırlığı (Offline ETL, Windows lokalde)

Üç kaynak için ayrı ayrı bronze→silver parse script'leri yazıldı (`parse_adsb_traces_from_tar_v2.py`, `parse_alfa.py`, `parse_uav_attack.py`), sonra ALFA için gold feature engineering (`alfa_gold_features.py`).

### 5.1 Bulunan ve düzeltilen kritik hatalar:

**A) UAV Attack etiketleme hatası (kritik):** `infer_label_from_path()` fonksiyonu ilk halinde TÜM path'i tarıyordu — `"Live GPS Spoofing and Jamming/Benign Flight/ace-benign-log_0..."` yolunda üst klasör adında "spoof" geçtiği için, gerçekte **benign** olan loglar yanlışlıkla `gps_spoofing` etiketleniyordu. **Düzeltme:** sadece EN YAKIN (bir alt) klasöre bakılacak şekilde değiştirildi. Doğrulama: `jamming_indicator` ortalaması artık `benign`'de ~0, `gps_jamming`'de ~27 — tutarlı.

**B) ALFA engine/engines birleşmesi:** `failure_status-engines.csv` dosya adından türetilen etiket "engines_fault" (çoğul) veriyordu, klasör-adı-tabanlı varsayılan ise "engine_fault" (tekil) — aynı şeyi ifade eden iki ayrı kategori olarak görünüyordu. Normalize edildi.

**C) RAM taşması (adsb tar parse):** İlk script tüm 47K dosyayı tek DataFrame'de biriktiriyordu → 8GB RAM'de çöktü. **Çözüm:** batch'ler halinde (300 dosyada bir) ayrı parquet parçalarına yazıp bellekten atma (`gc.collect()`).

**D) Timestamp yanlış yorumlama (dashboard'da, ayrı bir bug — Bölüm 9'da detaylı):** pandas `to_json()` varsayılan olarak datetime'ı epoch-milisaniye tam sayıya çeviriyor, `pd.to_datetime()` bunu nanosaniye sanıyor (1.000.000 kat fark) → grafikte "Jan 1970" + mikro-saniyelik sahte zikzak. Çözüm: `date_format="iso"` + garanti için açıkça ISO string'e dönüştürme.

### 5.2 ALFA Gold Feature Engineering
Ham `roll`/`pitch`/`airspeed` değil, **commanded vs measured farkı** (`roll_deviation` vb.) asıl arıza sinyali — çünkü normal manevrada ikisi birlikte hareket eder, arızada ayrışır. Rolling std + rate-of-change eklendi (ani/sert sapmaları yakalamak için, ALFA arızaları "sudden").

### 5.3 UAVAttackData'nın asıl değerli sinyali
`vehicle_gps_position`'daki `jamming_indicator`, `noise_per_ms`, `hdop`, `satellites_used`, `s_variance_m_s` — GPS jamming'i net ayırıyor (benign≈0.5, jamming≈27) ama **spoofing'i AYIRT ETMİYOR** (spoofing≈0.45, benign'e yakın) çünkü spoofing gürültü değil, temiz-ama-sahte sinyal enjekte ediyor. Spoofing tespiti için konum sıçraması gibi başka bir özellik gerekecek (henüz yapılmadı, model ekibinin işi).

---

## 6. Faz 3 — Yerel Windows Dashboard Kurulumu (Docker'sız)

**Neden Docker yok:** Kullanıcının Windows hesabı **admin değil** (okul/kurum bilgisayarı, `PC_4150_YD26` kullanıcı adı, `Xagt` güvenlik ajanı çalışıyor). MSI kurulumları gerçek admin şifresi istiyor (UAC "Evet/Hayır" değil, kullanıcı adı+şifre ekranı) — bu şifre yok. **Çözüm stratejisi:** her servis için, kurulum gerektirmeyen, sadece indirip klasöre açılan (zip/tar) taşınabilir sürümler kullanıldı.

### 6.1 Redis → Memurai başarısız, taşınabilir sürüme geçildi
- Önce Memurai (Redis'in resmi Windows portu) denendi, `winget install` **admin şifresi istedi** → vazgeçildi.
- **Çözüm:** `tporadowski/redis` (GitHub, taşınabilir zip, Redis 5.0.14, Win32 native) — `C:\redis\redis-server.exe` doğrudan çalıştırılıyor.
- **Yeni sorun:** Python `redis` kütüphanesi varsayılan olarak `HELLO 3` (RESP3) komutuyla bağlanmaya çalışıyor, Redis 5.0.14 bunu bilmiyor (`HELLO` komutu Redis 6.0+'ta eklendi) → `unknown command HELLO` hatası.
- **Çözüm:** Tüm `redis.Redis(...)` ve `redis.ConnectionPool(...)` çağrılarına `protocol=2` eklendi.

### 6.2 InfluxDB → 403 Forbidden
- `dl.influxdata.com` Python'un varsayılan `urllib` User-Agent'ini bot sayıp reddediyordu.
- **Çözüm:** Tarayıcı gibi görünen bir `User-Agent` header'ı eklenip stream indirme yapıldı.
- Kurulum: native Windows zip (influxdb2-2.9.1), `C:\...\Dashboard\influxdb\influxd.exe` — REST API ile org/bucket/token kurulumu (Colab'daki aynı yöntem).
- Bucket: `adsb-history`, retention 7 gün (script: `update_retention.py` ile değiştirilebilir, `--days 0` = sınırsız).

### 6.3 Kafka → en çok sorun çıkan kısım
Sırasıyla yaşanan ve çözülen sorunlar:

1. **Java 8 → 17 geçişi:** Kafka 3.x Java 8'i desteklemiyor. Java 17 kuruldu ama:
2. **JAVA_HOME klasör adı karışıklığı:** Komutlarda `jdk-17...` yazıldı ama gerçek klasör adı `jre-17...` idi (JRE kurulmuştu, JDK değil) — path geçersiz kaldı, sistem sessizce eski Java 8'e düştü.
3. **Oracle'ın eski `javapath` sistem PATH girişi:** Java 8 kurulumu `C:\Program Files (x86)\Common Files\Oracle\Java\javapath`'i SİSTEM PATH'ine kalıcı eklemiş. Windows PATH'i önce sistem, sonra kullanıcı PATH'i sırayla tarıyor — kullanıcı PATH'ine ne eklenirse eklensin, sistem PATH'teki bu eski giriş hep önce bulunuyordu. **Kalıcı çözüm:** `setup_kafka_windows.py` artık `java -version`'a hiç güvenmiyor — `find_java17()` fonksiyonu `jdk-17*`/`jre-17*` klasörlerini bilinen kurulum yollarında arayıp buluyor, sadece KENDİ başlattığı process'lere özel bir `env` dict'i (`build_env()`) veriyor.
4. **Windows classpath uzunluk sınırı:** Kafka'nın `.bat` script'leri her jar'ın TAM YOLUNU birleştiriyor — uzun proje yolunda (`Desktop\Dashboard\kafka\...`) bu, Windows'un ~8191 karakter komut satırı sınırını aşıp `"The input line is too long"` hatası veriyor. **Çözüm:** Kafka klasörü sabit kısa yola (`C:\kafka`) taşındı, script de hep bu sabit yolu kullanacak şekilde güncellendi.
5. **"Turkish locale bug" (en ilginç bug):** Zookeeper/Kafka açılırken `java.lang.IllegalArgumentException: No enum constant ... GroupType.CLASS¦C` hatası. Kök neden: Türkçe Windows'ta `String.toUpperCase()` küçük `i`'yi normal `I` yerine noktalı büyük `İ` (U+0130) yapıyor — Kafka'nın iç kodu `"classic"` değerini büyük harfe çevirip enum ile eşleştirirken `CLASSIC` yerine `CLASSİC` üretiyor, eşleşme patlıyor. **Çözüm:** `KAFKA_OPTS="-Duser.language=en -Duser.country=US"` — JVM'i sadece bu process'ler için İngilizce locale'e zorluyoruz, Windows'un kendi dilini değiştirmiyoruz.
6. **Stale broker registration:** `stop_all.bat`'ın `Stop-Process -Force` ile zorla kapatması, Kafka'ya Zookeeper'dan nazikçe kayıt silme fırsatı vermiyor → sonraki açılışta `NodeExistsException`. **Çözüm:** `start_kafka()` her seferinde önce `zookeeper-shell.bat` ile `/brokers/ids/0` kaydını proaktif temizliyor.

### 6.4 Neden Kafka var (Docker yok ama Kafka var) — önemli mimari karar
Tek kaynak (adsb.lol) olsa da EKİP bağımsız çalışıyor: sen dashboard, Yusuf MinIO arşivi, Anıl/model ekibi anomali tespiti — hepsi aynı `adsb.flights` topic'ini kendi `group.id`'siyle bağımsız okuyabilir, birbirinin koduna dokunmadan. Kafka'nın gerekçesi hacim değil, **organizasyonel ayrışma**.

---

## 7. Mimarinin Son Hâli (Uçtan Uca)

```
adsb.lol API (15sn'de bir, lat=39/lon=35/radius=500nm)
        |
        v
adsb_producer.py  --(Kafka: adsb.flights, 3 partition)-->  Kafka broker (C:\kafka, KRaft degil Zookeeper modu)
                                                                      |
                                                    dashboard_consumer.py (group=dashboard-consumer)
                                                                      |
                                                        +-------------+-------------+
                                                        v                           v
                                                     Redis                     InfluxDB
                                                  (canli state,              (bucket=adsb-history,
                                                   TTL 120sn)                  7 gun retention)
                                                        |                           |
                                                        +-------------+-------------+
                                                                      v
                                                        app.py (FastAPI + Dash, tek process)
                                                        FastAPI: iç API (port 8000)
                                                        Dash: arayuz (port 8050)
                                                                      |
                                                                      v
                                                            Tarayici (localhost:8050)

[Ekip - bağımsız, ayrı zamanda eklenecek]
Kafka (adsb.flights) --> Yusuf'un MinIO bronze consumer'i (henuz yazilmadi, plan var)
Kafka (adsb.flights) --> Model ekibinin anomali consumer'i (henuz yazilmadi)
                                  |
                                  v
                          Kafka (adsb.alerts) --> dashboard_consumer.py otomatik yakalar
                                                   (KOD DEGISIKLIGI GEREKMEZ)
```

## 8. Dosya Envanteri (proje klasörü: `C:\Users\PC_4150_YD26\Desktop\Dashboard`)

| Dosya | Görev |
|---|---|
| `setup_local_windows.py` | Redis (portable, elle) + InfluxDB (native zip) kurulumu/doğrulaması |
| `setup_kafka_windows.py` | Java 17 bulma, Kafka indirme, Zookeeper+broker başlatma (arka planda, log dosyasına), topic oluşturma, stale-broker temizliği |
| `adsb_producer.py` | adsb.lol'den çekip Kafka'ya yazan tek iş: fetch + parse + produce |
| `dashboard_consumer.py` | Kafka'dan okuyup Redis+InfluxDB'ye yazan consumer (group=dashboard-consumer) |
| `app.py` | FastAPI (iç API) + Dash (arayüz) — tüm dashboard mantığı burada |
| `update_retention.py` | InfluxDB bucket saklama süresini değiştirme scripti |
| `start_all.bat` / `stop_all.bat` | Tek tıkla tüm sistemi başlatma/durdurma |
| `KAFKA_SCHEMA.md` | Ekip için: `adsb.flights`/`adsb.alerts` topic şema sözleşmesi |
| `STARTUP.md` | Eski başlangıç prosedürü referansı (bu dosyadan önce yazılmıştı, kısmen güncelliğini yitirdi) |
| `influx_token.txt` | InfluxDB API token (setup script tarafından üretilir) |
| `C:\redis\` | Taşınabilir Redis 5.0.14 (proje klasörü dışında, sabit yol) |
| `C:\kafka\` | Kafka 3.9.2 + Zookeeper (proje klasörü dışında, sabit kısa yol — classpath sınırı için) |

**Veri hazırlığı klasörü (ayrı, `Desktop\Data`):** `parse_adsb_traces_from_tar_v2.py`, `parse_alfa.py`, `parse_uav_attack.py`, `alfa_gold_features.py`, `verify_silver.py`, `verify_adsb.py` — `silver/` ve `gold/` alt klasörlerine parquet üretiyor. Bu klasör dashboard'dan bağımsız, model eğitimi içindir.

---

## 9. Dashboard Özellikleri — Kronolojik Gelişim

**v1 (ilk çalışan hâl):** Plotly Scattermapbox harita (renkli noktalar) + geçmiş grafiği (irtifa/hız, dropdown ile uçak seçimi + saat kaydırıcısı) + trafik hacmi grafiği (son 24h benzersiz uçak/saat) + alarm listesi paneli.

**Timestamp bug:** pandas `to_json()` datetime'ı epoch-ms tam sayıya çeviriyor, Dash tarafında `pd.to_datetime()` bunu nanosaniye sanıyor → grafik "Jan 1970" + zikzak gösteriyordu. `date_format="iso"` + garanti için açık string dönüşümü ile çözüldü (hem `get_history` hem `traffic_stats` endpoint'lerinde).

**Uçak ikonu rotasyonu — büyük mimari değişiklik:** Plotly Scattermapbox marker'ları döndürülemiyor (native kısıt, Mapbox GL tabanlı, per-marker angle prop'u yok). **Leaflet'e geçildi** (`dash-leaflet` paketi, v1.1.3). Sürüm uyumsuzlukları yaşandı:
- `dl.DivIcon` diye bir bileşen YOK bu sürümde → `Marker(icon=<dict>)` ile denendi, marker hiç render olmadı
- Doğrusu: `dl.DivMarker(iconOptions=<dict>)` — ayrı, özel bir bileşen, GitHub kaynak kodundan (`iconOptions` PropTypes şeması) doğrulandı
- SVG uçak silüeti, `transform: rotate({heading}deg)` ile CSS döndürmesi, `track` alanına göre

**False-zero veri bug'ları (üç kez tekrarlayan desen):** `ac.get("gs", 0) or 0` gibi kod, adsb.lol bir alanı (hız/yön/dikey hız) o mesajda göndermediğinde sahte `0` yazıyordu — grafikte gerçek olmayan düşüşler görünüyordu. Sırayla `velocity`, sonra `track`, sonra kullanıcının sorusu üzerine kontrol edilip `lat/lon/alt`'ın GÜVENLİ olduğu doğrulandı (onlar eksikse zaten tüm mesaj atlanıyor). Çözüm deseni hep aynı: eksikse `None` bırak, InfluxDB'ye o alanı hiç yazma (gerçek boşluk, sahte sıfır değil), gösterim katmanında `None`'ı `"—"` olarak göster.

**Uçuş rotası (polyline):** Seçili uçağın son N saatlik `lat/lon` geçmişini haritada çizgi olarak çiziyor — mevcut `/api/history` endpoint'i zaten `lat/lon` döndürdüğü için backend değişikliği gerekmedi.

**Kalkış/Varış bilgisi:** ADS-B protokolü bunu taşımıyor (bilinen kısıt) — **adsbdb.com** (ücretsiz, topluluk veritabanı) entegre edildi. İki ayrı endpoint kullanılıyor:
- `/v0/callsign/{callsign}` → rota (kalkış/varış havalimanı)
- `/v0/aircraft/{icao24}` → uçak tipi/üretici/tescil/sahip + fotoğraf
İkisi de sadece uçak SEÇİLDİĞİNDE sorgulanıyor (her tick'te değil), Redis'te 7 gün/12 saat cache'leniyor.

**Tooltip yeniden tasarımı:** Düz metinden, çağrı kodu + kategori + 2 sütunlu istatistik gridine geçildi. Leaflet'in varsayılan beyaz tooltip kutusu da CSS ile koyu temaya çevrildi (`.leaflet-tooltip` override).

**TAM EKRAN YENİDEN TASARIM (büyük değişiklik):** Kullanıcı isteği: "ekranın tamamını kaplasın, kaydırma olmasın, kocaman harita, tıklamadan grafik görünmesin, tıklayınca sol kayan panel + sağ alt grafik paneli." Layout tamamen `position: fixed` + `overflow: hidden` + mutlak-konumlu overlay'lere çevrildi:
- Harita artık tek temel katman (100%×100%)
- Sol panel: `transform: translateX()` ile kayarak açılıp kapanıyor (canlı bilgi + rota + uçak tipi)
- Sağ-alt panel: `transform: translateY()` ile (geçmiş grafiği, sabit 24 saat)
- Dropdown kaldırıldı (`dcc.Store` ile görünmez state), sadece haritaya tıklayarak seçim
- Trafik hacmi grafiği ve saat kaydırıcısı KALDIRILDI (backend endpoint duruyor, sadece arayüzden kaldırıldı, geri eklenebilir)
- Kapatma butonu (`×`) — pattern-matching callback + close-button'ı BİRLEŞTİREN tek callback (aynı Output'u iki farklı buton yazıyor, `dash.callback_context` ile hangisinin tetiklediği ayırt ediliyor)

**Ayarlar paneli:** Alarm listesi kaldırıldı, yerine dişli (⚙) buton + açılır panel geldi. İki ayar:
- **Rota izi (saat):** polyline'ın kaç saatlik geçmişi göstereceği, varsayılan 2 saat (önceden sabit 1'di). ÖNEMLİ NOT: bu SADECE haritadaki iz çizgisini etkiliyor, sağ-alttaki geçmiş grafiği hâlâ sabit 24 saat (kasıtlı ayrım, "trace" kelimesi haritadaki iz olarak yorumlandı).
- **Saat dilimi:** Üç deneme sonucu `dcc.Slider` ile çözüldü:
  1. İlk deneme: `dcc.Dropdown` + CSS class override (`.Select-control` vb.) — dropdown'ın arka planı beyaz kaldı, CSS sınıf adları bu Dash sürümünde tutmadı
  2. İkinci deneme: `html.Select` + `value` prop — bu Dash sürümünde (4.3.0) `html.Select` hiç `value` prop'unu desteklemiyor, `TypeError` verdi
  3. **Çözüm:** `dcc.Slider(min=-12, max=14, step=1)` — eski/stabil bir bileşen, garanti çalıştı. Saat dilimi artık IANA isim değil (`Europe/Istanbul` yerine), düz UTC ofseti (`int`, örn. `3`).
- Artı/eksi butonları başta native `<input type=number>` spinner'larıydı, tıklanınca "siyahta takılı kalma" sorunu vardı (tarayıcı native davranışı, güvenilir CSS ile düzeltilemedi) — native spinner tamamen CSS ile gizlenip (`!important` gerekti, ilk deneme yetmedi), kendi `html.Button` `+`/`−` çiftimiz eklendi.

**Kafka stale-broker fix:** Bölüm 6.3'te detaylı — `stop_all.bat`'ın zorla kapatmasının bir yan etkisi olarak ortaya çıktı, script'e otomatik temizlik eklendi.

---

## 10. Bilinen Ortam Sabitleri (bir dahaki oturumda hemen lazım olacak)

```
Java 17 konumu     : C:\Users\PC_4150_YD26\AppData\Local\Programs\Eclipse Adoptium\jre-17.0.19.10-hotspot
Kafka klasörü       : C:\kafka   (sabit, kısa yol -- classpath sinir sorunu icin)
Redis (portable)    : C:\redis   (tporadowski/redis 5.0.14, RESP2, protocol=2 sart)
Proje klasörü       : C:\Users\PC_4150_YD26\Desktop\Dashboard
InfluxDB            : {proje}\influxdb\influxd.exe, bucket=adsb-history, 7 gun retention
Zookeeper portu     : 2181
Kafka broker portu  : 9092
Redis portu         : 6379
InfluxDB portu      : 8086
FastAPI portu       : 8000
Dash portu          : 8050
Kafka topics        : adsb.flights (3 partition), adsb.alerts (1 partition)
KAFKA_OPTS gerekli   : -Duser.language=en -Duser.country=US  (Turkish locale bug icin, script'te otomatik)
```

**Kullanıcı ortamı notları:**
- Windows hesabı **admin değil** — hiçbir MSI/winget kurulumu (gerçek şifre isteyenler) çalışmaz, hep taşınabilir/zip çözümler kullan.
- Python: `pythoncore-3.14-64` (çok yeni sürüm) — bazı paketlerde (confluent-kafka gibi) Windows wheel'i olmayabilir ihtimaline karşı hep tetikte ol, şu ana kadar sorun çıkmadı.
- Sistem dili Türkçe — `.toUpperCase()` türü işlemler her yerde potansiyel risk (Turkish locale bug benzerleri başka yerde de çıkabilir).

## 11. Ekip Entegrasyonu — PIPELINE_PLAN.md Analizi

Ekip, ayrı bir **Bronze/Silver/Gold lakehouse** mimarisi kuruyor (MinIO tabanlı), amacı model eğitimi için temiz veri seti — bizim canlı serving pipeline'ımızdan (Kafka→Redis/InfluxDB→dashboard) **farklı ama tamamlayıcı**. Öne çıkanlar:
- **Generic parser deseni** (`parse_generic.py`) — format otomatik algılama + provenance, bizim üç ayrı elle-yazılmış parser'dan daha esnek.
- **Provenance kolonları** (`_source_type`, `_ingest_ts_utc`, `_source_file`) — biz bunu yapmadık, ileride eklenebilir.
- ALFA/UAV Attack etiketleme mantığı **bizim bulduğumuz "en yakın klasör" bug fix'iyle birebir aynı** — muhtemelen aynı kod tabanının türevi veya bağımsız yakınsama.
- Yusuf'un Kafka consumer'ı (adsb.flights → MinIO bronze JSONL) **tam olarak** bizim `KAFKA_SCHEMA.md`'de öngördüğümüz "arkadaşların kendi group.id'siyle bağlanabilir" senaryosu — entegrasyon noktası hazır, henüz gerçekleşmedi.

## 12. Açık / Bekleyen Konular

- [ ] Trafik hacmi grafiği arayüzden kaldırıldı, backend endpoint (`/api/traffic_stats`) duruyor — istenirse geri eklenebilir.
- [ ] Sağ-alt geçmiş grafiği hâlâ sabit 24 saat, "rota izi" ayarına bağlı değil (kasıtlı ayrım, ama kullanıcı isterse birleştirilebilir).
- [ ] MinIO hiç kurulmadı bu dashboard'da (Yusuf'un işi, ayrı).
- [ ] Model ekibi henüz `adsb.alerts` topic'ine hiçbir şey yazmıyor — alarm sistemi altyapısı hazır ama şu an boş.
- [ ] Oracle Cloud (7/24 çalıştırma için) konuşuldu ama henüz denenmedi — laptop kapanınca pipeline durur, bilinen kısıt.
- [ ] React/Next.js/MapLibre GL (küresel harita) denendi, **bilinçli olarak vazgeçildi** — mevcut Leaflet mimarisiyle uyumsuz, efor/fayda dengesi kötü.
- [ ] Spark Streaming hiç kullanılmadı, kullanılmasına gerek yok (ölçek çok küçük) — soru olarak geldi, cevaplandı, karar: kullanma.

## 13. Hızlı Başlatma (özet — detay için `STARTUP.md`'ye bak, kısmen eski ama genel akış doğru)

```
1. C:\redis'te:     redis-server.exe redis.windows.conf   (kendi terminali)
2. {proje}\influxdb: influxd.exe                            (kendi terminali)
3. {proje}'de:       python setup_kafka_windows.py           (Zookeeper+Kafka arka planda baslar, script kapanir)
4. {proje}'de:       python adsb_producer.py                 (kendi terminali)
5. {proje}'de:       python dashboard_consumer.py             (kendi terminali)
6. {proje}'de:       python app.py                            (kendi terminali)
7. Tarayici:         http://localhost:8050
```
Ya da tek tık: `start_all.bat` (aynı adımları otomatik, doğru sırayla açar).
Kapatmak için: `stop_all.bat` (görünür pencereleri + arka plandaki Java process'lerini kapatır).
