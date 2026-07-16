# Kafka Topic Sözleşmesi

Bu proje **iki Kafka topic'i** üzerinden çalışıyor. Kendi consumer'ını veya
producer'ını yazacaksan, sadece bu şemaya uyman yeterli — dashboard koduna
dokunmana gerek yok, ben de senin koduna dokunmam.

Bağlantı: `bootstrap.servers = localhost:9092`

**NOT (kalıcı olarak yüksek hacim):** `uav_producer.py` artık her zaman DÜNYA
çapında sorgu yapıyor (bölge seçimi kaldırıldı) — bu topic'e düşen mesaj sayısı
cycle başına birkaç bin ile on binler arasında olabilir (sabit, sürekli). Kendi
consumer'ını yazarken bunu hesaba kat (özellikle MinIO arşivleyici ve model
consumer'ı için) — throughput/batch boyutu varsayımların buna göre olmalı.

**NOT (KIRICI DEĞİŞİKLİK — `ttl_hint` alanı kaldırıldı):** Daha önce her kayıtta
bir `ttl_hint` alanı vardı (Redis TTL'i için önerilen süre). Bu tamamen kaldırıldı —
eğer kendi consumer'ın bu alanı okuyorsa, artık gelmeyecek. Sebep: dashboard
consumer'ı artık Redis'te TTL kullanmıyor, bunun yerine PENCERE TABANLI bir
"gör-yoksa-sil" modeline geçti (belirli bir zaman penceresinde hiç görünmeyen
kayıtları doğrudan siliyor). Bu, sadece BİZİM Redis kullanımımızla ilgili bir
detaydı — kendi consumer'ın (MinIO arşivleyici, model eğitimi) muhtemelen zaten
bu alanı hiç kullanmıyordu, ama olur da bir yerde referans varsa diye belirtiyorum.

---

## 1. `uav.flights`

**Kim yazıyor:** `uav_producer.py` (ben) — adsb.lol'den dünya çapında çekip yazıyor
(hedef aralık 15sn, ama gerçek cycle süresi veri hacmine göre daha uzun olabilir).
**Kim okuyabilir:** herkes — kendi `group.id`'ni verirsen benim dashboard consumer'ımdan
bağımsız, kendi hızında okursun. Aynı mesajı ikimiz de görürüz, biri diğerini bloklamaz.

**Partition sayısı:** 3
**Key:** `icao24` (string, örn. `"4b1a2c"`)
**Value:** JSON

```json
{
  "icao24": "4b1a2c",
  "callsign": "THY1234",
  "lat": 41.015,
  "lon": 28.979,
  "alt": 10668.0,
  "velocity": 230.5,
  "track": 271.3,
  "vertical_rate": 0.0,
  "category": "A3",
  "is_military": false,
  "source": "adsblol",
  "cycle_id": 42,
  "signal_age_sec": 4.5,
  "ts": "2026-07-01T12:34:56.789012+00:00"
}
```

| Alan | Tip | Açıklama |
|---|---|---|
| `icao24` | string | 24-bit ICAO transponder ID, küçük harf hex |
| `callsign` | string | Uçuş çağrı kodu, boş olabilir |
| `lat`, `lon` | float | Derece |
| `alt` | float | Metre (barometrik) |
| `velocity` | float | m/s (ground speed) |
| `track` | float | Derece, 0-360 (kuzeyden saat yönünde) |
| `vertical_rate` | float | m/s, pozitif=tırmanış |
| `category` | string | Emitter kategorisi (A0-D7, ADS-B standardı). OpenSky kaynağında hep boş |
| `is_military` | bool | adsb.lol'ün `dbFlags` bit alanının 1. biti (`dbFlags & 1`) — topluluk veritabanına dayanır, %100 kapsama garantisi yok, alan gelmezse `false`. OpenSky kaynağında hep `false` |
| `source` | string | `"adsblol"` veya `"opensky"` — hangi kaynaktan geldiği |
| `cycle_id` | int | Bu kaydın hangi üretim cycle'ına ait olduğu (1'den başlayıp artan sayaç). Aynı cycle_id'ye sahip tüm kayıtlar AYNI fetch işleminden geliyor. Dashboard consumer'ı bunu, bir cycle tamamlanınca o cycle'da görünmeyen eski kayıtları anında temizlemek için kullanıyor (saniye tahminine dayalı TTL/pencere YOK) — kendi consumer'ında "bu cycle'ın verisi tamam" sinyaline ihtiyacın olursa bunu kullan |
| `signal_age_sec` | float veya null | Bu pozisyonun GERÇEKTE kaç saniye önce alındığı (adsb.lol/readsb'nin kendi "seen" alanı, OpenSky'de "last_contact"tan hesaplanıyor). ÖNEMLİ: adsb.lol bir uçaktan mesaj kesilse bile onu 60 saniyeye kadar listede TUTAR — yani `icao24`'ün kayıtta olması, sinyalin taze olduğu anlamına gelmez. Bu alan büyükse (30-40sn+), uçak hâlâ görünse bile sinyali aslında eskimiş/kesilmiş olabilir. Kaynak bu bilgiyi sağlamıyorsa `null` |
| `ts` | string | ISO 8601, UTC |

**Model consumer'ın için öneri:** `group.id="anomaly-model"` kullan, `auto.offset.reset="latest"`
ile abone ol. Kendi feature engineering'ini yapıp skorunu hesapla, anomali tespit edersen
aşağıdaki `uav.alerts` şemasına göre yaz.

**MinIO arşivleyici için öneri:** `group.id="minio-archiver"` kullan, mesajları
biriktirip (örn. 100 mesajda bir veya 60 saniyede bir) Parquet batch olarak yaz —
daha önceki Colab pipeline'ındaki `_flush_alerts` fonksiyonu aynı mantıkla uyarlanabilir.

---

## 2. `uav.alerts`

**Kim yazacak:** model ekibi (henüz kimse yazmıyor — topic hazır, bekliyor).
**Kim okuyor:** `dashboard_consumer.py` (ben) — bu topic'e bir şey düştüğü anda
otomatik olarak Redis'e yazıyorum, dashboard'daki "Model Alarmları" paneli
otomatik dolmaya başlıyor. **Benim tarafımda hiçbir kod değişikliği gerekmez.**

**Partition sayısı:** 1
**Key:** `icao24` (string)
**Value:** JSON — aşağıdaki alanlar **zorunlu**, istediğin kadar ek alan da ekleyebilirsin
(dashboard onları görmez ama Redis'te saklı kalır, ileride kullanılabilir):

```json
{
  "icao24": "4b1a2c",
  "alert_type": "gps_spoofing",
  "score": 0.87,
  "ts": "2026-07-01T12:35:10.123456+00:00"
}
```

| Alan | Tip | Zorunlu mu | Açıklama |
|---|---|---|---|
| `icao24` | string | evet | Hangi uçak (alarm panelinde eşleştirme için) |
| `alert_type` | string | evet | Serbest metin, örn. `"gps_spoofing"`, `"engine_fault"` |
| `score` | float | hayır | Model güven skoru (0-1 arası önerilir) |
| `ts` | string | hayır | ISO 8601 UTC, yoksa consumer kendi zamanını kullanır |

**Basit örnek üretim kodu** (kendi modelin bunu tetikleyince):

```python
from confluent_kafka import Producer
import json
from datetime import datetime, timezone

producer = Producer({"bootstrap.servers": "localhost:9092"})

alert = {
    "icao24": "4b1a2c",
    "alert_type": "gps_spoofing",
    "score": 0.87,
    "ts": datetime.now(timezone.utc).isoformat(),
}
producer.produce("uav.alerts", key=alert["icao24"],
                 value=json.dumps(alert).encode())
producer.flush()
```

---

## Test etmek için

Kendi consumer'ını yazmadan önce topic'te gerçekten veri akıp akmadığını
görmek istersen, Kafka'nın kendi CLI aracıyla (kurulum sonrası `kafka/`
klasöründe) kontrol edebilirsin:

```
kafka\bin\windows\kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic uav.flights --from-beginning
```

Ctrl+C ile durdurulur, veriyi bozmaz (sadece okur).
