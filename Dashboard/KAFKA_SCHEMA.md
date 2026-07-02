# Kafka Topic Sözleşmesi

Bu proje **iki Kafka topic'i** üzerinden çalışıyor. Kendi consumer'ını veya
producer'ını yazacaksan, sadece bu şemaya uyman yeterli — dashboard koduna
dokunmana gerek yok, ben de senin koduna dokunmam.

Bağlantı: `bootstrap.servers = localhost:9092`

---

## 1. `adsb.flights`

**Kim yazıyor:** `adsb_producer.py` (ben) — adsb.lol'den 15 saniyede bir çekip yazıyor.
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
| `category` | string | Emitter kategorisi (A0-D7, ADS-B standardı) |
| `is_military` | bool | adsb.lol'ün `dbFlags` bit alanının 1. biti (`dbFlags & 1`) — topluluk veritabanına dayanır, %100 kapsama garantisi yok, alan gelmezse `false` |
| `source` | string | Sabit `"adsblol"` |
| `ts` | string | ISO 8601, UTC |

**Model consumer'ın için öneri:** `group.id="anomaly-model"` kullan, `auto.offset.reset="latest"`
ile abone ol. Kendi feature engineering'ini yapıp skorunu hesapla, anomali tespit edersen
aşağıdaki `adsb.alerts` şemasına göre yaz.

**MinIO arşivleyici için öneri:** `group.id="minio-archiver"` kullan, mesajları
biriktirip (örn. 100 mesajda bir veya 60 saniyede bir) Parquet batch olarak yaz —
daha önceki Colab pipeline'ındaki `_flush_alerts` fonksiyonu aynı mantıkla uyarlanabilir.

---

## 2. `adsb.alerts`

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
producer.produce("adsb.alerts", key=alert["icao24"],
                 value=json.dumps(alert).encode())
producer.flush()
```

---

## Test etmek için

Kendi consumer'ını yazmadan önce topic'te gerçekten veri akıp akmadığını
görmek istersen, Kafka'nın kendi CLI aracıyla (kurulum sonrası `kafka/`
klasöründe) kontrol edebilirsin:

```
kafka\bin\windows\kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic adsb.flights --from-beginning
```

Ctrl+C ile durdurulur, veriyi bozmaz (sadece okur).
