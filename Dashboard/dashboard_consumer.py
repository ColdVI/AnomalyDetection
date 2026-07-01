"""
dashboard_consumer.py
Bu SENIN (dashboard) tuketicin. Kendi group.id'si var ("dashboard-consumer"),
yani ekip arkadaslarinin ekleyecegi baska consumer'lar (MinIO arsivleyici,
model anomali tespiti) bu script'ten tamamen bagimsiz calisir -- ayni
mesaji herkes kendi hizinda okur, birbirini etkilemez.

Iki topic dinler:
  - adsb.flights  -> Redis (canli durum) + InfluxDB (7 gunluk gecmis)
  - adsb.alerts   -> Redis (son alert listesi) -- SIMDILIK BOS, model ekibi
                     hazir olunca buraya yazmaya baslayacak, bu consumer
                     otomatik olarak onlari da yakalayip dashboard'a
                     yansitmaya baslayacak, kod degisikligi gerekmez.

Kullanim:
    python dashboard_consumer.py
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import redis
from confluent_kafka import Consumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

BOOTSTRAP = "localhost:9092"
FLIGHTS_TOPIC = "adsb.flights"
ALERTS_TOPIC = "adsb.alerts"

TOKEN_FILE = Path("influx_token.txt")
INFLUX_HOST = "http://localhost:8086"
INFLUX_ORG = "iha-org"
INFLUX_BUCKET = "adsb-history"

REDIS_TTL_SEC = 120


def load_token() -> str:
    if not TOKEN_FILE.exists():
        raise SystemExit("influx_token.txt bulunamadi. Once setup_local_windows.py calistir.")
    return TOKEN_FILE.read_text().strip()


def main():
    token = load_token()

    rdb = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True,
                      protocol=2)
    rdb.ping()
    print("Redis baglandi.")

    influx = InfluxDBClient(url=INFLUX_HOST, token=token, org=INFLUX_ORG)
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    print(f"InfluxDB baglandi (bucket={INFLUX_BUCKET}).")

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "dashboard-consumer",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([FLIGHTS_TOPIC, ALERTS_TOPIC])
    print(f"Kafka consumer hazir (group=dashboard-consumer, "
          f"topics=[{FLIGHTS_TOPIC}, {ALERTS_TOPIC}])")
    print("Durdurmak icin Ctrl+C\n")

    stats = {"flights": 0, "alerts": 0}

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"  [uyari] kafka hatasi: {msg.error()}")
                continue

            topic = msg.topic()
            try:
                data = json.loads(msg.value())
            except Exception:
                continue

            if topic == FLIGHTS_TOPIC:
                icao = data.get("icao24", "")
                if not icao:
                    continue
                rdb.set(f"iha:state:{icao}", json.dumps(data), ex=REDIS_TTL_SEC)
                rdb.sadd("iha:active_flights", icao)

                write_api.write(bucket=INFLUX_BUCKET, record=(
                    Point("flights")
                    .tag("icao24", icao)
                    .field("lat", data.get("lat", 0.0))
                    .field("lon", data.get("lon", 0.0))
                    .field("alt", data.get("alt", 0.0))
                    .field("velocity", data.get("velocity", 0.0))
                    .field("track", data.get("track", 0.0))
                    .field("vertical_rate", data.get("vertical_rate", 0.0))
                    .time(datetime.now(timezone.utc))
                ))
                stats["flights"] += 1

            elif topic == ALERTS_TOPIC:
                # sema: KAFKA_SCHEMA.md'deki adsb.alerts bolumune bakiniz
                rdb.lpush("iha:recent_alerts", json.dumps(data))
                rdb.ltrim("iha:recent_alerts", 0, 19)
                stats["alerts"] += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"YENI ALERT: {data.get('icao24','?')} "
                      f"{data.get('alert_type','?')}")

            if (stats["flights"] + stats["alerts"]) % 200 == 0 and stats["flights"] > 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"islenen: {stats['flights']} flight, {stats['alerts']} alert")

    except KeyboardInterrupt:
        print(f"\nDurduruldu. flights={stats['flights']}, alerts={stats['alerts']}")
    finally:
        consumer.close()
        influx.close()


if __name__ == "__main__":
    main()
