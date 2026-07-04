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
from influxdb_client.client.write_api import WriteOptions

BOOTSTRAP = "localhost:9092"
FLIGHTS_TOPIC = "adsb.flights"
ALERTS_TOPIC = "adsb.alerts"

TOKEN_FILE = Path("influx_token.txt")
INFLUX_HOST = "http://localhost:8086"
INFLUX_ORG = "iha-org"
INFLUX_BUCKET = "adsb-history"

# ONEMLI: bu artik SABIT bir deger olarak KULLANILMIYOR -- sadece kaydin
# icinde "ttl_hint" yoksa (eski/farkli bir producer'dan gelen kayit vb.)
# devreye giren GERI DUSUS degeri. Gercek TTL, adsb_producer.py'nin o
# CALISMA icin kullandigi --interval'den turetilip her kaydin icine
# "ttl_hint" olarak konuyor -- boylece TTL, kaynak/hiz ne olursa olsun
# HER ZAMAN dogru kalir (eskiden bu sabitti, OpenSky gibi yavas
# kaynaklarda TTL'nin tazeleme araligindan KISA kalmasi -- 120sn < 300sn
# -- ucaklarin "kaybolup geri gelmesine" yol aciyordu).
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
    # ONEMLI: ONCEDEN SYNCHRONOUS modundaydi -- her mesaj icin write_api.write()
    # cagrisi InfluxDB'nin HTTP yanitini BEKLIYORDU (tek tek, bloklayarak).
    # "Dunya" modunda tek bir cycle'da binlerce mesaj gelince (orn. 5660),
    # consumer bu kuyruguyu dakikalarca surecek sekilde tek tek isliyordu --
    # bu sirada Kafka'dan gelen SONRAKI (orn. "Turkiye" moduna donulduktan
    # sonraki) mesajlar da bu kuyrugun ARKASINDA bekliyordu (Kafka mesajlari
    # sirayla islenir, atlanamaz), yani mod degistirseniz bile dashboard
    # eski kuyruk bitene kadar donuk kaliyordu.
    #
    # COZUM: batching (toplu) yazim -- write_api.write() artik ANINDA doner
    # (sadece bir kuyruga ekler), gercek HTTP yazimi ARKA PLANDAKI bir thread
    # tarafindan batch_size'a ulasildiginda VEYA flush_interval doldugunda
    # yapilir. Boylece consumer dongusu InfluxDB'nin ag gecikmesinden tamamen
    # ayrisiyor, Kafka'yi native hizinda tuketebiliyor.
    write_api = influx.write_api(write_options=WriteOptions(
        batch_size=500,           # 500 noktaya ulasinca gonder
        flush_interval=2_000,     # yoksa 2 saniyede bir gonder (ms)
        jitter_interval=0,
        retry_interval=5_000,     # basarisiz batch'i 5sn sonra tekrar dene (ms)
        max_retries=3,
        max_retry_delay=30_000,
        exponential_base=2,
    ))
    print(f"InfluxDB baglandi (bucket={INFLUX_BUCKET}, batch yazim aktif).")

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "dashboard-consumer",
        "auto.offset.reset": "latest",
        # ONEMLI: varsayilan max.poll.interval.ms=300000 (5dk) -- eger
        # tek bir mesajin islenmesi (Redis/InfluxDB yazimi) herhangi bir
        # nedenle (gecici yavaslik, InfluxDB retry vb.) beklenenden uzun
        # surerse, Kafka bu consumer'i "oldu" sanip GRUPTAN ATIYOR
        # ("leaving group" hatasi, MAX_POLL_EXCEEDED) -- bunu canli
        # gozlemledik. 900000ms (15dk) ile cok daha genis bir tolerans
        # payi taniyoruz -- gecici bir yavaslik artik consumer'i
        # dusurmuyor, sadece bekliyor.
        "max.poll.interval.ms": 900000,
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
                rdb.set(f"iha:state:{icao}", json.dumps(data),
                       ex=data.get("ttl_hint", REDIS_TTL_SEC))
                rdb.sadd("iha:active_flights", icao)

                point = (
                    Point("flights")
                    .tag("icao24", icao)
                    .field("lat", float(data.get("lat", 0.0)))
                    .field("lon", float(data.get("lon", 0.0)))
                    .field("alt", float(data.get("alt", 0.0)))
                    .time(datetime.now(timezone.utc))
                )
                # ONEMLI: velocity/vertical_rate/track icin data.get(..., 0.0)
                # KULLANMIYORUZ -- deger gercekten None ise (kaynak o anki
                # mesajda gondermemis) alani hic yazmiyoruz, InfluxDB o
                # noktada dogal bir bosluk birakiyor. Aksi halde grafikte
                # gercek olmayan sahte "sifira dusus" gorunuyordu.
                #
                # ONEMLI (2. kez basimiza geldi -- OpenSky adaptorunde float()
                # unutulmustu): InfluxDB bir alanin tipini (int/float) ILK
                # YAZIMDA sabitler, sonra farkli tip gelince TUM batch'i
                # reddeder ("field type conflict"). Kaynaklar (adsb.lol,
                # OpenSky, ileride eklenecekler) JSON'dan gelen sayilari
                # bazen int bazen float birakabilir (orn. "0" vs "0.5").
                # BURADA, son yazim noktasinda, MUTLAKA float() ile
                # zorluyoruz -- hangi kaynak/adaptor unutursa unutsun,
                # InfluxDB'ye giden tip HER ZAMAN tutarli oluyor.
                if data.get("velocity") is not None:
                    point = point.field("velocity", float(data["velocity"]))
                if data.get("vertical_rate") is not None:
                    point = point.field("vertical_rate", float(data["vertical_rate"]))
                if data.get("track") is not None:
                    point = point.field("track", float(data["track"]))

                write_api.write(bucket=INFLUX_BUCKET, record=point)
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
        # ONEMLI: write_api.close() -- influx.close()'dan ONCE cagrilmali.
        # Batching modunda bekleyen (henuz gonderilmemis) noktalar bu
        # cagriyla flush edilir; atlanirsa son birkaç saniyenin verisi
        # kaybolabilir.
        write_api.close()
        influx.close()


if __name__ == "__main__":
    main()
