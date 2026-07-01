"""
adsb_producer.py
SADECE adsb.lol'den veri ceker ve Kafka'ya yazar. Redis/InfluxDB/MinIO
gibi hicbir depoyu bilmez -- o is tuketicilerin (consumer) sorumlulugunda.
Bu ayrim sayesinde ekip arkadaslari kendi consumer'larini bu script'e
hic dokunmadan yazabilir.

Kullanim:
    python adsb_producer.py [--interval 15] [--lat 39] [--lon 35] [--radius 500]
"""
import argparse
import json
import time
from datetime import datetime, timezone

import requests
from confluent_kafka import Producer

BOOTSTRAP = "localhost:9092"
TOPIC = "adsb.flights"


def fetch_adsblol(lat: float, lon: float, radius_nm: int):
    url = f"https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if data.get("ac"):
                return data
    except Exception as e:
        print(f"  [uyari] fetch hatasi: {e}")
    return None


def parse_aircraft(ac: dict):
    lat = ac.get("lat")
    lon = ac.get("lon")
    alt = ac.get("alt_baro", ac.get("alt_geom", 0))
    if not lat or not lon or not alt or str(alt).lower() == "ground":
        return None
    icao = ac.get("hex", "").strip().lower()
    if not icao:
        return None
    return {
        "icao24": icao,
        "callsign": ac.get("flight", "").strip(),
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "alt": round(float(alt) * 0.3048, 1),
        "velocity": round(float(ac.get("gs", 0) or 0) * 0.5144, 1),
        "track": float(ac.get("track", 0) or 0),
        "vertical_rate": round(float(ac.get("baro_rate", 0) or 0) * 0.00508, 2),
        "category": ac.get("category", ""),
        "source": "adsblol",
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def delivery_report(err, msg):
    if err is not None:
        print(f"  [uyari] teslim hatasi: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--lat", type=float, default=39.0)
    ap.add_argument("--lon", type=float, default=35.0)
    ap.add_argument("--radius", type=int, default=500)
    args = ap.parse_args()

    producer = Producer({"bootstrap.servers": BOOTSTRAP})
    print(f"Kafka producer hazir (topic={TOPIC})")
    print(f"Polling: merkez=({args.lat},{args.lon}) yaricap={args.radius}nm "
          f"araligi={args.interval}sn")
    print("Durdurmak icin Ctrl+C\n")

    stats = {"cycles": 0, "total": 0}

    try:
        while True:
            t0 = time.time()
            raw = fetch_adsblol(args.lat, args.lon, args.radius)

            if raw:
                n = 0
                for ac in raw.get("ac", []):
                    rec = parse_aircraft(ac)
                    if not rec:
                        continue
                    producer.produce(
                        TOPIC, key=rec["icao24"],
                        value=json.dumps(rec).encode(),
                        callback=delivery_report,
                    )
                    n += 1
                producer.flush()

                stats["cycles"] += 1
                stats["total"] += n
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{n} ucus -> Kafka (cycle={stats['cycles']}, "
                      f"toplam={stats['total']})")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] veri alinamadi")

            sleep_time = max(0, args.interval - (time.time() - t0))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nDurduruldu. Toplam: {stats['cycles']} cycle, {stats['total']} kayit")


if __name__ == "__main__":
    main()
