"""
adsb_producer.py
SADECE adsb.lol'den veri ceker ve Kafka'ya yazar. Redis/InfluxDB/MinIO
gibi hicbir depoyu bilmez -- o is tuketicilerin (consumer) sorumlulugunda.
Bu ayrim sayesinde ekip arkadaslari kendi consumer'larini bu script'e
hic dokunmadan yazabilir.

ISTISNA: Redis'e KUCUK bir kontrol amaciyla bagliyoruz -- Dash ayarlar
panelindeki "Bolge" (Turkiye/Dunya) secimini okumak icin. Bu, veri
YAZMIYOR (hala sadece tuketicilerin isi), sadece bir ayari OKUYOR.

Kullanim:
    python adsb_producer.py [--interval 15] [--lat 39] [--lon 35] [--radius 500]
"""
import argparse
import json
import time
from datetime import datetime, timezone

import redis
import requests
from confluent_kafka import Producer

BOOTSTRAP = "localhost:9092"
TOPIC = "adsb.flights"

# Dash ayarlar panelinin yazdigi, bizim okudugumuz kontrol anahtari.
REGION_KEY = "iha:settings:region"

# BOLGE: Turkiye modunda --radius (varsayilan 500nm) kullanilir. Dunya
# modunda AYNI merkez noktadan (--lat/--lon) cok daha buyuk bir yaricap
# istiyoruz. Herhangi bir noktadan Dunya uzerindeki EN UZAK nokta (tam
# karsi/antipodal nokta) ~10.800 deniz mili -- bu degeri asan bir yaricap
# matematiksel olarak "tum dunya" istemis oluyor, merkezi degistirmeye
# gerek yok.
#
# ONEMLI BELIRSIZLIK: adsb.lol'un "dist" parametresi icin resmi/dokumante
# edilmis bir ust siniri yok (API dokumantasyonu net degil), bu deger de
# canli test EDILEMEDI. Ilk calistirmada asagidaki "N ucus -> Kafka"
# satirina bak: binlerce/on binlerce ucus goruyorsan calisiyor demektir.
# Hala yuzlerle sinirliysa, adsb.lol daha kucuk bir ust sinir uyguluyor
# demektir -- o zaman farkli bir veri kaynagina (orn. OpenSky Network'un
# states/all uc noktasi) gecmemiz gerekir.
WORLD_RADIUS_NM = 12000


def get_region_mode(rdb) -> str:
    """Dash ayarlar panelinden yazilan bolge tercihini okur. Redis'e
    erisilemezse (henuz baslamadi, gecici kesinti vb.) GUVENLI VARSAYILAN
    olarak "turkey" doner -- yani bir sorun oldugunda sessizce dunya
    moduna GECMIYORUZ, bilinen/stabil davranista kaliyoruz."""
    try:
        mode = rdb.get(REGION_KEY)
        return mode if mode in ("turkey", "world") else "turkey"
    except Exception:
        return "turkey"


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

    # ONEMLI: adsb.lol her mesajda "gs" (ground speed) alanini
    # gondermiyor -- bazen pozisyon guncellemesi gelir ama hiz verisi
    # o an mevcut degildir (zayif sinyal, MLAT-only fix vb). Eskiden
    # "gs" eksikse 0 m/s YAZIYORDUK, bu sahte bir dususu grafige
    # yansitiyordu. Simdi eksikse None birakiyoruz -- tuketici bu
    # deger None ise InfluxDB'ye hic yazmayacak, gercek bir bosluk
    # olusacak, sahte sifir degil.
    raw_gs = ac.get("gs")
    velocity = round(float(raw_gs) * 0.5144, 1) if raw_gs is not None else None

    raw_baro_rate = ac.get("baro_rate")
    vertical_rate = (round(float(raw_baro_rate) * 0.00508, 2)
                     if raw_baro_rate is not None else None)

    # ONEMLI: ayni sekilde "track" (yon) eksikse sahte 0 (kuzey) YAZMIYORUZ.
    # Bu ozellikle harita uzerindeki ucak ikonu rotasyonunu etkiliyor --
    # yon bilinmiyorsa ikon yanlislikla "kuzeye bakiyor" gibi gorunmemeli.
    raw_track = ac.get("track")
    track = float(raw_track) if raw_track is not None else None

    # ASKERI/SIVIL AYRIMI: adsb.lol, ADSBExchange/readsb ile ayni "dbFlags"
    # bit alanini kullaniyor -- 1. bit (dbFlags & 1) askeri ucak demek
    # (topluluk tarafindan tutulan bir veritabanina dayanir, orn. bilinen
    # askeri ICAO hex araliklari/kayitlari). Alan gelmezse (KeyError/None/
    # beklenmeyen tip) guvenli varsayilan olarak False -- "bilinmiyor"u
    # "askeri" gibi gostermek istemiyoruz, sivil sayilir.
    try:
        is_military = bool(int(ac.get("dbFlags", 0) or 0) & 1)
    except (TypeError, ValueError):
        is_military = False

    return {
        "icao24": icao,
        "callsign": (ac.get("flight") or "").strip(),
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "alt": round(float(alt) * 0.3048, 1),
        "velocity": velocity,
        "track": track,
        "vertical_rate": vertical_rate,
        "category": ac.get("category", ""),
        "squawk": ac.get("squawk", ""),
        "emergency": ac.get("emergency", "none"),
        "is_military": is_military,
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
    # decode_responses=True + protocol=2 -- projede Redis icin kullanilan
    # standart baglanti sekli (bkz. dashboard_consumer.py, Redis 5.0.14
    # portable surumu RESP3/HELLO komutunu bilmiyor).
    rdb = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True,
                      protocol=2)
    print(f"Kafka producer hazir (topic={TOPIC})")
    print(f"Polling: merkez=({args.lat},{args.lon}) yaricap={args.radius}nm "
          f"araligi={args.interval}sn")
    print("Bolge ayari Redis'ten okunuyor -- Dash Ayarlar panelinden "
          "Turkiye/Dunya arasinda gecis yapilabilir.")
    print("Durdurmak icin Ctrl+C\n")

    stats = {"cycles": 0, "total": 0}
    current_mode = None  # ilk cycle'da mutlaka bir kez log basmak icin

    try:
        while True:
            t0 = time.time()

            mode = get_region_mode(rdb)
            if mode != current_mode:
                radius_desc = (f"{args.radius}nm (Turkiye)" if mode == "turkey"
                               else f"{WORLD_RADIUS_NM}nm (Dunya)")
                print(f"[bolge] '{mode}' moduna geçildi -- yaricap={radius_desc}")
                current_mode = mode
            radius = args.radius if mode == "turkey" else WORLD_RADIUS_NM

            raw = fetch_adsblol(args.lat, args.lon, radius)

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
                      f"toplam={stats['total']}, bolge={mode})")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] veri alinamadi")

            sleep_time = max(0, args.interval - (time.time() - t0))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nDurduruldu. Toplam: {stats['cycles']} cycle, {stats['total']} kayit")


if __name__ == "__main__":
    main()
