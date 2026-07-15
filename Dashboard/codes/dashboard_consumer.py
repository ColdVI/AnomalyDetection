"""
dashboard_consumer.py
Bu SENIN (dashboard) tuketicin. Kendi group.id'si var ("dashboard-consumer"),
yani ekip arkadaslarinin ekleyecegi baska consumer'lar (MinIO arsivleyici,
model anomali tespiti) bu script'ten tamamen bagimsiz calisir -- ayni
mesaji herkes kendi hizinda okur, birbirini etkilemez.

Iki topic dinler:
  - uav.flights  -> Redis (canli durum) + InfluxDB (7 gunluk gecmis)
  - uav.alerts   -> Redis (son alert listesi) -- SIMDILIK BOS, model ekibi
                     hazir olunca buraya yazmaya baslayacak, bu consumer
                     otomatik olarak onlari da yakalayip dashboard'a
                     yansitmaya baslayacak, kod degisikligi gerekmez.

Kullanim:
    python dashboard_consumer.py
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import redis
from confluent_kafka import Consumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import WriteOptions

# ONEMLI: logging_setup ciplak import edilebilsin diye -- bu dosya hem
# Docker'da "python codes/dashboard_consumer.py" ile __main__ olarak, hem de
# testlerde "from Dashboard.codes import dashboard_consumer" ile calisiyor
# (bkz. app.py'deki ayni sekildeki sys.path shim yorumu, ayni sebep).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_setup import enable_file_logging

# ONEMLI: Docker'da "localhost" container'in KENDI icini isaret eder --
# BOOTSTRAP/REDIS_HOST/INFLUX_HOST bu yuzden ortam degiskeniyle
# ayarlanabilir (docker-compose.yml servis adlarini enjekte eder).
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
FLIGHTS_TOPIC = "uav.flights"
ALERTS_TOPIC = "uav.alerts"

TOKEN_FILE = Path("influx_token.txt")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
INFLUX_HOST = os.environ.get("INFLUX_HOST", "http://localhost:8086")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "iha-org")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "uav-history")

# ============================================================
# CYCLE-ID TABANLI "GOR-YOKSA-SIL" MODELI (once TTL, sonra zaman-pencereli
# yaklasim denendi, ikisi de kaldirildi)
#
# 1. DENEME (TTL): her ucak kaydi Redis'e bir TTL (yasam suresi) ile
# yaziliyordu. "iha:active_flights" kumesinde SREM hic yapilmadigi icin,
# tekil key TTL ile silinse bile icao24 kumede kalmaya devam ediyordu
# ("hayalet kayit") -- periyodik bakimla temizleniyordu ama Redis'in
# burada "kalici bir depo" gibi davranmasini gerektiriyordu.
#
# 2. DENEME (zaman penceresi, WINDOW_SEC): TTL yerine, belirli bir
# SANIYE penceresi boyunca goruleni takip edip pencere kapaninca
# gorulmeyeni silme. Hayalet sorununu cozdu ama YENI bir sorun getirdi:
# pencere suresi, kaynagin GERCEK cycle suresine gore TAHMIN edilmek
# zorundaydi (110-600sn arasi, kaynaga gore elle ayarlaniyordu) -- pencere
# birden fazla cycle'i kapsadiginda (orn. 180sn/55sn≈3.3 cycle), ekranda
# "birden fazla cycle'in BIRLESIMI" gosteriliyordu (%13'e kadar fazlalik
# olcduk). Tek dogru saniye degeri yoktu -- ya fazlalik ya da titreme
# riski araninda hep bir denge kurulmasi gerekiyordu.
#
# 3. (GUNCEL) CYCLE-ID: saniye TAHMIN ETMEK yerine, producer'in KENDI
# dogal cycle sinirini kullaniyoruz. uav_producer.py her kayda hangi
# cycle'a ait oldugunu gosteren bir "cycle_id" ekliyor (stats["cycles"]).
# Bu consumer, gelen mesajlardaki cycle_id DEGISTIGINDE "onceki cycle
# TAMAMLANDI" sinyalini alir ve o cycle'da hic gorulmeyen eski kayitlari
# ANINDA siler. SANIYE TAHMINI YOK -- her zaman TAM OLARAK bir onceki
# TAMAMLANMIS cycle ile karsilastirilir, bu da fazlaligi SIFIRA indirir.
# Bir cycle basarisiz olursa (adsb.lol baglanti hatasi vb.), producer o
# cycle icin HICBIR mesaj/cycle_id degisikligi uretmez -- consumer da
# sessizce mevcut veriyi korur, saniyeye dayali bir zaman asimi riski YOK.
#
# 4. DENENIP GERI ALINAN (regresyon!): mesajlar geldikce Redis'e TEK TEK
# yazildigi icin, bir cycle akarken (birkaç saniye) Redis KISA SURELIGINE
# "eskinin bir kismi + yeninin bir kismi" karisik bir durumda kaliyordu --
# bunu COZMEK icin bir cycle'in TUM verisini once BELLEKTE toplayip,
# cycle TAMAMLANINCA TEK BIR ATOMIK islemde yazma denendi. AMA bu, bir
# cycle'in verisinin gorunur olmasini bir SONRAKI cycle baslayana kadar
# (yani ~1 TAM CYCLE SURESI, orn. ~56sn) ERTELEDI -- ekran surekli
# producer'in BIR ONCEKI cycle'ini gosterir hale geldi (kullanici bunu
# "ekran = producer - 1" seklinde fark etti). Birkaç saniyelik KUCUK bir
# tutarsizligi cozeyim derken, COK DAHA BUYUK bir gecikme (56sn) yaratildi
# -- bu bir REGRESYONDU, GERI ALINDI. Mesajlar yine GELDIKCE ANINDA
# yaziliyor (dusuk gecikme onceligi), birkac saniyelik gecis-penceresi
# tutarsizligi KABUL EDILEN bir bedel (0 gecikmeye kiyasla cok daha iyi).


def sweep_stale_flights(rdb, baseline_set, seen_this_cycle, batch_size=500):
    """Bir cycle tamamlaninca cagrilir. "baseline_set" -- BU cycle
    BASLAMADAN HEMEN ONCE Redis'in durumu (bir onceki TAMAMLANMIS
    cycle'in tam verisi). "seen_this_cycle" -- BU cycle'da GERCEKTEN
    gorulen icao24'ler. baseline_set - seen_this_cycle farki (eski
    cycle'da vardi ama bu cycle'da hic gelmedi) artik gecersiz sayilip
    SILINIR -- hem tekil "iha:state:{icao}" key'i (DEL) hem kume
    uyeligi (SREM).

    ONEMLI (gercek bir hata buradan cikti, dikkat!): sweep CAGIRILDIGI
    ANDA Redis'in GUNCEL durumunu (SMEMBERS) baseline olarak KULLANMA --
    o an Redis zaten BU cycle'in kendi verisini icerir (mesajlar
    geldikce ANINDA yaziliyor), yani "cycle'i kendisiyle" karsilastirmis
    olursun, fark HICBIR ZAMAN bulunmaz. Baseline, cycle BASLAMADAN
    ONCE ayrica alinip saklanmis olmali (bkz. main() icindeki
    baseline_before_cycle degiskeni)."""
    try:
        stale = baseline_set - seen_this_cycle
        if not stale:
            return 0
        stale_list = list(stale)
        pipe = rdb.pipeline()
        for icao in stale_list:
            pipe.delete(f"iha:state:{icao}")
        for i in range(0, len(stale_list), batch_size):
            pipe.srem("iha:active_flights", *stale_list[i:i + batch_size])
        pipe.execute()
        return len(stale)
    except Exception as e:
        print(f"  [uyari] pencere temizligi hatasi: {e}")
        return 0


def load_token() -> str:
    env_token = os.environ.get("INFLUX_TOKEN")
    if env_token:
        return env_token
    if not TOKEN_FILE.exists():
        raise SystemExit("influx_token.txt bulunamadi ve INFLUX_TOKEN ortam degiskeni yok. "
                          "Once setup_local_windows.py calistir (native) ya da INFLUX_TOKEN set et (docker).")
    return TOKEN_FILE.read_text().strip()


def main():
    token = load_token()

    rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True,
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
    current_cycle_id = None
    seen_this_cycle = set()
    baseline_before_cycle = None  # bu cycle BASLAMADAN once Redis'in durumu

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

                # ONEMLI: producer'in gonderdigi "cycle_id" DEGISTIYSE,
                # bir onceki cycle TAMAMLANMIS demektir -- o cycle'i,
                # KENDI basladigi andaki Redis durumuyla (baseline_before_cycle)
                # karsilastirip farkini SIMDI temizliyoruz. Ilk mesajda
                # (current_cycle_id=None) veya cycle_id gelmeyen eski/farkli
                # bir kaynaktan gelen kayitlarda GUVENLI VARSAYILAN: temizlik
                # yapma, sadece takip etmeye basla.
                cycle_id = data.get("cycle_id")
                if cycle_id is not None and cycle_id != current_cycle_id:
                    if current_cycle_id is not None and baseline_before_cycle is not None:
                        removed = sweep_stale_flights(rdb, baseline_before_cycle, seen_this_cycle)
                        print(f"[cycle] {current_cycle_id} tamamlandi -- "
                              f"{len(seen_this_cycle)} ucak goruldu, "
                              f"{removed} eski kayit silindi")
                    # YENI cycle basliyor -- Redis'in SU ANKI (henuz bu
                    # yeni cycle'in verisi hic yazilmamis) durumunu, BIR
                    # SONRAKI temizlik icin baseline olarak sakla.
                    baseline_before_cycle = set(rdb.smembers("iha:active_flights"))
                    current_cycle_id = cycle_id
                    seen_this_cycle = set()

                # ONEMLI: mesaj geldigi anda ANINDA yaziliyor -- DUSUK
                # GECIKME buradaki oncelik (bkz. yukaridaki "4. DENENIP
                # GERI ALINAN" notu). Gecerlilik sadece "bu cycle'da
                # goruldu mu" sorusuna dayaniyor, TTL YOK.
                rdb.set(f"iha:state:{icao}", json.dumps(data))
                rdb.sadd("iha:active_flights", icao)
                seen_this_cycle.add(icao)

                point = (
                    Point("flights")
                    .tag("icao24", icao)
                    .field("lat", float(data.get("lat", 0.0)))
                    .field("lon", float(data.get("lon", 0.0)))
                    .field("alt", float(data.get("alt", 0.0)))
                    .field("is_ground", bool(data.get("is_ground", False)))
                    # 2026-07-10 (kullanici istegi): uav_producer.py zaten dbFlags'ten
                    # is_military hesaplayip Kafka'ya koyuyor -- burada da yaziyoruz ki
                    # individual/metehan_geo'nun realtime (live/24h/7d) haritalari
                    # sivil/askeri filtresini uygulayabilsin. Sadece BUNDAN SONRA yazilan
                    # noktalarda olacak -- gecmis InfluxDB verisinde bu alan yok.
                    .field("is_military", bool(data.get("is_military", False)))
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
                # sema: docs/KAFKA_SCHEMA.md'deki uav.alerts bolumune bakiniz
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
    enable_file_logging("dashboard_consumer")
    main()
