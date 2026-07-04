"""
adsb_producer.py
SADECE veri kaynagindan (adsb.lol, OpenSky, ...) veri ceker ve Kafka'ya
yazar. Redis/InfluxDB/MinIO gibi hicbir depoyu bilmez -- o is
tuketicilerin (consumer) sorumlulugunda. Bu ayrim sayesinde ekip
arkadaslari kendi consumer'larini bu script'e hic dokunmadan yazabilir.

============================================================
VERI KAYNAGI ADAPTORLERI -- yeni bir kaynak eklemek KOLAY:
  1) "fetch_all_<isim>(args) -> list[dict]" seklinde bir fonksiyon yaz.
     Donen HER dict, ASAGIDAKI ORTAK SEMAYA uymali (bkz. _normalize_common):
       icao24, callsign, lat, lon, alt, velocity, track, vertical_rate,
       category, squawk, emergency, is_military, source, ts
     Kaynagin desteklemedigi alanlar icin makul varsayilan kullan:
     sayisal alanlar None, metin alanlari "", is_military False,
     emergency "none". _normalize_common() bunu senin yerine yapar,
     sadece elindeki degerleri ona ver.
  2) Dosyanin sonundaki SOURCES sozlugune "isim": fetch_all_<isim> ekle.
  3) --source <isim> (veya DATA_SOURCE ortam degiskeni) ile sec.
main() dongusu SOURCES disinda hicbir yerde kaynaga OZGU kod icermez --
hangi kaynak secilirse secilsin ayni sekilde calisir.
============================================================

YAPILANDIRMA -- KOMUT SATIRI *VEYA* ORTAM DEGISKENI (ikisi de calisir,
CLI verilirse o kazanir): bu, script'in NASIL baslatildigindan bagimsiz
calismasi icin -- bugun start_all.bat (Windows "set" = ortam degiskeni),
yarin Docker (docker run -e / docker-compose "environment:") -- AYNI
DATA_SOURCE/INTERVAL/... isimleri, hic kod degisikligi gerekmeden calisir.

  DATA_SOURCE   (--source)    varsayilan: adsblol
  INTERVAL      (--interval)  varsayilan: 15
  LAT           (--lat)       varsayilan: 39.0   (sadece adsblol)
  LON           (--lon)       varsayilan: 35.0   (sadece adsblol)
  RADIUS        (--radius)    varsayilan: 12000  (sadece adsblol)
  KAFKA_BOOTSTRAP              varsayilan: localhost:9092 -- ONEMLI:
      Docker'da "localhost" container'in KENDI icini isaret eder, Kafka
      container'ina degil -- o zaman bunu servis adina (orn. "kafka:9092")
      ayarlaman GEREKECEK.

Kullanim (CLI):
    python adsb_producer.py [--source adsblol] [--interval 15]
    python adsb_producer.py --source opensky --interval 300

Kullanim (ortam degiskeni -- Docker'da boyle kullanilacak):
    set DATA_SOURCE=opensky & set INTERVAL=300 & python adsb_producer.py
    # Docker: docker run -e DATA_SOURCE=opensky -e INTERVAL=300 ...
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests
from confluent_kafka import Producer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "adsb.flights"

DEFAULT_WORLD_RADIUS_NM = 12000


def _normalize_common(icao24, callsign, lat, lon, alt_m, velocity_ms, track_deg,
                       vertical_rate_ms, category, squawk, emergency,
                       is_military, source):
    """Tum kaynaklarin USTUNDE birlestigi ortak sema. Her adaptor kendi
    ham verisini normalize etmek icin bunu cagirir -- main() ve
    downstream (Kafka -> consumer -> dashboard) sadece bu ciktiyi gorur,
    hangi kaynaktan geldigini bilmesi gerekmez."""
    return {
        "icao24": icao24,
        "callsign": callsign,
        "lat": lat,
        "lon": lon,
        "alt": alt_m,
        "velocity": velocity_ms,
        "track": track_deg,
        "vertical_rate": vertical_rate_ms,
        "category": category,
        "squawk": squawk,
        "emergency": emergency,
        "is_military": is_military,
        "source": source,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================ adsb.lol --
#
# === HIZ ARASTIRMASI GECMISI (onemli, tekrar denenmesin) ===
# Once TEK bu dev 12000nm sorgu denendi -- fetch ~57-61s, produce ~0.1-0.2s
# (Kafka darbogaz DEGIL). 15-30sn hedefiyle, dunyayi 8 kucuk bolgeye bolup
# PARALEL cekmek denendi (Turkiye'nin 500nm sorgusu hizli oldugu icin
# "sure yaricaptan degil yogunluktan etkileniyor" varsayimiyla). Iki
# bulgu cikti: (1) adsb.lol ayni IP'den ESZAMANLI istegi sinirliyor --
# 8 istegi ayni anda atinca sadece 1-2'si gercek veri donuyordu, gerisi
# suphelice hizli "0 ucak" donuyordu. (2) --workers 1 (tam SIRALI) ile
# TUM bolgeler dogru veri donduruyordu ama toplam sure ~154-166 saniyeye
# cikti (Avrupa/Afrika/Rusya gibi yogun bolgelerin HER BIRI tek basina
# 37-44sn suruyor). SONUC: bolgesel bolme YAVASLATTI -- sorgu suresi
# donen KAYIT SAYISINDAN (yogunluktan) etkileniyor, 8 parcaya bolmek tek
# dev sorguda birlestirmekten daha pahaliya geliyor + paralellik yasak.
# Tek dev sorguya donuldu, 15-30sn hedefi bu API'yle ULASILAMAZ -- bu
# sunucu tarafinda bir sinir, istemci tarafinda asilamiyor.
# === /HIZ ARASTIRMASI GECMISI ===

def _fetch_adsblol_raw(lat, lon, radius_nm, max_retries=2, retry_delay=3):
    """adsb.lol'e HTTP GET. ONEMLI: buyuk (dunya capinda) yanitlarda
    bazen baglanti veri aktarimi TAMAMLANMADAN kesiliyor (ConnectionReset
    veya IncompleteRead hatasi) -- bu, temiz bir red (403/429) DEGIL,
    aktarim SIRASINDA kopma. Sebebi kesin degil (adsb.lol sunucu tarafi
    gecici yuk/instabilite VEYA yerel agdaki bir guvenlik/proxy katmani
    uzun sureli/buyuk indirmeleri kesiyor olabilir -- ikisini buradan
    ayirt edemiyoruz). Boyle bir kopma olursa, KISA bir bekleme sonrasi
    birkac kez tekrar deniyoruz -- gecici bir aksaklıksa bu genelde
    yeterli oluyor, degilse (max_retries tukenirse) None donup normal
    hata mesaji ile bir sonraki cycle'a devam ediliyor (crash YOK)."""
    url = f"https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}"
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=90)  # dunya capinda sorgu icin yuksek timeout
            if r.status_code == 200:
                data = r.json()
                if data.get("ac"):
                    return data
            return None  # HTTP hatasi (403/429/vb.) -- tekrar denemenin faydasi yok
        except Exception as e:
            if attempt < max_retries:
                print(f"  [uyari] adsb.lol fetch hatasi (deneme {attempt + 1}/"
                      f"{max_retries + 1}, {retry_delay}sn sonra tekrar denenecek): {e}")
                time.sleep(retry_delay)
            else:
                print(f"  [uyari] adsb.lol fetch hatasi (son deneme, vazgeciliyor): {e}")
    return None


def _parse_adsblol_aircraft(ac: dict):
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
    # o an mevcut degildir (zayif sinyal, MLAT-only fix vb). "gs" eksikse
    # None birakiyoruz -- tuketici bu deger None ise InfluxDB'ye hic
    # yazmiyor, gercek bir bosluk oluyor, sahte sifir degil.
    raw_gs = ac.get("gs")
    velocity = round(float(raw_gs) * 0.5144, 1) if raw_gs is not None else None

    raw_baro_rate = ac.get("baro_rate")
    vertical_rate = (round(float(raw_baro_rate) * 0.00508, 2)
                     if raw_baro_rate is not None else None)

    # ONEMLI: ayni sekilde "track" (yon) eksikse sahte 0 (kuzey) YAZMIYORUZ.
    raw_track = ac.get("track")
    track = float(raw_track) if raw_track is not None else None

    # ASKERI/SIVIL AYRIMI: adsb.lol, ADSBExchange/readsb ile ayni "dbFlags"
    # bit alanini kullaniyor -- 1. bit (dbFlags & 1) askeri ucak demek.
    try:
        is_military = bool(int(ac.get("dbFlags", 0) or 0) & 1)
    except (TypeError, ValueError):
        is_military = False

    return _normalize_common(
        icao24=icao, callsign=(ac.get("flight") or "").strip(),
        lat=round(float(lat), 6), lon=round(float(lon), 6),
        alt_m=round(float(alt) * 0.3048, 1),
        velocity_ms=velocity, track_deg=track, vertical_rate_ms=vertical_rate,
        category=ac.get("category", ""), squawk=ac.get("squawk", ""),
        emergency=ac.get("emergency", "none"), is_military=is_military,
        source="adsblol",
    )


def fetch_all_adsblol(args):
    """adsb.lol -- tek buyuk yaricapli (--radius, varsayilan 12000nm =
    tum dunya) sorgu. ~57-61sn suruyor, bkz. yukaridaki HIZ ARASTIRMASI."""
    raw = _fetch_adsblol_raw(args.lat, args.lon, args.radius)
    if not raw:
        return []
    out = []
    for ac in raw.get("ac", []):
        rec = _parse_adsblol_aircraft(ac)
        if rec:
            out.append(rec)
    return out


# ============================================================ OpenSky --
#
# ONEMLI KISITLAR (denenmeden once oku):
#   - Anonim kullanici gunde SADECE 400 "kredi" alir -- surekli 15-60sn'de
#     bir sorgulamaya YETMEZ, birkac dakika icinde tukenir ve 429 (Too
#     Many Requests) almaya baslarsin. --interval'i BUYUK tut (orn. 300+).
#   - Kimlik dogrulamali (OAuth2 client credentials) erisim gunluk krediyi
#     4000'e cikarir ama bu adaptor SADECE anonim erisimi uyguluyor --
#     token yonetimi (client_id/secret -> access_token, 30dk'da bir
#     yenileme) eklenmedi, kapsam disi birakildi. Gerekirse ayrica
#     eklenebilir (bkz. openskynetwork.github.io/opensky-api).
#   - is_military VE category alanlari OpenSky'de yok -- her zaman
#     sirasiyla False ve "" donuyor (adsb.lol'deki gibi zengin degil).
#   - Bu, DUNYA capinda TEK istekle calisir (bbox verilmezse tum aktif
#     ucaklari doner) -- adsb.lol'deki gibi yaricap/bolge kavramı yok.

OPENSKY_EMERGENCY_SQUAWKS = {"7500": "unlawful", "7600": "nordo", "7700": "general"}


def _parse_opensky_state(state):
    """OpenSky state vector -- DICT DEGIL, POZISYONEL DIZI. Index'ler
    resmi API dokumantasyonuna gore (openskynetwork.github.io/opensky-api):
    0=icao24 1=callsign 2=origin_country 3=time_position 4=last_contact
    5=longitude 6=latitude 7=baro_altitude 8=on_ground 9=velocity
    10=true_track 11=vertical_rate 12=sensors 13=geo_altitude 14=squawk
    15=spi 16=position_source [17=category, sadece extended=true ile]."""
    try:
        icao24 = (state[0] or "").strip().lower()
        callsign = (state[1] or "").strip()
        lon, lat = state[5], state[6]
        baro_alt = state[7]
        on_ground = state[8]
        velocity, track, vertical_rate = state[9], state[10], state[11]
        squawk = state[14] or ""
    except (IndexError, TypeError):
        return None

    if not icao24 or lat is None or lon is None or on_ground:
        return None

    return _normalize_common(
        icao24=icao24, callsign=callsign,
        lat=round(float(lat), 6), lon=round(float(lon), 6),
        alt_m=round(float(baro_alt), 1) if baro_alt is not None else 0.0,
        # ONEMLI: OpenSky JSON'unda bir deger tam sayiysa (orn. 0, 0.5
        # degil) Python bunu int olarak parse eder. adsb.lol tarafi bu
        # alanlari hep float(...) ile aciyordu, burada UNUTULMUSTU --
        # InfluxDB bir alanin tipini (int/float) ILK YAZIMDA sabitliyor,
        # sonra farkli tip gelince "field type conflict" hatasi veriyor.
        # None'u KORUYORUZ (sahte 0 yazmiyoruz), ama None degilse MUTLAKA
        # float'a ceviriyoruz.
        velocity_ms=float(velocity) if velocity is not None else None,
        track_deg=float(track) if track is not None else None,
        vertical_rate_ms=float(vertical_rate) if vertical_rate is not None else None,
        category="",  # OpenSky kategori kodlari adsb.lol'unkiyle eslesmiyor
        squawk=squawk, emergency=OPENSKY_EMERGENCY_SQUAWKS.get(squawk, "none"),
        is_military=False,  # OpenSky bu bilgiyi vermiyor
        source="opensky",
    )


def fetch_all_opensky(args):
    """OpenSky Network states/all -- anonim erisim (kimlik dogrulama
    YOK). Rate limit'e takilirsan --interval'i (orn. 300sn) artir."""
    url = "https://opensky-network.org/api/states/all"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            print("  [uyari] OpenSky: rate limit asildi (429) -- "
                  "--interval'i artir (anonim: gunde 400 kredi)")
            return []
        if r.status_code != 200:
            print(f"  [uyari] OpenSky fetch hatasi: HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  [uyari] OpenSky fetch hatasi: {e}")
        return []

    out = []
    for state in (data.get("states") or []):
        rec = _parse_opensky_state(state)
        if rec:
            out.append(rec)
    return out


# ============================================================ Kayit --
# Yeni kaynak eklemek icin buraya "isim": fetch_all_isim ekle -- baska
# hicbir yeri degistirmen gerekmiyor.
SOURCES = {
    "adsblol": fetch_all_adsblol,
    "opensky": fetch_all_opensky,
}


def delivery_report(err, msg):
    if err is not None:
        print(f"  [uyari] teslim hatasi: {err}")


def main():
    # ONEMLI: her arg once CLI'dan (--source vb.), verilmezse ortam
    # degiskeninden (DATA_SOURCE vb.), o da yoksa sabit varsayilandan
    # okunuyor. Boylece bu script'i HIC DEGISTIRMEDEN hem "python
    # adsb_producer.py --source opensky" ile hem de Docker'da
    # "docker run -e DATA_SOURCE=opensky ..." ile calistirabilirsin.
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES.keys()),
                    default=os.environ.get("DATA_SOURCE", "adsblol"),
                    help="Veri kaynagi (varsayilan: adsblol; ortam degiskeni: DATA_SOURCE)")
    ap.add_argument("--interval", type=int,
                    default=int(os.environ.get("INTERVAL", "15")),
                    help="Ortam degiskeni: INTERVAL")
    ap.add_argument("--lat", type=float, default=float(os.environ.get("LAT", "39.0")),
                    help="Sadece --source adsblol icin. Ortam degiskeni: LAT")
    ap.add_argument("--lon", type=float, default=float(os.environ.get("LON", "35.0")),
                    help="Sadece --source adsblol icin. Ortam degiskeni: LON")
    ap.add_argument("--radius", type=int,
                    default=int(os.environ.get("RADIUS", str(DEFAULT_WORLD_RADIUS_NM))),
                    help="Sadece --source adsblol icin. Ortam degiskeni: RADIUS")
    args = ap.parse_args()

    fetch_fn = SOURCES[args.source]

    print(f"Kafka producer hazir (topic={TOPIC})")
    print(f"Veri kaynagi: {args.source}  hedef-araligi={args.interval}sn")
    print("NOT: gercek cycle suresi hedef araliktan uzun olabilir -- asagidaki")
    print("     [timing] satirindan fetch/produce'un ne kadar surdugunu izle.")
    print("Durdurmak icin Ctrl+C\n")

    # ONEMLI: linger.ms/batch.size/compression.type -- binlerce mesajlik
    # burst'leri daha az sayida, daha buyuk ag paketiyle gondermek icin.
    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 50,
        "batch.size": 262144,       # 256 KB (varsayilan 16 KB)
        "compression.type": "lz4",
        "queue.buffering.max.messages": 200000,
    })

    stats = {"cycles": 0, "total": 0}

    try:
        while True:
            t0 = time.time()

            t_fetch_start = time.time()
            records = fetch_fn(args)
            t_fetch = time.time() - t_fetch_start

            if records:
                # ONEMLI: Redis'teki her ucagin kaydi bir TTL (yasam suresi)
                # ile tutulur -- tuketici (dashboard_consumer.py), YENI veri
                # gelmezse "sinyal kaybetti" varsayimiyla bunu otomatik siler.
                # Bu TTL, kaynagin GERCEK tazeleme araligindan KISA olursa,
                # her ucak bir sonraki veri gelmeden ONCE Redis'ten duser --
                # harita bosalir, sonra yeni veri gelince tekrar dolar --
                # "kaybolup geri geliyor" hatasinin TAM SEBEBI budur.
                #
                # ILK DENEMEDE BUYUK BIR HATA YAPILDI: ttl_hint = args.interval
                # * 3 hesaplanmisti -- ama args.interval sadece HEDEF, adsblol
                # gibi kaynaklarda GERCEK cycle suresi (t_fetch) hedeften COK
                # daha uzun olabiliyor (dunya sorgusunda hedef=15sn ama gercek
                # ~57-61sn olculdu!) -- interval'e gore hesaplayinca TTL yine
                # cok kisa kaliyordu, AYNI hatayi bu sefer adsblol icin
                # yeniden yaratmis olduk. DOGRUSU: hedef ile GERCEK olculen
                # fetch suresinden BUYUK OLANI kullanmak.
                #
                # IKINCI AYAR: guvenlik payi 3x'ten dusuruldu. 3x ile ekranda
                # "aktif ucus" sayisi tek cycle'dan ~%24 fazla cikiyordu
                # (OpenSky'de TTL=900sn=15dk -- son 3 cycle'in BIRLESIMI
                # gosteriliyordu, hem sayiyi sisiriyor hem de bazi ucaklarin
                # 15dk'ya kadar BAYAT pozisyonda gorunmesine yol aciyordu).
                #
                # UCUNCU AYAR (tek bir sabit carpan YETERSIZDI): OpenSky ile
                # adsblol'un DAVRANISI FARKLI -- OpenSky'de gercek cekme
                # suresi hep hizli (~1sn), bekleme SADECE sabit --interval'dan
                # geliyor (COK KARARLI) -- kucuk bir pay (1.5x) yeterli.
                # adsblol'de ise (dunya sorgusu) gercek sure hedefi ZATEN
                # asiyor ve KENDISI degisken (57-166sn arasi olculmustu) --
                # buraya sadece 1.5x uygulamak riskli olurdu (58sn'lik tipik
                # sureye sadece ~30sn pay birakirdi). Iki rejimi ayirdik:
                #   - gercek sure hedefi asiyorsa (degisken, is-agirlikli):
                #     gercek surenin 2 kati -- daha genis pay
                #   - hedef domine ediyorsa (kararli, OpenSky gibi):
                #     hedefin 1.5 kati -- dar pay yeterli
                if t_fetch > args.interval:
                    ttl_hint = max(60, int(t_fetch * 2))
                else:
                    ttl_hint = max(60, int(args.interval * 1.5))

                t_produce_start = time.time()
                for rec in records:
                    rec["ttl_hint"] = ttl_hint
                    producer.produce(
                        TOPIC, key=rec["icao24"],
                        value=json.dumps(rec).encode(),
                        callback=delivery_report,
                    )
                producer.flush()
                t_produce = time.time() - t_produce_start

                stats["cycles"] += 1
                stats["total"] += len(records)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{len(records)} ucus -> Kafka (kaynak={args.source}, "
                      f"cycle={stats['cycles']}, toplam={stats['total']})")
                print(f"  [timing] fetch={t_fetch:.1f}s  produce={t_produce:.1f}s  "
                      f"cycle-toplam={time.time() - t0:.1f}s  "
                      f"(hedef araligi={args.interval}s, ttl_hint={ttl_hint}s)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] veri alinamadi "
                      f"(kaynak={args.source}, fetch={t_fetch:.1f}s)")

            sleep_time = max(0, args.interval - (time.time() - t0))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nDurduruldu. Toplam: {stats['cycles']} cycle, {stats['total']} kayit")


if __name__ == "__main__":
    main()
