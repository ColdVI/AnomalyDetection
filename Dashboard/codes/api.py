"""app.py'den cikarildi, adim 4 -- FastAPI endpoint'leri ve onlara ozel
yardimci fonksiyonlar. _run_api() ve if __name__ == "__main__": bloğu
BILEREK burada DEGIL, app.py'de kaliyor (bkz. server.py'nin ayni kuralı).

ONEMLI: _query_history_df ve get_replay, app.py'deki Dash callback'leri
(download_history_csv, load_replay_data) tarafindan HTTP round-trip'ten
kacinmak icin DOGRUDAN da cagriliyor -- app.py bunlari (ve testlerin
dogrudan cagirdigi digerlerini) "from api import ..." ile GERI import
ediyor, boylece hem monkeypatch'ler hem dogrudan cagrilar app.py'nin
KENDI namespace'i uzerinden calismaya devam ediyor."""

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from datetime import datetime, timedelta, timezone

import pandas as pd
import redis
import requests
from fastapi import Response

from server import app_api, _rpool, _query_api, INFLUX_BUCKET
from constants import (
    DATA_SOURCES, DEFAULT_DATA_SOURCE, REDIS_DATA_SOURCE_KEY,
    REDIS_PRODUCER_STATUS_KEY, REPLAY_MAX_RANGE_HOURS, GEOCODE_CACHE_TTL,
    GEOCODE_MAX_LOOKUPS_PER_REQUEST, FLIGHT_GAP_THRESHOLD_MIN,
)

def _get_flights():
    r = redis.Redis(connection_pool=_rpool)
    icaos = list(r.smembers("iha:active_flights"))
    if not icaos:
        return []
    # ONEMLI: "Dunya" modunda binlerce ucak olabilir -- N adet ayri ayri
    # r.get() cagirmak (N round-trip) o olcekte ciddi yavasliga yol acardi.
    # Tek bir MGET ile hepsini bir seferde cekiyoruz (2 round-trip toplam:
    # SMEMBERS + MGET). TTL'i gecmis/silinmis key'ler icin MGET None doner,
    # onlari filtreliyoruz.
    raws = r.mget([f"iha:state:{icao}" for icao in icaos])
    out = [json.loads(raw) for raw in raws if raw]
    return sorted(out, key=lambda x: x.get("icao24", ""))


@app_api.get("/api/flights")
def get_flights():
    return _get_flights()


@app_api.get("/api/alerts")
def get_alerts():
    r = redis.Redis(connection_pool=_rpool)
    return [json.loads(a) for a in r.lrange("iha:recent_alerts", 0, 9)]


def _fetch_adsblol_route(callsign: str, lat: float, lon: float):
    """adsb.lol'un KENDI rota API'si -- adsbdb.com'dan FARKLI, muhtemelen
    daha guncel bir kaynak (VRS standing data + adsb.lol'un kendi
    plausibility filtresi). Kullanici WZZ43 testinde adsbdb.com YANLIS,
    adsb.lol'un kendi sitesi DOGRU rota gosterdigi icin bu tekrar denendi.

    ONCEKI DENEMEDE (6 varyasyon) HEPSI ayni bos "201 text/html" yanitini
    veriyordu -- COZULDU: sebep istek govdesi degil, EKSIK Origin/Referer
    basliklariydi (muhtemelen bir CORS/bot-koruma katmani, sadece
    adsb.lol'un KENDI sitesinden gelen isteklere gercek yanit veriyor).
    Origin+Referer eklenince endpoint GERCEK bir 422 hatasi verdi --
    bu da GERCEK semayi ortaya cikardi: "callsign" YETMIYOR, "lat" ve
    "lng" de ZORUNLU (muhtemelen adsb.lol kendi "plausible" filtresini
    ucagin GERCEK konumuna gore hesapliyor -- bkz. "plausible" alani).
    Bu yuzden bu fonksiyon artik lat/lon PARAMETRE olarak ALIYOR --
    caginin bunlari saglamasi sart, yoksa istek zaten calismaz."""
    try:
        resp = requests.post(
            "https://api.adsb.lol/api/0/routeset",
            json={"planes": [{"callsign": callsign, "lat": lat, "lng": lon}]},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json",
                "Origin": "https://adsb.lol",
                "Referer": "https://adsb.lol/",
            },
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        route = data[0]
        if not route.get("airport_codes") or route.get("airport_codes") == "unknown":
            return None
        airports = route.get("_airports") or []
        if len(airports) < 2:
            return None
        origin, dest = airports[0], airports[-1]  # ara durak varsa atlaniyor, ilk/son alinir
        return {
            "found": True,
            "airline": route.get("airline_code"),  # ICAO kodu -- adsbdb kadar zengin (tam isim) degil
            "origin_name": origin.get("name"),
            "origin_iata": origin.get("iata"),
            "origin_city": origin.get("location"),
            "origin_lat": origin.get("lat"),
            "origin_lon": origin.get("lon"),
            "dest_name": dest.get("name"),
            "dest_iata": dest.get("iata"),
            "dest_city": dest.get("location"),
            "dest_lat": dest.get("lat"),
            "dest_lon": dest.get("lon"),
            # adsb.lol'un KENDI guven bayragi -- update_route_info/line
            # bunu bizim _route_is_plausible() kontrolumuzle BIRLIKTE
            # kullaniyor.
            "source_plausible": bool(route.get("plausible", True)),
        }
    except Exception:
        return None


@app_api.get("/api/route/{callsign}")
def get_route(callsign: str, lat: float = None, lon: float = None):
    """Kalkis/varis rotasi -- ONCE adsb.lol'un kendi route API'sini
    dener (dogru sema + basliklarla artik CALISIYOR, bkz.
    _fetch_adsblol_route docstring'i), basarisiz olursa VEYA lat/lon
    saglanmamissa adsbdb.com'a DUSER. Rota nadiren degistigi icin
    Redis'te 12 saat cache'liyoruz, her secimde dis API'ye vurmayalim."""
    callsign = callsign.strip().upper()
    if not callsign:
        return {"found": False}

    r = redis.Redis(connection_pool=_rpool)
    cache_key = f"iha:route:{callsign}"
    cached = r.get(cache_key)
    if cached is not None:
        return json.loads(cached)

    result = None
    if lat is not None and lon is not None:
        result = _fetch_adsblol_route(callsign, lat, lon)

    if result is None:
        result = {"found": False}
        try:
            resp = requests.get(f"https://api.adsbdb.com/v0/callsign/{callsign}", timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("response", {})
                route = data.get("flightroute")
                if route:
                    origin = route.get("origin") or {}
                    dest = route.get("destination") or {}
                    result = {
                        "found": True,
                        "airline": (route.get("airline") or {}).get("name"),
                        "origin_name": origin.get("name"),
                        "origin_iata": origin.get("iata_code"),
                        "origin_city": origin.get("municipality"),
                        "origin_lat": origin.get("latitude"),
                        "origin_lon": origin.get("longitude"),
                        "dest_name": dest.get("name"),
                        "dest_iata": dest.get("iata_code"),
                        "dest_city": dest.get("municipality"),
                        "dest_lat": dest.get("latitude"),
                        "dest_lon": dest.get("longitude"),
                    }
        except Exception:
            pass  # bulunamadi/erisilemedi -- found:False donuyoruz, cache'lemiyoruz

    # 12 saat cache -- bulunamadi sonucunu da cache'liyoruz (ayni callsign icin
    # tekrar tekrar bosuna sorgu atmayalim), ama daha kisa sureli (1 saat)
    ttl = 43200 if result["found"] else 3600
    r.set(cache_key, json.dumps(result), ex=ttl)
    return result


@app_api.get("/api/aircraft_info/{icao24}")
def get_aircraft_info(icao24: str):
    """adsbdb.com'un ayni ucretsiz veritabaninin BASKA bir endpoint'i --
    bu sefer callsign degil, ICAO24 hex (mode_s) ile uçak tipi/uretici/
    tescil/sahip bilgisi donduruyor. Bu veri neredeyse hic degismedigi
    icin (sahiplik degisikligi disinda) 7 gun cache'liyoruz."""
    icao24 = icao24.strip().lower()
    if not icao24:
        return {"found": False}

    r = redis.Redis(connection_pool=_rpool)
    cache_key = f"iha:aircraft_info:{icao24}"
    cached = r.get(cache_key)
    if cached is not None:
        return json.loads(cached)

    result = {"found": False}
    try:
        resp = requests.get(f"https://api.adsbdb.com/v0/aircraft/{icao24}", timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("response", {})
            aircraft = data.get("aircraft")
            if aircraft:
                result = {
                    "found": True,
                    "type": aircraft.get("type"),
                    "manufacturer": aircraft.get("manufacturer"),
                    "registration": aircraft.get("registration"),
                    "owner": aircraft.get("registered_owner"),
                    "owner_country": aircraft.get("registered_owner_country_name"),
                    "photo_thumb": aircraft.get("url_photo_thumbnail"),
                }
    except Exception:
        pass

    ttl = 604800 if result["found"] else 3600  # bulunursa 7 gun, bulunamazsa 1 saat
    r.set(cache_key, json.dumps(result), ex=ttl)
    return result


@app_api.get("/api/health")
def health():
    fl = _get_flights()
    return {"status": "ok", "active_flights": len(fl)}


@app_api.get("/api/data_source")
def get_data_source():
    """Dashboard'un kaynak butonlarinin okudugu endpoint -- HEM istenen
    (dashboard'dan en son yazilan) HEM de GERCEKTE aktif (uav_producer.py
    kendi cycle'inda yazdigi) kaynagi ayri ayri donduruyor. Producer bir
    sonraki cycle'a kadar (60/300sn) istenen degisikligi henuz uygulamamis
    olabilir -- ikisi FARKLIYSA arayuz "gecis bekleniyor" gosterebiliyor."""
    r = redis.Redis(connection_pool=_rpool)
    requested = r.get(REDIS_DATA_SOURCE_KEY) or DEFAULT_DATA_SOURCE
    status_raw = r.get(REDIS_PRODUCER_STATUS_KEY)
    active = json.loads(status_raw) if status_raw else None
    return {"requested": requested, "active": active}


@app_api.post("/api/data_source")
def set_data_source(source: str):
    if source not in DATA_SOURCES:
        return {"error": f"gecersiz kaynak: {source}"}
    r = redis.Redis(connection_pool=_rpool)
    r.set(REDIS_DATA_SOURCE_KEY, source)
    return {"requested": source}


def _query_history_df(icao24: str, hours: int = 24, start: str = None, end: str = None):
    """/api/history/{icao24} (JSON) VE /api/history/{icao24}/csv (CSV indirme)
    AYNI Flux sorgusunu paylasir -- bu ikisi arasinda tekrarlanmasin diye
    ortak DataFrame-donduren govde buraya cikarildi. Hata durumunda
    (gecersiz start/end VEYA sorgu hatasi) None ile birlikte bir hata
    mesaji donuyor, cagiran taraf (JSON->{"error":...}, CSV->HTTP 400)
    kendi formatina cevirir.

    ONEMLI: start/end verilirse (tarih araligi secici) hours YOK SAYILIR.
    start/end ham string'i DOGRUDAN flux sorgusuna GOMMUYORUZ (injection
    riski -- kullanicidan gelen deger) -- once datetime.fromisoformat ile
    PARSE edip, KENDI ISO string'imize (guvenli, tek format, sadece
    dogrulanmis tarih/saat bilgisi) geri cevirip OYLE gomuyoruz."""
    if start and end:
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            return None, "invalid start/end (ISO8601 bekleniyor)"
        range_clause = (
            f'range(start: {start_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, '
            f'stop: {end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})'
        )
    else:
        hours = min(hours, 24 * 7)  # bucket zaten 7 gunden fazlasini tutmuyor
        range_clause = f'range(start: -{hours}h)'
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> {range_clause}
      |> filter(fn: (r) => r["_measurement"] == "flights")
      |> filter(fn: (r) => r["icao24"] == "{icao24}")
      |> filter(fn: (r) => r["_field"] == "alt" or r["_field"] == "velocity"
                         or r["_field"] == "lat" or r["_field"] == "lon")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        tables = _query_api.query_data_frame(flux)
        if isinstance(tables, list):
            tables = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if tables.empty:
            return pd.DataFrame(columns=["_time", "lat", "lon", "alt", "velocity"]), None
        tables = tables.sort_values("_time")
        # ONEMLI: date_format="iso" tek basina yetmeyebilir -- influxdb_client
        # bazen "_time" kolonunu object dtype (duz Python datetime) olarak
        # donduruyor, bu durumda pandas'in date_format parametresi devreye
        # girmiyor. Kolonu JSON'a cevirmeden ONCE acikca ISO string'e
        # ceviriyoruz, boylece hicbir belirsizlik kalmiyor.
        tables["_time"] = pd.to_datetime(tables["_time"], utc=True) \
                             .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        # ONEMLI: bir ucak icin secilen aralikta HIC velocity verisi
        # yoksa (her poll'da eksikmis), pivot sonucunda "velocity" kolonu
        # tamamen olusmayabiliyor -- once eksikse ekleyip NaN ile
        # dolduruyoruz, yoksa asagidaki satir KeyError verirdi.
        for col in ["lat", "lon", "alt", "velocity"]:
            if col not in tables.columns:
                tables[col] = None
        return tables[["_time", "lat", "lon", "alt", "velocity"]], None
    except Exception as e:
        return None, str(e)


@app_api.get("/api/history/{icao24}")
def get_history(icao24: str, hours: int = 24, start: str = None, end: str = None):
    df, err = _query_history_df(icao24, hours=hours, start=start, end=end)
    if err:
        return {"error": err}
    return json.loads(df.to_json(orient="records"))


@app_api.get("/api/history/{icao24}/csv")
def get_history_csv(icao24: str, hours: int = 24, start: str = None, end: str = None):
    """Ayni gecmis veriyi (JSON endpoint'iyle AYNI Flux sorgusu, bkz.
    _query_history_df) CSV dosyasi olarak indirir -- dashboard'daki
    "İndir" butonu bu URL'e dogrudan yonlendiriyor (tarayici indirme
    olarak isliyor, Content-Disposition: attachment sayesinde)."""
    df, err = _query_history_df(icao24, hours=hours, start=start, end=end)
    if err:
        return Response(content=f"error: {err}", media_type="text/plain", status_code=400)
    csv_bytes = df.rename(columns={"_time": "timestamp_utc"}).to_csv(index=False)
    filename = f"{icao24}_history.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



@app_api.get("/api/replay")
def get_replay(start: str, end: str, step_sec: int = 30):
    """Belirli bir zaman araligi icin, GERCEKTEN veri olan sabit-araliklarla
    (step_sec) "adim" zaman damgalarini dondurur -- tekrar oynatma (replay)
    ozelligi bunlar arasinda adim adim ilerler.

    ONEMLI (canli testte YAKALANAN gercek sorun -- tekrar denenmesin):
    ONCEDEN bu endpoint HER adimin TUM ucak listesini ("frames") TEK bir
    payload'da donduruyordu. Kuresel trafikte (~7000 ucak) 40 adim bile
    ~30MB'a cikti -- tarayicida indirip parse etmek pratik olarak
    calismadi (kullanicidan "hiçbir şey olmuyor" geri bildirimi bu
    yuzdendi). Artik SADECE hafif zaman damgasi listesi donuyor, her
    adimin ucak listesi AYRI/hafif bir istekle (/api/replay_frame)
    cekiliyor -- update_map'in her tick'te /api/flights'i cagirmasiyla
    AYNI desen (canli trafikte zaten calisan bir boyut/hiz).

    ONEMLI (kapsam kasitli sinirli): en fazla REPLAY_MAX_RANGE_HOURS
    saatlik aralik -- bu bir universite staj projesi ozelligi,
    prodüksiyon-olcekli bir replay motoru degil."""
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return {"error": "invalid start/end (ISO8601 bekleniyor)"}
    if end_dt <= start_dt:
        return {"error": "end, start'tan sonra olmali"}
    if (end_dt - start_dt) > timedelta(hours=REPLAY_MAX_RANGE_HOURS):
        end_dt = start_dt + timedelta(hours=REPLAY_MAX_RANGE_HOURS)
    step_sec = max(5, min(step_sec, 300))

    epoch0 = start_dt.astimezone(timezone.utc)
    start_s = epoch0.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_s = end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # SADECE hangi zaman damgalarinda veri VAR oldugunu ogrenmek icin --
    # tek bir field (lat) yeterli, ucaklarin kendisi burada DONMUYOR.
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start_s}, stop: {end_s})
      |> filter(fn: (r) => r["_measurement"] == "flights")
      |> filter(fn: (r) => r["_field"] == "lat")
      |> keep(columns: ["_time"])
    '''
    try:
        tables = _query_api.query_data_frame(flux)
        if isinstance(tables, list):
            tables = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    except Exception as e:
        return {"error": str(e)}

    if tables.empty:
        return {"steps": [], "step_sec": step_sec}

    tables["_time"] = pd.to_datetime(tables["_time"], utc=True)
    buckets = sorted(((tables["_time"] - epoch0).dt.total_seconds() // step_sec)
                      .astype(int).unique().tolist())
    steps = [(epoch0 + timedelta(seconds=b * step_sec)).strftime("%Y-%m-%dT%H:%M:%SZ")
             for b in buckets]
    return {"steps": steps, "step_sec": step_sec}


@lru_cache(maxsize=512)
def _query_replay_frame_cached(start_s: str, end_s: str) -> str:
    """get_replay_frame'in asil sorgusu -- SONUCU cache'ler (JSON STRING
    olarak, liste DEGIL -- cagiran taraf kazayla mutasyona ugratamasin
    diye). ONEMLI (performans, kullanicidan "çok yavaş" geri bildirimi
    uzerine eklendi): gecmis bir zaman penceresindeki InfluxDB verisi
    ARTIK DEGISMEZ (retroaktif yazma yok) -- ayni (start_s, end_s) icin
    tekrar sorgu atmak tamamen gereksiz. advance_replay_tick sona gelince
    BASA SARDIGI icin (bkz. o fonksiyonun yorumu) ayni kareler defalarca
    istenir -- ilk turdan sonraki turlar artik InfluxDB'ye HIC gitmeden,
    ANINDA cache'ten donuyor."""
    # ONEMLI (performans -- kullanici geri bildirimi "1de takılı kalıyor,
    # durdurunca 8e geçiyor", olcumle DOGRULANDI: bu sorgu tek basina ~5sn
    # surerken oynatma tick'i sabit 2sn'de bir -- yani NORMAL 1x oynatmada
    # bile HER ZAMAN ust uste binen istekler oluyordu). Kok neden: bu
    # pencerede (step_sec, genelde 30sn) TUM ucaklarin TUM ham noktalari
    # cekilip pandada groupby(...).last() ile "ucak basina EN SON nokta"
    # cikariliyordu -- binlerce ucagin HER BIRI icin 5-10+ ham nokta
    # InfluxDB'den tarayiciya/pandaya tasiniyor, sadece SONUNCUSU
    # tutuluyordu. Flux'in KENDI last() agregasyonuyla (group+last, alan
    # bazinda) ayni indirgeme InfluxDB TARAFINDA yapiliyor -- agdan/
    # pandadan gecen veri miktari dusuyor, sorgu suresi buna gore kisaliyor.
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start_s}, stop: {end_s})
      |> filter(fn: (r) => r["_measurement"] == "flights")
      |> filter(fn: (r) => r["_field"] == "lat" or r["_field"] == "lon"
                         or r["_field"] == "alt" or r["_field"] == "track"
                         or r["_field"] == "velocity")
      |> group(columns: ["icao24", "_field"])
      |> last()
      |> group()
      |> pivot(rowKey: ["_time", "icao24"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        tables = _query_api.query_data_frame(flux)
        if isinstance(tables, list):
            tables = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    except Exception as e:
        return json.dumps({"error": str(e)})

    if tables.empty or "icao24" not in tables.columns:
        return "[]"

    for col in ["lat", "lon", "alt", "track", "velocity"]:
        if col not in tables.columns:
            tables[col] = None
    tables = tables.dropna(subset=["lat", "lon"])
    if tables.empty:
        return "[]"
    tables["_time"] = pd.to_datetime(tables["_time"], utc=True)
    # Bu dar pencerede (step_sec) ayni ucaktan birden fazla nokta varsa
    # en sonuncusu -- get_replay'in ONCEKI (artik kaldirilan) bucket
    # mantigiyla AYNI netice, sadece tek adim icin.
    latest = tables.sort_values("_time").groupby("icao24", as_index=False).last()
    cols = ["icao24", "lat", "lon", "alt", "track", "velocity"]
    # ONEMLI: df.to_json() -- NaN'i otomatik null'a cevirir (bkz. get_history/
    # get_replay'deki AYNI duzeltme, elle None kontrolu GEREKMEZ).
    return latest[cols].to_json(orient="records")


@app_api.get("/api/replay_frame")
def get_replay_frame(ts: str, step_sec: int = 30):
    """TEK BIR replay karesi -- get_replay()'in dondurdugu "steps"
    listesindeki bir zaman damgasi (ts) icin, [ts, ts+step_sec) penceresindeki
    TUM ucaklarin en son bilinen konumu. render_replay_frame (Dash callback)
    her adim degistiginde bunu cagirir -- update_map'in /api/flights'i her
    tick'te cagirmasiyla AYNI desen (bkz. get_replay docstring'i -- boyut
    sorunu bu ikiye bolmeyle cozuldu). Asil sorgu _query_replay_frame_cached'te
    -- tekrarlanan (orn. dongude basa saran) istekler InfluxDB'ye gitmez."""
    try:
        start_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return {"error": "invalid ts (ISO8601 bekleniyor)"}
    step_sec = max(5, min(step_sec, 300))
    end_dt = start_dt + timedelta(seconds=step_sec)
    start_s = start_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_s = end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.loads(_query_replay_frame_cached(start_s, end_s))


def _reverse_geocode(lat, lon):
    """lat/lon -> kisa yer adi (sehir/kasaba/bolge), OpenStreetMap Nominatim
    (ucretsiz, API key gerektirmiyor -- projede adsbdb.com icin de ayni
    'ucretsiz topluluk servisi' yaklasimi kullanildi). Redis'te KABA bir
    hassasiyetle (2 ondalik, ~1km) 30 gun cache'leniyor -- hem Nominatim'in
    adil kullanim politikasina (agir otomatik sorgu YOK, bu sadece
    kullanici bir ucak SECTIGINDE calisan interaktif bir ozellik) saygi
    icin, hem de ayni havalimani/bolgeye yakin cok sayida ucusun AYNI
    cache kaydini paylasabilmesi icin. Basarisiz olursa (ag hatasi, zaman
    asimi, sonuc yok) None doner -- cagiran taraf koordinati fallback
    olarak gosterir, hata FIRLATMAZ (bu bir "olsa iyi olur" zenginlestirme,
    segment listesinin CALISMASI buna bagli DEGIL)."""
    r = redis.Redis(connection_pool=_rpool)
    cache_key = f"iha:geocode:{round(lat, 2)}:{round(lon, 2)}"
    cached = r.get(cache_key)
    if cached is not None:
        return cached or None
    name = None
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
            headers={"User-Agent": "iha-anomali-dashboard/1.0 (universite staj projesi)"},
            timeout=4,
        )
        if resp.status_code == 200:
            addr = resp.json().get("address", {})
            name = (addr.get("city") or addr.get("town") or addr.get("village")
                    or addr.get("county") or addr.get("state"))
    except Exception:
        pass
    r.set(cache_key, name or "", ex=GEOCODE_CACHE_TTL)
    return name


@app_api.get("/api/flight_segments/{icao24}")
def get_flight_segments(icao24: str):
    """Bucket'in tuttugu TUM gecmisi (7 gune kadar) ayri UCUSLARA boler --
    sol paneldeki 'onceki ucuslar' listesi icin. AYNI gap-tabanli heuristik
    (bkz. FLIGHT_GAP_THRESHOLD_MIN, update_flight_path'teki 'son ucus'
    mantigiyla TUTARLI olmasi icin) -- ardisik iki nokta arasinda
    esikten BUYUK bir bosluk, "onceki ucus bitti, yenisi basladi" sayilir.
    Her segmentin baslangic/bitis noktasi icin (mumkunse) bir yer adi da
    donuyor (bkz. _reverse_geocode) -- bulunamazsa frontend koordinati
    fallback olarak gosterebilsin diye start_lat/lon, end_lat/lon HER
    ZAMAN dahil."""
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -168h)
      |> filter(fn: (r) => r["_measurement"] == "flights")
      |> filter(fn: (r) => r["icao24"] == "{icao24}")
      |> filter(fn: (r) => r["_field"] == "lat" or r["_field"] == "lon")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        tables = _query_api.query_data_frame(flux)
        if isinstance(tables, list):
            tables = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if tables.empty:
            return []
        tables["_time"] = pd.to_datetime(tables["_time"], utc=True)
        tables = tables.sort_values("_time").reset_index(drop=True)

        gap = pd.Timedelta(minutes=FLIGHT_GAP_THRESHOLD_MIN)
        break_idx = tables.index[tables["_time"].diff() > gap].tolist()
        starts = [0] + break_idx
        ends = break_idx + [len(tables)]

        segments = []
        for s, e in zip(starts, ends):
            if e - s < 3:
                continue  # tek/iki noktalik "segment" muhtemelen gurultu, gercek ucus degil
            start_row, end_row = tables.iloc[s], tables.iloc[e - 1]
            segments.append({
                "start": start_row["_time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "end": end_row["_time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "duration_min": round((end_row["_time"] - start_row["_time"]).total_seconds() / 60, 1),
                "points": int(e - s),
                "start_lat": float(start_row["lat"]), "start_lon": float(start_row["lon"]),
                "end_lat": float(end_row["lat"]), "end_lon": float(end_row["lon"]),
            })
        segments.sort(key=lambda s: s["start"], reverse=True)  # en yeni once

        # ONEMLI: geocoding DIS bir servise gidiyor -- kotu durumda (hepsi
        # cache-miss) yavaslamayi sinirlamak icin sadece EN YENI N segmenti
        # zenginlestiriyoruz (GEOCODE_MAX_LOOKUPS_PER_REQUEST / 2, cunku
        # her segment 2 lookup -- baslangic+bitis). Geri kalanlar icin
        # frontend start_lat/lon'dan koordinat gosterir.
        #
        # ONEMLI (kullanici geri bildirimi -- "geçmiş uçuşlar çok geç
        # çalışıyor"): bu 16'ya kadar lookup ESKIDEN SIRAYLA (tek tek)
        # yapiliyordu -- her biri Nominatim'e giden GERCEK bir HTTP istegi,
        # 4sn timeout'lu. Sogutulmus (hic cache'lenmemis) bir ucak icin
        # bu 16 x ~0.3-4sn = TOPLAMDA onlarca saniye SIRAYLA bekleme
        # demekti. Lookup'lar birbirinden BAGIMSIZ (farkli lat/lon) --
        # ThreadPoolExecutor ile PARALEL calistiriliyor, toplam sure artik
        # ~16 istegin TOPLAMI degil, EN YAVAS TEKIL istegin suresi kadar.
        to_geocode = segments[:GEOCODE_MAX_LOOKUPS_PER_REQUEST // 2]
        rest = segments[GEOCODE_MAX_LOOKUPS_PER_REQUEST // 2:]
        if to_geocode:
            with ThreadPoolExecutor(max_workers=GEOCODE_MAX_LOOKUPS_PER_REQUEST) as pool:
                futures = {}
                for seg in to_geocode:
                    futures[pool.submit(_reverse_geocode, seg["start_lat"], seg["start_lon"])] = (seg, "start_place")
                    futures[pool.submit(_reverse_geocode, seg["end_lat"], seg["end_lon"])] = (seg, "end_place")
                for fut, (seg, key) in futures.items():
                    seg[key] = fut.result()
        for seg in rest:
            seg["start_place"] = None
            seg["end_place"] = None

        return segments
    except Exception as e:
        return {"error": str(e)}


@app_api.get("/api/traffic_stats")
def traffic_stats(hours: int = 24):
    hours = min(hours, 24 * 7)
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{hours}h)
      |> filter(fn: (r) => r["_measurement"] == "flights" and r["_field"] == "alt")
      |> aggregateWindow(every: 1h, fn: count, createEmpty: false)
      |> group(columns: ["_time"])
      |> count(column: "icao24")
    '''
    # basitlestirilmis alternatif: saat basina benzersiz icao24 sayisi
    flux_simple = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{hours}h)
      |> filter(fn: (r) => r["_measurement"] == "flights" and r["_field"] == "alt")
      |> group(columns: ["icao24"])
      |> aggregateWindow(every: 1h, fn: count, createEmpty: false)
    '''
    try:
        tables = _query_api.query_data_frame(flux_simple)
        if isinstance(tables, list):
            tables = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if tables.empty:
            return []
        grouped = tables.groupby("_time")["icao24"].nunique().reset_index()
        grouped.columns = ["time", "unique_aircraft"]
        # Ayni dtype-bagimsiz ISO string donusumu (bkz. get_history() notu)
        grouped["time"] = pd.to_datetime(grouped["time"], utc=True) \
                             .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return json.loads(grouped.to_json(orient="records"))
    except Exception as e:
        return {"error": str(e)}
