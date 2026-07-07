"""
app.py
Canli harita (Redis) + model alarmlari + secili ucagin gecmis grafigi
(InfluxDB, 7 gune kadar) tek Dash uygulamasinda. FastAPI arka planda
thread olarak calisir, Dash ondan besleniyor.

ONEMLI: Bunu calistirmadan once dashboard_consumer.py'nin AYRI bir
terminalde calisiyor olmasi lazim, yoksa Redis/InfluxDB'de veri olmaz.
Alert paneli, model ekibi "adsb.alerts" topic'ine yazmaya baslayana
kadar bos gorunur -- bu normaldir, kod degisikligi gerekmeyecek.

Kullanim:
    python app.py
Sonra tarayicida: http://localhost:8050
"""
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import redis
import requests
import uvicorn
import dash
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign
from dash import Dash, dcc, html, Output, Input, State
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient

def _resolve_tz(offset_str):
    """Ayarlardan gelen UTC ofsetini (orn. '3', '-5') sabit-ofsetli bir
    timezone nesnesine cevirir. Gecersiz/bos deger gelirse (ilk yukleme,
    hata vb.) guvenli varsayilana (Turkiye, UTC+3) duser."""
    try:
        return timezone(timedelta(hours=int(offset_str)))
    except Exception:
        return timezone(timedelta(hours=3))


# ADS-B/Mode-S emitter kategori kodlari (standart, ilk harf sinif, rakam alt tip)
# Dil destegi icin TR/EN ayri sozlukler -- CATEGORY_LABELS[lang][kod] seklinde kullanilir.
CATEGORY_LABELS = {
    "tr": {
        "A0": "Bilinmiyor", "A1": "Hafif uçak", "A2": "Küçük uçak",
        "A3": "Büyük uçak", "A4": "Büyük uçak (yüksek vorteks)", "A5": "Ağır uçak",
        "A6": "Yüksek performans", "A7": "Helikopter",
        "B0": "Bilinmiyor", "B1": "Planör", "B2": "Balon/Zeplin",
        "B3": "Paraşütçü", "B4": "Ultralight/Yamaç paraşütü",
        "B6": "İHA/Drone", "B7": "Uzay aracı",
        "C0": "Bilinmiyor", "C1": "Yer taşıtı (acil)", "C2": "Yer taşıtı (servis)",
        "C3": "Sabit engel", "C4": "Engel kümesi", "C5": "Hat engeli",
    },
    "en": {
        "A0": "Unknown", "A1": "Light aircraft", "A2": "Small aircraft",
        "A3": "Large aircraft", "A4": "Large aircraft (high vortex)", "A5": "Heavy aircraft",
        "A6": "High performance", "A7": "Helicopter",
        "B0": "Unknown", "B1": "Glider", "B2": "Balloon/Airship",
        "B3": "Parachutist", "B4": "Ultralight/Paraglider",
        "B6": "UAV/Drone", "B7": "Spacecraft",
        "C0": "Unknown", "C1": "Ground vehicle (emergency)", "C2": "Ground vehicle (service)",
        "C3": "Fixed obstacle", "C4": "Cluster obstacle", "C5": "Line obstacle",
    },
}

# ADS-B acil durum kodlari (Mode S emergency/priority status)
EMERGENCY_LABELS = {
    "tr": {
        "none": None,  # normal durum, gosterme
        "general": "GENEL ACİL DURUM",
        "lifeguard": "SAĞLIK ACİL DURUMU",
        "minfuel": "YAKIT KRİTİK",
        "nordo": "RADYO ARIZASI",
        "unlawful": "KAÇIRMA (HİJACK)",
        "downed": "DÜŞTÜ/İNİŞ ZORUNLU",
        "reserved": "REZERVE KOD",
    },
    "en": {
        "none": None,
        "general": "GENERAL EMERGENCY",
        "lifeguard": "MEDICAL EMERGENCY",
        "minfuel": "FUEL CRITICAL",
        "nordo": "RADIO FAILURE",
        "unlawful": "HIJACKING",
        "downed": "DOWNED / FORCED LANDING",
        "reserved": "RESERVED CODE",
    },
}

DEFAULT_LANGUAGE = "tr"

# Arayuzdeki tum sabit/dinamik metinler -- TEXTS[lang]["anahtar"] seklinde
# kullanilir. Yeni bir dil eklemek icin buraya ucuncu bir blok (orn. "de")
# eklemek yeterli, kodun geri kalani degismeden calisir.
TEXTS = {
    "tr": {
        "settings_title": "Ayarlar",
        "trace_hours_label": "Rota izi (saat)",
        "timezone_label": "Saat dilimi (UTC farkı)",
        "language_label": "Dil",
        "aircraft_info_title": "Uçak Bilgisi",
        "history_panel_title": "Geçmiş (son 24 saat)",
        "click_aircraft": "Haritada bir uçağa tıklayın.",
        "no_signal": "{icao} şu anda sinyal göndermiyor (kapsama alanından çıkmış olabilir).",
        "no_callsign": "Çağrı kodu yok, rota sorgulanamıyor.",
        "route_not_found": "'{callsign}' için rota bilgisi bulunamadı (adsbdb.com veritabanında yok).",
        "aircraft_info_not_found": "Uçak tipi/tescil bilgisi bulunamadı.",
        "no_details": "Detay yok",
        "registration": "Tescil",
        "field_icao": "ICAO24",
        "field_callsign": "Çağrı Kodu",
        "field_lat": "Enlem",
        "field_lon": "Boylam",
        "field_alt": "İrtifa",
        "field_speed": "Hız",
        "field_track": "Yön",
        "field_vspeed": "Dikey Hız",
        "field_category": "Kategori",
        "field_squawk": "Squawk",
        "field_last_update": "Son Güncelleme",
        "emergency_squawk": "ACİL DURUM SQUAWK: {squawk}",
        "status_bar": "{ts} | {n} aktif uçuş | {a} alarm",
        "history_alt_label": "İrtifa (m)",
        "history_speed_label": "Hız (m/s)",
        "no_data": "Veri bulunamadı",
        "tooltip_alt": "İrtifa",
        "tooltip_speed": "Hız",
        "tooltip_track": "Yön",
        "tooltip_vspeed": "Dikey",
        "tooltip_signal_age": "Sinyal yaşı",
        "tz_default_suffix": " (Türkiye)",
        "filter_civil_label": "Sivil",
        "filter_military_label": "Askeri",
        "field_military": "Askeri mi",
        "military_yes": "Evet",
        "military_no": "Hayır",
        "tooltip_military_tag": "Askeri",
        "map_style_label": "Harita Türü",
        "map_style_street": "Sokak",
        "map_style_satellite": "Uydu",
        "route_uncertain": "⚠ Rota güncel olmayabilir (uçak farklı bir yöne gidiyor)",
    },
    "en": {
        "settings_title": "Settings",
        "trace_hours_label": "Track trail (hours)",
        "timezone_label": "Time zone (UTC offset)",
        "language_label": "Language",
        "aircraft_info_title": "Aircraft Info",
        "history_panel_title": "History (last 24h)",
        "click_aircraft": "Click an aircraft on the map.",
        "no_signal": "{icao} is not currently transmitting (may have left coverage area).",
        "no_callsign": "No callsign, route lookup unavailable.",
        "route_not_found": "No route found for '{callsign}' (not in the adsbdb.com database).",
        "aircraft_info_not_found": "No aircraft type/registration info found.",
        "no_details": "No details",
        "registration": "Registration",
        "field_icao": "ICAO24",
        "field_callsign": "Callsign",
        "field_lat": "Latitude",
        "field_lon": "Longitude",
        "field_alt": "Altitude",
        "field_speed": "Speed",
        "field_track": "Heading",
        "field_vspeed": "Vertical Speed",
        "field_category": "Category",
        "field_squawk": "Squawk",
        "field_last_update": "Last Update",
        "emergency_squawk": "EMERGENCY SQUAWK: {squawk}",
        "status_bar": "{ts} | {n} active flights | {a} alerts",
        "history_alt_label": "Altitude (m)",
        "history_speed_label": "Speed (m/s)",
        "no_data": "No data found",
        "tooltip_alt": "Alt",
        "tooltip_speed": "Speed",
        "tooltip_track": "Heading",
        "tooltip_vspeed": "V/S",
        "tooltip_signal_age": "Signal age",
        "tz_default_suffix": " (Turkey)",
        "filter_civil_label": "Civilian",
        "filter_military_label": "Military",
        "field_military": "Military",
        "military_yes": "Yes",
        "military_no": "No",
        "tooltip_military_tag": "Military",
        "map_style_label": "Map Style",
        "map_style_street": "Street",
        "map_style_satellite": "Satellite",
        "route_uncertain": "⚠ Route may be outdated (aircraft heading a different way)",
    },
}

# Harita katmani secenekleri -- ayarlardan degistirilebilir. "street"
# varsayilan (mevcut OpenStreetMap katmani, davranis degismiyor).
#
# GECMIS: Once Esri World Imagery denendi (server.arcgisonline.com) --
# kullanicinin agindan gri ekran cikti, AYNI sorun adsb.lol'un KENDI Esri
# katmaninda da gorulduyu icin bu bizim kodumuzdaki bir hata degildi
# (URL, leaflet-providers'daki kanonik Esri adresiyle birebir ayniydi).
# Google Satellite'e gecildi ama O DA gri cikti -- bu sefer GERCEK bir kod
# hatasiydi: iki katman arasinda "subdomains" degerini de (street icin
# a/b/c, google icin mt0-mt3) dinamik degistirmeye calisiyorduk, ama
# react-leaflet/dash-leaflet TileLayer'da sadece "url" prop'u calisma
# zamaninda guvenilir sekilde uygulaniyor (Leaflet'in setUrl() metoduyla);
# "subdomains" ise SADECE ILK YUKLEMEDE okunuyor, sonradan degisse de
# Leaflet tarafinda yeniden uygulanmiyor. Sonuc: "Uydu"ya gecilince URL
# degisiyordu ama subdomain hala ilk yuklemedeki 'a/b/c' kaliyordu --
# "a.google.com/vt/..." gibi GECERSIZ adreslere istek atiliyordu (Google'in
# gercek subdomain'leri mt0-mt3), hepsi basarisiz oluyordu -> gri ekran.
#
# COZUM: "{s}" sablonunu tamamen kaldirip SABIT tek bir subdomain
# kullanmaya gecildi -- boylece dinamik "subdomains" prop'una hic ihtiyac
# kalmiyor, bu hata sinifi kokten ortadan kalkiyor (paralel-istek
# optimizasyonu kaybediliyor ama bizim trafik hacmimizde onemsiz).
TILE_LAYERS = {
    "street": {
        "url": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap",
    },
    "satellite": {
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attribution": "Map data © Google",
    },
}
DEFAULT_MAP_STYLE = "street"

# ONEMLI: REDIS_HOST/INFLUX_HOST/INFLUX_TOKEN artik ortam degiskeniyle
# ayarlanabilir -- Windows'ta native calisirken (setup_local_windows.py)
# varsayilanlar (localhost + dosyadan token) hala gecerli, Docker'da
# docker-compose.yml servis adlarini (redis, influxdb) ve sabit token'i
# (DOCKER_INFLUXDB_INIT_ADMIN_TOKEN) enjekte eder.
TOKEN_FILE = Path("influx_token.txt")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
INFLUX_HOST = os.environ.get("INFLUX_HOST", "http://localhost:8086")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "iha-org")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "adsb-history")

INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN")
if not INFLUX_TOKEN:
    if not TOKEN_FILE.exists():
        raise SystemExit("influx_token.txt bulunamadi ve INFLUX_TOKEN ortam degiskeni yok. "
                          "Once setup_local_windows.py calistir (native) ya da INFLUX_TOKEN set et (docker).")
    INFLUX_TOKEN = TOKEN_FILE.read_text().strip()

# ------------------------------------------------------------------ FastAPI --

app_api = FastAPI(title="ADS-B Local API")
app_api.add_middleware(CORSMiddleware, allow_origins=["*"],
                        allow_methods=["*"], allow_headers=["*"])

_rpool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0,
                               decode_responses=True, protocol=2)
_influx = InfluxDBClient(url=INFLUX_HOST, token=INFLUX_TOKEN, org=INFLUX_ORG)
_query_api = _influx.query_api()


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


@app_api.get("/api/history/{icao24}")
def get_history(icao24: str, hours: int = 24):
    hours = min(hours, 24 * 7)  # bucket zaten 7 gunden fazlasini tutmuyor
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{hours}h)
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
            return []
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
        return json.loads(tables[["_time", "lat", "lon", "alt", "velocity"]]
                          .to_json(orient="records"))
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


def _run_api():
    uvicorn.run(app_api, host="0.0.0.0", port=8000, log_level="warning")


threading.Thread(target=_run_api, daemon=True).start()
time.sleep(2)
print("FastAPI hazir (port 8000)")

# --------------------------------------------------------------------- Dash --

app_dash = Dash(__name__, title="ADS-B Local Dashboard")

# --------------------------------------------------------------------------
# UCAK KATMANI RENDER MANTIGI -- artik CLIENT-SIDE (JavaScript)
#
# ONEMLI MIMARI DEGISIKLIK: eskiden her ucak icin Python'da bir dl.DivMarker
# + ic ice Div'li dl.Tooltip nesnesi INSA EDILIYOR, tumu JSON'a cevrilip
# tarayiciya gonderiliyordu. "Dunya" modunda (5.000-11.000+ ucak) bu, hem
# Python tarafinda hem tarayicinin React/Leaflet reconciliation'inda
# donmaya yol aciyordu (bkz. proje sohbet gecmisi) -- CPU hizindan bagimsiz,
# mimari bir sinir (tek JS thread'inde on binlerce DOM elemani senkron
# insa etmek). Flightradar24 ve adsb.lol'un kendi haritalari da AYNI
# nedenle boyle calismiyor.
#
# COZUM: dl.GeoJSON(cluster=True) -- veri hafif GeoJSON olarak gidiyor
# (Python'da ic ice component AGACI yok, duz sozluk listesi), gercek
# marker/tooltip olusturma islemi supercluster (hizli JS kutuphanesi) +
# asagidaki iki KUCUK JS fonksiyonuyla TARAYICIDA yapiliyor. Uzaktan
# bakildiginda (dunya zoom'u) binlerce nokta birkaç yuz "cluster balonu"na
# indirgeniyor -- sadece o an ekranda gorunecek kadar gercek marker olusuyor.
#
# ONEMLI: bu iki isim (_POINT_TO_LAYER_JS / _ON_EACH_FEATURE_JS) asagidaki
# app_dash.layout icinde KULLANILIYOR -- Python modul seviyesinde yukaridan
# asagiya calistigi icin layout'tan ONCE tanimlanmis olmalari sart.

# Ucak SVG ikonunu (yon rotasyonlu) client-side olusturan fonksiyon.
# Eski Python-tarafi _airplane_icon() ile BIREBIR AYNI SVG/CSS -- sadece
# JS'e tasindi, gorsel sonuc degismiyor. Cluster'lar (feature.properties
# icao24 tasimayan sentetik gruplar) buraya hic ugramiyor, supercluster
# onlari kendi varsayilan balon ikonuyla ayrica render ediyor.
_POINT_TO_LAYER_JS = assign("""
function(feature, latlng, context){
    const p = feature.properties;
    const heading = p.track || 0;
    const color = p.color || '#00b4d8';
    const opacity = (p.opacity === undefined || p.opacity === null) ? 1 : p.opacity;
    const html = '<div style="transform: rotate(' + heading + 'deg); ' +
        'transform-origin: center; width: 22px; height: 22px; opacity: ' + opacity + ';">' +
        '<svg width="22" height="22" viewBox="0 0 24 24">' +
        '<path d="M12 1 L15 13 L23 18 L15 16 L15 20.5 L18.5 22.5 L12 21 ' +
        'L5.5 22.5 L9 20.5 L9 16 L1 18 L9 13 Z" fill="' + color + '" ' +
        'stroke="#07070e" stroke-width="0.5"/></svg></div>';
    const icon = L.divIcon({html: html, className: '', iconSize: [22, 22], iconAnchor: [11, 11]});
    return L.marker(latlng, {icon: icon});
}
""")

# Tooltip'i client-side baglayan fonksiyon -- Python'daki eski ic ice
# Div grid'iyle AYNI icerik, duz HTML string olarak. Tum metinler
# (etiketler, formatlanmis degerler) update_map icinde Python'da onceden
# hazirlanip feature.properties'e konuyor -- JS sadece bunlari yerlestiriyor,
# ceviri/formatlama mantigi burada TEKRARLANMIYOR.
_ON_EACH_FEATURE_JS = assign("""
function(feature, layer, context){
    const p = feature.properties;
    if(!p.icao24){ return; }  // cluster balonu -- ucak degil, tooltip yok
    let signalRow = '';
    if(p.signal_age_text){
        signalRow = '<div style="grid-column: 1 / -1;"><span style="color:#666">' +
            p.lbl_signal_age + ' </span><span style="color:#f7b731">' + p.signal_age_text + '</span></div>';
    }
    const html = '<div style="min-width:150px">' +
        '<div style="font-size:14px; font-weight:700; color:' + p.color +
        '; margin-bottom:1px;">' + p.callsign + '</div>' +
        '<div style="font-size:10px; color:#888; margin-bottom:6px;">' + p.subtitle + '</div>' +
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:3px 12px; font-size:11px;">' +
        '<div><span style="color:#666">' + p.lbl_alt + ' </span><span>' + p.alt_text + '</span></div>' +
        '<div><span style="color:#666">' + p.lbl_speed + ' </span><span>' + p.speed_text + '</span></div>' +
        '<div><span style="color:#666">' + p.lbl_track + ' </span><span>' + p.track_text + '</span></div>' +
        '<div><span style="color:#666">' + p.lbl_vspeed + ' </span><span>' + p.vspeed_text + '</span></div>' +
        signalRow +
        '</div></div>';
    layer.bindTooltip(html, {direction: 'top', offset: [0, -14]});
}
""")

# Tarayicinin varsayilan <body> kenar bosluguyla (genelde 8px) koyu tema
# etrafinda beyaz cerceve olusuyordu -- Dash'in index sablonunu gecersiz
# kilip body/html marjinini sifirliyoruz. overflow:hidden ile de sayfa
# hicbir zaman kaydirilamiyor -- tam ekran uygulama.
app_dash.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #07070e;
            }
            /* Leaflet'in varsayilan tooltip'i beyaz kutu/siyah yazi --
               koyu temaya uydurmak icin gecersiz kiliyoruz. */
            .leaflet-tooltip {
                background-color: #161625 !important;
                border: 1px solid #2a2a4a !important;
                color: #c8d0e0 !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
                padding: 8px 10px !important;
            }
            .leaflet-tooltip-top:before   { border-top-color: #2a2a4a !important; }
            .leaflet-tooltip-bottom:before{ border-bottom-color: #2a2a4a !important; }
            .leaflet-tooltip-left:before  { border-left-color: #2a2a4a !important; }
            .leaflet-tooltip-right:before { border-right-color: #2a2a4a !important; }

            /* Tarayicinin kendi (native) sayi giris +/- oklarini gizliyoruz --
               bunlarin tiklaninca "siyahta takili kalma" sorunu vardi, artik
               yerlerine kendi html.Button'larimizi (trace-hours-minus/plus)
               kullaniyoruz, native oklara gerek yok. */
            input[type=number]::-webkit-inner-spin-button,
            input[type=number]::-webkit-outer-spin-button {
                -webkit-appearance: none !important;
                appearance: none !important;
                margin: 0 !important;
            }
            input[type=number] {
                -moz-appearance: textfield !important;
                appearance: textfield !important;
            }

            /* Saat dilimi secimi: dcc.Dropdown eski react-select tabanli,
               varsayilan beyaz/acik tema kullaniyor. Kapali kutu, acik
               menu ve secenek satirlarinin HEPSINI ayri ayri koyu temaya
               ceviriyoruz -- daha once sadece disaridaki kutu (.Select-control)
               denenmisti, acilan menu (.Select-menu-outer / secenekler)
               beyaz kalmisti, bu yuzden burada eksiksiz kapsiyoruz. */
            .dark-dropdown .Select-control,
            .dark-dropdown.is-open .Select-control,
            .dark-dropdown.is-focused .Select-control,
            .dark-dropdown.is-focused:not(.is-open) .Select-control {
                background-color: #161625 !important;
                border: 1px solid #2a2a4a !important;
                border-radius: 6px !important;
                color: #c8d0e0 !important;
                box-shadow: none !important;
            }
            .dark-dropdown .Select-value-label,
            .dark-dropdown .Select-placeholder,
            .dark-dropdown .Select-input > input {
                color: #c8d0e0 !important;
            }
            .dark-dropdown .Select-arrow {
                border-color: #c8d0e0 transparent transparent !important;
            }
            .dark-dropdown .Select-menu-outer {
                background-color: #161625 !important;
                border: 1px solid #2a2a4a !important;
                border-radius: 6px !important;
                z-index: 1500 !important;
                box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
            }
            .dark-dropdown .Select-menu {
                background-color: #161625 !important;
            }
            .dark-dropdown .Select-option {
                background-color: #161625 !important;
                color: #c8d0e0 !important;
            }
            .dark-dropdown .Select-option.is-focused {
                background-color: #22224a !important;
                color: #ffffff !important;
            }
            .dark-dropdown .Select-option.is-selected {
                background-color: #00b4d8 !important;
                color: #07070e !important;
            }
            /* Dash Dropdown buyuk listelerde react-virtualized-select
               kullanabiliyor, o zaman secenekler yukaridaki .Select-option
               yerine bu siniflarla geliyor -- ikisini de kapsiyoruz. */
            .dark-dropdown .VirtualizedSelectOption {
                background-color: #161625 !important;
                color: #c8d0e0 !important;
            }
            .dark-dropdown .VirtualizedSelectFocusedOption {
                background-color: #22224a !important;
                color: #ffffff !important;
            }
            .dark-dropdown .VirtualizedSelectSelectedOption {
                background-color: #00b4d8 !important;
                color: #07070e !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Panel stil sabitleri -- hem layout'taki baslangic (gizli) hali hem de
# toggle_panels callback'i ayni degerleri kullaniyor.
LEFT_PANEL_BASE = {
    "position": "absolute", "top": 0, "left": 0, "bottom": 0,
    "width": "320px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "boxShadow": "4px 0 24px rgba(0,0,0,0.6)",
    "transition": "transform 0.3s ease",
    "zIndex": 800, "padding": "18px", "overflowY": "auto",
}
HISTORY_PANEL_BASE = {
    "position": "absolute", "bottom": 0, "right": 0,
    "width": "460px", "height": "260px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "boxShadow": "-2px -2px 24px rgba(0,0,0,0.6)",
    "borderTopLeftRadius": "10px",
    "transition": "transform 0.3s ease",
    "zIndex": 800, "padding": "12px",
}
SETTINGS_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "12px",
    "width": "230px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #2a2a4a",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}

DEFAULT_TRACE_HOURS = 6
DEFAULT_TIMEZONE = 3  # UTC+3, Turkiye -- dropdown "value" olarak int kullanilir

STEPPER_BTN_STYLE = {
    "width": "28px", "height": "28px", "borderRadius": "5px",
    "border": "1px solid #2a2a4a", "backgroundColor": "#161625",
    "color": "#c8d0e0", "fontSize": "16px", "cursor": "pointer",
    "display": "flex", "alignItems": "center", "justifyContent": "center",
    "padding": 0, "lineHeight": "1", "flexShrink": 0,
}

# Dil secim butonlari (TR/EN) -- iki durumlu (aktif/pasif) stil, hangisinin
# secili oldugu update_language_buttons callback'inde belirleniyor.
LANG_BTN_BASE_STYLE = {
    "flex": "1", "padding": "6px 0", "borderRadius": "5px",
    "border": "1px solid #2a2a4a", "fontSize": "12px", "fontWeight": "600",
    "cursor": "pointer", "letterSpacing": "0.5px",
}
LANG_BTN_ACTIVE_STYLE = {**LANG_BTN_BASE_STYLE,
    "backgroundColor": "#00b4d8", "color": "#07070e", "border": "1px solid #00b4d8"}
LANG_BTN_INACTIVE_STYLE = {**LANG_BTN_BASE_STYLE,
    "backgroundColor": "#161625", "color": "#888"}

# Askeri ucaklari haritada ayirt etmek icin ayri bir renk -- alarm kirmizisi
# (#e63946) ve varsayilan sivil rengiyle (#00b4d8) karismasin diye hakiki/
# zeytin yesili secildi. Oncelik sirasi: alarm > askeri > sivil (bkz.
# update_map, bir ucak hem alarmli hem askeri olabilir, alarm once gelir).
DEFAULT_AIRCRAFT_COLOR = "#00b4d8"
MILITARY_COLOR = "#8a9a5b"
ALERT_COLOR = "#e63946"

# Sol-ust askeri/sivil filtre butonlari -- haritanin kendi zoom (+/-)
# kontrolu de sol-ustte oldugu icin (Leaflet varsayilani, ~10px kenar
# bosluklu, iki dugme ~52px yukseklik), bu butonlar bilerek onun ALTINA
# (top: 72px) yerlestiriliyor, ustune degil -- cakismalari onlemek icin.
FILTER_BTN_BASE_STYLE = {
    "width": "92px", "padding": "6px 8px", "borderRadius": "6px",
    "fontSize": "11px", "fontWeight": "600", "cursor": "pointer",
    "textAlign": "center", "letterSpacing": "0.3px",
}
FILTER_BTN_CIVIL_ACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": DEFAULT_AIRCRAFT_COLOR, "color": "#07070e",
    "border": f"1px solid {DEFAULT_AIRCRAFT_COLOR}"}
FILTER_BTN_MILITARY_ACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": MILITARY_COLOR, "color": "#07070e",
    "border": f"1px solid {MILITARY_COLOR}"}
FILTER_BTN_INACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": "#161625", "color": "#888", "border": "1px solid #2a2a4a"}

app_dash.layout = html.Div(id="app-root", style={
    "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
    "overflow": "hidden", "backgroundColor": "#07070e",
    "fontFamily": "sans-serif", "color": "#c8d0e0",
}, children=[

    dcc.Interval(id="tick", interval=15000, n_intervals=0),
    dcc.Store(id="aircraft-select", data=None),  # secili ucak (gorunmez state)

    # ------------------------------------------------ Tam ekran harita --
    dl.Map(
        id="map",
        center=[39.0, 35.0], zoom=5,
        style={"width": "100%", "height": "100%"},
        children=[
            dl.TileLayer(
                id="base-tile-layer",
                url=TILE_LAYERS[DEFAULT_MAP_STYLE]["url"],
                attribution=TILE_LAYERS[DEFAULT_MAP_STYLE]["attribution"],
            ),
            dl.LayerGroup(id="flight-path-layer"),
            # Secili ucagin kalkis-varis havalimanlari arasindaki referans
            # cizgisi -- flight-path-layer'dan (GERCEK izlenen yol) ayri,
            # cunku bu "kus ucusu" bir referans, gercek rota degil.
            dl.LayerGroup(id="route-line-layer"),
            # ONEMLI: eskiden dl.LayerGroup + N adet dl.DivMarker cocugu idi.
            # Simdi TEK bir dl.GeoJSON, cluster=True ile -- render supercluster
            # (JS) + _POINT_TO_LAYER_JS/_ON_EACH_FEATURE_JS tarafindan
            # CLIENT-SIDE yapiliyor (bkz. yukaridaki yorum blogu). update_map
            # callback'i artik Python component agaci degil, duz GeoJSON
            # sozlugu donduruyor (Output("aircraft-geojson", "data")).
            dl.GeoJSON(
                id="aircraft-geojson",
                data=dlx.dicts_to_geojson([]),
                cluster=True,
                superClusterOptions={"radius": 80, "maxZoom": 14},
                zoomToBoundsOnClick=True,
                pointToLayer=_POINT_TO_LAYER_JS,
                onEachFeature=_ON_EACH_FEATURE_JS,
            ),
        ],
    ),

    # ------------------------------- Askeri/Sivil filtre butonlari (overlay) --
    # Leaflet'in kendi zoom (+/-) kontrolu sol-ustte durur (~10px kenar
    # boslugu, ~26-30px genislik, iki dugme ~52px yukseklik). Bu butonlari
    # ONUN SAGINA, ayni ust hizaya (top: 12px) koyuyoruz -- zoom kontrolunun
    # genisligi kadar (~46px) icerden baslatarak cakismalari onluyoruz.
    html.Div(id="type-filter-controls", style={
        "position": "absolute", "top": "12px", "left": "48px",
        "display": "flex", "flexDirection": "column", "gap": "6px",
        "zIndex": 700,
    }, children=[
        html.Button("Sivil", id="filter-civil-btn", n_clicks=0,
                   style=FILTER_BTN_CIVIL_ACTIVE_STYLE),
        html.Button("Askeri", id="filter-military-btn", n_clicks=0,
                   style=FILTER_BTN_MILITARY_ACTIVE_STYLE),
    ]),
    dcc.Store(id="show-civil", data=True),
    dcc.Store(id="show-military", data=True),

    # ------------------------------------------- Durum cubugu (overlay) --
    html.Div(id="status", style={
        "position": "absolute", "top": "12px", "left": "50%",
        "transform": "translateX(-50%)",
        "backgroundColor": "rgba(15,15,25,0.85)",
        "padding": "6px 16px", "borderRadius": "20px",
        "fontSize": "13px", "color": "#c8d0e0", "zIndex": 500,
        "pointerEvents": "none",  # altindaki haritayi engellemesin
    }),

    # ------------------------------------------- Ayarlar butonu (overlay) --
    html.Button("⚙", id="settings-btn", n_clicks=0, style={
        "position": "absolute", "top": "12px", "right": "12px",
        "width": "40px", "height": "40px", "borderRadius": "50%",
        "backgroundColor": "#000000", "border": "1px solid #2a2a4a",
        "color": "#c8d0e0", "fontSize": "18px", "cursor": "pointer", "zIndex": 900,
    }),

    # ------------------------------------------- Ayarlar paneli (overlay) --
    html.Div(id="settings-panel", style={**SETTINGS_PANEL_BASE, "display": "none"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "12px"}, children=[
            html.H4("Ayarlar", id="settings-title",
                    style={"margin": 0, "fontSize": "14px", "color": "#00b4d8"}),
            html.Button("×", id="close-settings-btn", n_clicks=0, style={
                "background": "none", "border": "none", "color": "#888",
                "fontSize": "20px", "cursor": "pointer", "lineHeight": "1",
                "padding": "0 4px",
            }),
        ]),
        html.Div(style={"marginBottom": "14px"}, children=[
            html.Label("Rota izi (saat)", id="trace-hours-label",
                       style={"fontSize": "12px", "color": "#888",
                              "display": "block", "marginBottom": "5px"}),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"},
                    children=[
                html.Button("−", id="trace-hours-minus", n_clicks=0, style=STEPPER_BTN_STYLE),
                # ONEMLI: type="text" (type="number" DEGIL) -- number input'un
                # tarayici-native +/- spinner oklari CSS ile guvenilir sekilde
                # gizlenemiyordu (bazi tarayicilarda hala gorunuyor, kendi
                # butonlarimizla yan yana "2 cift +/-" gibi gorunuyordu, ustelik
                # native olan daha yavas tepki veriyordu). type="text" +
                # inputMode="numeric" ile native spinner hic olusmuyor,
                # tek kontrol kendi -/+ butonlarimiz oluyor. Deger dogrulama
                # (sayi mi, 1-24 araliginda mi) Python tarafinda yapiliyor.
                dcc.Input(id="trace-hours-input", type="text", inputMode="numeric",
                         value=str(DEFAULT_TRACE_HOURS), style={
                    "flex": "1", "textAlign": "center", "padding": "6px 4px",
                    "borderRadius": "5px", "border": "1px solid #2a2a4a",
                    "backgroundColor": "#161625", "color": "#c8d0e0",
                    "boxSizing": "border-box",
                }),
                html.Button("+", id="trace-hours-plus", n_clicks=0, style=STEPPER_BTN_STYLE),
            ]),
        ]),
        html.Div(style={"marginBottom": "14px"}, children=[
            html.Label("Saat dilimi (UTC farkı)", id="timezone-label",
                       style={"fontSize": "12px", "color": "#888",
                              "display": "block", "marginBottom": "6px"}),
            # ONEMLI: dcc.Slider yerine dcc.Dropdown -- kullanici listeden
            # secim istedi. Onceki denemede dropdown'un arka plani beyaz
            # kalmisti çünkü sadece disaridaki kutu (.Select-control)
            # stillendirilmisti; simdi acilan menu ve secenekler de dahil
            # tum react-select siniflari "dark-dropdown" CSS'iyle kapsaniyor
            # (bkz. index_string). clearable=False -- saat dilimi hep bir
            # deger tasimali, bos birakilamaz.
            dcc.Dropdown(
                id="timezone-dropdown",
                options=[],  # update_timezone_options callback'i dolduruyor (dil'e gore etiket)
                value=DEFAULT_TIMEZONE,
                clearable=False, searchable=False,
                className="dark-dropdown",
            ),
        ]),
        html.Div(style={"marginBottom": "14px"}, children=[
            html.Label("Harita Türü", id="map-style-label",
                       style={"fontSize": "12px", "color": "#888",
                              "display": "block", "marginBottom": "6px"}),
            html.Div(style={"display": "flex", "gap": "6px"}, children=[
                html.Button("Sokak", id="map-style-street-btn", n_clicks=0,
                           style=LANG_BTN_ACTIVE_STYLE),
                html.Button("Uydu", id="map-style-satellite-btn", n_clicks=0,
                           style=LANG_BTN_INACTIVE_STYLE),
            ]),
        ]),
        html.Div(children=[
            html.Label("Dil", id="language-label",
                       style={"fontSize": "12px", "color": "#888",
                              "display": "block", "marginBottom": "6px"}),
            html.Div(style={"display": "flex", "gap": "6px"}, children=[
                html.Button("TR", id="language-tr-btn", n_clicks=0,
                           style=LANG_BTN_ACTIVE_STYLE),
                html.Button("EN", id="language-en-btn", n_clicks=0,
                           style=LANG_BTN_INACTIVE_STYLE),
            ]),
        ]),
    ]),

    dcc.Store(id="settings-open", data=False),
    dcc.Store(id="trace-hours-setting", data=DEFAULT_TRACE_HOURS),
    dcc.Store(id="timezone-setting", data=DEFAULT_TIMEZONE),
    dcc.Store(id="language-setting", data=DEFAULT_LANGUAGE),
    dcc.Store(id="map-style-setting", data=DEFAULT_MAP_STYLE),

    # ---------------------------------- Sol kayan panel (varsayilan gizli) --
    html.Div(id="left-panel", style={**LEFT_PANEL_BASE, "transform": "translateX(-100%)"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "12px"}, children=[
            html.H3("Uçak Bilgisi", id="aircraft-info-title",
                    style={"color": "#00b4d8", "margin": 0, "fontSize": "18px"}),
            html.Button("×", id="close-panel-btn", n_clicks=0, style={
                "background": "none", "border": "none", "color": "#888",
                "fontSize": "26px", "cursor": "pointer", "lineHeight": "1",
                "padding": "0 4px",
            }),
        ]),
        html.Div(id="live-aircraft-panel"),
        html.Div(id="route-info", style={"marginTop": "10px"}),
        html.Div(id="aircraft-info", style={"marginTop": "10px"}),
    ]),

    # --------------------------- Sag-alt gecmis paneli (varsayilan gizli) --
    html.Div(id="history-panel", style={**HISTORY_PANEL_BASE, "transform": "translateY(100%)"},
             children=[
        html.H4("Geçmiş (son 24 saat)", id="history-panel-title",
                style={"color": "#90e0ef", "marginTop": 0, "fontSize": "13px"}),
        dcc.Graph(id="history-chart", style={"height": "200px"},
                  config={"displayModeBar": False}),
    ]),
])


@app_dash.callback(
    [Output("aircraft-geojson", "data"), Output("status", "children")],
    [Input("tick", "n_intervals"), Input("timezone-setting", "data"),
     Input("language-setting", "data"), Input("show-civil", "data"),
     Input("show-military", "data")]
)
def update_map(n, tz_name, lang, show_civil, show_military):
    tz = _resolve_tz(tz_name)
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    cat_labels = CATEGORY_LABELS.get(lang, CATEGORY_LABELS[DEFAULT_LANGUAGE])
    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []
    try:
        alerts = requests.get("http://localhost:8000/api/alerts", timeout=3).json()
    except Exception:
        alerts = []

    # Alarm listesi artik ayri bir panelde gosterilmiyor (yerine ayarlar
    # butonu koyuldu) -- ama kirmizi marker renklendirmesi icin
    # alert_icaos hala gerekli, o yuzden fetch etmeye devam ediyoruz.
    alert_icaos = {a.get("icao24") for a in alerts}

    # ONEMLI: askeri/sivil filtre sol-ust butonlarindan geliyor. Ikisi de
    # acikken (varsayilan) davranis eskisiyle birebir ayni -- hicbir ucak
    # elenmiyor. Biri kapatilinca o gruptaki ucaklar hem haritadan hem de
    # asagidaki "aktif ucus" sayacindan (status bar) dusuyor, cunku sayac
    # goruntulenen/secilebilir ucaklari yansitmali.
    def _passes_filter(f):
        is_mil = bool(f.get("is_military"))
        return (show_military if is_mil else show_civil)

    flights = [f for f in flights if _passes_filter(f)]

    # Her ucak icin GOSTERIME HAZIR (formatlanmis) degerleri Python'da
    # hesaplayip duz bir sozluge koyuyoruz -- ceviri (t[...]) ve sayi
    # formatlama SADECE burada yapiliyor, JS tarafinda (_POINT_TO_LAYER_JS /
    # _ON_EACH_FEATURE_JS) tekrarlanmiyor, JS sadece bu degerleri yerlestiriyor.
    points = []
    for f in flights:
        icao = f.get("icao24", "")
        is_military = bool(f.get("is_military"))
        if icao in alert_icaos:
            color = ALERT_COLOR
        elif is_military:
            color = MILITARY_COLOR
        else:
            color = DEFAULT_AIRCRAFT_COLOR
        callsign = f.get("callsign", "").strip()
        category_label = cat_labels.get(f.get("category", ""), None)
        subtitle_parts = [icao.upper()]
        if category_label:
            subtitle_parts.append(category_label)
        if is_military:
            subtitle_parts.append(t["tooltip_military_tag"])
        subtitle = "  ·  ".join(subtitle_parts)

        # ONEMLI: adsb.lol/readsb, bir ucaktan mesaj kesilse bile onu 60
        # saniyeye kadar listede TUTAR ("seen" alani = mesajin GERCEKTE
        # kac saniye once alindigi). Bu sureyi KULLANMADAN once, sinyali
        # onlarca saniyedir kesilmis bir ucak bile haritada "taze"
        # gorunuyordu -- kullanicinin fark ettigi "olu sinyal" sorunu bu.
        # 10sn'nin altinda tam opak, 40sn+ icin belirgin soluk (0.35),
        # arasinda dogrusal geciyor. signal_age_sec None ise (kaynak
        # saglamiyorsa) GUVENLI VARSAYILAN: tam opak (dim etmeyecek kadar
        # bilgimiz yok).
        signal_age = f.get("signal_age_sec")
        if signal_age is None:
            opacity = 1.0
            signal_age_text = None
        else:
            opacity = max(0.35, min(1.0, 1.0 - (signal_age - 10) / 30))
            signal_age_text = f"{signal_age:.0f}sn" if signal_age >= 10 else None

        points.append(dict(
            lat=f.get("lat", 39), lon=f.get("lon", 35),
            icao24=icao,
            callsign=callsign or icao.upper(),
            color=color,
            opacity=round(opacity, 2),
            track=f.get("track") or 0,
            subtitle=subtitle,
            alt_text=f"{f.get('alt', 0):.0f} m",
            speed_text=(f"{f.get('velocity'):.0f} m/s"
                       if f.get('velocity') is not None else "—"),
            track_text=(f"{f.get('track'):.0f}°"
                       if f.get('track') is not None else "—"),
            vspeed_text=(f"{f.get('vertical_rate'):+.1f} m/s"
                        if f.get('vertical_rate') is not None else "—"),
            lbl_alt=t["tooltip_alt"], lbl_speed=t["tooltip_speed"],
            lbl_track=t["tooltip_track"], lbl_vspeed=t["tooltip_vspeed"],
            signal_age_text=signal_age_text, lbl_signal_age=t["tooltip_signal_age"],
        ))

    geojson_data = dlx.dicts_to_geojson(points)

    ts = datetime.now(tz).strftime("%H:%M:%S")
    status = t["status_bar"].format(ts=ts, n=len(flights), a=len(alerts))

    return geojson_data, status


@app_dash.callback(
    Output("aircraft-select", "data"),
    [Input("aircraft-geojson", "n_clicks"), Input("close-panel-btn", "n_clicks")],
    State("aircraft-geojson", "clickData"),
    prevent_initial_call=True
)
def select_or_close(geojson_clicks, close_clicks, feature):
    """Ucak marker'ina tiklaninca secim ayarlar, kapatma (x) butonuna
    tiklaninca secimi temizler -- ikisi de ayni Output'u (aircraft-select)
    yazdigi icin tek callback'te birlestirildi (Dash coklu-callback ayni
    Output kisitini boyle asiyoruz, allow_duplicate'a gerek kalmadan).

    ONEMLI: eskiden pattern-matching marker id'leri (ALL) dinleniyordu,
    artik TEK bir dl.GeoJSON bileseninin n_clicks/clickData ciftini
    kullaniyoruz -- clickData, tiklanan GeoJSON feature'ini (properties
    dahil) iceriyor. Cluster balonuna tiklanirsa feature'da "icao24"
    property'si OLMAZ (supercluster'in sentetik grup feature'i) -- bu
    durumda secim degistirmiyoruz, zoomToBoundsOnClick zaten kendiliginden
    o kumeye yakinlastiriyor."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_prop_id = ctx.triggered[0]["prop_id"]
    if trigger_prop_id == "close-panel-btn.n_clicks":
        return None

    if not feature:
        return dash.no_update
    icao24 = (feature.get("properties") or {}).get("icao24")
    return icao24 if icao24 else dash.no_update


@app_dash.callback(
    [Output("left-panel", "style"), Output("history-panel", "style")],
    Input("aircraft-select", "data"),
)
def toggle_panels(icao24):
    """Bir ucak secilince iki paneli de kaydirarak icine getirir,
    secim temizlenince (kapatma butonu) tekrar disari kaydirir."""
    left = dict(LEFT_PANEL_BASE)
    history = dict(HISTORY_PANEL_BASE)
    if icao24:
        left["transform"] = "translateX(0)"
        history["transform"] = "translateY(0)"
    else:
        left["transform"] = "translateX(-100%)"
        history["transform"] = "translateY(100%)"
    return left, history


@app_dash.callback(
    Output("settings-open", "data"),
    [Input("settings-btn", "n_clicks"), Input("close-settings-btn", "n_clicks")],
    State("settings-open", "data"),
    prevent_initial_call=True,
)
def toggle_settings_open(gear_clicks, close_clicks, is_open):
    """Disli butonuna tiklaninca ac/kapa (toggle), panel icindeki x
    butonuna tiklaninca kesin kapat -- ayni Output'u iki farkli butondan
    yonetmek icin (sol paneldeki kapatma mantigiyla ayni desen)."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "close-settings-btn.n_clicks":
        return False
    if trigger == "settings-btn.n_clicks":
        return not is_open
    return dash.no_update


@app_dash.callback(
    Output("settings-panel", "style"),
    Input("settings-open", "data"),
)
def show_settings_panel(is_open):
    style = dict(SETTINGS_PANEL_BASE)
    style["display"] = "block" if is_open else "none"
    return style


@app_dash.callback(
    Output("trace-hours-input", "value"),
    [Input("trace-hours-minus", "n_clicks"), Input("trace-hours-plus", "n_clicks")],
    State("trace-hours-input", "value"),
    prevent_initial_call=True,
)
def step_trace_hours(minus_clicks, plus_clicks, current):
    """Kendi +/- butonlarimiz -- tek kontrol bu, native tarayici spinner'i
    input type="text" oldugu icin hic olusmuyor (eskiden type="number"
    ile hem native ok hem bu butonlar birlikte gorunuyordu, "2 cift +/-"
    izlenimi veriyordu, native olan da daha yavas tepki veriyordu).
    Bu callback sadece input'un GORUNEN degerini (string) degistiriyor;
    Store'a yazma islemini update_trace_hours_setting halleder."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    try:
        current_val = int(current)
    except (TypeError, ValueError):
        current_val = DEFAULT_TRACE_HOURS
    if trigger == "trace-hours-minus.n_clicks":
        return str(max(1, current_val - 1))
    if trigger == "trace-hours-plus.n_clicks":
        return str(min(24, current_val + 1))
    return dash.no_update


@app_dash.callback(
    Output("trace-hours-setting", "data"),
    Input("trace-hours-input", "value"),
    prevent_initial_call=True,
)
def update_trace_hours_setting(value):
    # value artik string (type="text") -- gecerli bir tam sayi degilse
    # (kullanici elle harf/bos deger girdiyse) Store'u bozmadan yok sayiyoruz.
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return dash.no_update
    if parsed < 1 or parsed > 24:
        return dash.no_update
    return parsed


@app_dash.callback(
    Output("timezone-setting", "data"),
    Input("timezone-dropdown", "value"),
    prevent_initial_call=True,
)
def update_timezone_setting(value):
    # ONEMLI: "if not value" DEGIL -- UTC+0 secildiginde value=0 olur ve
    # Python'da 0 falsy'dir, eski kod bu durumda guncellemeyi sessizce
    # reddediyordu (UTC+0 hicbir zaman secilemiyordu). None kontrolu dogru.
    if value is None:
        return dash.no_update
    return value


@app_dash.callback(
    Output("timezone-dropdown", "options"),
    Input("language-setting", "data"),
)
def update_timezone_options(lang):
    """Saat dilimi listesinin etiketlerini secili dile gore uretir --
    deger (UTC ofseti, int) hep ayni kalir, sadece gorunen metin degisir."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    options = []
    for h in range(-12, 15):
        label = f"UTC{h:+d}"
        if h == DEFAULT_TIMEZONE:
            label += t["tz_default_suffix"]
        options.append({"label": label, "value": h})
    return options


@app_dash.callback(
    Output("language-setting", "data"),
    [Input("language-tr-btn", "n_clicks"), Input("language-en-btn", "n_clicks")],
    prevent_initial_call=True,
)
def update_language_setting(tr_clicks, en_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "language-tr-btn.n_clicks":
        return "tr"
    if trigger == "language-en-btn.n_clicks":
        return "en"
    return dash.no_update


@app_dash.callback(
    [Output("language-tr-btn", "style"), Output("language-en-btn", "style")],
    Input("language-setting", "data"),
)
def update_language_buttons(lang):
    if lang == "en":
        return LANG_BTN_INACTIVE_STYLE, LANG_BTN_ACTIVE_STYLE
    return LANG_BTN_ACTIVE_STYLE, LANG_BTN_INACTIVE_STYLE


@app_dash.callback(
    Output("map-style-setting", "data"),
    [Input("map-style-street-btn", "n_clicks"), Input("map-style-satellite-btn", "n_clicks")],
    prevent_initial_call=True,
)
def update_map_style_setting(street_clicks, satellite_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "map-style-street-btn.n_clicks":
        return "street"
    if trigger == "map-style-satellite-btn.n_clicks":
        return "satellite"
    return dash.no_update


@app_dash.callback(
    [Output("map-style-street-btn", "style"), Output("map-style-satellite-btn", "style")],
    Input("map-style-setting", "data"),
)
def update_map_style_buttons(style):
    if style == "satellite":
        return LANG_BTN_INACTIVE_STYLE, LANG_BTN_ACTIVE_STYLE
    return LANG_BTN_ACTIVE_STYLE, LANG_BTN_INACTIVE_STYLE


@app_dash.callback(
    [Output("base-tile-layer", "url"), Output("base-tile-layer", "attribution")],
    Input("map-style-setting", "data"),
)
def update_base_tile_layer(style):
    """Harita altligini degistirir -- varsayilan "street" (mevcut
    OpenStreetMap katmani, davranis eskisiyle ayni), "satellite" secilirse
    Google Satellite tile'larina geciliyor (API anahtari gerekmiyor).
    ONEMLI: her iki katman da SABIT tek subdomain'li URL kullaniyor (bkz.
    TILE_LAYERS yorumu) -- boylece burada sadece "url" degisiyor, ki bu
    react-leaflet'in guvenilir sekilde calisma zamaninda uyguladigi tek
    TileLayer prop'u (Leaflet setUrl() ile)."""
    layer = TILE_LAYERS.get(style, TILE_LAYERS[DEFAULT_MAP_STYLE])
    return layer["url"], layer["attribution"]


@app_dash.callback(
    [Output("settings-title", "children"), Output("trace-hours-label", "children"),
     Output("timezone-label", "children"), Output("language-label", "children"),
     Output("aircraft-info-title", "children"), Output("history-panel-title", "children"),
     Output("filter-civil-btn", "children"), Output("filter-military-btn", "children"),
     Output("map-style-label", "children"), Output("map-style-street-btn", "children"),
     Output("map-style-satellite-btn", "children")],
    Input("language-setting", "data"),
)
def update_static_texts(lang):
    """Sabit basliklari/etiketleri secili dile gore gunceller. Diger tum
    dinamik metinler (panel icerikleri, tooltip, grafik) kendi
    callback'lerinde language-setting'i dogrudan Input olarak aliyor."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    return (t["settings_title"], t["trace_hours_label"], t["timezone_label"],
            t["language_label"], t["aircraft_info_title"], t["history_panel_title"],
            t["filter_civil_label"], t["filter_military_label"],
            t["map_style_label"], t["map_style_street"], t["map_style_satellite"])


@app_dash.callback(
    Output("show-civil", "data"),
    Input("filter-civil-btn", "n_clicks"),
    State("show-civil", "data"),
    prevent_initial_call=True,
)
def toggle_show_civil(n_clicks, current):
    return not current


@app_dash.callback(
    Output("show-military", "data"),
    Input("filter-military-btn", "n_clicks"),
    State("show-military", "data"),
    prevent_initial_call=True,
)
def toggle_show_military(n_clicks, current):
    return not current


@app_dash.callback(
    Output("filter-civil-btn", "style"),
    Input("show-civil", "data"),
)
def style_civil_filter_btn(visible):
    # Aktif (gorunuyor) -- sivil rengiyle dolu; pasif (gizli) -- soluk gri.
    return FILTER_BTN_CIVIL_ACTIVE_STYLE if visible else FILTER_BTN_INACTIVE_STYLE


@app_dash.callback(
    Output("filter-military-btn", "style"),
    Input("show-military", "data"),
)
def style_military_filter_btn(visible):
    return FILTER_BTN_MILITARY_ACTIVE_STYLE if visible else FILTER_BTN_INACTIVE_STYLE


@app_dash.callback(
    Output("flight-path-layer", "children"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data"),
     Input("trace-hours-setting", "data")]
)
def update_flight_path(n, icao24, trace_hours):
    """Secili ucagin son N saatlik konum gecmisini haritada cizgi olarak
    cizer (N, ayarlar panelinden degistirilebilir, varsayilan 2 saat).
    Mevcut /api/history endpoint'ini aynen kullaniyor (lat/lon zaten
    donuyordu), backend'e hic dokunmadan calisir."""
    if not icao24:
        return []
    hours = trace_hours or DEFAULT_TRACE_HOURS
    try:
        data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                            params={"hours": hours}, timeout=5).json()
    except Exception:
        data = []
    if not data or isinstance(data, dict):
        return []

    positions = [[d["lat"], d["lon"]] for d in data
                 if d.get("lat") is not None and d.get("lon") is not None]
    if len(positions) < 2:
        return []

    return [dl.Polyline(positions=positions, color="#00b4d8",
                        weight=3, opacity=0.6)]


def _bearing_deg(lat1, lon1, lat2, lon2):
    """(lat1,lon1)'den (lat2,lon2)'ye buyuk daire uzerindeki BASLANGIC
    yonu (derece, 0-360, kuzeyden saat yonunde). Rota tutarlilik
    kontrolu icin kullanilir -- bkz. _route_is_plausible()."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _route_is_plausible(ac_lat, ac_lon, ac_track, dest_lat, dest_lon, threshold_deg=90):
    """adsbdb.com'un cagri kodu -> rota eslemesi GERCEK ZAMANLI degil --
    STATIK/gecmise dayali. Bazi hava yollari (orn. Wizz Air) ayni ucus
    numarasini farkli gunlerde FARKLI sehir ciftleri icin yeniden
    kullaniyor -- bu durumda adsbdb eski/yanlis bir rota donebiliyor
    (orn. WZZ43: gercekte Krakow->Stavanger ama adsbdb Londra->Budapeste
    diyor).

    UCAGIN GERCEK ANLIK YONUYLE (track) BASIT BIR TUTARLILIK KONTROLU:
    ucagin su anki konumundan iddia edilen varisa olan yon ile ucagin
    GERCEKTEN ucmakta oldugu yon (track) arasindaki fark threshold_deg'i
    (varsayilan 90) asarsa, rota SUPHELI sayilir -- WZZ43 orneginde bu
    fark 180 derece cikiyordu (tam ters yon).

    ONEMLI: bu KUSURSUZ bir dogrulama DEGIL, sadece bir HEURISTIK --
    kalkistan hemen sonraki yon degisiklikleri (SID prosedurleri),
    holding pattern'ler veya rotadan gecici sapmalar YANLIS ALARM
    verebilir. Ama acik yon celiskilerini (WZZ43 gibi tam ters yon)
    yakalamakta etkili."""
    if ac_track is None:
        return True  # yon bilinmiyorsa kontrol edemeyiz, varsayilan: guven
    expected_bearing = _bearing_deg(ac_lat, ac_lon, dest_lat, dest_lon)
    diff = abs((expected_bearing - ac_track + 180) % 360 - 180)
    return diff <= threshold_deg


def _great_circle_points(lat1, lon1, lat2, lon2, n=64):
    """Iki nokta arasindaki BUYUK DAIRE (great circle) yayi uzerinde n+1
    ara nokta uretir -- Ed Williams'in "Aviation Formulary"sindeki
    standart kure-uzeri enterpolasyon formulu (havacilikta yaygin
    kullanilir). Duz cizgiden farki: uzun menzilli ucuslarda gercek
    ucus rotasina (great circle, en kisa mesafe) cok daha yakin bir
    gorsel verir -- orn. NYC->Londra rotasinin neden Gronland'a yakin
    kuzeyden kavis yaptigini gosterir.

    ONEMLI: 180. meridyeni (tarih degistirme hatti) gecen rotalarda
    (orn. Tokyo->Los Angeles gibi Pasifik ucuslari) ardisik noktalarin
    boylami -180/+180 SINIRINDA ANI SICRAMA yapabilir (orn. 179->-179) --
    bu, Leaflet'te haritanin YANLIS tarafina cizilen uzun, hatali bir
    cizgiye yol acar. "Unwrap" ile SUREKLILIK sagliyoruz -- boylam
    -180/180 araligina sikistirilmiyor, gerekirse 360'in katlariyla
    kaydiriliyor (Leaflet, araligin disindaki boylam degerlerini de
    doğru sekilde projekte eder)."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    d = 2 * math.asin(min(1, math.sqrt(a)))

    if d < 1e-9:  # ayni nokta / neredeyse ayni -- egriye gerek yok
        return [[lat1, lon1], [lat2, lon2]]

    points = []
    prev_lon = None
    for i in range(n + 1):
        f = i / n
        A = math.sin((1 - f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
        y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
        z = A * math.sin(lat1r) + B * math.sin(lat2r)
        lat_i = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
        lon_i = math.degrees(math.atan2(y, x))

        if prev_lon is not None:
            while lon_i - prev_lon > 180:
                lon_i -= 360
            while lon_i - prev_lon < -180:
                lon_i += 360
        prev_lon = lon_i

        points.append([lat_i, lon_i])

    return points


@app_dash.callback(
    Output("route-line-layer", "children"),
    Input("aircraft-select", "data"),
)
def update_route_line(icao24):
    """Secili ucagin kalkis-varis havalimanlari arasina GREAT CIRCLE
    (buyuk daire) egrisi cizer -- gercek ucus rotasina (en kisa kure-uzeri
    mesafe) duz cizgiden cok daha yakin bir gorsel. Hesaplama icin
    _great_circle_points() kullanilir (bkz. o fonksiyonun docstring'i).

    ADS-B protokolu rota tasimadigi icin adsbdb.com'dan gelen havalimani
    koordinatlarini kullaniyoruz -- update_route_info ile AYNI /api/route
    endpoint'i (Redis'te 12 saat cache'li), yani bunun icin EKSTRA dis
    API cagrisi yok, zaten cekilen veriyi tekrar kullaniyoruz."""
    if not icao24:
        return []
    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []
    match = next((f for f in flights if f.get("icao24") == icao24), None)
    callsign = (match.get("callsign") or "").strip() if match else ""
    if not callsign:
        return []

    try:
        route = requests.get(
            f"http://localhost:8000/api/route/{callsign}",
            params={"lat": match.get("lat"), "lon": match.get("lon")},
            timeout=5,
        ).json()
    except Exception:
        route = {"found": False}

    if not route.get("found"):
        return []

    o_lat, o_lon = route.get("origin_lat"), route.get("origin_lon")
    d_lat, d_lon = route.get("dest_lat"), route.get("dest_lon")
    # ONEMLI: adsbdb'deki rotalarin KUCUK bir kismi koordinatsiz
    # donebiliyor (havalimani veritabaninda eksik veri) -- bu durumda
    # cizgiyi hic cizmiyoruz (yanlis/varsayilan 0,0 koordinatina
    # cizmek yaniltici olurdu), metin paneli (update_route_info) yine
    # de havalimani ismini gosterebiliyor olabilir.
    if None in (o_lat, o_lon, d_lat, d_lon):
        return []

    origin_label = f"{route.get('origin_city','')} ({route.get('origin_iata','')})"
    dest_label = f"{route.get('dest_city','')} ({route.get('dest_iata','')})"
    positions = _great_circle_points(o_lat, o_lon, d_lat, d_lon, n=64)

    # ONEMLI: IKI sinyali birlikte kullaniyoruz -- (1) adsb.lol'un route'u
    # KENDI SUNUCUSUNDA zaten olasilik filtresinden gecirmis olabilir
    # (bkz. "source_plausible" -- adsbdb.com fallback'inde bu alan yok,
    # o zaman varsayilan True), (2) bizim KENDI ucak-yonu tutarlilik
    # kontrolumuz (bkz. _route_is_plausible() docstring'i). Ikisinden
    # BIRI supheli derse, supheli sayiyoruz. Supheli rotalarda cizgiyi
    # SILMIYORUZ (belki dogrudur), ama daha soluk/kesikli cizerek
    # "bu kesin degil" gorsel sinyali veriyoruz.
    own_check = _route_is_plausible(
        match.get("lat"), match.get("lon"), match.get("track"), d_lat, d_lon
    ) if match else True
    plausible = route.get("source_plausible", True) and own_check
    line_opacity = 0.75 if plausible else 0.3
    dash = "6, 6" if plausible else "2, 8"

    return [
        dl.Polyline(positions=positions,
                   color="#f7b731", weight=2, opacity=line_opacity, dashArray=dash),
        dl.CircleMarker(center=[o_lat, o_lon], radius=5, color="#f7b731",
                        fillColor="#f7b731", fillOpacity=0.9,
                        children=[dl.Tooltip(origin_label)]),
        dl.CircleMarker(center=[d_lat, d_lon], radius=5, color="#f7b731",
                        fillColor="#f7b731", fillOpacity=0.9,
                        children=[dl.Tooltip(dest_label)]),
    ]


@app_dash.callback(
    Output("route-info", "children"),
    [Input("aircraft-select", "data"), Input("language-setting", "data")],
)
def update_route_info(icao24, lang):
    """Kalkis/varis bilgisi. SADECE secim veya dil degistiginde calisir
    (tick'e bagli degil) -- rota veritabani nadiren degistigi icin her
    15 saniyede tekrar sorgulamaya gerek yok, hem dis API'yi hem Redis'i
    gereksiz yormayalim."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    if not icao24:
        return None

    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []
    match = next((f for f in flights if f.get("icao24") == icao24), None)
    callsign = (match.get("callsign") or "").strip() if match else ""

    if not callsign:
        return html.Div(t["no_callsign"],
                        style={"color": "#666", "fontSize": "12px"})

    try:
        route = requests.get(
            f"http://localhost:8000/api/route/{callsign}",
            params={"lat": match.get("lat"), "lon": match.get("lon")} if match else {},
            timeout=5,
        ).json()
    except Exception:
        route = {"found": False}

    if not route.get("found"):
        return html.Div(t["route_not_found"].format(callsign=callsign),
                        style={"color": "#666", "fontSize": "12px"})

    origin = f"{route.get('origin_city','?')} ({route.get('origin_iata','—')})"
    dest = f"{route.get('dest_city','?')} ({route.get('dest_iata','—')})"
    airline = route.get("airline")

    # ONEMLI: IKI sinyali birlikte kullaniyoruz -- (1) adsb.lol'un route'u
    # KENDI SUNUCUSUNDA zaten olasilik filtresinden gecirmis olabilir
    # (bkz. "source_plausible" -- adsbdb.com fallback'inde bu alan yok,
    # o zaman varsayilan True), (2) bizim KENDI ucak-yonu tutarlilik
    # kontrolumuz (bkz. _route_is_plausible() docstring'i, WZZ43 ornegi).
    # Ikisinden BIRI supheli derse, supheli sayiyoruz. Rotayi
    # GIZLEMIYORUZ (belki dogrudur, heuristik kesin degil), sadece
    # supheli oldugunu isaretliyoruz.
    plausible = route.get("source_plausible", True)
    d_lat, d_lon = route.get("dest_lat"), route.get("dest_lon")
    if match and d_lat is not None and d_lon is not None:
        plausible = plausible and _route_is_plausible(
            match.get("lat"), match.get("lon"), match.get("track"), d_lat, d_lon)

    return html.Div(style={
        "background": "#161625", "padding": "8px 10px",
        "borderRadius": "6px", "fontSize": "13px",
    }, children=[
        html.Div(f"✈ {airline}" if airline else "",
                 style={"color": "#666", "fontSize": "11px", "marginBottom": "4px"}),
        html.Div([
            html.Span(origin, style={"color": "#00b4d8"}),
            html.Span("  →  ", style={"color": "#666"}),
            html.Span(dest, style={"color": "#00b4d8"}),
        ]),
        html.Div(t["route_uncertain"], style={
            "color": "#f7b731", "fontSize": "11px", "marginTop": "4px",
        }) if not plausible else None,
    ])


@app_dash.callback(
    Output("aircraft-info", "children"),
    [Input("aircraft-select", "data"), Input("language-setting", "data")],
)
def update_aircraft_info(icao24, lang):
    """Ucak tipi/uretici/tescil/sahip bilgisi. SADECE secim veya dil
    degistiginde calisir -- icao24 hex sabit oldugu icin ekstra flights
    sorgusuna gerek yok, dogrudan yeni endpoint'e sorulur."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    if not icao24:
        return None

    try:
        info = requests.get(f"http://localhost:8000/api/aircraft_info/{icao24}",
                            timeout=5).json()
    except Exception:
        info = {"found": False}

    if not info.get("found"):
        return html.Div(t["aircraft_info_not_found"],
                        style={"color": "#666", "fontSize": "12px"})

    rows = []
    if info.get("manufacturer") or info.get("type"):
        rows.append(f"{info.get('manufacturer','')} {info.get('type','')}".strip())
    if info.get("registration"):
        rows.append(f"{t['registration']}: {info['registration']}")
    if info.get("owner"):
        owner_line = info["owner"]
        if info.get("owner_country"):
            owner_line += f" ({info['owner_country']})"
        rows.append(owner_line)

    children = [
        html.Div(style={"background": "#161625", "padding": "8px 10px",
                        "borderRadius": "6px", "fontSize": "13px"}, children=[
            html.Div(row, style={"color": "#00b4d8" if i == 0 else "#c8d0e0",
                                  "marginBottom": "3px",
                                  "fontWeight": "500" if i == 0 else "400"})
            for i, row in enumerate(rows)
        ] if rows else [html.Div(t["no_details"], style={"color": "#666"})])
    ]

    if info.get("photo_thumb"):
        children.append(html.Img(src=info["photo_thumb"], style={
            "width": "100%", "borderRadius": "6px", "marginTop": "8px",
        }))

    return html.Div(children)


@app_dash.callback(
    Output("live-aircraft-panel", "children"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data"),
     Input("timezone-setting", "data"), Input("language-setting", "data")]
)
def update_live_panel(n, icao24, tz_name, lang):
    tz = _resolve_tz(tz_name)
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    cat_labels = CATEGORY_LABELS.get(lang, CATEGORY_LABELS[DEFAULT_LANGUAGE])
    emg_labels = EMERGENCY_LABELS.get(lang, EMERGENCY_LABELS[DEFAULT_LANGUAGE])

    if not icao24:
        return html.Div(t["click_aircraft"],
                        style={"color": "#666", "fontSize": "13px"})

    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []

    match = next((f for f in flights if f.get("icao24") == icao24), None)
    if not match:
        return html.Div(t["no_signal"].format(icao=icao24),
                        style={"color": "#e63946", "fontSize": "13px"})

    ts_raw = match.get("ts", "")
    if ts_raw:
        try:
            ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts_display = ts_dt.astimezone(tz).strftime("%H:%M:%S")
        except Exception:
            ts_display = ts_raw[:19].replace("T", " ")
    else:
        ts_display = "—"

    raw_category = match.get("category", "") or ""
    category_label = cat_labels.get(raw_category, raw_category or "—")

    squawk = match.get("squawk", "") or "—"
    emergency_code = match.get("emergency", "none")
    emergency_label = emg_labels.get(emergency_code)

    fields = [
        (t["field_icao"], match.get("icao24", "").upper()),
        (t["field_callsign"], match.get("callsign", "").strip() or "—"),
        (t["field_lat"], f"{match.get('lat', 0):.4f}°"),
        (t["field_lon"], f"{match.get('lon', 0):.4f}°"),
        (t["field_alt"], f"{match.get('alt', 0):.0f} m"),
        (t["field_speed"], f"{match.get('velocity'):.0f} m/s" if match.get('velocity') is not None else "—"),
        (t["field_track"], f"{match.get('track'):.0f}°" if match.get('track') is not None else "—"),
        (t["field_vspeed"], f"{match.get('vertical_rate'):+.1f} m/s"
                      if match.get('vertical_rate') is not None else "—"),
        (t["field_category"], category_label),
        (t["field_squawk"], squawk),
        (t["field_military"], t["military_yes"] if match.get("is_military") else t["military_no"]),
        (t["field_last_update"], ts_display),
    ]

    grid = html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "repeat(2, 1fr)",
        "gap": "8px",
    }, children=[
        html.Div([
            html.Div(label, style={"fontSize": "11px", "color": "#666"}),
            html.Div(str(value), style={"fontSize": "15px", "color": "#00b4d8",
                                         "fontWeight": "500"}),
        ], style={"background": "#161625", "padding": "8px 10px", "borderRadius": "6px"})
        for label, value in fields
    ])

    children = [grid]

    # Acil durum varsa (squawk 7500/7600/7700 veya emergency alani "none"
    # degilse) belirgin kirmizi bir uyari banner'i gosteriyoruz.
    if emergency_label or squawk in ("7500", "7600", "7700"):
        warning_text = emergency_label or t["emergency_squawk"].format(squawk=squawk)
        children.insert(0, html.Div(f"⚠ {warning_text}", style={
            "background": "#e63946", "color": "#fff", "padding": "8px 10px",
            "borderRadius": "6px", "marginBottom": "8px",
            "fontSize": "13px", "fontWeight": "600", "textAlign": "center",
        }))

    return html.Div(children)


@app_dash.callback(
    Output("history-chart", "figure"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data"),
     Input("timezone-setting", "data"), Input("language-setting", "data")]
)
def update_history(n, icao24, tz_name, lang):
    HOURS_DEFAULT = 24  # sabit -- bu grafik "rota izi" ayarindan bagimsiz,
                        # her zaman son 24 saati gosterir

    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    tz = _resolve_tz(tz_name)
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#0f0f19", plot_bgcolor="#0f0f19",
                      font=dict(color="#c8d0e0", size=10),
                      margin=dict(t=10, b=30, l=40, r=40))

    if not icao24:
        return fig  # panel zaten gizli, bos figure yeterli

    try:
        data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                            params={"hours": HOURS_DEFAULT}, timeout=10).json()
    except Exception:
        data = []

    if not data or isinstance(data, dict):
        fig.add_annotation(text=t["no_data"], showarrow=False,
                           font=dict(color="#666"))
        return fig

    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"]).dt.tz_convert(tz)

    fig.add_trace(go.Scatter(x=df["_time"], y=df["alt"], name=t["history_alt_label"],
                             line=dict(color="#00b4d8", width=1.5), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["_time"], y=df["velocity"], name=t["history_speed_label"],
                             line=dict(color="#f77f00", width=1.5), yaxis="y2"))

    fig.update_layout(
        yaxis=dict(title=t["history_alt_label"], side="left"),
        yaxis2=dict(title=t["history_speed_label"], side="right", overlaying="y"),
        legend=dict(orientation="h", font=dict(size=9), y=1.15),
    )
    return fig


if __name__ == "__main__":
    print("Dash başlıyor: http://localhost:8050")
    app_dash.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
