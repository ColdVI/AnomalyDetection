"""
app.py
Canli harita (Redis) + model alarmlari + secili ucagin gecmis grafigi
(InfluxDB, 7 gune kadar) tek Dash uygulamasinda. FastAPI arka planda
thread olarak calisir, Dash ondan besleniyor.

ONEMLI: Bunu calistirmadan once dashboard_consumer.py'nin AYRI bir
terminalde calisiyor olmasi lazim, yoksa Redis/InfluxDB'de veri olmaz.
Alert paneli, model ekibi "uav.alerts" topic'ine yazmaya baslayana
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import redis
import requests
import uvicorn
import dash
import dash_leaflet as dl
from dash_extensions.javascript import assign
from dash import Dash, dcc, html, Output, Input, State, ALL
from fastapi import FastAPI, Response
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
        "timezone_label": "Saat dilimi (UTC farkı)",
        "language_label": "Dil",
        "aircraft_info_title": "Uçak Bilgisi",
        "history_panel_title": "Geçmiş",
        "history_range_placeholder_start": "Başlangıç",
        "history_range_placeholder_end": "Bitiş",
        "history_day_placeholder": "Gün",
        "history_hour_placeholder": "Saat",
        "history_calculate_label": "Hesapla",
        "callsign_search_placeholder": "Çağrı kodu ara...",
        "callsign_not_found": "'{callsign}' bulunamadı",
        "stats_title": "İstatistikler",
        "stats_yaxis_label": "Benzersiz uçak sayısı",
        "emergency_panel_title": "Acil Durumlar",
        "no_emergency": "Aktif acil durum yok",
        "replay_panel_title": "Tekrar Oynatma",
        "replay_load_label": "Yükle",
        "replay_no_data": "⚠ Bu aralıkta veri bulunamadı — tarih/saat seçili mi?",
        "replay_loaded_label": "✓ {n} kare yüklendi — ▶ ile başlat",
        "replay_live_label": "Canlıya Dön",
        "altitude_legend_title": "İRTİFA (m)",
        "data_source_label": "Veri Kaynağı",
        "data_source_active": "Aktif: {source}",
        "data_source_pending": "İsteniyor: {requested} · aktif: {active} (geçiş bekleniyor)",
        "previous_flights_title": "Önceki Uçuşlar (7 gün)",
        "no_previous_flights": "Bu uçak için geçmiş uçuş bulunamadı.",
        "flight_duration_min": "{min} dk",
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
        "status_bar_main": "{ts} | {n} aktif uçuş",
        "status_bar_alarm": " | {a} alarm",
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
        "filter_ground_label": "Yerde",
        "airline_filter_placeholder": "Firma...",
        "field_military": "Askeri mi",
        "military_yes": "Evet",
        "military_no": "Hayır",
        "field_ground": "Yerde mi",
        "ground_yes": "Evet",
        "ground_no": "Hayır",
        "tooltip_military_tag": "Askeri",
        "tooltip_ground_tag": "Yerde",
        "map_style_label": "Harita Türü",
        "map_style_street": "Sokak",
        "map_style_satellite": "Uydu",
        "map_style_dark": "Karanlık",
        "signal_staleness_label": "Sinyal Yaşı Eşiği",
    },
    "en": {
        "settings_title": "Settings",
        "timezone_label": "Time zone (UTC offset)",
        "language_label": "Language",
        "aircraft_info_title": "Aircraft Info",
        "history_panel_title": "History",
        "history_range_placeholder_start": "Start",
        "history_range_placeholder_end": "End",
        "history_day_placeholder": "Day",
        "history_hour_placeholder": "Hour",
        "history_calculate_label": "Calculate",
        "callsign_search_placeholder": "Search callsign...",
        "callsign_not_found": "'{callsign}' not found",
        "stats_title": "Statistics",
        "stats_yaxis_label": "Unique aircraft count",
        "emergency_panel_title": "Emergency Alerts",
        "no_emergency": "No active emergencies",
        "replay_panel_title": "Replay",
        "replay_load_label": "Load",
        "replay_no_data": "⚠ No data found in this range — is a date/time selected?",
        "replay_loaded_label": "✓ {n} frames loaded — press ▶ to start",
        "replay_live_label": "Return to Live",
        "altitude_legend_title": "ALTITUDE (m)",
        "data_source_label": "Data Source",
        "data_source_active": "Active: {source}",
        "data_source_pending": "Requested: {requested} · active: {active} (switch pending)",
        "previous_flights_title": "Previous Flights (7 days)",
        "no_previous_flights": "No previous flights found for this aircraft.",
        "flight_duration_min": "{min} min",
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
        "status_bar_main": "{ts} | {n} active flights",
        "status_bar_alarm": " | {a} alerts",
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
        "filter_ground_label": "Ground",
        "airline_filter_placeholder": "Airline...",
        "field_military": "Military",
        "military_yes": "Yes",
        "military_no": "No",
        "field_ground": "On Ground",
        "ground_yes": "Yes",
        "ground_no": "No",
        "tooltip_military_tag": "Military",
        "tooltip_ground_tag": "Ground",
        "map_style_label": "Map Style",
        "map_style_street": "Street",
        "map_style_satellite": "Satellite",
        "map_style_dark": "Dark",
        "signal_staleness_label": "Signal Age Threshold",
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
    # CARTO Dark Matter -- API anahtari gerektirmiyor, "satellite"teki AYNI
    # sebeple (bkz. yukaridaki {s} subdomain yorumu) TEK sabit subdomain
    # ("a.basemaps...") kullaniliyor, dinamik subdomain rotasyonu YOK.
    "dark": {
        "url": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors © CARTO",
    },
}
DEFAULT_MAP_STYLE = "dark"

# Kullanicinin ayarlardan sectigi "bayat sinyal" esigi (saniye) -- bu
# esigin ALTINDAKI ucaklar tam opak, USTUNDEKILER soluk (bkz. update_map
# icindeki opacity hesabi). Onceden SABIT bir 10-40sn dogrusal soluklasma
# vardi -- adsb.lol (60sn'de bir sorgulama) icin makuldu ama OpenSky
# (90-300sn'de bir sorgulama, bkz. uav_producer.py SOURCES) icin neredeyse
# HER ucak daha ilk fetch'te esigin ustune cikip ekrandaki NEREDEYSE TUM
# filo soluk gorunuyordu -- kullanici geri bildirimi bu. Kaynaga gore SABIT
# bir esik yerine, kullanicinin secili kaynaga gore kendi esigini ayarlar
# panelinden secmesini sagliyoruz.
SIGNAL_STALENESS_OPTIONS = [30, 60, 120, 300, 600, 1800, 3600]  # saniye --
                            # 30sn/1dk/2dk/5dk/10dk/30dk/1sa
DEFAULT_SIGNAL_STALENESS_SEC = 60
STALE_SIGNAL_OPACITY = 0.35  # esigi asan ucaklarin gorunecegi soluk deger
                              # (0 degil -- tamamen kaybolmasin, hep bir iz kalsin)


def _signal_opacity(signal_age_sec, staleness_threshold_sec):
    """update_map icindeki ucak opacity hesabi -- test edilebilirlik icin
    ayri bir pure fonksiyona cikarildi. signal_age_sec None ise (kaynak
    saglamiyorsa) GUVENLI VARSAYILAN: tam opak (dim etmeyecek kadar
    bilgimiz yok). Degilse ikili: esigin ALTI tam opak, USTU sabit soluk."""
    if signal_age_sec is None:
        return 1.0
    return 1.0 if signal_age_sec <= staleness_threshold_sec else STALE_SIGNAL_OPACITY


def _format_staleness_label(sec, lang):
    """Saniyeyi kullaniciya okunakli sn/dk/sa (TR) ya da s/m/h (EN)
    etiketine cevirir -- SIGNAL_STALENESS_OPTIONS listesindeki tum
    degerler 60'in ya da 3600'un tam kati oldugu icin kesirli sonuc
    gelmez, ekstra yuvarlama gerekmiyor."""
    if sec < 60:
        return f"{sec}sn" if lang == "tr" else f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}dk" if lang == "tr" else f"{sec // 60}m"
    return f"{sec // 3600}sa" if lang == "tr" else f"{sec // 3600}h"


def _build_signal_staleness_options(lang):
    return [{"label": _format_staleness_label(s, lang), "value": s}
            for s in SIGNAL_STALENESS_OPTIONS]

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
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "uav-history")

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


# ============================================================ Kayit --
# Yeni bir kaynak eklemek icin buraya BIR giris ekle (ayrica
# uav_producer.py'deki SOURCES sozlugune ayni "key" ile fetch fonksiyonunu
# ekle) -- butonlar ve callback bu listeden turetiliyor, baska hicbir yeri
# degistirmen gerekmiyor.
DATA_SOURCE_DEFS = [
    {"key": "adsblol", "label": "adsb.lol"},
    {"key": "opensky", "label": "OpenSky"},
]
DATA_SOURCES = tuple(d["key"] for d in DATA_SOURCE_DEFS)
DEFAULT_DATA_SOURCE = DATA_SOURCE_DEFS[0]["key"]
REDIS_DATA_SOURCE_KEY = "iha:settings:data_source"
REDIS_PRODUCER_STATUS_KEY = "iha:producer_status"


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


REPLAY_MAX_RANGE_HOURS = 2  # sorgu/payload boyutunu makul tutmak icin sabit ust sinir


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


GEOCODE_CACHE_TTL = 60 * 60 * 24 * 30  # 30 gun -- yer adlari neredeyse hic degismez
GEOCODE_MAX_LOOKUPS_PER_REQUEST = 16  # guvenlik siniri, bkz. _reverse_geocode


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


def _run_api():
    uvicorn.run(app_api, host="0.0.0.0", port=8000, log_level="warning")


# ONEMLI (test edilebilirlik): bu THREAD BASLATMA + BLOKLAYICI sleep ONCEDEN
# modul seviyesindeydi -- yani sadece "import app" yapmak (orn. bir test
# dosyasindan pure bir fonksiyonu kullanmak icin) bile GERCEK bir HTTP
# sunucusunu arka planda ANINDA baslatiyordu, hic istenmeden. Docker'daki
# CMD zaten "python app.py" oldugu icin (bkz. Dockerfile) bu kod hala
# TAM OLARAK AYNI ANDA calisir -- __main__ bloguna tasinmasi uretim
# davranisini DEGISTIRMEZ, sadece "import Dashboard.app" artik yan etkisiz
# (hermetik test edilebilir) hale gelir.

# --------------------------------------------------------------------- Dash --

app_dash = Dash(__name__, title="ADS-B Local Dashboard")

# --------------------------------------------------------------------------
# UCAK KATMANI RENDER MANTIGI -- artik CLIENT-SIDE (JavaScript)
#
# ONEMLI MIMARI DEGISIKLIK (1. asama): eskiden her ucak icin Python'da bir
# dl.DivMarker + ic ice Div'li dl.Tooltip nesnesi INSA EDILIYOR, tumu
# JSON'a cevrilip tarayiciya gonderiliyordu. "Dunya" modunda (5.000-
# 11.000+ ucak) bu, hem Python tarafinda hem tarayicinin React/Leaflet
# reconciliation'inda donmaya yol aciyordu -- CPU hizindan bagimsiz,
# mimari bir sinir (tek JS thread'inde on binlerce DOM elemani senkron
# insa etmek). COZUM: dl.GeoJSON + supercluster (JS) ile "dunya zoom'unda
# binlerce nokta birkac yuz cluster balonuna indirgeniyordu.
#
# ONEMLI MIMARI DEGISIKLIK (2. asama, GUNCEL): kullanici "sekil+yon kalsin
# ama balon degil, GERCEK ucak gorunsun, donma olmasin" istedi -- tar1090
# (adsb.lol'un de kullandigi arayuz) incelendi, onlarin da ayni sorunu
# CANVAS/WebGL tabanli render ile (DOM marker DEGIL) coz-dugu goruldu
# (bkz. proje sohbet gecmisi). Bu yuzden kumeleme TAMAMEN KALDIRILDI --
# her ucak artik GERCEK bir GeoJSON Polygon (kucuk, heading'e gore
# ONCEDEN Python'da dondurulmus bir ok/dart sekli, bkz.
# _rotated_aircraft_polygon()), TEK BIR PAYLASILAN L.canvas() renderer'a
# atanarak (asagidaki _GEOJSON_STYLE_JS) render ediliyor -- Leaflet, ayni
# renderer'a atanmis TUM sekilleri TEK <canvas> elemaninda birlestiriyor,
# binlerce DOM node YERINE tek canvas + tek repaint.
#
# ONEMLI: bu iki isim (_GEOJSON_STYLE_JS / _ON_EACH_FEATURE_JS) asagidaki
# app_dash.layout icinde KULLANILIYOR -- Python modul seviyesinde yukaridan
# asagiya calistigi icin layout'tan ONCE tanimlanmis olmalari sart.

# GeoJSON Polygon feature'larini (ucak sekilleri) STIL'lendiren fonksiyon
# -- Leaflet'in GeoJSON "style" callback'i. Kose noktalari (rotasyon
# dahil) zaten Python'da hesaplanmis geliyor, burasi SADECE renk/opaklik
# atiyor ve TUM sekilleri ayni paylasilan canvas renderer'a bagliyor
# (window.__aircraftCanvasRenderer -- modul-seviyesi tekil nesne, her
# cagride yeniden OLUSTURULMUYOR, aksi halde her tick'te YENI bir canvas
# acilir/eskisi terk edilirdi).
_GEOJSON_STYLE_JS = assign("""
function(feature, context){
    if (!window.__aircraftCanvasRenderer) {
        window.__aircraftCanvasRenderer = L.canvas({padding: 0.5});
    }
    const p = feature.properties;
    const color = p.color || '#00b4d8';
    const opacity = (p.opacity === undefined || p.opacity === null) ? 1 : p.opacity;
    return {
        fillColor: color, fillOpacity: opacity,
        color: '#07070e', weight: 0.6, opacity: opacity,
        renderer: window.__aircraftCanvasRenderer,
    };
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
    if(!p.icao24){ return; }  // guvenlik icin birakildi, artik hep dolu (kumeleme kalkti)
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
            /* Koyu tema renk paleti -- asagida TEK TEK hex kodu tekrarlamak
               yerine (once boyleydi, ~40 tekrar) burada bir kez tanimlanip
               var(--...) ile kullaniliyor: hem tema degisikligi tek yerden
               yapilabiliyor hem de bir renk kodunun ANLAMI (orn. --dash-border)
               ham "#2a2a4a"den daha okunakli. */
            :root {
                --dash-bg: #161625;          /* panel/kutu zemin */
                --dash-border: #2a2a4a;      /* kenarlik */
                --dash-text: #c8d0e0;        /* birincil acik metin */
                --dash-text-strong: #ffffff; /* hover/vurgu metin */
                --dash-bg-hover: #22224a;    /* hover/secili zemin */
                --dash-accent: #00b4d8;      /* camgobegi vurgu */
                --dash-bg-page: #07070e;     /* en koyu, sayfa/harita zemin */
            }
            html, body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: var(--dash-bg-page);
            }
            /* Leaflet'in varsayilan tooltip'i beyaz kutu/siyah yazi --
               koyu temaya uydurmak icin gecersiz kiliyoruz. */
            .leaflet-tooltip {
                background-color: var(--dash-bg) !important;
                border: 1px solid var(--dash-border) !important;
                color: var(--dash-text) !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
                padding: 8px 10px !important;
            }
            .leaflet-tooltip-top:before   { border-top-color: var(--dash-border) !important; }
            .leaflet-tooltip-bottom:before{ border-bottom-color: var(--dash-border) !important; }
            .leaflet-tooltip-left:before  { border-left-color: var(--dash-border) !important; }
            .leaflet-tooltip-right:before { border-right-color: var(--dash-border) !important; }

            /* Sol-ust +/- yakinlastirma butonlari -- Leaflet'in varsayilani
               beyaz kutu/siyah yazi, koyu temaya uydurmak icin gecersiz
               kiliyoruz (digerleriyle AYNI renk paleti). */
            .leaflet-control-zoom {
                border: 1px solid var(--dash-border) !important;
            }
            .leaflet-control-zoom-in,
            .leaflet-control-zoom-out {
                background-color: var(--dash-bg) !important;
                color: var(--dash-text) !important;
                border-color: var(--dash-border) !important;
            }
            .leaflet-control-zoom-in:hover,
            .leaflet-control-zoom-out:hover {
                background-color: var(--dash-bg-hover) !important;
                color: var(--dash-text-strong) !important;
            }

            /* dcc.Dropdown -- Dash 4.x KENDI bilesenini kullaniyor (Radix UI
               tabanli, sinif isimleri dash-dropdown-*, RangeSlider'daki AYNI
               surum degisikligi -- bkz. .altitude-slider yorumu asagida).
               ONEMLI: acilan panel (.dash-dropdown-content) PORTAL'a
               (document.body'ye) render ediliyor -- .dark-dropdown'in
               ALTINDA/icinde DEGIL, o yuzden panel kurallari .dark-dropdown
               ile KAPSANMIYOR, GLOBAL uygulaniyor (projede zaten baska/acik
               temali bir dropdown yok, hepsi koyu tema kullaniyor). ESKI
               .Select-x / .VirtualizedSelectOption kurallari (react-select
               dönemınden kalma) gercek DOM'da hic eslesmiyordu, kaldirildi.
               Gercek sinif adlari, calisan container icindeki
               dash/dcc/async-dropdown.js dosyasindan dogrulandi. */
            .dark-dropdown.dash-dropdown {
                background-color: var(--dash-bg) !important;
                border: 1px solid var(--dash-border) !important;
                border-radius: 6px !important;
                color: var(--dash-text) !important;
            }
            .dark-dropdown .dash-dropdown-value,
            .dark-dropdown .dash-dropdown-placeholder,
            .dark-dropdown .dash-dropdown-trigger-icon {
                color: var(--dash-text) !important;
            }
            /* ONEMLI (kullanici geri bildirimi -- "firmalari kaydirdigimizda
               search'un altinda kaliyorlar"): panelin KENDISI (.dash-dropdown-content,
               inline max-height'i olan disaridaki kutu) tek parca olarak
               kayiyordu -- search/actions/liste HEPSI birlikte scroll
               oluyordu. flex column + overflow:hidden ile panelin kendisi
               ARTIK KAYMIYOR; search-container ve actions flex-shrink:0 ile
               SABIT kaliyor, SADECE .dash-dropdown-options (liste) kendi
               ic scroll'unu aliyor (overflow-y:auto + flex:1). */
            .dash-dropdown-content {
                background-color: var(--dash-bg) !important;
                border: 1px solid var(--dash-border) !important;
                border-radius: 6px !important;
                z-index: 2000 !important;
                box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
                display: flex !important;
                flex-direction: column !important;
                overflow: hidden !important;
            }
            .dash-dropdown-search-container {
                background-color: var(--dash-bg) !important;
                border-bottom: 1px solid var(--dash-border) !important;
                flex-shrink: 0 !important;
            }
            /* ONEMLI (kullanici geri bildirimi -- "kutunun kendisi hala
               beyaz"): arama <input type="search"> Chrome'un varsayilan
               beyaz kutu gorunumunu KENDI UA stiliyle getiriyor,
               background-color:transparent tek basina bunu SILMIYOR --
               -webkit-appearance:none ile varsayilan gorunum tamamen
               kaldirilip KENDI koyu arka planimiz veriliyor. */
            .dash-dropdown-search {
                background-color: var(--dash-bg) !important;
                color: var(--dash-text) !important;
                border: none !important;
                -webkit-appearance: none !important;
                appearance: none !important;
                box-shadow: none !important;
            }
            .dash-dropdown-search::placeholder {
                color: #666 !important;
            }
            .dash-dropdown-search-icon {
                color: #888 !important;
            }
            .dash-dropdown-actions {
                background-color: var(--dash-bg) !important;
                border-bottom: 1px solid var(--dash-border) !important;
                flex-shrink: 0 !important;
            }
            .dash-dropdown-action-button {
                background: transparent !important;
                color: var(--dash-accent) !important;
            }
            .dash-dropdown-options {
                overflow-y: auto !important;
                flex: 1 1 auto !important;
                min-height: 0 !important;
            }
            .dash-dropdown-option {
                background-color: var(--dash-bg) !important;
                color: var(--dash-text) !important;
            }
            .dash-dropdown-option:hover {
                background-color: var(--dash-bg-hover) !important;
                color: var(--dash-text-strong) !important;
            }
            .dash-dropdown-option[aria-selected="true"] {
                background-color: #0d3a45 !important;
                color: var(--dash-accent) !important;
            }
            .dash-options-list-option-checkbox {
                accent-color: var(--dash-accent);
            }
            /* ONEMLI (kullanici geri bildirimi -- "yazilarin altinda koyu
               mavi gibi bir sey kalmis"): secili deger(ler) .dash-dropdown-
               value-item span'ina sariliyor; buna ayrica bir arka plan
               (--dash-bg-hover) verilince disaridaki kutunun (--dash-bg)
               icinde "kutu icinde kutu" gorunumu olusuyordu -- hem tekli
               (saat dilimi) hem coklu (firma) secimde. Arka plani seffaf
               yapip metnin dogrudan disaridaki koyu zemin uzerinde
               durmasini sagliyoruz. */
            .dash-dropdown-value-item {
                background-color: transparent !important;
                color: var(--dash-text) !important;
            }
            /* Irtifa filtre kaydiricisi -- Dash 4.x KENDI slider bilesenini
               kullaniyor (Radix UI tabanli), sinif isimleri dash-slider-*
               (bkz. site-packages/dash/dcc/dash_core_components.js). */
            .altitude-slider .dash-range-slider-min-input,
            .altitude-slider .dash-range-slider-max-input {
                display: none !important;
            }
            .altitude-slider .dash-slider-container {
                gap: 0 !important;
            }
            .altitude-slider .dash-slider-root {
                padding: 0 !important;
                height: 12px !important;
            }
            .altitude-slider .dash-slider-track {
                background-color: rgba(7, 7, 14, 0.72) !important;
                height: 12px !important;
                border-radius: 3px !important;
            }
            .altitude-slider .dash-slider-range {
                background-color: transparent !important;
            }
            .altitude-slider .dash-slider-mark,
            .altitude-slider .dash-slider-dot {
                display: none !important;
            }
            .altitude-slider .dash-slider-tooltip {
                display: none !important;
            }
            .altitude-slider .dash-slider-thumb {
                width: 15px !important;
                height: 15px !important;
                background-color: var(--dash-text-strong) !important;
                border: 2px solid var(--dash-bg-page) !important;
                box-shadow: 0 0 4px rgba(0, 0, 0, 0.7) !important;
            }
            .altitude-slider .dash-slider-thumb:hover,
            .altitude-slider .dash-slider-thumb:focus {
                border-color: var(--dash-accent) !important;
                box-shadow: 0 0 0 4px rgba(0, 180, 216, 0.3) !important;
                transform: scale(1.125);
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
# "Onceki Ucuslar" listesindeki her satir icin -- secili olan (haritada
# su an gosterilen segment) FILTER_BTN'lerle AYNI aktif/pasif desenini
# kullanir (bkz. style_flight_segment_buttons).
FLIGHT_SEGMENT_BTN_STYLE = {
    "width": "100%", "textAlign": "left", "padding": "6px 8px",
    "borderRadius": "5px", "border": "1px solid #2a2a4a",
    "backgroundColor": "#161625", "cursor": "pointer", "marginBottom": "4px",
}
FLIGHT_SEGMENT_BTN_ACTIVE_STYLE = {**FLIGHT_SEGMENT_BTN_STYLE,
    "border": "1px solid #00b4d8", "backgroundColor": "#0d2830"}
HISTORY_PANEL_BASE = {
    "position": "absolute", "bottom": 0, "right": 0,
    "width": "540px", "height": "300px",
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
STATS_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "60px",
    "width": "420px", "height": "260px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #2a2a4a",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
EMERGENCY_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "108px",
    "width": "300px", "maxHeight": "320px", "overflowY": "auto",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #e63946",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
EMERGENCY_ROW_STYLE = {
    "width": "100%", "textAlign": "left", "padding": "8px 10px",
    "borderRadius": "5px", "border": "1px solid #e63946",
    "backgroundColor": "#2a0f13", "cursor": "pointer", "marginBottom": "6px",
    "color": "#fff",
}
REPLAY_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "156px",
    "width": "300px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #2a2a4a",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
REPLAY_COLOR = "#f7b731"  # canli haritadaki (#00b4d8) mavi/askeri yesilden AYRI -- kullanici
                          # bunun bir "gecmis" gorunum oldugunu tek bakista ayirt etsin

DEFAULT_TIMEZONE = 3  # UTC+3, Turkiye -- dropdown "value" olarak int kullanilir


def _build_timezone_options(lang):
    """update_timezone_options callback'iyle AYNI liste -- ilk render'da da
    kullanilir (bkz. asagidaki dcc.Dropdown). ONEMLI: dropdown ilk acilista
    options=[] ile baslarsa, value=DEFAULT_TIMEZONE hicbir secenekle
    eslesmedigi icin bilesen kendi icinde value'yu null'a sifirliyordu --
    saat dilimi kutusu kullanici elle bir sey secene kadar BOS gorunuyordu.
    Liste ilk render'da dolu geldigi icin bu yaris artik olusmuyor."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    options = []
    for h in range(-12, 15):
        label = f"UTC{h:+d}"
        if h == DEFAULT_TIMEZONE:
            label += t["tz_default_suffix"]
        options.append({"label": label, "value": h})
    return options

# Haritadaki rota izi (polyline) artik sabit saat degil, "son ucus" --
# gercek ucus/blok verisi olmadigi icin bir HEURISTIK: ardisik iki konum
# noktasi arasinda bu esikten (dakika) BUYUK bir bosluk, "onceki ucus
# bitti, yenisi basladi" sayilir (bkz. update_flight_path). Normal ucus
# icinde ardisik nokta araligi (~15-90sn, producer cycle suresine bagli)
# bu esigin COK altinda kalir.
FLIGHT_GAP_THRESHOLD_MIN = 20

# Gecmis grafigi tarih araligi secicisi icin saat secenekleri (0-23) --
# sabit, dile/tarihe bagli degil, gun secenekleri gibi callback'te
# yeniden hesaplanmasina gerek yok.
HISTORY_HOUR_OPTIONS = [{"label": f"{h:02d}", "value": h} for h in range(24)]

# Gecmis grafigi "Hesapla" butonu -- tarih araligi dropdown'lari artik
# Input DEGIL State (bkz. update_history) -- secim yapmak TEK BASINA
# grafigi guncellemiyor, kullanici bu butona basana kadar bekliyor.
# Boylece 4 dropdown'u tek tek secerken (gun/saat x baslangic/bitis) her
# ara adimda gereksiz sorgu atilmiyor, sadece kullanici hazir oldugunda.
HISTORY_CALC_BTN_STYLE = {
    "padding": "6px 12px", "borderRadius": "5px", "border": "1px solid #00b4d8",
    "backgroundColor": "#00b4d8", "color": "#07070e", "fontSize": "11px",
    "fontWeight": "700", "cursor": "pointer", "flexShrink": 0, "whiteSpace": "nowrap",
}
HISTORY_DOWNLOAD_BTN_STYLE = {
    "padding": "6px 10px", "borderRadius": "5px", "border": "1px solid #2a2a4a",
    "backgroundColor": "#161625", "color": "#c8d0e0", "fontSize": "13px",
    "cursor": "pointer", "flexShrink": 0,
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

# ONEMLI (kullanici geri bildirimi -- bkz. proje sohbet gecmisi: "900ler
# hala 2000den 4000e geçerken gidiyor"): duraklar ESKIDEN feet cinsindeydi
# (adsb.lol/tar1090 esintili), ama ucak bilgi paneli irtifayi HER YERDE
# METRE gosteriyor -- iki farkli birimin ayni sayilarmis gibi (900 ile
# 2000/4000) karsilastirilmasi kafa karistiriyordu (900m ~= 2953ft, yani
# GERCEKTEN 2000ft-4000ft arasinda kaliyordu -- filtre matematigi dogruydu,
# birim gosterimi tutarsizdi). COZUM: TUM sistem (duraklar, kaydirici,
# lejant, renk esikleri) artik METRE uzerinden calisiyor, ft<->m cevirisi
# TAMAMEN kaldirildi -- gosterilen sayi ile ic hesaplama HER ZAMAN ayni
# birimde. Duraklarin SIKLIGI yine dusuk irtifada YUKSEK, yuksek irtifada
# DUSUK (alcak irtifadaki -- havaalani yakini, trafik yogun -- renk
# degisimini daha hassas gostermek icin).
ALTITUDE_COLOR_STOPS = [
    (0,     "#e8551f"),
    (200,   "#ed7a1f"),
    (500,   "#f0971f"),
    (1000,  "#e8c81f"),
    (2000,  "#c8d820"),
    (3000,  "#78c840"),
    (4000,  "#38b868"),
    (6000,  "#20a8a0"),
    (8000,  "#2078c8"),
    (10000, "#3050d8"),
    (12000, "#9040c8"),  # 12000m+ -- legend'de "12 000+" olarak sabit
]
_ALT_STOP_M = [s[0] for s in ALTITUDE_COLOR_STOPS]
_ALT_STOP_RGB = [tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5)) for _, hexc in ALTITUDE_COLOR_STOPS]


def _altitude_to_color(alt_m):
    """Metre cinsinden irtifayi ALTITUDE_COLOR_STOPS'a (o da metre) gore
    bir hex renge cevirir. None/eksikse (kaynak saglamiyorsa) notr bir gri
    doner -- sahte bir irtifa varsaymiyoruz."""
    if alt_m is None:
        return "#888888"
    alt_m = max(0.0, alt_m)
    if alt_m >= _ALT_STOP_M[-1]:
        r, g, b = _ALT_STOP_RGB[-1]
        return f"#{r:02x}{g:02x}{b:02x}"
    i = 0
    while i < len(_ALT_STOP_M) - 1 and alt_m > _ALT_STOP_M[i + 1]:
        i += 1
    lo_m, hi_m = _ALT_STOP_M[i], _ALT_STOP_M[i + 1]
    lo_rgb, hi_rgb = _ALT_STOP_RGB[i], _ALT_STOP_RGB[i + 1]
    frac = (alt_m - lo_m) / (hi_m - lo_m) if hi_m > lo_m else 0.0
    r = round(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * frac)
    g = round(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * frac)
    b = round(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def _format_altitude_tick(m):
    """Lejant altindaki tik etiketi -- kullanicinin verdigi legend
    goruntusundeki bicimle AYNI (bosluklu binlik ayirici, son durak
    '12 000+')."""
    if m >= _ALT_STOP_M[-1]:
        return f"{m:,}".replace(",", " ") + "+"
    return f"{m:,}".replace(",", " ")


# ONEMLI: lejant, ALTITUDE_COLOR_STOPS'tan TUReTILIYOR (elle ayri renk
# listesi yazmiyoruz) -- boylece _altitude_to_color() ile lejant HICBIR
# ZAMAN birbirinden SAPAMAZ, tek gercek kaynak burasi. CSS coklu-durak
# linear-gradient, her durak ARALIGINA ESIT genislik veriyor (kullanicinin
# verdigi goruntudeki gibi -- durak degerleri ESIT ARALIKLI DEGIL ama
# CUBUKTAKI GENISLIKLERI esit, alcak irtifadaki yogun duraklara daha
# fazla gorsel yer ayirmak icin).
_ALT_LEGEND_N = len(ALTITUDE_COLOR_STOPS) - 1
ALTITUDE_LEGEND_GRADIENT = "linear-gradient(to right, " + ", ".join(
    f"{hexc} {i * 100 / _ALT_LEGEND_N:.4f}%" for i, (_, hexc) in enumerate(ALTITUDE_COLOR_STOPS)
) + ")"


def _slider_value_to_altitude_m(v):
    """Lejant uzerindeki irtifa filtre kaydiricisinin (0.._ALT_LEGEND_N,
    kesirli -- her tam sayi bir ALTITUDE_COLOR_STOPS durağina karsilik
    gelir) degerini irtifaya (metre) cevirir. _altitude_to_color'daki AYNI
    esit-segment-genisligi + durak-arasi-dogrusal mantigini kullanir --
    boylece daire, gradyandaki (ve alttaki tik etiketlerindeki) irtifayla
    HER ZAMAN tutarli bir noktada durur."""
    v = max(0.0, min(v, _ALT_LEGEND_N))
    i = int(v)
    if i >= _ALT_LEGEND_N:
        return _ALT_STOP_M[-1]
    frac = v - i
    lo, hi = _ALT_STOP_M[i], _ALT_STOP_M[i + 1]
    return lo + (hi - lo) * frac

# ONEMLI: yerde/havada askeri-sivil ekseninden BAGIMSIZ, ayri bir boyut --
# bir ucak ayni anda hem askeri hem yerde olabilir. Bu yuzden GROUND_COLOR
# rengi DEGISTIRMEZ (oncelik hala alarm > askeri > sivil), sadece tooltip'e
# "Yerde" etiketi ekler (bkz. update_map) -- ayri bir renk yerine filtre
# butonunun kendisi (asagida) tarafsiz bir kum/toprak tonu kullanir.
GROUND_COLOR = "#e0a458"

# Firma (havayolu) filtresi -- ICAO 3 harfli cagri kodu on-eki -> firma adi.
# KUCUK, ELLE KURATORLU bir liste (bu projede sik gorulen/taninan ~40
# havayolu) -- kapsamli bir ICAO veritabani DEGIL, kullanici tercihi boyle
# (bkz. proje sohbet gecmisi). Listede OLMAYAN cagri kodlari (askeri, genel
# havacilik, taninmayan/bolgesel havayollari) filtre BOSKEN her zaman
# gorunur kalir -- bir veya daha fazla firma SECILINCE haritada SADECE o
# firma(lar) gosterilir (bkz. asagidaki clientside_callback, filtreleme
# TARAYICIDA/JS'te yapiliyor -- update_map'e YENI bir Input EKLENMEDI,
# boylece bu oturumda iki kez duzeltilen "pahali sunucu isteği + yarış
# durumu" hatasi SINIFI bastan hic olusmuyor -- callsign zaten cekilen
# veride mevcut, ekstra ag gidis-donusune gerek yok).
AIRLINE_PREFIXES = {
    "THY": "Turkish Airlines", "PGT": "Pegasus", "SXS": "SunExpress",
    "RYR": "Ryanair", "EZY": "easyJet", "WZZ": "Wizz Air", "VLG": "Vueling",
    "DLH": "Lufthansa", "AFR": "Air France", "BAW": "British Airways",
    "KLM": "KLM", "IBE": "Iberia", "AZA": "ITA Airways", "SWR": "Swiss",
    "AUA": "Austrian Airlines", "LOT": "LOT Polish Airlines",
    "TAP": "TAP Air Portugal", "SAS": "SAS", "FIN": "Finnair",
    "ELY": "El Al", "AFL": "Aeroflot", "BEL": "Brussels Airlines",
    "UAE": "Emirates", "QTR": "Qatar Airways", "ETD": "Etihad Airways",
    "SVA": "Saudia", "MEA": "Middle East Airlines",
    "DAL": "Delta Air Lines", "AAL": "American Airlines",
    "UAL": "United Airlines", "ACA": "Air Canada", "JAL": "Japan Airlines",
    "ANA": "All Nippon Airways", "CPA": "Cathay Pacific",
    "SIA": "Singapore Airlines", "QFA": "Qantas", "KAL": "Korean Air",
    "THA": "Thai Airways", "GIA": "Garuda Indonesia",
    "CES": "China Eastern", "CSN": "China Southern", "CCA": "Air China",
    "ETH": "Ethiopian Airlines", "MSR": "EgyptAir",
    "RJA": "Royal Jordanian",
}

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
FILTER_BTN_GROUND_ACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": GROUND_COLOR, "color": "#07070e",
    "border": f"1px solid {GROUND_COLOR}"}
FILTER_BTN_INACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": "#161625", "color": "#888", "border": "1px solid #2a2a4a"}

app_dash.layout = html.Div(id="app-root", style={
    "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
    "overflow": "hidden", "backgroundColor": "#07070e",
    "fontFamily": "sans-serif", "color": "#c8d0e0",
}, children=[

    dcc.Interval(id="tick", interval=15000, n_intervals=0),
    dcc.Store(id="aircraft-select", data=None),  # secili ucak (gorunmez state)
    # Python'un doldurdugu HAM nokta verisi (lat/lon/track) -- gercek
    # poligon geometrisi clientside_callback'te (JS, sayfa sonunda)
    # hesaplanip "aircraft-geojson"a yaziliyor, bkz. update_map yorumu.
    dcc.Store(id="aircraft-raw", data={"type": "FeatureCollection", "features": []}),
    # Acil durum listesi (squawk 7500/7600/7700 veya emergency alani dolu
    # olan ucaklar) -- update_map her tick'te dolduruyor, panel bunu okuyor
    # (ayri bir /api/flights istegi atmadan, bkz. update_map yorumu).
    dcc.Store(id="emergency-alerts-data", data=[]),
    # Hangi icao24'ler icin oto-odaklama (bkz. auto_focus_new_emergency)
    # zaten yapildigini hatirlar -- ayni acil durum icin TEKRAR tetiklenmesin.
    dcc.Store(id="emergency-seen", data=[]),
    # "Onceki Ucuslar" listesi -- HAM segment verisi (start/end/duration,
    # bkz. /api/flight_segments) burada tutuluyor, gorunen liste (butonlar)
    # bunun uzerinden index'le esleniyor. Secili segment (haritada su an
    # cizilen ucus) -- None ise update_flight_path VARSAYILANA (son ucus,
    # gap-tabanli) doner.
    dcc.Store(id="flight-segments-store", data=[]),
    dcc.Store(id="selected-flight-segment", data=None),

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
            # ONEMLI: eskiden dl.LayerGroup + N adet dl.DivMarker, sonra
            # dl.GeoJSON(cluster=True) + supercluster balonlari (bkz.
            # yukaridaki buyuk yorum blogu). GUNCEL: kumeleme YOK -- her
            # ucak GERCEK bir Polygon (Python'da onceden dondurulmus ok/dart
            # sekli), TEK paylasilan canvas renderer'a (_GEOJSON_STYLE_JS)
            # atanarak render ediliyor. update_map callback'i Python
            # component agaci degil, duz GeoJSON sozlugu donduruyor
            # (Output("aircraft-geojson", "data")).
            dl.GeoJSON(
                id="aircraft-geojson",
                data={"type": "FeatureCollection", "features": []},
                style=_GEOJSON_STYLE_JS,
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
        "display": "flex", "flexDirection": "row", "gap": "6px",
        "alignItems": "flex-start",
        "zIndex": 700,
    }, children=[
        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "6px"}, children=[
            html.Button("Sivil", id="filter-civil-btn", n_clicks=0,
                       style=FILTER_BTN_CIVIL_ACTIVE_STYLE),
            html.Button("Askeri", id="filter-military-btn", n_clicks=0,
                       style=FILTER_BTN_MILITARY_ACTIVE_STYLE),
            # ONEMLI: varsayilan ACIK (kullanici karariyla) -- eskiden
            # yerdeki ucaklar varsayilan GIZLIYDI (bkz. show-ground Store
            # yorumu), ama kullanici bunu kafa karistirici buldu ("aktif
            # uçuş ile gösterilen fark neden" sorusu -- bkz. proje sohbet
            # gecmisi) ve varsayilani ACIK'a cevirmemizi istedi.
            html.Button("Yerde", id="filter-ground-btn", n_clicks=0,
                       style=FILTER_BTN_GROUND_ACTIVE_STYLE),
        ]),
        # Firma (havayolu) filtresi -- AIRLINE_PREFIXES'ten turetilen,
        # aranabilir/coklu-secim dropdown. Filtreleme TARAYICIDA yapiliyor
        # (bkz. AIRLINE_PREFIXES yorumu ve asagidaki clientside_callback) --
        # bu yuzden update_map'e YENI bir Input EKLENMEDI, secim degistiginde
        # sunucuya HIC istek gitmiyor. ONEMLI: Sivil/Askeri/Yerde sutununun
        # SAGINA (kullanici istegi) -- ayni satirda, ustten hizali.
        dcc.Dropdown(
            id="airline-filter-dropdown",
            options=[{"label": name, "value": code}
                     for code, name in sorted(AIRLINE_PREFIXES.items(), key=lambda kv: kv[1])],
            value=[], multi=True, placeholder="Firma...",
            className="dark-dropdown airline-filter-dropdown",
            style={"width": "190px", "fontSize": "11px"},
        ),
    ]),
    dcc.Store(id="show-civil", data=True),
    dcc.Store(id="show-military", data=True),
    dcc.Store(id="show-ground", data=True),

    # -------------------------------------- Irtifa renk lejandi (overlay) --
    # Durum cubugunun (ustte, ortada) alt-orta esligi -- sol/sag panellerin
    # (LEFT_PANEL_BASE/HISTORY_PANEL_BASE, ikisi de kenarlara sabit) ARASINDA
    # kalan orta bosluga sigacak sekilde dar tutuldu.
    html.Div(id="altitude-legend", style={
        "position": "absolute", "bottom": "12px", "left": "50%",
        "transform": "translateX(-50%)",
        "width": "460px",
        "backgroundColor": "rgba(15,15,25,0.85)",
        "border": "1px solid #2a2a4a", "borderRadius": "8px",
        "padding": "6px 14px 10px", "zIndex": 500,
        # ONEMLI: artik pointerEvents:none YOK -- kaydiricilar (asagida)
        # gercek bir kontrol, tiklanabilir/surunebilir olmasi gerekiyor.
    }, children=[
        html.Div("İRTİFA (m)", id="altitude-legend-title", style={
            "fontSize": "10px", "color": "#888", "marginBottom": "4px",
            "letterSpacing": "0.5px",
        }),
        # ONEMLI: gradyan CUBUGU (asagida, GERCEK renk skalasi) ile ONUN
        # TAM UZERINE bindirilen dcc.RangeSlider AYNI position:relative
        # kutuda. Slider'in kendi track/range'i CSS ile SEFFAF yapiliyor
        # (bkz. index_string ".altitude-slider" kurallari, Dash 4.x'in
        # dash-slider-* sinif adlarina gore). Sadece IKI daire tutamac
        # gorunur kaliyor.
        html.Div(style={"position": "relative", "height": "12px"}, children=[
            html.Div(style={
                "height": "12px", "borderRadius": "3px",
                "background": ALTITUDE_LEGEND_GRADIENT,
                "border": "1px solid rgba(255,255,255,0.15)",
            }),
            html.Div(style={"position": "absolute", "top": "0", "left": "0", "right": "0",
                            "height": "12px"},
                     children=[
                dcc.RangeSlider(
                    id="altitude-filter-slider",
                    # ONEMLI (kullanici geri bildirimi -- "ayarladigin
                    # sayilar barda gozukenle uyusmuyor"): step=0.01 iken
                    # tutamac GORSEL olarak herhangi bir ARA pozisyonda
                    # durabiliyordu (orn. "1000" ile "2000" arasinda),
                    # AMA sunucuya giden deger clientside_callback'te EN
                    # YAKIN duraga yuvarlaniyordu -- goruneni (tutamacin
                    # GERCEK pikseldeki yeri) ile UYGULANANI (yuvarlanmis
                    # durak) BIRBIRINDEN FARKLI olabiliyordu, bu da tam
                    # bu kafa karisikligina yol aciyordu. step=1 ile
                    # Radix'in KENDISI tutamaci SADECE 11 durağa (tam
                    # sayi degerlere) KILITLIYOR -- ara bir pozisyonda
                    # durmasi ARTIK MUMKUN DEGIL, goruneni ile uygulanani
                    # HER ZAMAN ayni.
                    min=0, max=_ALT_LEGEND_N, step=1,
                    value=[0, _ALT_LEGEND_N],
                    marks=None, allowCross=True,
                    updatemode="drag",
                    className="altitude-slider",
                ),
            ]),
        ]),
        # ONEMLI (gercek tarayici olcumuyle BULUNAN bir hata -- kullanici
        # geri bildirimi: "1000-40000 secince 300 falan gozukuyor"):
        # "justifyContent: space-between" bir flex satirda, FARKLI
        # genislikte metinleri (orn. "0" vs "40 000+") ARALARINDAKI
        # BOSLUGU esitler, MERKEZLERINI DEGIL -- "1 000" etiketi olmasi
        # gereken %20 yerine olcumle %16'da cikiyordu. Kullanici gozle
        # bir etikete hizalayinca, kaydiricinin/gradyanin GERCEK (dogru)
        # yuzdesiyle etiketin (yanlis) yuzdesi UYUSMUYORDU. Duzeltme:
        # her etiketi, kaydirici/gradyan ile AYNI dogrusal 0%-100%
        # olcegine gore MUTLAK konumlandiriyoruz (position:absolute,
        # left:i*10%, translateX(-50%)) -- boylece UCUNCU bir hizalama
        # sistemi degil, TEK bir dogru kaynak var.
        html.Div(style={"position": "relative", "height": "12px", "marginTop": "3px"},
                 children=[
            html.Span(_format_altitude_tick(ft), style={
                "position": "absolute", "left": f"{i * 100 / _ALT_LEGEND_N:.4f}%",
                "transform": "translateX(-50%)",
                "fontSize": "9px", "color": "#c8d0e0", "whiteSpace": "nowrap",
            })
            for i, (ft, _) in enumerate(ALTITUDE_COLOR_STOPS)
        ]),
    ]),
    # ONEMLI (gercek kullanicidan gelen "1000'e gelince 500 hala orada,
    # 2000'e gelince 500 gidiyor ama 1200 orda kaliyor" geri bildirimi
    # uzerine bulundu -- YARIS DURUMU): updatemode="drag" surukleme
    # sirasinda ONLARCA ARA (kesirli) deger gonderiyor, HER biri
    # update_map'i (binlerce ucagi yeniden ceken/filtreleyen PAHALI bir
    # islem) tetikliyordu -- yanitlar GONDERILDIKLERI sirada DEGIL
    # TAMAMLANDIKLARI sirada donuyor, o yuzden GEC gelen ESKI bir yanit
    # YENI olani EZEBILIYORDU. Onceki cozum bir "Uygula" butonuydu;
    # kullanici bunun yerine "sadece 11 duraga (0/500/1000/.../40000+)
    # GECISTE tetiklensin, buton olmadan" istedi. Asagidaki clientside_callback
    # (JS, sunucuya HIC gitmeden) HAM slider degerini EN YAKIN duraga
    # yuvarlar ve SADECE bir onceki yuvarlanmis degerden FARKLIYSA
    # "altitude-filter-snapped" Store'unu gunceller -- ayni durak icindeki
    # onlarca ara deger boylece sunucuya HIC gitmiyor, sadece GERCEKTEN
    # yeni bir duraga geçince (en fazla 11 kez) bir istek atiliyor --
    # yaris penceresi ~1000 olasi tetikleyiciden 11'e dusuyor.
    dcc.Store(id="altitude-filter-snapped", data=[0, _ALT_LEGEND_N]),
    # Filtrelenmis irtifa araligi (metre) -- update_map bunu okuyup
    # araligin DISINDAKI ucaklari haritadan gizliyor (bkz. _passes_filter).
    # Varsayilan [-1_000_000, 1_000_000] = "filtre yok" (bkz.
    # update_altitude_filter_range ust VE ALT sinir yorumu -- ikisi de
    # gercek metre degeri DEGIL, sinirsiz demek).
    dcc.Store(id="altitude-filter-range", data=[-1_000_000, 1_000_000]),

    # ------------------------------------------- Durum cubugu (overlay) --
    # ONEMLI: UC AYRI parcadan olusuyor, DOM SIRASI = GORSEL SIRA (kullanici
    # istegi -- "gösterilen'le aktif uçuş yan yana olsun, alarmı en sağa
    # al"): "status-main" (ts + TOPLAM ucak) ve "status-alarm" (alarm sayisi,
    # EN SAGDA) update_map'te (Python, sunucu) hesaplaniyor; aradaki
    # "status-shown" (filtreler sonrasi KALAN ucak sayisi) ise
    # clientside_callback'te (JS, sayfa sonu) hesaplaniyor -- firma filtresi
    # TAMAMEN tarayicida uygulaniyor (bkz. AIRLINE_PREFIXES yorumu), sunucu
    # o filtreden HABERSIZ, o yuzden "kalan" sayiyi SADECE JS taraf dogru
    # hesaplayabilir (sivil/askeri/yerde/irtifa sunucuda elendikten SONRA,
    # firma da JS'te elendikten sonraki nihai sourceFeatures.length).
    html.Div(id="status", style={
        "position": "absolute", "top": "12px", "left": "50%",
        "transform": "translateX(-50%)",
        "backgroundColor": "rgba(15,15,25,0.85)",
        "padding": "6px 16px", "borderRadius": "20px",
        "fontSize": "13px", "color": "#c8d0e0", "zIndex": 500,
        "pointerEvents": "none",  # altindaki haritayi engellemesin
    }, children=[
        html.Span(id="status-main"),
        html.Span(id="status-shown", style={"color": "#00b4d8", "marginLeft": "8px"}),
        html.Span(id="status-alarm"),
    ]),

    # ----------------------------- Cagri kodu arama cubugu (durum cubugunun ALTI) --
    html.Div(style={
        "position": "absolute", "top": "50px", "left": "50%",
        "transform": "translateX(-50%)",
        "display": "flex", "alignItems": "center", "gap": "6px",
        "backgroundColor": "rgba(15,15,25,0.85)",
        "padding": "6px 8px 6px 12px", "borderRadius": "20px", "zIndex": 500,
    }, children=[
        dcc.Input(id="callsign-search-input", type="text",
                 placeholder="Çağrı kodu ara...", debounce=True, style={
            "width": "150px", "padding": "5px 10px", "borderRadius": "14px",
            "border": "1px solid #2a2a4a", "backgroundColor": "#161625",
            "color": "#c8d0e0", "fontSize": "12px", "boxSizing": "border-box",
            "outline": "none",
        }),
        html.Button("🔍", id="callsign-search-btn", n_clicks=0, style={
            "width": "28px", "height": "28px", "borderRadius": "50%",
            "border": "1px solid #2a2a4a", "backgroundColor": "#161625",
            "color": "#c8d0e0", "fontSize": "13px", "cursor": "pointer",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "padding": 0, "flexShrink": 0,
        }),
    ]),
    html.Div(id="callsign-search-feedback", style={
        "position": "absolute", "top": "90px", "left": "50%",
        "transform": "translateX(-50%)",
        "fontSize": "11px", "color": "#e63946", "zIndex": 500,
        "pointerEvents": "none",
    }),

    # ------------------------------------------- Ayarlar butonu (overlay) --
    html.Button("⚙", id="settings-btn", n_clicks=0, style={
        "position": "absolute", "top": "12px", "right": "12px",
        "width": "40px", "height": "40px", "borderRadius": "50%",
        "backgroundColor": "#000000", "border": "1px solid #2a2a4a",
        "color": "#c8d0e0", "fontSize": "18px", "cursor": "pointer", "zIndex": 900,
    }),

    # -------------------------------------- Istatistik butonu (overlay) --
    # Ayarlar diliginin HEMEN SOLUNDA (right: 60px = 12 + 40 + 8 bosluk) --
    # ayni yukseklik/boyut, tutarli gorunum icin ayni stil.
    html.Button("📊", id="stats-btn", n_clicks=0, style={
        "position": "absolute", "top": "12px", "right": "60px",
        "width": "40px", "height": "40px", "borderRadius": "50%",
        "backgroundColor": "#000000", "border": "1px solid #2a2a4a",
        "color": "#c8d0e0", "fontSize": "18px", "cursor": "pointer", "zIndex": 900,
    }),

    # -------------------------------------- Istatistik paneli (overlay) --
    # Simdilik tek grafik: saat basina benzersiz ucak sayisi (son 24 saat)
    # -- /api/traffic_stats zaten mevcuttu (eskiden arayuzden kaldirilmisti,
    # bkz. FULL_PROJECT_HANDOFF.md Bolum 12), burada geri kullaniliyor.
    html.Div(id="stats-panel", style={**STATS_PANEL_BASE, "display": "none"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "10px"}, children=[
            html.H4("İstatistikler", id="stats-title",
                    style={"margin": 0, "fontSize": "14px", "color": "#00b4d8"}),
            html.Button("×", id="close-stats-btn", n_clicks=0, style={
                "background": "none", "border": "none", "color": "#888",
                "fontSize": "20px", "cursor": "pointer", "lineHeight": "1",
                "padding": "0 4px",
            }),
        ]),
        dcc.Graph(id="stats-chart", style={"height": "200px"},
                  config={"displayModeBar": False}),
    ]),
    dcc.Store(id="stats-open", data=False),

    # ---------------------------------------- Acil durum butonu (overlay) --
    # Istatistik dugmesinin HEMEN SOLUNDA (right: 108px = 60 + 40 + 8).
    # emergency-alerts-data doluyken kirmizi kenarlik/rozet ile vurgulanir
    # (bkz. update_emergency_badge) -- panel kapaliyken bile fark edilsin diye.
    html.Button("🚨", id="emergency-btn", n_clicks=0, style={
        "position": "absolute", "top": "12px", "right": "108px",
        "width": "40px", "height": "40px", "borderRadius": "50%",
        "backgroundColor": "#000000", "border": "1px solid #2a2a4a",
        "color": "#c8d0e0", "fontSize": "18px", "cursor": "pointer", "zIndex": 900,
    }),

    # ---------------------------------------- Acil durum paneli (overlay) --
    # /api/alerts (ML ekibinin ileride dolduracagi, henuz BOS anomali
    # kanalindan) TAMAMEN AYRI -- bu, ucagin KENDI yaydigi ADS-B acil
    # durum sinyalini (squawk 7500/7600/7700 / emergency alani) gosterir,
    # veri bugun bile mevcut. Kaynak: emergency-alerts-data (update_map
    # her tick'te dolduruyor, bkz. o callback'teki yorum).
    html.Div(id="emergency-panel", style={**EMERGENCY_PANEL_BASE, "display": "none"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "10px"}, children=[
            html.H4("Acil Durumlar", id="emergency-title",
                    style={"margin": 0, "fontSize": "14px", "color": "#e63946"}),
            html.Button("×", id="close-emergency-btn", n_clicks=0, style={
                "background": "none", "border": "none", "color": "#888",
                "fontSize": "20px", "cursor": "pointer", "lineHeight": "1",
                "padding": "0 4px",
            }),
        ]),
        html.Div(id="emergency-panel-list"),
    ]),
    dcc.Store(id="emergency-open", data=False),

    # -------------------------------------- Tekrar oynatma butonu (overlay) --
    # Acil durum dugmesinin HEMEN SOLUNDA (right: 156px = 108 + 40 + 8).
    html.Button("🎬", id="replay-btn", n_clicks=0, style={
        "position": "absolute", "top": "12px", "right": "156px",
        "width": "40px", "height": "40px", "borderRadius": "50%",
        "backgroundColor": "#000000", "border": "1px solid #2a2a4a",
        "color": "#c8d0e0", "fontSize": "18px", "cursor": "pointer", "zIndex": 900,
    }),

    # -------------------------------------- Tekrar oynatma paneli (overlay) --
    # InfluxDB'deki gecmis veriyi (7 gune kadar) secilen bir zaman
    # araliginda haritada "oynatiyor" -- update_map ile AYNI GeoJSON/canvas
    # render hattini kullanir (bkz. render_replay_frame), sadece veri
    # kaynagi canli tick yerine onceden yuklenmis "frames" listesi olur.
    # replay-mode acikken update_map kendi cizimini YAPMIYOR (bkz. o
    # callback'teki kontrol) -- ikisi ayni Store'a (aircraft-raw) yazdigi
    # icin CARPISMASINLAR diye.
    html.Div(id="replay-panel", style={**REPLAY_PANEL_BASE, "display": "none"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "10px"}, children=[
            html.H4("Tekrar Oynatma", id="replay-title",
                    style={"margin": 0, "fontSize": "14px", "color": REPLAY_COLOR}),
            html.Button("×", id="close-replay-btn", n_clicks=0, style={
                "background": "none", "border": "none", "color": "#888",
                "fontSize": "20px", "cursor": "pointer", "lineHeight": "1",
                "padding": "0 4px",
            }),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px",
                        "marginBottom": "6px"}, children=[
            html.Span("Başlangıç", id="replay-start-label",
                     style={"fontSize": "10px", "color": "#888", "width": "48px"}),
            dcc.Dropdown(id="replay-start-day", options=[], value=None,
                        placeholder="Gün", clearable=True, searchable=False,
                        className="dark-dropdown", style={"width": "112px", "fontSize": "11px"}),
            dcc.Dropdown(id="replay-start-hour", options=HISTORY_HOUR_OPTIONS, value=None,
                        placeholder="Saat", clearable=True, searchable=False,
                        className="dark-dropdown", style={"width": "88px", "fontSize": "11px"}),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px",
                        "marginBottom": "10px"}, children=[
            html.Span("Bitiş", id="replay-end-label",
                     style={"fontSize": "10px", "color": "#888", "width": "48px"}),
            dcc.Dropdown(id="replay-end-day", options=[], value=None,
                        placeholder="Gün", clearable=True, searchable=False,
                        className="dark-dropdown", style={"width": "112px", "fontSize": "11px"}),
            dcc.Dropdown(id="replay-end-hour", options=HISTORY_HOUR_OPTIONS, value=None,
                        placeholder="Saat", clearable=True, searchable=False,
                        className="dark-dropdown", style={"width": "88px", "fontSize": "11px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "6px", "marginBottom": "8px"}, children=[
            html.Button("Yükle", id="replay-load-btn", n_clicks=0,
                       style={**HISTORY_CALC_BTN_STYLE, "flex": "1",
                              "backgroundColor": REPLAY_COLOR, "border": f"1px solid {REPLAY_COLOR}"}),
            html.Button("▶", id="replay-play-btn", n_clicks=0,
                       style={**HISTORY_DOWNLOAD_BTN_STYLE, "width": "36px"}),
            # ONEMLI: hiz, tick ARALIGINI (sabit 2sn -- bkz. replay-tick
            # tanimi) DEGISTIRMIYOR, her tick'te KAC ADIM birden ilerlenecegini
            # degistiriyor (bkz. advance_replay_tick). Boylece render/InfluxDB
            # istegi HER ZAMAN ayni (guvenli) hizda kalir -- yuksek hizda
            # tekrar "çok yavaş" sorununa donmeyiz, sadece ekranda daha fazla
            # gercek zaman atlanir.
            dcc.Dropdown(
                id="replay-speed-dropdown",
                options=[{"label": "0.5×", "value": 0.5}, {"label": "1×", "value": 1},
                        {"label": "2×", "value": 2}, {"label": "4×", "value": 4},
                        {"label": "8×", "value": 8}],
                value=1, clearable=False, searchable=False,
                className="dark-dropdown", style={"width": "68px", "fontSize": "11px", "flexShrink": 0},
            ),
        ]),
        html.Div(id="replay-feedback", style={"fontSize": "12px", "color": "#c8d0e0",
                                              "marginBottom": "6px", "fontWeight": "600"}),
        html.Div(id="replay-step-label", style={"fontSize": "11px", "color": "#888",
                                                "marginBottom": "6px"}),
        html.Button("Canlıya Dön", id="replay-live-btn", n_clicks=0, style={
            "width": "100%", "padding": "6px 0", "borderRadius": "5px",
            "border": "1px solid #2a2a4a", "backgroundColor": "#161625",
            "color": "#888", "fontSize": "11px", "cursor": "pointer",
        }),
    ]),
    dcc.Store(id="replay-open", data=False),
    dcc.Store(id="replay-data", data={"steps": [], "frames": []}),
    dcc.Store(id="replay-index", data=0),
    dcc.Store(id="replay-playing", data=False),
    dcc.Store(id="replay-mode", data=False),
    dcc.Store(id="replay-speed", data=1),
    # Hiz 1'den kucukse (0.5x) bir adim BIRDEN FAZLA tick surer -- kalan
    # kesirli ilerlemeyi tick'ler arasi tasir (bkz. advance_replay_tick).
    dcc.Store(id="replay-progress", data=0.0),
    # ONEMLI (performans, "çok yavaş" geri bildirimi uzerine 1000ms'den
    # yukseltildi): her adimda hem InfluxDB sorgusu hem de ~7000 ucagin
    # tarayicida yeniden geometrisi hesaplaniyor (bkz. clientside_callback,
    # sayfa sonu) -- 1sn bu isi bitirmeye yetmeyip istekler birikince
    # tarayici "donuyormus" gibi gorunuyordu. 2sn, bir onceki adimin
    # tamamlanmasina yetecek payi biraktigi icin daha akici.
    dcc.Interval(id="replay-tick", interval=2000, disabled=True, n_intervals=0),

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
                options=_build_timezone_options(DEFAULT_LANGUAGE),  # bkz. fonksiyon yorumu -- bos [] ile
                                                                     # baslarsa value null'a sifirlaniyordu
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
                           style=LANG_BTN_INACTIVE_STYLE),
                html.Button("Uydu", id="map-style-satellite-btn", n_clicks=0,
                           style=LANG_BTN_INACTIVE_STYLE),
                html.Button("Karanlık", id="map-style-dark-btn", n_clicks=0,
                           style=LANG_BTN_ACTIVE_STYLE),
            ]),
        ]),
        html.Div(style={"marginBottom": "14px"}, children=[
            # ONEMLI: adsb.lol/OpenSky secimi digerlerinden (dil/harita/
            # saat dilimi) FARKLI -- pure client-side bir dcc.Store degil,
            # AYRI BIR PROCESS'E (uav_producer.py) Redis uzerinden
            # iletilen GERCEK/paylasilan bir ayar. Bu yuzden tik'te
            # (15sn'de bir) backend'den okunup gosteriliyor -- bkz.
            # manage_data_source callback'i.
            html.Label("Veri Kaynağı", id="data-source-label",
                       style={"fontSize": "12px", "color": "#888",
                              "display": "block", "marginBottom": "6px"}),
            html.Div(style={"display": "flex", "gap": "6px"}, children=[
                html.Button(d["label"], id={"type": "data-source-btn", "index": d["key"]},
                           n_clicks=0,
                           style=LANG_BTN_ACTIVE_STYLE if d["key"] == DEFAULT_DATA_SOURCE
                                 else LANG_BTN_INACTIVE_STYLE)
                for d in DATA_SOURCE_DEFS
            ]),
            html.Div(id="data-source-status", style={
                "fontSize": "10px", "color": "#666", "marginTop": "5px",
            }),
        ]),
        html.Div(style={"marginBottom": "14px"}, children=[
            # ONEMLI: kaynaga gore SABIT 10-40sn soluklasma esigi
            # kaldirildi -- OpenSky'nin dogal sorgulama araligi (90-300sn,
            # bkz. uav_producer.py SOURCES) bu sabit esigi neredeyse HER
            # ucak icin asiyordu, ekrandaki TUM filo soluk gorunuyordu.
            # Kullanici artik esigi kendi secili kaynagina gore kendisi
            # ayarliyor (bkz. SIGNAL_STALENESS_OPTIONS, update_map).
            html.Label("Sinyal Yaşı Eşiği", id="signal-staleness-label",
                       style={"fontSize": "12px", "color": "#888",
                              "display": "block", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="signal-staleness-dropdown",
                options=_build_signal_staleness_options(DEFAULT_LANGUAGE),
                value=DEFAULT_SIGNAL_STALENESS_SEC,
                clearable=False, searchable=False,
                className="dark-dropdown",
            ),
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
    dcc.Store(id="timezone-setting", data=DEFAULT_TIMEZONE),
    dcc.Store(id="language-setting", data=DEFAULT_LANGUAGE),
    dcc.Store(id="map-style-setting", data=DEFAULT_MAP_STYLE),
    dcc.Store(id="signal-staleness-setting", data=DEFAULT_SIGNAL_STALENESS_SEC),

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
        html.Div(style={"marginTop": "14px"}, children=[
            html.H4("Önceki Uçuşlar (7 gün)", id="previous-flights-title",
                    style={"color": "#90e0ef", "margin": "0 0 8px 0", "fontSize": "13px"}),
            html.Div(id="flight-segments-list"),
        ]),
    ]),

    # --------------------------- Sag-alt gecmis paneli (varsayilan gizli) --
    # ONEMLI: eskiden irtifa+hiz IKISI birden (cift eksenli) sabit son-24s
    # cizdiriliyordu. Artik SADECE BIRI (varsayilan irtifa, sag-ustteki
    # dropdown'dan degistirilebilir) ve tarih araligi gun+saat
    # dropdown'lariyla SECILEBILIR (ikisi de bos birakilirsa varsayilan
    # davranis -- son 24 saat -- korunur, bkz. update_history). Takvim
    # (dcc.DatePickerRange) YERINE dropdown kullaniliyor -- kullanici
    # istegi: "tıklayınca seçenekler çıksın", ayrica gun secenekleri zaten
    # sadece bugun-7gun araligini (InfluxDB'nin 7 gunluk saklama suresi)
    # kapsadigi icin bir takvimin sundugu serbestlik gereksiz.
    #
    # ONEMLI: gun/saat dropdown'lari update_history'de Input DEGIL State --
    # secim yapmak TEK BASINA grafigi guncellemiyor (kullanici geri
    # bildirimi: "tarih seçsek de bişey olmuyor" -- kafası karışıyordu,
    # cunku 4 dropdown'u tek tek secerken her ara adimda farkli/gecici bir
    # sorgu atiliyordu). Artik "Hesapla" butonuna basana kadar hicbir sey
    # olmuyor, basinca O ANKI 4 secim BIRLIKTE uygulaniyor.
    html.Div(id="history-panel", style={**HISTORY_PANEL_BASE, "transform": "translateY(100%)"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "10px"}, children=[
            html.H4("Geçmiş", id="history-panel-title",
                    style={"color": "#90e0ef", "margin": 0, "fontSize": "13px"}),
            dcc.Dropdown(
                id="history-metric-dropdown",
                options=[{"label": "İrtifa", "value": "alt"}, {"label": "Hız", "value": "velocity"}],
                value="alt", clearable=False, searchable=False,
                className="dark-dropdown", style={"width": "110px", "fontSize": "12px", "flexShrink": 0},
            ),
        ]),
        html.Div(style={"display": "flex", "alignItems": "flex-end", "gap": "10px",
                        "marginBottom": "8px", "flexWrap": "nowrap"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                html.Span("Başlangıç", id="history-start-label",
                         style={"fontSize": "10px", "color": "#888", "whiteSpace": "nowrap"}),
                dcc.Dropdown(id="history-start-day", options=[], value=None,
                            placeholder="Gün", clearable=True, searchable=False,
                            className="dark-dropdown", style={"width": "84px", "fontSize": "11px"}),
                dcc.Dropdown(id="history-start-hour", options=HISTORY_HOUR_OPTIONS, value=None,
                            placeholder="Saat", clearable=True, searchable=False,
                            className="dark-dropdown", style={"width": "68px", "fontSize": "11px"}),
            ]),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                html.Span("Bitiş", id="history-end-label",
                         style={"fontSize": "10px", "color": "#888", "whiteSpace": "nowrap"}),
                dcc.Dropdown(id="history-end-day", options=[], value=None,
                            placeholder="Gün", clearable=True, searchable=False,
                            className="dark-dropdown", style={"width": "84px", "fontSize": "11px"}),
                dcc.Dropdown(id="history-end-hour", options=HISTORY_HOUR_OPTIONS, value=None,
                            placeholder="Saat", clearable=True, searchable=False,
                            className="dark-dropdown", style={"width": "68px", "fontSize": "11px"}),
            ]),
            html.Button("Hesapla", id="history-calculate-btn", n_clicks=0,
                       style=HISTORY_CALC_BTN_STYLE),
            html.Button("⬇", id="history-download-btn", n_clicks=0,
                       title="CSV olarak indir", style=HISTORY_DOWNLOAD_BTN_STYLE),
            dcc.Download(id="history-download"),
        ]),
        dcc.Graph(id="history-chart", style={"height": "160px"},
                  config={"displayModeBar": False}),
    ]),
])


# ONEMLI (GECMIS -- iki basarisiz deneme, bkz. proje sohbet gecmisi):
# 1) Poligon kose noktalari Python'da, Input("map","zoom") ile HER zoom
#    degisiminde SUNUCUDAN yeniden hesaplatildi -- calisiyordu ama her
#    adimda bir ag gidis-donusu oldugu icin GOZLE GORULUR GECIKME vardi
#    ("donma yok ama boyut degisimi yavas/kotu").
# 2) SABIT bir referans zoom'a (5, dl.Map'in baslangic zoom'uyla ayni)
#    gore hesaplanip Leaflet'in kendi native zoom-reprojeksiyonuna
#    birakildi -- gecikme gitti ama kullanici "boyutlar hic degismiyor"
#    bildirdi (Leaflet'in canvas Path reprojeksiyonu, dash-leaflet'in
#    GeoJSON sarmalayicisi uzerinden guvenilir sekilde tetiklenmiyor
#    olabilir -- kesin sebep dogrulanamadi, ama sonuc gozlemlenebilir
#    sekilde calismiyordu).
#
# GUNCEL COZUM: geometri hesaplamasi TAMAMEN CLIENT-SIDE (JS) bir
# clientside_callback'e tasindi (bkz. asagidaki app_dash.clientside_callback
# cagrisi, sayfa sonunda) -- HEM sunucuya gitmiyor (sifir gecikme) HEM
# de GERCEK GUNCEL zoom'u kullanir (dogru boyut). Python (update_map)
# artik SADECE HAM nokta verisini (lat/lon/track + tum tooltip/renk
# bilgisi) "aircraft-raw" store'una yaziyor; clientside callback bunu
# ("aircraft-raw" VEYA "map"."zoom" degistiginde) donup poligona cevirip
# "aircraft-geojson"a yaziyor -- boylece hem yeni veri geldiginde hem
# zoom degistiginde HER ZAMAN o anki GERCEK zoom kullanilir, "eski
# referansa donme" sorunu olmaz.


@app_dash.callback(
    Output("altitude-filter-range", "data"),
    Input("altitude-filter-snapped", "data"),
)
def update_altitude_filter_range(value):
    """"altitude-filter-snapped" (bkz. asagidaki clientside_callback --
    ham slider degerini EN YAKIN 11 duraga yuvarlayip SADECE degisince
    gunceller) SADECE GERCEKTEN bir duraga geçince degistigi icin bu
    Python callback'i de sadece o zaman (en fazla 11 kez) calisir --
    surukleme sirasindaki yuzlerce ara deger sunucuya HIC ULASMAZ, bkz.
    proje sohbet gecmisindeki YARIS DURUMU aciklamasi.

    Iki durak-index'ini (0.._ALT_LEGEND_N) gercek irtifaya (metre --
    ic semamiz zaten metre kullaniyor, ft<->m cevirisi ARTIK YOK, bkz.
    ALTITUDE_COLOR_STOPS ust yorumu) cevirir -- update_map bu Store'u
    okuyup araligin DISINDAKI ucaklari haritadan gizliyor. Tutamaclarin
    SIRASI ne olursa olsun (allowCross=True) SIRALAMAYI HER ZAMAN burada
    kendimiz garanti ediyoruz.

    ONEMLI: MAKS tutamac EN SONDAYSA (_ALT_LEGEND_N, "12000+") ust sinir
    GERCEK 12000m DEGIL -- bircok jet 12000m'yi gecebiliyor, bu durumda
    hala YANLISLIKLA gizlenirlerdi. Sonsuzluk yerine guvenli buyuk bir
    sayi (1_000_000 m) kullaniyoruz -- float('inf') JSON'da GECERSIZ
    (bkz. replay'deki ayni NaN/inf dersi).

    ONEMLI (kullanici geri bildirimi -- "aktif uçuş ile gösterilen
    farklı, Yerde açıkken de böyle"): AYNI sorun ALT sinirda da vardi --
    MIN tutamac EN BASTAYSA (0, ilk durak) alt sinir GERCEK 0m
    kullaniliyordu, ama barometrik irtifa (QNH kalibrasyonu / pist
    yuksekligi farki) YERDEKI/PISTTEKI bircok ucakta dogal olarak HAFIF
    NEGATIF gelir (orn. -7.6m, -45.7m) -- "Yerde" filtresiyle HICBIR
    ILGISI YOK (bu ucaklarin coğu is_ground=False), SADECE 0'in alti
    oldugu icin yanlislikla tamamen gizleniyorlardi. Ust sinirdaki AYNI
    "sinirsiz" mantigini simetrik olarak alt sinira da uyguluyoruz."""
    if not value or len(value) != 2:
        return dash.no_update
    lo_v, hi_v = min(value), max(value)
    lo_m = -1_000_000 if lo_v <= 0 else _slider_value_to_altitude_m(lo_v)
    if hi_v >= _ALT_LEGEND_N:
        hi_m = 1_000_000
    else:
        hi_m = _slider_value_to_altitude_m(hi_v)
    return [lo_m, hi_m]


# ONEMLI: bu callback CLIENTSIDE (JS, tarayicida calisir, Python'a HIC
# UGRAMAZ) -- irtifa kaydiricisinin HAM (kesirli, updatemode="drag" ile
# surukleme sirasinda ONLARCA kez gelen) degerini EN YAKIN durağa (0..10
# tam sayi, ALTITUDE_COLOR_STOPS index'i) yuvarlar. window.__altSnapPrev
# modul-seviyesi degiskeninde bir onceki yuvarlanmis degeri saklar --
# YENI yuvarlanmis deger ONCEKIYLE AYNIYSA no_update doner, Store HIC
# guncellenmez, Python tarafindaki update_altitude_filter_range (ve onun
# tetikledigi pahali update_map) calismaz. Sadece GERCEKTEN farkli bir
# duraga geçildiginde (en fazla 11 kez) bir deger yayinlanir -- boylece
# ekstra bir "Uygula" butonuna gerek kalmadan, kullanici istegi uzerine
# (bkz. proje sohbet gecmisi) yaris durumu riski ~1000 olasi tetikleyiciden
# 11'e dusuruluyor.
app_dash.clientside_callback(
    """
    function(value) {
        if (!value || value.length !== 2) {
            return window.dash_clientside.no_update;
        }
        const snapped = [Math.round(value[0]), Math.round(value[1])];
        const prev = window.__altSnapPrev;
        if (prev && prev[0] === snapped[0] && prev[1] === snapped[1]) {
            return window.dash_clientside.no_update;
        }
        window.__altSnapPrev = snapped;
        return snapped;
    }
    """,
    Output("altitude-filter-snapped", "data"),
    Input("altitude-filter-slider", "value"),
)


# ONEMLI (gercek surukleme testiyle DOGRULANMIS yaris durumu -- bkz. proje
# sohbet gecmisi): irtifa kaydiricisi hizli surukleninde (tek surukleme
# hareketinde 6-7 durak gecilebiliyor) update_map'e ust uste BINEN cagrilar
# gidiyor -- her biri /api/flights'i cekip binlerce ucagi filtreleyen PAHALI
# bir islem, ve YANITLAR GONDERILDIKLERI SIRAYLA DEGIL TAMAMLANDIKLARI SIRAYLA
# donuyor (Playwright ile olculdu: 7 istek 0.3sn icinde gonderildi ama yanitlar
# 9.7-10.4sn arasinda TAMAMEN KARISIK sirada geldi -- 3. gonderilen istek EN
# SON yanit verdi ve Store'u kendi (ARTIK ESKI) degeriyle EZDI). Sonuc:
# kaydirici GORSEL olarak dogru son durakta dursa bile, haritadaki GERCEK
# filtre surukleme sirasindaki ESKI bir ARA durağa donebiliyordu -- kullanici
# geri bildirimi tam bu ("tutamaçlar filtreyi doğru uygulamıyor").
#
# COZUM: her update_map cagrisi baslarken KENDI sira numarasini alir
# (_altitude_map_seq, kilitli/atomik artan sayac). Cagri (ozellikle
# /api/flights istegi) TAMAMLANDIKTAN SONRA, sonucu Store'a YAZMADAN HEMEN
# ONCE kontrol eder: "ben hala EN SON BASLATILAN cagri miyim?" Eger bu
# calisirken DAHA YENI bir cagri BASLAMISSA (ne sebeple olursa olsun --
# irtifa filtresi, tick, askeri/sivil butonu...), bu cagri ARTIK BAYAT
# demektir -- sonucunu ATAR (dash.no_update), boylece daha yeni ama daha ERKEN
# biten bir cagrinin sonucunu asla EZMEZ. Bu, HANGI Input'un tetikledigine
# bakmaksizin genel bir "en son BASLATILAN kazanir" garantisi saglar.
_altitude_map_seq_lock = threading.Lock()
_altitude_map_latest_seq = 0


def _passes_filter(f, show_civil, show_military, show_ground, alt_lo_m, alt_hi_m):
    """update_map icindeki ucak-listesi filtresi -- test edilebilirlik icin
    (eskiden update_map'in icinde bir closure'du) modul seviyesine tasindi.

    - Sivil/askeri: sol-ust butonlardan geliyor, ikisi de acikken (varsayilan)
      hicbir ucak elenmiyor.
    - Yerde: sivil/askeri ekseninden BAGIMSIZ (bir ucak ayni anda hem askeri
      hem yerde olabilir) -- IKI kosul AYRI AYRI uygulanir, havadaki bir ucak
      show_ground'dan hic etkilenmez.
    - Irtifa: alt=None (irtifasi bilinmeyen ucak) FILTRELENMIYOR (guvenli
      varsayilan). alt_lo_m/alt_hi_m'i caniran taraf (update_map) saglamali --
      "sinirsiz" sentinel degerleri -1_000_000/1_000_000'dir, GERCEK 0 DEGIL
      (0 kullanilsaydi, yerdeki/pistteki ucaklarin dogal olarak HAFIF NEGATIF
      gelen barometrik irtifasi yanlislikla filtrelenirdi -- gercekten
      yasanmis bir hataydi, bkz. update_altitude_filter_range)."""
    is_mil = bool(f.get("is_military"))
    if not (show_military if is_mil else show_civil):
        return False
    if f.get("is_ground") and not show_ground:
        return False
    alt = f.get("alt")
    if alt is not None and not (alt_lo_m <= alt <= alt_hi_m):
        return False
    return True


@app_dash.callback(
    [Output("aircraft-raw", "data"), Output("status-main", "children"),
     Output("emergency-alerts-data", "data"), Output("status-alarm", "children")],
    [Input("tick", "n_intervals"), Input("timezone-setting", "data"),
     Input("language-setting", "data"), Input("show-civil", "data"),
     Input("show-military", "data"), Input("show-ground", "data"),
     Input("replay-mode", "data"), Input("altitude-filter-range", "data"),
     Input("signal-staleness-setting", "data")]
)
def update_map(n, tz_name, lang, show_civil, show_military, show_ground, replay_mode,
               altitude_filter_range, staleness_threshold):
    global _altitude_map_latest_seq
    with _altitude_map_seq_lock:
        _altitude_map_latest_seq += 1
        my_seq = _altitude_map_latest_seq
    # ONEMLI: bu artik haritaya DOGRUDAN cizilen "aircraft-geojson"u degil,
    # HAM nokta verisini ("aircraft-raw") dolduruyor -- ucak sekli/boyutu
    # (poligon geometrisi) artik clientside_callback'te (asagida, sayfa
    # sonunda) JS'de hesaplaniyor, bkz. yukaridaki buyuk yorum blogu
    # (2 basarisiz Python-tarafi deneme sonrasi).
    #
    # ONEMLI (replay): replay-mode acikken (bkz. load_replay_data/
    # exit_replay_mode) CANLI veriyi HARITAYA yazmiyoruz -- render_replay_frame
    # zaten "aircraft-raw"i kendi kareleriyle besliyor, ikisi CARPISMASIN.
    # replay-mode Input oldugu icin (State degil) kullanici "Canlıya Dön"e
    # basar basmaz, bir sonraki tick'i beklemeden ANINDA canli goruntuye
    # doner.
    if replay_mode:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    tz = _resolve_tz(tz_name)
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    cat_labels = CATEGORY_LABELS.get(lang, CATEGORY_LABELS[DEFAULT_LANGUAGE])
    staleness_threshold = staleness_threshold or DEFAULT_SIGNAL_STALENESS_SEC
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

    # ONEMLI: bu, /api/alerts'ten (ML ekibinin ileride "uav.alerts" Kafka
    # topic'ine yazacagi, henuz BOS olan anomali alarmlari) TAMAMEN AYRI --
    # burada ucagin KENDI yaydigi ADS-B acil durum sinyali (squawk 7500/
    # 7600/7700 veya emergency alani "none" degil) kontrol ediliyor. Ayni
    # kirmizi renklendirmeyi (alert_icaos) paylasiyor, ayrica "Acil Durum"
    # panelini (asagida ayri bir Store/callback) besliyor. update_map zaten
    # her tick'te flights'i cektigi icin panel icin AYRI bir /api/flights
    # istegi atmiyoruz -- panel bu Store'u okuyor.
    emg_labels_map = EMERGENCY_LABELS.get(lang, EMERGENCY_LABELS[DEFAULT_LANGUAGE])
    emergency_rows = []
    for f in flights:
        squawk = f.get("squawk") or ""
        emg_label = emg_labels_map.get(f.get("emergency") or "none")
        if not (emg_label or squawk in ("7500", "7600", "7700")):
            continue
        icao = f.get("icao24", "")
        alert_icaos.add(icao)
        ts_raw = f.get("ts", "")
        try:
            ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts_display = ts_dt.astimezone(tz).strftime("%H:%M:%S")
        except Exception:
            ts_display = "—"
        emergency_rows.append({
            "icao24": icao,
            "callsign": (f.get("callsign") or "").strip() or icao.upper(),
            "squawk": squawk or "—",
            "label": emg_label or t["emergency_squawk"].format(squawk=squawk),
            "ts": ts_display,
        })

    # Sivil/askeri/yerde/irtifa filtrelerinin SEMANTIGI icin bkz. _passes_filter
    # docstring'i -- burada sadece irtifa lejandindan (bkz.
    # update_altitude_filter_range) gelen araligi coziyoruz, [alt_min_m, alt_max_m].
    alt_lo_m, alt_hi_m = (altitude_filter_range if altitude_filter_range
                          and len(altitude_filter_range) == 2 else (-1_000_000, 1_000_000))

    total_flight_count = len(flights)  # HICBIR filtre (sivil/askeri/yerde/
                                        # irtifa/firma) uygulanmadan ONCEKI
                                        # toplam -- status cubugundaki
                                        # "toplam" sayisi bu (bkz. asagida)
    flights = [f for f in flights
               if _passes_filter(f, show_civil, show_military, show_ground, alt_lo_m, alt_hi_m)]

    # Her ucak icin GOSTERIME HAZIR (formatlanmis) degerleri Python'da
    # hesaplayip duz bir sozluge koyuyoruz -- ceviri (t[...]) ve sayi
    # formatlama SADECE burada yapiliyor, JS tarafinda (_POINT_TO_LAYER_JS /
    # _ON_EACH_FEATURE_JS) tekrarlanmiyor, JS sadece bu degerleri yerlestiriyor.
    features = []
    for f in flights:
        icao = f.get("icao24", "")
        is_military = bool(f.get("is_military"))
        # ONEMLI: adsb.lol/tar1090 gibi -- varsayilan renk artik IRTIFAYA
        # gore (bkz. _altitude_to_color), askeri/sivil ayrimi RENKTE
        # ARTIK YOK (kullanici karariyla -- "adsb.lol'daki gibi olsun").
        # Acil durum (ALERT_COLOR) TEK istisna, guvenlik-kritik oldugu
        # icin irtifa renginin USTUNE geciyor.
        if icao in alert_icaos:
            color = ALERT_COLOR
        else:
            color = _altitude_to_color(f.get("alt"))
        callsign = f.get("callsign", "").strip()
        category_label = cat_labels.get(f.get("category", ""), None)
        subtitle_parts = [icao.upper()]
        if category_label:
            subtitle_parts.append(category_label)
        if is_military:
            subtitle_parts.append(t["tooltip_military_tag"])
        if f.get("is_ground"):
            subtitle_parts.append(t["tooltip_ground_tag"])
        subtitle = "  ·  ".join(subtitle_parts)

        # ONEMLI: adsb.lol/readsb, bir ucaktan mesaj kesilse bile onu 60
        # saniyeye kadar listede TUTAR ("seen" alani = mesajin GERCEKTE
        # kac saniye once alindigi) -- OpenSky'de bu daha da uzun (90-300sn,
        # bkz. uav_producer.py SOURCES). Once SABIT bir 10-40sn esik vardi,
        # OpenSky'nin dogal sorgulama araligi bunu HER ucak icin astigi icin
        # ekrandaki NEREDEYSE TUM filo soluk gorunuyordu -- artik esik
        # ayarlardan seciliyor (bkz. _signal_opacity, hesabin kendisi).
        signal_age = f.get("signal_age_sec")
        opacity = _signal_opacity(signal_age, staleness_threshold)
        signal_age_text = (f"{signal_age:.0f}sn"
                           if signal_age is not None and signal_age >= 10 else None)

        lat, lon = f.get("lat", 39), f.get("lon", 35)
        track = f.get("track") or 0

        # ONEMLI: geometri hala Point (Polygon DEGIL) -- ok/dart seklinin
        # kose noktalarini clientside_callback (JS) hesaplayacak, "track"
        # (heading) bu yuzden properties'te HAM olarak tasiniyor.
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": dict(
                icao24=icao,
                callsign=callsign or icao.upper(),
                color=color,
                opacity=round(opacity, 2),
                track=track,
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
            ),
        })

    raw_data = {"type": "FeatureCollection", "features": features}

    ts = datetime.now(tz).strftime("%H:%M:%S")
    status_main = t["status_bar_main"].format(ts=ts, n=total_flight_count)
    status_alarm = t["status_bar_alarm"].format(a=len(alerts))

    # ONEMLI: yukaridaki _altitude_map_seq aciklamasina bkz -- eger bu cagri
    # calisirken (esp. /api/flights isteği surerken) DAHA YENI bir update_map
    # cagrisi BASLAMISSA, bu sonuc ARTIK BAYAT -- yazmadan at, daha yeni
    # cagrinin (ne zaman bitecegi onemli degil) sonucunu asla ezme.
    if my_seq != _altitude_map_latest_seq:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return raw_data, status_main, emergency_rows, status_alarm


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
    dahil) iceriyor. Kumeleme kaldirildigi icin her feature artik gercek
    bir ucak -- "icao24" hep dolu, guvenlik icin yine de kontrol ediliyor."""
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
    [Output("aircraft-select", "data", allow_duplicate=True),
     Output("callsign-search-feedback", "children"),
     Output("map", "viewport")],
    [Input("callsign-search-btn", "n_clicks"), Input("callsign-search-input", "n_submit")],
    [State("callsign-search-input", "value"), State("language-setting", "data"),
     State("map", "zoom")],
    prevent_initial_call=True,
)
def search_by_callsign(n_clicks, n_submit, query, lang, current_zoom):
    """Cagri koduna gore arama -- tiklama secimiyle AYNI Output'u
    (aircraft-select) yazdigi icin allow_duplicate=True gerekiyor (Dash,
    ayni Output'u birden fazla callback'in yazmasina bunu isaretlersen
    izin veriyor). Once TAM eslesme (bosluk/buyuk-kucuk harf normalize
    edilerek -- adsb.lol callsign'lari bazen "VATOZ 16" gibi ic bosluklu
    donuyor), yoksa KISMI eslesme (kullanici callsign'in bir kismini
    yazmis olabilir) denenir. Bulunamazsa secim DEGISTIRILMEZ, sadece
    kisa bir "bulunamadi" mesaji gosterilir.

    ONEMLI (kullanici istegi, YANLIS YERE zoom hatasi DUZELTILDI): ucak
    bulununca harita da OTOMATIK o ucaga kayiyor. ILK DENEMEDE
    Output("map","center")/Output("map","zoom") kullanilmisti -- ama
    dash-leaflet'in kendi dokumantasyonu acikca soyluyor: "center"
    SADECE haritanin ILK KURULUMUNDA kullanilir, sonradan degistirmek
    Leaflet'e YANSIMAZ (harita zaten kurulu oldugu icin "yanlis yere
    zoom" gibi gorunuyordu -- aslinda hicbir yere gitmiyordu, harita
    oldugu yerde kaliyordu VEYA eski/varsayilan degere donuyordu).
    Mount-SONRASI navigasyon icin ozel bir "viewport" prop'u var (bkz.
    site-packages/dash_leaflet/MapContainer.py), transition="flyTo" ile
    duzgun ve animasyonlu sekilde gercekten o konuma gidiyor."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    query = (query or "").strip().upper()
    if not query:
        return dash.no_update, "", dash.no_update

    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []

    def norm(cs):
        return (cs or "").strip().upper()

    match = next((f for f in flights if norm(f.get("callsign")) == query), None)
    if not match:
        match = next((f for f in flights if query in norm(f.get("callsign"))), None)

    if match and match.get("lat") is not None and match.get("lon") is not None:
        zoom = max(current_zoom or 5, 8)
        viewport = {"center": [match["lat"], match["lon"]], "zoom": zoom, "transition": "flyTo"}
        return match["icao24"], "", viewport
    if match:
        return match["icao24"], "", dash.no_update
    return dash.no_update, t["callsign_not_found"].format(callsign=query), dash.no_update


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
    Output("stats-open", "data"),
    [Input("stats-btn", "n_clicks"), Input("close-stats-btn", "n_clicks")],
    State("stats-open", "data"),
    prevent_initial_call=True,
)
def toggle_stats_open(stats_clicks, close_clicks, is_open):
    """Ayarlar panelinin ac/kapa deseniyle BIREBIR ayni (bkz.
    toggle_settings_open) -- ayri bir Store/panel oldugu icin
    birbirinden bagimsiz calisiyorlar, ikisi ayni anda acik kalabilir."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "close-stats-btn.n_clicks":
        return False
    if trigger == "stats-btn.n_clicks":
        return not is_open
    return dash.no_update


@app_dash.callback(
    Output("stats-panel", "style"),
    Input("stats-open", "data"),
)
def show_stats_panel(is_open):
    style = dict(STATS_PANEL_BASE)
    style["display"] = "block" if is_open else "none"
    return style


@app_dash.callback(
    Output("emergency-open", "data"),
    [Input("emergency-btn", "n_clicks"), Input("close-emergency-btn", "n_clicks")],
    State("emergency-open", "data"),
    prevent_initial_call=True,
)
def toggle_emergency_open(open_clicks, close_clicks, is_open):
    """toggle_stats_open ile BIREBIR ayni ac/kapa deseni."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "close-emergency-btn.n_clicks":
        return False
    if trigger == "emergency-btn.n_clicks":
        return not is_open
    return dash.no_update


@app_dash.callback(
    Output("emergency-panel", "style"),
    Input("emergency-open", "data"),
)
def show_emergency_panel(is_open):
    style = dict(EMERGENCY_PANEL_BASE)
    style["display"] = "block" if is_open else "none"
    return style


@app_dash.callback(
    Output("emergency-btn", "style"),
    Input("emergency-alerts-data", "data"),
)
def update_emergency_badge(rows):
    """Panel kapaliyken bile aktif acil durum oldugunu belli etmek icin
    dugmenin kenarligini/rengini kirmiziya ceviriyor -- kullanici paneli
    hic acmasa da fark etsin diye."""
    style = {
        "position": "absolute", "top": "12px", "right": "108px",
        "width": "40px", "height": "40px", "borderRadius": "50%",
        "fontSize": "18px", "cursor": "pointer", "zIndex": 900,
    }
    if rows:
        style.update({"backgroundColor": "#e63946", "border": "1px solid #e63946",
                      "color": "#fff"})
    else:
        style.update({"backgroundColor": "#000000", "border": "1px solid #2a2a4a",
                      "color": "#c8d0e0"})
    return style


@app_dash.callback(
    Output("emergency-panel-list", "children"),
    [Input("emergency-alerts-data", "data"), Input("language-setting", "data")],
)
def update_emergency_list(rows, lang):
    """emergency-alerts-data'yi (update_map dolduruyor) salt-okunur bir
    listeye ceviriyor -- bilgi amacli, TIKLANAMAZ (bkz. auto_focus_new_emergency:
    ilgili ucak zaten ILK GORULDUGUNDE otomatik seciliyor, ayrica tiklama
    gerekmiyor -- eskiden buton+n_clicks vardi ama bu liste her tick'te
    yeniden kuruldugu icin n_clicks 0'a sifirlaniyor, bu da Dash'te
    "tiklama" gibi algilanip kullanici paneli kapatsa bile ucagi TEKRAR
    seciyordu -- kullanici geri bildirimi: "kapatsak bile yeniden
    açılıyor". Kok neden buydu, en saglam cozum tiklamayi TAMAMEN
    kaldirmak oldu)."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    if not rows:
        return html.Div(t["no_emergency"], style={"color": "#666", "fontSize": "13px"})

    return [
        html.Div(
            [
                html.Div(f"{r['callsign']}  ·  {r['icao24'].upper()}",
                        style={"fontWeight": "600", "fontSize": "13px"}),
                html.Div(f"{r['label']}  ·  squawk {r['squawk']}  ·  {r['ts']}",
                        style={"fontSize": "11px", "color": "#ffb3ba", "marginTop": "2px"}),
            ],
            style={**EMERGENCY_ROW_STYLE, "cursor": "default"},
        )
        for r in rows
    ]


@app_dash.callback(
    [Output("aircraft-select", "data", allow_duplicate=True),
     Output("emergency-seen", "data")],
    Input("emergency-alerts-data", "data"),
    State("emergency-seen", "data"),
    prevent_initial_call=True,
)
def auto_focus_new_emergency(rows, seen):
    """Kullanici istegi: bir ucak acil durum yayinlamaya BASLADIGINDA
    otomatik olarak o ucaga git (sol panel acilsin), ama kullanici
    paneli KAPATTIKTAN SONRA ayni (hala devam eden) acil durum icin bir
    daha KENDILIGINDEN acilmasin.

    "emergency-seen" -- hangi icao24'ler icin zaten oto-odaklama
    yapildigini hatirliyor (session boyunca kalici). emergency-alerts-data
    HER tick'te (15sn) yeniden dolduruluyor -- ayni ucak hala acil
    durumda olsa bile "seen" listesindeyse BIR DAHA tetiklenmiyor, sadece
    GERCEKTEN yeni bir icao24 gorununce (once hic emergency olmayan bir
    ucak simdi emergency oldu) devreye giriyor."""
    seen_set = set(seen or [])
    new_icaos = [r["icao24"] for r in (rows or []) if r["icao24"] not in seen_set]
    if not new_icaos:
        return dash.no_update, dash.no_update
    seen_set.update(new_icaos)
    return new_icaos[0], sorted(seen_set)


@app_dash.callback(
    Output("stats-chart", "figure"),
    [Input("tick", "n_intervals"), Input("stats-open", "data"),
     Input("timezone-setting", "data"), Input("language-setting", "data")],
)
def update_stats_chart(n, is_open, tz_name, lang):
    """Saat basina benzersiz ucak sayisi (son 24 saat) -- panel KAPALIYKEN
    hicbir sey sorgulamiyoruz (is_open=False ise erken donuyor), her
    15sn'de bir gereksiz InfluxDB sorgusu atmayalim."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    tz = _resolve_tz(tz_name)
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#0f0f19", plot_bgcolor="#0f0f19",
                      font=dict(color="#c8d0e0", size=10),
                      margin=dict(t=10, b=30, l=50, r=10))

    if not is_open:
        return fig

    try:
        data = requests.get("http://localhost:8000/api/traffic_stats",
                            params={"hours": 24}, timeout=10).json()
    except Exception:
        data = []

    if not data or isinstance(data, dict):
        fig.add_annotation(text=t["no_data"], showarrow=False,
                           font=dict(color="#666"))
        return fig

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(tz)

    fig.add_trace(go.Scatter(x=df["time"], y=df["unique_aircraft"],
                             mode="lines+markers",
                             line=dict(color="#00b4d8", width=1.5)))
    fig.update_layout(yaxis=dict(title=t["stats_yaxis_label"]))
    return fig


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
    return _build_timezone_options(lang)


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
    [Input("map-style-street-btn", "n_clicks"), Input("map-style-satellite-btn", "n_clicks"),
     Input("map-style-dark-btn", "n_clicks")],
    prevent_initial_call=True,
)
def update_map_style_setting(street_clicks, satellite_clicks, dark_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "map-style-street-btn.n_clicks":
        return "street"
    if trigger == "map-style-satellite-btn.n_clicks":
        return "satellite"
    if trigger == "map-style-dark-btn.n_clicks":
        return "dark"
    return dash.no_update


@app_dash.callback(
    [Output("map-style-street-btn", "style"), Output("map-style-satellite-btn", "style"),
     Output("map-style-dark-btn", "style")],
    Input("map-style-setting", "data"),
)
def update_map_style_buttons(style):
    styles = [LANG_BTN_INACTIVE_STYLE] * 3
    idx = {"street": 0, "satellite": 1, "dark": 2}.get(style, 0)
    styles[idx] = LANG_BTN_ACTIVE_STYLE
    return tuple(styles)


@app_dash.callback(
    Output("signal-staleness-setting", "data"),
    Input("signal-staleness-dropdown", "value"),
)
def update_signal_staleness_setting(value):
    # ONEMLI: "if not value" DEGIL -- timezone-dropdown'daki ayni hataya
    # dusmemek icin None kontrolu (bkz. update_timezone_setting yorumu);
    # buradaki en kucuk secenek (30) zaten falsy degil ama tutarlilik icin
    # aynı deseni koruyoruz.
    if value is None:
        return dash.no_update
    return value


@app_dash.callback(
    Output("signal-staleness-dropdown", "options"),
    Input("language-setting", "data"),
)
def update_signal_staleness_options(lang):
    return _build_signal_staleness_options(lang)


@app_dash.callback(
    [Output({"type": "data-source-btn", "index": ALL}, "style"),
     Output("data-source-status", "children")],
    [Input("tick", "n_intervals"),
     Input({"type": "data-source-btn", "index": ALL}, "n_clicks"),
     Input("language-setting", "data")],
    State({"type": "data-source-btn", "index": ALL}, "id"),
)
def manage_data_source(n, btn_clicks, lang, btn_ids):
    """Diger ayarlardan (dil/harita/saat dilimi) FARKLI -- bu, AYRI BIR
    PROCESS'in (uav_producer.py) okudugu paylasilan bir ayar, pure
    client-side degil. Butona basilinca ONCE backend'e YAZIYORUZ
    (POST /api/data_source -- Redis'e yaziyor), SONRA (butonlar da dahil
    her tetiklemede) GERCEK durumu OKUYUP gosteriyoruz -- boylece "istenen"
    ile "producer'in su an gercekten kullandigi" (henuz bir sonraki
    cycle'a -- 60/300sn -- gecmemis olabilir) birbirinden AYRI gosterilir,
    kullanici "tikladim ama degismedi" sanip yanilmaz.

    Butonlar DATA_SOURCE_DEFS'ten pattern-matching id ile uretiliyor (bkz.
    layout) -- buton SAYISI hep DATA_SOURCE_DEFS'e esit ve SABIT (dinamik
    olarak degismiyor), bu yuzden Output/Input ALL burada guvenli; flight-
    segment-btn'deki gibi AYRI bir liste-uretme callback'i ile karisip
    IndexError riski yok (bkz. proje sohbet gecmisi)."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "data-source-btn":
        try:
            requests.post("http://localhost:8000/api/data_source",
                          params={"source": triggered_id["index"]}, timeout=3)
        except Exception:
            pass

    try:
        info = requests.get("http://localhost:8000/api/data_source", timeout=3).json()
    except Exception:
        info = {}

    requested = info.get("requested", DEFAULT_DATA_SOURCE)
    active = info.get("active") or {}
    active_source = active.get("source")

    styles = [LANG_BTN_ACTIVE_STYLE if b["index"] == requested else LANG_BTN_INACTIVE_STYLE
              for b in btn_ids]

    if active_source and active_source != requested:
        status = t["data_source_pending"].format(requested=requested, active=active_source)
    elif active_source:
        status = t["data_source_active"].format(source=active_source)
    else:
        status = ""

    return styles, status


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
    [Output("settings-title", "children"),
     Output("timezone-label", "children"), Output("language-label", "children"),
     Output("aircraft-info-title", "children"), Output("history-panel-title", "children"),
     Output("filter-civil-btn", "children"), Output("filter-military-btn", "children"),
     Output("filter-ground-btn", "children"),
     Output("map-style-label", "children"), Output("map-style-street-btn", "children"),
     Output("map-style-satellite-btn", "children"), Output("map-style-dark-btn", "children"),
     Output("history-start-label", "children"), Output("history-end-label", "children"),
     Output("history-start-day", "placeholder"), Output("history-end-day", "placeholder"),
     Output("history-start-hour", "placeholder"), Output("history-end-hour", "placeholder"),
     Output("history-calculate-btn", "children"),
     Output("callsign-search-input", "placeholder"),
     Output("stats-title", "children"),
     Output("data-source-label", "children"),
     Output("previous-flights-title", "children"),
     Output("emergency-title", "children"),
     Output("replay-title", "children"),
     Output("replay-start-label", "children"), Output("replay-end-label", "children"),
     Output("replay-load-btn", "children"), Output("replay-live-btn", "children"),
     Output("altitude-legend-title", "children"),
     Output("airline-filter-dropdown", "placeholder"),
     Output("signal-staleness-label", "children")],
    Input("language-setting", "data"),
)
def update_static_texts(lang):
    """Sabit basliklari/etiketleri secili dile gore gunceller. Diger tum
    dinamik metinler (panel icerikleri, tooltip, grafik) kendi
    callback'lerinde language-setting'i dogrudan Input olarak aliyor."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    return (t["settings_title"], t["timezone_label"],
            t["language_label"], t["aircraft_info_title"], t["history_panel_title"],
            t["filter_civil_label"], t["filter_military_label"], t["filter_ground_label"],
            t["map_style_label"], t["map_style_street"], t["map_style_satellite"], t["map_style_dark"],
            t["history_range_placeholder_start"], t["history_range_placeholder_end"],
            t["history_day_placeholder"], t["history_day_placeholder"],
            t["history_hour_placeholder"], t["history_hour_placeholder"],
            t["history_calculate_label"], t["callsign_search_placeholder"],
            t["stats_title"], t["data_source_label"], t["previous_flights_title"],
            t["emergency_panel_title"], t["replay_panel_title"],
            t["history_range_placeholder_start"], t["history_range_placeholder_end"],
            t["replay_load_label"], t["replay_live_label"],
            t["altitude_legend_title"], t["airline_filter_placeholder"],
            t["signal_staleness_label"])


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
    Output("show-ground", "data"),
    Input("filter-ground-btn", "n_clicks"),
    State("show-ground", "data"),
    prevent_initial_call=True,
)
def toggle_show_ground(n_clicks, current):
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
    Output("filter-ground-btn", "style"),
    Input("show-ground", "data"),
)
def style_ground_filter_btn(visible):
    return FILTER_BTN_GROUND_ACTIVE_STYLE if visible else FILTER_BTN_INACTIVE_STYLE


def _altitude_colored_segments(points, weight=3, opacity=0.85):
    """points: ardisik zaman sirali lat/lon/alt kayitlari (dict, "lat"/
    "lon"/"alt" anahtarlariyla). adsb.lol/tar1090'daki gibi "iz boyunca
    irtifaya gore renk degisen" gorunum icin, dash-leaflet'in Polyline'i
    TEK renk destekledigi icin ardisik her nokta CIFTINI, o ciftin
    ORTALAMA irtifasina gore renklendirilmis AYRI/kucuk bir dl.Polyline
    olarak dondurur."""
    segments = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if p1["lat"] is None or p1["lon"] is None or p2["lat"] is None or p2["lon"] is None:
            continue
        alt1, alt2 = p1.get("alt"), p2.get("alt")
        if alt1 is not None and alt2 is not None:
            avg_alt = (alt1 + alt2) / 2
        else:
            avg_alt = alt1 if alt1 is not None else alt2
        segments.append(dl.Polyline(
            positions=[[p1["lat"], p1["lon"]], [p2["lat"], p2["lon"]]],
            color=_altitude_to_color(avg_alt), weight=weight, opacity=opacity,
        ))
    return segments


@app_dash.callback(
    Output("flight-path-layer", "children"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data"),
     Input("selected-flight-segment", "data")]
)
def update_flight_path(n, icao24, segment):
    """Haritada cizgi olarak gosterilecek ucus: "Onceki Ucuslar"
    listesinden bir segment SECILMISSE (bkz. update_selected_segment)
    TAM O ucusun konum gecmisini ceker (start/end ile /api/history).
    SECILI DEGILSE (varsayilan, segment=None) eskisi gibi "SON UCUS"
    HEURISTIGINE doner -- son 24 saatlik gecmisi cekip, EN SON noktadan
    GERIYE dogru tarayarak ardisik iki nokta arasinda
    FLIGHT_GAP_THRESHOLD_MIN (20 dakika) esigini asan ilk bosluga kadar
    devam eder (bir bosluk = "onceki ucus bitti, yenisi basladi")."""
    if not icao24:
        return []

    if segment and segment.get("start") and segment.get("end"):
        try:
            data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                                params={"start": segment["start"], "end": segment["end"]},
                                timeout=5).json()
        except Exception:
            data = []
        if not data or isinstance(data, dict):
            return []
        points = [d for d in data if d.get("lat") is not None and d.get("lon") is not None]
        if len(points) < 2:
            return []
        return _altitude_colored_segments(points)

    try:
        data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                            params={"hours": 24}, timeout=5).json()
    except Exception:
        data = []
    if not data or isinstance(data, dict):
        return []

    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"])
    df = df.sort_values("_time").reset_index(drop=True)

    gap = pd.Timedelta(minutes=FLIGHT_GAP_THRESHOLD_MIN)
    gap_positions = df.index[df["_time"].diff() > gap]
    cutoff_idx = gap_positions[-1] if len(gap_positions) else 0
    last_flight = df.iloc[cutoff_idx:]

    points = [{"lat": row.lat, "lon": row.lon, "alt": row.alt}
              for row in last_flight.itertuples()
              if pd.notna(row.lat) and pd.notna(row.lon)]
    if len(points) < 2:
        return []

    return _altitude_colored_segments(points)


@app_dash.callback(
    [Output("flight-segments-list", "children"), Output("flight-segments-store", "data")],
    [Input("aircraft-select", "data"), Input("language-setting", "data"),
     Input("timezone-setting", "data"), Input("selected-flight-segment", "data")],
)
def update_flight_segments(icao24, lang, tz_name, selected):
    """Sol paneldeki 'Onceki Ucuslar' listesini doldurur --
    /api/flight_segments/{icao24}'ten gelen HAM (UTC) segmentleri
    kullanicinin sectigi saat dilimine cevirip goruntuluyor. Her satir
    pattern-matching bir id tasiyor ({"type":"flight-segment-btn",
    "index":i}) -- update_selected_segment hangisine tiklandigini bu
    index'le buluyor, "flight-segments-store"taki HAM veriden start/end
    okuyor.

    ONEMLI: "aktif" (haritada su an cizilen) satirin stili AYRI bir
    Output({"type":...,"index":ALL}, "style") callback'i OLARAK degil,
    BURADA (liste OLUSTURULURKEN) belirleniyor -- ALL-pattern bir Output,
    layout'taki GERCEK bilesen SAYISIYLA (o an DOM'da kac tane
    "flight-segment-btn" varsa) tam eslesmezse Dash "IndexError: list
    index out of range" hatasi veriyor (bkz. proje sohbet gecmisi, canli
    tespit edildi) -- iki AYRI callback (liste olusturan + stilleyen)
    arasinda bu sayi senkron kalmayabiliyordu. Tek callback'te
    birlestirmek bu riski TAMAMEN ortadan kaldiriyor."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    if not icao24:
        return [], []

    tz = _resolve_tz(tz_name)
    try:
        segments = requests.get(f"http://localhost:8000/api/flight_segments/{icao24}",
                                timeout=5).json()
    except Exception:
        segments = []
    if not segments or isinstance(segments, dict):
        return html.Div(t["no_previous_flights"],
                        style={"color": "#666", "fontSize": "12px"}), []

    rows = []
    for i, seg in enumerate(segments):
        start_local = pd.to_datetime(seg["start"]).tz_convert(tz)
        end_local = pd.to_datetime(seg["end"]).tz_convert(tz)
        same_day = start_local.date() == end_local.date()
        date_label = start_local.strftime("%d.%m.%Y")
        time_label = (f"{start_local.strftime('%H:%M')} → {end_local.strftime('%H:%M')}"
                     if same_day else
                     f"{start_local.strftime('%d.%m %H:%M')} → {end_local.strftime('%d.%m %H:%M')}")
        duration_text = t["flight_duration_min"].format(min=f"{seg['duration_min']:.0f}")

        # ONEMLI: yer adi bulunamadiysa (Nominatim'e hic sorulmadi -- bkz.
        # GEOCODE_MAX_LOOKUPS_PER_REQUEST -- veya sorulup sonuc gelmedi)
        # KOORDINATA duseriz, bos birakmayiz -- "mumkunse" isteniyordu,
        # tam olmasa da HER ZAMAN bir konum bilgisi gosterilsin diye.
        def _place_or_coords(place, lat, lon):
            return place if place else f"{lat:.1f}°,{lon:.1f}°"
        start_place = _place_or_coords(seg.get("start_place"), seg["start_lat"], seg["start_lon"])
        end_place = _place_or_coords(seg.get("end_place"), seg["end_lat"], seg["end_lon"])
        place_text = f"{start_place} → {end_place}"

        # secim YOKSA (None, varsayilan "son ucus" gosteriliyor demektir)
        # listedeki EN YENI (index 0) satiri aktif gosteriyoruz -- segmentler
        # de ayni gap-esigiyle hesaplandigi icin normalde ayni ucusa denk gelir.
        if selected:
            is_active = (selected.get("start") == seg["start"]
                        and selected.get("end") == seg["end"])
        else:
            is_active = (i == 0)

        rows.append(html.Button(
            [
                html.Div(f"{date_label}  ·  {time_label}",
                        style={"fontSize": "12px", "color": "#c8d0e0"}),
                html.Div(place_text, style={"fontSize": "11px", "color": "#00b4d8",
                                            "marginTop": "2px"}),
                html.Div(duration_text, style={"fontSize": "10px", "color": "#888",
                                               "marginTop": "2px"}),
            ],
            id={"type": "flight-segment-btn", "index": i}, n_clicks=0,
            style=FLIGHT_SEGMENT_BTN_ACTIVE_STYLE if is_active else FLIGHT_SEGMENT_BTN_STYLE,
        ))
    return rows, segments


@app_dash.callback(
    Output("selected-flight-segment", "data"),
    [Input("aircraft-select", "data"),
     Input({"type": "flight-segment-btn", "index": ALL}, "n_clicks")],
    State("flight-segments-store", "data"),
    prevent_initial_call=True,
)
def update_selected_segment(icao24, segment_clicks, segments):
    """Yeni bir ucak SECILINCE segment secimini SIFIRLAR (None ->
    update_flight_path varsayilana, "son ucus"a doner) -- eski ucagin
    secili segmenti yeni ucak icin anlamsiz kalirdi. Listeden bir
    satira TIKLANINCA o segmentin start/end'ini dondurur."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    triggered_id = ctx.triggered_id
    if triggered_id == "aircraft-select":
        return None
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "flight-segment-btn":
        idx = triggered_id["index"]
        if segments and 0 <= idx < len(segments):
            return {"start": segments[idx]["start"], "end": segments[idx]["end"]}
    return dash.no_update


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
        (t["field_ground"], t["ground_yes"] if match.get("is_ground") else t["ground_no"]),
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
    Output("history-metric-dropdown", "options"),
    Input("language-setting", "data"),
)
def update_history_metric_options(lang):
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    return [{"label": t["field_alt"], "value": "alt"},
            {"label": t["field_speed"], "value": "velocity"}]


@app_dash.callback(
    [Output("history-start-day", "options"), Output("history-end-day", "options"),
     Output("replay-start-day", "options"), Output("replay-end-day", "options")],
    [Input("tick", "n_intervals"), Input("timezone-setting", "data")],
)
def update_history_day_options(n, tz_name):
    """Gun secenekleri (bugun -> 7 gun once) her tick'te YENIDEN
    hesaplanir -- InfluxDB bucket'i zaten 7 gunden fazlasini tutmuyor,
    ve gece yarisi gecince ("bugun" degisince) sayfa yenilenmeden kendini
    duzeltsin diye sabit/bir kerelik hesaplanmiyor. Kullanicinin SU AN
    sectigi saat dilimine gore -- UTC'ye gore hesaplansaydi "bugun" sinir
    kullanicinin yerel gece yarisindan saatlerce once/sonra kayardi.
    Ayni secenek listesi replay panelinin gun dropdown'larinda da
    kullaniliyor (aircraft-select'e bagli DEGIL, o yuzden paylasilabilir)."""
    tz = _resolve_tz(tz_name)
    today = datetime.now(tz).date()
    days = [today - timedelta(days=i) for i in range(7, -1, -1)]
    options = [{"label": d.strftime("%d.%m"), "value": d.isoformat()} for d in days]
    return options, options, options, options


@app_dash.callback(
    Output("history-chart", "figure"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data"),
     Input("timezone-setting", "data"), Input("language-setting", "data"),
     Input("history-metric-dropdown", "value"), Input("history-calculate-btn", "n_clicks")],
    [State("history-start-day", "value"), State("history-start-hour", "value"),
     State("history-end-day", "value"), State("history-end-hour", "value")]
)
def update_history(n, icao24, tz_name, lang, metric, calc_clicks,
                    start_day, start_hour, end_day, end_hour):
    HOURS_DEFAULT = 24  # tarih araligi secilmemisse (varsayilan) kullanilir

    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    tz = _resolve_tz(tz_name)
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#0f0f19", plot_bgcolor="#0f0f19",
                      font=dict(color="#c8d0e0", size=10),
                      margin=dict(t=10, b=30, l=40, r=40))

    if not icao24:
        return fig  # panel zaten gizli, bos figure yeterli

    # ONEMLI: gun+saat DORDU DE secilmisse (baslangic gun/saat, bitis
    # gun/saat) kullanicinin SU AN sectigi saat dilimine gore YEREL
    # zamani hesaplayip UTC'ye ceviriyoruz, InfluxDB'ye o gidiyor.
    # Dorduncuden biri bile eksikse (varsayilan, kullanici hic secim
    # yapmadiysa VEYA sadece kismen sectiyse) eski davranis -- son 24 saat.
    params = {"hours": HOURS_DEFAULT}
    if None not in (start_day, start_hour, end_day, end_hour):
        # ONEMLI (duzeltildi -- kullanici geri bildirimi: "11 ile 12 arasi
        # istedim ama 11.40-12.40 geldi"): eskiden bitis saatine +1 SAAT
        # ekleniyordu ("o saatin TAMAMINI kapsasin" niyetiyle), yani
        # start=11/end=12 secince aslinda 11:00-13:00 sorgulaniyordu --
        # "su an" (ornegin 12:42) bu araligin ICINDE kaldigi icin veri
        # "su ana kadar" kesiliyor, kullaniciya sanki yanlis/kaymis bir
        # aralik gibi gorunuyordu. Artik end_hour TAM SINIR -- "11 ile 12
        # arasi" = tam olarak 11:00-12:00, +1 saat YOK.
        start_utc = datetime.fromisoformat(start_day).replace(hour=start_hour, tzinfo=tz) \
                            .astimezone(timezone.utc)
        end_utc = datetime.fromisoformat(end_day).replace(hour=end_hour, tzinfo=tz) \
                          .astimezone(timezone.utc)
        params = {"start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}

    try:
        data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                            params=params, timeout=10).json()
    except Exception:
        data = []

    if not data or isinstance(data, dict):
        fig.add_annotation(text=t["no_data"], showarrow=False,
                           font=dict(color="#666"))
        return fig

    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"]).dt.tz_convert(tz)

    # ONEMLI: eskiden irtifa+hiz IKISI birden cift eksenli cizdiriliyordu --
    # kullanici istegiyle artik SADECE SECILI OLAN TEKI (bkz. sag-ustteki
    # "history-metric-dropdown") cizdiriliyor.
    if metric == "velocity":
        col, label, color = "velocity", t["history_speed_label"], "#f77f00"
    else:
        col, label, color = "alt", t["history_alt_label"], "#00b4d8"

    fig.add_trace(go.Scatter(x=df["_time"], y=df[col], name=label,
                             line=dict(color=color, width=1.5)))
    fig.update_layout(yaxis=dict(title=label), showlegend=False)
    return fig


@app_dash.callback(
    Output("history-download", "data"),
    Input("history-download-btn", "n_clicks"),
    [State("aircraft-select", "data"), State("timezone-setting", "data"),
     State("history-start-day", "value"), State("history-start-hour", "value"),
     State("history-end-day", "value"), State("history-end-hour", "value")],
    prevent_initial_call=True,
)
def download_history_csv(n_clicks, icao24, tz_name, start_day, start_hour, end_day, end_hour):
    """update_history'deki gun/saat -> UTC cozumlemesiyle AYNI mantik --
    grafikte GORULEN aralikla indirilen CSV'nin aralik AYNI olsun diye.
    FastAPI endpoint'ine (/api/history/{icao24}/csv) HTTP ile gitmek yerine
    _query_history_df'i DOGRUDAN cagiriyoruz -- ayni process, ekstra ag
    gidis-donusune gerek yok."""
    if not icao24:
        return dash.no_update

    HOURS_DEFAULT = 24
    tz = _resolve_tz(tz_name)
    params = {"hours": HOURS_DEFAULT}
    if None not in (start_day, start_hour, end_day, end_hour):
        start_utc = datetime.fromisoformat(start_day).replace(hour=start_hour, tzinfo=tz) \
                            .astimezone(timezone.utc)
        end_utc = datetime.fromisoformat(end_day).replace(hour=end_hour, tzinfo=tz) \
                          .astimezone(timezone.utc)
        params = {"start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}

    df, err = _query_history_df(icao24, **params)
    if err or df is None or df.empty:
        return dash.no_update

    csv_str = df.rename(columns={"_time": "timestamp_utc"}).to_csv(index=False)
    return dcc.send_string(csv_str, filename=f"{icao24}_history.csv")


@app_dash.callback(
    [Output("replay-open", "data"),
     Output("replay-mode", "data", allow_duplicate=True),
     Output("replay-playing", "data", allow_duplicate=True)],
    [Input("replay-btn", "n_clicks"), Input("close-replay-btn", "n_clicks")],
    State("replay-open", "data"),
    prevent_initial_call=True,
)
def toggle_replay_open(open_clicks, close_clicks, is_open):
    """toggle_stats_open/toggle_emergency_open ile AYNI ac/kapa deseni.

    ONEMLI (2026-07-09, kullanici bulgusu -- "bazen ozellik degistirince
    veri gelmiyor, F5 gerekiyor"): bu callback ONCEDEN sadece paneli
    (replay-open) kapatiyordu -- replay-mode SADECE "Canliya Don"
    (replay-live-btn -> exit_replay_mode) ile False'a donuyordu. Kullanici
    paneli "x" ile (close-replay-btn) veya replay-btn'e TEKRAR basarak
    kapatirsa, replay-mode SESSIZCE True KALIYORDU -- update_map bunu
    gordugu surece (asagida, erken "return dash.no_update" blogu) canli
    tick'ler 15sn'de bir gelmeye devam etse bile haritayi hic
    guncellemiyordu, harita donuk kaliyordu. Sadece F5 (Store'un
    varsayilan False'a donmesiyle) duzeltiyordu. Simdi panel HANGI
    yoldan kapanirsa kapansin (x, toggle veya "Canliya Don"),
    replay-mode/replay-playing da BIRLIKTE False'a donuyor.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "close-replay-btn.n_clicks":
        return False, False, False
    if trigger == "replay-btn.n_clicks":
        new_open = not is_open
        if new_open:
            return True, dash.no_update, dash.no_update
        return False, False, False
    return dash.no_update, dash.no_update, dash.no_update


@app_dash.callback(
    Output("replay-panel", "style"),
    Input("replay-open", "data"),
)
def show_replay_panel(is_open):
    style = dict(REPLAY_PANEL_BASE)
    style["display"] = "block" if is_open else "none"
    return style


@app_dash.callback(
    [Output("replay-start-day", "value"), Output("replay-start-hour", "value"),
     Output("replay-end-day", "value"), Output("replay-end-hour", "value")],
    Input("replay-open", "data"),
    [State("replay-start-day", "value"), State("replay-start-hour", "value"),
     State("replay-end-day", "value"), State("replay-end-hour", "value"),
     State("timezone-setting", "data")],
    prevent_initial_call=True,
)
def set_replay_default_range(is_open, start_day, start_hour, end_day, end_hour, tz_name):
    """ONEMLI (canli testte YAKALANAN gercek kullanim sorunu): panel ilk
    acildiginda 4 dropdown da BOS geliyordu -- kullanici hicbirine
    dokunmadan "Yükle"ye basinca load_replay_data sessizce ("Bu aralıkta
    veri bulunamadı" -- kucuk, 11px gri yazi, kolayca gozden kaciyor)
    hicbir sey yapmiyordu, "hic calismiyor" gibi gorunuyordu. Artik panel
    her acildiginda (VE 4 alan hala bossa -- kullanicinin KENDI secimini
    EZMIYORUZ) "son 1 saat" gibi makul bir varsayilan aralik otomatik
    dolduruluyor, boylece hic dokunmadan "Yükle" calisir."""
    if not is_open:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if None not in (start_day, start_hour, end_day, end_hour):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    tz = _resolve_tz(tz_name)
    now = datetime.now(tz)
    one_hour_ago = now - timedelta(hours=1)
    return (one_hour_ago.date().isoformat(), one_hour_ago.hour,
            now.date().isoformat(), now.hour)


@app_dash.callback(
    [Output("replay-data", "data"), Output("replay-index", "data"),
     Output("replay-playing", "data"), Output("replay-mode", "data"),
     Output("replay-feedback", "children"), Output("replay-progress", "data")],
    Input("replay-load-btn", "n_clicks"),
    [State("replay-start-day", "value"), State("replay-start-hour", "value"),
     State("replay-end-day", "value"), State("replay-end-hour", "value"),
     State("timezone-setting", "data"), State("language-setting", "data")],
    prevent_initial_call=True,
)
def load_replay_data(n_clicks, start_day, start_hour, end_day, end_hour, tz_name, lang):
    """"Yükle" butonu -- secilen gun/saat araligini get_replay()'e (AYNI
    process icinde, HTTP round-trip OLMADAN -- bkz. download_history_csv'deki
    ayni yaklasim) gecirip donen kareleri Store'a koyar. Basariyla yuklenirse
    replay-mode'u ACAR (update_map artik kendi cizimini durdurur, bkz. o
    callback), YUKLENEMEZSE (4 dropdown'dan biri bos VEYA aralikta veri
    yoksa) replay-mode'a hic DOKUNMAZ -- bos bir sonuc canli haritayi
    donduryormus gibi YANILTMASIN. replay-progress (hiz kontrolundeki
    kesirli ilerleme, bkz. advance_replay_tick) her yeni yuklemede
    sifirlanir -- bir onceki oturumdan kalma kesir yeni veriyle karismasin."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    empty = {"steps": [], "frames": []}
    if None in (start_day, start_hour, end_day, end_hour):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
            t["replay_no_data"], dash.no_update

    tz = _resolve_tz(tz_name)
    start_utc = datetime.fromisoformat(start_day).replace(hour=start_hour, tzinfo=tz) \
                        .astimezone(timezone.utc)
    end_utc = datetime.fromisoformat(end_day).replace(hour=end_hour, tzinfo=tz) \
                      .astimezone(timezone.utc)
    if end_utc <= start_utc:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
            t["replay_no_data"], dash.no_update

    result = get_replay(start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
    steps = result.get("steps") or []
    if not steps:
        return empty, 0, False, False, t["replay_no_data"], 0.0

    return result, 0, False, True, t["replay_loaded_label"].format(n=len(steps)), 0.0


@app_dash.callback(
    Output("replay-playing", "data", allow_duplicate=True),
    Input("replay-play-btn", "n_clicks"),
    [State("replay-playing", "data"), State("replay-data", "data")],
    prevent_initial_call=True,
)
def toggle_replay_play(n_clicks, is_playing, data):
    if not (data and data.get("steps")):
        return dash.no_update
    return not is_playing


@app_dash.callback(
    Output("replay-play-btn", "children"),
    Input("replay-playing", "data"),
)
def update_replay_play_label(is_playing):
    return "⏸" if is_playing else "▶"


@app_dash.callback(
    Output("replay-tick", "disabled"),
    Input("replay-playing", "data"),
)
def sync_replay_interval(is_playing):
    return not is_playing


@app_dash.callback(
    Output("replay-speed", "data"),
    Input("replay-speed-dropdown", "value"),
)
def set_replay_speed(v):
    return v or 1


@app_dash.callback(
    [Output("replay-index", "data", allow_duplicate=True),
     Output("replay-progress", "data")],
    Input("replay-tick", "n_intervals"),
    [State("replay-index", "data"), State("replay-data", "data"),
     State("replay-speed", "data"), State("replay-progress", "data")],
    prevent_initial_call=True,
)
def advance_replay_tick(n, index, data, speed, progress):
    """Her tick'te (sabit 2sn -- bkz. replay-tick tanimi) "progress"a
    HIZ kadar eklenir, TAM SAYI kismi kadar adim ilerlenir, kalan kesir
    bir sonraki tick'e TASINIR. Boylece:
      - 0.5x gibi 1'den kucuk hizlar duzgun calisir (2 tick'te 1 adim).
      - 4x/8x gibi yuksek hizlarda render/InfluxDB istek SIKLIGI
        DEGISMEZ (tick hala 2sn'de bir) -- sadece TEK seferde daha fazla
        adim atlanir. Boylece hiz kontrolu, daha once "çok yavaş" sorununa
        yol acan "her saniye render" sorununu GERI GETIRMEZ.
    Sona gelince BASA SARAR (bkz. eski yorum -- demo/sunum icin surekli
    donen bir goruntu, "Canlıya Dön" ile istedigi an cikilabiliyor)."""
    steps = (data or {}).get("steps") or []
    if not steps:
        return dash.no_update, dash.no_update
    speed = speed or 1
    progress = (progress or 0.0) + speed
    advance = int(progress)
    progress -= advance
    if advance <= 0:
        return dash.no_update, progress
    return (index + advance) % len(steps), progress


@app_dash.callback(
    [Output("replay-mode", "data", allow_duplicate=True),
     Output("replay-playing", "data", allow_duplicate=True)],
    Input("replay-live-btn", "n_clicks"),
    prevent_initial_call=True,
)
def exit_replay_mode(n_clicks):
    return False, False


# ONEMLI (kullanici geri bildirimi -- "1de takılı kalıyor, durdurunca 8e
# geçiyor"): update_map'teki AYNI yaris durumu (bkz. _altitude_map_seq
# yorumu) burada da var -- replay-tick her 2sn'de bir "replay-index"i
# ilerletiyor, HER index degisimi render_replay_frame'i tetikleyip
# /api/replay_frame'e YENI bir HTTP istegi atiyor. Bu istek 2sn'den UZUN
# surerse (yavas InfluxDB sorgusu / agir kare), bir SONRAKI tick zaten
# YENI bir istek baslatir -- BIRDEN FAZLA istek CAKISIR ve YANITLAR
# GONDERILDIKLERI SIRAYLA DEGIL TAMAMLANDIKLARI SIRAYLA doner. Sonuc:
# ekran ESKI bir karede "takili" kalir (surekli daha eski bir yanitla
# EZILIYOR), oynatmayi DURDURUNCA (yeni istek baslamayi kesince) en son
# baslatilan istek nihayet doner ve GORUNURDE bir anda ileri "atlar."
# AYNI cozum: her cagri kendi sira numarasini alir, sonucu YAZMADAN once
# hala EN SON BASLATILAN cagri oldugunu dogrular -- degilse atar.
_replay_frame_seq_lock = threading.Lock()
_replay_frame_latest_seq = 0


@app_dash.callback(
    [Output("aircraft-raw", "data", allow_duplicate=True),
     Output("replay-step-label", "children")],
    [Input("replay-index", "data"), Input("replay-data", "data")],
    State("language-setting", "data"),
    prevent_initial_call=True,
)
def render_replay_frame(index, data, lang):
    global _replay_frame_latest_seq
    with _replay_frame_seq_lock:
        _replay_frame_latest_seq += 1
        my_seq = _replay_frame_latest_seq
    """index'teki adimin ucak listesini AYRI/hafif bir istekle
    (/api/replay_frame) ceker, update_map'in ciktisiyla AYNI GeoJSON-Point/
    properties semasinda "aircraft-raw"a yazar -- boylece haritanin geri
    kalani (clientside geometri hesabi, canvas render, tooltip) HICBIR SEY
    degistirmeden canli/replay ikisinde de ayni sekilde calisir.

    ONEMLI: eskiden TUM kareler "replay-data" Store'unda onceden
    yuklenmisti -- kuresel trafikte bu Store'un kendisi ~30MB'a cikip
    tarayicida pratik olarak calismiyordu (bkz. get_replay docstring'i).
    Artik SADECE o anki adimin verisi, update_map'in /api/flights'i her
    tick'te cekmesiyle AYNI boyut/hizda cekiliyor.

    callsign InfluxDB'de YOK (bkz. get_replay docstring'i) -- icao24
    fallback olarak kullanilir, canli haritadaki "callsign or
    icao.upper()" ile ayni desen."""
    t = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])
    steps = (data or {}).get("steps") or []
    step_sec = (data or {}).get("step_sec", 30)
    if not steps or index is None or not (0 <= index < len(steps)):
        return dash.no_update, ""

    try:
        frame = requests.get("http://localhost:8000/api/replay_frame",
                             params={"ts": steps[index], "step_sec": step_sec},
                             timeout=5).json()
        if isinstance(frame, dict):  # {"error": ...}
            frame = []
    except Exception:
        frame = []

    features = []
    for f in frame:
        icao = f.get("icao24", "")
        lat, lon = f.get("lat"), f.get("lon")
        if lat is None or lon is None:
            continue
        track = f.get("track") or 0
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": dict(
                icao24=icao, callsign=icao.upper(), color=_altitude_to_color(f.get("alt")), opacity=0.9,
                track=track, subtitle=f"{icao.upper()}  ·  REPLAY",
                alt_text=(f"{f.get('alt'):.0f} m" if f.get("alt") is not None else "—"),
                speed_text=(f"{f.get('velocity'):.0f} m/s"
                           if f.get("velocity") is not None else "—"),
                track_text=f"{track:.0f}°",
                vspeed_text="—",
                lbl_alt=t["tooltip_alt"], lbl_speed=t["tooltip_speed"],
                lbl_track=t["tooltip_track"], lbl_vspeed=t["tooltip_vspeed"],
                signal_age_text=None, lbl_signal_age=t["tooltip_signal_age"],
            ),
        })

    raw_data = {"type": "FeatureCollection", "features": features}
    ts_display = steps[index][:19].replace("T", " ")
    label = f"{index + 1}/{len(steps)}  ·  {ts_display}"

    # ONEMLI: yukaridaki _replay_frame_seq aciklamasina bkz -- bu istek
    # surerken DAHA YENI bir tick/kare zaten baslamissa, bu sonuc ARTIK
    # BAYAT -- yazmadan at, ekranin eski bir kareye "geri sicramasina"
    # izin verme.
    if my_seq != _replay_frame_latest_seq:
        return dash.no_update, dash.no_update

    return raw_data, label


# ONEMLI: bu callback CLIENTSIDE (JS, tarayicida calisir, Python'a HIC
# UGRAMAZ) -- ucak sekli/boyutu (yon + zoom'a gore olcek) burada
# hesaplanip "aircraft-geojson"a yaziliyor. Iki tetikleyicisi var:
# "aircraft-raw" (her tick'te yeni veri) VEYA "map"."zoom" (kullanici
# yakinlasip/uzaklastiginda) -- HANGISI degisirse degissin, HER ZAMAN
# o anki GERCEK zoom'u kullanir, boylece "eski referans zoom'a donme"
# sorunu olmaz VE ag gidis-donusu olmadigi icin gecikme SIFIRDIR (bkz.
# update_map yorumundaki iki basarisiz Python-tarafi deneme). Matematik
# Python'daki (kaldirilan) _rotated_aircraft_polygon() ile BIREBIR AYNI
# -- sadece JS'e tasindi.
app_dash.clientside_callback(
    """
    function(rawData, zoom, airlineCodes, lang) {
        if (!rawData || !rawData.features) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        const z = (zoom === null || zoom === undefined) ? 5 : zoom;
        // ONEMLI (firma filtresi -- bkz. AIRLINE_PREFIXES Python yorumu):
        // secili firma YOKSA (bos liste) hicbir sey elenmez -- tabloda
        // olmayan cagri kodlari (askeri/genel havacilik/taninmayan) da
        // varsayilan olarak HEP gorunur kalir. Bu TAMAMEN tarayicida
        // calisiyor -- update_map'e (sunucu) HICBIR yeni istek gitmiyor.
        const wanted = (airlineCodes && airlineCodes.length) ? new Set(airlineCodes) : null;
        const sourceFeatures = wanted
            ? rawData.features.filter(function(f) {
                const cs = (f.properties && f.properties.callsign) || "";
                return wanted.has(cs.trim().slice(0, 3).toUpperCase());
            })
            : rawData.features;
        const features = sourceFeatures.map(function(f) {
            const p = f.properties;
            const lon = f.geometry.coordinates[0];
            const lat = f.geometry.coordinates[1];
            const heading = p.track || 0;

            const latRad = lat * Math.PI / 180;
            const mpp = 156543.03392 * Math.cos(latRad) / Math.pow(2, z);
            // ONEMLI: 9 -> 11px hedef "yaricap" -- kullanici geri bildirimi
            // "uçaklara tıklamak zor" idi, biraz daha buyuk hem daha
            // tiklanabilir hem eski DOM ikonun (~11px yari-boyut) gercek
            // olcusune daha yakin.
            const unitM = 11 * mpp;

            // ONEMLI: eskiden 4 noktali basit ok/dart sekliydi -- kullanici
            // istegi uzerine ESKI DOM/SVG ucak siluetiyle (burun+kanatlar+
            // kuyruk) AYNI oranlarda 12 noktali sekle cevrildi (asagidaki
            // oranlar, orijinal SVG path'in 24x24 viewBox'ndan MERKEZE
            // GORE normalize edilerek turetildi -- bkz. proje sohbet
            // gecmisi). Daha genis kanatlar (once ±0.55 iken simdi ±0.96)
            // AYRICA tiklama alanini da buyutuyor.
            const localPts = [
                [0.0000 * unitM, 0.9565 * unitM],    // burun
                [0.2609 * unitM, -0.0870 * unitM],   // burun-sag govde
                [0.9565 * unitM, -0.5217 * unitM],   // sag kanat ucu
                [0.2609 * unitM, -0.3478 * unitM],   // sag govde-kuyruk
                [0.2609 * unitM, -0.7391 * unitM],   // sag kuyruk govde
                [0.5652 * unitM, -0.9130 * unitM],   // sag kuyruk kanadi
                [0.0000 * unitM, -0.7826 * unitM],   // kuyruk merkez
                [-0.5652 * unitM, -0.9130 * unitM],  // sol kuyruk kanadi
                [-0.2609 * unitM, -0.7391 * unitM],  // sol kuyruk govde
                [-0.2609 * unitM, -0.3478 * unitM],  // sol govde-kuyruk
                [-0.9565 * unitM, -0.5217 * unitM],  // sol kanat ucu
                [-0.2609 * unitM, -0.0870 * unitM],  // burun-sol govde
            ];

            const hRad = heading * Math.PI / 180;
            const cosH = Math.cos(hRad), sinH = Math.sin(hRad);
            const cosLat = Math.max(Math.abs(Math.cos(latRad)), 1e-6);

            const ring = localPts.map(function(pt) {
                const east = pt[0], north = pt[1];
                const rEast = east * cosH + north * sinH;
                const rNorth = -east * sinH + north * cosH;
                const dLat = rNorth / 111320.0;
                const dLon = rEast / (111320.0 * cosLat);
                return [lon + dLon, lat + dLat];
            });
            ring.push(ring[0]);

            return {
                type: 'Feature',
                geometry: {type: 'Polygon', coordinates: [ring]},
                properties: p,
            };
        });
        // ONEMLI (kullanici geri bildirimi -- "firma filtresi bu sayiyi
        // hic etkilemiyor"): status cubugundaki "toplam" Python'da
        // hesaplaniyor (hicbir filtre yok) ama firma filtresi SADECE burada
        // (JS) uygulandigi icin "kalan/gosterilen" sayiyi da BURADA
        // yazmak zorundayiz -- sourceFeatures.length TAM OLARAK sivil/
        // askeri/yerde/irtifa (sunucuda) + firma (burada) filtrelerinin
        // HEPSI uygulandiktan SONRAKI nihai sayidir.
        const shownLabel = (lang === 'en') ? 'shown' : 'gösteriliyor';
        const shownText = sourceFeatures.length + ' ' + shownLabel;
        return [{type: 'FeatureCollection', features: features}, shownText];
    }
    """,
    [Output("aircraft-geojson", "data"), Output("status-shown", "children")],
    [Input("aircraft-raw", "data"), Input("map", "zoom"),
     Input("airline-filter-dropdown", "value"), Input("language-setting", "data")],
)


if __name__ == "__main__":
    threading.Thread(target=_run_api, daemon=True).start()
    time.sleep(2)
    print("FastAPI hazir (port 8000)")
    print("Dash başlıyor: http://localhost:8050")
    app_dash.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
