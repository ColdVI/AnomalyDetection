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
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import redis
import requests
import uvicorn
import dash
import dash_leaflet as dl
from dash import Dash, dcc, html, Output, Input, State, ALL
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient

TR_TZ = ZoneInfo("Europe/Istanbul")  # Turkiye UTC+3, DST yok (2016'dan beri sabit)

# ADS-B/Mode-S emitter kategori kodlari (standart, ilk harf sinif, rakam alt tip)
CATEGORY_LABELS = {
    "A0": "Bilinmiyor", "A1": "Hafif uçak", "A2": "Küçük uçak",
    "A3": "Büyük uçak", "A4": "Büyük uçak (yüksek vorteks)", "A5": "Ağır uçak",
    "A6": "Yüksek performans", "A7": "Helikopter",
    "B0": "Bilinmiyor", "B1": "Planör", "B2": "Balon/Zeplin",
    "B3": "Paraşütçü", "B4": "Ultralight/Yamaç paraşütü",
    "B6": "İHA/Drone", "B7": "Uzay aracı",
    "C0": "Bilinmiyor", "C1": "Yer taşıtı (acil)", "C2": "Yer taşıtı (servis)",
    "C3": "Sabit engel", "C4": "Engel kümesi", "C5": "Hat engeli",
}

# ADS-B acil durum kodlari (Mode S emergency/priority status)
EMERGENCY_LABELS = {
    "none": None,  # normal durum, gosterme
    "general": "GENEL ACİL DURUM",
    "lifeguard": "SAĞLIK ACİL DURUMU",
    "minfuel": "YAKIT KRİTİK",
    "nordo": "RADYO ARIZASI",
    "unlawful": "KAÇIRMA (HİJACK)",
    "downed": "DÜŞTÜ/İNİŞ ZORUNLU",
    "reserved": "REZERVE KOD",
}

TOKEN_FILE = Path("influx_token.txt")
INFLUX_HOST = "http://localhost:8086"
INFLUX_ORG = "iha-org"
INFLUX_BUCKET = "adsb-history"

if not TOKEN_FILE.exists():
    raise SystemExit("influx_token.txt bulunamadi. Once setup_local_windows.py calistir.")
INFLUX_TOKEN = TOKEN_FILE.read_text().strip()

# ------------------------------------------------------------------ FastAPI --

app_api = FastAPI(title="ADS-B Local API")
app_api.add_middleware(CORSMiddleware, allow_origins=["*"],
                        allow_methods=["*"], allow_headers=["*"])

_rpool = redis.ConnectionPool(host="localhost", port=6379, db=0,
                               decode_responses=True, protocol=2)
_influx = InfluxDBClient(url=INFLUX_HOST, token=INFLUX_TOKEN, org=INFLUX_ORG)
_query_api = _influx.query_api()


def _get_flights():
    r = redis.Redis(connection_pool=_rpool)
    out = []
    for icao in r.smembers("iha:active_flights"):
        raw = r.get(f"iha:state:{icao}")
        if raw:
            out.append(json.loads(raw))
    return sorted(out, key=lambda x: x.get("icao24", ""))


@app_api.get("/api/flights")
def get_flights():
    return _get_flights()


@app_api.get("/api/alerts")
def get_alerts():
    r = redis.Redis(connection_pool=_rpool)
    return [json.loads(a) for a in r.lrange("iha:recent_alerts", 0, 9)]


@app_api.get("/api/route/{callsign}")
def get_route(callsign: str):
    """adsb.lol/ADS-B protokolu kalkis/varis tasimiyor -- adsbdb.com'un
    ucretsiz, topluluk tarafindan tutulan callsign->rota veritabanini
    kullaniyoruz. Rota nadiren degistigi icin Redis'te 12 saat cache'liyoruz,
    her secimde dis API'ye vurmayalim."""
    callsign = callsign.strip().upper()
    if not callsign:
        return {"found": False}

    r = redis.Redis(connection_pool=_rpool)
    cache_key = f"iha:route:{callsign}"
    cached = r.get(cache_key)
    if cached is not None:
        return json.loads(cached)

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
                    "dest_name": dest.get("name"),
                    "dest_iata": dest.get("iata_code"),
                    "dest_city": dest.get("municipality"),
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
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution="© OpenStreetMap"
            ),
            dl.LayerGroup(id="flight-path-layer"),
            dl.LayerGroup(id="aircraft-layer"),
        ],
    ),

    # ------------------------------------------- Durum cubugu (overlay) --
    html.Div(id="status", style={
        "position": "absolute", "top": "12px", "left": "50%",
        "transform": "translateX(-50%)",
        "backgroundColor": "rgba(15,15,25,0.85)",
        "padding": "6px 16px", "borderRadius": "20px",
        "fontSize": "13px", "color": "#c8d0e0", "zIndex": 500,
        "pointerEvents": "none",  # altindaki haritayi engellemesin
    }),

    # ------------------------------------------- Alarm paneli (overlay) --
    html.Div(id="alerts-overlay", style={
        "position": "absolute", "top": "12px", "right": "12px",
        "width": "260px", "maxHeight": "220px", "overflowY": "auto",
        "backgroundColor": "rgba(15,15,25,0.92)",
        "borderRadius": "8px", "padding": "10px", "zIndex": 500,
    }, children=[
        html.H4("Model Alarmları", style={"color": "#e63946", "margin": "0 0 8px 0",
                                           "fontSize": "13px"}),
        html.Div(id="alerts"),
    ]),

    # ---------------------------------- Sol kayan panel (varsayilan gizli) --
    html.Div(id="left-panel", style={**LEFT_PANEL_BASE, "transform": "translateX(-100%)"},
             children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "marginBottom": "12px"}, children=[
            html.H3("Uçak Bilgisi", style={"color": "#00b4d8", "margin": 0,
                                            "fontSize": "18px"}),
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
        html.H4("Geçmiş (son 24 saat)", style={"color": "#90e0ef", "marginTop": 0,
                                                 "fontSize": "13px"}),
        dcc.Graph(id="history-chart", style={"height": "200px"},
                  config={"displayModeBar": False}),
    ]),
])


def _airplane_icon(heading: float, color: str) -> dict:
    """Ucus yonune (heading, 0-360 derece, kuzeyden saat yonunde) gore
    CSS ile dondurulmus SVG ucak ikonu. Ikon 0 derecede kuzeye (yukari)
    bakacak sekilde cizildigi icin heading degeri dogrudan rotate()
    aciysa kullanilabiliyor, ekstra offset gerekmiyor.

    ONEMLI: bu dict, dl.DivMarker'in "iconOptions" parametresine veriliyor
    (dl.Marker + icon dict DEGIL -- o kombinasyon calismiyor, marker hic
    render olmuyordu). DivMarker, HTML/CSS tabanli ikonlar icin ayri,
    ozel bir bilesen."""
    svg = f'''
    <div style="transform: rotate({heading}deg); transform-origin: center;
                width: 22px; height: 22px;">
      <svg width="22" height="22" viewBox="0 0 24 24">
        <path d="M12 1 L15 13 L23 18 L15 16 L15 20.5 L18.5 22.5 L12 21
                 L5.5 22.5 L9 20.5 L9 16 L1 18 L9 13 Z"
              fill="{color}" stroke="#07070e" stroke-width="0.5"/>
      </svg>
    </div>
    '''
    return {"html": svg, "className": "", "iconSize": [22, 22], "iconAnchor": [11, 11]}


@app_dash.callback(
    [Output("aircraft-layer", "children"), Output("status", "children"),
     Output("alerts", "children")],
    Input("tick", "n_intervals")
)
def update_map(n):
    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []
    try:
        alerts = requests.get("http://localhost:8000/api/alerts", timeout=3).json()
    except Exception:
        alerts = []

    alert_icaos = {a.get("icao24") for a in alerts}

    markers = []
    for f in flights:
        icao = f.get("icao24", "")
        color = "#e63946" if icao in alert_icaos else "#00b4d8"
        markers.append(dl.DivMarker(
            position=[f.get("lat", 39), f.get("lon", 35)],
            iconOptions=_airplane_icon(f.get("track") or 0, color),
            id={"type": "aircraft-marker", "index": icao},
            children=[
                dl.Tooltip(
                    f"{icao} | {f.get('callsign','').strip() or '—'} | "
                    f"alt={f.get('alt',0):.0f}m | "
                    f"{f.get('velocity') if f.get('velocity') is not None else 0:.0f}m/s"
                ),
            ],
        ))

    ts = datetime.now(TR_TZ).strftime("%H:%M:%S")
    status = f"{ts} | {len(flights)} aktif uçuş | {len(alerts)} alarm"

    alert_divs = [html.Div(
        f"🔴 {a.get('icao24','?')}  {a.get('alert_type','anomali')}",
        style={"color": "#e63946", "padding": "5px 8px",
               "borderLeft": "3px solid #e63946",
               "marginBottom": "4px", "fontSize": "13px"}
    ) for a in alerts] or [html.Div("Model henüz alarm üretmedi",
                                     style={"color": "#666", "fontSize": "13px"})]

    return markers, status, alert_divs


@app_dash.callback(
    Output("aircraft-select", "data"),
    [Input({"type": "aircraft-marker", "index": ALL}, "n_clicks"),
     Input("close-panel-btn", "n_clicks")],
    prevent_initial_call=True
)
def select_or_close(marker_clicks, close_clicks):
    """Ucak marker'ina tiklaninca secim ayarlar, kapatma (x) butonuna
    tiklaninca secimi temizler -- ikisi de ayni Output'u (aircraft-select)
    yazdigi icin tek callback'te birlestirildi (Dash coklu-callback ayni
    Output kisitini boyle asiyoruz, allow_duplicate'a gerek kalmadan)."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_prop_id = ctx.triggered[0]["prop_id"]
    if trigger_prop_id == "close-panel-btn.n_clicks":
        return None

    if not ctx.triggered[0]["value"]:
        return dash.no_update
    triggered_id_str = trigger_prop_id.rsplit(".", 1)[0]
    triggered_id = json.loads(triggered_id_str)
    icao24 = triggered_id.get("index")
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
    Output("flight-path-layer", "children"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data")]
)
def update_flight_path(n, icao24):
    """Secili ucagin son 1 saatlik konum gecmisini haritada cizgi olarak
    cizer. Mevcut /api/history endpoint'ini aynen kullaniyor (lat/lon zaten
    donuyordu), backend'e hic dokunmadan calisir."""
    if not icao24:
        return []
    try:
        data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                            params={"hours": 1}, timeout=5).json()
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


@app_dash.callback(
    Output("route-info", "children"),
    Input("aircraft-select", "data"),
)
def update_route_info(icao24):
    """Kalkis/varis bilgisi. SADECE secim degistiginde calisir (tick'e
    bagli degil) -- rota veritabani nadiren degistigi icin her 15 saniyede
    tekrar sorgulamaya gerek yok, hem dis API'yi hem Redis'i gereksiz
    yormayalim."""
    if not icao24:
        return None

    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []
    match = next((f for f in flights if f.get("icao24") == icao24), None)
    callsign = (match.get("callsign") or "").strip() if match else ""

    if not callsign:
        return html.Div("Çağrı kodu yok, rota sorgulanamıyor.",
                        style={"color": "#666", "fontSize": "12px"})

    try:
        route = requests.get(f"http://localhost:8000/api/route/{callsign}",
                             timeout=5).json()
    except Exception:
        route = {"found": False}

    if not route.get("found"):
        return html.Div(f"'{callsign}' için rota bilgisi bulunamadı "
                        f"(adsbdb.com veritabanında yok).",
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
    Input("aircraft-select", "data"),
)
def update_aircraft_info(icao24):
    """Ucak tipi/uretici/tescil/sahip bilgisi. SADECE secim degistiginde
    calisir -- icao24 hex sabit oldugu icin ekstra flights sorgusuna
    gerek yok, dogrudan yeni endpoint'e sorulur."""
    if not icao24:
        return None

    try:
        info = requests.get(f"http://localhost:8000/api/aircraft_info/{icao24}",
                            timeout=5).json()
    except Exception:
        info = {"found": False}

    if not info.get("found"):
        return html.Div("Uçak tipi/tescil bilgisi bulunamadı.",
                        style={"color": "#666", "fontSize": "12px"})

    rows = []
    if info.get("manufacturer") or info.get("type"):
        rows.append(f"{info.get('manufacturer','')} {info.get('type','')}".strip())
    if info.get("registration"):
        rows.append(f"Tescil: {info['registration']}")
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
        ] if rows else [html.Div("Detay yok", style={"color": "#666"})])
    ]

    if info.get("photo_thumb"):
        children.append(html.Img(src=info["photo_thumb"], style={
            "width": "100%", "borderRadius": "6px", "marginTop": "8px",
        }))

    return html.Div(children)


@app_dash.callback(
    Output("live-aircraft-panel", "children"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data")]
)
def update_live_panel(n, icao24):
    if not icao24:
        return html.Div("Haritada bir uçağa tıklayın.",
                        style={"color": "#666", "fontSize": "13px"})

    try:
        flights = requests.get("http://localhost:8000/api/flights", timeout=3).json()
    except Exception:
        flights = []

    match = next((f for f in flights if f.get("icao24") == icao24), None)
    if not match:
        return html.Div(f"{icao24} şu anda sinyal göndermiyor "
                        f"(kapsama alanından çıkmış olabilir).",
                        style={"color": "#e63946", "fontSize": "13px"})

    ts_raw = match.get("ts", "")
    if ts_raw:
        try:
            ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts_display = ts_dt.astimezone(TR_TZ).strftime("%H:%M:%S")
        except Exception:
            ts_display = ts_raw[:19].replace("T", " ")
    else:
        ts_display = "—"

    raw_category = match.get("category", "") or ""
    category_label = CATEGORY_LABELS.get(raw_category, raw_category or "—")

    squawk = match.get("squawk", "") or "—"
    emergency_code = match.get("emergency", "none")
    emergency_label = EMERGENCY_LABELS.get(emergency_code)

    fields = [
        ("ICAO24", match.get("icao24", "").upper()),
        ("Çağrı Kodu", match.get("callsign", "").strip() or "—"),
        ("Enlem", f"{match.get('lat', 0):.4f}°"),
        ("Boylam", f"{match.get('lon', 0):.4f}°"),
        ("İrtifa", f"{match.get('alt', 0):.0f} m"),
        ("Hız", f"{match.get('velocity'):.0f} m/s" if match.get('velocity') is not None else "—"),
        ("Yön", f"{match.get('track'):.0f}°" if match.get('track') is not None else "—"),
        ("Dikey Hız", f"{match.get('vertical_rate'):+.1f} m/s"
                      if match.get('vertical_rate') is not None else "—"),
        ("Kategori", category_label),
        ("Squawk", squawk),
        ("Son Güncelleme", ts_display),
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
        warning_text = emergency_label or f"ACİL DURUM SQUAWK: {squawk}"
        children.insert(0, html.Div(f"⚠ {warning_text}", style={
            "background": "#e63946", "color": "#fff", "padding": "8px 10px",
            "borderRadius": "6px", "marginBottom": "8px",
            "fontSize": "13px", "fontWeight": "600", "textAlign": "center",
        }))

    return html.Div(children)


@app_dash.callback(
    Output("history-chart", "figure"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "data")]
)
def update_history(n, icao24):
    HOURS_DEFAULT = 24  # sabit -- kaydirici kaldirildi, sonra geri eklenebilir

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
        fig.add_annotation(text="Veri bulunamadı", showarrow=False,
                           font=dict(color="#666"))
        return fig

    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"]).dt.tz_convert(TR_TZ)

    fig.add_trace(go.Scatter(x=df["_time"], y=df["alt"], name="İrtifa (m)",
                             line=dict(color="#00b4d8", width=1.5), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["_time"], y=df["velocity"], name="Hız (m/s)",
                             line=dict(color="#f77f00", width=1.5), yaxis="y2"))

    fig.update_layout(
        yaxis=dict(title="İrtifa (m)", side="left"),
        yaxis2=dict(title="Hız (m/s)", side="right", overlaying="y"),
        legend=dict(orientation="h", font=dict(size=9), y=1.15),
    )
    return fig


if __name__ == "__main__":
    print("Dash başlıyor: http://localhost:8050")
    app_dash.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
