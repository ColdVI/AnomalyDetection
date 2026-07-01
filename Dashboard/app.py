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
P = {"background": "#0f0f19", "padding": "12px", "borderRadius": "8px", "margin": "6px"}

app_dash.layout = html.Div(style={"backgroundColor": "#07070e", "fontFamily": "sans-serif",
                                   "color": "#c8d0e0", "minHeight": "100vh"}, children=[
    html.Div(style={"textAlign": "center", "padding": "14px"}, children=[
        html.H2("✈️ ADS-B Local Dashboard", style={"color": "#00b4d8", "margin": 0}),
        html.Div(id="status", style={"color": "#666", "fontSize": "13px"}),
    ]),
    dcc.Interval(id="tick", interval=15000, n_intervals=0),

    html.Div(style={"display": "flex", "gap": "8px", "padding": "0 8px"}, children=[
        html.Div(style={**P, "flex": "2"}, children=[
            html.H4("Canlı Konum (Redis)", style={"color": "#90e0ef", "marginTop": 0}),
            dl.Map(
                id="map",
                center=[39.0, 35.0], zoom=5,
                style={"height": "420px", "width": "100%", "borderRadius": "6px"},
                children=[
                    dl.TileLayer(
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                        attribution="© OpenStreetMap"
                    ),
                    dl.LayerGroup(id="aircraft-layer"),
                ],
            ),
        ]),
        html.Div(style={**P, "flex": "1", "minWidth": "260px"}, children=[
            html.H4("Model Alarmları", style={"color": "#e63946", "marginTop": 0}),
            html.Div(id="alerts", style={"marginBottom": "12px"}),
            html.Hr(style={"borderColor": "#222"}),
            html.H4("Geçmiş Sorgula (InfluxDB, 7 gün)", style={"color": "#f77f00", "marginTop": 0}),
            dcc.Dropdown(id="aircraft-select", placeholder="Uçak seç (ICAO24)",
                         style={"color": "#000"}),
            html.Div(style={"marginTop": "10px"}, children=[
                html.Label("Kaç saat geriye?", style={"fontSize": "12px"}),
                dcc.Slider(id="hours-slider", min=1, max=168, step=1, value=24,
                          marks={1: "1s", 24: "1g", 72: "3g", 168: "7g"}),
            ]),
        ]),
    ]),

    html.Div(style={**P, "margin": "6px 8px"}, children=[
        html.H4("Seçili Uçak — Canlı Bilgi", style={"color": "#00b4d8", "marginTop": 0}),
        html.Div(id="live-aircraft-panel", children=[
            html.Div("Haritada bir uçağa tıklayın ya da sağdaki listeden seçin.",
                     style={"color": "#666", "fontSize": "13px"})
        ]),
    ]),

    html.Div(style={**P, "margin": "6px 8px"}, children=[
        html.H4("Seçili Uçağın Geçmişi", style={"color": "#90e0ef", "marginTop": 0}),
        dcc.Graph(id="history-chart", style={"height": "280px"},
                  config={"displayModeBar": False}),
    ]),

    html.Div(style={**P, "margin": "6px 8px"}, children=[
        html.H4("Trafik Hacmi (son 24 saat, benzersiz uçak/saat)",
                style={"color": "#90e0ef", "marginTop": 0}),
        dcc.Graph(id="traffic-chart", style={"height": "220px"},
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
     Output("aircraft-select", "options"), Output("alerts", "children")],
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

    ts = datetime.now().strftime("%H:%M:%S")
    status = f"{ts} | {len(flights)} aktif uçuş | {len(alerts)} alarm"
    options = [{"label": f.get("icao24", ""), "value": f.get("icao24", "")} for f in flights]

    alert_divs = [html.Div(
        f"🔴 {a.get('icao24','?')}  {a.get('alert_type','anomali')}",
        style={"color": "#e63946", "padding": "5px 8px",
               "borderLeft": "3px solid #e63946",
               "marginBottom": "4px", "fontSize": "13px"}
    ) for a in alerts] or [html.Div("Model henüz alarm üretmedi",
                                     style={"color": "#666", "fontSize": "13px"})]

    return markers, status, options, alert_divs


@app_dash.callback(
    Output("aircraft-select", "value"),
    Input({"type": "aircraft-marker", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def select_from_marker_click(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return dash.no_update
    triggered_id_str = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    triggered_id = json.loads(triggered_id_str)
    icao24 = triggered_id.get("index")
    return icao24 if icao24 else dash.no_update


@app_dash.callback(
    Output("live-aircraft-panel", "children"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "value")]
)
def update_live_panel(n, icao24):
    if not icao24:
        return html.Div("Haritada bir uçağa tıklayın ya da sağdaki listeden seçin.",
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
    ts_display = ts_raw[:19].replace("T", " ") if ts_raw else "—"

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
        ("Kategori", match.get("category", "") or "—"),
        ("Son Güncelleme", ts_display),
    ]

    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
        "gap": "8px",
    }, children=[
        html.Div([
            html.Div(label, style={"fontSize": "11px", "color": "#666"}),
            html.Div(str(value), style={"fontSize": "16px", "color": "#00b4d8",
                                         "fontWeight": "500"}),
        ], style={"background": "#161625", "padding": "8px 10px", "borderRadius": "6px"})
        for label, value in fields
    ])


@app_dash.callback(
    Output("history-chart", "figure"),
    [Input("tick", "n_intervals"), Input("aircraft-select", "value"),
     Input("hours-slider", "value")]
)
def update_history(n, icao24, hours):
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#0f0f19", plot_bgcolor="#0f0f19",
                      font=dict(color="#c8d0e0"), margin=dict(t=20, b=20))

    if not icao24:
        fig.add_annotation(text="Bir uçak seç", showarrow=False,
                           font=dict(color="#666"))
        return fig

    try:
        data = requests.get(f"http://localhost:8000/api/history/{icao24}",
                            params={"hours": hours}, timeout=10).json()
    except Exception:
        data = []

    if not data or isinstance(data, dict):
        fig.add_annotation(text="Veri bulunamadı (henüz geçmiş birikmemiş olabilir)",
                           showarrow=False, font=dict(color="#666"))
        return fig

    df = pd.DataFrame(data)
    df["_time"] = pd.to_datetime(df["_time"])

    fig.add_trace(go.Scatter(x=df["_time"], y=df["alt"], name="İrtifa (m)",
                             line=dict(color="#00b4d8"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["_time"], y=df["velocity"], name="Hız (m/s)",
                             line=dict(color="#f77f00"), yaxis="y2"))

    fig.update_layout(
        yaxis=dict(title="İrtifa (m)", side="left"),
        yaxis2=dict(title="Hız (m/s)", side="right", overlaying="y"),
        legend=dict(orientation="h"),
    )
    return fig


@app_dash.callback(
    Output("traffic-chart", "figure"),
    Input("tick", "n_intervals")
)
def update_traffic(n):
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#0f0f19", plot_bgcolor="#0f0f19",
                      font=dict(color="#c8d0e0"), margin=dict(t=10, b=10))
    try:
        data = requests.get("http://localhost:8000/api/traffic_stats",
                            params={"hours": 24}, timeout=10).json()
    except Exception:
        data = []

    if not data or isinstance(data, dict):
        fig.add_annotation(text="Henüz yeterli veri birikmedi",
                           showarrow=False, font=dict(color="#666"))
        return fig

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    fig.add_trace(go.Bar(x=df["time"], y=df["unique_aircraft"],
                         marker_color="#00b4d8"))
    return fig


if __name__ == "__main__":
    print("Dash başlıyor: http://localhost:8050")
    app_dash.run(host="0.0.0.0", port=8050, debug=False, use_reloader=False)
