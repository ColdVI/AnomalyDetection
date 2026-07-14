"""app.py'den cikarildi, adim 2 -- Dash layout agaci, build_layout() olarak.

_build_timezone_options/_build_signal_staleness_options/_format_altitude_tick
BILEREK burada degil app.py'de tanimli kaliyor -- callback'lerden de TEKRAR
cagriliyorlar (update_timezone_options/update_signal_staleness_options), bu
yuzden buraya tasinip app.py'nin geri import etmesi dairesel bagimliliga yol
acardi. Bunun yerine build_layout() bu fonksiyonlari/degerleri PARAMETRE
olarak aliyor -- app.py hala tek sahibi, layout.py sadece cagiriyor."""

from dash import dcc, html
import dash_leaflet as dl
from dash_extensions.javascript import assign

from texts import DEFAULT_LANGUAGE
from styles import (
    LEFT_PANEL_BASE, HISTORY_PANEL_BASE, SETTINGS_PANEL_BASE, STATS_PANEL_BASE,
    EMERGENCY_PANEL_BASE, REPLAY_PANEL_BASE, REPLAY_COLOR,
    HISTORY_CALC_BTN_STYLE, HISTORY_DOWNLOAD_BTN_STYLE, LANG_BTN_ACTIVE_STYLE,
    LANG_BTN_INACTIVE_STYLE, DEFAULT_MAP_STYLE, TILE_LAYERS, ALTITUDE_COLOR_STOPS,
    FILTER_BTN_CIVIL_ACTIVE_STYLE, FILTER_BTN_MILITARY_ACTIVE_STYLE,
    FILTER_BTN_GROUND_ACTIVE_STYLE,
)
from constants import (
    DEFAULT_TIMEZONE, DEFAULT_DATA_SOURCE, DATA_SOURCE_DEFS,
    DEFAULT_SIGNAL_STALENESS_SEC, HISTORY_HOUR_OPTIONS,
    AIRLINE_PREFIXES,
)

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
# CANVAS/WebGL tabanli render ile (DOM marker DEGIL) cozdugu goruldu
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


def build_layout(*, build_timezone_options, build_signal_staleness_options,
                  format_altitude_tick, alt_legend_n, altitude_legend_gradient):
    return html.Div(id="app-root", style={
        "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
        "overflow": "hidden", "backgroundColor": "#07070e",
        "fontFamily": "sans-serif", "color": "#c8d0e0",
    }, children=[

        # ONEMLI (15sn'nin kaynagi): bu deger OLCUME/hesaba dayanmiyor, bastan
        # beri sabit pragmatik bir varsayilan -- "kullanici icin yeterince
        # duyarli hissettirsin ama sunucuyu/Redis'i gereksiz yormasin" dengesi.
        # GERCEK veri uretim hizina (bkz. uav_producer.py SOURCES: adsblol=60sn;
        # opensky KIMLIK DOGRULAMASIZ 300sn ama credential'lar (.env) DOLUYSA --
        # bu ortamda oyle -- interval_override devreye girip OPENSKY_AUTH_INTERVAL'e,
        # yani 90sn'ye duser) KASITLI olarak BAGLI DEGIL -- coğu tick'te (orn.
        # adsblol'de 4 sorgudan 3'unde) Redis'te henuz YENI veri yok, bir onceki
        # cycle'in ayni sonucu geri donuyor. Bu ZARARSIZ (ucuz bir HTTP+Redis
        # round-trip'i) ama "tick" TEK amacli bir "ucus verisini yenile" sinyali
        # degil -- asagidaki 7 farkli callback (update_map, update_stats_chart,
        # manage_data_source, update_flight_path, update_live_panel,
        # update_history_day_options, update_history) hepsi AYNI ritimle
        # tetikleniyor, genel amacli bir "periyodik olarak her seyi tazele"
        # kalp atisi olarak kullaniliyor.
        dcc.Interval(id="tick", interval=15000, n_intervals=0),
        # ONEMLI: durum cubugundaki saat ESKIDEN "tick"in (15sn) bir parcasiydi --
        # yani sadece 15 saniyede bir, yeni ucus verisi geldiginde ilerliyordu,
        # normal bir saat gibi saniye saniye AKMIYORDU (kullanici geri bildirimi).
        # Ayri, hizli (1sn) bir Interval + saf clientside_callback (asagida,
        # sayfa sonunda) ile cozuldu -- sunucuya HICBIR istek gitmiyor, saat
        # tamamen tarayicida Date() ile hesaplanip yaziliyor.
        dcc.Interval(id="clock-tick", interval=1000, n_intervals=0),
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
                    "background": altitude_legend_gradient,
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
                        min=0, max=alt_legend_n, step=1,
                        value=[0, alt_legend_n],
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
                html.Span(format_altitude_tick(ft), style={
                    "position": "absolute", "left": f"{i * 100 / alt_legend_n:.4f}%",
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
        dcc.Store(id="altitude-filter-snapped", data=[0, alt_legend_n]),
        # Filtrelenmis irtifa araligi (metre) -- update_map bunu okuyup
        # araligin DISINDAKI ucaklari haritadan gizliyor (bkz. _passes_filter).
        # Varsayilan [-1_000_000, 1_000_000] = "filtre yok" (bkz.
        # update_altitude_filter_range ust VE ALT sinir yorumu -- ikisi de
        # gercek metre degeri DEGIL, sinirsiz demek).
        dcc.Store(id="altitude-filter-range", data=[-1_000_000, 1_000_000]),

        # ------------------------------------------- Durum cubugu (overlay) --
        # ONEMLI: DORT AYRI parcadan olusuyor, DOM SIRASI = GORSEL SIRA (kullanici
        # istegi -- "gösterilen'le aktif uçuş yan yana olsun, alarmı en sağa
        # al"): "status-clock" (saat, bkz. asagidaki clientside_callback --
        # "clock-tick" Interval'i ile saniye saniye ilerliyor, sunucuya HIC
        # istek atmiyor) ve "status-main" (TOPLAM ucak) ve "status-alarm"
        # (alarm sayisi, EN SAGDA) update_map'te (Python, sunucu) hesaplaniyor;
        # aradaki "status-shown" (filtreler sonrasi KALAN ucak sayisi) ise
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
            html.Span(id="status-clock"),
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
                    options=build_timezone_options(DEFAULT_LANGUAGE),  # bkz. fonksiyon yorumu -- bos [] ile
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
                    options=build_signal_staleness_options(DEFAULT_LANGUAGE),
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
