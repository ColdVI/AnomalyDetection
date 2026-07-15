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
import math
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import redis  # ONEMLI: app.py'nin kendi kodu artik bunu cagirmiyor -- SADECE
# testlerin monkeypatch.setattr(dashapp.redis, "Redis", ...) ile PAYLASILAN
# redis MODULUNUN attribute'unu degistirebilmesi icin burada import edilmis
# olmasi gerekiyor (bkz. test_dashboard_api_endpoints.py, test_dashboard_geocoding.py).
import requests
import uvicorn
import dash
import dash_leaflet as dl
from dash import dcc, html, Output, Input, State, ALL

# --------------------------------------------------------------- sys.path shim --
# Kardes moduller (texts.py, styles.py, constants.py, layout.py) CIPLAK
# import ile (from texts import ...) kullanilabilsin diye -- app.py hem
# Docker'da "python app.py" ile __main__ olarak (Dashboard/ icerigi /app'e
# duz kopyalaniyor, sarmalayan bir Dashboard paketi YOK) hem de testlerde
# "from Dashboard import app" ile Dashboard.app alt modulu olarak calisiyor.
# Bu iki mod, ciplak importlarin calismasi icin TERS sys.path durumu
# gerektiriyor -- __file__'in dizini iki modda da GERCEK Dashboard/ klasoru
# oldugundan, burada ekliyoruz: Docker'da zaten var olani tekrarlayan
# zararsiz bir no-op, pytest'te ise gercekten gerekli.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------- kardes moduller --
from texts import CATEGORY_LABELS, EMERGENCY_LABELS, DEFAULT_LANGUAGE, TEXTS
from styles import (
    TILE_LAYERS, DEFAULT_MAP_STYLE, LEFT_PANEL_BASE, FLIGHT_SEGMENT_BTN_STYLE,
    FLIGHT_SEGMENT_BTN_ACTIVE_STYLE, HISTORY_PANEL_BASE, SETTINGS_PANEL_BASE,
    STATS_PANEL_BASE, EMERGENCY_PANEL_BASE, EMERGENCY_ROW_STYLE, REPLAY_PANEL_BASE,
    LANG_BTN_ACTIVE_STYLE, LANG_BTN_INACTIVE_STYLE, ALERT_COLOR, ALTITUDE_COLOR_STOPS,
    FILTER_BTN_CIVIL_ACTIVE_STYLE, FILTER_BTN_MILITARY_ACTIVE_STYLE,
    FILTER_BTN_GROUND_ACTIVE_STYLE, FILTER_BTN_INACTIVE_STYLE,
)
from layout import build_layout
from server import app_api, app_dash, _query_api
# ONEMLI: _query_api de app.py'nin kendi kodunda cagrilmiyor -- testler
# monkeypatch.setattr(dashapp._query_api, "query_data_frame", ...) ile bu
# NESNENIN attribute'unu degistiriyor (server.py'de tanimlansa da app.py'nin
# import ettigi AYNI nesne oldugu icin patch gorunur kaliyor).
# ONEMLI: asagidaki 9 isim app.py'nin KENDI kodunda hic cagrilmiyor --
# SADECE testlerin dogrudan cagirdigi/patch'ledigi (health, get_data_source,
# set_data_source, get_history, get_flight_segments, _get_flights,
# _reverse_geocode) veya Dash callback'lerinin HTTP round-trip'ten kacinmak
# icin dogrudan cagirdigi (get_replay, _query_history_df) fonksiyonlar
# icin "geri import" ediliyor -- geri kalan endpoint'ler (get_flights,
# get_alerts, get_route, get_aircraft_info, get_history_csv,
# get_replay_frame, traffic_stats, _fetch_adsblol_route,
# _query_replay_frame_cached) burada GEREKMEZ: api.py zaten TEK BASINA
# import edildiginde TUM endpoint'lerini app_api'ye kaydediyor, app.py'nin
# bunlari AYRICA import etmesi gerekmiyor.
from api import (
    health, get_data_source, set_data_source, get_history, get_replay,
    get_flight_segments, _get_flights, _query_history_df, _reverse_geocode,
)
from constants import (
    SIGNAL_STALENESS_OPTIONS, DEFAULT_SIGNAL_STALENESS_SEC, STALE_SIGNAL_OPACITY,
    DEFAULT_DATA_SOURCE, REDIS_DATA_SOURCE_KEY, REDIS_PRODUCER_STATUS_KEY,
    GEOCODE_MAX_LOOKUPS_PER_REQUEST, DEFAULT_TIMEZONE, FLIGHT_GAP_THRESHOLD_MIN,
    AIRLINE_PREFIXES,
)
# ONEMLI: REDIS_DATA_SOURCE_KEY, REDIS_PRODUCER_STATUS_KEY, AIRLINE_PREFIXES
# ve GEOCODE_MAX_LOOKUPS_PER_REQUEST app.py'nin kendi kodunda sadece
# YORUM/JS string'i icinde geciyor -- gercek kullanimlari testlerde
# dashapp.<isim> uzerinden (bkz. test_dashboard_api_endpoints.py,
# test_dashboard_airline_filter.py, test_dashboard_geocoding.py). Silinmemeli.


def _resolve_tz(offset_str):
    """Ayarlardan gelen UTC ofsetini (orn. '3', '-5') sabit-ofsetli bir
    timezone nesnesine cevirir. Gecersiz/bos deger gelirse (ilk yukleme,
    hata vb.) guvenli varsayilana (Turkiye, UTC+3) duser."""
    try:
        return timezone(timedelta(hours=int(offset_str)))
    except Exception:
        return timezone(timedelta(hours=3))


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


# ONEMLI: lejant, ALTITUDE_COLOR_STOPS'tan TURETILIYOR (elle ayri renk
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

app_dash.layout = build_layout(
    build_timezone_options=_build_timezone_options,
    build_signal_staleness_options=_build_signal_staleness_options,
    format_altitude_tick=_format_altitude_tick,
    alt_legend_n=_ALT_LEGEND_N,
    altitude_legend_gradient=ALTITUDE_LEGEND_GRADIENT,
)


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
        # ARTIK YOK (kullanici karariyla).
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

    # ONEMLI: saat artik burada hesaplanmiyor -- "status-clock" span'i
    # ayri, hizli bir clientside_callback ile saniye saniye ilerliyor
    # (bkz. layout'taki "clock-tick" Interval yorumu).
    status_main = t["status_bar_main"].format(n=total_flight_count)
    status_alarm = t["status_bar_alarm"].format(a=len(alerts))

    # ONEMLI: yukaridaki _altitude_map_seq aciklamasina bkz -- eger bu cagri
    # calisirken (esp. /api/flights isteği surerken) DAHA YENI bir update_map
    # cagrisi BASLAMISSA, bu sonuc ARTIK BAYAT -- yazmadan at, daha yeni
    # cagrinin (ne zaman bitecegi onemli degil) sonucunu asla ezme.
    if my_seq != _altitude_map_latest_seq:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return raw_data, status_main, emergency_rows, status_alarm


# ONEMLI: bu callback CLIENTSIDE (JS, tarayicida calisir) -- durum
# cubugundaki saati "clock-tick" Interval'iyle (1sn, bkz. layout) her
# saniye gunceller. Python tarafinda BENZER bir Interval EKLEMEDIK --
# saniyede bir sunucuya HTTP istegi atip _resolve_tz + strftime calistirmak
# (ve Dash'in kendi callback donen-degeri islemesini tetiklemek) tamamen
# gereksiz yuk olurdu, saat sadece yerel saatin GORSEL akisi, veriyle
# ilgisi yok. tzOffset, Python'daki _resolve_tz ile AYNI semantige sahip
# (UTC ofseti, saat cinsinden) -- iki taraf da ayni "timezone-setting"
# Store'unu okuyor, o yuzden JS burada KENDI hesabini yapiyor.
app_dash.clientside_callback(
    """
    function(n_intervals, tzOffsetHours) {
        const offset = (tzOffsetHours === null || tzOffsetHours === undefined)
            ? 3 : Number(tzOffsetHours);
        const shifted = new Date(Date.now() + offset * 3600 * 1000);
        const pad = (x) => String(x).padStart(2, '0');
        return pad(shifted.getUTCHours()) + ':' + pad(shifted.getUTCMinutes())
             + ':' + pad(shifted.getUTCSeconds()) + ' | ';
    }
    """,
    Output("status-clock", "children"),
    [Input("clock-tick", "n_intervals"), Input("timezone-setting", "data")],
)


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
