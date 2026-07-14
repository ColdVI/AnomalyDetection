"""app.py'den cikarildi, adim 3 -- paylasilan client/config tekilleri
(Redis pool, InfluxDB client, FastAPI/Dash app nesneleri). _run_api() ve
if __name__ == "__main__": bloğu BILEREK burada DEGIL, app.py'de kaliyor --
bu modulun import edilmesi HICBIR sunucu/thread baslatmamali (bkz. app.py'deki
hermetik test edilebilirlik yorumu)."""

import os
from pathlib import Path

import redis
from dash import Dash
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient

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

# --------------------------------------------------------------------- Dash --

app_dash = Dash(__name__, title="Dashboard")


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
