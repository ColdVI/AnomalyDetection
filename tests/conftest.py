"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

from src.common.fakes import FakeMinioClient

# ONEMLI: Dashboard/codes/app.py modul SEVIYESINDE INFLUX_TOKEN'i okuyup (env
# degiskeni yoksa VE influx_token.txt dosyasi da yoksa) SystemExit firlatiyor
# (bkz. Dashboard/codes/app.py, TOKEN_FILE kontrolu) -- bu satir, "Dashboard.app"i
# import eden HERHANGI bir test dosyasindan ONCE (conftest.py, pytest
# tarafindan tum test modullerinden ONCE yuklenir) calismali. Gercek bir
# InfluxDB baglantisi KURULMUYOR (InfluxDBClient kurucusu lazy, sadece
# gercek sorgu/yazma caginca aga cikar), bu yuzden sahte bir token yeterli.
os.environ.setdefault("INFLUX_TOKEN", "test-token-for-pytest")

# ONEMLI: Dashboard/codes/app.py, clientside JS callback'lerini kaydetmek icin
# dash_extensions.javascript.assign() kullaniyor (bkz. _GEOJSON_STYLE_JS/
# _ON_EACH_FEATURE_JS) -- assign() her cagrildiginda (modul import
# ANINDA) su anki CALISMA DIZINine GORECELI bir "assets/dashExtensions_
# default.js" dosyasi YAZIYOR (bkz. dash_extensions.javascript.Namespace.
# dump). pytest repo KOKUNDEN calistirildigi icin (Dashboard/'dan degil),
# bu, HER test calistirmasinda repo kokunde ISTENMEYEN bir "assets/"
# klasoru olusturuyordu (git status'u kirletiyordu). Testler bu dosyanin
# ICERIGINE hic bakmadigi icin, dump()'i no-op yapmak GUVENLI -- assign()
# yine de dogru donus degerini uretmeye devam ediyor (dosyaya yazmak
# haric).
try:
    from dash_extensions.javascript import Namespace as _DashExtNamespace
    _DashExtNamespace.dump = lambda self, assets_folder="assets": None
except ImportError:
    pass


@pytest.fixture
def fake_minio_client() -> FakeMinioClient:
    return FakeMinioClient()


# ============================================================ E2E fixtures --
# ONEMLI: bu fixture'lar SADECE @pytest.mark.e2e testleri icin -- diger 420
# test tamamen hermetik (Docker/ag GEREKMEZ), bunlar ise GERCEK bir tarayici
# VE calisan dashboard-app konteynerini (localhost:8050) gerektirir. Bilerek
# AYRI tutuldu: "pytest -m 'not e2e'" ile varsayilan/hizli calistirmadan
# tamamen disarida birakilabilirler.
DASHBOARD_URL = "http://localhost:8050"


@pytest.fixture(scope="session")
def e2e_browser():
    """Docker stack ayakta degilse (dashboard-app localhost:8050'de yanit
    vermiyorsa) TUM e2e testlerini nazikce SKIP eder -- CI/hizli local
    kosumlarda "Docker calismiyor" diye testler KIRMIZI olmasin, sadece
    atlanmis gorunsun."""
    import urllib.request
    try:
        urllib.request.urlopen(DASHBOARD_URL, timeout=3)
    except Exception as exc:
        pytest.skip(f"dashboard-app {DASHBOARD_URL} adresinde yanit vermiyor "
                   f"(Docker stack calismiyor olabilir): {exc}")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(e2e_browser):
    """HER teste kendi TEMIZ sayfasini (context) verir -- bir testin
    sectigi firma filtresi/dil/ayar bir SONRAKI testi ETKILEMESIN diye
    (pytest-playwright eklentisindeki 'page' fixture'iyla AYNI isim/
    davranis, ama harici bir bagimlilik eklemeden kendi yazdik)."""
    context = e2e_browser.new_context(viewport={"width": 1600, "height": 900})
    pg = context.new_page()
    pg.goto(DASHBOARD_URL, wait_until="networkidle")
    yield pg
    context.close()
