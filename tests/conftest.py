"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

from src.common.fakes import FakeMinioClient

# ONEMLI: Dashboard/app.py modul SEVIYESINDE INFLUX_TOKEN'i okuyup (env
# degiskeni yoksa VE influx_token.txt dosyasi da yoksa) SystemExit firlatiyor
# (bkz. Dashboard/app.py, TOKEN_FILE kontrolu) -- bu satir, "Dashboard.app"i
# import eden HERHANGI bir test dosyasindan ONCE (conftest.py, pytest
# tarafindan tum test modullerinden ONCE yuklenir) calismali. Gercek bir
# InfluxDB baglantisi KURULMUYOR (InfluxDBClient kurucusu lazy, sadece
# gercek sorgu/yazma caginca aga cikar), bu yuzden sahte bir token yeterli.
os.environ.setdefault("INFLUX_TOKEN", "test-token-for-pytest")

# ONEMLI: Dashboard/app.py, clientside JS callback'lerini kaydetmek icin
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
