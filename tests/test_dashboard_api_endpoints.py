"""Dashboard/app.py -- FastAPI endpoint'lerinin (Redis'e dayanan) saf is
mantigi testleri.

ONEMLI: gercek bir HTTP istegi/TestClient KULLANMIYORUZ -- app.py'deki
endpoint fonksiyonlari (health, get_flights, get_data_source...) zaten
DUZ Python fonksiyonlari (FastAPI dekoratoru sadece routing icin), o
yuzden DOGRUDAN cagirip icerideki redis.Redis(...) cagrisini FakeRedis
ile degistirmek yeterli -- gercek bir sunucu/socket'e gerek yok."""

from __future__ import annotations

import json

from Dashboard.codes import app as dashapp
from dashboard_fakes import FakeRedis


def _patch_redis(monkeypatch, fake: FakeRedis) -> None:
    monkeypatch.setattr(dashapp.redis, "Redis", lambda **kwargs: fake)


# -------------------------------------------------------------- _get_flights --

def test_get_flights_empty_when_no_active_flights(monkeypatch):
    _patch_redis(monkeypatch, FakeRedis())
    assert dashapp._get_flights() == []


def test_get_flights_returns_parsed_records_sorted_by_icao(monkeypatch):
    fake = FakeRedis()
    for icao, alt in [("ccc", 3000), ("aaa", 1000), ("bbb", 2000)]:
        fake.sadd("iha:active_flights", icao)
        fake.set(f"iha:state:{icao}", json.dumps({"icao24": icao, "alt": alt}))
    _patch_redis(monkeypatch, fake)

    result = dashapp._get_flights()
    assert [r["icao24"] for r in result] == ["aaa", "bbb", "ccc"]


def test_get_flights_skips_expired_or_missing_state_keys(monkeypatch):
    """Kume uyeligi var ama tekil key TTL ile silinmis/hic yazilmamis
    olabilir -- MGET bunlar icin None doner, cikti listesinde hic
    gorunmemeliler (bkz. _get_flights docstring'i)."""
    fake = FakeRedis()
    fake.sadd("iha:active_flights", "ghost", "real")
    fake.set("iha:state:real", json.dumps({"icao24": "real"}))
    # "ghost" icin iha:state:ghost KASITLI olarak hic yazilmadi
    _patch_redis(monkeypatch, fake)

    result = dashapp._get_flights()
    assert [r["icao24"] for r in result] == ["real"]


# -------------------------------------------------------------------- health --

def test_health_reports_ok_status_and_flight_count(monkeypatch):
    fake = FakeRedis()
    for icao in ("a", "b", "c"):
        fake.sadd("iha:active_flights", icao)
        fake.set(f"iha:state:{icao}", json.dumps({"icao24": icao}))
    _patch_redis(monkeypatch, fake)

    assert dashapp.health() == {"status": "ok", "active_flights": 3}


# ------------------------------------------------------------- data_source --

def test_get_data_source_defaults_when_nothing_set(monkeypatch):
    _patch_redis(monkeypatch, FakeRedis())
    result = dashapp.get_data_source()
    assert result["requested"] == dashapp.DEFAULT_DATA_SOURCE
    assert result["active"] is None


def test_set_data_source_persists_the_requested_value(monkeypatch):
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)

    resp = dashapp.set_data_source("opensky")
    assert resp == {"requested": "opensky"}
    assert fake.get(dashapp.REDIS_DATA_SOURCE_KEY) == "opensky"


def test_set_data_source_rejects_unknown_source(monkeypatch):
    _patch_redis(monkeypatch, FakeRedis())
    resp = dashapp.set_data_source("not-a-real-source")
    assert "error" in resp


def test_get_data_source_reflects_producer_status_when_present(monkeypatch):
    fake = FakeRedis()
    fake.set(dashapp.REDIS_DATA_SOURCE_KEY, "opensky")
    fake.set(dashapp.REDIS_PRODUCER_STATUS_KEY, json.dumps({"source": "adsblol"}))
    _patch_redis(monkeypatch, fake)

    result = dashapp.get_data_source()
    assert result["requested"] == "opensky"
    assert result["active"] == {"source": "adsblol"}  # gecis bekleniyor -- farkli
