"""Tests for src/ingestion/adsblol_producer.py.

ONEMLI: bu dosyanin daha once HIC testi yoktu -- run()/main loop disindaki
tum saf mantik (_build_headers, fetch_point, poll_once) burada ilk kez
test ediliyor. run() (sinyal isleme + sonsuz dongu) kasitli olarak
kapsanmiyor -- projedeki diger butun producer/consumer main loop'lariyla
(Dashboard/uav_producer.py, dashboard_consumer.py, minio_archiver.py) AYNI
kural: gercek is mantigi ayri, saf fonksiyonlara cikariliyor, sadece
onlar test ediliyor."""

from __future__ import annotations

import pytest
import requests

from src.ingestion.adsblol_producer import _build_headers, fetch_point, poll_once


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """(lat, lon, dist) -> _FakeResponse veya Exception esleyen sahte oturum."""

    def __init__(self, routes: dict[tuple[float, float, int], object]):
        self.routes = routes
        self.calls: list[dict] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        for (lat, lon, dist), result in self.routes.items():
            if f"/lat/{lat}/lon/{lon}/dist/{dist}" in url:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"No fake route configured for {url}")


# -------------------------------------------------------------- _build_headers --

def test_build_headers_empty_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ADSBLOL_API_KEY", raising=False)
    assert _build_headers() == {}


def test_build_headers_includes_api_key_when_set(monkeypatch):
    monkeypatch.setenv("ADSBLOL_API_KEY", "secret123")
    assert _build_headers() == {"api-auth": "secret123"}


def test_build_headers_empty_when_api_key_is_whitespace_only(monkeypatch):
    monkeypatch.setenv("ADSBLOL_API_KEY", "   ")
    assert _build_headers() == {}


# ---------------------------------------------------------------- fetch_point --

def test_fetch_point_builds_correct_url_and_returns_ac_list():
    session = _FakeSession({(41.0, 29.0, 250): _FakeResponse({"ac": [{"hex": "abc"}]})})

    result = fetch_point(41.0, 29.0, 250, session=session)

    assert result == [{"hex": "abc"}]
    assert session.calls[0]["url"] == "https://api.adsb.lol/v2/lat/41.0/lon/29.0/dist/250"
    assert session.calls[0]["timeout"] == 15


def test_fetch_point_returns_empty_list_when_ac_key_missing():
    session = _FakeSession({(1.0, 1.0, 100): _FakeResponse({})})
    assert fetch_point(1.0, 1.0, 100, session=session) == []


def test_fetch_point_passes_auth_header(monkeypatch):
    monkeypatch.setenv("ADSBLOL_API_KEY", "secret123")
    session = _FakeSession({(1.0, 1.0, 100): _FakeResponse({"ac": []})})

    fetch_point(1.0, 1.0, 100, session=session)

    assert session.calls[0]["headers"] == {"api-auth": "secret123"}


def test_fetch_point_raises_on_http_error():
    session = _FakeSession({(1.0, 1.0, 100): _FakeResponse({}, status_code=500)})
    with pytest.raises(requests.HTTPError):
        fetch_point(1.0, 1.0, 100, session=session)


# ----------------------------------------------------------------- poll_once --

def test_poll_once_deduplicates_by_hex_across_points():
    """Ayni hex iki farkli sorgu noktasindan donerse (kapsama alanlari
    cakisiyorsa), tek bir kayit kalmali -- SON gelen kazanir (bkz. fonksiyon
    govdesi, dict atamasi)."""
    points = ((1.0, 1.0, 100), (2.0, 2.0, 100))
    session = _FakeSession({
        (1.0, 1.0, 100): _FakeResponse({"ac": [{"hex": "dup1", "lat": 1.0}]}),
        (2.0, 2.0, 100): _FakeResponse({"ac": [{"hex": "dup1", "lat": 2.0}, {"hex": "uniq2"}]}),
    })

    result = poll_once(session, points=points)

    assert set(result.keys()) == {"dup1", "uniq2"}
    assert result["dup1"]["lat"] == 2.0  # ikinci nokta kazandi


def test_poll_once_skips_entries_without_hex():
    points = ((1.0, 1.0, 100),)
    session = _FakeSession({
        (1.0, 1.0, 100): _FakeResponse({"ac": [{"hex": "ok1"}, {"lat": 5.0}]}),
    })

    result = poll_once(session, points=points)

    assert list(result.keys()) == ["ok1"]


def test_poll_once_one_point_failing_does_not_abort_the_others():
    """Bir sorgu noktasi adsb.lol'e ulasamazsa (RequestException), diger
    noktalar YINE DE islenmeye devam etmeli -- tek bir bolgenin gecici
    hatasi TUM poll'u iptal etmemeli."""
    points = ((1.0, 1.0, 100), (2.0, 2.0, 100))
    session = _FakeSession({
        (1.0, 1.0, 100): requests.ConnectionError("network down"),
        (2.0, 2.0, 100): _FakeResponse({"ac": [{"hex": "survivor"}]}),
    })

    result = poll_once(session, points=points)

    assert list(result.keys()) == ["survivor"]


def test_poll_once_empty_points_returns_empty_dict():
    session = _FakeSession({})
    assert poll_once(session, points=()) == {}
