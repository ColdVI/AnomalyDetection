"""Dashboard/app.py -- update_route_info (kalkis/varis bilgi paneli)
testleri."""

from __future__ import annotations

from Dashboard import app as dashapp
from dashboard_fakes import FakeRequestsRouter


def _patch(monkeypatch, flights=None, route=None):
    router = FakeRequestsRouter(routes={
        "flights": flights if flights is not None else [],
        "route": route if route is not None else {"found": False},
    })
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))
    return router


def test_update_route_info_no_selection_returns_none():
    assert dashapp.update_route_info(None, "tr") is None


def test_update_route_info_no_callsign_shows_message(monkeypatch):
    _patch(monkeypatch, flights=[{"icao24": "abc", "callsign": "  "}])
    result = dashapp.update_route_info("abc", "tr")
    assert result.children == dashapp.TEXTS["tr"]["no_callsign"]


def test_update_route_info_route_not_found_shows_message_with_callsign(monkeypatch):
    _patch(monkeypatch, flights=[{"icao24": "abc", "callsign": "THY123"}],
          route={"found": False})
    result = dashapp.update_route_info("abc", "tr")
    assert "THY123" in result.children


def test_update_route_info_found_route_shows_origin_and_destination(monkeypatch):
    _patch(monkeypatch, flights=[{"icao24": "abc", "callsign": "THY123", "lat": 41.0, "lon": 29.0}],
          route={"found": True, "origin_city": "Istanbul", "origin_iata": "IST",
                 "dest_city": "London", "dest_iata": "LHR", "airline": "Turkish Airlines"})
    result = dashapp.update_route_info("abc", "tr")
    # children[0] = havayolu satiri, children[1] = origin -> dest satiri
    airline_row, route_row = result.children
    assert "Turkish Airlines" in airline_row.children
    origin_span, arrow_span, dest_span = route_row.children
    assert origin_span.children == "Istanbul (IST)"
    assert dest_span.children == "London (LHR)"


def test_update_route_info_missing_iata_shows_em_dash(monkeypatch):
    _patch(monkeypatch, flights=[{"icao24": "abc", "callsign": "THY123"}],
          route={"found": True, "origin_city": "Istanbul", "dest_city": "London"})
    result = dashapp.update_route_info("abc", "tr")
    _, route_row = result.children
    origin_span, _, dest_span = route_row.children
    assert "—" in origin_span.children


def test_update_route_info_missing_airline_hides_airline_row(monkeypatch):
    _patch(monkeypatch, flights=[{"icao24": "abc", "callsign": "THY123"}],
          route={"found": True, "origin_city": "A", "dest_city": "B"})
    result = dashapp.update_route_info("abc", "tr")
    airline_row, _ = result.children
    assert airline_row.children == ""


def test_update_route_info_network_failure_treated_as_not_found(monkeypatch):
    def boom(url, *args, **kwargs):
        raise ConnectionError("koptu")
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(boom)}))
    result = dashapp.update_route_info("abc", "tr")
    assert result.children == dashapp.TEXTS["tr"]["no_callsign"]


