"""Dashboard/app.py -- update_map'in uctan uca (mock HTTP ile) davranis
testleri: filtreleme + renklendirme + acil durum tespiti + durum metni
tek bir cagrida DOGRU BIRLESIYOR mu.

ONEMLI: bunlar test_dashboard_altitude.py/test_dashboard_filters.py'deki
IZOLE _passes_filter/_signal_opacity testlerinden FARKLI -- oradakiler
"parca dogru mu", burdakiler "parcalar update_map icinde DOGRU
BAGLANIYOR mu" sorusunu cevapliyor (orn. _passes_filter'in GERCEKTEN
dogru sirayla/argumanlarla cagrildigini, ALERT_COLOR'in irtifa rengini
GERCEKTEN eze bildigini)."""

from __future__ import annotations

from Dashboard.codes import app as dashapp
from dashboard_fakes import FakeRequestsRouter


def _patch_requests(monkeypatch, flights=None, alerts=None):
    router = FakeRequestsRouter(routes={
        "flights": flights if flights is not None else [],
        "alerts": alerts if alerts is not None else [],
    })
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))
    return router


def _call_update_map(**overrides):
    kwargs = dict(
        n=1, tz_name="3", lang="tr", show_civil=True, show_military=True,
        show_ground=True, replay_mode=False,
        altitude_filter_range=[-1_000_000, 1_000_000], staleness_threshold=60,
    )
    kwargs.update(overrides)
    return dashapp.update_map(**kwargs)


# ------------------------------------------------------------ ag hatasi/bosluk --

def test_update_map_network_failure_treated_as_empty_not_crash(monkeypatch):
    def boom(url, timeout=3, **kw):
        raise ConnectionError("baglanti koptu")
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(boom)}))

    raw_data, status_main, emergency_rows, status_alarm = _call_update_map()
    assert raw_data["features"] == []
    assert emergency_rows == []
    assert "0 aktif uçuş" in status_main


# ----------------------------------------------------------------- filtreleme --

def test_update_map_total_count_ignores_filters_but_features_respects_them(monkeypatch):
    flights = [
        {"icao24": "civ1", "is_military": False, "alt": 1000.0, "callsign": "THY1",
         "lat": 41.0, "lon": 29.0},
        {"icao24": "mil1", "is_military": True, "alt": 1000.0, "callsign": "TUAF1",
         "lat": 41.0, "lon": 29.0},
    ]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, status_main, _, _ = _call_update_map(show_military=False)
    assert "2 aktif uçuş" in status_main  # toplam FILTRESIZ
    assert len(raw_data["features"]) == 1  # ama gosterilen filtrelenmis
    assert raw_data["features"][0]["properties"]["icao24"] == "civ1"


def test_update_map_altitude_range_excludes_outside_aircraft(monkeypatch):
    flights = [
        {"icao24": "low", "is_military": False, "alt": 500.0, "lat": 41.0, "lon": 29.0},
        {"icao24": "high", "is_military": False, "alt": 15000.0, "lat": 41.0, "lon": 29.0},
    ]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, _, _ = _call_update_map(altitude_filter_range=[0, 10000])
    icaos = [f["properties"]["icao24"] for f in raw_data["features"]]
    assert icaos == ["low"]


# --------------------------------------------------------- renklendirme/alarm --

def test_update_map_squawk_emergency_overrides_altitude_color(monkeypatch):
    flights = [{"icao24": "hijack", "is_military": False, "alt": 3000.0,
                "squawk": "7500", "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, emergency_rows, _ = _call_update_map()
    assert raw_data["features"][0]["properties"]["color"] == dashapp.ALERT_COLOR
    assert len(emergency_rows) == 1
    assert emergency_rows[0]["icao24"] == "hijack"
    assert emergency_rows[0]["squawk"] == "7500"


def test_update_map_emergency_field_also_triggers_alert_regardless_of_squawk(monkeypatch):
    flights = [{"icao24": "medic", "is_military": False, "alt": 3000.0,
                "squawk": "1200", "emergency": "lifeguard", "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights)

    _, _, emergency_rows, _ = _call_update_map()
    assert len(emergency_rows) == 1
    assert emergency_rows[0]["label"] == dashapp.EMERGENCY_LABELS["tr"]["lifeguard"]


def test_update_map_normal_squawk_no_emergency_row(monkeypatch):
    flights = [{"icao24": "normal", "is_military": False, "alt": 3000.0,
                "squawk": "1200", "emergency": "none", "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, emergency_rows, _ = _call_update_map()
    assert emergency_rows == []
    assert raw_data["features"][0]["properties"]["color"] != dashapp.ALERT_COLOR


def test_update_map_alerts_endpoint_also_colors_aircraft_red(monkeypatch):
    """/api/alerts (ML ekibinin uav.alerts topic'i, henuz bos) -- oradan
    gelen bir icao24 de kirmizi renklendirilmeli, kendi squawk/emergency
    alanindan BAGIMSIZ."""
    flights = [{"icao24": "ml-flagged", "is_military": False, "alt": 3000.0,
                "squawk": "1200", "emergency": "none", "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights, alerts=[{"icao24": "ml-flagged"}])

    raw_data, _, _, status_alarm = _call_update_map()
    assert raw_data["features"][0]["properties"]["color"] == dashapp.ALERT_COLOR
    assert "1" in status_alarm


# ------------------------------------------------------------- alan formatlama --

def test_update_map_missing_optional_fields_show_em_dash(monkeypatch):
    flights = [{"icao24": "sparse", "is_military": False, "alt": 3000.0,
                "lat": 41.0, "lon": 29.0}]  # velocity/track/vertical_rate yok
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, _, _ = _call_update_map()
    props = raw_data["features"][0]["properties"]
    assert props["speed_text"] == "—"
    assert props["track_text"] == "—"
    assert props["vspeed_text"] == "—"


def test_update_map_callsign_falls_back_to_icao_when_blank(monkeypatch):
    flights = [{"icao24": "nocall", "is_military": False, "alt": 3000.0,
                "callsign": "   ", "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, _, _ = _call_update_map()
    assert raw_data["features"][0]["properties"]["callsign"] == "NOCALL"


def test_update_map_signal_age_text_hidden_below_ten_seconds(monkeypatch):
    """Kisa/taze sinyaller icin metin gosterilmez (bkz. update_map'teki
    '>= 10' esigi) -- sadece opacity etkilenir, metin degil."""
    flights = [{"icao24": "fresh", "is_military": False, "alt": 3000.0,
                "signal_age_sec": 3.0, "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, _, _ = _call_update_map()
    assert raw_data["features"][0]["properties"]["signal_age_text"] is None
    assert raw_data["features"][0]["properties"]["opacity"] == 1.0


def test_update_map_signal_age_text_shown_and_opacity_dimmed_when_stale(monkeypatch):
    flights = [{"icao24": "stale", "is_military": False, "alt": 3000.0,
                "signal_age_sec": 120.0, "lat": 41.0, "lon": 29.0}]
    _patch_requests(monkeypatch, flights=flights)

    raw_data, _, _, _ = _call_update_map(staleness_threshold=60)
    props = raw_data["features"][0]["properties"]
    assert props["signal_age_text"] == "120sn"
    assert props["opacity"] == dashapp.STALE_SIGNAL_OPACITY


# ------------------------------------------------------------------- dil/etiket --

