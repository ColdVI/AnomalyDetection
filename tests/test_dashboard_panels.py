"""Dashboard/app.py -- panel ac/kapa, ucak secimi, cagri kodu arama ve
acil durum oto-odaklama testleri.

ONEMLI: toggle_settings_open/toggle_stats_open/toggle_emergency_open (ve
show_*_panel esleri) KAYNAK KODDA birebir ayni ac/kapa desenini UC KEZ
tekrar ediyor -- sadece "settings" (en sik kullanilan) temsilci olarak
test ediliyor, digger ikisi AYNI kod yolu oldugu icin atlandi."""

from __future__ import annotations

from Dashboard.codes import app as dashapp
from dashboard_fakes import FakeRequestsRouter, simulate_trigger


# -------------------------------------------------------------- select_or_close --

def test_select_or_close_click_close_button_clears_selection():
    with simulate_trigger("close-panel-btn.n_clicks"):
        assert dashapp.select_or_close(1, 1, {"properties": {"icao24": "aaa"}}) is None


def test_select_or_close_clicking_feature_selects_its_icao24():
    with simulate_trigger("aircraft-geojson.clickData"):
        result = dashapp.select_or_close(1, None, {"properties": {"icao24": "abc123"}})
    assert result == "abc123"


def test_select_or_close_no_feature_is_no_update():
    with simulate_trigger("aircraft-geojson.clickData"):
        result = dashapp.select_or_close(1, None, None)
    assert result is dashapp.dash.no_update


def test_select_or_close_feature_missing_icao24_is_no_update():
    with simulate_trigger("aircraft-geojson.clickData"):
        result = dashapp.select_or_close(1, None, {"properties": {}})
    assert result is dashapp.dash.no_update


# ------------------------------------------------------------- toggle_panels --

def test_toggle_panels_slides_in_when_aircraft_selected():
    left, history = dashapp.toggle_panels("abc123")
    assert left["transform"] == "translateX(0)"
    assert history["transform"] == "translateY(0)"


def test_toggle_panels_slides_out_when_no_selection():
    left, history = dashapp.toggle_panels(None)
    assert left["transform"] == "translateX(-100%)"
    assert history["transform"] == "translateY(100%)"


# ------------------------------------------------------------- ayarlar paneli --
# (settings-btn/stats-btn/emergency-btn UCUNUN de AYNI toggle deseni --
# tek temsilci olarak sadece settings test ediliyor)

def test_toggle_settings_open_gear_button_toggles():
    with simulate_trigger("settings-btn.n_clicks"):
        assert dashapp.toggle_settings_open(1, None, False) is True
    with simulate_trigger("settings-btn.n_clicks"):
        assert dashapp.toggle_settings_open(1, None, True) is False


def test_toggle_settings_open_close_button_always_closes():
    with simulate_trigger("close-settings-btn.n_clicks"):
        assert dashapp.toggle_settings_open(None, 1, True) is False


def test_show_settings_panel_display_follows_open_state():
    assert dashapp.show_settings_panel(True)["display"] == "block"
    assert dashapp.show_settings_panel(False)["display"] == "none"


def test_update_emergency_badge_color_follows_active_alerts():
    assert dashapp.update_emergency_badge([{"icao24": "x"}])["backgroundColor"] == "#e63946"
    assert dashapp.update_emergency_badge([])["backgroundColor"] == "#000000"


def test_update_emergency_list_shows_placeholder_when_empty():
    result = dashapp.update_emergency_list([], "tr")
    assert result.children == dashapp.TEXTS["tr"]["no_emergency"]


def test_update_emergency_list_renders_a_row_per_alert():
    rows = [
        {"icao24": "aaa", "callsign": "THY1", "label": "GENEL ACİL DURUM",
         "squawk": "7700", "ts": "10:00:00"},
        {"icao24": "bbb", "callsign": "PGT2", "label": "YAKIT KRİTİK",
         "squawk": "1200", "ts": "10:01:00"},
    ]
    result = dashapp.update_emergency_list(rows, "tr")
    assert len(result) == 2


# --------------------------------------------------------- auto_focus_new_emergency --

def test_auto_focus_new_emergency_selects_the_first_genuinely_new_icao():
    rows = [{"icao24": "aaa"}, {"icao24": "bbb"}]
    icao, seen = dashapp.auto_focus_new_emergency(rows, seen=[])
    assert icao == "aaa"
    assert seen == ["aaa", "bbb"]


def test_auto_focus_new_emergency_no_op_when_nothing_new_regression():
    """Regresyon (kullanici geri bildirimi -- 'kapatsak bile yeniden
    açılıyor'): AYNI (zaten 'seen' listesindeki) acil durum devam
    ediyor diye TEKRAR oto-odaklama YAPILMAMALI."""
    rows = [{"icao24": "aaa"}]
    icao, seen = dashapp.auto_focus_new_emergency(rows, seen=["aaa"])
    assert icao is dashapp.dash.no_update
    assert seen is dashapp.dash.no_update


def test_auto_focus_new_emergency_triggers_again_for_a_different_new_aircraft():
    rows = [{"icao24": "aaa"}, {"icao24": "ccc"}]
    icao, seen = dashapp.auto_focus_new_emergency(rows, seen=["aaa"])
    assert icao == "ccc"
    assert seen == ["aaa", "ccc"]


# -------------------------------------------------------------- search_by_callsign --

def _search(monkeypatch, flights, query, zoom=5):
    router = FakeRequestsRouter(routes={"flights": flights})
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))
    return dashapp.search_by_callsign(1, None, query, "tr", zoom)


def test_search_by_callsign_exact_match(monkeypatch):
    flights = [{"icao24": "abc", "callsign": "THY123", "lat": 41.0, "lon": 29.0}]
    icao, feedback, viewport = _search(monkeypatch, flights, "THY123")
    assert icao == "abc"
    assert feedback == ""
    assert viewport["center"] == [41.0, 29.0]


def test_search_by_callsign_case_and_whitespace_insensitive(monkeypatch):
    """adsb.lol callsign'lari bazen ic bosluklu doner (orn. 'VATOZ 16') --
    arama normalize edilmis karsilastirma yapmali."""
    flights = [{"icao24": "abc", "callsign": " thy123 ", "lat": 41.0, "lon": 29.0}]
    icao, _, _ = _search(monkeypatch, flights, "THY123")
    assert icao == "abc"


def test_search_by_callsign_falls_back_to_partial_match(monkeypatch):
    flights = [{"icao24": "abc", "callsign": "THY123EXTRA", "lat": 41.0, "lon": 29.0}]
    icao, _, _ = _search(monkeypatch, flights, "THY123")
    assert icao == "abc"


def test_search_by_callsign_not_found_returns_message_without_changing_selection(monkeypatch):
    icao, feedback, viewport = _search(monkeypatch, [], "GHOST99")
    assert icao is dashapp.dash.no_update
    assert "GHOST99" in feedback
    assert viewport is dashapp.dash.no_update


def test_search_by_callsign_empty_query_is_no_op(monkeypatch):
    icao, feedback, viewport = _search(monkeypatch, [], "   ")
    assert icao is dashapp.dash.no_update
    assert feedback == ""
    assert viewport is dashapp.dash.no_update


def test_search_by_callsign_zoom_never_decreases_below_eight(monkeypatch):
    """Kullanici istegi: bulunan ucaga en az 8 zoom seviyesinde gidilmeli
    (zaten daha yakinsa mevcut zoom korunur) -- bkz. fonksiyon docstring'i."""
    flights = [{"icao24": "abc", "callsign": "THY1", "lat": 41.0, "lon": 29.0}]
    _, _, viewport = _search(monkeypatch, flights, "THY1", zoom=3)
    assert viewport["zoom"] == 8

    _, _, viewport2 = _search(monkeypatch, flights, "THY1", zoom=12)
    assert viewport2["zoom"] == 12


def test_search_by_callsign_match_without_position_skips_viewport(monkeypatch):
    flights = [{"icao24": "abc", "callsign": "THY1", "lat": None, "lon": None}]
    icao, _, viewport = _search(monkeypatch, flights, "THY1")
    assert icao == "abc"
    assert viewport is dashapp.dash.no_update
