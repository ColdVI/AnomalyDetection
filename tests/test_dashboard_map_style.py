"""Dashboard/app.py -- harita türü (Sokak/Uydu/Karanlık) testleri.

ONEMLI: test_tile_layers_never_use_dynamic_subdomain_regression, bu
oturumda gercekten yasanan bir hatanin regresyon testidir -- "{s}"
sablonu iceren bir tile URL'i, Leaflet setUrl() ile CALISMA ZAMANINDA
degistirildiginde "subdomains" prop'u ilk yuklemedeki degerde KALIYORDU
("Uydu"ya gecince URL degisiyordu ama subdomain hala 'a/b/c' kaliyordu
-> "a.google.com/vt/..." gibi GECERSIZ adreslere istek atiliyordu ->
gri ekran). Cozum: TUM katmanlar SABIT tek subdomain kullaniyor, "{s}"
hic yok -- bu test o kararin sessizce bozulmasini engeller."""

from __future__ import annotations

import pytest

from Dashboard.codes import app as dashapp
from dashboard_fakes import simulate_trigger


# ------------------------------------------------------------- TILE_LAYERS --

def test_tile_layers_has_street_satellite_and_dark():
    assert set(dashapp.TILE_LAYERS.keys()) == {"street", "satellite", "dark"}


def test_tile_layers_never_use_dynamic_subdomain_regression():
    for name, layer in dashapp.TILE_LAYERS.items():
        assert "{s}" not in layer["url"], (
            f"TILE_LAYERS['{name}']['url'] dinamik {{s}} subdomain sablonu "
            "iceriyor -- bu daha once 'Uydu'ya gecince gri ekran' hatasina "
            "yol acmisti (Leaflet setUrl() subdomains'i calisma zamaninda "
            "guncellemiyor). Sabit tek subdomain kullanilmali."
        )


def test_tile_layers_urls_contain_required_placeholders():
    for name, layer in dashapp.TILE_LAYERS.items():
        for placeholder in ("{z}", "{x}", "{y}"):
            assert placeholder in layer["url"], f"{name} url'inde {placeholder} eksik"


def test_tile_layers_all_have_attribution():
    for name, layer in dashapp.TILE_LAYERS.items():
        assert layer.get("attribution"), f"{name} icin attribution eksik/bos"


def test_default_map_style_is_dark():
    """Kullanici karariyla varsayilan artik 'Karanlık' (CARTO Dark Matter)."""
    assert dashapp.DEFAULT_MAP_STYLE == "dark"


def test_dark_layer_uses_carto_no_api_key_needed():
    assert "cartocdn.com" in dashapp.TILE_LAYERS["dark"]["url"]


# ------------------------------------------------------- update_base_tile_layer --

def test_update_base_tile_layer_returns_correct_layer_for_each_style():
    for style in dashapp.TILE_LAYERS:
        url, attribution = dashapp.update_base_tile_layer(style)
        assert url == dashapp.TILE_LAYERS[style]["url"]
        assert attribution == dashapp.TILE_LAYERS[style]["attribution"]


def test_update_base_tile_layer_unknown_style_falls_back_to_default():
    url, _ = dashapp.update_base_tile_layer("does-not-exist")
    assert url == dashapp.TILE_LAYERS[dashapp.DEFAULT_MAP_STYLE]["url"]


# ---------------------------------------------------- update_map_style_setting --

def test_update_map_style_setting_street_button():
    with simulate_trigger("map-style-street-btn.n_clicks"):
        assert dashapp.update_map_style_setting(1, None, None) == "street"


def test_update_map_style_setting_dark_button():
    with simulate_trigger("map-style-dark-btn.n_clicks"):
        assert dashapp.update_map_style_setting(None, None, 1) == "dark"


def test_update_map_style_setting_unrelated_trigger_is_no_update():
    with simulate_trigger("some-other-btn.n_clicks"):
        assert dashapp.update_map_style_setting(1, 1, 1) is dashapp.dash.no_update


# ---------------------------------------------------- update_map_style_buttons --

@pytest.mark.parametrize("style,active_index", [("street", 0), ("dark", 2)])
def test_update_map_style_buttons_exactly_one_active(style, active_index):
    styles = dashapp.update_map_style_buttons(style)
    assert styles[active_index] == dashapp.LANG_BTN_ACTIVE_STYLE
    for i, s in enumerate(styles):
        if i != active_index:
            assert s == dashapp.LANG_BTN_INACTIVE_STYLE
