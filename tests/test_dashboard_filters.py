"""Dashboard/app.py -- Sivil/Askeri/Yerde buton Store toggle'lari ve
buton stil callback'leri."""

from __future__ import annotations

from Dashboard import app as dashapp


# ------------------------------------------------------------- toggle_show_* --

def test_toggle_show_civil_flips_current_value():
    assert dashapp.toggle_show_civil(1, True) is False
    assert dashapp.toggle_show_civil(1, False) is True


def test_toggle_show_military_flips_current_value():
    assert dashapp.toggle_show_military(1, True) is False
    assert dashapp.toggle_show_military(1, False) is True


def test_toggle_show_ground_flips_current_value():
    assert dashapp.toggle_show_ground(1, True) is False
    assert dashapp.toggle_show_ground(1, False) is True


# ---------------------------------------------------- style_*_filter_btn --

def test_style_civil_filter_btn_active_vs_inactive():
    assert dashapp.style_civil_filter_btn(True) == dashapp.FILTER_BTN_CIVIL_ACTIVE_STYLE
    assert dashapp.style_civil_filter_btn(False) == dashapp.FILTER_BTN_INACTIVE_STYLE


def test_style_military_filter_btn_active_vs_inactive():
    assert dashapp.style_military_filter_btn(True) == dashapp.FILTER_BTN_MILITARY_ACTIVE_STYLE
    assert dashapp.style_military_filter_btn(False) == dashapp.FILTER_BTN_INACTIVE_STYLE


def test_style_ground_filter_btn_active_vs_inactive():
    assert dashapp.style_ground_filter_btn(True) == dashapp.FILTER_BTN_GROUND_ACTIVE_STYLE
    assert dashapp.style_ground_filter_btn(False) == dashapp.FILTER_BTN_INACTIVE_STYLE


def test_filter_button_active_styles_are_visually_distinct():
    """Uc aktif stil de birbirinden FARKLI olmali -- kullanici hangi
    filtrenin acik oldugunu renkten ayirt edebilmeli."""
    styles = [
        dashapp.FILTER_BTN_CIVIL_ACTIVE_STYLE,
        dashapp.FILTER_BTN_MILITARY_ACTIVE_STYLE,
        dashapp.FILTER_BTN_GROUND_ACTIVE_STYLE,
    ]
    colors = [s.get("backgroundColor") for s in styles]
    assert len(set(colors)) == len(colors)


def test_ground_filter_defaults_to_visible():
    """Kullanici karariyla varsayilan ACIK (bkz. show-ground Store yorumu,
    proje sohbet gecmisi) -- bu varsayilanin sessizce eski (kapali)
    davranisa donmedigini garanti eden regresyon testi."""
    import re
    src = open(dashapp.__file__, encoding="utf-8").read()
    match = re.search(r'dcc\.Store\(id="show-ground", data=(\w+)\)', src)
    assert match is not None, "show-ground Store tanimi bulunamadi"
    assert match.group(1) == "True"
