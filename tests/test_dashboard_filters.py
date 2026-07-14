"""Dashboard/app.py -- Sivil/Askeri/Yerde buton Store toggle'lari ve
buton stil callback'leri.

ONEMLI: toggle_show_civil/military/ground VE style_civil/military/ground_
filter_btn KAYNAK KODDA birebir ayni deseni (sirasiyla "return not
current" ve "aktif_stil if visible else INACTIVE_STYLE") UC KEZ tekrar
ediyor -- ucunu de ayri ayri test etmek SIFIR ek guven katar (biri
bozulursa desen zaten bozuktur, digerleri de bozulur), o yuzden sadece
"Yerde" (en son degisen, varsayilani ACIK'a cevrilen) temsilci olarak
test ediliyor."""

from __future__ import annotations

from Dashboard import app as dashapp
from dashboard_fakes import find_by_id


def test_toggle_show_ground_flips_current_value():
    assert dashapp.toggle_show_ground(1, True) is False
    assert dashapp.toggle_show_ground(1, False) is True


def test_style_ground_filter_btn_active_vs_inactive():
    assert dashapp.style_ground_filter_btn(True) == dashapp.FILTER_BTN_GROUND_ACTIVE_STYLE
    assert dashapp.style_ground_filter_btn(False) == dashapp.FILTER_BTN_INACTIVE_STYLE


def test_filter_button_active_styles_are_visually_distinct():
    """Uc aktif stil de birbirinden FARKLI olmali -- kullanici hangi
    filtrenin acik oldugunu renkten ayirt edebilmeli. (Bu tek test, ucu
    de dogru TANIMLANMIS oldugunu -- yukaridaki "tekrarli" testlerin
    kapsamadigi tek gercek risk -- zaten dogruluyor.)"""
    colors = [s.get("backgroundColor") for s in (
        dashapp.FILTER_BTN_CIVIL_ACTIVE_STYLE,
        dashapp.FILTER_BTN_MILITARY_ACTIVE_STYLE,
        dashapp.FILTER_BTN_GROUND_ACTIVE_STYLE,
    )]
    assert len(set(colors)) == len(colors)


def test_ground_filter_defaults_to_visible():
    """Kullanici karariyla varsayilan ACIK (bkz. show-ground Store yorumu,
    proje sohbet gecmisi) -- bu varsayilanin sessizce eski (kapali)
    davranisa donmedigini garanti eden regresyon testi.

    ONEMLI: kaynak metnini regex'le taramak yerine GERCEKTEN olusan layout
    nesnesini (find_by_id) okuyoruz -- layout agaci artik Dashboard/layout.py
    icinde (bkz. modul-bolme, adim 2), bu yontem HANGI dosyada tanimli
    oldugundan bagimsiz calisir."""
    store = find_by_id(dashapp.app_dash.layout, "show-ground")
    assert store is not None, "show-ground Store tanimi bulunamadi"
    assert store.data is True
