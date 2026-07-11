"""Dashboard/app.py -- saat dilimi dropdown'u testleri.

ONEMLI: test_build_timezone_options_* testleri, bu oturumda gercekten
yasanan bir hatanin regresyon testidir -- dropdown options=[] ile
baslarsa (bos liste), value=DEFAULT_TIMEZONE hicbir secenekle
eslesmedigi icin dropdown kendi icinde value'yu null'a sifirliyordu,
saat dilimi kutusu kullanici elle secim yapana kadar BOS gorunuyordu.
Duzeltme: ilk render'da da _build_timezone_options() ile DOLU bir
options listesi veriliyor (bkz. Dashboard/app.py, timezone-dropdown
tanimi)."""

from __future__ import annotations

import pytest

from Dashboard import app as dashapp


def test_build_timezone_options_covers_full_utc_range():
    options = dashapp._build_timezone_options("tr")
    values = [o["value"] for o in options]
    assert values == list(range(-12, 15))


def test_build_timezone_options_never_empty_regression():
    """Ilk render'da bos [] DONMEMELI -- bkz. modul docstring'i."""
    assert len(dashapp._build_timezone_options("tr")) > 0
    assert len(dashapp._build_timezone_options("en")) > 0


def test_build_timezone_options_default_value_is_always_present():
    options = dashapp._build_timezone_options(dashapp.DEFAULT_LANGUAGE)
    values = [o["value"] for o in options]
    assert dashapp.DEFAULT_TIMEZONE in values


def test_build_timezone_options_default_timezone_has_turkey_suffix_in_tr():
    options = dashapp._build_timezone_options("tr")
    default_option = next(o for o in options if o["value"] == dashapp.DEFAULT_TIMEZONE)
    assert "Türkiye" in default_option["label"]


def test_build_timezone_options_default_timezone_has_turkey_suffix_in_en():
    options = dashapp._build_timezone_options("en")
    default_option = next(o for o in options if o["value"] == dashapp.DEFAULT_TIMEZONE)
    assert "Turkey" in default_option["label"]


def test_build_timezone_options_non_default_entries_have_no_suffix():
    options = dashapp._build_timezone_options("tr")
    non_default = [o for o in options if o["value"] != dashapp.DEFAULT_TIMEZONE]
    assert all(o["label"] == f"UTC{o['value']:+d}" for o in non_default)


def test_build_timezone_options_falls_back_to_default_language_for_unknown_lang():
    assert dashapp._build_timezone_options("xx") == dashapp._build_timezone_options(dashapp.DEFAULT_LANGUAGE)


def test_update_timezone_options_matches_build_helper():
    assert dashapp.update_timezone_options("en") == dashapp._build_timezone_options("en")


# ---------------------------------------------------- update_timezone_setting --

def test_update_timezone_setting_accepts_utc_plus_zero_regression():
    """Regresyon: eski kod 'if not value' kullaniyordu, value=0 (UTC+0)
    Python'da falsy oldugu icin guncelleme sessizce reddediliyordu --
    UTC+0 hicbir zaman secilemiyordu."""
    assert dashapp.update_timezone_setting(0) == 0


@pytest.mark.parametrize("value", [3, -5, 12, -12, 14])
def test_update_timezone_setting_passes_through_valid_values(value):
    assert dashapp.update_timezone_setting(value) == value


def test_update_timezone_setting_none_is_no_update():
    assert dashapp.update_timezone_setting(None) is dashapp.dash.no_update
