"""Dashboard/app.py -- saat dilimi dropdown'u testleri.

ONEMLI: test_build_timezone_options_covers_full_utc_range ve
test_update_timezone_setting_accepts_utc_plus_zero_regression, bu
oturumda gercekten yasanan iki AYRI hatanin regresyon testidir --
(1) dropdown options=[] ile baslarsa value null'a sifirlaniyordu (bkz.
_build_timezone_options), (2) UTC+0 secilemiyordu ('if not value'
falsy-0 hatasi)."""

from __future__ import annotations

import pytest

from Dashboard.codes import app as dashapp


def test_build_timezone_options_covers_full_utc_range():
    options = dashapp._build_timezone_options("tr")
    values = [o["value"] for o in options]
    assert values == list(range(-12, 15))


def test_build_timezone_options_default_timezone_has_turkey_suffix():
    options = dashapp._build_timezone_options("tr")
    default_option = next(o for o in options if o["value"] == dashapp.DEFAULT_TIMEZONE)
    assert "Türkiye" in default_option["label"]


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


@pytest.mark.parametrize("value", [3, -12])
def test_update_timezone_setting_passes_through_valid_values(value):
    assert dashapp.update_timezone_setting(value) == value


def test_update_timezone_setting_none_is_no_update():
    assert dashapp.update_timezone_setting(None) is dashapp.dash.no_update
