"""Dashboard/app.py -- sinyal yaşı eşiği (opacity) ve dropdown testleri.

ONEMLI: test_signal_opacity_* testleri, oturumda "OpenSky'de neredeyse
her uçak saydam" olarak fark edilen sorunun (sabit 10-40sn dogrusal
soluklasma esigi, OpenSky'nin 90-300sn'lik dogal sorgulama araligina
gore cok kucuk kaliyordu) yerini alan kullanici-secilebilir esik
mantiginin dogrulugunu test eder."""

from __future__ import annotations

import pytest

from Dashboard.codes import app as dashapp


# ------------------------------------------------------------- _signal_opacity --

def test_signal_opacity_none_is_fully_opaque():
    """signal_age_sec None ise (kaynak saglamiyorsa) GUVENLI VARSAYILAN:
    tam opak -- dim etmeyecek kadar bilgimiz yok."""
    assert dashapp._signal_opacity(None, 60) == 1.0


@pytest.mark.parametrize("age,threshold", [(0, 60), (60, 60)])
def test_signal_opacity_at_or_below_threshold_is_opaque(age, threshold):
    assert dashapp._signal_opacity(age, threshold) == 1.0


@pytest.mark.parametrize("age,threshold", [(61, 60), (301, 300)])
def test_signal_opacity_above_threshold_is_stale_value(age, threshold):
    assert dashapp._signal_opacity(age, threshold) == dashapp.STALE_SIGNAL_OPACITY


def test_signal_opacity_stale_value_never_fully_invisible():
    assert dashapp.STALE_SIGNAL_OPACITY > 0.0


def test_signal_opacity_opensky_typical_age_no_longer_always_stale_with_wide_threshold():
    """Regresyon: eskiden sabit 40sn ustu her zaman soluktu -- OpenSky'nin
    tipik (~90-200sn) sinyal yasiyla gelen bir ucak, kullanici 5dk esigi
    sectiginde artik tam opak gorunebilmeli."""
    assert dashapp._signal_opacity(150, 300) == 1.0


# ------------------------------------------------------ _format_staleness_label --

@pytest.mark.parametrize("sec,lang,expected", [
    (30, "tr", "30sn"), (30, "en", "30s"),
    (60, "tr", "1dk"), (3600, "en", "1h"),
])
def test_format_staleness_label(sec, lang, expected):
    assert dashapp._format_staleness_label(sec, lang) == expected


def test_format_staleness_label_all_options_are_whole_units():
    """SIGNAL_STALENESS_OPTIONS'taki her deger tam dakika/saat siniri
    ustunde olmali -- aksi halde _format_staleness_label kesirli/cirkin
    bir etiket uretebilir (bkz. fonksiyon docstring'i)."""
    for sec in dashapp.SIGNAL_STALENESS_OPTIONS:
        label = dashapp._format_staleness_label(sec, "tr")
        assert label[-2:] in ("sn", "dk", "sa")


# --------------------------------------------------- _build_signal_staleness_options --

def test_build_signal_staleness_options_matches_constant_list():
    options = dashapp._build_signal_staleness_options("tr")
    assert [o["value"] for o in options] == dashapp.SIGNAL_STALENESS_OPTIONS


def test_build_signal_staleness_options_has_seven_choices():
    """Kullanici istegi: 'sadece 3 secenek olmasin, 1 saate kadar birkaç
    seçenek daha ekle' -- 30sn/1dk/2dk/5dk/10dk/30dk/1sa."""
    assert len(dashapp.SIGNAL_STALENESS_OPTIONS) == 7
    assert dashapp.SIGNAL_STALENESS_OPTIONS[-1] == 3600


def test_update_signal_staleness_options_matches_build_helper():
    assert dashapp.update_signal_staleness_options("en") == dashapp._build_signal_staleness_options("en")


# ---------------------------------------------------- update_signal_staleness_setting --

@pytest.mark.parametrize("value", [30, 3600])
def test_update_signal_staleness_setting_passes_through_valid_values(value):
    assert dashapp.update_signal_staleness_setting(value) == value


def test_update_signal_staleness_setting_none_is_no_update():
    assert dashapp.update_signal_staleness_setting(None) is dashapp.dash.no_update


def test_default_signal_staleness_is_a_valid_option():
    assert dashapp.DEFAULT_SIGNAL_STALENESS_SEC in dashapp.SIGNAL_STALENESS_OPTIONS
