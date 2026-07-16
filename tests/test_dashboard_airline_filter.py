"""Dashboard/app.py -- AIRLINE_PREFIXES veri butunlugu ve dropdown
tanimi testleri.

ONEMLI: gercek eslesme mantigi (cagri kodunun ilk 3 harfini AIRLINE_PREFIXES
ile karsilastirma) CLIENTSIDE JS'te calisir -- bunun GERCEK davranisi
artik tests/test_dashboard_e2e.py::test_airline_filter_reduces_shown_count_but_not_total
tarafindan (gercek tarayicida) dogrulaniyor. Burasi sadece JS'in
CALISABILMESI icin gereken Python tarafindaki VERI SOZLESMESINI (3
harfli kodlar, dropdown yapisi) kapsar."""

from __future__ import annotations

from Dashboard.codes import app as dashapp
from dashboard_fakes import find_by_id


def test_airline_prefixes_keys_are_three_letter_uppercase_codes():
    for code in dashapp.AIRLINE_PREFIXES:
        assert len(code) == 3 and code == code.upper() and code.isalpha(), code


def test_airline_prefixes_is_a_small_curated_list():
    """Kullanici karariyla: kapsamli bir veritabani DEGIL, kucuk/elle
    kuratorlu bir liste (~30-50 havayolu) -- bkz. proje sohbet gecmisi."""
    assert 20 <= len(dashapp.AIRLINE_PREFIXES) <= 80


def test_airline_prefixes_names_are_unique():
    """Iki farkli kod AYNI goruntulenen isme sahip OLMAMALI -- kullanici
    dropdown'da ayni ismi iki kez gormemeli (kafa karistirici olurdu)."""
    names = list(dashapp.AIRLINE_PREFIXES.values())
    assert len(names) == len(set(names))


def test_airline_dropdown_is_multi_select():
    """Kullanici karariyla: coklu secim (Recommended) -- bkz. proje
    sohbet gecmisi."""
    dropdown = find_by_id(dashapp.app_dash.layout, "airline-filter-dropdown")
    assert dropdown is not None
    assert dropdown.multi is True
    assert dropdown.value == []  # bos secim = hicbir sey elenmez


def test_airline_dropdown_options_sorted_alphabetically_by_name():
    """Kullanicinin listede havayolunu bulmasi kolaylassin diye isme gore
    alfabetik sirali olmali (koda gore degil)."""
    dropdown = find_by_id(dashapp.app_dash.layout, "airline-filter-dropdown")
    labels = [o["label"] for o in dropdown.options]
    assert labels == sorted(labels)


def test_airline_dropdown_options_cover_every_prefix():
    dropdown = find_by_id(dashapp.app_dash.layout, "airline-filter-dropdown")
    option_values = {o["value"] for o in dropdown.options}
    assert option_values == set(dashapp.AIRLINE_PREFIXES.keys())
