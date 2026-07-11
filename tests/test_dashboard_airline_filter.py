"""Dashboard/app.py -- AIRLINE_PREFIXES veri butunlugu ve dropdown
tanimi testleri.

ONEMLI: gercek eslesme mantigi (cagri kodunun ilk 3 harfini AIRLINE_PREFIXES
ile karsilastirma) CLIENTSIDE JS'te (bkz. app.py sonundaki
clientside_callback) -- Python pytest bunu DOGRUDAN calistiramaz. Bu
dosya bu yuzden JS mantiginin dogru CALISABILMESI icin gereken Python
tarafindaki VERI SOZLESMESINI (3 harfli buyuk-harf kodlar, bos olmayan
isimler, dropdown'un multi-select ve doğru sirali olmasi) dogrular."""

from __future__ import annotations

from Dashboard import app as dashapp
from dashboard_fakes import find_by_id


def test_airline_prefixes_keys_are_three_letter_uppercase_codes():
    for code in dashapp.AIRLINE_PREFIXES:
        assert len(code) == 3, f"{code!r} 3 harfli degil"
        assert code == code.upper(), f"{code!r} tamami buyuk harf degil"
        assert code.isalpha(), f"{code!r} sadece harf icermiyor"


def test_airline_prefixes_values_are_nonempty_names():
    for code, name in dashapp.AIRLINE_PREFIXES.items():
        assert isinstance(name, str) and name.strip(), f"{code!r} icin isim bos"


def test_airline_prefixes_keys_are_unique_by_construction():
    # dict zaten anahtar celismesine izin vermez, ama JS tarafinda "ilk 3
    # harf" karsilastirmasi 1-1 oldugu icin AYNI KOD IKI FARKLI ISME
    # eslenmemeli -- dict semantigiyle bu zaten garanti, burada sadece
    # sayim tutarliligi dogrulaniyor.
    assert len(dashapp.AIRLINE_PREFIXES) == len(set(dashapp.AIRLINE_PREFIXES.keys()))


def test_airline_prefixes_is_a_small_curated_list():
    """Kullanici karariyla: kapsamli bir veritabani DEGIL, kucuk/elle
    kuratorlu bir liste (~30-50 havayolu) -- bkz. proje sohbet gecmisi."""
    assert 20 <= len(dashapp.AIRLINE_PREFIXES) <= 80


def test_airline_prefixes_names_are_unique():
    """Iki farkli kod AYNI goruntulenen isme sahip OLMAMALI -- kullanici
    dropdown'da ayni ismi iki kez gormemeli (kafa karistirici olurdu)."""
    names = list(dashapp.AIRLINE_PREFIXES.values())
    assert len(names) == len(set(names))


# ------------------------------------------------------- dropdown layout --

def test_airline_dropdown_is_multi_select():
    """Kullanici karariyla: coklu secim (Recommended) -- bkz. proje
    sohbet gecmisi."""
    dropdown = find_by_id(dashapp.app_dash.layout, "airline-filter-dropdown")
    assert dropdown is not None
    assert dropdown.multi is True


def test_airline_dropdown_default_value_is_empty_list():
    """Bos secim = hicbir sey elenmez (tum ucaklar gorunur) -- bkz.
    clientside_callback yorumu."""
    dropdown = find_by_id(dashapp.app_dash.layout, "airline-filter-dropdown")
    assert dropdown.value == []


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


def test_airline_dropdown_option_labels_match_prefix_names():
    dropdown = find_by_id(dashapp.app_dash.layout, "airline-filter-dropdown")
    for opt in dropdown.options:
        assert opt["label"] == dashapp.AIRLINE_PREFIXES[opt["value"]]


# ------------------------------------------------ JS eslesme mantiginin taklidi --

def _matches_airline(callsign: str, selected_codes: set[str]) -> bool:
    """clientside_callback'teki JS mantiginin (rawData.features.filter)
    BIREBIR Python karsiligi -- gercek JS'i calistirmadan, en azindan
    ALGORITMANIN dogru tanimlandigini test edebilmek icin."""
    if not selected_codes:
        return True
    return (callsign or "").strip()[:3].upper() in selected_codes


def test_airline_matching_logic_empty_selection_shows_everything():
    assert _matches_airline("THY1234", set()) is True
    assert _matches_airline("", set()) is True


def test_airline_matching_logic_matches_prefix_case_insensitively():
    assert _matches_airline("thy1234", {"THY"}) is True


def test_airline_matching_logic_ignores_leading_whitespace():
    assert _matches_airline("  THY1234", {"THY"}) is True


def test_airline_matching_logic_rejects_non_matching_callsign():
    assert _matches_airline("PGT5678", {"THY"}) is False


def test_airline_matching_logic_unknown_callsign_hidden_when_filter_active():
    """Bilinmeyen/askeri cagri kodu, BIR filtre secildiginde gizlenir --
    sadece secim BOSKEN her sey gorunur (bkz. onceki test)."""
    assert _matches_airline("F16-01", {"THY"}) is False
