"""Dashboard/app.py -- irtifa filtresi, irtifa->renk eslemesi, saat dilimi
cozumleme testleri.

ONEMLI: update_altitude_filter_range ve _passes_filter testleri, bu
oturumda GERCEKTEN yasanan bir hatanin (yerdeki ucaklarin barometrik
irtifasi dogal olarak hafif NEGATIF gelebiliyor -- QNH kalibrasyonu --
ama alt sinir tam 0 oldugu icin bu ucaklar "Yerde" filtresiyle HICBIR
ILGISI OLMADAN yanlislikla gizleniyordu) regresyon testidir -- bkz.
proje sohbet gecmisi, "aktif uçuş ile gösterilen fark neden" sorusu.
"""

from __future__ import annotations

from datetime import timezone, timedelta

import pytest

from Dashboard.codes import app as dashapp


# --------------------------------------------------------------- _resolve_tz --

@pytest.mark.parametrize("offset_str, expected_hours", [("3", 3), ("-5", -5)])
def test_resolve_tz_parses_valid_offsets(offset_str, expected_hours):
    tz = dashapp._resolve_tz(offset_str)
    assert tz == timezone(timedelta(hours=expected_hours))


@pytest.mark.parametrize("bad_value", [None, "abc"])
def test_resolve_tz_falls_back_to_turkey_on_invalid_input(bad_value):
    assert dashapp._resolve_tz(bad_value) == timezone(timedelta(hours=3))


# ---------------------------------------------------------- _altitude_to_color --

def test_altitude_to_color_none_is_neutral_gray():
    assert dashapp._altitude_to_color(None) == "#888888"


def test_altitude_to_color_matches_exact_stops():
    for alt_m, hexc in dashapp.ALTITUDE_COLOR_STOPS:
        assert dashapp._altitude_to_color(alt_m) == hexc


def test_altitude_to_color_negative_altitude_clamped_to_ground_stop():
    """Yerdeki ucaklarin dogal negatif baro-irtifasi (-7.6m gibi) renk
    haritasinda hata vermemeli -- 0m durağina esdeger davranmali."""
    assert dashapp._altitude_to_color(-45.7) == dashapp._altitude_to_color(0)


def test_altitude_to_color_above_top_stop_uses_last_color():
    top_m, top_color = dashapp.ALTITUDE_COLOR_STOPS[-1]
    assert dashapp._altitude_to_color(top_m + 5000) == top_color


def test_altitude_to_color_interpolates_between_stops():
    lo_m, lo_hex = dashapp.ALTITUDE_COLOR_STOPS[0]
    hi_m, hi_hex = dashapp.ALTITUDE_COLOR_STOPS[1]
    mid_color = dashapp._altitude_to_color((lo_m + hi_m) / 2)
    assert mid_color not in (lo_hex, hi_hex)  # tam ortada, iki uc renkten de FARKLI olmali
    assert mid_color.startswith("#") and len(mid_color) == 7


# ------------------------------------------------------- _format_altitude_tick --

def test_format_altitude_tick_uses_space_thousands_separator():
    assert dashapp._format_altitude_tick(1000) == "1 000"


def test_format_altitude_tick_top_stop_has_plus_suffix():
    top_m = dashapp._ALT_STOP_M[-1]
    assert dashapp._format_altitude_tick(top_m) == f"{top_m:,}".replace(",", " ") + "+"


def test_format_altitude_tick_zero_has_no_separator_artifact():
    assert dashapp._format_altitude_tick(0) == "0"


# --------------------------------------------------- _slider_value_to_altitude_m --

def test_slider_value_to_altitude_m_matches_stops_at_integer_positions():
    for i, alt_m in enumerate(dashapp._ALT_STOP_M):
        assert dashapp._slider_value_to_altitude_m(i) == alt_m


def test_slider_value_to_altitude_m_interpolates_fractional_positions():
    result = dashapp._slider_value_to_altitude_m(0.5)
    lo, hi = dashapp._ALT_STOP_M[0], dashapp._ALT_STOP_M[1]
    assert lo < result < hi


def test_slider_value_to_altitude_m_clamps_out_of_range_values():
    assert dashapp._slider_value_to_altitude_m(-5) == dashapp._ALT_STOP_M[0]
    assert dashapp._slider_value_to_altitude_m(dashapp._ALT_LEGEND_N + 5) == dashapp._ALT_STOP_M[-1]


# ---------------------------------------------- update_altitude_filter_range --

def test_update_altitude_filter_range_full_slider_means_unlimited_both_ends():
    """Kaydirici hem en baştan hem en sona ayarliyken (varsayilan/dokunulmamis
    durum) filtre GERCEKTEN sinirsiz olmali -- ne 0'in altindaki ne de en ust
    durağin uzerindeki irtifalar elenmemeli."""
    lo_m, hi_m = dashapp.update_altitude_filter_range([0, dashapp._ALT_LEGEND_N])
    assert lo_m == -1_000_000
    assert hi_m == 1_000_000


def test_update_altitude_filter_range_lower_bound_not_hard_zero_regression():
    """Regresyon testi: alt sinir tam 0 KULLANILMAMALI (yerdeki ucaklarin
    dogal negatif baro-irtifasini yanlislikla filtreler)."""
    lo_m, _ = dashapp.update_altitude_filter_range([0, dashapp._ALT_LEGEND_N])
    assert lo_m < 0
    assert lo_m <= -45.7  # gercekte gozlemlenen en negatif ornek deger


def test_update_altitude_filter_range_upper_bound_unlimited_when_at_top_stop():
    """Ust sinirin GERCEK en ust durak (orn. 12000m) DEGIL, sinirsiz (buyuk
    bir sayi) olmasi gerekiyor -- bircok jet o duraği asiyor."""
    _, hi_m = dashapp.update_altitude_filter_range([2, dashapp._ALT_LEGEND_N])
    assert hi_m == 1_000_000
    assert hi_m > dashapp._ALT_STOP_M[-1]


def test_update_altitude_filter_range_middle_positions_use_real_meters():
    lo_m, hi_m = dashapp.update_altitude_filter_range([2, 5])
    assert lo_m == dashapp._ALT_STOP_M[2]
    assert hi_m == dashapp._ALT_STOP_M[5]


def test_update_altitude_filter_range_handles_crossed_handles():
    """allowCross=True -- tutamaclar yer degistirebilir, SIRALAMA burada
    garanti edilmeli (min/max)."""
    lo_m, hi_m = dashapp.update_altitude_filter_range([5, 2])
    assert lo_m == dashapp._ALT_STOP_M[2]
    assert hi_m == dashapp._ALT_STOP_M[5]


@pytest.mark.parametrize("bad_value", [None, [], [1], [1, 2, 3]])
def test_update_altitude_filter_range_invalid_input_is_no_update(bad_value):
    result = dashapp.update_altitude_filter_range(bad_value)
    assert result is dashapp.dash.no_update


# ------------------------------------------------------------- _passes_filter --

def _flight(**overrides):
    base = {"is_military": False, "is_ground": False, "alt": 1000.0}
    base.update(overrides)
    return base


def test_passes_filter_negative_altitude_not_excluded_by_default_range_regression():
    """Asil regresyon testi -- yerdeki bir ucak (-7.6m baro-irtifa, is_ground
    isaretli bile olmasa), "Yerde" acikken VE varsayilan (sinirsiz) irtifa
    araligiyla, ASLA irtifa yuzunden filtrelenmemeli."""
    f = _flight(alt=-7.6, is_ground=False)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=True,
        alt_lo_m=-1_000_000, alt_hi_m=1_000_000,
    ) is True


def test_passes_filter_negative_altitude_excluded_only_if_range_explicitly_set_above_zero():
    f = _flight(alt=-7.6)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=True,
        alt_lo_m=0, alt_hi_m=1_000_000,
    ) is False


def test_passes_filter_altitude_none_never_filtered_by_range():
    f = _flight(alt=None)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=True,
        alt_lo_m=5000, alt_hi_m=6000,
    ) is True


def test_passes_filter_military_hidden_when_show_military_false():
    f = _flight(is_military=True)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=False, show_ground=True,
        alt_lo_m=-1_000_000, alt_hi_m=1_000_000,
    ) is False


def test_passes_filter_civil_hidden_when_show_civil_false():
    f = _flight(is_military=False)
    assert dashapp._passes_filter(
        f, show_civil=False, show_military=True, show_ground=True,
        alt_lo_m=-1_000_000, alt_hi_m=1_000_000,
    ) is False


def test_passes_filter_ground_hidden_only_when_show_ground_false():
    f = _flight(is_ground=True)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=False,
        alt_lo_m=-1_000_000, alt_hi_m=1_000_000,
    ) is False
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=True,
        alt_lo_m=-1_000_000, alt_hi_m=1_000_000,
    ) is True


def test_passes_filter_airborne_aircraft_unaffected_by_show_ground():
    """ONEMLI: "Yerde" filtresi askeri/sivil ekseninden BAGIMSIZ -- havadaki
    bir ucak show_ground'dan hic etkilenmemeli."""
    f = _flight(is_ground=False)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=False,
        alt_lo_m=-1_000_000, alt_hi_m=1_000_000,
    ) is True


def test_passes_filter_altitude_range_excludes_outside_values():
    f = _flight(alt=15000)
    assert dashapp._passes_filter(
        f, show_civil=True, show_military=True, show_ground=True,
        alt_lo_m=0, alt_hi_m=10000,
    ) is False


