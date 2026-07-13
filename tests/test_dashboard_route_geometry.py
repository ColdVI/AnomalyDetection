"""Dashboard/app.py -- rota cizgisi geometrisi (_bearing_deg,
_route_is_plausible, _great_circle_points) testleri.

ONEMLI: test_route_is_plausible_catches_wzz43_style_reversed_route,
fonksiyonun KENDI docstring'inde anlatilan gercek dunya ornegini
(WZZ43: adsbdb Londra->Budapeste diyordu ama ucak gercekte Krakow->
Stavanger ucuyordu, 180 derece ters yon) dogrudan regresyon testine
cevirir."""

from __future__ import annotations

import pytest

from Dashboard import app as dashapp


# --------------------------------------------------------------- _bearing_deg --

def test_bearing_deg_cardinal_directions():
    assert dashapp._bearing_deg(0, 0, 0, 10) == pytest.approx(90, abs=0.5)  # dogu
    assert dashapp._bearing_deg(0, 0, 10, 0) == pytest.approx(0, abs=0.5)   # kuzey
    assert dashapp._bearing_deg(10, 0, 0, 0) == pytest.approx(180, abs=0.5)  # guney
    assert dashapp._bearing_deg(0, 0, 0, -10) == pytest.approx(270, abs=0.5)  # bati


def test_bearing_deg_always_in_0_360_range():
    for lat2, lon2 in [(-10, -10), (10, 10), (-89, 179), (89, -179)]:
        b = dashapp._bearing_deg(0, 0, lat2, lon2)
        assert 0 <= b < 360


# ---------------------------------------------------------- _route_is_plausible --

def test_route_is_plausible_none_track_always_trusts():
    """Yon bilinmiyorsa kontrol edilemez -- guvenli varsayilan: guven
    (bkz. fonksiyon docstring'i)."""
    assert dashapp._route_is_plausible(41.0, 29.0, None, 50.0, 20.0) is True


def test_route_is_plausible_matching_direction_is_plausible():
    # Istanbul'dan (41,29) doguya dogru ucan bir ucak, dogudaki bir
    # varisa (41, 60) makul gorunmeli.
    bearing = dashapp._bearing_deg(41.0, 29.0, 41.0, 60.0)
    assert dashapp._route_is_plausible(41.0, 29.0, bearing, 41.0, 60.0) is True


def test_route_is_plausible_catches_wzz43_style_reversed_route():
    """Regresyon: WZZ43 orneği -- ucak GERCEKTE Krakow'dan (50.08, 19.79)
    Stavanger'a (58.88, 5.64) doğru batı-kuzeybatı yönünde uçarken,
    adsbdb rotayı Londra->Budapeşte (yani ters/güneydoğu yönü) olarak
    döndürüyordu. Gerçek track ile iddia edilen varış yönü arasındaki
    ~180 derecelik fark SÜPHELI sayılmalı."""
    actual_krakow_to_stavanger_bearing = dashapp._bearing_deg(50.08, 19.79, 58.88, 5.64)
    # adsbdb'nin iddia ettigi (yanlis) varis: Budapeste, Krakow'un GUNEYINDE
    claimed_dest_lat, claimed_dest_lon = 47.5, 19.0  # Budapeste
    assert dashapp._route_is_plausible(
        50.08, 19.79, actual_krakow_to_stavanger_bearing,
        claimed_dest_lat, claimed_dest_lon,
    ) is False


def test_route_is_plausible_boundary_at_threshold():
    # Kuzeye ucan bir ucak (track=0), doguya (bearing=90) varis --
    # fark tam 90, varsayilan threshold_deg=90 ile SINIRDA (<=) hala makul.
    assert dashapp._route_is_plausible(0, 0, 0, 0, 10, threshold_deg=90) is True


def test_route_is_plausible_just_over_threshold_is_implausible():
    assert dashapp._route_is_plausible(0, 0, 0, -0.001, 10, threshold_deg=89.9) is False


def test_route_is_plausible_custom_threshold_is_stricter():
    bearing = dashapp._bearing_deg(41.0, 29.0, 41.0, 60.0)
    # Tam dogru yon (fark 0) -- her esikte makul olmali.
    assert dashapp._route_is_plausible(41.0, 29.0, bearing, 41.0, 60.0, threshold_deg=1) is True
    # 45 derece sapma -- gevsek esikte makul, siki esikte degil.
    assert dashapp._route_is_plausible(41.0, 29.0, bearing + 45, 41.0, 60.0, threshold_deg=60) is True
    assert dashapp._route_is_plausible(41.0, 29.0, bearing + 45, 41.0, 60.0, threshold_deg=10) is False


# -------------------------------------------------------- _great_circle_points --

def test_great_circle_points_starts_and_ends_at_given_coordinates():
    points = dashapp._great_circle_points(41.0, 29.0, 51.5, -0.1, n=64)
    assert points[0] == pytest.approx([41.0, 29.0], abs=1e-6)
    assert points[-1] == pytest.approx([51.5, -0.1], abs=1e-6)


def test_great_circle_points_returns_n_plus_one_points():
    points = dashapp._great_circle_points(0, 0, 10, 10, n=20)
    assert len(points) == 21


def test_great_circle_points_identical_endpoints_returns_two_point_line():
    """d < 1e-9 kisayolu -- ayni/neredeyse ayni nokta icin egriye gerek
    yok, sadece 2 nokta donmeli (bkz. fonksiyon govdesi)."""
    points = dashapp._great_circle_points(41.0, 29.0, 41.0, 29.0, n=64)
    assert points == [[41.0, 29.0], [41.0, 29.0]]


def test_great_circle_points_no_sudden_longitude_jump_across_antimeridian():
    """Regresyon (fonksiyon docstring'i): 180. meridyeni gecen rotalarda
    (orn. Tokyo->Los Angeles) ardisik noktalar arasinda ANI (>180 derece)
    boylam sicramasi OLMAMALI -- 'unwrap' surekliligi saglamali."""
    # Tokyo (35.6, 139.7) -> Los Angeles (34.0, -118.2), Pasifik'i gecen rota
    points = dashapp._great_circle_points(35.6, 139.7, 34.0, -118.2, n=64)
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        assert abs(lon2 - lon1) < 180, f"ani boylam sicramasi: {lon1} -> {lon2}"


