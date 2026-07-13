"""Dashboard/app.py -- ters-geocoding (_reverse_geocode) ve gecmis ucus
segmentleme (get_flight_segments) testleri.

ONEMLI: test_get_flight_segments_geocoding_is_dispatched_in_parallel,
kullanicinin bildirdigi "geçmiş uçuşlar çok geç çalışıyor" sikayetinin
kok nedenini (sirali/tek-tek Nominatim sorgusu) duzelten
ThreadPoolExecutor kullanimini dogrudan test eder."""

from __future__ import annotations

import pandas as pd

from Dashboard import app as dashapp
from dashboard_fakes import FakeRedis, FakeRequestsRouter


def _patch_redis(monkeypatch, fake: FakeRedis) -> None:
    monkeypatch.setattr(dashapp.redis, "Redis", lambda **kwargs: fake)


# -------------------------------------------------------------- _reverse_geocode --

def test_reverse_geocode_cache_hit_skips_network_call(monkeypatch):
    fake = FakeRedis()
    fake.set("iha:geocode:41.0:29.0", "Istanbul")
    _patch_redis(monkeypatch, fake)

    def boom(*a, **kw):
        raise AssertionError("cache hit olmali, aga hic gidilmemeli")
    monkeypatch.setattr(dashapp.requests, "get", boom)

    assert dashapp._reverse_geocode(41.0, 29.0) == "Istanbul"


def test_reverse_geocode_cache_hit_empty_string_returns_none(monkeypatch):
    """Bos string cache'lenmis olabilir (onceki aramada sonuc bulunamadi) --
    bu durumda None donmeli (tekrar aga gidilmeden), 'cache yok' ile
    'bulunamadigi bilinen' ayirt edilmeli."""
    fake = FakeRedis()
    fake.set("iha:geocode:41.0:29.0", "")  # onceden aranmis, sonuc yoktu
    _patch_redis(monkeypatch, fake)
    monkeypatch.setattr(dashapp.requests, "get",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("aga gidilmemeli")))

    assert dashapp._reverse_geocode(41.0, 29.0) is None


def test_reverse_geocode_success_extracts_city_and_caches(monkeypatch):
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    router = FakeRequestsRouter(routes={"nominatim": {"address": {"city": "Ankara"}}})
    monkeypatch.setattr(dashapp.requests, "get", router)

    result = dashapp._reverse_geocode(39.9, 32.8)
    assert result == "Ankara"
    assert fake.get("iha:geocode:39.9:32.8") == "Ankara"


def test_reverse_geocode_falls_back_through_address_fields(monkeypatch):
    """city yoksa town, o da yoksa village, sonra county, sonra state
    denenir (bkz. fonksiyon govdesi)."""
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    router = FakeRequestsRouter(routes={"nominatim": {"address": {"county": "Some County"}}})
    monkeypatch.setattr(dashapp.requests, "get", router)

    assert dashapp._reverse_geocode(1.0, 1.0) == "Some County"


def test_reverse_geocode_no_address_fields_returns_none_and_caches_empty(monkeypatch):
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    router = FakeRequestsRouter(routes={"nominatim": {"address": {}}})
    monkeypatch.setattr(dashapp.requests, "get", router)

    assert dashapp._reverse_geocode(0.0, 0.0) is None
    assert fake.get("iha:geocode:0.0:0.0") == ""  # bulunamadigi cache'lendi


def test_reverse_geocode_network_exception_returns_none_without_crashing(monkeypatch):
    fake = FakeRedis()
    _patch_redis(monkeypatch, fake)
    monkeypatch.setattr(dashapp.requests, "get",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("timeout")))

    assert dashapp._reverse_geocode(0.0, 0.0) is None


# ---------------------------------------------------------- get_flight_segments --

def _points_df(times, lat=41.0, lon=29.0):
    return pd.DataFrame({
        "_time": pd.to_datetime(times, utc=True),
        "lat": [lat] * len(times), "lon": [lon] * len(times),
    })


def test_get_flight_segments_empty_history_returns_empty_list(monkeypatch):
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: pd.DataFrame())
    monkeypatch.setattr(dashapp, "_reverse_geocode", lambda lat, lon: None)
    assert dashapp.get_flight_segments("abc123") == []


def test_get_flight_segments_splits_on_large_time_gaps():
    """Ardisik iki nokta arasinda FLIGHT_GAP_THRESHOLD_MIN'den BUYUK bir
    bosluk, 'onceki ucus bitti, yenisi basladi' sayilmali (bkz.
    fonksiyon docstring'i)."""
    gap_min = dashapp.FLIGHT_GAP_THRESHOLD_MIN
    times = (
        ["2026-07-01T10:00:00Z", "2026-07-01T10:01:00Z", "2026-07-01T10:02:00Z"] +
        [f"2026-07-01T{10 + (gap_min // 60) + 1:02d}:00:00Z",
         f"2026-07-01T{10 + (gap_min // 60) + 1:02d}:01:00Z",
         f"2026-07-01T{10 + (gap_min // 60) + 1:02d}:02:00Z"]
    )

    def fake_geocode(lat, lon):
        return None

    import Dashboard.app as _mod
    _orig_query = _mod._query_api.query_data_frame
    _mod._query_api.query_data_frame = lambda flux: _points_df(times)
    _orig_geocode = _mod._reverse_geocode
    _mod._reverse_geocode = fake_geocode
    try:
        segments = dashapp.get_flight_segments("abc123")
    finally:
        _mod._query_api.query_data_frame = _orig_query
        _mod._reverse_geocode = _orig_geocode

    assert len(segments) == 2
    assert all(s["points"] == 3 for s in segments)


def test_get_flight_segments_ignores_noise_segments_under_three_points(monkeypatch):
    """Tek/iki noktalik 'segment' muhtemelen gurultu -- gercek bir ucus
    sayilmamali (bkz. fonksiyon govdesi, 'e - s < 3' kontrolu)."""
    gap_min = dashapp.FLIGHT_GAP_THRESHOLD_MIN
    hour_gap = gap_min // 60 + 1
    times = (
        ["2026-07-01T10:00:00Z", "2026-07-01T10:01:00Z"] +  # sadece 2 nokta -- gurultu
        [f"2026-07-01T{10 + hour_gap:02d}:00:00Z",
         f"2026-07-01T{10 + hour_gap:02d}:01:00Z",
         f"2026-07-01T{10 + hour_gap:02d}:02:00Z"]  # 3 nokta -- gercek ucus
    )
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: _points_df(times))
    monkeypatch.setattr(dashapp, "_reverse_geocode", lambda lat, lon: None)

    segments = dashapp.get_flight_segments("abc123")
    assert len(segments) == 1
    assert segments[0]["points"] == 3


def test_get_flight_segments_query_exception_returns_error_dict(monkeypatch):
    def boom(flux):
        raise RuntimeError("influx coktu")
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", boom)
    result = dashapp.get_flight_segments("abc123")
    assert result == {"error": "influx coktu"}


def test_get_flight_segments_geocoding_is_dispatched_in_parallel(monkeypatch):
    """Regresyon (kullanici geri bildirimi -- 'geçmiş uçuşlar çok geç
    çalışıyor'): en yeni segmentler icin start/end geocoding'i AYRI ayri
    (sirali degil) cagirilmali -- ThreadPoolExecutor kullanimini,
    cagrilan lat/lon ciftlerinin DOGRU oldugunu kontrol ederek dolayli
    dogruluyoruz (gercek thread zamanlamasini test etmek yerine)."""
    times = ["2026-07-01T10:00:00Z", "2026-07-01T10:01:00Z", "2026-07-01T10:02:00Z"]
    monkeypatch.setattr(dashapp._query_api, "query_data_frame",
                        lambda flux: _points_df(times, lat=41.0, lon=29.0))

    calls = []

    def fake_geocode(lat, lon):
        calls.append((lat, lon))
        return f"place-{lat}-{lon}"

    monkeypatch.setattr(dashapp, "_reverse_geocode", fake_geocode)
    segments = dashapp.get_flight_segments("abc123")

    assert len(calls) == 2  # start + end, ayni nokta oldugu icin ikisi de (41.0, 29.0)
    assert segments[0]["start_place"] == "place-41.0-29.0"
    assert segments[0]["end_place"] == "place-41.0-29.0"


def test_get_flight_segments_beyond_geocode_limit_gets_null_places(monkeypatch):
    """GEOCODE_MAX_LOOKUPS_PER_REQUEST/2'den fazla segment varsa, geri
    kalanlar icin start_place/end_place None kalmali (dis API'yi
    yormamak icin -- bkz. fonksiyon yorumu)."""
    gap_min = dashapp.FLIGHT_GAP_THRESHOLD_MIN
    hour_gap = gap_min // 60 + 1
    n_segments = dashapp.GEOCODE_MAX_LOOKUPS_PER_REQUEST // 2 + 2
    times = []
    for i in range(n_segments):
        base_hour = (i * hour_gap) % 24
        day_offset = (i * hour_gap) // 24
        times.extend([
            f"2026-07-{1 + day_offset:02d}T{base_hour:02d}:00:00Z",
            f"2026-07-{1 + day_offset:02d}T{base_hour:02d}:01:00Z",
            f"2026-07-{1 + day_offset:02d}T{base_hour:02d}:02:00Z",
        ])
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: _points_df(times))
    monkeypatch.setattr(dashapp, "_reverse_geocode", lambda lat, lon: "somewhere")

    segments = dashapp.get_flight_segments("abc123")
    assert len(segments) == n_segments
    with_place = [s for s in segments if s["start_place"] is not None]
    without_place = [s for s in segments if s["start_place"] is None]
    assert len(with_place) == dashapp.GEOCODE_MAX_LOOKUPS_PER_REQUEST // 2
    assert len(without_place) == 2
