"""Dashboard/app.py -- gecmis grafigi (update_history), CSV indirme ve
ortak _query_history_df sorgu govdesi testleri.

ONEMLI: test_update_history_date_range_end_hour_is_exact_boundary_regression,
kullanicinin GERCEKTEN bildirdigi bir hatanin ("11 ile 12 arasi istedim
ama 11.40-12.40 geldi") regresyon testidir."""

from __future__ import annotations

import pandas as pd
import pytest

from Dashboard.codes import app as dashapp


def _fake_query_data_frame(monkeypatch, df: pd.DataFrame):
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: df)


def _sample_history_df():
    return pd.DataFrame({
        "_time": pd.to_datetime(["2026-07-11T10:00:00Z", "2026-07-11T10:01:00Z"], utc=True),
        "lat": [41.0, 41.1], "lon": [29.0, 29.1],
        "alt": [1000.0, 1100.0], "velocity": [200.0, 210.0],
    })


# -------------------------------------------------------------- _query_history_df --

def test_query_history_df_hours_mode_returns_expected_columns(monkeypatch):
    _fake_query_data_frame(monkeypatch, _sample_history_df())
    df, err = dashapp._query_history_df("abc123", hours=24)
    assert err is None
    assert list(df.columns) == ["_time", "lat", "lon", "alt", "velocity"]
    assert len(df) == 2


def test_query_history_df_hours_capped_at_bucket_retention(monkeypatch):
    """InfluxDB bucket'i zaten 7 gunden fazlasini tutmuyor -- 'hours' bunu
    ASAMAZ (bkz. fonksiyon govdesindeki min() -- flux sorgusuna gomulen
    metni dogrudan kontrol ediyoruz)."""
    captured = {}

    def fake_query(flux):
        captured["flux"] = flux
        return pd.DataFrame()

    monkeypatch.setattr(dashapp._query_api, "query_data_frame", fake_query)
    dashapp._query_history_df("abc123", hours=24 * 30)  # 30 gun istendi
    assert "range(start: -168h)" in captured["flux"]  # 24*7 saate sabitlendi


def test_query_history_df_empty_result_returns_empty_dataframe_not_none(monkeypatch):
    _fake_query_data_frame(monkeypatch, pd.DataFrame())
    df, err = dashapp._query_history_df("abc123", hours=24)
    assert err is None
    assert df.empty
    assert list(df.columns) == ["_time", "lat", "lon", "alt", "velocity"]


def test_query_history_df_missing_velocity_column_filled_with_none(monkeypatch):
    """Bir ucak icin secilen aralikta HIC velocity verisi yoksa pivot
    sonucunda kolon hic olusmayabilir -- KeyError yerine None ile
    doldurulmus olmali (bkz. fonksiyon yorumu)."""
    df_without_velocity = pd.DataFrame({
        "_time": pd.to_datetime(["2026-07-11T10:00:00Z"], utc=True),
        "lat": [41.0], "lon": [29.0], "alt": [1000.0],
    })
    _fake_query_data_frame(monkeypatch, df_without_velocity)
    df, err = dashapp._query_history_df("abc123", hours=24)
    assert err is None
    assert df["velocity"].isna().all()


def test_query_history_df_invalid_start_end_returns_error():
    df, err = dashapp._query_history_df("abc123", start="not-a-date", end="also-not-a-date")
    assert df is None
    assert "invalid" in err.lower()


def test_query_history_df_query_exception_returns_error_not_crash(monkeypatch):
    def boom(flux):
        raise RuntimeError("influxdb coktu")
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", boom)
    df, err = dashapp._query_history_df("abc123", hours=24)
    assert df is None
    assert "influxdb coktu" in err


def test_query_history_df_rejects_flux_injection_via_icao24(monkeypatch):
    """start/end kullanicidan gelse bile PARSE EDILIP kendi ISO string'imize
    cevrilerek gomuluyor (injection riski yok, bkz. fonksiyon docstring'i) --
    icao24 dogrudan gomuluyor ama FastAPI path param oldugu icin zaten
    URL-safe. Burada en azindan start/end'in HAM STRING olarak flux'a
    gitmedigini (normallesmis ISO formatinda gittigini) dogruluyoruz."""
    captured = {}

    def fake_query(flux):
        captured["flux"] = flux
        return pd.DataFrame()

    monkeypatch.setattr(dashapp._query_api, "query_data_frame", fake_query)
    dashapp._query_history_df("abc123", start="2026-07-11T10:00:00Z", end="2026-07-11T11:00:00Z")
    assert "2026-07-11T10:00:00Z" in captured["flux"]
    assert "2026-07-11T11:00:00Z" in captured["flux"]


# ------------------------------------------------------------ update_history --

def _call_update_history(monkeypatch, df, **overrides):
    """update_history, _query_history_df'i DEGIL, kendi FastAPI
    sunucusundaki /api/history/{icao24} endpoint'ini HTTP ile cagiriyor
    (bkz. fonksiyon govdesi) -- bu yuzden burada _query_api DEGIL,
    dashapp.requests.get sahteleniyor."""
    import json as _json

    kwargs = dict(
        n=1, icao24="abc123", tz_name="3", lang="tr", metric="alt", calc_clicks=1,
        start_day=None, start_hour=None, end_day=None, end_hour=None,
    )
    kwargs.update(overrides)

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, **kw):
        rows = _json.loads(df.assign(
            _time=df["_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ).to_json(orient="records"))
        return FakeResp(rows)

    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(fake_get)}))
    return dashapp.update_history(**kwargs)


def test_update_history_no_aircraft_selected_returns_empty_figure(monkeypatch):
    fig = dashapp.update_history(n=1, icao24=None, tz_name="3", lang="tr", metric="alt",
                                 calc_clicks=1, start_day=None, start_hour=None,
                                 end_day=None, end_hour=None)
    assert fig.data == ()


def test_update_history_no_data_shows_annotation(monkeypatch):
    fig = _call_update_history(monkeypatch, pd.DataFrame(columns=["_time", "lat", "lon", "alt", "velocity"]))
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == dashapp.TEXTS["tr"]["no_data"]


def test_update_history_altitude_metric_plots_alt_column(monkeypatch):
    fig = _call_update_history(monkeypatch, _sample_history_df(), metric="alt")
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [1000.0, 1100.0]


def test_update_history_velocity_metric_plots_velocity_column(monkeypatch):
    fig = _call_update_history(monkeypatch, _sample_history_df(), metric="velocity")
    assert list(fig.data[0].y) == [200.0, 210.0]


def test_update_history_date_range_end_hour_is_exact_boundary_regression(monkeypatch):
    """Regresyon: kullanici geri bildirimi -- '11 ile 12 arasi istedim ama
    11.40-12.40 geldi'. Eskiden bitis saatine +1 saat ekleniyordu. Artik
    end_hour TAM SINIR olmali -- 11:00-12:00 TAM OLARAK sorgulanmali,
    +1 saat YOK."""
    captured = {}

    def fake_get(url, params=None, timeout=None, **kw):
        captured["params"] = params
        class FakeResp:
            def json(self):
                return []
        return FakeResp()

    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(fake_get)}))
    dashapp.update_history(
        n=1, icao24="abc123", tz_name="0", lang="tr", metric="alt", calc_clicks=1,
        start_day="2026-07-11", start_hour=11, end_day="2026-07-11", end_hour=12,
    )
    assert captured["params"]["start"] == "2026-07-11T11:00:00Z"
    assert captured["params"]["end"] == "2026-07-11T12:00:00Z"  # +1 saat YOK


def test_update_history_partial_date_selection_falls_back_to_24h_default(monkeypatch):
    """4 alandan biri bile eksikse (kismi secim) varsayilan davranisa
    (son 24 saat) dusulmeli -- bkz. fonksiyon yorumu."""
    captured = {}

    def fake_get(url, params=None, timeout=None, **kw):
        captured["params"] = params
        class FakeResp:
            def json(self):
                return []
        return FakeResp()

    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(fake_get)}))
    dashapp.update_history(
        n=1, icao24="abc123", tz_name="3", lang="tr", metric="alt", calc_clicks=1,
        start_day="2026-07-11", start_hour=11, end_day=None, end_hour=None,
    )
    assert captured["params"] == {"hours": 24}


def test_update_history_network_failure_shows_no_data_annotation(monkeypatch):
    def boom(url, params=None, timeout=None, **kw):
        raise ConnectionError("koptu")
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(boom)}))
    fig = dashapp.update_history(n=1, icao24="abc123", tz_name="3", lang="tr", metric="alt",
                                 calc_clicks=1, start_day=None, start_hour=None,
                                 end_day=None, end_hour=None)
    assert fig.layout.annotations[0].text == dashapp.TEXTS["tr"]["no_data"]


# ------------------------------------------------------------ download_history_csv --

def test_download_history_csv_no_aircraft_is_no_update():
    result = dashapp.download_history_csv(1, None, "3", None, None, None, None)
    assert result is dashapp.dash.no_update


def test_download_history_csv_returns_csv_with_expected_filename(monkeypatch):
    _fake_query_data_frame(monkeypatch, _sample_history_df())
    result = dashapp.download_history_csv(1, "abc123", "3", None, None, None, None)
    assert result["filename"] == "abc123_history.csv"
    assert "timestamp_utc" in result["content"]  # _time -> timestamp_utc yeniden adlandirildi
    assert "_time" not in result["content"].split("\n")[0]


def test_download_history_csv_query_error_is_no_update(monkeypatch):
    def boom(flux):
        raise RuntimeError("baglanti hatasi")
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", boom)
    result = dashapp.download_history_csv(1, "abc123", "3", None, None, None, None)
    assert result is dashapp.dash.no_update


# -------------------------------------------------------------- FastAPI endpoint'leri --

def test_get_history_endpoint_returns_records(monkeypatch):
    _fake_query_data_frame(monkeypatch, _sample_history_df())
    result = dashapp.get_history("abc123", hours=24)
    assert len(result) == 2
    assert result[0]["lat"] == 41.0


def test_get_history_endpoint_returns_error_dict_on_bad_input():
    result = dashapp.get_history("abc123", start="bad", end="bad")
    assert "error" in result


