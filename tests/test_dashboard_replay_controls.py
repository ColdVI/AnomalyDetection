"""Dashboard/app.py -- tekrar oynatma (replay) veri yukleme ve oynatma
kontrolu testleri.

ONEMLI: test_advance_replay_tick_* testleri, oturumda tartisilan "hız
kontrolü çok yavaş render/InfluxDB isteği sıklığı yaratıyordu" sorununu
cozen kesirli-ilerleme (fractional progress carry) mantigini kapsar --
0.5x/4x/8x hizlarda render SIKLIGININ (tick=2sn sabit) DEGISMEDIGINI,
sadece adim sayisinin degistigini dogrular."""

from __future__ import annotations

import pandas as pd
import pytest

from Dashboard import app as dashapp


# -------------------------------------------------------------------- get_replay --

def test_get_replay_invalid_dates_returns_error():
    result = dashapp.get_replay("not-a-date", "also-not-a-date")
    assert "error" in result


def test_get_replay_end_before_start_returns_error():
    result = dashapp.get_replay("2026-07-11T12:00:00Z", "2026-07-11T10:00:00Z")
    assert "error" in result


def test_get_replay_range_capped_at_max_hours(monkeypatch):
    """REPLAY_MAX_RANGE_HOURS'i asan bir aralik istenirse, sorgu/payload
    boyutunu makul tutmak icin sessizce kirpilmali (hata degil) -- bkz.
    fonksiyon docstring'i."""
    captured = {}

    def fake_query(flux):
        captured["flux"] = flux
        return pd.DataFrame()

    monkeypatch.setattr(dashapp._query_api, "query_data_frame", fake_query)
    dashapp.get_replay("2026-07-11T00:00:00Z", "2026-07-11T23:59:00Z")  # ~24 saat istendi
    # REPLAY_MAX_RANGE_HOURS (2) sonra bitmesi gerekiyordu -- 02:00'dan
    # sonraki bir zaman damgasi flux'ta GORUNMEMELI.
    assert "T23:59:00Z" not in captured["flux"]
    assert "T02:00:00Z" in captured["flux"]


def test_get_replay_step_sec_clamped_between_5_and_300(monkeypatch):
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: pd.DataFrame())
    assert dashapp.get_replay("2026-07-11T00:00:00Z", "2026-07-11T01:00:00Z",
                              step_sec=1)["step_sec"] == 5
    assert dashapp.get_replay("2026-07-11T00:00:00Z", "2026-07-11T01:00:00Z",
                              step_sec=9999)["step_sec"] == 300


def test_get_replay_no_data_returns_empty_steps(monkeypatch):
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: pd.DataFrame())
    result = dashapp.get_replay("2026-07-11T00:00:00Z", "2026-07-11T01:00:00Z")
    assert result["steps"] == []


def test_get_replay_buckets_timestamps_into_step_intervals(monkeypatch):
    df = pd.DataFrame({"_time": pd.to_datetime([
        "2026-07-11T00:00:05Z", "2026-07-11T00:00:12Z",  # ayni 30sn kovasi (0)
        "2026-07-11T00:00:35Z",  # 2. kova (30)
    ], utc=True)})
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", lambda flux: df)
    result = dashapp.get_replay("2026-07-11T00:00:00Z", "2026-07-11T01:00:00Z", step_sec=30)
    assert result["steps"] == ["2026-07-11T00:00:00Z", "2026-07-11T00:00:30Z"]


def test_get_replay_query_exception_returns_error(monkeypatch):
    def boom(flux):
        raise RuntimeError("influx coktu")
    monkeypatch.setattr(dashapp._query_api, "query_data_frame", boom)
    result = dashapp.get_replay("2026-07-11T00:00:00Z", "2026-07-11T01:00:00Z")
    assert "error" in result


# --------------------------------------------------------------- load_replay_data --

def test_load_replay_data_missing_field_shows_no_data_feedback():
    result = dashapp.load_replay_data(1, None, None, "2026-07-11", 10, "3", "tr")
    assert result[4] == dashapp.TEXTS["tr"]["replay_no_data"]
    assert result[0] is dashapp.dash.no_update  # replay-mode'a DOKUNULMADI


def test_load_replay_data_no_steps_found_disables_replay_mode(monkeypatch):
    monkeypatch.setattr(dashapp, "get_replay", lambda *a, **kw: {"steps": [], "step_sec": 30})
    data, index, playing, mode, feedback, progress = dashapp.load_replay_data(
        1, "2026-07-11", 10, "2026-07-11", 11, "3", "tr",
    )
    assert mode is False
    assert playing is False
    assert feedback == dashapp.TEXTS["tr"]["replay_no_data"]


def test_load_replay_data_success_enables_replay_mode(monkeypatch):
    steps = ["2026-07-11T10:00:00Z", "2026-07-11T10:00:30Z"]
    monkeypatch.setattr(dashapp, "get_replay",
                        lambda *a, **kw: {"steps": steps, "step_sec": 30})
    data, index, playing, mode, feedback, progress = dashapp.load_replay_data(
        1, "2026-07-11", 10, "2026-07-11", 11, "3", "tr",
    )
    assert mode is True
    assert playing is False  # yuklenince otomatik OYNATILMAZ, kullanici ▶'a basmali
    assert index == 0
    assert progress == 0.0
    assert data["steps"] == steps
    assert "2" in feedback  # "2 kare yuklendi"


def test_load_replay_data_resets_progress_from_previous_session(monkeypatch):
    """Onceki oturumdan kalma kesirli ilerleme (replay-progress), yeni
    yuklemeyle KARISMAMALI -- her yeni 'Yükle' 0.0'dan baslamali."""
    monkeypatch.setattr(dashapp, "get_replay",
                        lambda *a, **kw: {"steps": ["x"], "step_sec": 30})
    result = dashapp.load_replay_data(1, "2026-07-11", 10, "2026-07-11", 11, "3", "tr")
    assert result[5] == 0.0


# ----------------------------------------------------------- oynatma kontrolleri --

def test_toggle_replay_play_flips_state_when_data_present():
    assert dashapp.toggle_replay_play(1, False, {"steps": ["x"]}) is True
    assert dashapp.toggle_replay_play(1, True, {"steps": ["x"]}) is False
    assert dashapp.toggle_replay_play(1, False, {"steps": []}) is dashapp.dash.no_update


def test_update_replay_play_label_shows_pause_when_playing():
    assert dashapp.update_replay_play_label(True) == "⏸"
    assert dashapp.update_replay_play_label(False) == "▶"


def test_sync_replay_interval_disabled_when_not_playing():
    assert dashapp.sync_replay_interval(True) is False
    assert dashapp.sync_replay_interval(False) is True


def test_set_replay_speed_defaults_to_one_when_falsy():
    assert dashapp.set_replay_speed(None) == 1
    assert dashapp.set_replay_speed(0) == 1
    assert dashapp.set_replay_speed(4) == 4


def test_exit_replay_mode_disables_mode_and_playback():
    assert dashapp.exit_replay_mode(1) == (False, False)


# ------------------------------------------------------------- advance_replay_tick --

def test_advance_replay_tick_no_steps_is_no_op():
    result = dashapp.advance_replay_tick(1, 0, {"steps": []}, 1, 0.0)
    assert result == (dashapp.dash.no_update, dashapp.dash.no_update)


def test_advance_replay_tick_1x_speed_advances_one_step_per_tick():
    index, progress = dashapp.advance_replay_tick(1, 0, {"steps": ["a", "b", "c"]}, 1, 0.0)
    assert index == 1
    assert progress == 0.0


def test_advance_replay_tick_half_speed_needs_two_ticks_for_one_step_regression():
    """0.5x hiz -- ilk tick'te ilerleme SADECE 0.5'e ulasir, adim
    DEGISMEMELI, sadece kesir tasinmali (bkz. fonksiyon docstring'i)."""
    index1, progress1 = dashapp.advance_replay_tick(1, 0, {"steps": ["a", "b", "c"]}, 0.5, 0.0)
    assert index1 is dashapp.dash.no_update
    assert progress1 == 0.5

    index2, progress2 = dashapp.advance_replay_tick(2, 0, {"steps": ["a", "b", "c"]}, 0.5, progress1)
    assert index2 == 1  # ikinci tick'te birikim 1.0'a ulasti, TAM OLARAK 1 adim
    assert progress2 == pytest.approx(0.0)


def test_advance_replay_tick_high_speed_jumps_multiple_steps_but_one_render():
    """4x hiz -- TEK bir tick'te 4 adim birden atlanmali (render/InfluxDB
    istek SIKLIGI degismez, tick hala 2sn'de bir -- bkz. docstring)."""
    index, progress = dashapp.advance_replay_tick(
        1, 0, {"steps": list("abcdefgh")}, 4, 0.0,
    )
    assert index == 4
    assert progress == 0.0


def test_advance_replay_tick_wraps_around_at_end_of_steps():
    index, _ = dashapp.advance_replay_tick(1, 8, {"steps": list("abcdefghij")}, 3, 0.0)
    assert index == 1  # (8+3) % 10 = 1


def test_advance_replay_tick_fractional_progress_persists_across_calls():
    """1.5x hizda kesir TEK adimi asmiyor ama BIRIKIYOR -- 3 tick'te tam
    olarak 4-5 adim ilerlenmis olmali (1.5*3=4.5)."""
    steps = list("abcdefghij")
    index, progress = 0, 0.0
    total_advanced = 0
    for n in range(1, 4):
        new_index, progress = dashapp.advance_replay_tick(n, index, {"steps": steps}, 1.5, progress)
        if new_index is not dashapp.dash.no_update:
            total_advanced += (new_index - index) % len(steps)
            index = new_index
    assert total_advanced == 4  # int(1.5)+int(1.5)+int(1.5+kesirler) toplami


# ------------------------------------------------------------ set_replay_default_range --

def test_set_replay_default_range_panel_closed_is_no_op():
    result = dashapp.set_replay_default_range(False, None, None, None, None, "3")
    assert result == (dashapp.dash.no_update,) * 4


def test_set_replay_default_range_all_fields_already_set_is_no_op():
    """Kullanicinin KENDI secimi EZILMEMELI (bkz. fonksiyon docstring'i)."""
    result = dashapp.set_replay_default_range(True, "2026-07-10", 5, "2026-07-10", 6, "3")
    assert result == (dashapp.dash.no_update,) * 4


def test_set_replay_default_range_fills_last_one_hour_when_empty():
    """Regresyon (canli testte yakalanan sorun): panel ilk acildiginda 4
    alan da bosken, 'son 1 saat' gibi makul bir varsayilan otomatik
    doldurulmali -- kullanici hic dokunmadan 'Yükle' calissin diye."""
    start_day, start_hour, end_day, end_hour = dashapp.set_replay_default_range(
        True, None, None, None, None, "3",
    )
    assert start_day is not dashapp.dash.no_update
    assert end_day is not dashapp.dash.no_update
    # baslangic <= bitis (ayni gun ya da bir onceki gun, saat farkina gore)
    assert (start_day, start_hour) <= (end_day, end_hour)
