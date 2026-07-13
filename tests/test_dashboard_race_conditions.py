"""Dashboard/app.py -- update_map ve render_replay_frame'deki sequence-guard
(yaris durumu koruma) mekanizmasi testleri.

ONEMLI: bu, bu oturumun EN ILK ve en cok debug edilen hatasinin (irtifa
kaydiricisi hizli surukleninde ust uste binen /api/flights istekleri
TAMAMLANDIKLARI SIRAYLA donuyordu -- daha ESKI ama DAHA GEC biten bir
yanit, daha YENI ama daha erken biten bir yaniti EZIYORDU) dogrudan
regresyon testidir. O zamana kadar bu mekanizma HICBIR testte
kapsanmiyordu -- ironik sekilde, oturumun en cok emek harcanan hatasi
test edilmeyen tek yerdi.

Test yontemi: gercek threading/concurrency kullanmiyoruz (flaky/
deterministik-olmayan olurdu) -- bunun yerine, sahte requests.get'in
"ag cagrisi sirasinda" (yani orijinal callback SU AN "beklerken") modul
seviyesindeki sayaci ELLE bir adim ileri alarak "bu sirada BASKA bir
cagri baslamis" durumunu KESIN/tekrarlanabilir sekilde simule ediyoruz."""

from __future__ import annotations

from Dashboard import app as dashapp
from dashboard_fakes import FakeRequestsRouter


def _bump_altitude_seq(_url):
    """update_map'in KENDI /api/flights istegi SURERKEN, BASKA (daha
    yeni) bir update_map cagrisinin BASLADIGINI simule eder."""
    with dashapp._altitude_map_seq_lock:
        dashapp._altitude_map_latest_seq += 1


def _bump_replay_seq(_url):
    with dashapp._replay_frame_seq_lock:
        dashapp._replay_frame_latest_seq += 1


# ------------------------------------------------------------------ update_map --

def test_update_map_discards_stale_result_when_newer_call_started_during_fetch(monkeypatch):
    """ASIL regresyon testi: /api/flights isteği SURERKEN daha yeni bir
    update_map cagrisi baslarsa, bu (yavas/eski) cagrinin sonucu ASLA
    yazilmamali -- 4 cikti da dash.no_update olmali."""
    router = FakeRequestsRouter(
        routes={"flights": [{"icao24": "aaa", "is_military": False, "alt": 1000.0}],
                "alerts": []},
        on_call=_bump_altitude_seq,
    )
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    result = dashapp.update_map(
        n=1, tz_name="3", lang="tr", show_civil=True, show_military=True,
        show_ground=True, replay_mode=False,
        altitude_filter_range=[-1_000_000, 1_000_000], staleness_threshold=60,
    )
    assert result == (dashapp.dash.no_update, dashapp.dash.no_update,
                      dashapp.dash.no_update, dashapp.dash.no_update)


def test_update_map_returns_real_data_when_no_interleaving_call(monkeypatch):
    """Karsit durum: ARADA baska bir cagri BASLAMAZSA (normal, izole
    cagri), sonuc GERCEKTEN yazilmali -- guard yanlislikla HER ZAMAN
    reddetmiyor, SADECE gercekten bayatladiginda reddediyor."""
    router = FakeRequestsRouter(routes={
        "flights": [{"icao24": "aaa", "is_military": False, "alt": 1000.0,
                     "callsign": "THY123", "lat": 41.0, "lon": 29.0}],
        "alerts": [],
    })
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    raw_data, status_main, emergency_rows, status_alarm = dashapp.update_map(
        n=1, tz_name="3", lang="tr", show_civil=True, show_military=True,
        show_ground=True, replay_mode=False,
        altitude_filter_range=[-1_000_000, 1_000_000], staleness_threshold=60,
    )
    assert raw_data is not dashapp.dash.no_update
    assert len(raw_data["features"]) == 1
    assert raw_data["features"][0]["properties"]["icao24"] == "aaa"
    assert "1 aktif uçuş" in status_main


def test_update_map_sequence_counter_always_advances(monkeypatch):
    """my_seq'in HER cagrida artmasi (asla ayni/geri gitmemesi) -- guard
    mantiginin temel varsayimi."""
    before = dashapp._altitude_map_latest_seq
    router = FakeRequestsRouter(routes={"flights": [], "alerts": []})
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    dashapp.update_map(
        n=1, tz_name="3", lang="tr", show_civil=True, show_military=True,
        show_ground=True, replay_mode=False,
        altitude_filter_range=[-1_000_000, 1_000_000], staleness_threshold=60,
    )
    after = dashapp._altitude_map_latest_seq
    assert after == before + 1


def test_update_map_replay_mode_skips_fetch_entirely(monkeypatch):
    """replay-mode acikken update_map HICBIR ag istegi atmamali (bkz.
    render_replay_frame ile carpismama yorumu) -- erken donus."""
    router = FakeRequestsRouter(routes={"flights": [], "alerts": []})
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    result = dashapp.update_map(
        n=1, tz_name="3", lang="tr", show_civil=True, show_military=True,
        show_ground=True, replay_mode=True,
        altitude_filter_range=[-1_000_000, 1_000_000], staleness_threshold=60,
    )
    assert result == (dashapp.dash.no_update,) * 4
    assert router.calls == []  # HICBIR istek atilmadi


# ------------------------------------------------------------ render_replay_frame --

def _replay_data(step_sec=30, n_steps=3):
    return {"steps": [f"2026-07-11T10:0{i}:00Z" for i in range(n_steps)],
            "step_sec": step_sec}


def test_render_replay_frame_discards_stale_result_when_newer_tick_started_during_fetch(monkeypatch):
    """update_map ile AYNI hata sinifi, replay tarafinda -- kullanici
    geri bildirimi: '1de takılı kalıyor, durdurunca 8e geçiyor' (bkz.
    modul ustundeki yorum)."""
    router = FakeRequestsRouter(
        routes={"replay_frame": [{"icao24": "bbb", "lat": 41.0, "lon": 29.0, "alt": 5000.0}]},
        on_call=_bump_replay_seq,
    )
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    result = dashapp.render_replay_frame(index=1, data=_replay_data(), lang="tr")
    assert result == (dashapp.dash.no_update, dashapp.dash.no_update)


def test_render_replay_frame_returns_real_data_when_no_interleaving(monkeypatch):
    router = FakeRequestsRouter(routes={
        "replay_frame": [{"icao24": "bbb", "lat": 41.0, "lon": 29.0, "alt": 5000.0}],
    })
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    raw_data, label = dashapp.render_replay_frame(index=1, data=_replay_data(), lang="tr")
    assert raw_data is not dashapp.dash.no_update
    assert len(raw_data["features"]) == 1
    assert label.startswith("2/3")


def test_render_replay_frame_out_of_range_index_short_circuits_without_fetch(monkeypatch):
    """index adim listesinin disindaysa, AG ISTEGI ATMADAN hemen
    no_update donmeli -- guard'in bile devreye girmesine gerek yok."""
    router = FakeRequestsRouter(routes={"replay_frame": []})
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    result = dashapp.render_replay_frame(index=99, data=_replay_data(n_steps=3), lang="tr")
    assert result == (dashapp.dash.no_update, "")
    assert router.calls == []


def test_render_replay_frame_empty_steps_short_circuits():
    result = dashapp.render_replay_frame(index=0, data={"steps": []}, lang="tr")
    assert result == (dashapp.dash.no_update, "")


def test_render_replay_frame_error_payload_treated_as_empty_frame(monkeypatch):
    """/api/replay_frame bir hata sozlugu {'error': ...} dondurebilir --
    bu bir ucak LISTESI degil, bos liste gibi ele alinmali (crash yerine)."""
    router = FakeRequestsRouter(routes={"replay_frame": {"error": "InfluxDB down"}})
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    raw_data, label = dashapp.render_replay_frame(index=0, data=_replay_data(), lang="tr")
    assert raw_data["features"] == []


def test_render_replay_frame_skips_features_missing_lat_lon(monkeypatch):
    router = FakeRequestsRouter(routes={
        "replay_frame": [
            {"icao24": "has-pos", "lat": 41.0, "lon": 29.0, "alt": 1000.0},
            {"icao24": "no-pos", "lat": None, "lon": 29.0, "alt": 1000.0},
        ],
    })
    monkeypatch.setattr(dashapp, "requests", type("R", (), {"get": staticmethod(router)}))

    raw_data, _ = dashapp.render_replay_frame(index=0, data=_replay_data(), lang="tr")
    icaos = [f["properties"]["icao24"] for f in raw_data["features"]]
    assert icaos == ["has-pos"]
