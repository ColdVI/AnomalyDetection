"""Dashboard/uav_producer.py -- kaynak-normalize edici (adsb.lol/OpenSky)
saf ayristirma fonksiyonlari testleri.

ONEMLI: signal_age_sec testleri, proje sohbet gecmisindeki "OpenSky'de
neredeyse her uçak saydam" tartismasinin dogrudan temelini olusturan
fonksiyonlari kapsar -- iki kaynagin da AYNI birimde (saniye once)
tutarli signal_age_sec uretmesi, dashboard'daki opacity hesabinin
(bkz. test_dashboard_signal_staleness.py) dogru calismasi icin sarttir."""

from __future__ import annotations

import time

import pytest

from Dashboard import uav_producer as prod


# ------------------------------------------------------------- _normalize_common --

def test_normalize_common_builds_expected_schema():
    rec = prod._normalize_common(
        icao24="abc123", callsign="THY123", lat=41.0, lon=29.0, alt_m=1000.0,
        velocity_ms=200.0, track_deg=90.0, vertical_rate_ms=0.5, category="A3",
        squawk="1200", emergency="none", is_military=False, source="test",
    )
    assert rec["icao24"] == "abc123"
    assert rec["is_ground"] is False  # varsayilan
    assert rec["signal_age_sec"] is None  # varsayilan
    assert "ts" in rec and rec["ts"].endswith("+00:00")


def test_normalize_common_preserves_is_ground_and_signal_age():
    rec = prod._normalize_common(
        icao24="x", callsign="", lat=0, lon=0, alt_m=0, velocity_ms=None,
        track_deg=None, vertical_rate_ms=None, category="", squawk="",
        emergency="none", is_military=False, source="test",
        is_ground=True, signal_age_sec=42.5,
    )
    assert rec["is_ground"] is True
    assert rec["signal_age_sec"] == 42.5


# ------------------------------------------------------- _parse_adsblol_aircraft --

def _adsblol_ac(**overrides):
    base = {
        "hex": "abc123", "lat": 41.0, "lon": 29.0, "alt_baro": 3000,
        "gs": 250, "baro_rate": 500, "track": 90, "flight": "THY123",
        "category": "A3", "squawk": "1200", "emergency": "none",
        "dbFlags": 0, "seen_pos": 5, "seen": 5,
    }
    base.update(overrides)
    return base


def test_parse_adsblol_aircraft_basic_fields():
    rec = prod._parse_adsblol_aircraft(_adsblol_ac())
    assert rec["icao24"] == "abc123"
    assert rec["callsign"] == "THY123"
    assert rec["is_ground"] is False
    assert rec["is_military"] is False


def test_parse_adsblol_aircraft_altitude_converted_feet_to_meters():
    rec = prod._parse_adsblol_aircraft(_adsblol_ac(alt_baro=10000))
    assert rec["alt"] == pytest.approx(10000 * 0.3048, abs=0.1)


def test_parse_adsblol_aircraft_ground_string_altitude_regression():
    """Regresyon: adsb.lol yerdeki bir ucak icin alt_baro yerine DUZ METIN
    'ground' donduruyor -- eskiden bu durumda kayit TAMAMEN ATILIYORDU
    (bkz. Dashboard/docs/HANDOFF_UPDATE_2026-07-07.md, 'adsb.lol toplam sayisi bizden
    yuksek cikiyor' kok nedeni). Artik is_ground=True ile isaretlenip
    irtifa 0 kabul edilmeli, kayit ASLA None DONMEMELI."""
    rec = prod._parse_adsblol_aircraft(_adsblol_ac(alt_baro="ground"))
    assert rec is not None
    assert rec["is_ground"] is True
    assert rec["alt"] == 0.0


def test_parse_adsblol_aircraft_missing_lat_lon_returns_none():
    assert prod._parse_adsblol_aircraft(_adsblol_ac(lat=None)) is None
    assert prod._parse_adsblol_aircraft(_adsblol_ac(lon=None)) is None


def test_parse_adsblol_aircraft_missing_hex_returns_none():
    assert prod._parse_adsblol_aircraft(_adsblol_ac(hex="")) is None


def test_parse_adsblol_aircraft_missing_ground_speed_stays_none_not_zero():
    """gs alani eksikse sahte 0 YAZILMAMALI -- gercek bir bosluk olarak
    None kalmali (bkz. fonksiyon yorumu)."""
    ac = _adsblol_ac()
    del ac["gs"]
    rec = prod._parse_adsblol_aircraft(ac)
    assert rec["velocity"] is None


def test_parse_adsblol_aircraft_military_bit_flag():
    rec = prod._parse_adsblol_aircraft(_adsblol_ac(dbFlags=1))
    assert rec["is_military"] is True
    rec2 = prod._parse_adsblol_aircraft(_adsblol_ac(dbFlags=2))
    assert rec2["is_military"] is False  # sadece 1. bit askeri anlamina gelir


def test_parse_adsblol_aircraft_prefers_seen_pos_over_seen_regression():
    """Regresyon: 'seen' (herhangi bir mesaj) dusuk kalirken ucak her
    cycle'da 'goruldu' sayilip cycle-temizligini hicbir zaman
    tetiklemeyebilir -- oysa GERCEK pozisyonu (seen_pos) dakikalarca
    guncellenmemis olabilir. signal_age_sec MUTLAKA seen_pos'u kullanmali."""
    rec = prod._parse_adsblol_aircraft(_adsblol_ac(seen_pos=180, seen=2))
    assert rec["signal_age_sec"] == 180.0


def test_parse_adsblol_aircraft_falls_back_to_seen_when_seen_pos_missing():
    ac = _adsblol_ac(seen=7)
    del ac["seen_pos"]
    rec = prod._parse_adsblol_aircraft(ac)
    assert rec["signal_age_sec"] == 7.0


# --------------------------------------------------------- _parse_opensky_state --

def _opensky_state(overrides: dict | None = None):
    # index: 0=icao24 1=callsign 2=country 3=time_position 4=last_contact
    # 5=lon 6=lat 7=baro_alt 8=on_ground 9=velocity 10=track 11=vrate
    # 12=sensors 13=geo_alt 14=squawk 15=spi 16=position_source
    state = ["abc123", "THY123 ", "Turkey", 1000, 1000, 29.0, 41.0,
             3000.0, False, 200.0, 90.0, 0.5, None, 3100.0, "1200", False, 0]
    for index, value in (overrides or {}).items():
        state[index] = value
    return state


def test_parse_opensky_state_basic_fields():
    rec = prod._parse_opensky_state(_opensky_state())
    assert rec["icao24"] == "abc123"
    assert rec["callsign"] == "THY123"  # strip edilmis
    assert rec["source"] == "opensky"
    assert rec["is_military"] is False  # OpenSky bu bilgiyi hic vermiyor


def test_parse_opensky_state_missing_lat_lon_returns_none():
    assert prod._parse_opensky_state(_opensky_state({5: None})) is None
    assert prod._parse_opensky_state(_opensky_state({6: None})) is None


def test_parse_opensky_state_integer_fields_always_cast_to_float_regression():
    """Regresyon: OpenSky JSON'da tam sayi gelen alanlar (orn. velocity=200,
    0.5 degil) Python'da int olarak parse edilebilir -- InfluxDB tip
    tutarliligi icin bunlar HER ZAMAN float'a cevrilmis olmali."""
    rec = prod._parse_opensky_state(_opensky_state({9: 200, 10: 90, 11: 0}))
    assert isinstance(rec["velocity"], float)
    assert isinstance(rec["track"], float)
    assert isinstance(rec["vertical_rate"], float)


def test_parse_opensky_state_signal_age_computed_from_last_contact():
    now = time.time()
    rec = prod._parse_opensky_state(_opensky_state({4: now - 42}))
    assert rec["signal_age_sec"] == pytest.approx(42, abs=1)


def test_parse_opensky_state_missing_last_contact_leaves_signal_age_none():
    rec = prod._parse_opensky_state(_opensky_state({4: None}))
    assert rec["signal_age_sec"] is None


def test_parse_opensky_state_emergency_squawk_mapping():
    rec = prod._parse_opensky_state(_opensky_state({14: "7700"}))
    assert rec["emergency"] == "general"
    rec2 = prod._parse_opensky_state(_opensky_state({14: "1200"}))
    assert rec2["emergency"] == "none"


def test_parse_opensky_state_missing_baro_altitude_defaults_to_zero_not_none():
    rec = prod._parse_opensky_state(_opensky_state({7: None}))
    assert rec["alt"] == 0.0


# ------------------------------------------------------------------- SOURCES --

def test_sources_has_adsblol_and_opensky():
    assert set(prod.SOURCES.keys()) == {"adsblol", "opensky"}


def test_sources_adsblol_interval_matches_its_own_measured_fetch_time():
    """adsb.lol'un tek dev sorgusu ~57-61sn suruyor (bkz. HIZ ARASTIRMASI
    yorumu) -- interval bundan KISA olamaz, aksi halde ust uste binen
    istekler baslar."""
    assert prod.SOURCES["adsblol"]["interval"] >= 60


def test_sources_opensky_anonymous_interval_respects_daily_credit_budget():
    """Anonim erisimde gunde 400 kredi, 4 kredi/istek -- 400/4=100 istek/gun
    -> 86400/100=864sn minimum guvenli aralik OLMASI GEREKMEZ (kullanici
    daha once 300sn'ye ayarlamayi secti, bariz kota asimi olmasin diye en
    az adil-kullanim seviyesinde kalmali)."""
    assert prod.SOURCES["opensky"]["interval"] >= 300


def test_opensky_auth_interval_stays_within_daily_credit_budget():
    """CANLI OLCUM (bkz. uav_producer.py yorumu): bbox'siz TEK istek 4
    kredi tuketiyor. Kimlik dogrulamali erisimde 4000 kredi/gun ->
    4000/4=1000 istek/gun -> minimum 86400/1000=86.4sn. OPENSKY_AUTH_INTERVAL
    bunun ALTINDA OLURSA kota gun ortasinda biter (regresyon)."""
    min_safe_interval = 86400 / (4000 / 4)
    assert prod.OPENSKY_AUTH_INTERVAL >= min_safe_interval
