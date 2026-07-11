"""Dashboard/dashboard_consumer.py -- cycle-id tabanli "gör-yoksa-sil"
temizlik mantigi (sweep_stale_flights) ve token yukleme testleri.

ONEMLI: sweep_stale_flights, proje sohbet gecmisinde ayrintili tartisilan
3 asamali bir tasarim surecinin (TTL -> zaman penceresi -> cycle-id)
SON hali -- bu testler ozellikle DAHA ONCE denenip TERK EDILEN iki
hatali yaklasimin (hayalet kayit, birden fazla cycle'in birlesimi)
GERI GELMEDIGINI dogrular."""

from __future__ import annotations

import pytest

from Dashboard import dashboard_consumer as consumer
from dashboard_fakes import FakeRedis


def _redis_with_active_flights(*icaos: str) -> FakeRedis:
    r = FakeRedis()
    for icao in icaos:
        r.sadd("iha:active_flights", icao)
        r.set(f"iha:state:{icao}", "{}")
    return r


# -------------------------------------------------------- sweep_stale_flights --

def test_sweep_removes_icao_not_seen_this_cycle():
    r = _redis_with_active_flights("aaa", "bbb", "ccc")
    removed = consumer.sweep_stale_flights(r, baseline_set={"aaa", "bbb", "ccc"},
                                           seen_this_cycle={"aaa", "bbb"})
    assert removed == 1
    assert r.smembers("iha:active_flights") == {"aaa", "bbb"}
    assert r.get("iha:state:ccc") is None


def test_sweep_deletes_both_the_state_key_and_set_membership():
    """Regresyon (1. deneme -- TTL): tekil key silinse bile icao24 kumede
    'hayalet' olarak kalmamali -- IKISI DE silinmeli."""
    r = _redis_with_active_flights("ghost")
    consumer.sweep_stale_flights(r, baseline_set={"ghost"}, seen_this_cycle=set())
    assert "ghost" not in r.smembers("iha:active_flights")
    assert r.get("iha:state:ghost") is None


def test_sweep_no_op_when_nothing_is_stale():
    r = _redis_with_active_flights("aaa", "bbb")
    removed = consumer.sweep_stale_flights(r, baseline_set={"aaa", "bbb"},
                                           seen_this_cycle={"aaa", "bbb"})
    assert removed == 0
    assert r.smembers("iha:active_flights") == {"aaa", "bbb"}


def test_sweep_no_op_when_baseline_is_empty():
    r = FakeRedis()
    assert consumer.sweep_stale_flights(r, baseline_set=set(), seen_this_cycle=set()) == 0


def test_sweep_keeps_icao_seen_in_this_cycle_even_if_new():
    """seen_this_cycle'da olup baseline'da OLMAYAN (yeni gorulen) bir
    icao24, sweep'ten hic etkilenmemeli -- fark sadece baseline-seen
    yonunde hesaplaniyor."""
    r = _redis_with_active_flights("aaa")
    removed = consumer.sweep_stale_flights(r, baseline_set={"aaa"},
                                           seen_this_cycle={"aaa", "brand-new"})
    assert removed == 0


def test_sweep_respects_batch_size_for_srem_chunking():
    """batch_size'i kucuk tutup COK sayida stale kayitla cagirinca,
    hepsi yine de dogru silinmeli (chunking mantigi bug icermemeli)."""
    icaos = [f"ac{i:03d}" for i in range(12)]
    r = _redis_with_active_flights(*icaos)
    removed = consumer.sweep_stale_flights(r, baseline_set=set(icaos),
                                           seen_this_cycle=set(), batch_size=5)
    assert removed == 12
    assert r.smembers("iha:active_flights") == set()


def test_sweep_swallows_exceptions_and_returns_zero():
    """Redis pipeline'i patlarsa (baglanti hatasi vb.) sweep CRASH
    ETMEMELI -- ana consumer dongusu devam edebilmeli."""

    class ExplodingRedis:
        def pipeline(self):
            raise ConnectionError("redis coktu")

    removed = consumer.sweep_stale_flights(ExplodingRedis(), baseline_set={"x"},
                                           seen_this_cycle=set())
    assert removed == 0


def test_sweep_baseline_must_be_captured_before_cycle_not_at_sweep_time():
    """Regresyon (proje sohbet gecmisindeki 'gercek bir hata buradan
    cikti' notu): eger baseline, sweep CAGIRILDIGI ANDA Redis'in GUNCEL
    durumundan alinsaydi (SMEMBERS o an), cycle kendisiyle karsilastirilir
    ve fark HICBIR ZAMAN bulunmazdi. Bu test, fonksiyonun bu YANLIS
    kullanimla (o anki durumu tekrar baseline olarak vermek) SESSIZCE
    sifir donecegini -- yani cagiran tarafin baseline'i ONCEDEN
    saklamasi GEREKTIGINI -- somutlastirir (dokumentasyon amacli)."""
    r = _redis_with_active_flights("aaa", "bbb")
    wrong_baseline = r.smembers("iha:active_flights")  # YANLIS kullanim ornegi
    removed = consumer.sweep_stale_flights(r, baseline_set=wrong_baseline,
                                           seen_this_cycle=wrong_baseline)
    assert removed == 0  # fark bulunamadi -- tam da yorumda anlatilan tuzak


# -------------------------------------------------------------------- load_token --

def test_load_token_prefers_env_var(monkeypatch):
    monkeypatch.setenv("INFLUX_TOKEN", "from-env")
    assert consumer.load_token() == "from-env"


def test_load_token_falls_back_to_file(monkeypatch, tmp_path):
    monkeypatch.delenv("INFLUX_TOKEN", raising=False)
    token_file = tmp_path / "influx_token.txt"
    token_file.write_text("from-file\n")
    monkeypatch.setattr(consumer, "TOKEN_FILE", token_file)
    assert consumer.load_token() == "from-file"


def test_load_token_raises_when_neither_env_nor_file_present(monkeypatch, tmp_path):
    monkeypatch.delenv("INFLUX_TOKEN", raising=False)
    monkeypatch.setattr(consumer, "TOKEN_FILE", tmp_path / "does-not-exist.txt")
    with pytest.raises(SystemExit):
        consumer.load_token()
