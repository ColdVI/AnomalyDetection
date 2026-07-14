"""Tests for src/silver/parse_adsblol_historical.py"""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pandas as pd
import pytest

from src.common.fakes import FakeMinioClient
from src.common.minio_io import list_layer_objects, write_silver
from src.silver.parse_adsblol_historical import (
    SILVER_SCHEMA_VERSION,
    _delete_uris,
    _load_checkpoint,
    _parse_tar_fileobj,
    _save_checkpoint,
    parse_trace_bytes,
    run,
)


def _make_tar_bytes(aircraft_records: list[dict]) -> bytes:
    """Build an in-memory tar with one gzip-compressed JSON per record."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i, rec in enumerate(aircraft_records):
            payload = gzip.compress(json.dumps(rec).encode("utf-8"))
            info = tarfile.TarInfo(name=f"traces/ab/{rec.get('icao', 'xx')}.json")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


SAMPLE_AIRCRAFT = {
    "icao": "abc123",
    "timestamp": 1_700_000_000,
    "r": "TC-ABC",
    "t": "B738",
    "trace": [
        [0, 40.0, 29.0, 35000, 450, 90, 0, 0, None, "adsb_icao", 35100, 0, None, None],
        [10, 40.1, 29.1, "ground", 0, 0, 0, 0, None, "adsb_icao", None, 0, None, None],
        [20, 40.2, 29.2, 36000, 460, 91, 0, -500, None, "adsb_icao", 36100, -500, 440, 2.5],
    ],
}


def test_parse_trace_bytes_unit_conversions():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert len(df) == 3

    # feet → metres: 35000 * 0.3048 = 10668.0
    assert df.loc[0, "alt"] == pytest.approx(35000 * 0.3048, abs=0.2)
    # on_ground row
    assert bool(df.loc[1, "on_ground"]) is True
    assert pd.isna(df.loc[1, "alt"])
    # ground_speed_ms: 450 knots → m/s
    assert df.loc[0, "ground_speed_ms"] == pytest.approx(450 * 0.5144, abs=0.01)
    # vertical_rate_ms: -500 fpm → m/s
    assert df.loc[2, "vertical_rate_ms"] == pytest.approx(-500 * 0.00508, abs=0.001)


@pytest.mark.parametrize("db_flags,expected", [
    (1, True), (3, True), (0, False), (2, False), (None, False), ("garbage", False),
])
def test_parse_trace_bytes_is_military_bit_flag(db_flags, expected):
    record = dict(SAMPLE_AIRCRAFT)
    if db_flags is not None:
        record["dbFlags"] = db_flags
    raw = gzip.compress(json.dumps(record).encode())
    df = parse_trace_bytes(raw)
    assert (df["is_military"] == expected).all()


def test_parse_trace_bytes_source_fields():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert (df["source_type"] == "adsblol_historical").all()
    assert (df["source_id"] == "abc123").all()
    assert (df["registration"] == "TC-ABC").all()
    assert (df["label"].isna()).all()


def test_parse_trace_bytes_timestamp():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert df.loc[0, "timestamp_utc"] == 1_700_000_000 + 0
    assert df.loc[2, "timestamp_utc"] == 1_700_000_000 + 20


def test_sparse_s2_updates_are_distinct_from_forward_filled_values():
    record = {
        "icao": "fresh1",
        "timestamp": 1_700_000_000,
        "trace": [
            [
                0, 40.0, 29.0, 10000, 200, 90, 0, 0,
                {
                    "squawk": "7700", "emergency": "general", "nic": 8,
                    "rc": 186, "nac_p": 9, "sil": 3, "version": 2,
                    "sil_type": "perhour", "sda": 2, "nac_v": 2,
                },
                "adsb_icao", 10100, 0, None, None,
            ],
            [
                10, 40.1, 29.1, 10010, 200, 90, 0, 0,
                {"nic": 7}, "adsb_icao", 10110, 0, None, None,
            ],
            [
                30, 40.2, 29.2, 10020, 200, 90, 2, 0,
                None, "adsb_icao", 10120, 0, None, None,
            ],
        ],
    }

    df = parse_trace_bytes(gzip.compress(json.dumps(record).encode()))

    # Values remain forward-filled, but only actual key presence is an update.
    assert df["squawk"].tolist() == ["7700", "7700", "7700"]
    assert df["squawk_updated"].tolist() == [True, False, False]
    assert df["squawk_update_timestamp_utc"].tolist() == [
        1_700_000_000,
        1_700_000_000,
        1_700_000_000,
    ]
    assert df["squawk_update_age_s"].tolist() == [0.0, 10.0, 30.0]

    assert df["nic"].tolist() == [8, 7, 7]
    assert df["nic_updated"].tolist() == [True, True, False]
    assert df["nic_update_timestamp_utc"].tolist() == [
        1_700_000_000,
        1_700_000_010,
        1_700_000_010,
    ]
    assert df["nic_update_age_s"].tolist() == [0.0, 0.0, 20.0]

    # A new-leg flag does not masquerade as a new S2 transmission.
    assert bool(df.loc[2, "flags_new_leg"]) is True
    assert bool(df.loc[2, "emergency_updated"]) is False
    assert df.loc[2, "emergency_update_age_s"] == 30.0

    for field in ("rc", "nac_p", "sil", "adsb_version", "sil_type", "sda", "nac_v"):
        assert bool(df.loc[0, f"{field}_updated"]) is True
        assert bool(df.loc[1, f"{field}_updated"]) is False
        assert df.loc[1, f"{field}_update_age_s"] == 10.0


def test_explicit_null_is_a_fresh_clear_not_an_absent_update():
    record = {
        "icao": "clear1",
        "timestamp": 1000,
        "trace": [
            [0, 1, 1, 1000, 100, 0, 0, 0, {"emergency": "general"}, "adsb_icao"],
            [5, 1, 1, 1000, 100, 0, 0, 0, {"emergency": None}, "adsb_icao"],
            [8, 1, 1, 1000, 100, 0, 0, 0, None, "adsb_icao"],
        ],
    }

    df = parse_trace_bytes(json.dumps(record).encode())

    assert df.loc[0, "emergency"] == "general"
    assert pd.isna(df.loc[1, "emergency"])
    assert pd.isna(df.loc[2, "emergency"])
    assert df["emergency_updated"].tolist() == [True, True, False]
    assert df["emergency_update_timestamp_utc"].tolist() == [1000.0, 1005.0, 1005.0]
    assert df["emergency_update_age_s"].tolist() == [0.0, 0.0, 3.0]


def test_s2_update_state_resets_at_each_trace_boundary():
    first = {
        "icao": "sameicao",
        "timestamp": 1000,
        "trace": [[0, 1, 1, 1000, 100, 0, 0, 0, {"squawk": "7600"}, "adsb_icao"]],
    }
    second = {
        "icao": "sameicao",
        "timestamp": 2000,
        "trace": [[0, 1, 1, 1000, 100, 0, 0, 0, None, "adsb_icao"]],
    }

    assert parse_trace_bytes(json.dumps(first).encode()).loc[0, "squawk"] == "7600"
    parsed_second = parse_trace_bytes(json.dumps(second).encode())
    assert pd.isna(parsed_second.loc[0, "squawk"])
    assert bool(parsed_second.loc[0, "squawk_updated"]) is False
    assert pd.isna(parsed_second.loc[0, "squawk_update_timestamp_utc"])
    assert pd.isna(parsed_second.loc[0, "squawk_update_age_s"])


def test_parse_tar_fileobj_writes_silver(fake_minio_client: FakeMinioClient):
    tar_bytes = _make_tar_bytes([SAMPLE_AIRCRAFT])
    uris = _parse_tar_fileobj(
        io.BytesIO(tar_bytes), "test.tar", batch_size=100, client=fake_minio_client
    )

    assert len(uris) == 1
    assert uris[0].startswith("s3://silver/")

    # Silver Parquet contains provenance
    stored = list(fake_minio_client.buckets["silver"].values())[0]
    df = pd.read_parquet(io.BytesIO(stored))
    assert "_source_type" in df.columns
    assert (df["_source_type"] == "adsblol_historical").all()
    assert (df["_source_file"] == "test.tar").all()
    assert (df["_schema_version"] == SILVER_SCHEMA_VERSION).all()


def test_parse_tar_handles_non_gzip_json(fake_minio_client: FakeMinioClient):
    """Members that are plain (non-gzip) JSON should also parse without error."""
    plain_rec = dict(SAMPLE_AIRCRAFT)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        payload = json.dumps(plain_rec).encode("utf-8")
        info = tarfile.TarInfo(name="traces/ab/plain.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    uris = _parse_tar_fileobj(
        io.BytesIO(buf.getvalue()), "plain.tar", batch_size=100, client=fake_minio_client
    )
    assert len(uris) == 1


def test_parse_tar_fileobj_calls_on_part_written_after_each_flush():
    """run() bu callback'i checkpoint'e HER Silver parcasi yazildikca
    ekliyor (bkz. modul docstring'i, 2026-07-13 karari) -- batch_size'i
    KUCUK tutup 3 uye ile 2 flush'a zorlayarak callback'in dogru sayida
    ve dogru URI'lerle cagirildigini dogruluyoruz."""
    tar_bytes = _make_tar_bytes([SAMPLE_AIRCRAFT, SAMPLE_AIRCRAFT, SAMPLE_AIRCRAFT])
    written = []

    _parse_tar_fileobj(
        io.BytesIO(tar_bytes), "test.tar", batch_size=1, client=FakeMinioClient(),
        on_part_written=written.append,
    )

    assert len(written) == 3
    assert all(uri.startswith("s3://silver/") for uri in written)


def test_parse_tar_fileobj_skips_broken_member_but_keeps_going():
    """Bozuk/parse edilemeyen TEK bir trace uyesi (orn. bozuk gzip) tum
    tar'i BATIRMAMALI -- digerleri normal islenmeye devam etmeli, sadece
    hata sayaci artmali (bkz. fonksiyon govdesi, 'errors' sayaci)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        broken = b"\x1f\x8b" + b"not-actually-gzip-data"  # gzip magic bytes ama gecersiz icerik
        info = tarfile.TarInfo(name="traces/ab/broken.json")
        info.size = len(broken)
        tf.addfile(info, io.BytesIO(broken))

        good_payload = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
        info2 = tarfile.TarInfo(name="traces/ab/good.json")
        info2.size = len(good_payload)
        tf.addfile(info2, io.BytesIO(good_payload))

    uris = _parse_tar_fileobj(
        io.BytesIO(buf.getvalue()), "mixed.tar", batch_size=100, client=FakeMinioClient()
    )

    assert len(uris) == 1  # "good" uyesinden gelen tek Silver parcasi


# ======================================================================
# Checkpoint/resume sistemi (2026-07-13, kullanici istegi -- "dur deyince
# dursam ertesi gun kaldigi yerden devam edemez miyiz"). Bu bolum eklenene
# kadar HICBIR testi yoktu -- oysa bu, PC'nin gun boyu acik kalamamasi
# gercek sorununu cozen, projenin en yeni ve en kritik parcalarindan biri.
# ======================================================================

def _put_tar(client: FakeMinioClient, bucket: str, name: str, aircraft: list[dict]) -> None:
    client.make_bucket(bucket)
    payload = _make_tar_bytes(aircraft)
    client.put_object(bucket, name, io.BytesIO(payload), length=len(payload))


def test_load_checkpoint_missing_file_returns_empty_state(tmp_path):
    state = _load_checkpoint(tmp_path / "does_not_exist.json")
    assert state == {"completed_tars": [], "in_progress": {}}


def test_save_and_load_checkpoint_round_trips(tmp_path):
    path = tmp_path / "state" / "checkpoint.json"
    state = {"completed_tars": ["a.tar"], "in_progress": {"b.tar": ["s3://silver/x"]}}

    _save_checkpoint(path, state)

    assert _load_checkpoint(path) == state


def test_load_checkpoint_corrupt_json_falls_back_to_empty_state(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert _load_checkpoint(path) == {"completed_tars": [], "in_progress": {}}


def test_load_checkpoint_fills_missing_keys_with_defaults(tmp_path):
    """Eski/elle duzenlenmis bir checkpoint dosyasinda anahtarlardan biri
    eksik olsa bile crash etmemeli -- setdefault ile tamamlanir."""
    path = tmp_path / "checkpoint.json"
    path.write_text('{"completed_tars": ["a.tar"]}', encoding="utf-8")

    state = _load_checkpoint(path)

    assert state == {"completed_tars": ["a.tar"], "in_progress": {}}


def test_delete_uris_removes_each_object(fake_minio_client: FakeMinioClient):
    write_silver(pd.DataFrame({"x": [1]}), "adsblol_historical", client=fake_minio_client)
    uris = list_layer_objects(fake_minio_client, "silver", "adsblol_historical")
    full_uris = [f"s3://silver/{name}" for name in uris]

    _delete_uris(fake_minio_client, full_uris)

    assert list_layer_objects(fake_minio_client, "silver", "adsblol_historical") == []


def test_delete_uris_continues_after_one_failure(fake_minio_client: FakeMinioClient):
    """Bir URI'nin silinmesi basarisiz olsa (orn. zaten yok), digerleri
    YINE DE silinmeye devam etmeli -- bkz. fonksiyon docstring'i."""
    write_silver(pd.DataFrame({"x": [1]}), "adsblol_historical", client=fake_minio_client)
    real_uri = f"s3://silver/{list_layer_objects(fake_minio_client, 'silver', 'adsblol_historical')[0]}"

    _delete_uris(fake_minio_client, ["s3://silver/does-not-exist.parquet", real_uri])

    assert list_layer_objects(fake_minio_client, "silver", "adsblol_historical") == []


def test_run_returns_empty_list_when_no_tar_objects(fake_minio_client: FakeMinioClient, tmp_path):
    fake_minio_client.make_bucket("bronze")
    assert run(client=fake_minio_client, bronze_bucket="bronze",
               checkpoint_path=tmp_path / "cp.json") == []


def test_run_marks_tar_completed_and_skips_it_on_rerun(fake_minio_client: FakeMinioClient, tmp_path):
    _put_tar(fake_minio_client, "bronze", "adsblol_historical/one.tar", [SAMPLE_AIRCRAFT])
    checkpoint_path = tmp_path / "cp.json"

    first_uris = run(client=fake_minio_client, bronze_bucket="bronze", checkpoint_path=checkpoint_path)
    assert len(first_uris) == 1
    assert _load_checkpoint(checkpoint_path)["completed_tars"] == ["one.tar"]

    # Ayni Bronze tar hala orada (run() sadece Silver'i yazar, Bronze
    # tar'lari silmez -- gecmis veri tekrar tekrar islenebilir kaynak
    # olarak kalir) -- ama checkpoint'te "tamamlandi" oldugu icin tekrar
    # calisinca YENI bir Silver parcasi eklenmemeli.
    second_uris = run(client=fake_minio_client, bronze_bucket="bronze", checkpoint_path=checkpoint_path)
    assert second_uris == []
    assert len(list_layer_objects(fake_minio_client, "silver", "adsblol_historical")) == 1


def test_run_fresh_flag_reprocesses_and_clears_prior_silver(fake_minio_client: FakeMinioClient, tmp_path):
    _put_tar(fake_minio_client, "bronze", "adsblol_historical/one.tar", [SAMPLE_AIRCRAFT])
    checkpoint_path = tmp_path / "cp.json"
    run(client=fake_minio_client, bronze_bucket="bronze", checkpoint_path=checkpoint_path)
    assert len(list_layer_objects(fake_minio_client, "silver", "adsblol_historical")) == 1

    run(client=fake_minio_client, bronze_bucket="bronze", checkpoint_path=checkpoint_path, fresh=True)

    # --fresh: checkpoint sifirlanir, tar YENIDEN islenir -- eski Silver
    # parcasi once temizlendigi icin toplam hala 1 (2 degil, kopya yok).
    assert len(list_layer_objects(fake_minio_client, "silver", "adsblol_historical")) == 1
    assert _load_checkpoint(checkpoint_path)["completed_tars"] == ["one.tar"]


def test_run_resumes_after_interrupted_tar_by_discarding_partial_parts(
    fake_minio_client: FakeMinioClient, tmp_path
):
    """Kesinti senaryosu: bir onceki calistirma bir tar'i islerken (kill/
    Ctrl+C/PC kapanmasi ile) YARIM kaldi -- checkpoint'te o tar hala
    'in_progress' ve KISMEN yazilmis bir Silver parcasi var. run() tekrar
    cagrildiginda: (1) o yarim parcayi silmeli, (2) tar'i SIFIRDAN
    islemeli -- 30 tar'in TAMAMINI degil, SADECE kesinti anindaki tek
    tar'i kaybetmeli (bkz. modul docstring'i, 2026-07-13 karari)."""
    _put_tar(fake_minio_client, "bronze", "adsblol_historical/interrupted.tar", [SAMPLE_AIRCRAFT])
    checkpoint_path = tmp_path / "cp.json"

    # Kesinti oncesi durumu simule et: tar'in KISMEN yazilmis (yanlis/eski
    # icerikli, "yarim kalmis" oldugunu ayirt etmek icin farkli boyutlu)
    # bir Silver parcasi var, checkpoint onu "in_progress" olarak biliyor.
    partial_uri = write_silver(
        pd.DataFrame({"stale": [1, 2, 3]}), "adsblol_historical", client=fake_minio_client
    )
    _save_checkpoint(checkpoint_path, {
        "completed_tars": [],
        "in_progress": {"interrupted.tar": [partial_uri]},
    })

    uris = run(client=fake_minio_client, bronze_bucket="bronze", checkpoint_path=checkpoint_path)

    assert len(uris) == 1
    # Yarim kalan eski parca gitti, yerine tar'in SIFIRDAN islenmis
    # (gercek SAMPLE_AIRCRAFT verisini iceren) TEK bir parcasi var.
    remaining = list_layer_objects(fake_minio_client, "silver", "adsblol_historical")
    assert len(remaining) == 1
    stored = fake_minio_client.buckets["silver"][remaining[0]]
    df = pd.read_parquet(io.BytesIO(stored))
    assert "stale" not in df.columns
    assert _load_checkpoint(checkpoint_path)["completed_tars"] == ["interrupted.tar"]
    assert _load_checkpoint(checkpoint_path)["in_progress"] == {}
