import gzip
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.adsblol_historical_loader import extract_turkey, merge_tar_parts


def _make_aircraft_member(icao: str, base_ts: int, trace: list[list]) -> bytes:
    payload = {
        "icao": icao,
        "r": "TC-TEST",
        "t": "B738",
        "timestamp": base_ts,
        "trace": trace,
    }
    return gzip.compress(json.dumps(payload).encode("utf-8"))


def _write_fake_tar(tar_path: Path) -> None:
    base_ts = 1750000000
    # One aircraft entirely inside the Turkey bbox (lat 36-42, lon 26-45).
    inside = _make_aircraft_member(
        "tc0001",
        base_ts,
        trace=[
            [0, 39.0, 35.0, 35000, 450.0, 90.0, 0, 0, None, "adsb_icao", 35100, 0, 420, 1.0],
            [10, 39.1, 35.1, 35000, 450.0, 91.0, 2, 0, {"flight": "THY123  "}, "adsb_icao", 35100, 0, 420, 1.0],
        ],
    )
    # One aircraft entirely outside (e.g. over the Atlantic).
    outside = _make_aircraft_member(
        "ab0002",
        base_ts,
        trace=[[0, 51.0, -10.0, 30000, 400.0, 270.0, 0, 0, None]],
    )
    # One aircraft with a "ground" altitude and a None lat (should be dropped safely).
    edge_case = _make_aircraft_member(
        "tc0003",
        base_ts,
        trace=[[0, None, None, "ground", None, 180.0, 0, None, None]],
    )

    with tarfile.open(tar_path, "w") as tar:
        for icao, blob in [("tc0001", inside), ("ab0002", outside), ("tc0003", edge_case)]:
            data = io.BytesIO(blob)
            info = tarfile.TarInfo(name=f"traces/{icao[:2]}/trace_full_{icao}.json")
            info.size = len(blob)
            tar.addfile(info, data)


def test_extract_turkey_keeps_only_in_bbox_points(tmp_path, fake_minio_client):
    tar_path = tmp_path / "v2026.06.15-planes-readsb-prod-0.tar"
    _write_fake_tar(tar_path)
    written = extract_turkey(tar_path, client=fake_minio_client, flush_every=100)

    assert len(written) == 1
    bucket, object_name = written[0].removeprefix("s3://").split("/", 1)
    df = pd.read_parquet(io.BytesIO(fake_minio_client.buckets[bucket][object_name]))

    # Only the two in-bbox points from tc0001 should survive.
    assert set(df["icao"]) == {"tc0001"}
    assert len(df) == 2
    assert df["timestamp_epoch_s"].tolist() == [1750000000, 1750000010]
    assert df.loc[df["trace_seconds_after_timestamp"] == 10, "aircraft_dict"].iloc[0] == json.dumps(
        {"flight": "THY123  "}
    )

    for column in ("_source_type", "_ingest_ts_utc", "_source_file", "_schema_version"):
        assert column in df.columns
    assert (df["_source_type"] == "adsblol_hist").all()


def test_extract_turkey_returns_empty_when_nothing_in_bbox(tmp_path, fake_minio_client):
    tar_path = tmp_path / "v2026.06.16-planes-readsb-prod-0.tar"
    with tarfile.open(tar_path, "w") as tar:
        blob = _make_aircraft_member("ab0002", 1750000000, trace=[[0, 51.0, -10.0, 30000, 400.0, 270.0, 0, 0, None]])
        data = io.BytesIO(blob)
        info = tarfile.TarInfo(name="traces/ab/trace_full_ab0002.json")
        info.size = len(blob)
        tar.addfile(info, data)

    written = extract_turkey(tar_path, client=fake_minio_client)
    assert written == []


def test_merge_tar_parts_concatenates_in_order(tmp_path):
    base = tmp_path / "release"
    (tmp_path / "release.tar.aa").write_bytes(b"AAAA")
    (tmp_path / "release.tar.ab").write_bytes(b"BBBB")

    merged = merge_tar_parts(base)

    assert merged == tmp_path / "release.tar"
    assert merged.read_bytes() == b"AAAABBBB"


def test_merge_tar_parts_raises_when_no_parts_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_tar_parts(tmp_path / "missing")
