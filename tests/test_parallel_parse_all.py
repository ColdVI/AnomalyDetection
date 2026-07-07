"""Tests for scripts/parallel_parse_all.py's process-pool orchestration.

Runs the real `--local-tar` subprocess (so the orchestration logic itself is
exercised end-to-end) but with STORAGE_BACKEND=local pointed at a temp dir, so
no real MinIO server is required and nothing touches the real bronze/silver
buckets.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.parallel_parse_all import parse_one_tar  # noqa: E402


def _make_trace_tar(path: Path, icao: str, n_points: int = 3) -> None:
    trace = [[float(i), 40.0 + i * 0.01, 29.0 + i * 0.01, 1000.0 + i, 100.0, 90.0,
              None, None, None, "adsb_icao", None, None, None, None] for i in range(n_points)]
    payload = json.dumps({"icao": icao, "timestamp": 1700000000, "trace": trace}).encode()
    gz = gzip.compress(payload)

    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(name=f"./traces/{icao[:2]}/trace_full_{icao}.json.gz")
        info.size = len(gz)
        tar.addfile(info, io.BytesIO(gz))


def test_parse_one_tar_succeeds_and_writes_silver(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objectstore"))

    tar_path = tmp_path / "fake_tar_1.tar"
    _make_trace_tar(tar_path, icao="abc123")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    tar_path_str, ok, log_path = parse_one_tar(str(tar_path), log_dir)

    assert ok, Path(log_path).read_text(encoding="utf-8", errors="replace")
    silver_dir = tmp_path / "objectstore" / "silver" / "adsblol_historical"
    assert silver_dir.is_dir()
    assert list(silver_dir.glob("*.parquet"))


def test_parse_one_tar_reports_failure_without_crashing_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objectstore"))

    bad_tar = tmp_path / "not_actually_a_tar.tar"
    bad_tar.write_bytes(b"this is not a valid tar file")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    tar_path_str, ok, log_path = parse_one_tar(str(bad_tar), log_dir)

    assert not ok
    assert Path(log_path).exists()


def test_two_tars_process_independently(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objectstore"))

    tar1 = tmp_path / "t1.tar"
    tar2 = tmp_path / "t2.tar"
    _make_trace_tar(tar1, icao="aaa111")
    _make_trace_tar(tar2, icao="bbb222")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    r1 = parse_one_tar(str(tar1), log_dir)
    r2 = parse_one_tar(str(tar2), log_dir)

    assert r1[1] and r2[1]
    silver_dir = tmp_path / "objectstore" / "silver" / "adsblol_historical"
    # both tars' output landed in Silver, neither run cleared the other's parts
    assert len(list(silver_dir.glob("*.parquet"))) == 2
