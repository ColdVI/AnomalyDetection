"""Tests for src/silver/parse_generic.py -- new format support."""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.common.fakes import FakeMinioClient
from src.silver.parse_generic import (
    _read_bytes_to_df,
    _read_gz_bytes,
    _read_sqlite_bytes,
    parse_bytes,
)


# ---------------------------------------------------------------------------
# 4a: .gz single-file support
# ---------------------------------------------------------------------------

def test_gz_wrapping_csv():
    """A .csv.gz should decompress to a CSV and parse correctly."""
    inner = b"a,b\n1,2\n3,4\n"
    compressed = gzip.compress(inner)
    df = _read_gz_bytes(compressed, "data.csv.gz")
    assert df is not None
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_gz_wrapping_jsonl():
    """A .jsonl.gz should decompress to JSONL and parse correctly."""
    inner = b'{"x": 1}\n{"x": 2}\n'
    compressed = gzip.compress(inner)
    df = _read_gz_bytes(compressed, "data.jsonl.gz")
    assert df is not None
    assert list(df["x"]) == [1, 2]


def test_gz_bad_bytes_returns_none():
    """Corrupt bytes that are not valid gzip should return None without raising."""
    df = _read_gz_bytes(b"not gzip data", "bad.csv.gz")
    assert df is None


def test_parse_bytes_dispatches_gz():
    """`parse_bytes` should handle .gz files and attach provenance."""
    inner = b"col1,col2\n10,20\n"
    data = gzip.compress(inner)
    frames = parse_bytes(data, "telemetry.csv.gz", "mydata")
    assert len(frames) == 1
    assert "col1" in frames[0].columns
    assert "_source_type" in frames[0].columns


# ---------------------------------------------------------------------------
# 4d: SQLite .db / .sqlite support
# ---------------------------------------------------------------------------

def _make_sqlite_bytes(tables: dict[str, list[dict]]) -> bytes:
    """Build an in-memory SQLite database and return its bytes."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    con = sqlite3.connect(tmp_path)
    for table_name, rows in tables.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_sql(table_name, con, index=False)
    con.close()
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


def test_sqlite_single_table():
    data = _make_sqlite_bytes({"telemetry": [{"ts": 1, "lat": 10.0}, {"ts": 2, "lat": 11.0}]})
    df = _read_sqlite_bytes(data)
    assert df is not None
    assert "lat" in df.columns
    assert len(df) == 2


def test_sqlite_multiple_tables_concatenated():
    data = _make_sqlite_bytes({
        "flight_a": [{"id": 1, "v": 10}],
        "flight_b": [{"id": 2, "v": 20}],
    })
    df = _read_sqlite_bytes(data)
    assert df is not None
    assert len(df) == 2


def test_sqlite_empty_db_returns_none():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    sqlite3.connect(tmp_path).close()
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    df = _read_sqlite_bytes(data)
    assert df is None


def test_parse_bytes_dispatches_sqlite():
    """`parse_bytes` should handle .db files and attach provenance."""
    data = _make_sqlite_bytes({"logs": [{"t": 0, "alt": 100}, {"t": 1, "alt": 110}]})
    frames = parse_bytes(data, "flight.db", "mydata")
    assert len(frames) == 1
    assert "_source_file" in frames[0].columns
    assert frames[0]["_source_file"].iloc[0] == "flight.db"


def test_parse_bytes_gz_inside_zip(tmp_path):
    """A .gz file inside a zip archive should be parsed correctly."""
    inner_csv = b"x,y\n1,2\n"
    gz_data = gzip.compress(inner_csv)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("data.csv.gz", gz_data)
    frames = parse_bytes(zip_buf.getvalue(), "bundle.zip", "mydata")
    assert len(frames) == 1
    assert "x" in frames[0].columns


# ---------------------------------------------------------------------------
# 4b: MAVLink (.bin/.tlog/.log) -- skip gracefully when pymavlink absent
# ---------------------------------------------------------------------------

def test_mavlink_skips_gracefully_without_pymavlink(monkeypatch):
    """When pymavlink is not installed, .bin files should return None (not raise)."""
    import sys
    monkeypatch.setitem(sys.modules, "pymavlink", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "pymavlink.mavutil", None)  # type: ignore[arg-type]
    df = _read_bytes_to_df(b"\x00" * 16, "log.bin")
    assert df is None


# ---------------------------------------------------------------------------
# 4c: MATLAB (.mat) -- skip gracefully when scipy absent
# ---------------------------------------------------------------------------

def test_mat_skips_gracefully_without_scipy(monkeypatch):
    """When scipy is not installed, .mat files should return None (not raise)."""
    import sys
    monkeypatch.setitem(sys.modules, "scipy", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "scipy.io", None)  # type: ignore[arg-type]
    df = _read_bytes_to_df(b"\x00" * 16, "data.mat")
    assert df is None
