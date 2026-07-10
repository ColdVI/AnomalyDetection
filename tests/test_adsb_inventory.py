"""Faz 0.1 envanter testleri -- kucuk sentetik bir tar uzerinde (gercek 3GB dosyalara bagimli degil)."""

from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

from adsb.inventory import list_trace_members, profile_tar


def _make_trace_json(icao: str, *, n_rows: int = 5, category: str = "A3") -> bytes:
    trace = []
    for i in range(n_rows):
        trace.append([
            i * 10,       # t_offset
            40.0 + i * 0.01,  # lat
            29.0 + i * 0.01,  # lon
            1000.0 + i * 50,  # alt
            100.0,            # ground_speed
            45.0,              # track
            0,                 # flags
            5.0,               # vertical_rate
            {"flight": "TEST123", "category": category, "squawk": "1200"} if i == 0 else None,
            "adsb_icao",
            1050.0,
            5.0,
            110.0,
            2.0,
        ])
    data = {"icao": icao, "r": "N12345", "timestamp": 1_700_000_000, "trace": trace}
    return gzip.compress(json.dumps(data).encode())


def _write_synthetic_tar(path: Path, *, n_aircraft: int = 10) -> Path:
    with tarfile.open(path, "w") as tar:
        for k in range(n_aircraft):
            icao = f"a{k:05x}"
            content = _make_trace_json(icao, category="B6" if k == 0 else "A3")
            member = tarfile.TarInfo(name=f"traces/{icao[:2]}/trace_full_{icao}.json.gz")
            member.size = len(content)
            import io
            tar.addfile(member, io.BytesIO(content))
    return path


def test_list_trace_members_finds_all(tmp_path):
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=7)
    members = list_trace_members(tar_path)
    assert len(members) == 7
    assert all("traces/" in m for m in members)


def test_profile_tar_basic_stats(tmp_path):
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=10)
    profile = profile_tar(tar_path, n_samples=10)

    assert profile.total_trace_members == 10
    assert profile.sampled_members == 10
    assert profile.sampled_rows == 50  # 10 aircraft * 5 rows
    assert profile.parse_errors == 0
    assert profile.trace_row_lengths == {14: 50}
    assert profile.file_field_presence["icao"] == 10
    assert profile.file_field_presence["timestamp"] == 10
    assert profile.ac_dict_field_presence["flight"] == 10
    assert profile.ac_dict_field_presence["category"] == 10
    assert profile.category_counts == {"B6": 1, "A3": 9}
    # her ucus icin ardisik t_offset farki = 10 (4 aralik * 10 ucak = 40 kayit)
    assert profile.sampling_interval_s == {10: 40}


def test_profile_tar_respects_sample_size(tmp_path):
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=20)
    profile = profile_tar(tar_path, n_samples=5)
    assert profile.total_trace_members == 20
    assert profile.sampled_members == 5
    assert profile.sampled_rows == 25


def test_profile_tar_as_dict_is_json_serializable(tmp_path):
    import json as _json
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=3)
    profile = profile_tar(tar_path, n_samples=3)
    serialized = _json.dumps(profile.as_dict())
    assert "sample.tar" in serialized
