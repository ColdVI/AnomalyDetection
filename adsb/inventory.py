"""Faz 0.1: bir adsb.lol/readsb tar arsivini TAMAMEN parse etmeden hizli profil cikarir.

Tam Silver parse'i (adsb/README.md 0.2) her trace dosyasini okur -- saatler surebilir.
Bu modul yalniz bir ORNEKLEM okuyup (varsayilan 500 ucak/tar) alan-gorulme sikligi,
trace-satiri uzunlugu, ornekleme araligi ve category dagilimini raporlar. Format
referansindaki (docs/adsblo_data_format_reference (1) 2026-07-10 amt 11.03.27.md, SS5.1)
2026-06-15 orneklemine benzer ama BU tar'lara ozel -- o tarih farkli, guncel olmayabilir.
"""

from __future__ import annotations

import gzip
import json
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_TRACE_LEN = 14

# format referansi SS2 + SS5 -- dosya-seviyesi ve aircraft_dict alanlari
FILE_LEVEL_FIELDS = ["icao", "r", "t", "desc", "ownOp", "year", "dbFlags", "noRegData", "timestamp", "trace"]
AC_DICT_FIELDS = [
    "flight", "category", "squawk", "emergency", "nic", "rc", "nac_p", "nac_v", "sil",
    "sil_type", "alert", "spi", "version", "type", "gva", "sda", "baro_rate", "geom_rate",
    "alt_geom", "nic_baro", "track", "nav_qnh", "nav_altitude_mcp", "nav_altitude_fms",
    "nav_heading", "mag_heading", "true_heading", "mach", "tas", "ias", "wd", "ws",
    "oat", "tat", "track_rate", "roll",
]


@dataclass
class TarProfile:
    tar_name: str
    total_trace_members: int
    sampled_members: int
    sampled_rows: int
    parse_errors: int
    file_field_presence: dict[str, int] = field(default_factory=dict)
    ac_dict_field_presence: dict[str, int] = field(default_factory=dict)
    trace_row_lengths: dict[int, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    sampling_interval_s: dict[int, int] = field(default_factory=dict)
    on_ground_rows: int = 0

    def as_dict(self) -> dict:
        return {
            "tar_name": self.tar_name,
            "total_trace_members": self.total_trace_members,
            "sampled_members": self.sampled_members,
            "sampled_rows": self.sampled_rows,
            "parse_errors": self.parse_errors,
            "on_ground_rows": self.on_ground_rows,
            "file_field_presence": self.file_field_presence,
            "ac_dict_field_presence": self.ac_dict_field_presence,
            "trace_row_lengths": self.trace_row_lengths,
            "category_counts": self.category_counts,
            "sampling_interval_s": self.sampling_interval_s,
        }


def list_trace_members(tar_path: str | Path) -> list[str]:
    """Yalniz uye ADLARINI listeler (header okur, icerik ACMAZ -- ucuz)."""
    with tarfile.open(tar_path, mode="r:*") as tar:
        return [
            m.name for m in tar.getmembers()
            if "traces" in m.name and (m.name.endswith(".json") or m.name.endswith(".json.gz"))
        ]


def _evenly_spaced_indices(n_total: int, n_samples: int) -> list[int]:
    """n_total uyeden n_samples'i esit araliklarla secer (yalniz basa yigilma onlenir)."""
    if n_samples >= n_total:
        return list(range(n_total))
    step = n_total / n_samples
    return sorted({int(i * step) for i in range(n_samples)})


def _load_trace_json(raw: bytes) -> dict:
    try:
        return json.loads(gzip.decompress(raw))
    except OSError:
        return json.loads(raw)


def profile_tar(tar_path: str | Path, *, n_samples: int = 500) -> TarProfile:
    tar_path = Path(tar_path)
    all_members = list_trace_members(tar_path)
    idx = _evenly_spaced_indices(len(all_members), n_samples)

    file_field_counter: Counter = Counter()
    ac_field_counter: Counter = Counter()
    row_len_counter: Counter = Counter()
    category_counter: Counter = Counter()
    interval_counter: Counter = Counter()
    sampled_rows = 0
    on_ground_rows = 0
    errors = 0

    with tarfile.open(tar_path, mode="r:*") as tar:
        for i in idx:
            name = all_members[i]
            try:
                f = tar.extractfile(name)
                if f is None:
                    continue
                data = _load_trace_json(f.read())
            except Exception:
                errors += 1
                continue

            for fld in FILE_LEVEL_FIELDS:
                if fld in data:
                    file_field_counter[fld] += 1

            trace = data.get("trace", [])
            last_t = None
            seen_ac_fields: set[str] = set()
            for row in trace:
                sampled_rows += 1
                row_len_counter[len(row)] += 1

                t_offset = row[0] if len(row) > 0 else None
                if t_offset is not None and last_t is not None:
                    interval_counter[round(t_offset - last_t)] += 1
                last_t = t_offset

                alt_raw = row[3] if len(row) > 3 else None
                if alt_raw == "ground":
                    on_ground_rows += 1

                ac_dict = row[8] if len(row) > 8 else None
                if isinstance(ac_dict, dict):
                    for fld in AC_DICT_FIELDS:
                        if fld in ac_dict and fld not in seen_ac_fields:
                            ac_field_counter[fld] += 1
                            seen_ac_fields.add(fld)
                    if "category" in ac_dict and ac_dict["category"]:
                        category_counter[str(ac_dict["category"])] += 1

    return TarProfile(
        tar_name=tar_path.name,
        total_trace_members=len(all_members),
        sampled_members=len(idx),
        sampled_rows=sampled_rows,
        parse_errors=errors,
        file_field_presence=dict(file_field_counter),
        ac_dict_field_presence=dict(ac_field_counter),
        trace_row_lengths=dict(row_len_counter),
        category_counts=dict(category_counter),
        sampling_interval_s=dict(interval_counter),
        on_ground_rows=on_ground_rows,
    )


def profile_all(tar_paths: list[str | Path], *, n_samples: int = 500) -> list[TarProfile]:
    return [profile_tar(p, n_samples=n_samples) for p in tar_paths]
