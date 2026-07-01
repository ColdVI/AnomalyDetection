"""Bronze loader for adsb.lol historical `globe_history` tar releases.

Phase 2 of docs/bronze_implementasyon_plani.md.

Each daily release is a tar archive (optionally split into .tar.aa/.tar.ab
parts) containing one gzip-compressed JSON file per aircraft under
`traces/<hex-prefix>/trace_full_<icao>.json.gz`. Each file has a base
`timestamp` (unix epoch seconds) and a `trace` array of fixed-position rows:

    [seconds_after_timestamp, lat, lon, altitude, ground_speed, track,
     flags, vertical_rate, aircraft_dict, source_type, geom_altitude,
     geom_vertical_rate, indicated_airspeed, roll_angle]

(indices 9-13 only present in 2022+ releases; see
docs/.../adsblo_data_format_reference for the full empirically-validated
field list).

Bronze rule: no unit conversion, no renaming of original values. We only
(a) resolve each point's absolute epoch time from the base timestamp + the
per-point offset -- without this the row is unusable, this is not a unit
conversion, just offset resolution -- and (b) keep only points inside the
Turkey bbox. Everything else is passed through as-is. The nested
`aircraft_dict` (sparse per-point extra fields) is kept as a JSON string
column rather than being flattened/decoded, since flattening + harmonizing
is Silver's job.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from src.common.bbox import in_turkey
from src.common.io import ObjectStoreClient, write_bronze
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "adsblol_hist"
_DATE_PATTERN = re.compile(r"v(\d{4}\.\d{2}\.\d{2})")
_TRACE_COLUMNS = (
    "trace_seconds_after_timestamp",
    "lat",
    "lon",
    "alt_baro",
    "gs",
    "track_or_heading",
    "flags",
    "vert_rate",
    "aircraft_dict",
    "point_source_type",
    "alt_geom",
    "geom_vert_rate",
    "indicated_airspeed",
    "roll_angle",
)
_FILE_LEVEL_OPTIONAL_FIELDS = ("r", "t", "desc", "ownOp", "year", "dbFlags", "noRegData")


def merge_tar_parts(base_path: str | Path, parts: tuple[str, ...] = ("aa", "ab")) -> Path:
    """Concatenate `<base_path>.tar.aa` + `.tar.ab` (etc.) into `<base_path>.tar`.

    No-op (returns existing path) if the merged file or a plain .tar already
    exists. Streams via shutil-style chunked copy so multi-GB archives don't
    need to fit in memory twice.
    """
    base_path = Path(base_path)
    merged_path = base_path.with_suffix(".tar") if base_path.suffix != ".tar" else base_path
    if merged_path.exists():
        return merged_path

    part_paths = [Path(f"{base_path}.tar.{p}") for p in parts if Path(f"{base_path}.tar.{p}").exists()]
    if not part_paths:
        raise FileNotFoundError(f"No split parts found for {base_path}.tar.*")

    with merged_path.open("wb") as out:
        for part_path in part_paths:
            logger.info("Merging part: %s", part_path)
            with part_path.open("rb") as part_file:
                while chunk := part_file.read(1024 * 1024 * 16):
                    out.write(chunk)
    return merged_path


def _infer_tar_date(tar_path: Path) -> str:
    match = _DATE_PATTERN.search(tar_path.name)
    return match.group(1) if match else "unknown_date"


def _read_member_json(tar: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any] | None:
    extracted: BinaryIO | None = tar.extractfile(member)
    if extracted is None:
        return None
    raw = extracted.read()
    try:
        return json.loads(gzip.decompress(raw))
    except OSError:
        # Some releases ship members that are .json-named but not gzipped.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse member as JSON or gzip+JSON: %s", member.name)
            return None


def _trace_rows_in_turkey(aircraft: dict[str, Any], member_name: str) -> list[dict[str, Any]]:
    icao = aircraft.get("icao")
    base_timestamp = aircraft.get("timestamp")
    trace = aircraft.get("trace") or []
    if icao is None or base_timestamp is None:
        logger.warning("Skipping member missing icao/timestamp: %s", member_name)
        return []

    file_level = {field: aircraft.get(field) for field in _FILE_LEVEL_OPTIONAL_FIELDS if field in aircraft}

    rows: list[dict[str, Any]] = []
    for point in trace:
        if len(point) < 8:
            continue  # malformed / pre-2022 truncated row, not enough fields to use
        lat, lon = point[1], point[2]
        if not in_turkey(lat, lon):
            continue

        padded = list(point) + [None] * (len(_TRACE_COLUMNS) - len(point))
        row: dict[str, Any] = {"icao": icao, "file_timestamp": base_timestamp}
        row.update(file_level)
        row.update(dict(zip(_TRACE_COLUMNS, padded[: len(_TRACE_COLUMNS)])))
        # Resolve absolute epoch -- offset resolution, not a unit conversion.
        offset = row["trace_seconds_after_timestamp"]
        row["timestamp_epoch_s"] = (
            base_timestamp + offset if isinstance(offset, (int, float)) else None
        )
        if isinstance(row.get("aircraft_dict"), dict):
            row["aircraft_dict"] = json.dumps(row["aircraft_dict"])
        rows.append(row)
    return rows


def extract_turkey(
    tar_path: str | Path,
    *,
    client: ObjectStoreClient | None = None,
    flush_every: int = 2000,
) -> list[str]:
    """Stream a (merged) historical tar, keep Turkey-bbox trace points, write Bronze.

    Processes the archive lazily (tarfile member-by-member) so a 3GB+ daily
    release doesn't need to be fully extracted to disk first. Returns the
    list of `s3://bronze/...` URIs written to MinIO (may be empty if nothing
    in the archive touched the Turkey bbox).
    """
    tar_path = Path(tar_path)
    tar_date = _infer_tar_date(tar_path)
    written: list[str] = []
    buffer: list[dict[str, Any]] = []
    aircraft_seen = 0

    def _flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        df = add_provenance(
            df,
            source_type=SOURCE_TYPE,
            source_file=f"{tar_path.name}",
            schema_version="bronze_v1",
        )
        written.append(write_bronze(df, "adsblol_historical", partition=tar_date, client=client))
        buffer = []

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if "traces" not in member.name or not member.name.endswith(".json"):
                continue
            aircraft = _read_member_json(tar, member)
            if aircraft is None:
                continue
            buffer.extend(_trace_rows_in_turkey(aircraft, member.name))
            aircraft_seen += 1
            if aircraft_seen % flush_every == 0:
                logger.info("Processed %d aircraft from %s, %d Turkey rows buffered", aircraft_seen, tar_path.name, len(buffer))
                _flush()

    _flush()
    logger.info("Done: %s -> %d aircraft scanned, %d Bronze parts written", tar_path.name, aircraft_seen, len(written))
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="adsb.lol historical -> Bronze (Turkey bbox)")
    parser.add_argument(
        "--input",
        default="data/bronze/adsblol_historical/_input",
        help="Directory containing .tar / .tar.aa+.tar.ab releases",
    )
    args = parser.parse_args()
    input_dir = Path(args.input)

    candidates = sorted(input_dir.glob("*.tar"))
    seen_bases = {p.stem for p in candidates}
    for part in sorted(input_dir.glob("*.tar.aa")):
        base = part.name[: -len(".tar.aa")]
        if base not in seen_bases:
            candidates.append(merge_tar_parts(input_dir / base))
            seen_bases.add(base)

    if not candidates:
        logger.warning("No .tar or .tar.aa+.tar.ab files found in %s", input_dir)
        return

    for tar_path in candidates:
        extract_turkey(tar_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
