"""parse_generic.py -- Generic Silver parser for any new dataset.

Automatically detects file format (CSV, JSON, JSONL, gz, zip, tar, Parquet,
Excel, SQLite, MATLAB .mat, MAVLink .bin/.tlog/.log), reads into a DataFrame,
attaches provenance, and writes Silver Parquet to MinIO.
Does NOT do unit conversions or label extraction -- those are domain-specific
and belong in a custom parser built on top of this one.

Optional dependencies (install separately if needed):
  - openpyxl   : .xlsx support  (pip install openpyxl)
  - pymavlink  : .bin/.tlog/.log MAVLink support  (pip install pymavlink)
  - scipy      : .mat MATLAB support  (pip install scipy)

Usage (MinIO Bronze prefix):
    python -m src.silver.parse_generic --bronze-prefix surveildrone/ --source surveildrone

Usage (local file):
    python -m src.silver.parse_generic --local-file ~/Downloads/data.csv --source mydata

To add a new dataset end-to-end:
  1. Upload to Bronze:  python -m src.ingestion.upload_raw --source mydata --input ~/data.csv
  2. Parse to Silver:   python -m src.silver.parse_generic --bronze-prefix mydata/ --source mydata
  3. Add Gold mapping:  edit COLUMN_MAPS in src/gold/unify.py
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import sqlite3
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    delete_layer_objects,
    download_raw_bytes,
    get_minio_client,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet",
    ".xlsx", ".xls", ".gz", ".db", ".sqlite", ".bin", ".tlog", ".log", ".mat",
}


# ---------------------------------------------------------------------------
# Single-file readers
# ---------------------------------------------------------------------------

def _read_bytes_to_df(data: bytes, filename: str) -> pd.DataFrame | None:
    """Read raw bytes into a DataFrame based on the filename extension."""
    name = Path(filename).name.lower()
    buf = io.BytesIO(data)

    if name.endswith(".csv"):
        return pd.read_csv(buf)
    if name.endswith(".tsv"):
        return pd.read_csv(buf, sep="\t")
    if name.endswith((".jsonl", ".ndjson")):
        return pd.read_json(buf, lines=True)
    if name.endswith(".json"):
        try:
            obj = json.loads(data)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            return pd.json_normalize(obj)
        except Exception:
            buf.seek(0)
            return pd.read_json(buf, lines=True)
    if name.endswith(".parquet"):
        return pd.read_parquet(buf)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if name.endswith(".gz"):
        return _read_gz_bytes(data, filename)
    if name.endswith((".db", ".sqlite")):
        return _read_sqlite_bytes(data)
    if name.endswith((".bin", ".tlog", ".log")):
        return _read_mavlink_bytes(data, filename)
    if name.endswith(".mat"):
        return _read_mat_bytes(data)

    logger.warning("Unsupported file format: %s -- skipping", filename)
    return None


def _read_gz_bytes(data: bytes, filename: str) -> pd.DataFrame | None:
    """Decompress a .gz file and re-dispatch by the inner extension."""
    try:
        inner_data = gzip.decompress(data)
    except Exception as exc:
        logger.warning("Failed to decompress %s: %s", filename, exc)
        return None
    # strip the .gz suffix to get the real extension (e.g. data.csv.gz -> data.csv)
    inner_name = filename[:-3] if filename.lower().endswith(".gz") else filename
    return _read_bytes_to_df(inner_data, inner_name)


def _read_sqlite_bytes(data: bytes) -> pd.DataFrame | None:
    """Load all tables from a SQLite database and concatenate into one DataFrame."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        con = sqlite3.connect(tmp_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        if tables.empty:
            logger.warning("SQLite file contains no tables")
            con.close()
            return None
        frames = [pd.read_sql(f"SELECT * FROM [{t}]", con) for t in tables["name"]]
        con.close()
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        return df if not df.empty else None
    except Exception as exc:
        logger.warning("Failed to read SQLite bytes: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _read_mavlink_bytes(data: bytes, filename: str) -> pd.DataFrame | None:
    """Parse a MAVLink binary log (.bin/.tlog/.log) into a flat DataFrame.

    Requires pymavlink (pip install pymavlink). Returns None without it.
    Each message becomes one row; message type goes into the 'mavtype' column.
    """
    try:
        from pymavlink import mavutil  # type: ignore
    except ImportError:
        logger.warning(
            "pymavlink not installed -- skipping %s. Install with: pip install pymavlink",
            filename,
        )
        return None

    import tempfile

    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        mlog = mavutil.mavlink_connection(tmp_path, robust_parsing=True)
        rows: list[dict] = []
        while True:
            msg = mlog.recv_msg()
            if msg is None:
                break
            d = msg.to_dict()
            d["mavtype"] = msg.get_type()
            rows.append(d)
        if not rows:
            logger.warning("No MAVLink messages found in %s", filename)
            return None
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("Failed to parse MAVLink file %s: %s", filename, exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _read_mat_bytes(data: bytes) -> pd.DataFrame | None:
    """Load a MATLAB .mat file into a DataFrame (one variable per column).

    Requires scipy (pip install scipy). Returns None without it.
    Only numeric/string arrays are included; structs and cell arrays are skipped.
    """
    try:
        import scipy.io as sio  # type: ignore
    except ImportError:
        logger.warning(
            "scipy not installed -- skipping .mat file. Install with: pip install scipy"
        )
        return None

    try:
        mat = sio.loadmat(io.BytesIO(data), squeeze_me=True, simplify_cells=True)
    except Exception as exc:
        logger.warning("Failed to load .mat file: %s", exc)
        return None

    columns: dict[str, object] = {}
    for key, val in mat.items():
        if key.startswith("_"):  # scipy metadata keys
            continue
        try:
            import numpy as np  # already a dependency

            arr = np.atleast_1d(val)
            if arr.ndim == 1 and arr.dtype.kind in ("f", "i", "u", "S", "U"):
                columns[key] = arr
        except Exception:
            pass  # skip complex/struct values

    if not columns:
        logger.warning(".mat file contained no readable scalar arrays")
        return None

    try:
        return pd.DataFrame(columns)
    except Exception as exc:
        logger.warning("Could not build DataFrame from .mat variables: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Archive iterators
# ---------------------------------------------------------------------------

def _iter_zip(data: bytes, zip_name: str):
    """Yield (member_name, member_bytes) for every supported file inside a zip."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in _SUPPORTED_EXTS:
                continue
            with zf.open(info) as f:
                yield info.filename, f.read()


def _iter_tar(data: bytes, tar_name: str):
    """Yield (member_name, member_bytes) for every supported file inside a tar."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            ext = Path(member.name).suffix.lower()
            if ext not in _SUPPORTED_EXTS:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            yield member.name, f.read()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_bytes(data: bytes, filename: str, source_type: str) -> list[pd.DataFrame]:
    """Parse raw bytes (any supported format incl. archives) into a list of DataFrames."""
    name_lower = filename.lower()
    frames = []

    if name_lower.endswith(".zip"):
        for member_name, member_bytes in _iter_zip(data, filename):
            df = _read_bytes_to_df(member_bytes, member_name)
            if df is not None and not df.empty:
                df = add_provenance(
                    df, source_type=source_type,
                    source_file=f"{filename}/{member_name}",
                    schema_version="silver_v1",
                )
                frames.append(df)
    elif name_lower.endswith((".tar", ".tar.gz", ".tgz")):
        for member_name, member_bytes in _iter_tar(data, filename):
            df = _read_bytes_to_df(member_bytes, member_name)
            if df is not None and not df.empty:
                df = add_provenance(
                    df, source_type=source_type,
                    source_file=f"{filename}/{member_name}",
                    schema_version="silver_v1",
                )
                frames.append(df)
    else:
        df = _read_bytes_to_df(data, filename)
        if df is not None and not df.empty:
            df = add_provenance(
                df, source_type=source_type,
                source_file=filename,
                schema_version="silver_v1",
            )
            frames.append(df)

    return frames


def parse_local_file(
    path: str | Path,
    source_type: str,
    *,
    client: ObjectStoreClient | None = None,
) -> list[str]:
    """Parse a local file (any supported format) and write Silver to MinIO."""
    path = Path(path)
    data = path.read_bytes()
    frames = parse_bytes(data, path.name, source_type)
    if not frames:
        logger.warning("No data parsed from %s", path)
        return []
    uris = []
    for df in frames:
        uri = write_silver(df, source_type, client=client)
        uris.append(uri)
    logger.info("Parsed %s -> %d Silver object(s)", path.name, len(uris))
    return uris


def run(
    bronze_prefix: str,
    source_type: str,
    *,
    client: ObjectStoreClient | None = None,
    bronze_bucket: str | None = None,
) -> list[str]:
    """Read all objects under `bronze_prefix` from Bronze, parse each, write Silver."""
    client = client or get_minio_client()
    bronze_bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", "bronze")

    objects = [
        obj.object_name
        for obj in client.list_objects(bronze_bucket, prefix=bronze_prefix, recursive=True)
    ]
    if not objects:
        logger.warning("No objects found under %s/%s", bronze_bucket, bronze_prefix)
        return []

    # ONEMLI: bkz. parse_alfa.py/parse_uav_attack.py/parse_uav_sead.py'deki ayni
    # yorum -- write_silver append-only, tekrar calistirmadan once bu source_type'in
    # ESKI Silver ciktisini (bir onceki run()'dan kalma) temizliyoruz (Bronze'a
    # DOKUNULMUYOR). Dongu BASLAMADAN once, TEK seferlik -- yoksa bu run'in kendi
    # yazdigi parcalari da silerdik.
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
    cleared = delete_layer_objects(client, silver_bucket, source_type)
    if cleared:
        logger.info("Onceki calismadan %d Silver parcasi temizlendi (yeniden uretiliyor)", cleared)

    logger.info("Found %d object(s) under %s/%s", len(objects), bronze_bucket, bronze_prefix)
    all_uris: list[str] = []

    for obj_name in objects:
        filename = obj_name.split("/")[-1]
        data = download_raw_bytes(client, obj_name, bucket=bronze_bucket)
        frames = parse_bytes(data, filename, source_type)
        for df in frames:
            uri = write_silver(df, source_type, client=client)
            all_uris.append(uri)
        logger.info("Parsed %s -> %d Silver object(s)", obj_name, len(frames))

    logger.info("Done: %d Silver object(s) written", len(all_uris))
    return all_uris


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generic Bronze → Silver parser (any format)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bronze-prefix", help="MinIO Bronze prefix to scan")
    group.add_argument("--local-file", help="Local file path to parse directly")
    parser.add_argument("--source", required=True, help="source_type slug (e.g. 'surveildrone')")
    args = parser.parse_args()

    if args.local_file:
        parse_local_file(args.local_file, args.source)
    else:
        run(args.bronze_prefix, args.source)
