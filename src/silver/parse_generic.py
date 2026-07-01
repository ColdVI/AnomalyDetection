"""parse_generic.py -- Generic Silver parser for any new dataset.

Automatically detects file format (CSV, JSON, JSONL, zip, tar), reads into
a DataFrame, attaches provenance, and writes Silver Parquet to MinIO.
Does NOT do unit conversions or label extraction -- those are domain-specific
and belong in a custom parser built on top of this one.

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

import io
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    download_raw_bytes,
    get_minio_client,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".xlsx", ".xls"}


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
        # try records list first, fall back to lines
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

    logger.warning("Unsupported file format: %s -- skipping", filename)
    return None


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
