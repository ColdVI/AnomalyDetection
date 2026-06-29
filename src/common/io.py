"""Consistent Parquet output for Bronze datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

DEFAULT_BRONZE_ROOT = Path("data/bronze")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.=-]*$")


def _validate_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must be a non-empty filesystem-safe name; got {value!r}"
        )
    return value


def write_bronze(
    df: pd.DataFrame,
    source_type: str,
    partition: str | None = None,
    *,
    bronze_root: str | Path = DEFAULT_BRONZE_ROOT,
) -> Path:
    """Write one immutable Snappy Parquet part and return its path."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    source = _validate_component(source_type, "source_type")
    output_dir = Path(bronze_root) / source
    if partition is not None:
        output_dir /= _validate_component(partition, "partition")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"part-{timestamp}-{uuid4().hex[:8]}.parquet"
    df.to_parquet(output_path, index=False, compression="snappy", engine="pyarrow")
    return output_path
