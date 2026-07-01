"""Helpers for attaching immutable provenance metadata.

Per ADR-003 (docs/PIPELINE_PLAN.md), provenance is now attached in Silver, not Bronze
(Bronze holds untouched raw files) -- hence the "silver_v1" default. Callers that still
attach provenance at Bronze time (pre-ADR-003 call sites) pass `schema_version="bronze_v1"`
explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

PROVENANCE_COLUMNS = (
    "_source_type",
    "_ingest_ts_utc",
    "_source_file",
    "_schema_version",
)


def add_provenance(
    df: pd.DataFrame,
    source_type: str,
    source_file: str,
    schema_version: str = "silver_v1",
) -> pd.DataFrame:
    """Return a copy with standard provenance columns; never mutate ``df``."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(source_type, str) or not source_type:
        raise ValueError("source_type must be a non-empty string")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("source_file must be a non-empty string")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version must be a non-empty string")

    result = df.copy(deep=True)
    ingest_ts = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    result["_source_type"] = source_type
    result["_ingest_ts_utc"] = ingest_ts
    result["_source_file"] = source_file
    result["_schema_version"] = schema_version
    return result
