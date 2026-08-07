"""Silver for ALFA: per-sequence wide merge of Bronze topic objects.

NOT the active pipeline (see docs/PIPELINE_PLAN.md / ADR-003, docs/decisions.md):
the team moved to Bronze=raw-zip-upload-only, with `src/silver/parse_alfa.py`
(narrower, source_id + label + a handful of nav_info fields) as the real ALFA
Silver parser. This module assumed an older Bronze design (`src/ingestion/
alfa_loader.py`, since deleted) that pre-parsed every topic CSV into its own
Bronze object. Kept as a validated reference -- every claim in this docstring
about the real ALFA data (47 sequences, real column names, fault label
mapping) still holds -- for whenever the team decides to enrich the Silver
schema beyond `parse_alfa.py`'s current narrow column set.

Original Bronze contract this assumed: one Parquet object per original
topic CSV, tagged with `_alfa_scenario` / `_alfa_topic` / `_alfa_failure_label`.
This module reads all of those back, and for each scenario:

  1. Splits `failure_status-*` topics (label source) from sensor topics.
  2. Picks the highest-frequency sensor topic as the time backbone (ALFA paper,
     Keipour et al.: `mavctrl/rpy` at 50 Hz is the intended reference axis).
  3. `merge_asof`-joins every other sensor topic onto the backbone (never a
     timestamp/key JOIN across scenarios -- this is strictly intra-sequence).
  4. Derives `fault_type` from the exhaustive, verified list of the 47 real
     ALFA scenario folder suffixes (see `_EXACT_FAULT_MAP` below), and
     `is_fault` from the onset of the matching `failure_status-*` topic.
  5. Interpolates numeric NaNs (ALFA paper Algorithm 1).

Sequences are then concatenated (UNION) into one wide-but-sparse ALFA Silver
table -- deliberately kept wide (every topic's columns, prefixed by topic
name) rather than trimmed to a feature subset, so feature selection stays a
modeling-time decision.

Verified against the real ALFA `processed/` collection on 2026-07-01: 47
scenario folders (matches the paper exactly), real column names
(`field.commanded`/`field.measured` for `nav_info/*`, `field.data` for
`failure_status/*`, `field.latitude`/`field.longitude`/`field.altitude` for
`global_position/global` -- already in real-world units, NOT the
`lat/lon x 1e-7` MAVLink wire scaling assumed in earlier planning docs).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    DEFAULT_BUCKET,
    ObjectStoreClient,
    get_minio_client,
    read_layer,
    write_silver,
)

logger = logging.getLogger(__name__)

SOURCE_TYPE = "alfa"

# merge_asof tolerance in nanoseconds (topics range ~1-50 Hz; 250ms safely
# covers the slowest topics -- e.g. mavros/time_reference at ~2 Hz).
DEFAULT_TOLERANCE_NS = 250_000_000

# Prefer the paper's stated 50 Hz reference; fall back to whichever sensor
# topic has the most rows if neither is present in a given scenario.
_PREFERRED_BACKBONE_TOPICS = ("mavctrl-rpy", "mavctrl-path_dev")

# ROS message-header boilerplate repeated across nearly every topic -- no
# analytical value once `%time` gives us the real timing axis.
_ROS_BOILERPLATE_SUFFIXES = ("header.seq", "header.stamp", "header.frame_id")

_NUMERIC_PREFIX_RE = re.compile(r"^\d+_")
_EMR_TRAJ_SUFFIX = "_with_emr_traj"

# Exhaustive map of the 47 real ALFA scenario folder suffixes (after stripping
# a leading "<n>_" sub-sequence index) to a normalized fault_type. Built by
# enumerating every folder under processed/processed/ on 2026-07-01 -- not a
# guess. Anything not in this table falls back to a substring heuristic below
# (logged as a warning) so newly-added ALFA data doesn't silently mis-label.
_EXACT_FAULT_MAP: dict[str, str] = {
    "no_ground_truth": "no_ground_truth",
    "no_failure": "no_failure",
    "engine_failure": "engine_failure",
    "engine_failure_with_emr_traj": "engine_failure",
    "elevator_failure": "elevator_failure",
    "rudder_right_failure": "rudder_right_failure",
    "rudder_left_failure": "rudder_left_failure",
    "left_aileron_failure": "left_aileron_failure",
    "right_aileron_failure": "right_aileron_failure",
    "right_aileron_failure_with_emr_traj": "right_aileron_failure",
    "both_ailerons_failure": "both_aileron_failure",
    "left_aileron__right_aileron__failure": "both_aileron_failure",
    "rudder_zero__left_aileron_failure": "rudder_aileron_failure",
}

_NON_FAULT_LABELS = ("no_failure", "no_ground_truth")


def normalize_fault_label(raw_label: str) -> tuple[str, bool]:
    """Map a raw `_alfa_failure_label` to `(fault_type, emergency_traj)`.

    Exact-matches the verified suffix table first; only unrecognised suffixes
    (e.g. from ALFA data added later) fall through to the heuristic.
    """
    stripped = _NUMERIC_PREFIX_RE.sub("", raw_label)
    if stripped in _EXACT_FAULT_MAP:
        return _EXACT_FAULT_MAP[stripped], _EMR_TRAJ_SUFFIX in stripped

    logger.warning("Unrecognised ALFA failure label %r; falling back to heuristic", raw_label)
    s = stripped.lower()
    emergency_traj = _EMR_TRAJ_SUFFIX in s
    s = s.removesuffix(_EMR_TRAJ_SUFFIX)

    if "no_ground_truth" in s:
        return "no_ground_truth", emergency_traj
    if "no_failure" in s:
        return "no_failure", emergency_traj
    if "engine" in s:
        return "engine_failure", emergency_traj
    if "elevator" in s:
        return "elevator_failure", emergency_traj

    has_rudder, has_aileron = "rudder" in s, "aileron" in s
    has_left, has_right, has_both = "left" in s, "right" in s, "both" in s

    if has_rudder and has_aileron:
        return "rudder_aileron_failure", emergency_traj
    if has_aileron and (has_both or (has_left and has_right)):
        return "both_aileron_failure", emergency_traj
    if has_aileron and has_left:
        return "left_aileron_failure", emergency_traj
    if has_aileron and has_right:
        return "right_aileron_failure", emergency_traj
    if has_rudder and has_left:
        return "rudder_left_failure", emergency_traj
    if has_rudder and has_right:
        return "rudder_right_failure", emergency_traj
    return "unknown", emergency_traj


def _drop_boilerplate(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if any(c.endswith(suf) for suf in _ROS_BOILERPLATE_SUFFIXES)]
    return df.drop(columns=drop_cols, errors="ignore")


def _prepare_topic_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Clean one topic's rows: real columns only, `%time` -> sorted int64 `ts_ns`."""
    df = df.dropna(axis=1, how="all")
    if "%time" not in df.columns:
        return None
    out = df.rename(columns={"%time": "ts_ns"})
    out["ts_ns"] = out["ts_ns"].astype("int64")
    out = _drop_boilerplate(out)
    meta_cols = [c for c in out.columns if c.startswith("_")]
    out = out.drop(columns=meta_cols, errors="ignore")
    return out.sort_values("ts_ns")


def build_scenario_table(
    scenario: str,
    topic_frames: dict[str, pd.DataFrame],
    failure_label: str,
    *,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
) -> pd.DataFrame | None:
    """Wide-merge one ALFA scenario's topics into a single per-timestamp table."""
    sensor_topics = {t: df for t, df in topic_frames.items() if not t.startswith("failure_status")}
    failure_topics = {t: df for t, df in topic_frames.items() if t.startswith("failure_status")}

    prepared: dict[str, pd.DataFrame] = {}
    for topic, df in sensor_topics.items():
        cleaned = _prepare_topic_frame(df)
        if cleaned is not None and len(cleaned):
            prepared[topic] = cleaned

    if not prepared:
        logger.warning("Scenario %s: no usable sensor topics, skipping", scenario)
        return None

    backbone_topic = next((t for t in _PREFERRED_BACKBONE_TOPICS if t in prepared), None)
    if backbone_topic is None:
        backbone_topic = max(prepared, key=lambda t: len(prepared[t]))

    base = prepared[backbone_topic].rename(
        columns={c: f"{backbone_topic}__{c}" for c in prepared[backbone_topic].columns if c != "ts_ns"}
    )

    for topic, df in prepared.items():
        if topic == backbone_topic:
            continue
        renamed = df.rename(columns={c: f"{topic}__{c}" for c in df.columns if c != "ts_ns"})
        base = pd.merge_asof(
            base.sort_values("ts_ns"),
            renamed.sort_values("ts_ns"),
            on="ts_ns",
            direction="nearest",
            tolerance=tolerance_ns,
        )

    fault_type, emergency_traj = normalize_fault_label(failure_label)
    base["fault_type"] = fault_type
    base["is_fault"] = False

    if fault_type not in _NON_FAULT_LABELS:
        onset_times = []
        for topic, fdf in failure_topics.items():
            if "%time" not in fdf.columns or "field.data" not in fdf.columns:
                continue
            active = fdf[fdf["field.data"].astype(float) != 0]
            if not active.empty:
                onset_times.append(active["%time"].astype("int64").min())

        if onset_times:
            fault_start = min(onset_times)
            base["is_fault"] = base["ts_ns"] >= fault_start
            logger.info(
                "Scenario %s: fault_type=%s onset=%d (%d/%d rows marked is_fault)",
                scenario, fault_type, fault_start, int(base["is_fault"].sum()), len(base),
            )
        else:
            logger.warning(
                "Scenario %s: fault_type=%s but no active failure_status rows found -- "
                "is_fault left all-False", scenario, fault_type,
            )

    numeric_cols = base.select_dtypes(include="number").columns
    base[numeric_cols] = base[numeric_cols].interpolate(method="linear", limit_direction="both")

    base["sequence_id"] = scenario
    base["source_type"] = SOURCE_TYPE
    base["_alfa_failure_label"] = failure_label
    base["_alfa_emergency_traj"] = emergency_traj
    base["timestamp_utc"] = base["ts_ns"] / 1e9

    return base


def _coerce_mixed_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force every `object` column to one physical type so Parquet/Arrow can write it.

    ROS free-form fields (mainly `diagnostics/*`, a diagnostic_msgs/DiagnosticArray
    flattening with variable-length, free-form key/value entries) can end up
    mixing numbers and strings in the same column once scenarios are
    concatenated, even though each scenario alone looked consistent. Never
    drop the column -- numeric-coerce it if every non-null value supports
    that losslessly, else fall back to plain strings.
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str).where(df[col].notna(), None)
    return df


def build_alfa_silver(
    client: ObjectStoreClient,
    *,
    bronze_bucket: str | None = None,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
) -> pd.DataFrame:
    """Read every ALFA Bronze object and return one wide, per-sequence Silver table."""
    bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", DEFAULT_BUCKET)
    bronze = read_layer(client, bucket, SOURCE_TYPE)
    if bronze.empty:
        logger.warning("No ALFA Bronze objects found under bucket=%s", bucket)
        return bronze

    required = {"_alfa_scenario", "_alfa_topic", "_alfa_failure_label"}
    missing = required - set(bronze.columns)
    if missing:
        raise ValueError(f"ALFA Bronze data is missing expected provenance columns: {missing}")

    scenario_tables = []
    for scenario, scenario_df in bronze.groupby("_alfa_scenario"):
        failure_label = scenario_df["_alfa_failure_label"].iloc[0]
        topic_frames = {topic: topic_df for topic, topic_df in scenario_df.groupby("_alfa_topic")}
        table = build_scenario_table(scenario, topic_frames, failure_label, tolerance_ns=tolerance_ns)
        if table is not None and len(table):
            scenario_tables.append(table)
        else:
            logger.warning("Scenario %s produced no Silver rows", scenario)

    if not scenario_tables:
        logger.error("No ALFA scenarios could be built into Silver tables")
        return pd.DataFrame()

    silver = pd.concat(scenario_tables, ignore_index=True, sort=False)
    silver = _coerce_mixed_object_columns(silver)
    logger.info(
        "ALFA Silver: %d rows, %d sequences, %d columns",
        len(silver), silver["sequence_id"].nunique(), silver.shape[1],
    )
    return silver


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ALFA Bronze -> Silver (wide per-sequence merge_asof)")
    parser.add_argument("--local-out", default=None, help="Optional local Parquet path, e.g. data/silver/alfa_silver.parquet")
    parser.add_argument("--local-out-csv", default=None, help="Optional local CSV path")
    args = parser.parse_args()

    client = get_minio_client()
    silver = build_alfa_silver(client)
    if silver.empty:
        logger.error("Nothing to write: ALFA Silver is empty (did Bronze run first?)")
        return

    uri = write_silver(silver, SOURCE_TYPE, client=client)
    logger.info("Wrote ALFA Silver -> %s", uri)

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(args.local_out, index=False)
        logger.info("Local copy written: %s", args.local_out)
    if args.local_out_csv:
        Path(args.local_out_csv).parent.mkdir(parents=True, exist_ok=True)
        silver.to_csv(args.local_out_csv, index=False)
        logger.info("Local CSV copy written: %s", args.local_out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
