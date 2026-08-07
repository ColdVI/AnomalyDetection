"""Silver for UAV Attack: per-log wide merge of Bronze topic objects.

NOT the active pipeline (see docs/PIPELINE_PLAN.md / ADR-003, docs/decisions.md):
the team moved to Bronze=raw-zip-upload-only, with `src/silver/parse_uav_attack.py`
(narrower column set, moved from the pre-ADR-003 `parse_uav_attack.py`) as the
real UAV Attack Silver parser -- it also picked up this module's log_id/topic
regex fix (see below). Kept as a validated reference -- the real-data findings
below (folder structure, column names, the regex bug) still hold -- for
whenever the team decides to enrich the Silver schema (e.g. the
`vehicle_global_position_groundtruth` / raw-GPS-position spoofing features
this module adds) beyond `parse_uav_attack.py`'s current narrow columns.

Verified against the real IEEE DataPort UAV Attack collection (683.9 MB zip,
extracted to Desktop\\UAVAttackData) on 2026-07-01: both the "Live GPS
Spoofing and Jamming" and "Simulated - OTU Survey" folder trees match the
structure `src/ingestion/uav_attack_loader.py` already assumed, and the real
`vehicle_global_position` / `vehicle_attitude` / `battery_status` /
`vehicle_gps_position` CSVs have exactly the columns this module expects.

One real-data correction from the original plan: PX4 log-id prefixes are NOT
uniform across collections (`log_12_2020-8-2-14-18-24_...` for Simulated,
`ace-benign-log_0_2033-8-19-16-27-30_...` for Live, `001-2021-01-27-09-08-37-
708_...` for live Ping DoS) and contain their own underscores, so a generic
"find the last `_<word>_<n>.csv`" regex (what the earlier
`src/bronze2silverParsers/parse_uav_attack.py` used, and what this module
originally ported) mis-splits on the FIRST underscore instead of the topic
boundary. Fixed here by matching each of the small set of known topic names
we actually merge, anchored to the filename's end, instead of guessing a
generic split point.

Bronze (`src/ingestion/uav_attack_loader.py`) writes one Parquet object per
original per-topic CSV (ulog2csv output), tagged with `_attack_label`
("benign"/"malicious"), `_attack_type` ("normal"/"gps_spoofing"/"gps_jamming"/
"ping_dos"), `_attack_platform`, `_attack_collection`, and `_source_file` (the
relative path of the original CSV). Bronze does NOT extract a log id / topic
column -- that happens here, in Silver, from `_source_file`.

PX4 caveat (docs/MEMORY.md): each CSV's `timestamp` field is microseconds
since autopilot boot, NOT wall-clock UTC. Real UTC is only available via
`vehicle_gps_position.time_utc_usec` when present; `timestamp_is_real_utc`
records which case applied per row.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.minio_io import (
    DEFAULT_BUCKET,
    ObjectStoreClient,
    get_minio_client,
    list_layer_objects,
    read_parquet_object,
    write_silver,
)

logger = logging.getLogger(__name__)

SOURCE_TYPE = "uav_attack"

# PX4 topics used here run well above ALFA's ROS topics; 200ms mirrors the
# tolerance parse_uav_attack.py already validated against real logs.
DEFAULT_TOLERANCE_US = 200_000

_ATTITUDE_COLS = ["q[0]", "q[1]", "q[2]", "q[3]"]
_BATTERY_COLS = ["voltage_v", "remaining", "current_a"]
_GPS_COLS = [
    "jamming_indicator", "noise_per_ms", "hdop", "vdop",
    "satellites_used", "s_variance_m_s", "eph", "epv", "time_utc_usec",
]
# vehicle_gps_position's own lat/lon are raw-receiver estimates, scaled x1e7
# (MAVLink wire convention) -- unlike vehicle_global_position's, which are
# already plain degrees. Kept as an independent cross-check signal (receiver
# vs. fused-estimate position divergence is a natural spoofing indicator).
_GPS_LATLON_COLS = ["lat", "lon"]
_ATTACK_META_COLS = ("_attack_label", "_attack_type", "_attack_platform", "_attack_collection")

# Known uORB topics this Silver step merges, matched by exact name anchored to
# the filename's end (`_<topic>_<instance>.csv`). `vehicle_global_position_
# groundtruth` (Simulated/SITL-only: the simulator's undisturbed position) is
# intentionally listed so it is never mistaken for plain
# `vehicle_global_position` by a looser match.
_KNOWN_TOPICS = (
    "vehicle_global_position_groundtruth",
    "vehicle_global_position",
    "vehicle_attitude",
    "battery_status",
    "vehicle_gps_position",
)
_TOPIC_PATTERNS = {
    topic: re.compile(rf"_{re.escape(topic)}_(\d+)\.csv$", re.IGNORECASE) for topic in _KNOWN_TOPICS
}


def _log_id_and_topic(source_file: str) -> tuple[str, str] | tuple[None, None]:
    """Recover (log_id, topic) from a Bronze `_source_file` path.

    Only recognises the topics in `_KNOWN_TOPICS` (this Silver step's inputs);
    everything else (the ~25 other uORB topics ulog2csv also exports, e.g.
    `actuator_controls_0`, `sensor_combined`) returns (None, None) and is
    dropped -- expected, not an error, since this Silver table doesn't use them.
    """
    name = Path(source_file).name
    for topic, pattern in _TOPIC_PATTERNS.items():
        m = pattern.search(name)
        if m:
            return name[: m.start()], topic
    return None, None


def _quat_to_euler_deg(q0, q1, q2, q3):
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def build_log_table(
    log_id: str,
    topic_frames: dict[str, pd.DataFrame],
    *,
    tolerance_us: int = DEFAULT_TOLERANCE_US,
) -> pd.DataFrame | None:
    """Wide-merge one UAV Attack log's topics onto a `vehicle_global_position` backbone."""
    pos_df = topic_frames.get("vehicle_global_position")
    if pos_df is None or "timestamp" not in pos_df.columns:
        logger.warning("Log %s: no vehicle_global_position topic, skipping", log_id)
        return None

    needed = ["timestamp", "lat", "lon", "alt"]
    if not all(c in pos_df.columns for c in needed):
        logger.warning("Log %s: vehicle_global_position missing lat/lon/alt, skipping", log_id)
        return None

    meta = pos_df.iloc[0]
    extra_cols = [c for c in ("eph", "epv") if c in pos_df.columns]
    base = pos_df[needed + extra_cols].sort_values("timestamp").reset_index(drop=True)

    attitude_df = topic_frames.get("vehicle_attitude")
    if attitude_df is not None and all(c in attitude_df.columns for c in _ATTITUDE_COLS):
        small = attitude_df[["timestamp"] + _ATTITUDE_COLS].sort_values("timestamp")
        base = pd.merge_asof(base.sort_values("timestamp"), small, on="timestamp",
                              direction="nearest", tolerance=tolerance_us)
        roll, pitch, yaw = _quat_to_euler_deg(base["q[0]"], base["q[1]"], base["q[2]"], base["q[3]"])
        base["roll_deg"], base["pitch_deg"], base["yaw_deg"] = roll, pitch, yaw
        base = base.drop(columns=_ATTITUDE_COLS)

    battery_df = topic_frames.get("battery_status")
    if battery_df is not None:
        cols_present = [c for c in _BATTERY_COLS if c in battery_df.columns]
        if cols_present:
            small = battery_df[["timestamp"] + cols_present].sort_values("timestamp")
            base = pd.merge_asof(base.sort_values("timestamp"), small, on="timestamp",
                                  direction="nearest", tolerance=tolerance_us)

    gps_df = topic_frames.get("vehicle_gps_position")
    if gps_df is not None:
        cols_present = [c for c in _GPS_COLS if c in gps_df.columns]
        latlon_present = [c for c in _GPS_LATLON_COLS if c in gps_df.columns]
        if cols_present or latlon_present:
            small = gps_df[["timestamp"] + cols_present + latlon_present].sort_values("timestamp")
            small = small.rename(columns={"eph": "raw_gps_eph", "epv": "raw_gps_epv"})
            if latlon_present:
                # MAVLink-wire scaling (x1e7 int degrees) -- vehicle_global_position's
                # lat/lon are already plain degrees, so these need converting to be
                # comparable as an independent receiver-vs-estimate cross-check.
                small = small.rename(columns={"lat": "raw_gps_lat", "lon": "raw_gps_lon"})
                for col in ("raw_gps_lat", "raw_gps_lon"):
                    if col in small.columns:
                        small[col] = small[col] / 1e7
            base = pd.merge_asof(base.sort_values("timestamp"), small, on="timestamp",
                                  direction="nearest", tolerance=tolerance_us)

    groundtruth_df = topic_frames.get("vehicle_global_position_groundtruth")
    if groundtruth_df is not None and all(c in groundtruth_df.columns for c in ("lat", "lon", "alt")):
        small = groundtruth_df[["timestamp", "lat", "lon", "alt"]].rename(
            columns={"lat": "gt_lat", "lon": "gt_lon", "alt": "gt_alt"}
        ).sort_values("timestamp")
        base = pd.merge_asof(base.sort_values("timestamp"), small, on="timestamp",
                              direction="nearest", tolerance=tolerance_us)

    base["log_id"] = log_id
    base["source_type"] = SOURCE_TYPE
    for col in _ATTACK_META_COLS:
        base[col] = meta.get(col)

    if "time_utc_usec" in base.columns and base["time_utc_usec"].notna().any():
        base["timestamp_utc"] = base["time_utc_usec"] / 1e6
        base["timestamp_is_real_utc"] = True
    else:
        base["timestamp_utc"] = base["timestamp"] / 1e6
        base["timestamp_is_real_utc"] = False

    numeric_cols = base.select_dtypes(include="number").columns
    base[numeric_cols] = base[numeric_cols].interpolate(method="linear", limit_direction="both")

    return base


def build_uav_attack_silver(
    client: ObjectStoreClient,
    *,
    bronze_bucket: str | None = None,
    tolerance_us: int = DEFAULT_TOLERANCE_US,
) -> pd.DataFrame:
    """Read UAV Attack Bronze objects for the known topics and return one wide, per-log Silver table.

    Unlike ALFA (small, ROS-rate data), UAV Attack Bronze holds ~30 uORB topics per log,
    several logged at very high rates (`sensor_combined`, `actuator_outputs`,
    `ekf2_innovations`...). Concatenating every object first (`read_layer`) and filtering
    after was tried and blew past available memory (~25M rows x the full ~50-column union
    for one real run). Instead, each object is read once, checked against `_KNOWN_TOPICS`
    immediately, and discarded if irrelevant -- only the handful of topics this Silver step
    actually merges are ever held in memory together.
    """
    bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", DEFAULT_BUCKET)
    object_names = list_layer_objects(client, bucket, SOURCE_TYPE)
    if not object_names:
        logger.warning("No UAV Attack Bronze objects found under bucket=%s", bucket)
        return pd.DataFrame()

    kept_frames = []
    skipped = 0
    for object_name in object_names:
        df = read_parquet_object(client, bucket, object_name)
        if df.empty or "_source_file" not in df.columns:
            skipped += 1
            continue
        log_id, topic = _log_id_and_topic(df["_source_file"].iloc[0])
        if log_id is None:
            skipped += 1
            continue
        kept_frames.append(df.assign(_log_id=log_id, _topic=topic))

    logger.info(
        "%d/%d Bronze objects matched known topics %s; rest are topics this Silver step "
        "doesn't merge, skipped",
        len(kept_frames), len(object_names), _KNOWN_TOPICS,
    )
    if not kept_frames:
        logger.error("No UAV Attack Bronze objects matched any known topic")
        return pd.DataFrame()

    bronze = pd.concat(kept_frames, ignore_index=True, sort=False)

    log_tables = []
    for log_id, log_df in bronze.groupby("_log_id"):
        topic_frames = {topic: topic_df.dropna(axis=1, how="all") for topic, topic_df in log_df.groupby("_topic")}
        table = build_log_table(log_id, topic_frames, tolerance_us=tolerance_us)
        if table is not None and len(table):
            log_tables.append(table)

    if not log_tables:
        logger.error("No UAV Attack logs could be built into Silver tables")
        return pd.DataFrame()

    silver = pd.concat(log_tables, ignore_index=True, sort=False)
    logger.info(
        "UAV Attack Silver: %d rows, %d logs, %d columns",
        len(silver), silver["log_id"].nunique(), silver.shape[1],
    )
    return silver


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UAV Attack Bronze -> Silver (wide per-log merge_asof)")
    parser.add_argument("--local-out", default=None, help="Optional local Parquet path")
    parser.add_argument("--local-out-csv", default=None, help="Optional local CSV path")
    args = parser.parse_args()

    client = get_minio_client()
    silver = build_uav_attack_silver(client)
    if silver.empty:
        logger.error("Nothing to write: UAV Attack Silver is empty (did Bronze run first?)")
        return

    uri = write_silver(silver, SOURCE_TYPE, client=client)
    logger.info("Wrote UAV Attack Silver -> %s", uri)

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
