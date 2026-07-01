"""Bronze loader for ALFA (Autonomous Learning Framework for Anomalies) dataset.

Phase 4 of docs/bronze_implementasyon_plani.md.

ALFA processed collection structure (verified from real Drive files 2026-07-01):
  processed/<scenario_folder>/<scenario_name>-<topic>.csv

Scenario folder name pattern:
  carbonZ_<YYYY-MM-DD-HH-MM-SS>_<failure_label>
  e.g. carbonZ_2018-07-18-12-10-11_no_ground_truth       -> normal flight
       carbonZ_2018-07-18-15-53-31_1_engine_failure       -> engine 1 failure
       carbonZ_2018-07-18-15-53-31_2_engine_failure       -> engine 2 failure

Each topic CSV has:
  - `%time`      : nanosecond ROS timestamps
  - `field.*`    : original ROS message fields (kept as-is per Bronze rule)
  - `failure_status-engines.csv` only exists for failure scenarios

Bronze adds three extra provenance columns beyond the standard four:
  _alfa_failure_label : failure_label extracted from folder name
                        ("no_ground_truth" | "1_engine_failure" | ...)
  _alfa_scenario      : full scenario folder name
  _alfa_topic         : topic suffix (e.g. "mavros-global_position-global")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.common.io import ObjectStoreClient, write_bronze
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "alfa"
# Matches: carbonZ_2018-07-18-12-10-11_no_ground_truth
#          carbonZ_2018-07-18-15-53-31_1_engine_failure
_SCENARIO_RE = re.compile(r"^carbonZ_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)$")


def _parse_scenario_folder(folder_name: str) -> str | None:
    """Return the failure label from the scenario folder name, or None if unrecognised."""
    m = _SCENARIO_RE.match(folder_name)
    return m.group(1) if m else None


def _topic_from_filename(csv_name: str, scenario_name: str) -> str:
    """Strip the scenario prefix and .csv suffix to get the topic string.

    e.g. 'carbonZ_..._1_engine_failure-mavros-global_position-global.csv'
         -> 'mavros-global_position-global'
    """
    prefix = scenario_name + "-"
    stem = csv_name.removesuffix(".csv")
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def load_scenario(
    scenario_dir: Path,
    *,
    client: ObjectStoreClient | None = None,
) -> list[str]:
    """Load all topic CSVs in one ALFA scenario folder into Bronze.

    Returns the list of `s3://` URIs written (one per topic CSV).
    """
    failure_label = _parse_scenario_folder(scenario_dir.name)
    if failure_label is None:
        logger.warning("Skipping unrecognised scenario folder: %s", scenario_dir.name)
        return []

    written: list[str] = []
    csv_files = sorted(scenario_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files in %s", scenario_dir)
        return []

    for csv_path in csv_files:
        topic = _topic_from_filename(csv_path.name, scenario_dir.name)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to read %s", csv_path)
            continue

        df = add_provenance(
            df,
            source_type=SOURCE_TYPE,
            source_file=str(csv_path.relative_to(csv_path.parents[3])),
            schema_version="bronze_v1",
        )
        df["_alfa_failure_label"] = failure_label
        df["_alfa_scenario"] = scenario_dir.name
        df["_alfa_topic"] = topic

        uri = write_bronze(df, "alfa", client=client)
        written.append(uri)
        logger.debug("Wrote %s -> %s", csv_path.name, uri)

    logger.info(
        "Scenario %s: %d CSV files -> %d Bronze objects (label=%s)",
        scenario_dir.name, len(csv_files), len(written), failure_label,
    )
    return written


def extract_alfa(
    input_dir: str | Path,
    *,
    client: ObjectStoreClient | None = None,
) -> list[str]:
    """Walk `<input_dir>/processed/` and load every scenario into Bronze.

    Returns all `s3://` URIs written across all scenarios.
    """
    input_dir = Path(input_dir)
    processed_dir = input_dir / "processed"
    if not processed_dir.exists():
        logger.warning("processed/ sub-directory not found under %s", input_dir)
        return []

    all_written: list[str] = []
    scenario_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())
    if not scenario_dirs:
        logger.warning("No scenario sub-directories found in %s", processed_dir)
        return []

    for scenario_dir in scenario_dirs:
        all_written.extend(load_scenario(scenario_dir, client=client))

    logger.info(
        "Done: %d scenarios processed, %d Bronze objects written",
        len(scenario_dirs), len(all_written),
    )
    return all_written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ALFA processed CSVs -> Bronze")
    parser.add_argument(
        "--input",
        default="data/bronze/alfa/_input",
        help="Directory containing the ALFA processed/ sub-folder",
    )
    args = parser.parse_args()

    extract_alfa(args.input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
