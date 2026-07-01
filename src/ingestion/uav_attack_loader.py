"""Bronze loader for the UAV Attack dataset (GPS Spoofing / Jamming / Ping DoS).

Phase 5 of docs/bronze_implementasyon_plani.md.

Verified folder structure (from real files, 2026-07-01):

  _input/
  ├── Live GPS Spoofing and Jamming/        collection = "live"
  │   ├── Benign Flight/                    label = "benign",    attack = "normal"
  │   │   └── ace-benign-log_*_<topic>_<n>.csv
  │   ├── GPS Jamming/                      label = "malicious", attack = "gps_jamming"
  │   │   ├── Processed/                    (merged/derived CSVs, same labels)
  │   │   └── ace-jamming-log_*_<topic>_<n>.csv
  │   └── GPS Spoofing/                     label = "malicious", attack = "gps_spoofing"
  │       ├── Processed/
  │       └── ace-spoofing-log_*_<topic>_<n>.csv
  └── Simulated - OTU Survey/               collection = "simulated"
      ├── PX4-H480-SITL/                    platform = "PX4-H480-SITL"
      │   ├── Normal/                       label = "benign",    attack = "normal"
      │   ├── GPS Spoofing/                 label = "malicious", attack = "gps_spoofing"
      │   └── Ping DoS/                     label = "malicious", attack = "ping_dos"
      ├── PX4-PLANE-SITL/ ...
      ├── PX4-QUAD-HITL/  ...
      ├── PX4-QUAD-SITL/  ...
      ├── PX4-TAIL-SITL/  ...
      └── PX4-VTOL-SITL/  ...

Bronze adds four extra provenance columns beyond the standard four:
  _attack_label      : "benign" | "malicious"
  _attack_type       : "normal" | "gps_spoofing" | "gps_jamming" | "ping_dos"
  _attack_platform   : e.g. "PX4-H480-SITL" | "live"
  _attack_collection : "live" | "simulated"
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.common.io import ObjectStoreClient, write_bronze
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "uav_attack"

# Maps attack-type folder name → (label, attack_type slug)
_ATTACK_FOLDER_MAP: dict[str, tuple[str, str]] = {
    "Normal":        ("benign",    "normal"),
    "Benign Flight": ("benign",    "normal"),
    "GPS Spoofing":  ("malicious", "gps_spoofing"),
    "GPS Jamming":   ("malicious", "gps_jamming"),
    "Ping DoS":      ("malicious", "ping_dos"),
}

_COLLECTION_PREFIXES = {
    "Live GPS Spoofing and Jamming": "live",
    "Simulated - OTU Survey":        "simulated",
}


def _parse_csv_path(csv_path: Path, input_dir: Path) -> dict[str, str] | None:
    """Derive collection / platform / label / attack_type from a CSV's path.

    Returns None if the path doesn't match any known structure.
    """
    try:
        rel = csv_path.relative_to(input_dir)
    except ValueError:
        return None

    parts = rel.parts  # e.g. ('Live GPS Spoofing and Jamming', 'GPS Jamming', 'Processed', 'x.csv')

    if len(parts) < 3:
        return None

    collection_folder = parts[0]
    collection = _COLLECTION_PREFIXES.get(collection_folder)
    if collection is None:
        logger.warning("Unknown collection folder: %s", collection_folder)
        return None

    if collection == "live":
        # parts: (collection, attack_folder[, 'Processed'], filename)
        attack_folder = parts[1]
        label_attack = _ATTACK_FOLDER_MAP.get(attack_folder)
        if label_attack is None:
            logger.warning("Unknown live attack folder: %s", attack_folder)
            return None
        label, attack_type = label_attack
        return {"collection": collection, "platform": "live", "label": label, "attack_type": attack_type}

    # simulated: parts: (collection, platform_folder, attack_folder[, ...], filename)
    if len(parts) < 4:
        return None
    platform = parts[1]
    attack_folder = parts[2]
    label_attack = _ATTACK_FOLDER_MAP.get(attack_folder)
    if label_attack is None:
        logger.warning("Unknown simulated attack folder: %s", attack_folder)
        return None
    label, attack_type = label_attack
    return {"collection": collection, "platform": platform, "label": label, "attack_type": attack_type}


def extract_uav_attack(
    input_dir: str | Path,
    *,
    client: ObjectStoreClient | None = None,
) -> list[str]:
    """Walk `input_dir`, load every CSV into Bronze with attack metadata.

    Returns all `s3://` URIs written.
    """
    input_dir = Path(input_dir)
    all_written: list[str] = []
    csv_files = sorted(input_dir.rglob("*.csv"))

    if not csv_files:
        logger.warning("No CSV files found under %s", input_dir)
        return []

    for csv_path in csv_files:
        meta = _parse_csv_path(csv_path, input_dir)
        if meta is None:
            logger.warning("Skipping unrecognised path: %s", csv_path)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to read %s", csv_path)
            continue

        rel_source = str(csv_path.relative_to(input_dir))
        df = add_provenance(
            df,
            source_type=SOURCE_TYPE,
            source_file=rel_source,
            schema_version="bronze_v1",
        )
        df["_attack_label"]      = meta["label"]
        df["_attack_type"]       = meta["attack_type"]
        df["_attack_platform"]   = meta["platform"]
        df["_attack_collection"] = meta["collection"]

        uri = write_bronze(df, "uav_attack", client=client)
        all_written.append(uri)
        logger.debug("Wrote %s -> %s", csv_path.name, uri)

    logger.info(
        "Done: %d CSV files processed, %d Bronze objects written",
        len(csv_files), len(all_written),
    )
    return all_written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UAV Attack dataset -> Bronze")
    parser.add_argument(
        "--input",
        default="data/bronze/uav_attack/_input",
        help="Directory containing Live/Simulated sub-folders",
    )
    args = parser.parse_args()
    extract_uav_attack(args.input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
