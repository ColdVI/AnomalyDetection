"""RflyMAD-Full v2 dataset contract and canonical flight manifest.

The v1 overnight pipeline intentionally optimized for resumable acquisition.  This
module is the model-facing boundary: one row is one ULog flight candidate, exact
duplicates share a canonical id, environmental wind is not silently promoted to
a system fault, and every split is assigned at group/flight level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from rfly_full.pipeline import ARTIFACT_ROOT, PARSED_ROOT, _atomic_json, case_id

V2_ROOT = ARTIFACT_ROOT / "v2"
DATASET_MANIFEST = V2_ROOT / "dataset_manifest.parquet"
DATASET_MANIFEST_CSV = V2_ROOT / "dataset_manifest.csv"
MANIFEST_SUMMARY = V2_ROOT / "dataset_manifest_summary.json"


SPLIT_REGISTRY = V2_ROOT / 'split_registry.json'


def _token(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def domain_of(package: str) -> str:
    upper = package.upper()
    if upper.startswith("SIL"):
        return "SIL"
    if upper.startswith("HIL"):
        return "HIL"
    if upper.startswith("REAL"):
        return "Real"
    return "Unknown"


def taxonomy(package: str, object_name: str) -> dict[str, object]:
    """Map published package/path names without conflating wind and faults."""
    joined = _token(f"{package}/{object_name}")
    if "nofault" in joined:
        family, system_fault, environment = "NoFault", False, "None"
    elif "wind" in joined:
        family, system_fault, environment = "Environment", False, "Wind"
    elif "motor" in joined:
        family, system_fault, environment = "Motor", True, "None"
    elif "prop" in joined:
        family, system_fault, environment = "Propeller", True, "None"
    elif "voltage" in joined:
        family, system_fault, environment = "Voltage", True, "None"
    elif "load" in joined:
        family, system_fault, environment = "Load", True, "None"
    elif "sensor" in joined or any(
        key in joined for key in ("accelerometer", "acce", "gyro", "magnet", "baro", "gps")
    ):
        family, system_fault, environment = "Sensor", True, "None"
    else:
        family, system_fault, environment = "Unknown", True, "None"

    subtype = "Unknown"
    subtype_rules = (
        ("motor1", "Motor_1"), ("motor2", "Motor_2"),
        ("prop", "Propeller"), ("voltage", "Low_Voltage"),
        ("load", "Load_Loss"), ("accelerometer", "Accelerometer"),
        ("acce", "Accelerometer"), ("gyro", "Gyroscope"),
        ("magnet", "Magnetometer"), ("baro", "Barometer"),
        ("gps", "GPS"), ("wind", "Wind"), ("nofault", "NoFault"),
    )
    for needle, value in subtype_rules:
        if needle in joined:
            subtype = value
            break
    return {
        "fault_family": family,
        "fault_subtype": subtype,
        "system_fault": system_fault,
        "environment_condition": environment,
        "evaluation_role": (
            "environment_robustness" if environment == "Wind"
            else "fault_detection" if system_fault
            else "normal_reference"
        ),
    }


def window_phase(fault_active: np.ndarray, *, is_system_fault: bool) -> str:
    """Classify a complete window; mixed windows are never silently relabelled."""
    active = np.asarray(fault_active, dtype=bool)
    if not len(active):
        return "empty"
    if active.all():
        return "fault_active"
    if active.any():
        return "transition"
    return "pre_or_post_fault" if is_system_fault else "normal"


def _source_records() -> tuple[dict[str, dict], dict[str, list[tuple[str, dict]]]]:
    """Return ULog records and package-indexed TestInfo records from both mirrors."""
    ulogs: dict[str, dict] = {}
    infos: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    sources = (
        (ARTIFACT_ROOT / "download_manifest.json", "xianglile/rflymad"),
        (ARTIFACT_ROOT / "expansion_manifest.json", None),
    )
    for path, default_source in sources:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_key, record in payload.get("files", {}).items():
            if default_source is None and ":" in raw_key:
                source, name = raw_key.split(":", 1)
            else:
                source, name = str(payload.get("dataset", default_source)), raw_key
            item = {**record, "source_archive": source, "object_name": name}
            if name.lower().endswith(".ulg"):
                ulogs[name] = item
            elif PurePosixPath(name).name.lower().startswith("testinfo"):
                infos[name.split("/", 1)[0]].append((name, item))
    return ulogs, infos


def _testinfo_for(object_name: str, infos: dict[str, list[tuple[str, dict]]]) -> dict | None:
    package = object_name.split("/", 1)[0]
    flight = case_id(object_name)
    candidates = []
    for name, record in infos.get(package, []):
        parent = PurePosixPath(name).parent.as_posix()
        if flight == parent or flight.startswith(parent + "/"):
            candidates.append((len(parent), record))
    return max(candidates, default=(0, None), key=lambda value: value[0])[1]


def _testinfo_fields(record: dict) -> dict:
    result = {
        'fault_id': None, 'fault_parameter': None, 'fault_severity': 'unknown',
        'planned_fault_start_s': np.nan, 'planned_fault_end_s': np.nan,
        'platform_metadata': 'unknown',
    }
    raw_path = record.get('path') if record else None
    if not raw_path or not Path(raw_path).exists():
        return result
    path = Path(raw_path)
    try:
        table = pd.read_excel(path, header=None) if path.suffix.lower() == '.xlsx' else pd.read_csv(path, header=None)
    except Exception:
        return result
    for row in table.fillna('').astype(str).to_numpy():
        if len(row) < 2:
            continue
        key = ''.join(character for character in row[0].lower() if character.isalnum())
        value = str(row[1]).strip()
        if key == 'faultid':
            result['fault_id'] = value or None
        elif key == 'faultparameter':
            result['fault_parameter'] = value or None
        elif key in {'faultseverity', 'severity'}:
            result['fault_severity'] = value or 'unknown'
        elif key == 'faultinjectiontime':
            try:
                result['planned_fault_start_s'] = float(value)
            except ValueError:
                pass
        elif key == 'testendtime':
            try:
                result['planned_fault_end_s'] = float(value)
            except ValueError:
                pass
        elif key in {'platform', 'vehicletype', 'uavtype'}:
            result['platform_metadata'] = value or 'unknown'
    return result


def _flight_rows() -> list[dict]:
    ulogs, infos = _source_records()
    rows: list[dict] = []
    columns = [
        "case_id", "object_name", "package", "label", "fault_active",
        "truth_source", "t_rel_s",
    ]
    for parquet in sorted(PARSED_ROOT.glob("*/*.parquet")):
        if parquet.parent.name == "bootstrap_normal":
            # The same real flights are present in the acquired package pool.
            continue
        frame = pd.read_parquet(parquet, columns=columns)
        for source_id, flight in frame.groupby("case_id", sort=False):
            first = flight.iloc[0]
            object_name = str(first["object_name"])
            package = str(first["package"])
            active = flight.loc[flight["fault_active"], "t_rel_s"]
            ulog = ulogs.get(object_name, {})
            info = _testinfo_for(object_name, infos) or {}
            info_fields = _testinfo_fields(info)
            taxonomy_fields = taxonomy(package, object_name)
            duration = float(flight["t_rel_s"].max() - flight["t_rel_s"].min())
            rows.append({
                "case_id": str(source_id),
                "object_name": object_name,
                "source_archive": ulog.get("source_archive", "unknown"),
                "dataset_version": "kaggle_snapshot_2026-07-20",
                "domain": domain_of(package),
                "representation": "ULog+TestInfo" if info else "ULog",
                "platform": "unknown",
                "flight_status": "fault" if taxonomy_fields["system_fault"] else "normal_or_environment",
                **info_fields,
                **taxonomy_fields,
                "fault_start_s": float(active.min()) if len(active) else np.nan,
                "fault_end_s": float(active.max()) if len(active) else np.nan,
                "duration_s": duration,
                "truth_source": str(first["truth_source"]),
                "ulog_sha256": ulog.get("sha256"),
                "testinfo_sha256": info.get("sha256"),
                "ulog_bytes": ulog.get("bytes"),
                "parsed_path": parquet.as_posix(),
            })
    return rows


def _session_key(row: pd.Series) -> str:
    if row.domain != 'Real':
        parts = PurePosixPath(str(row.object_name)).parts
        boundary = next(
            (index for index, value in enumerate(parts) if value.startswith('TestCase')),
            len(parts) - 1,
        )
        scenario = '/'.join(parts[:boundary])
        return f'simulation_scenario:{scenario}'
    dated_session = re.search(r'20\d{2}[-_]\d{1,2}[-_]\d{1,2}', str(row.object_name))
    if dated_session:
        return f'real_session:{row.fault_family}:{dated_session.group(0)}'
    if row["domain"] != "Real":
        return str(row["canonical_case_id"])
    found = re.search(r"20\d{2}[-_]\d{1,2}[-_]\d{1,2}", str(row["object_name"]))
    return f"real_session:{found.group(0)}" if found else str(row["canonical_case_id"])


def _assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["split_group_id"] = frame.apply(_session_key, axis=1)
    representatives = frame.drop_duplicates("split_group_id").copy()
    representatives["stratum"] = representatives[
        ["domain", "fault_family", "environment_condition"]
    ].astype(str).agg("|".join, axis=1)
    flight_counts = frame.groupby('split_group_id').size().to_dict()
    assignments: dict[str, tuple[str, int]] = {}
    for stratum, group in representatives.groupby("stratum", sort=True):
        ordered = sorted(
            group["split_group_id"].astype(str),
            key=lambda value: hashlib.sha256(f"{stratum}|{value}".encode()).hexdigest(),
        )
        sizes = {group_id: int(flight_counts[group_id]) for group_id in ordered}
        target = round(sum(sizes.values()) * 0.20)
        choices: dict[int, tuple[str, ...]] = {0: ()}
        for group_id in ordered:
            size = int(sizes[group_id])
            for total, chosen_ids in list(choices.items()):
                candidate_total = total + size
                candidate_ids = (*chosen_ids, group_id)
                if candidate_total not in choices or candidate_ids < choices[candidate_total]:
                    choices[candidate_total] = candidate_ids
        valid_totals = [
            total for total in choices
            if total > 0 and (len(ordered) == 1 or total < sum(sizes.values()))
        ]
        best_total = min(valid_totals or list(choices), key=lambda total: (abs(total - target), total))
        locked_ids = set(choices[best_total])
        offset = int(hashlib.sha256(stratum.encode()).hexdigest()[:4], 16) % 5
        for rank, group_id in enumerate(ordered):
            locked = (rank + offset) % 5 == 0
            # Stable under future dataset growth: adding a case must never move
            # an existing group into or out of the locked test partition.
            locked_bucket = int(
                hashlib.sha256(f'locked|{group_id}'.encode()).hexdigest()[:8], 16
            ) % 100
            locked = locked_bucket < 20
            locked = group_id in locked_ids
            fold = int(hashlib.sha256(f"cv|{group_id}".encode()).hexdigest()[:8], 16) % 5
            assignments[group_id] = ("locked_test" if locked else "development", fold)
    frame["split"] = frame["split_group_id"].map(lambda value: assignments[str(value)][0])
    frame["cv_fold"] = frame["split_group_id"].map(lambda value: assignments[str(value)][1]).astype(int)
    if SPLIT_REGISTRY.exists():
        frozen = json.loads(SPLIT_REGISTRY.read_text(encoding='utf-8')).get('groups', {})
        for group_id, value in frozen.items():
            mask = frame.split_group_id.eq(group_id)
            if mask.any():
                frame.loc[mask, 'split'] = str(value['split'])
                frame.loc[mask, 'cv_fold'] = int(value['cv_fold'])
    return frame


def build_manifest() -> pd.DataFrame:
    frame = pd.DataFrame(_flight_rows())
    if frame.empty:
        raise RuntimeError("no parsed RflyMAD flights found")
    fallback = frame.apply(
        lambda row: hashlib.sha256(
            f"{row.source_archive}|{row.object_name}".encode()
        ).hexdigest(), axis=1,
    )
    identity = frame["ulog_sha256"].where(frame["ulog_sha256"].notna(), fallback)
    frame["canonical_case_id"] = "rfly_" + identity.str[:20]
    multiplicity = Counter(identity)
    frame["duplicate_group"] = [
        f"dup_{value[:16]}" if multiplicity[value] > 1 else None for value in identity
    ]
    quality = []
    for row in frame.itertuples(index=False):
        flags = []
        if row.ulog_sha256 is None:
            flags.append("missing_ulog_hash")
        if row.system_fault and not np.isfinite(row.fault_start_s):
            flags.append("missing_fault_interval")
        if row.system_fault and row.truth_source == "test_info":
            flags.append("provisional_testinfo_truth")
        if row.duplicate_group:
            flags.append("exact_duplicate")
        quality.append("ok" if not flags else ";".join(flags))
    frame["quality_status"] = quality
    frame = _assign_splits(frame)
    registry_rows = frame[
        ['split_group_id', 'split', 'cv_fold', 'domain', 'fault_family']
    ].drop_duplicates('split_group_id')
    _atomic_json(SPLIT_REGISTRY, {
        'schema_version': 1,
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'groups': {
            str(row.split_group_id): {
                'split': str(row.split), 'cv_fold': int(row.cv_fold),
                'domain': str(row.domain), 'fault_family': str(row.fault_family),
            }
            for row in registry_rows.itertuples(index=False)
        },
    })
    ordered = [
        'platform_metadata', 'fault_id', 'fault_parameter', 'fault_severity',
        'planned_fault_start_s', 'planned_fault_end_s',
        "case_id", "canonical_case_id", "source_archive", "dataset_version",
        "domain", "representation", "platform", "flight_status", "fault_family",
        "fault_subtype", "system_fault", "environment_condition", "evaluation_role",
        "fault_start_s", "fault_end_s", "duration_s", "truth_source", "ulog_sha256",
        "testinfo_sha256", "duplicate_group", "quality_status", "split_group_id",
        "split", "cv_fold", "object_name", "parsed_path", "ulog_bytes",
    ]
    frame = frame[ordered].sort_values(["domain", "fault_family", "case_id"]).reset_index(drop=True)
    V2_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATASET_MANIFEST, index=False)
    frame.to_csv(DATASET_MANIFEST_CSV, index=False)
    unique = frame.drop_duplicates("canonical_case_id")
    summary = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flight_candidates": int(len(frame)),
        "canonical_flights": int(unique["canonical_case_id"].nunique()),
        "exact_duplicate_candidates": int(frame["duplicate_group"].notna().sum()),
        "locked_test_canonical_flights": int(
            unique.loc[unique["split"].eq("locked_test"), "canonical_case_id"].nunique()
        ),
        "quality_status": frame["quality_status"].value_counts().to_dict(),
        "by_domain_family": (
            unique.groupby(["domain", "fault_family"], dropna=False)
            .size().rename("flights").reset_index().to_dict("records")
        ),
        "contract": {
            "test": "locked grouped 20% allocation within domain/family/environment strata",
            "development": "remaining groups with deterministic five-fold assignment",
            "wind": "environment robustness; not a system-fault positive",
            "duplicates": "exact ULog SHA-256 shares canonical_case_id and split",
            "truth_priority_v2": "rfly_ctrl_lxl, then TestInfo fallback; manifest records current parsed truth",
        },
    }
    summary['locked_test_fraction'] = float(unique.split.eq('locked_test').mean())
    summary['split_registry'] = str(SPLIT_REGISTRY.relative_to(ARTIFACT_ROOT.parent.parent)).replace('\\', '/')
    _atomic_json(MANIFEST_SUMMARY, summary)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    frame = build_manifest()
    if args.print_summary:
        print(MANIFEST_SUMMARY.read_text(encoding="utf-8"))
    else:
        print(f"wrote {DATASET_MANIFEST} rows={len(frame)}")


if __name__ == "__main__":
    main()
