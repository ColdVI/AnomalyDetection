"""Checkpointed RflyMAD download, 1 Hz parsing and lightweight evaluation.

This namespace is intentionally independent from archived RFLY tracks. Raw
data stays outside git; compact resumable state is written under artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

DATASET = "xianglile/rflymad"
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/rfly_full"
LISTING_CSV = ARTIFACT_ROOT / "kaggle_file_listing.csv"
LISTING_STATE = ARTIFACT_ROOT / "kaggle_listing_state.json"
RUN_STATE = ARTIFACT_ROOT / "overnight_state.json"
DEFAULT_RAW_ROOT = Path(r"D:\AnomalyDetectionData\rflymad")
LEGACY_RAW_ROOT = ROOT / "data/objectstore/bronze/rflymad"
PARSED_ROOT = ARTIFACT_ROOT / "parsed_batches"
REPORT_ROOT = ARTIFACT_ROOT / "batch_reports"
MANIFEST = ARTIFACT_ROOT / "download_manifest.json"

ESSENTIAL = (".ulg", "testinfo.csv", "testinfo.xlsx")
BATCH_SIZE = 16
RETRY_S = (15, 30, 60, 120, 300)
IDLE_SENTINEL = 1500
FEATURES = (
    "local_x", "local_y", "local_z", "local_vx", "local_vy", "local_vz",
    "local_ax", "local_ay", "local_az", "roll_deg", "pitch_deg", "yaw_deg",
    "act_roll", "act_pitch", "act_yaw", "act_thrust", "output_mean",
    "output_std", "output_range", "battery_voltage", "battery_current",
    "battery_remaining", "vel_test_ratio", "pos_test_ratio", "hgt_test_ratio",
    "mag_test_ratio", "gps_eph", "gps_epv", "gps_satellites",
)
LOG = logging.getLogger("rfly_full")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            # Windows may briefly hold the destination while a status reader
            # closes it. The temporary file is complete, so retrying replace
            # preserves the atomic checkpoint contract.
            time.sleep(0.05 * (attempt + 1))


def _load_json(path: Path, default: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _retry(call, description: str):
    error = None
    for delay in (0, *RETRY_S):
        if delay:
            LOG.warning("retry %s after %ss", description, delay)
            time.sleep(delay)
        try:
            return call()
        except Exception as exc:
            error = exc
            LOG.warning("%s failed: %s", description, exc)
    raise RuntimeError(f"retry budget exhausted: {description}") from error


def build_listing() -> Path:
    """Resume the paginated Kaggle listing, checkpointing every page."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    state = _load_json(LISTING_STATE, {"token": None, "done": False, "rows": 0})
    if state.get("done") and LISTING_CSV.exists():
        return LISTING_CSV
    known: dict[str, int] = {}
    if LISTING_CSV.exists():
        with LISTING_CSV.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                known[row["name"]] = int(row["bytes"] or 0)
    api = _api()
    token = state.get("token")
    while True:
        result = _retry(
            lambda: api.dataset_list_files(DATASET, page_token=token, page_size=200),
            "dataset listing page",
        )
        files = result.files or []
        for item in files:
            known[str(item.name)] = int(item.total_bytes or 0)
        token = getattr(result, "next_page_token", None)
        with LISTING_CSV.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("name", "bytes"))
            writer.writerows(sorted(known.items()))
        state = {"token": token, "done": not bool(token), "rows": len(known)}
        _atomic_json(LISTING_STATE, state)
        LOG.info("listing checkpoint rows=%d remaining=%s", len(known), bool(token))
        if not token or not files:
            break
        time.sleep(0.25)
    return LISTING_CSV


def load_listing() -> list[tuple[str, int]]:
    build_listing()
    with LISTING_CSV.open(encoding="utf-8", newline="") as stream:
        return [(row["name"], int(row["bytes"] or 0)) for row in csv.DictReader(stream)]


def case_id(name: str) -> str:
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        if part.startswith("TestCase") or part.startswith("log_"):
            return "/".join(parts[: index + 1])
    return "/".join(parts[:-1])


def is_essential(name: str) -> bool:
    path = PurePosixPath(name)
    lowered = path.name.lower()
    return path.suffix.lower() == ".ulg" or (
        lowered.startswith("testinfo") and path.suffix.lower() in {".csv", ".xlsx"}
    )


def _safe_target(root: Path, name: str) -> Path:
    relative = Path(*PurePosixPath(name).parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe Kaggle object path: {name}")
    return root / relative


def _existing_path(raw_root: Path, name: str, expected_size: int) -> Path | None:
    for root in (raw_root, LEGACY_RAW_ROOT):
        path = _safe_target(root, name)
        if path.exists() and (expected_size <= 0 or path.stat().st_size == expected_size):
            return path
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(api, raw_root: Path, name: str, expected_size: int) -> tuple[Path, bool]:
    existing = _existing_path(raw_root, name, expected_size)
    if existing is not None:
        return existing, False
    target = _safe_target(raw_root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    _retry(
        lambda: api.dataset_download_file(
            DATASET, name, path=str(target.parent), quiet=True, force=True
        ),
        f"download {name}",
    )
    zipped = target.parent / f"{target.name}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as archive:
            members = [member for member in archive.namelist() if not member.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"unexpected single-file archive: {name}")
            with archive.open(members[0]) as source, partial.open("wb") as destination:
                while block := source.read(4 * 1024 * 1024):
                    destination.write(block)
        zipped.unlink()
        os.replace(partial, target)
    if not target.exists():
        raise FileNotFoundError(f"Kaggle returned no target for {name}")
    if expected_size > 0 and target.stat().st_size != expected_size:
        target.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {name}: expected={expected_size}")
    return target, True


def _topic(ulog, name: str, fields: tuple[str, ...], prefix: str = "") -> pd.DataFrame | None:
    try:
        data = ulog.get_dataset(name).data
    except (KeyError, IndexError, StopIteration):
        return None
    if "timestamp" not in data:
        return None
    columns = {"timestamp": np.asarray(data["timestamp"], dtype=np.int64)}
    for field in fields:
        if field in data:
            columns[prefix + field] = np.asarray(data[field])
    return pd.DataFrame(columns).sort_values("timestamp") if len(columns) > 1 else None


def _merge(base: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    if extra is None:
        return base
    return pd.merge_asof(base, extra, on="timestamp", direction="nearest", tolerance=600_000)


def _euler(frame: pd.DataFrame) -> pd.DataFrame:
    q0, q1, q2, q3 = (frame[f"q[{i}]"] for i in range(4))
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return pd.DataFrame({"timestamp": frame["timestamp"], "roll_deg": np.degrees(roll),
                         "pitch_deg": np.degrees(pitch), "yaw_deg": np.degrees(yaw)})


def _test_info_interval(ulg_path: Path) -> tuple[float, float] | None:
    """Read the published injection/test-end interval beside a case ULog."""
    case_root = ulg_path.parent.parent
    matches = sorted(case_root.glob("TestInfo*"))
    if not matches:
        return None
    path = matches[0]
    try:
        table = pd.read_excel(path, header=None) if path.suffix.lower() == ".xlsx" else pd.read_csv(path, header=None)
    except Exception:
        return None
    found: dict[str, float] = {}
    for row in table.fillna("").astype(str).to_numpy():
        if len(row) < 2:
            continue
        key = "".join(character for character in row[0].lower() if character.isalnum())
        if key in {"faultinjectiontime", "testendtime"}:
            try:
                found[key] = float(str(row[1]).strip())
            except ValueError:
                pass
    onset, end = found.get("faultinjectiontime"), found.get("testendtime")
    if onset is None or end is None or not (0 <= onset <= end):
        return None
    return onset, end


def parse_ulg(path: Path, object_name: str, package: str) -> pd.DataFrame:
    """Parse deployment-parity ULog fields onto causal 1-second endpoints."""
    from pyulog import ULog
    ulog = ULog(str(path))
    local = _topic(ulog, "vehicle_local_position", ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"), "local_")
    if local is None or len(local) < 2:
        raise ValueError("vehicle_local_position unavailable")
    start, end = int(local.timestamp.min()), int(local.timestamp.max())
    timestamps = np.arange(math.ceil(start / 1e6) * 1_000_000, end + 1, 1_000_000, dtype=np.int64)
    base = _merge(pd.DataFrame({"timestamp": timestamps}), local)
    attitude = _topic(ulog, "vehicle_attitude", tuple(f"q[{i}]" for i in range(4)))
    if attitude is not None and all(f"q[{i}]" in attitude for i in range(4)):
        base = _merge(base, _euler(attitude))
    controls = _topic(ulog, "actuator_controls_0", tuple(f"control[{i}]" for i in range(4)))
    if controls is not None:
        controls = controls.rename(columns={f"control[{i}]": name for i, name in enumerate(("act_roll", "act_pitch", "act_yaw", "act_thrust"))})
    base = _merge(base, controls)
    outputs = _topic(ulog, "actuator_outputs", tuple(f"output[{i}]" for i in range(16)))
    if outputs is not None:
        cols = [column for column in outputs if column.startswith("output[")]
        values = outputs[cols].replace(0, np.nan)
        outputs = outputs[["timestamp"]].assign(output_mean=values.mean(axis=1), output_std=values.std(axis=1), output_range=values.max(axis=1) - values.min(axis=1))
    base = _merge(base, outputs)
    base = _merge(base, _topic(ulog, "battery_status", ("voltage_v", "current_a", "remaining"), "battery_"))
    base = _merge(base, _topic(ulog, "estimator_status", ("vel_test_ratio", "pos_test_ratio", "hgt_test_ratio", "mag_test_ratio")))
    base = _merge(base, _topic(ulog, "vehicle_gps_position", ("eph", "epv", "satellites_used"), "gps_"))
    token = package.lower().replace("_", "")
    normal = "nofault" in token or "no-fault" in token
    base["t_rel_s"] = (base["timestamp"] - base["timestamp"].iloc[0]) / 1e6
    is_simulation = package.startswith(("SIL", "HIL"))
    published_interval = _test_info_interval(path) if is_simulation else None
    if normal:
        active = np.zeros(len(base), dtype=bool)
        truth_source = "normal_no_fault"
    elif published_interval is not None:
        onset, end = published_interval
        active = base["t_rel_s"].between(onset, end, inclusive="both").to_numpy()
        truth_source = "test_info"
    else:
        fault = _topic(ulog, "rfly_ctrl_lxl", ("id", "mode"), "ctrl_")
        if fault is None or "ctrl_id" not in fault or "ctrl_mode" not in fault:
            active = np.zeros(len(base), dtype=bool)
            truth_source = "missing"
        else:
            aligned = _merge(base[["timestamp"]], fault)
            sentinel = 0 if is_simulation else IDLE_SENTINEL
            active = ((aligned["ctrl_id"] != sentinel) | (aligned["ctrl_mode"] != sentinel)).fillna(False).to_numpy()
            truth_source = "rfly_ctrl_lxl"
    base["case_id"] = case_id(object_name)
    base["object_name"] = object_name
    base["package"] = package
    base["label"] = "normal" if normal else package
    base["fault_active"] = False if normal else active
    base["truth_source"] = truth_source
    for feature in FEATURES:
        if feature not in base:
            base[feature] = np.nan
    return base[["timestamp", "t_rel_s", "case_id", "object_name", "package", "label", "fault_active", "truth_source", *FEATURES]]


def _k_of_n(binary: np.ndarray, k: int = 4, n: int = 6) -> np.ndarray:
    counts = np.convolve(binary.astype(np.int8), np.ones(n, dtype=np.int8), mode="full")[: len(binary)]
    state = counts >= k
    return state & ~np.r_[False, state[:-1]]


def smoke_evaluate(current: pd.DataFrame, output: Path) -> dict:
    """Flight-isolated normal calibration; preliminary, never a final claim."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score, roc_auc_score
    parsed_files = sorted(PARSED_ROOT.glob("**/*.parquet"))
    frames = [pd.read_parquet(path) for path in parsed_files]
    pool = pd.concat(frames, ignore_index=True) if frames else current
    evaluation_ids = set(current["case_id"].unique())
    normal_ids = sorted(
        set(pool.loc[pool.label.eq("normal"), "case_id"].unique())
        - evaluation_ids
    )
    report: dict = {"status": "skipped", "reason": "need >=6 normal flights", "normal_flights": len(normal_ids)}
    if len(normal_ids) < 6:
        _atomic_json(output, report)
        return report
    cut_train = max(3, int(len(normal_ids) * 0.6))
    cut_val = max(cut_train + 1, int(len(normal_ids) * 0.8))
    train_ids, val_ids = set(normal_ids[:cut_train]), set(normal_ids[cut_train:cut_val])
    train, val = pool[pool.case_id.isin(train_ids)], pool[pool.case_id.isin(val_ids)]
    coverage = train[list(FEATURES)].notna().mean()
    columns = [column for column in FEATURES if coverage[column] >= 0.5]
    if len(columns) < 4:
        report["reason"] = "fewer than four usable common features"
        _atomic_json(output, report)
        return report
    medians = train[columns].median()
    scale = (train[columns].quantile(0.75) - train[columns].quantile(0.25)).replace(0, 1).fillna(1)
    def transform(frame):
        return ((frame[columns].fillna(medians) - medians) / scale).clip(-10, 10)
    model = IsolationForest(n_estimators=80, max_samples=min(2048, len(train)), contamination="auto", random_state=20260720, n_jobs=1)
    model.fit(transform(train))
    threshold = float(np.quantile(-model.score_samples(transform(val)), 0.995))
    evaluation = current.copy()
    evaluation["score"] = -model.score_samples(transform(evaluation))
    rows = []
    tp = fn = fp = tn = false_events = 0
    normal_hours = 0.0
    for flight_id, flight in evaluation.groupby("case_id", sort=False):
        alarm = _k_of_n(flight.score.to_numpy() > threshold)
        truth = flight.fault_active.to_numpy(dtype=bool)
        hit = bool((alarm & truth).any())
        is_fault = flight.label.iloc[0] != "normal"
        false = int((alarm & ~truth).sum())
        false_events += false
        normal_hours += float((~truth).sum() / 3600)
        if is_fault:
            tp += int(hit); fn += int(not hit)
        else:
            predicted = bool(alarm.any()); fp += int(predicted); tn += int(not predicted)
        rows.append({"case_id": flight_id, "label": flight.label.iloc[0], "detected": hit, "false_alarm_events": false})
    truth = evaluation.fault_active.to_numpy(dtype=bool)
    scores = evaluation.score.to_numpy()
    report = {
        "status": "ok", "scope": "preliminary_batch_smoke", "model": "isolation_forest_80",
        "decision": "4_of_6_at_1hz", "threshold_source": "normal_validation_q99.5",
        "features": columns, "train_normal_flights": len(train_ids), "validation_normal_flights": len(val_ids),
        "evaluation_flights": int(evaluation.case_id.nunique()), "threshold": threshold,
        "row_auroc": float(roc_auc_score(truth, scores)) if truth.any() and (~truth).any() else None,
        "row_auprc": float(average_precision_score(truth, scores)) if truth.any() and (~truth).any() else None,
        "event_recall": tp / (tp + fn) if tp + fn else None,
        "false_alarms_per_hour": false_events / normal_hours if normal_hours else None,
        "confusion_flight": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "warning": "Smoke sonucu; sabit blind holdout veya nihai fizibilite kaniti degildir.",
        "flights": rows,
    }
    _atomic_json(output, report)
    return report


def _package_priority(name: str) -> tuple[int, str]:
    token = name.lower().replace("_", "").replace("-", "")
    order = ("nofault", "motor", "sensors", "prop", "voltage", "load", "wind")
    rank = next((index for index, item in enumerate(order) if item in token), len(order))
    domain = 0 if name.startswith("SIL") else 1 if name.startswith("HIL") else 2
    return rank * 3 + domain, name


def _queue(rows: list[tuple[str, int]]) -> dict[str, dict[str, list[tuple[str, int]]]]:
    grouped: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    info_files: list[tuple[str, int]] = []
    for name, size in rows:
        if name.lower().endswith(".ulg"):
            grouped[name.split("/", 1)[0]][case_id(name)].append((name, size))
        elif is_essential(name):
            info_files.append((name, size))
    # Real TestInfo is above log_*; simulation TestInfo is in TestCase_*.
    # Attach metadata only to ULog cases to avoid metadata-only pseudo-flights.
    for name, size in info_files:
        package = name.split("/", 1)[0]
        parent = PurePosixPath(name).parent.as_posix()
        for flight in grouped.get(package, {}):
            if flight == parent or flight.startswith(parent + "/"):
                grouped[package][flight].append((name, size))
    return grouped


def _bootstrap_existing_normal(limit: int = 12) -> None:
    if any(PARSED_ROOT.glob("bootstrap_normal/*.parquet")):
        return
    frames = []
    # Tail selection keeps the bootstrap disjoint from the first sorted batch.
    for path in sorted(LEGACY_RAW_ROOT.glob("Real-No_Fault/**/*.ulg"))[-limit:]:
        name = path.relative_to(LEGACY_RAW_ROOT).as_posix()
        try:
            frames.append(parse_ulg(path, name, "Real-No_Fault"))
        except Exception as exc:
            LOG.warning("bootstrap parse failed %s: %s", name, exc)
    if frames:
        target = PARSED_ROOT / "bootstrap_normal" / "batch_0000.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
        LOG.info("bootstrapped %d existing normal flights", len(frames))


def run(raw_root: Path, deadline: datetime | None, batch_size: int = BATCH_SIZE, max_batches: int | None = None) -> None:
    queue = _queue(load_listing())
    state = _load_json(RUN_STATE, {"completed_batches": [], "failed_cases": {}, "started_at": datetime.now().astimezone().isoformat()})
    manifest = _load_json(MANIFEST, {"dataset": DATASET, "files": {}})
    api = _api()
    _bootstrap_existing_normal()
    completed_now = 0
    for package in sorted(queue, key=_package_priority):
        cases = sorted(queue[package])
        for offset in range(0, len(cases), batch_size):
            batch_index = offset // batch_size
            key = f"{package}/batch_{batch_index:04d}"
            if key in state["completed_batches"]:
                continue
            if deadline and datetime.now().astimezone() >= deadline:
                state.update(stopped_at=datetime.now().astimezone().isoformat(), stop_reason="deadline")
                _atomic_json(RUN_STATE, state)
                return
            parsed = []
            for case in cases[offset : offset + batch_size]:
                items = queue[package][case]
                failed = False
                for name, size in sorted(items):
                    try:
                        path, downloaded = download_file(api, raw_root, name, size)
                        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": _sha256(path), "path": str(path), "downloaded_now": downloaded}
                        _atomic_json(MANIFEST, manifest)
                    except Exception as exc:
                        state["failed_cases"][case] = f"download: {exc}"
                        _atomic_json(RUN_STATE, state)
                        LOG.exception("case download failed %s", case)
                        failed = True
                        break
                if failed:
                    continue
                for name, size in items:
                    if not name.lower().endswith(".ulg"):
                        continue
                    path = _existing_path(raw_root, name, size)
                    if path is not None:
                        try:
                            parsed.append(parse_ulg(path, name, package))
                        except Exception as exc:
                            state["failed_cases"][case] = f"parse: {exc}"
                            LOG.warning("parse failed %s: %s", case, exc)
            if parsed:
                parsed_path = PARSED_ROOT / package / f"batch_{batch_index:04d}.parquet"
                parsed_path.parent.mkdir(parents=True, exist_ok=True)
                frame = pd.concat(parsed, ignore_index=True)
                frame.to_parquet(parsed_path, index=False)
                report_path = REPORT_ROOT / package / f"batch_{batch_index:04d}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report = smoke_evaluate(frame, report_path)
                LOG.info("batch evaluated %s flights=%d status=%s", key, frame.case_id.nunique(), report["status"])
            else:
                LOG.warning("batch has no parsed flights: %s", key)
            state["completed_batches"].append(key)
            state.update(last_completed=key, updated_at=datetime.now().astimezone().isoformat(), downloaded_files=len(manifest["files"]))
            _atomic_json(RUN_STATE, state)
            completed_now += 1
            if max_batches is not None and completed_now >= max_batches:
                return
    state.update(stopped_at=datetime.now().astimezone().isoformat(), stop_reason="queue_complete")
    _atomic_json(RUN_STATE, state)


def describe_listing() -> dict:
    rows = load_listing()
    queue = _queue(rows)
    return {"files": len(rows), "bytes_total": sum(size for _, size in rows),
            "packages": {package: {"cases": len(cases), "essential_bytes": sum(size for files in cases.values() for _, size in files)} for package, cases in sorted(queue.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--deadline", help="local ISO-8601 datetime")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.list_only:
        print(json.dumps(describe_listing(), indent=2))
    else:
        run(args.raw_root, datetime.fromisoformat(args.deadline) if args.deadline else None, args.batch_size, args.max_batches)


if __name__ == "__main__":
    main()
