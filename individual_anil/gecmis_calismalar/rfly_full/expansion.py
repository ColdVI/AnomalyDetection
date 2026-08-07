"""Checkpointed ingestion of the separate official RflyMAD SIL/HIL mirrors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from gecmis_calismalar.rfly_full.pipeline import (
    ARTIFACT_ROOT, PARSED_ROOT, REPORT_ROOT, _atomic_json, _load_json,
    _package_priority, _retry, case_id, is_essential, parse_ulg, smoke_evaluate,
)

DATASETS = ("xianglile/rflymad-sil", "xianglile/rflymad-hil")
SKIP_PACKAGES = {"SIL-Wind", "HIL-Wind"}
DEFAULT_RAW_ROOT = Path(r"D:\AnomalyDetectionData\rflymad_expanded")
SOURCE_ROOT = ARTIFACT_ROOT / "expanded_sources"
STATE = ARTIFACT_ROOT / "expansion_state.json"
MANIFEST = ARTIFACT_ROOT / "expansion_manifest.json"
LOG = logging.getLogger("rfly_expansion")


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _slug(dataset: str) -> str:
    return dataset.rsplit("/", 1)[-1]


def build_listing(api, dataset: str) -> Path:
    root = SOURCE_ROOT / _slug(dataset)
    listing, state_path = root / "listing.csv", root / "listing_state.json"
    state = _load_json(state_path, {"token": None, "done": False, "rows": 0})
    if state.get("done") and listing.exists():
        return listing
    known: dict[str, int] = {}
    if listing.exists():
        with listing.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                known[row["name"]] = int(row["bytes"] or 0)
    token = state.get("token")
    while True:
        result = _retry(
            lambda: api.dataset_list_files(dataset, page_token=token, page_size=200),
            f"listing {dataset}",
        )
        files = result.files or []
        for item in files:
            known[str(item.name)] = int(item.total_bytes or 0)
        token = getattr(result, "next_page_token", None)
        root.mkdir(parents=True, exist_ok=True)
        with listing.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("name", "bytes"))
            writer.writerows(sorted(known.items()))
        _atomic_json(state_path, {"token": token, "done": not bool(token), "rows": len(known)})
        LOG.info("listing %s rows=%d remaining=%s", dataset, len(known), bool(token))
        if not token or not files:
            return listing
        time.sleep(0.4)


def load_queue(api) -> dict[str, dict[str, list[tuple[str, str, int]]]]:
    grouped: dict[str, dict[str, list[tuple[str, str, int]]]] = defaultdict(lambda: defaultdict(list))
    metadata: list[tuple[str, str, int]] = []
    for dataset in DATASETS:
        listing = build_listing(api, dataset)
        with listing.open(encoding="utf-8", newline="") as stream:
            rows = [(row["name"], int(row["bytes"] or 0)) for row in csv.DictReader(stream)]
        for name, size in rows:
            package = name.split("/", 1)[0]
            if package in SKIP_PACKAGES:
                continue
            if name.lower().endswith(".ulg"):
                grouped[package][case_id(name)].append((dataset, name, size))
            elif is_essential(name):
                metadata.append((dataset, name, size))
    for dataset, name, size in metadata:
        package = name.split("/", 1)[0]
        parent = Path(name.replace("/", os.sep)).parent.as_posix()
        for flight in grouped.get(package, {}):
            if flight == parent or flight.startswith(parent + "/"):
                grouped[package][flight].append((dataset, name, size))
    return grouped


def _target(raw_root: Path, dataset: str, name: str) -> Path:
    parts = Path(name.replace("/", os.sep))
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError(f"unsafe object path: {name}")
    return raw_root / _slug(dataset) / parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(api, raw_root: Path, dataset: str, name: str, size: int) -> tuple[Path, bool]:
    target = _target(raw_root, dataset, name)
    if target.exists() and (size <= 0 or target.stat().st_size == size):
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    _retry(
        lambda: api.dataset_download_file(dataset, name, path=str(target.parent), quiet=True, force=True),
        f"download {dataset}:{name}",
    )
    zipped = target.parent / f"{target.name}.zip"
    if zipped.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        with zipfile.ZipFile(zipped) as archive:
            members = [member for member in archive.namelist() if not member.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"unexpected archive members: {name}")
            with archive.open(members[0]) as source, temporary.open("wb") as destination:
                while block := source.read(4 * 1024 * 1024):
                    destination.write(block)
        zipped.unlink()
        os.replace(temporary, target)
    if not target.exists() or (size > 0 and target.stat().st_size != size):
        target.unlink(missing_ok=True)
        raise IOError(f"download size mismatch: {dataset}:{name}")
    return target, True


def run(raw_root: Path, deadline: datetime | None, batch_size: int) -> None:
    api = _api()
    queue = load_queue(api)
    state = _load_json(STATE, {"completed_batches": [], "failed_batches": {}, "started_at": datetime.now().astimezone().isoformat()})
    manifest = _load_json(MANIFEST, {"datasets": list(DATASETS), "files": {}})
    for package in sorted(queue, key=_package_priority):
        flights = sorted(queue[package])
        for offset in range(0, len(flights), batch_size):
            batch_number = offset // batch_size
            key = f"{package}/batch_{batch_number:04d}"
            if key in state["completed_batches"]:
                continue
            if deadline and datetime.now().astimezone() >= deadline:
                state.update(stop_reason="deadline", stopped_at=datetime.now().astimezone().isoformat())
                _atomic_json(STATE, state)
                return
            frames, errors = [], []
            for flight in flights[offset : offset + batch_size]:
                items = queue[package][flight]
                paths: dict[str, Path] = {}
                for dataset, name, size in sorted(items):
                    try:
                        path, fetched = download(api, raw_root, dataset, name, size)
                        paths[name] = path
                        manifest["files"][f"{dataset}:{name}"] = {
                            "bytes": path.stat().st_size, "sha256": _sha256(path),
                            "path": str(path), "downloaded_now": fetched,
                        }
                        _atomic_json(MANIFEST, manifest)
                    except Exception as exc:
                        errors.append({"case_id": flight, "file": name, "error": str(exc)})
                        break
                if errors and errors[-1]["case_id"] == flight:
                    continue
                for dataset, name, _ in items:
                    if name.lower().endswith(".ulg"):
                        try:
                            frames.append(parse_ulg(paths[name], name, package))
                        except Exception as exc:
                            errors.append({"case_id": flight, "file": name, "error": f"parse: {exc}"})
            if frames:
                parsed_path = PARSED_ROOT / package / f"batch_{batch_number:04d}.parquet"
                parsed_path.parent.mkdir(parents=True, exist_ok=True)
                frame = pd.concat(frames, ignore_index=True)
                frame.to_parquet(parsed_path, index=False)
                report_path = REPORT_ROOT / package / f"batch_{batch_number:04d}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report = smoke_evaluate(frame, report_path)
                LOG.info("evaluated %s flights=%d status=%s", key, frame.case_id.nunique(), report["status"])
            if errors:
                state["failed_batches"][key] = errors
                LOG.error("batch incomplete %s errors=%d", key, len(errors))
            else:
                state["completed_batches"].append(key)
                state["failed_batches"].pop(key, None)
            state.update(last_attempted=key, updated_at=datetime.now().astimezone().isoformat(), files_manifested=len(manifest["files"]))
            _atomic_json(STATE, state)
    state.update(
        stop_reason="queue_complete" if not state["failed_batches"] else "queue_complete_with_failures",
        stopped_at=datetime.now().astimezone().isoformat(),
    )
    _atomic_json(STATE, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--deadline")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.raw_root, datetime.fromisoformat(args.deadline) if args.deadline else None, args.batch_size)
    from gecmis_calismalar.rfly_full.summary import main as write_combined_summary
    write_combined_summary()


if __name__ == "__main__":
    main()
