"""RflyMAD Bronze .ulg files -> PX4-compatible Silver table.

RFLY-0 keeps RflyMAD as a separate source while reusing the established PX4
ULog parser from UAV-SEAD. The first pass is real-flight only: Real-NoFault,
Real-Motor, and Real-Sensors. SampleData is accepted for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd

from src.common.provenance import add_provenance
from src.silver.parse_uav_sead import parse_ulg_bytes as parse_px4_ulg_bytes

logger = logging.getLogger(__name__)

SOURCE_TYPE = "rflymad"
DEFAULT_BRONZE_DIR = Path("data/objectstore/bronze/rflymad")
DEFAULT_LOCAL_OUT = Path("data/silver/rflymad_silver.parquet")
DEFAULT_REPORT_OUT = Path("artifacts/rfly0/rflymad/parse_report.json")
DEFAULT_SUBSETS = ("SampleData", "Real-NoFault", "Real-Motor", "Real-Sensors")
REAL_SUBSETS = {"Real-NoFault", "Real-Motor", "Real-Sensors"}
_SAFE_LABEL = re.compile(r"[^a-z0-9]+")


def _parts(path: str | Path) -> tuple[str, ...]:
    return PurePosixPath(str(path).replace("\\", "/")).parts


def case_root_from_object(path: str | Path) -> str:
    """Return the stable flight/case id for a RflyMAD object path."""
    parts = _parts(path)
    for i, part in enumerate(parts):
        if part.startswith("TestCase"):
            return "/".join(parts[: i + 1])
    for i, part in enumerate(parts):
        if part.startswith("log_"):
            return "/".join(parts[: i + 1])
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return "/".join(parts[:-1] or parts)


def rflymad_session_of(source_id: str) -> str:
    """Session key for RflyMAD split isolation."""
    parts = _parts(source_id)
    for i, part in enumerate(parts):
        if part.startswith("TestCase"):
            return "/".join(parts[: i + 1])
    if parts and parts[0] in REAL_SUBSETS | {"SampleData"} and len(parts) >= 3:
        return "/".join(parts[:3])
    return source_id.split("/")[0] if "/" in source_id else source_id


def _normalise_token(value: str) -> str:
    token = _SAFE_LABEL.sub("_", value.lower()).strip("_")
    return token or "unknown"


def infer_label_from_case(case_id: str) -> str:
    """Map RflyMAD case paths to the fixed RFLY-0 label taxonomy."""
    parts = _parts(case_id)
    joined = "/".join(parts).lower()
    subset = parts[0] if parts else ""
    if subset == "Real-NoFault" or "nofault" in joined or "no-fault" in joined:
        return "normal"
    if subset == "Real-Motor" or "motor" in joined:
        return "motor_fault"
    if subset == "Real-Sensors":
        fault = _normalise_token(parts[1] if len(parts) > 1 else "sensor")
        return f"sensor_{fault}_fault"
    if "sensor" in joined and len(parts) > 1:
        return f"sensor_{_normalise_token(parts[1])}_fault"
    return "unknown"


def _subdataset(case_id: str) -> str:
    parts = _parts(case_id)
    return parts[0] if parts else "unknown"


def _flight_mode(case_id: str) -> str:
    parts = _parts(case_id)
    return parts[1] if len(parts) > 1 else "unknown"


def _test_info_for_case(bronze_dir: Path, case_id: str) -> str | None:
    case_path = bronze_dir / Path(case_id)
    search_root = case_path.parent if case_path.name.startswith("log_") else case_path
    if not search_root.exists():
        return None
    matches = sorted(
        p for p in search_root.iterdir()
        if p.is_file()
        and p.name.startswith("TestInfo")
        and p.suffix.lower() in {".csv", ".xlsx"}
    )
    if not matches:
        return None
    return matches[0].relative_to(bronze_dir).as_posix()


def _manifest_cases(bronze_dir: Path) -> list[dict[str, str]]:
    manifest_path = bronze_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for case_id, entry in sorted(manifest.get("cases", {}).items()):
        for name in sorted(entry.get("files", {})):
            if not name.endswith(".ulg"):
                continue
            rows.append({
                "source_id": case_id,
                "object_name": name,
                "subdataset": entry.get("subdataset") or _subdataset(case_id),
            })
    return rows


def discover_cases(
    bronze_dir: str | Path = DEFAULT_BRONZE_DIR,
    *,
    subsets: Iterable[str] = DEFAULT_SUBSETS,
) -> list[dict[str, str]]:
    """Discover downloaded RflyMAD ULog cases from manifest or filesystem."""
    bronze_dir = Path(bronze_dir)
    allowed = set(subsets)
    rows = _manifest_cases(bronze_dir)
    if not rows:
        rows = [
            {
                "source_id": case_root_from_object(path.relative_to(bronze_dir)),
                "object_name": path.relative_to(bronze_dir).as_posix(),
                "subdataset": _subdataset(path.relative_to(bronze_dir).as_posix()),
            }
            for path in sorted(bronze_dir.rglob("*.ulg"))
        ]
    filtered = []
    for row in rows:
        if row["subdataset"] not in allowed:
            continue
        path = bronze_dir / Path(row["object_name"])
        if not path.exists():
            logger.warning("%s: manifestte var ama dosya yok", row["object_name"])
            continue
        row = dict(row)
        row["label"] = infer_label_from_case(row["source_id"])
        row["flight_mode"] = _flight_mode(row["source_id"])
        row["test_info"] = _test_info_for_case(bronze_dir, row["source_id"])
        filtered.append(row)
    return filtered


def build_rflymad_silver_from_directory(
    bronze_dir: str | Path = DEFAULT_BRONZE_DIR,
    *,
    subsets: Iterable[str] = DEFAULT_SUBSETS,
    limit_cases: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse downloaded RflyMAD cases from a local Bronze tree."""
    bronze_dir = Path(bronze_dir)
    cases = discover_cases(bronze_dir, subsets=subsets)
    if limit_cases is not None:
        cases = cases[:limit_cases]
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for i, case in enumerate(cases, 1):
        object_name = case["object_name"]
        path = bronze_dir / Path(object_name)
        frame = parse_px4_ulg_bytes(
            path.read_bytes(),
            source_id=case["source_id"],
            label=case["label"],
        )
        if frame is None or frame.empty:
            skipped.append({"source_id": case["source_id"], "object_name": object_name})
            continue
        frame = frame.copy()
        frame["source_type"] = SOURCE_TYPE
        frame["source_id"] = case["source_id"]
        frame["label"] = case["label"]
        frame["rflymad_subdataset"] = case["subdataset"]
        frame["rflymad_flight_mode"] = case["flight_mode"]
        frame["rflymad_test_info"] = case.get("test_info")
        frames.append(frame)
        logger.info("[%d/%d] %s: %d satir label=%s",
                    i, len(cases), case["source_id"], len(frame), case["label"])
    report = {
        "source": SOURCE_TYPE,
        "bronze_dir": str(bronze_dir).replace("\\", "/"),
        "candidate_cases": len(cases),
        "parsed_cases": len(frames),
        "skipped_cases": skipped,
        "blind_holdout_read": False,
        "subsets": list(subsets),
    }
    if not frames:
        return pd.DataFrame(), report
    full = pd.concat(frames, ignore_index=True, sort=False)
    return add_provenance(full, source_type=SOURCE_TYPE, source_file="rflymad/*.ulg"), report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="RflyMAD Bronze .ulg -> Silver")
    parser.add_argument("--local-bronze-dir", default=str(DEFAULT_BRONZE_DIR))
    parser.add_argument("--local-out", default=str(DEFAULT_LOCAL_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subsets = tuple(s.strip() for s in args.subsets.split(",") if s.strip())
    if args.dry_run:
        cases = discover_cases(args.local_bronze_dir, subsets=subsets)
        if args.limit_cases is not None:
            cases = cases[:args.limit_cases]
        print(json.dumps({"candidate_cases": len(cases), "cases": cases[:10]}, indent=2))
        return

    silver, report = build_rflymad_silver_from_directory(
        args.local_bronze_dir,
        subsets=subsets,
        limit_cases=args.limit_cases,
    )
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if silver.empty:
        logger.error("Nothing to write: RflyMAD Silver is empty")
        return
    out = Path(args.local_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out, index=False)
    logger.info("Local copy written: %s", out)


if __name__ == "__main__":
    main()
