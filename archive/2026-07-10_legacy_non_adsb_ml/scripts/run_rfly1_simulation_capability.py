"""RFLY-1 simulation-only capability preflight and runner scaffold.

This track is intentionally separate from Real-* RflyMAD and UAV-SEAD. It may
inspect only SIL-Wind/HIL-Wind simulation cases and writes artifacts under
artifacts/rfly1/simulation_capability/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.silver.parse_rflymad import (  # noqa: E402
    DEFAULT_BRONZE_DIR,
    SIM_SUBSETS,
    discover_cases,
    infer_label_from_case,
)

OUT_DIR = ROOT / "artifacts/rfly1/simulation_capability"
LISTING_CSV = ROOT / "artifacts/rflymad/kaggle_file_listing.csv"
SIMULATION_SUBSETS = tuple(sorted(SIM_SUBSETS))


def _parts(value: str) -> tuple[str, ...]:
    return tuple(Path(value.replace("\\", "/")).parts)


def simulation_family_from_case(case_id: str) -> str:
    parts = _parts(case_id)
    if not parts or parts[0] not in SIM_SUBSETS:
        raise ValueError(f"not an RFLY simulation case: {case_id}")
    if parts[0] == "SIL-Wind" and len(parts) >= 3 and parts[1] == "SIL-Wind":
        return parts[2]
    if len(parts) >= 2:
        return parts[1]
    return "unknown-wind"


def assert_simulation_only_cases(case_ids: list[str] | set[str] | tuple[str, ...]) -> None:
    bad = [case_id for case_id in case_ids if _parts(case_id)[0] not in SIM_SUBSETS]
    if bad:
        raise AssertionError("RFLY-1 simulation track received non-simulation cases: " + ", ".join(sorted(bad)[:5]))


def listing_summary(listing_csv: Path = LISTING_CSV) -> dict:
    if not listing_csv.exists():
        return {"status": "listing_missing", "listing_csv": str(listing_csv)}
    cases: dict[str, dict] = {}
    with listing_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["name"]
            parts = name.split("/")
            if not parts or parts[0] not in SIM_SUBSETS:
                continue
            case_parts = []
            for part in parts:
                case_parts.append(part)
                if part.startswith("TestCase"):
                    break
            if not case_parts or not case_parts[-1].startswith("TestCase"):
                continue
            case_id = "/".join(case_parts)
            entry = cases.setdefault(case_id, {"subset": parts[0], "family": simulation_family_from_case(case_id), "files": 0})
            entry["files"] += 1
    by_subset = Counter(entry["subset"] for entry in cases.values())
    by_family = Counter(entry["subset"] + "/" + entry["family"] for entry in cases.values())
    return {
        "status": "ok",
        "listing_csv": str(listing_csv),
        "simulation_case_count": len(cases),
        "by_subset": dict(sorted(by_subset.items())),
        "by_family": dict(sorted(by_family.items())),
    }


def local_case_summary(bronze_dir: Path = DEFAULT_BRONZE_DIR) -> dict:
    cases = discover_cases(bronze_dir, subsets=SIMULATION_SUBSETS)
    case_ids = sorted({case["source_id"] for case in cases})
    assert_simulation_only_cases(case_ids)
    by_subset = Counter(case["subdataset"] for case in cases)
    by_label = Counter(infer_label_from_case(case_id) for case_id in case_ids)
    return {
        "local_case_count": len(case_ids),
        "local_ulg_count": len(cases),
        "by_subset": dict(sorted(by_subset.items())),
        "by_label": dict(sorted(by_label.items())),
        "cases_present": bool(case_ids),
    }


def write_preflight(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local = local_case_summary()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "RFLY-1 simulation-only capability preflight",
        "status": "ready" if local["cases_present"] else "blocked_no_simulation_data",
        "real_data_mixed": False,
        "simulation_subsets": SIMULATION_SUBSETS,
        "listing": listing_summary(),
        "local": local,
        "next_step": (
            "run simulation-only split/model evaluation"
            if local["cases_present"]
            else "download or provide SIL-Wind/HIL-Wind essential files; Real-* data will not be used as fallback"
        ),
    }
    path = out_dir / "preflight.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="write data-availability and isolation report")
    args = parser.parse_args()
    if args.preflight:
        print(write_preflight())
        return
    report_path = write_preflight()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "ready":
        raise SystemExit(f"simulation capability run blocked: {report['status']} ({report_path})")
    raise SystemExit("full simulation capability evaluation is intentionally not run without an explicit implementation gate")


if __name__ == "__main__":
    main()
