"""Build the contextual_physics_v2 fit-expansion manifest.

Locks the exact set of NEW Silver parts (the 3 previously-unused days added in
2026-07-23) that contextual_physics_v2 adds to the fit role, on top of the
untouched Step-5 manifest (fit/calibration/development/rehearsal/validation
roles all stay exactly as contextual_physics_v1 defined them -- see
docs/adsb_contextual_physics_v2_prereg_20260723.md Section 1).

"New" is computed structurally (current Silver parts minus every part already
catalogued by any role in the frozen Step-5 manifest), not by trusting a
hand-typed file list -- if parse_adsblol_historical produced anything other
than exactly the 3 expected days, this script fails loudly instead of silently
using a different fit set than the pre-registration described.

Usage:
    python scripts/adsb_build_fit_expansion_manifest_v2.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

STEP5_MANIFEST = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json")
SILVER_DIR = Path("data/objectstore/silver/adsblol_historical")
OUT_DIR = Path("artifacts/adsb/runs/20260723_step5_v2_fit_expansion")

EXPECTED_DAYS = {
    "v2024.09.01-planes-readsb-prod-0.tar": "2024-09-01",
    "v2025.02.15-planes-readsb-prod-0.tar": "2025-02-15",
    "v2025.06.15-planes-readsb-prod-0-003.tar": "2025-06-15",
}
TAR_DAY_PATTERN = re.compile(r"^v(\d{4})\.(\d{2})\.(\d{2})-")


class FitExpansionManifestError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _already_catalogued_names(manifest: dict[str, Any]) -> set[str]:
    return {Path(record["path"]).name for record in manifest["inputs"]}


def _day_for_source_file(source_file: str) -> str:
    if source_file not in EXPECTED_DAYS:
        match = TAR_DAY_PATTERN.match(source_file)
        if not match:
            raise FitExpansionManifestError(f"Unrecognized source tar name: {source_file!r}")
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return EXPECTED_DAYS[source_file]


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Manifest run dir already exists: {OUT_DIR}")

    manifest = json.loads(STEP5_MANIFEST.read_text(encoding="utf-8"))
    catalogued = _already_catalogued_names(manifest)

    all_parts = sorted(SILVER_DIR.glob("*.parquet"))
    new_parts = [p for p in all_parts if p.name not in catalogued]
    if not new_parts:
        raise FitExpansionManifestError(
            "No new Silver parts found beyond the Step-5 manifest -- did Faz A "
            "(parallel_parse_all.py) finish?"
        )

    by_day: dict[str, list[Path]] = {}
    for part in new_parts:
        source_file = pd.read_parquet(part, columns=["_source_file"])["_source_file"].iloc[0]
        day = _day_for_source_file(source_file)
        by_day.setdefault(day, []).append(part)

    expected_day_set = set(EXPECTED_DAYS.values())
    found_day_set = set(by_day)
    if found_day_set != expected_day_set:
        raise FitExpansionManifestError(
            f"Expected exactly {sorted(expected_day_set)}, found {sorted(found_day_set)} -- "
            "prereg Section 1 must be revised (new dated pre-registration) before proceeding, "
            "not silently trained on an unexpected day set."
        )

    days_payload = []
    for day in sorted(by_day):
        parts = sorted(by_day[day])
        records = []
        for part in parts:
            rel = part.relative_to(Path.cwd()) if part.is_absolute() else part
            records.append(
                {
                    "path": rel.as_posix(),
                    "bytes": part.stat().st_size,
                    "sha256": _sha256_file(part),
                }
            )
        days_payload.append(
            {
                "source_day": day,
                "part_count": len(records),
                "files": records,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    out_manifest = {
        "schema_version": 1,
        "candidate_namespace": "contextual_physics_v2",
        "role": "fit_expansion",
        "fit_flight_sample_probability": 1.0,
        "base_step5_manifest": str(STEP5_MANIFEST).replace("\\", "/"),
        "base_step5_manifest_sha256": _sha256_file(STEP5_MANIFEST),
        "days": days_payload,
        "total_new_parts": len(new_parts),
    }
    out_manifest["fit_expansion_sha256"] = _canonical_json_sha256(out_manifest)
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"days": {d["source_day"]: d["part_count"] for d in days_payload}}, indent=2))


if __name__ == "__main__":
    main()
