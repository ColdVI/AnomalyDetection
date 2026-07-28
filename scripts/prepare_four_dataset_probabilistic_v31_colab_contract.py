"""Build the v3.1 RflyMAD Colab contract ZIP and reuse frozen telemetry ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer"
V1_INDEX = TRANSFER / "transfer_index.json"
V31_INDEX = TRANSFER / "transfer_index_v31.json"
V31_MANIFEST = TRANSFER / "bundle_manifest_v31.json"
V31_CODE = TRANSFER / "four_dataset_probabilistic_v31_code_and_contract.zip"
RFLY_ARCHIVE = "four_dataset_probabilistic_gpu_v1_rflymad_development_folds_0_4.zip"
MEMBERS = (
    Path("configs/four_dataset_probabilistic_v31_rflymad.json"),
    Path("configs/four_dataset_probabilistic_v31_runner_roles.json"),
    Path("configs/four_dataset_probabilistic_v31_split_manifest.json"),
    Path("configs/four_dataset_probabilistic_v31_evaluation_contract.json"),
    Path("docs/FOUR_DATASET_NORMAL_ONLY_V31_PROTOCOL_DECISIONS_20260728.md"),
    Path("scripts/four_dataset_probabilistic_gpu_v1_runner.py"),
    Path("scripts/four_dataset_probabilistic_v31_runner.py"),
    Path("artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet"),
    Path("artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = (V31_CODE, V31_INDEX, V31_MANIFEST)
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError("v3.1 transfer outputs already exist")
    old = json.loads(V1_INDEX.read_text(encoding="utf-8"))
    rfly_records = [record for record in old["archives"] if record["path"] == RFLY_ARCHIVE]
    if len(rfly_records) != 1:
        raise RuntimeError("Frozen RflyMAD telemetry archive missing from v1 transfer")
    rfly_path = TRANSFER / RFLY_ARCHIVE
    if _sha256(rfly_path) != rfly_records[0]["sha256"]:
        raise RuntimeError("Frozen RflyMAD telemetry ZIP hash mismatch")

    member_records = []
    with zipfile.ZipFile(V31_CODE, "w", compression=zipfile.ZIP_STORED) as bundle:
        for relative in MEMBERS:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            bundle.write(source, relative.as_posix())
            member_records.append(
                {"path": relative.as_posix(), "bytes": source.stat().st_size, "sha256": _sha256(source)}
            )
    code_record = {
        "path": V31_CODE.name,
        "bytes": V31_CODE.stat().st_size,
        "sha256": _sha256(V31_CODE),
        "compression": "ZIP_STORED",
        "member_count": len(MEMBERS),
        "source_bytes": sum(row["bytes"] for row in member_records),
    }
    manifest = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "reuse_policy": "frozen v1 RflyMAD telemetry ZIP reused byte-for-byte",
        "final_fault_policy": (
            "telemetry archive contains the frozen corpus, but v3.1 runner roles omit "
            "all final_fault_test sources during development"
        ),
        "code_archive": {**code_record, "members": member_records},
        "reused_data_archive": rfly_records[0],
    }
    _write_json(V31_MANIFEST, manifest)
    index = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "bundle_role": "rflymad_v31_colab_transfer",
        "archives_built": True,
        "archives": [code_record, rfly_records[0]],
        "bundle_manifest": {
            "path": V31_MANIFEST.name,
            "bytes": V31_MANIFEST.stat().st_size,
            "sha256": _sha256(V31_MANIFEST),
        },
    }
    _write_json(V31_INDEX, index)
    print(
        json.dumps(
            {
                "new_uploads": [
                    {"path": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)}
                    for p in (V31_CODE, V31_INDEX, V31_MANIFEST)
                ],
                "reused_upload": rfly_records[0],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
