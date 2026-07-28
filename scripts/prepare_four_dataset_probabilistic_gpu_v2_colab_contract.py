"""Build the small v2 Colab contract ZIP while reusing the existing data ZIPs."""

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
V2_INDEX = TRANSFER / "transfer_index_v2.json"
V2_MANIFEST = TRANSFER / "bundle_manifest_v2.json"
V2_CODE = TRANSFER / "four_dataset_probabilistic_gpu_v2_code_and_contract.zip"
MEMBERS = (
    Path("configs/four_dataset_probabilistic_gpu_v2.json"),
    Path("configs/four_dataset_probabilistic_gpu_v2_split_manifest.json"),
    Path("docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_PREREG_20260727.md"),
    Path("scripts/four_dataset_probabilistic_gpu_v1_runner.py"),
    Path("scripts/four_dataset_probabilistic_gpu_v2_runner.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = (V2_CODE, V2_INDEX, V2_MANIFEST)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(existing)
    old = json.loads(V1_INDEX.read_text(encoding="utf-8"))
    data_records = [
        record
        for record in old["archives"]
        if "code_and_contract" not in str(record["path"])
    ]
    member_records = []
    with zipfile.ZipFile(V2_CODE, "w", compression=zipfile.ZIP_STORED) as bundle:
        for relative in MEMBERS:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            bundle.write(source, relative.as_posix())
            member_records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
    code_record = {
        "bytes": V2_CODE.stat().st_size,
        "compression": "ZIP_STORED",
        "member_count": len(MEMBERS),
        "path": V2_CODE.name,
        "sha256": _sha256(V2_CODE),
        "source_bytes": sum(record["bytes"] for record in member_records),
    }
    manifest = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_gpu_v2",
        "reuse_policy": "v1 dataset ZIP bytes are reused unchanged; only contract ZIP is new",
        "code_archive": {**code_record, "members": member_records},
        "reused_data_archives": data_records,
    }
    _write_json(V2_MANIFEST, manifest)
    index = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_gpu_v2",
        "bundle_role": "four_dataset_probabilistic_gpu_v2_colab_transfer",
        "archives_built": True,
        "archives": [code_record, *data_records],
        "bundle_manifest": {
            "path": V2_MANIFEST.name,
            "bytes": V2_MANIFEST.stat().st_size,
            "sha256": _sha256(V2_MANIFEST),
        },
    }
    _write_json(V2_INDEX, index)
    print(
        json.dumps(
            {
                "new_uploads": [
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in (V2_CODE, V2_INDEX, V2_MANIFEST)
                ],
                "reused_data_archives": [record["path"] for record in data_records],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
