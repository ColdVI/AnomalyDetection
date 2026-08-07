"""Build deterministic Colab transfer ZIPs for the four-dataset GPU pilot.

The transfer is deliberately split into five archives:

* one common code/contract archive;
* one Gold parquet archive for each of ALFA, UAV-Attack, and UAV-SEAD; and
* one RflyMAD archive containing exactly the preregistered development roles
  (normal train folds 2--4, normal validation fold 0, all test fold 1), plus
  the frozen dataset manifest and split registry.

Every source member is size/hash locked in ``bundle_manifest.json``.  ZIP
members are stored without recompression, ordered lexicographically, and use
fixed metadata so identical inputs produce byte-identical archives.  Archive
sizes and SHA-256 digests are recorded in both ``bundle_manifest.json`` and
``transfer_index.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_NAMESPACE = "four_dataset_probabilistic_gpu_v1"
BUNDLE_ROLE = "four_dataset_probabilistic_gpu_v1_colab_transfer"

CONFIG = Path("configs/four_dataset_probabilistic_gpu_v1.json")
RUNNER = Path("scripts/four_dataset_probabilistic_gpu_v1_runner.py")
PREREG = Path("docs/FOUR_DATASET_PROBABILISTIC_GPU_V1_PREREG_20260727.md")
SPLIT_MANIFEST = Path("data/gold/ml_features/split_manifest.json")

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer"
)

COMMON_ARCHIVE = "four_dataset_probabilistic_gpu_v1_code_and_contract.zip"
GOLD_ARCHIVES = {
    "alfa": "four_dataset_probabilistic_gpu_v1_alfa_gold.zip",
    "uav_attack": "four_dataset_probabilistic_gpu_v1_uav_attack_gold.zip",
    "uav_sead": "four_dataset_probabilistic_gpu_v1_uav_sead_gold.zip",
}
RFLY_ARCHIVE = (
    "four_dataset_probabilistic_gpu_v1_rflymad_development_folds_0_4.zip"
)

EXPECTED_GOLD_PATHS = {
    "alfa": Path("data/gold/ml_features/alfa/alfa_ml_features.parquet"),
    "uav_attack": Path(
        "data/gold/ml_features/uav_attack/uav_attack_ml_features.parquet"
    ),
    "uav_sead": Path(
        "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
    ),
}
EXPECTED_RFLY_DATA_ROOT = Path("artifacts/rfly_full/v2/parsed_10hz")
EXPECTED_RFLY_MANIFEST = Path("artifacts/rfly_full/v2/dataset_manifest.parquet")
EXPECTED_RFLY_SPLIT_REGISTRY = Path("artifacts/rfly_full/v2/split_registry.json")
EXPECTED_RFLY_FOLDS = frozenset(range(5))

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
HASH_BLOCK_BYTES = 4 * 1024 * 1024


class ColabBundleError(RuntimeError):
    """Raised when a source or transfer invariant is violated."""


@dataclass(frozen=True)
class BundleMember:
    """One source file and its deterministic path inside an archive."""

    source: Path
    archive_path: str
    bytes: int
    sha256: str

    def manifest_record(self) -> dict[str, Any]:
        return {
            "path": self.archive_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArchivePlan:
    """A fully hash-locked archive plan."""

    path: str
    kind: str
    dataset: str | None
    members: tuple[BundleMember, ...]
    selection: Mapping[str, Any] | None = None

    @property
    def source_bytes(self) -> int:
        return sum(member.bytes for member in self.members)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_stable_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    if not path.is_file():
        raise ColabBundleError(f"Expected a regular source file: {path}")
    digest = _sha256_file(path)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ColabBundleError(f"Source changed while hashing: {path}")
    return after.st_size, digest


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ColabBundleError(f"Expected a JSON object: {path}")
    return value


def _normal_relative(value: str | Path) -> str:
    raw = str(value).replace("\\", "/")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ColabBundleError(f"Unsafe relative path: {value}")
    normalized = relative.as_posix()
    if normalized in {".", ""}:
        raise ColabBundleError(f"Unsafe relative path: {value}")
    return normalized


def _safe_source(root: Path, relative: str | Path) -> tuple[Path, str]:
    normalized = _normal_relative(relative)
    source = (root / Path(*PurePosixPath(normalized).parts)).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ColabBundleError(f"Source escapes repository root: {normalized}") from exc
    if not source.is_file():
        raise ColabBundleError(f"Expected a source file: {normalized}")
    return source, normalized


def _member_from_relative(root: Path, relative: str | Path) -> BundleMember:
    source, normalized = _safe_source(root, relative)
    size, digest = _hash_stable_file(source)
    return BundleMember(
        source=source,
        archive_path=normalized,
        bytes=size,
        sha256=digest,
    )


def _validate_unique_members(members: Sequence[BundleMember], archive_name: str) -> None:
    paths = [member.archive_path for member in members]
    if len(paths) != len(set(paths)):
        raise ColabBundleError(f"Duplicate member path in {archive_name}")
    if paths != sorted(paths):
        raise ColabBundleError(f"Members are not lexicographically ordered: {archive_name}")


def _validate_contract(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path, _ = _safe_source(root, CONFIG)
    split_path, _ = _safe_source(root, SPLIT_MANIFEST)
    config = _load_json_object(config_path)
    split_manifest = _load_json_object(split_path)

    if config.get("candidate_namespace") != CANDIDATE_NAMESPACE:
        raise ColabBundleError("Unexpected candidate namespace")
    if _normal_relative(config.get("split_manifest", "")) != SPLIT_MANIFEST.as_posix():
        raise ColabBundleError("The configured frozen split manifest changed")
    if config.get("split_name") != "split_00":
        raise ColabBundleError("Only the preregistered split_00 may be bundled")
    prohibitions = config.get("prohibitions")
    if not isinstance(prohibitions, dict) or not prohibitions:
        raise ColabBundleError("Missing experiment prohibitions")
    if any(value is not True for value in prohibitions.values()):
        raise ColabBundleError("Every experiment prohibition must remain true")

    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ColabBundleError("Missing dataset contracts")
    for dataset, expected_path in EXPECTED_GOLD_PATHS.items():
        spec = datasets.get(dataset)
        if not isinstance(spec, dict):
            raise ColabBundleError(f"Missing dataset contract: {dataset}")
        observed_path = _normal_relative(spec.get("data_path", ""))
        if observed_path != expected_path.as_posix():
            raise ColabBundleError(f"Unexpected Gold parquet path for {dataset}")

        try:
            split = split_manifest["sources"][dataset]["splits"]["split_00"]
        except (KeyError, TypeError) as exc:
            raise ColabBundleError(f"Missing frozen split_00 for {dataset}") from exc
        if not isinstance(split, dict):
            raise ColabBundleError(f"Malformed frozen split_00 for {dataset}")
        role_sets: dict[str, set[str]] = {}
        for role in ("train", "val", "test"):
            values = split.get(role)
            if not isinstance(values, list):
                raise ColabBundleError(f"Missing {dataset} split role: {role}")
            role_sets[role] = {str(value) for value in values}
        if role_sets["train"] & role_sets["val"]:
            raise ColabBundleError(f"Train/validation overlap in {dataset}")
        if (role_sets["train"] | role_sets["val"]) & role_sets["test"]:
            raise ColabBundleError(f"Development/test overlap in {dataset}")

    rfly = datasets.get("rflymad")
    if not isinstance(rfly, dict):
        raise ColabBundleError("Missing RflyMAD dataset contract")
    expected_rfly_values = {
        "data_path": EXPECTED_RFLY_DATA_ROOT.as_posix(),
        "manifest_path": EXPECTED_RFLY_MANIFEST.as_posix(),
        "split_registry_path": EXPECTED_RFLY_SPLIT_REGISTRY.as_posix(),
    }
    for key, expected in expected_rfly_values.items():
        if _normal_relative(rfly.get(key, "")) != expected:
            raise ColabBundleError(f"Unexpected RflyMAD {key}")
    if rfly.get("data_mode") != "rfly_full_v2_parsed":
        raise ColabBundleError("Unexpected RflyMAD data mode")

    train_folds = {int(value) for value in rfly.get("train_folds", [])}
    validation_fold = int(rfly.get("validation_fold", -1))
    test_fold = int(rfly.get("test_fold", -1))
    if train_folds & {validation_fold, test_fold} or validation_fold == test_fold:
        raise ColabBundleError("RflyMAD train/validation/test folds overlap")
    declared_folds = train_folds | {validation_fold, test_fold}
    if declared_folds != EXPECTED_RFLY_FOLDS:
        raise ColabBundleError(
            f"RflyMAD contract must cover folds 0--4, observed {sorted(declared_folds)}"
        )

    return config, split_manifest, rfly


def _common_plan(root: Path) -> ArchivePlan:
    members = tuple(
        sorted(
            (
                _member_from_relative(root, relative)
                for relative in (RUNNER, CONFIG, PREREG, SPLIT_MANIFEST)
            ),
            key=lambda member: member.archive_path,
        )
    )
    _validate_unique_members(members, COMMON_ARCHIVE)
    return ArchivePlan(
        path=COMMON_ARCHIVE,
        kind="code_and_contract",
        dataset=None,
        members=members,
    )


def _gold_plans(root: Path) -> list[ArchivePlan]:
    plans: list[ArchivePlan] = []
    for dataset in ("alfa", "uav_attack", "uav_sead"):
        member = _member_from_relative(root, EXPECTED_GOLD_PATHS[dataset])
        members = (member,)
        _validate_unique_members(members, GOLD_ARCHIVES[dataset])
        plans.append(
            ArchivePlan(
                path=GOLD_ARCHIVES[dataset],
                kind="gold_parquet",
                dataset=dataset,
                members=members,
            )
        )
    return plans


def _rfly_manifest_rows(
    root: Path, rfly_contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path, _ = _safe_source(root, EXPECTED_RFLY_MANIFEST)
    registry_path, _ = _safe_source(root, EXPECTED_RFLY_SPLIT_REGISTRY)
    registry = _load_json_object(registry_path)
    groups = registry.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ColabBundleError("RflyMAD split registry has no groups")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific failure
        raise ColabBundleError("pyarrow is required to read the RflyMAD manifest") from exc

    columns = [
        "canonical_case_id",
        "domain",
        "fault_family",
        "split",
        "cv_fold",
        "split_group_id",
    ]
    table = pq.read_table(manifest_path, columns=columns)
    values = table.to_pydict()
    rows: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()
    observed_development_folds: set[int] = set()
    manifest_split_counts: dict[str, int] = {}
    selected_role_counts: dict[str, int] = {}
    normal_label = str(rfly_contract["normal_label"])
    train_folds = {int(value) for value in rfly_contract["train_folds"]}
    validation_fold = int(rfly_contract["validation_fold"])
    test_fold = int(rfly_contract["test_fold"])

    for index in range(table.num_rows):
        canonical_case_id = str(values["canonical_case_id"][index])
        domain = str(values["domain"][index])
        fault_family = str(values["fault_family"][index])
        split = str(values["split"][index])
        split_group_id = str(values["split_group_id"][index])
        fold_value = values["cv_fold"][index]
        if fold_value is None:
            raise ColabBundleError(f"Null RflyMAD cv_fold at manifest row {index}")
        fold = int(fold_value)

        if (
            not canonical_case_id
            or PurePosixPath(canonical_case_id).name != canonical_case_id
            or "\\" in canonical_case_id
        ):
            raise ColabBundleError(f"Unsafe RflyMAD canonical_case_id: {canonical_case_id}")
        if not domain or PurePosixPath(domain).name != domain or "\\" in domain:
            raise ColabBundleError(f"Unsafe RflyMAD domain: {domain}")

        identity = (domain, canonical_case_id)
        if identity in seen_sources:
            raise ColabBundleError(f"Duplicate RflyMAD manifest identity: {identity}")
        seen_sources.add(identity)

        group = groups.get(split_group_id)
        if not isinstance(group, dict):
            raise ColabBundleError(f"RflyMAD group missing from registry: {split_group_id}")
        if (
            group.get("split") != split
            or int(group.get("cv_fold", -1)) != fold
            or group.get("domain") != domain
        ):
            raise ColabBundleError(
                f"RflyMAD manifest/registry disagreement for group {split_group_id}"
            )

        manifest_split_counts[split] = manifest_split_counts.get(split, 0) + 1
        if split != "development":
            continue
        if fold not in EXPECTED_RFLY_FOLDS:
            raise ColabBundleError(f"Development row has out-of-contract fold: {fold}")
        observed_development_folds.add(fold)
        if fold in train_folds and fault_family == normal_label:
            selected_role = "train_normal"
        elif fold == validation_fold and fault_family == normal_label:
            selected_role = "validation_normal"
        elif fold == test_fold:
            selected_role = "test_all"
        else:
            continue
        selected_role_counts[selected_role] = selected_role_counts.get(selected_role, 0) + 1
        rows.append(
            {
                "canonical_case_id": canonical_case_id,
                "domain": domain,
                "fault_family": fault_family,
                "split": split,
                "cv_fold": fold,
                "selected_role": selected_role,
            }
        )

    if observed_development_folds != EXPECTED_RFLY_FOLDS:
        raise ColabBundleError(
            "RflyMAD development manifest must contain every fold 0--4; "
            f"observed {sorted(observed_development_folds)}"
        )
    if not rows:
        raise ColabBundleError("RflyMAD development selection is empty")

    rows.sort(key=lambda row: (row["domain"], row["canonical_case_id"]))
    summary = {
        "manifest_rows": table.num_rows,
        "manifest_rows_by_split": dict(sorted(manifest_split_counts.items())),
        "included_split": "development",
        "included_folds": sorted(EXPECTED_RFLY_FOLDS),
        "included_parquet_files": len(rows),
        "included_parquet_files_by_role": dict(sorted(selected_role_counts.items())),
        "locked_test_parquet_files_included": 0,
    }
    return rows, summary


def _rfly_plan(root: Path, rfly_contract: Mapping[str, Any]) -> ArchivePlan:
    rows, selection = _rfly_manifest_rows(root, rfly_contract)
    fold_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    members: list[BundleMember] = []

    total = len(rows)
    for number, row in enumerate(rows, start=1):
        fold_key = str(row["cv_fold"])
        domain = str(row["domain"])
        fold_counts[fold_key] = fold_counts.get(fold_key, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        relative = (
            EXPECTED_RFLY_DATA_ROOT
            / domain
            / f"{row['canonical_case_id']}.parquet"
        )
        members.append(_member_from_relative(root, relative))
        if number % 250 == 0 or number == total:
            print(f"rfly_member_hash_progress={number}/{total}", flush=True)

    members.extend(
        [
            _member_from_relative(root, EXPECTED_RFLY_MANIFEST),
            _member_from_relative(root, EXPECTED_RFLY_SPLIT_REGISTRY),
        ]
    )
    ordered = tuple(sorted(members, key=lambda member: member.archive_path))
    _validate_unique_members(ordered, RFLY_ARCHIVE)

    selection = {
        **selection,
        "included_parquet_files_by_fold": dict(sorted(fold_counts.items())),
        "included_parquet_files_by_domain": dict(sorted(domain_counts.items())),
        "metadata_files": [
            EXPECTED_RFLY_MANIFEST.as_posix(),
            EXPECTED_RFLY_SPLIT_REGISTRY.as_posix(),
        ],
    }
    return ArchivePlan(
        path=RFLY_ARCHIVE,
        kind="rflymad_development_parquet",
        dataset="rflymad",
        members=ordered,
        selection=selection,
    )


def _zip_info(member: BundleMember) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member.archive_path, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _write_member(
    archive: zipfile.ZipFile,
    member: BundleMember,
) -> None:
    before = member.source.stat()
    digest = hashlib.sha256()
    copied = 0
    force_zip64 = member.bytes >= getattr(zipfile, "ZIP64_LIMIT", 2**31 - 1)
    with member.source.open("rb") as source, archive.open(
        _zip_info(member), "w", force_zip64=force_zip64
    ) as target:
        for block in iter(lambda: source.read(HASH_BLOCK_BYTES), b""):
            target.write(block)
            digest.update(block)
            copied += len(block)
    after = member.source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ColabBundleError(f"Source changed while archiving: {member.archive_path}")
    if copied != member.bytes:
        raise ColabBundleError(f"Member byte-size drift: {member.archive_path}")
    if digest.hexdigest() != member.sha256:
        raise ColabBundleError(f"Member SHA-256 drift: {member.archive_path}")


def _validate_zip_directory(path: Path, members: Sequence[BundleMember]) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
    if [info.filename for info in infos] != [member.archive_path for member in members]:
        raise ColabBundleError(f"ZIP member order/path mismatch: {path.name}")
    for info, member in zip(infos, members):
        if info.compress_type != zipfile.ZIP_STORED:
            raise ColabBundleError(f"ZIP member was compressed: {info.filename}")
        if info.file_size != member.bytes:
            raise ColabBundleError(f"ZIP member size mismatch: {info.filename}")
        if info.date_time != FIXED_ZIP_TIMESTAMP:
            raise ColabBundleError(f"ZIP member timestamp drift: {info.filename}")


def _write_zip_atomic(
    destination: Path,
    plan: ArchivePlan,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    final_path = destination / plan.path
    if final_path.exists() and not overwrite:
        raise ColabBundleError(f"Refusing to overwrite existing archive: {final_path}")
    if final_path.exists() and not final_path.is_file():
        raise ColabBundleError(f"Archive target is not a file: {final_path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{plan.path}.", suffix=".partial", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for number, member in enumerate(plan.members, start=1):
                _write_member(archive, member)
                if len(plan.members) >= 250 and (
                    number % 250 == 0 or number == len(plan.members)
                ):
                    print(
                        f"archive_member_progress={plan.path}:{number}/{len(plan.members)}",
                        flush=True,
                    )
        _validate_zip_directory(temporary, plan.members)
        archive_bytes = temporary.stat().st_size
        archive_sha256 = _sha256_file(temporary)
        os.replace(temporary, final_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "path": plan.path,
        "bytes": archive_bytes,
        "sha256": archive_sha256,
        "compression": "ZIP_STORED",
        "member_count": len(plan.members),
        "source_bytes": plan.source_bytes,
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ColabBundleError(f"Refusing to overwrite existing file: {path}")
    if path.exists() and not path.is_file():
        raise ColabBundleError(f"Output target is not a file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _archive_manifest_record(
    plan: ArchivePlan,
    built: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": plan.path,
        "kind": plan.kind,
        "dataset": plan.dataset,
        "compression": "ZIP_STORED",
        "member_count": len(plan.members),
        "source_bytes": plan.source_bytes,
        "members": [member.manifest_record() for member in plan.members],
    }
    if plan.selection is not None:
        record["selection"] = dict(plan.selection)
    if built is None:
        record["archive_status"] = "planned_not_built"
    else:
        record.update(
            {
                "archive_status": "built",
                "archive_bytes": int(built["bytes"]),
                "archive_sha256": str(built["sha256"]),
            }
        )
    return record


def _ensure_targets_available(
    destination: Path,
    names: Iterable[str],
    *,
    overwrite: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ColabBundleError(f"Output path is not a directory: {destination}")
    if overwrite:
        return
    existing = [str(destination / name) for name in names if (destination / name).exists()]
    if existing:
        raise ColabBundleError(
            "Refusing to overwrite existing bundle outputs: " + ", ".join(existing)
        )


def build_bundle(
    *,
    root: Path,
    output_dir: Path,
    build_archives: bool,
    overwrite: bool,
) -> dict[str, Any]:
    repo_root = root.resolve(strict=True)
    if not repo_root.is_dir():
        raise ColabBundleError(f"Repository root is not a directory: {repo_root}")
    destination = (
        output_dir.resolve(strict=False)
        if output_dir.is_absolute()
        else (repo_root / output_dir).resolve(strict=False)
    )

    _, _, rfly_contract = _validate_contract(repo_root)
    plans = [
        _common_plan(repo_root),
        *_gold_plans(repo_root),
        _rfly_plan(repo_root, rfly_contract),
    ]
    archive_names = [plan.path for plan in plans]
    target_names = ["bundle_manifest.json", "transfer_index.json"]
    if build_archives:
        target_names.extend(archive_names)
    _ensure_targets_available(destination, target_names, overwrite=overwrite)

    built_by_path: dict[str, dict[str, Any]] = {}
    if build_archives:
        for number, plan in enumerate(plans, start=1):
            print(
                f"archive_start={number}/{len(plans)} path={plan.path} "
                f"members={len(plan.members)} source_bytes={plan.source_bytes}",
                flush=True,
            )
            built = _write_zip_atomic(destination, plan, overwrite=overwrite)
            built_by_path[plan.path] = built
            print(
                f"archive_done={number}/{len(plans)} path={plan.path} "
                f"sha256={built['sha256']}",
                flush=True,
            )

    bundle_manifest = {
        "schema_version": 1,
        "bundle_role": BUNDLE_ROLE,
        "candidate_namespace": CANDIDATE_NAMESPACE,
        "archives_built": build_archives,
        "determinism": {
            "archive_format": "ZIP",
            "compression": "ZIP_STORED",
            "member_order": "lexicographic_posix_path",
            "member_timestamp": "1980-01-01T00:00:00",
            "member_mode": "0644",
            "generated_timestamp_in_manifest": False,
        },
        "archives": [
            _archive_manifest_record(plan, built_by_path.get(plan.path)) for plan in plans
        ],
        "notes": [
            "The three Gold archives contain exactly one frozen feature parquet each.",
            "RflyMAD telemetry contains only the preregistered normal train/validation and all-test development roles.",
            "The full RflyMAD dataset_manifest and split_registry are metadata members; locked-test parsed telemetry is excluded.",
            "Parquet is already compressed, so every ZIP member uses ZIP_STORED.",
        ],
    }
    manifest_payload = _json_bytes(bundle_manifest)
    manifest_path = destination / "bundle_manifest.json"
    _write_bytes_atomic(manifest_path, manifest_payload, overwrite=overwrite)
    manifest_record = {
        "path": manifest_path.name,
        "bytes": len(manifest_payload),
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }

    transfer_index = {
        "schema_version": 1,
        "bundle_role": BUNDLE_ROLE,
        "candidate_namespace": CANDIDATE_NAMESPACE,
        "archives_built": build_archives,
        "bundle_manifest": manifest_record,
        "archives": [built_by_path[plan.path] for plan in plans if plan.path in built_by_path],
    }
    transfer_payload = _json_bytes(transfer_index)
    _write_bytes_atomic(
        destination / "transfer_index.json",
        transfer_payload,
        overwrite=overwrite,
    )

    return {
        "output_dir": str(destination),
        "archives_built": build_archives,
        "archive_count": len(built_by_path),
        "planned_archive_count": len(plans),
        "member_count": sum(len(plan.members) for plan in plans),
        "source_bytes": sum(plan.source_bytes for plan in plans),
        "bundle_manifest": manifest_record,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Transfer output directory, relative to --repo-root unless absolute.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Hash and validate all members but do not write ZIP archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace bundle outputs that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_bundle(
        root=args.repo_root,
        output_dir=args.output_dir,
        build_archives=not args.manifest_only,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": "ok", **result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
