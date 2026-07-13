"""Immutable, auditable run manifests for the clean ADS-B pipeline.

The manifest is deliberately created *before* a training/evaluation run writes any
other artifact.  It records exact inputs, the explicit flight-level split contract,
the configuration and the Git state.  A run directory is never reused.

This module does not discover inputs on its own.  In particular, it has no default
or implicit access to the sealed raw holdout pool under ``Downloads``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
HASH_ALGORITHM = "sha256"

# 2026-07-13 review measurement.  This is provenance, not a silent replacement
# of the older documented value; both values and the unresolved difference are
# intentionally carried in every manifest.
SILVER_FOOTER_ROWS_REVIEWED = 256_155_009
SILVER_ROWS_DOCUMENTED = 256_150_550
SILVER_ROW_DELTA = SILVER_FOOTER_ROWS_REVIEWED - SILVER_ROWS_DOCUMENTED

ALLOWED_INPUT_ROLES = frozenset(
    {
        "fit",
        "train",
        "calibration",
        "development",
        "validation",
        "rehearsal",
        "natural_evaluation",
        "synthetic_evaluation",
        "test",
        "reference",
    }
)
PROTECTED_SPLIT_ROLES = frozenset({"fit", "train", "calibration"})


class ManifestError(ValueError):
    """The requested run manifest violates an auditable run contract."""


@dataclass(frozen=True)
class InputSpec:
    """One input and the role it is permitted to have in a run."""

    path: Path
    role: str

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_INPUT_ROLES:
            allowed = ", ".join(sorted(ALLOWED_INPUT_ROLES))
            raise ManifestError(f"Unknown input role {self.role!r}; allowed roles: {allowed}")
        object.__setattr__(self, "path", Path(self.path))


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical JSON encoding used by manifest hashes."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Value is not canonical-JSON serializable: {exc}") from exc
    return encoded.encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a value after canonical JSON encoding."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a complete file into SHA-256 without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _encoded_metadata(metadata: Mapping[bytes, bytes] | None) -> list[dict[str, str]]:
    """Represent Arrow metadata without depending on mapping iteration order."""

    if not metadata:
        return []
    return [
        {
            "key_base64": base64.b64encode(key).decode("ascii"),
            "value_base64": base64.b64encode(value).decode("ascii"),
        }
        for key, value in sorted(metadata.items())
    ]


def arrow_schema_sha256(schema: pa.Schema) -> str:
    """Hash Arrow field order/types/nullability and schema/field metadata."""

    schema_contract = {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "metadata": _encoded_metadata(field.metadata),
            }
            for field in schema
        ],
        "metadata": _encoded_metadata(schema.metadata),
    }
    return sha256_json(schema_contract)


def _manifest_path(path: Path, repo_root: Path) -> str:
    """Use a repo-relative POSIX path when possible, otherwise an absolute path."""

    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def inspect_input_file(path: str | Path, *, role: str, repo_root: str | Path) -> dict[str, Any]:
    """Measure a single input's bytes/hash and, for Parquet, footer/schema.

    Non-Parquet files are supported for future raw-file freeze manifests.  Their
    ``footer_rows`` and ``schema_sha256`` fields are explicitly ``None`` rather
    than guessed.
    """

    spec = InputSpec(Path(path), role)
    root = Path(repo_root).resolve(strict=True)
    resolved = spec.path.resolve(strict=True)
    if not resolved.is_file():
        raise ManifestError(f"Input is not a regular file: {resolved}")

    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": _manifest_path(resolved, root),
        "role": spec.role,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
        "footer_rows": None,
        "schema_sha256": None,
        "schema_hash_contract": None,
        "format": "binary",
    }
    if resolved.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(resolved)
        record.update(
            {
                "footer_rows": parquet.metadata.num_rows,
                "schema_sha256": arrow_schema_sha256(parquet.schema_arrow),
                "schema_hash_contract": "sha256(canonical Arrow fields+metadata JSON), v1",
                "format": "parquet",
            }
        )
    return record


def _canonical_flight_ids(flight_ids: Iterable[object], *, label: str) -> list[str]:
    result: set[str] = set()
    for raw_id in flight_ids:
        flight_id = str(raw_id)
        if not flight_id or flight_id.strip() != flight_id:
            raise ManifestError(f"{label} contains an empty or whitespace-padded flight ID")
        result.add(flight_id)
    return sorted(result)


def make_deterministic_split(
    flight_ids: Iterable[object],
    split_weights: Mapping[str, float],
    *,
    seed: int,
    excluded_flight_ids: Iterable[object] = (),
) -> dict[str, list[str]]:
    """Create an order-invariant, exact-count hash-ranked flight split.

    Split names are sorted before allocation, IDs are ranked by SHA-256 of
    ``seed + NUL + flight_id``, and largest remainders determine exact counts.
    The algorithm therefore does not depend on input iteration order.
    """

    if not split_weights:
        raise ManifestError("split_weights cannot be empty")
    names = sorted(split_weights)
    if any(name not in ALLOWED_INPUT_ROLES for name in names):
        unknown = sorted(set(names) - ALLOWED_INPUT_ROLES)
        raise ManifestError(f"Unknown split roles: {unknown}")
    weights = {name: float(split_weights[name]) for name in names}
    if any(weight <= 0 for weight in weights.values()):
        raise ManifestError("Every split weight must be positive")

    excluded = set(_canonical_flight_ids(excluded_flight_ids, label="excluded_flight_ids"))
    ids = [
        flight_id
        for flight_id in _canonical_flight_ids(flight_ids, label="flight_ids")
        if flight_id not in excluded
    ]
    total_weight = sum(weights.values())
    exact = {name: len(ids) * weights[name] / total_weight for name in names}
    counts = {name: int(exact[name]) for name in names}
    remainder_order = sorted(names, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in remainder_order[: len(ids) - sum(counts.values())]:
        counts[name] += 1

    def rank_key(flight_id: str) -> tuple[str, str]:
        payload = f"{seed}\0{flight_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), flight_id

    ranked = sorted(ids, key=rank_key)
    result: dict[str, list[str]] = {}
    offset = 0
    for name in names:
        result[name] = sorted(ranked[offset : offset + counts[name]])
        offset += counts[name]
    return result


def build_split_contract(
    splits: Mapping[str, Iterable[object]],
    *,
    algorithm: str,
    seed: int | None,
) -> dict[str, Any]:
    """Validate, store and hash the exact flight IDs assigned to every split."""

    if not splits:
        raise ManifestError("At least one explicit split is required")
    normalized: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for name in sorted(splits):
        if name not in ALLOWED_INPUT_ROLES:
            raise ManifestError(f"Unknown split role: {name!r}")
        ids = _canonical_flight_ids(splits[name], label=f"split {name!r}")
        for flight_id in ids:
            if flight_id in owner:
                raise ManifestError(
                    f"Flight ID {flight_id!r} occurs in both {owner[flight_id]!r} and {name!r}"
                )
            owner[flight_id] = name
        normalized[name] = ids

    hash_payload = {
        "algorithm": algorithm,
        "seed": seed,
        "splits": normalized,
    }
    return {
        "algorithm": algorithm,
        "seed": seed,
        "contract_sha256": sha256_json(hash_payload),
        "splits": {
            name: {
                "flight_id_count": len(ids),
                "flight_ids_sha256": sha256_json(ids),
                "flight_ids": ids,
            }
            for name, ids in normalized.items()
        },
    }


def _path_has_synthetic_marker(path: Path) -> bool:
    return any("synthetic" in part.lower() for part in path.resolve(strict=False).parts)


def build_synthetic_guard(
    input_specs: Sequence[InputSpec],
    split_contract: Mapping[str, Any],
    *,
    synthetic_flight_ids: Iterable[object],
) -> dict[str, Any]:
    """Reject synthetic fit paths and synthetic-source IDs in protected splits."""

    path_checks = []
    for spec in sorted(input_specs, key=lambda item: (item.role, str(item.path))):
        marked = _path_has_synthetic_marker(spec.path)
        path_checks.append(
            {
                "path": spec.path.resolve(strict=False).as_posix(),
                "role": spec.role,
                "synthetic_path_marker_found": marked,
            }
        )
        if marked and spec.role in PROTECTED_SPLIT_ROLES:
            raise ManifestError(
                f"Synthetic path is forbidden for protected role {spec.role!r}: {spec.path}"
            )

    excluded_ids = _canonical_flight_ids(
        synthetic_flight_ids, label="synthetic_flight_ids"
    )
    excluded_set = set(excluded_ids)
    protected_ids: set[str] = set()
    protected_roles_present: list[str] = []
    split_records = split_contract["splits"]
    for role in sorted(PROTECTED_SPLIT_ROLES):
        if role in split_records:
            protected_roles_present.append(role)
            protected_ids.update(split_records[role]["flight_ids"])
    overlap = sorted(protected_ids & excluded_set)
    if overlap:
        preview = ", ".join(repr(value) for value in overlap[:5])
        raise ManifestError(
            f"Synthetic-source flight IDs overlap protected splits ({len(overlap)}): {preview}"
        )

    return {
        "policy": "synthetic_never_fit_or_calibrate_v1",
        "status": "passed",
        "protected_split_roles": sorted(PROTECTED_SPLIT_ROLES),
        "protected_roles_present": protected_roles_present,
        "path_marker": "synthetic (case-insensitive path-component substring)",
        "path_checks": path_checks,
        "excluded_flight_id_count": len(excluded_ids),
        "excluded_flight_ids_sha256": sha256_json(excluded_ids),
        "excluded_flight_ids": excluded_ids,
        "protected_flight_id_count": len(protected_ids),
        "overlap_count": 0,
    }


def _run_git(repo_root: Path, *args: str) -> str:
    command = ["git", "-C", str(repo_root), *args]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError(f"Cannot collect Git provenance with {command!r}: {exc}") from exc
    return completed.stdout.rstrip("\n")


def collect_git_state(repo_root: str | Path) -> dict[str, Any]:
    """Capture the exact commit plus a hashed and explicit porcelain dirty state."""

    root = Path(repo_root).resolve(strict=True)
    commit = _run_git(root, "rev-parse", "HEAD").strip()
    if len(commit) != 40:
        raise ManifestError(f"Unexpected Git commit identifier: {commit!r}")
    status_text = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    entries = status_text.splitlines() if status_text else []
    return {
        "commit": commit,
        "dirty": bool(entries),
        "status_porcelain_sha256": sha256_json(entries),
        "status_porcelain": entries,
    }


def silver_row_count_provenance() -> dict[str, Any]:
    """Carry the approved unresolved Silver row-count discrepancy explicitly."""

    return {
        "status": "unresolved_do_not_silently_correct",
        "review_date": "2026-07-13",
        "footer_observed_rows": SILVER_FOOTER_ROWS_REVIEWED,
        "documented_rows": SILVER_ROWS_DOCUMENTED,
        "delta_rows": SILVER_ROW_DELTA,
        "footer_scope": "638 Silver Parquet files across 2026-02-28/03-01/03-16",
        "evidence": [
            "adsb_parse_02_28.log:475-476",
            "adsb_parse_03_01.log:433-434",
            "adsb_parse_03_16.log:371-372",
            "docs/codex_review_findings_2026-07-13.md:38-50",
        ],
        "note": (
            "The 256,155,009 footer measurement and the previously documented "
            "256,150,550 total differ by 4,459 rows. The cause was not measured; "
            "both values remain recorded until reconciled."
        ),
    }


def create_immutable_run_manifest(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    inputs: Sequence[InputSpec],
    splits: Mapping[str, Iterable[object]],
    split_algorithm: str,
    split_seed: int | None,
    synthetic_flight_ids: Iterable[object],
    config: Mapping[str, Any],
) -> Path:
    """Create ``run_manifest.json`` in a new directory, refusing all reuse.

    All validation and input hashing happens before the directory is created.  The
    final ``mkdir(exist_ok=False)`` and ``open('x')`` provide fail-if-exists
    semantics even under a race.
    """

    destination = Path(run_dir).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists and is immutable: {destination}")
    if not destination.name:
        raise ManifestError("run_dir must name a run directory")
    if not inputs:
        raise ManifestError("At least one explicit input is required")

    root = Path(repo_root).resolve(strict=True)
    normalized_inputs = [InputSpec(item.path, item.role) for item in inputs]
    input_keys = [(item.path.resolve(strict=False), item.role) for item in normalized_inputs]
    if len(set(input_keys)) != len(input_keys):
        raise ManifestError("Duplicate input path/role entries are not allowed")

    split_contract = build_split_contract(
        splits,
        algorithm=split_algorithm,
        seed=split_seed,
    )
    synthetic_guard = build_synthetic_guard(
        normalized_inputs,
        split_contract,
        synthetic_flight_ids=synthetic_flight_ids,
    )
    input_records = [
        inspect_input_file(item.path, role=item.role, repo_root=root)
        for item in sorted(normalized_inputs, key=lambda value: (value.role, str(value.path)))
    ]
    config_value = dict(config)
    config_sha256 = sha256_json(config_value)
    git_state = collect_git_state(root)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": destination.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutability": {
            "policy": "new_directory_and_exclusive_manifest_create",
            "fail_if_exists": True,
        },
        "hash_algorithm": HASH_ALGORITHM,
        "git": git_state,
        "config": {
            "sha256": config_sha256,
            "value": config_value,
        },
        "inputs": input_records,
        "input_contract_sha256": sha256_json(input_records),
        "split_contract": split_contract,
        "synthetic_guard": synthetic_guard,
        "silver_row_count_provenance": silver_row_count_provenance(),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    manifest_path = destination / MANIFEST_FILENAME
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path

