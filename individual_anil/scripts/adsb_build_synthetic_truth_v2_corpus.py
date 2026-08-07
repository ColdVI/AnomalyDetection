"""Build a fail-if-exists truth-v2 corpus from the immutable v1 paired corpus.

The corrupt observations are copied from v1 rather than regenerated.  Therefore
Step 3 can change only truth/evaluation while keeping the frozen rule's score
surface byte-for-value equivalent.  The old dropout injector's exact RNG block
is reconstructed explicitly; its old onset-to-end proxy label is not reused as
truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.run_manifest import sha256_file, sha256_json  # noqa: E402
from adsb.synthetic import PHYSICS_BREAK_RECIPES  # noqa: E402
from adsb.truth import attach_clean_truth_v2, attach_event_truth_v2  # noqa: E402

DEFAULT_SOURCE = Path("data/objectstore/synthetic/adsb")
DEFAULT_OUTPUT = Path("data/objectstore/synthetic/adsb_v2_20260713_01")
ONSET_FRACTION = 0.5
DROPOUT_BLOCK_FRACTION = 0.3
DROPOUT_LEGACY_SEED = 0
BATCH_FLIGHTS = 250

OBSERVABLE_COLUMNS = {
    "vertical_rate_frozen": ["vertical_rate_ms"],
    "ground_speed_biased": ["ground_speed_ms"],
    "track_frozen": ["track_deg"],
    "position_ramp_stealthy": ["lat", "lon"],
    "altitude_dropout": ["alt"],
}


def _active_mask(n_rows: int, recipe: str) -> np.ndarray:
    i0 = int(n_rows * ONSET_FRACTION)
    active = np.zeros(n_rows, dtype=bool)
    if recipe != "altitude_dropout":
        active[i0:] = True
        return active

    n_after_onset = n_rows - i0
    n_drop = int(n_after_onset * DROPOUT_BLOCK_FRACTION)
    if n_drop:
        # This exactly mirrors the v1 implementation, including its exclusive
        # upper bound.  It is reconstruction of existing data, not a new choice.
        rng = np.random.default_rng(DROPOUT_LEGACY_SEED)
        start = i0 + int(rng.integers(0, max(n_after_onset - n_drop, 1)))
        active[start : start + n_drop] = True
    return active


def _normalize_truth_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("event_id", "event_type"):
        out[column] = out[column].astype("string")
    for column in ("attack_onset", "observable_onset", "event_end"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    for column in ("injection_active", "observable_changed", "evaluable_truth"):
        out[column] = out[column].astype(bool)
    return out


def _write_batches(path: Path, batches, *, compression: str = "zstd") -> dict:
    if path.exists():
        raise FileExistsError(f"Truth-v2 output already exists: {path}")
    writer: pq.ParquetWriter | None = None
    n_rows = n_flights = 0
    n_active = n_changed = n_evaluable = 0
    event_ids: set[str] = set()
    try:
        for frame in batches:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(_normalize_truth_dtypes(frame), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression)
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            n_rows += len(frame)
            n_active += int(frame['injection_active'].sum())
            n_changed += int(frame['observable_changed'].sum())
            n_evaluable += int(frame['evaluable_truth'].sum())
            event_ids.update(frame['event_id'].dropna().astype(str).unique())
            n_flights += int(frame["flight_id"].nunique())
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"No rows were produced for {path.name}")
    return {
        "n_rows": n_rows,
        "n_flights": n_flights,
        "n_active_rows": n_active,
        "n_observable_changed_rows": n_changed,
        "n_evaluable_rows": n_evaluable,
        "n_events": len(event_ids),
    }


def _clean_batches(clean: pd.DataFrame):
    groups = clean.groupby("flight_id", sort=False)
    pending: list[pd.DataFrame] = []
    for _, flight in groups:
        pending.append(attach_clean_truth_v2(flight.reset_index(drop=True)))
        if len(pending) >= BATCH_FLIGHTS:
            yield pd.concat(pending, ignore_index=True)
            pending.clear()
    if pending:
        yield pd.concat(pending, ignore_index=True)


def _recipe_batches(clean: pd.DataFrame, corrupt: pd.DataFrame, recipe: str):
    clean_grouped = clean.groupby("flight_id", sort=False)
    clean_ids = {str(fid) for fid in clean_grouped.groups}
    pending: list[pd.DataFrame] = []
    seen: set[str] = set()
    for flight_id, bad in corrupt.groupby("flight_id", sort=False):
        key = str(flight_id)
        if key not in clean_ids:
            raise ValueError(f"{recipe}: corrupt flight missing from clean pair: {key}")
        good = clean_grouped.get_group(flight_id).reset_index(drop=True)
        bad = bad.reset_index(drop=True)
        if len(good) != len(bad):
            raise ValueError(f"{recipe}/{key}: paired row-count mismatch")
        if not np.array_equal(
            good["timestamp_utc"].to_numpy(), bad["timestamp_utc"].to_numpy(), equal_nan=True
        ):
            raise ValueError(f"{recipe}/{key}: paired timestamp mismatch")
        truth = attach_event_truth_v2(
            good,
            bad,
            event_type=recipe,
            event_id=f"{recipe}:{key}",
            injection_active=_active_mask(len(bad), recipe),
            observable_cols=OBSERVABLE_COLUMNS[recipe],
        )
        pending.append(truth)
        seen.add(key)
        if len(pending) >= BATCH_FLIGHTS:
            yield pd.concat(pending, ignore_index=True)
            pending.clear()
    if seen != clean_ids:
        missing = sorted(clean_ids - seen)
        raise ValueError(f"{recipe}: {len(missing)} clean flights lack corrupt pairs")
    if pending:
        yield pd.concat(pending, ignore_index=True)


def _annotate_recipe_vectorized(
    clean: pd.DataFrame,
    corrupt: pd.DataFrame,
    recipe: str,
) -> pd.DataFrame:
    """Attach the same per-flight truth contract without per-flight copies."""

    if len(clean) != len(corrupt):
        raise ValueError(f"{recipe}: paired row-count mismatch")
    good_flight = clean["flight_id"].astype("string").reset_index(drop=True)
    bad_flight = corrupt["flight_id"].astype("string").reset_index(drop=True)
    if not good_flight.equals(bad_flight):
        raise ValueError(f"{recipe}: paired flight/order mismatch")
    good_time = pd.to_numeric(clean["timestamp_utc"], errors="coerce").to_numpy(dtype=float)
    bad_time = pd.to_numeric(corrupt["timestamp_utc"], errors="coerce").to_numpy(dtype=float)
    if not np.array_equal(good_time, bad_time, equal_nan=True):
        raise ValueError(f"{recipe}: paired timestamp/order mismatch")

    grouped = bad_flight.groupby(bad_flight, sort=False)
    position = grouped.cumcount().to_numpy(dtype=np.int64)
    size = grouped.transform("size").to_numpy(dtype=np.int64)
    onset = np.floor(size * ONSET_FRACTION).astype(np.int64)
    if recipe == "altitude_dropout":
        starts_by_size: dict[int, int] = {}
        for n_rows in np.unique(size):
            i0 = int(n_rows * ONSET_FRACTION)
            n_after = int(n_rows - i0)
            n_drop = int(n_after * DROPOUT_BLOCK_FRACTION)
            rng = np.random.default_rng(DROPOUT_LEGACY_SEED)
            start = i0 + int(rng.integers(0, max(n_after - n_drop, 1))) if n_drop else i0
            starts_by_size[int(n_rows)] = start
        start = np.fromiter((starts_by_size[int(value)] for value in size), dtype=np.int64, count=len(size))
        n_drop = np.floor((size - onset) * DROPOUT_BLOCK_FRACTION).astype(np.int64)
        active = (position >= start) & (position < start + n_drop)
    else:
        active = position >= onset

    observable_cols = OBSERVABLE_COLUMNS[recipe]
    left, right = clean[observable_cols], corrupt[observable_cols]
    same = left.eq(right) | (left.isna() & right.isna())
    changed = ~same.fillna(False).all(axis=1).to_numpy(dtype=bool)

    out = corrupt.copy()
    out["event_id"] = (recipe + ":" + bad_flight).astype("string")
    out["event_type"] = pd.Series(recipe, index=out.index, dtype="string")
    time = pd.Series(bad_time, index=out.index)
    ids = out["flight_id"]
    out["attack_onset"] = time.where(active).groupby(ids, sort=False).transform("min")
    out["observable_onset"] = time.where(changed).groupby(ids, sort=False).transform("min")
    out["event_end"] = time.where(active).groupby(ids, sort=False).transform("max")
    out["injection_active"] = active
    out["observable_changed"] = changed
    out["evaluable_truth"] = True
    return out


def build_corpus(source_dir: Path, output_dir: Path) -> Path:
    source_dir = source_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if not any(part.lower() == "synthetic" for part in output_dir.parts):
        raise ValueError("output_dir must contain an exact 'synthetic' path component")
    if output_dir.exists():
        raise FileExistsError(f"Truth-v2 corpus directory already exists: {output_dir}")

    expected = [source_dir / "clean.parquet"] + [
        source_dir / f"{name}.parquet" for name in PHYSICS_BREAK_RECIPES
    ]
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    clean = pd.read_parquet(source_dir / "clean.parquet")
    if clean["flight_id"].isna().any():
        raise ValueError("v1 clean flight IDs must be complete")

    outputs: list[dict] = []
    clean_path = output_dir / "clean.parquet"
    clean_truth = attach_clean_truth_v2(clean)
    clean_stats = _write_batches(clean_path, [clean_truth])
    del clean_truth
    outputs.append({"recipe": "clean", "path": clean_path.name, **clean_stats})

    for recipe in PHYSICS_BREAK_RECIPES:
        corrupt = pd.read_parquet(source_dir / f"{recipe}.parquet")
        path = output_dir / f"{recipe}.parquet"
        truth = _annotate_recipe_vectorized(clean, corrupt, recipe)
        stats = _write_batches(path, [truth])
        outputs.append({"recipe": recipe, "path": path.name, **stats})
        del corrupt, truth

    for item in outputs:
        path = output_dir / item["path"]
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
        item["footer_rows"] = pq.ParquetFile(path).metadata.num_rows

    source_records = [
        {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in expected
    ]
    manifest = {
        "schema_version": "adsb_synthetic_truth_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "paired v1 observations copied; no corrupt observation regenerated",
        "source_files": source_records,
        "source_contract_sha256": sha256_json(source_records),
        "truth_columns": [
            "event_id", "event_type", "attack_onset", "observable_onset", "event_end",
            "injection_active", "observable_changed", "evaluable_truth",
        ],
        "window_truth_contract": {
            "primary": "y_any = 1[q_w > 0]",
            "secondary": "q_w in {0,1}; mixed windows reported separately",
            "rule_ae_support": "all window rows",
            "forecaster_support": "explicit final target rows only",
        },
        "dropout_exact_block_reconstruction": {
            "onset_fraction": ONSET_FRACTION,
            "block_fraction": DROPOUT_BLOCK_FRACTION,
            "legacy_rng_seed_per_flight": DROPOUT_LEGACY_SEED,
            "legacy_upper_bound_contract": "integers(0, max(n_after_onset - n_drop, 1))",
        },
        "synthetic_never_training": True,
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(build_corpus(args.source_dir, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
