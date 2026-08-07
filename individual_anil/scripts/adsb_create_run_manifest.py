"""Create an immutable ADS-B run manifest before producing run artifacts.

Example::

    python scripts/adsb_create_run_manifest.py \
      --run-dir artifacts/adsb/runs/rule_v2_001 \
      --input fit=data/objectstore/silver/adsblol_historical/part-00000.parquet \
      --config configs/rule_v2_001.json \
      --splits configs/rule_v2_001_splits.json \
      --synthetic-flight-ids configs/synthetic_source_flight_ids.json

The command has no input discovery and no Downloads/holdout default.  Every input
must be named explicitly as ``ROLE=PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.run_manifest import InputSpec, create_immutable_run_manifest  # noqa: E402


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_spec(raw: str) -> InputSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("input must be ROLE=PATH")
    role, path = raw.split("=", 1)
    if not role or not path:
        raise argparse.ArgumentTypeError("input must have non-empty ROLE and PATH")
    try:
        return InputSpec(Path(path), role)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_splits(path: Path) -> tuple[dict[str, list[object]], str, int | None]:
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError("splits JSON must be an object")
    if "splits" in value:
        splits = value["splits"]
        algorithm = value.get("algorithm", "precomputed_explicit_v1")
        seed = value.get("seed")
    else:
        splits = value
        algorithm = "precomputed_explicit_v1"
        seed = None
    if not isinstance(splits, dict) or not all(isinstance(ids, list) for ids in splits.values()):
        raise ValueError("splits must map each role to a JSON list of flight IDs")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("split algorithm must be a non-empty string")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("split seed must be an integer or null")
    return splits, algorithm, seed


def _load_flight_ids(path: Path) -> list[object]:
    if path.suffix.lower() == ".json":
        value = _json(path)
        if isinstance(value, dict):
            value = value.get("flight_ids")
        if not isinstance(value, list):
            raise ValueError("synthetic flight-ID JSON must be a list or {'flight_ids': [...]} object")
        return value
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=_input_spec,
        required=True,
        metavar="ROLE=PATH",
        help="Explicit input; repeat once per file",
    )
    parser.add_argument("--config", type=Path, required=True, help="Canonical JSON run config")
    parser.add_argument("--splits", type=Path, required=True, help="Explicit flight split JSON")
    parser.add_argument(
        "--synthetic-flight-ids",
        type=Path,
        required=True,
        help="JSON list/object or one-ID-per-line source-flight exclusion file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _json(args.config)
    if not isinstance(config, dict):
        raise ValueError("config JSON must be an object")
    splits, algorithm, seed = _load_splits(args.splits)
    synthetic_flight_ids = _load_flight_ids(args.synthetic_flight_ids)
    manifest_path = create_immutable_run_manifest(
        run_dir=args.run_dir,
        repo_root=args.repo_root,
        inputs=args.inputs,
        splits=splits,
        split_algorithm=algorithm,
        split_seed=seed,
        synthetic_flight_ids=synthetic_flight_ids,
        config=config,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

