from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.features.align import align_to_clock, default_tolerances, observed_tolerances
from gecmis_calismalar.residual_v1.features.build import build_xy
from gecmis_calismalar.residual_v1.features.phases import label_phases
from gecmis_calismalar.residual_v1.features.spec import ALFA_SPECS, RFLY_SPECS, descriptor_schema_sha256
from gecmis_calismalar.residual_v1.features.waypoints import load_waypoint_config
from gecmis_calismalar.residual_v1.ingest.alfa import load_alfa_flight
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, sha256_file, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def _stale_from_profile(path: Path | None) -> dict[str, list[dict[str, float]]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stale: dict[str, list[dict[str, float]]] = {}
    for topic in payload["topics"].values():
        for channel, values in topic["channels"].items():
            if values["stale_segments"]:
                stale[channel] = values["stale_segments"]
    return stale


def _load_topics(flight_root: Path) -> dict[str, pd.DataFrame]:
    return {
        path.stem: pd.read_parquet(path)
        for path in sorted(flight_root.glob("*.parquet"))
        if not path.stem.startswith("failure_status-")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RESIDUAL-V1 X/y matrices")
    parser.add_argument("--dataset", choices=("alfa", "rfly"), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--silver-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--profile-root")
    parser.add_argument("--output-root", default="artifacts/residual_v1/features")
    args = parser.parse_args()

    silver_root = Path(args.silver_root or f"artifacts/residual_v1/silver/{args.dataset}")
    split_path = Path(
        args.split_manifest or f"artifacts/residual_v1/splits/{args.dataset}_seed{args.seed}.json"
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_ids = split["partitions"]["development"]["flight_ids"]
    waypoint_config_path = Path("configs/residual_v1_waypoint_mask.json")
    config_paths = ["configs/residual_v1_phases.json", "configs/residual_v1_g0.json"]
    if args.dataset == "alfa":
        config_paths.append(waypoint_config_path)
    run_dir, _ = create_run_dir(
        f"phaseC_build_features_{args.dataset}_seed{args.seed}",
        seed=args.seed,
        config_paths=config_paths,
        input_paths=[split_path],
    )
    output = Path(args.output_root) / args.dataset / f"seed{args.seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    specs = ALFA_SPECS if args.dataset == "alfa" else RFLY_SPECS
    clock_topic = "mavros-nav_info-roll" if args.dataset == "alfa" else "vehicle_attitude"
    tolerances = default_tolerances(args.dataset)
    total_rows = 0
    dropped_rows = 0
    flights_written = 0
    per_flight = []
    profile_root = Path(args.profile_root) if args.profile_root else None
    boundary_configs = (
        {"waypoint": load_waypoint_config(waypoint_config_path)}
        if args.dataset == "alfa"
        else {}
    )
    for flight_id in development_ids:
        flight_root = silver_root / Path(flight_id)
        if not flight_root.exists():
            raise FileNotFoundError(flight_root)
        topics = _load_topics(flight_root)
        profile_path = profile_root / Path(flight_id).with_suffix(".json") if profile_root else None
        aligned = align_to_clock(
            topics,
            clock_topic,
            observed_tolerances(topics, tolerances),
            stale=_stale_from_profile(profile_path),
        )
        phases = label_phases(aligned, dataset=args.dataset)
        events = json.loads((flight_root / "events.json").read_text(encoding="utf-8"))
        flight_output = output / Path(flight_id)
        flight_output.mkdir(parents=True)
        channel_rows = {}
        for spec in specs:
            X, y, meta = build_xy(
                aligned,
                spec,
                phases,
                events=events,
                flight_id=flight_id,
                boundary_configs=boundary_configs,
            )
            row_meta = meta.pop("row_meta")
            matrix = pd.concat(
                [row_meta.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)],
                axis=1,
            )
            matrix.to_parquet(flight_output / f"{spec.name}.parquet", index=False)
            channel_rows[spec.name] = int(len(matrix))
            total_rows += int(len(matrix))
            dropped_rows += int(meta["nan_drop_count"])
        write_json(flight_output / "feature_report.json", {"flight_id": flight_id, "channel_rows": channel_rows})
        per_flight.append({"flight_id": flight_id, "channel_rows": channel_rows})
        flights_written += 1
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "development_flights": flights_written,
        "feature_rows": total_rows,
        "nan_dropped_rows": dropped_rows,
        "descriptor_schema_residual_v1": descriptor_schema_sha256(),
        "boundary_mask_configs": (
            {
                "waypoint": {
                    "path": str(waypoint_config_path),
                    "sha256": sha256_file(waypoint_config_path),
                    "applies_to": ["R6_xtrack_error"],
                }
            }
            if args.dataset == "alfa"
            else {}
        ),
        "flights": per_flight,
    }
    write_json(output / "summary.json", summary, fail_if_exists=True)
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name=f"phaseC_build_features_{args.dataset}",
        metrics={
            "development_flights": flights_written,
            "feature_rows": total_rows,
            "nan_dropped_rows": dropped_rows,
        },
        params={"dataset": args.dataset, "seed": args.seed},
    )
    update_manifest(
        run_dir,
        descriptor_schema_residual_v1=descriptor_schema_sha256(),
        boundary_mask_configs=summary["boundary_mask_configs"],
        feature_output=str(output),
        mlflow_status=tracking["status"],
    )


if __name__ == "__main__":
    main()
