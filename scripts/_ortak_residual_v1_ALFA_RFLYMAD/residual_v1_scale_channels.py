from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.decision.scaling import ZeroMADChannel, fit_robust_scaler, robust_z
from gecmis_calismalar.residual_v1.features.align import align_to_clock, default_tolerances, observed_tolerances
from gecmis_calismalar.residual_v1.features.phases import label_phases
from gecmis_calismalar.residual_v1.features.spec import ALFA_SPECS, descriptor_schema_sha256
from gecmis_calismalar.residual_v1.features.waypoints import label_waypoint_boundaries
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, sha256_file, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run

R6_CHANNEL = "R6_xtrack_error"
R6_PROXY_FORMULA = "sqrt((roll/rad(25deg))^2 + (roll_rate/rad(15deg_s))^2)"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix(feature_root: Path, channel: str) -> pd.DataFrame:
    paths = sorted(feature_root.rglob(f"{channel}.parquet"))
    if not paths:
        raise FileNotFoundError(f"no feature matrices found for {channel}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _rfly_scaled_frame(
    *,
    feature_root: Path,
    g1_run: Path,
    channel: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    matrix = _load_matrix(feature_root, channel)
    residuals = pd.read_parquet(g1_run / "residuals" / f"{channel}.parquet")
    if len(matrix) != len(residuals):
        raise ValueError(f"{channel}: feature/residual row count mismatch")
    if not (
        matrix["flight_id"].astype(str).to_numpy() == residuals["flight_id"].astype(str).to_numpy()
    ).all() or not np.allclose(matrix["t"], residuals["t"], rtol=0.0, atol=1e-9):
        raise ValueError(f"{channel}: feature/residual row identity mismatch")
    values = matrix[feature_columns].to_numpy(float)
    output = residuals.copy()
    output["input_magnitude"] = np.linalg.norm(values, axis=1)
    params = fit_robust_scaler(
        output["r"], output["train_eligible"], channel=channel, clip=8.0
    )
    output["z"] = robust_z(output["r"], params)
    return output, params.to_dict()


def _load_alfa_topics(flight_root: Path) -> dict[str, pd.DataFrame]:
    names = (
        "mavros-nav_info-roll",
        "mavros-imu-data",
        "mavros-vfr_hud",
        "mavros-global_position-global",
        "mavros-nav_info-errors",
    )
    return {name: pd.read_parquet(flight_root / f"{name}.parquet") for name in names}


def _r6_direct_frame(silver_root: Path, development_ids: list[str]) -> pd.DataFrame:
    parts = []
    for flight_id in development_ids:
        flight_root = silver_root / Path(flight_id)
        topics = _load_alfa_topics(flight_root)
        aligned = align_to_clock(
            topics,
            "mavros-nav_info-roll",
            observed_tolerances(topics, default_tolerances("alfa")),
        )
        phases = label_phases(aligned, dataset="alfa")
        waypoint = label_waypoint_boundaries(aligned)
        events = _read_json(flight_root / "events.json")
        onset = min((float(event["onset_s"]) for event in events), default=float("inf"))
        xtrack = pd.to_numeric(aligned["xtrack_error"], errors="coerce")
        roll = pd.to_numeric(aligned["roll"], errors="coerce")
        roll_rate = pd.to_numeric(aligned["roll_rate"], errors="coerce")
        proxy = np.sqrt(
            (roll / np.deg2rad(25.0)) ** 2
            + (roll_rate / np.deg2rad(15.0)) ** 2
        )
        keep = (
            (phases["phase"] != "ground")
            & ~phases["phase_boundary"].astype(bool)
            & ~waypoint["waypoint_boundary"].astype(bool)
            & xtrack.notna()
        )
        retained_t = pd.to_numeric(aligned.loc[keep, "t"], errors="raise")
        parts.append(
            pd.DataFrame(
                {
                    "flight_id": flight_id,
                    "t": retained_t.to_numpy(float),
                    "phase": phases.loc[keep, "phase"].to_numpy(),
                    "train_eligible": (retained_t < onset - 10.0).to_numpy(bool),
                    "channel": R6_CHANNEL,
                    "r": xtrack.loc[keep].to_numpy(float),
                    "input_magnitude": proxy.loc[keep].to_numpy(float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale S-4-eligible RESIDUAL-V1 channels")
    parser.add_argument("--rfly-feature-root", required=True)
    parser.add_argument("--rfly-g1-run", required=True)
    parser.add_argument("--s4-run", required=True)
    parser.add_argument("--alfa-silver-root", required=True)
    parser.add_argument("--rfly-split", default="artifacts/residual_v1/splits/rfly_seed11.json")
    parser.add_argument("--alfa-split", default="artifacts/residual_v1/splits/alfa_seed11.json")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    feature_root = Path(args.rfly_feature_root)
    g1_run = Path(args.rfly_g1_run)
    s4_run = Path(args.s4_run)
    alfa_silver_root = Path(args.alfa_silver_root)
    rfly_split_path = Path(args.rfly_split)
    alfa_split_path = Path(args.alfa_split)
    feature_summary = _read_json(feature_root / "summary.json")
    g1_summary = _read_json(g1_run / "summary.json")
    s4_flags = _read_json(s4_run / "flags.json")
    expected_hash = descriptor_schema_sha256()
    if feature_summary.get("descriptor_schema_residual_v1") != expected_hash:
        raise ValueError("RFLY feature descriptor hash is stale")
    if g1_summary.get("descriptor_schema_residual_v1") != expected_hash:
        raise ValueError("RFLY G1 descriptor hash is stale")
    if not bool(g1_summary.get("development_only")) or not bool(s4_flags.get("development_only")):
        raise ValueError("scaling requires development-only G1 and S-4 inputs")
    rfly_development = set(_read_json(rfly_split_path)["partitions"]["development"]["flight_ids"])
    if {record["flight_id"] for record in feature_summary["flights"]} != rfly_development:
        raise ValueError("RFLY feature root is not exactly the development partition")
    alfa_development = _read_json(alfa_split_path)["partitions"]["development"]["flight_ids"]

    waypoint_config_path = Path("configs/residual_v1_waypoint_mask.json")
    run_dir, _ = create_run_dir(
        f"phaseE_scaling_seed{args.seed}",
        seed=args.seed,
        config_paths=[waypoint_config_path],
        input_paths=[
            feature_root / "summary.json",
            g1_run / "summary.json",
            s4_run / "flags.json",
            rfly_split_path,
            alfa_split_path,
        ],
    )
    (run_dir / "scaled" / "rfly").mkdir(parents=True)
    (run_dir / "scaled" / "alfa").mkdir(parents=True)
    scalers = {}
    excluded = []

    for channel in s4_flags["decision_eligible_channels"]:
        source_report = _read_json(g1_run / "channel_reports" / f"{channel}.json")
        try:
            frame, params = _rfly_scaled_frame(
                feature_root=feature_root,
                g1_run=g1_run,
                channel=channel,
                feature_columns=source_report["feature_columns"],
            )
        except ZeroMADChannel as error:
            excluded.append({"dataset": "rfly", "channel": channel, "reason": str(error)})
            continue
        if set(frame["flight_id"].astype(str)) - rfly_development:
            raise ValueError(f"non-development flight reached scaling for {channel}")
        frame.to_parquet(run_dir / "scaled" / "rfly" / f"{channel}.parquet", index=False)
        scalers[channel] = {"dataset": "rfly", **params, "input_magnitude": "l2_norm_of_full_g1_feature_vector"}
        del frame

    r6 = _r6_direct_frame(alfa_silver_root, alfa_development)
    try:
        params = fit_robust_scaler(r6["r"], r6["train_eligible"], channel=R6_CHANNEL, clip=8.0)
    except ZeroMADChannel as error:
        excluded.append({"dataset": "alfa", "channel": R6_CHANNEL, "reason": str(error)})
    else:
        r6["z"] = robust_z(r6["r"], params)
        r6.to_parquet(run_dir / "scaled" / "alfa" / f"{R6_CHANNEL}.parquet", index=False)
        scalers[R6_CHANNEL] = {
            "dataset": "alfa",
            **params.to_dict(),
            "input_magnitude": R6_PROXY_FORMULA,
            "waypoint_mask_config_sha256": sha256_file(waypoint_config_path),
        }

    for channel in s4_flags["flagged_channels"]:
        excluded.append({"dataset": "rfly", "channel": channel, "reason": "s4_flagged"})
    for channel, report in s4_flags["channels"].items():
        if report["status"] == "not_evaluable":
            excluded.append({"dataset": "rfly", "channel": channel, "reason": "model_unavailable"})
    for spec in ALFA_SPECS[:5]:
        excluded.append({"dataset": "alfa", "channel": spec.name, "reason": "model_unavailable"})

    write_json(run_dir / "scalers.json", scalers, fail_if_exists=True)
    write_json(run_dir / "excluded_channels.json", {"excluded_channels": excluded}, fail_if_exists=True)
    summary = {
        "seed": args.seed,
        "development_only": True,
        "descriptor_schema_residual_v1": expected_hash,
        "active_channels": {
            "rfly": sorted(k for k, v in scalers.items() if v["dataset"] == "rfly"),
            "alfa": sorted(k for k, v in scalers.items() if v["dataset"] == "alfa"),
        },
        "excluded_channel_count": len(excluded),
        "s4_run": str(s4_run),
        "g1_run": str(g1_run),
    }
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name="phaseE_scaling",
        metrics={"active_channels": len(scalers), "excluded_channels": len(excluded)},
        params={"seed": args.seed, "z_clip": 8.0, "scale": "median_raw_mad"},
    )
    update_manifest(
        run_dir,
        descriptor_schema_residual_v1=expected_hash,
        development_only=True,
        active_channels=summary["active_channels"],
        mlflow_status=tracking["status"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
