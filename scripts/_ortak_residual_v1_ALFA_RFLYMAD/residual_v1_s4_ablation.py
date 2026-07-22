from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from residual_v1.eval.s4_ablation import command_ablation_report
from residual_v1.features.spec import RFLY_SPECS, descriptor_schema_sha256
from residual_v1.ingest.common import write_json
from residual_v1.run import create_run_dir, update_manifest
from residual_v1.tracking import log_run


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix(feature_root: Path, channel: str) -> pd.DataFrame:
    paths = sorted(feature_root.rglob(f"{channel}.parquet"))
    if not paths:
        raise FileNotFoundError(f"no feature matrices found for {channel}")
    parts = [pd.read_parquet(path) for path in paths]
    expected = parts[0].columns.tolist()
    if any(part.columns.tolist() != expected for part in parts[1:]):
        raise ValueError(f"{channel}: feature columns differ between flights")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RESIDUAL-V1 S-4 command ablation")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--g1-run", required=True)
    parser.add_argument("--split-manifest", default="artifacts/residual_v1/splits/rfly_seed11.json")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    g1_run = Path(args.g1_run)
    split_path = Path(args.split_manifest)
    feature_summary_path = feature_root / "summary.json"
    feature_summary = _read_json(feature_summary_path)
    g1_summary = _read_json(g1_run / "summary.json")
    split = _read_json(split_path)
    expected_hash = descriptor_schema_sha256()
    if feature_summary.get("dataset") != "rfly" or g1_summary.get("dataset") != "rfly":
        raise ValueError("S-4 is frozen for RFLY G1 channels only")
    if feature_summary.get("descriptor_schema_residual_v1") != expected_hash:
        raise ValueError("feature descriptor hash is stale")
    if g1_summary.get("descriptor_schema_residual_v1") != expected_hash:
        raise ValueError("G1 descriptor hash is stale")
    development_ids = set(split["partitions"]["development"]["flight_ids"])
    feature_ids = {record["flight_id"] for record in feature_summary["flights"]}
    if feature_ids != development_ids:
        raise ValueError("S-4 feature root is not exactly the frozen development partition")
    if not bool(g1_summary.get("development_only")):
        raise ValueError("S-4 refuses a G1 run not marked development_only")

    run_dir, _ = create_run_dir(
        f"phaseE_s4_ablation_rfly_seed{args.seed}",
        seed=args.seed,
        input_paths=[feature_summary_path, g1_run / "summary.json", split_path],
    )
    (run_dir / "channel_reports").mkdir()
    reports: dict[str, dict] = {}
    for spec in RFLY_SPECS:
        source_report = _read_json(g1_run / "channel_reports" / f"{spec.name}.json")
        if source_report.get("status") != "trained":
            reports[spec.name] = {
                "gate": "S-4",
                "channel": spec.name,
                "status": "not_evaluable",
                "reason": "model_unavailable",
                "source_model_status": source_report.get("status"),
            }
        else:
            matrix = _load_matrix(feature_root, spec.name)
            unknown = set(matrix["flight_id"].astype(str)) - development_ids
            if unknown:
                raise ValueError(f"non-development flights reached S-4: {sorted(unknown)}")
            reports[spec.name] = command_ablation_report(
                matrix,
                spec=spec,
                selected_alpha=float(source_report["selected_alpha"]),
                full_feature_columns=source_report["feature_columns"],
            )
        write_json(
            run_dir / "channel_reports" / f"{spec.name}.json",
            reports[spec.name],
            fail_if_exists=True,
        )

    flags = {
        "gate": "S-4",
        "dataset": "rfly",
        "development_only": True,
        "threshold": 1.15,
        "channels": reports,
        "flagged_channels": sorted(
            channel for channel, report in reports.items() if report.get("flagged")
        ),
        "decision_eligible_channels": sorted(
            channel for channel, report in reports.items() if report.get("status") == "passed"
        ),
    }
    write_json(run_dir / "flags.json", flags, fail_if_exists=True)
    summary = {
        "dataset": "rfly",
        "seed": args.seed,
        "development_only": True,
        "descriptor_schema_residual_v1": expected_hash,
        "g1_run": str(g1_run),
        "feature_root": str(feature_root),
        "passed_channels": flags["decision_eligible_channels"],
        "flagged_channels": flags["flagged_channels"],
        "not_evaluable_channels": sorted(
            channel for channel, report in reports.items() if report["status"] == "not_evaluable"
        ),
    }
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    lines = [
        "# RESIDUAL-V1 S-4 — RFLY Development",
        "",
        "Komut özellikleri çıkarıldı; aynı seçilmiş ridge alpha ile yalnız bağlam özellikleri yeniden fit edildi.",
        "Test ve holdout okunmadı.",
        "",
        "| Kanal | Durum | Var(sakat)/Var(tam) | Eşik |",
        "|---|---:|---:|---:|",
    ]
    for channel, report in reports.items():
        ratio = report.get("variance_ratio")
        lines.append(f"| {channel} | {report['status']} | {ratio if ratio is not None else '—'} | 1.15 |")
    (run_dir / "S4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tracking = log_run(
        run_dir,
        run_name="phaseE_s4_ablation_rfly",
        metrics={
            "passed_channels": len(summary["passed_channels"]),
            "flagged_channels": len(summary["flagged_channels"]),
        },
        params={"dataset": "rfly", "seed": args.seed, "threshold": 1.15},
    )
    update_manifest(
        run_dir,
        descriptor_schema_residual_v1=expected_hash,
        development_only=True,
        g1_run=str(g1_run),
        feature_root=str(feature_root),
        mlflow_status=tracking["status"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
