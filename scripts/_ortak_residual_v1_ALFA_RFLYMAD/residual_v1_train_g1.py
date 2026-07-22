from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from residual_v1.features.spec import (
    ALFA_SPECS,
    RFLY_SPECS,
    descriptor_schema_sha256,
)
from residual_v1.ingest.common import write_json
from residual_v1.models.g1_ridge import (
    InsufficientSessionCoverage,
    fit_g1_channel,
)
from residual_v1.run import create_run_dir, update_manifest
from residual_v1.tracking import log_run


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix(feature_root: Path, channel: str) -> pd.DataFrame:
    paths = sorted(feature_root.rglob(f"{channel}.parquet"))
    if not paths:
        return pd.DataFrame()
    parts = [pd.read_parquet(path) for path in paths]
    expected = parts[0].columns.tolist()
    for path, part in zip(paths, parts, strict=True):
        if part.columns.tolist() != expected:
            raise ValueError(f"{channel}: feature columns differ at {path}")
    return pd.concat(parts, ignore_index=True)


def _session_map(silver_root: Path, development_ids: set[str]) -> dict[str, str]:
    mapping = {}
    for flight_id in sorted(development_ids):
        metadata_path = silver_root / Path(flight_id) / "flight.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = _read_json(metadata_path)
        if str(metadata.get("flight_id")) != flight_id:
            raise ValueError(f"flight metadata ID mismatch: {metadata_path}")
        mapping[flight_id] = str(metadata["session"])
    return mapping


def _skipped_report(
    *,
    channel: str,
    response: str,
    reason: str,
    matrix: pd.DataFrame,
    sessions: dict[str, str],
) -> dict:
    if matrix.empty:
        train = matrix
    else:
        train = matrix.loc[matrix["train_eligible"].astype(bool)]
    flight_ids = (
        sorted(train["flight_id"].astype(str).unique().tolist())
        if "flight_id" in train
        else []
    )
    session_ids = sorted({sessions[flight_id] for flight_id in flight_ids})
    return {
        "channel": channel,
        "response": response,
        "status": "skipped",
        "reason": reason,
        "coverage": {
            "matrix_rows": int(len(matrix)),
            "train_rows": int(len(train)),
            "matrix_flights": (
                int(matrix["flight_id"].astype(str).nunique())
                if "flight_id" in matrix
                else 0
            ),
            "train_flights": len(flight_ids),
            "train_sessions": len(session_ids),
            "session_ids": session_ids,
            "cv_folds": 0,
        },
    }


def _markdown_report(dataset: str, summary: dict, reports: dict, sanity: dict) -> str:
    lines = [
        f"# RESIDUAL-V1 G1 Ridge — {dataset.upper()} Development",
        "",
        f"Descriptor hash: {summary['descriptor_schema_residual_v1']}",
        "",
        "Model seçimi yalnız development train maskesi ve oturum-bazlı CV ile yapıldı. "
        "Test ve holdout telemetrisi okunmadı.",
        "",
    ]
    if dataset == "alfa":
        lines.extend(
            [
                "**ALFA headline iddiası tek test oturumuna dayanır ve holdout'ta "
                "R1–R5 kapsamı beklenmez.**",
                "",
            ]
        )
    lines.extend(
        [
            "| Kanal | Durum | Train satır | Uçuş | Oturum | Fold | Alpha | CV R² | Train R² | İşaret |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for channel, report in reports.items():
        coverage = report["coverage"]
        sign_status = sanity.get(channel, {}).get("status", "n/a")
        lines.append(
            "| {channel} | {status} | {rows} | {flights} | {sessions} | {folds} | "
            "{alpha} | {cv_r2} | {train_r2} | {sign} |".format(
                channel=channel,
                status=report["status"],
                rows=coverage["train_rows"],
                flights=coverage["train_flights"],
                sessions=coverage["train_sessions"],
                folds=coverage["cv_folds"],
                alpha=report.get("selected_alpha", "—"),
                cv_r2=(
                    f"{report['cv_r2']:.4f}"
                    if report.get("cv_r2") is not None
                    else "—"
                ),
                train_r2=(
                    f"{report['train_r2']:.4f}"
                    if report.get("train_r2") is not None
                    else "—"
                ),
                sign=sign_status,
            )
        )
    lines.extend(["", "## Atlanan kanallar", ""])
    skipped = [
        f"- {channel}: {report['reason']}"
        for channel, report in reports.items()
        if report["status"] == "skipped"
    ]
    lines.extend(skipped or ["- Yok."])
    if dataset == "alfa":
        lines.extend(
            [
                "",
                "R6 tasarım gereği G1'e sokulmaz; Görev 5.1'de doğrudan robust-z "
                "olarak ele alınır.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RESIDUAL-V1 G1 ridge models")
    parser.add_argument("--dataset", choices=("alfa", "rfly"), required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--silver-root", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    silver_root = Path(args.silver_root)
    split_path = Path(
        args.split_manifest
        or f"artifacts/residual_v1/splits/{args.dataset}_seed{args.seed}.json"
    )
    summary_path = feature_root / "summary.json"
    feature_summary = _read_json(summary_path)
    expected_hash = descriptor_schema_sha256()
    observed_hash = feature_summary.get("descriptor_schema_residual_v1")
    if observed_hash != expected_hash:
        raise ValueError(
            f"stale descriptor schema: expected {expected_hash}, observed {observed_hash}"
        )
    if feature_summary.get("dataset") != args.dataset:
        raise ValueError("feature dataset does not match requested dataset")

    split = _read_json(split_path)
    development_ids = set(split["partitions"]["development"]["flight_ids"])
    feature_ids = {item["flight_id"] for item in feature_summary["flights"]}
    if feature_ids != development_ids:
        raise ValueError("feature root is not exactly the frozen development partition")
    sessions = _session_map(silver_root, development_ids)
    specs = ALFA_SPECS if args.dataset == "alfa" else RFLY_SPECS

    run_dir, _ = create_run_dir(
        f"phaseD_g1_ridge_{args.dataset}_seed{args.seed}",
        seed=args.seed,
        input_paths=[summary_path, split_path],
    )
    (run_dir / "models").mkdir()
    (run_dir / "residuals").mkdir()
    (run_dir / "channel_reports").mkdir()

    reports = {}
    coefficients = {}
    sanity = {}
    metrics = {}
    for spec in specs:
        matrix = _load_matrix(feature_root, spec.name)
        if spec.name.startswith("R6_"):
            report = _skipped_report(
                channel=spec.name,
                response=spec.response,
                reason="direct_channel_excluded_from_g1_per_k6",
                matrix=matrix,
                sessions=sessions,
            )
        elif matrix.empty:
            report = _skipped_report(
                channel=spec.name,
                response=spec.response,
                reason="no_feature_rows",
                matrix=matrix,
                sessions=sessions,
            )
        elif not matrix["train_eligible"].astype(bool).any():
            report = _skipped_report(
                channel=spec.name,
                response=spec.response,
                reason="no_train_eligible_rows",
                matrix=matrix,
                sessions=sessions,
            )
        else:
            unknown = set(matrix["flight_id"].astype(str)) - development_ids
            if unknown:
                raise ValueError(f"non-development flights reached G1: {sorted(unknown)}")
            try:
                fitted = fit_g1_channel(
                    matrix,
                    channel=spec.name,
                    response=spec.response,
                    session_by_flight=sessions,
                )
            except InsufficientSessionCoverage as error:
                report = _skipped_report(
                    channel=spec.name,
                    response=spec.response,
                    reason=f"insufficient_session_coverage: {error}",
                    matrix=matrix,
                    sessions=sessions,
                )
            else:
                report = fitted.report
                coefficients[spec.name] = {
                    "intercept": report["intercept"],
                    "coefficients": fitted.coefficients,
                }
                sanity[spec.name] = fitted.coefficient_sanity
                joblib.dump(fitted.model, run_dir / "models" / f"{spec.name}.joblib")
                fitted.residuals.to_parquet(
                    run_dir / "residuals" / f"{spec.name}.parquet",
                    index=False,
                )
                metrics[f"{spec.name}_cv_r2"] = report["cv_r2"]
                metrics[f"{spec.name}_train_r2"] = report["train_r2"]
                metrics[f"{spec.name}_train_rows"] = report["coverage"]["train_rows"]
                del fitted
        reports[spec.name] = report
        write_json(
            run_dir / "channel_reports" / f"{spec.name}.json",
            report,
            fail_if_exists=True,
        )
        del matrix

    trained = sum(report["status"] == "trained" for report in reports.values())
    skipped = len(reports) - trained
    result_summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "status": "trained" if trained else "no_trainable_channels",
        "descriptor_schema_residual_v1": expected_hash,
        "development_only": True,
        "trained_channels": trained,
        "skipped_channels": skipped,
        "feature_root": str(feature_root),
        "silver_root": str(silver_root),
    }
    write_json(run_dir / "summary.json", result_summary, fail_if_exists=True)
    write_json(run_dir / "coefficients.json", coefficients, fail_if_exists=True)
    write_json(run_dir / "coeff_sanity.json", sanity, fail_if_exists=True)
    write_json(run_dir / "coverage.json", reports, fail_if_exists=True)
    (run_dir / "G1_REPORT.md").write_text(
        _markdown_report(args.dataset, result_summary, reports, sanity),
        encoding="utf-8",
    )
    metrics.update({"trained_channels": trained, "skipped_channels": skipped})
    tracking = log_run(
        run_dir,
        run_name=f"phaseD_g1_ridge_{args.dataset}",
        metrics=metrics,
        params={
            "dataset": args.dataset,
            "seed": args.seed,
            "descriptor_schema_residual_v1": expected_hash,
            "alpha_grid": "0.1,1,10,100",
            "cv_group": "session",
        },
    )
    update_manifest(
        run_dir,
        descriptor_schema_residual_v1=expected_hash,
        development_only=True,
        feature_root=str(feature_root),
        silver_root=str(silver_root),
        mlflow_status=tracking["status"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
