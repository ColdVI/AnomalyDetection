"""Small optional MLflow adapter with an explicit local fallback record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from residual_v1.ingest.common import write_json


def log_run(
    run_dir: str | Path,
    *,
    run_name: str,
    metrics: Mapping[str, float | int],
    params: Mapping[str, object] | None = None,
) -> dict:
    root = Path(run_dir)
    record = {
        "experiment": "residual_v1",
        "run_name": run_name,
        "metrics": dict(metrics),
        "params": dict(params or {}),
    }
    try:
        import mlflow
    except ImportError:
        record["status"] = "unavailable"
        record["reason"] = "mlflow package is not installed in this environment"
        write_json(root / "mlflow_record.json", record, fail_if_exists=True)
        return record

    mlflow.set_experiment("residual_v1")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key: str(value) for key, value in record["params"].items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in root.rglob("*"):
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path=str(artifact.parent.relative_to(root)))
        record["status"] = "logged"
        record["mlflow_run_id"] = mlflow.active_run().info.run_id
    write_json(root / "mlflow_record.json", record, fail_if_exists=True)
    return record

