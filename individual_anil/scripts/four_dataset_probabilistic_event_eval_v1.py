"""Frozen-checkpoint window, event, and flight evaluation for four UAV datasets.

This script never trains a model.  It accepts a completed
``four_dataset_probabilistic_gpu_v2`` run, verifies its contract, exports the
validation/test window scores with timestamps, and evaluates the frozen event
policy grid from ``configs/four_dataset_probabilistic_event_eval_v1.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import four_dataset_probabilistic_gpu_v1_runner as core
except ModuleNotFoundError:  # pragma: no cover - import path used by tests
    from scripts import four_dataset_probabilistic_gpu_v1_runner as core


EVAL_CONFIG_PATH = Path("configs/four_dataset_probabilistic_event_eval_v1.json")
BASE_CONFIG_PATH = Path("configs/four_dataset_probabilistic_gpu_v2.json")
BASE_PREREG_PATH = Path("docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_PREREG_20260727.md")
BASE_NAMESPACE = "four_dataset_probabilistic_gpu_v2"
DATASETS = ("alfa", "uav_attack", "uav_sead", "rflymad")
ALFA_SILVER_PATH = Path("data/silver/alfa_silver.parquet")
ALFA_BRONZE_PATH = Path("data/objectstore/bronze/alfa/processed.zip")
UAV_SEAD_SILVER_PATH = Path("data/silver/uav_sead_silver.parquet")
UAV_SEAD_LABELS_PATH = Path("data/objectstore/bronze/uav_sead/labels.json")


class EventEvalV1Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EventEvalV1Error(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EventEvalV1Error(f"Expected JSON object in {path}")
    return value


def _configure_core() -> None:
    core.CONFIG_PATH = BASE_CONFIG_PATH
    core.PREREG_PATH = BASE_PREREG_PATH
    core.EXPECTED_NAMESPACE = BASE_NAMESPACE


def _load_eval_config(root: Path) -> dict[str, Any]:
    path = root / EVAL_CONFIG_PATH
    config = _read_json(path)
    if config.get("candidate_namespace") != "four_dataset_probabilistic_event_eval_v1":
        raise EventEvalV1Error("Unexpected event-evaluation namespace")
    frozen = config["frozen_inputs"]
    for key in ("base_config", "split_manifest"):
        evidence = frozen[key]
        target = root / evidence["path"]
        if not target.is_file():
            raise EventEvalV1Error(f"Missing frozen input: {target}")
        actual = _sha256(target)
        if actual != str(evidence["sha256"]).lower():
            raise EventEvalV1Error(
                f"Frozen input hash mismatch for {target}: {actual}"
            )
    if not all(config["prohibitions"].values()):
        raise EventEvalV1Error("Every event-evaluation prohibition must remain true")
    return config


def _scaler_from_dict(value: dict[str, Any]) -> core.RobustScaler:
    features = tuple(map(str, value["features"]))
    median = value["median"]
    scale = value["scale"]
    if set(features) != set(map(str, median)) or set(features) != set(map(str, scale)):
        raise EventEvalV1Error("Scaler feature/median/scale keys do not agree")
    return core.RobustScaler(
        features=features,
        median=np.asarray([median[name] for name in features], dtype=np.float64),
        scale=np.asarray([scale[name] for name in features], dtype=np.float64),
        excluded=tuple(map(str, value.get("excluded", []))),
        clip=float(value["clip"]),
    )


@dataclass(frozen=True)
class AcceptedRun:
    config: dict[str, Any]
    spec: dict[str, Any]
    access: core.DatasetAccess
    scaler: core.RobustScaler
    report: dict[str, Any]
    model_payload: dict[str, Any]
    acceptance: dict[str, Any]


@dataclass(frozen=True)
class TruthContext:
    t0_by_source: dict[str, float]
    intervals_by_source: dict[str, tuple[tuple[float, float], ...]]
    categories_by_source: dict[str, tuple[str, ...]]
    labels_by_source: dict[str, str]
    evidence: dict[str, Any]


def _verify_run(root: Path, dataset: str, run_dir: Path) -> AcceptedRun:
    event_config = _load_eval_config(root)
    required = tuple(event_config["frozen_inputs"]["required_run_files"])
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise EventEvalV1Error(f"Run is missing required files: {missing}")

    _configure_core()
    try:
        base_config, spec = core._load_contract(root, dataset)
        access = core._load_access(root, base_config, spec, dataset)
        evidence = core._contract_evidence(root, base_config, access)
    except core.ProbabilisticV1Error as exc:
        raise EventEvalV1Error(f"Base contract verification failed: {exc}") from exc

    manifest = _read_json(run_dir / "run_manifest.json")
    report = _read_json(run_dir / "training_report.json")
    scaler_value = _read_json(run_dir / "scaler.json")
    checks = {
        "manifest_namespace": manifest.get("candidate_namespace") == BASE_NAMESPACE,
        "report_namespace": report.get("candidate_namespace") == BASE_NAMESPACE,
        "manifest_dataset": manifest.get("dataset") == dataset,
        "report_dataset": report.get("dataset") == dataset,
        "manifest_complete": manifest.get("status") == "complete",
        "manifest_completed_epochs": int(manifest.get("completed_epochs", -1))
        == int(base_config["common"]["epochs"]),
        "manifest_contract": manifest.get("contract_sha256") == evidence["contract_sha256"],
        "report_contract": report.get("contract_sha256") == evidence["contract_sha256"],
        "completed_epochs": int(report.get("completed_epochs", -1))
        == int(base_config["common"]["epochs"]),
        "magnitude_gate": report.get("magnitude_domination_flagged_at_0_8") is False,
    }

    recomputed_scaler = core._fit_scaler_access(
        access, base_config["common"], str(spec["normal_label"])
    )
    checks["scaler_matches_frozen_train"] = (
        core._canonical_sha(recomputed_scaler.to_dict())
        == core._canonical_sha(scaler_value)
    )

    artifact_hash_checks: dict[str, bool] = {}
    declared_hashes = report.get("artifact_hashes", {})
    for name in required:
        if name == "training_report.json":
            continue
        declared = declared_hashes.get(name)
        artifact_hash_checks[name] = bool(
            isinstance(declared, dict)
            and declared.get("sha256") == _sha256(run_dir / name)
            and int(declared.get("bytes", -1)) == (run_dir / name).stat().st_size
        )
    checks["declared_artifact_hashes"] = all(artifact_hash_checks.values())

    try:
        model_payload = core._checkpoint_load(run_dir / "model_state.pt", torch.device("cpu"))
    except Exception as exc:  # torch raises several format-specific exceptions
        raise EventEvalV1Error(f"Cannot load model_state.pt: {exc}") from exc
    if not isinstance(model_payload, dict):
        raise EventEvalV1Error("model_state.pt payload is not a mapping")
    checks["model_dataset"] = model_payload.get("dataset") == dataset
    checks["model_contract"] = model_payload.get("contract_sha256") == evidence["contract_sha256"]
    checks["model_features"] = list(model_payload.get("features", [])) == list(
        scaler_value["features"]
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    acceptance = {
        "schema_version": 1,
        "event_namespace": event_config["candidate_namespace"],
        "base_model_namespace": BASE_NAMESPACE,
        "dataset": dataset,
        "run_dir": str(run_dir),
        "accepted": not failed,
        "checks": checks,
        "artifact_hash_checks": artifact_hash_checks,
        "failed_checks": failed,
        "contract_sha256": evidence["contract_sha256"],
    }
    if failed:
        raise EventEvalV1Error(f"Run artifact verification failed: {failed}")
    return AcceptedRun(
        config=base_config,
        spec=spec,
        access=access,
        scaler=_scaler_from_dict(scaler_value),
        report=report,
        model_payload=model_payload,
        acceptance=acceptance,
    )


def _rfly_metadata(root: Path, accepted: AcceptedRun) -> pd.DataFrame:
    manifest_path = root / accepted.spec["manifest_path"]
    wanted = [
        "canonical_case_id",
        "domain",
        "flight_status",
        "fault_family",
        "truth_source",
    ]
    schema = set(pd.read_parquet(manifest_path, engine="pyarrow").columns)
    columns = [name for name in wanted if name in schema]
    frame = pd.read_parquet(manifest_path, columns=columns)
    return frame.drop_duplicates("canonical_case_id").set_index("canonical_case_id")


def _truth_file_evidence(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    evidence = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise EventEvalV1Error(f"Required truth input is missing: {path}")
        evidence.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return evidence


def _load_alfa_truth_context(root: Path) -> TruthContext:
    inputs = _truth_file_evidence(root, (ALFA_SILVER_PATH, ALFA_BRONZE_PATH))
    silver = pd.read_parquet(root / ALFA_SILVER_PATH, columns=["source_id", "ts_ns"])
    t0 = (
        silver.groupby("source_id", sort=False)["ts_ns"]
        .min()
        .astype(float)
        .to_dict()
    )
    onsets: dict[str, list[float]] = {}
    categories: dict[str, set[str]] = {}
    active_files = 0
    with zipfile.ZipFile(root / ALFA_BRONZE_PATH) as archive:
        for member in archive.namelist():
            if "failure_status-" not in member or not member.lower().endswith(".csv"):
                continue
            source_id = Path(member).parent.name
            try:
                status = pd.read_csv(
                    BytesIO(archive.read(member)), usecols=["%time", "field.data"]
                )
            except (KeyError, ValueError) as exc:
                raise EventEvalV1Error(
                    f"Malformed ALFA failure-status file {member}: {exc}"
                ) from exc
            active = status[pd.to_numeric(status["field.data"], errors="coerce") != 0]
            if active.empty:
                continue
            if source_id not in t0:
                continue
            onset_ns = float(pd.to_numeric(active["%time"], errors="raise").min())
            onset_s = (onset_ns - float(t0[source_id])) / 1e9
            onsets.setdefault(source_id, []).append(onset_s)
            category = Path(member).stem.split("failure_status-", 1)[-1]
            categories.setdefault(source_id, set()).add(category)
            active_files += 1
    intervals = {
        source_id: ((min(values), float("inf")),)
        for source_id, values in onsets.items()
    }
    return TruthContext(
        t0_by_source=t0,
        intervals_by_source=intervals,
        categories_by_source={
            source_id: tuple(sorted(values)) for source_id, values in categories.items()
        },
        labels_by_source={},
        evidence={
            "dataset": "alfa",
            "status": "mapped",
            "input_files": inputs,
            "time_mapping": "target_time_s = (raw_failure_status_%time - Silver_source_min_ts_ns) / 1e9",
            "boundary_semantics": "inclusive onset; a latched airframe failure remains active through the final Gold sample",
            "mapped_sources": len(intervals),
            "active_failure_status_files": active_files,
        },
    )


def _uav_sead_ranges(
    labels: dict[str, Any],
) -> tuple[
    dict[str, tuple[tuple[float, float], ...]],
    dict[str, tuple[str, ...]],
    dict[str, str],
]:
    ranges: dict[str, tuple[tuple[float, float], ...]] = {}
    categories: dict[str, tuple[str, ...]] = {}
    flight_labels: dict[str, str] = {}
    for source_id, metadata in labels.items():
        if not isinstance(metadata, dict):
            raise EventEvalV1Error(f"Invalid UAV-SEAD label record for {source_id}")
        spans: list[tuple[float, float]] = []
        names: set[str] = set()
        for annotation in metadata.get("ranges", []):
            if not isinstance(annotation, list):
                continue
            for entry in annotation:
                if not isinstance(entry, list) or len(entry) != 2:
                    continue
                category, intervals = entry
                names.add(str(category))
                for interval in intervals:
                    if not isinstance(interval, list) or len(interval) != 2:
                        raise EventEvalV1Error(
                            f"Invalid UAV-SEAD interval for {source_id}: {interval!r}"
                        )
                    start, end = map(float, interval)
                    if not (math.isfinite(start) and math.isfinite(end) and end >= start):
                        raise EventEvalV1Error(
                            f"Invalid UAV-SEAD interval bounds for {source_id}: {interval!r}"
                        )
                    spans.append((start, end))
        ranges[str(source_id)] = tuple(sorted(set(spans)))
        categories[str(source_id)] = tuple(sorted(names))
        flight_labels[str(source_id)] = str(metadata.get("label"))
    return ranges, categories, flight_labels


def _load_uav_sead_truth_context(root: Path) -> TruthContext:
    inputs = _truth_file_evidence(root, (UAV_SEAD_SILVER_PATH, UAV_SEAD_LABELS_PATH))
    silver = pd.read_parquet(
        root / UAV_SEAD_SILVER_PATH, columns=["source_id", "timestamp"]
    )
    t0 = (
        silver.groupby("source_id", sort=False)["timestamp"]
        .min()
        .astype(float)
        .to_dict()
    )
    labels = _read_json(root / UAV_SEAD_LABELS_PATH)
    intervals, categories, flight_labels = _uav_sead_ranges(labels)
    mapped = sum(bool(value) for value in intervals.values())
    return TruthContext(
        t0_by_source=t0,
        intervals_by_source=intervals,
        categories_by_source=categories,
        labels_by_source=flight_labels,
        evidence={
            "dataset": "uav_sead",
            "status": "mapped",
            "input_files": inputs,
            "time_mapping": "absolute_us = Silver_source_min_timestamp + target_time_s * 1e6",
            "boundary_semantics": "dataset-provided anomaly intervals are inclusive and unioned across annotation channels",
            "label_records": len(flight_labels),
            "sources_with_intervals": mapped,
        },
    )


def _load_truth_context(root: Path, dataset: str) -> TruthContext:
    if dataset == "alfa":
        return _load_alfa_truth_context(root)
    if dataset == "uav_sead":
        return _load_uav_sead_truth_context(root)
    return TruthContext({}, {}, {}, {}, {"dataset": dataset, "status": "native_or_unavailable"})


def _attach_truth(
    root: Path,
    accepted: AcceptedRun | None,
    dataset: str,
    source_id: str,
    label: str,
    times: np.ndarray,
    normal_label: str,
    rfly_meta: pd.DataFrame | None,
    truth_context: TruthContext,
) -> dict[str, Any]:
    count = len(times)
    result: dict[str, Any] = {
        "truth_available": np.full(count, label == normal_label, dtype=bool),
        "truth_active": np.zeros(count, dtype=bool),
        "fault_family": np.full(count, label, dtype=object),
        "domain": np.full(count, None, dtype=object),
        "flight_status": np.full(count, None, dtype=object),
        "truth_source": np.full(count, "normal_flight_label" if label == normal_label else None, dtype=object),
    }
    if dataset == "alfa":
        if label == normal_label:
            return result
        intervals = truth_context.intervals_by_source.get(source_id, ())
        if not intervals:
            result["truth_source"] = np.full(count, "alfa_truth_unavailable", dtype=object)
            return result
        onset_s = min(start for start, _ in intervals)
        result["truth_available"] = np.ones(count, dtype=bool)
        result["truth_active"] = times >= onset_s
        result["truth_source"] = np.full(
            count, "alfa_failure_status_first_active_latched", dtype=object
        )
        categories = truth_context.categories_by_source.get(source_id, ())
        if categories:
            result["fault_family"] = np.full(count, "+".join(categories), dtype=object)
        return result

    if dataset == "uav_sead":
        recorded_label = truth_context.labels_by_source.get(source_id)
        if recorded_label is None or source_id not in truth_context.t0_by_source:
            result["truth_source"] = np.full(count, "uav_sead_mapping_unavailable", dtype=object)
            return result
        if recorded_label != label:
            result["truth_source"] = np.full(count, "uav_sead_label_mismatch", dtype=object)
            return result
        spans = truth_context.intervals_by_source.get(source_id, ())
        if label != normal_label and not spans:
            result["truth_source"] = np.full(count, "uav_sead_ranges_unavailable", dtype=object)
            return result
        absolute_us = float(truth_context.t0_by_source[source_id]) + times * 1e6
        active = np.zeros(count, dtype=bool)
        for start, end in spans:
            active |= (absolute_us >= start) & (absolute_us <= end)
        result["truth_available"] = np.ones(count, dtype=bool)
        result["truth_active"] = active
        result["truth_source"] = np.full(
            count, "uav_sead_labels_json_absolute_us", dtype=object
        )
        categories = truth_context.categories_by_source.get(source_id, ())
        if categories:
            result["fault_family"] = np.full(count, "+".join(categories), dtype=object)
        return result

    if dataset != "rflymad":
        return result

    if accepted is None or accepted.access.source_paths is None:
        raise EventEvalV1Error("RflyMAD source paths are unavailable")
    path = accepted.access.source_paths[source_id]
    truth = pd.read_parquet(
        path,
        columns=[
            "t_rel_s",
            "fault_active",
            "condition_active",
            "truth_source",
            "truth_crosscheck_disagreement_v2",
        ],
    ).drop_duplicates("t_rel_s", keep="last")
    lookup = truth.set_index("t_rel_s")
    aligned = lookup.reindex(pd.Index(times, name="t_rel_s"))
    if aligned[["fault_active", "condition_active"]].isna().any().any():
        raise EventEvalV1Error(f"RflyMAD truth/time alignment failed for {source_id}")
    disagreement = aligned["truth_crosscheck_disagreement_v2"].fillna(False).astype(bool)
    available = ~disagreement
    result["truth_available"] = available.to_numpy(bool)
    result["truth_active"] = (
        aligned["fault_active"].fillna(False).astype(bool)
        | aligned["condition_active"].fillna(False).astype(bool)
    ).to_numpy(bool)
    result["truth_source"] = aligned["truth_source"].astype("string").to_numpy(object)
    if rfly_meta is not None and source_id in rfly_meta.index:
        row = rfly_meta.loc[source_id]
        for column in ("domain", "flight_status", "fault_family"):
            if column in row.index:
                result[column] = np.full(count, row[column], dtype=object)
    return result


def _score_frame(
    root: Path,
    accepted: AcceptedRun,
    dataset: str,
    role: str,
    scored: core.ScoreSet,
    truth_context: TruthContext,
) -> pd.DataFrame:
    normal_label = str(accepted.spec["normal_label"])
    rfly_meta = _rfly_metadata(root, accepted) if dataset == "rflymad" else None
    records: list[pd.DataFrame] = []
    for code, (source_id, label) in enumerate(zip(scored.sources, scored.labels)):
        index = scored.source_codes == code
        times = scored.target_times[index].astype(np.float64, copy=False)
        truth = _attach_truth(
            root,
            accepted,
            dataset,
            source_id,
            label,
            times,
            normal_label,
            rfly_meta,
            truth_context,
        )
        records.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "role": role,
                    "source_id": source_id,
                    "target_time_s": times,
                    "score": scored.scores[index].astype(np.float64, copy=False),
                    "target_magnitude": scored.magnitudes[index].astype(np.float64, copy=False),
                    "flight_label": label,
                    "is_anomaly_flight": label != normal_label,
                    **truth,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def _build_model(accepted: AcceptedRun, device: torch.device) -> core.GaussianForecaster:
    common = accepted.config["common"]
    model = core.GaussianForecaster(
        input_size=len(accepted.scaler.features) * 2 + 1,
        channels=len(accepted.scaler.features),
        common=common,
    ).to(device)
    try:
        model.load_state_dict(accepted.model_payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise EventEvalV1Error(f"Model state/architecture mismatch: {exc}") from exc
    model.eval()
    return model


def _export_scores(
    root: Path,
    dataset: str,
    run_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    accepted = _verify_run(root, dataset, run_dir)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise EventEvalV1Error("CUDA requested but unavailable")
    device = torch.device(device_name)
    model = _build_model(accepted, device)
    truth_context = _load_truth_context(root, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "artifact_acceptance.json", accepted.acceptance)

    ledgers: dict[str, Any] = {}
    for role in ("val", "test"):
        scored = core._score_sources(
            accepted.access,
            role,
            accepted.scaler,
            model,
            accepted.config["common"],
            device,
            output_dir,
            stage=f"event_v1_{role}_score_export",
        )
        frame = _score_frame(root, accepted, dataset, role, scored, truth_context)
        path = output_dir / f"{role}_window_scores.parquet"
        frame.to_parquet(path, index=False)
        ledgers[role] = {
            "path": path.name,
            "rows": int(len(frame)),
            "sources": int(frame["source_id"].nunique()),
            "truth_available_rows": int(frame["truth_available"].sum()),
            "truth_active_rows": int(frame["truth_active"].sum()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    report = {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_event_eval_v1",
        "dataset": dataset,
        "device": str(device),
        "base_run_dir": str(run_dir),
        "artifact_acceptance": accepted.acceptance,
        "truth_mapping_evidence": truth_context.evidence,
        "ledgers": ledgers,
    }
    _write_json_atomic(output_dir / "score_export_report.json", report)
    return report


def _score_payload_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "role",
        "source_id",
        "target_time_s",
        "score",
        "target_magnitude",
        "flight_label",
        "is_anomaly_flight",
    ]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EventEvalV1Error(f"Score ledger is missing immutable columns: {missing}")
    hashed = pd.util.hash_pandas_object(frame[columns], index=True).to_numpy(np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _truth_mapping_summary(
    dataset: str,
    context: TruthContext,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for role, ledger in frames.items():
        for source_id, frame in ledger.groupby("source_id", sort=True):
            anomaly = bool(frame["is_anomaly_flight"].iloc[0])
            available = frame["truth_available"].to_numpy(bool)
            active = frame["truth_active"].to_numpy(bool)
            records.append(
                {
                    "role": role,
                    "source_id": str(source_id),
                    "flight_label": str(frame["flight_label"].iloc[0]),
                    "is_anomaly_flight": anomaly,
                    "truth_fully_available": bool(available.all()),
                    "truth_available_rows": int(available.sum()),
                    "truth_active_rows": int(active.sum()),
                    "truth_sources": sorted(
                        str(value)
                        for value in frame["truth_source"].dropna().unique().tolist()
                    ),
                    "target_time_min_s": float(frame["target_time_s"].min()),
                    "target_time_max_s": float(frame["target_time_s"].max()),
                }
            )
    mapped_anomaly = sum(
        row["is_anomaly_flight"] and row["truth_fully_available"] for row in records
    )
    unavailable = [
        row["source_id"]
        for row in records
        if row["is_anomaly_flight"] and not row["truth_fully_available"]
    ]
    return {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_event_eval_v1",
        "dataset": dataset,
        "evidence": context.evidence,
        "sources": len(records),
        "anomaly_sources": sum(row["is_anomaly_flight"] for row in records),
        "mapped_anomaly_sources": mapped_anomaly,
        "unavailable_anomaly_sources": unavailable,
        "source_records": records,
    }


def _refresh_truth(root: Path, dataset: str, ledger_dir: Path) -> dict[str, Any]:
    if dataset not in {"alfa", "uav_sead"}:
        raise EventEvalV1Error("refresh-truth is only needed for ALFA or UAV-SEAD")
    context = _load_truth_context(root, dataset)
    normal_label = str(_read_json(root / BASE_CONFIG_PATH)["datasets"][dataset]["normal_label"])
    frames: dict[str, pd.DataFrame] = {}
    ledger_records: dict[str, Any] = {}
    for role in ("val", "test"):
        path = ledger_dir / f"{role}_window_scores.parquet"
        if not path.is_file():
            raise EventEvalV1Error(f"Score ledger is missing: {path}")
        frame = pd.read_parquet(path)
        if set(frame["dataset"].astype(str).unique()) != {dataset}:
            raise EventEvalV1Error(f"{role} ledger dataset mismatch")
        before = _score_payload_sha256(frame)
        for source_id, indexes in frame.groupby("source_id", sort=False).groups.items():
            position = np.asarray(indexes, dtype=np.int64)
            label_values = frame.loc[position, "flight_label"].astype(str).unique()
            if len(label_values) != 1:
                raise EventEvalV1Error(f"Multiple flight labels for {source_id}")
            truth = _attach_truth(
                root,
                None,
                dataset,
                str(source_id),
                str(label_values[0]),
                frame.loc[position, "target_time_s"].to_numpy(float),
                normal_label,
                None,
                context,
            )
            for column, values in truth.items():
                frame.loc[position, column] = values
        after = _score_payload_sha256(frame)
        if before != after:
            raise EventEvalV1Error(f"Immutable score payload changed while attaching {role} truth")
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
        frames[role] = frame
        ledger_records[role] = {
            "path": path.name,
            "rows": len(frame),
            "sources": int(frame["source_id"].nunique()),
            "immutable_score_payload_sha256": after,
            "truth_available_rows": int(frame["truth_available"].sum()),
            "truth_active_rows": int(frame["truth_active"].sum()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    report = _truth_mapping_summary(dataset, context, frames)
    report["ledgers"] = ledger_records
    _write_json_atomic(ledger_dir / "truth_mapping_report.json", report)
    return report


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float
    alarm_s: float


def _contiguous_true_intervals(
    times: np.ndarray,
    active: np.ndarray,
    max_gap_s: float,
    minimum_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> list[Interval]:
    times = np.asarray(times, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if len(times) != len(active):
        raise EventEvalV1Error("Time/boolean array length mismatch")
    if len(times) == 0:
        return []
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise EventEvalV1Error("Times must be finite and nondecreasing")

    positions = np.flatnonzero(active)
    raw: list[tuple[int, int]] = []
    if len(positions):
        start = previous = int(positions[0])
        for position in positions[1:]:
            position = int(position)
            adjacent = position == previous + 1
            gap_ok = float(times[position] - times[previous]) <= max_gap_s
            if adjacent and gap_ok:
                previous = position
                continue
            raw.append((start, previous))
            start = previous = position
        raw.append((start, previous))

    qualified: list[Interval] = []
    for start, end in raw:
        duration = float(times[end] - times[start])
        if duration + 1e-12 < minimum_duration_s:
            continue
        alarm_target = float(times[start] + minimum_duration_s)
        alarm_index = start + int(np.searchsorted(times[start : end + 1], alarm_target, side="left"))
        alarm_index = min(alarm_index, end)
        qualified.append(
            Interval(float(times[start]), float(times[end]), float(times[alarm_index]))
        )

    merged: list[Interval] = []
    for interval in qualified:
        if merged and interval.start_s - merged[-1].end_s <= merge_gap_s:
            previous = merged[-1]
            merged[-1] = Interval(
                previous.start_s,
                max(previous.end_s, interval.end_s),
                previous.alarm_s,
            )
        else:
            merged.append(interval)
    return merged


def _exposure_seconds(times: np.ndarray, max_gap_s: float) -> float:
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2:
        return 0.0
    delta = np.diff(times)
    return float(delta[(delta >= 0.0) & (delta <= max_gap_s)].sum())


def _intersection_seconds(left: Iterable[Interval], right: Iterable[Interval]) -> float:
    total = 0.0
    for a in left:
        for b in right:
            total += max(0.0, min(a.end_s, b.end_s) - max(a.alarm_s, b.start_s))
    return total


def _eventize_source(
    frame: pd.DataFrame,
    threshold: float,
    persistence_s: float,
    merge_gap_s: float,
    max_gap_s: float,
) -> list[Interval]:
    ordered = frame.sort_values("target_time_s", kind="mergesort")
    return _contiguous_true_intervals(
        ordered["target_time_s"].to_numpy(float),
        ordered["score"].to_numpy(float) >= threshold,
        max_gap_s=max_gap_s,
        minimum_duration_s=persistence_s,
        merge_gap_s=merge_gap_s,
    )


def _false_event_rate(
    validation: pd.DataFrame,
    threshold: float,
    persistence_s: float,
    merge_gap_s: float,
    max_gap_s: float,
) -> tuple[float | None, int, float]:
    event_count = 0
    exposure_s = 0.0
    for _, frame in validation.groupby("source_id", sort=True):
        if bool(frame["is_anomaly_flight"].iloc[0]):
            raise EventEvalV1Error("Validation ledger is not normal-only")
        times = frame["target_time_s"].to_numpy(float)
        exposure_s += _exposure_seconds(times, max_gap_s)
        event_count += len(
            _eventize_source(frame, threshold, persistence_s, merge_gap_s, max_gap_s)
        )
    hours = exposure_s / 3600.0
    return ((event_count / hours) if hours > 0 else None), event_count, hours


def _safe_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(truth)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
    }


def _flight_metrics(test: pd.DataFrame) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for source_id, frame in test.groupby("source_id", sort=True):
        score = frame["score"].to_numpy(float)
        top_means = {}
        for fraction, name in (
            (0.005, "top_0_5pct_mean"),
            (0.01, "top_1pct_mean"),
            (0.02, "top_2pct_mean"),
        ):
            count = max(1, int(math.ceil(fraction * len(score))))
            top = np.partition(score, len(score) - count)[-count:]
            top_means[name] = float(np.mean(top))
        records.append(
            {
                "source_id": source_id,
                "truth": int(bool(frame["is_anomaly_flight"].iloc[0])),
                "max": float(np.max(score)),
                "mean": float(np.mean(score)),
                "q99": float(np.quantile(score, 0.99)),
                "q995": float(np.quantile(score, 0.995)),
                **top_means,
            }
        )
    flights = pd.DataFrame.from_records(records)
    truth = flights["truth"].to_numpy(np.int8)
    return {
        name: _safe_metrics(truth, flights[name].to_numpy(float))
        for name in (
            "max",
            "mean",
            "q99",
            "q995",
            "top_0_5pct_mean",
            "top_1pct_mean",
            "top_2pct_mean",
        )
    }


def _percentile_interval(
    values: list[float], confidence_level: float
) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return None
    tail = (1.0 - confidence_level) / 2.0
    return [float(value) for value in np.quantile(finite, [tail, 1.0 - tail])]


def _bootstrap_source_intervals(
    records: list[dict[str, Any]],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any] | None:
    if replicates <= 0:
        return None
    normal = [row for row in records if not row["is_anomaly_flight"]]
    anomaly = [row for row in records if row["is_anomaly_flight"]]
    if not records:
        return None
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "event_recall": [],
        "false_events_per_normal_hour": [],
        "median_detection_delay_s": [],
        "p90_detection_delay_s": [],
        "range_precision_time_weighted": [],
        "range_recall_time_weighted": [],
    }
    for _ in range(replicates):
        sampled_normal = (
            [normal[index] for index in rng.integers(0, len(normal), len(normal))]
            if normal
            else []
        )
        sampled_anomaly = (
            [anomaly[index] for index in rng.integers(0, len(anomaly), len(anomaly))]
            if anomaly
            else []
        )
        sampled = sampled_normal + sampled_anomaly
        events = sum(row["event_count"] for row in sampled_anomaly)
        if events:
            values["event_recall"].append(
                sum(row["detected_events"] for row in sampled_anomaly) / events
            )
        normal_hours = sum(row["normal_exposure_s"] for row in sampled_normal) / 3600.0
        if normal_hours > 0:
            values["false_events_per_normal_hour"].append(
                sum(row["false_events"] for row in sampled_normal) / normal_hours
            )
        delays = [
            delay
            for row in sampled_anomaly
            for delay in row["detection_delays_s"]
        ]
        if delays:
            values["median_detection_delay_s"].append(float(np.median(delays)))
            values["p90_detection_delay_s"].append(float(np.quantile(delays, 0.9)))
        predicted = sum(row["predicted_duration_s"] for row in sampled)
        truth = sum(row["truth_duration_s"] for row in sampled_anomaly)
        overlap = sum(row["overlap_duration_s"] for row in sampled_anomaly)
        if predicted > 0:
            values["range_precision_time_weighted"].append(overlap / predicted)
        if truth > 0:
            values["range_recall_time_weighted"].append(overlap / truth)
    return {
        "unit": "source_id",
        "stratification": "normal_vs_anomaly_flight",
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "percentile_intervals": {
            name: _percentile_interval(metric_values, confidence_level)
            for name, metric_values in values.items()
        },
    }


def _evaluate_point(
    test: pd.DataFrame,
    threshold: float,
    persistence_s: float,
    merge_gap_s: float,
    max_gap_s: float,
    bootstrap_replicates: int = 0,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260728,
) -> dict[str, Any]:
    false_events = 0
    normal_exposure_s = 0.0
    event_count = 0
    detected = 0
    delays: list[float] = []
    pred_duration = 0.0
    eligible_exposure_s = 0.0
    predicted_event_count = 0
    longest_alarm_event_s = 0.0
    truth_duration = 0.0
    overlap_duration = 0.0
    unavailable_anomaly_sources = 0
    source_records: list[dict[str, Any]] = []

    for source_id, frame in test.groupby("source_id", sort=True):
        frame = frame.sort_values("target_time_s", kind="mergesort")
        times = frame["target_time_s"].to_numpy(float)
        pred = _eventize_source(frame, threshold, persistence_s, merge_gap_s, max_gap_s)
        anomaly_flight = bool(frame["is_anomaly_flight"].iloc[0])
        source_normal_exposure_s = 0.0
        source_false_events = 0
        if not anomaly_flight:
            source_false_events = len(pred)
            source_normal_exposure_s = _exposure_seconds(times, max_gap_s)
            false_events += source_false_events
            normal_exposure_s += source_normal_exposure_s

        available = frame["truth_available"].to_numpy(bool)
        if anomaly_flight and not bool(available.all()):
            unavailable_anomaly_sources += 1
            continue
        eligible_exposure_s += _exposure_seconds(times, max_gap_s)
        durations = [max(0.0, item.end_s - item.alarm_s) for item in pred]
        pred_duration += sum(durations)
        predicted_event_count += len(pred)
        if durations:
            longest_alarm_event_s = max(longest_alarm_event_s, max(durations))
        active = frame["truth_active"].to_numpy(bool) & available
        truth = _contiguous_true_intervals(times, active, max_gap_s=max_gap_s)
        source_truth_duration = sum(
            max(0.0, item.end_s - item.start_s) for item in truth
        )
        source_overlap_duration = _intersection_seconds(pred, truth)
        truth_duration += source_truth_duration
        overlap_duration += source_overlap_duration
        source_detected = 0
        source_delays: list[float] = []
        for event in truth:
            event_count += 1
            inside = [
                alarm.alarm_s
                for alarm in pred
                if event.start_s <= alarm.alarm_s <= event.end_s
            ]
            if inside:
                detected += 1
                source_detected += 1
                delay = min(inside) - event.start_s
                delays.append(delay)
                source_delays.append(delay)
        source_records.append(
            {
                "source_id": str(source_id),
                "is_anomaly_flight": anomaly_flight,
                "event_count": len(truth),
                "detected_events": source_detected,
                "false_events": source_false_events,
                "normal_exposure_s": source_normal_exposure_s,
                "detection_delays_s": source_delays,
                "predicted_duration_s": sum(durations),
                "truth_duration_s": source_truth_duration,
                "overlap_duration_s": source_overlap_duration,
            }
        )

    normal_hours = normal_exposure_s / 3600.0
    result = {
        "threshold": float(threshold),
        "minimum_above_threshold_seconds": float(persistence_s),
        "event_count": event_count,
        "detected_events": detected,
        "event_recall": (detected / event_count) if event_count else None,
        "false_events": false_events,
        "normal_exposure_hours": normal_hours,
        "false_events_per_normal_hour": (
            false_events / normal_hours if normal_hours > 0 else None
        ),
        "median_detection_delay_s": float(np.median(delays)) if delays else None,
        "p90_detection_delay_s": float(np.quantile(delays, 0.9)) if delays else None,
        "range_precision_time_weighted": (
            overlap_duration / pred_duration if pred_duration > 0 else None
        ),
        "range_recall_time_weighted": (
            overlap_duration / truth_duration if truth_duration > 0 else None
        ),
        "predicted_alarm_duration_s": pred_duration,
        "truth_duration_s": truth_duration,
        "overlap_duration_s": overlap_duration,
        "alarm_time_fraction": (
            pred_duration / eligible_exposure_s if eligible_exposure_s > 0 else None
        ),
        "longest_alarm_event_seconds": longest_alarm_event_s,
        "alarm_event_count_per_observed_hour": (
            predicted_event_count / (eligible_exposure_s / 3600.0)
            if eligible_exposure_s > 0
            else None
        ),
        "unavailable_anomaly_sources": unavailable_anomaly_sources,
    }
    result["source_bootstrap"] = _bootstrap_source_intervals(
        source_records,
        bootstrap_replicates,
        confidence_level,
        bootstrap_seed,
    )
    return result


def _evaluate_ledgers(root: Path, dataset: str, ledger_dir: Path) -> dict[str, Any]:
    config = _load_eval_config(root)
    validation_path = ledger_dir / "val_window_scores.parquet"
    test_path = ledger_dir / "test_window_scores.parquet"
    if not validation_path.is_file() or not test_path.is_file():
        raise EventEvalV1Error("Score ledgers are missing; run export-scores first")
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)
    if set(validation["dataset"].astype(str).unique()) != {dataset}:
        raise EventEvalV1Error("Validation ledger dataset mismatch")
    if set(test["dataset"].astype(str).unique()) != {dataset}:
        raise EventEvalV1Error("Test ledger dataset mismatch")

    event_grid = config["event_policy_grid"]
    calibration = config["threshold_calibration"]
    max_gap_s = float(_read_json(root / BASE_CONFIG_PATH)["common"]["max_gap_s"])
    merge_gap_s = float(event_grid["merge_gap_seconds"])
    quantiles = list(map(float, calibration["candidate_quantiles"]))
    budgets = list(map(float, calibration["false_event_budgets_per_hour"]))
    persistence_values = list(map(float, event_grid["minimum_above_threshold_seconds"]))
    thresholds = {
        str(quantile): float(np.quantile(validation["score"].to_numpy(float), quantile))
        for quantile in quantiles
    }

    calibration_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for persistence_s in persistence_values:
        candidates: list[dict[str, Any]] = []
        for quantile in quantiles:
            threshold = thresholds[str(quantile)]
            rate, count, hours = _false_event_rate(
                validation, threshold, persistence_s, merge_gap_s, max_gap_s
            )
            row = {
                "minimum_above_threshold_seconds": persistence_s,
                "quantile": quantile,
                "threshold": threshold,
                "validation_false_events": count,
                "validation_normal_hours": hours,
                "validation_false_events_per_hour": rate,
            }
            calibration_rows.append(row)
            candidates.append(row)
        for budget in budgets:
            eligible = [
                row
                for row in candidates
                if row["validation_false_events_per_hour"] is not None
                and row["validation_false_events_per_hour"] <= budget
            ]
            chosen = min(eligible, key=lambda row: row["threshold"]) if eligible else None
            selected.append(
                {
                    "minimum_above_threshold_seconds": persistence_s,
                    "budget_false_events_per_hour": budget,
                    "available": chosen is not None,
                    "quantile": chosen["quantile"] if chosen else None,
                    "threshold": chosen["threshold"] if chosen else None,
                    "validation_false_events_per_hour": (
                        chosen["validation_false_events_per_hour"] if chosen else None
                    ),
                }
            )

    reporting = config["reporting"]
    bootstrap_replicates = int(reporting["bootstrap_replicates"])
    confidence_level = float(reporting["confidence_level"])
    evaluation_cache: dict[tuple[float, float], dict[str, Any]] = {}
    test_rows = []
    for choice in selected:
        if not choice["available"]:
            test_rows.append({**choice, "evaluation": None})
            continue
        threshold = float(choice["threshold"])
        persistence_s = float(choice["minimum_above_threshold_seconds"])
        cache_key = (threshold, persistence_s)
        if cache_key not in evaluation_cache:
            seed_material = f"{dataset}|{threshold:.17g}|{persistence_s:.17g}"
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            evaluation_cache[cache_key] = _evaluate_point(
                test,
                threshold,
                persistence_s,
                merge_gap_s,
                max_gap_s,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                bootstrap_seed=seed,
            )
        evaluation = evaluation_cache[cache_key]
        test_rows.append({**choice, "evaluation": evaluation})

    normal_validation_flights = int(validation["source_id"].nunique())
    interval_eligible = test["truth_available"].astype(bool)
    interval_metrics = _safe_metrics(
        test.loc[interval_eligible, "truth_active"].to_numpy(bool),
        test.loc[interval_eligible, "score"].to_numpy(float),
    )
    mapping_path = ledger_dir / "truth_mapping_report.json"
    mapping_evidence = None
    if mapping_path.is_file():
        mapping = _read_json(mapping_path)
        mapping_evidence = {
            "path": mapping_path.name,
            "sha256": _sha256(mapping_path),
            "mapped_anomaly_sources": mapping.get("mapped_anomaly_sources"),
            "unavailable_anomaly_sources": mapping.get("unavailable_anomaly_sources", []),
        }
    report = {
        "schema_version": 1,
        "candidate_namespace": config["candidate_namespace"],
        "dataset": dataset,
        "claim_eligible_normal_validation_count": normal_validation_flights
        >= int(calibration["minimum_normal_validation_flights_for_claim"]),
        "normal_validation_flights": normal_validation_flights,
        "truth_contract": config["dataset_truth_contract"][dataset],
        "truth_mapping_evidence": mapping_evidence,
        "bootstrap_contract": {
            "unit": reporting["bootstrap_unit"],
            "replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "method": "source-stratified percentile bootstrap",
        },
        "calibration_grid": calibration_rows,
        "selected_budget_points": test_rows,
        "flight_aggregations_exploratory": True,
        "flight_metrics": _flight_metrics(test),
        "interval_truth_window_metrics": interval_metrics,
    }
    _write_json_atomic(ledger_dir / "event_evaluation_report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("verify-run", "export-scores", "refresh-truth", "evaluate")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.repo_root.resolve()
    try:
        if args.command == "verify-run":
            if args.run_dir is None:
                raise EventEvalV1Error("--run-dir is required")
            accepted = _verify_run(root, args.dataset, args.run_dir.resolve())
            print(json.dumps(accepted.acceptance, indent=2, ensure_ascii=False))
            return 0
        if args.command == "export-scores":
            if args.run_dir is None or args.output_dir is None:
                raise EventEvalV1Error("--run-dir and --output-dir are required")
            report = _export_scores(
                root,
                args.dataset,
                args.run_dir.resolve(),
                args.output_dir.resolve(),
                args.device,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.command == "refresh-truth":
            if args.output_dir is None:
                raise EventEvalV1Error("--output-dir is required")
            report = _refresh_truth(root, args.dataset, args.output_dir.resolve())
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.output_dir is None:
            raise EventEvalV1Error("--output-dir is required")
        report = _evaluate_ledgers(root, args.dataset, args.output_dir.resolve())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (EventEvalV1Error, core.ProbabilisticV1Error) as exc:
        print(f"EVENT CONTRACT ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
