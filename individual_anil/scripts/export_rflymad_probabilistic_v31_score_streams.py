"""Export immutable per-window score/truth streams for the accepted RflyMAD v3.1 B0 run.

This command never trains and never opens the sealed final-fault sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import four_dataset_probabilistic_gpu_v1_runner as core
    import four_dataset_probabilistic_v31_runner as v31
except ModuleNotFoundError:  # pragma: no cover
    from scripts import four_dataset_probabilistic_gpu_v1_runner as core
    from scripts import four_dataset_probabilistic_v31_runner as v31


GRID_PATH = Path("configs/rflymad_probabilistic_v31_event_grid.json")
PREREG_PATH = Path("docs/FOUR_DATASET_PROBABILISTIC_V31_RFLYMAD_B0_EVENT_PREREG_20260728.md")
PROTOCOL_PATH = Path("configs/four_dataset_probabilistic_v31_split_manifest.json")
RUNNER_ROLES_PATH = Path("configs/four_dataset_probabilistic_v31_runner_roles.json")
DEFAULT_RUN = Path("artifacts/four_dataset_probabilistic_v31/runs/rflymad_colab_l4_20260728")
DEFAULT_OUTPUT = Path("artifacts/four_dataset_probabilistic_v31/rflymad_b0_event_eval")
EXPECTED_NAMESPACE = "four_dataset_probabilistic_v31"
EVENT_NAMESPACE = "rflymad_probabilistic_v31_b0_event_v1"
VISIBLE_ROLES = ("val_normal", "normal_test", "anomaly_dev")


class ScoreExportError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreExportError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScoreExportError(f"Expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _scaler_from_dict(value: dict[str, Any]) -> core.RobustScaler:
    features = tuple(map(str, value["features"]))
    return core.RobustScaler(
        features=features,
        median=np.asarray([value["median"][name] for name in features], dtype=np.float64),
        scale=np.asarray([value["scale"][name] for name in features], dtype=np.float64),
        excluded=tuple(map(str, value.get("excluded", []))),
        clip=float(value["clip"]),
    )


@dataclass(frozen=True)
class AcceptedRun:
    config: dict[str, Any]
    spec: dict[str, Any]
    access: core.DatasetAccess
    scaler: core.RobustScaler
    model_payload: dict[str, Any]
    evidence: dict[str, Any]
    checks: dict[str, bool]


def _verify_contract(root: Path, run_dir: Path) -> AcceptedRun:
    for relative in (GRID_PATH, PREREG_PATH, PROTOCOL_PATH, RUNNER_ROLES_PATH):
        if not (root / relative).is_file():
            raise ScoreExportError(f"Missing frozen input: {relative}")
    grid = _read_json(root / GRID_PATH)
    if grid.get("candidate_namespace") != EVENT_NAMESPACE:
        raise ScoreExportError("Unexpected event-grid namespace")
    if not all(bool(value) for value in grid.get("prohibitions", {}).values()):
        raise ScoreExportError("Every event-grid prohibition must remain true")

    v31._configure("rflymad", 0)
    try:
        config, spec = core._load_contract(root, "rflymad")
        access = core._load_access(root, config, spec, "rflymad")
        evidence = core._contract_evidence(root, config, access)
    except core.ProbabilisticV1Error as exc:
        raise ScoreExportError(f"Base contract verification failed: {exc}") from exc

    required = ("run_manifest.json", "training_report.json", "scaler.json", "model_state.pt")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ScoreExportError(f"Accepted run is missing: {missing}")
    manifest = _read_json(run_dir / "run_manifest.json")
    report = _read_json(run_dir / "training_report.json")
    scaler_value = _read_json(run_dir / "scaler.json")
    model_payload = core._checkpoint_load(run_dir / "model_state.pt", torch.device("cpu"))
    if not isinstance(model_payload, dict):
        raise ScoreExportError("model_state.pt is not a mapping")

    roles_doc = _read_json(root / RUNNER_ROLES_PATH)["sources"]["rflymad"]["roles"]
    protocol = _read_json(root / PROTOCOL_PATH)["datasets"]["rflymad"]
    visible = {
        "val_normal": tuple(map(str, protocol["source_ids"]["val"])),
        "normal_test": tuple(map(str, protocol["source_ids"]["normal_test"])),
        "anomaly_dev": tuple(map(str, protocol["source_ids"]["anomaly_dev"])),
    }
    # Deliberately do not dereference protocol['source_ids']['final_fault_test'].
    runner_test = tuple(map(str, roles_doc["test"]))
    frozen_evidence = manifest.get("contract_evidence", {})
    current_evidence_except_prereg = {
        key: evidence[key]
        for key in ("candidate_namespace", "config_sha256", "data_fingerprint", "roles_sha256", "role_counts")
    }
    frozen_evidence_except_prereg = {
        key: frozen_evidence.get(key)
        for key in current_evidence_except_prereg
    }
    frozen_evidence_payload = dict(frozen_evidence)
    frozen_declared_contract = frozen_evidence_payload.pop("contract_sha256", None)
    checks = {
        "grid_base_run": Path(str(grid["base_run"])) == run_dir.relative_to(root),
        "manifest_namespace": manifest.get("candidate_namespace") == EXPECTED_NAMESPACE,
        "report_namespace": report.get("candidate_namespace") == EXPECTED_NAMESPACE,
        "manifest_status": manifest.get("status") == "complete",
        "epochs_complete": int(report.get("completed_epochs", -1)) == int(config["common"]["epochs"]) == 30,
        "magnitude_gate_pass": report.get("magnitude_domination_flagged_at_0_8") is False,
        "frozen_contract_self_consistent": core._canonical_sha(frozen_evidence_payload) == frozen_declared_contract,
        "current_non_prereg_contract_inputs_match": current_evidence_except_prereg == frozen_evidence_except_prereg,
        "manifest_contract": manifest.get("contract_sha256") == frozen_declared_contract,
        "report_contract": report.get("contract_sha256") == frozen_declared_contract,
        "model_contract": model_payload.get("contract_sha256") == frozen_declared_contract,
        "model_dataset": model_payload.get("dataset") == "rflymad",
        "runner_val_matches_protocol": tuple(map(str, roles_doc["val"])) == visible["val_normal"],
        "runner_test_matches_visible_union": set(runner_test) == set(visible["normal_test"]) | set(visible["anomaly_dev"]),
        "visible_roles_disjoint": sum(map(len, visible.values())) == len(set().union(*map(set, visible.values()))),
        "visible_counts": [len(visible[name]) for name in VISIBLE_ROLES] == [90, 91, 557],
        "final_access_prohibited": grid["prohibitions"].get("no_final_fault_test_access") is True,
    }
    declared = report.get("artifact_hashes", {})
    for name in ("run_manifest.json", "scaler.json", "model_state.pt"):
        item = declared.get(name, {})
        checks[f"hash_{name}"] = (
            item.get("sha256") == _sha256(run_dir / name)
            and int(item.get("bytes", -1)) == (run_dir / name).stat().st_size
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ScoreExportError(f"CONTRACT ERROR: failed checks: {failed}")

    access.roles.update(visible)
    return AcceptedRun(
        config=config,
        spec=spec,
        access=access,
        scaler=_scaler_from_dict(scaler_value),
        model_payload=model_payload,
        evidence={
            **frozen_evidence,
            "current_preregistration_sha256": evidence["preregistration_sha256"],
            "preregistration_hash_matches_current_checkout": (
                evidence["preregistration_sha256"] == frozen_evidence.get("preregistration_sha256")
            ),
            "known_provenance_defect": (
                "accepted Colab artifact embeds a preregistration byte hash that does not match "
                "the later committed checkout; config, roles, data fingerprints and all run artifacts match"
                if evidence["preregistration_sha256"] != frozen_evidence.get("preregistration_sha256")
                else None
            ),
        },
        checks=checks,
    )


def _build_model(accepted: AcceptedRun, device: torch.device) -> core.GaussianForecaster:
    model = core.GaussianForecaster(
        input_size=len(accepted.scaler.features) * 2 + 1,
        channels=len(accepted.scaler.features),
        common=accepted.config["common"],
    ).to(device)
    model.load_state_dict(accepted.model_payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _metadata(root: Path) -> pd.DataFrame:
    registry = pd.read_parquet(
        root / "artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet",
        columns=["dataset", "source_id", "group_id", "domain", "flight_status", "fault_family", "fault_subtype"],
    )
    registry = registry[registry["dataset"].astype(str) == "rflymad"].copy()
    return registry.drop_duplicates("source_id").set_index("source_id")


def _source_frame(
    accepted: AcceptedRun,
    role: str,
    scored: core.ScoreSet,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    normal_label = str(accepted.spec["normal_label"])
    if accepted.access.source_paths is None:
        raise ScoreExportError("RflyMAD source paths unavailable")
    for code, (source_id, label) in enumerate(zip(scored.sources, scored.labels)):
        use = scored.source_codes == code
        times = scored.target_times[use].astype(np.float64, copy=False)
        path = accepted.access.source_paths[source_id]
        truth = pd.read_parquet(
            path,
            columns=["t_rel_s", "fault_active", "condition_active", "truth_source", "truth_crosscheck_disagreement_v2"],
        ).drop_duplicates("t_rel_s", keep="last").set_index("t_rel_s")
        aligned = truth.reindex(pd.Index(times, name="t_rel_s"))
        if aligned[["fault_active", "condition_active"]].isna().any().any():
            raise ScoreExportError(f"Truth/time alignment failed: {source_id}")
        disagreement = aligned["truth_crosscheck_disagreement_v2"].fillna(False).astype(bool).to_numpy()
        fault = aligned["fault_active"].fillna(False).astype(bool).to_numpy()
        condition = aligned["condition_active"].fillna(False).astype(bool).to_numpy()
        active_times = times[(fault | condition) & ~disagreement]
        meta = metadata.loc[source_id] if source_id in metadata.index else pd.Series(dtype=object)
        family = meta.get("fault_family", label)
        subtype = meta.get("fault_subtype", None)
        records.append(pd.DataFrame({
            "source_id": source_id,
            "group_id": meta.get("group_id", None),
            "role": role,
            "domain": meta.get("domain", None),
            "flight_status": meta.get("flight_status", None),
            "fault_family": family,
            "fault_subtype": subtype,
            "flight_label": label,
            "is_anomaly_flight": label != normal_label,
            "t_rel_s": times,
            "gaussian_nll_score": scored.scores[use].astype(np.float64, copy=False),
            "target_magnitude": scored.magnitudes[use].astype(np.float64, copy=False),
            "truth_available": ~disagreement,
            "fault_active": fault,
            "condition_active": condition,
            "truth_active": (fault | condition) & ~disagreement,
            "fault_start_s": float(active_times.min()) if len(active_times) else np.nan,
            "fault_end_s": float(active_times.max()) if len(active_times) else np.nan,
            "truth_source": aligned["truth_source"].astype("string").to_numpy(object),
        }))
    return pd.concat(records, ignore_index=True)


def export(root: Path, run_dir: Path, output_dir: Path, device_name: str) -> dict[str, Any]:
    accepted = _verify_contract(root, run_dir)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ScoreExportError("CUDA requested but unavailable")
    device = torch.device(device_name)
    model = _build_model(accepted, device)
    metadata = _metadata(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for role in VISIBLE_ROLES:
        scored = core._score_sources(
            accepted.access, role, accepted.scaler, model, accepted.config["common"],
            device, output_dir, stage=f"rflymad_v31_{role}_score_export",
        )
        frames.append(_source_frame(accepted, role, scored, metadata))
    ledger = pd.concat(frames, ignore_index=True)
    validation = ledger.loc[ledger["role"] == "val_normal", "gaussian_nll_score"].to_numpy(float)
    location = float(np.median(validation))
    scale = max(1.4826 * float(np.median(np.abs(validation - location))), 1e-9)
    ledger["standardized_nll"] = (ledger["gaussian_nll_score"] - location) / scale
    ledger = ledger.sort_values(["role", "source_id", "t_rel_s"], kind="mergesort").reset_index(drop=True)
    path = output_dir / "score_streams.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    ledger.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)
    manifest = {
        "schema_version": 1,
        "candidate_namespace": EVENT_NAMESPACE,
        "status": "complete",
        "device": str(device),
        "base_run": run_dir.relative_to(root).as_posix(),
        "contract_sha256": accepted.evidence["contract_sha256"],
        "contract_checks": accepted.checks,
        "contract_provenance": {
            "embedded_preregistration_sha256": accepted.evidence["preregistration_sha256"],
            "current_preregistration_sha256": accepted.evidence["current_preregistration_sha256"],
            "preregistration_hash_matches_current_checkout": accepted.evidence["preregistration_hash_matches_current_checkout"],
            "known_provenance_defect": accepted.evidence["known_provenance_defect"],
        },
        "final_fault_test_accessed": False,
        "score_transform": {"fit_role": "val_normal", "median": location, "scale_1_4826_mad": scale},
        "roles": {
            role: {
                "sources": int(frame["source_id"].nunique()),
                "windows": int(len(frame)),
                "truth_active_windows": int(frame["truth_active"].sum()),
            }
            for role, frame in ledger.groupby("role", sort=True)
        },
        "artifacts": {
            "event_grid": {"path": GRID_PATH.as_posix(), "sha256": _sha256(root / GRID_PATH)},
            "preregistration": {"path": PREREG_PATH.as_posix(), "sha256": _sha256(root / PREREG_PATH)},
            "model_state": {"path": (run_dir / "model_state.pt").relative_to(root).as_posix(), "sha256": _sha256(run_dir / "model_state.pt")},
            "scaler": {"path": (run_dir / "scaler.json").relative_to(root).as_posix(), "sha256": _sha256(run_dir / "scaler.json")},
            "score_streams": {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size},
        },
    }
    _write_json_atomic(output_dir / "score_export_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        result = export(root, run_dir.resolve(), output_dir.resolve(), args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ScoreExportError, core.ProbabilisticV1Error, KeyError, ValueError, RuntimeError) as exc:
        print(f"CONTRACT ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
