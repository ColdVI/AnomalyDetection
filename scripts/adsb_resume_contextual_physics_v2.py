"""Resume contextual_physics_v2 from a completed-epoch recovery checkpoint.

This is an operational recovery entrypoint.  It does not alter the frozen
training configuration, source ordering, model, loss, optimizer, RNG stream,
or epoch count.  The interrupted parent remains immutable and incomplete; a
new run directory records the complete parent/checkpoint lineage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.adsb_train_contextual_physics_v2 as base  # noqa: E402
from adsb.contextual_scaling import (  # noqa: E402
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)
from adsb.models.contextual_residual_forecaster import (  # noqa: E402
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    channelwise_gaussian_nll,
    weighted_masked_channel_loss,
)


class RecoveryContractError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scaler(parent: Path, config: dict[str, Any]) -> tuple[StrictNaturalRobustScaler, dict[str, Any]]:
    payload = _read_json(parent / "fit_scaler.json")
    frozen = payload["scaler"]
    if float(frozen["clip"]) != float(config["scaling"]["clip"]):
        raise RecoveryContractError("Parent scaler clip differs from frozen config")
    channels = tuple(config["channels"])
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=float(frozen["clip"])))
    scaler.fit_from_statistics(
        frozen["calibration"],
        tuple(frozen["excluded_channels"]),
        channels,
        data_role=NATURAL_FIT_ROLE,
        contains_synthetic=False,
    )
    if scaler.to_dict() != frozen:
        raise RecoveryContractError("Parent scaler does not round-trip exactly")
    return scaler, payload


def _verify_parent(parent: Path, root: Path, config_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not parent.is_dir():
        raise RecoveryContractError(f"Parent run is missing: {parent}")
    if (parent / "training_report.json").exists() or (parent / "model_state.pt").exists():
        raise RecoveryContractError("A completed parent run must not be resumed")
    manifest = _read_json(parent / "run_manifest.json")
    if manifest.get("candidate_namespace") != "contextual_physics_v2":
        raise RecoveryContractError("Unexpected parent candidate namespace")
    if base._sha256_file(config_file) != manifest["config_sha256"]:
        raise RecoveryContractError("Frozen config hash differs from parent")
    for relative, expected in manifest["code_sha256"].items():
        if base._sha256_file(root / relative) != expected:
            raise RecoveryContractError(f"Parent code hash mismatch: {relative}")
    source_manifest = root / manifest["source_step5_manifest"]
    if base._sha256_file(source_manifest) != manifest["source_step5_manifest_sha256"]:
        raise RecoveryContractError("Parent Step-5 manifest hash mismatch")
    checkpoint_path = parent / "training_epoch_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "artifact_role",
        "completed_epochs",
        "history",
        "model_config",
        "model_state_dict",
        "optimizer_state_dict",
        "torch_rng_state",
    }
    if not required.issubset(checkpoint):
        raise RecoveryContractError("Recovery checkpoint is missing required fields")
    if checkpoint["artifact_role"] != "incomplete_training_recovery_only":
        raise RecoveryContractError("Unexpected recovery checkpoint role")
    completed = int(checkpoint["completed_epochs"])
    if completed <= 0 or completed != len(checkpoint["history"]):
        raise RecoveryContractError("Invalid completed epoch history")
    if [int(row["epoch"]) for row in checkpoint["history"]] != list(range(1, completed + 1)):
        raise RecoveryContractError("Parent epoch history is not contiguous")
    return manifest, checkpoint


def _sources(root: Path, config: dict[str, Any]) -> tuple[list[base.FitSource], list[base.FitSource], dict[str, Any]]:
    source = base._verify_source_manifest(root, config)
    step5_paths = base._verify_fit_inputs(root, source["fit_inputs"])
    split_records = source["manifest"]["split_contract"]["splits"]
    step5_fit_ids = tuple(split_records["fit"]["flight_ids"])
    calibration_ids = tuple(split_records["calibration"]["flight_ids"])
    step5_selected = base._sample_flights(
        step5_fit_ids,
        probability=float(config["data"]["fit_flight_sample_probability"]),
        seed=int(config["data"]["fit_flight_sample_seed"]),
        purpose="contextual_physics_v2_fit_step5",
    )
    calibration_selected = base._sample_flights(
        calibration_ids,
        probability=float(config["data"]["calibration_diagnostic_sample_probability"]),
        seed=int(config["data"]["calibration_diagnostic_sample_seed"]),
        purpose="contextual_physics_v2_calibration_diagnostic",
    )
    if set(step5_selected) & set(calibration_selected):
        raise RecoveryContractError("Fit and calibration diagnostic flights overlap")
    expansion = base._load_fit_expansion_manifest(root, config["source_step5_manifest_sha256"])
    fit_sources = base._build_fit_sources(
        step5_paths=step5_paths,
        step5_selected_flights=step5_selected,
        expansion_manifest=expansion,
        repo_root=root,
    )
    calibration_sources = [
        base.FitSource(path, base.STEP5_FIT_DAY, selected_flights=set(calibration_selected), selected_sources=None)
        for path in step5_paths
    ]
    evidence = {
        "step5_fit_input_count": len(step5_paths),
        "step5_fit_flights_total": len(step5_fit_ids),
        "step5_fit_flights_selected": len(step5_selected),
        "step5_fit_flight_ids_sha256": base._canonical_json_sha256(list(step5_selected)),
        "calibration_flights_total": len(calibration_ids),
        "calibration_diagnostic_flights_selected": len(calibration_selected),
        "calibration_diagnostic_flight_ids_sha256": base._canonical_json_sha256(list(calibration_selected)),
        "expansion": expansion,
    }
    return fit_sources, calibration_sources, evidence


def _resume_train(
    sources: list[base.FitSource],
    *,
    scaler: StrictNaturalRobustScaler,
    config: dict[str, Any],
    destination: Path,
    checkpoint: dict[str, Any],
    parent_derived: dict[str, Any],
) -> tuple[ContextualResidualForecaster, dict[str, Any]]:
    training = config["training"]
    total_epochs = int(training["epochs"])
    completed = int(checkpoint["completed_epochs"])
    if completed >= total_epochs:
        raise RecoveryContractError("Recovery checkpoint already reached the frozen epoch count")

    model_config = ContextualForecasterConfig(**checkpoint["model_config"])
    if model_config.__dict__ != parent_derived["model_config"]:
        raise RecoveryContractError("Checkpoint and derived model configs differ")
    model = ContextualResidualForecaster(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    weights = torch.ones(model_config.target_channels, dtype=torch.float32)
    input_features = tuple(parent_derived["input_features"])
    history = [dict(row) for row in checkpoint["history"]]
    epoch_checkpoint_path = destination / "training_epoch_checkpoint.pt"

    for epoch in range(completed, total_epochs):
        epoch_loss_sum = 0.0
        epoch_windows = 0
        epoch_batches = 0
        for _, features in base._iter_fit_features(sources):
            batch = base._make_batch(features, scaler=scaler, config=config)
            if len(batch.X) == 0:
                continue
            if batch.input_features != input_features:
                raise RecoveryContractError("Input feature order changed during recovery")
            if base._model_config(config, batch) != model_config:
                raise RecoveryContractError("Frozen model config changed during recovery")
            permutation = torch.randperm(len(batch.X))
            for start in range(0, len(batch.X), int(training["batch_size"])):
                index = base._numpy_batch_indices(permutation, start, int(training["batch_size"]))
                xb = torch.from_numpy(batch.X[index])
                mb = torch.from_numpy(batch.X_mask[index])
                yb = torch.from_numpy(batch.y[index])
                ymb = torch.from_numpy(batch.y_mask[index])
                optimizer.zero_grad()
                location, scale = model(xb, mb)
                nll, _ = channelwise_gaussian_nll(yb, location, scale, ymb)
                loss = weighted_masked_channel_loss(nll, ymb, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(training["gradient_clip_norm"]))
                optimizer.step()
                epoch_loss_sum += float(loss.detach()) * len(index)
                epoch_windows += len(index)
                epoch_batches += 1
        if epoch_windows == 0:
            raise RecoveryContractError("No scoreable recovery training windows were produced")
        record = {
            "epoch": epoch + 1,
            "mean_weighted_gaussian_nll": epoch_loss_sum / epoch_windows,
            "windows": epoch_windows,
            "batches": epoch_batches,
        }
        history.append(record)
        base._write_epoch_checkpoint_atomic(
            epoch_checkpoint_path, model=model, optimizer=optimizer, history=history
        )
        print(
            f"epoch={record['epoch']} windows={epoch_windows} "
            f"mean_nll={record['mean_weighted_gaussian_nll']:.8f} "
            f"checkpoint={epoch_checkpoint_path.name}",
            flush=True,
        )
    return model, {
        "epochs": history,
        "derived_training_config": "derived_training_config.json",
        "derived_training_config_sha256": base._sha256_file(destination / "derived_training_config.json"),
        "epoch_checkpoint": {
            "path": epoch_checkpoint_path.name,
            "artifact_role": "incomplete_training_recovery_only",
            "completed_epochs": len(history),
            "bytes": epoch_checkpoint_path.stat().st_size,
            "sha256": base._sha256_file(epoch_checkpoint_path),
        },
    }


def run(*, repo_root: Path, config_path: Path, parent_run: Path, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = repo_root.resolve(strict=True)
    config_file = config_path.resolve(strict=True)
    parent = parent_run.resolve(strict=True)
    destination = run_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}")
    if {part.lower() for part in destination.parts} & {"archive", "downloads", "raw"}:
        raise RecoveryContractError("Run directory uses a forbidden path component")

    config = base._load_config(config_file)
    parent_manifest, checkpoint = _verify_parent(parent, root, config_file)
    if base._canonical_json_sha256(config) != parent_manifest["config_payload_sha256"]:
        raise RecoveryContractError("Frozen config payload differs from parent")
    scaler, scaler_payload = _load_scaler(parent, config)
    parent_derived = _read_json(parent / "derived_training_config.json")
    if parent_derived["scaler"] != scaler.to_dict():
        raise RecoveryContractError("Parent derived config and scaler differ")
    fit_sources, calibration_sources, source_evidence = _sources(root, config)
    expansion = source_evidence.pop("expansion")
    if [row["source_day"] for row in expansion["days"]] != parent_manifest["fit_expansion_days"]:
        raise RecoveryContractError("Fit-expansion day set/order differs from parent")
    if expansion["fit_expansion_sha256"] != parent_manifest["fit_expansion_manifest_sha256"]:
        raise RecoveryContractError("Fit-expansion manifest hash differs from parent")

    git_start = base._git_state(root)
    code_start = base._code_hashes(root)
    recovery_script = Path(__file__).resolve()
    code_start[recovery_script.relative_to(root).as_posix()] = base._sha256_file(recovery_script)
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint_path = parent / "training_epoch_checkpoint.pt"
    lineage = {
        "parent_run_id": parent.name,
        "parent_run_manifest_sha256": base._sha256_file(parent / "run_manifest.json"),
        "parent_checkpoint_path": checkpoint_path.name,
        "parent_checkpoint_sha256": base._sha256_file(checkpoint_path),
        "parent_completed_epochs": int(checkpoint["completed_epochs"]),
        "resume_preserves_model_optimizer_and_torch_rng": True,
    }
    run_manifest = {
        **{k: v for k, v in parent_manifest.items() if k not in {"run_id", "git", "code_sha256"}},
        "run_id": destination.name,
        "git": git_start,
        "code_sha256": code_start,
        "recovery_lineage": lineage,
        **source_evidence,
    }
    base._write_json_exclusive(destination / "run_manifest.json", run_manifest)
    base._write_json_exclusive(destination / "fit_scaler.json", scaler_payload)
    base._write_json_exclusive(destination / "derived_training_config.json", parent_derived)

    model, training_report = _resume_train(
        fit_sources,
        scaler=scaler,
        config=config,
        destination=destination,
        checkpoint=checkpoint,
        parent_derived=parent_derived,
    )
    final_model = destination / "model_state.pt"
    torch.save(model.state_dict(), final_model)
    diagnostics = base._natural_diagnostics(
        model, calibration_sources, scaler=scaler, config=config, seed_offset=1000
    )
    code_end = base._code_hashes(root)
    code_end[recovery_script.relative_to(root).as_posix()] = base._sha256_file(recovery_script)
    git_end = base._git_state(root)
    report = {
        "run_id": destination.name,
        "status": "trained_not_thresholded",
        "elapsed_seconds": time.perf_counter() - started,
        "synthetic_training_rows": 0,
        "threshold_selection_performed": False,
        "fit_scaler": scaler.to_dict(),
        "training": training_report,
        "natural_calibration_diagnostic": diagnostics,
        "model_checkpoint": {
            "path": final_model.name,
            "bytes": final_model.stat().st_size,
            "sha256": base._sha256_file(final_model),
        },
        "recovery_lineage": lineage,
        "provenance": {
            "git_start": git_start,
            "git_end": git_end,
            "code_sha256_start": code_start,
            "code_sha256_end": code_end,
            "code_and_git_unchanged": code_end == code_start and git_end == git_start,
            "changes_are_recorded_not_a_training_gate": True,
        },
        "next_gate": "conformal + CUSUM + persistence_v2 calibration (Faz D)",
    }
    base._write_json_exclusive(destination / "training_report.json", report)
    base._write_checksums(destination)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_contextual_physics_v2_train.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(
            repo_root=args.repo_root,
            config_path=args.config,
            parent_run=args.parent_run,
            run_dir=args.run_dir,
        )
    except Exception as exc:
        destination = args.run_dir.resolve(strict=False)
        if destination.exists():
            marker = destination / "INCOMPLETE_DO_NOT_USE.md"
            if not marker.exists():
                marker.write_text(
                    f"# INCOMPLETE -- DO NOT USE\n\nRecovery stopped before a complete checksum chain: "
                    f"{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
        raise
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "magnitude_flagged": report["natural_calibration_diagnostic"][
                    "magnitude_domination_flagged_at_0_8"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
