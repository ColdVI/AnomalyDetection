"""Preregistered development-only Wind/Real robustness experiments.

The experiment contract is frozen in
``docs/RFLYMAD_V2_ROBUSTNESS_SOZLESMESI_20260722.md``.  This module never
loads locked-test feature parquet files: the manifest query is predicate-
filtered to the development split before paths are constructed.
"""

from __future__ import annotations

import json
import shutil
import gc
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rfly_full.contract import DATASET_MANIFEST, V2_ROOT
from rfly_full.normal_ae import (
    BUDGETS,
    DOMAINS,
    SEED,
    TRAIN_STRIDE_SECONDS,
    VALIDATION_STRIDE_SECONDS,
    TemporalConvAutoencoder,
    _evaluate,
    _fa_rate,
    _masked_loss,
    _normal_windows,
    _ranked_groups,
    _score_manifest,
    _train,
    _validation_loss,
)
from rfly_full.pipeline import _atomic_json
from rfly_full.supervised import PARSED_10HZ_ROOT, _fit_scaler
from rfly_full.v2_parser import V2_FEATURES


OUTPUT_ROOT = V2_ROOT / "normal_temporal_ae" / "robustness"
ROTATIONS = (0, 1, 2, 3, 4)
POLICY_NAMES = tuple(BUDGETS)
BASELINE_CRITICAL_RECALL = 0.6043117744610281
BASELINE_REAL_MACRO_RECALL = (0.2020 + 0.0835) / 2
BASELINE_WIND_FA_PER_HOUR = 28.461784635934823
REAL_CRITICAL_BUDGET = 4.0
FROZEN_BASELINE_ROOT = OUTPUT_ROOT.parent / "sweep_20260722_093049"
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_SAMPLES = 1000
CANDIDATES = ("R1", "W1", "W2", "R2", "R3")
CONVERGED_CANDIDATE = "R4"
CONVERGENCE_MAX_EPOCHS = 100
CONVERGENCE_PATIENCE = 12
CONVERGENCE_MIN_DELTA = 1e-4
CONVERGENCE_EXTENSION_CEILINGS = (500, 2000)


def _development_manifest() -> pd.DataFrame:
    manifest = pd.read_parquet(
        DATASET_MANIFEST,
        filters=[("split", "==", "development")],
    ).drop_duplicates("canonical_case_id")
    available = []
    for row in manifest.itertuples(index=False):
        path = PARSED_10HZ_ROOT / str(row.domain) / f"{row.canonical_case_id}.parquet"
        available.append(path.exists())
    result = manifest.loc[available].copy()
    if not result["split"].eq("development").all():
        raise RuntimeError("robustness manifest escaped development split")
    return result


def _nested_normal_split(
    available: pd.DataFrame, rotation: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    normal = available.loc[available["evaluation_role"].eq("normal_reference")].copy()
    train_parts = []
    inner_parts = []
    outer_parts = []
    groups: dict[str, dict[str, str]] = {}
    for domain in DOMAINS:
        domain_normal = normal.loc[normal["domain"].eq(domain)]
        ranked = _ranked_groups(
            domain,
            domain_normal["split_group_id"].dropna().astype(str).unique().tolist(),
        )
        if len(ranked) < 3:
            raise RuntimeError(f"nested robustness needs three {domain} normal groups")
        outer_group = ranked[rotation % len(ranked)]
        inner_group = ranked[(rotation + 1) % len(ranked)]
        groups[domain] = {"outer": outer_group, "inner": inner_group}
        outer_mask = domain_normal["split_group_id"].eq(outer_group)
        inner_mask = domain_normal["split_group_id"].eq(inner_group)
        outer_parts.append(domain_normal.loc[outer_mask])
        inner_parts.append(domain_normal.loc[inner_mask])
        train_parts.append(domain_normal.loc[~outer_mask & ~inner_mask])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(inner_parts, ignore_index=True),
        pd.concat(outer_parts, ignore_index=True),
        groups,
    )


def _nested_wind_split(
    available: pd.DataFrame, rotation: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    wind = available.loc[available["evaluation_role"].eq("environment_robustness")].copy()
    train_parts = []
    inner_parts = []
    outer_parts = []
    groups: dict[str, dict[str, str]] = {}
    for domain in ("HIL", "SIL"):
        domain_wind = wind.loc[wind["domain"].eq(domain)]
        ranked = _ranked_groups(
            f"{domain}:Wind",
            domain_wind["split_group_id"].dropna().astype(str).unique().tolist(),
        )
        if len(ranked) < 3:
            raise RuntimeError(f"nested robustness needs three {domain} Wind groups")
        outer_group = ranked[rotation % len(ranked)]
        inner_group = ranked[(rotation + 1) % len(ranked)]
        groups[domain] = {"outer": outer_group, "inner": inner_group}
        outer_mask = domain_wind["split_group_id"].eq(outer_group)
        inner_mask = domain_wind["split_group_id"].eq(inner_group)
        outer_parts.append(domain_wind.loc[outer_mask])
        inner_parts.append(domain_wind.loc[inner_mask])
        train_parts.append(domain_wind.loc[~outer_mask & ~inner_mask])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(inner_parts, ignore_index=True),
        pd.concat(outer_parts, ignore_index=True),
        groups,
    )


def _combine_windows(
    normal_windows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    wind_windows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([normal_windows[0], wind_windows[0]]),
        np.concatenate([normal_windows[1], wind_windows[1]]),
        np.concatenate([normal_windows[2], wind_windows[2]]),
        np.asarray(
            ["NoFault"] * len(normal_windows[0]) + ["Wind"] * len(wind_windows[0]),
            dtype=object,
        ),
    )


def _weighted_environment_rate(
    normal_scores: pd.DataFrame,
    wind_scores: pd.DataFrame,
    threshold: float,
) -> tuple[int, float, float, float]:
    normal_events, normal_rate = _fa_rate(normal_scores, threshold)
    wind_events, wind_rate = _fa_rate(wind_scores, threshold)
    weighted_rate = 0.5 * normal_rate + 0.5 * wind_rate
    return normal_events + wind_events, weighted_rate, normal_rate, wind_rate


def _fit_candidate_policies(
    scored: pd.DataFrame,
    *,
    inner_normal_ids: set[str],
    inner_wind_ids: set[str],
    real_relaxed: bool,
    environment_aware: bool,
) -> dict:
    policies = {}
    normal = scored.loc[scored["canonical_case_id"].isin(inner_normal_ids)]
    wind = scored.loc[scored["canonical_case_id"].isin(inner_wind_ids)]
    for domain in DOMAINS:
        normal_domain = normal.loc[normal["domain"].eq(domain)]
        if normal_domain.empty:
            raise RuntimeError(f"no inner normal calibration scores for {domain}")
        wind_domain = wind.loc[wind["domain"].eq(domain)]
        calibration = (
            pd.concat([normal_domain, wind_domain], ignore_index=True)
            if environment_aware and not wind_domain.empty else normal_domain
        )
        candidates = np.unique(
            np.quantile(calibration["score"], np.linspace(0.80, 1.0, 121))
        )
        candidates = np.r_[candidates, np.nextafter(candidates[-1], np.inf)]
        for policy_name, base_budget in BUDGETS.items():
            budget = (
                REAL_CRITICAL_BUDGET
                if real_relaxed and domain == "Real" and policy_name == "critical"
                else base_budget
            )
            rows = []
            for threshold in candidates:
                if environment_aware and not wind_domain.empty:
                    events, rate, normal_rate, wind_rate = _weighted_environment_rate(
                        normal_domain, wind_domain, float(threshold)
                    )
                else:
                    events, rate = _fa_rate(normal_domain, float(threshold))
                    normal_rate, wind_rate = rate, None
                rows.append((float(threshold), events, rate, normal_rate, wind_rate))
            feasible = [row for row in rows if row[2] <= budget]
            chosen = min(feasible, key=lambda row: row[0]) if feasible else max(
                rows, key=lambda row: row[0]
            )
            policies[f"{domain}:{policy_name}"] = {
                "domain": domain,
                "policy": policy_name,
                "budget_fa_per_hour": budget,
                "threshold": chosen[0],
                "validation_alarm_events": chosen[1],
                "validation_fa_per_hour": chosen[2],
                "normal_component_fa_per_hour": chosen[3],
                "wind_component_fa_per_hour": chosen[4],
                "environment_aware": bool(environment_aware and not wind_domain.empty),
                "alarm": "4-of-6 seconds; 30-second refractory",
            }
    return policies


def _evaluation_scores(
    scored: pd.DataFrame,
    *,
    outer_normal_ids: set[str],
    outer_wind_ids: set[str] | None,
) -> pd.DataFrame:
    result = scored.copy()
    result["normal_calibration_holdout"] = result["canonical_case_id"].isin(
        outer_normal_ids
    )
    if outer_wind_ids is not None:
        environment = result["evaluation_role"].eq("environment_robustness")
        result = result.loc[
            ~environment | result["canonical_case_id"].isin(outer_wind_ids)
        ].copy()
    return result


def _rotation_metrics(
    candidate: str,
    rotation: int,
    scored: pd.DataFrame,
    policies: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics, per_flight = _evaluate(scored, policies)
    family = (
        per_flight.loc[per_flight["evaluation_role"].eq("fault_detection")]
        .groupby(["policy", "domain", "fault_family"], observed=True)
        .agg(flights=("canonical_case_id", "nunique"), detected=("detected", "sum"))
        .reset_index()
    )
    family["recall"] = family["detected"] / family["flights"]
    rows = []
    for policy_name in POLICY_NAMES:
        metric = metrics.loc[metrics["policy"].eq(policy_name)].iloc[0]
        real_family = family.loc[
            family["policy"].eq(policy_name)
            & family["domain"].eq("Real")
            & family["fault_family"].isin(["Motor", "Sensor"])
        ].set_index("fault_family")["recall"]
        real_normal = per_flight.loc[
            per_flight["policy"].eq(policy_name)
            & per_flight["domain"].eq("Real")
            & per_flight["evaluation_role"].eq("normal_reference")
            & per_flight["normal_calibration_holdout"]
        ]
        exposure = real_normal["normal_exposure_hours"].sum()
        real_fa = (
            float(real_normal["false_alarm_events"].sum() / exposure)
            if exposure else float("nan")
        )
        real_alarm_flight_rate = (
            float(real_normal["alarm_events"].gt(0).mean())
            if len(real_normal) else float("nan")
        )
        motor = float(real_family.get("Motor", np.nan))
        sensor = float(real_family.get("Sensor", np.nan))
        rows.append({
            "candidate": candidate,
            "rotation": rotation,
            "policy": policy_name,
            "event_recall": float(metric["event_recall"]),
            "all_nonfault_fa_per_hour": float(metric["all_nonfault_fa_per_hour"]),
            "wind_fa_per_hour": float(metric["environment_fa_per_hour"]),
            "real_motor_recall": motor,
            "real_sensor_recall": sensor,
            "real_macro_recall": float(np.nanmean([motor, sensor])),
            "real_normal_fa_per_hour": real_fa,
            "real_normal_alarm_flight_rate": real_alarm_flight_rate,
        })
    return pd.DataFrame(rows), per_flight, {"metrics": metrics, "family": family}


def _gate_summary(rotation_metrics: pd.DataFrame) -> dict:
    critical = rotation_metrics.loc[rotation_metrics["policy"].eq("critical")]
    aggregate = {
        column: {
            "mean": float(critical[column].mean()),
            "std": float(critical[column].std()),
            "min": float(critical[column].min()),
            "max": float(critical[column].max()),
        }
        for column in (
            "event_recall", "all_nonfault_fa_per_hour", "wind_fa_per_hour",
            "real_motor_recall", "real_sensor_recall", "real_macro_recall",
            "real_normal_fa_per_hour", "real_normal_alarm_flight_rate",
        )
    }
    real_conditions = {
        "macro_mean_ge_0_40": aggregate["real_macro_recall"]["mean"] >= 0.40,
        "motor_mean_ge_0_30": aggregate["real_motor_recall"]["mean"] >= 0.30,
        "sensor_mean_ge_0_30": aggregate["real_sensor_recall"]["mean"] >= 0.30,
        "macro_min_ge_0_25": aggregate["real_macro_recall"]["min"] >= 0.25,
        "real_fa_mean_le_4": aggregate["real_normal_fa_per_hour"]["mean"] <= 4.0,
        "real_fa_max_le_8": aggregate["real_normal_fa_per_hour"]["max"] <= 8.0,
        "overall_recall_mean_preserved": aggregate["event_recall"]["mean"]
        >= BASELINE_CRITICAL_RECALL - 0.05,
        "overall_recall_min_ge_0_50": aggregate["event_recall"]["min"] >= 0.50,
        "all_nonfault_fa_mean_le_2": aggregate["all_nonfault_fa_per_hour"]["mean"] <= 2.0,
    }
    wind_conditions = {
        "wind_fa_mean_le_15": aggregate["wind_fa_per_hour"]["mean"] <= 15.0,
        "wind_fa_max_le_20": aggregate["wind_fa_per_hour"]["max"] <= 20.0,
        "wind_reduction_ge_40pct": aggregate["wind_fa_per_hour"]["mean"]
        <= BASELINE_WIND_FA_PER_HOUR * 0.60,
        "overall_recall_mean_preserved": aggregate["event_recall"]["mean"]
        >= BASELINE_CRITICAL_RECALL - 0.05,
        "real_macro_loss_le_0_05": aggregate["real_macro_recall"]["mean"]
        >= BASELINE_REAL_MACRO_RECALL - 0.05,
        "all_nonfault_fa_mean_le_2": aggregate["all_nonfault_fa_per_hour"]["mean"] <= 2.0,
    }
    feasibility_conditions = {
        "macro_mean_ge_0_50": aggregate["real_macro_recall"]["mean"] >= 0.50,
        "motor_mean_ge_0_40": aggregate["real_motor_recall"]["mean"] >= 0.40,
        "sensor_mean_ge_0_40": aggregate["real_sensor_recall"]["mean"] >= 0.40,
        "real_fa_mean_le_2": aggregate["real_normal_fa_per_hour"]["mean"] <= 2.0,
        "real_fa_max_le_4": aggregate["real_normal_fa_per_hour"]["max"] <= 4.0,
    }
    wind_final_conditions = {
        "wind_fa_mean_le_5": aggregate["wind_fa_per_hour"]["mean"] <= 5.0,
        "wind_fa_max_le_8": aggregate["wind_fa_per_hour"]["max"] <= 8.0,
    }
    return {
        "critical_aggregate": aggregate,
        "real_research_gate": {
            "passed": bool(all(real_conditions.values())), "conditions": real_conditions,
        },
        "wind_intermediate_gate": {
            "passed": bool(all(wind_conditions.values())), "conditions": wind_conditions,
        },
        "real_feasibility_gate_without_bootstrap": {
            "passed": bool(all(feasibility_conditions.values())),
            "conditions": feasibility_conditions,
            "note": "bootstrap lower-bound condition is applied after cluster bootstrap",
        },
        "wind_final_research_target": {
            "passed": bool(all(wind_final_conditions.values())),
            "conditions": wind_final_conditions,
        },
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_checkpoint(model: nn.Module, scaler: dict) -> dict:
    return {
        "state_dict": model.state_dict(),
        "features": scaler["features"],
        "model": "temporal_conv_autoencoder",
        "training_seed": SEED,
    }


def _load_model(path: Path, scaler: dict) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["features"] != scaler["features"]:
        raise RuntimeError(f"checkpoint/scaler feature mismatch: {path}")
    feature_count = len(scaler["features"])
    model = TemporalConvAutoencoder(feature_count * 2, feature_count)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _fine_tune(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    epochs: int,
) -> tuple[nn.Module, list[dict]]:
    x_train, y_train, m_train, _ = train
    x_val, y_val, m_val, _ = validation
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(m_train)
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    history: list[dict] = []
    torch.manual_seed(SEED)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            loss = _masked_loss(yb, model(xb), mb).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = _validation_loss(model, x_val, y_val, m_val)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Real fine-tune produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _fine_tune_until_convergence(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    max_epochs: int = CONVERGENCE_MAX_EPOCHS,
    patience: int = CONVERGENCE_PATIENCE,
    min_delta: float = CONVERGENCE_MIN_DELTA,
) -> tuple[nn.Module, list[dict], dict]:
    x_train, y_train, m_train, _ = train
    x_val, y_val, m_val, _ = validation
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(m_train)
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    initial_loss = _validation_loss(model, x_val, y_val, m_val)
    best_loss = initial_loss
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    bad_epochs = 0
    history: list[dict] = [{
        "epoch": 0,
        "train_loss": None,
        "validation_loss": initial_loss,
        "is_best": True,
        "epochs_without_improvement": 0,
    }]
    torch.manual_seed(SEED)
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            loss = _masked_loss(yb, model(xb), mb).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = _validation_loss(model, x_val, y_val, m_val)
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "is_best": improved,
            "epochs_without_improvement": bad_epochs,
        })
        if bad_epochs >= patience:
            stop_reason = "validation_patience_exhausted"
            break
    model.load_state_dict(best_state)
    convergence = {
        "stop_reason": stop_reason,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "epochs_completed": int(history[-1]["epoch"]),
        "best_epoch": best_epoch,
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best_loss,
        "best_is_unmodified_base": best_epoch == 0,
    }
    return model, history, convergence


def initialize_experiment(run_name: str) -> Path:
    root = OUTPUT_ROOT / run_name
    state_path = root / "experiment_state.json"
    if state_path.exists():
        state = _read_json(state_path)
        if state.get("status") not in {
            "initialized", "base_prepared", "candidates_in_progress", "complete",
            "convergence_followup_in_progress", "convergence_followup_complete",
            "convergence_ceiling_extension_in_progress",
        }:
            raise RuntimeError(f"unrecognized experiment state: {state.get('status')}")
        return root
    root.mkdir(parents=True, exist_ok=False)
    available = _development_manifest()
    available.to_parquet(root / "development_manifest.parquet", index=False)
    contract = Path("docs/RFLYMAD_V2_ROBUSTNESS_SOZLESMESI_20260722.md")
    shutil.copy2(contract, root / "frozen_contract.md")
    _atomic_json(state_path, {
        "status": "initialized",
        "created_at": _now(),
        "run_name": run_name,
        "development_flights": int(available["canonical_case_id"].nunique()),
        "rotations": list(ROTATIONS),
        "candidates": list(CANDIDATES),
        "selection_rule": (
            "stop at the first passing candidate per target"
        ),
        "locked_test_features_read": False,
        "status_label": "development_only_robustness",
        "operational_claim_allowed": False,
    })
    return root


def _experiment_manifest(root: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(root / "development_manifest.parquet")
    if manifest.empty or not manifest["split"].eq("development").all():
        raise RuntimeError("experiment manifest is not development-only")
    return manifest


def _update_experiment_state(root: Path, **updates) -> None:
    path = root / "experiment_state.json"
    state = _read_json(path)
    state.update(updates)
    state["updated_at"] = _now()
    _atomic_json(path, state)


def prepare_base_models(
    root: Path, *, epochs: int = 25, torch_threads: int = 4,
) -> None:
    torch.set_num_threads(max(1, torch_threads))
    available = _experiment_manifest(root)
    base_root = root / "base"
    base_root.mkdir(exist_ok=True)
    completed = []
    for rotation in ROTATIONS:
        output = base_root / f"rotation_{rotation}"
        if (output / "summary.json").exists():
            completed.append(rotation)
            continue
        output.mkdir(parents=True, exist_ok=True)
        train_manifest, inner_manifest, outer_manifest, normal_groups = (
            _nested_normal_split(available, rotation)
        )
        _, inner_wind, outer_wind, wind_groups = _nested_wind_split(
            available, rotation
        )
        scaler = _fit_scaler(train_manifest, list(V2_FEATURES))
        train_windows = _normal_windows(
            train_manifest, scaler, TRAIN_STRIDE_SECONDS
        )
        validation_windows = _normal_windows(
            inner_manifest, scaler, VALIDATION_STRIDE_SECONDS
        )
        model, history = _train(
            train_windows, validation_windows, epochs=epochs
        )
        scored = _score_manifest(model, available, scaler)
        torch.save(_model_checkpoint(model, scaler), output / "model.pt")
        _atomic_json(output / "scaler.json", scaler)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        scored.to_parquet(output / "development_scores.parquet", index=False)
        summary = {
            "status": "development_only_robustness_base",
            "created_at": _now(),
            "rotation": rotation,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "normal_groups": normal_groups,
            "wind_groups": wind_groups,
            "train_normal_flights": int(train_manifest["canonical_case_id"].nunique()),
            "inner_normal_flights": int(inner_manifest["canonical_case_id"].nunique()),
            "outer_normal_flights": int(outer_manifest["canonical_case_id"].nunique()),
            "inner_wind_flights": int(inner_wind["canonical_case_id"].nunique()),
            "outer_wind_flights": int(outer_wind["canonical_case_id"].nunique()),
            "train_windows": int(len(train_windows[0])),
            "validation_windows": int(len(validation_windows[0])),
            "locked_test_features_read": False,
            "operational_claim_allowed": False,
        }
        _atomic_json(output / "summary.json", summary)
        completed.append(rotation)
        _update_experiment_state(
            root, status="initialized", base_rotations_completed=completed
        )
        del model, scored, train_windows, validation_windows
        gc.collect()
    _update_experiment_state(
        root, status="base_prepared", base_rotations_completed=completed
    )


def _candidate_model_and_scores(
    root: Path,
    candidate: str,
    rotation: int,
    available: pd.DataFrame,
) -> tuple[nn.Module, dict, pd.DataFrame, list[dict], str]:
    base = root / "base" / f"rotation_{rotation}"
    if not (base / "summary.json").exists():
        raise RuntimeError(f"base rotation {rotation} has not been prepared")
    scaler = _read_json(base / "scaler.json")
    if candidate in {"R1", "W1"}:
        model = _load_model(base / "model.pt", scaler)
        scored = pd.read_parquet(base / "development_scores.parquet")
        return model, scaler, scored, [], str(base / "model.pt")

    train_normal, inner_normal, _, _ = _nested_normal_split(available, rotation)
    model = _load_model(base / "model.pt", scaler)
    if candidate == "W2":
        train_wind, inner_wind, _, _ = _nested_wind_split(available, rotation)
        train_windows = _combine_windows(
            _normal_windows(train_normal, scaler, TRAIN_STRIDE_SECONDS),
            _normal_windows(train_wind, scaler, TRAIN_STRIDE_SECONDS),
        )
        validation_windows = _combine_windows(
            _normal_windows(inner_normal, scaler, VALIDATION_STRIDE_SECONDS),
            _normal_windows(inner_wind, scaler, VALIDATION_STRIDE_SECONDS),
        )
        model, history = _train(train_windows, validation_windows, epochs=25)
        source = "normal+Wind 1:1 AE training from deterministic initialization"
    elif candidate in {"R2", "R3"}:
        real_train = train_normal.loc[train_normal["domain"].eq("Real")]
        real_inner = inner_normal.loc[inner_normal["domain"].eq("Real")]
        train_windows = _normal_windows(
            real_train, scaler, TRAIN_STRIDE_SECONDS
        )
        validation_windows = _normal_windows(
            real_inner, scaler, VALIDATION_STRIDE_SECONDS
        )
        fine_tune_epochs = 3 if candidate == "R2" else 8
        model, history = _fine_tune(
            model, train_windows, validation_windows, epochs=fine_tune_epochs
        )
        source = f"base AE fine-tuned on Real NoFault for {fine_tune_epochs} epochs"
    else:
        raise ValueError(f"unsupported candidate: {candidate}")
    scored = _score_manifest(model, available, scaler)
    del train_windows, validation_windows
    gc.collect()
    return model, scaler, scored, history, source


def _cluster_bootstrap(per_flight: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_names = (
        "event_recall", "real_motor_recall", "real_sensor_recall",
        "real_macro_recall", "real_normal_fa_per_hour",
        "real_normal_alarm_flight_rate", "all_nonfault_fa_per_hour",
        "wind_fa_per_hour",
    )
    for policy_name in POLICY_NAMES:
        frame = per_flight.loc[per_flight["policy"].eq(policy_name)].copy()
        role = frame["evaluation_role"]
        fault = role.eq("fault_detection")
        outer_normal = role.eq("normal_reference") & frame["normal_calibration_holdout"]
        wind = role.eq("environment_robustness")
        detected = frame["detected"].eq(True) | frame["detected"].astype(str).str.lower().eq("true")
        real_motor = fault & frame["domain"].eq("Real") & frame["fault_family"].eq("Motor")
        real_sensor = fault & frame["domain"].eq("Real") & frame["fault_family"].eq("Sensor")
        nonfault = fault | outer_normal
        values = pd.DataFrame({
            "canonical_case_id": frame["canonical_case_id"],
            "event_num": (fault & detected).astype(float),
            "event_den": fault.astype(float),
            "motor_num": (real_motor & detected).astype(float),
            "motor_den": real_motor.astype(float),
            "sensor_num": (real_sensor & detected).astype(float),
            "sensor_den": real_sensor.astype(float),
            "real_fa_num": frame["false_alarm_events"].where(
                outer_normal & frame["domain"].eq("Real"), 0.0
            ),
            "real_fa_den": frame["normal_exposure_hours"].where(
                outer_normal & frame["domain"].eq("Real"), 0.0
            ),
            "real_alarm_num": frame["alarm_events"].gt(0).where(
                outer_normal & frame["domain"].eq("Real"), False
            ).astype(float),
            "real_alarm_den": (
                outer_normal & frame["domain"].eq("Real")
            ).astype(float),
            "nonfault_num": frame["false_alarm_events"].where(nonfault, 0.0),
            "nonfault_den": frame["normal_exposure_hours"].where(nonfault, 0.0),
            "wind_num": frame["false_alarm_events"].where(wind, 0.0),
            "wind_den": frame["normal_exposure_hours"].where(wind, 0.0),
        })
        clusters = values.groupby("canonical_case_id", sort=False).sum()
        arrays = {name: clusters[name].to_numpy(float) for name in clusters.columns}
        samples = {name: [] for name in metric_names}
        cluster_count = len(clusters)
        for _ in range(BOOTSTRAP_SAMPLES):
            chosen = rng.integers(0, cluster_count, size=cluster_count)

            def ratio(numerator: str, denominator: str) -> float:
                den = arrays[denominator][chosen].sum()
                return float(arrays[numerator][chosen].sum() / den) if den else np.nan

            motor = ratio("motor_num", "motor_den")
            sensor = ratio("sensor_num", "sensor_den")
            samples["event_recall"].append(ratio("event_num", "event_den"))
            samples["real_motor_recall"].append(motor)
            samples["real_sensor_recall"].append(sensor)
            samples["real_macro_recall"].append(float(np.nanmean([motor, sensor])))
            samples["real_normal_fa_per_hour"].append(ratio("real_fa_num", "real_fa_den"))
            samples["real_normal_alarm_flight_rate"].append(
                ratio("real_alarm_num", "real_alarm_den")
            )
            samples["all_nonfault_fa_per_hour"].append(ratio("nonfault_num", "nonfault_den"))
            samples["wind_fa_per_hour"].append(ratio("wind_num", "wind_den"))
        result[policy_name] = {
            name: {
                "lower_95": float(np.nanquantile(values_, 0.025)),
                "upper_95": float(np.nanquantile(values_, 0.975)),
            }
            for name, values_ in samples.items()
        }
    return {
        "method": "canonical-flight cluster bootstrap; rotations stay in cluster",
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "policies": result,
    }


def run_candidate(
    root: Path, candidate: str, *, torch_threads: int = 4,
) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"candidate must be one of {CANDIDATES}")
    _assert_candidate_is_due(root, candidate)
    torch.set_num_threads(max(1, torch_threads))
    available = _experiment_manifest(root)
    output_root = root / "candidates" / candidate
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for rotation in ROTATIONS:
        output = output_root / f"rotation_{rotation}"
        if (output / "summary.json").exists():
            completed.append(rotation)
            continue
        output.mkdir(parents=True, exist_ok=True)
        _, inner_normal, outer_normal, normal_groups = _nested_normal_split(
            available, rotation
        )
        _, inner_wind, outer_wind, wind_groups = _nested_wind_split(
            available, rotation
        )
        model, scaler, scored, history, model_source = _candidate_model_and_scores(
            root, candidate, rotation, available
        )
        policies = _fit_candidate_policies(
            scored,
            inner_normal_ids=set(inner_normal["canonical_case_id"].astype(str)),
            inner_wind_ids=set(inner_wind["canonical_case_id"].astype(str)),
            real_relaxed=candidate == "R1",
            environment_aware=candidate == "W1",
        )
        evaluated = _evaluation_scores(
            scored,
            outer_normal_ids=set(outer_normal["canonical_case_id"].astype(str)),
            outer_wind_ids=set(outer_wind["canonical_case_id"].astype(str)),
        )
        rotation_metrics, per_flight, auxiliary = _rotation_metrics(
            candidate, rotation, evaluated, policies
        )
        per_flight.insert(0, "rotation", rotation)
        per_flight.insert(0, "candidate", candidate)
        rotation_metrics.to_csv(output / "rotation_metrics.csv", index=False)
        per_flight.to_csv(output / "per_flight_metrics.csv", index=False)
        auxiliary["metrics"].to_csv(output / "operational_metrics.csv", index=False)
        auxiliary["family"].to_csv(output / "domain_family_metrics.csv", index=False)
        _atomic_json(output / "policies.json", policies)
        if history:
            pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
            torch.save(_model_checkpoint(model, scaler), output / "model.pt")
            _atomic_json(output / "scaler.json", scaler)
        _atomic_json(output / "summary.json", {
            "status": "development_only_robustness",
            "created_at": _now(),
            "candidate": candidate,
            "rotation": rotation,
            "model_source": model_source,
            "normal_groups": normal_groups,
            "wind_groups": wind_groups,
            "inner_calibration_fault_flights": 0,
            "outer_evaluation_used_for_selection": False,
            "locked_test_features_read": False,
            "operational_claim_allowed": False,
        })
        completed.append(rotation)
        _update_experiment_state(
            root,
            status="candidates_in_progress",
            active_candidate=candidate,
            active_candidate_rotations_completed=completed,
        )
        del model, scored, evaluated, per_flight
        gc.collect()

    all_metrics = pd.concat([
        pd.read_csv(output_root / f"rotation_{rotation}" / "rotation_metrics.csv")
        for rotation in ROTATIONS
    ], ignore_index=True)
    all_per_flight = pd.concat([
        pd.read_csv(output_root / f"rotation_{rotation}" / "per_flight_metrics.csv")
        for rotation in ROTATIONS
    ], ignore_index=True)
    gates = _gate_summary(all_metrics)
    bootstrap = _cluster_bootstrap(all_per_flight)
    bootstrap_lower = bootstrap["policies"]["critical"]["real_macro_recall"]["lower_95"]
    feasibility = gates.pop("real_feasibility_gate_without_bootstrap")
    feasibility["conditions"]["bootstrap_macro_lower_ge_0_35"] = bootstrap_lower >= 0.35
    feasibility["passed"] = bool(all(feasibility["conditions"].values()))
    feasibility.pop("note", None)
    gates["real_feasibility_gate"] = feasibility
    gates["frozen_baseline"] = {
        "critical_event_recall_mean": BASELINE_CRITICAL_RECALL,
        "critical_real_macro_recall": BASELINE_REAL_MACRO_RECALL,
        "critical_wind_fa_per_hour": BASELINE_WIND_FA_PER_HOUR,
    }
    all_metrics.to_csv(output_root / "all_rotation_metrics.csv", index=False)
    _atomic_json(output_root / "bootstrap_ci.json", bootstrap)
    _atomic_json(output_root / "gate_summary.json", gates)
    summary = {
        "status": "development_only_robustness",
        "created_at": _now(),
        "candidate": candidate,
        "rotations_completed": list(ROTATIONS),
        "real_research_gate_passed": gates["real_research_gate"]["passed"],
        "real_feasibility_gate_passed": gates["real_feasibility_gate"]["passed"],
        "wind_intermediate_gate_passed": gates["wind_intermediate_gate"]["passed"],
        "wind_final_research_target_passed": gates["wind_final_research_target"]["passed"],
        "locked_test_features_read": False,
        "operational_claim_allowed": False,
    }
    _atomic_json(output_root / "summary.json", summary)
    completed_candidates = []
    for name in CANDIDATES:
        if (root / "candidates" / name / "summary.json").exists():
            completed_candidates.append(name)
    _update_experiment_state(
        root,
        status="candidates_in_progress",
        active_candidate=None,
        candidates_completed=completed_candidates,
    )
    return summary


def run_converged_real_candidate(
    root: Path, *, torch_threads: int = 4,
) -> dict:
    for prerequisite in CANDIDATES:
        if _candidate_summary(root, prerequisite) is None:
            raise RuntimeError(f"{prerequisite} must complete before {CONVERGED_CANDIDATE}")
    torch.set_num_threads(max(1, torch_threads))
    available = _experiment_manifest(root)
    output_root = root / "candidates" / CONVERGED_CANDIDATE
    output_root.mkdir(parents=True, exist_ok=True)
    extension_contract = Path(
        "docs/RFLYMAD_V2_CONVERGENCE_EK_SOZLESME_20260722.md"
    )
    shutil.copy2(extension_contract, root / "convergence_extension_contract.md")
    convergence_rows = []
    for rotation in ROTATIONS:
        output = output_root / f"rotation_{rotation}"
        if (output / "summary.json").exists():
            convergence_rows.append(_read_json(output / "summary.json")["convergence"])
            continue
        output.mkdir(parents=True, exist_ok=True)
        train_normal, inner_normal, outer_normal, normal_groups = (
            _nested_normal_split(available, rotation)
        )
        _, inner_wind, outer_wind, wind_groups = _nested_wind_split(
            available, rotation
        )
        base = root / "base" / f"rotation_{rotation}"
        scaler = _read_json(base / "scaler.json")
        model = _load_model(base / "model.pt", scaler)
        real_train = train_normal.loc[train_normal["domain"].eq("Real")]
        real_inner = inner_normal.loc[inner_normal["domain"].eq("Real")]
        train_windows = _normal_windows(
            real_train, scaler, TRAIN_STRIDE_SECONDS
        )
        validation_windows = _normal_windows(
            real_inner, scaler, VALIDATION_STRIDE_SECONDS
        )
        model, history, convergence = _fine_tune_until_convergence(
            model, train_windows, validation_windows
        )
        scored = _score_manifest(model, available, scaler)
        policies = _fit_candidate_policies(
            scored,
            inner_normal_ids=set(inner_normal["canonical_case_id"].astype(str)),
            inner_wind_ids=set(inner_wind["canonical_case_id"].astype(str)),
            real_relaxed=False,
            environment_aware=False,
        )
        evaluated = _evaluation_scores(
            scored,
            outer_normal_ids=set(outer_normal["canonical_case_id"].astype(str)),
            outer_wind_ids=set(outer_wind["canonical_case_id"].astype(str)),
        )
        rotation_metrics, per_flight, auxiliary = _rotation_metrics(
            CONVERGED_CANDIDATE, rotation, evaluated, policies
        )
        per_flight.insert(0, "rotation", rotation)
        per_flight.insert(0, "candidate", CONVERGED_CANDIDATE)
        rotation_metrics.to_csv(output / "rotation_metrics.csv", index=False)
        per_flight.to_csv(output / "per_flight_metrics.csv", index=False)
        auxiliary["metrics"].to_csv(output / "operational_metrics.csv", index=False)
        auxiliary["family"].to_csv(output / "domain_family_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        torch.save(_model_checkpoint(model, scaler), output / "model.pt")
        _atomic_json(output / "scaler.json", scaler)
        _atomic_json(output / "policies.json", policies)
        rotation_summary = {
            "status": "development_only_convergence_followup",
            "created_at": _now(),
            "candidate": CONVERGED_CANDIDATE,
            "rotation": rotation,
            "model_source": "nested base AE; Real NoFault convergence fine-tune",
            "normal_groups": normal_groups,
            "wind_groups": wind_groups,
            "convergence": convergence,
            "inner_calibration_fault_flights": 0,
            "outer_evaluation_used_for_selection": False,
            "locked_test_features_read": False,
            "operational_claim_allowed": False,
        }
        _atomic_json(output / "summary.json", rotation_summary)
        convergence_rows.append(convergence)
        _update_experiment_state(
            root,
            status="convergence_followup_in_progress",
            active_candidate=CONVERGED_CANDIDATE,
            active_candidate_rotations_completed=list(range(rotation + 1)),
        )
        del model, scored, evaluated, per_flight, train_windows, validation_windows
        gc.collect()
    summary = {
        "status": "development_only_convergence_followup",
        "created_at": _now(),
        "candidate": CONVERGED_CANDIDATE,
        "rotations_completed": list(ROTATIONS),
        "convergence_protocol": {
            "max_epochs": CONVERGENCE_MAX_EPOCHS,
            "patience": CONVERGENCE_PATIENCE,
            "min_delta": CONVERGENCE_MIN_DELTA,
            "checkpoint": "minimum inner Real-NoFault validation loss; epoch 0 allowed",
        },
        "rotation_convergence": convergence_rows,
        "locked_test_features_read": False,
        "operational_claim_allowed": False,
    }
    _atomic_json(output_root / "summary.json", summary)
    summary = _refresh_candidate_report(root, CONVERGED_CANDIDATE)
    render_convergence_plots(root)
    completed = list(CANDIDATES) + [CONVERGED_CANDIDATE]
    _update_experiment_state(
        root,
        status="convergence_followup_complete",
        active_candidate=None,
        candidates_completed=completed,
    )
    return summary


def extend_convergence_ceiling(
    root: Path, *, max_epochs: int, torch_threads: int = 4,
) -> dict:
    if max_epochs not in CONVERGENCE_EXTENSION_CEILINGS:
        raise ValueError(
            f"extension ceiling must be one of {CONVERGENCE_EXTENSION_CEILINGS}"
        )
    torch.set_num_threads(max(1, torch_threads))
    available = _experiment_manifest(root)
    output_root = root / "candidates" / CONVERGED_CANDIDATE
    extended_rotations = []
    for rotation in ROTATIONS:
        output = output_root / f"rotation_{rotation}"
        old_summary = _read_json(output / "summary.json")
        old_convergence = old_summary["convergence"]
        previous_ceiling = int(old_convergence["max_epochs"])
        boundary_hit = (
            old_convergence["stop_reason"] == "max_epochs"
            and int(old_convergence["best_epoch"])
            >= int(old_convergence["epochs_completed"]) - CONVERGENCE_PATIENCE
        )
        if not boundary_hit:
            continue
        if max_epochs <= previous_ceiling:
            continue
        snapshot = output / f"cap_{previous_ceiling}_snapshot"
        snapshot.mkdir(exist_ok=True)
        for source in output.iterdir():
            if source.is_file():
                shutil.copy2(source, snapshot / source.name)
        old_history = pd.read_csv(output / "training_history.csv")

        train_normal, inner_normal, outer_normal, normal_groups = (
            _nested_normal_split(available, rotation)
        )
        _, inner_wind, outer_wind, wind_groups = _nested_wind_split(
            available, rotation
        )
        base = root / "base" / f"rotation_{rotation}"
        scaler = _read_json(base / "scaler.json")
        model = _load_model(base / "model.pt", scaler)
        real_train = train_normal.loc[train_normal["domain"].eq("Real")]
        real_inner = inner_normal.loc[inner_normal["domain"].eq("Real")]
        train_windows = _normal_windows(
            real_train, scaler, TRAIN_STRIDE_SECONDS
        )
        validation_windows = _normal_windows(
            real_inner, scaler, VALIDATION_STRIDE_SECONDS
        )
        model, history, convergence = _fine_tune_until_convergence(
            model, train_windows, validation_windows, max_epochs=max_epochs
        )
        history_frame = pd.DataFrame(history)
        overlap = min(len(old_history), len(history_frame))
        max_validation_replay_delta = float(np.max(np.abs(
            old_history["validation_loss"].iloc[:overlap].to_numpy(float)
            - history_frame["validation_loss"].iloc[:overlap].to_numpy(float)
        )))
        scored = _score_manifest(model, available, scaler)
        policies = _fit_candidate_policies(
            scored,
            inner_normal_ids=set(inner_normal["canonical_case_id"].astype(str)),
            inner_wind_ids=set(inner_wind["canonical_case_id"].astype(str)),
            real_relaxed=False,
            environment_aware=False,
        )
        evaluated = _evaluation_scores(
            scored,
            outer_normal_ids=set(outer_normal["canonical_case_id"].astype(str)),
            outer_wind_ids=set(outer_wind["canonical_case_id"].astype(str)),
        )
        rotation_metrics, per_flight, auxiliary = _rotation_metrics(
            CONVERGED_CANDIDATE, rotation, evaluated, policies
        )
        per_flight.insert(0, "rotation", rotation)
        per_flight.insert(0, "candidate", CONVERGED_CANDIDATE)
        rotation_metrics.to_csv(output / "rotation_metrics.csv", index=False)
        per_flight.to_csv(output / "per_flight_metrics.csv", index=False)
        auxiliary["metrics"].to_csv(output / "operational_metrics.csv", index=False)
        auxiliary["family"].to_csv(output / "domain_family_metrics.csv", index=False)
        history_frame.to_csv(output / "training_history.csv", index=False)
        torch.save(_model_checkpoint(model, scaler), output / "model.pt")
        _atomic_json(output / "scaler.json", scaler)
        _atomic_json(output / "policies.json", policies)
        convergence["extended_from_ceiling"] = previous_ceiling
        convergence["validation_replay_max_abs_delta"] = max_validation_replay_delta
        _atomic_json(output / "summary.json", {
            "status": "development_only_convergence_followup",
            "created_at": old_summary["created_at"],
            "extended_at": _now(),
            "candidate": CONVERGED_CANDIDATE,
            "rotation": rotation,
            "model_source": "nested base AE; Real NoFault convergence fine-tune",
            "normal_groups": normal_groups,
            "wind_groups": wind_groups,
            "convergence": convergence,
            "inner_calibration_fault_flights": 0,
            "outer_evaluation_used_for_selection": False,
            "locked_test_features_read": False,
            "operational_claim_allowed": False,
        })
        extended_rotations.append(rotation)
        _update_experiment_state(
            root,
            status="convergence_ceiling_extension_in_progress",
            active_candidate=CONVERGED_CANDIDATE,
            convergence_extension_ceiling=max_epochs,
            convergence_extended_rotations=extended_rotations,
        )
        del model, scored, evaluated, per_flight, train_windows, validation_windows
        gc.collect()
    rotation_convergence = [
        _read_json(output_root / f"rotation_{rotation}" / "summary.json")["convergence"]
        for rotation in ROTATIONS
    ]
    summary = _read_json(output_root / "summary.json")
    summary.update({
        "status": "development_only_convergence_followup",
        "convergence_protocol": {
            "initial_max_epochs": CONVERGENCE_MAX_EPOCHS,
            "extension_ceiling": max_epochs,
            "patience": CONVERGENCE_PATIENCE,
            "min_delta": CONVERGENCE_MIN_DELTA,
            "checkpoint": "minimum inner Real-NoFault validation loss; epoch 0 allowed",
        },
        "rotation_convergence": rotation_convergence,
        "last_ceiling_extension_at": _now(),
    })
    _atomic_json(output_root / "summary.json", summary)
    summary = _refresh_candidate_report(root, CONVERGED_CANDIDATE)
    render_convergence_plots(root)
    _update_experiment_state(
        root,
        status="convergence_followup_complete",
        active_candidate=None,
        convergence_extension_ceiling=max_epochs,
        convergence_extended_rotations=extended_rotations,
    )
    return summary


def render_convergence_plots(root: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root = root / "candidates" / CONVERGED_CANDIDATE
    colors = {"R2": "#4c78a8", "R3": "#f58518", "R4": "#54a24b"}
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    for rotation, axis in enumerate(axes):
        for candidate in ("R2", "R3", CONVERGED_CANDIDATE):
            history = pd.read_csv(
                root / "candidates" / candidate / f"rotation_{rotation}"
                / "training_history.csv"
            )
            history = history.loc[history["epoch"].le(25)]
            axis.plot(
                history["epoch"], history["validation_loss"], marker="o",
                markersize=2.5, linewidth=1.4, label=candidate,
                color=colors[candidate],
            )
        axis.set_title(f"Rotation {rotation}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Inner Real validation loss")
    axes[-1].legend()
    fig.suptitle("Real fine-tune validation loss: first 25 epochs")
    fig.tight_layout()
    zoom_path = output_root / "00_validation_loss_R2_R3_R4_first25.png"
    fig.savefig(zoom_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    for rotation, axis in enumerate(axes):
        for candidate in ("R2", "R3", CONVERGED_CANDIDATE):
            history = pd.read_csv(
                root / "candidates" / candidate / f"rotation_{rotation}"
                / "training_history.csv"
            )
            axis.plot(
                history["epoch"], history["validation_loss"], marker="o",
                markersize=2.5, linewidth=1.4, label=candidate,
                color=colors[candidate],
            )
        axis.set_title(f"Rotation {rotation}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Inner Real validation loss")
    axes[-1].legend()
    fig.suptitle("Real fine-tune validation loss by epoch")
    fig.tight_layout()
    comparison_path = output_root / "01_validation_loss_R2_R3_R4.png"
    fig.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    best_epochs = []
    stop_epochs = []
    for rotation, axis in enumerate(axes):
        history = pd.read_csv(
            output_root / f"rotation_{rotation}" / "training_history.csv"
        )
        trained = history.loc[history["epoch"].gt(0)]
        axis.plot(
            trained["epoch"], trained["train_loss"], label="train",
            color="#4c78a8", linewidth=1.4,
        )
        axis.plot(
            history["epoch"], history["validation_loss"], label="validation",
            color="#e45756", linewidth=1.4,
        )
        summary = _read_json(output_root / f"rotation_{rotation}" / "summary.json")
        best_epoch = int(summary["convergence"]["best_epoch"])
        stop_epoch = int(summary["convergence"]["epochs_completed"])
        best_epochs.append(best_epoch)
        stop_epochs.append(stop_epoch)
        axis.axvline(best_epoch, color="#54a24b", linestyle="--", linewidth=1)
        axis.set_title(f"Rot {rotation}: best={best_epoch}, stop={stop_epoch}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Masked reconstruction loss")
    axes[-1].legend()
    fig.suptitle("R4 convergence fine-tune: train and inner validation loss")
    fig.tight_layout()
    curve_path = output_root / "02_R4_train_validation_by_epoch.png"
    fig.savefig(curve_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(ROTATIONS))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(x - 0.18, best_epochs, 0.36, label="best epoch", color="#54a24b")
    axis.bar(x + 0.18, stop_epochs, 0.36, label="stop epoch", color="#e45756")
    axis.set_xticks(x, [str(value) for value in ROTATIONS])
    axis.set_xlabel("Rotation")
    axis.set_ylabel("Epoch")
    axis.set_title("R4 best-checkpoint and early-stop epochs")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    epoch_path = output_root / "03_R4_best_and_stop_epochs.png"
    fig.savefig(epoch_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [zoom_path, comparison_path, curve_path, epoch_path]


def _candidate_summary(root: Path, candidate: str) -> dict | None:
    path = root / "candidates" / candidate / "summary.json"
    return _read_json(path) if path.exists() else None


def _assert_candidate_is_due(root: Path, candidate: str) -> None:
    if candidate == "R1":
        return
    r1 = _candidate_summary(root, "R1")
    if r1 is None:
        raise RuntimeError("R1 must complete before later candidates")
    if candidate == "W1":
        return
    w1 = _candidate_summary(root, "W1")
    if w1 is None:
        raise RuntimeError("W1 must complete before later candidates")
    if candidate == "W2":
        if w1["wind_intermediate_gate_passed"]:
            raise RuntimeError("W2 is skipped because W1 passed its Wind gate")
        return
    if not w1["wind_intermediate_gate_passed"] and _candidate_summary(root, "W2") is None:
        raise RuntimeError("W2 must complete after a failing W1 and before Real fine-tunes")
    if r1["real_research_gate_passed"]:
        raise RuntimeError(f"{candidate} is skipped because R1 passed its Real gate")
    if candidate == "R2":
        return
    r2 = _candidate_summary(root, "R2")
    if r2 is None:
        raise RuntimeError("R2 must complete before R3")
    if r2["real_research_gate_passed"]:
        raise RuntimeError("R3 is skipped because R2 passed its Real gate")


def select_rw1_components(root: Path) -> dict:
    real = []
    wind = []
    r1 = _candidate_summary(root, "R1")
    w1 = _candidate_summary(root, "W1")
    if r1 is None or w1 is None:
        raise RuntimeError("R1 and W1 must complete before the RW1 decision")
    if r1["real_research_gate_passed"]:
        real.append("R1")
    else:
        r2 = _candidate_summary(root, "R2")
        if r2 is None:
            raise RuntimeError("R2 is required after R1 fails")
        if r2["real_research_gate_passed"]:
            real.append("R2")
        else:
            r3 = _candidate_summary(root, "R3")
            if r3 is None:
                raise RuntimeError("R3 is required after R1 and R2 fail")
            if r3["real_research_gate_passed"]:
                real.append("R3")
            else:
                r4 = _candidate_summary(root, CONVERGED_CANDIDATE)
                if r4 is not None and r4["real_research_gate_passed"]:
                    real.append(CONVERGED_CANDIDATE)
    if w1["wind_intermediate_gate_passed"]:
        wind.append("W1")
    else:
        w2 = _candidate_summary(root, "W2")
        if w2 is None:
            raise RuntimeError("W2 is required after W1 fails")
        if w2["wind_intermediate_gate_passed"]:
            wind.append("W2")
    decision = {
        "created_at": _now(),
        "selection_rule_frozen_before_results": "stop at first passing candidate per target",
        "passing_real_candidates": real,
        "passing_wind_candidates": wind,
        "real_component": real[0] if real else None,
        "wind_component": wind[0] if wind else None,
        "rw1_required": bool(real and wind),
        "locked_test_features_read": False,
    }
    _atomic_json(root / "rw1_decision.json", decision)
    return decision


def run_rw1(root: Path, *, torch_threads: int = 4) -> dict:
    decision = select_rw1_components(root)
    if not decision["rw1_required"]:
        return decision
    torch.set_num_threads(max(1, torch_threads))
    available = _experiment_manifest(root)
    real_component = str(decision["real_component"])
    wind_component = str(decision["wind_component"])
    output_root = root / "candidates" / "RW1"
    output_root.mkdir(parents=True, exist_ok=True)
    for rotation in ROTATIONS:
        output = output_root / f"rotation_{rotation}"
        if (output / "summary.json").exists():
            continue
        output.mkdir(parents=True, exist_ok=True)
        train_normal, inner_normal, outer_normal, normal_groups = (
            _nested_normal_split(available, rotation)
        )
        _, inner_wind, outer_wind, wind_groups = _nested_wind_split(
            available, rotation
        )
        if wind_component == "W2":
            source_root = root / "candidates" / "W2" / f"rotation_{rotation}"
        else:
            source_root = root / "base" / f"rotation_{rotation}"
        scaler = _read_json(source_root / "scaler.json")
        model = _load_model(source_root / "model.pt", scaler)
        history: list[dict] = []
        if real_component in {"R2", "R3"}:
            if wind_component == "W1":
                source_root = root / "candidates" / real_component / f"rotation_{rotation}"
                scaler = _read_json(source_root / "scaler.json")
                model = _load_model(source_root / "model.pt", scaler)
            else:
                real_train = train_normal.loc[train_normal["domain"].eq("Real")]
                real_inner = inner_normal.loc[inner_normal["domain"].eq("Real")]
                train_windows = _normal_windows(
                    real_train, scaler, TRAIN_STRIDE_SECONDS
                )
                validation_windows = _normal_windows(
                    real_inner, scaler, VALIDATION_STRIDE_SECONDS
                )
                epochs = 3 if real_component == "R2" else 8
                model, history = _fine_tune(
                    model, train_windows, validation_windows, epochs=epochs
                )
                del train_windows, validation_windows
        scored = _score_manifest(model, available, scaler)
        policies = _fit_candidate_policies(
            scored,
            inner_normal_ids=set(inner_normal["canonical_case_id"].astype(str)),
            inner_wind_ids=set(inner_wind["canonical_case_id"].astype(str)),
            real_relaxed=real_component == "R1",
            environment_aware=wind_component == "W1",
        )
        evaluated = _evaluation_scores(
            scored,
            outer_normal_ids=set(outer_normal["canonical_case_id"].astype(str)),
            outer_wind_ids=set(outer_wind["canonical_case_id"].astype(str)),
        )
        rotation_metrics, per_flight, auxiliary = _rotation_metrics(
            "RW1", rotation, evaluated, policies
        )
        per_flight.insert(0, "rotation", rotation)
        per_flight.insert(0, "candidate", "RW1")
        rotation_metrics.to_csv(output / "rotation_metrics.csv", index=False)
        per_flight.to_csv(output / "per_flight_metrics.csv", index=False)
        auxiliary["metrics"].to_csv(output / "operational_metrics.csv", index=False)
        auxiliary["family"].to_csv(output / "domain_family_metrics.csv", index=False)
        _atomic_json(output / "policies.json", policies)
        torch.save(_model_checkpoint(model, scaler), output / "model.pt")
        _atomic_json(output / "scaler.json", scaler)
        if history:
            pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        _atomic_json(output / "summary.json", {
            "status": "development_only_robustness",
            "created_at": _now(),
            "candidate": "RW1",
            "rotation": rotation,
            "real_component": real_component,
            "wind_component": wind_component,
            "normal_groups": normal_groups,
            "wind_groups": wind_groups,
            "inner_calibration_fault_flights": 0,
            "outer_evaluation_used_for_selection": False,
            "locked_test_features_read": False,
            "operational_claim_allowed": False,
        })
        del model, scored, evaluated, per_flight
        gc.collect()
    return _aggregate_rw1(root, decision)


def _aggregate_rw1(root: Path, decision: dict) -> dict:
    output_root = root / "candidates" / "RW1"
    all_metrics = pd.concat([
        pd.read_csv(output_root / f"rotation_{rotation}" / "rotation_metrics.csv")
        for rotation in ROTATIONS
    ], ignore_index=True)
    all_per_flight = pd.concat([
        pd.read_csv(output_root / f"rotation_{rotation}" / "per_flight_metrics.csv")
        for rotation in ROTATIONS
    ], ignore_index=True)
    gates = _gate_summary(all_metrics)
    bootstrap = _cluster_bootstrap(all_per_flight)
    bootstrap_lower = bootstrap["policies"]["critical"]["real_macro_recall"]["lower_95"]
    feasibility = gates.pop("real_feasibility_gate_without_bootstrap")
    feasibility["conditions"]["bootstrap_macro_lower_ge_0_35"] = bootstrap_lower >= 0.35
    feasibility["passed"] = bool(all(feasibility["conditions"].values()))
    feasibility.pop("note", None)
    gates["real_feasibility_gate"] = feasibility
    gates["frozen_baseline"] = {
        "critical_event_recall_mean": BASELINE_CRITICAL_RECALL,
        "critical_real_macro_recall": BASELINE_REAL_MACRO_RECALL,
        "critical_wind_fa_per_hour": BASELINE_WIND_FA_PER_HOUR,
    }
    all_metrics.to_csv(output_root / "all_rotation_metrics.csv", index=False)
    _atomic_json(output_root / "bootstrap_ci.json", bootstrap)
    _atomic_json(output_root / "gate_summary.json", gates)
    summary = {
        "status": "development_only_robustness",
        "created_at": _now(),
        "candidate": "RW1",
        "real_component": decision["real_component"],
        "wind_component": decision["wind_component"],
        "rotations_completed": list(ROTATIONS),
        "real_research_gate_passed": gates["real_research_gate"]["passed"],
        "real_feasibility_gate_passed": gates["real_feasibility_gate"]["passed"],
        "wind_intermediate_gate_passed": gates["wind_intermediate_gate"]["passed"],
        "wind_final_research_target_passed": gates["wind_final_research_target"]["passed"],
        "locked_test_features_read": False,
        "operational_claim_allowed": False,
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


def _refresh_candidate_report(root: Path, candidate: str) -> dict:
    output_root = root / "candidates" / candidate
    for rotation in ROTATIONS:
        rotation_root = output_root / f"rotation_{rotation}"
        metrics = pd.read_csv(rotation_root / "rotation_metrics.csv")
        per_flight = pd.read_csv(rotation_root / "per_flight_metrics.csv")
        for policy_name in POLICY_NAMES:
            real_normal = per_flight.loc[
                per_flight["policy"].eq(policy_name)
                & per_flight["domain"].eq("Real")
                & per_flight["evaluation_role"].eq("normal_reference")
                & per_flight["normal_calibration_holdout"]
            ]
            rate = (
                float(real_normal["alarm_events"].gt(0).mean())
                if len(real_normal) else float("nan")
            )
            metrics.loc[
                metrics["policy"].eq(policy_name),
                "real_normal_alarm_flight_rate",
            ] = rate
        metrics.to_csv(rotation_root / "rotation_metrics.csv", index=False)
    all_metrics = pd.concat([
        pd.read_csv(output_root / f"rotation_{rotation}" / "rotation_metrics.csv")
        for rotation in ROTATIONS
    ], ignore_index=True)
    all_per_flight = pd.concat([
        pd.read_csv(output_root / f"rotation_{rotation}" / "per_flight_metrics.csv")
        for rotation in ROTATIONS
    ], ignore_index=True)
    gates = _gate_summary(all_metrics)
    bootstrap = _cluster_bootstrap(all_per_flight)
    bootstrap_lower = bootstrap["policies"]["critical"]["real_macro_recall"]["lower_95"]
    feasibility = gates.pop("real_feasibility_gate_without_bootstrap")
    feasibility["conditions"]["bootstrap_macro_lower_ge_0_35"] = bootstrap_lower >= 0.35
    feasibility["passed"] = bool(all(feasibility["conditions"].values()))
    feasibility.pop("note", None)
    gates["real_feasibility_gate"] = feasibility
    gates["frozen_baseline"] = {
        "critical_event_recall_mean": BASELINE_CRITICAL_RECALL,
        "critical_real_macro_recall": BASELINE_REAL_MACRO_RECALL,
        "critical_wind_fa_per_hour": BASELINE_WIND_FA_PER_HOUR,
    }
    all_metrics.to_csv(output_root / "all_rotation_metrics.csv", index=False)
    _atomic_json(output_root / "bootstrap_ci.json", bootstrap)
    _atomic_json(output_root / "gate_summary.json", gates)
    summary = _read_json(output_root / "summary.json")
    summary.update({
        "report_refreshed_at": _now(),
        "real_research_gate_passed": gates["real_research_gate"]["passed"],
        "real_feasibility_gate_passed": gates["real_feasibility_gate"]["passed"],
        "wind_intermediate_gate_passed": gates["wind_intermediate_gate"]["passed"],
        "wind_final_research_target_passed": gates["wind_final_research_target"]["passed"],
    })
    _atomic_json(output_root / "summary.json", summary)
    return summary


def refresh_candidate_reports(root: Path) -> None:
    for candidate in (*CANDIDATES, CONVERGED_CANDIDATE, "RW1"):
        if (root / "candidates" / candidate / "summary.json").exists():
            _refresh_candidate_report(root, candidate)


def _frozen_baseline_rotation_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(FROZEN_BASELINE_ROOT / "all_metrics.csv").rename(columns={
        "validation_rotation": "rotation",
        "environment_fa_per_hour": "wind_fa_per_hour",
    })
    progress = _read_json(FROZEN_BASELINE_ROOT / "progress.json")
    additions = []
    for completed in progress["completed"]:
        rotation = int(completed["rotation"])
        run_root = FROZEN_BASELINE_ROOT.parent / Path(completed["output"]).name
        family = pd.read_csv(run_root / "domain_family_metrics.csv")
        per_flight = pd.read_csv(run_root / "per_flight_metrics.csv")
        for policy_name in POLICY_NAMES:
            selected_family = family.loc[
                family["policy"].eq(policy_name)
                & family["domain"].eq("Real")
                & family["fault_family"].isin(["Motor", "Sensor"])
            ].set_index("fault_family")["recall"]
            real_normal = per_flight.loc[
                per_flight["policy"].eq(policy_name)
                & per_flight["domain"].eq("Real")
                & per_flight["evaluation_role"].eq("normal_reference")
                & per_flight["normal_calibration_holdout"]
            ]
            exposure = real_normal["normal_exposure_hours"].sum()
            motor = float(selected_family.get("Motor", np.nan))
            sensor = float(selected_family.get("Sensor", np.nan))
            additions.append({
                "rotation": rotation,
                "policy": policy_name,
                "real_motor_recall": motor,
                "real_sensor_recall": sensor,
                "real_macro_recall": float(np.nanmean([motor, sensor])),
                "real_normal_fa_per_hour": (
                    float(real_normal["false_alarm_events"].sum() / exposure)
                    if exposure else float("nan")
                ),
                "real_normal_alarm_flight_rate": (
                    float(real_normal["alarm_events"].gt(0).mean())
                    if len(real_normal) else float("nan")
                ),
            })
    return metrics.merge(pd.DataFrame(additions), on=["rotation", "policy"])


def _write_policy_comparison(root: Path, candidate_names: list[str]) -> Path:
    metric_names = (
        "event_recall", "all_nonfault_fa_per_hour", "wind_fa_per_hour",
        "real_motor_recall", "real_sensor_recall", "real_macro_recall",
        "real_normal_fa_per_hour", "real_normal_alarm_flight_rate",
    )
    frames = [("frozen_baseline", "frozen_single_holdout", _frozen_baseline_rotation_metrics())]
    for candidate in candidate_names:
        frame = pd.read_csv(
            root / "candidates" / candidate / "all_rotation_metrics.csv"
        )
        frames.append((candidate, "nested_inner_outer", frame))
    rows = []
    for candidate, protocol, frame in frames:
        for policy_name in POLICY_NAMES:
            selected = frame.loc[frame["policy"].eq(policy_name)]
            row: dict[str, object] = {
                "candidate": candidate,
                "protocol": protocol,
                "policy": policy_name,
            }
            for metric_name in metric_names:
                row[f"{metric_name}_mean"] = float(selected[metric_name].mean())
                row[f"{metric_name}_std"] = float(selected[metric_name].std())
                row[f"{metric_name}_min"] = float(selected[metric_name].min())
                row[f"{metric_name}_max"] = float(selected[metric_name].max())
            if candidate == "frozen_baseline" or policy_name != "critical":
                row["real_gate_passed"] = None
                row["wind_gate_passed"] = None
            else:
                summary = _read_json(root / "candidates" / candidate / "summary.json")
                row["real_gate_passed"] = summary["real_research_gate_passed"]
                row["wind_gate_passed"] = summary["wind_intermediate_gate_passed"]
            rows.append(row)
    path = root / "candidate_comparison_by_policy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def finalize_experiment(root: Path) -> dict:
    decision = select_rw1_components(root)
    candidate_names = [
        candidate for candidate in (*CANDIDATES, CONVERGED_CANDIDATE)
        if _candidate_summary(root, candidate) is not None
    ]
    if decision["rw1_required"]:
        if not (root / "candidates" / "RW1" / "summary.json").exists():
            summary = {
                "status": "awaiting_conditional_rw1",
                "rw1_decision": decision,
                "locked_test_features_read": False,
                "operational_claim_allowed": False,
            }
            _atomic_json(root / "final_summary.json", summary)
            _update_experiment_state(root, status="candidates_in_progress")
            return summary
        candidate_names.append("RW1")
    rows = [{
        "candidate": "frozen_baseline",
        "critical_event_recall_mean": BASELINE_CRITICAL_RECALL,
        "critical_all_nonfault_fa_mean": 1.278806935954617,
        "critical_wind_fa_mean": BASELINE_WIND_FA_PER_HOUR,
        "critical_real_macro_recall_mean": BASELINE_REAL_MACRO_RECALL,
        "real_gate_passed": None,
        "wind_gate_passed": None,
    }]
    summaries = {}
    for candidate in candidate_names:
        candidate_root = root / "candidates" / candidate
        summary = _read_json(candidate_root / "summary.json")
        gates = _read_json(candidate_root / "gate_summary.json")
        aggregate = gates["critical_aggregate"]
        rows.append({
            "candidate": candidate,
            "critical_event_recall_mean": aggregate["event_recall"]["mean"],
            "critical_all_nonfault_fa_mean": aggregate["all_nonfault_fa_per_hour"]["mean"],
            "critical_wind_fa_mean": aggregate["wind_fa_per_hour"]["mean"],
            "critical_real_macro_recall_mean": aggregate["real_macro_recall"]["mean"],
            "real_gate_passed": summary["real_research_gate_passed"],
            "wind_gate_passed": summary["wind_intermediate_gate_passed"],
        })
        summaries[candidate] = summary
    pd.DataFrame(rows).to_csv(root / "candidate_comparison.csv", index=False)
    policy_comparison = _write_policy_comparison(root, candidate_names)
    any_real = any(value["real_research_gate_passed"] for value in summaries.values())
    any_wind = any(value["wind_intermediate_gate_passed"] for value in summaries.values())
    final = {
        "status": "development_only_robustness_complete",
        "created_at": _now(),
        "rw1_decision": decision,
        "candidate_summaries": summaries,
        "policy_comparison": str(policy_comparison),
        "real_conclusion": (
            "research-promotion signal observed; not a feasibility or operational claim"
            if any_real else "current data/representation did not demonstrate Real transfer"
        ),
        "wind_conclusion": (
            "intermediate development gate passed; not an operational alarm-load claim"
            if any_wind else "Wind robustness remains unresolved"
        ),
        "locked_test_features_read": False,
        "operational_claim_allowed": False,
    }
    _atomic_json(root / "final_summary.json", final)
    _update_experiment_state(root, status="complete", candidates_completed=candidate_names)
    return final
