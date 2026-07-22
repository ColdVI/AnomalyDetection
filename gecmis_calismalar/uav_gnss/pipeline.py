"""Decision-frozen execution pipeline for the UAV GNSS-integrity feasibility pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from anomaly_core.calibration import (
    ConditionalCalibrationConfig,
    HierarchicalConformalCalibrator,
    NATURAL_CALIBRATION_ROLE,
)
from anomaly_core.forecaster import (
    NATURAL_FIT_ROLE,
    ForecasterConfig,
    ResidualForecaster,
    score_forecaster,
    train_forecaster,
)
from anomaly_core.sequential import MultiChannelPageCUSUM, PageCUSUMConfig
from uav_gnss.catalog import build_role_manifest, cases_for_role
from uav_gnss.evaluation import deadline_event_metrics, natural_burden
from uav_gnss.features import MODEL_INPUT_CHANNELS, Z_CHANNELS, load_role_features

NAMESPACE = "uav_gnss_integrity_v1"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class RobustScaler:
    clip: float
    calibration: dict[str, dict[str, float]] | None = None
    excluded: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> "RobustScaler":
        calibration: dict[str, dict[str, float]] = {}
        excluded: list[str] = []
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                excluded.append(column)
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)) * 1.4826)
            if not np.isfinite(mad) or mad == 0:
                excluded.append(column)
            else:
                calibration[column] = {"median": median, "mad": mad}
        if not calibration:
            raise ValueError("all model inputs were excluded by natural-only robust scaling")
        self.calibration = calibration
        self.excluded = tuple(excluded)
        return self

    @property
    def active(self) -> tuple[str, ...]:
        if self.calibration is None:
            raise RuntimeError("scaler is not fitted")
        return tuple(self.calibration)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.calibration is None:
            raise RuntimeError("scaler is not fitted")
        result = pd.DataFrame(index=frame.index)
        for column, values in self.calibration.items():
            numeric = pd.to_numeric(frame[column], errors="coerce")
            result[column] = ((numeric - values["median"]) / values["mad"]).clip(
                -self.clip, self.clip
            )
        return result


@dataclass
class ForecastBatch:
    X: np.ndarray
    X_mask: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    meta: pd.DataFrame
    input_channels: tuple[str, ...]
    target_channels: tuple[str, ...]


def build_windows(
    frame: pd.DataFrame,
    *,
    scaler: RobustScaler,
    history_rows: int,
    max_gap_s: float,
    target_channels: tuple[str, ...],
) -> ForecastBatch:
    scaled = scaler.transform(frame)
    input_channels = scaler.active
    target_active = tuple(channel for channel in target_channels if channel in input_channels)
    if not target_active:
        raise ValueError("no target channel survived strict scaling")
    matrix = scaled.loc[:, input_channels].to_numpy(float)
    matrix_mask = np.isfinite(matrix)
    matrix = np.where(matrix_mask, matrix, 0.0)
    target_positions = [input_channels.index(channel) for channel in target_active]
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for flight_id, group in frame.groupby("flight_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        times = group["timestamp_s"].to_numpy(float)
        for local_target in range(history_rows, len(group)):
            start = local_target - history_rows
            interval = times[start : local_target + 1]
            if np.any(np.diff(interval) <= 0) or np.any(np.diff(interval) > max_gap_s):
                continue
            target_index = positions[local_target]
            if not bool(frame.at[target_index, "evaluable"]):
                continue
            history = positions[start:local_target]
            windows.append(matrix[history])
            masks.append(matrix_mask[history].astype(float))
            y = matrix[target_index, target_positions]
            ym = matrix_mask[target_index, target_positions].astype(float)
            targets.append(np.where(ym > 0, y, 0.0))
            target_masks.append(ym)
            meta.append(
                {
                    "frame_index": int(target_index),
                    "flight_id": flight_id,
                    "timestamp_s": float(times[local_target]),
                    "context_phase": str(group["context_phase"].iloc[local_target]),
                    "context_cadence": str(group["context_cadence"].iloc[local_target]),
                }
            )
    feature_count = len(input_channels)
    target_count = len(target_active)
    return ForecastBatch(
        X=np.asarray(windows, dtype=np.float32).reshape((-1, history_rows, feature_count)),
        X_mask=np.asarray(masks, dtype=np.float32).reshape((-1, history_rows, feature_count)),
        y=np.asarray(targets, dtype=np.float32).reshape((-1, target_count)),
        y_mask=np.asarray(target_masks, dtype=np.float32).reshape((-1, target_count)),
        meta=pd.DataFrame.from_records(meta),
        input_channels=input_channels,
        target_channels=target_active,
    )


def long_scores(batch: ForecastBatch, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for channel_index, channel in enumerate(batch.target_channels):
        valid = batch.y_mask[:, channel_index] > 0
        rows.append(
            pd.DataFrame(
                {
                    "frame_index": batch.meta.loc[valid, "frame_index"].to_numpy(),
                    "flight_id": batch.meta.loc[valid, "flight_id"].to_numpy(),
                    "timestamp_s": batch.meta.loc[valid, "timestamp_s"].to_numpy(),
                    "channel": channel,
                    "context_phase": batch.meta.loc[valid, "context_phase"].to_numpy(),
                    "context_cadence": batch.meta.loc[valid, "context_cadence"].to_numpy(),
                    "score": scores[valid, channel_index],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _episodes_per_hour(frame: pd.DataFrame, alarms: np.ndarray, merge_gap_s: float) -> float:
    value = natural_burden(
        frame, alarms, merge_gap_s=merge_gap_s
    )["episodes_per_scoreable_flight_hour"]
    return float(value) if value is not None else float("inf")


def select_cusum_threshold(
    calibration: pd.DataFrame,
    scores: pd.Series,
    *,
    budget_per_hour: float,
    merge_gap_s: float,
) -> dict[str, Any]:
    finite = pd.to_numeric(scores, errors="coerce").to_numpy(float)
    finite = finite[np.isfinite(finite)]
    candidates = sorted(
        set(np.quantile(finite, np.linspace(0.0, 0.999, 200)).astype(float).tolist())
    )
    candidates.append(float("inf"))
    measured = []
    for threshold in candidates:
        alarms = pd.to_numeric(scores, errors="coerce").to_numpy(float) > threshold
        measured.append(
            {
                "threshold": threshold,
                "episodes_per_hour": _episodes_per_hour(
                    calibration, alarms, merge_gap_s
                ),
            }
        )
    feasible = [row for row in measured if row["episodes_per_hour"] <= budget_per_hour]
    return {"selected": min(feasible, key=lambda row: row["threshold"]), "curve": measured}


def lstm_alarm_vector(
    frame: pd.DataFrame,
    scored_long: pd.DataFrame,
    *,
    total_alpha: float,
    persistence_s: float,
    max_gap_s: float,
) -> np.ndarray:
    alarms = np.zeros(len(frame), dtype=bool)
    channels = tuple(sorted(scored_long["channel"].unique()))
    per_channel_alpha = total_alpha / len(channels)
    for _, channel_rows in scored_long.groupby("channel", sort=False):
        for _, group in channel_rows.groupby("flight_id", sort=False):
            evidence = 0.0
            previous_time: float | None = None
            for row in group.sort_values("timestamp_s").itertuples(index=False):
                timestamp = float(row.timestamp_s)
                if (
                    previous_time is None
                    or timestamp - previous_time <= 0
                    or timestamp - previous_time > max_gap_s
                ):
                    evidence = 0.0
                elif float(row.conformal_p_value) <= per_channel_alpha:
                    evidence += timestamp - previous_time
                else:
                    evidence = 0.0
                if evidence >= persistence_s:
                    alarms[int(row.frame_index)] = True
                previous_time = timestamp
    return alarms


def select_lstm_alpha(
    calibration: pd.DataFrame,
    scored_long: pd.DataFrame,
    *,
    budget_per_hour: float,
    persistence_s: float,
    max_gap_s: float,
    merge_gap_s: float,
) -> dict[str, Any]:
    measured = []
    for alpha in np.geomspace(1e-5, 0.5, 120):
        alarms = lstm_alarm_vector(
            calibration,
            scored_long,
            total_alpha=float(alpha),
            persistence_s=persistence_s,
            max_gap_s=max_gap_s,
        )
        measured.append(
            {
                "total_alpha": float(alpha),
                "episodes_per_hour": _episodes_per_hour(
                    calibration, alarms, merge_gap_s
                ),
            }
        )
    feasible = [row for row in measured if row["episodes_per_hour"] <= budget_per_hour]
    selected = max(feasible, key=lambda row: row["total_alpha"]) if feasible else {
        "total_alpha": 1e-12,
        "episodes_per_hour": 0.0,
    }
    return {"selected": selected, "curve": measured}


def method_metrics(
    normal: pd.DataFrame,
    fault: pd.DataFrame,
    normal_alarms: np.ndarray,
    fault_alarms: np.ndarray,
    *,
    deadline_s: float,
    merge_gap_s: float,
) -> dict[str, Any]:
    return {
        "burden": natural_burden(normal, normal_alarms, merge_gap_s=merge_gap_s),
        "events": deadline_event_metrics(fault, fault_alarms, deadline_s=deadline_s),
    }


def passes_gate(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    burden = metrics["burden"]
    events = metrics["events"]
    by_mode = events["by_fault_mode"]
    return bool(
        burden["scoreable_coverage"] is not None
        and burden["scoreable_coverage"] >= gate["normal_scoreable_coverage_min"]
        and events["event_evaluable_coverage_macro"] is not None
        and events["event_evaluable_coverage_macro"] >= gate["fault_evaluable_coverage_min"]
        and events["recall"] is not None
        and events["recall"] >= gate["recall_min"]
        and events["wilson_95"]["lower"] is not None
        and events["wilson_95"]["lower"] >= gate["wilson_lower_min"]
        and all(
            by_mode[str(mode)]["recall"] is not None
            and by_mode[str(mode)]["recall"] >= gate["per_mode_recall_min"]
            for mode in (3, 4)
        )
        and burden["episodes_per_scoreable_flight_hour"] is not None
        and burden["episodes_per_scoreable_flight_hour"]
        <= gate["max_false_alarms_per_hour"]
    )


class PilotRunner:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.config_sha256 = sha256_payload(self.config)
        self.root = Path(self.config["data"]["bronze_root"])
        self.out = Path(self.config["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)

    def preflight(self) -> dict[str, Any]:
        manifest = build_role_manifest(self.root, self.config_sha256)
        write_json(self.out / "role_manifest.json", manifest)
        result = {
            "status": "ready",
            "candidate_namespace": NAMESPACE,
            "config_path": self.config_path.as_posix(),
            "config_sha256": self.config_sha256,
            "role_counts": manifest["role_counts"],
            "quarantine_policy": "fault_id != 123456 is excluded from GPS metrics",
            "holdout_unsealed": False,
        }
        write_json(self.out / "preflight.json", result)
        return result

    def _load_roles(
        self, roles: tuple[str, ...], *, allow_holdout: bool = False
    ) -> dict[str, pd.DataFrame]:
        max_gap = float(self.config["features"]["max_gap_s"])
        return {
            role: load_role_features(
                cases_for_role(self.root, role, allow_holdout=allow_holdout),
                max_gap_s=max_gap,
            ).reset_index(drop=True)
            for role in roles
        }

    def _fit_models(self, fit: pd.DataFrame) -> dict[str, Any]:
        model_cfg = self.config["lstm"]
        scaler = RobustScaler(clip=float(model_cfg["robust_clip"])).fit(
            fit, MODEL_INPUT_CHANNELS
        )
        batch = build_windows(
            fit,
            scaler=scaler,
            history_rows=int(model_cfg["history_rows"]),
            max_gap_s=float(self.config["features"]["max_gap_s"]),
            target_channels=Z_CHANNELS,
        )
        forecaster_config = ForecasterConfig(
            input_features=len(batch.input_channels),
            target_channels=len(batch.target_channels),
            hidden_size=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["num_layers"]),
            min_scale=float(model_cfg["min_scale"]),
            max_scale=float(model_cfg["max_scale"]),
        )
        model, history = train_forecaster(
            batch.X,
            batch.X_mask,
            batch.y,
            batch.y_mask,
            config=forecaster_config,
            channel_weights=tuple(1.0 for _ in batch.target_channels),
            data_role=NATURAL_FIT_ROLE,
            contains_synthetic=False,
            epochs=int(model_cfg["epochs"]),
            batch_size=int(model_cfg["batch_size"]),
            learning_rate=float(model_cfg["learning_rate"]),
            seed=int(model_cfg["seed"]),
        )
        cusum = MultiChannelPageCUSUM(
            PageCUSUMConfig(
                channels=Z_CHANNELS,
                reference_shift_z=float(self.config["cusum"]["reference_shift_z"]),
                z_clip=float(self.config["cusum"]["z_clip"]),
                max_gap_s=float(self.config["features"]["max_gap_s"]),
            )
        ).fit(fit)
        torch.save(model.state_dict(), self.out / "lstm_state.pt")
        fit_result = {
            "data_role": NATURAL_FIT_ROLE,
            "contains_synthetic": False,
            "n_flights": int(fit["flight_id"].nunique()),
            "n_rows": int(len(fit)),
            "n_windows": int(len(batch.X)),
            "scaler": asdict(scaler),
            "forecaster_config": asdict(forecaster_config),
            "target_channels": list(batch.target_channels),
            "training_loss": history,
            "cusum": cusum.to_dict(),
        }
        write_json(self.out / "fit_result.json", fit_result)
        return {"scaler": scaler, "model": model, "cusum": cusum, "fit_result": fit_result}

    def _lstm_scored(
        self, frame: pd.DataFrame, fitted: dict[str, Any]
    ) -> tuple[pd.DataFrame, ForecastBatch, np.ndarray]:
        batch = build_windows(
            frame,
            scaler=fitted["scaler"],
            history_rows=int(self.config["lstm"]["history_rows"]),
            max_gap_s=float(self.config["features"]["max_gap_s"]),
            target_channels=Z_CHANNELS,
        )
        scores = score_forecaster(
            fitted["model"], batch.X, batch.X_mask, batch.y, batch.y_mask
        )
        return long_scores(batch, scores), batch, scores

    def _calibrate(self, calibration: pd.DataFrame, fitted: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        cal_long, _, _ = self._lstm_scored(calibration, fitted)
        calibrator = HierarchicalConformalCalibrator(
            ConditionalCalibrationConfig(
                min_group_size=int(cfg["lstm"]["min_calibration_group"])
            )
        ).fit(
            cal_long,
            data_role=NATURAL_CALIBRATION_ROLE,
            contains_synthetic=False,
        )
        transformed = calibrator.transform(cal_long)
        cal_long = cal_long.copy()
        cal_long["conformal_p_value"] = transformed["conformal_p_value"].to_numpy()
        cusum_scored = fitted["cusum"].score(calibration)
        result: dict[str, Any] = {
            "data_role": NATURAL_CALIBRATION_ROLE,
            "contains_synthetic": False,
            "n_flights": int(calibration["flight_id"].nunique()),
            "n_rows": int(len(calibration)),
            "conformal": calibrator.to_dict(),
            "contracts": {},
        }
        for name in ("critical", "advisory"):
            contract = cfg["contracts"][name]
            result["contracts"][name] = {
                "cusum": select_cusum_threshold(
                    calibration,
                    cusum_scored["cusum_score"],
                    budget_per_hour=float(contract["max_false_alarms_per_hour"]),
                    merge_gap_s=float(cfg["episodes"]["merge_gap_s"]),
                ),
                "lstm": select_lstm_alpha(
                    calibration,
                    cal_long,
                    budget_per_hour=float(contract["max_false_alarms_per_hour"]),
                    persistence_s=float(cfg["lstm"]["persistence_s"][name]),
                    max_gap_s=float(cfg["features"]["max_gap_s"]),
                    merge_gap_s=float(cfg["episodes"]["merge_gap_s"]),
                ),
            }
        write_json(self.out / "calibration_result.json", result)
        return {
            "calibrator": calibrator,
            "cal_long": cal_long,
            "cusum_scored": cusum_scored,
            "calibration_result": result,
        }

    def _alarms(
        self,
        frame: pd.DataFrame,
        fitted: dict[str, Any],
        calibrated: dict[str, Any],
        contract_name: str,
    ) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
        result = calibrated["calibration_result"]["contracts"][contract_name]
        native_frame = frame.copy().reset_index(drop=True)
        native = native_frame["px4_native_alarm"].to_numpy(bool)
        cusum_frame = frame.copy().reset_index(drop=True)
        cusum_scored = fitted["cusum"].score(cusum_frame)
        cusum_frame["evaluable"] = cusum_scored["cusum_evaluable"].to_numpy(bool)
        threshold = float(result["cusum"]["selected"]["threshold"])
        cusum = cusum_scored["cusum_score"].to_numpy(float) > threshold
        lstm_frame = frame.copy().reset_index(drop=True)
        lstm_long, batch, _ = self._lstm_scored(lstm_frame, fitted)
        conformal = calibrated["calibrator"].transform(lstm_long)
        lstm_long["conformal_p_value"] = conformal["conformal_p_value"].to_numpy()
        scoreable_indices = set(batch.meta["frame_index"].astype(int).tolist())
        lstm_frame["evaluable"] = (
            lstm_frame["evaluable"]
            & lstm_frame.index.to_series().isin(scoreable_indices)
        )
        alpha = float(result["lstm"]["selected"]["total_alpha"])
        lstm = lstm_alarm_vector(
            lstm_frame,
            lstm_long,
            total_alpha=alpha,
            persistence_s=float(self.config["lstm"]["persistence_s"][contract_name]),
            max_gap_s=float(self.config["features"]["max_gap_s"]),
        )
        return {
            "px4_native": (native_frame, native),
            "cusum": (cusum_frame, cusum),
            "lstm": (lstm_frame, lstm),
        }

    def _evaluate_role(
        self,
        normal: pd.DataFrame,
        fault: pd.DataFrame,
        fitted: dict[str, Any],
        calibrated: dict[str, Any],
        role_name: str,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {"role": role_name, "contracts": {}}
        for contract_name, contract in self.config["contracts"].items():
            normal_methods = self._alarms(normal, fitted, calibrated, contract_name)
            fault_methods = self._alarms(fault, fitted, calibrated, contract_name)
            methods = {}
            for method in ("px4_native", "cusum", "lstm"):
                normal_frame, normal_alarm = normal_methods[method]
                fault_frame, fault_alarm = fault_methods[method]
                metrics = method_metrics(
                    normal_frame,
                    fault_frame,
                    normal_alarm,
                    fault_alarm,
                    deadline_s=float(contract["deadline_s"]),
                    merge_gap_s=float(self.config["episodes"]["merge_gap_s"]),
                )
                metrics["passes_gate"] = passes_gate(metrics, contract)
                methods[method] = metrics
            report["contracts"][contract_name] = {"methods": methods}
        critical = report["contracts"]["critical"]["methods"]
        advisory = report["contracts"]["advisory"]["methods"]
        selected = None
        if critical["px4_native"]["passes_gate"]:
            selected = "px4_native"
        elif critical["cusum"]["passes_gate"]:
            selected = "cusum"
        elif (
            critical["lstm"]["passes_gate"]
            and critical["lstm"]["events"]["recall"]
            >= critical["cusum"]["events"]["recall"] + 0.10
        ):
            selected = "lstm"
        report["selected_critical_method"] = selected
        report["preliminary_status"] = (
            "critical_candidate"
            if selected
            else "advisory_candidate"
            if any(value["passes_gate"] for value in advisory.values())
            else "no_go_on_current_role"
        )
        return report

    def _magnitude_diagnostic(
        self, development: pd.DataFrame, fitted: dict[str, Any]
    ) -> dict[str, Any]:
        _, batch, trained_scores = self._lstm_scored(development, fitted)
        random_model = ResidualForecaster(fitted["model"].config)
        random_scores = score_forecaster(
            random_model, batch.X, batch.X_mask, batch.y, batch.y_mask
        )
        valid = batch.y_mask > 0
        trained = trained_scores[valid]
        random = random_scores[valid]
        magnitude = np.abs(batch.y)[valid]
        random_rho = float(spearmanr(trained, random).statistic) if len(trained) > 2 else None
        magnitude_rho = (
            float(spearmanr(trained, magnitude).statistic) if len(trained) > 2 else None
        )
        passed = bool(
            random_rho is not None
            and magnitude_rho is not None
            and random_rho < 0.80
            and magnitude_rho < 0.80
        )
        return {
            "n_cells": int(len(trained)),
            "trained_vs_random_spearman": random_rho,
            "trained_vs_magnitude_spearman": magnitude_rho,
            "gate_threshold": 0.80,
            "passes": passed,
        }

    def run_through(self, stage: str) -> dict[str, Any]:
        self.preflight()
        required_roles = ["fit"]
        if stage in {"calibrate", "develop", "rehearse", "holdout"}:
            required_roles.append("calibration")
        if stage in {"develop", "rehearse", "holdout"}:
            required_roles.append("development")
        if stage in {"rehearse", "holdout"}:
            required_roles.append("rehearsal")
        allow_holdout = stage == "holdout"
        if allow_holdout:
            self._assert_unsealed()
            required_roles.append("holdout")
        role_frames = self._load_roles(
            tuple(dict.fromkeys(required_roles)), allow_holdout=allow_holdout
        )
        fitted = self._fit_models(role_frames["fit"])
        if stage == "fit":
            return fitted["fit_result"]
        calibrated = self._calibrate(role_frames["calibration"], fitted)
        if stage == "calibrate":
            return calibrated["calibration_result"]
        development = role_frames["development"]
        development_result = self._evaluate_role(
            development.loc[development["class"] == "normal"].reset_index(drop=True),
            development.loc[development["class"] == "gps_fault"].reset_index(drop=True),
            fitted,
            calibrated,
            "development",
        )
        diagnostic = self._magnitude_diagnostic(development, fitted)
        development_result["lstm_magnitude_diagnostic"] = diagnostic
        if not diagnostic["passes"]:
            for contract in development_result["contracts"].values():
                contract["methods"]["lstm"]["passes_gate"] = False
            if development_result["selected_critical_method"] == "lstm":
                development_result["selected_critical_method"] = None
        write_json(self.out / "development_result.json", development_result)
        if stage == "develop":
            return development_result
        rehearsal = role_frames["rehearsal"]
        rehearsal_result = self._evaluate_role(
            rehearsal.loc[rehearsal["class"] == "normal"].reset_index(drop=True),
            rehearsal.loc[rehearsal["class"] == "gps_fault"].reset_index(drop=True),
            fitted,
            calibrated,
            "rehearsal",
        )
        rehearsal_result["lstm_magnitude_diagnostic"] = diagnostic
        if not diagnostic["passes"]:
            for contract in rehearsal_result["contracts"].values():
                contract["methods"]["lstm"]["passes_gate"] = False
            if rehearsal_result["selected_critical_method"] == "lstm":
                rehearsal_result["selected_critical_method"] = None
        rehearsal_result["holdout_status"] = "sealed_pending_separate_approval"
        write_json(self.out / "rehearsal_result.json", rehearsal_result)
        if stage == "rehearse":
            return rehearsal_result
        holdout = role_frames["holdout"]
        holdout_result = self._evaluate_role(
            holdout.loc[holdout["class"] == "normal"].reset_index(drop=True),
            holdout.loc[holdout["class"] == "gps_fault"].reset_index(drop=True),
            fitted,
            calibrated,
            "blind_holdout",
        )
        holdout_result["lstm_magnitude_diagnostic"] = diagnostic
        if not diagnostic["passes"]:
            for contract in holdout_result["contracts"].values():
                contract["methods"]["lstm"]["passes_gate"] = False
        holdout_result["final_status"] = (
            "GO_narrow_GNSS_integrity_candidate"
            if holdout_result["selected_critical_method"]
            else "LIMITED_advisory_research_demonstrator"
            if any(
                method["passes_gate"]
                for method in holdout_result["contracts"]["advisory"]["methods"].values()
            )
            else "NO_GO_not_achievable_with_current_data_and_instrumentation"
        )
        write_json(self.out / "holdout_result.json", holdout_result)
        return holdout_result

    def _assert_unsealed(self) -> None:
        path = self.out / "HOLDOUT_UNSEAL.json"
        if not path.exists():
            raise PermissionError(
                f"holdout is sealed; create {path} only after separate approval"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "candidate_namespace": NAMESPACE,
            "config_sha256": self.config_sha256,
            "approval": "UNSEAL HOLDOUT",
        }
        if payload != expected:
            raise PermissionError("holdout unseal record does not match frozen config")

    def report(self) -> Path:
        rehearsal_path = self.out / "rehearsal_result.json"
        if not rehearsal_path.exists():
            raise FileNotFoundError("run --stage rehearse before --stage report")
        rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
        development = json.loads(
            (self.out / "development_result.json").read_text(encoding="utf-8")
        )
        tex_path = self.out / "uav_gnss_integrity_v1_rehearsal_report.tex"
        tex_path.write_text(
            render_latex_report(
                rehearsal, development, config_sha256=self.config_sha256
            ),
            encoding="utf-8",
        )
        return tex_path


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def _num(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def render_latex_report(
    rehearsal: dict[str, Any],
    development: dict[str, Any],
    *,
    config_sha256: str,
) -> str:
    rows = []
    for contract_name in ("critical", "advisory"):
        for method, metrics in rehearsal["contracts"][contract_name]["methods"].items():
            rows.append(
                " & ".join(
                    [
                        contract_name,
                        method.replace("_", "\\_"),
                        _pct(metrics["events"]["recall"]),
                        _pct(metrics["events"]["wilson_95"]["lower"]),
                        _num(metrics["burden"]["episodes_per_scoreable_flight_hour"]),
                        "PASS" if metrics["passes_gate"] else "FAIL",
                    ]
                )
                + r" \\"
            )
    diagnostic = development["lstm_magnitude_diagnostic"]
    status = rehearsal["preliminary_status"].replace("_", r"\_")
    rows_text = "\n".join(rows)
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[turkish]{{babel}}
\usepackage{{geometry,booktabs,xcolor,hyperref}}
\geometry{{margin=2.2cm}}
\title{{UAV GNSS Bütünlük Fizibilitesi v1\\Rehearsal Sonuç Raporu}}
\author{{AnomalyDetection Projesi}}
\date{{}}
\begin{{document}}
\maketitle
\section{{Karar Durumu}}
Bu belge kör holdout sonucu değildir. Holdout ayrı onay verilene kadar mühürlüdür.
Ürün için nihai GO kararı verilemez. Rehearsal ön sonucu: \texttt{{{status}}}.

\section{{Dondurulmuş Sözleşme}}
Tek ürün iddiası, PX4 üzerinde mevcut telemetriyle GNSS noise ve scale-factor
arızalarının kabul edilebilir gecikme ve operatör yüküyle tespitidir. Motor sağlığı
bu turda kapsam dışıdır; mevcut gerçek loglarda ESC RPM/akım/sıcaklık değerleri
ölçülmemiştir. Config SHA-256: \texttt{{{config_sha256}}}.

\section{{Rehearsal Sonuçları}}
\begin{{tabular}}{{llrrrr}}
\toprule
Sözleşme & Yöntem & Deadline recall & Wilson alt & Alarm/saat & Kapı \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}

\section{{LSTM Geçerlilik Tanısı}}
Eğitilmiş skor ile random-init skor Spearman korelasyonu:
\textbf{{{_num(diagnostic["trained_vs_random_spearman"])}}}.
Eğitilmiş skor ile ham büyüklük korelasyonu:
\textbf{{{_num(diagnostic["trained_vs_magnitude_spearman"])}}}.
Ön-kayıt kapısı 0.80'dir. Sonuç:
\textbf{{{"PASS" if diagnostic["passes"] else "FAIL"}}}.
FAIL durumunda LSTM recall değeri ürün kanıtı olarak kullanılamaz.

\section{{Yorumlama Sınırları}}
RflyMAD gerçek GPS-fault havuzu küçüktür. SIL/HIL rüzgâr verileri GPS
ground-truth'u değildir ve yalnız çevresel yanlış-alarm stres testi olabilir.
Sentetik veya simülasyon recall'ı ürün kanıtı sayılmaz. \texttt{{not\_evaluable}}
süreleri normal veya anomaly sınıfına zorlanmamıştır.

\section{{Sonraki Kapı}}
Holdout yalnız \texttt{{HOLDOUT\_UNSEAL.json}} içinde frozen config hash'ine bağlı
ayrı onay kaydı oluşturulduğunda açılabilir. Sonuç görüldükten sonra eşik, özellik,
pencere veya model değiştirmek yeni prereg gerektirir.
\end{{document}}
"""
