"""Governance-hardened runner built on the v1 implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gecmis_calismalar.uav_gnss.evaluation import natural_burden
from gecmis_calismalar.uav_gnss.features import load_role_features
from gecmis_calismalar.uav_gnss.pipeline import PilotRunner, write_json


def require_prior_role(current: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    """Require both development and current-role gates before candidate selection."""

    for contract_name in ("critical", "advisory"):
        for method in ("px4_native", "cusum", "lstm"):
            metrics = current["contracts"][contract_name]["methods"][method]
            current_pass = bool(metrics["passes_gate"])
            prior_pass = bool(
                prior["contracts"][contract_name]["methods"][method]["passes_gate"]
            )
            metrics["current_role_passes_gate"] = current_pass
            metrics["prior_role_passes_gate"] = prior_pass
            metrics["passes_gate"] = current_pass and prior_pass
    critical = current["contracts"]["critical"]["methods"]
    advisory = current["contracts"]["advisory"]["methods"]
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
    current["selected_critical_method"] = selected
    current["preliminary_status"] = (
        "critical_candidate"
        if selected
        else "advisory_candidate"
        if any(value["passes_gate"] for value in advisory.values())
        else "no_go_on_current_role"
    )
    return current


def simulation_wind_cases(bronze_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"SIL-Wind": [], "HIL-Wind": []}
    for domain in result:
        for path in sorted((bronze_root / domain).rglob("*.ulg")):
            result[domain].append(
                {
                    "flight_id": path.relative_to(bronze_root).as_posix(),
                    "path": path.as_posix(),
                    "domain": domain,
                    "class": "wind_stress",
                    "flight_mode": next(
                        (
                            mode
                            for mode in ("acce", "circling", "hover", "velocity", "waypoint", "dece")
                            if mode in path.as_posix().lower()
                        ),
                        "unknown",
                    ),
                    "role": "stress",
                    "fault_id": None,
                    "fault_mode": None,
                    "fault_mode_name": None,
                    "fault_onset_s": None,
                    "fault_end_s": None,
                }
            )
    return result


class FrozenPilotRunner(PilotRunner):
    """Adds prior-role gating and a frozen environmental burden stress stage."""

    def run_through(self, stage: str) -> dict[str, Any]:
        if stage == "stress":
            return self.stress()
        result = super().run_through(stage)
        if stage == "rehearse":
            development = json.loads(
                (self.out / "development_result.json").read_text(encoding="utf-8")
            )
            result = require_prior_role(result, development)
            result["holdout_status"] = "sealed_pending_separate_approval"
            write_json(self.out / "rehearsal_result.json", result)
        elif stage == "holdout":
            rehearsal = json.loads(
                (self.out / "rehearsal_result.json").read_text(encoding="utf-8")
            )
            result = require_prior_role(result, rehearsal)
            result["final_status"] = (
                "GO_narrow_GNSS_integrity_candidate"
                if result["selected_critical_method"]
                else "LIMITED_advisory_research_demonstrator"
                if any(
                    method["passes_gate"]
                    for method in result["contracts"]["advisory"]["methods"].values()
                )
                else "NO_GO_not_achievable_with_current_data_and_instrumentation"
            )
            write_json(self.out / "holdout_result.json", result)
        return result

    def stress(self) -> dict[str, Any]:
        self.preflight()
        role_frames = self._load_roles(("fit", "calibration"))
        fitted = self._fit_models(role_frames["fit"])
        calibrated = self._calibrate(role_frames["calibration"], fitted)
        cases = simulation_wind_cases(self.root)
        report: dict[str, Any] = {
            "role": "environmental_false_alarm_stress_only",
            "is_gnss_ground_truth": False,
            "domains": {},
        }
        max_gap = float(self.config["features"]["max_gap_s"])
        for domain, domain_cases in cases.items():
            frame = load_role_features(domain_cases, max_gap_s=max_gap).reset_index(drop=True)
            domain_report: dict[str, Any] = {
                "n_cases": len(domain_cases),
                "contracts": {},
            }
            for contract_name, contract in self.config["contracts"].items():
                methods = self._alarms(frame, fitted, calibrated, contract_name)
                method_report = {}
                for method, (method_frame, alarms) in methods.items():
                    burden = natural_burden(
                        method_frame,
                        alarms,
                        merge_gap_s=float(self.config["episodes"]["merge_gap_s"]),
                    )
                    burden["warning_above_twice_real_budget"] = bool(
                        burden["episodes_per_scoreable_flight_hour"] is not None
                        and burden["episodes_per_scoreable_flight_hour"]
                        > 2 * float(contract["max_false_alarms_per_hour"])
                    )
                    method_report[method] = burden
                domain_report["contracts"][contract_name] = {"methods": method_report}
            report["domains"][domain] = domain_report
        write_json(self.out / "wind_stress_result.json", report)
        return report

