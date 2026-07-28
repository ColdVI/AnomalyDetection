"""Audit RflyMAD v3.1 normal training quality and coverage before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/four_dataset_probabilistic_v31_split_manifest.json"
MODEL_CONFIG = ROOT / "configs/four_dataset_probabilistic_gpu_v2.json"
RFLY_MANIFEST = ROOT / "artifacts/rfly_full/v2/dataset_manifest.parquet"
DATA_ROOT = ROOT / "artifacts/rfly_full/v2/parsed_10hz"
OUTPUT = ROOT / "artifacts/four_dataset_probabilistic_v31/rflymad_normal_train_audit.json"


class NormalTrainAuditError(RuntimeError):
    """Raised for a hard identity, label, or frozen-input contract failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _continuous_window_count(times: np.ndarray, history: int, max_gap_s: float) -> int:
    finite = np.isfinite(times)
    breaks = np.ones(len(times), dtype=bool)
    if len(times) > 1:
        delta = np.diff(times)
        breaks[1:] = (~finite[1:]) | (~finite[:-1]) | (delta <= 0) | (delta > max_gap_s)
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(times)]
    return int(sum(max(0, int(end - start) - history) for start, end in zip(starts, ends)))


def audit(
    protocol_path: Path,
    config_path: Path,
    manifest_path: Path,
    data_root: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["inputs"]["group_registry_sha256"] != _sha256(
        ROOT / protocol["inputs"]["group_registry_path"]
    ):
        raise NormalTrainAuditError("v3.1 registry hash mismatch")
    train_ids = list(protocol["datasets"]["rflymad"]["source_ids"]["train"])
    features = list(config["datasets"]["rflymad"]["features"])
    history = int(config["common"]["history_rows"])
    max_gap_s = float(config["common"]["max_gap_s"])
    manifest = pd.read_parquet(manifest_path)
    manifest["canonical_case_id"] = manifest["canonical_case_id"].astype(str)
    selected = manifest.loc[manifest["canonical_case_id"].isin(train_ids)].copy()
    if len(selected) != len(train_ids) or selected["canonical_case_id"].duplicated().any():
        raise NormalTrainAuditError("Train manifest coverage/uniqueness failure")
    if set(selected["fault_family"].astype(str)) != {"NoFault"}:
        raise NormalTrainAuditError("Fault-labelled source entered normal train")
    if not selected["quality_status"].astype(str).eq("ok").all():
        raise NormalTrainAuditError("Non-ok parser quality_status entered normal train")
    for column in ("fault_start_s", "fault_end_s"):
        if column in selected and selected[column].notna().any():
            raise NormalTrainAuditError(f"Normal train contains a fault interval: {column}")
    planned_interval_placeholders = int(
        (
            selected["planned_fault_start_s"].notna()
            | selected["planned_fault_end_s"].notna()
        ).sum()
    )
    nonempty_fault_ids = selected["fault_id"].map(
        lambda value: str(value).strip() not in {"", "[]", "nan", "None"}
    )
    if nonempty_fault_ids.any():
        raise NormalTrainAuditError("Normal train contains a non-empty fault_id")

    source_rows: list[dict[str, Any]] = []
    feature_finite = Counter()
    feature_total = Counter()
    for row in selected.sort_values("canonical_case_id").itertuples(index=False):
        source_id = str(row.canonical_case_id)
        path = data_root / str(row.domain) / f"{source_id}.parquet"
        if not path.is_file():
            raise NormalTrainAuditError(f"Parsed source missing: {path}")
        requested_columns = list(
            dict.fromkeys(
                [
                "t_rel_s",
                "canonical_case_id",
                "fault_family",
                "fault_active",
                "condition_active",
                "local_vx",
                "local_vy",
                "local_vz",
                *features,
                ]
            )
        )
        frame = pd.read_parquet(path, columns=requested_columns)
        if set(frame["canonical_case_id"].dropna().astype(str)) != {source_id}:
            raise NormalTrainAuditError(f"Parsed source identity mismatch: {source_id}")
        if set(frame["fault_family"].dropna().astype(str)) != {"NoFault"}:
            raise NormalTrainAuditError(f"Parsed fault label in normal train: {source_id}")
        if frame["fault_active"].fillna(False).astype(bool).any() or frame[
            "condition_active"
        ].fillna(False).astype(bool).any():
            raise NormalTrainAuditError(f"Active fault/condition in normal train: {source_id}")
        numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(float))
        for index, feature in enumerate(features):
            feature_finite[feature] += int(finite[:, index].sum())
            feature_total[feature] += int(len(frame))
        times = pd.to_numeric(frame["t_rel_s"], errors="coerce").to_numpy(float)
        velocity = frame[["local_vx", "local_vy", "local_vz"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        horizontal = np.sqrt(np.square(velocity[:, 0]) + np.square(velocity[:, 1]))
        vertical = velocity[:, 2]
        source_rows.append(
            {
                "source_id": source_id,
                "group_id": str(row.split_group_id),
                "domain": str(row.domain),
                "rows": int(len(frame)),
                "duration_s": float(np.nanmax(times) - np.nanmin(times)),
                "candidate_windows": _continuous_window_count(times, history, max_gap_s),
                "all_feature_missing_fraction": float((~finite).all(axis=1).mean()),
                "low_motion_fraction": float(
                    np.nanmean((horizontal < 0.25) & (np.abs(vertical) < 0.25))
                ),
                "translation_fraction": float(np.nanmean(horizontal > 2.0)),
                "climb_proxy_fraction": float(np.nanmean(vertical < -0.5)),
                "descent_proxy_fraction": float(np.nanmean(vertical > 0.5)),
            }
        )
    source_frame = pd.DataFrame(source_rows)
    windows = source_frame["candidate_windows"].to_numpy(int)
    if (windows <= 0).any():
        bad = source_frame.loc[source_frame["candidate_windows"].le(0), "source_id"].tolist()
        raise NormalTrainAuditError(f"Train sources without causal windows: {bad[:5]}")
    group_windows = source_frame.groupby("group_id")["candidate_windows"].sum()
    total_windows = int(windows.sum())
    return {
        "schema_version": 1,
        "candidate_namespace": "four_dataset_probabilistic_v31",
        "artifact_role": "normal-train quality and coverage preflight; no test data read",
        "created_date": "2026-07-28",
        "contract_passed": True,
        "training_started": False,
        "test_data_read": False,
        "inputs": {
            "protocol_path": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "model_config_path": config_path.relative_to(ROOT).as_posix(),
            "model_config_sha256": _sha256(config_path),
            "rfly_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "rfly_manifest_sha256": _sha256(manifest_path),
        },
        "hard_checks": {
            "source_identity": "pass",
            "normal_label_only": "pass",
            "realized_fault_ranges_absent": "pass",
            "nonempty_fault_ids_absent": "pass",
            "fault_and_condition_flags_inactive": "pass",
            "quality_status_ok": "pass",
            "causal_windows_nonempty": "pass",
        },
        "metadata_findings": {
            "planned_interval_placeholder_source_count": planned_interval_placeholders,
            "interpretation": (
                "NoFault simulation TestInfo carries 0-to-duration planned interval "
                "placeholders despite empty fault_id; realized fault_start/fault_end, "
                "fault_active and condition_active are absent/inactive. Preserved as "
                "a documented schema quirk, not treated as a realized fault range."
            ),
        },
        "coverage": {
            "source_count": int(len(source_frame)),
            "group_count": int(source_frame["group_id"].nunique()),
            "domain_source_counts": dict(sorted(Counter(source_frame["domain"]).items())),
            "rows": int(source_frame["rows"].sum()),
            "duration_hours_sum": float(source_frame["duration_s"].sum() / 3600),
            "candidate_windows": total_windows,
            "candidate_windows_per_source": {
                "min": int(np.min(windows)),
                "median": float(np.median(windows)),
                "p95": float(np.quantile(windows, 0.95)),
                "max": int(np.max(windows)),
                "largest_source_share": float(np.max(windows) / total_windows),
            },
            "candidate_windows_per_group": {
                "min": int(group_windows.min()),
                "median": float(group_windows.median()),
                "max": int(group_windows.max()),
                "largest_group_share": float(group_windows.max() / total_windows),
            },
            "feature_finite_fraction": {
                feature: float(feature_finite[feature] / feature_total[feature])
                for feature in features
            },
            "kinematic_proxy_note": (
                "Coverage proxies are descriptive only; they are not the frozen flight-phase classifier."
            ),
            "kinematic_proxy_source_medians": {
                column: float(source_frame[column].median())
                for column in (
                    "low_motion_fraction",
                    "translation_fraction",
                    "climb_proxy_fraction",
                    "descent_proxy_fraction",
                )
            },
        },
        "optimizer_weighting_contract": {
            "rule": "equal epoch quota across groups, then equal quota across sources within each group",
            "purpose": "prevent long flights and large scenario groups from dominating",
            "derived_from_test": False,
            "must_be_implemented_by_v31_runner": True,
        },
        "preprocessing_contract": {
            "fit_roles": ["train"],
            "includes": [
                "imputation constants",
                "scaler",
                "feature availability/degeneracy selection",
                "phase normalization",
                "residual coefficients",
            ],
            "validation_or_test_distribution_access": False,
        },
        "sources": source_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--rfly-manifest", type=Path, default=RFLY_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    report = audit(
        args.protocol.resolve(),
        args.model_config.resolve(),
        args.rfly_manifest.resolve(),
        args.data_root.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"contract_passed": True, **report["coverage"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
