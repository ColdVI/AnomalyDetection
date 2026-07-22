"""Watch parsed RflyMAD batches, train one frozen Dense AE, then evaluate them."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from gecmis_calismalar.rfly_dl.models import DenseAutoencoder, reconstruction_scores, train_model
from gecmis_calismalar.rfly_full.pipeline import ARTIFACT_ROOT, FEATURES, PARSED_ROOT, _atomic_json, _k_of_n

EXPANSION_STATE = ARTIFACT_ROOT / "expansion_state.json"
DL_ROOT = ARTIFACT_ROOT / "dl"
MODEL_PATH = DL_ROOT / "model.pt"
TRAINING_REPORT = DL_ROOT / "training_report.json"
WORKER_STATE = DL_ROOT / "worker_state.json"
BATCH_ROOT = DL_ROOT / "batches"
SUMMARY_PATH = DL_ROOT / "summary.json"
WINDOW = 20
TRAIN_STRIDE = 5
EVAL_STRIDE = 1
SEED = 20260720
LOG = logging.getLogger("rfly_dl_worker")


def split_of(source_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(source_id.encode("utf-8")).digest()[:2], "big") % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def _normal_ready() -> bool:
    if not EXPANSION_STATE.exists():
        return False
    state = json.loads(EXPANSION_STATE.read_text(encoding="utf-8"))
    completed = set(state.get("completed_batches", []))
    expected: dict[str, int] = {}
    for package in ("SIL-NoFault", "HIL-NoFault"):
        listing = ARTIFACT_ROOT / "expanded_sources" / ("rflymad-sil" if package.startswith("SIL") else "rflymad-hil") / "listing.csv"
        if not listing.exists():
            return False
        rows = pd.read_csv(listing, usecols=["name"])
        flights = int(rows["name"].str.startswith(package + "/").where(rows["name"].str.lower().str.endswith(".ulg"), False).sum())
        expected[package] = math.ceil(flights / 16)
    return all(sum(key.startswith(package + "/") for key in completed) >= count for package, count in expected.items())


def _normal_frames() -> pd.DataFrame:
    paths = []
    for package in ("Real-No_Fault", "SIL-NoFault", "HIL-NoFault"):
        paths.extend(sorted((PARSED_ROOT / package).glob("*.parquet")))
    if not paths:
        raise RuntimeError("no parsed normal flights")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _windows(frame: pd.DataFrame, columns: list[str], scaler: dict, stride: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows, masks, metadata = [], [], []
    medians = pd.Series(scaler["medians"])
    scales = pd.Series(scaler["scales"])
    for source_id, flight in frame.groupby("case_id", sort=False):
        flight = flight.sort_values("t_rel_s").reset_index(drop=True)
        observed = flight[columns].notna().to_numpy(dtype=np.float32)
        values = ((flight[columns].fillna(medians) - medians) / scales).clip(-10, 10).to_numpy(dtype=np.float32)
        for end in range(WINDOW - 1, len(flight), stride):
            start = end - WINDOW + 1
            windows.append(values[start : end + 1])
            masks.append(observed[start : end + 1])
            metadata.append({
                "case_id": str(source_id), "t_end": float(flight.loc[end, "t_rel_s"]),
                "label": str(flight.loc[end, "label"]),
                "fault_active": bool(flight.loc[end, "fault_active"]),
            })
    if not windows:
        empty_meta = pd.DataFrame(
            metadata, columns=["case_id", "t_end", "label", "fault_active"]
        )
        return np.empty((0, WINDOW, len(columns)), np.float32), np.empty((0, WINDOW, len(columns)), np.float32), empty_meta
    return np.stack(windows), np.stack(masks), pd.DataFrame(metadata)


def _cap(x: np.ndarray, mask: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= limit:
        return x, mask
    index = np.random.default_rng(seed).choice(len(x), limit, replace=False)
    return x[index], mask[index]


def train_dense_ae() -> dict:
    normal = _normal_frames()
    ids = {part: {source_id for source_id in normal.case_id.unique() if split_of(str(source_id)) == part} for part in ("train", "val", "test")}
    train = normal[normal.case_id.isin(ids["train"])]
    val = normal[normal.case_id.isin(ids["val"])]
    coverage = train[list(FEATURES)].notna().mean()
    columns = [column for column in FEATURES if coverage[column] >= 0.70]
    if len(columns) < 6:
        raise RuntimeError(f"only {len(columns)} DL features meet 70% completeness")
    medians = train[columns].median()
    scales = (train[columns].quantile(0.75) - train[columns].quantile(0.25)).replace(0, 1).fillna(1)
    scaler = {"medians": medians.to_dict(), "scales": scales.to_dict()}
    x_train, m_train, _ = _windows(train, columns, scaler, TRAIN_STRIDE)
    x_val, m_val, _ = _windows(val, columns, scaler, TRAIN_STRIDE)
    x_train, m_train = _cap(x_train, m_train, 16000, SEED)
    x_val, m_val = _cap(x_val, m_val, 5000, SEED + 1)
    model = DenseAutoencoder(WINDOW, len(columns), hidden=64, latent=16)
    result = train_model(
        "dense_ae", model, x_train, m_train, x_val, m_val,
        seed=SEED, max_epochs=12, patience=3, batch_size=128,
        learning_rate=1e-3, device="cpu",
    )
    val_scores = reconstruction_scores("dense_ae", result.model, x_val, m_val)
    threshold = float(np.quantile(val_scores, 0.995))
    DL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result.model.state_dict(), "columns": columns, "scaler": scaler,
                "threshold": threshold, "window": WINDOW, "hidden": 64, "latent": 16}, MODEL_PATH)
    report = {
        "status": "trained", "trained_at": datetime.now().astimezone().isoformat(),
        "model": "dense_autoencoder", "window_seconds": WINDOW,
        "train_stride_seconds": TRAIN_STRIDE, "evaluation_stride_seconds": EVAL_STRIDE,
        "split": "sha256(case_id): 70/15/15", "normal_flights": {key: len(value) for key, value in ids.items()},
        "features": columns, "train_windows": len(x_train), "validation_windows": len(x_val),
        "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
        "parameters": result.parameter_count, "threshold": threshold,
        "threshold_source": "normal_validation_reconstruction_q99.5",
        "history": result.history,
    }
    _atomic_json(TRAINING_REPORT, report)
    return report


def _load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = DenseAutoencoder(checkpoint["window"], len(checkpoint["columns"]), checkpoint["hidden"], checkpoint["latent"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def evaluate_batch(path: Path, model, checkpoint: dict) -> dict:
    frame = pd.read_parquet(path)
    if frame.label.eq("normal").all():
        frame = frame[frame.case_id.map(lambda value: split_of(str(value)) == "test")]
    x, mask, meta = _windows(frame, checkpoint["columns"], checkpoint["scaler"], EVAL_STRIDE)
    scores = reconstruction_scores("dense_ae", model, x, mask)
    meta["score"] = scores
    threshold = float(checkpoint["threshold"])
    tp = fn = fp = tn = false_events = 0
    exposure_hours = 0.0
    flights = []
    for source_id, group in meta.groupby("case_id", sort=False):
        alarm = _k_of_n(group.score.to_numpy() > threshold)
        truth = group.fault_active.to_numpy(dtype=bool)
        hit = bool((alarm & truth).any())
        fault_flight = group.label.iloc[0] != "normal"
        false_count = int((alarm & ~truth).sum())
        false_events += false_count
        exposure_hours += float((~truth).sum() / 3600)
        if fault_flight:
            tp += int(hit); fn += int(not hit)
        else:
            predicted = bool(alarm.any()); fp += int(predicted); tn += int(not predicted)
        flights.append({"case_id": source_id, "label": group.label.iloc[0], "detected": hit, "false_alarm_events": false_count})
    truth = meta.fault_active.to_numpy(dtype=bool) if len(meta) else np.empty(0, bool)
    return {
        "status": "ok", "model": "frozen_dense_autoencoder", "batch": path.as_posix(),
        "evaluated_flights": int(meta.case_id.nunique()) if len(meta) else 0,
        "windows": len(meta), "decision": "4_of_6_at_1hz",
        "window_auroc": float(roc_auc_score(truth, scores)) if truth.any() and (~truth).any() else None,
        "window_auprc": float(average_precision_score(truth, scores)) if truth.any() and (~truth).any() else None,
        "event_recall": tp / (tp + fn) if tp + fn else None,
        "false_alarm_events": false_events, "normal_exposure_hours": exposure_hours,
        "false_alarms_per_hour": false_events / exposure_hours if exposure_hours else None,
        "confusion_flight": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "flights": flights,
    }


def write_summary() -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in BATCH_ROOT.glob("**/*.json")]
    confusion = {key: int(sum(report["confusion_flight"][key] for report in reports)) for key in ("tp", "fn", "fp", "tn")}
    false_events = int(sum(report["false_alarm_events"] for report in reports))
    hours = float(sum(report["normal_exposure_hours"] for report in reports))
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(), "model": "frozen_dense_autoencoder",
        "evaluated_batches": len(reports), "evaluated_flights": int(sum(report["evaluated_flights"] for report in reports)),
        "confusion_flight": confusion,
        "event_recall": confusion["tp"] / (confusion["tp"] + confusion["fn"]) if confusion["tp"] + confusion["fn"] else None,
        "false_alarm_events": false_events, "normal_exposure_hours": hours,
        "false_alarms_per_hour": false_events / hours if hours else None,
    }
    _atomic_json(SUMMARY_PATH, summary)
    return summary


def run() -> None:
    while not MODEL_PATH.exists():
        if _normal_ready():
            LOG.info("normal SIL/HIL pool complete; training frozen Dense AE")
            train_dense_ae()
            break
        time.sleep(30)
    model, checkpoint = _load_model()
    state = json.loads(WORKER_STATE.read_text(encoding="utf-8")) if WORKER_STATE.exists() else {"completed": []}
    while True:
        for path in sorted(PARSED_ROOT.glob("*/*.parquet")):
            relative = path.relative_to(PARSED_ROOT).as_posix()
            if relative in state["completed"] or path.parent.name == "bootstrap_normal":
                continue
            try:
                report = evaluate_batch(path, model, checkpoint)
            except Exception as exc:
                # The producer may still be atomically completing a new parquet.
                # Leave it uncheckpointed and retry on the next polling cycle.
                state["last_error"] = {"batch": relative, "error": str(exc)}
                _atomic_json(WORKER_STATE, state)
                LOG.exception("DL batch deferred %s", relative)
                continue
            target = BATCH_ROOT / path.parent.name / f"{path.stem}.json"
            _atomic_json(target, report)
            state["completed"].append(relative)
            state.pop("last_error", None)
            state["updated_at"] = datetime.now().astimezone().isoformat()
            _atomic_json(WORKER_STATE, state)
            LOG.info("DL evaluated %s flights=%d", relative, report["evaluated_flights"])
        expansion = json.loads(EXPANSION_STATE.read_text(encoding="utf-8")) if EXPANSION_STATE.exists() else {}
        if expansion.get("stop_reason"):
            write_summary()
            return
        time.sleep(30)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
