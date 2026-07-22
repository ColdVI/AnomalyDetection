"""Deterministic session-level RESIDUAL-V1 split manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.ingest.profile import find_flight_roots
from gecmis_calismalar.residual_v1.run import sha256_file

DEFAULT_SEEDS = (11, 23, 37, 41, 53)
PARTITIONS = ("development", "test", "holdout")


def load_flight_metadata(silver_root: str | Path) -> list[dict]:
    quarantine_path = Path(silver_root) / "quarantine.json"
    quarantined = (
        set(json.loads(quarantine_path.read_text(encoding="utf-8")))
        if quarantine_path.exists()
        else set()
    )
    rows = []
    for root in find_flight_roots(silver_root):
        row = json.loads((root / "flight.json").read_text(encoding="utf-8"))
        if row["flight_id"] not in quarantined:
            rows.append(row)
    return rows


def _target_partition_counts(session_count: int) -> dict[str, int]:
    raw = {"development": 0.70 * session_count, "test": 0.15 * session_count, "holdout": 0.15 * session_count}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    for partition in ("test", "holdout"):
        if session_count >= 3 and counts[partition] == 0:
            counts[partition] = 1
    while sum(counts.values()) > session_count:
        counts["development"] -= 1
    remainder_order = sorted(PARTITIONS, key=lambda name: raw[name] - np.floor(raw[name]), reverse=True)
    index = 0
    while sum(counts.values()) < session_count:
        counts[remainder_order[index % len(remainder_order)]] += 1
        index += 1
    return counts


def _class_counts(rows: Iterable[Mapping[str, object]]) -> Counter:
    return Counter(
        str(row["fault_class"])
        for row in rows
        if str(row.get("fault_class", "normal")) not in {"normal", "unknown"}
    )


def _valid_assignment(
    assignment: Mapping[str, str],
    session_classes: Mapping[str, set[str]],
    rare_classes: set[str],
    headline_classes: set[str],
) -> bool:
    for session, classes in session_classes.items():
        if classes & rare_classes and assignment.get(session) != "development":
            return False
    for fault_class in headline_classes:
        for partition in PARTITIONS:
            if not any(
                assignment.get(session) == partition and fault_class in classes
                for session, classes in session_classes.items()
            ):
                return False
    return True


def split_flights(
    flights: Iterable[Mapping[str, object]],
    *,
    seed: int,
    rare_class_max_events: int = 7,
    attempts: int = 20_000,
) -> dict:
    rows = [dict(row) for row in flights]
    if not rows:
        raise ValueError("at least one flight is required")
    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sessions[str(row["session"])].append(row)
    if len(sessions) < 3:
        raise ValueError("session-level 70/15/15 split requires at least three sessions")

    counts = _class_counts(rows)
    rare_classes = {name for name, count in counts.items() if count <= rare_class_max_events}
    headline_classes = set(counts) - rare_classes
    session_classes = {
        session: {
            str(row["fault_class"])
            for row in group
            if str(row.get("fault_class", "normal")) not in {"normal", "unknown"}
        }
        for session, group in sessions.items()
    }
    for fault_class in headline_classes:
        supporting = [session for session, classes in session_classes.items() if fault_class in classes]
        if len(supporting) < 3:
            raise ValueError(
                f"headline class {fault_class!r} spans {len(supporting)} sessions; three are required"
            )

    forced_development = {
        session for session, classes in session_classes.items() if classes & rare_classes
    }
    targets = _target_partition_counts(len(sessions))
    targets["development"] = max(targets["development"], len(forced_development))
    overflow = sum(targets.values()) - len(sessions)
    for partition in ("test", "holdout"):
        reduction = min(overflow, max(0, targets[partition] - 1))
        targets[partition] -= reduction
        overflow -= reduction
    if overflow:
        raise ValueError("rare-class development constraint leaves no test/holdout session")

    rng = np.random.default_rng(seed)
    unforced = sorted(set(sessions) - forced_development)
    assignment: dict[str, str] | None = None
    slots = [
        partition
        for partition in PARTITIONS
        for _ in range(targets[partition] - (len(forced_development) if partition == "development" else 0))
    ]
    if len(slots) != len(unforced):
        raise RuntimeError("internal split slot mismatch")
    for _ in range(attempts):
        shuffled_sessions = list(rng.permutation(unforced))
        shuffled_slots = list(rng.permutation(slots))
        candidate = {session: "development" for session in forced_development}
        candidate.update(dict(zip(shuffled_sessions, shuffled_slots, strict=True)))
        if _valid_assignment(candidate, session_classes, rare_classes, headline_classes):
            assignment = candidate
            break
    if assignment is None:
        raise ValueError(
            "no session assignment satisfies class coverage and rare-class constraints"
        )

    partitions: dict[str, dict] = {}
    for partition in PARTITIONS:
        selected_sessions = sorted(session for session, value in assignment.items() if value == partition)
        selected_rows = [row for session in selected_sessions for row in sessions[session]]
        partitions[partition] = {
            "sessions": selected_sessions,
            "flight_ids": sorted(str(row["flight_id"]) for row in selected_rows),
            "class_counts": dict(sorted(_class_counts(selected_rows).items())),
        }
    return {
        "seed": int(seed),
        "ratios": {"development": 0.70, "test": 0.15, "holdout": 0.15},
        "rare_class_max_events": rare_class_max_events,
        "rare_classes_development_only": sorted(rare_classes),
        "headline_classes": sorted(headline_classes),
        "partitions": partitions,
    }


def write_split_manifests(
    dataset: str,
    flights: Iterable[Mapping[str, object]],
    output_root: str | Path = "artifacts/residual_v1/splits",
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> dict[str, str]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    rows = list(flights)
    for seed in seeds:
        path = output / f"{dataset}_seed{seed}.json"
        manifest = {"dataset": dataset, **split_flights(rows, seed=seed)}
        write_json(path, manifest, fail_if_exists=True)
        hashes[path.name] = sha256_file(path)
    write_json(output / f"{dataset}_hashes.json", hashes, fail_if_exists=True)
    return hashes

