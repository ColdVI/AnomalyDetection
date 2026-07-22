"""Aşama A truth audit for the RflyMAD-Full v2 10 Hz parse.

`dataset_manifest.parquet` (`rfly_full.contract`) still computes its
`truth_source` / `fault_start_s` / `fault_end_s` / `quality_status` columns
from the historical 1 Hz `parsed_batches` output (`rfly_full.pipeline`),
whose truth priority is TestInfo-first with `rfly_ctrl_lxl` only as a
fallback. The v2 10 Hz parser (`rfly_full.v2_parser`) inverts that priority
(`rfly_ctrl_lxl` first, TestInfo fallback) and stores the resulting
`truth_source` / `fault_active` / `truth_crosscheck_disagreement` directly
on every parsed flight. This module recomputes the truth-quality picture
from that v2 per-flight data instead of trusting the stale manifest fields.

Only `split_group_id` / `split` are borrowed from the manifest, because
those are derived purely from taxonomy/session logic and do not depend on
which parser produced the feature data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.parquet import read_schema

from rfly_full.contract import DATASET_MANIFEST, V2_ROOT
from rfly_full.pipeline import _atomic_json
from rfly_full.v2_parser import (
    CROSSCHECK_COLUMNS,
    CROSSCHECK_ONSET_TOLERANCE_S,
    PARSE_STATE,
    PARSED_10HZ_ROOT,
    V2_FEATURES,
)

AUDIT_ROOT = V2_ROOT / "truth_audit"
DEVELOPMENT_AUDIT_ROOT = V2_ROOT / "truth_audit_development"
PER_FLIGHT_CSV = AUDIT_ROOT / "truth_audit_per_flight.csv"
DOMAIN_FAMILY_CSV = AUDIT_ROOT / "truth_source_by_domain_family.csv"
NEAR_DUPLICATE_CSV = AUDIT_ROOT / "near_duplicate_clusters.csv"
SUMMARY_MD = AUDIT_ROOT / "truth_audit_summary.md"
SUMMARY_JSON = AUDIT_ROOT / "truth_audit_summary.json"

READ_COLUMNS = [
    "t_rel_s", "canonical_case_id", "object_name", "package", "domain",
    "fault_family", "fault_subtype", "system_fault", "environment_condition",
    "fault_active", "environment_active", "truth_source",
    "truth_crosscheck_disagreement", *CROSSCHECK_COLUMNS,
    "local_x", "local_y", "local_z",
]


def _first_value(frame: pd.DataFrame, column: str, default):
    if column not in frame or frame.empty:
        return default
    return frame[column].iloc[0]


def _audit_flight_frame(frame: pd.DataFrame, *, missing_features: list[str]) -> dict:
    """Pure aggregation over one already-loaded flight frame. Testable without I/O."""
    t = frame["t_rel_s"].to_numpy(dtype=float)
    active = frame["fault_active"].to_numpy(dtype=bool)
    duration_s = float(t.max() - t.min()) if len(t) else float("nan")
    if active.any():
        fault_start_s = float(t[active].min())
        fault_end_s = float(t[active].max())
    else:
        fault_start_s = float("nan")
        fault_end_s = float("nan")
    system_fault = bool(frame["system_fault"].iloc[0]) if len(frame) else False
    violations = []
    if np.isfinite(fault_start_s):
        if fault_start_s < -1e-6:
            violations.append("negative_start")
        if fault_end_s < fault_start_s - 1e-6:
            violations.append("end_before_start")
        if fault_end_s > duration_s + 1e-6:
            violations.append("end_overflows_duration")
    missing_active_interval = system_fault and not active.any()
    # A fault flagged active on the very first sample is suspicious: it means
    # `rfly_ctrl_lxl` never observed an idle/sentinel state, which can be a
    # genuine "fault present from arm" protocol OR a sentinel-value mismatch
    # in `_active_control()` misreading the idle state as already-active.
    active_from_first_sample = bool(
        system_fault and active.any() and abs(fault_start_s - (t.min() if len(t) else 0.0)) < 1e-6
    )
    return {
        "canonical_case_id": str(frame["canonical_case_id"].iloc[0]) if len(frame) else "",
        "object_name": str(frame["object_name"].iloc[0]) if len(frame) else "",
        "package": str(frame["package"].iloc[0]) if len(frame) else "",
        "domain": str(frame["domain"].iloc[0]) if len(frame) else "",
        "fault_family": str(frame["fault_family"].iloc[0]) if len(frame) else "",
        "fault_subtype": str(frame["fault_subtype"].iloc[0]) if len(frame) else "",
        "system_fault": system_fault,
        "truth_source": str(frame["truth_source"].iloc[0]) if len(frame) else "",
        "truth_crosscheck_disagreement": bool(frame["truth_crosscheck_disagreement"].any()),
        "truth_crosscheck_eligible_v2": bool(
            _first_value(frame, "truth_crosscheck_eligible_v2", False)
        ),
        "truth_crosscheck_onset_delta_s": float(
            _first_value(frame, "truth_crosscheck_onset_delta_s", float("nan"))
        ),
        "truth_crosscheck_offset_delta_s": float(
            _first_value(frame, "truth_crosscheck_offset_delta_s", float("nan"))
        ),
        "truth_crosscheck_overlap_s": float(
            _first_value(frame, "truth_crosscheck_overlap_s", 0.0)
        ),
        "truth_crosscheck_disagreement_v2": bool(
            _first_value(frame, "truth_crosscheck_disagreement_v2", False)
        ),
        "truth_crosscheck_schema_version": int(
            _first_value(frame, "truth_crosscheck_schema_version", 0)
        ),
        "row_count": int(len(frame)),
        "duration_s": duration_s,
        "fault_start_s": fault_start_s,
        "fault_end_s": fault_end_s,
        "missing_active_interval": bool(missing_active_interval),
        "active_from_first_sample": active_from_first_sample,
        "interval_violation": ";".join(violations),
        "missing_v2_features": ";".join(missing_features),
        "trajectory_fingerprint": _trajectory_fingerprint(frame),
    }


def _trajectory_fingerprint(frame: pd.DataFrame) -> str:
    """Coarse content signature from five evenly spaced local-position samples.

    Distinguishes flights that only coincide on duration/row-count (e.g. many
    SIL/HIL runs of the same standardized test protocol share an identical
    flight length by design) from flights that are plausibly the same
    physical recording. Not a cryptographic hash — just a compact, stable
    string key so two flights with near-identical trajectories collide.
    """
    if not len(frame) or not {"local_x", "local_y", "local_z"}.issubset(frame.columns):
        return "no_position_data"
    positions = frame[["local_x", "local_y", "local_z"]].to_numpy(dtype=float)
    indices = np.linspace(0, len(positions) - 1, num=5).round().astype(int)
    sampled = np.nan_to_num(positions[indices], nan=-9999.0).round(1)
    return ";".join(",".join(str(value) for value in row) for row in sampled)


def _audit_one(path: Path) -> dict:
    schema_names = set(read_schema(path).names)
    missing_features = sorted(set(V2_FEATURES) - schema_names)
    frame = pd.read_parquet(
        path, columns=[column for column in READ_COLUMNS if column in schema_names]
    )
    return _audit_flight_frame(frame, missing_features=missing_features)


_CLUSTER_COLUMNS = [
    "tier", "signature", "flight_count", "distinct_split_group_ids",
    "spans_multiple_split_groups", "spans_locked_and_development",
    "canonical_case_ids", "split_group_ids",
]


def _cluster_by(frame: pd.DataFrame, signature: pd.Series, tier: str) -> pd.DataFrame:
    working = frame.assign(signature=signature)
    clusters = []
    for signature_value, group in working.groupby("signature"):
        if len(group) < 2:
            continue
        group_ids = sorted(group["split_group_id"].astype(str).unique())
        splits = sorted(group["split"].astype(str).unique())
        clusters.append({
            "tier": tier,
            "signature": signature_value,
            "flight_count": int(len(group)),
            "distinct_split_group_ids": len(group_ids),
            "spans_multiple_split_groups": len(group_ids) > 1,
            "spans_locked_and_development": len(splits) > 1,
            "canonical_case_ids": ";".join(group["canonical_case_id"].astype(str)),
            "split_group_ids": ";".join(group_ids),
        })
    return pd.DataFrame(clusters, columns=_CLUSTER_COLUMNS)


def _near_duplicate_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    """Two-tier near-duplicate audit beyond exact ULog SHA-256 matches.

    Tier "duration_signature": flights sharing (domain, fault_family,
    fault_subtype, rounded duration, row_count). This alone is NOT reliable
    evidence of duplication — manual inspection during this audit showed
    standardized SIL/HIL batch test protocols legitimately share an
    identical flight length by design (e.g. many distinct
    `SIL-Sensors/<pair>` scenarios all run for exactly 101.2s), so this tier
    over-clusters and is kept only for context.

    Tier "trajectory_signature": duration-signature AND a coarse local-
    position trajectory fingerprint both match. This is the tier the
    leakage-risk claim (`spans_locked_and_development`) should be read from;
    it requires the flights to also traverse a similar physical path, not
    just share a duration.
    """
    duration_signature = (
        frame["domain"] + "|" + frame["fault_family"] + "|" + frame["fault_subtype"]
        + "|" + frame["duration_s"].round(1).astype(str)
        + "|" + frame["row_count"].astype(str)
    )
    trajectory_signature = duration_signature + "|" + frame["trajectory_fingerprint"]
    clusters = pd.concat([
        _cluster_by(frame, duration_signature, "duration_signature"),
        _cluster_by(frame, trajectory_signature, "trajectory_signature"),
    ], ignore_index=True)
    if clusters.empty:
        return clusters
    return clusters.sort_values(
        ["tier", "spans_locked_and_development", "spans_multiple_split_groups", "flight_count"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def run_audit(*, split: str | None = None) -> pd.DataFrame:
    if split not in {None, "development"}:
        raise ValueError("truth audit split must be None or development")
    manifest_columns = ["canonical_case_id", "domain", "split_group_id", "split"]
    manifest = pd.read_parquet(
        DATASET_MANIFEST,
        columns=manifest_columns,
        filters=[("split", "==", split)] if split is not None else None,
    ).drop_duplicates("canonical_case_id")
    files = [
        PARSED_10HZ_ROOT / str(row.domain) / f"{row.canonical_case_id}.parquet"
        for row in manifest.itertuples(index=False)
    ]
    if not files:
        raise RuntimeError(f"no v2 10 Hz parquet files under {PARSED_10HZ_ROOT}")
    rows = [_audit_one(path) for path in files]
    frame = pd.DataFrame(rows)

    frame = frame.merge(
        manifest[["canonical_case_id", "split_group_id", "split"]],
        on="canonical_case_id", how="left",
    )
    unmatched_split = int(frame["split_group_id"].isna().sum())

    audit_root = DEVELOPMENT_AUDIT_ROOT if split == "development" else AUDIT_ROOT
    audit_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(audit_root / "truth_audit_per_flight.csv", index=False)

    domain_family = (
        frame.groupby(["domain", "fault_family", "truth_source"])
        .size().rename("flights").reset_index()
        .sort_values(["domain", "fault_family", "truth_source"])
    )
    domain_family.to_csv(audit_root / "truth_source_by_domain_family.csv", index=False)

    clusters = _near_duplicate_clusters(frame)
    clusters.to_csv(audit_root / "near_duplicate_clusters.csv", index=False)

    parse_state = json.loads(PARSE_STATE.read_text(encoding="utf-8")) if PARSE_STATE.exists() else {}
    system_fault_frame = frame.loc[frame["system_fault"]]
    disagreement_eligible = frame.loc[frame["truth_source"].eq("rfly_ctrl_lxl")]
    disagreement_v2_eligible = frame.loc[frame["truth_crosscheck_eligible_v2"]]
    active_from_start_frame = frame.loc[frame["active_from_first_sample"]]
    violation_frame = frame.loc[frame["interval_violation"].ne("")]
    schema_violation_frame = frame.loc[frame["missing_v2_features"].ne("")]
    duration_tier = clusters.loc[clusters["tier"].eq("duration_signature")] if len(clusters) else clusters
    trajectory_tier = clusters.loc[clusters["tier"].eq("trajectory_signature")] if len(clusters) else clusters
    trajectory_locked_dev = (
        trajectory_tier.loc[trajectory_tier["spans_locked_and_development"]]
        if len(trajectory_tier) else trajectory_tier
    )

    summary = {
        "scope_split": split or "all",
        "locked_test_features_read": split is None,
        "flights_audited": int(len(frame)),
        "parse_state_stop_reason": parse_state.get("stop_reason"),
        "parse_state_completed": len(parse_state.get("completed", [])),
        "parse_state_failed": len(parse_state.get("failed", {})),
        "parse_state_truth_schema_version": parse_state.get("truth_schema_version"),
        "parse_state_truth_reparse_invalidated": parse_state.get(
            "truth_reparse_invalidated", 0
        ),
        "manifest_split_group_join_misses": unmatched_split,
        "truth_source_distribution": frame["truth_source"].value_counts().to_dict(),
        "system_fault_flights": int(len(system_fault_frame)),
        "missing_active_interval_flights": int(frame["missing_active_interval"].sum()),
        "missing_active_interval_by_domain_family": (
            {
                f"{domain}|{family}": int(count)
                for (domain, family), count in system_fault_frame.loc[
                    system_fault_frame["missing_active_interval"]
                ].groupby(["domain", "fault_family"]).size().items()
            }
            if len(system_fault_frame) else {}
        ),
        "rfly_ctrl_lxl_flights": int(len(disagreement_eligible)),
        "truth_crosscheck_disagreement_flights": int(disagreement_eligible["truth_crosscheck_disagreement"].sum()),
        "truth_crosscheck_v2_onset_tolerance_s": CROSSCHECK_ONSET_TOLERANCE_S,
        "truth_crosscheck_v2_eligible_flights": int(len(disagreement_v2_eligible)),
        "truth_crosscheck_disagreement_v2_flights": int(
            disagreement_v2_eligible["truth_crosscheck_disagreement_v2"].sum()
        ),
        "truth_crosscheck_v2_disagreement_by_domain_family": {
            f"{domain}|{family}": int(count)
            for (domain, family), count in disagreement_v2_eligible.loc[
                disagreement_v2_eligible["truth_crosscheck_disagreement_v2"]
            ].groupby(["domain", "fault_family"]).size().items()
        },
        "truth_crosscheck_v2_onset_delta_abs_quantiles_s": {
            str(quantile): float(value)
            for quantile, value in disagreement_v2_eligible[
                "truth_crosscheck_onset_delta_s"
            ].abs().dropna().quantile([0.0, 0.5, 0.9, 0.99, 1.0]).items()
        },
        "truth_crosscheck_v2_offset_delta_quantiles_s": {
            str(quantile): float(value)
            for quantile, value in disagreement_v2_eligible[
                "truth_crosscheck_offset_delta_s"
            ].dropna().quantile([0.0, 0.5, 0.9, 0.99, 1.0]).items()
        },
        "active_from_first_sample_flights": int(len(active_from_start_frame)),
        "active_from_first_sample_by_package": (
            active_from_start_frame["package"].value_counts().to_dict()
            if len(active_from_start_frame) else {}
        ),
        "interval_violation_flights": int(len(violation_frame)),
        "schema_missing_v2_features_flights": int(len(schema_violation_frame)),
        "near_duplicate_duration_tier_clusters": int(len(duration_tier)),
        "near_duplicate_trajectory_tier_clusters": int(len(trajectory_tier)),
        "near_duplicate_trajectory_tier_spanning_locked_and_development": int(len(trajectory_locked_dev)),
    }
    summary_json = audit_root / "truth_audit_summary.json"
    summary_md = audit_root / "truth_audit_summary.md"
    _atomic_json(summary_json, summary)

    _write_markdown(summary, frame, summary_md=summary_md)
    return frame


def _write_markdown(summary: dict, frame: pd.DataFrame, *, summary_md: Path = SUMMARY_MD) -> None:
    lines = [
        "# RflyMAD-Full v2 truth audit (Aşama A)",
        "",
        "Bu rapor `dataset_manifest.parquet` yerine v2 10 Hz parse çıktısının "
        "kendi `truth_source`/`fault_active`/`truth_crosscheck_disagreement` "
        "kolonlarından üretildi. Manifest'in truth-ile-ilgili alanları hâlâ "
        "eski 1 Hz parser'a (`rfly_full/pipeline.py`, TestInfo-önce/"
        "rfly_ctrl_lxl-fallback) dayanıyor; v2 parser (`rfly_full/v2_parser.py`) "
        "önceliği tersine çevirdi (`rfly_ctrl_lxl`-önce, TestInfo-fallback). "
        "Bu iki öncelik SIRASI farklı olduğu için manifest özetindeki eski "
        "`provisional_testinfo_truth` sayısı gerçek v2 dağılımını yansıtmıyor.",
        "",
        f"- Denetlenen uçuş: **{summary['flights_audited']}**",
        f"- Kapsam split: `{summary['scope_split']}`",
        f"- Locked-test feature okundu: `{str(summary['locked_test_features_read']).lower()}`",
        f"- Parse state: `{summary['parse_state_stop_reason']}` "
        f"({summary['parse_state_completed']} completed, {summary['parse_state_failed']} failed)",
        f"- Truth schema: v{summary['parse_state_truth_schema_version']} "
        f"({summary['parse_state_truth_reparse_invalidated']} selectively invalidated/reparsed)",
        f"- Manifest split_group_id join kaçırma: {summary['manifest_split_group_join_misses']}",
        "",
        "## Gerçek v2 truth_source dağılımı",
        "",
        "| truth_source | uçuş |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["truth_source_distribution"].items(), key=lambda item: -item[1]):
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        f"- Sistem-arızalı uçuş: {summary['system_fault_flights']}",
        f"- `rfly_ctrl_lxl` kaynaklı uçuş (crosscheck-uygun): {summary['rfly_ctrl_lxl_flights']}",
        f"- Crosscheck disagreement (`rfly_ctrl_lxl` vs TestInfo, >%1 örnek uyuşmazlığı): "
        f"{summary['truth_crosscheck_disagreement_flights']} / {summary['rfly_ctrl_lxl_flights']}",
        f"- Crosscheck v2 disagreement (|onset delta| <= "
        f"{summary['truth_crosscheck_v2_onset_tolerance_s']:.0f}s ve interval overlap): "
        f"{summary['truth_crosscheck_disagreement_v2_flights']} / "
        f"{summary['truth_crosscheck_v2_eligible_flights']}",
        f"- Crosscheck v2 onset |delta| quantilleri (s): "
        f"`{summary['truth_crosscheck_v2_onset_delta_abs_quantiles_s']}`",
        f"- Crosscheck v2 signed offset delta quantilleri (s, yalnız tanısal): "
        f"`{summary['truth_crosscheck_v2_offset_delta_quantiles_s']}`",
        "",
        "  V2 boolean sample-by-sample eşitlik istemez. SIL'deki doğrulanmış zaman "
        "kaymasını onset toleransıyla ele alır ve iki aktif intervalin gerçekten "
        "örtüşmesini zorunlu tutar. Offset farkı aileye göre farklı bitiş "
        "tanımlarından etkilendiği için görünür raporlanır, tek başına boolean'ı "
        "kırmızıya çevirmez. Legacy alan geriye dönük karşılaştırma için korunur.",
        f"- Eksik aktif aralık (`system_fault=True` ama hiç `fault_active` yok): "
        f"{summary['missing_active_interval_flights']}",
        f"- **İlk örnekten itibaren aktif (t=0'dan başlıyor, şüpheli sentinel-değer sorunu adayı): "
        f"{summary['active_from_first_sample_flights']} / {summary['rfly_ctrl_lxl_flights']}**",
        "",
        "  Paket kırılımı:",
        "",
        "  | package | uçuş |",
        "  |---|---:|",
    ]
    for package, count in sorted(summary["active_from_first_sample_by_package"].items(), key=lambda item: -item[1]):
        lines.append(f"  | {package} | {count} |")
    lines += [
        "",
        "  Bu sayaç truth-quality guard'ıdır. Truth schema v2, `SIL_Motor_*`, "
        "`HIL_Motor_*`, `SIL_Prop` ve `HIL_Prop` paketlerini canonical domain "
        "ile yorumlar; önceki sahte t=0 yoğunluğunun kök nedeni underscore "
        "paket adlarının yanlış domain/sentinel seçmesiydi. Düzeltme sonrasında "
        "burada kalan paketler ayrı ham ULog incelemesi gerektirir.",
        "",
        f"- Aktif aralık negatif/taşma ihlali: {summary['interval_violation_flights']} "
        "(beklenen: 0)",
        f"- V2_FEATURES şema eksikliği olan uçuş: {summary['schema_missing_v2_features_flights']} "
        "(beklenen: 0)",
        "",
        "## Eksik aktif aralık — domain/aile kırılımı",
        "",
        "| domain | fault_family | uçuş |",
        "|---|---|---:|",
    ]
    for key, count in sorted(summary["missing_active_interval_by_domain_family"].items()):
        domain, family = key.split("|", 1)
        lines.append(f"| {domain} | {family} | {count} |")
    lines += [
        "",
        "## Near-duplicate audit (iki katmanlı heuristik)",
        "",
        "**Tier 1 — `duration_signature`**: `(domain, fault_family, fault_subtype, "
        "süre~0.1s, satır sayısı)` eşleşmesi. Bu tier TEK BAŞINA GÜVENİLİR "
        "DEĞİL: bu denetim sırasında en büyük kümeler elle incelendi ve "
        "SIL/HIL'de standardize batch test protokollerinin (ör. birçok farklı "
        "`SIL-Sensors/<çift>` senaryosu) aynı sabit uçuş süresini paylaştığı, "
        "ama FARKLI senaryolar olduğu görüldü — yani bu tier aşırı-kümeleniyor "
        "ve yalnız bağlam için tutuluyor.",
        "",
        "**Tier 2 — `trajectory_signature`**: Tier 1 + 5 eşit aralıklı örnekte "
        "kaba `local_x/y/z` konum parmak izi de eşleşmeli. Sızıntı riski iddiası "
        "bu tier'den okunmalı; yine de kriptografik hash değil, kaba bir "
        "içerik imzasıdır — kesin duplicate kanıtı değildir.",
        "",
        f"- Tier 1 (duration_signature) küme sayısı: {summary['near_duplicate_duration_tier_clusters']} "
        "(bilgi amaçlı, güvenilmez)",
        f"- Tier 2 (trajectory_signature) küme sayısı: {summary['near_duplicate_trajectory_tier_clusters']}",
        f"- **Tier 2'de locked_test ve development'a birlikte yayılan küme "
        f"(gerçek sızıntı riski adayı): "
        f"{summary['near_duplicate_trajectory_tier_spanning_locked_and_development']}**",
        "",
        "Ayrıntılar: `near_duplicate_clusters.csv`, `truth_audit_per_flight.csv`, "
        "`truth_source_by_domain_family.csv`.",
        "",
    ]
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    split = "development" if args.development_only else None
    frame = run_audit(split=split)
    summary = (
        DEVELOPMENT_AUDIT_ROOT / "truth_audit_summary.md"
        if split == "development" else SUMMARY_MD
    )
    print(f"audited {len(frame)} flights -> {summary}")


if __name__ == "__main__":
    main()
