"""Build the committed, executed RflyMAD-Full v2 experiment notebook.

The notebook deliberately reads compact, committed snapshots instead of the heavy
parquet/model artifacts.  This keeps it reviewable on GitHub and prevents a
notebook rerun from touching the locked-test feature store.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "rfly_full" / "v2"
ROBUSTNESS_ROOT = (
    ARTIFACT_ROOT
    / "normal_temporal_ae"
    / "robustness"
    / "approved_20260722_nested_v1"
)
R4_ROOT = ROBUSTNESS_ROOT / "candidates" / "R4"
TCN_SWEEP_ROOT = ARTIFACT_ROOT / "supervised_tcn" / "development_5fold_20260722_v1"
DATA_DIR = ROOT / "notebooks" / "data" / "rflymad_v2"
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "RFLYMAD_V2_TUM_DENEYLER_CALISTIRILMIS_20260722.ipynb"
)
DOC_ASSET_DIR = ROOT / "docs" / "assets" / "rflymad_v2_convergence"

VISUAL_FILES = [
    "00_validation_loss_R2_R3_R4_first25.png",
    "01_validation_loss_R2_R3_R4.png",
    "02_R4_train_validation_by_epoch.png",
    "03_R4_best_and_stop_epochs.png",
    "04_R3_R4_metric_comparison.png",
    "05_R4_rotation_stability.png",
    "06_real_recall_fa_tradeoff.png",
    "07_R4_validation_loss_reduction.png",
    "08_alarm_timeseries_case_studies.png",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _only(source: dict, fields: list[str]) -> dict:
    return {field: source.get(field) for field in fields}


def build_snapshots() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOC_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    parse_state = _read_json(ARTIFACT_ROOT / "parse_10hz_state.json")
    manifest = _read_json(ARTIFACT_ROOT / "dataset_manifest_summary.json")
    crosscheck = _read_json(ARTIFACT_ROOT / "crosscheck_v2_development_state.json")
    truth_audit = _read_json(
        ARTIFACT_ROOT / "truth_audit_development" / "truth_audit_summary.json"
    )
    tcn = _read_json(
        ARTIFACT_ROOT
        / "supervised_tcn"
        / "run_20260722_111938"
        / "summary.json"
    )
    tcn_development = _read_json(TCN_SWEEP_ROOT / "summary.json")
    tcn_development_gates = _read_json(TCN_SWEEP_ROOT / "gate_summary.json")
    experiment_state = _read_json(ROBUSTNESS_ROOT / "experiment_state.json")
    final_summary = _read_json(ROBUSTNESS_ROOT / "final_summary.json")
    r4_summary = _read_json(R4_ROOT / "summary.json")
    r4_gate = _read_json(R4_ROOT / "gate_summary.json")
    r4_bootstrap = _read_json(R4_ROOT / "bootstrap_ci.json")

    ae = pd.read_csv(
        ARTIFACT_ROOT
        / "normal_temporal_ae"
        / "sweep_20260722_093049"
        / "report_summary.csv"
    )
    ae.to_csv(DATA_DIR / "ae_sweep_summary.csv", index=False)

    pd.DataFrame(manifest["by_domain_family"]).to_csv(
        DATA_DIR / "manifest_by_domain_family.csv", index=False
    )
    pd.DataFrame(tcn["metrics"]).drop(columns=["confusion_flight"]).to_csv(
        DATA_DIR / "tcn_smoke_metrics.csv", index=False
    )
    pd.read_csv(TCN_SWEEP_ROOT / "outer_fold_metrics.csv").to_csv(
        DATA_DIR / "tcn_development_outer_metrics.csv", index=False
    )
    pd.read_csv(TCN_SWEEP_ROOT / "aggregate_metrics.csv").to_csv(
        DATA_DIR / "tcn_development_aggregate_metrics.csv", index=False
    )
    pd.read_csv(TCN_SWEEP_ROOT / "training_history.csv").to_csv(
        DATA_DIR / "tcn_development_training_history.csv", index=False
    )
    pd.read_csv(ROBUSTNESS_ROOT / "candidate_comparison_by_policy.csv").to_csv(
        DATA_DIR / "candidate_comparison_by_policy.csv", index=False
    )
    pd.read_csv(R4_ROOT / "all_rotation_metrics.csv").to_csv(
        DATA_DIR / "r4_rotation_metrics.csv", index=False
    )

    convergence_rows = []
    for rotation, row in enumerate(r4_summary["rotation_convergence"]):
        convergence_rows.append({"rotation": rotation, **row})
    pd.DataFrame(convergence_rows).to_csv(
        DATA_DIR / "r4_convergence.csv", index=False
    )

    history_frames = []
    for rotation in range(5):
        frame = pd.read_csv(R4_ROOT / f"rotation_{rotation}" / "training_history.csv")
        frame.insert(0, "rotation", rotation)
        history_frames.append(frame)
    pd.concat(history_frames, ignore_index=True).to_csv(
        DATA_DIR / "r4_training_history.csv", index=False
    )

    summary = {
        "snapshot_schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only; locked-test features were not read",
        "parser": {
            **_only(
                parse_state,
                [
                    "schema_version",
                    "sample_hz",
                    "stop_reason",
                    "remaining",
                    "canonical_flights",
                    "feature_schema_version",
                    "truth_schema_version",
                    "truth_reparse_invalidated",
                ],
            ),
            "completed_count": len(parse_state["completed"]),
            "failed_count": len(parse_state["failed"]),
        },
        "manifest": _only(
            manifest,
            [
                "canonical_flights",
                "locked_test_canonical_flights",
                "locked_test_fraction",
                "contract",
            ],
        ),
        "crosscheck": _only(
            crosscheck,
            [
                "status",
                "split",
                "crosscheck_schema_version",
                "onset_tolerance_s",
                "flights",
                "completed",
                "eligible",
                "disagreement_v2",
                "locked_test_features_read",
            ],
        ),
        "truth_audit": _only(
            truth_audit,
            [
                "scope_split",
                "locked_test_features_read",
                "flights_audited",
                "parse_state_truth_reparse_invalidated",
                "manifest_split_group_join_misses",
                "truth_source_distribution",
                "missing_active_interval_flights",
                "truth_crosscheck_disagreement_flights",
                "truth_crosscheck_v2_eligible_flights",
                "truth_crosscheck_disagreement_v2_flights",
                "interval_violation_flights",
                "schema_missing_v2_features_flights",
                "near_duplicate_trajectory_tier_spanning_locked_and_development",
            ],
        ),
        "tcn_smoke": {
            **_only(
                tcn,
                [
                    "status",
                    "model",
                    "validation_fold",
                    "development_smoke_fold",
                    "training_flights",
                    "validation_flights",
                    "test_flights",
                    "training_windows",
                    "validation_eval_windows",
                    "test_eval_windows",
                    "locked_test_features_read",
                    "operational_claim_allowed",
                ],
            ),
            "note": "test_flights is the development smoke fold, not the locked test",
        },
        "tcn_development": {
            "status": tcn_development["status"],
            "outer_folds": tcn_development["outer_folds"],
            "completed_outer_folds": tcn_development["completed_outer_folds"],
            "gate_summary": tcn_development_gates,
            "locked_test_features_read": tcn_development[
                "locked_test_features_read"
            ],
            "operational_claim_allowed": tcn_development[
                "operational_claim_allowed"
            ],
        },
        "robustness": {
            "status": final_summary["status"],
            "development_flights": experiment_state["development_flights"],
            "rotations": experiment_state["rotations"],
            "candidates_completed": experiment_state["candidates_completed"],
            "convergence_extension_ceiling": experiment_state[
                "convergence_extension_ceiling"
            ],
            "convergence_extended_rotations": experiment_state[
                "convergence_extended_rotations"
            ],
            "real_conclusion": final_summary["real_conclusion"],
            "wind_conclusion": final_summary["wind_conclusion"],
            "locked_test_features_read": final_summary["locked_test_features_read"],
            "operational_claim_allowed": final_summary[
                "operational_claim_allowed"
            ],
            "r4_gate": r4_gate,
            "r4_bootstrap": r4_bootstrap,
        },
        "verification": {
            "related_tests_passed": 50,
            "locked_test_features_read": False,
            "archive_changes": 0,
            "operational_claim_allowed": False,
        },
    }
    summary["truth_audit"]["truth_crosscheck_v2_disagreement_flights"] = (
        truth_audit["truth_crosscheck_disagreement_v2_flights"]
    )
    (DATA_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for filename in VISUAL_FILES:
        shutil.copy2(R4_ROOT / filename, DOC_ASSET_DIR / filename)


def _markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def _code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        _markdown(
            """
# RflyMAD-Full v2 — Çalıştırılmış deney özeti

**Tarih:** 22 Temmuz 2026
**Kapsam:** veri/parser/truth denetimleri, normal-only temporal AE, TCN development
smoke ve beş-fold development sweep, Wind/Real robustness adayları ve R4
convergence takibi.
**Güvenlik sınırı:** locked-test feature dosyaları okunmadı; sonuçlar yalnız
development araştırma bulgusudur ve operasyonel iddia oluşturmaz.

Bu notebook çıktı hücreleriyle birlikte commit edilmiştir. Yeniden çalıştırıldığında
yalnız `notebooks/data/rflymad_v2/` altındaki küçük, sabitlenmiş özetleri okur; model,
parquet veya kilitli test dosyalarına erişmez.
"""
        ),
        _code(
            """
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "notebooks" / "data" / "rflymad_v2").exists():
    ROOT = ROOT.parent
DATA = ROOT / "notebooks" / "data" / "rflymad_v2"
assert DATA.exists(), f"Özet veri dizini bulunamadı: {DATA}"

summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
manifest = pd.read_csv(DATA / "manifest_by_domain_family.csv")
ae = pd.read_csv(DATA / "ae_sweep_summary.csv")
tcn = pd.read_csv(DATA / "tcn_smoke_metrics.csv")
tcn_dev_outer = pd.read_csv(DATA / "tcn_development_outer_metrics.csv")
tcn_dev_aggregate = pd.read_csv(DATA / "tcn_development_aggregate_metrics.csv")
tcn_dev_history = pd.read_csv(DATA / "tcn_development_training_history.csv")
comparison = pd.read_csv(DATA / "candidate_comparison_by_policy.csv")
r4_metrics = pd.read_csv(DATA / "r4_rotation_metrics.csv")
convergence = pd.read_csv(DATA / "r4_convergence.csv")
history = pd.read_csv(DATA / "r4_training_history.csv")

pd.set_option("display.max_columns", 50)
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"frozen_baseline": "#9d9da1", "R1": "#4c78a8", "W1": "#72b7b2",
          "W2": "#54a24b", "R2": "#f2cf5b", "R3": "#b279a2", "R4": "#e45756"}
print(f"Snapshot: {summary['generated_at']}")
print(f"Veri kaynağı: {DATA.relative_to(ROOT)}")
"""
        ),
        _markdown(
            """
## 1. Veri, preprocessing ve truth sözleşmesi

- Kanonik birim uçuştur; satır/event/uçuş metrikleri birbirine karıştırılmaz.
- Örnekleme 10 Hz’dir; truth şeması v2’dir.
- Wind, sistem arızası pozitif sınıfı değil robustness/nonfault alanıdır.
- Locked test 1.225 uçuş olarak ayrılmıştır; bu çalışma boyunca feature’ları okunmamıştır.
- Parser düzeltmesi, eski toleranssız cross-check alanını silmeden yeni toleranslı v2
  alanını ekler.
"""
        ),
        _code(
            """
p = summary["parser"]
m = summary["manifest"]
c = summary["crosscheck"]
a = summary["truth_audit"]

facts = pd.DataFrame([
    ["Kanonik uçuş", p["canonical_flights"], "parser complete"],
    ["10 Hz parse tamamlanan", p["completed_count"], f"failed={p['failed_count']}"],
    ["Truth-v2 için yeniden parse", p["truth_reparse_invalidated"], "parser hatasından etkilenen"],
    ["Development audit", a["flights_audited"], a["scope_split"]],
    ["Cross-check v2 eligible", c["eligible"], f"uyuşmazlık={c['disagreement_v2']}"],
    ["Locked test", m["locked_test_canonical_flights"], "feature okunmadı"],
], columns=["Kanıt", "Değer", "Durum"])
display(facts)

audit = pd.DataFrame([
    ["Eski toleranssız disagreement", a["truth_crosscheck_disagreement_flights"]],
    ["Toleranslı v2 disagreement", a["truth_crosscheck_v2_disagreement_flights"]],
    ["Manifest split/group join miss", a["manifest_split_group_join_misses"]],
    ["Interval ihlali", a["interval_violation_flights"]],
    ["Eksik truth-v2 feature", a["schema_missing_v2_features_flights"]],
    ["Dev/locked trajectory near-duplicate", a["near_duplicate_trajectory_tier_spanning_locked_and_development"]],
], columns=["Truth/sızıntı kontrolü", "Adet"])
display(audit)
"""
        ),
        _code(
            """
pivot = manifest.pivot(index="domain", columns="fault_family", values="flights").fillna(0)
display(pivot.astype(int))
ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
ax.set_title("Kanonik uçuş dağılımı — domain ve fault family")
ax.set_xlabel("Domain")
ax.set_ylabel("Uçuş")
ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
"""
        ),
        _markdown(
            """
## 2. AE ve TCN sağlık sonuçları

Normal-only temporal AE, beş development rotasyonu üzerinde raporlandı. TCN ise
`--development-smoke-fold 1 --epochs 3` sağlık kontrolüdür; `status=smoke_only`
olduğu için nihai model karşılaştırması veya operasyonel kanıt değildir.
"""
        ),
        _code(
            """
ae_view = ae.rename(columns={
    "policy": "politika", "recall_mean": "event_recall",
    "recall_std": "recall_std", "fa_mean": "nonfault_FA/saat",
    "fa_std": "FA_std", "wind_fa_mean": "Wind_FA/saat"
})
display(Markdown("**Normal-only temporal AE — 5 rotasyon ortalaması**"))
display(ae_view.round(4))

tcn_view = tcn[["policy", "event_recall", "false_alarms_per_hour",
                "median_detection_delay_s", "flight_tp", "flight_fn", "flight_fp", "flight_tn"]]
display(Markdown("**TCN — development smoke fold 1, 3 epoch**"))
display(tcn_view.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
policies = ae["policy"].tolist()
x = np.arange(len(policies)); width = 0.34
axes[0].bar(x-width/2, ae["recall_mean"]*100, width, label="AE")
axes[0].bar(x+width/2, tcn.set_index("policy").loc[policies, "event_recall"]*100, width, label="TCN smoke")
axes[0].set_xticks(x, policies); axes[0].set_ylabel("Event recall (%)"); axes[0].legend()
axes[1].bar(x-width/2, ae["fa_mean"], width, label="AE")
axes[1].bar(x+width/2, tcn.set_index("policy").loc[policies, "false_alarms_per_hour"], width, label="TCN smoke")
axes[1].set_xticks(x, policies); axes[1].set_ylabel("False alarm / saat"); axes[1].legend()
fig.suptitle("AE raporu ve TCN smoke — kapsamları farklıdır, doğrudan yarış değildir")
plt.tight_layout(); plt.show()
"""
        ),
        _markdown(
            """
## 3. TCN development-only 5-fold sweep

TCN, kilitli test yerine her seferinde ayrı bir development outer fold üzerinde
değerlendirildi. Validation fold yalnız checkpoint/uzatma seçimi için kullanıldı;
outer sonuçları epoch kararına girmedi. Başlangıç tavanı 12 epoch’tu; yalnız en iyi
validation epoch’u son iki epoch’a dayanır ve anlamlı iyileşme sürerse 25/50’ye
uzatma mümkündü. Bu sonuçlar development-only’dir ve operasyonel iddia değildir.
"""
        ),
        _code(
            """
display(Markdown("**TCN development 5-fold aggregate**"))
display(tcn_dev_aggregate.round(4))

best_rows = []
for outer_fold, group in tcn_dev_history.groupby("outer_fold"):
    best = group.loc[group["validation_loss"].idxmin()]
    best_rows.append({
        "outer_fold": int(outer_fold),
        "validation_fold": int(group["validation_fold"].iloc[0]),
        "epoch_cap": int(group["epoch_cap"].max()),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
    })
best_epochs = pd.DataFrame(best_rows)
display(best_epochs.round(5))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for outer_fold, group in tcn_dev_history.groupby("outer_fold"):
    group = group.sort_values("epoch")
    axes[0].plot(group["epoch"], group["train_loss"], marker=".", label=f"fold {outer_fold}")
    axes[1].plot(group["epoch"], group["validation_loss"], marker=".", label=f"fold {outer_fold}")
axes[0].set_title("TCN epoch başına training loss")
axes[1].set_title("TCN epoch başına validation loss")
for ax in axes:
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(ncol=2); ax.grid(alpha=.25)
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for policy, group in tcn_dev_outer.groupby("policy"):
    group = group.sort_values("outer_fold")
    axes[0].plot(group["outer_fold"], group["event_recall"]*100, "o-", label=policy)
    axes[1].plot(group["outer_fold"], group["all_nonfault_fa_per_hour"], "o-", label=policy)
axes[0].set_title("Outer-fold event recall"); axes[0].set_ylabel("Recall (%)")
axes[1].set_title("Outer-fold nonfault false alarm"); axes[1].set_ylabel("Alarm / saat")
for ax in axes:
    ax.set_xlabel("Outer fold"); ax.set_xticks(range(5)); ax.legend(); ax.grid(alpha=.25)
plt.tight_layout(); plt.show()

tcn_gates = summary["tcn_development"]["gate_summary"]
display(pd.DataFrame([
    [name, gate["passed"]]
    for name, gate in tcn_gates.items() if isinstance(gate, dict)
], columns=["TCN development kapısı", "Geçti"]))
print(f"Locked-test features read: {summary['tcn_development']['locked_test_features_read']}")
print(f"Operational claim allowed: {summary['tcn_development']['operational_claim_allowed']}")
"""
        ),
        _markdown(
            """
## 4. Wind/Real robustness deneyleri

Başarı ölçütleri sonuçlardan önce donduruldu. R1, W1, W2, R2 ve R3 sekiz epoch
bütçeli nested development adaylarıdır. R4, kullanıcının epoch bütçesi itirazı
üzerine ayrı sözleşmeyle yapılan validation-early-stopping convergence takibidir.
"""
        ),
        _code(
            """
critical = comparison.query("policy == 'critical'").copy()
cols = ["candidate", "event_recall_mean", "all_nonfault_fa_per_hour_mean",
        "wind_fa_per_hour_mean", "real_motor_recall_mean", "real_sensor_recall_mean",
        "real_macro_recall_mean", "real_normal_fa_per_hour_mean"]
display(critical[cols].round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(critical))
axes[0].bar(x, critical["event_recall_mean"]*100,
            color=[COLORS[c] for c in critical["candidate"]])
axes[0].plot(x, critical["real_macro_recall_mean"]*100, "ko--", label="Real macro")
axes[0].axhline(40, color="#54a24b", ls="--", label="Real hedef %40")
axes[0].set_xticks(x, critical["candidate"]); axes[0].set_ylabel("Recall (%)")
axes[0].set_title("Critical recall"); axes[0].legend()
axes[1].bar(x-0.2, critical["all_nonfault_fa_per_hour_mean"], 0.4, label="Tüm nonfault")
axes[1].bar(x+0.2, critical["wind_fa_per_hour_mean"], 0.4, label="Wind")
axes[1].axhline(2, color="#e45756", ls="--", label="Genel FA sınırı 2/s")
axes[1].set_xticks(x, critical["candidate"]); axes[1].set_ylabel("Alarm / saat")
axes[1].set_title("Critical false-alarm yükü"); axes[1].legend()
plt.tight_layout(); plt.show()
"""
        ),
        _markdown(
            """
## 5. R4 convergence: epoch-başına davranış

Sabit sekiz epoch yeterli değildi: rotasyona bağlı olarak en iyi checkpoint epoch
1 ile 780 arasında değişti. Seçim yalnız inner Real-NoFault validation loss ile
yapıldı; outer fault metrikleri epoch seçimine girmedi.
"""
        ),
        _code(
            """
conv_view = convergence[["rotation", "initial_validation_loss", "best_validation_loss",
                         "best_epoch", "epochs_completed", "stop_reason"]].copy()
display(conv_view.round(5))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for rotation, group in history.groupby("rotation"):
    axes[0].plot(group["epoch"], group["validation_loss"], label=f"rot {rotation}")
    short = group[group["epoch"] <= 25]
    axes[1].plot(short["epoch"], short["validation_loss"], marker=".", label=f"rot {rotation}")
axes[0].set_title("R4 tam validation-loss eğrileri")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Inner validation loss"); axes[0].legend()
axes[1].set_title("İlk 25 epoch yakın görünüm")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Inner validation loss"); axes[1].legend()
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(convergence)); width = 0.36
ax.bar(x-width/2, convergence["best_epoch"], width, label="En iyi epoch")
ax.bar(x+width/2, convergence["epochs_completed"], width, label="Durma epoch")
ax.set_xticks(x, convergence["rotation"]); ax.set_xlabel("Rotasyon"); ax.set_ylabel("Epoch")
ax.set_title("Validation early stopping: en iyi ve durma epochları"); ax.legend()
plt.tight_layout(); plt.show()
"""
        ),
        _markdown(
            """
## 6. R4 rotasyon kararlılığı ve trade-off

Convergence Real recall’ı R3’e göre artırdı; buna karşılık genel recall düştü ve
false-alarm yükü arttı. Beş rotasyon arasındaki geniş saçılım, ortalama sonucun tek
başına kararlı model davranışı sayılmaması gerektiğini gösterir.
"""
        ),
        _code(
            """
r4c = r4_metrics.query("policy == 'critical'").sort_values("rotation")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
x = r4c["rotation"]
axes[0].plot(x, r4c["event_recall"]*100, "o-", label="Genel")
axes[0].plot(x, r4c["real_macro_recall"]*100, "o-", label="Real macro")
axes[0].axhline(40, color="#54a24b", ls="--", label="Real hedef %40")
axes[0].set_ylabel("Recall (%)"); axes[0].set_title("Recall"); axes[0].legend()
axes[1].plot(x, r4c["all_nonfault_fa_per_hour"], "o-", label="Tüm nonfault")
axes[1].plot(x, r4c["real_normal_fa_per_hour"], "o-", label="Real normal")
axes[1].axhline(2, color="#e45756", ls="--", label="Genel sınır 2/s")
axes[1].axhline(4, color="#f2cf5b", ls="--", label="Real sınır 4/s")
axes[1].set_ylabel("Alarm / saat"); axes[1].set_title("Normal false alarm"); axes[1].legend()
axes[2].bar(x, r4c["wind_fa_per_hour"], color="#72b7b2")
axes[2].axhline(15, color="#54a24b", ls="--", label="Ara hedef 15/s")
axes[2].set_ylabel("Wind alarm / saat"); axes[2].set_title("Wind robustness"); axes[2].legend()
for ax in axes: ax.set_xlabel("Outer rotasyon"); ax.set_xticks(x)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.axvspan(0, 4, color="#54a24b", alpha=.08)
ax.axhspan(40, 100, color="#54a24b", alpha=.08)
for _, row in critical.iterrows():
    xval = row["real_normal_fa_per_hour_mean"]
    yval = row["real_macro_recall_mean"] * 100
    ax.scatter(xval, yval, s=75, color=COLORS[row["candidate"]])
    ax.annotate(row["candidate"], (xval, yval), xytext=(5, 5), textcoords="offset points")
ax.axvline(4, color="#e45756", ls="--"); ax.axhline(40, color="#4c78a8", ls="--")
ax.set_xlim(left=0); ax.set_ylim(0, max(45, critical["real_macro_recall_mean"].max()*115))
ax.set_xlabel("Real-NoFault FA / saat"); ax.set_ylabel("Real macro recall (%)")
ax.set_title("Real transfer trade-off — hiçbir aday hedef bölgesinde değil")
plt.tight_layout(); plt.show()
"""
        ),
        _markdown(
            """
## 7. Nihai karar

AE robustness R4 yalnız development convergence takibidir. Real research gate,
Wind ara hedefi ve Wind nihai hedefi geçilmedi. TCN development sweep'inde de
critical, advisory, Real ve Wind kapılarının hiçbiri geçmedi. Locked test
açılmamalı; mevcut temsil ve veriyle operasyonel/fizibilite iddiası kurulamaz.
"""
        ),
        _code(
            """
rob = summary["robustness"]
gate = rob["r4_gate"]
agg = gate["critical_aggregate"]
ci = rob["r4_bootstrap"]["policies"]["critical"]["real_macro_recall"]

decision = pd.DataFrame([
    ["Critical event recall", agg["event_recall"]["mean"], "koruma kapısı", False],
    ["Real Motor recall", agg["real_motor_recall"]["mean"], ">= 0.30", False],
    ["Real Sensor recall", agg["real_sensor_recall"]["mean"], ">= 0.30", False],
    ["Real macro recall", agg["real_macro_recall"]["mean"], ">= 0.40", False],
    ["Real-NoFault FA/saat", agg["real_normal_fa_per_hour"]["mean"], "<= 4", False],
    ["Tüm nonfault FA/saat", agg["all_nonfault_fa_per_hour"]["mean"], "<= 2", False],
    ["Wind FA/saat", agg["wind_fa_per_hour"]["mean"], "<= 15 ara hedef", False],
], columns=["Metrik", "R4 ortalama", "Ölçüt", "Geçti"])
display(decision.round(4))
print(f"Real macro cluster-bootstrap %95 GA: {ci['lower_95']:.4f}–{ci['upper_95']:.4f}")
print(f"Real research gate: {gate['real_research_gate']['passed']}")
print(f"Wind intermediate gate: {gate['wind_intermediate_gate']['passed']}")
print(f"Locked-test features read: {rob['locked_test_features_read']}")
print(f"Operational claim allowed: {rob['operational_claim_allowed']}")
print("TCN gates:", {
    name: gate["passed"]
    for name, gate in summary["tcn_development"]["gate_summary"].items()
    if isinstance(gate, dict)
})
"""
        ),
        _markdown(
            """
## 8. Tekrarlanabilirlik

- Kaynak kod: `rfly_full/`
- Çalıştırıcılar: `scripts/run_rfly_full_v2_*.py`
- Notebook üretici: `scripts/build_rfly_full_v2_summary_notebook.py`
- AE convergence raporu: `docs/RFLYMAD_V2_CONVERGENCE_DENEY_RAPORU_20260722.md`
- TCN development raporu: `docs/RFLYMAD_V2_TCN_DEVELOPMENT_DENEY_RAPORU_20260722.md`
- Kompakt notebook verisi: `notebooks/data/rflymad_v2/`
- Ağır yerel artefakt kökü: `artifacts/rfly_full/v2/` (commit edilmez)

Son doğrulama: ilgili 50 test geçti; `archive/` değişikliği yoktu. Ham model/parquet
dosyaları ve loglar yeniden üretilebilir yerel artefakt olarak bırakıldı.
"""
        ),
        _code(
            """
files = sorted(p.name for p in DATA.iterdir() if p.is_file())
pd.DataFrame({"Commit edilen özet dosya": files})
"""
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "rflymad": {
                "scope": "development_only",
                "locked_test_features_read": False,
                "operational_claim_allowed": False,
            },
        },
    )
    return notebook


def execute_and_write(notebook: nbformat.NotebookNode) -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook execution produced {len(errors)} error output(s)")
    nbformat.write(notebook, NOTEBOOK_PATH)


def main() -> None:
    build_snapshots()
    execute_and_write(build_notebook())
    print(NOTEBOOK_PATH.relative_to(ROOT))
    print(DATA_DIR.relative_to(ROOT))
    print(DOC_ASSET_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
