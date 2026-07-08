"""Ucus (source_id) bazli train/val/test split'leri (ML-0 fazi).

ASLA satir bazli random split yapilmaz: ayni ucusun t ve t+0.2s ornekleri
train/test'e bolunurse model ucusu ezberler ve sahte yuksek skor verir.
Split birimi source_id'dir (GroupShuffleSplit mantigi).

Yari-gozetimli (novelty detection) kurgusu:
- TRAIN yalnizca tamamen-normal ucuslardan olusur (IF/LSTM-AE normal'i ogrenir).
- VAL normal ucuslardir: threshold = percentile(val_scores, 99) buradan secilir.
- TEST = kalan normal ucuslar + TUM anomali ucuslari.
- ALFA 'unknown' etiketli ucus train/val/test'e GIRMEZ -- ayri "exploration"
  kumesi olarak isaretlenir (etiketi guvenilmez, skorlanip incelenebilir).
- Etiketler feature DEGILDIR; yalnizca degerlendirmede kullanilir.

Az sayida normal ucus oldugu icin (ALFA: 10, UAV Attack: 6) tek split yeterli
degil: N_SEEDS farkli seed'le tekrarlanir (rapor: ortalama +- std) ve ayrica
leave-one-flight-out (LOFO) listesi uretilir. Hepsi split_manifest.json'a yazilir.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_SEEDS = 5

# Kaynak basina normal etiket adi (ALFA 'normal', PX4 kaynaklari 'benign'/'Normal').
NORMAL_LABELS = {"normal", "benign", "Normal"}
EXPLORATION_LABELS = {"unknown"}


def flight_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ucus basina tek etiket: tum satirlari normal ise 'normal', degilse
    ucusta gorulen ilk anomali etiketi (ALFA'da ariza onset'ten sonra baslar,
    oncesi normal gorunur -- ucus yine de anomalili ucustur)."""
    rows = []
    for source_id, g in df.groupby("source_id"):
        labels = set(g["label"].dropna().unique())
        anomaly = sorted(labels - NORMAL_LABELS - EXPLORATION_LABELS)
        if labels & EXPLORATION_LABELS and not anomaly:
            flight_label = "unknown"
        elif anomaly:
            flight_label = anomaly[0]
        else:
            flight_label = "normal"
        rows.append({"source_id": source_id, "flight_label": flight_label, "n_rows": len(g)})
    return pd.DataFrame(rows)


def session_of(source_id: str) -> str:
    """Ucusun oturum anahtari: UAV-SEAD'de 'tarih-klasoru/saat' formatindaki id'nin
    tarih kismi (ayni gun/oturum ucuslari birbirine cok benzer -- ayni oturumun
    biri train'e biri test'e duserse sizintiya benzer iyimserlik dogar).
    Klasorsuz id'ler (ALFA, UAV Attack, SEAD'in '00_02_49' gibi kokleri) kendi
    baslarina oturumdur (davranis ucus-bazli split'e esdeger kalir)."""
    return source_id.split("/")[0] if "/" in source_id else source_id


def _anomaly_dev_holdout(anomalous: list[str], *, by_session: bool,
                         holdout_fraction: float, holdout_seed: int) -> tuple[list[str], list[str]]:
    """Anomalileri sabit development/final-holdout gruplarina ayir.

    Split seed'lerinden bagimsiz ``holdout_seed`` kullanilir; boylece bes deney
    seed'i boyunca final set degismez ve model seciminde dolayli kullanilmaz.
    """
    if holdout_fraction <= 0.0 or not anomalous:
        return sorted(anomalous), []
    rng = np.random.default_rng(holdout_seed)
    if by_session:
        groups = sorted({session_of(f) for f in anomalous})
        rng.shuffle(groups)
        n_hold = min(len(groups) - 1, max(1, round(len(groups) * holdout_fraction)))
        held_groups = set(groups[:n_hold])
        final = [f for f in anomalous if session_of(f) in held_groups]
    else:
        order = list(anomalous)
        rng.shuffle(order)
        n_hold = min(len(order) - 1, max(1, round(len(order) * holdout_fraction)))
        final = order[:n_hold]
    final_set = set(final)
    return sorted(f for f in anomalous if f not in final_set), sorted(final)


def _normalise_frozen_holdout(spec: Any) -> dict[str, set[str]] | None:
    """Return previous final/development/known flights for incremental holdout.

    ML-14 passes the previous source manifest config, but accepting a plain
    final_holdout list keeps the public parameter small for simpler callers.
    With only a list we can preserve old final flights, while full old-dev
    contamination asserts require the previous split/source metadata.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        if "splits" in spec:
            split = spec["splits"]["split_00"]
            final = set(split.get("final_holdout", []))
            exploration = set(split.get("exploration", []))
            labels = set(spec.get("flight_labels", {}))
            development = labels - final - exploration if labels else set().union(
                *(set(split.get(part, [])) for part in ("train", "val", "test"))
            )
            known = labels or development | final | exploration
            return {
                "final": final,
                "development": development,
                "known": known,
            }
        final = set(spec.get("final_holdout", spec.get("old_final_holdout", [])))
        development = set(spec.get(
            "development",
            spec.get("old_development", spec.get("development_flights", [])),
        ))
        known = set(spec.get("known", spec.get("old_known_flights", [])))
        if not known:
            known = final | development
        return {
            "final": final,
            "development": development,
            "known": known,
        }
    final = set(spec)
    return {"final": final, "development": set(), "known": set(final)}


def _partition_holdout(
    normal: list[str],
    all_anomalous: list[str],
    *,
    by_session: bool,
    holdout_fraction: float,
    holdout_seed: int,
    frozen_holdout: Any = None,
) -> tuple[list[str], list[str], set[str], dict[str, set[str]] | None]:
    frozen = _normalise_frozen_holdout(frozen_holdout)
    if frozen is None:
        anomalous, final_anomalous = _anomaly_dev_holdout(
            all_anomalous,
            by_session=by_session,
            holdout_fraction=holdout_fraction,
            holdout_seed=holdout_seed,
        )
        final_groups = (
            {session_of(f) for f in final_anomalous}
            if by_session else set(final_anomalous)
        )
        return anomalous, final_anomalous, final_groups, None

    current = set(normal) | set(all_anomalous)
    missing = frozen["final"] - current
    if missing:
        raise AssertionError(
            "Frozen holdout flights are missing from the refreshed table: "
            f"{sorted(missing)}"
        )

    if by_session:
        frozen_groups = {session_of(f) for f in frozen["final"]}
        known_groups = (
            {session_of(f) for f in frozen["known"]}
            if frozen["known"] else set(frozen_groups)
        )
        frozen_final = [
            f for f in all_anomalous if session_of(f) in frozen_groups
        ]
        old_development = [
            f for f in all_anomalous
            if session_of(f) in known_groups and session_of(f) not in frozen_groups
        ]
        new_anomalous = [
            f for f in all_anomalous if session_of(f) not in known_groups
        ]
        new_dev, new_final = _anomaly_dev_holdout(
            new_anomalous,
            by_session=True,
            holdout_fraction=holdout_fraction,
            holdout_seed=holdout_seed,
        )
        final_groups = frozen_groups | {session_of(f) for f in new_final}
        return (
            sorted(old_development + new_dev),
            sorted(frozen_final + new_final),
            final_groups,
            frozen,
        )

    frozen_final = [f for f in all_anomalous if f in frozen["final"]]
    old_development = [
        f for f in all_anomalous if f in frozen["known"] and f not in frozen["final"]
    ]
    new_anomalous = [f for f in all_anomalous if f not in frozen["known"]]
    new_dev, new_final = _anomaly_dev_holdout(
        new_anomalous,
        by_session=False,
        holdout_fraction=holdout_fraction,
        holdout_seed=holdout_seed,
    )
    final_groups = set(frozen["final"]) | set(new_final)
    return sorted(old_development + new_dev), sorted(frozen_final + new_final), final_groups, frozen


def _assert_frozen_holdout_contract(split: dict, frozen: dict[str, set[str]] | None) -> None:
    if frozen is None:
        return
    final = set(split["final_holdout"])
    development = set(split["train"]) | set(split["val"]) | set(split["test"])
    if not frozen["final"] <= final:
        missing = sorted(frozen["final"] - final)
        raise AssertionError(f"Old final_holdout is not preserved: {missing}")
    if frozen["development"] & final:
        overlap = sorted(frozen["development"] & final)
        raise AssertionError(f"Old development leaked into new holdout: {overlap}")
    if final & development:
        overlap = sorted(final & development)
        raise AssertionError(f"New final_holdout overlaps development: {overlap}")


def make_group_split(flights: pd.DataFrame, *, seed: int,
                     n_val: int = 1, n_test_normal: int = 1,
                     by_session: bool = False,
                     final_holdout_fraction: float = 0.0,
                     holdout_seed: int = 20260703,
                     frozen_holdout: Any = None) -> dict:
    """Tek bir seed icin ucus-bazli (veya oturum-bazli) split sozlugu uretir.

    Normal ucuslar karistirilip val/test-normal ayrilir, kalani train olur.
    Anomalili ucuslarin TAMAMI test'e gider; 'unknown' exploration'a.

    by_session=True (UAV-SEAD): split birimi ucus degil OTURUM olur ve
    icinde anomalili ucus bulunan oturumlarin normal ucuslari train/val'e
    ALINMAZ (test-normal'e gider) -- ayni oturum iki tarafa dusemez.
    Kotalar oturum sayisi degil, yaklasik ucus sayisi olarak yorumlanir.
    """
    normal = flights[flights["flight_label"] == "normal"]["source_id"].tolist()
    all_anomalous = flights[~flights["flight_label"].isin(["normal", "unknown"])]["source_id"].tolist()
    exploration = flights[flights["flight_label"] == "unknown"]["source_id"].tolist()
    anomalous, final_anomalous, final_groups, frozen = _partition_holdout(
        normal,
        all_anomalous,
        by_session=by_session,
        holdout_fraction=final_holdout_fraction,
        holdout_seed=holdout_seed,
        frozen_holdout=frozen_holdout,
    )

    if len(normal) < n_val + n_test_normal + 1:
        raise ValueError(
            f"Yetersiz normal ucus: {len(normal)} adet, en az {n_val + n_test_normal + 1} gerek")

    rng = np.random.default_rng(seed)

    if not by_session:
        shuffled = list(normal)
        rng.shuffle(shuffled)
        val = sorted(shuffled[:n_val])
        test_normal = sorted(shuffled[n_val : n_val + n_test_normal])
        train = sorted(shuffled[n_val + n_test_normal :])
        final_normal: list[str] = []
    else:
        dev_anomalous_sessions = {session_of(f) for f in anomalous}
        final_anomalous_sessions = set(final_groups)
        anomalous_sessions = dev_anomalous_sessions | final_anomalous_sessions
        # Anomalili oturumdaki normaller train/val'e giremez. Development
        # anomaly kardesleri test-normal, final anomaly kardesleri blind final olur.
        tainted = [f for f in normal if session_of(f) in dev_anomalous_sessions]
        final_normal = [f for f in normal if session_of(f) in final_anomalous_sessions]
        clean_sessions: dict[str, list[str]] = {}
        for f in normal:
            s = session_of(f)
            if s not in anomalous_sessions:
                clean_sessions.setdefault(s, []).append(f)
        order = sorted(clean_sessions)
        rng.shuffle(order)
        val, test_normal, train = [], list(tainted), []
        for s in order:
            fl = clean_sessions[s]
            if len(val) < n_val:
                val += fl
            elif len(test_normal) - len(tainted) < n_test_normal:
                test_normal += fl
            else:
                train += fl
        val, test_normal, train = sorted(val), sorted(test_normal), sorted(train)
        if not train or not val:
            raise ValueError("Oturum-bazli split icin yeterli temiz oturum yok")

    split = {
        "seed": seed,
        "train": train,
        "val": val,
        "test": sorted(test_normal + anomalous),
        "test_normal": test_normal,
        "test_anomalous": sorted(anomalous),
        "development_anomalous": sorted(anomalous),
        "final_holdout": sorted(final_normal + final_anomalous),
        "final_holdout_normal": sorted(final_normal),
        "final_holdout_anomalous": sorted(final_anomalous),
        "exploration": sorted(exploration),
    }
    _assert_frozen_holdout_contract(split, frozen)
    return split


def make_lofo_splits(flights: pd.DataFrame, *, by_session: bool = False,
                     final_holdout_fraction: float = 0.0,
                     holdout_seed: int = 20260703,
                     frozen_holdout: Any = None) -> list[dict]:
    """Normal validation fold'lari uret.

    ``by_session=False`` klasik leave-one-flight-out'tur. SEAD icin
    ``by_session=True`` kullanilir ve bir oturumun tum normal ucuslari birlikte
    validation'a ayrilir; kardes ucuslar train'de kalamaz.
    """
    normal = sorted(flights[flights["flight_label"] == "normal"]["source_id"].tolist())
    all_anomalous = sorted(flights[~flights["flight_label"].isin(["normal", "unknown"])]["source_id"].tolist())
    anomalous, final_anomalous, final_groups, _ = _partition_holdout(
        normal,
        all_anomalous,
        by_session=by_session,
        holdout_fraction=final_holdout_fraction,
        holdout_seed=holdout_seed,
        frozen_holdout=frozen_holdout,
    )
    if by_session:
        dev_sessions = {session_of(f) for f in anomalous}
        final_sessions = set(final_groups)
        tainted = sorted(f for f in normal if session_of(f) in dev_sessions)
        final_normal = sorted(f for f in normal if session_of(f) in final_sessions)
        clean = [f for f in normal if session_of(f) not in dev_sessions | final_sessions]
        clean_sessions = sorted({session_of(f) for f in clean})
        out = []
        for held_session in clean_sessions:
            val = [f for f in clean if session_of(f) == held_session]
            train = [f for f in clean if session_of(f) != held_session]
            if not train:
                continue
            out.append({
                "held_out_session": held_session,
                "train": sorted(train),
                "val": sorted(val),
                "test_normal": tainted,
                "test_anomalous": anomalous,
                "final_holdout_normal": final_normal,
                "final_holdout_anomalous": final_anomalous,
            })
        return out

    out = []
    for held_out in normal:
        out.append({
            "held_out_flight": held_out,
            "train": [f for f in normal if f != held_out],
            "val": [held_out],
            "test_anomalous": anomalous,
            "final_holdout_normal": [],
            "final_holdout_anomalous": final_anomalous,
        })
    return out


# Kaynak basina (n_val, n_test_normal) kotalari -- PIPELINE karari (GPT/FableChat):
# ALFA 10 normal ucus -> 6 train / 2 val / 2 test-normal; UAV Attack 6 benign ->
# 4/1/1; UAV-SEAD (ML-4 buyutmesi sonrasi ~60 normal) -> 40/10/10.
# ML-14 yenilemesinde uav_sead icin bu deger rebuild oncesi
# development-normal sayisindan n_val=n_test_normal=max(30, round(0.15 * dev_normal))
# olarak guncellenir ve rebuild_report.json'a yazilir.
# Listede olmayan kaynak icin (1, 1).
SPLIT_QUOTAS: dict[str, tuple[int, int]] = {
    "alfa": (2, 2),
    "uav_attack": (1, 1),
    "uav_sead": (10, 10),
}

# Oturum-bazli split kullanan kaynaklar (bkz. session_of): UAV-SEAD ucus id'leri
# "tarih/saat" formatinda -- ayni gunun ucuslari tek split tarafinda kalmali.
SESSION_SPLIT_SOURCES = {"uav_sead"}
FINAL_HOLDOUT_FRACTIONS: dict[str, float] = {"uav_sead": 0.30}


def build_split_manifest(feature_tables: dict[str, pd.DataFrame], *,
                         quotas: dict[str, tuple[int, int]] | None = None,
                         frozen_holdout: dict[str, Any] | None = None) -> dict:
    """Kaynak basina seed'li split'ler + LOFO listesi iceren manifest."""
    quotas = quotas or SPLIT_QUOTAS
    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": N_SEEDS,
        "split_unit": "source-specific; see sources.<name>.split_unit",
        "train_policy": "normal-only (novelty detection)",
        "sources": {},
    }
    for source, df in feature_tables.items():
        flights = flight_label_table(df)
        n_val, n_test_normal = quotas.get(source, (1, 1))
        by_session = source in SESSION_SPLIT_SOURCES
        holdout_fraction = FINAL_HOLDOUT_FRACTIONS.get(source, 0.0)
        frozen_spec = (frozen_holdout or {}).get(source)
        seeds = [make_group_split(flights, seed=s, n_val=n_val, n_test_normal=n_test_normal,
                                  by_session=by_session,
                                  final_holdout_fraction=holdout_fraction,
                                  frozen_holdout=frozen_spec)
                 for s in range(N_SEEDS)]
        manifest["sources"][source] = {
            "n_flights": len(flights),
            "split_unit": "session" if by_session else "source_id",
            "evaluation_status": ("blind-final-holdout"
                                  if holdout_fraction else "development-only; anomaly data too scarce"),
            "flight_labels": flights.set_index("source_id")["flight_label"].to_dict(),
            "splits": {f"split_{s['seed']:02d}": s for s in seeds},
            "lofo": make_lofo_splits(
                flights, by_session=by_session,
                final_holdout_fraction=holdout_fraction,
                frozen_holdout=frozen_spec),
        }
        n_normal = int((flights["flight_label"] == "normal").sum())
        logger.info("%s: %d ucus (%d normal), %d seed split + %d %s fold",
                    source, len(flights), n_normal, N_SEEDS,
                    len(manifest["sources"][source]["lofo"]),
                    "LOSO" if by_session else "LOFO")
    return manifest


def write_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Split manifest yazildi: %s", path)


def _partition_supervised_groups(groups: list[str], *, seed: int) -> tuple[set[str], set[str], set[str]]:
    """Deterministically partition one label stratum 60/20/20 by group."""

    if len(groups) < 3:
        raise ValueError("Supervised train/val/test icin her sinifta en az 3 grup gerekli")
    order = np.asarray(sorted(groups), dtype=object)
    np.random.default_rng(seed).shuffle(order)
    n_test = max(1, round(len(order) * 0.20))
    n_val = max(1, round(len(order) * 0.20))
    if n_test + n_val >= len(order):
        n_test = n_val = 1
    test = set(order[:n_test].tolist())
    val = set(order[n_test:n_test + n_val].tolist())
    train = set(order[n_test + n_val:].tolist())
    return train, val, test


def add_supervised_splits(manifest: dict, *, sources: tuple[str, ...] = ("alfa", "uav_sead")) -> dict:
    """Add ML-8A binary supervised folds without changing novelty folds.

    The original manifest was intentionally normal-only for novelty detection,
    so it cannot train a supervised classifier.  ML-8A folds partition normal
    and anomalous development groups separately, preserve session isolation,
    and exclude the fixed final holdout and exploration sets completely.
    """

    out = deepcopy(manifest)
    for source in sources:
        config = out["sources"][source]
        labels: dict[str, str] = config["flight_labels"]
        reference = config["splits"]["split_00"]
        final_holdout = set(reference.get("final_holdout", []))
        exploration = set(reference.get("exploration", []))
        development = set(labels) - final_holdout - exploration
        group_of = session_of if config.get("split_unit") == "session" else (lambda value: value)

        anomalous_groups = {
            group_of(sid) for sid in development if labels[sid] not in NORMAL_LABELS
        }
        all_groups = {group_of(sid) for sid in development}
        normal_only_groups = all_groups - anomalous_groups

        folds: dict[str, dict] = {}
        for seed in range(int(out.get("n_seeds", N_SEEDS))):
            an_train, an_val, an_test = _partition_supervised_groups(
                sorted(anomalous_groups), seed=seed
            )
            no_train, no_val, no_test = _partition_supervised_groups(
                sorted(normal_only_groups), seed=10_000 + seed
            )
            group_sets = {
                "train": an_train | no_train,
                "val": an_val | no_val,
                "test": an_test | no_test,
            }
            fold: dict[str, object] = {
                "seed": seed,
                "split_unit": config.get("split_unit", "source_id"),
                "policy": "stratified 60/20/20 development groups; final_holdout excluded",
                "final_holdout": sorted(final_holdout),
            }
            for part, groups in group_sets.items():
                ids = sorted(sid for sid in development if group_of(sid) in groups)
                fold[part] = ids
                fold[f"{part}_normal"] = sorted(sid for sid in ids if labels[sid] in NORMAL_LABELS)
                fold[f"{part}_anomalous"] = sorted(sid for sid in ids if labels[sid] not in NORMAL_LABELS)
            folds[f"split_{seed:02d}"] = fold
        config["supervised_splits"] = folds
        config["supervised_split_note"] = (
            "ML-8A override: original splits are normal-only novelty folds; "
            "these folds are group-isolated and never include final_holdout."
        )
    return out


def assert_no_flight_overlap(split: dict) -> None:
    """Ayni ucusun iki sete dusmedigini dogrular (ML-0 kabul kriteri)."""
    sets = {k: set(split[k]) for k in ("train", "val", "test", "final_holdout", "exploration")
            if k in split}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(f"Ucus sizintisi: {a} ve {b} kesisiyor: {sorted(overlap)}")
