"""Nedensel alarm persistence ve event-duzeyi anomaly metrikleri.

Ucus ROC tek basina alarm kalitesini anlatmaz. Bu modul, satir/pencere skoru
uzerinden canli kullanima uygun K-of-N alarmi ve anomaly-event yakalama,
gecikme, kapsama ile saat basina yanlis alarm metriklerini uretir.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_uav_sead_ranges(labels_path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Load UAV-SEAD ranges as absolute PX4-microsecond intervals.

    ML-6 event evaluation and ML-8A window labeling share this one parser.
    """

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    result: dict[str, list[tuple[float, float]]] = {}
    for flight, meta in labels.items():
        spans: list[tuple[float, float]] = []
        for annotation in meta.get("ranges", []):
            for _, intervals in annotation:
                spans.extend((float(a), float(b)) for a, b in intervals)
        result[flight] = spans
    return result


def load_uav_sead_ranges_by_category(
        labels_path: str | Path,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Load ``flight -> annotation category -> absolute-us intervals``.

    Kategori etiketi kaybolmadan korunur; eski ``load_uav_sead_ranges``
    davranisi degismez ve tum kategorileri havuzlamaya devam eder.
    """
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    result: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for flight, meta in labels.items():
        categories: dict[str, list[tuple[float, float]]] = {}
        for annotation in meta.get("ranges", []):
            for category, intervals in annotation:
                categories.setdefault(str(category), []).extend(
                    (float(start), float(end)) for start, end in intervals
                )
        result[flight] = categories
    return result


def uav_sead_absolute_us(t_rel_s, t0_us: float) -> np.ndarray:
    """Reconstruct absolute PX4 time using the established ML-6 mapping."""

    return float(t0_us) + np.asarray(t_rel_s, dtype=float) * 1e6


def range_mask(values, intervals) -> np.ndarray:
    """Mark values inside any inclusive interval."""

    values = np.asarray(values, dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    for start, end in intervals:
        mask |= (values >= float(start)) & (values <= float(end))
    return mask


def k_of_n_alarm(scores, threshold: float, *, k: int = 1, n: int = 1) -> np.ndarray:
    """Son N skorun en az K'si esik ustundeyse alarm ver (yalnizca gecmise bakar)."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores tek boyutlu olmali")
    if n < 1 or k < 1 or k > n:
        raise ValueError("1 <= k <= n olmali")
    exceed = np.isfinite(scores) & (scores > float(threshold))
    counts = np.convolve(exceed.astype(int), np.ones(n, dtype=int), mode="full")[:len(exceed)]
    # Ilk N-1 noktada da o ana kadar elde olan gecmis kullanilir; gelecek yoktur.
    return counts >= k


def persistent_alarm(t_s, scores, threshold: float, *, k: int = 1, n: int = 1,
                     clear_s: float = 0.0,
                     cooldown_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """K-of-N tetigini nedensel latch/cooldown alarm durumuna cevir.

    ``clear_s`` kadar kesintisiz sakinlik gorulmeden aktif alarm kapanmaz.
    Kapandiktan sonra ``cooldown_s`` boyunca yeni bildirim uretilmez. Donen
    ikinci maske yalniz yeni alarm baslangiclarini (operator bildirimlerini)
    isaretler; boylece tek fiziksel sapma alarm firtinasina donusmez.
    """
    t = np.asarray(t_s, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if t.ndim != 1 or scores.ndim != 1 or len(t) != len(scores):
        raise ValueError("t_s ve scores esit uzunlukta tek boyutlu olmali")
    if len(t) and (not np.isfinite(t).all() or np.any(np.diff(t) < 0)):
        raise ValueError("t_s sonlu ve artan sirada olmali")
    if clear_s < 0 or cooldown_s < 0:
        raise ValueError("clear_s ve cooldown_s negatif olamaz")

    trigger = k_of_n_alarm(scores, threshold, k=k, n=n)
    alarm = np.zeros(len(trigger), dtype=bool)
    onsets = np.zeros(len(trigger), dtype=bool)
    active = False
    below_since: float | None = None
    cooldown_until = -np.inf

    for i, (now, is_triggered) in enumerate(zip(t, trigger)):
        if active:
            if is_triggered:
                below_since = None
            elif clear_s == 0:
                active = False
                cooldown_until = now + cooldown_s
            else:
                if below_since is None:
                    below_since = now
                if now - below_since >= clear_s:
                    active = False
                    cooldown_until = now + cooldown_s
                    below_since = None

        if not active and is_triggered and now >= cooldown_until:
            active = True
            below_since = None
            onsets[i] = True
        alarm[i] = active

    return alarm, onsets


def _episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Boolean maskeyi inclusive (start, end) episode indekslerine cevir."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(t_s, y_true, scores, threshold: float, *,
                  k: int = 1, n: int = 1, clear_s: float = 0.0,
                  cooldown_s: float = 0.0,
                  max_gap_s: float | None = None) -> dict[str, float | int]:
    """Tek ucus icin event recall/gecikme/kapsama/yanlis-alarm metrikleri.

    Event, ard arda gelen ``y_true=True`` orneklerinden olusur. Ana recall,
    event araliginda yeni alarm baslangici uretilmesini ister. Onceden acik
    alarm ile yalniz overlap ayri metrikte tutulur. Yanlis alarm, normal anda
    uretilen yeni operator bildirimidir.
    """
    t = np.asarray(t_s, dtype=float)
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if not (t.ndim == y.ndim == s.ndim == 1 and len(t) == len(y) == len(s)):
        raise ValueError("t_s, y_true ve scores esit uzunlukta tek boyutlu olmali")
    if len(t) and np.any(np.diff(t) < 0):
        raise ValueError("t_s artan sirada olmali")
    if max_gap_s is not None and max_gap_s <= 0:
        raise ValueError("max_gap_s pozitif olmali")

    alarm, alarm_onsets = persistent_alarm(
        t, s, threshold, k=k, n=n,
        clear_s=clear_s, cooldown_s=cooldown_s)
    true_events = _episodes(y)

    delays: list[float] = []
    detected = 0
    overlap_detected = 0
    preexisting_alarm_events = 0
    for start, end in true_events:
        # Event baslamadan once verilmis bir alarmi basari sayma: gercek bir
        # tespit, event araliginda yeni bir operator bildirimi uretmelidir.
        hit = np.flatnonzero(alarm_onsets[start:end + 1])
        overlaps = bool(alarm[start:end + 1].any())
        overlap_detected += int(overlaps)
        if len(hit):
            detected += 1
            delays.append(float(t[start + hit[0]] - t[start]))
        elif overlaps:
            preexisting_alarm_events += 1

    false_alarm_events = int((alarm_onsets & ~y).sum())
    if len(t) > 1:
        dt = np.diff(t)
        # Bir intervali baslangic orneginin durumu ile iliskilendir.
        valid_exposure = ~y[:-1]
        if max_gap_s is not None:
            valid_exposure &= dt <= max_gap_s
        normal_hours = float(dt[valid_exposure].sum() / 3600.0)
    else:
        normal_hours = 0.0

    return {
        "n_events": len(true_events),
        "detected_events": detected,
        "event_recall": float(detected / len(true_events)) if true_events else np.nan,
        "overlap_detected_events": overlap_detected,
        "event_overlap_recall": (float(overlap_detected / len(true_events))
                                 if true_events else np.nan),
        "preexisting_alarm_events": preexisting_alarm_events,
        "mean_detection_delay_s": float(np.mean(delays)) if delays else np.nan,
        "max_detection_delay_s": float(np.max(delays)) if delays else np.nan,
        "detection_delays_s": delays,
        "anomaly_coverage": float(alarm[y].mean()) if y.any() else np.nan,
        "false_alarm_events": false_alarm_events,
        "alarm_onsets": int(alarm_onsets.sum()),
        "normal_hours": normal_hours,
        "false_alarms_per_hour": (float(false_alarm_events / normal_hours)
                                  if normal_hours > 0 else np.nan),
        "alarm_fraction": float(alarm.mean()) if len(alarm) else np.nan,
    }
