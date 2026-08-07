"""Evaluation primitives with explicit window/event/flight units.

The functions in this module deliberately do not perform threshold selection.  A
caller supplies an already frozen alarm decision and receives diagnostics whose
denominators are explicit.  In particular, synthetic detection results are
paired with a burden measured on the single, unmodified clean reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class EpisodeContract:
    """Frozen alarm-event semantics.

    Window scores become observable at ``t_end``.  Consecutive emissions for the
    same flight are one operator-facing episode when their separation is no more
    than ``merge_gap_s``.
    """

    merge_gap_s: float = 60.0
    emission_time_col: str = "t_end"


def diagnostic_window_metrics(q_w: np.ndarray, scores: np.ndarray) -> dict:
    """Return primary ``q>0`` and secondary steady-state window diagnostics.

    Mixed-support windows (``0 < q < 1``) are positive in the primary metric and
    reported, not silently converted to negatives, in the secondary metric.
    Non-finite truth or scores are unscoreable and excluded with counts retained.
    """

    q = np.asarray(q_w, dtype=float)
    score = np.asarray(scores, dtype=float)
    if q.shape != score.shape:
        raise ValueError("q_w and scores must have identical shapes")

    finite = np.isfinite(q) & np.isfinite(score)
    qf, sf = q[finite], score[finite]
    qf_clipped = np.clip(qf, 0.0, 1.0)
    if not np.array_equal(qf, qf_clipped):
        raise ValueError('q_w outside unit interval')
    primary_y = qf > 0.0
    steady = (qf == 0.0) | (qf == 1.0)

    def _binary(y: np.ndarray, s: np.ndarray) -> dict:
        n_pos = int(np.count_nonzero(y))
        n_neg = int(len(y) - n_pos)
        return {
            "n": int(len(y)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "auroc": float(roc_auc_score(y, s)) if n_pos and n_neg else None,
            "auprc": float(average_precision_score(y, s)) if n_pos and n_neg else None,
        }

    return {
        "unit": "window",
        "n_input": int(len(q)),
        "n_unscoreable": int(len(q) - np.count_nonzero(finite)),
        "q_strata": {
            "q_eq_0": int(np.count_nonzero(qf == 0.0)),
            "q_mixed": int(np.count_nonzero((qf > 0.0) & (qf < 1.0))),
            "q_eq_1": int(np.count_nonzero(qf == 1.0)),
        },
        "primary_y_any": _binary(primary_y, sf),
        "secondary_steady_state": _binary(qf[steady] == 1.0, sf[steady]),
    }


def sampled_roc(q_w: np.ndarray, scores: np.ndarray, *, max_points: int = 201) -> dict:
    '''Return a bounded-size ROC diagnostic for y_any = 1[q_w > 0].

    Exact AUROC remains in diagnostic_window_metrics; sampling only bounds JSON
    size and never feeds a decision or threshold selection.
    '''

    if max_points < 2:
        raise ValueError('max_points must be at least 2')
    q = np.asarray(q_w, dtype=float)
    score = np.asarray(scores, dtype=float)
    if q.shape != score.shape:
        raise ValueError('q_w and scores must have identical shapes')
    finite = np.isfinite(q) & np.isfinite(score)
    q, score = q[finite], score[finite]
    q_clipped = np.clip(q, 0.0, 1.0)
    if not np.array_equal(q, q_clipped):
        raise ValueError('q_w outside unit interval')
    labels = q > 0.0
    if not np.any(labels) or np.all(labels):
        return {'unit': 'window', 'n_exact_points': 0, 'fpr': [], 'tpr': []}
    fpr, tpr, _ = roc_curve(labels, score)
    if len(fpr) <= max_points:
        indices = np.arange(len(fpr))
    else:
        indices = np.unique(np.linspace(0, len(fpr) - 1, max_points, dtype=int))
    return {
        'unit': 'window',
        'n_exact_points': int(len(fpr)),
        'fpr': fpr[indices].astype(float).tolist(),
        'tpr': tpr[indices].astype(float).tolist(),
    }


def _merged_intervals(intervals: list[tuple[float, float]], *, gap_s: float = 0.0):
    clean = sorted((float(a), float(b)) for a, b in intervals if np.isfinite(a) and np.isfinite(b) and b >= a)
    if not clean:
        return []
    merged: list[list[float]] = [[clean[0][0], clean[0][1]]]
    for start, end in clean[1:]:
        if start <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def scoreable_exposure(meta: pd.DataFrame, *, flight_id_col: str = "flight_id") -> dict:
    """Union overlapping score-support intervals; never count window overlap twice."""

    required = {flight_id_col, "t_start", "t_end"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"missing exposure columns: {sorted(missing)}")

    seconds_by_flight: dict[str, float] = {}
    for flight_id, group in meta.groupby(flight_id_col, sort=False, dropna=False):
        merged = _merged_intervals(list(group[["t_start", "t_end"]].itertuples(index=False, name=None)))
        seconds_by_flight[str(flight_id)] = float(sum(end - start for start, end in merged))
    total_seconds = float(sum(seconds_by_flight.values()))
    return {
        "unit": "scoreable_flight_time",
        "n_scoreable_flights": int(len(seconds_by_flight)),
        "scoreable_seconds": total_seconds,
        "scoreable_flight_hours": total_seconds / 3600.0,
        "seconds_by_flight": seconds_by_flight,
    }


def alarm_episodes(
    meta: pd.DataFrame,
    alarm: np.ndarray,
    *,
    contract: EpisodeContract = EpisodeContract(),
    flight_id_col: str = "flight_id",
) -> pd.DataFrame:
    """Collapse true alarm emissions into operator-facing flight-local episodes."""

    flags = np.asarray(alarm, dtype=bool)
    if len(flags) != len(meta):
        raise ValueError("alarm and meta must have identical lengths")
    if contract.emission_time_col not in meta or flight_id_col not in meta:
        raise KeyError("metadata lacks flight or emission-time column")

    active = meta.loc[flags, [flight_id_col, contract.emission_time_col]].copy()
    rows: list[dict] = []
    for flight_id, group in active.groupby(flight_id_col, sort=False, dropna=False):
        times = np.sort(group[contract.emission_time_col].to_numpy(dtype=float))
        times = times[np.isfinite(times)]
        if not len(times):
            continue
        start = end = float(times[0])
        n_emissions = 1
        for value in times[1:]:
            value = float(value)
            if value - end <= contract.merge_gap_s:
                end = value
                n_emissions += 1
            else:
                rows.append({"flight_id": flight_id, "episode_start": start, "episode_end": end, "n_emissions": n_emissions})
                start = end = value
                n_emissions = 1
        rows.append({"flight_id": flight_id, "episode_start": start, "episode_end": end, "n_emissions": n_emissions})
    return pd.DataFrame(rows, columns=["flight_id", "episode_start", "episode_end", "n_emissions"])


def natural_alert_burden(
    meta: pd.DataFrame,
    alarm: np.ndarray,
    *,
    contract: EpisodeContract = EpisodeContract(),
    flight_id_col: str = "flight_id",
) -> dict:
    """Nominal alert burden on one unmodified reference; this is not labelled FP."""

    exposure = scoreable_exposure(meta, flight_id_col=flight_id_col)
    episodes = alarm_episodes(meta, alarm, contract=contract, flight_id_col=flight_id_col)
    n_episodes = int(len(episodes))
    hours = float(exposure["scoreable_flight_hours"])
    alerted_flights = int(episodes["flight_id"].nunique()) if n_episodes else 0
    n_flights = int(exposure["n_scoreable_flights"])
    return {
        "unit": "natural_alert_burden",
        "episode_contract": {
            "emission_time": contract.emission_time_col,
            "merge_gap_s": contract.merge_gap_s,
        },
        "n_alert_episodes": n_episodes,
        "n_scoreable_flights": n_flights,
        "n_alerted_flights": alerted_flights,
        "alerted_flight_fraction": (alerted_flights / n_flights) if n_flights else None,
        "scoreable_flight_hours": hours,
        "alert_episodes_per_scoreable_flight_hour": (n_episodes / hours) if hours > 0 else None,
    }


def truth_event_table(
    truth_rows: pd.DataFrame,
    *,
    flight_id_col: str = 'flight_id',
) -> pd.DataFrame:
    '''Collapse row truth-v2 into one explicit observability row per event.

    attack_eligible and observable_eligible are deliberately distinct. An
    injected event can have evaluable active rows yet make no observable change;
    it remains reported but cannot enter a detection-recall denominator.
    '''

    required = {
        'event_id',
        'event_type',
        flight_id_col,
        'attack_onset',
        'observable_onset',
        'event_end',
        'injection_active',
        'observable_changed',
        'evaluable_truth',
    }
    missing = required - set(truth_rows.columns)
    if missing:
        raise KeyError(f'missing truth-v2 event columns: {sorted(missing)}')

    rows: list[dict] = []
    event_rows = truth_rows.loc[truth_rows['event_id'].notna()]
    for event_id, group in event_rows.groupby('event_id', sort=False, dropna=False):
        scalar_columns = [
            'event_type',
            flight_id_col,
            'attack_onset',
            'observable_onset',
            'event_end',
        ]
        scalars: dict[str, object] = {}
        for column in scalar_columns:
            values = group[column].dropna().unique()
            if len(values) > 1:
                raise ValueError(f'event {event_id!r} has inconsistent {column}')
            scalars[column] = values[0] if len(values) else None

        active = group['injection_active'].fillna(False).to_numpy(dtype=bool)
        changed = group['observable_changed'].fillna(False).to_numpy(dtype=bool)
        evaluable = group['evaluable_truth'].fillna(False).to_numpy(dtype=bool)
        attack_active_rows = int(active.sum())
        evaluable_active_rows = int((active & evaluable).sum())
        observable_changed_rows = int((changed & evaluable).sum())
        attack_eligible = attack_active_rows > 0 and evaluable_active_rows > 0
        observable_eligible = (
            attack_eligible
            and observable_changed_rows > 0
            and scalars['observable_onset'] is not None
            and scalars['event_end'] is not None
        )
        rows.append(
            {
                'event_id': event_id,
                'event_type': scalars['event_type'],
                'flight_id': scalars[flight_id_col],
                'attack_onset': scalars['attack_onset'],
                'observable_onset': scalars['observable_onset'],
                'event_end': scalars['event_end'],
                'attack_active_rows': attack_active_rows,
                'evaluable_active_rows': evaluable_active_rows,
                'observable_changed_rows': observable_changed_rows,
                'attack_eligible': bool(attack_eligible),
                'observable_eligible': bool(observable_eligible),
                'active_but_unobservable': bool(attack_eligible and not observable_eligible),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            'event_id',
            'event_type',
            'flight_id',
            'attack_onset',
            'observable_onset',
            'event_end',
            'attack_active_rows',
            'evaluable_active_rows',
            'observable_changed_rows',
            'attack_eligible',
            'observable_eligible',
            'active_but_unobservable',
        ],
    )


def event_observability_denominators(events: pd.DataFrame) -> dict:
    '''Report injected/evaluable/observable event counts without conflation.'''

    required = {'attack_eligible', 'observable_eligible', 'active_but_unobservable'}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f'missing event eligibility columns: {sorted(missing)}')
    attack = events['attack_eligible'].fillna(False).to_numpy(dtype=bool)
    observable = events['observable_eligible'].fillna(False).to_numpy(dtype=bool)
    active_unobservable = events['active_but_unobservable'].fillna(False).to_numpy(dtype=bool)
    return {
        'unit': 'event',
        'n_declared_events': int(len(events)),
        'n_attack_eligible_events': int(attack.sum()),
        'n_observable_eligible_events': int(observable.sum()),
        'n_active_but_unobservable_events': int(active_unobservable.sum()),
        'detection_recall_denominator': 'observable_eligible_events',
    }


def active_interval_coverage(
    events: pd.DataFrame,
    alarm_meta: pd.DataFrame,
    alarm: np.ndarray,
    *,
    onset_col: str = 'observable_onset',
    event_end_col: str = 'event_end',
    flight_id_col: str = 'flight_id',
) -> dict:
    '''Measure alerted-window support time inside each observable event.

    Only actual alerted-window support intervals are intersected with the event;
    one hit never fills the rest of an event. This descriptive support-time
    measure does not replace the causal t_end first-alarm delay.
    '''

    required_events = {'event_id', flight_id_col, onset_col, event_end_col}
    required_meta = {flight_id_col, 't_start', 't_end'}
    missing_events = required_events - set(events.columns)
    missing_meta = required_meta - set(alarm_meta.columns)
    if missing_events:
        raise KeyError(f'missing event columns: {sorted(missing_events)}')
    if missing_meta:
        raise KeyError(f'missing window support columns: {sorted(missing_meta)}')
    flags = np.asarray(alarm, dtype=bool)
    if len(flags) != len(alarm_meta):
        raise ValueError('alarm and alarm_meta must have identical lengths')

    alerted = alarm_meta.loc[flags, [flight_id_col, 't_start', 't_end']]
    alerted_by_flight = {
        flight_id: list(group[['t_start', 't_end']].itertuples(index=False, name=None))
        for flight_id, group in alerted.groupby(flight_id_col, sort=False)
    }
    per_event: list[dict] = []
    total_covered = total_duration = 0.0
    finite_fractions: list[float] = []
    for event in events.itertuples(index=False):
        row = event._asdict()
        onset = float(row[onset_col])
        end = float(row[event_end_col])
        duration = max(0.0, end - onset)
        intervals = []
        for start, stop in alerted_by_flight.get(row[flight_id_col], ()):
            start = max(float(start), onset)
            stop = min(float(stop), end)
            if stop >= start:
                intervals.append((start, stop))
        covered = float(sum(stop - start for start, stop in _merged_intervals(intervals)))
        fraction = covered / duration if duration > 0.0 else None
        total_covered += covered
        total_duration += duration
        if fraction is not None:
            finite_fractions.append(fraction)
        per_event.append(
            {
                'event_id': row['event_id'],
                'flight_id': row[flight_id_col],
                'observable_active_duration_s': duration,
                'alerted_window_support_seconds': covered,
                'alerted_window_support_fraction': fraction,
            }
        )

    return {
        'unit': 'alerted_window_support_time_within_observable_event',
        'n_events': int(len(events)),
        'n_positive_duration_events': int(len(finite_fractions)),
        'macro_mean_fraction': float(np.mean(finite_fractions)) if finite_fractions else None,
        'micro_fraction': total_covered / total_duration if total_duration > 0.0 else None,
        'total_observable_active_duration_s': total_duration,
        'total_alerted_window_support_seconds': total_covered,
        'point_adjustment': False,
        'per_event': per_event,
    }


def event_detection_metrics(
    events: pd.DataFrame,
    alarm_meta: pd.DataFrame,
    alarm: np.ndarray,
    *,
    onset_col: str = "observable_onset",
    event_end_col: str = "event_end",
    emission_time_col: str = "t_end",
    flight_id_col: str = "flight_id",
) -> dict:
    """Event-macro recall and causal first-alarm delay without point adjustment."""

    required = {"event_id", flight_id_col, onset_col, event_end_col}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"missing event columns: {sorted(missing)}")
    flags = np.asarray(alarm, dtype=bool)
    if len(flags) != len(alarm_meta):
        raise ValueError("alarm and alarm_meta must have identical lengths")

    emissions = alarm_meta.loc[flags, [flight_id_col, emission_time_col]].copy()
    emissions_by_flight = {
        flight_id: np.sort(group[emission_time_col].to_numpy(dtype=float))
        for flight_id, group in emissions.groupby(flight_id_col, sort=False)
    }
    detected = 0
    delays: list[float] = []
    per_event: list[dict] = []
    for event in events.itertuples(index=False):
        row = event._asdict()
        onset, end = float(row[onset_col]), float(row[event_end_col])
        flight_emissions = emissions_by_flight.get(row[flight_id_col], np.empty(0))
        left = int(np.searchsorted(flight_emissions, onset, side='left'))
        hit = bool(left < len(flight_emissions) and flight_emissions[left] <= end)
        delay = float(flight_emissions[left] - onset) if hit else None
        if hit:
            detected += 1
            delays.append(delay)
        per_event.append({"event_id": row["event_id"], "detected": hit, "first_alarm_delay_s": delay})

    n_events = int(len(events))
    return {
        "unit": "event",
        "n_events": n_events,
        "n_detected_events": detected,
        "event_recall": (detected / n_events) if n_events else None,
        "first_alarm_delay_s": {
            "median": float(np.median(delays)) if delays else None,
            "p95": float(np.quantile(delays, 0.95)) if delays else None,
            "n_detected": int(len(delays)),
        },
        "per_event": per_event,
    }
