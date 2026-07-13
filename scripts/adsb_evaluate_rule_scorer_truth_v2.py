'''Corrected truth-v2 rescore of the already frozen ADS-B residual rule.

This step changes truth and evaluation only. It never fits the scorer, changes
its equation, or selects a threshold. Synthetic files are explicit evaluation
inputs and can never acquire a fit/calibration role in the run manifest.
'''

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.diagnostics import z_score_confidence  # noqa: E402
from adsb.evaluation import (  # noqa: E402
    EpisodeContract,
    active_interval_coverage,
    diagnostic_window_metrics,
    event_detection_metrics,
    event_observability_denominators,
    natural_alert_burden,
    sampled_roc,
    truth_event_table,
)
from adsb.features import build_feature_table  # noqa: E402
from adsb.rules import CAP, Z0, ResidualRuleScorer  # noqa: E402
from adsb.run_manifest import (  # noqa: E402
    InputSpec,
    create_immutable_run_manifest,
    sha256_file,
    sha256_json,
)
from adsb.synthetic import PHYSICS_BREAK_RECIPES  # noqa: E402
from adsb.windowing import build_windows  # noqa: E402


DEFAULT_CORPUS = Path('data/objectstore/synthetic/adsb_v2_20260713_01')
DEFAULT_FROZEN_REPORT = Path('artifacts/adsb/models/rule_scorer_report.json')
DEFAULT_HISTORICAL_NN_REPORT = Path('artifacts/adsb/models/baseline_training_report.json')
WINDOW = 12
STRIDE = 6
MAX_GAP_S = 60.0
CONFIDENCE_THRESHOLD = 0.95
ROC_MAX_POINTS = 201

SOURCE_COLUMNS = [
    'flight_id',
    'timestamp_utc',
    'lat',
    'lon',
    'alt',
    'alt_geom_m',
    'ground_speed_ms',
    'track_deg',
    'vertical_rate_ms',
    'roll_deg',
    'event_id',
    'event_type',
    'attack_onset',
    'observable_onset',
    'event_end',
    'injection_active',
    'observable_changed',
    'evaluable_truth',
]


@dataclass(frozen=True)
class ScoredTruthFile:
    scores: np.ndarray
    meta: pd.DataFrame
    events: pd.DataFrame


def _finite_or_none(value):
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _write_json_exclusive(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(
            _finite_or_none(value),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write('\n')


def _required_columns(path: Path, requested: Iterable[str]) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    required = set(requested)
    missing = required - available
    if missing:
        raise KeyError(f'{path}: missing columns {sorted(missing)}')
    return list(requested)


def iter_parquet_flight_batches(path: Path, columns: Sequence[str]):
    '''Yield bounded batches containing only complete contiguous flights.'''

    parquet = pq.ParquetFile(path)
    selected = _required_columns(path, columns)
    pending: pd.DataFrame | None = None
    seen: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=selected).to_pandas()
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
            pending = None
        if frame.empty:
            continue
        if frame['flight_id'].isna().any():
            raise ValueError(f'{path}: null flight_id')
        block = frame['flight_id'].ne(frame['flight_id'].shift()).cumsum()
        block_ids = frame.groupby(block, sort=False)['flight_id'].first().astype(str).tolist()
        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f'{path}: flight rows are not contiguous')
        last_id = frame['flight_id'].iloc[-1]
        pending = frame.loc[frame['flight_id'] == last_id].copy()
        complete = frame.loc[frame['flight_id'] != last_id]
        complete_ids = set(complete['flight_id'].astype(str).unique())
        repeated = sorted(complete_ids & seen)
        if repeated:
            raise ValueError(f'{path}: flight {repeated[0]!r} crosses a non-adjacent block')
        seen.update(complete_ids)
        if not complete.empty:
            yield complete.reset_index(drop=True)
    if pending is not None:
        key = str(pending['flight_id'].iloc[0])
        if key in seen:
            raise ValueError(f'{path}: flight {key!r} repeated at EOF')
        yield pending.reset_index(drop=True)


def iter_parquet_flights(path: Path, columns: Sequence[str]):
    '''Yield individual flights for exact ID contracts and small consumers.'''

    for batch in iter_parquet_flight_batches(path, columns):
        for _, flight in batch.groupby('flight_id', sort=False):
            yield flight.reset_index(drop=True)


def read_flight_ids(path: Path) -> list[str]:
    return sorted(
        str(flight['flight_id'].iloc[0])
        for flight in iter_parquet_flights(path, ['flight_id'])
    )


def score_truth_file(
    path: Path,
    scorer: ResidualRuleScorer,
    *,
    window: int,
    stride: int,
    max_gap_s: float,
) -> ScoredTruthFile:
    score_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    for flight_batch in iter_parquet_flight_batches(path, SOURCE_COLUMNS):
        flight_batch = flight_batch.sort_values(
            ['flight_id', 'timestamp_utc'], kind='stable'
        ).reset_index(drop=True)
        events = truth_event_table(flight_batch)
        if not events.empty:
            event_parts.append(events)
        features = build_feature_table(flight_batch)
        features['rule_penalty'] = scorer.row_penalties(features)
        x_values, masks, meta = build_windows(
            features,
            ['rule_penalty'],
            window=window,
            stride=stride,
            max_gap_s=max_gap_s,
            truth_architecture='rule',
        )
        if len(meta):
            denominator = np.clip(masks.sum(axis=(1, 2)), 1.0, None)
            scores = (x_values * masks).sum(axis=(1, 2)) / denominator
            score_parts.append(np.asarray(scores, dtype=float))
            meta_parts.append(meta)

    if score_parts:
        scores = np.concatenate(score_parts)
        meta = pd.concat(meta_parts, ignore_index=True)
    else:
        scores = np.zeros(0, dtype=float)
        meta = pd.DataFrame(
            columns=[
                'flight_id',
                't_start',
                't_end',
                'q_w',
                'truth_scoreable',
                'y_any',
                'steady_subset',
                'steady_label',
                'history_contaminated',
            ]
        )
    events = pd.concat(event_parts, ignore_index=True) if event_parts else truth_event_table(
        pd.DataFrame(columns=SOURCE_COLUMNS)
    )
    if not events.empty and events['event_id'].duplicated().any():
        raise ValueError(f'{path}: duplicate event_id across flights')
    if len(scores) != len(meta) or not np.isfinite(scores).all():
        raise ValueError(f'{path}: invalid score output')
    return ScoredTruthFile(scores=scores, meta=meta, events=events)


def load_frozen_rule(path: Path) -> tuple[dict, ResidualRuleScorer, dict, float]:
    report = json.loads(path.read_text(encoding='utf-8'))
    scorer_record = report['scorer']
    if float(scorer_record['z0']) != Z0 or float(scorer_record['cap']) != CAP:
        raise ValueError('frozen scorer constants differ from ResidualRuleScorer.from_dict')
    confidence = float(report['confidence_threshold'])
    if confidence != CONFIDENCE_THRESHOLD:
        raise ValueError('frozen confidence threshold is not the approved 0.95')
    baseline = {
        'median': float(report['score_baseline_median_mad']['median']),
        'mad': float(report['score_baseline_median_mad']['mad']),
    }
    if not np.isfinite(list(baseline.values())).all() or baseline['mad'] <= 0.0:
        raise ValueError('frozen score baseline must have finite median and positive MAD')
    return report, ResidualRuleScorer.from_dict(scorer_record), baseline, confidence


def _fixed_alarms(scores: np.ndarray, baseline: dict, confidence: float) -> np.ndarray:
    return np.asarray(z_score_confidence(scores, baseline) >= confidence, dtype=bool)


def _assert_paired_windows(clean: ScoredTruthFile, corrupt: ScoredTruthFile) -> None:
    if len(clean.meta) != len(corrupt.meta):
        raise ValueError('paired clean/corrupt window counts differ')
    for column in ('flight_id', 't_start', 't_end'):
        left = clean.meta[column].reset_index(drop=True)
        right = corrupt.meta[column].reset_index(drop=True)
        if column == 'flight_id':
            equal = np.array_equal(left.astype(str).to_numpy(), right.astype(str).to_numpy())
        else:
            equal = np.array_equal(left.to_numpy(dtype=float), right.to_numpy(dtype=float))
        if not equal:
            raise ValueError(f'paired clean/corrupt window {column} differs')


def _q_strata(q_w: np.ndarray) -> dict:
    q = np.asarray(q_w, dtype=float)
    finite = np.isfinite(q)
    values = q[finite]
    if not np.array_equal(values, np.clip(values, 0.0, 1.0)):
        raise ValueError('q_w outside unit interval')
    return {
        'n_input': int(len(q)),
        'n_unscoreable': int((~finite).sum()),
        'q_eq_0': int((values == 0.0).sum()),
        'q_mixed': int(((values > 0.0) & (values < 1.0)).sum()),
        'q_eq_1': int((values == 1.0).sum()),
    }


def _score_stats(scores: np.ndarray, alarm: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=float)
    flags = np.asarray(alarm, dtype=bool)
    if len(values) != len(flags):
        raise ValueError('scores and alarms differ in length')
    if not len(values):
        return {
            'n': 0,
            'n_alarms': 0,
            'alarm_fraction': None,
            'mean_score': None,
            'median_score': None,
            'p05_score': None,
            'p95_score': None,
        }
    return {
        'n': int(len(values)),
        'n_alarms': int(flags.sum()),
        'alarm_fraction': float(flags.mean()),
        'mean_score': float(values.mean()),
        'median_score': float(np.median(values)),
        'p05_score': float(np.quantile(values, 0.05)),
        'p95_score': float(np.quantile(values, 0.95)),
    }


def corrupt_q0_timeline_sanity(
    scored: ScoredTruthFile,
    alarm: np.ndarray,
) -> dict:
    q = scored.meta['q_w'].to_numpy(dtype=float)
    scoreable = scored.meta['truth_scoreable'].fillna(False).to_numpy(dtype=bool)
    q0 = scoreable & (q == 0.0)
    onset_by_flight = scored.events.set_index('flight_id')['observable_onset'].to_dict()
    onsets = pd.to_numeric(
        scored.meta['flight_id'].map(onset_by_flight), errors='coerce'
    ).to_numpy(dtype=float)
    ends = scored.meta['t_end'].to_numpy(dtype=float)
    pre_observable = q0 & np.isfinite(onsets) & (ends < onsets)
    at_or_after = q0 & np.isfinite(onsets) & (ends >= onsets)
    no_observable_onset = q0 & ~np.isfinite(onsets)
    summary = _score_stats(scored.scores[q0], np.asarray(alarm)[q0])
    summary.update(
        {
            'unit': 'corrupt_q_eq_0_window_timeline_sanity_only',
            'included_as_auc_negative': False,
            'n_before_observable_onset': int(pre_observable.sum()),
            'n_at_or_after_observable_onset': int(at_or_after.sum()),
            'n_without_observable_event_onset': int(no_observable_onset.sum()),
        }
    )
    return summary


def event_results(
    scored: ScoredTruthFile,
    alarm: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    denominators = event_observability_denominators(scored.events)
    eligible = scored.events.loc[
        scored.events['observable_eligible'].fillna(False)
    ].copy()
    detection = event_detection_metrics(eligible, scored.meta, alarm)
    coverage = active_interval_coverage(eligible, scored.meta, alarm)
    detection_rows = pd.DataFrame(detection.pop('per_event'))
    coverage_rows = pd.DataFrame(coverage.pop('per_event'))
    details = scored.events.copy()
    if not detection_rows.empty:
        details = details.merge(
            detection_rows[['event_id', 'detected', 'first_alarm_delay_s']],
            on='event_id',
            how='left',
        )
    else:
        details['detected'] = pd.NA
        details['first_alarm_delay_s'] = np.nan
    if not coverage_rows.empty:
        details = details.merge(
            coverage_rows[
                [
                    'event_id',
                    'observable_active_duration_s',
                    'alerted_window_support_seconds',
                    'alerted_window_support_fraction',
                ]
            ],
            on='event_id',
            how='left',
        )
    else:
        details['observable_active_duration_s'] = np.nan
        details['alerted_window_support_seconds'] = np.nan
        details['alerted_window_support_fraction'] = np.nan
    return {
        'observability_denominators': denominators,
        'fixed_threshold_detection': detection,
        'active_interval_coverage': coverage,
    }, details


def evaluate_recipe(
    clean: ScoredTruthFile,
    corrupt: ScoredTruthFile,
    *,
    baseline: dict,
    confidence: float,
    clean_burden: dict,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    '''Evaluate one paired recipe without treating corrupt q=0 as negatives.'''

    _assert_paired_windows(clean, corrupt)
    q = corrupt.meta['q_w'].to_numpy(dtype=float)
    scoreable = corrupt.meta['truth_scoreable'].fillna(False).to_numpy(dtype=bool)
    positives = scoreable & np.isfinite(q) & (q > 0.0)
    clean_q = clean.meta['q_w'].to_numpy(dtype=float)
    clean_scoreable = clean.meta['truth_scoreable'].fillna(False).to_numpy(dtype=bool)
    if not clean_scoreable.all() or not np.all(clean_q == 0.0):
        raise ValueError('clean truth-v2 windows must all be scoreable q=0 negatives')

    metric_q = np.concatenate([np.zeros(len(clean.scores)), q[positives]])
    metric_scores = np.concatenate([clean.scores, corrupt.scores[positives]])
    corrupt_alarm = _fixed_alarms(corrupt.scores, baseline, confidence)
    event_summary, event_table = event_results(corrupt, corrupt_alarm)
    positive_alarm = corrupt_alarm[positives]
    window_metrics = diagnostic_window_metrics(metric_q, metric_scores)
    window_metrics['sampled_roc_y_any'] = sampled_roc(
        metric_q, metric_scores, max_points=ROC_MAX_POINTS
    )
    return {
        'window_diagnostics': window_metrics,
        'corrupt_truth_scoreable_q_strata': _q_strata(
            np.where(scoreable, q, np.nan)
        ),
        'auc_input_contract': {
            'negative_source': 'single_unmodified_clean_reference_only',
            'n_clean_negative_windows': int(len(clean.scores)),
            'positive_source': 'corrupt_truth_scoreable_q_w_gt_0_only',
            'n_corrupt_positive_windows': int(positives.sum()),
            'corrupt_q_eq_0_included_as_negative': False,
        },
        'fixed_threshold_positive_window_recall': {
            'unit': 'truth_scoreable_corrupt_q_w_gt_0_window',
            'n_positive_windows': int(positives.sum()),
            'n_alerted_positive_windows': int(positive_alarm.sum()),
            'recall': float(positive_alarm.mean()) if len(positive_alarm) else None,
        },
        'corrupt_q_eq_0_timeline_sanity': corrupt_q0_timeline_sanity(
            corrupt, corrupt_alarm
        ),
        'event_evaluation': event_summary,
        'natural_burden_same_unmodified_clean_reference': clean_burden,
    }, event_table, q[positives], corrupt.scores[positives]


def _input_record(manifest: dict, path: Path, repo_root: Path) -> dict:
    resolved = path.resolve(strict=True)
    for record in manifest['inputs']:
        recorded = Path(record['path'])
        candidate = recorded if recorded.is_absolute() else repo_root / recorded
        if candidate.resolve(strict=True) == resolved:
            return record
    raise KeyError(f'input missing from run manifest: {path}')


def run_evaluation(
    *,
    repo_root: Path,
    corpus_dir: Path,
    frozen_report_path: Path,
    historical_nn_report_path: Path,
    run_dir: Path,
    recipes: Sequence[str] = tuple(PHYSICS_BREAK_RECIPES),
    window: int = WINDOW,
    stride: int = STRIDE,
    max_gap_s: float = MAX_GAP_S,
) -> Path:
    repo_root = repo_root.resolve(strict=True)
    corpus_dir = corpus_dir.resolve(strict=True)
    frozen_report_path = frozen_report_path.resolve(strict=True)
    historical_nn_report_path = historical_nn_report_path.resolve(strict=True)
    corpus_manifest_path = (corpus_dir / 'manifest.json').resolve(strict=True)
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding='utf-8'))
    if corpus_manifest.get('schema_version') != 'adsb_synthetic_truth_v2':
        raise ValueError('corpus manifest is not adsb_synthetic_truth_v2')
    if not recipes:
        raise ValueError('at least one recipe is required')
    unknown = sorted(set(recipes) - set(PHYSICS_BREAK_RECIPES))
    if unknown:
        raise ValueError(f'unknown recipes: {unknown}')

    clean_path = (corpus_dir / 'clean.parquet').resolve(strict=True)
    recipe_paths = {
        recipe: (corpus_dir / f'{recipe}.parquet').resolve(strict=True)
        for recipe in recipes
    }
    flight_ids = read_flight_ids(clean_path)
    frozen_report, scorer, baseline, confidence = load_frozen_rule(frozen_report_path)
    threshold_score = baseline['median'] + float(norm.ppf(confidence)) * baseline['mad']

    inputs = [
        InputSpec(corpus_manifest_path, 'reference'),
        InputSpec(clean_path, 'natural_evaluation'),
        *[
            InputSpec(recipe_paths[recipe], 'synthetic_evaluation')
            for recipe in recipes
        ],
        InputSpec(frozen_report_path, 'reference'),
        InputSpec(historical_nn_report_path, 'reference'),
    ]
    config = {
        'step': 'step_3_corrected_truth_v2_frozen_rule_rescore',
        'architecture_truth_support': 'rule_full_window',
        'window': window,
        'stride': stride,
        'max_gap_s': max_gap_s,
        'recipes': list(recipes),
        'frozen_confidence_threshold': confidence,
        'frozen_equivalent_score_threshold': threshold_score,
        'episode_contract': {'emission_time': 't_end', 'merge_gap_s': 60.0},
        'auc_contract': {
            'clean_negative_once_per_recipe': True,
            'corrupt_q_eq_0_negative': False,
            'primary': 'y_any = 1[q_w > 0]',
            'secondary': 'q_w in {0,1}',
        },
        'paired_flight_contract': {
            'same_ids_across_clean_and_recipes': True,
            'flight_id_count': len(flight_ids),
            'flight_ids_sha256': sha256_json(flight_ids),
            'manifest_split_role': 'test_unique_ids_once',
        },
        'synthetic_usage': 'evaluation_and_reference_only_never_fit_or_calibrate',
        'frozen_scorer': frozen_report['scorer'],
        'historical_nn_status': 'historical_label_bugged_no_checkpoint_rescore',
    }
    manifest_path = create_immutable_run_manifest(
        run_dir=run_dir,
        repo_root=repo_root,
        inputs=inputs,
        splits={'test': flight_ids},
        split_algorithm='paired_synthetic_all_flights_unique_id_role_v1',
        split_seed=None,
        synthetic_flight_ids=flight_ids,
        config=config,
    )
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    clean = score_truth_file(
        clean_path,
        scorer,
        window=window,
        stride=stride,
        max_gap_s=max_gap_s,
    )
    clean_alarm = _fixed_alarms(clean.scores, baseline, confidence)
    clean_burden = natural_alert_burden(
        clean.meta,
        clean_alarm,
        contract=EpisodeContract(merge_gap_s=60.0, emission_time_col='t_end'),
    )

    per_recipe: dict[str, dict] = {}
    event_tables: list[pd.DataFrame] = []
    pooled_q: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    for recipe in recipes:
        corrupt = score_truth_file(
            recipe_paths[recipe],
            scorer,
            window=window,
            stride=stride,
            max_gap_s=max_gap_s,
        )
        result, event_table, positive_q, positive_scores = evaluate_recipe(
            clean,
            corrupt,
            baseline=baseline,
            confidence=confidence,
            clean_burden=clean_burden,
        )
        event_table.insert(0, 'recipe', recipe)
        per_recipe[recipe] = result
        event_tables.append(event_table)
        pooled_q.append(positive_q)
        pooled_scores.append(positive_scores)

    pooled_metric_q = np.concatenate(
        [np.zeros(len(clean.scores)), *pooled_q]
    )
    pooled_metric_scores = np.concatenate([clean.scores, *pooled_scores])
    pooled = diagnostic_window_metrics(pooled_metric_q, pooled_metric_scores)
    pooled['sampled_roc_y_any'] = sampled_roc(
        pooled_metric_q, pooled_metric_scores, max_points=ROC_MAX_POINTS
    )
    pooled['negative_contract'] = 'one_clean_pool_not_duplicated_per_recipe'

    event_table = pd.concat(event_tables, ignore_index=True)
    for column in ('recipe', 'event_id', 'event_type', 'flight_id'):
        event_table[column] = event_table[column].astype('string')
    event_table['detected'] = event_table['detected'].astype('boolean')
    event_table_path = run_dir / 'event_table.parquet'
    if event_table_path.exists():
        raise FileExistsError(event_table_path)
    event_table.to_parquet(event_table_path, index=False)

    frozen_input = _input_record(manifest, frozen_report_path, repo_root)
    nn_input = _input_record(manifest, historical_nn_report_path, repo_root)
    report = {
        'schema_version': 'adsb_corrected_truth_v2_rule_rescore_v1',
        'run_manifest': manifest_path.name,
        'status': 'corrected_truth_v2_frozen_rule_rescore',
        'units_kept_separate': ['window', 'event', 'flight', 'scoreable_flight_hour'],
        'truth_contract': {
            'architecture': 'rule_full_window',
            'q_w': 'observable_changed / evaluable_truth on full window support',
            'primary_window_label': 'y_any = 1[q_w > 0]',
            'secondary_window_subset': 'q_w in {0,1}',
            'corrupt_q_eq_0_role': 'timeline_sanity_only_not_auc_negative',
        },
        'frozen_rule': {
            'path': frozen_input['path'],
            'sha256': frozen_input['sha256'],
            'scorer': frozen_report['scorer'],
            'score_baseline_median_mad': baseline,
            'confidence_threshold': confidence,
            'equivalent_score_threshold': threshold_score,
            'fit_performed_in_this_run': False,
        },
        'historical_neural_baseline': {
            'path': nn_input['path'],
            'sha256': nn_input['sha256'],
            'status': 'historical_label_bugged_no_checkpoint_rescore',
            'corrected_truth_claim': False,
        },
        'clean_reference': {
            'path': _input_record(manifest, clean_path, repo_root)['path'],
            'n_windows': int(len(clean.scores)),
            'fixed_threshold_score_stats': _score_stats(clean.scores, clean_alarm),
            'natural_alert_burden': clean_burden,
        },
        'per_recipe': per_recipe,
        'pooled_all_recipes_vs_single_clean_reference': pooled,
        'event_table': {
            'path': event_table_path.name,
            'sha256': sha256_file(event_table_path),
            'footer_rows': int(len(event_table)),
            'unit': 'one_row_per_flight_event',
        },
    }
    summary_path = run_dir / 'summary.json'
    _write_json_exclusive(summary_path, report)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--corpus-dir', type=Path, default=DEFAULT_CORPUS)
    parser.add_argument('--frozen-report', type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument(
        '--historical-nn-report', type=Path, default=DEFAULT_HISTORICAL_NN_REPORT
    )
    parser.add_argument('--run-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_evaluation(
        repo_root=args.repo_root,
        corpus_dir=args.corpus_dir,
        frozen_report_path=args.frozen_report,
        historical_nn_report_path=args.historical_nn_report,
        run_dir=args.run_dir,
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
