from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from adsb.evaluation import (
    active_interval_coverage,
    event_observability_denominators,
    truth_event_table,
)
from adsb.synthetic import inject_freeze
from adsb.truth import attach_clean_truth_v2
from scripts.adsb_evaluate_rule_scorer_truth_v2 import (
    ScoredTruthFile,
    evaluate_recipe,
    run_evaluation,
)


def test_event_truth_keeps_active_but_unobservable_out_of_recall_denominator():
    rows = pd.DataFrame(
        {
            'event_id': ['seen', 'seen', 'hidden', 'hidden'],
            'event_type': ['x'] * 4,
            'flight_id': ['a', 'a', 'b', 'b'],
            'attack_onset': [0.0, 0.0, 10.0, 10.0],
            'observable_onset': [1.0, 1.0, np.nan, np.nan],
            'event_end': [2.0, 2.0, 11.0, 11.0],
            'injection_active': [True, True, True, True],
            'observable_changed': [False, True, False, False],
            'evaluable_truth': [True, True, True, True],
        }
    )
    events = truth_event_table(rows)
    result = event_observability_denominators(events)
    assert result['n_attack_eligible_events'] == 2
    assert result['n_observable_eligible_events'] == 1
    assert result['n_active_but_unobservable_events'] == 1


def test_active_coverage_unions_only_alerted_window_support():
    events = pd.DataFrame(
        {
            'event_id': ['e'],
            'flight_id': ['a'],
            'observable_onset': [10.0],
            'event_end': [30.0],
        }
    )
    meta = pd.DataFrame(
        {
            'flight_id': ['a', 'a', 'a'],
            't_start': [0.0, 12.0, 18.0],
            't_end': [15.0, 20.0, 25.0],
        }
    )
    result = active_interval_coverage(events, meta, np.array([True, False, True]))
    assert result['point_adjustment'] is False
    assert result['total_alerted_window_support_seconds'] == 12.0
    assert result['micro_fraction'] == 0.6


def test_recipe_auc_contract_excludes_corrupt_q0_duplicate():
    clean_meta = pd.DataFrame(
        {
            'flight_id': ['f'] * 3,
            't_start': [0.0, 10.0, 20.0],
            't_end': [5.0, 15.0, 25.0],
            'q_w': [0.0, 0.0, 0.0],
            'truth_scoreable': [True] * 3,
        }
    )
    corrupt_meta = clean_meta.copy()
    corrupt_meta['q_w'] = [0.0, 0.5, 1.0]
    events = pd.DataFrame(
        {
            'event_id': ['e'],
            'event_type': ['x'],
            'flight_id': ['f'],
            'attack_onset': [10.0],
            'observable_onset': [10.0],
            'event_end': [25.0],
            'attack_active_rows': [2],
            'evaluable_active_rows': [2],
            'observable_changed_rows': [2],
            'attack_eligible': [True],
            'observable_eligible': [True],
            'active_but_unobservable': [False],
        }
    )
    clean = ScoredTruthFile(np.array([0.0, 0.1, 0.2]), clean_meta, events.iloc[0:0])
    corrupt = ScoredTruthFile(np.array([10.0, 1.0, 2.0]), corrupt_meta, events)
    result, _, positive_q, _ = evaluate_recipe(
        clean,
        corrupt,
        baseline={'median': 0.0, 'mad': 1.0},
        confidence=0.95,
        clean_burden={'unit': 'fixture'},
    )
    contract = result['auc_input_contract']
    assert contract['n_clean_negative_windows'] == 3
    assert contract['n_corrupt_positive_windows'] == 2
    assert contract['corrupt_q_eq_0_included_as_negative'] is False
    assert positive_q.tolist() == [0.5, 1.0]
    assert result['corrupt_q_eq_0_timeline_sanity']['n_alarms'] == 1


def _clean_flight(flight_id: str, offset: float) -> pd.DataFrame:
    timestamp = np.arange(8, dtype=float) + offset
    return pd.DataFrame(
        {
            'flight_id': flight_id,
            'timestamp_utc': timestamp,
            'lat': 40.0 + timestamp * 1e-5,
            'lon': 29.0 + timestamp * 1e-5,
            'alt': 1000.0 + timestamp,
            'alt_geom_m': 1010.0 + timestamp,
            'ground_speed_ms': np.full(8, 60.0),
            'track_deg': np.linspace(10.0, 17.0, 8),
            'vertical_rate_ms': np.linspace(1.0, 2.0, 8),
            'roll_deg': np.zeros(8),
        }
    )


def test_small_truth_v2_run_is_immutable_and_writes_event_table(tmp_path: Path):
    corpus = tmp_path / 'synthetic' / 'adsb_v2_fixture'
    corpus.mkdir(parents=True)
    clean_raw = pd.concat(
        [_clean_flight('f1', 0.0), _clean_flight('f2', 100.0)], ignore_index=True
    )
    clean = pd.concat(
        [attach_clean_truth_v2(group) for _, group in clean_raw.groupby('flight_id')],
        ignore_index=True,
    )
    corrupt = pd.concat(
        [
            inject_freeze(
                group.reset_index(drop=True),
                'vertical_rate_ms',
                event_type='vertical_rate_frozen',
            )
            for _, group in clean_raw.groupby('flight_id')
        ],
        ignore_index=True,
    )
    clean.to_parquet(corpus / 'clean.parquet', index=False)
    corrupt.to_parquet(corpus / 'vertical_rate_frozen.parquet', index=False)
    (corpus / 'manifest.json').write_text(
        json.dumps({'schema_version': 'adsb_synthetic_truth_v2'}), encoding='utf-8'
    )

    frozen = tmp_path / 'frozen_rule.json'
    frozen.write_text(
        json.dumps(
            {
                'scorer': {
                    'channels': ['vertical_rate_residual'],
                    'weights': {'vertical_rate_residual': 1.0},
                    'z0': 3.0,
                    'cap': 10.0,
                    'calibration': {
                        'vertical_rate_residual': {'median': 0.0, 'mad': 1.0}
                    },
                    'excluded_channels': [],
                },
                'confidence_threshold': 0.95,
                'score_baseline_median_mad': {'median': 0.0, 'mad': 1.0},
            }
        ),
        encoding='utf-8',
    )
    historical = tmp_path / 'historical_nn.json'
    historical.write_text('{}', encoding='utf-8')
    run_dir = tmp_path / 'run'
    summary_path = run_evaluation(
        repo_root=Path.cwd(),
        corpus_dir=corpus,
        frozen_report_path=frozen,
        historical_nn_report_path=historical,
        run_dir=run_dir,
        recipes=('vertical_rate_frozen',),
        window=4,
        stride=2,
        max_gap_s=60.0,
    )
    report = json.loads(summary_path.read_text(encoding='utf-8'))
    assert report['status'] == 'corrected_truth_v2_frozen_rule_rescore'
    assert report['historical_neural_baseline']['corrected_truth_claim'] is False
    assert (run_dir / 'run_manifest.json').is_file()
    assert (run_dir / 'event_table.parquet').is_file()
    try:
        run_evaluation(
            repo_root=Path.cwd(),
            corpus_dir=corpus,
            frozen_report_path=frozen,
            historical_nn_report_path=historical,
            run_dir=run_dir,
            recipes=('vertical_rate_frozen',),
            window=4,
            stride=2,
            max_gap_s=60.0,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError('run directory reuse must fail')
