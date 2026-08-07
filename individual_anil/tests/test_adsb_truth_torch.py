"""ADS-B truth/duzeltilmis-kural/windowing testleri (torch gerektirir)

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from __future__ import annotations

import numpy as np

import pandas as pd

from adsb.truth import (
    TRUTH_V2_COLUMNS,
    attach_clean_truth_v2,
    attach_event_truth_v2,
    paired_observable_changed,
    refresh_observable_truth_v2,
    score_support_mask,
    summarize_window_truth,
)

from adsb.windowing import build_windows

import json

from pathlib import Path

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

import torch

from adsb.windowing import build_windows, masked_mse, masked_mse_per_channel



# ===== kaynak: test_adsb_truth =====

def test_paired_observable_changed_treats_paired_null_as_unchanged():
    clean = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    corrupt = pd.DataFrame({"x": [1.0, np.nan, np.nan]})

    changed = paired_observable_changed(clean, corrupt, columns=["x"])

    assert changed.tolist() == [False, False, True]


def test_observation_fn_applies_feature_or_serialization_resolution():
    clean = pd.DataFrame({"x": [1.01, 1.11]})
    corrupt = pd.DataFrame({"x": [1.02, 1.19]})

    def one_decimal(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"observed": frame["x"].round(1)})

    changed = paired_observable_changed(
        clean,
        corrupt,
        columns=["observed"],
        observation_fn=one_decimal,
    )

    assert changed.tolist() == [False, True]


def test_attach_event_truth_separates_active_from_observable_and_bounds_event():
    clean = pd.DataFrame({"timestamp_utc": [0.0, 10.0, 20.0], "x": [0.0, np.nan, 2.0]})
    corrupt = clean.copy()
    corrupt.loc[2, "x"] = 5.0

    truth = attach_event_truth_v2(
        clean,
        corrupt,
        event_type="example",
        event_id="event-1",
        injection_active=[False, True, True],
        observable_cols=["x"],
    )

    assert truth["injection_active"].tolist() == [False, True, True]
    assert truth["observable_changed"].tolist() == [False, False, True]
    assert truth["evaluable_truth"].all()
    assert truth["event_id"].eq("event-1").all()
    assert truth["event_type"].eq("example").all()
    assert truth["attack_onset"].iloc[0] == 10.0
    assert truth["observable_onset"].iloc[0] == 20.0
    assert truth["event_end"].iloc[0] == 20.0


def test_clean_truth_has_complete_negative_contract():
    truth = attach_clean_truth_v2(pd.DataFrame({"x": [1.0, 2.0]}))

    assert set(TRUTH_V2_COLUMNS).issubset(truth.columns)
    assert not truth["injection_active"].any()
    assert not truth["observable_changed"].any()
    assert truth["evaluable_truth"].all()
    assert truth["event_id"].isna().all()


def test_refresh_observable_truth_uses_post_transform_pair():
    clean = pd.DataFrame({"timestamp_utc": [0.0, 1.0], "x": [1.01, 1.11]})
    corrupt_raw = clean.copy()
    corrupt_raw["x"] = [1.02, 1.19]
    truth = attach_event_truth_v2(
        clean,
        corrupt_raw,
        event_type="quantized",
        injection_active=[True, True],
        observable_cols=["x"],
    )

    refreshed = refresh_observable_truth_v2(
        clean,
        truth,
        observable_cols=["observed"],
        observation_fn=lambda frame: pd.DataFrame({"observed": frame["x"].round(1)}),
    )

    assert refreshed["observable_changed"].tolist() == [False, True]
    assert refreshed["observable_onset"].iloc[0] == 1.0


def test_rule_and_ae_use_full_window_q():
    truth = pd.DataFrame({
        "observable_changed": [False, False, True, True],
        "evaluable_truth": [True, True, True, True],
    })

    summary = summarize_window_truth(truth, architecture="rule")

    assert summary["q_w"] == 0.5
    assert summary["y_any"] is True
    assert summary["steady_subset"] is False
    assert summary["history_contaminated"] is False
    assert score_support_mask(4, architecture="dense_ae").tolist() == [True] * 4


def test_forecaster_uses_only_target_and_separates_contaminated_history():
    truth = pd.DataFrame({
        "observable_changed": [True, True, False, False],
        "evaluable_truth": [True, True, True, True],
    })

    summary = summarize_window_truth(
        truth,
        architecture="forecaster",
        forecast_target_rows=2,
    )

    assert summary["q_w"] == 0.0
    assert summary["y_any"] is False
    assert summary["steady_subset"] is True
    assert summary["steady_label"] is False
    assert summary["history_contaminated"] is True


def test_zero_evaluable_support_is_unscoreable_truth():
    truth = pd.DataFrame({
        "observable_changed": [True, False, False, False],
        "evaluable_truth": [True, True, False, False],
    })

    summary = summarize_window_truth(
        truth,
        architecture="lstm_forecaster",
        forecast_target_rows=2,
    )

    assert np.isnan(summary["q_w"])
    assert summary["truth_scoreable"] is False
    assert pd.isna(summary["y_any"])
    assert summary["steady_subset"] is False
    assert summary["history_contaminated"] is True


def test_build_windows_integrates_forecaster_support_without_history_leakage():
    df = pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": np.arange(6, dtype=float),
        "f1": np.arange(6, dtype=float),
        "observable_changed": [True, True, False, False, False, False],
        "evaluable_truth": True,
    })

    _, _, meta = build_windows(
        df,
        ["f1"],
        window=4,
        stride=2,
        max_gap_s=60.0,
        truth_architecture="forecaster",
        forecast_target_rows=2,
    )

    assert meta["q_w"].tolist() == [0.0, 0.0]
    assert meta["y_any"].tolist() == [False, False]
    assert meta["history_contaminated"].tolist() == [True, False]
    assert meta["t_end"].tolist() == [3.0, 5.0]



# ===== kaynak: test_adsb_corrected_rule_evaluation =====

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



# ===== kaynak: test_adsb_windowing =====

def _flight_df(flight_id: str, n: int, dt: float = 10.0, gap_at: int | None = None) -> pd.DataFrame:
    t = np.arange(n) * dt
    if gap_at is not None:
        t = t.astype(float)
        t[gap_at:] += 5000.0  # buyuk boşluk
    return pd.DataFrame({
        "flight_id": flight_id,
        "timestamp_utc": t,
        "f1": np.arange(n, dtype=float),
        "f2": np.arange(n, dtype=float) * 2,
    })


def test_build_windows_basic_shapes():
    df = _flight_df("A", 20)
    X, M, meta = build_windows(df, ["f1", "f2"], window=5, stride=5, max_gap_s=1800.0)
    assert X.shape == (4, 5, 2)
    assert M.shape == (4, 5, 2)
    assert len(meta) == 4
    assert (M == 1.0).all()  # eksik yok


def test_build_windows_skips_short_flights():
    df = _flight_df("A", 3)
    X, M, meta = build_windows(df, ["f1", "f2"], window=5, stride=5, max_gap_s=1800.0)
    assert len(X) == 0


def test_build_windows_skips_windows_crossing_gap():
    df = _flight_df("A", 10, gap_at=5)  # index 5'ten sonra 5000s sıçrama
    X, M, meta = build_windows(df, ["f1", "f2"], window=5, stride=1, max_gap_s=1800.0)
    # 6 aday pencereden (start=0..5) yalniz bosluğu icermeyen ikisi kalir: [0:5) ve [5:10)
    assert len(X) == 2
    assert meta["t_start"].tolist() == [0.0, 5050.0]


def test_build_windows_nan_handled_with_mask():
    df = _flight_df("A", 10)
    df.loc[3, "f1"] = np.nan
    X, M, meta = build_windows(df, ["f1", "f2"], window=5, stride=5, max_gap_s=1800.0)
    assert X[0, 3, 0] == 0.0  # NaN -> 0
    assert M[0, 3, 0] == 0.0  # maske eksik isaretli
    assert M[0, 3, 1] == 1.0  # diger kanal etkilenmedi


def test_masked_mse_matches_manual_computation():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    recon = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
    mask = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    # hatalar: (1-1)^2=0, (2-0)^2=4, (3-3)^2=0, (4-0)^2=16 -> toplam 20, /4 eleman = 5.0
    result = masked_mse(x, recon, mask)
    assert result.shape == (1,)
    assert torch.allclose(result, torch.tensor([5.0]))


def test_masked_mse_ignores_masked_out_entries():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    recon = torch.tensor([[[1.0, 999.0], [3.0, 4.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])  # 2. eleman maskeli (999 farkı görmezden gelinir)
    result = masked_mse(x, recon, mask)
    assert torch.allclose(result, torch.tensor([0.0]))


def test_masked_mse_per_channel_recombines_to_masked_mse():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    recon = torch.randn(4, 6, 3)
    mask = (torch.rand(4, 6, 3) > 0.2).float()

    total = masked_mse(x, recon, mask)
    numerator, denominator = masked_mse_per_channel(x, recon, mask)
    recombined = numerator.sum(dim=-1) / denominator.sum(dim=-1).clamp(min=1.0)

    assert torch.allclose(total, recombined, atol=1e-6)

