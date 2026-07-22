import numpy as np
import pandas as pd

from rfly_full.v2_parser import (
    TRUTH_SCHEMA_VERSION,
    _binned_stats,
    _control_domain,
    _migrate_truth_state,
    _truth_crosscheck_metrics,
)


def test_binned_stats_preserve_high_frequency_variation():
    result = _binned_stats(
        np.array([10_000, 40_000, 90_000, 110_000]),
        np.array([1.0, 3.0, 5.0, 9.0]),
        step_us=100_000,
        prefix="signal",
    )
    first = result.iloc[0]
    assert first["timestamp"] == 100_000
    assert first["signal_mean"] == 3.0
    assert first["signal_ptp"] == 4.0
    assert first["signal_rms"] > first["signal_mean"]


def test_control_domain_accepts_hyphen_and_underscore_packages():
    assert _control_domain('HIL-Sensors') == 'HIL'
    assert _control_domain('HIL_Motor_1') == 'HIL'
    assert _control_domain('SIL-Load') == 'SIL'
    assert _control_domain('SIL_Prop') == 'SIL'
    assert _control_domain('Real-Motor') == 'REAL'


def test_truth_schema_migration_invalidates_only_underscore_sil_hil_packages():
    manifest = pd.DataFrame({
        'canonical_case_id': ['hil_motor', 'sil_prop', 'hil_sensor', 'real_motor'],
        'case_id': [
            'HIL_Motor_1/case-a',
            'SIL_Prop/case-b',
            'HIL-Sensors/case-c',
            'Real-Motor/case-d',
        ],
    })
    state = {
        'completed': ['hil_motor', 'sil_prop', 'hil_sensor', 'real_motor'],
        'failed': {'hil_motor': 'old failure', 'hil_sensor': 'keep failure'},
    }

    affected = _migrate_truth_state(state, manifest)

    assert affected == {'hil_motor', 'sil_prop'}
    assert state['completed'] == ['hil_sensor', 'real_motor']
    assert state['failed'] == {'hil_sensor': 'keep failure'}
    assert state['truth_schema_version'] == TRUTH_SCHEMA_VERSION
    assert state['truth_reparse_invalidated'] == 2


def test_truth_schema_migration_is_idempotent():
    manifest = pd.DataFrame({
        'canonical_case_id': ['hil_motor'],
        'case_id': ['HIL_Motor_1/case-a'],
    })
    state = {
        'completed': ['hil_motor'],
        'failed': {},
        'truth_schema_version': TRUTH_SCHEMA_VERSION,
    }

    assert _migrate_truth_state(state, manifest) == set()
    assert state['completed'] == ['hil_motor']


def test_crosscheck_v2_accepts_bounded_shift_with_interval_overlap():
    t = np.arange(0.0, 61.0)
    planned = (t >= 10.0) & (t <= 30.0)
    control = (t >= 25.0) & (t <= 45.0)

    result = _truth_crosscheck_metrics(t, control, planned)

    assert result['truth_crosscheck_eligible_v2'] is True
    assert result['truth_crosscheck_onset_delta_s'] == 15.0
    assert result['truth_crosscheck_offset_delta_s'] == 15.0
    assert result['truth_crosscheck_overlap_s'] > 0.0
    assert result['truth_crosscheck_disagreement_v2'] is False


def test_crosscheck_v2_rejects_different_nonoverlapping_onset():
    t = np.arange(0.0, 61.0)
    planned = (t >= 10.0) & (t <= 20.0)
    control = (t >= 40.0) & (t <= 50.0)

    result = _truth_crosscheck_metrics(t, control, planned)

    assert result['truth_crosscheck_onset_delta_s'] == 30.0
    assert result['truth_crosscheck_offset_delta_s'] == 30.0
    assert result['truth_crosscheck_overlap_s'] == 0.0
    assert result['truth_crosscheck_disagreement_v2'] is True
