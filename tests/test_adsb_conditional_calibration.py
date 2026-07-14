import pandas as pd
import pytest

from adsb.conditional_calibration import (
    NATURAL_CALIBRATION_ROLE,
    ConditionalCalibrationConfig,
    HierarchicalConformalCalibrator,
)


def _calibration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["speed"] * 6 + ["track"] * 3,
            "context_phase": ["level"] * 3 + ["climb"] * 3 + ["level"] * 3,
            "context_cadence": ["cadence_0"] * 3 + ["cadence_1"] * 3 + ["cadence_0"] * 3,
            "score": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.5, 2.0],
        }
    )


def _fit() -> HierarchicalConformalCalibrator:
    return HierarchicalConformalCalibrator(ConditionalCalibrationConfig(min_group_size=3)).fit(
        _calibration(), data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False
    )


def test_exact_empirical_tail_and_hierarchical_fallback():
    scored = pd.DataFrame(
        {
            "channel": ["speed", "speed", "speed"],
            "context_phase": ["level", "level", "unknown"],
            "context_cadence": ["cadence_0", "unseen", "unseen"],
            "score": [2.0, 2.0, 2.0],
        }
    )
    result = _fit().transform(scored)
    assert result.loc[0, "conformal_p_value"] == pytest.approx(0.75)
    assert result.loc[0, "calibration_level"] == "channel+context_phase+context_cadence"
    assert result.loc[1, "calibration_level"] == "channel+context_phase"
    assert result.loc[2, "calibration_level"] == "channel"


def test_fit_rejects_synthetic_or_wrong_role():
    calibrator = HierarchicalConformalCalibrator(ConditionalCalibrationConfig(min_group_size=3))
    with pytest.raises(ValueError, match="Synthetic"):
        calibrator.fit(
            _calibration(), data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=True
        )
    with pytest.raises(ValueError, match="Only"):
        calibrator.fit(_calibration(), data_role="truth_v2", contains_synthetic=False)


def test_alarm_threshold_is_mandatory_and_explicit():
    scored = _calibration().iloc[[0]].copy()
    calibrator = _fit()
    with pytest.raises(ValueError, match="mandatory"):
        calibrator.alarms(scored, alpha=None)
    result = calibrator.alarms(scored, alpha=0.2)
    assert result.loc[0, "alert_alpha"] == 0.2
    assert bool(result.loc[0, "alarm"]) is False
