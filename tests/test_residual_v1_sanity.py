import pandas as pd

from scripts.residual_v1_sanity_plots import _spearman_or_none


def test_sanity_spearman_marks_constant_input_without_nan():
    statistic, status = _spearman_or_none(
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series([1.0, 2.0, 3.0]),
    )
    assert statistic is None
    assert status == "constant_input"
