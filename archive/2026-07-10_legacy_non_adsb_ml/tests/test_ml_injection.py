import numpy as np
import pandas as pd

from src.ml.injection import (
    inject_bias,
    inject_drift,
    inject_dropout,
    inject_freeze,
    inject_gps_ramp,
    inject_noise,
)


def _flight():
    n = 100
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "timestamp": np.arange(n) * 200_000,  # 5 Hz, us
        "ts_ns": np.arange(n) * 250_000_000,
        "lat": 40.0 + rng.normal(0, 1e-6, n),
        "roll_measured": rng.normal(0, 2.0, n),
        "vel_m_s": 5.0,
        "label": ["normal"] * n,
    })


def test_freeze_holds_last_value_and_marks_label():
    out = inject_freeze(_flight(), "roll_measured", onset_frac=0.5)
    assert out["roll_measured"].iloc[50:].nunique() == 1
    assert (out["label"].iloc[50:] == "inj_freeze").all()
    assert (out["label"].iloc[:50] == "normal").all()


def test_bias_shifts_by_sigma_multiple():
    df = _flight()
    out = inject_bias(df, "roll_measured", sigma_mult=4.0, onset_frac=0.5)
    shift = (out["roll_measured"].iloc[50:] - df["roll_measured"].iloc[50:]).mean()
    assert abs(shift - 4.0 * df["roll_measured"].std()) < 1e-6


def test_drift_grows_linearly():
    df = _flight()
    out = inject_drift(df, "roll_measured", "ts_ns", sigma_per_min=3.0, onset_frac=0.5)
    delta = out["roll_measured"] - df["roll_measured"]
    assert delta.iloc[:50].abs().max() == 0.0
    assert delta.iloc[-1] > delta.iloc[60] > 0


def test_gps_ramp_moves_position_but_not_velocity():
    df = _flight()
    out = inject_gps_ramp(df, meters_per_s=2.0, onset_frac=0.5)
    # onset oncesi degismedi; sonrasi kuzey yonlu kayma buyuyor
    assert (out["lat"].iloc[:50] == df["lat"].iloc[:50]).all()
    assert out["lat"].iloc[-1] > df["lat"].iloc[-1]
    # 100 ornek * 0.2s = 10s ucus; onset sonrasi ~10s * 2 m/s = ~20 m kayma bekleriz (son ornekte)
    moved_m = (out["lat"].iloc[-1] - df["lat"].iloc[-1]) * 111_320.0
    assert 15 < moved_m < 25
    assert (out["vel_m_s"] == df["vel_m_s"]).all()  # receiver hizi sabit -> residual dogar


def test_dropout_nans_block_after_onset():
    out = inject_dropout(_flight(), ["roll_measured"], onset_frac=0.5, block_frac=0.3)
    assert out["roll_measured"].iloc[:50].notna().all()
    assert out["roll_measured"].iloc[50:].isna().sum() == 15


def test_noise_increases_variance_only_after_onset():
    df = _flight()
    out = inject_noise(df, "roll_measured", sigma_mult=3.0, onset_frac=0.5)
    assert (out["roll_measured"].iloc[:50] == df["roll_measured"].iloc[:50]).all()
    assert out["roll_measured"].iloc[50:].std() > 2 * df["roll_measured"].iloc[50:].std()
