import numpy as np
import pandas as pd

from src.ml.data.windowing import build_windows


def _df():
    n = 25
    return pd.DataFrame({
        "source_id": ["f1"] * n,
        "t_rel_s": np.arange(n) * 0.25,
        "a": np.arange(n, dtype=float),
        "b": [np.nan if i == 3 else 1.0 for i in range(n)],
        "label": ["normal"] * 20 + ["engine_fault"] * 5,
    })


def test_build_windows_shapes_and_mask():
    X, M, meta = build_windows(_df(), ["a", "b"], window=8, stride=4, max_gap_s=1.0)
    assert X.shape == M.shape and X.shape[1:] == (8, 2)
    # NaN 0'a cevrildi, maskede 0 olarak isaretli
    assert X[0, 3, 1] == 0.0 and M[0, 3, 1] == 0.0 and M[0, 2, 1] == 1.0


def test_window_label_any_anomalous_row_marks_window():
    X, M, meta = build_windows(_df(), ["a"], window=8, stride=4, max_gap_s=1.0)
    # 20. satirdan itibaren fault: son pencereler anomali olmali, ilkler normal
    assert not meta["is_anomaly"].iloc[0]
    assert meta["is_anomaly"].iloc[-1]
    assert meta[meta["is_anomaly"]]["label"].iloc[0] == "engine_fault"


def test_windows_with_time_gap_are_dropped():
    df = _df()
    df.loc[12:, "t_rel_s"] += 30.0  # kayit kopmasi
    X, _, meta = build_windows(df, ["a"], window=8, stride=1, max_gap_s=1.0)
    # kopmayi kapsayan hicbir pencere kalmamali
    assert all((meta["t_end"] - meta["t_start"]) < 10)


def test_windows_never_cross_flights():
    df = pd.concat([_df().assign(source_id="f1"), _df().assign(source_id="f2")])
    X, _, meta = build_windows(df, ["a"], window=8, stride=4, max_gap_s=1.0)
    assert set(meta["source_id"]) == {"f1", "f2"}
    # her pencere tek ucusa ait (gruplama garanti eder) -- toplam pencere 2 kat
    single = build_windows(_df(), ["a"], window=8, stride=4, max_gap_s=1.0)[2]
    assert len(meta) == 2 * len(single)
