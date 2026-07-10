"""adsb/windowing.py testleri -- sentetik veri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from adsb.windowing import build_windows, masked_mse, masked_mse_per_channel


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
