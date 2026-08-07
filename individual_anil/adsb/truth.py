"""Sentetik ADS-B olaylari icin satir ve pencere truth v2 sozlesmesi.

Truth, enjektorun calistirilmis olmasi ile gozlemde gercekten bir sey degismesi
arasindaki farki korur. Ozellikle mevcut degeri zaten NaN olan bir dropout
satiri ya da position-ramp'in ``dt == 0`` onset satiri enjekte edilmis olabilir,
ama ``observable_changed`` degildir.

Pencere etiketi model mimarisinden bagimsizmis gibi varsayilmaz. Rule/AE skoru
butun pencereyi, forecaster skoru ise yalniz hedef satirlarini destekler.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

TRUTH_V2_COLUMNS = (
    "event_id",
    "event_type",
    "attack_onset",
    "observable_onset",
    "event_end",
    "injection_active",
    "observable_changed",
    "evaluable_truth",
)

WINDOW_TRUTH_META_COLUMNS = (
    "q_w",
    "truth_scoreable",
    "y_any",
    "steady_subset",
    "steady_label",
    "history_contaminated",
)

_FULL_WINDOW_ARCHITECTURES = {
    "rule", "rule_scorer", "ae", "autoencoder", "dense_ae",
    "dense_autoencoder", "lstm_ae", "lstm_autoencoder", "usad",
}
_FORECASTER_ARCHITECTURES = {"forecaster", "lstm_forecaster"}


def _as_bool_mask(values: Iterable[bool], *, length: int, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.ndim != 1 or len(mask) != length:
        raise ValueError(f"{name} uzunlugu satir sayisiyla ayni olmali ({length}).")
    if pd.isna(mask).any():
        raise ValueError(f"{name} null deger iceremez.")
    return mask.astype(bool, copy=False)


def paired_observable_changed(
    clean: pd.DataFrame,
    corrupt: pd.DataFrame,
    *,
    columns: Iterable[str],
    observation_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> np.ndarray:
    """Temiz/bozuk cifti gozlem yuzeyinde satir satir karsilastirir.

    ``observation_fn`` model feature'larini ureten ya da Parquet round-trip'ini
    temsil eden saf bir donusum olabilir. Iki tarafta da null olan hucre
    degismemis, tek tarafta null olan hucre degismis kabul edilir.
    """
    if len(clean) != len(corrupt):
        raise ValueError("clean ve corrupt ayni sayida, eslesmis satir icermeli.")
    observe = observation_fn or (lambda frame: frame)
    clean_obs = observe(clean.copy()).reset_index(drop=True)
    corrupt_obs = observe(corrupt.copy()).reset_index(drop=True)
    if len(clean_obs) != len(clean) or len(corrupt_obs) != len(corrupt):
        raise ValueError("observation_fn satir sayisini veya hizasini degistiremez.")

    selected = list(columns)
    if not selected:
        raise ValueError("observable karsilastirma icin en az bir kolon gerekli.")
    missing_clean = sorted(set(selected) - set(clean_obs.columns))
    missing_corrupt = sorted(set(selected) - set(corrupt_obs.columns))
    if missing_clean or missing_corrupt:
        raise KeyError(
            "observable kolonlar eksik: "
            f"clean={missing_clean}, corrupt={missing_corrupt}"
        )
    left = clean_obs[selected]
    right = corrupt_obs[selected]
    same = left.eq(right) | (left.isna() & right.isna())
    return ~same.fillna(False).all(axis=1).to_numpy(dtype=bool)


def _time_at(df: pd.DataFrame, position: int | None, time_col: str) -> Any:
    if position is None:
        return pd.NA
    if time_col not in df.columns:
        return position
    return df[time_col].iloc[position]


def _default_event_id(
    df: pd.DataFrame, *, event_type: str, attack_position: int | None, time_col: str
) -> str:
    flight_token = "single_flight"
    if "flight_id" in df.columns:
        flight_ids = df["flight_id"].dropna().astype(str).unique()
        if len(flight_ids) > 1:
            raise ValueError("Tek bir sentetik event birden fazla flight_id kapsayamaz.")
        if len(flight_ids) == 1:
            flight_token = flight_ids[0]
    onset_token = _time_at(df, attack_position, time_col)
    return f"{event_type}:{flight_token}:{onset_token}"


def attach_event_truth_v2(
    clean: pd.DataFrame,
    corrupt: pd.DataFrame,
    *,
    event_type: str,
    injection_active: Iterable[bool],
    observable_cols: Iterable[str],
    event_id: str | None = None,
    evaluable_truth: Iterable[bool] | None = None,
    observation_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    time_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Eslesmis bir temiz/bozuk cifte truth v2 kolonlarini ekler.

    Olay sinirlari inclusive satir zamanlaridir. ``attack_onset`` ilk gercek
    ``injection_active`` satiridir; dropout icin bu, onset aday bolgesinin basi
    degil, rastgele secilen gercek blogun basidir. ``observable_onset`` ilk
    eslesmis ve gozlenebilir farktir; hic fark yoksa null kalir.
    """
    if len(clean) != len(corrupt):
        raise ValueError("clean ve corrupt ayni sayida, eslesmis satir icermeli.")
    if not event_type:
        raise ValueError("event_type bos olamaz.")

    n_rows = len(corrupt)
    active = _as_bool_mask(injection_active, length=n_rows, name="injection_active")
    if evaluable_truth is None:
        evaluable = np.ones(n_rows, dtype=bool)
    else:
        evaluable = _as_bool_mask(
            evaluable_truth, length=n_rows, name="evaluable_truth"
        )
    changed = paired_observable_changed(
        clean,
        corrupt,
        columns=observable_cols,
        observation_fn=observation_fn,
    )

    active_positions = np.flatnonzero(active)
    attack_position = int(active_positions[0]) if len(active_positions) else None
    event_end_position = int(active_positions[-1]) if len(active_positions) else None
    observable_positions = np.flatnonzero(changed & evaluable)
    observable_position = int(observable_positions[0]) if len(observable_positions) else None

    out = corrupt.copy()
    out["event_id"] = event_id or _default_event_id(
        out,
        event_type=event_type,
        attack_position=attack_position,
        time_col=time_col,
    )
    out["event_type"] = event_type
    out["attack_onset"] = _time_at(out, attack_position, time_col)
    out["observable_onset"] = _time_at(out, observable_position, time_col)
    out["event_end"] = _time_at(out, event_end_position, time_col)
    out["injection_active"] = active
    out["observable_changed"] = changed
    out["evaluable_truth"] = evaluable
    return out


def attach_clean_truth_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Sentetik korpusun eslesmis clean tarafina negatif truth v2 ekler."""
    out = df.copy()
    out["event_id"] = pd.NA
    out["event_type"] = pd.NA
    out["attack_onset"] = pd.NA
    out["observable_onset"] = pd.NA
    out["event_end"] = pd.NA
    out["injection_active"] = False
    out["observable_changed"] = False
    out["evaluable_truth"] = True
    return out


def refresh_observable_truth_v2(
    clean: pd.DataFrame,
    corrupt: pd.DataFrame,
    *,
    observable_cols: Iterable[str],
    observation_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    time_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Serialization/feature donusumunden sonra observable truth'u yeniler."""
    missing = sorted(set(TRUTH_V2_COLUMNS) - set(corrupt.columns))
    if missing:
        raise KeyError(f"truth v2 kolonlari eksik: {missing}")
    changed = paired_observable_changed(
        clean,
        corrupt,
        columns=observable_cols,
        observation_fn=observation_fn,
    )
    evaluable = _as_bool_mask(
        corrupt["evaluable_truth"], length=len(corrupt), name="evaluable_truth"
    )
    observable_positions = np.flatnonzero(changed & evaluable)
    observable_position = int(observable_positions[0]) if len(observable_positions) else None

    out = corrupt.copy()
    out["observable_changed"] = changed
    out["observable_onset"] = _time_at(out, observable_position, time_col)
    return out


def score_support_mask(
    window: int,
    *,
    architecture: str,
    forecast_target_rows: int | None = None,
) -> np.ndarray:
    """Mimarinin bir pencerede gercekten skorladigi satirlari dondurur."""
    if window <= 0:
        raise ValueError("window pozitif olmali.")
    architecture_key = architecture.strip().lower()
    if architecture_key in _FULL_WINDOW_ARCHITECTURES:
        if forecast_target_rows is not None:
            raise ValueError("Full-window mimaride forecast_target_rows verilmez.")
        return np.ones(window, dtype=bool)
    if architecture_key in _FORECASTER_ARCHITECTURES:
        if forecast_target_rows is None:
            raise ValueError("Forecaster icin forecast_target_rows acikca verilmelidir.")
        if not 0 < forecast_target_rows <= window:
            raise ValueError("forecast_target_rows 1..window araliginda olmali.")
        support = np.zeros(window, dtype=bool)
        support[-forecast_target_rows:] = True
        return support
    raise ValueError(f"Bilinmeyen truth score architecture: {architecture!r}")


def summarize_window_truth(
    window_truth: pd.DataFrame,
    *,
    architecture: str,
    forecast_target_rows: int | None = None,
) -> dict[str, Any]:
    """Bir pencere icin q_w ve on-kayitli etiket/tabaka alanlarini hesaplar."""
    required = {"observable_changed", "evaluable_truth"}
    missing = sorted(required - set(window_truth.columns))
    if missing:
        raise KeyError(f"window truth kolonlari eksik: {missing}")
    n_rows = len(window_truth)
    support = score_support_mask(
        n_rows,
        architecture=architecture,
        forecast_target_rows=forecast_target_rows,
    )
    changed = _as_bool_mask(
        window_truth["observable_changed"], length=n_rows, name="observable_changed"
    )
    evaluable = _as_bool_mask(
        window_truth["evaluable_truth"], length=n_rows, name="evaluable_truth"
    )

    supported_evaluable = support & evaluable
    denominator = int(supported_evaluable.sum())
    history_contaminated = bool((~support & evaluable & changed).any())
    if denominator == 0:
        return {
            "q_w": np.nan,
            "truth_scoreable": False,
            "y_any": pd.NA,
            "steady_subset": False,
            "steady_label": pd.NA,
            "history_contaminated": history_contaminated,
        }

    numerator = int((supported_evaluable & changed).sum())
    q_w = numerator / denominator
    y_any = q_w > 0.0
    steady_subset = q_w == 0.0 or q_w == 1.0
    return {
        "q_w": q_w,
        "truth_scoreable": True,
        "y_any": y_any,
        "steady_subset": steady_subset,
        "steady_label": (q_w == 1.0) if steady_subset else pd.NA,
        "history_contaminated": history_contaminated,
    }
