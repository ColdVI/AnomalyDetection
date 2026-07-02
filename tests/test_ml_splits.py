import pandas as pd
import pytest

from src.ml.data.scaling import apply_scaler_params, fit_scaler_params
from src.ml.data.splits import (
    assert_no_flight_overlap,
    build_split_manifest,
    flight_label_table,
    make_group_split,
    make_lofo_splits,
)


def _flights_df() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows += [{"source_id": f"normal_{i}", "label": "normal"}] * 5
    for i in range(4):
        rows += [{"source_id": f"fault_{i}", "label": "engine_fault"}] * 5
    rows += [{"source_id": "mixed", "label": "normal"}] * 2
    rows += [{"source_id": "mixed", "label": "rudder_fault"}] * 3
    rows += [{"source_id": "unk", "label": "unknown"}] * 5
    return pd.DataFrame(rows)


def test_flight_label_table_mixed_flight_is_anomalous():
    flights = flight_label_table(_flights_df())
    lookup = flights.set_index("source_id")["flight_label"]
    assert lookup["mixed"] == "rudder_fault"  # onset oncesi normal satirlar ucusu normal yapmaz
    assert lookup["unk"] == "unknown"
    assert lookup["normal_0"] == "normal"


def test_make_group_split_train_is_normal_only_and_no_overlap():
    flights = flight_label_table(_flights_df())
    split = make_group_split(flights, seed=0, n_val=2, n_test_normal=2)
    labels = flights.set_index("source_id")["flight_label"]
    assert all(labels[f] == "normal" for f in split["train"])
    assert all(labels[f] == "normal" for f in split["val"])
    # tum anomalili ucuslar test'te, unknown exploration'da
    assert set(split["test_anomalous"]) == {f"fault_{i}" for i in range(4)} | {"mixed"}
    assert split["exploration"] == ["unk"]
    assert_no_flight_overlap(split)


def test_make_group_split_is_seed_deterministic():
    flights = flight_label_table(_flights_df())
    a = make_group_split(flights, seed=3)
    b = make_group_split(flights, seed=3)
    assert a == b


def test_make_group_split_raises_when_too_few_normals():
    df = pd.DataFrame({"source_id": ["a"] * 3 + ["b"] * 3,
                       "label": ["normal"] * 3 + ["engine_fault"] * 3})
    flights = flight_label_table(df)
    with pytest.raises(ValueError):
        make_group_split(flights, seed=0)


def test_lofo_covers_every_normal_flight_once():
    flights = flight_label_table(_flights_df())
    folds = make_lofo_splits(flights)
    held = [f["held_out_flight"] for f in folds]
    assert sorted(held) == sorted(flights[flights["flight_label"] == "normal"]["source_id"])
    for fold in folds:
        assert fold["held_out_flight"] not in fold["train"]


def test_build_split_manifest_has_seeds_and_lofo():
    manifest = build_split_manifest({"alfa": _flights_df()})
    entry = manifest["sources"]["alfa"]
    assert len(entry["splits"]) == manifest["n_seeds"]
    assert len(entry["lofo"]) == 10  # 10 normal ucus; mixed anomalili, fold degil
    for split in entry["splits"].values():
        assert_no_flight_overlap(split)


def test_scaler_fit_on_train_only_and_constant_column_safe():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
    params = fit_scaler_params(train, ["a", "b"])
    assert params["columns"]["b"]["scale"] == 1.0  # sabit kolon: bolme hatasi yok
    test = pd.DataFrame({"a": [2.5, None], "b": [5.0, 5.0]})
    scaled = apply_scaler_params(test, params)
    assert scaled["a"].iloc[0] == 0.0  # medyan merkezli
    assert scaled["a"].notna().all()  # NaN train medyaniyla impute edildi
