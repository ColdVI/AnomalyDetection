import pandas as pd
import pytest

from src.ml.data.scaling import apply_scaler_params, fit_scaler_params
from src.ml.data.splits import (
    add_supervised_splits,
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


def test_supervised_splits_have_both_classes_and_exclude_holdout():
    manifest = build_split_manifest({"alfa": _flights_df()})
    holdout = "fault_0"
    for split in manifest["sources"]["alfa"]["splits"].values():
        split["final_holdout"] = [holdout]
        split["final_holdout_anomalous"] = [holdout]
        split["development_anomalous"] = [f"fault_{i}" for i in range(1, 4)] + ["mixed"]
    extended = add_supervised_splits(manifest, sources=("alfa",))

    for split in extended["sources"]["alfa"]["supervised_splits"].values():
        assert holdout not in split["train"] + split["val"] + split["test"]
        for part in ("train", "val", "test"):
            assert split[f"{part}_normal"]
            assert split[f"{part}_anomalous"]
        assert not set(split["train"]) & set(split["val"])
        assert not set(split["train"]) & set(split["test"])
        assert not set(split["val"]) & set(split["test"])


def test_scaler_fit_on_train_only_and_constant_column_safe():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
    params = fit_scaler_params(train, ["a", "b"])
    assert params["columns"]["b"]["scale"] == 1.0  # sabit kolon: bolme hatasi yok
    test = pd.DataFrame({"a": [2.5, None], "b": [5.0, 5.0]})
    scaled = apply_scaler_params(test, params)
    assert scaled["a"].iloc[0] == 0.0  # medyan merkezli
    assert scaled["a"].notna().all()  # NaN train medyaniyla impute edildi


def _session_flights_df():
    rows = []
    # 4 temiz-normal oturum (2'ser ucus), 1 karisik oturum (1 normal + 1 anomalili)
    for d in ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]:
        for h in ["10_00_00", "11_00_00"]:
            rows += [{"source_id": f"{d}/{h}", "label": "normal"}] * 3
    rows += [{"source_id": "2018-02-01/09_00_00", "label": "normal"}] * 3
    rows += [{"source_id": "2018-02-01/09_30_00", "label": "altitude_anomaly"}] * 3
    return pd.DataFrame(rows)


def test_session_split_keeps_sessions_together_and_quarantines_tainted():
    from src.ml.data.splits import make_group_split, flight_label_table, session_of

    flights = flight_label_table(_session_flights_df())
    split = make_group_split(flights, seed=0, n_val=2, n_test_normal=2, by_session=True)
    assert_no_flight_overlap(split)
    # ayni oturumun ucuslari ayni tarafta olmali
    side = {}
    for part in ["train", "val", "test"]:
        for f in split[part]:
            s = session_of(f)
            assert side.get(s, part) == part, f"oturum {s} iki tarafa dustu"
            side[s] = part
    # anomalili oturumun normali train/val'de OLMAMALI
    assert "2018-02-01/09_00_00" not in split["train"]
    assert "2018-02-01/09_00_00" not in split["val"]
    assert "2018-02-01/09_00_00" in split["test_normal"]


def test_session_split_deterministic_and_train_normal_only():
    from src.ml.data.splits import make_group_split, flight_label_table

    flights = flight_label_table(_session_flights_df())
    a = make_group_split(flights, seed=1, by_session=True)
    b = make_group_split(flights, seed=1, by_session=True)
    assert a == b
    labels = flights.set_index("source_id")["flight_label"]
    assert all(labels[f] == "normal" for f in a["train"])


def test_session_lofo_never_keeps_sibling_flight_in_train():
    from src.ml.data.splits import flight_label_table, make_lofo_splits, session_of

    flights = flight_label_table(_session_flights_df())
    folds = make_lofo_splits(flights, by_session=True)
    assert folds
    for fold in folds:
        held = fold["held_out_session"]
        assert all(session_of(f) == held for f in fold["val"])
        assert all(session_of(f) != held for f in fold["train"])


def test_blind_anomaly_holdout_is_fixed_across_seeds_and_session_disjoint():
    from src.ml.data.splits import flight_label_table, make_group_split, session_of

    rows = []
    for i in range(8):
        rows += [{"source_id": f"normal-{i}/flight", "label": "normal"}] * 2
    for i in range(6):
        # Her anomaly oturumunda bir normal kardes de var; final secilirse ikisi
        # birlikte blind holdout'a gitmeli.
        rows += [{"source_id": f"anom-{i}/normal", "label": "normal"}] * 2
        rows += [{"source_id": f"anom-{i}/fault", "label": "altitude_anomaly"}] * 2
    flights = flight_label_table(pd.DataFrame(rows))
    a = make_group_split(flights, seed=0, by_session=True,
                         final_holdout_fraction=0.34)
    b = make_group_split(flights, seed=4, by_session=True,
                         final_holdout_fraction=0.34)
    assert a["final_holdout_anomalous"] == b["final_holdout_anomalous"]
    assert a["final_holdout_anomalous"]
    final_sessions = {session_of(f) for f in a["final_holdout"]}
    assert not final_sessions & {session_of(f) for f in a["train"] + a["val"] + a["test"]}
    assert_no_flight_overlap(a)
