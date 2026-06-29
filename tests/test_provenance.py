import pandas as pd

from src.common.provenance import PROVENANCE_COLUMNS, add_provenance


def test_add_provenance_preserves_source_data_without_mutating_input():
    original = pd.DataFrame({"lat": [41.0], "raw_name": ["unchanged"]})

    result = add_provenance(original, "alfa", "flight/example.csv")

    assert list(original.columns) == ["lat", "raw_name"]
    assert result.loc[0, "lat"] == 41.0
    assert result.loc[0, "raw_name"] == "unchanged"
    assert result.loc[0, "_source_type"] == "alfa"
    assert result.loc[0, "_source_file"] == "flight/example.csv"
    assert result.loc[0, "_schema_version"] == "bronze_v1"
    assert result.loc[0, "_ingest_ts_utc"].endswith("Z")
    assert all(column in result.columns for column in PROVENANCE_COLUMNS)


def test_add_provenance_uses_one_timestamp_for_the_batch():
    result = add_provenance(pd.DataFrame({"value": [1, 2, 3]}), "adsblol_rt", "api")
    assert result["_ingest_ts_utc"].nunique() == 1


def test_add_provenance_handles_empty_dataframe():
    result = add_provenance(pd.DataFrame(columns=["raw"]), "uav_attack", "empty.csv")
    assert result.empty
    assert all(column in result.columns for column in PROVENANCE_COLUMNS)
