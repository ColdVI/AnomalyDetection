import pandas as pd

from src.common.minio_io import write_silver
from src.gold.unify import GOLD_COLUMNS, unify


def _alfa_silver_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_utc": [1.0, 2.0],
        "lat": [40.0, 40.001],
        "lon": [29.0, 29.001],
        "alt": [100.0, 101.0],
        "velocity_measured": [12.0, 12.5],
        "yaw_measured": [10.0, 11.0],
        "source_type": ["alfa", "alfa"],
        "source_id": ["seq1", "seq1"],
        "label": ["normal", "normal"],
    })


def _uav_attack_silver_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_utc": [5.0, 6.0],
        "lat": [41.0, 41.001],
        "lon": [30.0, 30.001],
        "alt": [200.0, 199.0],
        "yaw_deg": [90.0, 91.0],
        "source_type": ["uav_attack", "uav_attack"],
        "source_id": ["log1", "log1"],
        "label": ["gps_spoofing", "gps_spoofing"],
    })


def _adsblol_hist_silver_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_utc": [9.0],
        "lat": [39.0],
        "lon": [35.0],
        "alt": [3000.0],
        "ground_speed_ms": [120.0],
        "track_deg": [270.0],
        "vertical_rate_ms": [2.5],
        "source_type": ["adsblol_hist"],
        "source_id": ["abc123"],
    })


def test_unify_aligns_every_source_to_7_plus_3_schema(fake_minio_client):
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)
    write_silver(_uav_attack_silver_df(), "uav_attack", client=fake_minio_client)
    write_silver(_adsblol_hist_silver_df(), "adsblol_hist", client=fake_minio_client)

    gold = unify(fake_minio_client)

    assert list(gold.columns) == GOLD_COLUMNS
    assert set(gold["source_type"]) == {"alfa", "uav_attack", "adsblol_hist"}
    assert len(gold) == 5


def test_unify_maps_alfa_columns_correctly(fake_minio_client):
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)

    gold = unify(fake_minio_client, source_types=("alfa",))

    row = gold.iloc[0]
    assert row["velocity_mps"] == 12.0
    assert row["heading_deg"] == 10.0
    assert pd.isna(row["vertical_rate_mps"])
    assert row["label"] == "normal"


def test_unify_leaves_missing_fields_null_for_adsblol(fake_minio_client):
    write_silver(_adsblol_hist_silver_df(), "adsblol_hist", client=fake_minio_client)

    gold = unify(fake_minio_client, source_types=("adsblol_hist",))

    row = gold.iloc[0]
    assert row["velocity_mps"] == 120.0
    assert row["vertical_rate_mps"] == 2.5
    assert pd.isna(row["label"])


def test_unify_leaves_velocity_null_for_uav_attack_known_gap(fake_minio_client):
    write_silver(_uav_attack_silver_df(), "uav_attack", client=fake_minio_client)

    gold = unify(fake_minio_client, source_types=("uav_attack",))

    row = gold.iloc[0]
    assert pd.isna(row["velocity_mps"])
    assert row["heading_deg"] == 90.0


def test_unify_skips_sources_with_no_silver_data(fake_minio_client):
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)

    gold = unify(fake_minio_client, source_types=("alfa", "uav_attack", "adsblol_hist", "adsblol_rt"))

    assert set(gold["source_type"]) == {"alfa"}


def test_unify_returns_empty_frame_with_gold_columns_when_nothing_available(fake_minio_client):
    gold = unify(fake_minio_client)

    assert gold.empty
    assert list(gold.columns) == GOLD_COLUMNS
