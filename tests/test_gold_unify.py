import io

import pandas as pd
import pytest

from src.common.minio_io import list_layer_objects, write_silver
from src.gold.unify import GOLD_COLUMNS, GOLD_NAME, clear_gold_before_unify, stream_unify, unify


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
        "is_military": [True],
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


def test_unify_maps_is_military_for_adsblol_sources(fake_minio_client):
    """2026-07-10 karari: is_military artik Gold semasinda -- SADECE adsb.lol
    kaynakli tabloda (dbFlags biti var) doldurulur, digerlerinde her zaman
    null olmali (COLUMN_MAPS'te None -- bkz. modul yorumu 'adsb.lol disi
    kaynak -- dbFlags yok')."""
    write_silver(_adsblol_hist_silver_df(), "adsblol_hist", client=fake_minio_client)

    gold = unify(fake_minio_client, source_types=("adsblol_hist",))

    assert bool(gold.iloc[0]["is_military"]) is True


def test_unify_is_military_is_null_for_non_adsblol_sources(fake_minio_client):
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)
    write_silver(_uav_attack_silver_df(), "uav_attack", client=fake_minio_client)

    gold = unify(fake_minio_client, source_types=("alfa", "uav_attack"))

    assert gold["is_military"].isna().all()


def test_unify_raises_for_unregistered_source_type(fake_minio_client):
    with pytest.raises(ValueError):
        unify(fake_minio_client, source_types=("not_a_real_source",))


def test_stream_unify_raises_for_unregistered_source_type(fake_minio_client):
    with pytest.raises(ValueError):
        stream_unify(fake_minio_client, source_types=("not_a_real_source",))


def test_clear_gold_before_unify_removes_prior_unified_parts(fake_minio_client):
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)
    stream_unify(fake_minio_client, source_types=("alfa",))
    assert len(list_layer_objects(fake_minio_client, "gold", GOLD_NAME)) == 1

    removed = clear_gold_before_unify(fake_minio_client)

    assert removed == 1
    assert list_layer_objects(fake_minio_client, "gold", GOLD_NAME) == []


def test_clear_gold_before_unify_returns_zero_when_nothing_to_clear(fake_minio_client):
    assert clear_gold_before_unify(fake_minio_client) == 0


def test_stream_unify_rerun_does_not_double_count_rows(fake_minio_client):
    """Regression test: stream_unify() must clear prior unified/ output before
    rewriting, otherwise a second run leaves the first run's parts in place and
    every downstream reader double-counts rows."""
    write_silver(_alfa_silver_df(), "alfa", client=fake_minio_client)

    first_total = stream_unify(fake_minio_client, source_types=("alfa",))
    second_total = stream_unify(fake_minio_client, source_types=("alfa",))

    assert first_total == second_total == 2
    gold_bucket = fake_minio_client.buckets.get("gold", {})
    # rerun must not leave the first run's parts alongside the new ones
    assert len(list_layer_objects(fake_minio_client, "gold", GOLD_NAME)) == 1
    total_rows_in_bucket = sum(
        len(pd.read_parquet(io.BytesIO(gold_bucket[name])))
        for name in gold_bucket
        if name.startswith(f"{GOLD_NAME}/")
    )
    assert total_rows_in_bucket == 2
