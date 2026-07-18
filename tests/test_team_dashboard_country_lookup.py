"""Tests for team_dashboard/country_lookup.py -- the incremental
Gold-aircraft -> ISO2-country cache builder."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common.fakes import FakeMinioClient
from src.common.minio_io import write_gold
from team_dashboard.country_lookup import build_country_lookup


def _seed_gold_part(client, *, source_id, source_type="adsblol_historical"):
    df = pd.DataFrame({
        "timestamp_utc": [1_700_000_000.0],
        "source_id": [source_id],
        "source_type": [source_type],
    })
    write_gold(df, "unified", client=client)


@pytest.fixture
def fake_client():
    c = FakeMinioClient()
    c.make_bucket("gold")
    return c


def test_build_country_lookup_resolves_known_hex(fake_client, tmp_path):
    # 4baa62 -> Turkey (bkz. src/common/hex_country.py / ICAOHexRange.csv)
    _seed_gold_part(fake_client, source_id="4baa62")
    lookup_path = tmp_path / "lookup.parquet"
    scanned_path = tmp_path / "scanned.json"

    result = build_country_lookup(fake_client, lookup_path=lookup_path, scanned_parts_path=scanned_path)

    assert dict(zip(result["source_id"], result["country_iso2"])) == {"4baa62": "TR"}
    assert json.loads(scanned_path.read_text()) != []


def test_build_country_lookup_ignores_non_adsblol_sources(fake_client, tmp_path):
    _seed_gold_part(fake_client, source_id="4baa62", source_type="alfa")
    lookup_path = tmp_path / "lookup.parquet"
    scanned_path = tmp_path / "scanned.json"

    result = build_country_lookup(fake_client, lookup_path=lookup_path, scanned_parts_path=scanned_path)

    assert result.empty


def test_build_country_lookup_second_run_only_scans_new_parts(fake_client, tmp_path):
    _seed_gold_part(fake_client, source_id="4baa62")  # Turkey
    lookup_path = tmp_path / "lookup.parquet"
    scanned_path = tmp_path / "scanned.json"

    build_country_lookup(fake_client, lookup_path=lookup_path, scanned_parts_path=scanned_path)
    scanned_after_first = set(json.loads(scanned_path.read_text()))
    assert len(scanned_after_first) == 1

    # Yeni bir parca eklenir (US hex) -- ikinci calistirma SADECE bunu taramali,
    # ilkinden gelen "4baa62 -> TR" eslesmesini KORUMALI.
    _seed_gold_part(fake_client, source_id="a133c7")  # US
    result = build_country_lookup(fake_client, lookup_path=lookup_path, scanned_parts_path=scanned_path)

    mapping = dict(zip(result["source_id"], result["country_iso2"]))
    assert mapping == {"4baa62": "TR", "a133c7": "US"}
    scanned_after_second = set(json.loads(scanned_path.read_text()))
    assert len(scanned_after_second) == 2
    assert scanned_after_first.issubset(scanned_after_second)


def test_build_country_lookup_fresh_ignores_existing_manifest(fake_client, tmp_path):
    _seed_gold_part(fake_client, source_id="4baa62")
    lookup_path = tmp_path / "lookup.parquet"
    scanned_path = tmp_path / "scanned.json"

    build_country_lookup(fake_client, lookup_path=lookup_path, scanned_parts_path=scanned_path)
    result = build_country_lookup(fake_client, lookup_path=lookup_path, scanned_parts_path=scanned_path, fresh=True)

    assert dict(zip(result["source_id"], result["country_iso2"])) == {"4baa62": "TR"}
