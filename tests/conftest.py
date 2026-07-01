"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.common.fakes import FakeMinioClient


@pytest.fixture
def fake_minio_client() -> FakeMinioClient:
    return FakeMinioClient()
