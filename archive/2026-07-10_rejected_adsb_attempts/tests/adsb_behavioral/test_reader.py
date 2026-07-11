from pathlib import Path

import pytest

from src.adsb_behavioral.reader import archive_date


def test_archive_date_from_real_filename_convention():
    assert archive_date(Path("v2026.03.01-planes-readsb-prod-0.tar")) == "2026-03-01"


def test_archive_date_rejects_unknown_name():
    with pytest.raises(ValueError):
        archive_date("data.tar")
