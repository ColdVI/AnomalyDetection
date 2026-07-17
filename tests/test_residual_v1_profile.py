import pandas as pd

from residual_v1.ingest.common import write_json
from residual_v1.ingest.profile import profile_dataset, stale_segments


def test_stale_segments_and_quarantine(tmp_path):
    silver = tmp_path / "silver"
    flight = silver / "flight_1"
    flight.mkdir(parents=True)
    write_json(
        flight / "flight.json",
        {"dataset": "alfa", "flight_id": "flight_1", "session": "s1"},
    )
    pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "airspeed": [100.0, 100.0, 100.0, 100.0],
        }
    ).to_parquet(flight / "mavros-nav_info-airspeed.parquet", index=False)

    segments = stale_segments(pd.Series([0.0, 1.0, 2.0]), pd.Series([1.0, 1.0, 1.0]))
    assert segments == [{"start_s": 0.0, "end_s": 2.0}]
    summary = profile_dataset(silver, tmp_path / "profile", dataset="alfa")
    assert summary["flight_count"] == 1
    assert summary["quarantine_count"] == 1
