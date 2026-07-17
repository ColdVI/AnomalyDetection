from residual_v1.ingest.splits import split_flights


def _flights():
    rows = []
    for session in range(12):
        rows.append(
            {
                "flight_id": f"normal_{session}",
                "session": f"session_{session}",
                "fault_class": "normal",
            }
        )
        rows.append(
            {
                "flight_id": f"engine_{session}",
                "session": f"session_{session}",
                "fault_class": "engine",
            }
        )
    rows.extend(
        {
            "flight_id": f"rudder_{index}",
            "session": f"rare_session_{index}",
            "fault_class": "rudder",
        }
        for index in range(4)
    )
    return rows


def test_split_is_session_isolated_deterministic_and_stratified():
    first = split_flights(_flights(), seed=11)
    second = split_flights(_flights(), seed=11)
    assert first == second
    partitions = first["partitions"]
    session_sets = {name: set(value["sessions"]) for name, value in partitions.items()}
    assert session_sets["development"].isdisjoint(session_sets["test"])
    assert session_sets["development"].isdisjoint(session_sets["holdout"])
    assert session_sets["test"].isdisjoint(session_sets["holdout"])
    assert all(session.startswith("rare_session_") for session in session_sets["development"] if session.startswith("rare_"))
    assert not any(session.startswith("rare_session_") for session in session_sets["test"] | session_sets["holdout"])
    for partition in ("development", "test", "holdout"):
        assert partitions[partition]["class_counts"]["engine"] > 0
