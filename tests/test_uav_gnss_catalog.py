from uav_gnss.catalog import FLIGHT_MODES, _assign_balanced


def _normal_rows():
    rows = []
    for mode in FLIGHT_MODES:
        for index in range(10 + (1 if mode == "hover" else 0)):
            rows.append(
                {
                    "flight_id": f"normal/{mode}/{index:02d}",
                    "class": "normal",
                    "flight_mode": mode,
                    "fault_mode": None,
                }
            )
    return rows


def _fault_rows():
    counts = {
        ("acce", 3): 2,
        ("acce", 4): 2,
        ("circling", 3): 2,
        ("circling", 4): 5,
        ("hover", 3): 5,
        ("hover", 4): 5,
        ("velocity", 3): 2,
        ("velocity", 4): 4,
        ("waypoint", 3): 5,
        ("waypoint", 4): 5,
    }
    rows = []
    for (mode, fault_mode), count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "flight_id": f"fault/{mode}/{fault_mode}/{index:02d}",
                    "class": "gps_fault",
                    "flight_mode": mode,
                    "fault_mode": fault_mode,
                }
            )
    return rows


def test_balanced_roles_match_frozen_contract():
    rows = _normal_rows() + _fault_rows()
    rows.append(
        {
            "flight_id": "quarantine/one",
            "class": "quarantine_magnetometer",
            "flight_mode": "hover",
            "fault_mode": 4,
        }
    )
    assignments = _assign_balanced(rows)
    counts = {}
    for role in assignments.values():
        counts[role] = counts.get(role, 0) + 1
    assert counts == {
        "fit": 20,
        "calibration": 10,
        "development": 23,
        "rehearsal": 15,
        "holdout": 20,
        "quarantine": 1,
    }
    for mode in FLIGHT_MODES:
        for fault_mode in (3, 4):
            cell = [
                row["flight_id"]
                for row in rows
                if row["class"] == "gps_fault"
                and row["flight_mode"] == mode
                and row["fault_mode"] == fault_mode
            ]
            assert sum(assignments[flight_id] == "rehearsal" for flight_id in cell) == 1
            assert sum(assignments[flight_id] == "holdout" for flight_id in cell) == 1

