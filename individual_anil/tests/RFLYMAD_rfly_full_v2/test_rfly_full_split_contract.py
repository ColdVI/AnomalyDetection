import pandas as pd

from gecmis_calismalar.rfly_full.contract import _assign_splits


def _rows(count: int) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "canonical_case_id": f"rfly_{index:04d}",
            "domain": "SIL",
            "fault_family": "Motor",
            "environment_condition": "None",
            "object_name": f"SIL_Motor/TestCase_{index}/Log/a.ulg",
        }
        for index in range(count)
    ])


def test_locked_split_does_not_move_when_new_flights_arrive():
    first = _assign_splits(_rows(20)).set_index("canonical_case_id")["split"]
    expanded = _assign_splits(_rows(50)).set_index("canonical_case_id")["split"]
    assert first.to_dict() == expanded.loc[first.index].to_dict()
    assert first.nunique() == 1
