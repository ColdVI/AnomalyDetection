from gecmis_calismalar.rfly_full.dl_worker import split_of


def test_flight_split_is_stable_and_valid():
    assert split_of("SIL-NoFault/example/TestCase_1") == split_of("SIL-NoFault/example/TestCase_1")
    assert split_of("SIL-NoFault/example/TestCase_1") in {"train", "val", "test"}
