"""RFLY-0 RflyMAD integration tests."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.ingestion.rflymad_downloader import is_essential_file
from src.ml.data.splits import build_split_manifest, session_of
from src.silver.parse_rflymad import (
    SOURCE_TYPE,
    build_rflymad_silver_from_directory,
    discover_cases,
    infer_label_from_case,
)


def _workspace_tmp() -> Path:
    root = Path(".tmp_test_rflymad") / uuid.uuid4().hex
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def test_rflymad_downloader_keeps_testinfo_xlsx_but_skips_bulk_xlsx():
    assert is_essential_file("Real-Motor/hover/10_1/log_4/log_4.ulg")
    assert is_essential_file("Real-Motor/hover/10_1/TestInfo_2023-05-17_15-29.xlsx")
    assert not is_essential_file("Real-Motor/hover/10_1/TrueData/UAVState_data.xlsx")


def test_rflymad_label_mapping_is_fixed_by_subset_and_fault_family():
    assert infer_label_from_case("Real-NoFault/hover/001_1/log_0") == "normal"
    assert infer_label_from_case("Real-No_Fault/hover/001_1/log_0") == "normal"
    assert infer_label_from_case("Real-Motor/acce/406_1/log_26") == "motor_fault"
    assert (
        infer_label_from_case("Real-Sensors/acce-GPS/447_1/log_71")
        == "sensor_acce_gps_fault"
    )


def test_rflymad_session_key_uses_case_root_not_whole_subset():
    assert (
        session_of("Real-Motor/acce/406_1/log_26_2023-6-2")
        == "Real-Motor/acce/406_1"
    )
    assert (
        session_of("Real-Sensors/acce-GPS/447_1/log_71")
        == "Real-Sensors/acce-GPS/447_1"
    )


def test_discover_cases_reads_manifest_and_filters_subsets():
    root = _workspace_tmp()
    try:
        bronze = root / "rflymad"
        ulg = bronze / "Real-Motor/acce/406_1/log_26/log_26.ulg"
        info = bronze / "Real-Motor/acce/406_1/TestInfo_2023-06-02_10-08.xlsx"
        ulg.parent.mkdir(parents=True)
        info.parent.mkdir(parents=True, exist_ok=True)
        ulg.write_bytes(b"ulg")
        info.write_bytes(b"xlsx")
        (bronze / "manifest.json").write_text(json.dumps({
            "dataset": "xianglile/rflymad",
            "cases": {
                "Real-Motor/acce/406_1/log_26": {
                    "subdataset": "Real-Motor",
                    "files": {
                        "Real-Motor/acce/406_1/log_26/log_26.ulg": {
                            "bytes": 3,
                            "sha256": "abc",
                        }
                    },
                }
            },
        }), encoding="utf-8")

        cases = discover_cases(bronze, subsets=("Real-Motor",))

        assert len(cases) == 1
        assert cases[0]["source_id"] == "Real-Motor/acce/406_1/log_26"
        assert cases[0]["label"] == "motor_fault"
        assert cases[0]["test_info"] == "Real-Motor/acce/406_1/TestInfo_2023-06-02_10-08.xlsx"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_build_rflymad_silver_relabels_reused_px4_parser_output():
    root = _workspace_tmp()
    try:
        bronze = root / "rflymad"
        ulg = bronze / "Real-Motor/acce/406_1/log_26/log_26.ulg"
        ulg.parent.mkdir(parents=True)
        ulg.write_bytes(b"ulg")
        (bronze / "manifest.json").write_text(json.dumps({
            "cases": {
                "Real-Motor/acce/406_1/log_26": {
                    "subdataset": "Real-Motor",
                    "files": {"Real-Motor/acce/406_1/log_26/log_26.ulg": {"bytes": 3}},
                }
            }
        }), encoding="utf-8")
        parsed = pd.DataFrame({
            "timestamp": [1],
            "source_type": ["uav_sead"],
            "source_id": ["old"],
            "label": ["old"],
            "timestamp_utc": [0.000001],
            "timestamp_is_real_utc": [False],
        })

        with patch("src.silver.parse_rflymad.parse_px4_ulg_bytes", return_value=parsed):
            silver, report = build_rflymad_silver_from_directory(bronze, subsets=("Real-Motor",))

        assert report["parsed_cases"] == 1
        assert silver["source_type"].tolist() == [SOURCE_TYPE]
        assert silver["source_id"].tolist() == ["Real-Motor/acce/406_1/log_26"]
        assert silver["label"].tolist() == ["motor_fault"]
        assert silver["_source_type"].tolist() == [SOURCE_TYPE]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rflymad_split_gets_session_holdout_from_first_day():
    rows = []
    for i in range(5):
        rows.append({"source_id": f"Real-NoFault/hover/n{i}/log_0", "label": "normal"})
    for i in range(5):
        rows.append({"source_id": f"Real-Motor/hover/m{i}/log_0", "label": "motor_fault"})
    table = pd.DataFrame(rows)

    manifest = build_split_manifest({"rflymad": table}, quotas={"rflymad": (1, 1)})
    source = manifest["sources"]["rflymad"]
    split = source["splits"]["split_00"]

    assert source["split_unit"] == "session"
    assert source["evaluation_status"] == "blind-final-holdout"
    assert split["final_holdout"]
    assert not set(split["final_holdout"]) & (set(split["train"]) | set(split["val"]) | set(split["test"]))
