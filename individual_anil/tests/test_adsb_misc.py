"""ADS-B envanter/manifest/IF/truth-v2 korpus testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from __future__ import annotations

import gzip

import json

import tarfile

from pathlib import Path

from adsb.inventory import list_trace_members, profile_tar

import hashlib

import subprocess

import sys

import pandas as pd

import pytest

from adsb.run_manifest import (
    InputSpec,
    ManifestError,
    build_split_contract,
    create_immutable_run_manifest,
    inspect_input_file,
    make_deterministic_split,
    sha256_json,
    silver_row_count_provenance,
)

import numpy as np

from adsb.models.isolation_forest_residual import (
    RESIDUAL_CHANNELS,
    fit_isolation_forest_residual,
    score_isolation_forest_residual,
)

from scripts.adsb_build_synthetic_truth_v2_corpus import (
    _active_mask,
    _annotate_recipe_vectorized,
)



# ===== kaynak: test_adsb_inventory =====

def _make_trace_json(icao: str, *, n_rows: int = 5, category: str = "A3") -> bytes:
    trace = []
    for i in range(n_rows):
        trace.append([
            i * 10,       # t_offset
            40.0 + i * 0.01,  # lat
            29.0 + i * 0.01,  # lon
            1000.0 + i * 50,  # alt
            100.0,            # ground_speed
            45.0,              # track
            0,                 # flags
            5.0,               # vertical_rate
            {"flight": "TEST123", "category": category, "squawk": "1200"} if i == 0 else None,
            "adsb_icao",
            1050.0,
            5.0,
            110.0,
            2.0,
        ])
    data = {"icao": icao, "r": "N12345", "timestamp": 1_700_000_000, "trace": trace}
    return gzip.compress(json.dumps(data).encode())


def _write_synthetic_tar(path: Path, *, n_aircraft: int = 10) -> Path:
    with tarfile.open(path, "w") as tar:
        for k in range(n_aircraft):
            icao = f"a{k:05x}"
            content = _make_trace_json(icao, category="B6" if k == 0 else "A3")
            member = tarfile.TarInfo(name=f"traces/{icao[:2]}/trace_full_{icao}.json.gz")
            member.size = len(content)
            import io
            tar.addfile(member, io.BytesIO(content))
    return path


def test_list_trace_members_finds_all(tmp_path):
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=7)
    members = list_trace_members(tar_path)
    assert len(members) == 7
    assert all("traces/" in m for m in members)


def test_profile_tar_basic_stats(tmp_path):
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=10)
    profile = profile_tar(tar_path, n_samples=10)

    assert profile.total_trace_members == 10
    assert profile.sampled_members == 10
    assert profile.sampled_rows == 50  # 10 aircraft * 5 rows
    assert profile.parse_errors == 0
    assert profile.trace_row_lengths == {14: 50}
    assert profile.file_field_presence["icao"] == 10
    assert profile.file_field_presence["timestamp"] == 10
    assert profile.ac_dict_field_presence["flight"] == 10
    assert profile.ac_dict_field_presence["category"] == 10
    assert profile.category_counts == {"B6": 1, "A3": 9}
    # her ucus icin ardisik t_offset farki = 10 (4 aralik * 10 ucak = 40 kayit)
    assert profile.sampling_interval_s == {10: 40}


def test_profile_tar_respects_sample_size(tmp_path):
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=20)
    profile = profile_tar(tar_path, n_samples=5)
    assert profile.total_trace_members == 20
    assert profile.sampled_members == 5
    assert profile.sampled_rows == 25


def test_profile_tar_as_dict_is_json_serializable(tmp_path):
    import json as _json
    tar_path = _write_synthetic_tar(tmp_path / "sample.tar", n_aircraft=3)
    profile = profile_tar(tar_path, n_samples=3)
    serialized = _json.dumps(profile.as_dict())
    assert "sample.tar" in serialized



# ===== kaynak: test_adsb_run_manifest =====

REPO_ROOT = Path(__file__).parent.parent


def _parquet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "flight_id": ["f1", "f1", "f2"],
            "timestamp_utc": [1.0, 2.0, 3.0],
            "alt": [1000.0, None, 1200.0],
        }
    ).to_parquet(path, index=False)
    return path


def _create(run_dir: Path, input_path: Path) -> Path:
    return create_immutable_run_manifest(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        inputs=[InputSpec(input_path, "fit")],
        splits={"fit": ["f2", "f1"], "development": ["f3"]},
        split_algorithm="test_explicit_v1",
        split_seed=7,
        synthetic_flight_ids=["s2", "s1"],
        config={"window": 12, "features": ["a", "b"]},
    )


def test_inspect_parquet_records_bytes_hash_footer_rows_and_schema(tmp_path: Path):
    input_path = _parquet(tmp_path / "input.parquet")
    record = inspect_input_file(input_path, role="fit", repo_root=REPO_ROOT)

    assert record["bytes"] == input_path.stat().st_size
    assert record["sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert record["footer_rows"] == 3
    assert record["format"] == "parquet"
    assert len(record["schema_sha256"]) == 64
    assert record["schema_hash_contract"].endswith("v1")


def test_deterministic_split_is_input_and_weight_order_invariant():
    ids = [f"f{i}" for i in range(11)]
    first = make_deterministic_split(
        ids,
        {"fit": 0.8, "development": 0.2},
        seed=19,
        excluded_flight_ids=["f10"],
    )
    second = make_deterministic_split(
        reversed(ids),
        {"development": 2, "fit": 8},
        seed=19,
        excluded_flight_ids=["f10"],
    )

    assert first == second
    assert len(first["fit"]) == 8
    assert len(first["development"]) == 2
    assert "f10" not in first["fit"] + first["development"]


def test_split_contract_stores_sorted_ids_and_order_invariant_hash():
    first = build_split_contract(
        {"fit": ["f2", "f1"], "development": ["d1"]},
        algorithm="explicit_v1",
        seed=0,
    )
    second = build_split_contract(
        {"development": ["d1"], "fit": ["f1", "f2"]},
        algorithm="explicit_v1",
        seed=0,
    )

    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["splits"]["fit"]["flight_ids"] == ["f1", "f2"]
    assert first["splits"]["fit"]["flight_ids_sha256"] == sha256_json(["f1", "f2"])


def test_split_contract_rejects_flight_id_overlap():
    with pytest.raises(ManifestError, match="occurs in both"):
        build_split_contract(
            {"fit": ["same"], "development": ["same"]},
            algorithm="explicit_v1",
            seed=0,
        )


def test_manifest_is_complete_and_run_directory_is_immutable(tmp_path: Path):
    input_path = _parquet(tmp_path / "silver" / "input.parquet")
    run_dir = tmp_path / "runs" / "run-001"
    manifest_path = _create(run_dir, input_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == run_dir / "run_manifest.json"
    assert manifest["run_id"] == "run-001"
    assert manifest["immutability"]["fail_if_exists"] is True
    assert len(manifest["git"]["commit"]) == 40
    assert isinstance(manifest["git"]["dirty"], bool)
    assert manifest["config"]["sha256"] == sha256_json(manifest["config"]["value"])
    assert manifest["split_contract"]["splits"]["fit"]["flight_ids"] == ["f1", "f2"]
    assert manifest["synthetic_guard"]["status"] == "passed"
    assert manifest["synthetic_guard"]["excluded_flight_ids"] == ["s1", "s2"]
    assert manifest["synthetic_guard"]["overlap_count"] == 0

    with pytest.raises(FileExistsError, match="immutable"):
        _create(run_dir, input_path)


def test_manifest_rejects_synthetic_path_before_opening_it(tmp_path: Path):
    # The file intentionally does not exist: the path-role guard must fail first,
    # proving that a protected synthetic path is not opened or hashed.
    synthetic_path = tmp_path / "synthetic" / "missing.parquet"
    with pytest.raises(ManifestError, match="Synthetic path is forbidden"):
        create_immutable_run_manifest(
            run_dir=tmp_path / "run",
            repo_root=REPO_ROOT,
            inputs=[InputSpec(synthetic_path, "fit")],
            splits={"fit": ["f1"]},
            split_algorithm="explicit_v1",
            split_seed=0,
            synthetic_flight_ids=[],
            config={},
        )
    assert not (tmp_path / "run").exists()


def test_manifest_rejects_synthetic_source_id_in_fit(tmp_path: Path):
    input_path = _parquet(tmp_path / "silver" / "input.parquet")
    with pytest.raises(ManifestError, match="overlap protected splits"):
        create_immutable_run_manifest(
            run_dir=tmp_path / "run",
            repo_root=REPO_ROOT,
            inputs=[InputSpec(input_path, "fit")],
            splits={"fit": ["f1"]},
            split_algorithm="explicit_v1",
            split_seed=0,
            synthetic_flight_ids=["f1"],
            config={},
        )
    assert not (tmp_path / "run").exists()


def test_silver_row_count_discrepancy_is_explicit_and_unresolved():
    provenance = silver_row_count_provenance()
    assert provenance["footer_observed_rows"] == 256_155_009
    assert provenance["documented_rows"] == 256_150_550
    assert provenance["delta_rows"] == 4_459
    assert provenance["status"] == "unresolved_do_not_silently_correct"


def test_cli_creates_manifest_without_input_discovery(tmp_path: Path):
    input_path = _parquet(tmp_path / "silver" / "input.parquet")
    config_path = tmp_path / "config.json"
    split_path = tmp_path / "splits.json"
    exclusions_path = tmp_path / "excluded.json"
    config_path.write_text('{"window": 12}', encoding="utf-8")
    split_path.write_text(
        json.dumps(
            {
                "algorithm": "explicit_cli_v1",
                "seed": 3,
                "splits": {"fit": ["f1"], "development": ["f2"]},
            }
        ),
        encoding="utf-8",
    )
    exclusions_path.write_text('["s1"]', encoding="utf-8")
    run_dir = tmp_path / "runs" / "cli-run"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "adsb_create_run_manifest.py"),
            "--run-dir",
            str(run_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--input",
            f"fit={input_path}",
            "--config",
            str(config_path),
            "--splits",
            str(split_path),
            "--synthetic-flight-ids",
            str(exclusions_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = Path(completed.stdout.strip())
    assert manifest_path == run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_contract"]["algorithm"] == "explicit_cli_v1"
    assert manifest["inputs"][0]["footer_rows"] == 3



# ===== kaynak: test_adsb_isolation_forest_residual =====

def _natural_fit_frame(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(0.0, 1.0, n) for c in RESIDUAL_CHANNELS})


def test_fit_rejects_synthetic_flag():
    df = _natural_fit_frame()
    with pytest.raises(ValueError):
        fit_isolation_forest_residual(df, contains_synthetic=True)


def test_fit_excludes_zero_mad_channel_without_floor():
    df = _natural_fit_frame()
    df["heading_residual"] = 0.0  # sabit -> MAD=0
    _, scaler, active_channels = fit_isolation_forest_residual(df)
    assert "heading_residual" not in active_channels
    assert "heading_residual" in scaler.excluded_channels_


def test_normal_rows_score_lower_than_extreme_outlier():
    df = _natural_fit_frame(n=1000)
    model, scaler, _ = fit_isolation_forest_residual(df)

    normal_probe = pd.DataFrame({c: [0.0] for c in RESIDUAL_CHANNELS})
    outlier_probe = pd.DataFrame({c: [50.0] for c in RESIDUAL_CHANNELS})  # tum kanallarda asiri

    normal_score = score_isolation_forest_residual(model, scaler, normal_probe).iloc[0]
    outlier_score = score_isolation_forest_residual(model, scaler, outlier_probe).iloc[0]
    assert outlier_score > normal_score


def test_nan_channel_row_scores_nan_not_dropped():
    df = _natural_fit_frame()
    model, scaler, _ = fit_isolation_forest_residual(df)

    probe = pd.DataFrame({c: [0.0, np.nan] for c in RESIDUAL_CHANNELS})
    scores = score_isolation_forest_residual(model, scaler, probe)
    assert len(scores) == 2
    assert np.isfinite(scores.iloc[0])
    assert np.isnan(scores.iloc[1])


def test_predict_binary_output_never_used():
    """Sozlesme: skorlama yalniz score_samples uzerinden, predict() kullanilmaz --
    bu, IsolationForest'in kendi contamination-tabanli ikili esigine bagimli olmadigimizi
    dogrudan test eder (skor fonksiyonu -1/1 degil surekli deger donduruyor mu)."""
    df = _natural_fit_frame()
    model, scaler, _ = fit_isolation_forest_residual(df)
    probe = pd.DataFrame({c: [0.0, 1.0, 2.0] for c in RESIDUAL_CHANNELS})
    scores = score_isolation_forest_residual(model, scaler, probe)
    assert scores.nunique() > 2  # ikili degil, surekli skor


def test_fit_is_deterministic_given_seed():
    df = _natural_fit_frame()
    model1, scaler1, _ = fit_isolation_forest_residual(df)
    model2, scaler2, _ = fit_isolation_forest_residual(df)
    probe = pd.DataFrame({c: [0.5] for c in RESIDUAL_CHANNELS})
    s1 = score_isolation_forest_residual(model1, scaler1, probe)
    s2 = score_isolation_forest_residual(model2, scaler2, probe)
    np.testing.assert_allclose(s1.to_numpy(), s2.to_numpy())



# ===== kaynak: test_adsb_truth_v2_corpus =====

def test_non_dropout_active_range_is_onset_to_end():
    assert _active_mask(10, "track_frozen").tolist() == [False] * 5 + [True] * 5


def test_dropout_active_range_is_exact_legacy_random_block_not_onset_to_end():
    active = _active_mask(10, "altitude_dropout")
    positions = np.flatnonzero(active)
    assert len(positions) == int((10 - 5) * 0.3)
    assert positions[0] >= 5
    assert positions[-1] < 10
    assert not active[-1] or len(positions) == 5


def test_vectorized_pair_annotation_keeps_exact_block_per_flight():
    clean = pd.DataFrame(
        {
            "flight_id": np.repeat(["a", "b"], 10),
            "timestamp_utc": np.tile(np.arange(10, dtype=float), 2),
            "alt": 1000.0,
            "label": None,
        }
    )
    corrupt = clean.copy()
    expected = np.tile(_active_mask(10, "altitude_dropout"), 2)
    corrupt.loc[expected, "alt"] = np.nan
    truth = _annotate_recipe_vectorized(clean, corrupt, "altitude_dropout")
    assert truth["injection_active"].tolist() == expected.tolist()
    assert truth["observable_changed"].tolist() == expected.tolist()
    assert truth.groupby("flight_id")["event_id"].nunique().eq(1).all()

