"""UAV-SEAD Silver parser: Bronze'daki .ulg dosyalari -> PX4 Silver tablosu.

parse_uav_attack.py ile AYNI kolon semantigi uretir (timestamp, lat/lon/alt,
roll/pitch/yaw_deg, GPS saglik kolonlari, vel_m_s, batarya) -- boylece
src/ml/features/uav_attack_features.py'nin build_px4_features'i degismeden
calisir ve leave-dataset-out deneyi ayni feature uzayinda yapilabilir.

Etiket Bronze'daki `uav_sead/labels.json`'dan gelir (ucus-seviyesi sinif;
mapping.json'daki sinyal-bazli zaman araliklari da ileride nokta-bazli
degerlendirme icin labels.json icinde saklanir).

PX4 .ulg okuma: pyulog.ULog. 2018 donemi loglari eski uORB surumleri tasir;
alan adlari (vehicle_global_position.lat, vehicle_gps_position.vel_m_s,
battery_status.voltage_v) bu donemde de aynidir -- eksik topic'ler NaN kalir.

Kullanim:
    python -m src.silver.parse_uav_sead [--local-out data/silver/uav_sead_silver.parquet]
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    download_raw_bytes,
    get_minio_client,
    list_layer_objects,
    write_silver,
)
from src.common.provenance import add_provenance
from src.silver.parse_uav_attack import quat_to_euler_deg

logger = logging.getLogger(__name__)

SOURCE_TYPE = "uav_sead"

_GPS_FIELDS = ["s_variance_m_s", "eph", "epv", "hdop", "vdop", "noise_per_ms",
               "jamming_indicator", "vel_m_s", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s",
               "cog_rad", "fix_type", "satellites_used", "time_utc_usec"]
_BATTERY_FIELDS = ["voltage_v", "remaining", "current_a"]
# H13 (docs/ML1_BULGULAR_VE_HATALAR.md): SEAD'in state-estimation anomalileri icin
# kaynak-uygun sinyaller. estimator_status test oranlari PX4'un KENDI innovation-
# tutarlilik kontrolu (ALFA nav_info/errors'un PX4 karsiligi -- hazir residual).
_ESTIMATOR_FIELDS = ["vel_test_ratio", "pos_test_ratio", "hgt_test_ratio",
                     "mag_test_ratio", "tas_test_ratio", "hagl_test_ratio", "beta_test_ratio",
                     "innovation_check_flags", "gps_check_fail_flags", "filter_fault_flags",
                     "solution_status_flags", "control_mode_flags", "timeout_flags",
                     "pre_flt_fail", "pos_horiz_accuracy", "pos_vert_accuracy",
                     "vibe[0]", "vibe[1]", "vibe[2]"]
_INNOV_FIELDS = ["vel_pos_innov[0]", "vel_pos_innov[1]", "vel_pos_innov[2]",
                 "vel_pos_innov[3]", "vel_pos_innov[4]", "vel_pos_innov[5]", "heading_innov"]
_LOCAL_FIELDS = [
    "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "z_deriv",
    "eph", "epv", "evh", "evv", "dist_bottom", "dist_bottom_rate",
    "dist_bottom_valid", "xy_valid", "z_valid", "v_xy_valid", "v_z_valid",
    "xy_global", "z_global", "xy_reset_counter", "z_reset_counter",
    "vxy_reset_counter", "vz_reset_counter", "ref_alt",
]
_ATTITUDE_SETPOINT_FIELDS = ["roll_body", "pitch_body", "yaw_body", "thrust"]
_RATE_SETPOINT_FIELDS = ["roll", "pitch", "yaw", "thrust"]
_ACTUATOR_CONTROL_FIELDS = [f"control[{i}]" for i in range(4)]
_ACTUATOR_OUTPUT_FIELDS = [f"output[{i}]" for i in range(16)] + ["noutputs"]
MERGE_TOLERANCE_US = 200_000  # parse_uav_attack ile ayni


def _topic_df(ulog, topic: str, fields: list[str]) -> pd.DataFrame | None:
    """ULog'dan bir topic'in istenen alanlarini DataFrame olarak ceker."""
    try:
        data = ulog.get_dataset(topic).data
    except (KeyError, IndexError, StopIteration):
        return None
    if "timestamp" not in data:
        return None
    cols = {"timestamp": data["timestamp"]}
    for f in fields:
        if f in data:
            cols[f] = data[f]
    if len(cols) == 1:
        return None
    return pd.DataFrame(cols).sort_values("timestamp")


def _merge(base: pd.DataFrame, extra: pd.DataFrame | None, rename: dict | None = None) -> pd.DataFrame:
    if extra is None:
        return base
    if rename:
        extra = extra.rename(columns=rename)
    return pd.merge_asof(base.sort_values("timestamp"), extra,
                          on="timestamp", direction="nearest", tolerance=MERGE_TOLERANCE_US)


def parse_ulg_bytes(data: bytes, source_id: str, label: str) -> pd.DataFrame | None:
    """Tek bir .ulg -> parse_uav_attack semantiginde duz tablo."""
    from pyulog import ULog

    try:
        ulog = ULog(BytesIO(data))
    except Exception:
        logger.exception("%s: ULog okunamadi", source_id)
        return None

    local_position = _topic_df(ulog, "vehicle_local_position", _LOCAL_FIELDS)
    base = _topic_df(ulog, "vehicle_global_position", ["lat", "lon", "alt", "eph", "epv"])
    if base is None or not {"lat", "lon", "alt"}.issubset(base.columns):
        # Ic mekan ucuslari (External Position sinifi: mocap/harici konum) GPS
        # tasimaz -- vehicle_local_position omurgaya alinir. x/y metre (NED),
        # z asagi-pozitif; vx/vy/vz'den hiz dogrudan gelir.
        base = None if local_position is None else local_position.copy()
        if base is None or not {"x", "y", "z"}.issubset(base.columns):
            logger.warning("%s: vehicle_global_position da vehicle_local_position da yok, atlandi", source_id)
            return None
        base = base.rename(columns={"x": "pos_x_m", "y": "pos_y_m"})
        base["alt"] = -base.pop("z")
        if {"vx", "vy"}.issubset(base.columns):
            base["vel_m_s"] = np.sqrt(base["vx"] ** 2 + base["vy"] ** 2)
        if "vz" in base.columns:
            base["vertical_rate_mps"] = -base["vz"]
        base = base.drop(columns=[c for c in ("vx", "vy", "vz") if c in base.columns])

    att = _topic_df(ulog, "vehicle_attitude", ["q[0]", "q[1]", "q[2]", "q[3]"])
    if att is not None and all(c in att.columns for c in ["q[0]", "q[1]", "q[2]", "q[3]"]):
        roll, pitch, yaw = quat_to_euler_deg(att["q[0]"], att["q[1]"], att["q[2]"], att["q[3]"])
        att = att.assign(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)[
            ["timestamp", "roll_deg", "pitch_deg", "yaw_deg"]]
    else:
        att = None
    base = _merge(base, att)

    base = _merge(base, _topic_df(ulog, "battery_status", _BATTERY_FIELDS))
    base = _merge(base, _topic_df(ulog, "vehicle_gps_position", _GPS_FIELDS),
                  rename={"eph": "raw_gps_eph", "epv": "raw_gps_epv"})

    # H13: EKF tutarlilik sinyalleri (state-estimation anomalilerinin dogal imzasi)
    base = _merge(base, _topic_df(ulog, "estimator_status", _ESTIMATOR_FIELDS))
    innov = _topic_df(ulog, "ekf2_innovations", _INNOV_FIELDS)
    base = _merge(base, innov)
    if all(f"vel_pos_innov[{i}]" in base.columns for i in range(6)):
        base["ekf_vel_innov_mag"] = np.sqrt(sum(base[f"vel_pos_innov[{i}]"] ** 2 for i in range(3)))
        base["ekf_pos_innov_mag"] = np.sqrt(sum(base[f"vel_pos_innov[{i}]"] ** 2 for i in range(3, 6)))
        # ML-9: pooled yatay+dikey buyukluklari korurken kategori-eslesmeli
        # dikey innovation'lari ayrica sakla. Mutlak deger, yon yerine
        # tutarsizlik siddetini temsil eder.
        base["ekf_vertical_vel_innov"] = base["vel_pos_innov[2]"].abs()
        base["ekf_alt_innov"] = base["vel_pos_innov[5]"].abs()
        base = base.drop(columns=[f"vel_pos_innov[{i}]" for i in range(6)])

    # ML-7: estimator bir olcumu reddettiginde test ratio dusuk/temiz gorunebilir.
    # Red bayraklari ile bagimsiz global-local-baro tutarliligi bu kor noktayi kapatir.
    if local_position is not None:
        local_rename = {
            "x": "local_x_m", "y": "local_y_m", "z": "local_z_down_m",
            "vx": "local_vx_mps", "vy": "local_vy_mps", "vz": "local_vz_down_mps",
            "ax": "local_ax_mps2", "ay": "local_ay_mps2", "az": "local_az_mps2",
            **{name: f"local_{name}" for name in _LOCAL_FIELDS
               if name not in {"x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"}},
        }
        base = _merge(base, local_position, rename=local_rename)
        if "local_z_down_m" in base:
            base["local_alt_m"] = -base["local_z_down_m"]
        if {"local_vx_mps", "local_vy_mps"}.issubset(base.columns):
            base["local_speed_mps"] = np.sqrt(
                base["local_vx_mps"] ** 2 + base["local_vy_mps"] ** 2)
        if "local_vz_down_mps" in base:
            base["local_vertical_rate_mps"] = -base["local_vz_down_mps"]

    base = _merge(
        base, _topic_df(ulog, "sensor_baro", ["altitude", "pressure", "error_count"]),
        rename={"altitude": "baro_alt_m", "pressure": "baro_pressure_pa",
                "error_count": "baro_error_count"})
    base = _merge(
        base, _topic_df(ulog, "vehicle_attitude_setpoint", _ATTITUDE_SETPOINT_FIELDS),
        rename={"roll_body": "roll_sp_rad", "pitch_body": "pitch_sp_rad",
                "yaw_body": "yaw_sp_rad", "thrust": "thrust_sp"})
    base = _merge(
        base, _topic_df(ulog, "vehicle_rates_setpoint", _RATE_SETPOINT_FIELDS),
        rename={"roll": "roll_rate_sp_rad_s", "pitch": "pitch_rate_sp_rad_s",
                "yaw": "yaw_rate_sp_rad_s", "thrust": "rate_thrust_sp"})
    base = _merge(
        base, _topic_df(ulog, "actuator_controls_0", _ACTUATOR_CONTROL_FIELDS),
        rename={f"control[{i}]": name for i, name in enumerate(
                ["actuator_roll_cmd", "actuator_pitch_cmd",
                 "actuator_yaw_cmd", "actuator_thrust_cmd"])})
    base = _merge(
        base, _topic_df(ulog, "actuator_outputs", _ACTUATOR_OUTPUT_FIELDS),
        rename={
            **{f"output[{i}]": f"actuator_output_{i}" for i in range(16)},
            "noutputs": "actuator_noutputs",
        })

    if "vel_d_m_s" in base.columns:
        base["vertical_rate_mps"] = -base["vel_d_m_s"]

    base["source_type"] = SOURCE_TYPE
    base["source_id"] = source_id
    base["label"] = label
    if "time_utc_usec" in base.columns and base["time_utc_usec"].notna().any():
        base["timestamp_utc"] = base["time_utc_usec"] / 1e6
        base["timestamp_is_real_utc"] = True
    else:
        base["timestamp_utc"] = base["timestamp"] / 1e6
        base["timestamp_is_real_utc"] = False
    return base


def build_uav_sead_silver(client: ObjectStoreClient) -> pd.DataFrame:
    """Bronze'daki tum uav_sead/*.ulg dosyalarini labels.json etiketleriyle parse eder."""
    try:
        labels = json.loads(download_raw_bytes(client, "uav_sead/labels.json").decode("utf-8"))
    except Exception:
        logger.error("bronze/uav_sead/labels.json yok -- once uav_sead_downloader calistirilmali")
        return pd.DataFrame()

    objects = {n for n in list_layer_objects(client, "bronze", SOURCE_TYPE) if n.endswith(".ulg")}
    results = []
    for i, (flight, meta) in enumerate(sorted(labels.items()), 1):
        object_name = meta["object_name"]
        if object_name not in objects:
            logger.warning("[%d/%d] %s: bronze objesi yok (%s), atlandi", i, len(labels), flight, object_name)
            continue
        data = download_raw_bytes(client, object_name)
        df = parse_ulg_bytes(data, source_id=flight, label=meta["label"])
        if df is not None and len(df):
            results.append(df)
            logger.info("[%d/%d] %s: %d satir, label=%s", i, len(labels), flight, len(df), meta["label"])

    if not results:
        logger.error("Hicbir UAV-SEAD ucusu parse edilemedi.")
        return pd.DataFrame()
    full = pd.concat(results, ignore_index=True)
    logger.info("UAV-SEAD Silver: %d satir, %d ucus", len(full), full["source_id"].nunique())
    return add_provenance(full, source_type=SOURCE_TYPE, source_file="uav_sead/*.ulg")


def build_uav_sead_silver_from_directory(raw_dir: str | Path) -> pd.DataFrame:
    """Yereldeki labels.json + ULog'lari MinIO gerektirmeden parse et.

    Downloader nesne yollarini Windows'ta ``2018-05-24__19_20_57.ulg`` gibi
    duzlestirdigi icin hem object_name'in goreli yolunu hem basename'ini dener.
    """
    raw_dir = Path(raw_dir)
    labels = json.loads((raw_dir / "labels.json").read_text(encoding="utf-8"))
    results = []
    for i, (flight, meta) in enumerate(sorted(labels.items()), 1):
        relative = Path(meta["object_name"])
        candidates = [raw_dir / relative, raw_dir / relative.name]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            logger.warning("[%d/%d] %s: yerel ULog yok", i, len(labels), flight)
            continue
        frame = parse_ulg_bytes(path.read_bytes(), source_id=flight, label=meta["label"])
        if frame is not None and len(frame):
            results.append(frame)
            logger.info("[%d/%d] %s: %d satir", i, len(labels), flight, len(frame))
    if not results:
        return pd.DataFrame()
    full = pd.concat(results, ignore_index=True)
    return add_provenance(full, source_type=SOURCE_TYPE, source_file=str(raw_dir / "*.ulg"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UAV-SEAD Bronze .ulg -> Silver")
    parser.add_argument("--local-out", default="data/silver/uav_sead_silver.parquet")
    parser.add_argument("--local-bronze-dir", default=None,
                        help="MinIO yerine labels.json ve .ulg iceren yerel klasor")
    args = parser.parse_args()

    if args.local_bronze_dir:
        silver = build_uav_sead_silver_from_directory(args.local_bronze_dir)
    else:
        client = get_minio_client()
        silver = build_uav_sead_silver(client)
    if silver.empty:
        logger.error("Nothing to write: UAV-SEAD Silver is empty")
        return

    if not args.local_bronze_dir:
        uri = write_silver(silver, SOURCE_TYPE, client=client)
        logger.info("Wrote UAV-SEAD Silver -> %s", uri)
    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(args.local_out, index=False)
        logger.info("Local copy written: %s", args.local_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
