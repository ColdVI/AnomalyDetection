"""Leakage-aware, flight-level raw telemetry handout generation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from gecmis_calismalar.residual_v1.features.physics import finite_difference
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import sha256_file

Progress = Callable[[str], None]
MAX_PLOT_POINTS = 2_000


def split_scope(split: Mapping[str, object]) -> tuple[dict[str, str], list[str]]:
    """Return development/test roles and sealed holdout IDs.

    Holdout IDs are deliberately returned separately so callers cannot process
    them through the same flight loop by accident.
    """

    partitions = split.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("split manifest is missing partitions")
    visible: dict[str, str] = {}
    sealed: list[str] = []
    seen: set[str] = set()
    for role in ("development", "test", "holdout"):
        partition = partitions.get(role)
        if not isinstance(partition, Mapping):
            raise ValueError(f"split manifest is missing {role}")
        flight_ids = partition.get("flight_ids")
        if not isinstance(flight_ids, list) or not all(isinstance(value, str) for value in flight_ids):
            raise ValueError(f"{role}.flight_ids must be a string list")
        overlap = seen.intersection(flight_ids)
        if overlap:
            raise ValueError(f"split roles overlap: {sorted(overlap)[:3]}")
        seen.update(flight_ids)
        if role == "holdout":
            sealed.extend(flight_ids)
        else:
            visible.update({flight_id: role for flight_id in flight_ids})
    return visible, sealed


def downsample(frame: pd.DataFrame, max_points: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    """Select deterministic observed rows without synthesising timestamps."""

    if max_points < 2:
        raise ValueError("max_points must be at least two")
    if len(frame) <= max_points:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return frame.iloc[indices].reset_index(drop=True)


def flight_slug(flight_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "__", flight_id).strip("_.-")
    digest = hashlib.sha256(flight_id.encode("utf-8")).hexdigest()[:8]
    return f"{base[:140]}__{digest}"


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_topic(root: Path, topic: str) -> pd.DataFrame:
    path = root / f"{topic}.parquet"
    if not path.exists():
        return pd.DataFrame({"t": pd.Series(dtype=float)})
    frame = pd.read_parquet(path)
    if "t" not in frame:
        raise ValueError(f"{path}: missing t")
    return frame.sort_values("t", kind="stable").reset_index(drop=True)


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sample_stats(frames: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for topic, frame in frames.items():
        sampled = downsample(frame, max_points=5_000)
        channels: dict[str, dict[str, float | int | None]] = {}
        for column in sampled.columns:
            if column == "t":
                continue
            values = pd.to_numeric(sampled[column], errors="coerce")
            valid = values[np.isfinite(values)]
            channels[column] = {
                "finite_sample_count": int(len(valid)),
                "min": _finite(valid.min()) if not valid.empty else None,
                "median": _finite(valid.median()) if not valid.empty else None,
                "max": _finite(valid.max()) if not valid.empty else None,
            }
        result[topic] = channels
    return result


def _duration(frames: Mapping[str, pd.DataFrame]) -> float:
    maxima = [
        _finite(pd.to_numeric(frame["t"], errors="coerce").max())
        for frame in frames.values()
        if not frame.empty
    ]
    valid = [value for value in maxima if value is not None]
    return max(valid, default=0.0)


def _shade_events(axes: Sequence[plt.Axes], events: Sequence[Mapping[str, object]]) -> None:
    for event_index, event in enumerate(events):
        onset = _finite(event.get("onset_s"))
        end = _finite(event.get("end_s"))
        if onset is None:
            continue
        for axis in axes:
            axis.axvline(onset, color="#d6278b", linestyle="--", linewidth=1.0)
            if end is not None and end >= onset:
                axis.axvspan(onset, end, color="#d6278b", alpha=0.07)
        if event_index == 0:
            axes[0].text(
                onset,
                0.98,
                f" onset {onset:.2f}s",
                transform=axes[0].get_xaxis_transform(),
                color="#a50f5b",
                ha="left",
                va="top",
                fontsize=8,
            )


def _plot_columns(
    axis: plt.Axes,
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    scale: float = 1.0,
    suffix: str = "",
    linestyle: str = "-",
    colors: Sequence[str] | None = None,
) -> None:
    if frame.empty:
        return
    sampled = downsample(frame)
    for index, column in enumerate(columns):
        if column not in sampled:
            continue
        color = colors[index] if colors and index < len(colors) else None
        axis.plot(
            sampled["t"],
            pd.to_numeric(sampled[column], errors="coerce") * scale,
            linewidth=0.75,
            linestyle=linestyle,
            color=color,
            label=f"{column}{suffix}",
        )


def _finalise_axes(
    figure: plt.Figure,
    axes: Sequence[plt.Axes],
    *,
    title: str,
    duration_s: float,
    role: str,
) -> None:
    for axis in axes:
        axis.grid(alpha=0.22)
        if axis.lines:
            axis.legend(loc="upper right", fontsize=7, ncol=3)
        if duration_s > 0:
            axis.set_xlim(0.0, duration_s)
    axes[-1].set_xlabel("flight time (s)")
    figure.suptitle(title, fontsize=11)
    if role == "test":
        figure.text(
            0.5,
            0.005,
            "TEST RAW VIEW — DO NOT USE FOR MODEL, FEATURE OR THRESHOLD TUNING",
            ha="center",
            color="#b2182b",
            fontsize=9,
            weight="bold",
        )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.97))


def render_alfa_flight(root: Path, role: str, output: Path) -> dict:
    metadata = _read_json(root / "flight.json")
    events = _read_json(root / "events.json")
    assert isinstance(metadata, dict) and isinstance(events, list)
    frames = {
        topic: _read_topic(root, topic)
        for topic in (
            "mavros-rc-out",
            "mavros-imu-data",
            "mavros-vfr_hud",
            "mavros-nav_info-airspeed",
            "mavros-nav_info-roll",
        )
    }
    duration_s = _duration(frames)
    topic_rows = {topic: int(len(frame)) for topic, frame in frames.items()}
    stats = _sample_stats(frames)
    figure, axes_array = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    axes = list(axes_array)
    _plot_columns(axes[0], frames["mavros-rc-out"], ("aileron_cmd", "elevator_cmd", "rudder_cmd"))
    axes[0].set_ylabel("surface PWM Δ")
    _plot_columns(
        axes[1],
        frames["mavros-imu-data"],
        ("roll_rate", "pitch_rate", "yaw_rate"),
        scale=180.0 / np.pi,
    )
    axes[1].set_ylabel("angular rate °/s")
    _plot_columns(axes[2], frames["mavros-vfr_hud"], ("throttle_cmd",))
    axes[2].set_ylabel("throttle ratio")
    _plot_columns(axes[3], frames["mavros-nav_info-airspeed"], ("airspeed", "airspeed_cmd"))
    axes[3].set_ylabel("airspeed m/s")
    airspeed = frames["mavros-nav_info-airspeed"]
    if not airspeed.empty and "airspeed" in airspeed:
        derivative = finite_difference(airspeed["t"], airspeed["airspeed"], window_s=0.5)
        derivative_frame = pd.DataFrame({"t": airspeed["t"], "dV_dt": derivative})
        _plot_columns(axes[4], derivative_frame, ("dV_dt",))
    axes[4].axhline(0.0, color="black", linewidth=0.5)
    axes[4].set_ylabel("dV/dt m/s²")
    _shade_events(axes, events)
    _finalise_axes(
        figure,
        axes,
        title=f"ALFA | {role} | {metadata['fault_class']} | {metadata['flight_id']}",
        duration_s=duration_s,
        role=role,
    )
    figure.savefig(output, dpi=95)
    plt.close(figure)
    return {
        "dataset": "alfa",
        "flight_id": metadata["flight_id"],
        "split_role": role,
        "fault_class": metadata["fault_class"],
        "is_anomalous": bool(metadata.get("is_anomalous")),
        "session": metadata.get("session"),
        "duration_s": duration_s,
        "events": events,
        "topic_rows": topic_rows,
        "observed_sample_stats": stats,
        "visual_semantics": "raw natural-rate telemetry; observed-row downsampling only; no model/residual/threshold",
    }


def render_rfly_flight(root: Path, role: str, output: Path) -> dict:
    metadata = _read_json(root / "flight.json")
    events = _read_json(root / "events.json")
    assert isinstance(metadata, dict) and isinstance(events, list)
    frames = {
        topic: _read_topic(root, topic)
        for topic in (
            "vehicle_angular_velocity",
            "vehicle_rates_setpoint",
            "actuator_outputs",
            "sensor_combined",
            "vehicle_local_position",
            "vehicle_local_position_setpoint",
        )
    }
    duration_s = _duration(frames)
    topic_rows = {topic: int(len(frame)) for topic, frame in frames.items()}
    stats = _sample_stats(frames)
    figure, axes_array = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes = list(axes_array)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    _plot_columns(
        axes[0],
        frames["vehicle_rates_setpoint"],
        ("roll_rate_sp", "pitch_rate_sp", "yaw_rate_sp"),
        scale=180.0 / np.pi,
        suffix=" (sp)",
        linestyle="--",
        colors=colors,
    )
    _plot_columns(
        axes[0],
        frames["vehicle_angular_velocity"],
        ("roll_rate", "pitch_rate", "yaw_rate"),
        scale=180.0 / np.pi,
        colors=colors,
    )
    axes[0].set_ylabel("angular rate °/s")
    _plot_columns(
        axes[1],
        frames["actuator_outputs"],
        ("motor_pwm_0", "motor_pwm_1", "motor_pwm_2", "motor_pwm_3"),
    )
    axes[1].set_ylabel("motor PWM")
    _plot_columns(axes[2], frames["sensor_combined"], ("accel_x", "accel_y", "accel_z"))
    axes[2].set_ylabel("acceleration m/s²")
    _plot_columns(
        axes[3],
        frames["vehicle_local_position_setpoint"],
        ("velocity_sp_x", "velocity_sp_y", "velocity_sp_z"),
        suffix=" (sp)",
        linestyle="--",
        colors=colors,
    )
    _plot_columns(
        axes[3],
        frames["vehicle_local_position"],
        ("local_vx", "local_vy", "local_vz"),
        colors=colors,
    )
    axes[3].set_ylabel("local velocity m/s")
    _shade_events(axes, events)
    _finalise_axes(
        figure,
        axes,
        title=f"RflyMAD | {role} | {metadata['fault_class']} | {metadata['flight_id']}",
        duration_s=duration_s,
        role=role,
    )
    figure.savefig(output, dpi=90)
    plt.close(figure)
    return {
        "dataset": "rfly",
        "flight_id": metadata["flight_id"],
        "split_role": role,
        "fault_class": metadata["fault_class"],
        "is_anomalous": bool(metadata.get("is_anomalous")),
        "session": metadata.get("session"),
        "duration_s": duration_s,
        "events": events,
        "topic_rows": topic_rows,
        "observed_sample_stats": stats,
        "visual_semantics": "raw natural-rate telemetry; observed-row downsampling only; no model/residual/threshold",
    }


def _flight_markdown(report: Mapping[str, object]) -> str:
    lines = [
        f"# {report['flight_id']}",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Split role: `{report['split_role']}`",
        f"- Fault class: `{report['fault_class']}`",
        f"- Session: `{report['session']}`",
        f"- Duration covered: `{float(report['duration_s']):.3f} s`",
        f"- Visual: [plot.png](plot.png)",
        "- Semantics: raw natural-rate telemetry; no trained model, residual, z-score, CUSUM or threshold.",
        "",
        "## Events",
        "",
    ]
    events = report["events"]
    if events:
        lines.extend(["| class | onset_s | end_s |", "|---|---:|---:|"])
        for event in events:
            lines.append(
                f"| {event.get('fault_class')} | {event.get('onset_s')} | {event.get('end_s')} |"
            )
    else:
        lines.append("No labelled interval event.")
    lines.extend(["", "## Topic rows", "", "| topic | rows |", "|---|---:|"])
    for topic, rows in sorted(report["topic_rows"].items()):
        lines.append(f"| {topic} | {rows} |")
    if report["split_role"] == "test":
        lines.extend(
            [
                "",
                "> TEST RAW VIEW: do not use this flight for model selection, feature tuning, threshold selection or miss analysis.",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_indexes(output: Path, reports: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "dataset",
        "flight_id",
        "split_role",
        "fault_class",
        "is_anomalous",
        "session",
        "duration_s",
        "plot",
        "report",
    )
    with (output / "flight_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow({field: report.get(field) for field in fields})

    markdown = [
        "# Flight Index",
        "",
        "All visuals are raw telemetry. Test rows are view-only and must not drive tuning.",
        "",
        "| dataset | role | class | flight | visual | report |",
        "|---|---|---|---|---|---|",
    ]
    html_rows = []
    for report in reports:
        markdown.append(
            f"| {report['dataset']} | {report['split_role']} | {report['fault_class']} | "
            f"`{report['flight_id']}` | [PNG]({report['plot']}) | [MD]({report['report']}) |"
        )
        html_rows.append(
            "<article>"
            f"<h3>{html.escape(str(report['flight_id']))}</h3>"
            f"<p>{html.escape(str(report['dataset']))} · {html.escape(str(report['split_role']))} · "
            f"{html.escape(str(report['fault_class']))}</p>"
            f"<a href=\"{html.escape(str(report['report']))}\"><img loading=\"lazy\" "
            f"src=\"{html.escape(str(report['plot']))}\" alt=\"flight plot\"></a>"
            "</article>"
        )
    (output / "FLIGHT_INDEX.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    gallery = """<!doctype html><html><head><meta charset="utf-8"><title>RESIDUAL-V1 Flight Gallery</title>
<style>body{font-family:system-ui;margin:20px;background:#f7f7f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}article{background:white;padding:12px;border:1px solid #ddd}img{width:100%;height:auto}h3{font-size:14px;overflow-wrap:anywhere}p{color:#555}</style>
</head><body><h1>RESIDUAL-V1 Flight Gallery</h1><p>Raw telemetry only. Holdout is sealed.</p><main>"""
    gallery += "\n".join(html_rows)
    gallery += "</main></body></html>\n"
    (output / "GALLERY.html").write_text(gallery, encoding="utf-8")


def _copy_evidence(output: Path, evidence_root: Path, sanity_run: Path, docs: Sequence[Path]) -> None:
    docs_output = output / "source_docs"
    docs_output.mkdir()
    for path in docs:
        shutil.copy2(path, docs_output / path.name)
    sanity_output = output / "checkpoint_task_3_5"
    shutil.copytree(sanity_run, sanity_output)
    run_evidence = []
    for manifest_path in sorted(evidence_root.glob("*/manifest.json")):
        record = {"run": manifest_path.parent.name, "manifest": _read_json(manifest_path)}
        summary_path = manifest_path.parent / "summary.json"
        if summary_path.exists():
            record["summary"] = _read_json(summary_path)
        run_evidence.append(record)
    write_json(output / "run_evidence.json", run_evidence, fail_if_exists=True)

    source_output = output / "implementation_source"
    shutil.copytree(
        Path("residual_v1"),
        source_output / "residual_v1",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for source_group, pattern in (("scripts", "residual_v1_*.py"), ("tests", "test_residual_v1_*.py")):
        destination = source_output / source_group
        destination.mkdir(parents=True)
        for path in sorted(Path(source_group).glob(pattern)):
            shutil.copy2(path, destination / path.name)
    config_output = source_output / "configs"
    config_output.mkdir()
    for path in sorted(Path("configs").glob("residual_v1_*.json")):
        shutil.copy2(path, config_output / path.name)
    shutil.copy2(Path("requirements.txt"), source_output / "requirements.txt")


def _package_manifest(output: Path) -> dict:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "file_count_excluding_manifest": len(files),
        "total_bytes_excluding_manifest": sum(record["size_bytes"] for record in files),
        "files": files,
    }


def create_zip(output: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent))


def chunked(values: Sequence[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _read_flight_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _flight_pdf(
    handout: Path,
    records: Sequence[dict[str, str]],
    destination: Path,
    *,
    part_label: str,
) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with PdfPages(destination) as pdf:
        for page, record in enumerate(records, start=1):
            image = plt.imread(handout / record["plot"])
            figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            axis = figure.add_axes((0.01, 0.055, 0.98, 0.935))
            axis.imshow(image)
            axis.axis("off")
            figure.text(
                0.012,
                0.018,
                f"{part_label} · page {page}/{len(records)} · {record['flight_id']}",
                fontsize=6.5,
                color="#333333",
            )
            figure.text(
                0.988,
                0.018,
                "RAW TELEMETRY — NO MODEL/RESIDUAL/THRESHOLD · HOLDOUT SEALED",
                fontsize=6.5,
                color="#8c2d04",
                ha="right",
            )
            pdf.savefig(figure, dpi=95)
            plt.close(figure)


def _combined_context(handout: Path) -> str:
    sections = [
        "RESIDUAL-V1 CLAUDE REVIEW CONTEXT",
        "=" * 80,
        "",
        "RECOMMENDED REVIEW REQUEST",
        "Review Task 1.1 through the mandatory Task 3.5 STOP. Check leakage locks,",
        "physical channel semantics, raw flight visuals, data-coverage limitations,",
        "and whether the evidence is sufficient to authorize Task 4.1. Do not treat",
        "test raw plots as tuning evidence and do not request/open holdout telemetry.",
        "",
    ]
    primary = (
        handout / "README_FOR_CLAUDE.md",
        handout / "SUMMARY.json",
        handout / "SEALED_HOLDOUT.json",
        handout / "checkpoint_task_3_5" / "SANITY_REPORT.md",
        handout / "checkpoint_task_3_5" / "sanity_metrics.json",
        handout / "run_evidence.json",
        handout / "source_docs" / "RESIDUAL_V1_DENEY_TASARIMI.md",
        handout / "source_docs" / "RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md",
    )
    source_files = sorted((handout / "implementation_source").rglob("*"))
    for path in [*primary, *[value for value in source_files if value.is_file()]]:
        relative = path.relative_to(handout).as_posix()
        sections.extend(
            [
                "",
                "=" * 80,
                f"FILE: {relative}",
                "=" * 80,
                path.read_text(encoding="utf-8", errors="replace"),
            ]
        )
    return "\n".join(sections)


def create_claude_upload_set(handout: Path, destination: Path) -> dict:
    """Create files that respect Claude chat-upload and visual-PDF limits."""

    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    records = _read_flight_index(handout / "flight_index.csv")
    alfa = [record for record in records if record["dataset"] == "alfa"]
    rfly = [record for record in records if record["dataset"] == "rfly"]
    pdf_specs: list[tuple[str, list[dict[str, str]], str]] = [
        ("01_ALFA_ALL_VISIBLE_FLIGHTS.pdf", alfa, "ALFA all visible flights")
    ]
    for part, values in enumerate(chunked(rfly, 87), start=1):
        pdf_specs.append(
            (
                f"{part + 1:02d}_RFLY_VISIBLE_FLIGHTS_PART_{part}.pdf",
                values,
                f"RflyMAD visible flights part {part}",
            )
        )
    page_counts: dict[str, int] = {}
    for filename, values, label in pdf_specs:
        if len(values) >= 100:
            raise ValueError(f"visual PDF must stay under 100 pages: {filename}")
        _flight_pdf(handout, values, destination / filename, part_label=label)
        page_counts[filename] = len(values)

    shutil.copy2(handout / "flight_index.csv", destination / "06_CLAUDE_FLIGHT_INDEX.csv")
    context_path = destination / "07_CLAUDE_CONTEXT_AND_SOURCE.txt"
    context_path.write_text(_combined_context(handout), encoding="utf-8")
    guide = [
        "RESIDUAL-V1 CLAUDE UPLOAD SET",
        "",
        "Upload all files in this directory to one Claude chat/project.",
        "Start by asking Claude to read 07_CLAUDE_CONTEXT_AND_SOURCE.txt and",
        "06_CLAUDE_FLIGHT_INDEX.csv, then inspect the five PDFs by fault class.",
        "",
        "Leakage warning: test pages are raw-view only and watermarked. Holdout",
        "telemetry is absent and must remain sealed. The work is at Task 3.5 STOP;",
        "no G1/G2 model, z-scaling, CUSUM calibration or final evaluation exists.",
    ]
    (destination / "00_UPLOAD_INSTRUCTIONS.txt").write_text("\n".join(guide) + "\n", encoding="utf-8")

    file_records = []
    max_bytes = 30 * 1024 * 1024
    for path in sorted(destination.iterdir()):
        if not path.is_file() or path.name == "08_UPLOAD_MANIFEST.json":
            continue
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"Claude file exceeds 30 MiB safety ceiling: {path.name} ({size})")
        file_records.append(
            {
                "name": path.name,
                "size_bytes": size,
                "sha256": sha256_file(path),
                "pdf_pages": page_counts.get(path.name),
            }
        )
    manifest = {
        "schema_version": 1,
        "upload_file_count_including_manifest": len(file_records) + 1,
        "claude_constraints": {
            "max_file_bytes_used_for_validation": max_bytes,
            "max_files_per_chat": 20,
            "visual_pdf_page_rule": "strictly fewer than 100 pages",
        },
        "holdout_opened_count": 0,
        "visible_flight_count": len(records),
        "pdf_flight_page_count": sum(page_counts.values()),
        "files": file_records,
    }
    write_json(destination / "08_UPLOAD_MANIFEST.json", manifest, fail_if_exists=True)
    return manifest


def build_handout(
    *,
    output: Path,
    alfa_root: Path,
    rfly_root: Path,
    alfa_split_path: Path,
    rfly_split_path: Path,
    evidence_root: Path,
    sanity_run: Path,
    docs: Sequence[Path],
    progress: Progress = print,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    split_paths = {"alfa": alfa_split_path, "rfly": rfly_split_path}
    silver_roots = {"alfa": alfa_root, "rfly": rfly_root}
    renderers = {"alfa": render_alfa_flight, "rfly": render_rfly_flight}
    visible_by_dataset: dict[str, dict[str, str]] = {}
    sealed_by_dataset: dict[str, list[str]] = {}
    split_payloads: dict[str, dict] = {}
    for dataset, split_path in split_paths.items():
        payload = _read_json(split_path)
        if not isinstance(payload, dict):
            raise ValueError(f"{split_path}: expected object")
        visible, sealed = split_scope(payload)
        visible_by_dataset[dataset] = visible
        sealed_by_dataset[dataset] = sealed
        split_payloads[dataset] = payload

    sealed_payload = {
        "policy": "holdout telemetry was not opened or plotted",
        "opened_holdout_count": 0,
        "datasets": {
            dataset: {
                "flight_count": len(flight_ids),
                "flight_ids_from_split_manifest_only": flight_ids,
                "class_counts_from_split_manifest": split_payloads[dataset]["partitions"]["holdout"]["class_counts"],
            }
            for dataset, flight_ids in sealed_by_dataset.items()
        },
    }
    write_json(output / "SEALED_HOLDOUT.json", sealed_payload, fail_if_exists=True)

    reports: list[dict] = []
    total_visible = sum(len(values) for values in visible_by_dataset.values())
    processed = 0
    for dataset in ("alfa", "rfly"):
        root = silver_roots[dataset]
        for flight_id, role in visible_by_dataset[dataset].items():
            processed += 1
            destination = output / "flights" / dataset / flight_slug(flight_id)
            destination.mkdir(parents=True)
            report = renderers[dataset](root / Path(flight_id), role, destination / "plot.png")
            relative_root = destination.relative_to(output).as_posix()
            report["plot"] = f"{relative_root}/plot.png"
            report["report"] = f"{relative_root}/REPORT.md"
            report["metrics"] = f"{relative_root}/metrics.json"
            write_json(destination / "metrics.json", report, fail_if_exists=True)
            (destination / "REPORT.md").write_text(_flight_markdown(report), encoding="utf-8")
            reports.append(report)
            if processed == 1 or processed % 10 == 0 or processed == total_visible:
                progress(f"handout {processed}/{total_visible}: {dataset}/{flight_id}")

    reports.sort(key=lambda value: (value["dataset"], value["split_role"], value["flight_id"]))
    _write_indexes(output, reports)
    _copy_evidence(output, evidence_root, sanity_run, docs)
    role_counts = Counter((report["dataset"], report["split_role"]) for report in reports)
    class_counts = Counter((report["dataset"], report["fault_class"]) for report in reports)
    summary = {
        "status": "TASK_3_5_STOP; no Phase-D model training performed",
        "visible_flight_count": len(reports),
        "sealed_holdout_flight_count": sum(len(values) for values in sealed_by_dataset.values()),
        "holdout_opened_count": 0,
        "role_counts": {f"{dataset}:{role}": count for (dataset, role), count in sorted(role_counts.items())},
        "class_counts": {f"{dataset}:{fault}": count for (dataset, fault), count in sorted(class_counts.items())},
        "flight_report_count": len(reports),
        "flight_plot_count": len(reports),
        "test_policy": "raw view only; prohibited for tuning/model selection/threshold selection/miss analysis",
        "visual_policy": "natural-rate observed samples; deterministic row selection only; no synthetic timestamps",
    }
    write_json(output / "SUMMARY.json", summary, fail_if_exists=True)
    readme = [
        "# RESIDUAL-V1 — Claude Handout",
        "",
        "This package is a complete raw-flight handout at the mandatory Task 3.5 STOP.",
        "It is not a final model evaluation: G1/G2 training, residual z-scaling, CUSUM, calibration and holdout scoring have not run.",
        "",
        f"- Flight visuals/reports: **{len(reports)}**",
        f"- Sealed holdout flights: **{summary['sealed_holdout_flight_count']}** (telemetry not read)",
        "- [Flight index](FLIGHT_INDEX.md)",
        "- [Browser gallery](GALLERY.html)",
        "- [Machine summary](SUMMARY.json)",
        "- [Sealed holdout inventory](SEALED_HOLDOUT.json)",
        "- [Task 3.5 checkpoint](checkpoint_task_3_5/SANITY_REPORT.md)",
        "- [Run provenance](run_evidence.json)",
        "- [Implementation snapshot](implementation_source/)",
        "",
        "## Current evidence",
        "",
        "- ALFA engine checkpoint: throttle drops to zero at onset; first 5 s median dV/dt = -1.281 m/s²; 78.7% of derivatives are negative.",
        "- Three aggressive normal ALFA raw R1 views have command/response Spearman rho = 0.672, 0.843 and 0.829.",
        "- ALFA September/October frozen airspeed masks learned R1-R5 feature rows; R6 remains available. This limitation must stay visible in later claims.",
        "- Corrected ALFA R1 uses the mean of observed left/right aileron servo outputs (RCOut channels4/5); the earlier channels0-derived artifacts are preserved as failed evidence.",
        "",
        "## Leakage rules",
        "",
        "- Development plots may be inspected for design work.",
        "- Test plots are included because the user explicitly requested a complete handout, but are watermarked and must not drive any tuning decision.",
        "- Holdout telemetry is absent. Opening it still requires the one-shot holdout command and explicit human approval.",
    ]
    (output / "README_FOR_CLAUDE.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _package_manifest(output)
    write_json(output / "PACKAGE_MANIFEST.json", manifest, fail_if_exists=True)
    return summary
