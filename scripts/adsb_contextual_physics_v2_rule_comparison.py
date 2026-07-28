"""Write the Phase-G contextual-v2 versus simple-rule comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.contextual_events import COMMON_EVENT_COLUMNS  # noqa: E402
from adsb.simple_anomaly import ALTITUDE_EVENT_COLUMNS, ROUTE_EVENT_COLUMNS  # noqa: E402

REPRESENTATIVE_BUDGETS = (0.1, 5.0, 50.0, 500.0)


class ComparisonContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonContractError(f"Expected JSON object: {path}")
    return value


def _fmt_percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _fmt_number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _point_at(points: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    matches = [row for row in points if float(row["pareto_v"]) == float(budget)]
    if len(matches) != 1:
        raise ComparisonContractError(f"Expected one frozen point for V={budget}")
    return matches[0]


def _model_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recipe, recipe_result in report["results"].items():
        if "profiles" in recipe_result:
            channel = recipe_result["channel"]
            for profile_name, profile in recipe_result["profiles"].items():
                rows.append(
                    {
                        "recipe": recipe,
                        "detector": (
                            "model"
                            if profile["mode"] == "instant"
                            else "persistence_v2"
                        ),
                        "channel_profile": f"{channel}/{profile_name}",
                        "points": profile["points"],
                    }
                )
        else:
            rows.append(
                {
                    "recipe": recipe,
                    "detector": "CUSUM",
                    "channel_profile": "east_north_velocity_residual/accumulation",
                    "points": recipe_result["points"],
                }
            )
    return rows


def _verify_contracts(
    eval_report: dict[str, Any], rule_summary: dict[str, Any]
) -> None:
    if eval_report.get("candidate_namespace") != "contextual_physics_v2":
        raise ComparisonContractError("Unexpected evaluation namespace")
    if eval_report.get("threshold_selection_performed_on_truth_v2") is not False:
        raise ComparisonContractError("Truth-v2 threshold-selection flag is not false")
    grid = list(map(float, eval_report["budget_grid"]))
    missing = sorted(set(REPRESENTATIVE_BUDGETS) - set(grid))
    if missing:
        raise ComparisonContractError(f"Representative frozen budgets missing: {missing}")
    if rule_summary.get("status") != "discovery_only":
        raise ComparisonContractError("Simple-rule result is not discovery-only")
    if rule_summary.get("ground_truth_available") is not False:
        raise ComparisonContractError("Simple-rule ground-truth flag unexpectedly true")
    expected = list(COMMON_EVENT_COLUMNS)
    if list(ALTITUDE_EVENT_COLUMNS[:6]) != expected:
        raise ComparisonContractError("Altitude event schema prefix differs")
    if list(ROUTE_EVENT_COLUMNS[:6]) != expected:
        raise ComparisonContractError("Route event schema prefix differs")


def _render(eval_report: dict[str, Any], rule_summary: dict[str, Any]) -> str:
    _verify_contracts(eval_report, rule_summary)
    lines = [
        "# ADS-B contextual_physics_v2 — model/CUSUM/persistence_v2 ile basit kural turu karşılaştırması",
        "",
        "> Tarih: 2026-07-27",
        "> Faz G; eşik veya bütçe seçimi değildir.",
        "",
        "## Sonuçların doğru okuma birimi",
        "",
        "Bu iki çalışma aynı veri ve ground-truth rejiminde değildir. Basit kural turu "
        "100 doğal uçuşluk keşif örneğidir ve ground truth içermez; contextual-v2 ise "
        "mevcut 8.910-uçuş truth-v2 corpusunda enjekte olay recall'ü ile, eşlenik temiz "
        "uçuşlarda doğal alarm yükünü ölçer. Bu nedenle aşağıdaki tablolar yan yana "
        "bağlam sağlar; doğrudan bir kazanan sıralaması değildir.",
        "",
        "## Ortak event şeması",
        "",
        "Model, persistence_v2 ve CUSUM alarm emisyonları 60 saniyelik episode "
        "birleştirmesinden sonra `adsb.contextual_events.contextual_alarm_events` "
        "ile aşağıdaki ortak prefix'e dönüştürülür:",
        "",
        "`event_id, flight_id, start_time, end_time, duration_s, n_samples`",
        "",
        "Bu prefix `adsb.simple_anomaly` içindeki hem irtifa hem rota event tablolarıyla "
        "birebir aynıdır. Detector/channel/profile/budget/threshold/recipe alanları "
        "ortak prefix'in arkasına eklenir; temel alanların anlamı değiştirilmez.",
        "",
        "## Basit kural turu — doğal keşif örneği",
        "",
        "| Kural | Değerlendirilebilir uçuş | Triggerlı uçuş | Event | Doğrulanmış anomaly | Ana bağlam |",
        "|---|---:|---:|---:|---:|---|",
        (
            f"| İrtifa sapması | {rule_summary['altitude']['evaluable_flights']} | "
            f"{rule_summary['altitude']['triggered_flights']} "
            f"({_fmt_percent(rule_summary['altitude']['triggered_flight_rate'])}) | "
            f"{rule_summary['altitude']['events']} | "
            f"{rule_summary['altitude']['manual_review_confirmed_anomalies']} | "
            "Faz sınırı / meşru seviye değişimi |"
        ),
        (
            f"| Rota sapması | {rule_summary['route']['evaluable_flights']} | "
            f"{rule_summary['route']['triggered_flights']} "
            f"({_fmt_percent(rule_summary['route']['triggered_flight_rate'])}) | "
            f"{rule_summary['route']['events']} | 0 | Düşük-hız bearing kararsızlığı |"
        ),
        "",
        "## contextual_physics_v2 — dondurulmuş bütçe noktaları",
        "",
        "| Recipe | Detector / kanal-profili | V=0.1 recall / temiz yük | V=5 recall / temiz yük | V=50 recall / temiz yük | V=500 recall / temiz yük |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in _model_rows(eval_report):
        cells = []
        for budget in REPRESENTATIVE_BUDGETS:
            point = _point_at(row["points"], budget)
            cells.append(
                f"{_fmt_percent(point['event_recall'])} / "
                f"{_fmt_number(point['paired_clean_natural_burden_per_hour'])} ep/saat"
            )
        lines.append(
            f"| {row['recipe']} | {row['detector']} / {row['channel_profile']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Karşılaştırmalı yorum",
            "",
            "- Kural turundaki event sayıları anomaly recall değildir; ground truth yoktur "
            "ve elle doğrulanmış anomaly sayısı sıfırdır.",
            "- contextual-v2 satırlarında recall yalnız observable-eligible enjekte olaylar "
            "üzerindedir; temiz yük aynı uçuşların enjeksiyonsuz eşlerinde episode/saat "
            "birimindedir.",
            "- V=0.1, 5, 50 ve 500 noktaları sonuç görüldükten sonra seçilmedi; önceden "
            "dondurulmuş 11-noktalı ızgaranın sabit temsilcileridir. Ara noktalar makine "
            "okunur evaluation artifact'ında aynen korunur.",
            "- Bu rapor threshold, epoch, grid, channel share veya persistence parametresi "
            "değiştirmez ve operasyonel başarı iddiası üretmez.",
            "",
            "## Kaynaklar",
            "",
            "- `docs/ADSB_BASIT_ANOMALI_KARSILASTIRMA_20260722.md`",
            "- `artifacts/adsb/simple_anomaly_20260722/summary.json`",
            "- Faz E `truth_v2_eval_report.json`",
            "- `adsb/contextual_events.py`",
            "",
        ]
    )
    return "\n".join(lines)


def run(eval_report_path: Path, rule_summary_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(output_path)
    text = _render(_load_json(eval_report_path), _load_json(rule_summary_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=Path(
            "artifacts/adsb/runs/20260724_contextual_physics_v2_truth_v2_eval_v1/"
            "truth_v2_eval_report.json"
        ),
    )
    parser.add_argument(
        "--rule-summary",
        type=Path,
        default=Path("artifacts/adsb/simple_anomaly_20260722/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ADSB_CONTEXTUAL_PHYSICS_V2_KURAL_KARSILASTIRMA_20260724.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = run(args.eval_report.resolve(), args.rule_summary.resolve(), args.output.resolve())
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
