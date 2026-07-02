"""UAV-SEAD alt kumesini HuggingFace'ten indirip Bronze'a yukler.

Dataset: aykutkabaoglu/uav-flight-anomaly-dataset (CC BY 4.0) -- 1396 etiketli
PX4 .ulg ucus logu, 29.9 GB. Tamami staj kapsaminda gereksiz: sinif basina
kota ile kucuk, dengeli bir alt kume secilir (yalnizca TEK-sinifli ucuslar --
cok-etiketli ucuslar degerlendirmeyi bulanikladigi icin atlanir).

mapping.json ucus -> {class, ranges} etiketlerini verir; secilen alt kumenin
etiketleri Bronze'a `uav_sead/labels.json` olarak birlikte yazilir (ADR-003:
Bronze ham dosya + yanina etiket sozlugu, parse Silver'da).

Kullanim:
    python -m src.ingestion.uav_sead_downloader [--normal 20] [--per-class 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from collections import defaultdict

from src.common.minio_io import get_minio_client, write_bronze_bytes

logger = logging.getLogger(__name__)

REPO = "aykutkabaoglu/uav-flight-anomaly-dataset"
MAPPING_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/mapping.json"
ULG_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/ulg_files/{{flight}}.ulg"

# UAV-SEAD sinif adi -> bizim etiket sozlugumuz (splits.NORMAL_LABELS ile uyumlu).
CLASS_TO_LABEL = {
    "Normal": "normal",
    "Mechanical": "mechanical_fault",
    "Global Position": "global_position_anomaly",
    "Altitude": "altitude_anomaly",
    "External Position": "external_position_anomaly",
}


def fetch_mapping() -> dict:
    with urllib.request.urlopen(MAPPING_URL, timeout=120) as r:
        return json.load(r)


def select_flights(mapping: dict, *, n_normal: int, n_per_class: int) -> dict[str, dict]:
    """Sinif basina kotayla tek-sinifli ucuslari secer (deterministik: sirali gezinme)."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for flight, meta in sorted(mapping.items()):
        classes = sorted({a["class"] for a in meta.get("annotations", [])})
        if len(classes) != 1 or classes[0] not in CLASS_TO_LABEL:
            continue  # cok-sinifli veya Uncategorized -- atla
        by_class[classes[0]].append(flight)

    selected: dict[str, dict] = {}
    for cls, flights in by_class.items():
        quota = n_normal if cls == "Normal" else n_per_class
        for flight in flights[:quota]:
            annotations = mapping[flight]["annotations"]
            selected[flight] = {
                "label": CLASS_TO_LABEL[cls],
                "class": cls,
                "ranges": [a.get("ranges", []) for a in annotations],
            }
    for cls in CLASS_TO_LABEL:
        n = sum(1 for v in selected.values() if v["class"] == cls)
        logger.info("secim: %s -> %d ucus", cls, n)
    return selected


def download_and_upload(selected: dict[str, dict], client) -> dict[str, dict]:
    """Secilen .ulg'leri indirir, Bronze'a `uav_sead/<flight>.ulg` olarak yukler.

    Indirilemeyen ucus loglanip atlanir (dataset agaci ile mapping.json arasinda
    ufak tutarsizliklar olabiliyor); basarili olanlar dondurulur.
    """
    ok: dict[str, dict] = {}
    for i, (flight, meta) in enumerate(sorted(selected.items()), 1):
        url = ULG_URL.format(flight=flight)
        object_name = f"uav_sead/{flight.replace('/', '__')}.ulg"
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
        except Exception as exc:
            logger.warning("[%d/%d] %s indirilemedi (%s), atlandi", i, len(selected), flight, exc)
            continue
        write_bronze_bytes(data, object_name, client=client)
        ok[flight] = {**meta, "object_name": object_name, "size_bytes": len(data)}
        logger.info("[%d/%d] %s (%.1f MB) -> bronze/%s", i, len(selected), flight, len(data) / 1e6, object_name)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV-SEAD alt kumesi -> Bronze")
    parser.add_argument("--normal", type=int, default=20, help="Normal ucus kotasi")
    parser.add_argument("--per-class", type=int, default=10, help="Anomali sinifi basina kota")
    args = parser.parse_args()

    client = get_minio_client()
    mapping = fetch_mapping()
    logger.info("mapping.json: %d etiketli ucus", len(mapping))

    selected = select_flights(mapping, n_normal=args.normal, n_per_class=args.per_class)
    uploaded = download_and_upload(selected, client)

    labels_json = json.dumps(uploaded, indent=2, ensure_ascii=False).encode("utf-8")
    write_bronze_bytes(labels_json, "uav_sead/labels.json", content_type="application/json", client=client)
    total_mb = sum(m["size_bytes"] for m in uploaded.values()) / 1e6
    logger.info("Tamam: %d/%d ucus Bronze'da (%.1f MB) + labels.json", len(uploaded), len(selected), total_mb)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
