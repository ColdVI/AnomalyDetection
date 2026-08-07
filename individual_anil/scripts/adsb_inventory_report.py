"""Faz 0.1 calistirici: 3 gercek adsb.lol tar arsivi icin hizli envanter profili uretir.

Kullanim:
    python scripts/adsb_inventory_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.inventory import profile_all  # noqa: E402

TAR_DIR = Path(r"C:\Users\PC_5812_YD26\Downloads")
TAR_NAMES = [
    "v2026.02.28-planes-readsb-prod-0.tar",
    "v2026.03.01-planes-readsb-prod-0.tar",
    "v2026.03.16-planes-readsb-prod-0.tar",
]
OUT_PATH = Path("adsb/reports/inventory_profile.json")


def main() -> None:
    tar_paths = [TAR_DIR / name for name in TAR_NAMES]
    missing = [p for p in tar_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Bulunamayan tar dosyalari: {missing}")

    profiles = profile_all(tar_paths, n_samples=500)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps([p.as_dict() for p in profiles], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for p in profiles:
        print(f"\n=== {p.tar_name} ===")
        print(f"  toplam trace dosyasi: {p.total_trace_members}")
        print(f"  ornekleme: {p.sampled_members} ucak, {p.sampled_rows} satir, {p.parse_errors} hata")
        print(f"  trace satir uzunluklari: {p.trace_row_lengths}")
        print(f"  category dagilimi: {p.category_counts}")
        print(f"  ornekleme araligi (s): {p.sampling_interval_s}")
        print(f"  on_ground satir: {p.on_ground_rows}")
    print(f"\nKayit: {OUT_PATH}")


if __name__ == "__main__":
    main()
