"""ALFA raw rosbag envanteri (ML-4 Faz B fizibilitesi).

Desktop/ALFA/raw/raw altindaki 39 .bag dosyasini tarar:
- processed sette KULLANILMAYAN ucuslari bulur (processed 31 ucus koku / 47 sequence),
- her eksik bag icin topic listesi + mesaj sayilari,
- parse edilebilirlik kriterleri: mavros/global_position/global var mi, nav_info/* var mi,
- failure_status/* topic'leri hic aktiflesmis mi (aktiflesmemisse NORMAL ADAYI).

Cikti: konsol tablosu -- parse_alfa_rosbag.py yazilip yazilmayacagini bu belirler.

Kullanim:
    python scripts/inventory_alfa_raw.py [--raw-dir "C:/Users/.../Desktop/ALFA/raw/raw"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rosbags.highlevel import AnyReader

DEFAULT_RAW = r"C:/Users/PC_5812_YD26/Desktop/ALFA/raw/raw"

REQUIRED_TOPICS = ("mavros/global_position/global",)
NAVINFO_PREFIX = "nav_info"
FAILURE_PREFIX = "failure_status"


def flight_root(name: str) -> str:
    return name.removesuffix(".bag")


def processed_roots(silver_path: str = "data/silver/alfa_silver.parquet") -> set[str]:
    a = pd.read_parquet(silver_path, columns=["source_id"])
    roots = set()
    for s in a["source_id"].unique():
        # sequence adi: carbonZ_<tarih>[_<n>]_<senaryo> -> kok: carbonZ_<tarih>
        parts = s.split("_")
        roots.add("_".join(parts[:2]))
    return roots


def inspect_bag(path: Path) -> dict:
    info = {"bag": path.stem, "okunabilir": False, "topics": 0, "global_position": False,
            "nav_info": 0, "failure_topics": 0, "failure_aktif": None, "sure_s": None, "hata": ""}
    try:
        with AnyReader([path]) as reader:
            info["okunabilir"] = True
            topics = {c.topic: c for c in reader.connections}
            info["topics"] = len(topics)
            info["sure_s"] = round((reader.end_time - reader.start_time) / 1e9, 1)
            info["global_position"] = any(t.rstrip("/").endswith("global_position/global") for t in topics)
            info["nav_info"] = sum(1 for t in topics if NAVINFO_PREFIX in t)
            fail_conns = [c for t, c in topics.items() if FAILURE_PREFIX in t]
            info["failure_topics"] = len(fail_conns)
            if fail_conns:
                aktif = False
                for conn, _, raw in reader.messages(connections=fail_conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    if bool(getattr(msg, "data", False)):
                        aktif = True
                        break
                info["failure_aktif"] = aktif
    except Exception as exc:  # bozuk/uyumsuz bag: fizibilite raporuna girer
        info["hata"] = str(exc)[:80]
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    bags = sorted(raw_dir.glob("*.bag"))
    used = processed_roots()
    print(f"raw bag sayisi: {len(bags)}, processed'de kullanilan ucus koku: {len(used)}\n")

    rows = []
    for bag in bags:
        root = flight_root(bag.name)
        in_processed = root in used
        row = {"bag": root, "processed_de": in_processed}
        if not in_processed:
            row.update(inspect_bag(bag))
            row["normal_adayi"] = (row.get("okunabilir") and row.get("global_position")
                                    and row.get("failure_aktif") is not True)
        rows.append(row)

    df = pd.DataFrame(rows)
    missing = df[~df["processed_de"]]
    print(f"processed'e girmemis bag: {len(missing)}\n")
    cols = [c for c in ["bag", "okunabilir", "topics", "sure_s", "global_position",
                         "nav_info", "failure_topics", "failure_aktif", "normal_adayi", "hata"]
            if c in missing.columns]
    print(missing[cols].to_string(index=False))
    n_aday = int(missing.get("normal_adayi", pd.Series(dtype=bool)).fillna(False).sum())
    print(f"\nSONUC: {n_aday} yeni normal-ucus adayi (failure_status hic aktiflesmemis + konum verisi var)")


if __name__ == "__main__":
    main()
