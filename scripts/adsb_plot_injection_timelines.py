"""Zaman-damgali enjeksiyon gozlemi: ayni ucusun temiz ve sentetik-bozuk
versiyonlarinda kural-penalty skorunun ZAMAN icindeki seyri, enjeksiyon onset'i
dikey cizgiyle isaretli. "Anomali enjekte edildiginde skor gercekten o anda mi
yukseliyor?" sorusunun dogrudan gorsel cevabi.

Kalibrasyonu rule_scorer_report.json'dan okur (yeniden 60 parca yuklemez) --
once scripts/adsb_evaluate_rule_scorer.py kosmus olmali.

Kullanim:
    python scripts/adsb_plot_injection_timelines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.features import build_feature_table  # noqa: E402
from adsb.rules import ResidualRuleScorer  # noqa: E402
from adsb.synthetic import PHYSICS_BREAK_RECIPES  # noqa: E402

SYNTHETIC_DIR = Path("data/objectstore/synthetic/adsb")
REPORT_PATH = Path("artifacts/adsb/models/rule_scorer_report.json")
OUT_DIR = Path("artifacts/adsb/plots/injection_timelines")
N_EXAMPLE_FLIGHTS = 3
MIN_ROWS = 120  # zaman-cizgisi anlamli olsun diye yeterince uzun ucuslar


def pick_flights(clean: pd.DataFrame) -> list[str]:
    sizes = clean.groupby("flight_id").size()
    candidates = sizes[sizes >= MIN_ROWS].index.tolist()
    return sorted(candidates)[:N_EXAMPLE_FLIGHTS]


def main() -> None:
    scorer = ResidualRuleScorer.from_dict(json.loads(REPORT_PATH.read_text(encoding="utf-8"))["scorer"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_all = pd.read_parquet(SYNTHETIC_DIR / "clean.parquet")
    flights = pick_flights(clean_all)
    print(f"ornek ucuslar: {flights}", flush=True)

    recipes = list(PHYSICS_BREAK_RECIPES)
    for fid in flights:
        clean = clean_all[clean_all["flight_id"] == fid].sort_values("timestamp_utc").reset_index(drop=True)
        clean_feat = build_feature_table(clean)
        clean_pen = scorer.row_penalties(clean_feat)
        t0 = clean["timestamp_utc"].iloc[0]
        t_clean = (clean["timestamp_utc"] - t0) / 60.0  # dakika

        fig, axes = plt.subplots(len(recipes), 1, figsize=(10, 2.2 * len(recipes)), sharex=True)
        for ax, recipe in zip(axes, recipes):
            bozuk_all = pd.read_parquet(SYNTHETIC_DIR / f"{recipe}.parquet")
            bozuk = bozuk_all[bozuk_all["flight_id"] == fid].sort_values("timestamp_utc").reset_index(drop=True)
            bozuk_feat = build_feature_table(bozuk)
            bozuk_pen = scorer.row_penalties(bozuk_feat)
            t_bozuk = (bozuk["timestamp_utc"] - t0) / 60.0

            ax.plot(t_clean, clean_pen, color="#4C956C", lw=1.0, label="temiz", alpha=0.8)
            ax.plot(t_bozuk, bozuk_pen, color="#C1121F", lw=1.0, label="bozuk", alpha=0.8)

            labeled = bozuk["label"].notna()
            if labeled.any():
                onset_min = (bozuk.loc[labeled, "timestamp_utc"].iloc[0] - t0) / 60.0
                ax.axvline(onset_min, color="black", ls="--", lw=1.2, label="enjeksiyon onset")
            ax.set_ylabel("penalty", fontsize=8)
            ax.set_title(recipe, fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("ucus suresi (dakika)")
        fig.suptitle(f"Enjeksiyon zaman-cizgisi -- {fid} (kural-penalty, satir bazi)")
        fig.tight_layout()
        path = OUT_DIR / f"{fid.replace(':', '_')}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
