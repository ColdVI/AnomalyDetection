"""ADS-B: gercek ucuslardan (Silver, val-split) kalici bir sentetik test/validation
korpusu uretir ve diske yazar (data/objectstore/synthetic/adsb/).

`scripts/adsb_train_baseline_models.py::run_synthetic_check()` bozulmayi her egitim
turunda bellekte, gecici olarak uretiyor (5 recipe x ilk 20 val ucusu, hicbir zaman
diske yazilmiyor). Bu script AYNI PHYSICS_BREAK_RECIPES'i daha genis bir ucus
kumesine uygulayip sonucu kalici parquet olarak saklar -- boylece farkli
model/egitim turlari ayni sabit sentetik test setini yeniden kullanabilir ve
sonuclar karsilastirilabilir kalir.

KURAL (adsb/synthetic.py ile ayni): bu korpus ASLA egitime girmez, yalniz
degerlendirme/test icin kullanilir. Cikti data/objectstore/synthetic/adsb/ altina
yazilir -- gercek Silver veriye (data/objectstore/silver/...) hicbir zaman
dokunulmaz (save_synthetic_batch bunu path guard ile zorlar).

Kullanim:
    python scripts/adsb_generate_synthetic_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.segmentation import segment_flights  # noqa: E402
from adsb.synthetic import PHYSICS_BREAK_RECIPES, save_synthetic_batch  # noqa: E402

SILVER_DIR = Path("data/objectstore/silver/adsblol_historical")
OUT_DIR = Path("data/objectstore/synthetic/adsb")
N_PARTS = 60  # egitimdeki 10'dan genis bir kesit -- tam-hacim (638 parca) sonraki adim
SEED = 0  # egitim scriptiyle AYNI seed -- ayni 80/20 train/val bolmesi, val kismi kullanilir
MIN_FLIGHT_LEN = 20  # cok kisa ucuslarda onset-oncesi/sonrasi ayrimi anlamsizlasir


def load_real_data(n_parts: int) -> pd.DataFrame:
    files = sorted(SILVER_DIR.glob("*.parquet"))[:n_parts]
    print(f"{len(files)} Silver parcasi okunuyor...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df)} satir, {df.source_id.nunique()} ucak")
    return df


def main() -> None:
    df = load_real_data(N_PARTS)
    seg = segment_flights(df, gap_s=1800.0)

    # .unique() bir ArrowStringArray dondurebiliyor (pyarrow-destekli string dtype) --
    # np.random.Generator.shuffle bunun icin "Sequence degil" uyarisi verip yavas/
    # guvenilmez calisabiliyor. Duz numpy object array'e cevirip guvenli hale getiriyoruz.
    flight_ids = np.array(seg["flight_id"].unique(), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(flight_ids)
    split = int(len(flight_ids) * 0.8)
    val_ids = sorted(set(flight_ids[split:]))  # egitimdeki AYNI bolme, val kismi -- egitimle asla ortusmez
    print(f"  {len(flight_ids)} ucus segmenti, {len(val_ids)} val ucusu aday", flush=True)

    # Her ucus/recipe icin AYRI parquet yazmak (>50k kucuk dosya) Windows'ta cok
    # yavas ve gereksiz -- recipe basina TEK dosyada biriktirip yaziyoruz
    # (flight_id kolonu zaten ucuslari ayirt ediyor).
    clean_batches: list[pd.DataFrame] = []
    corrupt_batches: dict[str, list[pd.DataFrame]] = {name: [] for name in PHYSICS_BREAK_RECIPES}
    manifest = []
    n_flights, n_skipped_short = 0, 0
    for i, fid in enumerate(val_ids):
        flight = seg[seg["flight_id"] == fid].sort_values("timestamp_utc").reset_index(drop=True)
        if len(flight) < MIN_FLIGHT_LEN:
            n_skipped_short += 1
            continue
        n_flights += 1
        flight = flight.assign(label=None)
        clean_batches.append(flight)

        for recipe_name, (fn, kwargs) in PHYSICS_BREAK_RECIPES.items():
            try:
                bozuk = fn(flight, onset_frac=0.5, **kwargs)
            except Exception as e:
                print(f"  atlandi ({fid}, {recipe_name}): {e}", flush=True)
                continue
            corrupt_batches[recipe_name].append(bozuk)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(val_ids)} ucus islendi...", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_df = pd.concat(clean_batches, ignore_index=True)
    clean_path = save_synthetic_batch(clean_df, out_dir=OUT_DIR, name="clean")
    manifest.append({"recipe": "clean", "n_flights": n_flights, "n_rows": len(clean_df), "path": str(clean_path)})
    print(f"  clean yazildi: {len(clean_df)} satir, {n_flights} ucus -> {clean_path}", flush=True)

    for recipe_name, batches in corrupt_batches.items():
        if not batches:
            continue
        recipe_df = pd.concat(batches, ignore_index=True)
        path = save_synthetic_batch(recipe_df, out_dir=OUT_DIR, name=recipe_name)
        manifest.append({
            "recipe": recipe_name, "n_flights": len(batches), "n_rows": len(recipe_df),
            "onset_frac": 0.5, "path": str(path),
        })
        print(f"  {recipe_name} yazildi: {len(recipe_df)} satir, {len(batches)} ucus -> {path}", flush=True)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{n_flights} val-ucusu isglendi ({n_skipped_short} cok kisa oldugu icin atlandi)")
    print(f"{len(manifest)} parquet dosyasi yazildi -> {OUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
