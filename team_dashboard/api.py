"""api.py -- Grup projesi Silver/Gold veri disa aktarma dashboard'unun backend'i.

Yusuf'un canli dashboard'undan (Dashboard/) TAMAMEN BAGIMSIZ: bu, canli
uçuş izleme degil, Silver VEYA Gold katmanindan, secilen dataset (source_type)
+ kolon + tarih araligini Parquet/CSV olarak indirmeye yarayan ayri bir arac.
Kendi portunda (8010) calisir, Dashboard'un hicbir dosyasina dokunmaz/import etmez.

Akis: once KATMAN (silver/gold) secilir, sonra o katmandaki DATASET
(adsblol_historical, alfa, uav_attack, ...) -- kolon listesi VE tarih araligi
secilen (layer, dataset) ikilisine gore DEGISIR (her ikisinin kendi semasi/
kapsami var).

Calistirma:
    python -m team_dashboard.layer_index         # (once, tek seferlik) indeks + kolon katalogu olustur
    python -m team_dashboard.api                 # sonra API'yi baslat
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.common.minio_io import get_minio_client, read_parquet_object
from team_dashboard.layer_index import (
    load_columns_catalog,
    load_index,
    parts_overlapping,
)

logger = logging.getLogger(__name__)

# 2026-07-14 (kullanici karari): tek seferlik export'larda sunucu bellegi/
# tarayici indirmesi kontrolsuz buyumesin diye ust sinir -- asilirsa istek
# net bir hata mesajiyla REDDEDILIR (sessizce kesilmez).
MAX_EXPORT_ROWS = 5_000_000

app = FastAPI(title="ADS-B Silver/Gold Veri Dışa Aktarma")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_client = None
_index_cache: pd.DataFrame | None = None
_catalog_cache: dict | None = None


def _get_client():
    global _client
    if _client is None:
        _client = get_minio_client()
    return _client


def _get_index() -> pd.DataFrame:
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index()
    return _index_cache


def _get_catalog() -> dict:
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = load_columns_catalog()
    return _catalog_cache


def _day_to_epoch(d: date, *, end_of_day: bool = False) -> float:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return dt.timestamp()


def _validate_layer_dataset(layer: str, dataset: str) -> None:
    if layer not in ("silver", "gold"):
        raise HTTPException(400, "layer 'silver' veya 'gold' olmali")
    catalog = _get_catalog()
    if dataset not in catalog.get(layer, {}):
        raise HTTPException(404, f"'{layer}' katmaninda '{dataset}' diye bir dataset yok. "
                                  f"Mevcutlar: {sorted(catalog.get(layer, {}).keys())}")


@app.get("/api/layers")
def get_layers():
    return {"layers": ["silver", "gold"]}


@app.get("/api/datasets")
def get_datasets(layer: str = Query(...)):
    """Secilen katmandaki dataset'leri, her birinin toplam satir sayisiyla
    birlikte dondurur -- frontend'in dropdown'u bunu dolduruyor."""
    if layer not in ("silver", "gold"):
        raise HTTPException(400, "layer 'silver' veya 'gold' olmali")
    index_df = _get_index()
    subset = index_df[index_df["layer"] == layer]
    if subset.empty:
        return {"datasets": []}
    summary = subset.groupby("dataset")["row_count"].sum().sort_values(ascending=False)
    return {"datasets": [{"name": name, "row_count": int(count)} for name, count in summary.items()]}


@app.get("/api/meta")
def get_meta(layer: str = Query(...), dataset: str = Query(...)):
    """Secilen (layer, dataset) ikilisi icin: gercek kolon listesi (o
    dataset'te GERCEKTEN var olanlar -- Silver'da her dataset farkli, Gold'da
    hepsi ayni 7+3(+is_military) semayi paylasir) + verinin kapsadigi tarih araligi."""
    _validate_layer_dataset(layer, dataset)
    index_df = _get_index()
    subset = index_df[(index_df["layer"] == layer) & (index_df["dataset"] == dataset)]
    valid = subset.dropna(subset=["min_ts", "max_ts"])
    if valid.empty:
        raise HTTPException(404, f"'{layer}/{dataset}' icin veri bulunamadi")
    min_ts, max_ts = float(valid["min_ts"].min()), float(valid["max_ts"].max())
    return {
        "columns": _get_catalog()[layer][dataset],
        "min_date": datetime.fromtimestamp(min_ts, tz=timezone.utc).date().isoformat(),
        "max_date": datetime.fromtimestamp(max_ts, tz=timezone.utc).date().isoformat(),
        "total_rows": int(subset["row_count"].sum()),
        "max_export_rows": MAX_EXPORT_ROWS,
    }


@app.get("/api/estimate")
def estimate(layer: str = Query(...), dataset: str = Query(...), start: date = Query(...), end: date = Query(...)):
    """Gercek veriye HIC dokunmadan, sadece indeksten tahmini satir sayisi --
    frontend indirme butonuna basmadan once kullaniciyi uyarabilsin diye.
    Tahmin, kesisen parcanin TOPLAM satir sayisini sayar (parcanin ne kadari
    istenen aralikla ortustugune bakmadan) -- yani GERCEK sayidan HER ZAMAN
    buyuk-esit bir ust sinirdir, asla kucuk cikmaz (guvenli tarafta abartili)."""
    _validate_layer_dataset(layer, dataset)
    if end < start:
        raise HTTPException(400, "Bitis tarihi baslangictan once olamaz")
    start_ts, end_ts = _day_to_epoch(start), _day_to_epoch(end, end_of_day=True)
    overlap = parts_overlapping(_get_index(), layer, dataset, start_ts, end_ts)
    return {
        "estimated_rows": int(overlap["row_count"].sum()),
        "parts_to_scan": len(overlap),
        "exceeds_limit": int(overlap["row_count"].sum()) > MAX_EXPORT_ROWS,
    }


@app.get("/api/export")
def export(
    layer: str = Query(...),
    dataset: str = Query(...),
    start: date = Query(...),
    end: date = Query(...),
    columns: str = Query(..., description="Virgulle ayrilmis kolon adlari"),
    fmt: str = Query("parquet", pattern="^(parquet|csv)$"),
    min_lat: float | None = Query(None, ge=-90, le=90),
    max_lat: float | None = Query(None, ge=-90, le=90),
    min_lon: float | None = Query(None, ge=-180, le=180),
    max_lon: float | None = Query(None, ge=-180, le=180),
):
    _validate_layer_dataset(layer, dataset)
    if end < start:
        raise HTTPException(400, "Bitis tarihi baslangictan once olamaz")

    available_cols = _get_catalog()[layer][dataset]
    requested_cols = [c.strip() for c in columns.split(",") if c.strip()]
    unknown = set(requested_cols) - set(available_cols)
    if unknown:
        raise HTTPException(400, f"'{layer}/{dataset}' icin bilinmeyen kolon(lar): {sorted(unknown)}")
    if not requested_cols:
        raise HTTPException(400, "En az bir kolon secilmeli")

    has_bbox = any(v is not None for v in (min_lat, max_lat, min_lon, max_lon))
    if has_bbox and not {"lat", "lon"}.issubset(available_cols):
        raise HTTPException(400, f"'{layer}/{dataset}' icin lat/lon kolonlari yok -- bolgesel filtre uygulanamaz")
    if min_lat is not None and max_lat is not None and min_lat > max_lat:
        raise HTTPException(400, "min_lat, max_lat'tan buyuk olamaz")
    if min_lon is not None and max_lon is not None and min_lon > max_lon:
        raise HTTPException(400, "min_lon, max_lon'dan buyuk olamaz")

    read_cols = list(dict.fromkeys(requested_cols + ["timestamp_utc"]))
    # Gold'da dataset secimi bir PREFIX degil, "unified/" icindeki bir
    # source_type DEGERI -- okurken ayrica bu kolona gore filtrelenmeli.
    if layer == "gold" and "source_type" not in read_cols:
        read_cols.append("source_type")
    if has_bbox:
        read_cols = list(dict.fromkeys(read_cols + ["lat", "lon"]))

    start_ts, end_ts = _day_to_epoch(start), _day_to_epoch(end, end_of_day=True)
    index_df = _get_index()
    overlap = parts_overlapping(index_df, layer, dataset, start_ts, end_ts)

    estimated_rows = int(overlap["row_count"].sum())
    if estimated_rows > MAX_EXPORT_ROWS:
        raise HTTPException(
            413,
            f"Tahmini {estimated_rows:,} satir, izin verilen {MAX_EXPORT_ROWS:,} satiri asiyor -- "
            "tarih araligini daraltin.",
        )
    if overlap.empty:
        raise HTTPException(404, "Secilen tarih araliginda veri bulunamadi")

    logger.info("Export: %s/%s %s->%s, %d parca, tahmini %d satir%s",
                layer, dataset, start, end, len(overlap), estimated_rows,
                f", bbox=({min_lat},{max_lat},{min_lon},{max_lon})" if has_bbox else "")

    client = _get_client()
    bucket = os.getenv("MINIO_SILVER_BUCKET" if layer == "silver" else "MINIO_GOLD_BUCKET",
                        "silver" if layer == "silver" else "gold")
    frames = []
    total = 0
    for object_name in overlap["object_name"]:
        df = read_parquet_object(client, bucket, object_name)
        mask = (df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)
        if layer == "gold":
            mask &= df["source_type"] == dataset
        if has_bbox:
            # NaN lat/lon karsilastirmalari otomatik False doner -- eksik
            # konum verisi olan satirlar bolgesel filtrede kendiliginden elenir.
            if min_lat is not None:
                mask &= df["lat"] >= min_lat
            if max_lat is not None:
                mask &= df["lat"] <= max_lat
            if min_lon is not None:
                mask &= df["lon"] >= min_lon
            if max_lon is not None:
                mask &= df["lon"] <= max_lon
        chunk = df.loc[mask, [c for c in read_cols if c in df.columns]]
        if chunk.empty:
            continue
        frames.append(chunk)
        total += len(chunk)
        if total > MAX_EXPORT_ROWS:
            raise HTTPException(413, f"Gerçek satır sayısı {MAX_EXPORT_ROWS:,} sınırını aştı, istek iptal edildi.")

    if not frames:
        raise HTTPException(404, "Secilen tarih araliginda veri bulunamadi")

    result = pd.concat(frames, ignore_index=True)
    result = result[[c for c in requested_cols if c in result.columns]]
    logger.info("Export tamamlandi: %d satir, %d kolon", len(result), len(requested_cols))

    stamp = f"{layer}_{dataset}_{start.isoformat()}_{end.isoformat()}"
    buffer = io.BytesIO()
    if fmt == "csv":
        result.to_csv(buffer, index=False)
        media_type, filename = "text/csv", f"export_{stamp}.csv"
    else:
        result.to_parquet(buffer, index=False, engine="pyarrow")
        media_type, filename = "application/octet-stream", f"export_{stamp}.parquet"
    buffer.seek(0)

    return StreamingResponse(
        buffer, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8010)
