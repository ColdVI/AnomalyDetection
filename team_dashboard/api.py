"""api.py -- Grup projesi Gold veri disa aktarma dashboard'unun backend'i.

Yusuf'un canli dashboard'undan (Dashboard/) TAMAMEN BAGIMSIZ: bu, canli
uçuş izleme degil, Gold katmanindan istenen kolon+tarih araligini
Parquet/CSV olarak indirmeye yarayan ayri bir arac. Kendi portunda
(8010) calisir, Dashboard'un hicbir dosyasina dokunmaz/import etmez.

Calistirma:
    python -m team_dashboard.gold_index         # (once, tek seferlik) indeks olustur
    python -m team_dashboard.api                # sonra API'yi baslat
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.common.minio_io import get_minio_client, read_parquet_object
from src.gold.unify import GOLD_COLUMNS
from team_dashboard.gold_index import INDEX_PATH, load_index, parts_overlapping

logger = logging.getLogger(__name__)

# 2026-07-14 (kullanici karari): tek seferlik export'larda sunucu bellegi/
# tarayici indirmesi kontrolsuz buyumesin diye ust sinir -- asilirsa istek
# net bir hata mesajiyla REDDEDILIR (sessizce kesilmez).
MAX_EXPORT_ROWS = 5_000_000

app = FastAPI(title="ADS-B Gold Veri Disa Aktarma")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_client = None
_index_cache: pd.DataFrame | None = None


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


def _day_to_epoch(d: date, *, end_of_day: bool = False) -> float:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return dt.timestamp()


@app.get("/api/meta")
def get_meta():
    """Frontend'in acilista gostermesi gereken bilgiler: secilebilir
    kolonlar (Gold'un GERCEK semasindan, elle kopyalanmis ikinci bir liste
    degil) ve indeksteki verinin kapsadigi tarih araligi."""
    index_df = _get_index()
    valid = index_df.dropna(subset=["min_ts", "max_ts"])
    if valid.empty:
        raise HTTPException(500, "Export indeksi bos -- once gold_index.build_index() calistirilmali.")
    min_ts, max_ts = float(valid["min_ts"].min()), float(valid["max_ts"].max())
    return {
        "columns": GOLD_COLUMNS,
        "min_date": datetime.fromtimestamp(min_ts, tz=timezone.utc).date().isoformat(),
        "max_date": datetime.fromtimestamp(max_ts, tz=timezone.utc).date().isoformat(),
        "total_rows_indexed": int(index_df["row_count"].sum()),
        "max_export_rows": MAX_EXPORT_ROWS,
    }


@app.get("/api/estimate")
def estimate(start: date = Query(...), end: date = Query(...)):
    """Gercek veriye HIC dokunmadan, sadece indeksten tahmini satir sayisi --
    frontend indirme butonuna basmadan once kullaniciyi uyarabilsin diye."""
    if end < start:
        raise HTTPException(400, "Bitis tarihi baslangictan once olamaz")
    start_ts, end_ts = _day_to_epoch(start), _day_to_epoch(end, end_of_day=True)
    overlap = parts_overlapping(_get_index(), start_ts, end_ts)
    return {
        "estimated_rows": int(overlap["row_count"].sum()),
        "parts_to_scan": len(overlap),
        "exceeds_limit": int(overlap["row_count"].sum()) > MAX_EXPORT_ROWS,
    }


@app.get("/api/export")
def export(
    start: date = Query(...),
    end: date = Query(...),
    columns: str = Query(..., description="Virgulle ayrilmis Gold kolon adlari"),
    fmt: str = Query("parquet", pattern="^(parquet|csv)$"),
):
    if end < start:
        raise HTTPException(400, "Bitis tarihi baslangictan once olamaz")

    requested_cols = [c.strip() for c in columns.split(",") if c.strip()]
    unknown = set(requested_cols) - set(GOLD_COLUMNS)
    if unknown:
        raise HTTPException(400, f"Bilinmeyen kolon(lar): {sorted(unknown)}")
    if not requested_cols:
        raise HTTPException(400, "En az bir kolon secilmeli")
    # timestamp_utc, tarih filtresi icin HER ZAMAN gerekli -- kullanici
    # secmemis olsa bile dahili olarak okunur, ciktiya sadece istenirse eklenir.
    read_cols = list(dict.fromkeys(requested_cols + ["timestamp_utc"]))

    start_ts, end_ts = _day_to_epoch(start), _day_to_epoch(end, end_of_day=True)
    index_df = _get_index()
    overlap = parts_overlapping(index_df, start_ts, end_ts)

    estimated_rows = int(overlap["row_count"].sum())
    if estimated_rows > MAX_EXPORT_ROWS:
        raise HTTPException(
            413,
            f"Tahmini {estimated_rows:,} satir, izin verilen {MAX_EXPORT_ROWS:,} satiri asiyor -- "
            "tarih araligini daraltin veya daha az kolon secin (kolon sayisi satir sinirini etkilemez "
            "ama daha dar bir tarih araligi satir sayisini dusurur).",
        )
    if overlap.empty:
        raise HTTPException(404, "Secilen tarih araliginda veri bulunamadi")

    logger.info("Export: %s->%s, %d parca, tahmini %d satir", start, end, len(overlap), estimated_rows)

    client = _get_client()
    gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")
    frames = []
    total = 0
    for object_name in overlap["object_name"]:
        df = read_parquet_object(client, gold_bucket, object_name)
        chunk = df.loc[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts), read_cols]
        if chunk.empty:
            continue
        frames.append(chunk)
        total += len(chunk)
        if total > MAX_EXPORT_ROWS:
            # Indeks tahmini yanilmis olabilir (parca-ici kismi kesisim) --
            # GERCEK satir sayisi da limiti asarsa yine net bir hatayla dur.
            raise HTTPException(413, f"Gerçek satır sayısı {MAX_EXPORT_ROWS:,} sınırını aştı, istek iptal edildi.")

    if not frames:
        raise HTTPException(404, "Secilen tarih araliginda veri bulunamadi")

    result = pd.concat(frames, ignore_index=True)[requested_cols]
    logger.info("Export tamamlandi: %d satir, %d kolon", len(result), len(requested_cols))

    stamp = f"{start.isoformat()}_{end.isoformat()}"
    buffer = io.BytesIO()
    if fmt == "csv":
        result.to_csv(buffer, index=False)
        media_type, filename = "text/csv", f"gold_export_{stamp}.csv"
    else:
        result.to_parquet(buffer, index=False, engine="pyarrow")
        media_type, filename = "application/octet-stream", f"gold_export_{stamp}.parquet"
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
