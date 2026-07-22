"""RflyMAD alt kumesini Kaggle'dan indirip Bronze'a yerlestirir.

Dataset: xianglile/rflymad (Kaggle mirror; resmi kaynak Beihang Yunpan).
Lisans: yalniz ticari olmayan kullanim. Kaggle mirror'i zip-shard DEGIL,
patlatilmis dosya agacidir (TestCase basina Log/*.ulg, TLog/*, TestInfo.csv,
TrueData/*.xlsx). Bu indirici CASE-BAZLI ve secicidir: varsayilan olarak
yalniz .ulg + TestInfo.csv ceker (TLog/TrueData atlanir — boyut kontrolu).

Disiplin (uav_sead_downloader ile ayni):
- skip-existing: Bronze'da olan dosya tekrar indirilmez (resume/idempotent).
- manifest: bronze/rflymad/manifest.json — case basina dosyalar, boyut,
  sha256, subdataset/fault bilgisi. Her kosuda birlestirilerek yazilir.
- listing checkpoint'li: 429 rate-limit'te bekleyip kaldigi sayfadan surer.

Kullanim:
  python -m src.ingestion.rflymad_downloader --list
  python -m src.ingestion.rflymad_downloader --download --subsets SampleData
  python -m src.ingestion.rflymad_downloader --download \
      --subsets Real-NoFault,Real-Motor,Real-Sensors
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
import zipfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET = "xianglile/rflymad"
ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data/objectstore/bronze/rflymad"
LISTING_DIR = ROOT / "artifacts/rflymad"
LISTING_CSV = LISTING_DIR / "kaggle_file_listing.csv"
LISTING_STATE = LISTING_DIR / "kaggle_listing_state.json"
MANIFEST = BRONZE / "manifest.json"

ESSENTIAL_SUFFIXES = (".ulg", "TestInfo.csv")
ESSENTIAL_PREFIXES = ("TestInfo_",)
RETRY_SLEEPS = [30, 60, 120, 300, 600]


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_retries(fn, what: str):
    for i, sleep_s in enumerate([0, *RETRY_SLEEPS]):
        if sleep_s:
            logger.warning("%s: yeniden deneme %d, %ds bekleniyor", what, i, sleep_s)
            time.sleep(sleep_s)
        try:
            return fn()
        except Exception as exc:  # 429 dahil
            last = exc
    raise last


def do_list() -> Path:
    """Tam dosya listesini sayfalayarak cikar; 429'da checkpoint'ten surer."""
    api = _api()
    LISTING_DIR.mkdir(parents=True, exist_ok=True)
    state = (json.loads(LISTING_STATE.read_text(encoding="utf-8"))
             if LISTING_STATE.exists() else {"token": None, "done": False, "rows": 0})
    if state.get("done") and LISTING_CSV.exists():
        logger.info("listing zaten tam: %s", LISTING_CSV)
        return LISTING_CSV

    mode = "a" if state["rows"] else "w"
    with LISTING_CSV.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if mode == "w":
            writer.writerow(["name", "bytes"])
        token = state["token"]
        page = 0
        while True:
            result = _with_retries(
                lambda: api.dataset_list_files(DATASET, page_token=token, page_size=1000),
                "listing sayfasi",
            )
            files = result.files or []
            for f in files:
                writer.writerow([f.name, f.total_bytes])
            state["rows"] += len(files)
            token = getattr(result, "next_page_token", None)
            state["token"] = token
            LISTING_STATE.write_text(json.dumps(state), encoding="utf-8")
            page += 1
            if page % 20 == 0:
                fh.flush()
                logger.info("listing: +%d sayfa, toplam %d dosya", page, state["rows"])
            if not token or not files:
                break
            time.sleep(0.5)
    state["done"] = True
    LISTING_STATE.write_text(json.dumps(state), encoding="utf-8")
    logger.info("listing tamam: %d dosya -> %s", state["rows"], LISTING_CSV)
    return LISTING_CSV


def _load_listing() -> list[tuple[str, int]]:
    with LISTING_CSV.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        return [(name, int(size or 0)) for name, size in reader]


def _case_id(name: str) -> str:
    """'Real-Motor/<durum>/TestCase_x_y/...' -> 'Real-Motor/<durum>/TestCase_x_y'"""
    parts = name.split("/")
    for i, part in enumerate(parts):
        if part.startswith("TestCase"):
            return "/".join(parts[: i + 1])
    return "/".join(parts[:-1])


def is_essential_file(name: str) -> bool:
    """Return whether a RflyMAD object is needed for the first Bronze pass."""
    filename = name.rsplit("/", 1)[-1]
    return (
        name.endswith(ESSENTIAL_SUFFIXES)
        or (filename.startswith(ESSENTIAL_PREFIXES) and filename.endswith(".xlsx"))
    )


def do_download(subsets: list[str], *, essential_only: bool = True,
                limit_cases: int | None = None) -> None:
    api = _api()
    rows = _load_listing()
    wanted = [
        (name, size) for name, size in rows
        if name.split("/", 1)[0] in subsets
        and (not essential_only or is_essential_file(name))
    ]
    by_case: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, size in wanted:
        by_case[_case_id(name)].append((name, size))
    cases = sorted(by_case)
    if limit_cases:
        cases = cases[:limit_cases]
    total_bytes = sum(size for c in cases for _, size in by_case[c])
    logger.info("secim: %d case, %d dosya, %.2f GB (%s)",
                len(cases), sum(len(by_case[c]) for c in cases),
                total_bytes / 1e9, ",".join(subsets))

    manifest = (json.loads(MANIFEST.read_text(encoding="utf-8"))
                if MANIFEST.exists() else {"dataset": DATASET, "cases": {}})
    downloaded = skipped = failed = 0
    for ci, case in enumerate(cases, 1):
        entry = manifest["cases"].setdefault(case, {
            "subdataset": case.split("/", 1)[0], "files": {}})
        for name, size in sorted(by_case[case]):
            target = BRONZE / name
            if target.exists() and target.stat().st_size == size and size > 0:
                if name not in entry["files"]:
                    entry["files"][name] = {"bytes": size, "sha256": _sha256(target)}
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                _with_retries(
                    lambda: api.dataset_download_file(
                        DATASET, name, path=str(target.parent), quiet=True),
                    f"indirme {name}",
                )
            except Exception as exc:
                logger.warning("indirilemedi, atlandi: %s (%s)", name, exc)
                failed += 1
                continue
            # kaggle bazen tekil dosyayi .zip sarar
            zipped = target.parent / (target.name + ".zip")
            if zipped.exists():
                with zipfile.ZipFile(zipped) as zf:
                    zf.extract(zf.namelist()[0], target.parent)
                zipped.unlink()
            entry["files"][name] = {"bytes": target.stat().st_size,
                                    "sha256": _sha256(target)}
            downloaded += 1
        if ci % 20 == 0 or ci == len(cases):
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            logger.info("[%d/%d case] indirilen=%d atlanan=%d hata=%d",
                        ci, len(cases), downloaded, skipped, failed)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    logger.info("bitti: indirilen=%d atlanan(zaten var)=%d hata=%d; manifest=%s",
                downloaded, skipped, failed, MANIFEST)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--subsets", default="SampleData",
                        help="virgullu: SampleData,Real-NoFault,Real-Motor,Real-Sensors")
    parser.add_argument("--all-files", action="store_true",
                        help="yalniz .ulg+TestInfo yerine TUM dosyalar")
    parser.add_argument("--limit-cases", type=int, default=None)
    args = parser.parse_args()
    if args.list:
        do_list()
    if args.download:
        if not LISTING_CSV.exists():
            do_list()
        do_download([s.strip() for s in args.subsets.split(",") if s.strip()],
                    essential_only=not args.all_files,
                    limit_cases=args.limit_cases)


if __name__ == "__main__":
    main()
