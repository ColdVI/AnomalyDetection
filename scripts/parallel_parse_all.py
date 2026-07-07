"""parallel_parse_all.py -- parse all adsb historical tars to Silver in parallel processes.

Thread NOT used deliberately: gzip decompress + JSON parse + DataFrame construction
is CPU-bound, and Python's GIL only lets one thread run Python bytecode at a time --
threads help with I/O-wait workloads, not CPU-bound ones. Real parallelism here needs
separate processes (`concurrent.futures.ProcessPoolExecutor`), each running
`python -m src.silver.parse_adsblol_historical --local-tar <path>` as a subprocess so the
existing single-tar CLI path is reused unchanged.

Usage:
    # MinIO must be running (docker compose up -d minio)

    # Start small -- verify 2 tars work before firing all 11 at once.
    python scripts/parallel_parse_all.py --tar-dir "C:\\path\\to\\tars" --workers 2 --limit 2

    # Full run
    python scripts/parallel_parse_all.py --tar-dir "C:\\path\\to\\tars" --workers 4
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Conservative default -- not all cores, so a bad run doesn't starve disk/RAM alongside
# whatever else is running. Override with --workers if the machine can take more.
DEFAULT_MAX_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))


def parse_one_tar(tar_path: str, log_dir: Path) -> tuple[str, bool, str]:
    """Run one tar through `parse_adsblol_historical --local-tar` in its own subprocess.

    Subprocess (not just a plain function call inside the pool worker) so each tar's
    stdout/stderr can be captured to its own log file instead of interleaving with the
    other workers' output on one terminal.

    Returns: (tar_path, succeeded, log_file_path)
    """
    log_path = log_dir / f"{Path(tar_path).stem}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        result = subprocess.run(
            [sys.executable, "-m", "src.silver.parse_adsblol_historical", "--local-tar", tar_path],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return tar_path, result.returncode == 0, str(log_path)


def clear_stale_silver_output() -> int:
    """Delete any existing `silver/adsblol_historical/` parts before a fresh parallel run.

    Each worker subprocess calls `parse_local_tar()` directly (not `run()`), so it never
    clears prior output itself -- clearing must happen exactly once, here, before any
    worker starts. Skipping this would leave stale parts (e.g. from a run killed by a
    shutdown mid-way) sitting alongside the new ones, double-counting rows downstream.
    """
    from src.common.minio_io import delete_layer_objects, get_minio_client

    client = get_minio_client()
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
    removed = delete_layer_objects(client, silver_bucket, "adsblol_historical")
    if removed:
        logger.info("Cleared %d stale Silver part(s) from a prior run before starting", removed)
    return removed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse all adsb historical tars to Silver in parallel")
    parser.add_argument("--tar-dir", required=True, help="Folder containing the .tar files")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--log-dir", default="logs/parallel_parse")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tars (smoke test)")
    parser.add_argument("--skip-clear", action="store_true", help="Skip clearing stale Silver output first")
    args = parser.parse_args()

    tar_dir = Path(args.tar_dir)
    tars = sorted(str(p) for p in tar_dir.glob("*.tar"))
    if not tars:
        print(f"HATA: {tar_dir} altinda .tar dosyasi bulunamadi.")
        return
    if args.limit:
        tars = tars[: args.limit]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_clear:
        clear_stale_silver_output()

    print(f"{len(tars)} tar, {args.workers} paralel process ile islenecek.")
    print(f"Her tar'in loglari: {log_dir}/<tar_adi>.log")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(parse_one_tar, t, log_dir): t for t in tars}
        for future in concurrent.futures.as_completed(futures):
            tar_path, ok, log_path = future.result()
            status = "OK" if ok else "HATA"
            print(f"[{status}] {tar_path} -> log: {log_path}")
            results.append((tar_path, ok))

    failed = [t for t, ok in results if not ok]
    print(f"\nToplam: {len(results)}, basarili: {len(results)-len(failed)}, basarisiz: {len(failed)}")
    if failed:
        print("Basarisiz olanlar (tek tek --local-tar ile tekrar dene):")
        for t in failed:
            print(f"  python -m src.silver.parse_adsblol_historical --local-tar {t}")


if __name__ == "__main__":
    main()
