"""Consume adsb.lol Kafka messages: land raw JSONL, write filtered Bronze Parquet.

Phase 3 of docs/bronze_implementasyon_plani.md, updated per ADR-002: every
Bronze artifact lives in MinIO, not local disk.

Two MinIO objects per flushed batch:
  - `bronze/adsblol_realtime/_landing/states-<batch-stamp>.jsonl`: every
    message in the batch, completely unfiltered, exactly as produced -- the
    true raw landing. MinIO has no native "append to object" operation, so
    landing is batched at the same cadence as the Parquet flush rather than
    streamed line-by-line.
  - `bronze/adsblol_realtime/part-*.parquet`: only the entries whose lat/lon
    fall inside the Turkey bbox, with standard provenance columns. This is
    where the Bronze "Turkey bbox filter for adsb sources" rule is actually
    applied for the realtime path (the producer does not filter).

No unit conversion or renaming happens here; original adsb.lol field names
(`hex`, `lat`, `lon`, `alt_baro`, `gs`, `track`, ...) are kept as-is.
"""

from __future__ import annotations

import json
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from confluent_kafka import Consumer
from dotenv import load_dotenv

from src.common.bbox import in_turkey
from src.common.io import ObjectStoreClient, write_bronze, write_bronze_bytes
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "adsblol_rt"
DEFAULT_TOPIC = "uav.raw.states"
BATCH_SIZE = 500


class _ShutdownFlag:
    def __init__(self) -> None:
        self.stop = False

    def request_stop(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %s, will stop after draining the buffer", signum)
        self.stop = True


def turkey_rows(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure filtering helper, kept separate from I/O so it's easy to unit test."""
    return [entry for entry in batch if in_turkey(entry.get("lat"), entry.get("lon"))]


def land_batch_raw(
    batch: list[dict[str, Any]],
    *,
    bucket: str | None = None,
    client: ObjectStoreClient | None = None,
) -> str | None:
    """Upload the unfiltered batch as one JSONL object. Returns the s3:// URI, or
    None if the batch is empty."""
    if not batch:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    object_name = f"adsblol_realtime/_landing/states-{stamp}.jsonl"
    payload = "\n".join(json.dumps(entry) for entry in batch).encode("utf-8") + b"\n"
    return write_bronze_bytes(payload, object_name, bucket=bucket, content_type="application/x-ndjson", client=client)


def flush_batch_to_bronze(
    batch: list[dict[str, Any]],
    *,
    source_file: str,
    client: ObjectStoreClient | None = None,
) -> str | None:
    """Write only the Turkey-bbox rows as a Bronze Parquet object."""
    rows = turkey_rows(batch)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = add_provenance(df, source_type=SOURCE_TYPE, source_file=source_file, schema_version="bronze_v1")
    return write_bronze(df, "adsblol_realtime", client=client)


def run(
    *,
    kafka_bootstrap: str | None = None,
    topic: str | None = None,
    client: ObjectStoreClient | None = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    load_dotenv()
    bootstrap = kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_name = topic or os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC)

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": "adsblol-bronze-consumer",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic_name])

    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request_stop)
    signal.signal(signal.SIGTERM, shutdown.request_stop)

    batch: list[dict[str, Any]] = []
    logger.info("Consuming %s @ %s, landing + Bronze writes go to MinIO", topic_name, bootstrap)

    def _flush() -> None:
        if not batch:
            return
        landing_uri = land_batch_raw(batch, client=client)
        bronze_uri = flush_batch_to_bronze(batch, source_file=topic_name, client=client)
        logger.info(
            "Flushed batch of %d -> landing=%s bronze=%s",
            len(batch),
            landing_uri,
            bronze_uri or "(no Turkey rows)",
        )
        batch.clear()

    try:
        while not shutdown.stop:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.warning("Kafka error: %s", msg.error())
                continue

            batch.append(json.loads(msg.value()))
            if len(batch) >= batch_size:
                _flush()
    finally:
        _flush()
        consumer.close()
        logger.info("Consumer stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
