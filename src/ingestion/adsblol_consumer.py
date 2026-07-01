"""Consume adsb.lol Kafka messages: land raw JSONL, write Bronze Parquet.

See docs/PIPELINE_PLAN.md (YUSUF REHBERİ). Per ADR-003, every Bronze artifact
lives in MinIO, not local disk, and there is no geographic filter -- every
message worldwide is kept.

Two MinIO objects per flushed batch:
  - `bronze/adsblol_realtime/_landing/states-<batch-stamp>.jsonl`: every
    message in the batch, exactly as produced -- the true raw landing. MinIO
    has no native "append to object" operation, so landing is batched at the
    same cadence as the Parquet flush rather than streamed line-by-line.
  - `bronze/adsblol_realtime/part-*.parquet`: the same batch, with standard
    provenance columns added.

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

from src.common.minio_io import ObjectStoreClient, write_bronze, write_bronze_bytes
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
    """Write the whole batch as a Bronze Parquet object (no geo filter, ADR-003)."""
    if not batch:
        return None
    df = pd.DataFrame(batch)
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
            bronze_uri or "(empty batch)",
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
