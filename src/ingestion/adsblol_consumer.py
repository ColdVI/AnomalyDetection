"""Consume adsb.lol Kafka messages and land raw JSONL to MinIO Bronze.

See docs/PIPELINE_PLAN.md (YUSUF REHBERİ). Per ADR-003, Bronze = raw only.
Every batch is written as ONE JSONL object -- no Parquet, no provenance, no
unit conversion, no geo filter. Silver (src/silver/parse_adsblol_realtime.py)
reads these JSONL files and produces the parsed Parquet.

MinIO landing path: bronze/adsblol_realtime/_landing/states-<timestamp>.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Consumer
from dotenv import load_dotenv

from src.common.minio_io import ObjectStoreClient, write_bronze_bytes

logger = logging.getLogger(__name__)

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
    """Upload the batch as one JSONL object to Bronze. Returns the s3:// URI or None."""
    if not batch:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    object_name = f"adsblol_realtime/_landing/states-{stamp}.jsonl"
    payload = "\n".join(json.dumps(entry) for entry in batch).encode("utf-8") + b"\n"
    return write_bronze_bytes(
        payload, object_name, bucket=bucket, content_type="application/x-ndjson", client=client
    )


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
    logger.info("Consuming %s @ %s, writing raw JSONL to MinIO bronze", topic_name, bootstrap)

    def _flush() -> None:
        if not batch:
            return
        uri = land_batch_raw(batch, client=client)
        logger.info("Flushed %d messages -> %s", len(batch), uri)
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
