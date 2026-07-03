"""
minio_archiver.py
Kafka adsb.flights → MinIO Bronze (adsblol_realtime/_landing/) arşivleyici.

dashboard_consumer.py ile AYRI group.id kullanır, aynı mesajları bağımsız
okur -- dashboard_consumer'ı etkilemez.

Her BATCH_SIZE mesajda bir (ya da FLUSH_SECS saniye geçince) JSONL dosyası
MinIO bronze/adsblol_realtime/_landing/states-<timestamp>.jsonl olarak yazar.
parse_adsblol_realtime.py bu dosyaları Silver Parquet'e çevirir.

7 günlük saklama: MinIO bucket'ına lifecycle kural ile ayarlanır (bkz. main()).

Kullanim:
    python minio_archiver.py
"""
import json
import time
import os
import io
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import Consumer
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP    = "localhost:9092"
TOPIC        = "adsb.flights"
GROUP_ID     = "minio-archiver"
BATCH_SIZE   = 500          # kaç mesajda bir dosya yaz
FLUSH_SECS   = 60           # en fazla kaç saniyede bir yaz (mesaj az olsa bile)
BRONZE_PREFIX = "adsblol_realtime/_landing/"
BUCKET       = os.getenv("MINIO_BRONZE_BUCKET", "bronze")


def get_minio() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )


def ensure_lifecycle(client: Minio, bucket: str) -> None:
    """7 günlük silme kuralı — sadece adsblol_realtime/_landing/ prefix'i için."""
    from minio.lifecycleconfig import LifecycleConfig, Rule, Filter, Expiration
    try:
        cfg = LifecycleConfig([
            Rule(
                "Enabled",
                rule_filter=Filter(prefix=BRONZE_PREFIX),
                rule_id="rt-7day-expire",
                expiration=Expiration(days=7),
            )
        ])
        client.set_bucket_lifecycle(bucket, cfg)
        print(f"Lifecycle kural ayarlandi: {BRONZE_PREFIX} -> 7 gun sonra silinir")
    except Exception as e:
        print(f"[uyari] lifecycle ayarlanamadi (elle yapilabilir): {e}")


def flush(client: Minio, bucket: str, lines: list[str]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    obj_name = f"{BRONZE_PREFIX}states-{ts}.jsonl"
    data = "\n".join(lines).encode("utf-8")
    client.put_object(bucket, obj_name, io.BytesIO(data), length=len(data),
                      content_type="application/x-ndjson")
    print(f"[{ts}] {len(lines)} mesaj → {obj_name}")


def main() -> None:
    client = get_minio()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)
    ensure_lifecycle(client, BUCKET)

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([TOPIC])
    print(f"MinIO arşivleyici hazır (topic={TOPIC}, batch={BATCH_SIZE}, flush={FLUSH_SECS}s)")
    print("Durdurmak için Ctrl+C\n")

    lines: list[str] = []
    last_flush = time.time()

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                pass
            elif msg.error():
                print(f"  [uyari] kafka: {msg.error()}")
            else:
                lines.append(msg.value().decode("utf-8", errors="replace"))

            now = time.time()
            if lines and (len(lines) >= BATCH_SIZE or now - last_flush >= FLUSH_SECS):
                flush(client, BUCKET, lines)
                lines = []
                last_flush = now

    except KeyboardInterrupt:
        if lines:
            flush(client, BUCKET, lines)
        print("Durduruldu.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
