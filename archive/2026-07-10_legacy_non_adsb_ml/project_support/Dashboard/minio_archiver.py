"""
minio_archiver.py
Kafka adsb.flights → MinIO Bronze (adsblol_realtime/_landing/) arşivleyici.

dashboard_consumer.py ile AYRI group.id kullanır, aynı mesajları bağımsız
okur -- dashboard_consumer'ı etkilemez.

Her BATCH_SIZE mesajda bir (ya da FLUSH_SECS saniye geçince) JSONL dosyası
MinIO bronze/adsblol_realtime/_landing/states-<timestamp>.jsonl olarak yazar.
parse_adsblol_realtime.py bu dosyaları Silver Parquet'e çevirir VE ISLENEN
dosyalari Bronze'dan siler (bkz. o script'teki _delete_processed).

2026-07-09 KARARI: MinIO'da 7 GUNLUK OTOMATIK SILME KURALI YOK/KALDIRILDI --
sadece InfluxDB'de (adsb-history bucket, 168h retention) gecici 7 gunluk
saklama var. MinIO'daki realtime landing verisi (bu dosya) kalici olmali:
parse_adsblol_realtime.py --loop calisir durumda tutularak duzenli araliklarla
Silver'a (ve oradan Gold'a) islenip KALICI hale getirilir -- silinmez, sadece
"islendi" (Bronze landing'den Silver'a tasindi) olur. --loop calismiyorsa
Bronze'da JSONL birikir (bu ZARARSIZ, sadece disk kullanir) ama KAYBOLMAZ.

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

BOOTSTRAP    = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
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


def remove_lifecycle_if_present(client: Minio, bucket: str) -> None:
    """MinIO'da otomatik silme kurali OLMAMALI (2026-07-09 karari, bkz. modul
    docstring) -- daha once ensure_lifecycle() ile ayarlanmis "rt-7day-expire"
    kurali varsa kaldirir. Kural hic yoksa sessizce gecer."""
    try:
        client.delete_bucket_lifecycle(bucket)
        print(f"Lifecycle kurali kaldirildi (varsa): {bucket}")
    except Exception as e:
        print(f"[bilgi] lifecycle kaldirma atlandi (muhtemelen zaten yoktu): {e}")


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
    remove_lifecycle_if_present(client, BUCKET)

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
