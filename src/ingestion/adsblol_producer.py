"""Poll adsb.lol for live aircraft state and publish raw entries to Kafka.

Phase 3 of docs/bronze_implementasyon_plani.md. Publishes each aircraft's
unmodified `ac[]` entry to Kafka, keyed by ICAO hex (`hex`). No bbox
filtering, unit conversion, or schema work happens here -- the consumer
applies the Turkey bbox filter when it writes Bronze Parquet; this producer
just relays whatever adsb.lol returns for the configured query points.

CAVEATS TO VERIFY AGAINST THE LIVE API BEFORE TRUSTING THIS IN PRODUCTION
(this sandbox has no network route to adsb.lol, so none of this could be
exercised against the real service):
  - Endpoint shape (`/v2/lat/{lat}/lon/{lon}/dist/{nm}`) matches the
    ADSBExchange-compatible v2 API adsb.lol documents itself as a drop-in
    replacement for. Check `https://api.adsb.lol/docs` for the current
    OpenAPI spec -- a v3 endpoint may now be preferred (mirrors what
    happened on the adsb.fi sibling service).
  - The Turkey bbox (lat 36-42, lon 26-45) is ~480nm corner-to-corner, which
    likely exceeds a single circle query's max radius (commonly 250nm on
    these APIs). QUERY_POINTS below is an untested 4-point grid guess to
    cover the bbox with overlap; re-tune radius/points once you can hit the
    API and see actual coverage + `total` counts in the response.
  - Confirm whether unauthenticated requests still work or whether an API
    key is now required (the docs note this may change).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Any

import requests
from confluent_kafka import Producer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

API_BASE = "https://api.adsb.lol/v2"
DEFAULT_TOPIC = "uav.raw.states"
POLL_INTERVAL_SECONDS = 60

# (lat, lon, radius_nm) -- see module docstring caveat on radius/coverage.
QUERY_POINTS: tuple[tuple[float, float, int], ...] = (
    (41.0, 29.0, 250),  # Istanbul / Marmara / Thrace
    (39.9, 41.0, 250),  # Erzurum / east Anatolia
    (37.0, 35.5, 250),  # Adana / south coast
    (39.0, 35.0, 250),  # Ankara / central Anatolia
)


class _ShutdownFlag:
    def __init__(self) -> None:
        self.stop = False

    def request_stop(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %s, will stop after the current poll", signum)
        self.stop = True


def _build_headers() -> dict[str, str]:
    api_key = os.getenv("ADSBLOL_API_KEY", "").strip()
    return {"api-auth": api_key} if api_key else {}


def fetch_point(lat: float, lon: float, dist_nm: int, *, session: requests.Session) -> list[dict[str, Any]]:
    """Return the raw, unmodified `ac` list for one query point."""
    url = f"{API_BASE}/lat/{lat}/lon/{lon}/dist/{dist_nm}"
    response = session.get(url, headers=_build_headers(), timeout=15)
    response.raise_for_status()
    return response.json().get("ac", [])


def poll_once(session: requests.Session, points: tuple[tuple[float, float, int], ...] = QUERY_POINTS) -> dict[str, dict[str, Any]]:
    """Query every configured point and de-duplicate aircraft by hex."""
    by_hex: dict[str, dict[str, Any]] = {}
    for lat, lon, dist in points:
        try:
            for entry in fetch_point(lat, lon, dist, session=session):
                hex_id = entry.get("hex")
                if hex_id:
                    by_hex[hex_id] = entry
        except requests.RequestException:
            logger.exception("adsb.lol request failed for point (%s, %s, %s)", lat, lon, dist)
    return by_hex


def run(
    *,
    kafka_bootstrap: str | None = None,
    topic: str | None = None,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> None:
    load_dotenv()
    bootstrap = kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_name = topic or os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC)

    producer = Producer({"bootstrap.servers": bootstrap})
    session = requests.Session()
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request_stop)
    signal.signal(signal.SIGTERM, shutdown.request_stop)

    logger.info("Polling adsb.lol every %ss, publishing to %s @ %s", poll_interval, topic_name, bootstrap)
    while not shutdown.stop:
        aircraft = poll_once(session)
        for hex_id, entry in aircraft.items():
            producer.produce(topic_name, key=hex_id.encode("utf-8"), value=json.dumps(entry).encode("utf-8"))
        producer.flush()
        logger.info("Published %d aircraft", len(aircraft))

        for _ in range(poll_interval):
            if shutdown.stop:
                break
            time.sleep(1)

    producer.flush()
    logger.info("Producer stopped cleanly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
