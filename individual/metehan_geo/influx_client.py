"""influx_client.py -- realtime (--live/--24h/--7d) veri kaynagi.

Yusuf'un dashboard_consumer.py'sinin zaten yazdigi InfluxDB bucket'ini
(adsb-history, measurement "flights", tag icao24) OKUR -- yeni bir yazici
kurmaya gerek yok. (docs/realtime_pipeline_prompt.md Soru #1: ayri modul mu --
evet, ama sadece OKUMA icin; yazma tarafi zaten Dashboard/dashboard_consumer.py'de var.)

2026-07-07 karari: Yusuf'un kendi makinesinde calisan instance'a agdan
baglanmiyoruz (kirilgan bagimlilik olurdu) -- bunun yerine AYNI docker-compose.yml
("streaming" profili) kendi makinemizde calistirilip BAGIMSIZ bir kopya
besleniyor. Sema/mimari ortak, instance degil.

historical (`data.py:load_adsb_gold_data`) 1 milyar satirlik Gold'u stream
ediyordu (RAM'e sigmiyordu); burada max 7 gunluk TEK-instance realtime veri
soz konusu (binler-milyonlar mertebesinde) -- TEK sorgu + TEK DataFrame
yeterli, generator/chunk gerekmiyor.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from influxdb_client import InfluxDBClient

logger = logging.getLogger(__name__)

INFLUX_HOST = os.environ.get("INFLUX_HOST", "http://localhost:8086")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "iha-org")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "adsb-history")


def _load_token() -> str:
    env_token = os.environ.get("INFLUX_TOKEN")
    if env_token:
        return env_token
    # docker-compose.yml'deki DOCKER_INFLUXDB_INIT_ADMIN_TOKEN ile ayni --
    # dashboard_consumer.py de bunu varsayilan olarak boyle kullaniyor.
    return "iha-influx-token-docker"


def get_influx_client() -> InfluxDBClient:
    # Varsayilan istemci timeout'u (10sn) baslangicta yeterliydi ama veri
    # birikince (2026-07-09: 7 gunluk pencere 4.38M satira ulasti) "-7d"
    # sorgusu bunu asip surekli ReadTimeoutError ile basarisiz oluyordu --
    # arka planda calisan realtime_density.py --mode 7d dongusu SESSIZCE
    # (try/except loglayip devam ediyordu) her turda basarisiz kaliyordu.
    # Olcum: ayni sorgu GERCEKTE ~170sn suruyor (InfluxDB'nin 1GB bellek
    # sinirli tek-node docker instance'i bu hacimde yavas) -- 240sn'lik
    # pay birakiyoruz.
    return InfluxDBClient(url=INFLUX_HOST, token=_load_token(), org=INFLUX_ORG, timeout=240_000)


def load_realtime_window(range_start: str, client: InfluxDBClient | None = None) -> pd.DataFrame:
    """InfluxDB'den `range_start` (Flux relatif sure, ör. "-2m"/"-24h"/"-7d")
    kadar geriye giden pencereyi ceker, tek bir DataFrame dondurur.

    Kolonlar: source_id (icao24 -- historical Gold ile AYNI isim, boylece
    clean_coordinates/assign_h3_cell/flight_count mantigi degismeden calisir),
    lat, lon, _time.
    """
    client = client or get_influx_client()
    # Flux'un SUNUCU-TARAFI pivot()'u BUYUK pencerelerde (2026-07-09: 7d
    # penceresi 4.36M satira ulasti) cok yavas/timeout oluyordu (120sn
    # istemci timeout'unu bile asiyordu). Duzeltme: pivot'u InfluxDB'ye
    # YAPTIRMAK yerine (lat, lon) satirlarini DUZ (long-format) cekip
    # pandas.pivot_table ile YERELDE (cok daha hizli) genisletiyoruz.
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r._measurement == "flights")
      |> filter(fn: (r) => r._field == "lat" or r._field == "lon")
      |> keep(columns: ["_time", "icao24", "_field", "_value"])
    '''
    result = client.query_api().query_data_frame(query, org=INFLUX_ORG)
    long_df = pd.concat(result, ignore_index=True) if isinstance(result, list) else result

    if long_df is None or long_df.empty:
        logger.warning("InfluxDB'den %s penceresinde hic veri donmedi", range_start)
        return pd.DataFrame(columns=["source_id", "lat", "lon"])

    df = long_df.pivot_table(
        index=["_time", "icao24"], columns="_field", values="_value", aggfunc="first",
    ).reset_index()
    df = df.rename(columns={"icao24": "source_id"})
    keep_cols = [c for c in ["source_id", "lat", "lon", "_time"] if c in df.columns]
    df = df[keep_cols]
    logger.info(
        "InfluxDB %s penceresi: %d satir, %d benzersiz ucak",
        range_start, len(df), df["source_id"].nunique() if "source_id" in df.columns else 0,
    )
    return df
