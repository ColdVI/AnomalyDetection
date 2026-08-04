"""Canli varis-tahmini servisi -- individual/metehan_geo/viz/index.html'in
"Canli Ucaklar" modu icin. individual/metehan_varis_tahmini/ altindaki
egitilmis GAT+Transformer iki-kule modelini (bkz. o klasordeki train_full.py)
CANLI (Redis/InfluxDB'den beslenen) uçak izlerine karsi calistirir.

Mimari: Redis/InfluxDB zaten Dashboard/codes/dashboard_consumer.py tarafindan
besleniyor (docker-compose --profile streaming up -d) -- burada YENI bir
yazici YOK, sadece OKUMA + model forward-pass. Havalimani embedding tablosu
(852x32) sorgu-bagimsiz oldugu icin SADECE ACILISTA bir kez hesaplanir
(bkz. train_full.py'deki ayni desen: item_tower(x, adj) her batch'te degil
bir kez cagrilir) -- her /api/predict cagrisi sadece kucuk QueryTower
forward-pass'i + 852'lik bir matmul yapar (CPU'da tek haneli milisaniye).

min_route_count/kesme-noktasi (bkz. gunluk_kayit.md 2026-08-01) politikasi
EGITIM/RAPORLAMA icindi -- canli serving'de gercek varisi (etiketi) onceden
BILMEDIGIMIZ icin "bu rota 10+ kez uculmus mu" diye bir filtre burada
UYGULANAMAZ. Sadece kesme-noktasi (>=15dk gozlem) politikasi dogrudan
tasinir -- AMA "gozlem" burada GERCEK kalkistan itibaren sayilir, "biz bu
ucagi ne zamandir izliyoruz"dan DEGIL (kullanici tespiti, 2026-08-01): canli
sistemde bir ucagi ilk gordugumuz an onun kalkisi olmak zorunda degil, zaten
havadayken de goruntuye girmis olabilir. Gercek kalkis, is_ground alanindaki
true->false gecisi DOGRUDAN gozlemlenerek tespit edilir (bkz.
_find_takeoff_index) -- bu gecis gozlemlenemezse (ucak zaten havadayken
goruldu) tahmin YAPILMAZ, "kalkis_gorulmedi" donulur.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import redis
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from individual.metehan_varis_tahmini.airport_gcn import load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH
from individual.metehan_varis_tahmini.airport_gat import AirportGAT, build_adjacency_mask
from individual.metehan_varis_tahmini.query_tower import QueryTower, POINT_DIM
from individual.metehan_geo.influx_client import get_influx_client, INFLUX_BUCKET, INFLUX_ORG

logger = logging.getLogger(__name__)

# Dashboard/codes/dashboard_consumer.py'nin AYNI ortam-degiskeni deseni --
# o modulu DOGRUDAN import ETMIYORUZ (confluent_kafka gibi bu servisin hic
# ihtiyaci olmayan agir bagimliliklari surukler, sadece 2 sabit icin).
import os
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

CHECKPOINT_NAME = "model_checkpoint_gat_aug_opensky_h64o32_best.pt"
CHECKPOINT_FILE = Path(__file__).resolve().parents[1] / "metehan_varis_tahmini" / CHECKPOINT_NAME
AIRPORTS_CSV = Path(__file__).resolve().parents[1] / "metehan_geo_country" / "data" / "airports.csv"
# 2026-08-01: her "ok" tahmini buraya (JSONL, bir satir = bir tahmin anı)
# kaydedilir -- check_prediction_accuracy.py gunler icinde bu kaydi okuyup
# ucak inince gercek varisin top-3'te olup olmadigini kontrol eder.
PREDICTION_LOG_PATH = Path(__file__).parent / "prediction_log.jsonl"

HIDDEN_DIM = 64
OUT_DIM = 32
MAX_SEQ_LEN = 300
MIN_POINTS = 3
MIN_ELAPSED_MIN = 15.0  # bkz. gunluk 2026-08-01: kesme<15dk icin tahmin GUVENILIR degil (olculdu)
# 2026-08-01 BUG DUZELTMESI (kullanici bulgusu: "uzun suredir live acik ama
# hala cogu ucak gri neden tahmin edemiyoruz?"): -2h onceki degeriyle, 2
# saatten UZUN suren HERHANGI bir ucusun gercek kalkisi (is_ground gecisi)
# bu pencerenin DISINDA kalip _find_takeoff_index None donuyordu --
# InfluxDB'de o gecis GERCEKTEN KAYITLIYDI, sadece sorgu yeterince geriye
# bakmiyordu. Bu, TAKEOFF'UN KENDISI gozlemlenemedigi durumdan (gercek
# "kalkis_gorulmedi") FARKLI bir sahte-negatif kaynagiydi. -8h'ye
# genisletildi: coğu gercek ucus (bolgesel/kita-ici) suresini kapsar, ve
# bugunku (Docker'in kendi ~birkac saatlik) toplam veri hacmi zaten bunu
# asmadigi icin sorgu maliyeti BUGUN icin degismedi -- InfluxDB'nin
# rolling penceresi zamanla buyudukce toplu sorgu (_fetch_all_live_trajectories)
# suresi olculup gerekirse yeniden ayarlanmali (bkz. PREDICTION_CACHE_REFRESH_SECONDS).
LIVE_WINDOW = "-8h"
TOP_K = 3

# 2026-08-01 (kullanici istegi): "henuz available olmayan ucaklari farkli
# renkle goster, sadece tahmin yapabildiklerimize tiklayabilelim" -- bunun
# icin HER ucagin tek tek /api/predict ile (ucak basina 1 InfluxDB sorgusu)
# durumunu bilmemiz lazim, ama su an ~7000+ aktif ucak var -- istek basina
# binlerce sorgu atmak (frontend her 5sn'de bir /api/live_aircraft cekiyor)
# InfluxDB'yi (tek-node, bellek sinirli) kilitler. Cozum: TEK bir bulk sorguyla
# (TUM ucaklarin izini bir kerede ceken _fetch_all_live_trajectories) periyodik
# (arka plan thread'i) bir "tahmin durumu" onbellegi hesaplanir, /api/live_aircraft
# bu onbellekten O(1) okur. Olculdu (2026-08-01, ~15.5k ucak): sorgu ~20sn,
# pivot+siniflandirma ~4sn -- REFRESH_SECONDS bunu rahat karsilayacak sekilde
# genis tutuldu (arka plan thread'i zaten sirali/kendini-kisitlayan bir dongu,
# ust uste binme riski yok).
# 2026-08-01 (olcum): tam dongu (sorgu+pivot+siniflandirma) ~15-20bin ucakta
# ~24sn suruyor -- bu sure boyunca (pandas GIL'i CPU-yogun asamalarda tutabiliyor)
# es zamanli /api/ istekleri gecikebiliyor (gozlemlendi: bazen 10-15sn'lik
# yanit gecikmesi). 60sn yerine 120sn secildi -- durum zaten dakikalar
# mertebesinde degisiyor (15dk esigi), saniye hassasiyeti gerekmiyor, boylece
# bu gecikme penceresi toplam surenin ~%20'sinden ~%10'una dusuyor.
PREDICTION_CACHE_REFRESH_SECONDS = 120

# Egitim verisinin gercek cografi kapsamindan (852 havalimaninin bbox'i)
# ne kadar disina "pay" birakilir -- kiyi/sinir havalimanlarina yakin
# ucuslari yanlislikla disarida birakmamak icin (bkz. startup'taki bbox
# hesabi). ABD gibi kitalar arasi mesafeler zaten bu paydan COK daha
# buyuk oldugu icin yanlis-pozitif (ABD'yi ice alma) riski yok.
SCOPE_MARGIN_DEG = 8.0

app = FastAPI(title="Varis Tahmini -- Canli Servis")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)  # kisisel/localhost arac -- internete acik degil, CORS kisitlamasi gerekmiyor

_state: dict = {}  # startup'ta doldurulur: item_emb, airport_to_idx, idx_to_airport, query_tower, airports_df, rdb, influx_client


def _load_airports_lookup() -> pd.DataFrame:
    df = pd.read_csv(AIRPORTS_CSV, usecols=["ident", "name", "latitude_deg", "longitude_deg"])
    return df.set_index("ident")


@app.on_event("startup")
def load_model():
    logger.info("Model yukleniyor: %s", CHECKPOINT_FILE)
    if not CHECKPOINT_FILE.exists():
        raise RuntimeError(f"Checkpoint bulunamadi: {CHECKPOINT_FILE}")

    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)

    raw = json.loads(CHECKPOINT_PATH.read_text())
    from collections import Counter
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    adj = build_adjacency_mask(route_counts, idents)

    item_tower = AirportGAT(in_dim=x.shape[1], hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, n_gat_layers=2, n_heads=4)
    query_tower = QueryTower(d_model=HIDDEN_DIM, nhead=4, num_layers=2, out_dim=OUT_DIM)

    ckpt = torch.load(CHECKPOINT_FILE, weights_only=False, map_location="cpu")
    item_tower.load_state_dict(ckpt["item_tower"])
    query_tower.load_state_dict(ckpt["query_tower"])
    item_tower.eval()
    query_tower.eval()

    # airport_to_idx CHECKPOINT'TEN okunuyor (parquet satir sirasindan DEGIL)
    # -- ileride airport_features.parquet yeniden uretilirse sira degisebilir,
    # checkpoint kendi egitildigi haritayi tasir (bkz. train_full.py kaydi).
    airport_to_idx: dict[str, int] = ckpt["airport_to_idx"]
    idx_to_airport = {i: a for a, i in airport_to_idx.items()}

    with torch.no_grad():
        # SORGU-BAGIMSIZ: bir kez hesaplanip TUM /api/predict cagrilarinda
        # yeniden kullanilir (train_full.py'nin evaluate() icindeki ayni desen).
        item_emb = item_tower(x, adj)

    _state["item_emb"] = item_emb
    _state["query_tower"] = query_tower
    _state["airport_to_idx"] = airport_to_idx
    _state["idx_to_airport"] = idx_to_airport
    _state["airports_df"] = _load_airports_lookup()
    _state["rdb"] = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True, protocol=2)
    _state["influx"] = get_influx_client()
    _state["prediction_status_cache"] = {}

    # 2026-08-01 (kullanici bulgusu -- "biz modeli ABD tarafi olmadan
    # egittik, model simdi neye gore ABD'deki ucaklari tahmin ediyor?"):
    # DOGRULANDI -- 852 havalimanlik egitim graf'inda TEK BIR ABD havalimani
    # (ICAO "K" onekli) yok, lat/lon araligi da Amerika kitasina hic
    # girmiyor (lon max 144.19E, ABD icin ~-125..-65 gerekirdi). Yani model
    # ABD ucaklari icin verdigi "tahmin" bile aslinda YANLIŞ -- gercek varis
    # zaten 852'lik cikti kumesinde YOK, ne cikarsa cıksin dogru olamaz.
    # Duzeltme: egitilen 852 havalimaninin gercek bbox'i (+ pay) HESAPLANIP
    # kaydediliyor, _classify_trajectory/predict() bu kapsamin DISINDAKI
    # ucaklar icin tahmin YAPMIYOR ("kapsam_disi" donuyor).
    trained_airports_df = _state["airports_df"].loc[_state["airports_df"].index.isin(airport_to_idx.keys())]
    margin = SCOPE_MARGIN_DEG
    _state["scope_bbox"] = (
        float(trained_airports_df["latitude_deg"].min()) - margin,
        float(trained_airports_df["latitude_deg"].max()) + margin,
        float(trained_airports_df["longitude_deg"].min()) - margin,
        float(trained_airports_df["longitude_deg"].max()) + margin,
    )
    logger.info(
        "Egitim kapsami (bbox, +%.0f derece pay): lat [%.1f, %.1f], lon [%.1f, %.1f]",
        margin, *_state["scope_bbox"],
    )
    logger.info("Model hazir: %d havalimani, embedding boyutu %s", item_emb.shape[0], tuple(item_emb.shape))

    threading.Thread(target=_prediction_status_loop, daemon=True).start()


@app.get("/api/live_aircraft")
def live_aircraft():
    """Redis'teki 'su an aktif' ucak kumesinden (Dashboard/codes/dashboard_consumer.py
    zaten yaziyor) marker cizimi icin hafif bir liste doner."""
    rdb = _state["rdb"]
    icaos = rdb.smembers("iha:active_flights")
    cache = _state.get("prediction_status_cache", {})
    out = []
    for icao in icaos:
        raw = rdb.get(f"iha:state:{icao}")
        if not raw:
            continue
        d = json.loads(raw)
        # onbellek arka planda periyodik (bkz. PREDICTION_CACHE_REFRESH_SECONDS)
        # hesaplaniyor -- bu ucak henuz siniflandirilmamissa (yeni gorulmus,
        # bir sonraki turu bekliyor) "bilinmiyor" donuyoruz, frontend bunu da
        # "henuz tiklanamaz" olarak ele aliyor.
        status_info = cache.get(icao, {"status": "bilinmiyor"})
        out.append({
            "icao24": icao,
            "callsign": (d.get("callsign") or "").strip(),
            "lat": d.get("lat"),
            "lon": d.get("lon"),
            "track": d.get("track"),
            "alt": d.get("alt"),
            "velocity": d.get("velocity"),
            "is_military": bool(d.get("is_military", False)),
            "predict_status": status_info["status"],
            "elapsed_min": status_info.get("elapsed_min"),
        })
    return {"aircraft": out}


def _fetch_live_trajectory(icao24: str) -> pd.DataFrame:
    """Tek bir ucagin InfluxDB'deki son izini (lat/lon/alt/velocity/track/
    vertical_rate/is_ground) zaman sirali ceker. influx_client.load_realtime_window
    SADECE lat/lon/is_military cekiyor (yogunluk haritasi icin yeterliydi) --
    burada modelin ihtiyac duydugu TUM alanlar gerektigi icin ayri bir sorgu.
    is_ground: GERCEK kalkis anini tespit etmek icin sart (bkz. _find_takeoff_index)."""
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {LIVE_WINDOW})
      |> filter(fn: (r) => r._measurement == "flights")
      |> filter(fn: (r) => r.icao24 == "{icao24}")
      |> filter(fn: (r) => r._field == "lat" or r._field == "lon" or r._field == "alt"
                          or r._field == "velocity" or r._field == "track" or r._field == "vertical_rate"
                          or r._field == "is_ground")
      |> keep(columns: ["_time", "_field", "_value"])
    '''
    result = _state["influx"].query_api().query_data_frame(query, org=INFLUX_ORG)
    long_df = pd.concat(result, ignore_index=True) if isinstance(result, list) else result
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    df = long_df.pivot_table(index="_time", columns="_field", values="_value", aggfunc="first").reset_index()
    df = df.sort_values("_time").reset_index(drop=True)
    return df


def _find_takeoff_index(df: pd.DataFrame) -> int | None:
    """GERCEK kalkis anini (is_ground: true -> false gecisi) DOGRUDAN
    gozlemleyerek bulur -- "biz bu ucagi ne zamandir izliyoruz" ile "ucak
    ne zamandir havada" AYNI SEY DEGIL (kullanici tespiti, 2026-08-01):
    egitim verisindeki her session gercekten kalkistan basliyordu, ama
    canli sistemde bir ucagi ILK gordugumuz an onun kalkisi olmak zorunda
    degil -- zaten havadayken de gorunmus olabilir. Boyle bir gecis
    BULUNAMAZSA (ör. is_ground hic yazilmamis, ya da ucak pencerenin
    basindan beri zaten havada -- kendi kalkisini hic gormedik), None
    doner ve tahmin YAPILMAZ (yaniltici bir dt_norm=0 uydurmak yerine)."""
    if "is_ground" not in df.columns:
        return None
    ground = df["is_ground"].astype(float).fillna(-1)  # NaN'i ne True ne False sayma
    for i in range(1, len(ground)):
        if ground.iloc[i - 1] > 0.5 and ground.iloc[i] == 0.0:
            return i
    return None


def _in_training_scope(lat: float, lon: float) -> bool:
    """852 havalimanlik egitim kumesinin cografi kapsami DISINDAKI ucaklar
    icin tahmin YAPMAMAK gerekiyor -- model o bolgeden HICBIR havalimani
    gormedi (bkz. startup'taki bbox hesabi ve kullanici bulgusu 2026-08-01),
    yani cikacagi "tahmin" ne olursa olsun dogru olamaz (gercek varis zaten
    852'lik cikti kumesinde yok)."""
    lat_min, lat_max, lon_min, lon_max = _state["scope_bbox"]
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _fetch_all_live_trajectories() -> pd.DataFrame:
    """_fetch_live_trajectory'nin TEK-ucak halinin bulk (TUM aktif ucaklar
    icin TEK sorgu) versiyonu -- her ucak icin ayri sorgu atmak yerine, TUM
    LIVE_WINDOW penceresini bir kerede cekip icao24'e gore grupluyoruz."""
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {LIVE_WINDOW})
      |> filter(fn: (r) => r._measurement == "flights")
      |> filter(fn: (r) => r._field == "lat" or r._field == "lon" or r._field == "alt"
                          or r._field == "velocity" or r._field == "track" or r._field == "vertical_rate"
                          or r._field == "is_ground")
      |> keep(columns: ["_time", "icao24", "_field", "_value"])
    '''
    result = _state["influx"].query_api().query_data_frame(query, org=INFLUX_ORG)
    long_df = pd.concat(result, ignore_index=True) if isinstance(result, list) else result
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    df = long_df.pivot_table(
        index=["_time", "icao24"], columns="_field", values="_value", aggfunc="first",
    ).reset_index()
    return df.sort_values(["icao24", "_time"])


def _classify_trajectory(df_g: pd.DataFrame) -> tuple[dict, pd.DataFrame | None, pd.Series | None]:
    """predict()'in durum-belirleme kismiyla AYNI mantik (model forward-pass'i
    HARIC) -- bulk onbellek icin, tek bir ucagin (zaten icao24'e gore
    gruplanmis) izini durum etiketine indirger. status "ok" ise, cagiran
    tarafin (bkz. _refresh_prediction_status_cache) modeli TEKRAR
    calistirmak icin veriyi yeniden hesaplamasina gerek kalmasin diye
    (df, elapsed_min) da doner -- aksi halde (None, None)."""
    if len(df_g) < MIN_POINTS:
        return {"status": "yetersiz_veri", "n_points": len(df_g)}, None, None
    takeoff_i = _find_takeoff_index(df_g)
    if takeoff_i is None:
        return {"status": "kalkis_gorulmedi", "n_points": len(df_g)}, None, None
    df = df_g.iloc[takeoff_i:].reset_index(drop=True)
    n = len(df)
    if n < MIN_POINTS:
        return {"status": "yetersiz_veri", "n_points": n}, None, None
    if not _in_training_scope(float(df["lat"].iloc[-1]), float(df["lon"].iloc[-1])):
        return {"status": "kapsam_disi", "n_points": n}, None, None
    elapsed_min = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds() / 60.0
    total_elapsed = float(elapsed_min.iloc[-1])
    if total_elapsed < MIN_ELAPSED_MIN:
        return {"status": "belirsiz_erken", "n_points": n, "elapsed_min": round(total_elapsed, 1)}, None, None
    return {"status": "ok", "n_points": n, "elapsed_min": round(total_elapsed, 1)}, df, elapsed_min


def _refresh_prediction_status_cache() -> None:
    try:
        df = _fetch_all_live_trajectories()
        if df.empty:
            logger.warning("Tahmin durumu onbellegi: InfluxDB'den veri gelmedi")
            return
        # 2026-08-01 (kullanici istegi -- "gercekten canli takip edip top3'e
        # indi mi baktik mi?"): her "ok" ucus icin (bir kez, ayni ucus
        # tekrar tekrar loglanmasin diye logged_icaos ile takip edilir)
        # model calistirilip tahmin diske kaydedilir -- gunler icinde bu
        # ucaklar inince check_prediction_accuracy.py gercek varisi
        # top-3'le karsilastirabilecek.
        logged_icaos: set[str] = _state.setdefault("logged_icaos", set())
        cache = {}
        n_logged_this_cycle = 0
        for icao, g in df.groupby("icao24", sort=False):
            status_info, post_takeoff_df, elapsed_min = _classify_trajectory(g.reset_index(drop=True))
            cache[icao] = status_info
            if status_info["status"] == "ok":
                if icao not in logged_icaos:
                    predictions = _run_model_prediction(post_takeoff_df, elapsed_min)
                    _log_prediction(
                        icao, float(post_takeoff_df["lat"].iloc[-1]), float(post_takeoff_df["lon"].iloc[-1]),
                        status_info["elapsed_min"], predictions,
                    )
                    logged_icaos.add(icao)
                    n_logged_this_cycle += 1
            else:
                # ucus artik "ok" degil (indi, kayboldu, vs.) -- bir SONRAKI
                # ucusu tekrar loglayabilelim diye takipten cikar.
                logged_icaos.discard(icao)
        _state["prediction_status_cache"] = cache
        n_ok = sum(1 for v in cache.values() if v["status"] == "ok")
        logger.info(
            "Tahmin durumu onbellegi guncellendi: %d ucak (%d 'ok', %d yeni loglandi)",
            len(cache), n_ok, n_logged_this_cycle,
        )
    except Exception:
        logger.exception("Tahmin durumu onbellegi guncellenirken hata -- bir sonraki turda tekrar denenecek")


def _prediction_status_loop() -> None:
    while True:
        _refresh_prediction_status_cache()
        time.sleep(PREDICTION_CACHE_REFRESH_SECONDS)


def _sample_indices(n: int, max_len: int) -> list[int]:
    if n <= max_len:
        return list(range(n))
    idx = np.linspace(0, n - 1, max_len)
    return idx.round().astype(int).tolist()


def _run_model_prediction(df: pd.DataFrame, elapsed_min: pd.Series) -> list[dict]:
    """Model forward-pass'i -- predict() VE bulk onbellek dongusu (kayit
    icin) AYNI kodu kullanir, iki kez yazilmasin diye 2026-08-01'de
    fonksiyona cikarildi."""
    n = len(df)
    idx = _sample_indices(n, MAX_SEQ_LEN)
    sub = df.iloc[idx]
    sub_elapsed = elapsed_min.iloc[idx]
    dt_norm = (sub_elapsed / max(sub_elapsed.max(), 1e-6)).to_numpy(dtype=np.float32)

    m = len(idx)
    points = np.zeros((m, POINT_DIM), dtype=np.float32)
    points[:, 0] = sub["lat"].to_numpy(dtype=np.float32)
    points[:, 1] = sub["lon"].to_numpy(dtype=np.float32)
    points[:, 2] = sub["alt"].fillna(0).to_numpy(dtype=np.float32) / 1000.0
    points[:, 3] = sub["velocity"].fillna(0).to_numpy(dtype=np.float32) / 100.0
    heading_rad = np.radians(sub["track"].fillna(0).to_numpy(dtype=np.float32))
    points[:, 4] = np.sin(heading_rad)
    points[:, 5] = np.cos(heading_rad)
    points[:, 6] = sub["vertical_rate"].fillna(0).to_numpy(dtype=np.float32) / 10.0
    points[:, 7] = dt_norm

    points_t = torch.from_numpy(points).unsqueeze(0)  # (1, m, 8)
    lengths_t = torch.tensor([m], dtype=torch.long)

    with torch.no_grad():
        query_emb = _state["query_tower"](points_t, lengths_t)
        logits = query_emb @ _state["item_emb"].T
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_probs, top_idx = probs.topk(TOP_K)

    airports_df = _state["airports_df"]
    idx_to_airport = _state["idx_to_airport"]
    predictions = []
    for prob, ai in zip(top_probs.tolist(), top_idx.tolist()):
        ident = idx_to_airport[ai]
        row = airports_df.loc[ident] if ident in airports_df.index else None
        predictions.append({
            "ident": ident,
            "name": (row["name"] if row is not None else ident),
            "lat": (float(row["latitude_deg"]) if row is not None else None),
            "lon": (float(row["longitude_deg"]) if row is not None else None),
            "prob": round(float(prob), 4),
        })
    return predictions


def _log_prediction(icao24: str, current_lat: float, current_lon: float, elapsed_min: float, predictions: list[dict]) -> None:
    """2026-08-01 (kullanici istegi -- "gercekten canli bir ucagi takip
    edip top3'e indi mi diye baktik mi?"): her "ok" tahmini diske (JSONL)
    kaydeder ki gunler icinde ucak inince check_prediction_accuracy.py ile
    gercek varisi top-3'le KARSILASTIRABILELIM -- gormeden/kaydetmeden
    sonradan dogrulamanin bir yolu yok."""
    try:
        entry = {
            "icao24": icao24,
            "logged_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "current_lat": current_lat,
            "current_lon": current_lon,
            "elapsed_min": elapsed_min,
            "predictions": predictions,
        }
        with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        logger.exception("Tahmin loglanirken hata (yoksayilip devam ediliyor)")


@app.get("/api/predict/{icao24}")
def predict(icao24: str):
    icao24 = icao24.lower()
    raw_df = _fetch_live_trajectory(icao24)
    if len(raw_df) < MIN_POINTS:
        return {"icao24": icao24, "status": "yetersiz_veri", "n_points": len(raw_df)}

    takeoff_i = _find_takeoff_index(raw_df)
    if takeoff_i is None:
        # is_ground:true->false gecisi GOZLEMLENEMEDI -- ucak zaten havadayken
        # goruntuye girmis olabilir (kalkisini biz izlemedik) ya da is_ground
        # hic yazilmamis. dt_norm=0'i "az once kalkti" gibi UYDURMAK yerine
        # durumu ACIKCA bildiriyoruz -- bu ucak, biz kalkisindan itibaren
        # takip etmeye baslayana kadar tahmin edilmeyecek.
        return {"icao24": icao24, "status": "kalkis_gorulmedi", "n_points": len(raw_df)}

    df = raw_df.iloc[takeoff_i:].reset_index(drop=True)  # SADECE havadaki (kalkis sonrasi) noktalar
    n = len(df)
    if n < MIN_POINTS:
        return {"icao24": icao24, "status": "yetersiz_veri", "n_points": n}

    if not _in_training_scope(float(df["lat"].iloc[-1]), float(df["lon"].iloc[-1])):
        # 2026-08-01 (kullanici bulgusu): egitim verisi ABD dahil hicbir
        # Amerika kitasi havalimani icermiyor (bkz. startup'taki bbox
        # hesabi) -- bu bolgedeki bir ucak icin model ne cikarirsa cıksin
        # YANLIS olur (gercek varis zaten 852'lik cikti kumesinde yok),
        # bu yuzden tahmin hic DENENMIYOR.
        return {"icao24": icao24, "status": "kapsam_disi", "n_points": n}

    elapsed_min = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds() / 60.0
    total_elapsed = float(elapsed_min.iloc[-1])
    if total_elapsed < MIN_ELAPSED_MIN:
        # bkz. gunluk 2026-08-01: evaluate_by_truncation.py ile olculdu --
        # ilk 15dk'da tahmin GUVENILIR degil, kesin cevap yerine bekletiyoruz.
        return {
            "icao24": icao24, "status": "belirsiz_erken",
            "n_points": n, "elapsed_min": round(total_elapsed, 1),
            "min_elapsed_min": MIN_ELAPSED_MIN,
        }

    predictions = _run_model_prediction(df, elapsed_min)
    _log_prediction(icao24, float(df["lat"].iloc[-1]), float(df["lon"].iloc[-1]), round(total_elapsed, 1), predictions)

    return {
        "icao24": icao24, "status": "ok",
        "n_points": n, "elapsed_min": round(total_elapsed, 1),
        "current_lat": float(df["lat"].iloc[-1]), "current_lon": float(df["lon"].iloc[-1]),
        "predictions": predictions,
    }


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8010)
