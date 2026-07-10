"""geo_clustering.py -- dbscan_geo_hotspot_prompt.md.

A-B havaalani eslestirmesinden BAGIMSIZ bir yaklasim: ham H3 hex
yogunlugunu (build_flight_density.py ciktisi, zaten var -- Gold'u TEKRAR
TARAMAYA gerek yok) dogrudan kumeleyip hava trafiginin "hub" bolgelerini
onceden tanimlanmis ulke/idari sinir KULLANMADAN kesfediyoruz.

Granularite karari (2026-07-09): hex centroid + flight_count. DBSCAN
agirlikli ornek desteklemiyor -- flight_count'a orantili tekrarlama (weighted
resampling) toplam 74.4M nokta cikarirdi (pratik degil). Bunun yerine:
DBSCAN kumelemesi ICIN hex'ler DUZ (agirliksiz) kullaniliyor, ama SADECE
belirli bir yogunluk esigini gecen ("gercekten yogun") hex'ler dahil
ediliyor -- boylece kitalararasi DUSUK trafikli koridorlar (ör. Atlantik
gecisi) ayri hub'lari birbirine baglamiyor, DBSCAN dogru sekilde AYRI
kumeler buluyor. flight_count her kumenin ICIN toplam agirlik/buyukluk
istatistigi olarak SONRADAN (post-hoc) kullaniliyor.

KRITIK DUZELTME (2026-07-09, kullanici bulgusu): ilk versiyon TEK, SABIT
GLOBAL esik (dunya capinda p95) kullaniyordu -- bu, Kuzey Amerika/Bati
Avrupa disindaki TUM hub'lari (Korfez, Guney Asya, Guney Amerika, Afrika)
sistematik olarak eledi. Kanit: bu bolgelerin KENDI p95'leri (1309/558/
229/137) global esigin (1723) ALTINDA -- yani Korfez'in "en yogun" %5'i
bile ABD/Avrupa'nin ortalama seviyesine erisemiyor. Bu, gercek trafik
farkindan CIDDI OLARAK BUYUK ihtimalle kaynak ADS-B verisinin (adsb.lol)
tarihsel olarak ABD/Avrupa'da cok daha yogun alici-istasyon agina sahip
olmasindan kaynaklaniyor -- Dubai/Mumbai/Sao Paulo gibi gercek buyuk
hub'lar "gorunmez" kaliyordu.

Duzeltme: TEK global esik yerine, dunya kaba bir grid'e (GRID_DEG x
GRID_DEG derece) bolunup her hucrenin hex'leri KENDI YEREL dagilimina
gore degerlendiriliyor (bkz. compute_regional_mask). Elle kita sinirlari
CIZMEK yerine (ozne/kesin olurdu) duzenli bir grid kullaniliyor --
boylece hala "onceden tanimlanmis idari sinir yok" ilkesine sadik
kalinirken, ABD/Avrupa'nin mutlak-sayi ustunlugu diger bolgeleri
gizlemiyor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

DENSITY_DIR = Path(__file__).parent / "viz" / "data"
EARTH_RADIUS_KM = 6371.0088


def load_hex_density(resolution: int = 5) -> pd.DataFrame:
    df = pd.read_parquet(DENSITY_DIR / f"density_flights_res{resolution}.parquet")
    lat, lon = zip(*(h3.cell_to_latlng(h) for h in df["h3_cell"]))
    df = df.assign(lat=lat, lon=lon)
    logger.info("load_hex_density: res%d, %d hex", resolution, len(df))
    return df


def filter_bbox(df: pd.DataFrame, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> pd.DataFrame:
    return df[df["lat"].between(min_lat, max_lat) & df["lon"].between(min_lon, max_lon)]


def compute_knn_local_mask(
    df: pd.DataFrame, *, k_neighbors: int = 1000, percentile: float = 0.95, min_absolute: int = 50,
    batch_size: int = 5000,
) -> pd.Series:
    """Her hex'i, KENDI en yakin `k_neighbors` komsusunun (haversine mesafe,
    SERT grid hucresi YOK) flight_count dagilimina gore degerlendirir.

    2026-07-10 GECMISI (onceki compute_regional_mask'in -- 25 derecelik grid
    -- YERINE gecti, kullanici bulgusu): grid yaklasimi calisiyordu ama iki
    somut sorunu vardi:
      1) SERT SINIR: Paris (48.99N) ile Frankfurt/Londra (50.49N/51.88N)
         SIRF 25 derecelik grid cizgisinin (50N) iki yaninda kaldiklari icin
         FARKLI bolgesel esiklerle (p95=2291 vs p95=1520, ~%50 fark)
         degerlendiriliyordu -- coğrafi olarak anlamsiz, keyfi bir kesim.
      2) Londra'nin boylami (-0.51) Greenwich meridyeninin (0 derece) HEMEN
         batisinda oldugu icin Frankfurt/Amsterdam'dan (0-25E hucresi) AYRI
         bir hucreye (-25-0E) dusuyordu.
    Once GRID'SIZ bir alternatif (VDBSCAN-tarzi, sadece uzaysal k-distance'a
    dayali, flight_count filtresi OLMADAN) denendi -- bu, DBSCAN'in bilinen
    "chaining" (zincirleme baglanabilirlik) zaafini ortaya cikardi: Avrupa+
    Ortadogu'da flight_count filtresi kaldirilinca Istanbul/Londra/Frankfurt/
    Paris/Bukres HEPSI TEK bir 2000-3000km yaricapli deve kumeye birlesti --
    dusuk/orta yogunluklu transit koridor hex'leri (ucaklarin GECIS guzergahi
    uzerinde biraktigi izler) ayri hub'lari birbirine "kopru" gibi bagliyordu.
    Bu, flight_count esiginin sadece "adil bolgesel karsilastirma" degil,
    AYNI ZAMANDA bu zincirlemeyi KESEN bir mekanizma oldugunu kanitladi --
    kaldirilamaz, sadece "yerel" tanimi degismeli.

    Cozum: grid hucresi yerine, her hex'in KNN komsulugu (k_neighbors) --
    sert sinir yok (surekli/yumusak), coverage-onyargisi duzeltmesi KORUNUYOR
    (uzak/seyrek bolgelerde komsular da dogal olarak uzak/seyrek oldugu icin
    kendiliginden "genis pencere" olusuyor). k_neighbors AMPIRIK olarak
    secildi: k=300 test edildiginde bazi GERCEK hub'lar (Frankfurt, Bukres,
    Dubai, Sao Paulo) TAMAMEN kayboldu -- cok dar bir komsulukta, bir hub'in
    kendi komsularinin COGU ZATEN o hub'in bir parcasi oldugu icin "komsulara
    gore p95" kendi icinde asiri siki bir esige donusuyor (hub kendi kendini
    eliyor). k=1000'de TUM bilinen hub'lar (Istanbul, Londra, Frankfurt,
    Paris, Bukres, Dubai, Sao Paulo, Johannesburg) dogru cikti VE en buyuk
    kume grid yontemindeki 432 hex/335km'den 50 hex/130km'ye KUCULDU (mega-
    blob riski BUYUMEDI, kucaldi) -- 243.835 hex icin ~170sn suruyor (tek
    seferlik batch is icin kabul edilebilir).
    """
    coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
    fc = df["flight_count"].to_numpy()
    n = len(df)

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, n), metric="haversine", algorithm="ball_tree").fit(coords_rad)
    mask = np.zeros(n, dtype=bool)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, idx = nn.kneighbors(coords_rad[start:end])
        neighbor_fc = fc[idx]
        threshold = np.maximum(np.quantile(neighbor_fc, percentile, axis=1), min_absolute)
        mask[start:end] = fc[start:end] >= threshold

    mask = pd.Series(mask, index=df.index)
    logger.info(
        "compute_knn_local_mask: k=%d, p%.0f, %d/%d hex secildi",
        k_neighbors, percentile * 100, int(mask.sum()), n,
    )
    return mask


def run_dbscan(dense: pd.DataFrame, *, eps_km: float, min_samples: int) -> pd.DataFrame:
    """`dense`: ONCEDEN filtrelenmis (esik gecmis -- global sabit VEYA
    compute_regional_mask ile bolgesel) hex'ler. Donus: "cluster" kolonu
    eklenmis hali (-1 = noise/izole)."""
    dense = dense.copy()
    if dense.empty:
        dense["cluster"] = pd.Series(dtype=int)
        return dense

    coords_rad = np.radians(dense[["lat", "lon"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine").fit(coords_rad)
    dense["cluster"] = db.labels_

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = int((db.labels_ == -1).sum())
    logger.info(
        "run_dbscan: %d hex, eps=%gkm min_samples=%d -> %d kume, %d hex noise (%.1f%%)",
        len(dense), eps_km, min_samples, n_clusters, n_noise, 100 * n_noise / len(dense),
    )
    return dense


def run_dbscan_two_pass(
    dense: pd.DataFrame, *, eps_km: float, min_samples_strict: int, min_samples_relaxed: int,
) -> pd.DataFrame:
    """Tek gecisli DBSCAN'in COZEMEDIGI bir gerginligi cozer (2026-07-09,
    kullanici bulgusu: "Istanbul gorunmuyor, Bukres gorunuyor, mantik
    yanlis"): Istanbul gibi COK BUYUK/YAYILMIS ama gercek hub'lar (iki kita,
    Bogaz'in iki yakasi), min_samples SIKI (30) iken HICBIR hex kendi 50km
    cevresinde yeterli komsu BULAMADIGI icin tamamen "gurultu" (-1) sayilip
    kayboluyordu -- oysa o bolgede toplamda esigi gecen 38 hex vardi, sadece
    birbirlerinden DBSCAN'in "cekirdek nokta" tanimini karsilayacak kadar
    yakin degillerdi. min_samples'i GLOBAL olarak gevsetmek (ör. 30->20)
    Istanbul'u yakaladi AMA Bati Avrupa'nin (Londra/Paris/Frankfurt/Brüksel)
    onceden AYRI ayrilan kumelerini de TEK bir 600km+ yaricapli mega-blob'a
    geri birlestirdi -- yani tek bir global parametre HEM kompakt (Bukres)
    HEM yayilmis (Istanbul) hub'lari ayni anda dogru cozemiyor.

    Cozum -- IKI GECIS:
      1) SIKI parametrelerle (min_samples_strict) normal DBSCAN -- kompakt
         hub'lar (cogu sehir) burada ZATEN dogru ayriliyor, bu kumeler
         SONRAKI adimda HIC DOKUNULMUYOR (Bati Avrupa bozulmuyor).
      2) Pass 1'de "gurultu" (-1) kalan hex'ler uzerinde, SADECE o alt-kume
         icinde, GEVSEK parametrelerle (min_samples_relaxed) IKINCI bir
         DBSCAN -- Istanbul gibi yayilmis ama GERCEKTEN yogun hub'lar burada
         yakalanir, cunku artik rakip/komsu YOGUN bolgelerle (ör. Bukres,
         zaten kendi kumesini Pass 1'de almisti) REKABET ETMIYOR, sadece
         kendi aralarinda degerlendiriliyorlar.
    Pass 2'nin kume ID'leri Pass 1'inkilerle CAKISMAMASI icin offsetleniyor.
    Ampirik dogrulama (2026-07-09): Istanbul kendi kumesini aldi (34 hex,
    86km yaricap), Bukres FARKLI bir kumede kaldi, Bati Avrupa'nin en buyuk
    kumesi (432 hex, 335km) Pass 1'den DEGISMEDEN geldi, ve Pass 2 ayrica
    120 baska gercekci hub yakaladi (Viyana, Zurih, Madrid, Tokyo, KL vb.)
    -- hepsi Istanbul'la AYNI "sprawling ama gercek" kategorisindeydi.
    """
    dense = dense.copy()
    if dense.empty:
        dense["cluster"] = pd.Series(dtype=int)
        return dense

    eps_rad = eps_km / EARTH_RADIUS_KM
    coords_rad = np.radians(dense[["lat", "lon"]].to_numpy())

    db1 = DBSCAN(eps=eps_rad, min_samples=min_samples_strict, metric="haversine").fit(coords_rad)
    dense["cluster"] = db1.labels_
    n_pass1 = len(set(db1.labels_) - {-1})

    noise_mask = (dense["cluster"] == -1).to_numpy()
    n_pass2 = 0
    if noise_mask.sum() >= min_samples_relaxed:
        db2 = DBSCAN(eps=eps_rad, min_samples=min_samples_relaxed, metric="haversine").fit(coords_rad[noise_mask])
        offset = int(dense["cluster"].max()) + 1
        new_labels = np.where(db2.labels_ == -1, -1, db2.labels_ + offset)
        dense.loc[noise_mask, "cluster"] = new_labels
        n_pass2 = len(set(new_labels) - {-1})

    n_clusters = dense["cluster"].nunique() - (1 if (dense["cluster"] == -1).any() else 0)
    n_noise = int((dense["cluster"] == -1).sum())
    logger.info(
        "run_dbscan_two_pass: %d hex, eps=%gkm -- Pass1(min_samples=%d): %d kume; "
        "Pass2(min_samples=%d, sadece %d Pass1-gurultusu hex uzerinde): %d YENI kume; "
        "toplam %d kume, %d hex hala gurultu (%.1f%%)",
        len(dense), eps_km, min_samples_strict, n_pass1, min_samples_relaxed,
        int(noise_mask.sum()), n_pass2, n_clusters, n_noise, 100 * n_noise / len(dense),
    )
    return dense


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    """Her kume icin: merkez (flight_count agirlikli), toplam ucus, hex
    sayisi, yaklasik yaricap (merkeze en uzak hex mesafesi)."""
    records = []
    for cluster_id, group in clustered[clustered["cluster"] != -1].groupby("cluster"):
        w = group["flight_count"].to_numpy()
        lat_c = np.average(group["lat"], weights=w)
        lon_c = np.average(group["lon"], weights=w)
        # Merkeze en uzak hex -- yaklasik kume yaricapi (haversine, km).
        lat_r, lon_r = np.radians(group["lat"]), np.radians(group["lon"])
        lat_cr, lon_cr = np.radians(lat_c), np.radians(lon_c)
        dlat, dlon = lat_r - lat_cr, lon_r - lon_cr
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_cr) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
        dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
        records.append({
            "cluster": cluster_id, "n_hexes": len(group),
            "total_flight_count": int(group["flight_count"].sum()),
            "center_lat": lat_c, "center_lon": lon_c,
            "radius_km": float(dist_km.max()),
        })
    columns = ["cluster", "n_hexes", "total_flight_count", "center_lat", "center_lon", "radius_km"]
    out = pd.DataFrame.from_records(records, columns=columns).sort_values("total_flight_count", ascending=False)
    logger.info("summarize_clusters: %d kume", len(out))
    return out
