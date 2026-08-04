"""Asil egitim: TUM v2 veri seti + session-bazli train/val/test ayrimi
(session_split.parquet, build_split.py ciktisi) ile item tower (GCN) +
query tower (Transformer) birlikte egitilir.

train.py'daki kucuk-ornek mekanik dogrulamasindan farki: (1) TUM
training_sessions/*.parquet okunur, (2) sadece session_split.parquet'te
KEPT (1-kez ucan rota degil) ve bir split'e (train/val/test) atanmis
session'lar kullanilir, (3) her epoch sonunda val loss/accuracy raporlanir,
(4) egitim sonunda model agirliklari kaydedilir.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from individual.metehan_varis_tahmini.airport_gcn import (
    AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
)
from individual.metehan_varis_tahmini.airport_gat import AirportGAT, build_adjacency_mask
from individual.metehan_varis_tahmini.query_tower import QueryTower, POINT_DIM

SESSIONS_DIR = Path(__file__).parent / "training_sessions"
SPLIT_PATH = Path(__file__).parent / "session_split.parquet"

# Feature ablation (mentor sorusu, 2026-07-27): airport_features.parquet'teki
# 22 kolonun 8'i traffic/graf-turevli (rota sayimlarindan HESAPLANIR -- veri
# degisirse degisir), 12'si gercekten STATIK (havalimaninin kendi fiziksel/
# cografi ozelligi, hic degismez). --feature-subset static12 bu 8'ini
# CIKARIP sadece 12 statik ozellikle egitir -- turetilmis ozelliklerin
# gercek katkisini olcmek icin.
STATIC_FEATURE_COLS = [
    "latitude_deg", "longitude_deg", "elevation_ft", "scheduled_service", "has_iata",
    "type_large_airport", "type_medium_airport", "type_nan",
    "continent_AF", "continent_AS", "continent_EU", "continent_nan",
]
# feature_subset="expanded30": mevcut 20 ozelligin uzerine 9 yeni graf-
# topoloji ozelligi (betweenness/closeness/HITS/reciprocity/... -- bkz.
# build_expanded_airport_features.py) eklenmis AYRI bir dosya -- kolon
# FILTRELEME degil, FARKLI bir feature dosyasindan okuma.
EXPANDED_FEATURES_PATH = Path(__file__).parent / "airport_features_expanded30.parquet"


def checkpoint_paths(item_tower_kind: str, loss_kind: str, extra: str = "") -> tuple[Path, Path]:
    """GCN/GAT ve CE/Focal farkli agirlik yapilari uretir -- AYNI dosya
    adini kullanmak, bir deneyin checkpoint'ini digerine yanlislikla
    'resume' etmeye (state_dict uyusmazligi/sessiz bozulma) yol acardi.
    Varsayilan (gcn+ce) ESKI sabit isimleri korur, boylece mevcut/devam
    eden GCN egitimi ETKILENMEZ. extra: loss_kind degismeden (ör. oversample/
    augment, ce uzerine) ayri bir checkpoint gerektiginde ek ayrimci."""
    suffix = ""
    if item_tower_kind != "gcn":
        suffix += f"_{item_tower_kind}"
    if loss_kind != "ce":
        suffix += f"_{loss_kind}"
    if extra:
        suffix += f"_{extra}"
    base = Path(__file__).parent
    return base / f"model_checkpoint{suffix}.pt", base / f"model_checkpoint{suffix}_best.pt"


# Varsayilan (gcn+ce) konfigurasyonun sabit yolu -- evaluate_by_route_frequency.py
# gibi diger scriptler bunu dogrudan import ediyor, geriye-uyumluluk icin korunuyor.
MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths("gcn", "ce")

# Yerel (Windows/CPU-only) calisirken davranis DEGISMEZ (cuda yoksa cpu secilir).
# Colab/GPU ortaminda gercek hizlanma icin sart -- daha once hicbir device
# yonetimi yoktu, her sey ORTUK olarak CPU'da calisiyordu.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 300


def _sample_indices(n, max_len, jitter=False):
    """jitter=True (sadece augment=True train ornekleri icin): downsample
    indekslerine kucuk rastgele ofset ekleyip her erisimde izin biraz farkli
    noktalarindan orneklenmesini saglar ("farkli dt_norm ornekleme"
    cesitlendirmesi). TUM alanlar (lat/lon/alt/hiz/yon/vrate/dt_norm) AYNI
    indeks kumesiyle kesilir (fiziksel tutarlilik icin) -- ayri ayri
    downsample edilirse noktalar birbirine karisirdi."""
    if n <= max_len:
        return list(range(n))
    idx = np.linspace(0, n - 1, max_len)
    if jitter:
        idx = np.clip(idx + np.random.uniform(-0.5, 0.5, size=max_len), 0, n - 1)
    return idx.round().astype(int).tolist()


class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, airport_to_idx: dict[str, int], require_origin: bool = False,
                 augment: bool = False):
        # require_origin: SADECE mask_routes=True iken True -- v1 (maskesiz)
        # ile BIREBIR ayni veri kompozisyonu (214105/26965/26676) korunsun
        # diye, origin filtresi sadece gercekten maskeleme icin gerekliyken
        # uygulanir.
        keep = df["dest_airport"].isin(airport_to_idx)
        if require_origin:
            keep &= df["origin_airport"].isin(airport_to_idx)
        df = df[keep].reset_index(drop=True)
        self.df = df
        self.airport_to_idx = airport_to_idx
        # augment: SADECE train_ds icin True olmali -- val/test HER ZAMAN
        # gercek/sabit izlere karsi degerlendirilmeli (2026-07-28: focal loss
        # denemesinde oldugu gibi, "sonuc" checkpoint'in gercek performansini
        # yansitsin diye egitim disi hicbir dinamik degisiklik yapilmiyor).
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = _sample_indices(row["n_points"], MAX_SEQ_LEN, jitter=self.augment)
        n = len(idx)
        lats = np.array([row["lats"][j] for j in idx], dtype=np.float32)
        lons = np.array([row["lons"][j] for j in idx], dtype=np.float32)
        alts = np.array([row["alts"][j] for j in idx], dtype=np.float32)
        vels = np.array([row["velocities"][j] for j in idx], dtype=np.float32)
        heads = np.array([row["headings"][j] for j in idx], dtype=np.float32)
        vrates = np.array([row["vrates"][j] for j in idx], dtype=np.float32)
        dt_norms = np.array([row["dt_norms"][j] for j in idx], dtype=np.float32)

        if self.augment:
            # Gercekci trajectory-level jitter: ADS-B/GPS olcum gurultusune
            # benzer kucuk rastgele sapmalar -- etiket (dest_airport) AYNI
            # kalir, model ayni rotayi her epoch'ta biraz farkli bir
            # "gorunumde" gorur (oversampling ile tekrarlanan nadir-rota
            # ornekleri boylece BIREBIR ayni kopya olmaz).
            lats = lats + np.random.normal(0, 0.002, n).astype(np.float32)   # ~200m std (lat)
            lons = lons + np.random.normal(0, 0.002, n).astype(np.float32)  # ~200m std (lon)
            vels = vels * np.random.normal(1.0, 0.02, n).astype(np.float32)  # ~%2 hiz gurultusu
            heads = heads + np.random.normal(0, 3.0, n).astype(np.float32)  # ~3 derece yon gurultusu
            vrates = vrates + np.random.normal(0, 0.3, n).astype(np.float32)  # kucuk dikey hiz gurultusu

        points = np.zeros((n, POINT_DIM), dtype=np.float32)
        points[:, 0] = lats
        points[:, 1] = lons
        points[:, 2] = alts / 1000.0
        points[:, 3] = vels / 100.0
        heading_rad = np.radians(heads)
        points[:, 4] = np.sin(heading_rad)
        points[:, 5] = np.cos(heading_rad)
        points[:, 6] = vrates / 10.0
        points[:, 7] = dt_norms
        label = self.airport_to_idx[row["dest_airport"]]
        # -1 (require_origin=False iken bazi origin'ler airport_to_idx'te
        # olmayabilir): route_mask=None oldugunda origin'e HIC BAKILMAZ
        # (apply_route_mask no-op), guvenli bir yer-tutucu yeterli.
        origin = self.airport_to_idx.get(row["origin_airport"], -1)
        return torch.from_numpy(points), label, origin


def collate(batch):
    points_list, labels, origins = zip(*batch)
    lengths = torch.tensor([p.shape[0] for p in points_list], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(points_list), max_len, POINT_DIM, dtype=torch.float32)
    for i, p in enumerate(points_list):
        padded[i, :p.shape[0]] = p
    return padded, lengths, torch.tensor(labels, dtype=torch.long), torch.tensor(origins, dtype=torch.long)


def load_all_sessions() -> pd.DataFrame:
    files = sorted(glob.glob(str(SESSIONS_DIR / "*.parquet")))
    print(f"{len(files)} parquet dosyasi okunuyor (tam veri seti)...", flush=True)
    t0 = time.time()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"-> {len(df)} satir, {time.time()-t0:.1f}sn", flush=True)
    return df


def build_route_mask_from_sessions(df: pd.DataFrame, airport_to_idx: dict[str, int]) -> torch.Tensor:
    """route_scan_full_archive.json'daki route_counts, bolge-filtresi/adsbdb
    duzeltmesinden ONCEKI eski taramaya ait -- bu yuzden v2 training_sessions'
    taki GERCEK bazi (origin,dest) ciftlerini (ör. adsbdb'nin kurtardigi
    LTFM->WSSS gibi) icermiyor. build_route_mask(route_counts, ...) kullanmak,
    boyle ornekler icin GERCEK etiketi maskeleyip asla secilemez hale getirir
    (test_route_mask.py ile YAKALANDI). Bunun yerine maske DOGRUDAN v2
    session'larinin kendi (origin_airport,dest_airport) ciftlerinden kurulur
    -- boylece egitim/dogrulama sirasinda kullanilacak HER gercek etiket
    maskede garanti bulunur."""
    n = len(airport_to_idx)
    mask = torch.zeros((n, n), dtype=torch.bool)
    pairs = df[["origin_airport", "dest_airport"]].drop_duplicates()
    for o, d in pairs.itertuples(index=False):
        if o in airport_to_idx and d in airport_to_idx:
            mask[airport_to_idx[o], airport_to_idx[d]] = True
    return mask


class FocalLoss(nn.Module):
    """Lin ve ark. 2017 (RetinaNet) -- literatur karsilastirmasinda (bkz.
    proje dokumantasyonu SS11) MobGT'nin nadir-rota sorunu icin kullandigi
    "Tail Loss"nin genel karsiligi. Duz CrossEntropyLoss'un aksine, modelin
    ZATEN dogru bildigi ("kolay") ornekleri (1-p_t)^gamma ile agirlik
    olarak KUCULTUP, hala yanlis bildigi ("zor" -- bizde COGUNLUKLA nadir
    rota) orneklere goreceli daha fazla agirlik verir. gamma=0 iken
    matematiksel olarak duz CrossEntropyLoss'a esittir (weight yoksa)."""

    def __init__(self, gamma: float = 2.0, class_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weight", class_weight, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.class_weight is not None:
            loss = loss * self.class_weight[targets]
        return loss.mean()


def class_balanced_weights(labels: pd.Series, n_classes: int, beta: float = 0.999) -> torch.Tensor:
    """Cui ve ark. 2019 "Class-Balanced Loss" -- her sinifin agirligini,
    ham 1/frekans yerine "etkin ornek sayisi" (1-beta^n)/(1-beta) uzerinden
    hesaplar (n cok buyudukce marjinal fayda azaldigi icin ham ters-frekans
    kadar asiri agirlik vermez). beta=0.999 makalenin onerdigi varsayilan.
    Agirliklar, ortalama 1 olacak sekilde normalize edilir (loss olcegi
    degismesin diye)."""
    counts = labels.value_counts()
    full_counts = np.array([counts.get(i, 0) for i in range(n_classes)], dtype=np.float64)
    effective_num = 1.0 - np.power(beta, full_counts)
    effective_num[full_counts == 0] = 1.0  # hic ornegi olmayan sinif icin bolme hatasi olmasin
    weights = (1.0 - beta) / effective_num
    weights = weights / weights[full_counts > 0].mean()
    return torch.tensor(weights, dtype=torch.float32)


def apply_route_mask(logits: torch.Tensor, origins: torch.Tensor, route_mask: torch.Tensor | None) -> torch.Tensor:
    """Origin-kosullu softmax: o origin'den GERCEKTE hic gozlemlenmemis
    destinasyonlarin logit'ini -inf'e cekerek tam-852 adaydan sadece
    gercekci/ulasilabilir olanlara indirger. route_mask=None ise (mask_routes
    kapali) HICBIR SEY yapmadan logits'i oldugu gibi dondurur -- v1 (tam-852
    softmax) davranisiyla birebir ayni."""
    if route_mask is None:
        return logits
    row_mask = route_mask[origins]  # (batch, n_airports)
    return logits.masked_fill(~row_mask, -1e9)


@torch.no_grad()
def evaluate(item_tower, query_tower, loader, x, adj, route_mask, loss_fn):
    item_tower.eval()
    query_tower.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    airport_emb = item_tower(x, adj)
    for points, lengths, labels, origins in loader:
        points, lengths = points.to(DEVICE), lengths.to(DEVICE)
        labels, origins = labels.to(DEVICE), origins.to(DEVICE)
        query_emb = query_tower(points, lengths)
        logits = query_emb @ airport_emb.T
        logits = apply_route_mask(logits, origins, route_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_n += len(labels)
    item_tower.train()
    query_tower.train()
    return total_loss / total_n, total_correct / total_n


def main(n_epochs: int = 5, batch_size: int = 32, limit_files: int | None = None, mask_routes: bool = False,
         item_tower_kind: str = "gcn", loss_kind: str = "ce", focal_gamma: float = 2.0, class_balanced: bool = False,
         patience: int | None = None, oversample: bool = False, augment: bool = False,
         feature_subset: str = "full", run_tag: str = "",
         hidden_dim: int = 32, out_dim: int = 16, weight_decay: float = 0.0, use_lr_scheduler: bool = False,
         min_route_count: int | None = None):
    # oversample/augment: Focal Loss + Class-Balanced'in YERINE denenen ikinci
    # yol (2026-07-28: o kombinasyon TUM dilimlerde -- nadir rotalar dahil --
    # duz GAT'tan daha kotu cikti, bkz. gunluk). loss_kind degismez (CE kalir,
    # class_balanced ile AYNI ANDA kullanilmaz), ayri checkpoint icin "aug" eki.
    # feature_subset="static12": ozellik SAYISI (girdi boyutu) degistigi icin
    # ONCEKI checkpoint'lerden warm-start EDILEMEZ (shape uyusmazligi) --
    # bu ablasyon her zaman epoch 0'dan, ayri bir checkpoint'e egitilir.
    # run_tag: ayni hiperparametre kombinasyonuyla (ör. oversample+augment)
    # FARKLI bir veri seti (ör. OpenSky-zenginlestirilmis) uzerinde ayri bir
    # checkpoint'e ihtiyac oldugunda serbest metin ayirici (ör. "opensky").
    # min_route_count: nadir-rota iyilestirme denemelerinden (Focal/CB,
    # Oversample+Augment, OpenSky, kapasite/hiperparametre) SONRA, referans
    # karsilastirma icin -- train setinde bu sayidan AZ tekrar eden (origin,
    # dest) rotalari TUM split'lerden (train/val/test) tamamen cikarir.
    # Nadir rotalarin zaten guvenilir bir tahmine katki saglamasi
    # beklenmedigi icin (bkz. gunluk 2026-07-30/31), bu deneyin nihai
    # metrigi kasitli olarak daha dar/kolay bir rota kumesi uzerinden
    # olculur -- onceki denemelerle DOGRUDAN karsilastirilamaz, kendi basina
    # ayri bir referans noktasidir.
    extra_parts = []
    if oversample or augment:
        extra_parts.append("aug")
    if run_tag:
        extra_parts.append(run_tag)
    if feature_subset != "full":
        extra_parts.append(feature_subset)
    # kapasite (hidden_dim/out_dim) varsayilandan farkliysa ayri checkpoint --
    # agirlik boyutlari degistigi icin ONCEKI checkpoint'lerden warm-start
    # EDILEMEZ (shape uyusmazligi), bu yuzden ayirici etikete ihtiyac var.
    if hidden_dim != 32 or out_dim != 16:
        extra_parts.append(f"h{hidden_dim}o{out_dim}")
    if min_route_count is not None:
        extra_parts.append(f"minroute{min_route_count}")
    extra = "_".join(extra_parts)
    MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths(item_tower_kind, loss_kind, extra=extra)
    print(f"item_tower={item_tower_kind}, loss={loss_kind}"
          + (f" (gamma={focal_gamma})" if loss_kind == "focal" else "")
          + (", class_balanced=True" if class_balanced else "")
          + (", oversample=True" if oversample else "")
          + (", augment=True" if augment else "")
          + f", feature_subset={feature_subset}"
          + f", hidden_dim={hidden_dim}, out_dim={out_dim}, weight_decay={weight_decay}, lr_scheduler={use_lr_scheduler}"
          + (f", min_route_count={min_route_count}" if min_route_count is not None else "")
          + f" -> checkpoint: {MODEL_OUT.name}", flush=True)

    print(f"device: {DEVICE}", flush=True)
    if feature_subset == "expanded30":
        feat_df = pd.read_parquet(EXPANDED_FEATURES_PATH)
        print(f"feature_subset=expanded30: {EXPANDED_FEATURES_PATH.name} okunuyor "
              f"(9 yeni graf-topoloji ozelligi eklendi: betweenness/closeness_centrality, "
              f"hub/authority_score, reciprocity, origin_entropy, max_out_share, "
              f"avg_neighbor_out_degree, n_countries_connected)", flush=True)
    else:
        feat_df = pd.read_parquet(FEATURES_PATH)
        if feature_subset == "static12":
            feat_df = feat_df[["ident", "iso_country"] + STATIC_FEATURE_COLS].copy()
            print(f"feature_subset=static12: sadece {len(STATIC_FEATURE_COLS)} statik ozellik kullaniliyor "
                  f"(8 traffic/graf-turevli ozellik CIKARILDI: out_degree, in_degree, total_volume, "
                  f"military_ratio, dest_entropy, avg_route_distance_km, pagerank, clustering_coef)", flush=True)
    x, idents = load_feature_tensor(feat_df)
    x = x.to(DEVICE)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    # GCN: rota-sayisina gore ONCEDEN AGIRLIKLI komsuluk matrisi.
    # GAT: agirligi model KENDI OGRENIR, burada sadece "kenar var mi" (Bool) yeterli.
    adj = build_normalized_adjacency(route_counts, idents) if item_tower_kind == "gcn" \
        else build_adjacency_mask(route_counts, idents)
    adj = adj.to(DEVICE)

    split_df = pd.read_parquet(SPLIT_PATH)
    split_map = split_df.set_index("session_id")["split"]

    all_df = load_all_sessions()
    all_df["split"] = all_df["session_id"].map(split_map)
    all_df = all_df.dropna(subset=["split"])  # 1-kez ucan rotalarin session'lari elenir
    print(f"Split sonrasi kullanilabilir satir: {len(all_df)}", flush=True)
    print(all_df["split"].value_counts(), flush=True)

    if min_route_count is not None:
        # Esik, SADECE train setindeki rota-tekrar sayisina gore hesaplanir
        # (evaluate_by_route_frequency.py'deki dilimlerle ayni granularite:
        # 2-4/5-9 dilimleri min_route_count=10 ile tamamen elenir). Ayni rota
        # kumesi TUM split'lerden (train/val/test) cikarilir -- aksi halde
        # model hic gormedigi nadir rotalarda test edilip GENEL dogruluk
        # yapay olarak dusurulur, bu da "nadir rotalari disarida birakinca
        # ne kadar iyilesiyor" sorusuna adil cevap vermez.
        train_route_counts = all_df[all_df["split"] == "train"].groupby(
            ["origin_airport", "dest_airport"]).size()
        keep_routes = set(train_route_counts[train_route_counts >= min_route_count].index)
        n_before = len(all_df)
        route_pairs = list(zip(all_df["origin_airport"], all_df["dest_airport"]))
        all_df = all_df[[p in keep_routes for p in route_pairs]].reset_index(drop=True)
        print(f"min_route_count={min_route_count}: {len(train_route_counts)} rotadan "
              f"{len(keep_routes)} tanesi tutuldu (train'de >= {min_route_count} kez uculmus), "
              f"{n_before} satirdan {len(all_df)} tanesi kaldi ({n_before - len(all_df)} satir elendi).",
              flush=True)
        print(all_df["split"].value_counts(), flush=True)

    # route_scan_full_archive.json (a_norm icin yukarida kullanilan) bolge-
    # filtresi/adsbdb duzeltmesinden ONCEKI eski taramaya ait -- maske bunun
    # yerine DOGRUDAN v2 session'larindan kuruluyor (bkz. fonksiyon docstring'i).
    # mask_routes=False iken maskeleme tamamen devre disi (tam-852 softmax,
    # v1 ile ayni davranis) -- iki deney de AYNI script'ten, koda dokunmadan
    # karsilastirilabilsin diye.
    route_mask = build_route_mask_from_sessions(all_df, airport_to_idx) if mask_routes else None
    if route_mask is not None:
        route_mask = route_mask.to(DEVICE)

    # augment: SADECE train_ds -- val/test her zaman gercek/sabit izlerle
    # degerlendirilir (bkz. TrajectoryDataset.augment dokumantasyonu).
    train_ds = TrajectoryDataset(all_df[all_df["split"] == "train"], airport_to_idx, require_origin=mask_routes,
                                  augment=augment)
    val_ds = TrajectoryDataset(all_df[all_df["split"] == "val"], airport_to_idx, require_origin=mask_routes)
    test_ds = TrajectoryDataset(all_df[all_df["split"] == "test"], airport_to_idx, require_origin=mask_routes)
    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

    if oversample:
        # Nadir ROTALARIN (origin,dest cifti -- evaluate_by_route_frequency.py
        # ile AYNI granularite) ornekleme agirligini train setinin KENDI
        # frekansindan hesapla. sqrt(1/frekans): Class-Balanced Loss'taki ham
        # ters-frekansa yakin agirliklarin (min=0.031, max=7.698) egitimi
        # dengesizlestirdigi gozlemine karsi (2026-07-28) daha ILIMLI bir
        # secim -- burada agirlik LOSS'u degil SADECE ornekleme sikligini
        # degistirdigi icin zaten daha az riskli, yine de asiri uc bir
        # tekrar sayisindan kacinmak icin sqrt kullanildi.
        route_counts_train = train_ds.df.groupby(["origin_airport", "dest_airport"]).size()
        weights = train_ds.df.apply(
            lambda r: 1.0 / route_counts_train[(r["origin_airport"], r["dest_airport"])] ** 0.5, axis=1
        ).to_numpy().copy()
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate)
        print(f"oversampling aktif (sqrt-ters-rota-frekansi agirlikli sampler, "
              f"agirlik min={weights.min():.4f} max={weights.max():.4f})", flush=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)

    torch.manual_seed(0)
    if item_tower_kind == "gcn":
        item_tower = AirportGCN(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, n_gcn_layers=2)
    else:
        item_tower = AirportGAT(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, n_gat_layers=2, n_heads=4)
    query_tower = QueryTower(d_model=hidden_dim, nhead=4, num_layers=2, out_dim=out_dim)
    item_tower = item_tower.to(DEVICE)
    query_tower = query_tower.to(DEVICE)
    optimizer = torch.optim.Adam(list(item_tower.parameters()) + list(query_tower.parameters()),
                                  lr=1e-3, weight_decay=weight_decay)
    # ReduceLROnPlateau: val_acc patience=2 boyunca iyilesmezse lr'i yariya
    # indirir -- sabit lr=1e-3'un platoya girince ince ayar yapamama sorununa
    # karsi (2026-07-30 gunlugunde tartisildi). mode='max' cunku val_acc
    # (loss degil) izleniyor.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2) if use_lr_scheduler else None

    # class_balanced: nadir rotalarin (destinasyonlarin) az ornekli olmasi
    # sorununa karsi, TRAIN setindeki gercek destinasyon dagilimindan
    # (Cui ve ark. 2019) agirlik hesaplanir -- hem duz CE hem Focal ile
    # birlikte kullanilabilir.
    class_weight = None
    if class_balanced:
        train_labels = train_ds.df["dest_airport"].map(airport_to_idx)
        class_weight = class_balanced_weights(train_labels, n_classes=len(idents))
        print(f"class_balanced agirliklar hesaplandi (ort=1.0, min={class_weight.min():.3f}, max={class_weight.max():.3f})", flush=True)

    if loss_kind == "focal":
        loss_fn = FocalLoss(gamma=focal_gamma, class_weight=class_weight)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    loss_fn = loss_fn.to(DEVICE)  # class_weight/buffer'i da tasir

    # yarida kesilirse (PC/oturum kapanmasi) EN AZINDAN son tamamlanan epoch'u
    # kaybetmemek icin -- var olan checkpoint'ten devam
    start_epoch = 0
    if MODEL_OUT.exists():
        ckpt = torch.load(MODEL_OUT, weights_only=False, map_location=DEVICE)
        item_tower.load_state_dict(ckpt["item_tower"])
        query_tower.load_state_dict(ckpt["query_tower"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        # eski format (checkpoint ozelligi eklenmeden once kaydedilmis):
        # "epoch" yok -- o kayit epoch 0'in tamamlanmis hali oldugu icin
        # epoch 1'den devam ediyoruz.
        start_epoch = ckpt["epoch"] + 1 if "epoch" in ckpt else 1
        print(f"Checkpoint bulundu, epoch {start_epoch}'den devam ediliyor.", flush=True)

    # En-iyi (val_acc'e gore) checkpoint AYRI bir dosyada takip edilir --
    # val_acc epoch'tan epoch'a dalgalanabiliyor (2026-07-24: epoch 5, epoch
    # 4'ten daha kotu cikti, epoch 6 toparladi) -- "son epoch" ile "en iyi
    # epoch" AYNI SEY DEGIL. Resume sirasinda onceki en-iyiyi kaybetmemek
    # icin once BEST_MODEL_OUT'a, o da yoksa normal MODEL_OUT'un kendi
    # val_acc'ine bakilir.
    if BEST_MODEL_OUT.exists():
        best_ckpt = torch.load(BEST_MODEL_OUT, weights_only=False, map_location=DEVICE)
        best_val_acc, best_epoch = best_ckpt.get("val_acc", -1.0), best_ckpt.get("epoch", -1)
    elif MODEL_OUT.exists() and "val_acc" in ckpt:
        # BEST_MODEL_OUT bu ozellik eklenmeden onceki bir checkpoint'ten
        # devam ediyoruz demektir -- MODEL_OUT su an bildigimiz TEK/en iyi
        # aday. HEMEN BEST_MODEL_OUT'a da yazilir, aksi halde sonraki
        # epoch'larin hicbiri onu gecemezse bu agirliklar (MODEL_OUT her
        # epoch UZERINE yazildigi icin) sessizce kaybolurdu.
        best_val_acc, best_epoch = ckpt["val_acc"], ckpt.get("epoch", -1)
        torch.save(ckpt, BEST_MODEL_OUT)
        print(f"BEST_MODEL_OUT ilk kez olusturuldu (epoch {best_epoch}, val_acc={best_val_acc:.4f}).", flush=True)
    else:
        best_val_acc, best_epoch = -1.0, -1
    print(f"Su ana kadarki en iyi: epoch {best_epoch}, val_acc={best_val_acc:.4f}", flush=True)

    # Erken durdurma (early stopping): kac epoch'tur yeni rekor gelmedigini
    # takip eder. Resume sirasinda da DOGRU baslatilir (ör. best_epoch=19,
    # start_epoch=22 ise, 21'e kadar zaten 2 epoch'tur iyilesme yok demektir)
    # -- sadece BU calistirmadaki sayaci sifirlamak, gecmisteki dalgalanmayi
    # gormezden gelip patience'i yanlis tetiklerdi/tetiklemezdi.
    epochs_without_improvement = max(0, (start_epoch - 1) - best_epoch) if start_epoch > 0 else 0
    if patience is not None and epochs_without_improvement > 0:
        print(f"(Resume: son {epochs_without_improvement} epoch'tur yeni rekor yok -- patience={patience})", flush=True)

    item_tower.train()
    query_tower.train()

    print(f"\n--- Egitim (train batch sayisi: {len(train_loader)}) ---", flush=True)
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        for i, (points, lengths, labels, origins) in enumerate(train_loader):
            points, lengths = points.to(DEVICE), lengths.to(DEVICE)
            labels, origins = labels.to(DEVICE), origins.to(DEVICE)
            optimizer.zero_grad()
            airport_emb = item_tower(x, adj)
            query_emb = query_tower(points, lengths)
            logits = query_emb @ airport_emb.T
            logits = apply_route_mask(logits, origins, route_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_n += len(labels)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta_min = (len(train_loader) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
                pct = 100 * (i + 1) / len(train_loader)
                print(f"  epoch={epoch} batch={i+1}/{len(train_loader)} (%{pct:.1f}) "
                      f"running_loss={epoch_loss/epoch_n:.4f} running_acc={epoch_correct/epoch_n:.4f} "
                      f"{rate:.2f} batch/sn, ETA bu epoch {eta_min:.1f} dk", flush=True)

        val_loss, val_acc = evaluate(item_tower, query_tower, val_loader, x, adj, route_mask, loss_fn)
        print(f"epoch={epoch} train_loss={epoch_loss/epoch_n:.4f} train_acc={epoch_correct/epoch_n:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}sn)", flush=True)

        if scheduler is not None:
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_acc)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after != lr_before:
                print(f"  (lr scheduler: {lr_before:.2e} -> {lr_after:.2e})", flush=True)

        # HER epoch sonunda kaydet -- yarida kesilirse (PC/oturum kapanmasi)
        # en azindan bu ana kadarki egitim kaybolmaz, bir sonraki calistirma
        # buradan devam eder.
        ckpt_dict = {
            "item_tower": item_tower.state_dict(),
            "query_tower": query_tower.state_dict(),
            "optimizer": optimizer.state_dict(),
            "airport_to_idx": airport_to_idx,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if scheduler is not None:
            ckpt_dict["scheduler"] = scheduler.state_dict()
        torch.save(ckpt_dict, MODEL_OUT)
        print(f"  (checkpoint kaydedildi: epoch {epoch})", flush=True)

        # val_acc epoch'tan epoch'a DUSEBILIR (2026-07-24: epoch 5) -- "son"
        # checkpoint her zaman "en iyi" degildir, bu yuzden ayri takip edilir.
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            epochs_without_improvement = 0
            torch.save(ckpt_dict, BEST_MODEL_OUT)
            print(f"  >>> YENI EN IYI: epoch {epoch}, val_acc={val_acc:.4f} ({BEST_MODEL_OUT} guncellendi)", flush=True)
        else:
            epochs_without_improvement += 1
            if patience is not None:
                print(f"  (en iyiyi gecemedi: {epochs_without_improvement}/{patience} epoch'tur iyilesme yok)", flush=True)

        # Erken durdurma: patience epoch UST USTE yeni rekor gelmezse dur --
        # 5/11/18'deki gibi TEK epoch'luk gecici dususleri (hep bir sonrakinde
        # asilmisti) yanlislikla "overfit" saymamak icin patience yeterince
        # genis (>1) secilmeli (bkz. --patience CLI aciklamasi).
        if patience is not None and epochs_without_improvement >= patience:
            print(f"\n>>> ERKEN DURDURMA: {patience} epoch'tur (epoch {epoch - patience + 1}-{epoch}) "
                  f"yeni rekor gelmedi -- gercek/kalici bir plato/overfitting isareti olarak degerlendirildi.", flush=True)
            break

    # Nihai TEST iki ayri sekilde raporlanir: (1) o an bellekte olan agirliklarla
    # (son TAMAMLANAN epoch -- erken durdurulduysa bu epoch'un KENDISI zaten
    # kotu/overfit olabilir), (2) BEST_MODEL_OUT'un agirliklari GERI YUKLENEREK
    # -- asil onemli olan/kullanilacak model budur.
    test_loss, test_acc = evaluate(item_tower, query_tower, test_loader, x, adj, route_mask, loss_fn)
    print(f"\nTEST (son tamamlanan epoch, {epoch}): loss={test_loss:.4f} acc={test_acc:.4f}")

    best_ckpt_final = torch.load(BEST_MODEL_OUT, weights_only=False, map_location=DEVICE)
    item_tower.load_state_dict(best_ckpt_final["item_tower"])
    query_tower.load_state_dict(best_ckpt_final["query_tower"])
    best_test_loss, best_test_acc = evaluate(item_tower, query_tower, test_loader, x, adj, route_mask, loss_fn)
    print(f"TEST (en iyi checkpoint, epoch {best_epoch}): loss={best_test_loss:.4f} acc={best_test_acc:.4f}")
    print(f"En iyi val_acc: epoch {best_epoch}, val_acc={best_val_acc:.4f} -> {BEST_MODEL_OUT}")
    print(f"Son epoch modeli: {MODEL_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mask-routes", action="store_true",
                         help="Origin-kosullu aday kisitlamasi (deneysel, v1 sonuclarla dogrudan karsilastirilamaz)")
    parser.add_argument("--item-tower", choices=["gcn", "gat"], default="gcn",
                         help="gcn: sabit rota-agirlikli komsuluk (varsayilan). gat: komsu agirligini model ogrenir.")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce",
                         help="ce: duz CrossEntropyLoss (varsayilan). focal: nadir-rota agirlikli Focal Loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="--loss focal icin odaklanma parametresi")
    parser.add_argument("--class-balanced", action="store_true",
                         help="Train setindeki gercek destinasyon dagilimindan (Cui ve ark. 2019) sinif agirligi ekler")
    parser.add_argument("--patience", type=int, default=None,
                         help="Bu kadar epoch ust uste yeni val_acc rekoru gelmezse egitimi erken durdurur "
                              "(2026-07-25: epoch 5/11/18'deki gecici tek-epoch dususleri yanlislikla overfit "
                              "saymamak icin >1 onerilir, ör. 5)")
    parser.add_argument("--oversample", action="store_true",
                         help="Nadir rotalari sqrt-ters-frekans agirlikli sampler ile daha sik ornekler "
                              "(class-balanced loss'un YERINE, onunla birlikte kullanilmaz)")
    parser.add_argument("--augment", action="store_true",
                         help="Train ornekleri icin dinamik trajectory-level jitter (lat/lon/hiz/yon/vrate + "
                              "farkli downsample noktalari) -- oversample ile birlikte kullanilmasi onerilir")
    parser.add_argument("--feature-subset", choices=["full", "static12", "expanded30"], default="full",
                         help="'static12': sadece 12 gercek-statik ozellik. 'expanded30': mevcut 20 ozellik + 9 "
                              "yeni graf-topoloji ozelligi (~29 ozellik). Girdi boyutu degistigi icin ikisinde de "
                              "warm-start YOK.")
    parser.add_argument("--run-tag", default="",
                         help="Ayni hiperparametrelerle farkli bir veri seti icin ayri checkpoint (ör. 'opensky')")
    parser.add_argument("--hidden-dim", type=int, default=32,
                         help="Item/query tower gizli boyutu (varsayilan 32). Degistirilirse warm-start YOK.")
    parser.add_argument("--out-dim", type=int, default=16,
                         help="Item/query tower cikti (embedding) boyutu (varsayilan 16). Degistirilirse warm-start YOK.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 regularizasyon katsayisi")
    parser.add_argument("--lr-scheduler", action="store_true",
                         help="ReduceLROnPlateau (val_acc, patience=2, factor=0.5) aktif eder")
    parser.add_argument("--min-route-count", type=int, default=None,
                         help="Train setinde bu sayidan AZ tekrar eden (origin,dest) rotalarini TUM "
                              "split'lerden (train/val/test) tamamen cikarir (ör. 10 -> 2-4 ve 5-9 "
                              "dilimleri elenir). Nadir-rota iyilestirme denemelerinden SONRA referans "
                              "karsilastirma icin.")
    args = parser.parse_args()
    main(n_epochs=args.epochs, batch_size=args.batch_size, mask_routes=args.mask_routes,
         item_tower_kind=args.item_tower, loss_kind=args.loss, focal_gamma=args.focal_gamma,
         class_balanced=args.class_balanced, patience=args.patience,
         oversample=args.oversample, augment=args.augment, feature_subset=args.feature_subset,
         run_tag=args.run_tag, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
         weight_decay=args.weight_decay, use_lr_scheduler=args.lr_scheduler,
         min_route_count=args.min_route_count)
