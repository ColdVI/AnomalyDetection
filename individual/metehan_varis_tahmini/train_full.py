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
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")
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
from torch.utils.data import Dataset, DataLoader

from individual.metehan_varis_tahmini.airport_gcn import (
    AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
)
from individual.metehan_varis_tahmini.airport_gat import AirportGAT, build_adjacency_mask
from individual.metehan_varis_tahmini.query_tower import QueryTower, POINT_DIM

SESSIONS_DIR = Path(__file__).parent / "training_sessions"
SPLIT_PATH = Path(__file__).parent / "session_split.parquet"


def checkpoint_paths(item_tower_kind: str, loss_kind: str) -> tuple[Path, Path]:
    """GCN/GAT ve CE/Focal farkli agirlik yapilari uretir -- AYNI dosya
    adini kullanmak, bir deneyin checkpoint'ini digerine yanlislikla
    'resume' etmeye (state_dict uyusmazligi/sessiz bozulma) yol acardi.
    Varsayilan (gcn+ce) ESKI sabit isimleri korur, boylece mevcut/devam
    eden GCN egitimi ETKILENMEZ."""
    suffix = ""
    if item_tower_kind != "gcn":
        suffix += f"_{item_tower_kind}"
    if loss_kind != "ce":
        suffix += f"_{loss_kind}"
    base = Path(__file__).parent
    return base / f"model_checkpoint{suffix}.pt", base / f"model_checkpoint{suffix}_best.pt"


# Varsayilan (gcn+ce) konfigurasyonun sabit yolu -- evaluate_by_route_frequency.py
# gibi diger scriptler bunu dogrudan import ediyor, geriye-uyumluluk icin korunuyor.
MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths("gcn", "ce")

MAX_SEQ_LEN = 300


def _downsample(arr, n, max_len):
    if n <= max_len:
        return arr
    idx = np.linspace(0, n - 1, max_len).round().astype(int)
    return [arr[i] for i in idx]


class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, airport_to_idx: dict[str, int], require_origin: bool = False):
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        n = min(row["n_points"], MAX_SEQ_LEN)
        lats = _downsample(row["lats"], row["n_points"], MAX_SEQ_LEN)
        lons = _downsample(row["lons"], row["n_points"], MAX_SEQ_LEN)
        alts = _downsample(row["alts"], row["n_points"], MAX_SEQ_LEN)
        vels = _downsample(row["velocities"], row["n_points"], MAX_SEQ_LEN)
        heads = _downsample(row["headings"], row["n_points"], MAX_SEQ_LEN)
        vrates = _downsample(row["vrates"], row["n_points"], MAX_SEQ_LEN)
        dt_norms = _downsample(row["dt_norms"], row["n_points"], MAX_SEQ_LEN)

        points = np.zeros((n, POINT_DIM), dtype=np.float32)
        points[:, 0] = lats
        points[:, 1] = lons
        points[:, 2] = np.array(alts, dtype=np.float32) / 1000.0
        points[:, 3] = np.array(vels, dtype=np.float32) / 100.0
        heading_rad = np.radians(heads)
        points[:, 4] = np.sin(heading_rad)
        points[:, 5] = np.cos(heading_rad)
        points[:, 6] = np.array(vrates, dtype=np.float32) / 10.0
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
         patience: int | None = None):
    MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths(item_tower_kind, loss_kind)
    print(f"item_tower={item_tower_kind}, loss={loss_kind}"
          + (f" (gamma={focal_gamma})" if loss_kind == "focal" else "")
          + (", class_balanced=True" if class_balanced else "")
          + f" -> checkpoint: {MODEL_OUT.name}", flush=True)

    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    # GCN: rota-sayisina gore ONCEDEN AGIRLIKLI komsuluk matrisi.
    # GAT: agirligi model KENDI OGRENIR, burada sadece "kenar var mi" (Bool) yeterli.
    adj = build_normalized_adjacency(route_counts, idents) if item_tower_kind == "gcn" \
        else build_adjacency_mask(route_counts, idents)

    split_df = pd.read_parquet(SPLIT_PATH)
    split_map = split_df.set_index("session_id")["split"]

    all_df = load_all_sessions()
    all_df["split"] = all_df["session_id"].map(split_map)
    all_df = all_df.dropna(subset=["split"])  # 1-kez ucan rotalarin session'lari elenir
    print(f"Split sonrasi kullanilabilir satir: {len(all_df)}", flush=True)
    print(all_df["split"].value_counts(), flush=True)

    # route_scan_full_archive.json (a_norm icin yukarida kullanilan) bolge-
    # filtresi/adsbdb duzeltmesinden ONCEKI eski taramaya ait -- maske bunun
    # yerine DOGRUDAN v2 session'larindan kuruluyor (bkz. fonksiyon docstring'i).
    # mask_routes=False iken maskeleme tamamen devre disi (tam-852 softmax,
    # v1 ile ayni davranis) -- iki deney de AYNI script'ten, koda dokunmadan
    # karsilastirilabilsin diye.
    route_mask = build_route_mask_from_sessions(all_df, airport_to_idx) if mask_routes else None

    train_ds = TrajectoryDataset(all_df[all_df["split"] == "train"], airport_to_idx, require_origin=mask_routes)
    val_ds = TrajectoryDataset(all_df[all_df["split"] == "val"], airport_to_idx, require_origin=mask_routes)
    test_ds = TrajectoryDataset(all_df[all_df["split"] == "test"], airport_to_idx, require_origin=mask_routes)
    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)

    torch.manual_seed(0)
    if item_tower_kind == "gcn":
        item_tower = AirportGCN(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gcn_layers=2)
    else:
        item_tower = AirportGAT(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gat_layers=2, n_heads=4)
    query_tower = QueryTower(d_model=32, nhead=4, num_layers=2, out_dim=16)
    optimizer = torch.optim.Adam(list(item_tower.parameters()) + list(query_tower.parameters()), lr=1e-3)

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

    # yarida kesilirse (PC/oturum kapanmasi) EN AZINDAN son tamamlanan epoch'u
    # kaybetmemek icin -- var olan checkpoint'ten devam
    start_epoch = 0
    if MODEL_OUT.exists():
        ckpt = torch.load(MODEL_OUT, weights_only=False)
        item_tower.load_state_dict(ckpt["item_tower"])
        query_tower.load_state_dict(ckpt["query_tower"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
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
        best_ckpt = torch.load(BEST_MODEL_OUT, weights_only=False)
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
                print(f"  epoch={epoch} batch={i+1}/{len(train_loader)} "
                      f"running_loss={epoch_loss/epoch_n:.4f} running_acc={epoch_correct/epoch_n:.4f} "
                      f"{rate:.2f} batch/sn, ETA bu epoch {eta_min:.1f} dk", flush=True)

        val_loss, val_acc = evaluate(item_tower, query_tower, val_loader, x, adj, route_mask, loss_fn)
        print(f"epoch={epoch} train_loss={epoch_loss/epoch_n:.4f} train_acc={epoch_correct/epoch_n:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}sn)", flush=True)

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

    best_ckpt_final = torch.load(BEST_MODEL_OUT, weights_only=False)
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
    args = parser.parse_args()
    main(n_epochs=args.epochs, batch_size=args.batch_size, mask_routes=args.mask_routes,
         item_tower_kind=args.item_tower, loss_kind=args.loss, focal_gamma=args.focal_gamma,
         class_balanced=args.class_balanced, patience=args.patience)
