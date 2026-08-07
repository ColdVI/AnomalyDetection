"""Kalici egitim izi kurali (ML-11 Bolum 5).

Bundan sonra egitilen HER model, epoch bazli train/val loss kaydini
``artifacts/training_logs/<source>/<model>/<run_id>/`` altina birakir:
``loss.csv`` (epoch, train_loss, val_loss) + otomatik ``loss.png``.

- LSTM-AE: ``train_lstm_autoencoder`` artik ``info["history"]`` dondurur;
  paketleme scriptleri bunu ``write_training_log``'a gecirir.
- Isolation Forest'in epoch kavrami yoktur; onun icin zorunlu iz yok.
- ML-10 Chronos zero-shot'tir (egitim adimi yok); kapsam disidir.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "artifacts/training_logs"


def write_training_log(history: list[dict], source: str, model_name: str,
                       run_id: str, *, root: str | Path = DEFAULT_ROOT) -> Path:
    """Epoch gecmisini loss.csv + loss.png olarak yaz, klasoru dondur.

    ``history`` her epoch icin {"epoch", "train_loss", "val_loss"} icerir
    (``train_lstm_autoencoder`` ciktisindaki ``info["history"]`` formati).
    """
    if not history:
        raise ValueError("Egitim gecmisi bos; loss izi yazilamaz")
    out = Path(root) / source / model_name / run_id
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(history)[["epoch", "train_loss", "val_loss"]]
    frame.to_csv(out / "loss.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["epoch"], frame["train_loss"], label="train loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="val loss")
    best = frame.loc[frame["val_loss"].idxmin()]
    ax.axvline(best["epoch"], color="tab:red", ls="--", lw=0.8,
               label=f"en iyi val (epoch {int(best['epoch'])})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("maskeli MSE loss")
    ax.set_yscale("log")
    ax.set_title(f"{source} / {model_name} / {run_id} — Egitim Egrisi "
                 f"(n={len(frame)} epoch)")
    ax.legend()
    fig.savefig(out / "loss.png", bbox_inches="tight")
    plt.close(fig)
    return out
