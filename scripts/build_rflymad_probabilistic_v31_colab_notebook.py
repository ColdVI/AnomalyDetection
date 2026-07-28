"""Generate the checksum-pinned RflyMAD v3.1 Colab notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer"
INDEX = TRANSFER / "transfer_index_v31.json"
OUTPUT = ROOT / "notebooks/rflymad_probabilistic_v31_colab.ipynb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def main() -> int:
    index_sha = _sha256(INDEX)
    cells = [
        _cell(
            "markdown",
            """# RflyMAD probabilistic v3.1 — group-safe Colab L4

Bu koşu yalnız 260 normal train kaynağında optimizer çalıştırır. 90 normal
validation checkpoint/eşik seçer. Geliştirme değerlendirmesi 91 normal test ve
557 `anomaly_dev` kaynağını kullanır; mühürlü 553 `final_fault_test` kaynağı
runner rol listesinde yoktur. Eğitim grup, sonra kaynak dengeli örneklenir.
""",
        ),
        _cell(
            "code",
            """from google.colab import drive
drive.mount('/content/drive')
""",
        ),
        _cell(
            "code",
            f"""from pathlib import Path
import hashlib, json, shutil, zipfile

TRANSFER = Path('/content/drive/MyDrive/bykr/four_dataset_probabilistic_v31_transfer')
REPO = Path('/content/four_dataset_probabilistic_v31')
RUN_DIR = Path('/content/drive/MyDrive/bykr/four_dataset_probabilistic_v31_runs/rflymad_v31')
EXPECTED_INDEX_SHA256 = '{index_sha}'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

index_path = TRANSFER / 'transfer_index_v31.json'
assert index_path.is_file(), index_path
assert sha256(index_path) == EXPECTED_INDEX_SHA256
index = json.loads(index_path.read_text(encoding='utf-8'))
assert index['candidate_namespace'] == 'four_dataset_probabilistic_v31'
for record in index['archives']:
    path = TRANSFER / record['path']
    assert path.is_file(), path
    assert path.stat().st_size == record['bytes']
    assert sha256(path) == record['sha256']
print('TRANSFER CONTRACT PASS')
""",
        ),
        _cell(
            "code",
            """if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
for record in index['archives']:
    with zipfile.ZipFile(TRANSFER / record['path']) as archive:
        archive.extractall(REPO)
print('Extracted:', REPO)
""",
        ),
        _cell(
            "code",
            """import subprocess, sys, torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('GPU runtime seç: Runtime > Change runtime type > L4/T4 GPU')
verify = [
    sys.executable, str(REPO / 'scripts/four_dataset_probabilistic_v31_runner.py'),
    'verify', '--repo-root', str(REPO), '--dataset', 'rflymad', '--device', 'cuda'
]
subprocess.run(verify, check=True)
""",
        ),
        _cell(
            "markdown",
            """## Eğitim

Bu hücre 30 epoch çalışır ve her tamamlanmış epoch sonunda Drive checkpointi
yazar. Bağlantı koparsa aynı hücreyi yeniden çalıştırmak kaldığı epoch'tan devam
eder. L4 için beklenen süre yaklaşık 5–15 dakikadır; ilk gerçek v3.1 ölçümü esas
alınacaktır.
""",
        ),
        _cell(
            "code",
            """RUN_DIR.mkdir(parents=True, exist_ok=True)
train = [
    sys.executable, str(REPO / 'scripts/four_dataset_probabilistic_v31_runner.py'),
    'train', '--repo-root', str(REPO), '--dataset', 'rflymad',
    '--device', 'cuda', '--run-dir', str(RUN_DIR)
]
subprocess.run(train, check=True)
""",
        ),
        _cell(
            "code",
            """report_path = RUN_DIR / 'training_report.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
summary = {
    'completed_epochs': report['completed_epochs'],
    'selected_checkpoint': report['selected_checkpoint'],
    'magnitude_domination_flagged_at_0_8': report['magnitude_domination_flagged_at_0_8'],
    'flight_level': report['evaluation']['flight_level'],
    'normal_flight_alarm_fraction': report['evaluation']['normal_flight_alarm_fraction'],
    'anomalous_flight_detection_rate': report['evaluation']['anomalous_flight_detection_rate'],
}
print(json.dumps(summary, indent=2))
if summary['magnitude_domination_flagged_at_0_8']:
    raise RuntimeError('Magnitude gate TRUE: v3.1 ablation aşamasına geçmeyin.')
""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "four_dataset_contract": {
                "namespace": "four_dataset_probabilistic_v31",
                "transfer_index_sha256": index_sha,
                "final_fault_test_visible": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT), _sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
