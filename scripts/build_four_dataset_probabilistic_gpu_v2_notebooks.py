"""Derive balanced-split v2 Colab notebooks from the verified v1 notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "artifacts/legacy_gpu/colab/four_dataset_probabilistic_v1_transfer"
INDEX = TRANSFER / "transfer_index_v2.json"
DATASETS = ("alfa", "uav_attack", "uav_sead", "rflymad")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown(dataset: str, counts: dict) -> str:
    names = {
        "alfa": "ALFA",
        "uav_attack": "UAV-Attack",
        "uav_sead": "UAV-SEAD",
        "rflymad": "RflyMAD",
    }
    return f"""# {names[dataset]} probabilistic GPU v2 — balanced Colab L4

Bu notebook yeni four_dataset_probabilistic_gpu_v2 sözleşmesini çalıştırır.
Tamamlanmış v1 sonucu değiştirilmez. Model geçmiş 32 satırdan bir sonraki
satırın kanal bazlı mean ve scale değerlerini öğrenir; optimizer yalnız normal
uçuşları görür.

Frozen source-level roller:

- normal train: {counts['train']}
- normal validation: {counts['val']}
- primary test: {counts['primary_test']} kaynak
- primary test normal/anomaly: {counts['primary_test_normal']}/{counts['primary_test_anomaly']}
- ayrı stress-test anomaly havuzu: {counts['stress_test_anomaly']}

Normal havuz 70/15/15 bölünmüştür. Ana test, sınıf oranının ROC/AP yorumunu
bozmaması için dengelenmiştir. Stress-test havuzu threshold, optimizer veya
primary sonuç seçiminde kullanılmaz.
"""


def main() -> int:
    index_sha = _sha256(INDEX)
    split = json.loads(
        (
            ROOT
            / "configs/four_dataset_probabilistic_gpu_v2_split_manifest.json"
        ).read_text(encoding="utf-8")
    )
    old_index_sha = "1da9acaa509941dbc3ff3b6db07bccc50f63b4b13063d15debcd59388ed11af0"
    for dataset in DATASETS:
        source = ROOT / "notebooks" / f"{dataset}_probabilistic_gpu_v1_colab.ipynb"
        target = ROOT / "notebooks" / f"{dataset}_probabilistic_gpu_v2_colab.ipynb"
        notebook = json.loads(source.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            text = "".join(cell.get("source", []))
            text = text.replace(
                "four_dataset_probabilistic_gpu_v1",
                "four_dataset_probabilistic_gpu_v2",
            )
            text = text.replace(
                "four_dataset_probabilistic_v1_runs",
                "four_dataset_probabilistic_v2_runs",
            )
            text = text.replace("_gpu_v1", "_gpu_v2")
            text = text.replace("transfer_index.json", "transfer_index_v2.json")
            text = text.replace(old_index_sha, index_sha)
            text = text.replace("aynı v1", "aynı v2")
            text = text.replace("Bu v1", "Bu v2")
            text = text.replace("bu v1", "bu v2")
            text = text.replace("fold registry", "balanced split manifest")
            cell["source"] = text.splitlines(keepends=True)
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        notebook["cells"][0]["source"] = _markdown(
            dataset, split["sources"][dataset]["counts"]
        ).splitlines(keepends=True)
        notebook["cells"][1]["source"] = (
            "## 1. Drive ve sabit yollar\n\n"
            "Mevcut four_dataset_probabilistic_v1_transfer klasöründeki dataset "
            "ZIP'i yeniden kullanılabilir. Aynı klasöre yalnız "
            "four_dataset_probabilistic_gpu_v2_code_and_contract.zip, "
            "transfer_index_v2.json ve bundle_manifest_v2.json dosyalarını "
            "ekleyin. Eski transfer_index.json dosyasını silmeyin; bu notebook "
            "yalnız transfer_index_v2.json dosyasını okur."
        ).splitlines(keepends=True)
        notebook["metadata"]["four_dataset_contract"] = {
            "namespace": "four_dataset_probabilistic_gpu_v2",
            "split": "balanced_v2",
            "transfer_index_sha256": index_sha,
        }
        target.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
