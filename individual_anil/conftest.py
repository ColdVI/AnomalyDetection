"""Bu klasörün testleri ve scriptleri iki farklı köke göreli import kullanır:

1. `individual_anil/` kökü -- `adsb.*` ve `gecmis_calismalar.*` paket
   import'ları için (bkz. `from adsb.features import ...`,
   `from gecmis_calismalar.residual_v1.ingest.alfa import ...`).
2. Repo kökü -- `gecmis_calismalar/{ALFA,UAV_ATTACK,UAV_SEAD,RFLYMAD}/` altındaki
   dondurulmuş arşiv kodu (`kaynak_kod_legacy/`, `kaynak_kod/`,
   `legacy_rfly0_1/parse_rflymad.py`) ortak veri hattının eski hâlinden
   `from src.common.minio_io import ...` / `from src.common.provenance import
   ...` şeklinde import ediyor.

Bu iki yolu da ekleyerek hem `cd individual_anil && pytest` hem repo
kökünden `pytest individual_anil/tests -q` çalışır hâle geliyor. NOT: (2)
altındaki import'ların bir kısmı yine de kırık olabilir -- bkz.
KOD_HARITASI.md, bu kod zaten çalıştırılabilir olması hedeflenmeyen
dondurulmuş arşivdir.

Ayrıca: `configs/...` ve `artifacts/...` gibi bazı yollar (bkz. örn.
`gecmis_calismalar/residual_v1/ingest/rfly.py`) `sys.path`'e değil,
**çalışma dizinine (cwd)** göreli -- bunlar bu klasörün dosyaları taşınmadan
önce de böyleydi (bkz. KRİTİK: import değişmezi notu, üst dizindeki görev
talimatı). Aşağıdaki autouse fixture, testler bu klasörün ALTINDA
toplandığında cwd'yi `individual_anil/`'e çevirip test bitince eski hâline
geri alıyor -- yalnız BU klasörün testlerini etkiler, `tests/` (takım)
testlerinin cwd'sine dokunmaz (pytest conftest fixture'ları kendi dizin
alt ağacıyla sınırlıdır).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_INDIVIDUAL_ANIL_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _INDIVIDUAL_ANIL_ROOT.parent

for _path in (_INDIVIDUAL_ANIL_ROOT, _REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


@pytest.fixture(autouse=True)
def _individual_anil_cwd():
    previous = os.getcwd()
    os.chdir(_INDIVIDUAL_ANIL_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)
