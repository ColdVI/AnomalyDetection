import pandas as pd
import pytest

from src.common.io import write_bronze


def test_write_bronze_round_trip(tmp_path):
    source = pd.DataFrame({"raw_value": [1, 2]})
    output = write_bronze(source, "alfa", "flight=demo", bronze_root=tmp_path)

    assert output.parent == tmp_path / "alfa" / "flight=demo"
    assert output.name.startswith("part-")
    assert pd.read_parquet(output).equals(source)


@pytest.mark.parametrize("unsafe", ["../alfa", "a/b", "", "with space"])
def test_write_bronze_rejects_unsafe_path_components(tmp_path, unsafe):
    with pytest.raises(ValueError):
        write_bronze(pd.DataFrame(), unsafe, bronze_root=tmp_path)
