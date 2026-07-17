from pathlib import Path


def test_residual_v1_contains_no_forbidden_interpolation_patterns():
    forbidden = ("interpolate" + "(", ".resample" + "(", "fillna" + "(method=")
    hits = []
    for path in Path("residual_v1").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in source:
                hits.append(f"{path}: {pattern}")
    assert not hits, "forbidden Silver interpolation patterns:\n" + "\n".join(hits)

