import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "chrf", Path(__file__).resolve().parents[1] / "metrics" / "chrf.py"
)
chrf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chrf)


def test_chrf_identical():
    assert chrf.compute(["سلام دنیا"], ["سلام دنیا"]) == 1.0


def test_chrf_different():
    assert chrf.compute(["abc"], ["xyz"]) == 0.0
