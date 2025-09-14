import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ter", Path(__file__).resolve().parents[1] / "metrics" / "ter.py"
)
ter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ter)


def test_ter_identical():
    assert ter.compute(["سلام دنیا"], ["سلام دنیا"]) == 1.0


def test_ter_different():
    assert ter.compute(["abc"], ["xyz"]) < 1.0
