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


def test_ter_empty_reference():
    assert ter.compute(["abc"], [""]) == 0.0


def test_ter_mismatched_lengths():
    assert ter.compute(["سلام", "دنیا"], ["سلام"]) == 1.0
    assert ter.compute(["سلام"], ["سلام", "دنیا"]) == 1.0


def test_ter_numeric_inputs():
    assert ter.compute([123], [123]) == 1.0
    assert ter.compute([123], [456]) < 1.0
