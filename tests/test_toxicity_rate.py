import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "toxicity_rate", Path(__file__).resolve().parents[1] / "metrics" / "toxicity_rate.py"
)
toxicity_rate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toxicity_rate)


def test_toxicity_rate_mixed():
    preds = ["toxic", "0", 1, "no", "positive", 0]
    assert toxicity_rate.compute(preds, [0] * len(preds)) == 0.5


def test_toxicity_rate_empty():
    assert toxicity_rate.compute([], []) == 0.0


def test_toxicity_rate_various_representations():
    preds = [1, "1", "true", "Yes", "positive", "toxic", "y"]
    assert toxicity_rate.compute(preds, [0] * len(preds)) == 1.0
