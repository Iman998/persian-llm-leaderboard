import importlib.util
import sys
import types
from pathlib import Path

import pytest

METRICS_PATH = Path(__file__).resolve().parents[1] / "metrics"
metrics_pkg = types.ModuleType("metrics")
metrics_pkg.__path__ = [str(METRICS_PATH)]
sys.modules.setdefault("metrics", metrics_pkg)

spec2 = importlib.util.spec_from_file_location(
    "metrics.rouge2", METRICS_PATH / "rouge2.py"
)
rouge2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rouge2)

spec = importlib.util.spec_from_file_location(
    "metrics.rouge", METRICS_PATH / "rouge.py"
)
rouge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rouge)


def test_rouge2_matches_rouge_n():
    preds = ["the cat sat on the mat"]
    labels = ["the cat is on the mat"]
    assert rouge2.compute(preds, labels) == pytest.approx(
        rouge.compute_rouge_n(preds, labels, n=2)
    )


def test_rouge2_empty_lists():
    assert rouge2.compute([], []) == 0.0


def test_rouge2_mismatched_lengths():
    preds = ["a b", "c d"]
    labels = ["a b", "c d", "e f"]
    assert rouge2.compute(preds, labels) == pytest.approx(
        rouge.compute_rouge_n(preds, labels, n=2)
    )
