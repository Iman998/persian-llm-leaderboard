import importlib.util
import sys
import types
from pathlib import Path
import pytest

# Create a pseudo-package for metrics so relative imports work
metrics_path = Path(__file__).resolve().parents[1] / "metrics"
metrics_pkg = types.ModuleType("metrics")
metrics_pkg.__path__ = [str(metrics_path)]
sys.modules.setdefault("metrics", metrics_pkg)


def _load_metric(name: str):
    spec = importlib.util.spec_from_file_location(f"metrics.{name}", metrics_path / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"metrics.{name}"] = module
    spec.loader.exec_module(module)
    return module


rouge1 = _load_metric("rouge1")
rouge = _load_metric("rouge")


def test_rouge1_matches_rouge_n():
    preds = ["this is a test", "another test"]
    labels = ["this is a test", "another one"]
    assert rouge1.compute(preds, labels) == rouge.compute_rouge_n(preds, labels, n=1)


def test_rouge1_empty_inputs():
    preds: list[str] = []
    labels: list[str] = []
    result1 = rouge1.compute(preds, labels)
    result2 = rouge.compute_rouge_n(preds, labels, n=1)
    assert result1 == result2 == 0.0


@pytest.mark.parametrize(
    "preds,labels",
    [
        (["a b c", "extra"], ["a b c"]),  # preds longer
        (["a b c"], ["a b c", "extra"]),  # labels longer
    ],
)
def test_rouge1_mismatched_lengths(preds, labels):
    assert rouge1.compute(preds, labels) == rouge.compute_rouge_n(preds, labels, n=1)

