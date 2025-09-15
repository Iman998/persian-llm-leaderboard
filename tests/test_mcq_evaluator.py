from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluators.mcq_evaluator import MCQEvaluator


@pytest.fixture
def evaluator(tmp_path):
    prompt_path = tmp_path / "template.jinja2"
    prompt_path.write_text("{{ question }}")
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text("choice_cols: ['a', 'b']\n")
    model_cfg = {"api_key": "test", "base_url": "http://localhost", "model": "dummy"}
    return MCQEvaluator(model_cfg=model_cfg, prompt_path=prompt_path, meta_path=meta_path)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("prefix **** 3 **** suffix", "3"),
        ("پاسخ: ****۳****", "3"),
        ("درست است ****۳.۰****", "3"),
    ],
)
def test_extract_valid(text, expected, evaluator):
    assert evaluator._extract(text) == expected


@pytest.mark.parametrize("text", [None, "", "****abc****", "no pattern here"])
def test_extract_invalid(text, evaluator):
    assert evaluator._extract(text) is None
