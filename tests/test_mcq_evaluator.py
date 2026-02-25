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


def test_chat_completion_options_without_thinking_flag(evaluator):
    options = evaluator._chat_completion_options()
    assert options["temperature"] == 0.01
    assert options["top_p"] == 0.01
    assert options["max_tokens"] == 4000
    assert "extra_body" not in options


def test_chat_completion_options_with_thinking_disabled(tmp_path):
    prompt_path = tmp_path / "template.jinja2"
    prompt_path.write_text("{{ question }}")
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text("choice_cols: ['a', 'b']\n")
    model_cfg = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "dummy",
        "enable_thinking": False,
    }

    eval_with_thinking = MCQEvaluator(
        model_cfg=model_cfg,
        prompt_path=prompt_path,
        meta_path=meta_path,
    )
    options = eval_with_thinking._chat_completion_options()
    assert options["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
