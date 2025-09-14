import subprocess
import sys
from pathlib import Path

def test_run_eval_cli(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("question,Key\nWhat is 2+2?,4\n")

    prompt = tmp_path / "prompt.jinja2"
    prompt.write_text("{{question}} ->")

    evaluator = tmp_path / "dummy_evaluator.py"
    evaluator.write_text(
        "import pandas as pd\n"
        "class DummyEvaluator:\n"
        "    def __init__(self, model_cfg, prompt_path, meta_path, shots, max_retries):\n"
        "        pass\n"
        "    def evaluate_df(self, df, max_workers):\n"
        "        df = df.copy()\n"
        "        df['pred'] = df['Key']\n"
        "        return df\n"
    )

    meta = tmp_path / "meta.yaml"
    meta.write_text(
        f"evaluator: {evaluator}\n"
        f"prompt_template: {prompt}\n"
        "answer_col: Key\n"
        "question_col: question\n"
    )

    model = tmp_path / "model.yaml"
    model.write_text("name: dummy\n")

    out = tmp_path / "out.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--dataset",
            str(dataset),
            "--meta",
            str(meta),
            "--model",
            str(model),
            "--prompt",
            str(prompt),
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert out.exists()
