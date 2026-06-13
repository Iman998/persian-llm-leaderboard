import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from leaderboard_lib import llm_judge_board


def test_board_uses_only_judge_results_and_normalizes_average(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    models = tmp_path / "models"
    datasets = tmp_path / "data"
    metrics = tmp_path / "metrics"
    judge_results = results / "zharfa_translate_judge_reference" / "candidate"
    no_reference_results = (
        results / "zharfa_translate_judge_no_reference" / "candidate"
    )
    normal_results = results / "ordinary" / "candidate"
    zharfa_data = datasets / "zharfa_translate"
    for path in (
        models,
        metrics,
        judge_results,
        no_reference_results,
        normal_results,
        zharfa_data,
    ):
        path.mkdir(parents=True, exist_ok=True)

    (models / "candidate.yaml").write_text(
        "display_name: Candidate\nmodel_type: Instruct\n"
    )
    (metrics / "llm_judge_score.py").write_text(
        "def compute(preds, labels):\n"
        "    values = [float(value) for value in preds]\n"
        "    return sum(values) / len(values)\n"
    )
    (zharfa_data / "meta.yaml").write_text(
        "answer_col: gold\n"
        "judge:\n"
        "  enabled: true\n"
        "  score_max: 100\n"
        "  metrics: [llm_judge_score]\n"
    )
    pd.DataFrame({"pred": [80, 70], "gold": ["a", "b"]}).to_csv(
        judge_results / "candidate.csv", index=False
    )
    pd.DataFrame({"pred": [60, 70], "gold": ["a", "b"]}).to_csv(
        no_reference_results / "candidate.csv", index=False
    )
    pd.DataFrame({"pred": [1], "gold": ["1"]}).to_csv(
        normal_results / "candidate.csv", index=False
    )

    output = tmp_path / "judge-board.csv"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_llm_judge_board.py",
            "--results_dir",
            str(results),
            "--datasets_dir",
            str(datasets),
            "--models_dir",
            str(models),
            "--out",
            str(output),
        ],
    )
    llm_judge_board.main()

    board = pd.read_csv(output)
    assert board.loc[0, "Model"] == "Candidate"
    assert board.loc[0, "zharfa_translate (Reference Score)"] == 75.0
    assert board.loc[0, "zharfa_translate (No-reference Score)"] == 65.0
    assert board.loc[0, "Average"] == 0.7
    assert not any("ordinary" in column for column in board.columns)
