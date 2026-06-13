from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from views import llm_judge


def test_discover_judge_datasets_includes_zharfa_result():
    original_datasets = llm_judge.DATASETS
    llm_judge.DATASETS = [
        "zharfa_translate",
        "zharfa_translate_judge",
        "zharfa_translate_judge_reference",
        "zharfa_translate_judge_no_reference",
        "translation_quality",
        "ordinary",
    ]
    try:
        assert llm_judge._discover_judge_datasets() == [
            "zharfa_translate_judge",
            "zharfa_translate_judge_reference",
            "zharfa_translate_judge_no_reference",
            "translation_quality",
        ]
    finally:
        llm_judge.DATASETS = original_datasets


def test_collect_judge_table_without_numeric_cols(tmp_path):
    df = pd.DataFrame({"text": ["a", "b"], "note": ["x", "y"]})
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    original_map = llm_judge.MAIN_MAP
    llm_judge.MAIN_MAP = {("fake_ds", "modelA"): csv_path}
    try:
        table_df, warnings = llm_judge._collect_judge_table("fake_ds")
        assert table_df.empty
        assert warnings  # ensure warning about missing columns
    finally:
        llm_judge.MAIN_MAP = original_map
