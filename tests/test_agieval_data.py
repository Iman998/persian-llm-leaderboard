from pathlib import Path
import hashlib
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from leaderboard_lib.agieval_utils import parse_agieval_scoring_key


def test_agieval_v11_data_integrity():
    dataset_dir = ROOT / "data" / "AGIEval"
    data_path = dataset_dir / "test.csv"
    meta = yaml.safe_load((dataset_dir / "meta.yaml").read_text(encoding="utf-8"))
    df = pd.read_csv(data_path, keep_default_na=False)

    assert len(df) == 7272
    assert df["task"].nunique() == 21
    assert set(df["question_type"]) == {"mcq", "cloze"}
    assert set(df["language"]) == {"en", "zh"}
    assert not df["question"].str.strip().eq("").any()
    assert not df["target"].str.strip().eq("").any()
    assert (
        hashlib.sha256(data_path.read_bytes()).hexdigest()
        == meta["converted_data_sha256"]
    )


def test_agieval_scoring_keys_match_row_metadata():
    data_path = ROOT / "data" / "AGIEval" / "test.csv"
    df = pd.read_csv(data_path, keep_default_na=False)

    for row in df.itertuples(index=False):
        key = parse_agieval_scoring_key(row.scoring_key)
        assert key["task"] == row.task
        assert key["question_type"] == row.question_type
        assert key["target"]


def test_agieval_mcq_rows_have_choices_and_cloze_rows_do_not():
    data_path = ROOT / "data" / "AGIEval" / "test.csv"
    df = pd.read_csv(data_path, keep_default_na=False)
    choice_columns = [f"choice{index}" for index in range(1, 6)]
    mcq = df[df["question_type"] == "mcq"]
    cloze = df[df["question_type"] == "cloze"]

    assert mcq["choice1"].str.strip().ne("").all()
    assert cloze[choice_columns].eq("").all(axis=None)
