from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "parsinlu-qqp" / "test.csv"
)


def test_parsinlu_qqp_matches_official_test_shape():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    assert df.shape == (1916, 4)
    assert df.columns.tolist() == [
        "question_1",
        "question_2",
        "is_duplicate",
        "data_source",
    ]


def test_parsinlu_qqp_records_are_structurally_valid():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    assert df[["question_1", "question_2"]].ne("").all().all()
    assert df["is_duplicate"].isin({"0", "1"}).all()
    assert df["data_source"].isin({"natural", "qqp"}).all()
    assert df["is_duplicate"].value_counts().to_dict() == {"0": 1082, "1": 834}
    assert df["data_source"].value_counts().to_dict() == {
        "natural": 1438,
        "qqp": 478,
    }
