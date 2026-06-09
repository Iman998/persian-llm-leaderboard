from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "parsinlu-entailment"
    / "test.csv"
)


def test_parsinlu_entailment_has_only_labeled_official_rows():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    assert df.shape == (1673, 5)
    assert df.columns.tolist() == [
        "id",
        "sentence_1",
        "sentence_2",
        "relation_label",
        "data_source",
    ]
    assert df[["id", "sentence_1", "sentence_2", "data_source"]].ne("").all().all()
    assert df["relation_label"].value_counts().to_dict() == {
        "entailment": 610,
        "contradiction": 561,
        "neutral": 502,
    }


def test_parsinlu_entailment_source_categories_are_preserved():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    assert df["data_source"].value_counts().to_dict() == {
        "translation-train": 713,
        "natural-wiki": 590,
        "natural-voa": 134,
        "natural-miras": 126,
        "translation-dev": 110,
    }
    assert not df["sentence_1"].str.contains(r"[\r\n]", regex=True).any()
    assert not df["sentence_2"].str.contains(r"[\r\n]", regex=True).any()
