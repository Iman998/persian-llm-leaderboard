from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "parsinlu-mcqa" / "test.csv"
)


def test_parsinlu_mcqa_matches_official_test_shape():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    assert df.shape == (1050, 8)
    assert df.columns.tolist() == [
        "id",
        "question",
        "option1",
        "option2",
        "option3",
        "option4",
        "answer",
        "category",
    ]


def test_parsinlu_mcqa_records_are_structurally_valid():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    assert df["id"].ne("").all()
    assert df["question"].ne("").all()
    assert df[["option1", "option2"]].ne("").all().all()
    assert df["answer"].isin({"1", "2", "3", "4"}).all()
    assert df["category"].isin(
        {"math_and_logic", "common_knowledge", "literature"}
    ).all()

    choice_cols = ["option1", "option2", "option3", "option4"]
    choice_counts = df[choice_cols].ne("").sum(axis=1)
    assert choice_counts.between(2, 4).all()
    assert all(
        row[f"option{int(row['answer'])}"] != ""
        for _, row in df.iterrows()
    )


def test_previously_split_questions_are_single_records():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    perfect_number = df[
        df["question"].str.contains("عدد 6 یک عدد کامل", regex=False)
    ]
    win_rate = df[df["question"].str.contains("رکورد 60% برد", regex=False)]

    assert len(perfect_number) == 1
    assert perfect_number.iloc[0]["answer"] == "2"
    assert perfect_number.iloc[0]["option4"] == "۱۶"
    assert len(win_rate) == 1
    assert win_rate.iloc[0]["answer"] == "3"
    assert win_rate.iloc[0]["option4"] == "نمیتوان تعیین کرد"
