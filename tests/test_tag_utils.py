import pandas as pd

from leaderboard_lib.data_utils import (
    explode_tag_column,
    filter_rows_by_tags,
    split_tag_value,
    tag_counts,
)


def test_split_tag_value_supports_pipe_and_comma():
    assert split_tag_value("iran | israel, syria | iran") == [
        "iran",
        "israel",
        "syria",
    ]
    assert split_tag_value(None) == []


def test_filter_rows_by_tags_matches_membership_in_multi_tag_cells():
    df = pd.DataFrame(
        {
            "region_tags": ["iran|israel", "syria, lebanon", "qatar"],
            "level": ["easy", "hard", "easy"],
        }
    )

    filtered = filter_rows_by_tags(
        df, {"region_tags": {"israel", "lebanon"}, "level": {"easy"}}
    )

    assert filtered.index.tolist() == [0]


def test_explode_and_count_treat_each_tag_as_a_separate_group():
    df = pd.DataFrame({"tags": ["a|b", "a, c", "a|a", None]})

    exploded = explode_tag_column(df, "tags", keep_empty=True)
    exploded_tags = exploded["tags"].tolist()
    assert exploded_tags[:-1] == ["a", "b", "a", "c", "a"]
    assert pd.isna(exploded_tags[-1])

    counts = tag_counts(df, "tags").set_index("tags")
    assert counts["Count"].to_dict() == {"a": 3, "b": 1, "c": 1}
    assert counts.loc["a", "Percent"] == 75.0
