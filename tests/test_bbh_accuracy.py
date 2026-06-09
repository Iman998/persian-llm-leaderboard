from leaderboard_lib.bbh_utils import (
    bbh_macro_accuracy,
    make_bbh_scoring_key,
    normalize_bbh_answer,
)


def _key(task, family, target):
    return make_bbh_scoring_key(task, family, target)


def test_normalization_handles_common_bbh_answer_formats():
    assert normalize_bbh_answer("  (A). ") == "(a)"
    assert normalize_bbh_answer(r"\boxed{42}") == "42"
    assert normalize_bbh_answer("]   }  ]") == "] } ]"


def test_macro_accuracy_weights_task_families_equally():
    labels = [
        _key("boolean_expressions", "boolean_expressions", "True"),
        _key("boolean_expressions", "boolean_expressions", "False"),
        _key("word_sorting", "word_sorting", "alpha beta"),
    ]
    predictions = [
        _key("boolean_expressions", "boolean_expressions", "true"),
        _key("boolean_expressions", "boolean_expressions", "True"),
        _key("word_sorting", "word_sorting", "alpha beta"),
    ]

    assert bbh_macro_accuracy(predictions, labels) == 0.75


def test_task_mismatch_is_incorrect():
    label = _key("navigate", "navigate", "Yes")
    prediction = _key("web_of_lies", "web_of_lies", "Yes")
    assert bbh_macro_accuracy([prediction], [label]) == 0.0
