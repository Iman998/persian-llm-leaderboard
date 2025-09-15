import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_runner import board_builder, paths


def test_rebuild_leaderboard_dry_run(capsys):
    base_cmd = [
        sys.executable,
        paths.BUILD_BOARD_SCRIPT,
        "--results_dir",
        paths.RESULTS_DIR,
        "--datasets_dir",
        paths.DATASETS_DIR,
    ]

    boards = [
        (paths.LEADERBOARD_OUT, "all", None, ["translat", "summar", "summary"]),
        (paths.LEADERBOARD_FA_OUT, "fa", None, ["translat", "summar", "summary"]),
        (paths.LEADERBOARD_EN_OUT, "en", None, ["translat", "summar", "summary"]),
        (paths.TRANSLATION_OUT, "all", ["translat"], ["translation_quality"]),
        (
            paths.SUMMARIZATION_OUT,
            "all",
            ["summar", "summary"],
            ["summarization_quality"],
        ),
        (
            paths.SUMMARIZATION_FA_OUT,
            "fa",
            ["summar", "summary"],
            ["summarization_quality"],
        ),
        (
            paths.SUMMARIZATION_EN_OUT,
            "en",
            ["summar", "summary"],
            ["summarization_quality"],
        ),
    ]

    expected = []
    for out_path, lang, include, exclude in boards:
        cmd = base_cmd + ["--out", out_path, "--lang", lang]
        if include:
            cmd += ["--include", *include]
        if exclude:
            cmd += ["--exclude", *exclude]
        expected.append(" ".join(map(str, cmd)))

    judge_cmd = [
        sys.executable,
        paths.BUILD_JUDGE_BOARD_SCRIPT,
        "--results_dir",
        paths.RESULTS_DIR,
        "--datasets_dir",
        paths.DATASETS_DIR,
        "--out",
        paths.LEADERBOARD_JUDGE_OUT,
    ]
    expected.append(" ".join(map(str, judge_cmd)))

    board_builder.rebuild_leaderboard(dry_run=True)
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines() == expected
