from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))
from leaderboard_lib.data_utils import _norm, _read_dataset


def test_norm_behaviour():
    assert _norm(" ۱۲۳ ") == "123"
    assert _norm(float("nan")) == ""
    assert _norm("۱۲a") == "12a"


def test_read_dataset_fallback_and_logging(tmp_path, caplog):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("col1,col2\n1,2\n3,\"4\n5,6\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        df = _read_dataset(str(bad_csv), verbose=True)
    assert df.shape == (1, 2)
    assert any("Falling back to engine='python'" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
