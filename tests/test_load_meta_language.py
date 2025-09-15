from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.io import load_meta


def test_load_meta_missing_language():
    meta = load_meta("translation")
    assert meta["language"] == ""
