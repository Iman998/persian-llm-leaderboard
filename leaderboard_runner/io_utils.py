"""
io_utils.py
~~~~~~~~~~~
Utility helpers for reading list arguments and sampling CSV files.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import List

import pandas as pd


def parse_csv_or_file(arg: str) -> List[str]:
    """
    Expand either a comma-separated string **or** a newline-delimited file
    into a list of non-empty, stripped tokens.
    """
    p = Path(arg)
    if p.is_file():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def sample_csv(src: Path, n_rows: int) -> Path:
    """
    Return a temporary CSV path containing *n_rows* random rows of *src*
    (header preserved).  Caller must delete the file afterwards.
    """
    df = pd.read_csv(
        src,
        engine="python",  # tolerant parser
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        escapechar="\\",
        on_bad_lines="skip",
    )
    df = df.sample(n=min(n_rows, len(df)), random_state=42)
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
    df.to_csv(tmp, index=False)
    return tmp
