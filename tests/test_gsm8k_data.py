from pathlib import Path
import hashlib
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from leaderboard_lib.gsm8k_utils import extract_gsm8k_answer


def test_gsm8k_official_test_data_integrity():
    dataset_dir = ROOT / "data" / "GSM8K"
    data_path = dataset_dir / "test.csv"
    meta = yaml.safe_load((dataset_dir / "meta.yaml").read_text(encoding="utf-8"))
    df = pd.read_csv(data_path, keep_default_na=False)

    assert len(df) == 1319
    assert not df["question"].str.strip().eq("").any()
    assert not df["solution"].str.strip().eq("").any()
    assert df["answer"].astype(str).str.fullmatch(r"-?\d+").all()
    assert (
        hashlib.sha256(data_path.read_bytes()).hexdigest()
        == meta["converted_data_sha256"]
    )


def test_gsm8k_solution_answers_match_canonical_column():
    data_path = ROOT / "data" / "GSM8K" / "test.csv"
    df = pd.read_csv(data_path, keep_default_na=False)

    extracted = df["solution"].map(extract_gsm8k_answer)
    assert extracted.tolist() == df["answer"].astype(str).tolist()
