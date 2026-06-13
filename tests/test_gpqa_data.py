from pathlib import Path
import hashlib

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_gpqa_main_data_integrity():
    dataset_dir = ROOT / "data" / "GPQA"
    data_path = dataset_dir / "test.csv"
    meta = yaml.safe_load((dataset_dir / "meta.yaml").read_text(encoding="utf-8"))
    df = pd.read_csv(data_path, keep_default_na=False)

    assert len(df) == 448
    assert df["record_id"].nunique() == 448
    assert df["question"].nunique() == 448
    assert df["is_diamond"].sum() == 198
    assert df["domain"].value_counts().to_dict() == {
        "Physics": 187,
        "Chemistry": 183,
        "Biology": 78,
    }
    assert df["subdomain"].nunique() == 16
    assert set(df["answer"]) <= {1, 2, 3, 4}
    assert df[[f"choice{i}" for i in range(1, 5)]].ne("").all(axis=None)
    assert df["canary_string"].eq(meta["canary_string"]).all()
    assert (
        hashlib.sha256(data_path.read_bytes()).hexdigest()
        == meta["converted_data_sha256"]
    )


def test_gpqa_license_is_included():
    license_text = (ROOT / "data" / "GPQA" / "LICENSE").read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0 International License" in license_text
