import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from leaderboard_lib import leaderboard


def test_translation_board_average(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    dataset = "trans_ds"
    model = "modelA"

    (results_dir / dataset / model).mkdir(parents=True)
    (data_dir / dataset).mkdir(parents=True)
    models_dir.mkdir(parents=True)

    (models_dir / f"{model}.yaml").write_text("model_type: instruct\n")

    meta = {
        "board": "translation",
        "answer_col": "target",
        "metrics": ["bleu", "meteor", "chrf", "ter"],
    }
    (data_dir / dataset / "meta.yaml").write_text(yaml.dump(meta))

    pd.DataFrame({"pred": ["hello world"], "target": ["hello world"]}).to_csv(
        results_dir / dataset / model / f"{model}.csv", index=False
    )

    out_file = tmp_path / "board.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leaderboard",
            "--results_dir",
            str(results_dir),
            "--datasets_dir",
            str(data_dir),
            "--models_dir",
            str(models_dir),
            "--out",
            str(out_file),
            "--board",
            "translation",
        ],
    )
    leaderboard.main()
    df = pd.read_csv(out_file)
    assert pd.api.types.is_numeric_dtype(df["Average"])
    assert df["Average"].notna().all()
