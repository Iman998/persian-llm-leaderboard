from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for path in (ROOT, APP):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from app.views.battle import _model_rate_table


def test_model_rate_table_has_win_loss_and_equal_for_both_models():
    row = pd.Series(
        {
            "Model 1": "First",
            "Model 2": "Second",
            "Model 1 Win Rate": 0.6,
            "Model 1 Loss Rate": 0.3,
            "Model 2 Win Rate": 0.3,
            "Model 2 Loss Rate": 0.6,
            "Equal Rate": 0.1,
        }
    )

    table = _model_rate_table(row)

    assert table["Model"].tolist() == ["First", "Second"]
    assert table["Win Rate"].tolist() == [0.6, 0.3]
    assert table["Loss Rate"].tolist() == [0.3, 0.6]
    assert table["Equal Rate"].tolist() == [0.1, 0.1]
