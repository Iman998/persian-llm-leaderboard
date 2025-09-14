from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.style import apply_gradient


def test_apply_gradient_handles_nan_in_average():
    df = pd.DataFrame(
        {
            "Average": [0.5, np.nan, 0.3],
            "Model": ["m1", "m2", "m3"],
        }
    )
    styler = apply_gradient(df)
    assert isinstance(styler, pd.io.formats.style.Styler)
