"""
meta_utils.py
~~~~~~~~~~~~~
Functions related to reading `meta.yaml` files.
"""

from pathlib import Path
from typing import Tuple

import yaml


def load_meta_fields(meta_file: Path) -> Tuple[str, str]:
    """
    Extract ``prompt_template`` and ``evaluator`` from *meta_file*.

    Fallback:
        prompt_template → prompts/mcq_fewshot.jinja2
        evaluator       → evaluators/mcq_evaluator.py
    """
    cfg = yaml.safe_load(meta_file.read_text()) or {}
    return (
        cfg.get("prompt_template", "prompts/mcq_fewshot.jinja2"),
        cfg.get("evaluator", "evaluators/mcq_evaluator.py"),
    )
