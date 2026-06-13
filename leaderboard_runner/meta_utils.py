"""
meta_utils.py
~~~~~~~~~~~~~
Functions related to reading `meta.yaml` files.
"""

from pathlib import Path
from typing import Any, Tuple

import yaml


DEFAULT_JUDGE_PROMPT = "prompts/judge.jinja2"
DEFAULT_JUDGE_EVALUATOR = "evaluators/judge_evaluator.py"
JUDGE_MODES = ("reference", "no-reference", "both")


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


def load_judge_configs(
    meta_cfg: dict[str, Any],
    *,
    candidate_model: str,
    judge_model_override: str | None = None,
    judge_mode: str = "reference",
) -> list[dict[str, Any]]:
    """Return one normalized config for each requested judge mode.

    ``judge: true`` remains supported for older datasets. New datasets should
    use a mapping so the evaluator model and rubric are explicit.
    """
    if judge_mode not in JUDGE_MODES:
        raise ValueError(f"Unsupported judge mode: {judge_mode}")

    raw = meta_cfg.get("judge", False)
    if isinstance(raw, dict):
        config = raw.copy()
        enabled = bool(config.pop("enabled", True))
    else:
        config = {}
        enabled = bool(raw)

    if not enabled:
        return []

    judge_model = judge_model_override or config.get("model") or candidate_model
    reference_prompt = config.get(
        "reference_prompt_template",
        config.get("prompt_template", DEFAULT_JUDGE_PROMPT),
    )
    no_reference_prompt = config.get(
        "no_reference_prompt_template",
        config.get("prompt_template", DEFAULT_JUDGE_PROMPT),
    )
    modes = ("reference", "no-reference") if judge_mode == "both" else (judge_mode,)

    return [
        {
            "mode": mode,
            "model": judge_model,
            "prompt_template": (
                reference_prompt if mode == "reference" else no_reference_prompt
            ),
            "evaluator": config.get("evaluator", DEFAULT_JUDGE_EVALUATOR),
            "use_reference": mode == "reference",
            "score_min": config.get("score_min"),
            "score_max": config.get("score_max"),
            "metrics": config.get("metrics", ["llm_judge_score"]),
        }
        for mode in modes
    ]
